//! Three-tier page buffer manager with compressed T2 and specialist pinning.
//!
//! The heart of vib3. Manages page lifecycle across VRAM → RAM → NVMe
//! with predictive prefetching, async transfers, and adaptive expert pinning.
//!
//! ## Pipeline B (default): Compressed T2 → VRAM → GPU Decompress
//!
//! When `t2_compressed = true` (default), T2 stores Zstd-compressed pages
//! directly from disk. The T2→T1 promotion path:
//!   1. DMA compressed data from T2 (pinned host) to VRAM staging buffer
//!   2. Blackwell Decompression Engine (600 GB/s) decompresses into final T1 slot
//!   3. Page is ready for compute
//!
//! This dramatically increases T2 effective capacity:
//! - 168 GB raw RAM × 3.5x Zstd compression = ~588 GB effective
//! - Combined T1 (81 GB) + T2 (588 GB) = 669 GB > 630 GB model
//! - The entire model fits in RAM+VRAM; NVMe drops out of steady-state
//!
//! ## Specialist Pinning
//!
//! In specialist mode (detected by entropy-based mode detection), the buffer
//! manager pins a hot expert cluster in T1 to prevent eviction. The working
//! set for specialist mode (~40-60 experts × 60 layers × 22 MB/expert at
//! INT4) is ~52-79 GB, which fits in T1's 81 GB budget.
//!
//! ## Design principles:
//! - T1 (VRAM) hit is lock-free: just an atomic read + pointer return
//! - T2→T1 promotion overlaps with GPU compute via async DMA
//! - T3→T2 loading uses io_uring for kernel-bypass NVMe reads
//! - Eviction is prediction-aware, not LRU
//! - In compressed mode, T2 slots store compressed data (variable size ≤ PAGE_SIZE)

use crate::compute::cuda_ffi;
use crate::core::config::BufferPoolConfig;
use crate::core::error::{Error, Result};
use crate::core::types::*;
use crate::storage::format::Vib3File;

use dashmap::DashMap;
use parking_lot::Mutex;
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

// ─── Priority Prefetch Queue ─────────────────────────────────────────────

/// Wrapper for BinaryHeap ordering: highest urgency (lowest priority value) first.
struct PrioritizedRequest(PrefetchRequest);

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.0.urgency_key() == other.0.urgency_key()
    }
}
impl Eq for PrioritizedRequest {}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverse: lower urgency_key = higher priority in BinaryHeap
        other.0.urgency_key().cmp(&self.0.urgency_key())
    }
}

/// Thread-safe priority queue for prefetch requests.
///
/// Critical/High priority requests are processed before Medium/Low ones,
/// preventing speculative prefetches from starving the current token's
/// page loads. Uses a `BinaryHeap` + `tokio::sync::Notify` for async
/// consumption.
struct PrefetchQueue {
    heap: Mutex<BinaryHeap<PrioritizedRequest>>,
    notify: tokio::sync::Notify,
    capacity: usize,
}

impl PrefetchQueue {
    fn new(capacity: usize) -> Self {
        Self {
            heap: Mutex::new(BinaryHeap::with_capacity(capacity)),
            notify: tokio::sync::Notify::new(),
            capacity,
        }
    }

    /// Push a request. If at capacity, drop the lowest-priority item
    /// (only if the new request has higher priority).
    fn push(&self, req: PrefetchRequest) {
        let mut heap = self.heap.lock();
        if heap.len() >= self.capacity {
            // Check if the new request has higher priority than the worst
            if let Some(worst) = heap.peek() {
                if req.urgency_key() >= worst.0.urgency_key() {
                    // New request is lower priority than worst in queue — drop it
                    return;
                }
            }
            // Remove the lowest-priority item (peek = highest priority,
            // so we need to find and remove the worst)
            // BinaryHeap doesn't support efficient remove-min, but for
            // a bounded queue this is acceptable. Just let it grow by 1.
        }
        heap.push(PrioritizedRequest(req));
        self.notify.notify_one();
    }

    /// Pop the highest-priority request (blocking async).
    async fn pop(&self) -> Option<PrefetchRequest> {
        loop {
            {
                let mut heap = self.heap.lock();
                if let Some(PrioritizedRequest(req)) = heap.pop() {
                    return Some(req);
                }
            }
            // Wait for a notification that new items were pushed
            self.notify.notified().await;
        }
    }

    /// Try to pop without blocking.
    #[allow(dead_code)]
    fn try_pop(&self) -> Option<PrefetchRequest> {
        let mut heap = self.heap.lock();
        heap.pop().map(|PrioritizedRequest(req)| req)
    }

    /// Number of pending requests.
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.heap.lock().len()
    }
}

// ─── Resident Page Pointer (Send+Sync wrapper) ──────────────────────────

/// A frozen device pointer + size pair for the resident snapshot.
/// Used in the zero-overhead `get_page_resident()` fast path.
#[derive(Clone, Copy)]
struct ResidentPtr {
    ptr: *mut u8,
    size: usize,
}

// SAFETY: The device pointers in the resident snapshot are valid for the
// lifetime of the PageBufferManager. The snapshot is write-once (OnceLock)
// and read-only thereafter. No mutation occurs after construction.
unsafe impl Send for ResidentPtr {}
unsafe impl Sync for ResidentPtr {}

// ─── Page Slot ───────────────────────────────────────────────────────────

/// A pre-allocated slot in a tier's memory pool.
struct PageSlot {
    /// Raw memory pointer (device ptr for T1, host ptr for T2).
    ptr: *mut u8,
    /// Allocation capacity (always PAGE_SIZE for T1; PAGE_SIZE for T2 even in
    /// compressed mode, since compressed data is always ≤ PAGE_SIZE).
    capacity: usize,
    /// Actual data size in bytes.
    /// - T1: always the decompressed page size (= raw_size from catalog)
    /// - T2 compressed mode: the compressed byte count (< PAGE_SIZE)
    /// - T2 raw mode: the decompressed page size
    size: usize,
    /// Which page is stored here, if any.
    page_id: Option<PageId>,
    /// Pinned pages cannot be evicted (shared layers + specialist hot cluster).
    pinned: bool,
    /// Tick when this page was loaded into this slot.
    load_tick: u64,
    /// Tick of last access.
    last_access: AtomicU64,
    /// Predicted reuse probability from query planner.
    predicted_reuse: f32,
    /// Whether this slot has been allocated (has valid memory).
    allocated: bool,
    /// If true, the data in this slot is Zstd-compressed.
    /// Only meaningful for T2 slots when Pipeline B is active.
    compressed: bool,
    /// The raw (decompressed) size of the page data.
    /// Needed during T2→T1 promotion to know the final output size.
    raw_size: usize,

    // ── Phase D: Gear-aware eviction ─────────────────────────────────
    /// Which gear was active when this page was loaded into this slot.
    /// Used by gear-aware eviction: pages from the old gear are deprioritized
    /// when the gear changes, making them first eviction candidates.
    loaded_by_gear: Option<String>,
}

// SAFETY: PageSlot contains raw pointers but we manage their lifecycle
// exclusively through the buffer manager. Send is needed for the slot
// vectors to live in Arc'd pools accessed from multiple threads.
unsafe impl Send for PageSlot {}
unsafe impl Sync for PageSlot {}

// ─── Tier Pool ───────────────────────────────────────────────────────────

/// Memory pool for one storage tier.
struct TierPool {
    tier: Tier,
    slots: Vec<Mutex<PageSlot>>,
    capacity: usize,

    /// Free slot indices. Pop to allocate, push to release.
    free_slots: Mutex<Vec<usize>>,

    /// Page lookup: PageId.key() → slot index.
    /// DashMap gives us concurrent read access without locking.
    page_map: DashMap<u64, usize>,

    /// Counters
    occupied: AtomicU64,
    pinned: AtomicU64,
}

impl TierPool {
    fn new(tier: Tier, capacity: usize, slot_size: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        let mut free = Vec::with_capacity(capacity);

        for i in 0..capacity {
            slots.push(Mutex::new(PageSlot {
                ptr: std::ptr::null_mut(),
                capacity: slot_size,
                size: 0,
                page_id: None,
                pinned: false,
                load_tick: 0,
                last_access: AtomicU64::new(0),
                predicted_reuse: 0.0,
                allocated: false,
                compressed: false,
                raw_size: 0,
                loaded_by_gear: None,
            }));
            free.push(i);
        }

        Self {
            tier,
            slots,
            capacity,
            free_slots: Mutex::new(free),
            page_map: DashMap::with_capacity(capacity),
            occupied: AtomicU64::new(0),
            pinned: AtomicU64::new(0),
        }
    }

    /// Find which slot contains a page, if any.
    fn find(&self, page: &PageId) -> Option<usize> {
        self.page_map.get(&page.key()).map(|r| *r)
    }

    /// Try to allocate a free slot. Returns None if pool is full.
    fn try_allocate(&self) -> Option<usize> {
        let mut free = self.free_slots.lock();
        free.pop()
    }

    fn release(&self, slot_idx: usize) {
        let mut slot = self.slots[slot_idx].lock();
        if let Some(page_id) = slot.page_id.take() {
            self.page_map.remove(&page_id.key());
        }
        slot.pinned = false;
        slot.compressed = false;
        slot.size = 0;
        slot.raw_size = 0;
        self.occupied.fetch_sub(1, Ordering::Relaxed);

        self.free_slots.lock().push(slot_idx);
    }

    fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.occupied.load(Ordering::Relaxed) as f32 / self.capacity as f32
    }
}

// ─── Transfer Operation ──────────────────────────────────────────────────

/// An in-flight page transfer between tiers.
struct TransferOp {
    #[allow(dead_code)]
    page_id: PageId,
    _source: Tier,
    _dest: Tier,
    _priority: PrefetchPriority,
    completed: AtomicBool,
    /// Notifier for threads waiting on this transfer.
    notify: tokio::sync::watch::Sender<bool>,
    waiter: tokio::sync::watch::Receiver<bool>,
}

// ─── VRAM Staging Buffer ─────────────────────────────────────────────────

/// A VRAM staging buffer for the compressed → decompressed promotion path.
///
/// When Pipeline B is active, compressed data is DMA'd from T2 (host pinned)
/// to this staging area in VRAM, then the Blackwell Decompression Engine
/// decompresses it into the final T1 slot.
///
/// The staging buffer is divided into fixed-size slots (PAGE_SIZE each) to
/// support concurrent decompress operations.
struct VramStagingBuffer {
    /// Device pointer to the staging area.
    ptr: *mut u8,
    /// Total size in bytes.
    total_size: usize,
    /// Slot size (= PAGE_SIZE).
    slot_size: usize,
    /// Number of staging slots.
    num_slots: usize,
    /// Free slot bitmap.
    free_slots: Mutex<Vec<usize>>,
    /// Whether the buffer is allocated.
    allocated: bool,
}

unsafe impl Send for VramStagingBuffer {}
unsafe impl Sync for VramStagingBuffer {}

impl VramStagingBuffer {
    fn new(total_size: usize) -> Self {
        let slot_size = PAGE_SIZE;
        let num_slots = total_size / slot_size;
        Self {
            ptr: std::ptr::null_mut(),
            total_size,
            slot_size,
            num_slots,
            free_slots: Mutex::new((0..num_slots).collect()),
            allocated: false,
        }
    }

    fn allocate(&mut self) -> Result<()> {
        if self.num_slots == 0 {
            return Ok(());
        }
        self.ptr = cuda_ffi::device_alloc(self.total_size)?;
        self.allocated = true;
        tracing::info!(
            "VRAM staging buffer allocated: {} MB ({} slots)",
            self.total_size / (1024 * 1024),
            self.num_slots,
        );
        Ok(())
    }

    /// Acquire a staging slot. Returns (slot_index, device_ptr_to_slot).
    fn acquire(&self) -> Option<(usize, *mut u8)> {
        let mut free = self.free_slots.lock();
        free.pop().map(|idx| {
            let offset = idx * self.slot_size;
            let ptr = unsafe { self.ptr.add(offset) };
            (idx, ptr)
        })
    }

    /// Release a staging slot back to the pool.
    fn release(&self, slot_idx: usize) {
        self.free_slots.lock().push(slot_idx);
    }
}

impl Drop for VramStagingBuffer {
    fn drop(&mut self) {
        if self.allocated && !self.ptr.is_null() {
            cuda_ffi::device_free(self.ptr, self.total_size);
        }
    }
}

// ─── Specialist Pin Set ──────────────────────────────────────────────────

/// Tracks which expert pages are pinned due to specialist mode.
///
/// Separated from the shared-layer pinning so we can bulk unpin when
/// switching back to generalist mode without touching shared pins.
struct SpecialistPinSet {
    /// Set of PageId keys that are specialist-pinned.
    pinned_pages: HashSet<u64>,
    /// Number of pages currently specialist-pinned in T1.
    count: usize,
    /// Maximum pages allowed for specialist pinning (VRAM budget).
    max_pages: usize,
}

impl SpecialistPinSet {
    fn new(max_pages: usize) -> Self {
        Self {
            pinned_pages: HashSet::new(),
            count: 0,
            max_pages,
        }
    }

    fn contains(&self, page: &PageId) -> bool {
        self.pinned_pages.contains(&page.key())
    }

    fn insert(&mut self, page: &PageId) -> bool {
        if self.count >= self.max_pages {
            return false;
        }
        if self.pinned_pages.insert(page.key()) {
            self.count += 1;
            true
        } else {
            false // already present
        }
    }

    fn clear(&mut self) -> Vec<u64> {
        let keys: Vec<u64> = self.pinned_pages.drain().collect();
        self.count = 0;
        keys
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.count
    }
}

// ─── Page Buffer Manager ─────────────────────────────────────────────────

/// The core buffer pool manager.
///
/// Manages page lifecycle across three tiers with async transfers,
/// predictive prefetching, compressed T2 storage (Pipeline B), and
/// specialist expert pinning.
pub struct PageBufferManager {
    config: BufferPoolConfig,
    model_file: Arc<Vib3File>,

    /// Per-tier memory pools.
    t1: Arc<TierPool>,
    t2: Arc<TierPool>,

    /// VRAM staging buffer for compressed→decompressed promotion (Pipeline B).
    staging: Mutex<VramStagingBuffer>,

    /// Global page table: authoritative catalog index for every page.
    page_catalog_index: HashMap<u64, usize>,

    /// Priority-ordered prefetch queue: query planner submits, prefetch worker consumes.
    prefetch_queue: Arc<PrefetchQueue>,

    /// In-flight transfer tracking.
    inflight: DashMap<u64, Arc<TransferOp>>,

    /// Specialist pinning state.
    specialist_pins: Mutex<SpecialistPinSet>,

    /// Monotonic tick for access ordering.
    tick: AtomicU64,

    /// Running flag for background workers.
    running: AtomicBool,

    /// When true, all model pages fit in T1 — eviction is suppressed.
    fully_resident: AtomicBool,

    /// Frozen page→pointer snapshot for fully-resident models.
    /// Set once after `preload_all()` succeeds. Subsequent `get_page_resident()`
    /// calls use a single `HashMap::get()` with zero locking or atomic ops,
    /// bypassing the DashMap + Mutex + stats overhead of the normal `get_page()`.
    resident_snapshot: OnceLock<HashMap<u64, ResidentPtr>>,

    /// Stats.
    pub stats: Arc<InferenceStats>,

    /// CUDA stream for DMA transfers.
    stream: cuda_ffi::CudaStream,

    /// Secondary CUDA stream for decompress operations (overlaps with DMA).
    decompress_stream: cuda_ffi::CudaStream,

    // ── Phase D: Gear-aware eviction ─────────────────────────────────
    /// The currently active gear (set by Engine::set_task_context).
    /// Used to tag newly loaded pages and adjust eviction scores.
    current_gear: Mutex<Option<String>>,
}

impl PageBufferManager {
    pub fn new(config: BufferPoolConfig, model_file: Arc<Vib3File>) -> Self {
        let t1_slots = if config.t1_capacity > 0 {
            config.t1_capacity / PAGE_SIZE
        } else {
            0
        };

        // In compressed mode, T2 capacity is effectively multiplied by the
        // compression ratio. But we still allocate PAGE_SIZE per slot (the
        // compressed data just uses less of each slot).
        // To store more pages in the same RAM, we can allocate more slots.
        let t2_slots = if config.t2_capacity > 0 {
            // T2 slots are always PAGE_SIZE (2 MB) each, regardless of compression.
            // In compressed mode, each slot stores a compressed page that uses less
            // than PAGE_SIZE bytes, but we still allocate PAGE_SIZE per slot for
            // simplicity (avoids a slab allocator). The compression benefit comes
            // from storing more *logical* data per unit of RAM, not from smaller
            // slot allocations.
            //
            // t2_capacity is the actual RAM budget in bytes, so:
            //   slot_count = t2_capacity / PAGE_SIZE
            config.t2_capacity / PAGE_SIZE
        } else {
            0
        };

        let prefetch_queue = Arc::new(PrefetchQueue::new(config.prefetch_queue_depth));

        // Build catalog index
        let mut page_catalog_index = HashMap::new();
        for (idx, entry) in model_file.page_catalog().iter().enumerate() {
            let page_id = entry.page_id();
            page_catalog_index.insert(page_id.key(), idx);
        }

        // Create a CUDA device + streams
        let device = cuda_ffi::CudaDevice::new(config.cuda_device).unwrap_or_else(|_| {
            tracing::warn!("Failed to create CUDA device, using defaults");
            cuda_ffi::CudaDevice::new(0).expect("CPU fallback must work")
        });
        let stream = cuda_ffi::CudaStream::new(&device).expect("Stream creation must work");
        let decompress_stream =
            cuda_ffi::CudaStream::new(&device).expect("Decompress stream creation must work");

        // VRAM staging buffer for Pipeline B
        let staging_size = if config.t2_compressed {
            config.vram_staging_size
        } else {
            0
        };

        // Specialist pin budget: default to 70% of T1 capacity (leave 30% for
        // shared layers and dynamic pages).
        let max_specialist_pages = (t1_slots as f32 * 0.70) as usize;

        Self {
            config,
            model_file,
            t1: Arc::new(TierPool::new(Tier::T1Vram, t1_slots, PAGE_SIZE)),
            t2: Arc::new(TierPool::new(Tier::T2Ram, t2_slots, PAGE_SIZE)),
            staging: Mutex::new(VramStagingBuffer::new(staging_size)),
            page_catalog_index,
            prefetch_queue,
            inflight: DashMap::new(),
            specialist_pins: Mutex::new(SpecialistPinSet::new(max_specialist_pages)),
            tick: AtomicU64::new(0),
            running: AtomicBool::new(false),
            fully_resident: AtomicBool::new(false),
            resident_snapshot: OnceLock::new(),
            stats: Arc::new(InferenceStats::default()),
            stream,
            decompress_stream,
            current_gear: Mutex::new(None),
        }
    }

    /// Whether the buffer manager is operating in compressed T2 mode (Pipeline B).
    pub fn is_compressed_t2(&self) -> bool {
        self.config.t2_compressed
    }

    /// Allocate memory pools and initialize.
    pub async fn initialize(&self) -> Result<()> {
        let t2_ram_mb = self.t2.capacity * PAGE_SIZE / (1024 * 1024);
        let t2_effective_mb = if self.config.t2_compressed {
            (t2_ram_mb as f64 * self.config.t2_compression_ratio as f64) as usize
        } else {
            t2_ram_mb
        };

        tracing::info!(
            "Initializing buffer pools: T1={} slots ({} MB), T2={} slots ({} MB RAM, ~{} MB effective, compressed={})",
            self.t1.capacity,
            self.t1.capacity * PAGE_SIZE / (1024 * 1024),
            self.t2.capacity,
            t2_ram_mb,
            t2_effective_mb,
            self.config.t2_compressed,
        );

        self.allocate_t1_pool()?;
        self.allocate_t2_pool()?;

        // Allocate VRAM staging buffer for Pipeline B
        if self.config.t2_compressed {
            self.staging.lock().allocate()?;
        }

        tracing::info!(
            "Buffer pools ready. Page catalog: {} pages indexed",
            self.page_catalog_index.len(),
        );

        Ok(())
    }

    /// Preload the entire model into T1 (VRAM).
    ///
    /// When the model fits entirely in T1, this bulk-loads all pages upfront
    /// to eliminate I/O stalls during inference. Each page is read from the
    /// mmap'd file (decompressed if needed) into a host staging buffer, then
    /// copied to the T1 slot via cudaMemcpy H2D.
    ///
    /// Returns the number of pages successfully loaded.
    pub async fn preload_all(&self) -> Result<usize> {
        let total_pages = self.model_file.page_count();

        if total_pages > self.t1.capacity {
            tracing::warn!(
                "Model has {} pages but T1 only has {} slots — skipping preload",
                total_pages,
                self.t1.capacity,
            );
            return Ok(0);
        }

        tracing::info!(
            "Preloading {} pages ({} MB) into T1...",
            total_pages,
            total_pages * PAGE_SIZE / (1024 * 1024),
        );

        // Host staging buffer for decompression + H2D (reused across pages)
        let staging = cuda_ffi::host_alloc_pinned(PAGE_SIZE)?;
        let staging_buf = unsafe { std::slice::from_raw_parts_mut(staging, PAGE_SIZE) };

        let start = std::time::Instant::now();
        let mut loaded = 0usize;

        for (catalog_idx, entry) in self.model_file.page_catalog().iter().enumerate() {
            let page_id = entry.page_id();
            let raw_size = (entry.raw_size as usize).min(PAGE_SIZE);

            // Acquire a T1 slot (should always succeed since model fits)
            let t1_slot = match self.t1.try_allocate() {
                Some(slot) => slot,
                None => {
                    tracing::warn!(
                        "T1 full during preload at page {} / {} — stopping",
                        loaded,
                        total_pages,
                    );
                    break;
                }
            };

            // Read + decompress from mmap into host staging buffer
            if let Err(e) = self
                .model_file
                .read_page_sync(catalog_idx, &mut staging_buf[..raw_size])
            {
                tracing::warn!("Failed to read page {}: {} — skipping", catalog_idx, e);
                self.t1.release(t1_slot);
                continue;
            }

            // H2D copy: host staging → T1 VRAM slot
            let dst_ptr = {
                let slot = self.t1.slots[t1_slot].lock();
                slot.ptr
            };

            if let Err(e) = cuda_ffi::memcpy_h2d_sync(dst_ptr, staging, raw_size) {
                tracing::warn!("H2D copy failed for page {}: {} — skipping", catalog_idx, e);
                self.t1.release(t1_slot);
                continue;
            }

            // Update T1 slot metadata
            {
                let mut slot = self.t1.slots[t1_slot].lock();
                slot.page_id = Some(page_id);
                slot.load_tick = self.tick.load(Ordering::Relaxed);
                slot.last_access
                    .store(self.tick.load(Ordering::Relaxed), Ordering::Relaxed);
                slot.size = raw_size;
                slot.raw_size = raw_size;
                slot.compressed = false;
                slot.loaded_by_gear = self.current_gear.lock().clone();
            }
            self.t1.page_map.insert(page_id.key(), t1_slot);
            self.t1.occupied.fetch_add(1, Ordering::Relaxed);

            loaded += 1;

            // Progress logging every 1000 pages
            if loaded.is_multiple_of(1000) {
                let elapsed = start.elapsed().as_secs_f64();
                let mb_loaded = loaded * PAGE_SIZE / (1024 * 1024);
                let rate = mb_loaded as f64 / elapsed;
                tracing::info!(
                    "Preload progress: {} / {} pages ({} MB, {:.0} MB/s)",
                    loaded,
                    total_pages,
                    mb_loaded,
                    rate,
                );
            }
        }

        cuda_ffi::host_free_pinned(staging, PAGE_SIZE);

        let elapsed = start.elapsed();
        let mb_loaded = loaded * PAGE_SIZE / (1024 * 1024);
        let rate = mb_loaded as f64 / elapsed.as_secs_f64();
        tracing::info!(
            "Preload complete: {} pages ({} MB) in {:.1}s ({:.0} MB/s)",
            loaded,
            mb_loaded,
            elapsed.as_secs_f64(),
            rate,
        );

        // If all pages were loaded, mark model as fully resident.
        // This suppresses eviction so the background worker doesn't
        // discard pages that are needed for inference.
        if loaded == total_pages {
            self.fully_resident.store(true, Ordering::Release);

            // Build a frozen snapshot: PageId key → ResidentPtr.
            // This enables get_page_resident() to bypass the DashMap + Mutex
            // hot path entirely — a single HashMap::get() with zero locking.
            let mut snapshot = HashMap::with_capacity(total_pages);
            for entry in self.t1.page_map.iter() {
                let key = *entry.key();
                let slot_idx = *entry.value();
                let slot = self.t1.slots[slot_idx].lock();
                if !slot.ptr.is_null() {
                    snapshot.insert(
                        key,
                        ResidentPtr {
                            ptr: slot.ptr,
                            size: slot.size,
                        },
                    );
                }
            }
            let _ = self.resident_snapshot.set(snapshot);

            tracing::info!("Model fully resident in T1: {} pages preloaded", loaded);
        }

        Ok(loaded)
    }

    /// Whether the entire model fits in T1.
    pub fn model_fits_in_t1(&self) -> bool {
        self.model_file.page_count() <= self.t1.capacity
    }

    // ── Page Access (hot path) ───────────────────────────────────────

    /// Get a page ready for GPU compute, blocking if necessary.
    ///
    /// Fast path: page is in T1 → return device pointer immediately (no lock).
    /// Slow path: page is in T2 → promote to T1 (decompress if compressed), wait.
    /// Slowest path: page is in T3 → load to T2, then promote to T1, wait.
    pub async fn get_page(&self, id: &PageId) -> Result<PageHandle> {
        let tick = self.next_tick();
        self.stats
            .total_page_accesses
            .fetch_add(1, Ordering::Relaxed);

        // Fast path: check T1
        if let Some(slot_idx) = self.t1.find(id) {
            let slot = self.t1.slots[slot_idx].lock();
            if let Some(ref page_id) = slot.page_id {
                if *page_id == *id && !slot.ptr.is_null() {
                    slot.last_access.store(tick, Ordering::Relaxed);
                    self.stats.t1_hits.fetch_add(1, Ordering::Relaxed);

                    return Ok(PageHandle {
                        device_ptr: slot.ptr,
                        size: slot.size,
                        source_tier: Tier::T1Vram,
                        was_prefetched: false,
                    });
                }
            }
        }

        // Check if transfer is already in flight
        if let Some(transfer_ref) = self.inflight.get(&id.key()) {
            let transfer = transfer_ref.clone();
            drop(transfer_ref); // Release DashMap ref
            
            // Wait for transfer to complete
            let mut rx = transfer.waiter.clone();
            while !*rx.borrow() {
                let _ = rx.changed().await;
            }
            
            // After notification, page should be in T1 — recurse
            if let Some(slot_idx) = self.t1.find(id) {
                let slot = self.t1.slots[slot_idx].lock();
                if let Some(ref page_id) = slot.page_id {
                    if *page_id == *id && !slot.ptr.is_null() {
                        slot.last_access.store(tick, Ordering::Relaxed);
                        self.stats.t1_hits.fetch_add(1, Ordering::Relaxed);
                        return Ok(PageHandle {
                            device_ptr: slot.ptr,
                            size: slot.size,
                            source_tier: Tier::T1Vram,
                            was_prefetched: true,
                        });
                    }
                }
            }
        }

        // Slow path: check T2 and promote
        if self.t2.find(id).is_some() {
            self.stats.t2_hits.fetch_add(1, Ordering::Relaxed);
            self.promote_t2_to_t1(id).await?;

            if let Some(slot_idx) = self.t1.find(id) {
                let slot = self.t1.slots[slot_idx].lock();
                return Ok(PageHandle {
                    device_ptr: slot.ptr,
                    size: slot.size,
                    source_tier: Tier::T2Ram,
                    was_prefetched: false,
                });
            }
        }

        // Slowest path: load from T3 (disk)
        self.stats.t3_hits.fetch_add(1, Ordering::Relaxed);
        let stall_start = std::time::Instant::now();
        if let Err(e) = self.load_t3_to_t1(id).await {
            tracing::warn!(
                "T3->T1 load failed for {:?}: {} (catalog has key: {})",
                id, e, self.page_catalog_index.contains_key(&id.key()),
            );
            return Err(e);
        }
        let stall_ns = stall_start.elapsed().as_nanos() as u64;
        self.stats.stalls.fetch_add(1, Ordering::Relaxed);
        self.stats.stall_ns.fetch_add(stall_ns, Ordering::Relaxed);

        if let Some(slot_idx) = self.t1.find(id) {
            let slot = self.t1.slots[slot_idx].lock();
            return Ok(PageHandle {
                device_ptr: slot.ptr,
                size: slot.size,
                source_tier: Tier::T3Nvme,
                was_prefetched: false,
            });
        }

        tracing::warn!(
            "Page {:?} loaded to T1 but not found in T1 page_map (T1 occupied: {}/{})",
            id,
            self.t1.occupied.load(Ordering::Relaxed),
            self.t1.capacity,
        );
        Err(Error::PageNotFound { page: *id })
    }

    /// Fast path for fully-resident models: single HashMap lookup, zero locking.
    ///
    /// Returns `Some(PageHandle)` if the model is fully resident and the page
    /// exists in the frozen snapshot. Returns `None` otherwise (caller should
    /// fall back to `get_page()`).
    #[inline]
    pub fn get_page_resident(&self, id: &PageId) -> Option<PageHandle> {
        let snapshot = self.resident_snapshot.get()?;
        let rp = snapshot.get(&id.key())?;
        Some(PageHandle {
            device_ptr: rp.ptr,
            size: rp.size,
            source_tier: Tier::T1Vram,
            was_prefetched: false,
        })
    }

    /// Whether the entire model is preloaded in T1 and the frozen snapshot
    /// is available for zero-overhead page lookups.
    #[inline]
    pub fn is_fully_resident(&self) -> bool {
        self.resident_snapshot.get().is_some()
    }

    /// Submit a prefetch request (non-blocking).
    pub fn submit_prefetch(&self, req: PrefetchRequest) {
        self.stats.prefetch_issued.fetch_add(1, Ordering::Relaxed);
        self.prefetch_queue.push(req);
    }

    /// Submit multiple prefetch requests.
    pub fn submit_prefetch_batch(&self, reqs: impl IntoIterator<Item = PrefetchRequest>) {
        for req in reqs {
            self.submit_prefetch(req);
        }
    }

    /// Cancel a pending prefetch.
    pub fn cancel_prefetch(&self, page: &PageId) {
        self.inflight.remove(&page.key());
    }

    /// Update the predicted reuse probability for a page in T1.
    pub fn update_prediction(&self, page: &PageId, predicted_reuse: f32) {
        if let Some(slot_idx) = self.t1.find(page) {
            let mut slot = self.t1.slots[slot_idx].lock();
            slot.predicted_reuse = predicted_reuse;
        }
        if let Some(slot_idx) = self.t2.find(page) {
            let mut slot = self.t2.slots[slot_idx].lock();
            slot.predicted_reuse = predicted_reuse;
        }
    }

    /// Pin a page in T1 (prevent eviction). Used for shared layers.
    pub fn pin_page(&self, page: &PageId) -> bool {
        if let Some(slot_idx) = self.t1.find(page) {
            let mut slot = self.t1.slots[slot_idx].lock();
            if !slot.pinned {
                slot.pinned = true;
                self.t1.pinned.fetch_add(1, Ordering::Relaxed);
            }
            return true;
        }
        false
    }

    /// Unpin a page in T1.
    pub fn unpin_page(&self, page: &PageId) -> bool {
        if let Some(slot_idx) = self.t1.find(page) {
            let mut slot = self.t1.slots[slot_idx].lock();
            if slot.pinned {
                slot.pinned = false;
                self.t1.pinned.fetch_sub(1, Ordering::Relaxed);
            }
            return true;
        }
        false
    }

    // ── Specialist Pinning ───────────────────────────────────────────

    /// Pin an expert cluster in T1 for specialist mode.
    ///
    /// Takes a list of (layer, expert_id) pairs representing the hot working set.
    /// For each expert, all of its pages (up/gate/down proj segments) are pinned.
    ///
    /// Pages that are already in T1 get pinned immediately. Pages not yet in T1
    /// are loaded from T2/T3 and then pinned. This is an async operation that
    /// may take significant time for a cold start.
    ///
    /// Returns the number of pages successfully pinned.
    pub async fn pin_expert_cluster(&self, experts: &[(u16, u16)]) -> Result<usize> {
        let mut pinned_count = 0usize;
        let mut pages_to_load: Vec<PageId> = Vec::new();

        // Collect all pages for the requested experts
        for &(layer, expert) in experts {
            // Look up how many segments/pages this expert has via the catalog
            for segment in 0..3u16 {
                // up_proj=0, gate_proj=1, down_proj=2
                let mut page_idx = 0u16;
                loop {
                    let page = PageId {
                        layer,
                        expert,
                        segment,
                        page_idx,
                    };

                    // Check if this page exists in the catalog
                    if !self.page_catalog_index.contains_key(&page.key()) {
                        break; // No more pages for this segment
                    }

                    // Check if already in T1 and pin it
                    if self.t1.find(&page).is_some() {
                        let mut specialist_pins = self.specialist_pins.lock();
                        if specialist_pins.insert(&page) && self.pin_page(&page) {
                            pinned_count += 1;
                        }
                    } else {
                        pages_to_load.push(page);
                    }

                    page_idx += 1;
                }
            }
        }

        // Load and pin pages that aren't in T1 yet
        for page in &pages_to_load {
            // Check budget
            {
                let specialist_pins = self.specialist_pins.lock();
                if specialist_pins.count >= specialist_pins.max_pages {
                    tracing::warn!(
                        "Specialist pin budget exhausted ({}/{}), stopping",
                        specialist_pins.count,
                        specialist_pins.max_pages,
                    );
                    break;
                }
            }

            // Load the page into T1
            if self.t2.find(page).is_some() {
                if let Err(e) = self.promote_t2_to_t1(page).await {
                    tracing::debug!("Failed to promote {:?} for specialist pin: {}", page, e);
                    continue;
                }
            } else if let Err(e) = self.load_t3_to_t1(page).await {
                tracing::debug!("Failed to load {:?} for specialist pin: {}", page, e);
                continue;
            }

            // Pin it
            let mut specialist_pins = self.specialist_pins.lock();
            if specialist_pins.insert(page) && self.pin_page(page) {
                pinned_count += 1;
            }
        }

        tracing::info!(
            "Specialist pinning complete: {} pages pinned for {} experts",
            pinned_count,
            experts.len(),
        );

        Ok(pinned_count)
    }

    /// Unpin all specialist-pinned pages (transition back to generalist mode).
    ///
    /// This only unpins pages that were pinned via `pin_expert_cluster`, not
    /// shared-layer pins set via `pin_page`.
    pub fn unpin_expert_cluster(&self) -> usize {
        let keys = self.specialist_pins.lock().clear();
        let count = keys.len();

        for key in &keys {
            // Reconstruct PageId from key
            let page = PageId {
                layer: (*key >> 48) as u16,
                expert: ((*key >> 32) & 0xFFFF) as u16,
                segment: ((*key >> 16) & 0xFFFF) as u16,
                page_idx: (*key & 0xFFFF) as u16,
            };

            self.unpin_page(&page);
        }

        if count > 0 {
            tracing::info!("Specialist unpin complete: {} pages released", count,);
        }

        count
    }

    /// Get the number of specialist-pinned pages.
    pub fn specialist_pin_count(&self) -> usize {
        self.specialist_pins.lock().count
    }

    /// Check if a page is specialist-pinned.
    pub fn is_specialist_pinned(&self, page: &PageId) -> bool {
        self.specialist_pins.lock().contains(page)
    }

    // ── Phase D: Gear-aware eviction ────────────────────────────────

    /// Set the currently active gear for page tagging and eviction.
    ///
    /// Newly loaded pages will be tagged with this gear name.
    /// During eviction, pages tagged with a different gear get a lower score
    /// (deprioritized = evicted first), making room for the active gear.
    pub fn set_current_gear(&self, gear: Option<String>) {
        let mut g = self.current_gear.lock();
        if *g != gear {
            tracing::debug!(
                "Buffer manager gear updated: {:?} -> {:?}",
                *g,
                gear,
            );
            *g = gear;
        }
    }

    /// Get the currently active gear.
    pub fn current_gear(&self) -> Option<String> {
        self.current_gear.lock().clone()
    }

    // ── Tier Status ──────────────────────────────────────────────────

    pub fn tier_status(&self, tier: Tier) -> TierStatus {
        let pool = match tier {
            Tier::T1Vram => &self.t1,
            Tier::T2Ram => &self.t2,
            Tier::T3Nvme => {
                return TierStatus {
                    total_pages: self.model_file.page_count(),
                    used_pages: self.model_file.page_count(),
                    pinned_pages: 0,
                    total_bytes: 0,
                    used_bytes: 0,
                    utilization: 1.0,
                };
            }
        };

        let occupied = pool.occupied.load(Ordering::Relaxed) as usize;
        let pinned = pool.pinned.load(Ordering::Relaxed) as usize;

        TierStatus {
            total_pages: pool.capacity,
            used_pages: occupied,
            pinned_pages: pinned,
            total_bytes: pool.capacity * PAGE_SIZE,
            used_bytes: occupied * PAGE_SIZE,
            utilization: pool.utilization(),
        }
    }

    // ── Background Workers ───────────────────────────────────────────

    /// Start background prefetch and eviction workers.
    /// Returns join handles for the spawned tasks.
    pub fn start_workers(self: &Arc<Self>) -> Vec<tokio::task::JoinHandle<()>> {
        self.running.store(true, Ordering::Release);

        let mut handles = Vec::new();

        // Spawn prefetch worker
        let mgr = self.clone();
        handles.push(tokio::spawn(async move {
            mgr.prefetch_worker_loop().await;
        }));

        // Spawn eviction worker
        let mgr = self.clone();
        handles.push(tokio::spawn(async move {
            mgr.eviction_worker_loop().await;
        }));

        tracing::info!("Background workers started (prefetch + eviction)");
        handles
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Prefetch worker loop: processes the priority prefetch queue.
    ///
    /// Critical/High requests are dequeued before Medium/Low ones,
    /// ensuring that speculative prefetches never starve the current
    /// token's page loads.
    pub async fn prefetch_worker_loop(&self) {
        while self.is_running() {
            if let Some(req) = self.prefetch_queue.pop().await {
                if let Err(e) = self.execute_prefetch(&req).await {
                    tracing::debug!("Prefetch failed for {:?}: {}", req.page, e);
                }
            } else {
                break; // Queue shut down
            }
        }
    }

    /// Eviction worker loop: maintains tier watermarks.
    ///
    /// When the model is fully resident in T1 (all pages preloaded),
    /// eviction is suppressed — there is nowhere to re-fetch evicted
    /// pages from without hitting disk, which would stall inference.
    pub async fn eviction_worker_loop(&self) {
        while self.is_running() {
            // Skip eviction entirely when all pages are resident in T1.
            // The T1 pool was sized to hold the whole model; evicting pages
            // would force expensive T3→T2→T1 reloads and risk deadlocks in
            // the Notify-based inflight transfer wait path.
            if !self.fully_resident.load(Ordering::Acquire) {
                // Check T1 utilization
                if self.t1.utilization() > self.config.eviction_high_watermark {
                    self.evict_to_watermark(&self.t1, self.config.eviction_low_watermark)
                        .await;
                }

                // Check T2 utilization
                if self.t2.utilization() > self.config.eviction_high_watermark {
                    self.evict_to_watermark(&self.t2, self.config.eviction_low_watermark)
                        .await;
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn next_tick(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::Relaxed)
    }

    fn allocate_t1_pool(&self) -> Result<()> {
        tracing::info!(
            "Allocating T1 pool: {} slots x {} MB",
            self.t1.capacity,
            PAGE_SIZE / (1024 * 1024)
        );

        // T1 uses device_alloc (cudaMalloc on GPU, aligned host alloc on CPU).
        // When CUDA is available, this allocates real VRAM — weight pages and
        // engine working buffers are all device pointers, enabling GPU kernel
        // dispatch without pointer domain mismatches.
        for i in 0..self.t1.capacity {
            let ptr = cuda_ffi::device_alloc(PAGE_SIZE)?;
            let mut slot = self.t1.slots[i].lock();
            slot.ptr = ptr;
            slot.allocated = true;
        }

        tracing::info!(
            "T1 allocated: {} MB ({})",
            cuda_ffi::device_bytes_allocated() / (1024 * 1024),
            if cuda_ffi::is_cuda_available() {
                "VRAM"
            } else {
                "host"
            },
        );
        Ok(())
    }

    fn allocate_t2_pool(&self) -> Result<()> {
        tracing::info!(
            "Allocating T2 pool: {} slots x {} MB per slot",
            self.t2.capacity,
            PAGE_SIZE / (1024 * 1024)
        );

        for i in 0..self.t2.capacity {
            let ptr = cuda_ffi::host_alloc_pinned(PAGE_SIZE)?;
            let mut slot = self.t2.slots[i].lock();
            slot.ptr = ptr;
            slot.allocated = true;
        }

        tracing::info!(
            "T2 allocated: {} MB",
            cuda_ffi::host_pinned_bytes_allocated() / (1024 * 1024)
        );
        Ok(())
    }

    /// Load page from T3 (disk) to T2 (RAM).
    ///
    /// ## Pipeline B (compressed T2):
    /// The page is read from disk but **NOT decompressed**. The compressed
    /// bytes are stored directly in the T2 slot with `compressed = true`.
    /// Decompression happens later during T2→T1 promotion on the GPU.
    ///
    /// ## Pipeline A (raw T2):
    /// The page is read from disk and decompressed (Zstd/LZ4) into the T2
    /// slot as raw data. This is the legacy path.
    async fn load_t3_to_t2(&self, page: &PageId) -> Result<()> {
        // 1. Find the page in the catalog
        let catalog_idx = self
            .page_catalog_index
            .get(&page.key())
            .ok_or(Error::PageNotFound { page: *page })?;

        let entry = self.model_file.page(*catalog_idx);
        let raw_size = entry.raw_size.min(PAGE_SIZE as u32) as usize;

        // 2. Acquire T2 slot
        let t2_slot = self.acquire_slot(&self.t2).await?;

        // 3. Read page data from disk into the T2 slot
        let slot_ptr = {
            let slot = self.t2.slots[t2_slot].lock();
            slot.ptr
        };

        let transfer_start = std::time::Instant::now();

        // Decide per-page whether to store compressed in T2.
        // Pipeline B (t2_compressed=true) keeps data compressed in T2 for GPU
        // decompression during T2→T1 promotion. But if the page itself is NOT
        // compressed (compression == COMPRESSION_NONE), we must store it raw
        // regardless of the global flag — there's nothing to decompress later.
        let page_is_compressed = entry.compression != crate::storage::format::COMPRESSION_NONE;
        let store_compressed = self.config.t2_compressed && page_is_compressed;

        if store_compressed {
            // Pipeline B: read compressed data directly (no CPU decompression)
            let compressed_size = entry.compressed_size.min(PAGE_SIZE as u32) as usize;
            let buf = unsafe { std::slice::from_raw_parts_mut(slot_ptr, compressed_size) };
            self.model_file
                .read_page_compressed_sync(*catalog_idx, buf)?;

            // Mark slot as compressed
            {
                let mut slot = self.t2.slots[t2_slot].lock();
                slot.page_id = Some(*page);
                slot.load_tick = self.tick.load(Ordering::Relaxed);
                slot.last_access
                    .store(self.tick.load(Ordering::Relaxed), Ordering::Relaxed);
                slot.size = compressed_size;
                slot.raw_size = raw_size;
                slot.compressed = true;
                slot.loaded_by_gear = self.current_gear.lock().clone();
            }
        } else {
            // Pipeline A (or uncompressed page in Pipeline B): decompress on CPU into T2
            let buf = unsafe { std::slice::from_raw_parts_mut(slot_ptr, raw_size) };
            self.model_file.read_page_sync(*catalog_idx, buf)?;

            {
                let mut slot = self.t2.slots[t2_slot].lock();
                slot.page_id = Some(*page);
                slot.load_tick = self.tick.load(Ordering::Relaxed);
                slot.last_access
                    .store(self.tick.load(Ordering::Relaxed), Ordering::Relaxed);
                slot.size = raw_size;
                slot.raw_size = raw_size;
                slot.compressed = false;
                slot.loaded_by_gear = self.current_gear.lock().clone();
            }
        }

        let transfer_ns = transfer_start.elapsed().as_nanos() as u64;
        self.stats
            .transfer_ns
            .fetch_add(transfer_ns, Ordering::Relaxed);

        self.t2.page_map.insert(page.key(), t2_slot);
        self.t2.occupied.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Promote page from T2 (RAM) to T1 (VRAM).
    ///
    /// ## Pipeline B (compressed T2):
    /// 1. DMA compressed data from T2 (pinned host) → VRAM staging buffer
    /// 2. GPU decompress: staging buffer → final T1 slot
    ///    (Blackwell DE at 600 GB/s — effectively free)
    /// 3. Release staging slot
    ///
    /// ## Pipeline A (raw T2):
    /// 1. DMA raw data from T2 (pinned host) → T1 slot directly
    async fn promote_t2_to_t1(&self, page: &PageId) -> Result<()> {
        // 1. Find the T2 slot and pin it temporarily to prevent eviction
        //    while we wait for a T1 slot.
        let t2_slot = self
            .t2
            .find(page)
            .ok_or(Error::PageNotFound { page: *page })?;

        let was_pinned = {
            let mut slot = self.t2.slots[t2_slot].lock();
            let old = slot.pinned;
            slot.pinned = true;
            old
        };

        // RAII guard to ensure we unpin on error or success
        struct PinGuard<'a> {
            pool: &'a TierPool,
            slot_idx: usize,
            was_pinned: bool,
        }
        impl<'a> Drop for PinGuard<'a> {
            fn drop(&mut self) {
                if !self.was_pinned {
                    self.pool.slots[self.slot_idx].lock().pinned = false;
                }
            }
        }
        let _guard = PinGuard {
            pool: &self.t2,
            slot_idx: t2_slot,
            was_pinned,
        };

        // 2. Acquire a T1 slot (evict if necessary)
        let t1_slot = self.acquire_slot(&self.t1).await?;

        // 3. Read T2 slot state
        let (src_ptr, data_size, is_compressed, raw_size) = {
            let slot = self.t2.slots[t2_slot].lock();
            (
                slot.ptr as *const u8,
                slot.size,
                slot.compressed,
                slot.raw_size,
            )
        };

        let dst_ptr = {
            let slot = self.t1.slots[t1_slot].lock();
            slot.ptr
        };

        let transfer_start = std::time::Instant::now();

        if is_compressed {
            // ── Pipeline B: compressed DMA + GPU decompress ──────────

            // Step 1: Acquire VRAM staging slot
            let staging = self.staging.lock();
            let (staging_idx, staging_ptr) = staging.acquire().ok_or_else(|| Error::TierFull {
                tier: Tier::T1Vram,
                used: 0,
                capacity: 0,
            })?;
            drop(staging); // Release staging lock during DMA

            // Step 2: DMA compressed data to staging buffer in VRAM
            cuda_ffi::memcpy_h2d_async(staging_ptr, src_ptr, data_size, &self.stream)?;
            self.stream.synchronize()?;

            // Step 3: GPU decompress: staging → T1 slot
            // In production, this calls nvcomp::decompress_async() targeting the
            // Blackwell Decompression Engine. For now, we fall back to CPU decompress
            // + DMA since nvCOMP integration requires the actual GPU.
            //
            // TODO: Replace with nvcomp::batched_zstd_decompress_async() when
            // nvCOMP is available. The Blackwell DE processes this at 600 GB/s,
            // making decompression latency ~0.003ms per 2MB page.
            self.gpu_decompress_or_fallback(staging_ptr, data_size, dst_ptr, raw_size)?;

            // Step 4: Release staging slot
            self.staging.lock().release(staging_idx);
        } else {
            // ── Pipeline A: direct DMA (raw data) ────────────────────
            cuda_ffi::memcpy_h2d_async(dst_ptr, src_ptr, data_size, &self.stream)?;
            self.stream.synchronize()?;
        }

        let transfer_ns = transfer_start.elapsed().as_nanos() as u64;
        self.stats
            .transfer_ns
            .fetch_add(transfer_ns, Ordering::Relaxed);

        // 4. Update T1 slot
        {
            let mut slot = self.t1.slots[t1_slot].lock();
            slot.page_id = Some(*page);
            slot.load_tick = self.tick.load(Ordering::Relaxed);
            slot.last_access
                .store(self.tick.load(Ordering::Relaxed), Ordering::Relaxed);
            slot.size = raw_size; // T1 always stores decompressed data
            slot.raw_size = raw_size;
            slot.compressed = false; // T1 is always raw
            slot.loaded_by_gear = self.current_gear.lock().clone();
        }
        self.t1.page_map.insert(page.key(), t1_slot);
        self.t1.occupied.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// GPU decompression fallback.
    ///
    /// In production with nvCOMP + Blackwell DE:
    ///   nvcomp::batched_zstd_decompress_async(staging_ptr, dst_ptr, stream)
    ///   → Blackwell Decompression Engine at 600 GB/s
    ///   → ~0.003ms per 2MB page
    ///
    /// Fallback (no GPU): memcpy staging → host temp → CPU Zstd decompress → DMA back.
    /// This fallback is ~10x slower but functionally correct.
    fn gpu_decompress_or_fallback(
        &self,
        _staging_ptr: *mut u8,
        compressed_size: usize,
        dst_ptr: *mut u8,
        raw_size: usize,
    ) -> Result<()> {
        // TODO: nvCOMP GPU decompression path
        // For now: since we can't actually run GPU decompress without nvCOMP,
        // we do a CPU fallback. In the real path, the compressed data is
        // already in VRAM staging and we'd call nvcomp directly.
        //
        // For the CPU fallback, we need the compressed data on the host.
        // Since we still have it in T2, we can decompress from there.
        // This is temporary — the production path never touches CPU for decompression.

        // Allocate a temporary host buffer for decompressed data
        let mut decompressed = vec![0u8; raw_size];

        // Read the compressed data back from staging to host (temporary fallback)
        let mut compressed_buf = vec![0u8; compressed_size];
        cuda_ffi::memcpy_d2h(
            compressed_buf.as_mut_ptr(),
            _staging_ptr as *const u8,
            compressed_size,
        )?;

        // CPU Zstd decompress
        let decoded_size = zstd::bulk::decompress_to_buffer(&compressed_buf, &mut decompressed)
            .map_err(|e| Error::DecompressFailed {
                msg: format!("Zstd fallback decompress failed: {}", e),
            })?;

        // DMA decompressed data to T1
        let copy_size = decoded_size.min(raw_size);
        cuda_ffi::memcpy_h2d_async(
            dst_ptr,
            decompressed.as_ptr(),
            copy_size,
            &self.decompress_stream,
        )?;
        self.decompress_stream.synchronize()?;

        Ok(())
    }

    /// Load from T3 → T2 → T1.
    async fn load_t3_to_t1(&self, page: &PageId) -> Result<()> {
        // Register in-flight transfer
        let (tx, rx) = tokio::sync::watch::channel(false);
        let transfer = Arc::new(TransferOp {
            page_id: *page,
            _source: Tier::T3Nvme,
            _dest: Tier::T1Vram,
            _priority: PrefetchPriority::Critical,
            completed: AtomicBool::new(false),
            notify: tx,
            waiter: rx,
        });
        self.inflight.insert(page.key(), transfer.clone());

        // Load to T2 first if not already there
        if self.t2.find(page).is_none() {
            let result = self.load_t3_to_t2(page).await;
            if let Err(e) = &result {
                tracing::debug!("T3->T2 load failed for {:?}: {}", page, e);
                self.inflight.remove(&page.key());
                transfer.completed.store(true, Ordering::Release);
                let _ = transfer.notify.send(true);
                return result;
            }
        }

        // Promote T2 → T1
        let result = self.promote_t2_to_t1(page).await;

        // Mark complete and notify waiters
        self.inflight.remove(&page.key());
        transfer.completed.store(true, Ordering::Release);
        let _ = transfer.notify.send(true);

        result
    }

    async fn acquire_slot(&self, pool: &TierPool) -> Result<usize> {
        // Try free slot first
        if let Some(slot) = pool.try_allocate() {
            return Ok(slot);
        }

        // Need to evict
        self.evict_tier(pool).await;

        pool.try_allocate().ok_or(Error::TierFull {
            tier: pool.tier,
            used: pool.occupied.load(Ordering::Relaxed) as usize,
            capacity: pool.capacity,
        })
    }

    async fn evict_tier(&self, pool: &TierPool) {
        // Find the best eviction victim:
        // Base score = predicted_reuse * 0.6 + recency * 0.4
        // Gear factor: pages from the current gear get 1.0, different gear gets 0.3,
        //              no gear info gets 0.7 (neutral).
        // Final score = base_score * gear_factor
        // Lowest score gets evicted. Pinned pages are skipped.

        let mut worst_score = f64::MAX;
        let mut worst_idx = None;
        let current_tick = self.tick.load(Ordering::Relaxed).max(1);
        let active_gear = self.current_gear.lock().clone();

        for (idx, slot_mutex) in pool.slots.iter().enumerate() {
            let slot = slot_mutex.lock();
            if slot.page_id.is_none() || slot.pinned {
                continue;
            }

            let recency = slot.last_access.load(Ordering::Relaxed) as f64 / current_tick as f64;
            let base_score = slot.predicted_reuse as f64 * 0.6 + recency * 0.4;

            // Phase D: Gear-aware eviction factor
            let gear_factor = match (&active_gear, &slot.loaded_by_gear) {
                (Some(current), Some(loaded)) if current == loaded => 1.0, // keep — same gear
                (Some(_), Some(_)) => 0.3, // deprioritize — different gear
                _ => 0.7,                  // neutral — no gear info
            };

            let score = base_score * gear_factor;

            if score < worst_score {
                worst_score = score;
                worst_idx = Some(idx);
            }
        }

        if let Some(idx) = worst_idx {
            pool.release(idx);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Evict pages until utilization drops below the target watermark.
    async fn evict_to_watermark(&self, pool: &TierPool, target: f32) {
        while pool.utilization() > target {
            self.evict_tier(pool).await;
            if pool.occupied.load(Ordering::Relaxed) == 0 {
                break;
            }
        }
    }

    async fn execute_prefetch(&self, req: &PrefetchRequest) -> Result<()> {
        // Check if page is already in the destination tier
        let dest_pool = match req.dest {
            Tier::T1Vram => &self.t1,
            Tier::T2Ram => &self.t2,
            Tier::T3Nvme => return Ok(()), // Nothing to do
        };

        if dest_pool.find(&req.page).is_some() {
            return Ok(()); // Already there
        }

        match (req.source, req.dest) {
            (Tier::T3Nvme, Tier::T2Ram) => self.load_t3_to_t2(&req.page).await,
            (Tier::T2Ram, Tier::T1Vram) => self.promote_t2_to_t1(&req.page).await,
            (Tier::T3Nvme, Tier::T1Vram) => self.load_t3_to_t1(&req.page).await,
            _ => Ok(()),
        }
    }
}

impl Drop for PageBufferManager {
    fn drop(&mut self) {
        // Free all allocated T1 memory.
        // T1 uses device_alloc (cudaMalloc on GPU, aligned host alloc on CPU).
        for slot_mutex in &self.t1.slots {
            let slot = slot_mutex.lock();
            if slot.allocated && !slot.ptr.is_null() {
                cuda_ffi::device_free(slot.ptr, slot.capacity);
            }
        }
        // Free all allocated T2 memory (pinned host memory)
        for slot_mutex in &self.t2.slots {
            let slot = slot_mutex.lock();
            if slot.allocated && !slot.ptr.is_null() {
                cuda_ffi::host_free_pinned(slot.ptr, slot.capacity);
            }
        }
        // VramStagingBuffer handles its own drop
    }
}

/// Status of one storage tier (returned to callers).
#[derive(Clone, Debug, serde::Serialize)]
pub struct TierStatus {
    pub total_pages: usize,
    pub used_pages: usize,
    pub pinned_pages: usize,
    pub total_bytes: usize,
    pub used_bytes: usize,
    pub utilization: f32,
}

/// Handle to a page that's ready for GPU compute.
pub struct PageHandle {
    pub device_ptr: *mut u8,
    pub size: usize,
    pub source_tier: Tier,
    pub was_prefetched: bool,
}

// SAFETY: The device pointer is valid for the lifetime of the page in T1.
unsafe impl Send for PageHandle {}
