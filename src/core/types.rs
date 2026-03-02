//! Foundational types for the vib3 storage engine.
//!
//! These types define the atomic units of the system: pages, experts,
//! tiers, and the metadata that connects them. Every other module
//! builds on these primitives.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

// ─── Constants ───────────────────────────────────────────────────────────

/// Page size: 2 MB, aligned to Linux hugepages and GPU allocation granularity.
pub const PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Minimum alignment for DMA transfers.
pub const PAGE_ALIGNMENT: usize = 4096;

/// Maximum experts per layer.
pub const MAX_EXPERTS: usize = 512;

/// Maximum layers in a model.
pub const MAX_LAYERS: usize = 128;

/// Maximum experts active per token.
pub const MAX_ACTIVE_EXPERTS: usize = 16;

// ─── Kimi K2.5 defaults ─────────────────────────────────────────────────

pub mod kimi_k25 {
    pub const NUM_EXPERTS: u32 = 384;
    pub const NUM_ACTIVE: u32 = 8;
    pub const NUM_SHARED_EXPERTS: u32 = 1;
    pub const NUM_LAYERS: u32 = 61;
    pub const NUM_MOE_LAYERS: u32 = 60;
    pub const DENSE_LAYERS: u32 = 1; // first_k_dense_replace = 1
    pub const HIDDEN_DIM: u32 = 7168;
    pub const EXPERT_HIDDEN_DIM: u32 = 2048; // moe_intermediate_size
    pub const SHARED_INTERMEDIATE_SIZE: u32 = 2048; // moe_intermediate_size * n_shared_experts
    pub const NUM_HEADS: u32 = 64;
    pub const NUM_KV_HEADS: u32 = 64; // With MLA, same as num_heads
    pub const VOCAB_SIZE: u32 = 163_840;
    pub const MAX_SEQ_LEN: u32 = 262_144;

    // MLA (Multi-head Latent Attention) config
    pub const KV_LORA_RANK: u32 = 512;
    pub const Q_LORA_RANK: u32 = 1536;
    pub const QK_ROPE_HEAD_DIM: u32 = 64;
    pub const QK_NOPE_HEAD_DIM: u32 = 128;
    pub const V_HEAD_DIM: u32 = 128;

    // RoPE config (YaRN scaling)
    pub const ROPE_THETA: f64 = 50_000.0;
    pub const ROPE_SCALING_FACTOR: f64 = 64.0;
    pub const ROPE_BETA_FAST: f64 = 32.0;
    pub const ROPE_BETA_SLOW: f64 = 1.0;
    pub const ROPE_ORIGINAL_MAX_POS: u32 = 4096;
    pub const MSCALE_ALL_DIM: f64 = 1.0;

    // Router config
    pub const ROUTED_SCALING_FACTOR: f64 = 2.827;
    pub const SCORING_FUNC: &str = "sigmoid"; // sigmoid gating, not softmax

    // Quantization config
    pub const QUANT_GROUP_SIZE: u32 = 32;
    pub const QUANT_BITS: u32 = 4;
    pub const QUANT_SYMMETRIC: bool = true;
}

// ─── Qwen3.5-122B-A10B defaults ────────────────────────────────────────

pub mod qwen35_122b {
    // Core dimensions
    pub const HIDDEN_DIM: u32 = 3072;
    pub const NUM_LAYERS: u32 = 48;
    pub const VOCAB_SIZE: u32 = 248_320;
    pub const MAX_SEQ_LEN: u32 = 262_144;

    // Full attention layers (every 4th layer: 3, 7, 11, ..., 47)
    pub const FULL_ATTN_INTERVAL: u32 = 4;
    pub const NUM_ATTN_LAYERS: u32 = 12; // 48 / 4
    pub const NUM_DELTANET_LAYERS: u32 = 36; // 48 - 12
    pub const NUM_ATTN_HEADS: u32 = 32;
    pub const NUM_KV_HEADS: u32 = 2;
    pub const HEAD_DIM: u32 = 256;
    pub const PARTIAL_ROTARY_FACTOR: f64 = 0.25; // 64 of 256 dims get RoPE
    pub const ROPE_THETA: f64 = 10_000_000.0;

    // DeltaNet (linear attention) config — 36 of 48 layers
    pub const DELTANET_NUM_KEY_HEADS: u32 = 16;
    pub const DELTANET_NUM_VALUE_HEADS: u32 = 64;
    pub const DELTANET_KEY_HEAD_DIM: u32 = 128;
    pub const DELTANET_VALUE_HEAD_DIM: u32 = 128;
    pub const DELTANET_CONV_KERNEL: u32 = 4;
    pub const DELTANET_INNER_DIM: u32 = 8192; // 64 * 128

    // MoE config — every layer (both DeltaNet and attention)
    pub const NUM_EXPERTS: u32 = 256;
    pub const NUM_ACTIVE_EXPERTS: u32 = 8;
    pub const NUM_SHARED_EXPERTS: u32 = 1;
    pub const EXPERT_INTERMEDIATE_SIZE: u32 = 1024;
    pub const SHARED_EXPERT_INTERMEDIATE_SIZE: u32 = 1024;

    // Router config
    pub const SCORING_FUNC: &str = "softmax";

    // Norm config
    pub const RMS_NORM_EPS: f64 = 1e-6;

    // IMRoPE (Interleaved Multi-RoPE) dimension sections
    pub const MROPE_SECTIONS: [u32; 3] = [11, 11, 10]; // temporal, height, width

    /// Returns true if layer `i` is a full attention layer.
    pub const fn is_attention_layer(layer_idx: u32) -> bool {
        (layer_idx + 1) % FULL_ATTN_INTERVAL == 0
    }
}

// ─── Qwen3.5-35B-A3B defaults ─────────────────────────────────────────
pub mod qwen35_35b {
    // Core dimensions
    pub const HIDDEN_DIM: u32 = 2048;
    pub const NUM_LAYERS: u32 = 40;
    pub const VOCAB_SIZE: u32 = 248_320;
    pub const MAX_SEQ_LEN: u32 = 262_144;

    // Full attention layers (every 4th layer: 3, 7, 11, ..., 39)
    pub const FULL_ATTN_INTERVAL: u32 = 4;
    pub const NUM_ATTN_LAYERS: u32 = 10; // 40 / 4
    pub const NUM_DELTANET_LAYERS: u32 = 30; // 40 - 10
    pub const NUM_ATTN_HEADS: u32 = 16;
    pub const NUM_KV_HEADS: u32 = 2;
    pub const HEAD_DIM: u32 = 256;
    pub const PARTIAL_ROTARY_FACTOR: f64 = 0.25; // 64 of 256 dims get RoPE
    pub const ROPE_THETA: f64 = 10_000_000.0;

    // DeltaNet (linear attention) config — 30 of 40 layers
    pub const DELTANET_NUM_KEY_HEADS: u32 = 16;
    pub const DELTANET_NUM_VALUE_HEADS: u32 = 32;
    pub const DELTANET_KEY_HEAD_DIM: u32 = 128;
    pub const DELTANET_VALUE_HEAD_DIM: u32 = 128;
    pub const DELTANET_CONV_KERNEL: u32 = 4;
    pub const DELTANET_INNER_DIM: u32 = 4096; // 32 * 128

    // MoE config — every layer (both DeltaNet and attention)
    pub const NUM_EXPERTS: u32 = 256;
    pub const NUM_ACTIVE_EXPERTS: u32 = 8;
    pub const NUM_SHARED_EXPERTS: u32 = 1;
    pub const EXPERT_INTERMEDIATE_SIZE: u32 = 512;
    pub const SHARED_EXPERT_INTERMEDIATE_SIZE: u32 = 512;

    // Router config
    pub const SCORING_FUNC: &str = "softmax";

    // Norm config
    pub const RMS_NORM_EPS: f64 = 1e-6;

    // IMRoPE (Interleaved Multi-RoPE) dimension sections
    pub const MROPE_SECTIONS: [u32; 3] = [11, 11, 10]; // temporal, height, width

    /// Returns true if layer `i` is a full attention layer.
    pub const fn is_attention_layer(layer_idx: u32) -> bool {
        (layer_idx + 1) % FULL_ATTN_INTERVAL == 0
    }
}

// ─── Page Identity ───────────────────────────────────────────────────────

/// Sentinel value for shared weight pages (attention, embeddings, norms).
pub const EXPERT_SHARED: u16 = 0xFFFF;

/// Sentinel value for KV cache pages (K and V vectors for attention).
pub const EXPERT_KV_CACHE: u16 = 0xFFFE;

/// KV cache segment: K vectors.
pub const KV_SEGMENT_K: u16 = 0;

/// KV cache segment: V vectors.
pub const KV_SEGMENT_V: u16 = 1;

/// Uniquely identifies a page in the unified storage engine.
///
/// A page is a 2 MB region of memory used for either:
/// - **Weight data**: a slice of a weight matrix (expert or shared)
/// - **KV cache data**: a block of K or V vectors for attention
///
/// The `expert` field distinguishes the page type:
/// - `0x0000..0xFFFD` → Expert weight page (layer, expert, segment, page_idx)
/// - `0xFFFF` → Shared weight page (attention, embeddings, norms)
/// - `0xFFFE` → KV cache page (layer, K/V segment, position block index)
///
/// For KV cache pages:
/// - `segment` = 0 for K vectors, 1 for V vectors
/// - `page_idx` = position block index (each page holds multiple KV positions)
///
/// This unification is the core insight: weight pages and KV cache pages
/// share the same lifecycle (tiered storage, predictive prefetch, eviction)
/// and can be managed by a single buffer pool.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct PageId {
    pub layer: u16,
    pub expert: u16,   // 0xFFFF = shared, 0xFFFE = KV cache, else expert ID
    pub segment: u16,  // weights: 0=up, 1=gate, 2=down; KV: 0=K, 1=V
    pub page_idx: u16, // weights: page within segment; KV: position block index
}

impl PageId {
    /// Create a PageId for a shared (non-expert) weight page.
    pub const fn shared(layer: u16, segment: u16, page_idx: u16) -> Self {
        Self {
            layer,
            expert: EXPERT_SHARED,
            segment,
            page_idx,
        }
    }

    /// Create a PageId for a KV cache page.
    ///
    /// - `layer`: transformer layer index
    /// - `kv_segment`: `KV_SEGMENT_K` (0) or `KV_SEGMENT_V` (1)
    /// - `block_idx`: position block index within the layer
    pub const fn kv_cache(layer: u16, kv_segment: u16, block_idx: u16) -> Self {
        Self {
            layer,
            expert: EXPERT_KV_CACHE,
            segment: kv_segment,
            page_idx: block_idx,
        }
    }

    /// Whether this page belongs to shared layers (not an expert or KV).
    pub const fn is_shared(&self) -> bool {
        self.expert == EXPERT_SHARED
    }

    /// Whether this page is a KV cache page.
    pub const fn is_kv_cache(&self) -> bool {
        self.expert == EXPERT_KV_CACHE
    }

    /// Whether this page is a weight page (expert or shared, not KV cache).
    pub const fn is_weight(&self) -> bool {
        self.expert != EXPERT_KV_CACHE
    }

    /// Whether this is a K (key) vector page. Only meaningful for KV cache pages.
    pub const fn is_k_page(&self) -> bool {
        self.expert == EXPERT_KV_CACHE && self.segment == KV_SEGMENT_K
    }

    /// Whether this is a V (value) vector page. Only meaningful for KV cache pages.
    pub const fn is_v_page(&self) -> bool {
        self.expert == EXPERT_KV_CACHE && self.segment == KV_SEGMENT_V
    }

    /// Pack into a u64 for use as a hash map key.
    pub const fn key(&self) -> u64 {
        (self.layer as u64) << 48
            | (self.expert as u64) << 32
            | (self.segment as u64) << 16
            | (self.page_idx as u64)
    }
}

impl fmt::Debug for PageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_kv_cache() {
            let kv_type = if self.segment == KV_SEGMENT_K {
                "K"
            } else {
                "V"
            };
            write!(
                f,
                "Page(L{}/KV:{}/blk{})",
                self.layer, kv_type, self.page_idx
            )
        } else if self.is_shared() {
            write!(
                f,
                "Page(L{}/shared/s{}/p{})",
                self.layer, self.segment, self.page_idx
            )
        } else {
            write!(
                f,
                "Page(L{}/E{}/s{}/p{})",
                self.layer, self.expert, self.segment, self.page_idx
            )
        }
    }
}

impl fmt::Display for PageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ─── KV Cache Page Geometry ──────────────────────────────────────────────

/// Number of KV positions stored per page.
///
/// Each position stores one K or V vector (head_dim floats = head_dim * 4 bytes).
///
/// For standard multi-head attention, head_dim = hidden_dim / num_heads.
/// For Kimi K2.5 with MLA, the KV cache stores compressed latents:
///   kv_lora_rank = 512 per layer (shared across all heads)
///   Per position: 512 * 4 = 2,048 bytes (latent vector)
///   Per page (2MB): 2,097,152 / 2,048 = 1,024 positions per page
///
/// For models without MLA (standard GQA), head_dim = hidden_dim / num_heads:
///   e.g., head_dim = 112 (7168/64): 112 * 4 = 448 bytes → 4,681 positions/page
///   e.g., head_dim = 128: 128 * 4 = 512 bytes → 4,096 positions/page
///
/// This default (4096) assumes head_dim=128 for non-MLA models.
/// MLA models should use kv_lora_rank-based geometry instead.
pub const KV_POSITIONS_PER_PAGE_DEFAULT: usize = 4096;

/// Maximum position blocks per layer (limits u16 page_idx range).
/// At 4096 positions/block, this supports up to 4096 * 65535 = ~268M positions,
/// well beyond any practical context length.
pub const MAX_KV_BLOCKS: usize = 65535;

// ─── Expert Identity ─────────────────────────────────────────────────────

/// Identifies a specific expert within the model.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExpertId {
    pub layer: u16,
    pub expert: u16,
}

impl ExpertId {
    pub const fn key(&self) -> u32 {
        (self.layer as u32) << 16 | (self.expert as u32)
    }
}

// ─── Storage Tiers ───────────────────────────────────────────────────────

/// The three tiers of the storage hierarchy.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Tier {
    /// GPU VRAM (GDDR7 on RTX PRO 6000) — hot compute tier. ~1,792 GB/s internal bandwidth.
    T1Vram = 0,
    /// System RAM — warm buffer pool. ~64 GB/s to GPU via PCIe 5.0.
    T2Ram = 1,
    /// NVMe SSD — cold store. ~12-50 GB/s depending on array config.
    T3Nvme = 2,
}

impl Tier {
    pub const fn name(&self) -> &'static str {
        match self {
            Tier::T1Vram => "T1:VRAM",
            Tier::T2Ram => "T2:RAM",
            Tier::T3Nvme => "T3:NVMe",
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ─── Page State ──────────────────────────────────────────────────────────

/// Lifecycle state of a page in the storage hierarchy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PageState {
    /// On disk only (T3).
    Cold = 0,
    /// Transfer in flight (T3→T2 or T2→T1).
    Loading = 1,
    /// In RAM (T2), ready for promotion to VRAM.
    Warm = 2,
    /// Transfer in flight (T2→T1).
    Promoting = 3,
    /// In VRAM (T1), ready for compute.
    Hot = 4,
    /// In VRAM (T1), pinned — will never be evicted.
    Pinned = 5,
    /// Being demoted (T1→T2 or T2→T3).
    Evicting = 6,
}

// ─── Page Table Entry ────────────────────────────────────────────────────

/// Tracks the current location and status of a single page across all tiers.
///
/// This is the core bookkeeping structure. The buffer manager maintains one
/// of these for every page in the model. It must be safe to read from
/// multiple threads concurrently (the atomic fields handle this).
pub struct PageTableEntry {
    pub id: PageId,

    /// Current state (atomic for lock-free reads on the hot path).
    state: AtomicU8,

    /// Pointer to page data in T1 (VRAM). `None` if not in T1.
    /// This is a raw device pointer — only valid for CUDA kernel launches.
    pub t1_ptr: Option<DevicePtr>,

    /// Pointer to page data in T2 (RAM). `None` if not in T2.
    /// This is pinned host memory for zero-copy DMA.
    pub t2_ptr: Option<HostPtr>,

    /// Byte offset in the .vib3 file on disk (T3).
    pub t3_offset: u64,

    /// Actual data size in bytes (may be < PAGE_SIZE).
    pub size_bytes: u32,

    /// Weight matrix row range covered by this page.
    pub row_start: u16,
    pub row_count: u16,

    // Access tracking (atomics for contention-free updates)
    access_count: AtomicU64,
    last_access_tick: AtomicU64,

    /// Predicted probability of reuse (updated by query planner).
    pub predicted_reuse: f32,
}

impl PageTableEntry {
    pub fn new(id: PageId, t3_offset: u64, size_bytes: u32) -> Self {
        Self {
            id,
            state: AtomicU8::new(PageState::Cold as u8),
            t1_ptr: None,
            t2_ptr: None,
            t3_offset,
            size_bytes,
            row_start: 0,
            row_count: 0,
            access_count: AtomicU64::new(0),
            last_access_tick: AtomicU64::new(0),
            predicted_reuse: 0.0,
        }
    }

    pub fn state(&self) -> PageState {
        // SAFETY: We only store valid PageState discriminants
        unsafe { std::mem::transmute(self.state.load(Ordering::Acquire)) }
    }

    pub fn set_state(&self, state: PageState) {
        self.state.store(state as u8, Ordering::Release);
    }

    pub fn record_access(&self, tick: u64) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access_tick.store(tick, Ordering::Relaxed);
    }

    pub fn access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }

    pub fn last_access_tick(&self) -> u64 {
        self.last_access_tick.load(Ordering::Relaxed)
    }

    /// Is this page currently available for GPU compute?
    pub fn is_compute_ready(&self) -> bool {
        matches!(self.state(), PageState::Hot | PageState::Pinned)
    }

    /// Current tier based on state.
    pub fn current_tier(&self) -> Option<Tier> {
        match self.state() {
            PageState::Hot | PageState::Pinned => Some(Tier::T1Vram),
            PageState::Warm => Some(Tier::T2Ram),
            PageState::Cold => Some(Tier::T3Nvme),
            _ => None, // In transit
        }
    }
}

// ─── Pointer Wrappers ────────────────────────────────────────────────────

/// Opaque handle to a GPU device pointer (VRAM).
///
/// This is deliberately not `Copy` — moving device pointers around
/// carelessly is how you get use-after-free in GPU memory.
#[derive(Debug)]
pub struct DevicePtr {
    ptr: *mut u8,
    size: usize,
}

impl DevicePtr {
    /// # Safety
    /// `ptr` must be a valid CUDA device pointer allocated via cudaMalloc.
    pub unsafe fn new(ptr: *mut u8, size: usize) -> Self {
        Self { ptr, size }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

// SAFETY: Device pointers can be sent across threads — CUDA operations
// are thread-safe when using the same context.
unsafe impl Send for DevicePtr {}
unsafe impl Sync for DevicePtr {}

/// Opaque handle to pinned host memory (for DMA to GPU).
#[derive(Debug)]
pub struct HostPtr {
    ptr: *mut u8,
    size: usize,
}

impl HostPtr {
    /// # Safety
    /// `ptr` must be a valid pointer to pinned memory (cudaMallocHost or mlock'd).
    pub unsafe fn new(ptr: *mut u8, size: usize) -> Self {
        Self { ptr, size }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: The pointer is valid and the size is correct by construction.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: We have exclusive access via &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

unsafe impl Send for HostPtr {}
unsafe impl Sync for HostPtr {}

// ─── Expert Activation ──────────────────────────────────────────────────

/// Output of the MoE router for one token at one layer.
///
/// Contains which experts were selected and their routing weights.
#[derive(Clone, Debug)]
pub struct ExpertActivation {
    /// (expert_id, routing_weight) pairs, sorted by weight descending.
    pub experts: Vec<(u16, f32)>,
}

impl ExpertActivation {
    pub fn new() -> Self {
        Self {
            experts: Vec::with_capacity(MAX_ACTIVE_EXPERTS),
        }
    }

    pub fn count(&self) -> usize {
        self.experts.len()
    }

    pub fn expert_ids(&self) -> impl Iterator<Item = u16> + '_ {
        self.experts.iter().map(|(id, _)| *id)
    }

    pub fn contains_expert(&self, expert_id: u16) -> bool {
        self.experts.iter().any(|(id, _)| *id == expert_id)
    }
}

impl Default for ExpertActivation {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Prefetch Request ────────────────────────────────────────────────────

/// Priority level for prefetch requests.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum PrefetchPriority {
    /// Needed for current token — stall if not ready.
    Critical = 0,
    /// Needed for next token — hide behind current compute.
    High = 1,
    /// Predicted for near-future tokens.
    Medium = 2,
    /// Speculative / domain-level prefetch.
    Low = 3,
}

/// A request to move a page between storage tiers.
#[derive(Clone, Debug)]
pub struct PrefetchRequest {
    pub page: PageId,
    pub source: Tier,
    pub dest: Tier,
    pub priority: PrefetchPriority,
    pub deadline_tick: u64,
    pub confidence: f32,
}

impl PrefetchRequest {
    /// Ordering for the priority queue: lower priority value = higher urgency.
    pub fn urgency_key(&self) -> (u8, u64) {
        (self.priority as u8, self.deadline_tick)
    }
}

// ─── Data Types ──────────────────────────────────────────────────────────

/// Weight data types supported by vib3.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum DType {
    FP16 = 0,
    BF16 = 1,
    FP8E4 = 2,
    FP8E5 = 3,
    INT8 = 4,
    INT4 = 5,
    NF4 = 6,
    NVFP4 = 7,
}

impl DType {
    /// Bits per element.
    pub const fn bits(&self) -> usize {
        match self {
            DType::INT4 | DType::NF4 | DType::NVFP4 => 4,
            DType::FP8E4 | DType::FP8E5 | DType::INT8 => 8,
            DType::FP16 | DType::BF16 => 16,
        }
    }

    /// Bytes for a given number of elements.
    pub const fn bytes_for(&self, elements: usize) -> usize {
        (elements * self.bits()).div_ceil(8)
    }
}

// ─── Activation Mode ─────────────────────────────────────────────────────

/// Expert activation mode — detected at runtime based on workload pattern.
///
/// The engine continuously monitors expert activation entropy and switches
/// between modes to optimize cache strategy and prefetch behavior.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum ActivationMode {
    /// Surface-level skimming: diverse expert activation, low locality.
    /// Working set is large (many experts touched infrequently).
    /// Strategy: aggressive prefetch, wider prediction, accept lower hit rate.
    /// Expected T1 hit rate: 60-70%.
    Generalist = 0,

    /// Deep domain focus: concentrated expert activation, high locality.
    /// Working set is small (~40-60 experts across layers, reused heavily).
    /// Strategy: pin hot expert cluster in T1, suppress eviction, hold steady.
    /// Expected T1 hit rate: 95%+.
    Specialist = 1,
}

impl ActivationMode {
    pub const fn name(&self) -> &'static str {
        match self {
            ActivationMode::Generalist => "generalist",
            ActivationMode::Specialist => "specialist",
        }
    }
}

impl fmt::Display for ActivationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A specialist profile: a pre-computed mapping from a domain to a set of
/// hot experts that should be pinned in T1.
///
/// These are the "materialized views" of the storage engine — pre-computed
/// working sets for known workload patterns.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpecialistProfile {
    /// Human-readable name (e.g., "low-level-systems", "mathematics", "creative-writing").
    pub name: String,

    /// Domain centroid embedding (used for matching).
    pub centroid: Vec<f32>,

    /// Per-layer hot expert IDs. `hot_experts[layer_idx]` = list of expert IDs to pin.
    /// Only MoE layers are included (indexed from 0).
    pub hot_experts: Vec<Vec<u16>>,

    /// Total number of unique pages in this profile's working set.
    pub total_pages: usize,

    /// Estimated T1 VRAM required to pin this entire profile (bytes).
    pub vram_required: usize,
}

impl SpecialistProfile {
    /// Count unique experts across all layers.
    pub fn unique_expert_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for layer_experts in &self.hot_experts {
            for &expert in layer_experts {
                seen.insert(expert);
            }
        }
        seen.len()
    }

    /// Check if a specific expert is hot in this profile at a given layer.
    pub fn is_hot(&self, layer: usize, expert: u16) -> bool {
        self.hot_experts
            .get(layer)
            .map(|experts| experts.contains(&expert))
            .unwrap_or(false)
    }
}

// ─── Statistics ──────────────────────────────────────────────────────────

/// Runtime inference statistics, updated atomically.
#[derive(Default)]
pub struct InferenceStats {
    pub tokens_generated: AtomicU64,
    pub total_page_accesses: AtomicU64,
    pub t1_hits: AtomicU64,
    pub t2_hits: AtomicU64,
    pub t3_hits: AtomicU64,
    pub prefetch_issued: AtomicU64,
    pub prefetch_useful: AtomicU64,
    pub prefetch_wasted: AtomicU64,
    pub evictions: AtomicU64,
    pub stalls: AtomicU64,
    pub compute_ns: AtomicU64,
    pub transfer_ns: AtomicU64,
    pub stall_ns: AtomicU64,

    // ── KV cache statistics ──────────────────────────────────────────
    /// Total KV page accesses (separate from weight page accesses).
    pub kv_page_accesses: AtomicU64,
    /// KV pages found in T1 (VRAM).
    pub kv_t1_hits: AtomicU64,
    /// KV pages found in T2 (RAM) and promoted.
    pub kv_t2_hits: AtomicU64,
    /// KV pages that had to be loaded from T3 (NVMe).
    pub kv_t3_hits: AtomicU64,
    /// Total positions stored in the KV cache across all layers.
    pub kv_total_positions: AtomicU64,
    /// KV pages evicted to make room for weight pages or newer KV.
    pub kv_evictions: AtomicU64,
    /// Sparse attention queries executed (vs exhaustive attention).
    pub sparse_attn_queries: AtomicU64,
    /// Positions retrieved via ANN search (subset of total KV).
    pub sparse_attn_retrieved: AtomicU64,
}

impl InferenceStats {
    pub fn t1_hit_rate(&self) -> f64 {
        let total = self.total_page_accesses.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.t1_hits.load(Ordering::Relaxed) as f64 / total as f64
    }

    pub fn combined_hit_rate(&self) -> f64 {
        let total = self.total_page_accesses.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let hits = self.t1_hits.load(Ordering::Relaxed) + self.t2_hits.load(Ordering::Relaxed);
        hits as f64 / total as f64
    }

    pub fn tokens_per_second(&self, wall_ns: u64) -> f64 {
        if wall_ns == 0 {
            return 0.0;
        }
        self.tokens_generated.load(Ordering::Relaxed) as f64 * 1e9 / wall_ns as f64
    }

    pub fn prefetch_efficiency(&self) -> f64 {
        let issued = self.prefetch_issued.load(Ordering::Relaxed);
        if issued == 0 {
            return 0.0;
        }
        self.prefetch_useful.load(Ordering::Relaxed) as f64 / issued as f64
    }

    /// Create a snapshot for reporting (non-atomic copy).
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            tokens_generated: self.tokens_generated.load(Ordering::Relaxed),
            total_page_accesses: self.total_page_accesses.load(Ordering::Relaxed),
            t1_hits: self.t1_hits.load(Ordering::Relaxed),
            t2_hits: self.t2_hits.load(Ordering::Relaxed),
            t3_hits: self.t3_hits.load(Ordering::Relaxed),
            prefetch_issued: self.prefetch_issued.load(Ordering::Relaxed),
            prefetch_useful: self.prefetch_useful.load(Ordering::Relaxed),
            prefetch_wasted: self.prefetch_wasted.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            stalls: self.stalls.load(Ordering::Relaxed),
            compute_ns: self.compute_ns.load(Ordering::Relaxed),
            transfer_ns: self.transfer_ns.load(Ordering::Relaxed),
            stall_ns: self.stall_ns.load(Ordering::Relaxed),
            kv_page_accesses: self.kv_page_accesses.load(Ordering::Relaxed),
            kv_t1_hits: self.kv_t1_hits.load(Ordering::Relaxed),
            kv_t2_hits: self.kv_t2_hits.load(Ordering::Relaxed),
            kv_t3_hits: self.kv_t3_hits.load(Ordering::Relaxed),
            kv_total_positions: self.kv_total_positions.load(Ordering::Relaxed),
            kv_evictions: self.kv_evictions.load(Ordering::Relaxed),
            sparse_attn_queries: self.sparse_attn_queries.load(Ordering::Relaxed),
            sparse_attn_retrieved: self.sparse_attn_retrieved.load(Ordering::Relaxed),
        }
    }
}

/// Non-atomic snapshot of stats for reporting/serialization.
#[derive(Clone, Debug, serde::Serialize)]
pub struct StatsSnapshot {
    pub tokens_generated: u64,
    pub total_page_accesses: u64,
    pub t1_hits: u64,
    pub t2_hits: u64,
    pub t3_hits: u64,
    pub prefetch_issued: u64,
    pub prefetch_useful: u64,
    pub prefetch_wasted: u64,
    pub evictions: u64,
    pub stalls: u64,
    pub compute_ns: u64,
    pub transfer_ns: u64,
    pub stall_ns: u64,

    // KV cache stats
    pub kv_page_accesses: u64,
    pub kv_t1_hits: u64,
    pub kv_t2_hits: u64,
    pub kv_t3_hits: u64,
    pub kv_total_positions: u64,
    pub kv_evictions: u64,
    pub sparse_attn_queries: u64,
    pub sparse_attn_retrieved: u64,
}

// ─── Gear Taxonomy ──────────────────────────────────────────────────────

/// The six-gear taxonomy from Clank's Gearbox routing.
///
/// Each gear represents a high-level task mode that biases expert selection
/// and drives cache management decisions. When a gear signal is available
/// (from Gearbox or an external classifier), vib3 uses it for:
/// - Immediate mode detection (skip entropy warmup)
/// - Proactive cache warming (pin gear's expert working set)
/// - Filtered HNSW search (narrow page retrieval by domain)
/// - Gear-aware eviction (deprioritize old gear's pages)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum Gear {
    /// Code generation, debugging, refactoring.
    Code,
    /// Image/video understanding, spatial reasoning, OCR.
    Vision,
    /// Chain-of-thought, math, planning, logical deduction.
    Reason,
    /// Tool calling, structured output, function invocation.
    Tool,
    /// Conversational dialogue, style, cultural knowledge.
    Chat,
    /// Long-term memory, retrieval, compression, factual recall.
    Memory,
}

impl Gear {
    /// All gear variants in canonical order.
    pub const ALL: [Gear; 6] = [
        Gear::Code,
        Gear::Vision,
        Gear::Reason,
        Gear::Tool,
        Gear::Chat,
        Gear::Memory,
    ];

    /// Parse a gear name string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "code" => Some(Gear::Code),
            "vision" => Some(Gear::Vision),
            "reason" | "reasoning" => Some(Gear::Reason),
            "tool" | "tools" => Some(Gear::Tool),
            "chat" | "conversation" => Some(Gear::Chat),
            "memory" | "retrieval" => Some(Gear::Memory),
            _ => None,
        }
    }

    /// Canonical string name.
    pub const fn name(&self) -> &'static str {
        match self {
            Gear::Code => "code",
            Gear::Vision => "vision",
            Gear::Reason => "reason",
            Gear::Tool => "tool",
            Gear::Chat => "chat",
            Gear::Memory => "memory",
        }
    }

    /// Expected activation mode for this gear.
    ///
    /// From the integration spec Section 3.6:
    /// - code, vision, reason, tool → Specialist (concentrated expert activation)
    /// - chat, memory → Generalist (diverse expert activation)
    pub const fn expected_mode(&self) -> ActivationMode {
        match self {
            Gear::Code => ActivationMode::Specialist,
            Gear::Vision => ActivationMode::Specialist,
            Gear::Reason => ActivationMode::Specialist,
            Gear::Tool => ActivationMode::Specialist,
            Gear::Chat => ActivationMode::Generalist,
            Gear::Memory => ActivationMode::Generalist,
        }
    }
}

impl fmt::Display for Gear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ─── Task Context ───────────────────────────────────────────────────────

/// Optional task context attached to an inference request.
///
/// This is the integration surface between Clank's Gearbox and vib3's storage
/// engine. When present, it provides an authoritative task signal that drives:
/// - Mode detection (skip entropy-based warmup)
/// - Cache warming (pin gear's working set before generation)
/// - Filtered HNSW search (narrow page retrieval by domain)
/// - Eviction policy (deprioritize old gear's pages)
///
/// When absent, vib3 falls back to entropy-based mode detection and unfiltered
/// search — the integration is an enhancement, not a dependency.
///
/// Passed via the OpenAI-compatible API as `extra_body.task_context`:
/// ```json
/// {
///   "model": "clank",
///   "messages": [...],
///   "extra_body": {
///     "task_context": {
///       "gear": "code",
///       "blend": { "code": 0.7, "reason": 0.3 },
///       "alpha": 0.8,
///       "phase": "executing"
///     }
///   }
/// }
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TaskContext {
    /// Primary gear (e.g., "code", "vision", "reason").
    /// When set, this is the dominant task mode.
    #[serde(default)]
    pub gear: Option<String>,

    /// Gear blend weights for mixed tasks (e.g., { "code": 0.7, "reason": 0.3 }).
    /// Keys are gear names, values are blend weights (should sum to ~1.0).
    /// When present alongside `gear`, the blend provides finer-grained control.
    #[serde(default)]
    pub blend: Option<HashMap<String, f32>>,

    /// Gearbox alpha override — how strongly to apply the gear bias.
    /// Higher values = more aggressive filtering/routing.
    /// If None, uses the engine's default alpha.
    #[serde(default)]
    pub alpha: Option<f32>,

    /// Execution phase hint (e.g., "planning", "executing", "verifying").
    /// Informational — may influence cache strategy in future phases.
    #[serde(default)]
    pub phase: Option<String>,
}

impl TaskContext {
    /// Create a task context with just a primary gear.
    pub fn with_gear(gear: &str) -> Self {
        Self {
            gear: Some(gear.to_string()),
            blend: None,
            alpha: None,
            phase: None,
        }
    }

    /// Create a task context with a gear blend.
    pub fn with_blend(blend: HashMap<String, f32>) -> Self {
        // Primary gear is the one with the highest blend weight
        let primary = blend
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone());
        Self {
            gear: primary,
            blend: Some(blend),
            alpha: None,
            phase: None,
        }
    }

    /// Resolve the primary gear to a `Gear` enum variant.
    /// Returns None if gear is unset or not a recognized gear name.
    pub fn primary_gear(&self) -> Option<Gear> {
        self.gear.as_deref().and_then(Gear::from_str)
    }

    /// Get the expected activation mode from the task context.
    /// Returns None if no gear is set or the gear is unrecognized.
    pub fn expected_mode(&self) -> Option<ActivationMode> {
        self.primary_gear().map(|g| g.expected_mode())
    }

    /// Get blend weights as (Gear, weight) pairs, filtering out unrecognized gears.
    pub fn gear_blend(&self) -> Vec<(Gear, f32)> {
        match &self.blend {
            Some(blend) => blend
                .iter()
                .filter_map(|(name, &weight)| Gear::from_str(name).map(|gear| (gear, weight)))
                .collect(),
            None => {
                // If no blend but a primary gear is set, treat it as 100% blend
                match self.primary_gear() {
                    Some(gear) => vec![(gear, 1.0)],
                    None => vec![],
                }
            }
        }
    }
}
