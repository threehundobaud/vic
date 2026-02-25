//! Tiered KV Cache — page-managed KV storage with ANN-indexed retrieval.
//!
//! This is Section 8 of the whitepaper, implemented. The tiered KV cache
//! treats K and V vectors as first-class pages in the unified storage engine,
//! sharing the same three-tier hierarchy (VRAM → RAM → NVMe) and eviction
//! policy as expert weight pages.
//!
//! ## Architecture
//!
//! ```text
//!                           KV Cache Pages
//!                          +-------------+
//! T1 (VRAM):               | Recent KV    |  ← sliding window + landmarks
//!                          | + ANN top-k  |  ← retrieved via sparse attention
//!                          +-------------+
//!                                |
//! T2 (RAM):                | Indexed KV   |  ← ANN index over K vectors
//!                          | (bulk store) |
//!                          +-------------+
//!                                |
//! T3 (NVMe):               | Cold KV      |  ← very long context overflow
//!                          | (archived)   |
//!                          +-------------+
//! ```
//!
//! ## Key Design Decisions
//!
//! 1. **Page granularity**: Each KV page stores `positions_per_page` positions
//!    for one KV head at one layer. At head_dim=128, that's 512 bytes/position,
//!    giving ~4096 positions per 2MB page.
//!
//! 2. **Per-head pages**: K and V are stored in separate pages (K page, V page).
//!    This enables ANN search over K pages without loading V data until needed.
//!
//! 3. **Three zones in T1**:
//!    - Recent window (always resident, sliding)
//!    - Landmarks (pinned positions with high aggregate attention)
//!    - ANN-retrieved (promoted from T2 on demand, per-query)
//!
//! 4. **ANN index lives in T2**: An index over K vectors stored in host RAM.
//!    When computing attention, Q is searched against this index to find the
//!    top-k most relevant positions. Their K/V pages are then promoted to T1.

use crate::core::config::KvCacheConfig;
use crate::core::types::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

// ─── KV Page Geometry ───────────────────────────────────────────────────

/// Compute how many KV positions fit in one 2MB page for a given head_dim.
///
/// Each position stores `head_dim` f32 values = `head_dim * 4` bytes.
pub fn positions_per_page(head_dim: usize) -> usize {
    let bytes_per_position = head_dim * std::mem::size_of::<f32>();
    if bytes_per_position == 0 {
        return 0;
    }
    PAGE_SIZE / bytes_per_position
}

/// Compute the block index for a given sequence position.
pub fn position_to_block(position: usize, positions_per_page: usize) -> u16 {
    if positions_per_page == 0 {
        return 0;
    }
    (position / positions_per_page) as u16
}

/// Compute the offset within a block for a given sequence position.
pub fn position_offset_in_block(position: usize, positions_per_page: usize) -> usize {
    if positions_per_page == 0 {
        return 0;
    }
    position % positions_per_page
}

// ─── KV Position Metadata ───────────────────────────────────────────────

/// Metadata for a single KV position across the tiered cache.
#[derive(Clone, Debug)]
struct PositionMeta {
    /// Which tier this position currently resides in.
    tier: Tier,
    /// Aggregate attention weight received (for landmark detection).
    attention_weight: f32,
    /// Tick of last access.
    last_access: u64,
}

// ─── Landmark Tracker ───────────────────────────────────────────────────

/// Tracks which positions receive the highest aggregate attention weight
/// and should be pinned as landmarks in T1.
///
/// Landmarks are the "materialized indexes" of the KV cache — positions
/// that every future query is likely to need (e.g., system prompt, key
/// instructions, important context).
struct LandmarkTracker {
    /// (position, aggregate_attention_weight) — maintained as a sorted set.
    landmarks: Vec<(usize, f32)>,
    max_landmarks: usize,
}

impl LandmarkTracker {
    fn new(max_landmarks: usize) -> Self {
        Self {
            landmarks: Vec::with_capacity(max_landmarks),
            max_landmarks,
        }
    }

    /// Update the attention weight for a position.
    fn update(&mut self, position: usize, weight: f32) {
        if let Some(entry) = self.landmarks.iter_mut().find(|(p, _)| *p == position) {
            entry.1 += weight;
        } else if self.landmarks.len() < self.max_landmarks {
            self.landmarks.push((position, weight));
        } else {
            // Replace the lowest-weight landmark if new weight is higher
            if let Some(min_idx) = self
                .landmarks
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1 .1
                        .partial_cmp(&b.1 .1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
            {
                if weight > self.landmarks[min_idx].1 {
                    self.landmarks[min_idx] = (position, weight);
                }
            }
        }
    }

    /// Get the current set of landmark positions, sorted by weight descending.
    fn positions(&self) -> Vec<usize> {
        let mut sorted = self.landmarks.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.iter().map(|(pos, _)| *pos).collect()
    }

    fn contains(&self, position: usize) -> bool {
        self.landmarks.iter().any(|(p, _)| *p == position)
    }

    fn len(&self) -> usize {
        self.landmarks.len()
    }
}

// ─── KV Index (ANN over K vectors) ─────────────────────────────────────

/// ANN index over K vectors for one KV head at one layer.
///
/// This is the core data structure that transforms O(n) attention into
/// O(k log n) retrieval. It stores K vectors from T2 positions and supports
/// fast approximate nearest-neighbor queries with Q vectors.
///
/// For the CPU/test path, this uses the same `AnnBackend` trait as the
/// expert vector index. The default is brute-force; production would use
/// HNSW (usearch) for million-scale vectors.
pub struct KvIndex {
    /// K vectors indexed by position. Dense array: index = position.
    k_vectors: Vec<Vec<f32>>,
    /// Set of positions currently in the index.
    indexed_positions: HashSet<usize>,
    /// Head dimension.
    head_dim: usize,
    /// Total indexed count.
    count: usize,
}

impl KvIndex {
    pub fn new(head_dim: usize) -> Self {
        Self {
            k_vectors: Vec::new(),
            indexed_positions: HashSet::new(),
            head_dim,
            count: 0,
        }
    }

    /// Insert a K vector for a position.
    pub fn insert(&mut self, position: usize, k_vector: &[f32]) {
        assert_eq!(k_vector.len(), self.head_dim);

        // Grow storage if needed
        if position >= self.k_vectors.len() {
            self.k_vectors.resize(position + 1, Vec::new());
        }

        if self.k_vectors[position].is_empty() {
            self.count += 1;
        }
        self.k_vectors[position] = k_vector.to_vec();
        self.indexed_positions.insert(position);
    }

    /// Remove a position from the index.
    pub fn remove(&mut self, position: usize) {
        if position < self.k_vectors.len() && !self.k_vectors[position].is_empty() {
            self.k_vectors[position].clear();
            self.indexed_positions.remove(&position);
            self.count -= 1;
        }
    }

    /// Search for the top-k most similar positions to a query vector.
    ///
    /// Returns (position, squared_distance) pairs sorted by distance.
    /// This is the brute-force path — O(n) where n is indexed positions.
    /// For production, replace with HNSW for O(k log n).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.count == 0 || k == 0 {
            return vec![];
        }

        let mut results: Vec<(usize, f32)> = self
            .indexed_positions
            .iter()
            .filter_map(|&pos| {
                let kv = &self.k_vectors[pos];
                if kv.is_empty() {
                    return None;
                }
                // Dot product (attention score proxy)
                let dot: f32 = query.iter().zip(kv.iter()).map(|(q, k)| q * k).sum();
                Some((pos, dot))
            })
            .collect();

        // Sort by dot product descending (higher = more relevant)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Number of indexed positions.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if a position is in the index.
    pub fn contains(&self, position: usize) -> bool {
        self.indexed_positions.contains(&position)
    }

    /// Get the K vector for a position (if indexed).
    pub fn get(&self, position: usize) -> Option<&[f32]> {
        if position < self.k_vectors.len() && !self.k_vectors[position].is_empty() {
            Some(&self.k_vectors[position])
        } else {
            None
        }
    }
}

// ─── Tiered KV Cache ────────────────────────────────────────────────────

/// Per-layer, per-head KV cache tier assignment.
///
/// Tracks which positions are in which tier and manages promotion/demotion.
struct HeadCache {
    /// Tier assignment for each position.
    positions: HashMap<usize, PositionMeta>,
    /// Positions currently in T1 (VRAM) — fast set for membership check.
    t1_positions: HashSet<usize>,
    /// Positions currently in T2 (RAM).
    t2_positions: HashSet<usize>,
    /// Positions currently in T3 (NVMe).
    t3_positions: HashSet<usize>,
    /// K vector data for positions in T1 and T2.
    k_data: Vec<Vec<f32>>,
    /// V vector data for positions in T1 and T2.
    v_data: Vec<Vec<f32>>,
    /// ANN index over K vectors in T2.
    kv_index: KvIndex,
    /// Landmark tracker.
    landmarks: LandmarkTracker,
}

impl HeadCache {
    fn new(head_dim: usize, landmark_count: usize) -> Self {
        Self {
            positions: HashMap::new(),
            t1_positions: HashSet::new(),
            t2_positions: HashSet::new(),
            t3_positions: HashSet::new(),
            k_data: Vec::new(),
            v_data: Vec::new(),
            kv_index: KvIndex::new(head_dim),
            landmarks: LandmarkTracker::new(landmark_count),
        }
    }
}

/// The tiered KV cache manager.
///
/// Manages K and V vectors across VRAM, RAM, and NVMe for all layers
/// and heads, integrated with the unified page buffer manager.
///
/// ## Attention-as-Query-Plan
///
/// When computing attention for a token, the tiered KV cache executes:
/// 1. **Recent window**: Always in T1, always attended. No search needed.
/// 2. **Landmarks**: Pinned in T1, always attended. No search needed.
/// 3. **ANN retrieval**: Search K index in T2 with Q vector, retrieve top-k
///    positions, promote their K/V pages to T1 for this attention pass.
/// 4. **Compute attention**: Over the union of (recent + landmarks + retrieved).
///
/// This transforms O(seq_len) attention into O(k log n) retrieval + O(k) attention.
pub struct TieredKvCache {
    /// Per-layer, per-head caches.
    /// Indexed as: heads[layer_idx][head_idx]
    heads: Vec<Vec<HeadCache>>,

    /// Configuration.
    config: KvCacheConfig,

    /// Current sequence length (positions appended so far).
    seq_len: usize,

    /// Head dimension.
    head_dim: usize,

    /// Number of layers.
    num_layers: usize,

    /// Number of KV heads per layer.
    num_kv_heads: usize,

    /// Monotonic tick for access ordering.
    tick: u64,

    /// Statistics.
    pub stats: TieredKvStats,
}

/// Statistics for the tiered KV cache.
#[derive(Default)]
pub struct TieredKvStats {
    /// Total append operations (one per token per layer per head).
    pub appends: AtomicU64,
    /// Positions currently in T1 across all heads/layers.
    pub t1_positions: AtomicU64,
    /// Positions currently in T2 across all heads/layers.
    pub t2_positions: AtomicU64,
    /// Positions currently in T3 across all heads/layers.
    pub t3_positions: AtomicU64,
    /// ANN search queries executed.
    pub ann_queries: AtomicU64,
    /// Positions retrieved via ANN search.
    pub ann_retrieved: AtomicU64,
    /// Positions demoted from T1 to T2.
    pub demotions_t1_t2: AtomicU64,
    /// Positions demoted from T2 to T3.
    pub demotions_t2_t3: AtomicU64,
    /// Landmark updates.
    pub landmark_updates: AtomicU64,
}

/// Result of attention computation over the tiered KV cache.
#[derive(Debug)]
pub struct AttentionResult {
    /// Positions that participated in attention (for stats/debugging).
    pub attended_positions: usize,
    /// How many came from the recent window.
    pub from_recent: usize,
    /// How many came from landmarks.
    pub from_landmarks: usize,
    /// How many came from ANN retrieval.
    pub from_ann: usize,
}

impl TieredKvCache {
    /// Create a new tiered KV cache.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        config: KvCacheConfig,
    ) -> Self {
        let landmark_count = config.landmark_count;
        let heads: Vec<Vec<HeadCache>> = (0..num_layers)
            .map(|_layer| {
                (0..num_kv_heads)
                    .map(|_head| HeadCache::new(head_dim, landmark_count))
                    .collect()
            })
            .collect();

        Self {
            heads,
            config,
            seq_len: 0,
            head_dim,
            num_layers,
            num_kv_heads,
            tick: 0,
            stats: TieredKvStats::default(),
        }
    }

    /// Append K and V vectors for a new position at one layer.
    ///
    /// The new position starts in T1 (hot). If T1 is full, the oldest
    /// non-landmark position is demoted to T2.
    ///
    /// `k_heads`: [num_kv_heads][head_dim]
    /// `v_heads`: [num_kv_heads][head_dim]
    pub fn append_layer(&mut self, layer: usize, k_heads: &[Vec<f32>], v_heads: &[Vec<f32>]) {
        assert_eq!(k_heads.len(), self.num_kv_heads);
        assert_eq!(v_heads.len(), self.num_kv_heads);

        let position = self.seq_len;
        let tick = self.tick;

        for head_idx in 0..self.num_kv_heads {
            let head_cache = &mut self.heads[layer][head_idx];

            // Grow data storage
            if position >= head_cache.k_data.len() {
                head_cache.k_data.resize(position + 1, Vec::new());
                head_cache.v_data.resize(position + 1, Vec::new());
            }

            // Store KV data
            head_cache.k_data[position] = k_heads[head_idx].clone();
            head_cache.v_data[position] = v_heads[head_idx].clone();

            // Add to T1
            head_cache.t1_positions.insert(position);
            head_cache.positions.insert(
                position,
                PositionMeta {
                    tier: Tier::T1Vram,
                    attention_weight: 0.0,
                    last_access: tick,
                },
            );

            self.stats.t1_positions.fetch_add(1, Ordering::Relaxed);
            self.stats.appends.fetch_add(1, Ordering::Relaxed);

            // Check if T1 is over capacity and demote if needed
            self.maybe_demote_t1(layer, head_idx);
        }
    }

    /// Advance the sequence position. Call after appending all layers for a token.
    pub fn advance_position(&mut self) {
        self.seq_len += 1;
        self.tick += 1;
    }

    /// Get the current sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Gather the positions to attend to for a given layer and head.
    ///
    /// Returns the set of positions from three sources:
    /// 1. Recent window (last `recent_window` positions) — always in T1
    /// 2. Landmarks — pinned in T1
    /// 3. ANN-retrieved positions — searched from T2 index
    ///
    /// The union of these sets defines the sparse attention pattern.
    pub fn gather_attention_positions(
        &mut self,
        layer: usize,
        head_idx: usize,
        query: &[f32],
    ) -> Vec<usize> {
        let head_cache = &self.heads[layer][head_idx];
        let mut positions = HashSet::new();

        // 1. Recent window: last N positions
        let recent_start = self.seq_len.saturating_sub(self.config.recent_window);
        for pos in recent_start..self.seq_len {
            if head_cache.t1_positions.contains(&pos) || head_cache.t2_positions.contains(&pos) {
                positions.insert(pos);
            }
        }
        let _from_recent = positions.len();

        // 2. Landmarks
        let landmark_positions = head_cache.landmarks.positions();
        for &pos in &landmark_positions {
            positions.insert(pos);
        }

        // 3. ANN retrieval from T2 index (if sparse attention enabled)
        let mut from_ann = 0;
        if self.config.sparse_attention && !head_cache.kv_index.is_empty() {
            let top_k = self.config.top_k_positions;
            let results = head_cache.kv_index.search(query, top_k);

            self.stats.ann_queries.fetch_add(1, Ordering::Relaxed);

            for (pos, _score) in &results {
                if positions.insert(*pos) {
                    from_ann += 1;
                }
            }

            self.stats
                .ann_retrieved
                .fetch_add(from_ann as u64, Ordering::Relaxed);
        }

        let mut result: Vec<usize> = positions.into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Get K vectors for a set of positions at a given layer and head.
    pub fn get_k_vectors(
        &self,
        layer: usize,
        head_idx: usize,
        positions: &[usize],
    ) -> Vec<Vec<f32>> {
        let head_cache = &self.heads[layer][head_idx];
        positions
            .iter()
            .filter_map(|&pos| {
                if pos < head_cache.k_data.len() && !head_cache.k_data[pos].is_empty() {
                    Some(head_cache.k_data[pos].clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get V vectors for a set of positions at a given layer and head.
    pub fn get_v_vectors(
        &self,
        layer: usize,
        head_idx: usize,
        positions: &[usize],
    ) -> Vec<Vec<f32>> {
        let head_cache = &self.heads[layer][head_idx];
        positions
            .iter()
            .filter_map(|&pos| {
                if pos < head_cache.v_data.len() && !head_cache.v_data[pos].is_empty() {
                    Some(head_cache.v_data[pos].clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Update attention weights for landmark tracking.
    ///
    /// Called after computing attention to record which positions received
    /// significant attention weight. Over time, positions that consistently
    /// receive high attention are promoted to landmarks.
    pub fn update_attention_weights(
        &mut self,
        layer: usize,
        head_idx: usize,
        position_weights: &[(usize, f32)],
    ) {
        let head_cache = &mut self.heads[layer][head_idx];

        for &(pos, weight) in position_weights {
            head_cache.landmarks.update(pos, weight);

            if let Some(meta) = head_cache.positions.get_mut(&pos) {
                meta.attention_weight += weight;
                meta.last_access = self.tick;
            }
        }

        self.stats
            .landmark_updates
            .fetch_add(position_weights.len() as u64, Ordering::Relaxed);
    }

    /// Demote oldest non-landmark, non-recent positions from T1 to T2.
    fn maybe_demote_t1(&mut self, layer: usize, head_idx: usize) {
        let max_t1 = self.config.t1_positions;
        let head_cache = &mut self.heads[layer][head_idx];

        while head_cache.t1_positions.len() > max_t1 {
            // Find the best eviction victim: oldest non-landmark, non-recent position
            let recent_start = self.seq_len.saturating_sub(self.config.recent_window);

            let victim = head_cache
                .t1_positions
                .iter()
                .filter(|&&pos| pos < recent_start && !head_cache.landmarks.contains(pos))
                .min_by_key(|&&pos| {
                    head_cache
                        .positions
                        .get(&pos)
                        .map(|m| m.last_access)
                        .unwrap_or(0)
                })
                .copied();

            match victim {
                Some(pos) => {
                    // Demote to T2
                    head_cache.t1_positions.remove(&pos);
                    head_cache.t2_positions.insert(pos);

                    if let Some(meta) = head_cache.positions.get_mut(&pos) {
                        meta.tier = Tier::T2Ram;
                    }

                    // Add K vector to ANN index
                    if pos < head_cache.k_data.len() && !head_cache.k_data[pos].is_empty() {
                        head_cache
                            .kv_index
                            .insert(pos, &head_cache.k_data[pos].clone());
                    }

                    self.stats.t1_positions.fetch_sub(1, Ordering::Relaxed);
                    self.stats.t2_positions.fetch_add(1, Ordering::Relaxed);
                    self.stats.demotions_t1_t2.fetch_add(1, Ordering::Relaxed);
                }
                None => break, // No evictable positions (all are recent or landmarks)
            }
        }
    }

    /// Demote positions from T2 to T3 when T2 is over capacity.
    pub fn maybe_demote_t2(&mut self, layer: usize, head_idx: usize) {
        let max_t2 = self.config.t2_positions;
        let head_cache = &mut self.heads[layer][head_idx];

        while head_cache.t2_positions.len() > max_t2 {
            // Evict lowest-attention-weight position from T2
            let victim = head_cache
                .t2_positions
                .iter()
                .min_by(|&&a, &&b| {
                    let wa = head_cache
                        .positions
                        .get(&a)
                        .map(|m| m.attention_weight)
                        .unwrap_or(0.0);
                    let wb = head_cache
                        .positions
                        .get(&b)
                        .map(|m| m.attention_weight)
                        .unwrap_or(0.0);
                    wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied();

            match victim {
                Some(pos) => {
                    head_cache.t2_positions.remove(&pos);
                    head_cache.t3_positions.insert(pos);

                    if let Some(meta) = head_cache.positions.get_mut(&pos) {
                        meta.tier = Tier::T3Nvme;
                    }

                    // Remove from ANN index
                    head_cache.kv_index.remove(pos);

                    // Free RAM data (keep position metadata for potential T3 retrieval)
                    if pos < head_cache.k_data.len() {
                        head_cache.k_data[pos].clear();
                        // Shrink allocated memory
                        head_cache.k_data[pos].shrink_to_fit();
                    }
                    if pos < head_cache.v_data.len() {
                        head_cache.v_data[pos].clear();
                        head_cache.v_data[pos].shrink_to_fit();
                    }

                    self.stats.t2_positions.fetch_sub(1, Ordering::Relaxed);
                    self.stats.t3_positions.fetch_add(1, Ordering::Relaxed);
                    self.stats.demotions_t2_t3.fetch_add(1, Ordering::Relaxed);
                }
                None => break,
            }
        }
    }

    /// Clear the entire KV cache (for new sequences).
    pub fn clear(&mut self) {
        for layer_heads in &mut self.heads {
            for head_cache in layer_heads {
                head_cache.positions.clear();
                head_cache.t1_positions.clear();
                head_cache.t2_positions.clear();
                head_cache.t3_positions.clear();
                head_cache.k_data.clear();
                head_cache.v_data.clear();
                head_cache.kv_index = KvIndex::new(self.head_dim);
                head_cache.landmarks = LandmarkTracker::new(self.config.landmark_count);
            }
        }
        self.seq_len = 0;
        self.tick = 0;
    }

    // ── Accessors ────────────────────────────────────────────────────

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Total positions in T1 for a given layer and head.
    pub fn t1_count(&self, layer: usize, head_idx: usize) -> usize {
        self.heads[layer][head_idx].t1_positions.len()
    }

    /// Total positions in T2 for a given layer and head.
    pub fn t2_count(&self, layer: usize, head_idx: usize) -> usize {
        self.heads[layer][head_idx].t2_positions.len()
    }

    /// Total positions in T3 for a given layer and head.
    pub fn t3_count(&self, layer: usize, head_idx: usize) -> usize {
        self.heads[layer][head_idx].t3_positions.len()
    }

    /// Number of landmarks for a given layer and head.
    pub fn landmark_count(&self, layer: usize, head_idx: usize) -> usize {
        self.heads[layer][head_idx].landmarks.len()
    }

    /// Number of positions in the ANN index for a given layer and head.
    pub fn index_count(&self, layer: usize, head_idx: usize) -> usize {
        self.heads[layer][head_idx].kv_index.len()
    }

    /// Whether the tiered KV cache is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the config.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Generate PageIds for all KV cache pages currently in T1.
    ///
    /// This is used by the unified eviction policy to reason about
    /// KV pages alongside weight pages.
    pub fn t1_page_ids(&self) -> Vec<PageId> {
        let ppp = positions_per_page(self.head_dim);
        let mut pages = Vec::new();

        for (layer_idx, layer_heads) in self.heads.iter().enumerate() {
            for head_cache in layer_heads {
                // Collect unique block indices for K and V pages in T1
                let mut k_blocks: HashSet<u16> = HashSet::new();
                for &pos in &head_cache.t1_positions {
                    k_blocks.insert(position_to_block(pos, ppp));
                }

                for &block in &k_blocks {
                    pages.push(PageId::kv_cache(layer_idx as u16, KV_SEGMENT_K, block));
                    pages.push(PageId::kv_cache(layer_idx as u16, KV_SEGMENT_V, block));
                }
            }
        }

        pages
    }

    /// Estimate total RAM (T2) bytes used by KV cache data.
    pub fn t2_bytes_used(&self) -> usize {
        let bytes_per_pos = self.head_dim * std::mem::size_of::<f32>() * 2; // K + V
        let mut total = 0;
        for layer_heads in &self.heads {
            for head_cache in layer_heads {
                total += head_cache.t2_positions.len() * bytes_per_pos;
            }
        }
        total
    }

    /// Estimate total VRAM (T1) bytes used by KV cache data.
    pub fn t1_bytes_used(&self) -> usize {
        let bytes_per_pos = self.head_dim * std::mem::size_of::<f32>() * 2; // K + V
        let mut total = 0;
        for layer_heads in &self.heads {
            for head_cache in layer_heads {
                total += head_cache.t1_positions.len() * bytes_per_pos;
            }
        }
        total
    }
}

// ─── Unified Eviction Policy ────────────────────────────────────────────

/// Pressure level for a memory tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPressure {
    /// Under watermark — no eviction needed.
    Low,
    /// Between low and high watermarks — gentle eviction.
    Medium,
    /// Over high watermark — aggressive eviction needed.
    High,
    /// Tier is full — stalling for eviction.
    Critical,
}

/// Recommendation from the unified eviction policy.
#[derive(Clone, Debug)]
pub struct EvictionRecommendation {
    /// Evict weight pages from T1?
    pub evict_weights: bool,
    /// Evict KV pages from T1?
    pub evict_kv: bool,
    /// Number of weight pages to evict.
    pub weight_pages_to_evict: usize,
    /// Number of KV positions to demote from T1 to T2.
    pub kv_positions_to_demote: usize,
    /// Current T1 memory pressure.
    pub pressure: MemoryPressure,
    /// Explanation (for logging).
    pub reason: String,
}

/// The unified eviction policy coordinates eviction across weight pages
/// and KV cache pages in the same VRAM/RAM budget.
///
/// Key insight: when T1 VRAM is under pressure, should we evict a cold
/// weight page or a cold KV page? No existing system can make this tradeoff
/// because no existing system manages both in the same pool.
///
/// ## Decision Factors
///
/// 1. **Cost of weight miss**: Loading a weight page from T2→T1 costs
///    ~0.13ms (2MB / 16GB/s PCIe). Weight pages are needed deterministically
///    (the router already decided which experts to activate).
///
/// 2. **Cost of KV miss**: Not having a KV position in T1 means either:
///    a. Skip it (lossy — reduces attention accuracy)
///    b. Load from T2 (adds latency to attention computation)
///    For sparse attention, many KV positions are never retrieved, so
///    evicting them is often free.
///
/// 3. **Predictability**: Weight page access is highly predictable (vector
///    index + coactivation). KV cache access depends on the query at each
///    layer, which we can't predict until the Q projection is computed.
///
/// **Conclusion**: Prefer evicting KV over weights when both are cold,
/// because weight misses have higher predictability and a stall cost,
/// while KV misses in sparse attention are often masked by ANN retrieval.
pub struct UnifiedEvictionPolicy {
    /// T1 budget fraction for KV cache.
    kv_fraction: f32,
    /// T2 budget fraction for KV cache.
    t2_kv_fraction: f32,
    /// T1 high watermark (start evicting above this).
    t1_high_watermark: f32,
    /// T1 low watermark (target after eviction).
    t1_low_watermark: f32,
}

impl UnifiedEvictionPolicy {
    pub fn new(kv_fraction: f32, t2_kv_fraction: f32) -> Self {
        Self {
            kv_fraction,
            t2_kv_fraction,
            t1_high_watermark: 0.95,
            t1_low_watermark: 0.80,
        }
    }

    /// Evaluate memory pressure and produce eviction recommendations.
    ///
    /// `t1_weight_used`: Weight pages occupying T1 (in page count).
    /// `t1_total`: Total T1 slots.
    /// `t1_kv_bytes`: Bytes used by KV cache in T1 (from TieredKvCache).
    pub fn evaluate(
        &self,
        t1_weight_used: usize,
        t1_total: usize,
        t1_kv_bytes: usize,
    ) -> EvictionRecommendation {
        if t1_total == 0 {
            return EvictionRecommendation {
                evict_weights: false,
                evict_kv: false,
                weight_pages_to_evict: 0,
                kv_positions_to_demote: 0,
                pressure: MemoryPressure::Low,
                reason: "No T1 pool".into(),
            };
        }

        let t1_kv_pages = t1_kv_bytes / PAGE_SIZE.max(1);
        let total_used = t1_weight_used + t1_kv_pages;
        let utilization = total_used as f32 / t1_total as f32;

        let pressure = if utilization >= 1.0 {
            MemoryPressure::Critical
        } else if utilization >= self.t1_high_watermark {
            MemoryPressure::High
        } else if utilization >= self.t1_low_watermark {
            MemoryPressure::Medium
        } else {
            MemoryPressure::Low
        };

        if pressure == MemoryPressure::Low {
            return EvictionRecommendation {
                evict_weights: false,
                evict_kv: false,
                weight_pages_to_evict: 0,
                kv_positions_to_demote: 0,
                pressure,
                reason: "T1 utilization OK".into(),
            };
        }

        // Target: reduce to low watermark
        let target_used = (t1_total as f32 * self.t1_low_watermark) as usize;
        let pages_to_free = total_used.saturating_sub(target_used);

        if pages_to_free == 0 {
            return EvictionRecommendation {
                evict_weights: false,
                evict_kv: false,
                weight_pages_to_evict: 0,
                kv_positions_to_demote: 0,
                pressure,
                reason: "At target".into(),
            };
        }

        // Budget allocation: KV should be <= kv_fraction of T1.
        // If KV is over-budget, prefer evicting KV first.
        let kv_budget_pages = (t1_total as f32 * self.kv_fraction) as usize;
        let kv_over_budget = t1_kv_pages.saturating_sub(kv_budget_pages);

        let (weight_evict, kv_demote) = if kv_over_budget > 0 {
            // KV is over budget — evict KV first, then weights if needed
            let kv_to_evict = kv_over_budget.min(pages_to_free);
            let weight_to_evict = pages_to_free.saturating_sub(kv_to_evict);
            (weight_to_evict, kv_to_evict)
        } else {
            // KV within budget — split eviction proportionally
            // But prefer evicting KV (lower cost of miss for sparse attention)
            let kv_share = (pages_to_free as f32 * 0.6) as usize; // 60% from KV
            let weight_share = pages_to_free - kv_share;
            // But don't evict more KV pages than exist
            let kv_evict = kv_share.min(t1_kv_pages);
            let weight_evict = weight_share + (kv_share - kv_evict);
            (weight_evict, kv_evict)
        };

        // Convert KV pages to positions
        // Rough: positions_per_page(128) = 4096, but we don't know head_dim here.
        // Use page count directly — the caller converts to positions.
        let kv_positions = kv_demote * KV_POSITIONS_PER_PAGE_DEFAULT;

        EvictionRecommendation {
            evict_weights: weight_evict > 0,
            evict_kv: kv_demote > 0,
            weight_pages_to_evict: weight_evict,
            kv_positions_to_demote: kv_positions,
            pressure,
            reason: format!(
                "T1 at {:.0}% ({}/{}): evict {} weight pages + {} KV positions",
                utilization * 100.0,
                total_used,
                t1_total,
                weight_evict,
                kv_positions,
            ),
        }
    }

    /// Get the KV fraction of T1.
    pub fn kv_fraction(&self) -> f32 {
        self.kv_fraction
    }

    /// Get the T2 KV fraction.
    pub fn t2_kv_fraction(&self) -> f32 {
        self.t2_kv_fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::KvCacheConfig;

    fn test_config() -> KvCacheConfig {
        KvCacheConfig {
            enabled: true,
            t1_positions: 8, // Small for testing
            t2_positions: 16,
            sparse_attention: true,
            top_k_positions: 4,
            recent_window: 4,
            landmark_count: 2,
            unified_pool: true,
            t1_kv_fraction: 0.15,
            t2_kv_fraction: 0.10,
        }
    }

    #[test]
    fn test_positions_per_page() {
        // head_dim=128, 4 bytes/float => 512 bytes/position
        // PAGE_SIZE = 2MB = 2,097,152 bytes
        // 2,097,152 / 512 = 4096
        assert_eq!(positions_per_page(128), 4096);

        // head_dim=64 => 256 bytes/position => 8192
        assert_eq!(positions_per_page(64), 8192);

        // head_dim=0 => 0
        assert_eq!(positions_per_page(0), 0);
    }

    #[test]
    fn test_position_to_block() {
        assert_eq!(position_to_block(0, 4096), 0);
        assert_eq!(position_to_block(4095, 4096), 0);
        assert_eq!(position_to_block(4096, 4096), 1);
        assert_eq!(position_to_block(8191, 4096), 1);
        assert_eq!(position_to_block(8192, 4096), 2);
    }

    #[test]
    fn test_page_id_kv_cache() {
        let k_page = PageId::kv_cache(5, KV_SEGMENT_K, 10);
        assert!(k_page.is_kv_cache());
        assert!(k_page.is_k_page());
        assert!(!k_page.is_v_page());
        assert!(!k_page.is_shared());
        assert!(!k_page.is_weight());
        assert_eq!(k_page.layer, 5);
        assert_eq!(k_page.page_idx, 10);

        let v_page = PageId::kv_cache(5, KV_SEGMENT_V, 10);
        assert!(v_page.is_kv_cache());
        assert!(v_page.is_v_page());
        assert!(!v_page.is_k_page());

        // Distinct keys
        assert_ne!(k_page.key(), v_page.key());

        // Debug formatting
        let dbg = format!("{:?}", k_page);
        assert!(dbg.contains("KV:K"));
        assert!(dbg.contains("blk10"));
    }

    #[test]
    fn test_kv_index_basic() {
        let mut idx = KvIndex::new(4);
        assert!(idx.is_empty());

        // Insert some K vectors
        idx.insert(0, &[1.0, 0.0, 0.0, 0.0]);
        idx.insert(1, &[0.0, 1.0, 0.0, 0.0]);
        idx.insert(2, &[0.0, 0.0, 1.0, 0.0]);
        idx.insert(5, &[0.0, 0.0, 0.0, 1.0]);

        assert_eq!(idx.len(), 4);
        assert!(idx.contains(0));
        assert!(idx.contains(5));
        assert!(!idx.contains(3));

        // Search: query closest to position 0
        let results = idx.search(&[0.9, 0.1, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Highest dot product with [1,0,0,0]

        // Remove
        idx.remove(0);
        assert_eq!(idx.len(), 3);
        assert!(!idx.contains(0));
    }

    #[test]
    fn test_kv_index_search_empty() {
        let idx = KvIndex::new(4);
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_tiered_kv_cache_append_and_tiers() {
        let config = test_config();
        let mut cache = TieredKvCache::new(2, 1, 4, config);

        // Append 12 positions (T1 max = 8, so 4 should demote to T2)
        for pos in 0..12 {
            let k = vec![vec![pos as f32; 4]];
            let v = vec![vec![(pos as f32) * 0.5; 4]];
            cache.append_layer(0, &k, &v);
            cache.advance_position();
        }

        // T1 should have at most 8 positions
        assert!(cache.t1_count(0, 0) <= 8);
        // Some positions should be in T2
        assert!(cache.t2_count(0, 0) > 0);
        // Total = 12
        assert_eq!(
            cache.t1_count(0, 0) + cache.t2_count(0, 0) + cache.t3_count(0, 0),
            12
        );
        assert_eq!(cache.seq_len(), 12);
    }

    #[test]
    fn test_tiered_kv_cache_gather_attention() {
        let config = test_config(); // recent_window=4, t1_positions=8, top_k=4
        let mut cache = TieredKvCache::new(1, 1, 4, config);

        // Append 16 positions
        for pos in 0..16 {
            let v = vec![vec![0.0; 4]];
            // Make position 3 a distinctive K vector
            let k = if pos == 3 {
                vec![vec![10.0, 0.0, 0.0, 0.0]]
            } else {
                vec![vec![pos as f32 * 0.1; 4]]
            };
            cache.append_layer(0, &k, &v);
            cache.advance_position();
        }

        // Query that should match position 3 via ANN
        let query = vec![10.0, 0.0, 0.0, 0.0];
        let positions = cache.gather_attention_positions(0, 0, &query);

        // Should include recent window (positions 12-15)
        assert!(positions.contains(&12));
        assert!(positions.contains(&15));

        // Should include position 3 via ANN if it was demoted to T2
        // (depends on whether T1 evicted it)
        // At minimum, we should have the recent window
        assert!(positions.len() >= 4);
    }

    #[test]
    fn test_tiered_kv_cache_landmarks() {
        let config = test_config();
        let mut cache = TieredKvCache::new(1, 1, 4, config);

        // Append positions
        for pos in 0..10 {
            let k = vec![vec![pos as f32; 4]];
            let v = vec![vec![0.0; 4]];
            cache.append_layer(0, &k, &v);
            cache.advance_position();
        }

        // Mark position 2 as high-attention (landmark candidate)
        cache.update_attention_weights(0, 0, &[(2, 10.0), (3, 0.1)]);

        assert_eq!(cache.landmark_count(0, 0), 2);
    }

    #[test]
    fn test_tiered_kv_cache_clear() {
        let config = test_config();
        let mut cache = TieredKvCache::new(1, 1, 4, config);

        for _pos in 0..5 {
            let k = vec![vec![0.0; 4]];
            let v = vec![vec![0.0; 4]];
            cache.append_layer(0, &k, &v);
            cache.advance_position();
        }

        assert_eq!(cache.seq_len(), 5);
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.t1_count(0, 0), 0);
    }

    #[test]
    fn test_tiered_kv_cache_t2_demotion() {
        let config = KvCacheConfig {
            enabled: true,
            t1_positions: 4,
            t2_positions: 4, // Small T2 too
            sparse_attention: false,
            top_k_positions: 2,
            recent_window: 2,
            landmark_count: 1,
            unified_pool: true,
            t1_kv_fraction: 0.15,
            t2_kv_fraction: 0.10,
        };
        let mut cache = TieredKvCache::new(1, 1, 4, config);

        // Append 12 positions — T1(4) + T2(4) = 8, so 4 should go to T3
        for i in 0..12 {
            let k = vec![vec![i as f32; 4]];
            let v = vec![vec![0.0; 4]];
            cache.append_layer(0, &k, &v);
            cache.advance_position();
            // Manually trigger T2 demotion
            cache.maybe_demote_t2(0, 0);
        }

        assert!(cache.t1_count(0, 0) <= 4);
        assert!(cache.t2_count(0, 0) <= 4);
        // Some must have gone to T3
        assert!(cache.t3_count(0, 0) > 0);
    }

    #[test]
    fn test_tiered_kv_cache_get_vectors() {
        let config = test_config();
        let mut cache = TieredKvCache::new(1, 1, 4, config);

        let k = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let v = vec![vec![5.0, 6.0, 7.0, 8.0]];
        cache.append_layer(0, &k, &v);
        cache.advance_position();

        let k_vecs = cache.get_k_vectors(0, 0, &[0]);
        assert_eq!(k_vecs.len(), 1);
        assert_eq!(k_vecs[0], vec![1.0, 2.0, 3.0, 4.0]);

        let v_vecs = cache.get_v_vectors(0, 0, &[0]);
        assert_eq!(v_vecs.len(), 1);
        assert_eq!(v_vecs[0], vec![5.0, 6.0, 7.0, 8.0]);

        // Non-existent position
        let empty = cache.get_k_vectors(0, 0, &[99]);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_tiered_kv_cache_page_ids() {
        let config = test_config();
        let mut cache = TieredKvCache::new(1, 1, 128, config);

        for _pos in 0..3 {
            let k = vec![vec![0.0; 128]];
            let v = vec![vec![0.0; 128]];
            cache.append_layer(0, &k, &v);
            cache.advance_position();
        }

        let page_ids = cache.t1_page_ids();
        // All 3 positions in the same block (block 0), so 2 pages (K + V)
        assert_eq!(page_ids.len(), 2);
        assert!(page_ids.iter().any(|p| p.is_k_page()));
        assert!(page_ids.iter().any(|p| p.is_v_page()));
    }

    #[test]
    fn test_tiered_kv_cache_bytes_used() {
        let config = test_config();
        let mut cache = TieredKvCache::new(1, 1, 4, config);

        for _pos in 0..3 {
            let k = vec![vec![0.0; 4]];
            let v = vec![vec![0.0; 4]];
            cache.append_layer(0, &k, &v);
            cache.advance_position();
        }

        // 3 positions * 4 dims * 4 bytes * 2 (K+V) = 96 bytes
        assert_eq!(cache.t1_bytes_used(), 96);
    }

    // ── Unified Eviction Policy Tests ───────────────────────────────

    #[test]
    fn test_eviction_policy_low_pressure() {
        let policy = UnifiedEvictionPolicy::new(0.15, 0.10);
        let rec = policy.evaluate(50, 100, 0);
        assert_eq!(rec.pressure, MemoryPressure::Low);
        assert!(!rec.evict_weights);
        assert!(!rec.evict_kv);
    }

    #[test]
    fn test_eviction_policy_high_pressure() {
        let policy = UnifiedEvictionPolicy::new(0.15, 0.10);
        // 96 weight pages + some KV in a 100-slot T1
        let kv_bytes = 4 * PAGE_SIZE; // 4 pages worth of KV
        let rec = policy.evaluate(96, 100, kv_bytes);
        assert!(matches!(
            rec.pressure,
            MemoryPressure::High | MemoryPressure::Critical
        ));
        // Should recommend evicting something
        assert!(rec.evict_weights || rec.evict_kv);
        assert!(rec.weight_pages_to_evict > 0 || rec.kv_positions_to_demote > 0);
    }

    #[test]
    fn test_eviction_policy_kv_over_budget() {
        let policy = UnifiedEvictionPolicy::new(0.15, 0.10);
        // KV using 30 pages (0.30 fraction) in a 100-slot T1, way over 0.15 budget
        let kv_bytes = 30 * PAGE_SIZE;
        let rec = policy.evaluate(70, 100, kv_bytes);
        // Should prefer evicting KV since it's over budget
        assert!(rec.evict_kv);
        assert!(rec.kv_positions_to_demote > 0);
    }

    #[test]
    fn test_eviction_policy_empty() {
        let policy = UnifiedEvictionPolicy::new(0.15, 0.10);
        let rec = policy.evaluate(0, 0, 0);
        assert_eq!(rec.pressure, MemoryPressure::Low);
    }
}
