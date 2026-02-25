//! Vector Index — Predictive expert activation and page-level lookup.
//!
//! The vector index maps regions of embedding space to predicted expert
//! activations and the specific weight pages they need. This enables
//! speculative prefetching: before a token even reaches an MoE layer,
//! we can predict which experts it will activate and start loading their
//! weight pages from NVMe into RAM or VRAM.
//!
//! ## ANN Backend
//!
//! The nearest-neighbor search is abstracted behind the [`AnnBackend`] trait.
//! The default path uses [`BruteForceBackend`] with no dynamic dispatch —
//! the compiler inlines the L2 scan directly. This is the right choice for
//! <10K centroids (typical for expert activation clustering).
//!
//! For larger indexes or the KV cache ANN search (Section 8), implement
//! the [`AnnBackend`] trait with an HNSW backend (usearch, Faiss) and pass
//! it via [`VectorIndex::load_with_backend`]. This path uses dynamic dispatch
//! (`Box<dyn AnnBackend>`) — acceptable since the backend-swap case is
//! already I/O-dominated.

use crate::core::config::ModelConfig;
use crate::core::error::{Error, Result};
use crate::core::types::*;
use crate::storage::format::{PageCatalogEntry, VectorIndexEntry, Vib3File};

// ─── ANN Backend Trait ──────────────────────────────────────────────────

/// Result of a nearest-neighbor search: (centroid_index, squared_distance).
#[derive(Clone, Copy, Debug)]
pub struct NnResult {
    pub index: usize,
    pub distance: f32,
}

/// Pluggable approximate nearest-neighbor search backend.
///
/// Implement this trait to swap in HNSW (usearch), IVF (Faiss), or other
/// ANN algorithms. The default [`BruteForceBackend`] is used when no
/// custom backend is provided.
///
/// # Performance Contract
///
/// The backend is called on the hot path: ~9,000 calls/sec during decode.
/// For the expert centroid use case (<10K centroids), brute-force is fine
/// (~2-10μs per call). The trait exists primarily for:
/// 1. The KV cache ANN search (Section 8) over millions of vectors
/// 2. Per-layer filtered search ("nearest centroid WHERE layer == 5")
/// 3. Very large centroid counts (>50K) from fine-grained profiling
pub trait AnnBackend: Send + Sync {
    /// Find the nearest centroid to the query vector.
    fn search(&self, query: &[f32]) -> NnResult;

    /// Find the k nearest centroids (optional, for multi-probe).
    fn search_k(&self, query: &[f32], _k: usize) -> Vec<NnResult> {
        vec![self.search(query)]
    }

    /// Number of centroids in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Human-readable backend name (for logging/diagnostics).
    fn backend_name(&self) -> &str;

    /// Access a centroid vector (needed for confidence scoring).
    /// Returns None if the backend doesn't store centroids directly.
    fn centroid(&self, index: usize) -> Option<&[f32]>;
}

// ─── Brute-Force Backend ────────────────────────────────────────────────

/// Brute-force O(n) nearest-neighbor search over centroids.
///
/// The default and recommended backend for expert activation prediction.
/// For <10K centroids at 64-256 dims, this is 2-10μs per search — well
/// within budget. No allocation on the search path.
pub struct BruteForceBackend {
    centroids: Vec<Vec<f32>>,
}

impl BruteForceBackend {
    pub fn new(centroids: Vec<Vec<f32>>) -> Self {
        Self { centroids }
    }

    /// Direct search without vtable indirection (used by VectorIndex
    /// when no custom backend is configured).
    #[inline]
    fn search_direct(&self, query: &[f32]) -> NnResult {
        if self.centroids.is_empty() {
            return NnResult {
                index: 0,
                distance: f32::MAX,
            };
        }

        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist: f32 = query
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        NnResult {
            index: best_idx,
            distance: best_dist,
        }
    }
}

impl AnnBackend for BruteForceBackend {
    fn search(&self, query: &[f32]) -> NnResult {
        self.search_direct(query)
    }

    fn search_k(&self, query: &[f32], k: usize) -> Vec<NnResult> {
        if self.centroids.is_empty() || k == 0 {
            return vec![];
        }

        let mut results: Vec<NnResult> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let dist: f32 = query
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| {
                        let d = a - b;
                        d * d
                    })
                    .sum();
                NnResult {
                    index: i,
                    distance: dist,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    fn len(&self) -> usize {
        self.centroids.len()
    }

    fn backend_name(&self) -> &str {
        "brute-force"
    }

    fn centroid(&self, index: usize) -> Option<&[f32]> {
        self.centroids.get(index).map(|v| v.as_slice())
    }
}

// ─── Search Backend Enum ────────────────────────────────────────────────

/// The search backend used by VectorIndex.
///
/// This enum avoids dynamic dispatch on the default (brute-force) path
/// while still allowing custom backends via `Box<dyn AnnBackend>`.
/// On the hot path with the default backend, the compiler sees through
/// the enum match and inlines the brute-force search directly.
enum SearchBackend {
    /// Default: brute-force L2 scan, no dynamic dispatch.
    BruteForce(BruteForceBackend),
    /// Custom backend (usearch, Faiss, etc.) via dynamic dispatch.
    Custom(Box<dyn AnnBackend>),
}

impl SearchBackend {
    #[inline]
    fn search(&self, query: &[f32]) -> NnResult {
        match self {
            SearchBackend::BruteForce(bf) => bf.search_direct(query),
            SearchBackend::Custom(backend) => backend.search(query),
        }
    }

    fn len(&self) -> usize {
        match self {
            SearchBackend::BruteForce(bf) => bf.centroids.len(),
            SearchBackend::Custom(backend) => backend.len(),
        }
    }

    fn backend_name(&self) -> &str {
        match self {
            SearchBackend::BruteForce(_) => "brute-force",
            SearchBackend::Custom(backend) => backend.backend_name(),
        }
    }

    fn centroid(&self, index: usize) -> Option<&[f32]> {
        match self {
            SearchBackend::BruteForce(bf) => bf.centroids.get(index).map(|v| v.as_slice()),
            SearchBackend::Custom(backend) => backend.centroid(index),
        }
    }
}

// ─── Activation Profile Types ───────────────────────────────────────────

/// Predicted expert activations for an embedding region.
#[derive(Clone, Debug)]
pub struct ActivationProfile {
    pub layers: Vec<LayerProfile>,
    pub domain_id: u32,
    pub domain_name: String,
    pub domain_confidence: f32,
    pub recommended_view: Option<String>,
}

#[derive(Clone, Debug)]
pub struct LayerProfile {
    pub layer: u16,
    pub experts: Vec<ExpertPrediction>,
    pub predicted_pages: Vec<PageId>,
    pub coverage: f32,
}

#[derive(Clone, Debug)]
pub struct ExpertPrediction {
    pub expert_id: u16,
    pub probability: f32,
}

// ─── Parsing Helpers ────────────────────────────────────────────────────

/// Parse VectorIndexEntry from potentially unaligned bytes.
fn parse_entries_unaligned(bytes: &[u8], entry_size: usize) -> Vec<VectorIndexEntry> {
    let count = bytes.len() / entry_size;
    if count == 0 {
        return vec![];
    }
    let needed = count * entry_size;
    let mut aligned = vec![0u8; needed];
    aligned[..needed].copy_from_slice(&bytes[..needed]);
    bytemuck::cast_slice(&aligned).to_vec()
}

/// Parse centroids from the binary section of a .vib3 file.
#[allow(clippy::type_complexity)]
fn parse_centroids(bytes: &[u8]) -> Option<(Vec<Vec<f32>>, usize, &[u8])> {
    if bytes.len() < 8 {
        return None;
    }

    let centroid_count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let dim = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let centroid_data_size = centroid_count * dim * 4;
    let header_size = 8 + centroid_data_size;
    let entry_size = std::mem::size_of::<VectorIndexEntry>();

    if dim == 0
        || centroid_count == 0
        || centroid_count > 100_000
        || dim > 65536
        || header_size + centroid_count * entry_size > bytes.len()
    {
        return None;
    }

    let centroid_bytes = &bytes[8..8 + centroid_data_size];
    let centroids: Vec<Vec<f32>> = (0..centroid_count)
        .map(|i| {
            (0..dim)
                .map(|d| {
                    let off = (i * dim + d) * 4;
                    f32::from_le_bytes([
                        centroid_bytes[off],
                        centroid_bytes[off + 1],
                        centroid_bytes[off + 2],
                        centroid_bytes[off + 3],
                    ])
                })
                .collect()
        })
        .collect();

    Some((centroids, dim, &bytes[header_size..]))
}

// ─── Vector Index ───────────────────────────────────────────────────────

/// The vector index: maps embedding space → expert activations → pages.
///
/// Default path uses brute-force L2 scan with no dynamic dispatch.
/// Custom ANN backends (usearch, Faiss) can be plugged in via
/// [`load_with_backend`](Self::load_with_backend) for the rare case
/// where centroid count exceeds ~10K or filtered search is needed.
pub struct VectorIndex {
    entries: Vec<VectorIndexEntry>,
    backend: SearchBackend,
    model_config: ModelConfig,
    page_catalog: Vec<PageCatalogEntry>,

    // ── Phase C: Gear-filtered HNSW search ──────────────────────────
    /// Domain tags per centroid index. Maps centroid_idx → list of domain tags
    /// (e.g., ["code", "math"]). When a gear filter is active, only centroids
    /// whose domain tags include the active gear's domains are returned.
    /// Empty means no filtering available (all centroids match all gears).
    domain_tags: std::collections::HashMap<usize, Vec<String>>,
}

impl VectorIndex {
    /// Load from a .vib3 file using brute-force search (default, zero-cost).
    pub fn load(file: &Vib3File) -> Result<Self> {
        if !file.has_vector_index() {
            return Err(Error::NoVectorIndex);
        }

        let bytes = file.vector_index_bytes();
        let entry_size = std::mem::size_of::<VectorIndexEntry>();

        let (centroids, entries) =
            if let Some((centroids, _dim, remaining)) = parse_centroids(bytes) {
                let entries = parse_entries_unaligned(remaining, entry_size);
                (centroids, entries)
            } else {
                let entries = parse_entries_unaligned(bytes, entry_size);
                (vec![], entries)
            };

        Ok(Self {
            entries,
            backend: SearchBackend::BruteForce(BruteForceBackend::new(centroids)),
            model_config: file.model_config().clone(),
            page_catalog: file.page_catalog().to_vec(),
            domain_tags: std::collections::HashMap::new(),
        })
    }

    /// Load from a .vib3 file with a custom ANN backend.
    ///
    /// The custom backend should already be built from the same centroids
    /// stored in the file. This path uses dynamic dispatch.
    pub fn load_with_backend(file: &Vib3File, backend: Box<dyn AnnBackend>) -> Result<Self> {
        if !file.has_vector_index() {
            return Err(Error::NoVectorIndex);
        }

        let bytes = file.vector_index_bytes();
        let entry_size = std::mem::size_of::<VectorIndexEntry>();

        let entries = if let Some((_centroids, _dim, remaining)) = parse_centroids(bytes) {
            parse_entries_unaligned(remaining, entry_size)
        } else {
            parse_entries_unaligned(bytes, entry_size)
        };

        Ok(Self {
            entries,
            backend: SearchBackend::Custom(backend),
            model_config: file.model_config().clone(),
            page_catalog: file.page_catalog().to_vec(),
            domain_tags: std::collections::HashMap::new(),
        })
    }

    /// Create from pre-built centroids and entries (default brute-force).
    pub fn from_parts(
        centroids: Vec<Vec<f32>>,
        entries: Vec<VectorIndexEntry>,
        model_config: ModelConfig,
        page_catalog: Vec<PageCatalogEntry>,
    ) -> Self {
        Self {
            entries,
            backend: SearchBackend::BruteForce(BruteForceBackend::new(centroids)),
            model_config,
            page_catalog,
            domain_tags: std::collections::HashMap::new(),
        }
    }

    /// Create with a custom ANN backend.
    pub fn from_parts_with_backend(
        entries: Vec<VectorIndexEntry>,
        backend: Box<dyn AnnBackend>,
        model_config: ModelConfig,
        page_catalog: Vec<PageCatalogEntry>,
    ) -> Self {
        Self {
            entries,
            backend: SearchBackend::Custom(backend),
            model_config,
            page_catalog,
            domain_tags: std::collections::HashMap::new(),
        }
    }

    // ── Phase C: Domain tags for gear-filtered search ────────────────

    /// Set domain tags for a centroid.
    ///
    /// Domain tags describe what task domains a centroid (and its associated
    /// pages) are relevant for. Used by gear-filtered search to narrow
    /// retrieval to the active gear's domains.
    pub fn set_domain_tags(&mut self, centroid_idx: usize, tags: Vec<String>) {
        self.domain_tags.insert(centroid_idx, tags);
    }

    /// Load domain tags from a gear profiles JSON.
    ///
    /// Maps expert IDs from gear profiles to centroid indices via the
    /// vector index entries. Each centroid that maps to an expert in a
    /// gear's hot set gets tagged with that gear's domain tags.
    pub fn load_domain_tags_from_gear_profiles(
        &mut self,
        gear_domains: &std::collections::HashMap<String, Vec<String>>,
        gear_profiles_json: &serde_json::Value,
    ) -> usize {
        let mut tagged = 0usize;

        let obj = match gear_profiles_json.as_object() {
            Some(o) => o,
            None => return 0,
        };

        for (gear_name, profile_val) in obj {
            let domains = match gear_domains.get(gear_name) {
                Some(d) => d.clone(),
                None => vec![gear_name.clone()], // Use gear name as domain if no mapping
            };

            let hot_experts = match profile_val.get("hot_experts").and_then(|v| v.as_array()) {
                Some(layers) => layers,
                None => continue,
            };

            // Collect all expert IDs across layers for this gear
            let mut expert_set = std::collections::HashSet::new();
            for layer_val in hot_experts {
                if let Some(experts) = layer_val.as_array() {
                    for e in experts {
                        if let Some(eid) = e.as_u64() {
                            expert_set.insert(eid as u16);
                        }
                    }
                }
            }

            // Tag centroids whose predicted experts overlap with this gear's hot set
            for (centroid_idx, entry) in self.entries.iter().enumerate() {
                let pred_count = { entry.prediction_count } as usize;
                let predictions = { entry.expert_predictions };
                let has_overlap = predictions[..pred_count.min(32)]
                    .iter()
                    .any(|&(eid, _prob)| expert_set.contains(&eid));

                if has_overlap {
                    let tags = self
                        .domain_tags
                        .entry(centroid_idx)
                        .or_insert_with(Vec::new);
                    for domain in &domains {
                        if !tags.contains(domain) {
                            tags.push(domain.clone());
                            tagged += 1;
                        }
                    }
                }
            }
        }

        tracing::info!(
            "Loaded {} domain tags across {} centroids",
            tagged,
            self.domain_tags.len(),
        );
        tagged
    }

    /// Check if a centroid matches any of the given domain tags.
    pub fn centroid_matches_domains(&self, centroid_idx: usize, domains: &[String]) -> bool {
        if domains.is_empty() {
            return true; // No filter = match all
        }
        match self.domain_tags.get(&centroid_idx) {
            Some(tags) => tags.iter().any(|t| domains.contains(t)),
            None => true, // No tags = unfiltered (matches all)
        }
    }

    /// Whether domain tags are loaded.
    pub fn has_domain_tags(&self) -> bool {
        !self.domain_tags.is_empty()
    }

    /// Number of centroids in the index.
    pub fn centroid_count(&self) -> usize {
        self.backend.len()
    }

    /// Number of entries (one per centroid).
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Name of the ANN backend in use.
    pub fn backend_name(&self) -> &str {
        self.backend.backend_name()
    }

    /// Predict expert activations and pages from an input embedding.
    pub fn predict(&self, embedding: &[f32]) -> ActivationProfile {
        if self.entries.is_empty() {
            return ActivationProfile {
                layers: vec![],
                domain_id: 0,
                domain_name: String::new(),
                domain_confidence: 0.0,
                recommended_view: None,
            };
        }

        // ── ANN lookup ──
        let nn = self.backend.search(embedding);
        let centroid_idx = nn.index;

        let entry = &self.entries[centroid_idx.min(self.entries.len() - 1)];

        let prediction_count = entry.prediction_count as usize;
        let predictions = entry.expert_predictions;
        let centroid_id = entry.centroid_id;

        let experts: Vec<ExpertPrediction> = predictions[..prediction_count]
            .iter()
            .map(|&(expert_id, prob_u8)| ExpertPrediction {
                expert_id,
                probability: prob_u8 as f32 / 255.0,
            })
            .collect();

        let hot_page_count = entry.hot_page_count as usize;
        let hot_pages = entry.hot_pages;
        let predicted_pages: Vec<PageId> = hot_pages[..hot_page_count]
            .iter()
            .filter_map(|&catalog_idx| {
                let idx = catalog_idx as usize;
                if idx < self.page_catalog.len() {
                    Some(self.page_catalog[idx].page_id())
                } else {
                    None
                }
            })
            .collect();

        let total_needed = experts.len() as f32
            * self.model_config.pages_per_expert() as f32
            * self.model_config.num_moe_layers as f32;
        let _coverage = if total_needed > 0.0 {
            (predicted_pages.len() as f32 / total_needed).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let num_moe_layers = self.model_config.num_moe_layers as usize;
        let num_experts = self.model_config.num_experts as usize;
        let mut layers: Vec<LayerProfile> = Vec::with_capacity(num_moe_layers);
        let dense_start = self.model_config.dense_layer_idx as u16;

        for l in 0..num_moe_layers {
            let layer_idx = dense_start + l as u16;

            let layer_pages: Vec<PageId> = predicted_pages
                .iter()
                .filter(|p| p.layer == layer_idx)
                .cloned()
                .collect();

            let layer_experts: Vec<ExpertPrediction> = experts
                .iter()
                .filter(|e| (e.expert_id as usize) < num_experts)
                .cloned()
                .collect();

            let layer_total =
                layer_experts.len() as f32 * self.model_config.pages_per_expert() as f32;
            let layer_coverage = if layer_total > 0.0 {
                (layer_pages.len() as f32 / layer_total).clamp(0.0, 1.0)
            } else {
                0.0
            };

            layers.push(LayerProfile {
                layer: layer_idx,
                experts: layer_experts,
                predicted_pages: layer_pages,
                coverage: layer_coverage,
            });
        }

        let domain_confidence = if let Some(nearest) = self.backend.centroid(centroid_idx) {
            cosine_similarity(embedding, nearest)
        } else {
            0.0
        };

        ActivationProfile {
            layers,
            domain_id: centroid_id,
            domain_name: format!("cluster_{}", centroid_id),
            domain_confidence: ((domain_confidence + 1.0) / 2.0).clamp(0.0, 1.0),
            recommended_view: None,
        }
    }

    /// Predict which specific pages are needed for a layer given router output.
    pub fn predict_pages(
        &self,
        layer: u16,
        activation: &ExpertActivation,
        file: &Vib3File,
    ) -> Vec<PageId> {
        let mut pages = Vec::new();
        for (expert_id, _weight) in &activation.experts {
            for entry in file.pages_for_expert(layer, *expert_id) {
                pages.push(entry.page_id());
            }
        }
        pages
    }

    /// Speculative prefetch: predict pages for future tokens.
    ///
    /// Uses trajectory of recent hidden states to extrapolate future
    /// expert activations via linear velocity estimation.
    pub fn speculative_prefetch(
        &self,
        recent_hidden_states: &[&[f32]],
        lookahead: usize,
    ) -> Vec<PrefetchRequest> {
        if recent_hidden_states.is_empty() {
            return vec![];
        }

        let predicted = if recent_hidden_states.len() >= 2 {
            let n = recent_hidden_states.len();
            let last = recent_hidden_states[n - 1];
            let prev = recent_hidden_states[n - 2];
            let velocity: Vec<f32> = last.iter().zip(prev.iter()).map(|(a, b)| a - b).collect();
            let scale = (lookahead as f32).min(3.0) * 0.5;
            last.iter()
                .zip(velocity.iter())
                .map(|(l, v)| l + v * scale)
                .collect::<Vec<f32>>()
        } else {
            recent_hidden_states[0].to_vec()
        };

        let profile = self.predict(&predicted);

        let mut requests: Vec<PrefetchRequest> = Vec::new();

        for layer_profile in &profile.layers {
            for (page_idx, page) in layer_profile.predicted_pages.iter().enumerate() {
                let priority = if page_idx < 4 {
                    PrefetchPriority::High
                } else if page_idx < 12 {
                    PrefetchPriority::Medium
                } else {
                    PrefetchPriority::Low
                };

                let confidence = layer_profile.coverage
                    * profile.domain_confidence
                    * (1.0 / (page_idx as f32 + 1.0));

                requests.push(PrefetchRequest {
                    page: *page,
                    source: Tier::T3Nvme,
                    dest: Tier::T2Ram,
                    priority,
                    deadline_tick: 0,
                    confidence: confidence.clamp(0.0, 1.0),
                });
            }
        }

        requests.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let max_requests = (lookahead * 16).max(32);
        requests.truncate(max_requests);

        requests
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
