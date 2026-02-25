//! HNSW-based ANN backend using usearch.
//!
//! Provides an embedded HNSW (Hierarchical Navigable Small World) graph index
//! for approximate nearest-neighbor search. This replaces brute-force O(n) scans
//! with O(log n) search, critical for two use cases:
//!
//! 1. **Page-level weight-space retrieval (Section 2.4):** With ~45,000 weight pages
//!    for Kimi K2.5, brute-force scan at 2-10μs becomes 90-450ms per token across
//!    all layers. HNSW reduces this to <1ms total.
//!
//! 2. **KV cache retrieval at scale (Section 8):** At 100K+ context positions,
//!    brute-force KvIndex becomes the bottleneck. HNSW keeps retrieval at ~10μs
//!    per head per layer regardless of context length.
//!
//! The backend wraps `usearch::Index` and implements the `AnnBackend` trait,
//! plugging into `VectorIndex` via `SearchBackend::Custom(Box<dyn AnnBackend>)`.
//! usearch is compiled into the binary — no external server, no network round-trips.
//!
//! ## Filtered Search
//!
//! usearch supports predicate-based filtered search (`filtered_search`), enabling
//! queries like "nearest weight pages WHERE layer == 5". This replaces the separate
//! `DomainClassifier` for domain-aware routing with a metadata predicate on the
//! same HNSW search.

use crate::index::vector_index::{AnnBackend, NnResult};

// ─── Configuration ──────────────────────────────────────────────────────

/// Configuration for building an HNSW index.
#[derive(Clone, Debug)]
pub struct HnswConfig {
    /// Distance metric: L2 (Euclidean) or Cosine.
    pub metric: HnswMetric,
    /// HNSW connectivity parameter (M). Higher values give better recall
    /// at the cost of memory and build time. Default: 16.
    pub connectivity: usize,
    /// Expansion factor during index construction. Higher values give
    /// better index quality at the cost of build time. Default: 128.
    pub expansion_add: usize,
    /// Expansion factor during search. Higher values give better recall
    /// at the cost of search latency. Default: 64.
    pub expansion_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            metric: HnswMetric::L2,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        }
    }
}

/// Distance metric for HNSW search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HnswMetric {
    /// Squared Euclidean distance (L2sq). Matches BruteForceBackend.
    L2,
    /// Cosine distance (1 - cosine_similarity).
    Cosine,
    /// Inner product (negative dot product as distance).
    InnerProduct,
}

impl HnswMetric {
    fn to_usearch(self) -> usearch::ffi::MetricKind {
        match self {
            HnswMetric::L2 => usearch::MetricKind::L2sq,
            HnswMetric::Cosine => usearch::MetricKind::Cos,
            HnswMetric::InnerProduct => usearch::MetricKind::IP,
        }
    }
}

// ─── HNSW Backend ───────────────────────────────────────────────────────

/// HNSW-based ANN backend wrapping `usearch::Index`.
///
/// Stores centroids alongside the usearch index because usearch does not
/// expose stored vectors through its API — the `centroid()` trait method
/// needs access to the original vectors for confidence scoring.
pub struct HnswBackend {
    /// The usearch HNSW index.
    index: usearch::Index,
    /// Original centroid vectors, kept for the `centroid()` method.
    /// Indexed by the same key used in usearch (0..n).
    centroids: Vec<Vec<f32>>,
    /// Number of dimensions per vector.
    dimensions: usize,
}

impl HnswBackend {
    /// Build an HNSW index from a set of centroid vectors.
    ///
    /// Each centroid is added to the usearch index with its array index as
    /// the key. The centroids are also stored for the `centroid()` method.
    ///
    /// # Errors
    ///
    /// Returns `None` if the centroid list is empty or if usearch fails to
    /// create the index.
    pub fn new(centroids: Vec<Vec<f32>>, config: &HnswConfig) -> Option<Self> {
        if centroids.is_empty() {
            return None;
        }

        let dimensions = centroids[0].len();
        if dimensions == 0 {
            return None;
        }

        let options = usearch::IndexOptions {
            dimensions,
            metric: config.metric.to_usearch(),
            connectivity: config.connectivity,
            quantization: usearch::ScalarKind::F32,
            ..Default::default()
        };

        let index = usearch::Index::new(&options).ok()?;
        index.reserve(centroids.len()).ok()?;
        index.change_expansion_add(config.expansion_add);
        index.change_expansion_search(config.expansion_search);

        for (i, centroid) in centroids.iter().enumerate() {
            if centroid.len() != dimensions {
                continue; // skip malformed vectors
            }
            index.add(i as u64, centroid).ok()?;
        }

        Some(Self {
            index,
            centroids,
            dimensions,
        })
    }

    /// Build from centroids with default configuration.
    pub fn with_defaults(centroids: Vec<Vec<f32>>) -> Option<Self> {
        Self::new(centroids, &HnswConfig::default())
    }

    /// Wrap a pre-built usearch index with its centroids.
    ///
    /// The caller must ensure that the centroids match the vectors stored
    /// in the index (same order, same keys 0..n).
    pub fn from_parts(index: usearch::Index, centroids: Vec<Vec<f32>>) -> Self {
        let dimensions = if centroids.is_empty() {
            index.dimensions()
        } else {
            centroids[0].len()
        };
        Self {
            index,
            centroids,
            dimensions,
        }
    }

    /// Number of dimensions per vector.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Memory usage of the HNSW graph (approximate lower bound).
    pub fn memory_usage(&self) -> usize {
        self.index.memory_usage()
    }

    /// Perform filtered search: find the k nearest vectors satisfying a predicate.
    ///
    /// The predicate receives the key (centroid index as u64) and returns true
    /// if the vector should be included in results.
    ///
    /// This enables queries like:
    /// - "nearest centroids WHERE layer == 5"
    /// - "nearest pages WHERE domain == 'code'"
    pub fn filtered_search_k<F>(&self, query: &[f32], k: usize, predicate: F) -> Vec<NnResult>
    where
        F: Fn(u64) -> bool,
    {
        if self.centroids.is_empty() || k == 0 {
            return vec![];
        }

        match self.index.filtered_search(query, k, predicate) {
            Ok(matches) => matches
                .keys
                .iter()
                .zip(matches.distances.iter())
                .map(|(&key, &dist)| NnResult {
                    index: key as usize,
                    distance: dist,
                })
                .collect(),
            Err(_) => vec![],
        }
    }

    /// Serialize the HNSW index to a byte buffer for storage in .vib3 files.
    pub fn save_to_buffer(&self) -> Option<Vec<u8>> {
        let len = self.index.serialized_length();
        let mut buffer = vec![0u8; len];
        self.index.save_to_buffer(&mut buffer).ok()?;
        Some(buffer)
    }

    /// Load an HNSW index from a byte buffer.
    ///
    /// The centroids must be provided separately (they are not stored in the
    /// usearch serialization format).
    pub fn load_from_buffer(
        buffer: &[u8],
        centroids: Vec<Vec<f32>>,
        config: &HnswConfig,
    ) -> Option<Self> {
        if centroids.is_empty() {
            return None;
        }

        let dimensions = centroids[0].len();
        let options = usearch::IndexOptions {
            dimensions,
            metric: config.metric.to_usearch(),
            connectivity: config.connectivity,
            quantization: usearch::ScalarKind::F32,
            ..Default::default()
        };

        let index = usearch::Index::new(&options).ok()?;
        index.load_from_buffer(buffer).ok()?;
        index.change_expansion_search(config.expansion_search);

        Some(Self {
            index,
            centroids,
            dimensions,
        })
    }
}

impl AnnBackend for HnswBackend {
    fn search(&self, query: &[f32]) -> NnResult {
        if self.centroids.is_empty() {
            return NnResult {
                index: 0,
                distance: f32::MAX,
            };
        }

        match self.index.search(query, 1) {
            Ok(matches) if !matches.keys.is_empty() => NnResult {
                index: matches.keys[0] as usize,
                distance: matches.distances[0],
            },
            _ => NnResult {
                index: 0,
                distance: f32::MAX,
            },
        }
    }

    fn search_k(&self, query: &[f32], k: usize) -> Vec<NnResult> {
        if self.centroids.is_empty() || k == 0 {
            return vec![];
        }

        match self.index.search(query, k) {
            Ok(matches) => matches
                .keys
                .iter()
                .zip(matches.distances.iter())
                .map(|(&key, &dist)| NnResult {
                    index: key as usize,
                    distance: dist,
                })
                .collect(),
            Err(_) => vec![],
        }
    }

    fn len(&self) -> usize {
        self.centroids.len()
    }

    fn backend_name(&self) -> &str {
        "hnsw-usearch"
    }

    fn centroid(&self, index: usize) -> Option<&[f32]> {
        self.centroids.get(index).map(|v| v.as_slice())
    }
}

// ─── Page Signatures ────────────────────────────────────────────────────

/// Method for computing a representative signature vector for a weight page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignatureMethod {
    /// Mean of weight rows in the page. Cheap, coarse.
    Mean,
    // Future: SVD first principal component, activation-based signatures.
}

/// Compute a signature vector for a weight page.
///
/// The signature is a low-dimensional representative vector used to index
/// the page in the HNSW graph. During inference, the hidden state is compared
/// against page signatures to retrieve the most relevant weight pages.
///
/// # Arguments
///
/// * `page_data` - Raw weight data (f32 values, row-major).
/// * `rows` - Number of rows in the page slice.
/// * `cols` - Number of columns per row.
/// * `target_dim` - Desired signature dimension (will truncate or pad).
/// * `method` - Signature computation method.
pub fn compute_page_signature(
    page_data: &[f32],
    rows: usize,
    cols: usize,
    target_dim: usize,
    method: SignatureMethod,
) -> Vec<f32> {
    match method {
        SignatureMethod::Mean => compute_mean_signature(page_data, rows, cols, target_dim),
    }
}

/// Mean-based page signature: average of all row vectors, projected to target_dim.
///
/// For a page with rows of dimension `cols`, this computes the element-wise mean
/// across all rows, then truncates or zero-pads to `target_dim`.
fn compute_mean_signature(
    page_data: &[f32],
    rows: usize,
    cols: usize,
    target_dim: usize,
) -> Vec<f32> {
    if rows == 0 || cols == 0 {
        return vec![0.0; target_dim];
    }

    let effective_rows = (page_data.len() / cols).min(rows);
    if effective_rows == 0 {
        return vec![0.0; target_dim];
    }

    // Compute mean of each column
    let mut mean = vec![0.0f64; cols];
    for row in 0..effective_rows {
        let start = row * cols;
        let end = (start + cols).min(page_data.len());
        for (j, val) in page_data[start..end].iter().enumerate() {
            mean[j] += *val as f64;
        }
    }
    let inv = 1.0 / effective_rows as f64;
    for m in &mut mean {
        *m *= inv;
    }

    // Truncate or pad to target_dim
    let mut signature = vec![0.0f32; target_dim];
    let copy_len = target_dim.min(cols);
    for i in 0..copy_len {
        signature[i] = mean[i] as f32;
    }

    // L2-normalize the signature for cosine compatibility
    let norm: f32 = signature.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for s in &mut signature {
            *s /= norm;
        }
    }

    signature
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_basic_construction() {
        let centroids = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let backend = HnswBackend::with_defaults(centroids.clone()).unwrap();
        assert_eq!(backend.len(), 4);
        assert_eq!(backend.dimensions(), 4);
        assert_eq!(backend.backend_name(), "hnsw-usearch");
    }

    #[test]
    fn test_hnsw_search_finds_nearest() {
        let centroids = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let backend = HnswBackend::with_defaults(centroids).unwrap();

        // Query close to centroid 0
        let result = backend.search(&[0.9, 0.1, 0.0]);
        assert_eq!(result.index, 0, "should find centroid 0 as nearest");

        // Query close to centroid 2
        let result = backend.search(&[0.0, 0.1, 0.9]);
        assert_eq!(result.index, 2, "should find centroid 2 as nearest");
    }

    #[test]
    fn test_hnsw_search_k() {
        let centroids = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let backend = HnswBackend::with_defaults(centroids).unwrap();

        // Search for top 2 nearest to [1.0, 0.0, 0.0]
        let results = backend.search_k(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // Centroids 0 and 1 should be the nearest
        let indices: Vec<usize> = results.iter().map(|r| r.index).collect();
        assert!(indices.contains(&0), "should contain centroid 0");
        assert!(indices.contains(&1), "should contain centroid 1");
    }

    #[test]
    fn test_hnsw_centroid_access() {
        let centroids = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let backend = HnswBackend::with_defaults(centroids.clone()).unwrap();

        assert_eq!(backend.centroid(0), Some(vec![1.0, 2.0, 3.0].as_slice()));
        assert_eq!(backend.centroid(1), Some(vec![4.0, 5.0, 6.0].as_slice()));
        assert_eq!(backend.centroid(2), None);
    }

    #[test]
    fn test_hnsw_empty_returns_none() {
        let result = HnswBackend::with_defaults(vec![]);
        assert!(result.is_none(), "empty centroids should return None");
    }

    #[test]
    fn test_hnsw_filtered_search() {
        let centroids = vec![
            vec![1.0, 0.0, 0.0], // key 0
            vec![0.9, 0.1, 0.0], // key 1
            vec![0.0, 1.0, 0.0], // key 2
            vec![0.0, 0.0, 1.0], // key 3
        ];

        let backend = HnswBackend::with_defaults(centroids).unwrap();

        // Search but exclude keys 0 and 1 (only allow even keys >= 2)
        let results = backend.filtered_search_k(&[1.0, 0.0, 0.0], 2, |key| key >= 2);

        // Should only return centroids 2 and 3
        assert!(!results.is_empty());
        for r in &results {
            assert!(
                r.index >= 2,
                "filtered result should only include keys >= 2, got {}",
                r.index
            );
        }
    }

    #[test]
    fn test_hnsw_matches_brute_force() {
        // Verify HNSW returns the same nearest as brute-force
        use crate::index::vector_index::BruteForceBackend;

        let centroids = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![0.9, 0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.9, 0.1],
            vec![0.2, 0.8, 0.1, 0.7],
        ];

        let bf = BruteForceBackend::new(centroids.clone());
        let hnsw = HnswBackend::with_defaults(centroids).unwrap();

        let queries = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        for query in &queries {
            let bf_result = bf.search(query);
            let hnsw_result = hnsw.search(query);
            assert_eq!(
                bf_result.index, hnsw_result.index,
                "HNSW and brute-force should agree on nearest for query {:?}",
                query
            );
        }
    }

    #[test]
    fn test_hnsw_serialization_roundtrip() {
        let centroids = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let config = HnswConfig::default();
        let backend = HnswBackend::new(centroids.clone(), &config).unwrap();

        // Serialize
        let buffer = backend
            .save_to_buffer()
            .expect("serialization should succeed");
        assert!(!buffer.is_empty());

        // Deserialize
        let restored = HnswBackend::load_from_buffer(&buffer, centroids, &config)
            .expect("deserialization should succeed");

        // Verify search still works
        let result = restored.search(&[0.9, 0.1, 0.0]);
        assert_eq!(result.index, 0, "restored index should find centroid 0");
    }

    #[test]
    fn test_page_signature_mean() {
        // 3 rows x 4 cols
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0, 6.0];

        let sig = compute_page_signature(&data, 3, 4, 4, SignatureMethod::Mean);
        assert_eq!(sig.len(), 4);

        // Mean should be [3.0, 4.0, 5.0, 6.0] before normalization
        let norm = (9.0 + 16.0 + 25.0 + 36.0f32).sqrt();
        let expected = vec![3.0 / norm, 4.0 / norm, 5.0 / norm, 6.0 / norm];
        for (a, b) in sig.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "signature mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_page_signature_truncation() {
        // 2 rows x 8 cols, target_dim = 4 (should truncate)
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ];

        let sig = compute_page_signature(&data, 2, 8, 4, SignatureMethod::Mean);
        assert_eq!(sig.len(), 4);
        // Should only use first 4 columns of the mean
    }

    #[test]
    fn test_page_signature_padding() {
        // 2 rows x 2 cols, target_dim = 4 (should pad with zeros)
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let sig = compute_page_signature(&data, 2, 2, 4, SignatureMethod::Mean);
        assert_eq!(sig.len(), 4);
        // Mean is [2.0, 3.0], padded to [2.0, 3.0, 0.0, 0.0], then normalized
        // After normalization: only first 2 dims are non-zero
        assert!((sig[2]).abs() < 1e-12, "padding should be zero");
        assert!((sig[3]).abs() < 1e-12, "padding should be zero");
    }

    #[test]
    fn test_page_signature_empty() {
        let sig = compute_page_signature(&[], 0, 0, 4, SignatureMethod::Mean);
        assert_eq!(sig.len(), 4);
        assert!(
            sig.iter().all(|&x| x == 0.0),
            "empty page should produce zero signature"
        );
    }

    #[test]
    fn test_hnsw_cosine_metric() {
        let centroids = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let config = HnswConfig {
            metric: HnswMetric::Cosine,
            ..Default::default()
        };

        let backend = HnswBackend::new(centroids, &config).unwrap();

        // Cosine: [0.9, 0.1, 0.0] is closest to [1.0, 0.0, 0.0]
        let result = backend.search(&[0.9, 0.1, 0.0]);
        assert_eq!(result.index, 0);
    }

    #[test]
    fn test_hnsw_memory_usage() {
        let centroids = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let backend = HnswBackend::with_defaults(centroids).unwrap();
        assert!(
            backend.memory_usage() > 0,
            "memory usage should be non-zero"
        );
    }
}
