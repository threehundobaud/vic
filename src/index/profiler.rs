//! Expert activation profiler.
//!
//! Records expert activations during a calibration pass over a dataset.
//! Produces:
//!   - Per-layer activation frequency histograms
//!   - Coactivation pairs (which experts fire together)
//!   - Centroid clusters for the vector index
//!
//! Usage during model conversion:
//! ```ignore
//! let mut profiler = ActivationProfiler::new(num_layers, num_experts, hidden_dim);
//!
//! for token_embedding in calibration_data {
//!     // Simulate or record router decisions per layer
//!     for layer in 0..num_layers {
//!         let active = router.route(layer, &hidden_state);
//!         profiler.record(layer, &active, &token_embedding);
//!     }
//! }
//!
//! let (centroids, entries) = profiler.build_vector_index(num_clusters);
//! let coactivation_entries = profiler.build_coactivation(min_correlation);
//! ```

use crate::storage::format::{CoactivationEntry, VectorIndexEntry};
use std::collections::HashMap;

/// Records expert activations during a calibration pass.
pub struct ActivationProfiler {
    num_layers: usize,
    num_experts: usize,
    hidden_dim: usize,

    /// Per-layer, per-expert activation count.
    /// `activation_counts[layer][expert]`
    activation_counts: Vec<Vec<u64>>,

    /// Per-layer coactivation pair counts.
    /// Key: `(layer, min(expert_a, expert_b), max(expert_a, expert_b))` → count
    coactivation_counts: HashMap<(u16, u16, u16), u64>,

    /// Collected (embedding, active_experts) samples per layer for clustering.
    /// We downsample to avoid memory blow-up.
    samples: Vec<ProfileSample>,

    /// Total tokens profiled.
    total_tokens: u64,

    /// Maximum number of samples to keep (reservoir sampling).
    max_samples: usize,

    /// RNG state for reservoir sampling.
    rng_state: u64,
}

/// One profiling sample: an embedding vector and its active experts at each layer.
struct ProfileSample {
    /// The input embedding (or hidden state at the router).
    embedding: Vec<f32>,
    /// Per-layer active expert IDs.
    layer_experts: Vec<Vec<u16>>,
}

impl ActivationProfiler {
    /// Create a new profiler.
    ///
    /// - `num_layers`: number of MoE layers to profile
    /// - `num_experts`: experts per layer
    /// - `hidden_dim`: embedding dimension (for centroid building)
    pub fn new(num_layers: usize, num_experts: usize, hidden_dim: usize) -> Self {
        Self {
            num_layers,
            num_experts,
            hidden_dim,
            activation_counts: vec![vec![0u64; num_experts]; num_layers],
            coactivation_counts: HashMap::new(),
            samples: Vec::new(),
            total_tokens: 0,
            max_samples: 10_000, // Keep up to 10K samples for clustering
            rng_state: 0xCAFE_BABE_DEAD_BEEF,
        }
    }

    /// Set the maximum number of samples to keep for clustering.
    pub fn set_max_samples(&mut self, max: usize) {
        self.max_samples = max;
    }

    /// Record one token's expert activations at one layer.
    ///
    /// - `layer`: MoE layer index (0-based within MoE layers)
    /// - `active_experts`: expert IDs selected by the router
    /// - `embedding`: the input embedding or hidden state (for clustering)
    pub fn record(&mut self, layer: usize, active_experts: &[u16], embedding: &[f32]) {
        if layer >= self.num_layers {
            return;
        }

        // Count activations
        for &expert in active_experts {
            if (expert as usize) < self.num_experts {
                self.activation_counts[layer][expert as usize] += 1;
            }
        }

        // Count coactivation pairs
        for i in 0..active_experts.len() {
            for j in (i + 1)..active_experts.len() {
                let a = active_experts[i].min(active_experts[j]);
                let b = active_experts[i].max(active_experts[j]);
                *self
                    .coactivation_counts
                    .entry((layer as u16, a, b))
                    .or_insert(0) += 1;
            }
        }

        // Reservoir sampling for embedding samples
        if layer == 0 {
            // Only sample once per token (at layer 0)
            self.total_tokens += 1;

            let sample = ProfileSample {
                embedding: embedding.to_vec(),
                layer_experts: vec![active_experts.to_vec()],
            };

            if self.samples.len() < self.max_samples {
                self.samples.push(sample);
            } else {
                // Reservoir sampling: replace with probability max_samples/total_tokens
                self.rng_state ^= self.rng_state << 13;
                self.rng_state ^= self.rng_state >> 7;
                self.rng_state ^= self.rng_state << 17;
                let idx = (self.rng_state % self.total_tokens) as usize;
                if idx < self.max_samples {
                    self.samples[idx] = sample;
                }
            }
        } else if let Some(last) = self.samples.last_mut() {
            // Append this layer's activations to the most recent sample
            while last.layer_experts.len() <= layer {
                last.layer_experts.push(Vec::new());
            }
            last.layer_experts[layer] = active_experts.to_vec();
        }
    }

    /// Record a complete token with activations at all layers at once.
    pub fn record_token(&mut self, embedding: &[f32], all_layer_activations: &[Vec<u16>]) {
        for (layer, experts) in all_layer_activations.iter().enumerate() {
            self.record(layer, experts, embedding);
        }
    }

    /// Get activation frequency for a specific expert.
    ///
    /// Returns the fraction of tokens that activated this expert (0.0 to 1.0).
    pub fn expert_frequency(&self, layer: usize, expert: usize) -> f64 {
        if layer >= self.num_layers || expert >= self.num_experts || self.total_tokens == 0 {
            return 0.0;
        }
        self.activation_counts[layer][expert] as f64 / self.total_tokens as f64
    }

    /// Get the total number of tokens profiled.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Get the number of embedding samples collected.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    // ── Index Building ───────────────────────────────────────────────

    /// Build coactivation entries from profiling data.
    ///
    /// Returns entries where the correlation exceeds `min_correlation`.
    /// Correlation is computed as: `co_count / min(count_a, count_b)`,
    /// representing how often the pair fires together relative to the
    /// less-frequent expert.
    pub fn build_coactivation(&self, min_correlation: f32) -> Vec<CoactivationEntry> {
        let mut entries = Vec::new();

        for (&(layer, expert_a, expert_b), &count) in &self.coactivation_counts {
            let count_a = self.activation_counts[layer as usize][expert_a as usize];
            let count_b = self.activation_counts[layer as usize][expert_b as usize];

            if count_a == 0 || count_b == 0 {
                continue;
            }

            // Correlation: how often they co-fire relative to the rarer one
            let correlation = count as f32 / count_a.min(count_b) as f32;

            if correlation >= min_correlation {
                entries.push(CoactivationEntry {
                    expert_a,
                    expert_b,
                    layer,
                    _pad: 0,
                    correlation,
                    sample_count: count as u32,
                });
            }
        }

        // Sort by correlation descending
        entries.sort_by(|a, b| {
            let ca = { a.correlation };
            let cb = { b.correlation };
            cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
        });

        entries
    }

    /// Build vector index centroids and entries using k-means clustering.
    ///
    /// Returns `(centroids, entries)` suitable for `Vib3Writer::set_vector_index`.
    ///
    /// - `num_clusters`: number of centroids to compute (e.g., 64-256)
    /// - `max_iterations`: k-means iterations (10-20 is usually enough)
    pub fn build_vector_index(
        &self,
        num_clusters: usize,
        max_iterations: usize,
    ) -> (Vec<Vec<f32>>, Vec<VectorIndexEntry>) {
        if self.samples.is_empty() || num_clusters == 0 {
            return (vec![], vec![]);
        }

        let dim = self.hidden_dim;
        let k = num_clusters.min(self.samples.len());

        // Initialize centroids with k-means++ (simplified: take evenly spaced samples)
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        let step = self.samples.len() / k;
        for i in 0..k {
            let idx = (i * step).min(self.samples.len() - 1);
            let mut c = self.samples[idx].embedding.clone();
            c.resize(dim, 0.0);
            centroids.push(c);
        }

        // k-means iterations
        let mut assignments = vec![0usize; self.samples.len()];

        for _iter in 0..max_iterations {
            // Assign each sample to the nearest centroid
            let mut changed = false;
            for (i, sample) in self.samples.iter().enumerate() {
                let embedding = &sample.embedding;
                let mut best_dist = f32::MAX;
                let mut best_cluster = 0;

                for (c, centroid) in centroids.iter().enumerate() {
                    let dist: f32 = embedding
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| {
                            let d = a - b;
                            d * d
                        })
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Recompute centroids
            let mut new_centroids = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];

            for (i, sample) in self.samples.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (d, &val) in sample.embedding.iter().enumerate() {
                    if d < dim {
                        new_centroids[c][d] += val;
                    }
                }
            }

            for c in 0..k {
                if counts[c] > 0 {
                    for val in new_centroids[c].iter_mut().take(dim) {
                        *val /= counts[c] as f32;
                    }
                    centroids[c] = new_centroids[c].clone();
                }
            }
        }

        // Build VectorIndexEntry per cluster
        let mut entries: Vec<VectorIndexEntry> = Vec::with_capacity(k);

        for c in 0..k {
            // Collect all expert activations in this cluster
            let cluster_samples: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &a)| a == c)
                .map(|(i, _)| i)
                .collect();

            let cluster_size = cluster_samples.len();
            if cluster_size == 0 {
                // Empty cluster: push a zero entry
                entries.push(VectorIndexEntry {
                    centroid_id: c as u32,
                    cluster_size: 0,
                    prediction_count: 0,
                    hot_page_count: 0,
                    expert_predictions: [(0, 0); 32],
                    hot_pages: [0; 64],
                });
                continue;
            }

            // Count expert frequencies in this cluster (across all layers)
            let mut expert_freq: HashMap<u16, u32> = HashMap::new();
            for &sample_idx in &cluster_samples {
                let sample = &self.samples[sample_idx];
                for layer_experts in &sample.layer_experts {
                    for &expert in layer_experts {
                        *expert_freq.entry(expert).or_insert(0) += 1;
                    }
                }
            }

            // Top-32 experts by frequency
            let mut sorted_experts: Vec<(u16, u32)> = expert_freq.into_iter().collect();
            sorted_experts.sort_by(|a, b| b.1.cmp(&a.1));
            sorted_experts.truncate(32);

            let max_count = sorted_experts.first().map(|(_, c)| *c).unwrap_or(1);

            let mut expert_predictions = [(0u16, 0u8); 32];
            for (i, (expert_id, count)) in sorted_experts.iter().enumerate() {
                let prob_u8 = ((*count as f32 / max_count as f32) * 255.0) as u8;
                expert_predictions[i] = (*expert_id, prob_u8);
            }

            entries.push(VectorIndexEntry {
                centroid_id: c as u32,
                cluster_size: cluster_size as u16,
                prediction_count: sorted_experts.len() as u8,
                hot_page_count: 0, // Pages filled in later by the converter
                expert_predictions,
                hot_pages: [0; 64],
            });
        }

        (centroids, entries)
    }

    /// Get the per-layer activation frequency histogram.
    ///
    /// Returns `[num_layers][num_experts]` with values in [0.0, 1.0].
    pub fn frequency_histogram(&self) -> Vec<Vec<f64>> {
        self.activation_counts
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .map(|&count| {
                        if self.total_tokens > 0 {
                            count as f64 / self.total_tokens as f64
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Summary statistics for display.
    pub fn summary(&self) -> ProfileSummary {
        let mut most_active = Vec::new();
        let mut least_active = Vec::new();

        for layer in 0..self.num_layers {
            let mut sorted: Vec<(usize, u64)> = self.activation_counts[layer]
                .iter()
                .copied()
                .enumerate()
                .collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));

            if let Some((expert, count)) = sorted.first() {
                most_active.push((layer, *expert, *count));
            }
            if let Some((expert, count)) = sorted.last() {
                least_active.push((layer, *expert, *count));
            }
        }

        ProfileSummary {
            total_tokens: self.total_tokens,
            num_layers: self.num_layers,
            num_experts: self.num_experts,
            num_samples: self.samples.len(),
            num_coactivation_pairs: self.coactivation_counts.len(),
            most_active,
            least_active,
        }
    }
}

/// Summary statistics from profiling.
pub struct ProfileSummary {
    pub total_tokens: u64,
    pub num_layers: usize,
    pub num_experts: usize,
    pub num_samples: usize,
    pub num_coactivation_pairs: usize,
    /// (layer, expert_id, count) — most activated expert per layer.
    pub most_active: Vec<(usize, usize, u64)>,
    /// (layer, expert_id, count) — least activated expert per layer.
    pub least_active: Vec<(usize, usize, u64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic() {
        let mut profiler = ActivationProfiler::new(2, 4, 4);

        // Simulate 100 tokens
        for token in 0..100u32 {
            let embedding = vec![token as f32 / 100.0, (100 - token) as f32 / 100.0, 0.5, 0.5];

            // Layer 0: experts 0 and 1 always active
            profiler.record(0, &[0, 1], &embedding);

            // Layer 1: expert depends on token
            let expert = (token % 4) as u16;
            profiler.record(1, &[expert], &embedding);
        }

        assert_eq!(profiler.total_tokens(), 100);

        // Layer 0: experts 0 and 1 should each have freq ~1.0
        assert!((profiler.expert_frequency(0, 0) - 1.0).abs() < 0.01);
        assert!((profiler.expert_frequency(0, 1) - 1.0).abs() < 0.01);
        assert!((profiler.expert_frequency(0, 2) - 0.0).abs() < 0.01);

        // Layer 1: each expert should have freq ~0.25
        for e in 0..4 {
            assert!(
                (profiler.expert_frequency(1, e) - 0.25).abs() < 0.05,
                "Expert {} freq: {}",
                e,
                profiler.expert_frequency(1, e)
            );
        }
    }

    #[test]
    fn test_profiler_coactivation() {
        let mut profiler = ActivationProfiler::new(1, 4, 4);
        let embedding = vec![0.0; 4];

        // Experts 0 and 1 always fire together
        for _ in 0..50 {
            profiler.record(0, &[0, 1], &embedding);
        }
        // Expert 2 fires alone sometimes
        for _ in 0..50 {
            profiler.record(0, &[2], &embedding);
        }

        let coact = profiler.build_coactivation(0.5);
        // Should have the (0, 1) pair with high correlation
        assert!(!coact.is_empty());
        let pair_01 = coact.iter().find(|e| {
            let a = { e.expert_a };
            let b = { e.expert_b };
            (a == 0 && b == 1) || (a == 1 && b == 0)
        });
        assert!(
            pair_01.is_some(),
            "Should find coactivation between 0 and 1"
        );
        let corr = { pair_01.unwrap().correlation };
        assert!(corr >= 0.9, "Correlation should be ~1.0, got {}", corr);
    }

    #[test]
    fn test_profiler_vector_index() {
        let mut profiler = ActivationProfiler::new(1, 8, 4);

        // Create two clusters of embeddings
        for i in 0..50 {
            // Cluster 0: embeddings near [1, 0, 0, 0], experts 0-3
            let embedding = vec![1.0 + i as f32 * 0.01, 0.1, 0.0, 0.0];
            profiler.record(0, &[0, 1, 2, 3], &embedding);
        }
        for i in 0..50 {
            // Cluster 1: embeddings near [0, 0, 1, 0], experts 4-7
            let embedding = vec![0.0, 0.1, 1.0 + i as f32 * 0.01, 0.0];
            profiler.record(0, &[4, 5, 6, 7], &embedding);
        }

        let (centroids, entries) = profiler.build_vector_index(2, 20);

        assert_eq!(centroids.len(), 2);
        assert_eq!(entries.len(), 2);

        // Each cluster should have predictions for its experts
        for entry in &entries {
            let pc = { entry.prediction_count };
            assert!(pc > 0, "Each cluster should have expert predictions");
        }
    }

    #[test]
    fn test_profiler_summary() {
        let mut profiler = ActivationProfiler::new(2, 4, 4);
        let embedding = vec![0.0; 4];

        for _ in 0..100 {
            profiler.record(0, &[0, 1], &embedding);
            profiler.record(1, &[2, 3], &embedding);
        }

        let summary = profiler.summary();
        assert_eq!(summary.total_tokens, 100);
        assert_eq!(summary.num_layers, 2);
        assert_eq!(summary.num_experts, 4);
    }

    #[test]
    fn test_profiler_reservoir_sampling() {
        let mut profiler = ActivationProfiler::new(1, 4, 2);
        profiler.set_max_samples(100);

        // Add 1000 tokens — should keep at most 100 samples
        for i in 0..1000 {
            let embedding = vec![i as f32, (1000 - i) as f32];
            profiler.record(0, &[(i % 4) as u16], &embedding);
        }

        assert_eq!(profiler.total_tokens(), 1000);
        assert!(profiler.sample_count() <= 100);
    }
}
