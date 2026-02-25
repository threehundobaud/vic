//! Validation scaffolding for end-to-end correctness testing.
//!
//! This module provides tools to verify that vib3's inference pipeline
//! produces correct results, without requiring real model weights or CUDA.
//!
//! ## Approach
//!
//! 1. **ReferenceModel**: A tiny deterministic model where every weight
//!    is derived from a known seed. We can compute the "correct" output
//!    for any input by running the same math in f32 on CPU.
//!
//! 2. **OutputComparison**: Compares engine output against reference output,
//!    tracking per-layer numerical divergence.
//!
//! 3. **QuantizationErrorTracker**: Measures how INT4 quantization affects
//!    output quality across layers.

use crate::core::config::ModelConfig;

// ─── Reference Model ────────────────────────────────────────────────────

/// A tiny reference model with deterministic weights for validation.
///
/// All weights are derived from a seed using a simple PRNG, making the
/// model fully reproducible. The reference forward pass runs in f32 on
/// CPU, giving us a "ground truth" to compare the engine's output against.
pub struct ReferenceModel {
    pub config: ModelConfig,
    /// Router weights: [num_moe_layers][num_experts][hidden_dim]
    pub router_weights: Vec<Vec<Vec<f32>>>,
    /// Expert up_proj: [num_moe_layers][num_experts][expert_hidden_dim][hidden_dim]
    pub up_proj: Vec<Vec<Vec<Vec<f32>>>>,
    /// Expert gate_proj: same shape as up_proj
    pub gate_proj: Vec<Vec<Vec<Vec<f32>>>>,
    /// Expert down_proj: [num_moe_layers][num_experts][hidden_dim][expert_hidden_dim]
    pub down_proj: Vec<Vec<Vec<Vec<f32>>>>,
    /// Embedding table: [vocab_size][hidden_dim]
    pub embeddings: Vec<Vec<f32>>,
}

/// Simple PRNG for deterministic weight generation.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Map to [-0.1, 0.1] for reasonable weight magnitudes
    let bits = (*state >> 33) as u32;
    (bits as f32 / u32::MAX as f32 - 0.5) * 0.2
}

impl ReferenceModel {
    /// Create a reference model with deterministic weights.
    pub fn new(config: ModelConfig, seed: u64) -> Self {
        let mut rng = seed;
        let hidden = config.hidden_dim as usize;
        let expert_hidden = config.expert_hidden_dim as usize;
        let num_experts = config.num_experts as usize;
        let num_moe = config.num_moe_layers as usize;
        let vocab_size = config.vocab_size as usize;

        // Generate embeddings
        let embeddings: Vec<Vec<f32>> = (0..vocab_size)
            .map(|_| (0..hidden).map(|_| lcg_next(&mut rng)).collect())
            .collect();

        // Generate router weights
        let router_weights: Vec<Vec<Vec<f32>>> = (0..num_moe)
            .map(|_| {
                (0..num_experts)
                    .map(|_| (0..hidden).map(|_| lcg_next(&mut rng)).collect())
                    .collect()
            })
            .collect();

        // Generate expert weights
        let mut up_proj = Vec::with_capacity(num_moe);
        let mut gate_proj = Vec::with_capacity(num_moe);
        let mut down_proj = Vec::with_capacity(num_moe);

        for _ in 0..num_moe {
            let mut layer_up = Vec::with_capacity(num_experts);
            let mut layer_gate = Vec::with_capacity(num_experts);
            let mut layer_down = Vec::with_capacity(num_experts);

            for _ in 0..num_experts {
                layer_up.push(
                    (0..expert_hidden)
                        .map(|_| (0..hidden).map(|_| lcg_next(&mut rng)).collect())
                        .collect(),
                );
                layer_gate.push(
                    (0..expert_hidden)
                        .map(|_| (0..hidden).map(|_| lcg_next(&mut rng)).collect())
                        .collect(),
                );
                layer_down.push(
                    (0..hidden)
                        .map(|_| (0..expert_hidden).map(|_| lcg_next(&mut rng)).collect())
                        .collect(),
                );
            }

            up_proj.push(layer_up);
            gate_proj.push(layer_gate);
            down_proj.push(layer_down);
        }

        Self {
            config,
            router_weights,
            up_proj,
            gate_proj,
            down_proj,
            embeddings,
        }
    }

    /// Run the reference forward pass for one token embedding through MoE layers.
    ///
    /// Returns the hidden state after all MoE layers, computed in f32.
    /// This is the "ground truth" that the engine should approximate.
    pub fn forward_moe(&self, input_embedding: &[f32]) -> Vec<f32> {
        let hidden = self.config.hidden_dim as usize;
        let expert_hidden = self.config.expert_hidden_dim as usize;
        let top_k = self.config.num_active_experts as usize;
        let num_experts = self.config.num_experts as usize;

        let mut state = input_embedding.to_vec();

        for layer in 0..self.config.num_moe_layers as usize {
            // 1. Router: compute scores and top-k
            let mut scores: Vec<(usize, f32)> = (0..num_experts)
                .map(|e| {
                    let dot: f32 = state
                        .iter()
                        .zip(self.router_weights[layer][e].iter())
                        .map(|(s, w)| s * w)
                        .sum();
                    (e, dot)
                })
                .collect();

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(top_k);

            // Softmax over top-k
            let max_s = scores
                .iter()
                .map(|(_, s)| *s)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|(_, s)| (s - max_s).exp()).sum();
            let weights: Vec<(usize, f32)> = scores
                .iter()
                .map(|(e, s)| (*e, (s - max_s).exp() / exp_sum))
                .collect();

            // 2. Expert computation: SwiGLU for each active expert
            let mut layer_output = vec![0.0f32; hidden];

            for &(expert_id, weight) in &weights {
                // up_proj: [expert_hidden] = state × up_proj^T
                let mut up_out = vec![0.0f32; expert_hidden];
                for (m, up_val) in up_out.iter_mut().enumerate() {
                    for (k, &s) in state.iter().enumerate().take(hidden) {
                        *up_val += s * self.up_proj[layer][expert_id][m][k];
                    }
                }

                // gate_proj: [expert_hidden] = state × gate_proj^T
                let mut gate_out = vec![0.0f32; expert_hidden];
                for (m, gate_val) in gate_out.iter_mut().enumerate() {
                    for (k, &s) in state.iter().enumerate().take(hidden) {
                        *gate_val += s * self.gate_proj[layer][expert_id][m][k];
                    }
                }

                // SwiGLU: up * silu(gate)
                let mut swiglu = vec![0.0f32; expert_hidden];
                for m in 0..expert_hidden {
                    let silu = gate_out[m] * sigmoid(gate_out[m]);
                    swiglu[m] = up_out[m] * silu;
                }

                // down_proj: [hidden] = swiglu × down_proj^T
                let mut down_out = vec![0.0f32; hidden];
                for (d, down_val) in down_out.iter_mut().enumerate() {
                    for (m, &sw) in swiglu.iter().enumerate() {
                        *down_val += sw * self.down_proj[layer][expert_id][d][m];
                    }
                }

                // Weighted accumulate
                for (lo, &dv) in layer_output.iter_mut().zip(down_out.iter()).take(hidden) {
                    *lo += weight * dv;
                }
            }

            // Residual connection
            for (s, &lo) in state.iter_mut().zip(layer_output.iter()).take(hidden) {
                *s += lo;
            }
        }

        state
    }

    /// Get the embedding for a token.
    pub fn embed(&self, token_id: u32) -> &[f32] {
        &self.embeddings[token_id as usize % self.embeddings.len()]
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─── Output Comparison ──────────────────────────────────────────────────

/// Result of comparing two output vectors.
#[derive(Clone, Debug)]
pub struct ComparisonResult {
    /// Mean absolute error.
    pub mae: f32,
    /// Max absolute error.
    pub max_error: f32,
    /// Root mean square error.
    pub rmse: f32,
    /// Cosine similarity (1.0 = identical direction).
    pub cosine_similarity: f32,
    /// Number of elements compared.
    pub num_elements: usize,
}

impl ComparisonResult {
    /// Whether the outputs are "close enough" for practical purposes.
    ///
    /// Thresholds are calibrated for INT4 quantization, which introduces
    /// significant per-element error but preserves the overall direction.
    pub fn is_acceptable(&self) -> bool {
        // Cosine similarity > 0.9 means the direction is preserved
        // MAE < 0.5 means individual values aren't too far off
        self.cosine_similarity > 0.9 && self.mae < 0.5
    }

    /// Whether the outputs are numerically identical (within floating point).
    pub fn is_exact(&self, tolerance: f32) -> bool {
        self.max_error < tolerance
    }
}

/// Compare two vectors element-wise.
pub fn compare_outputs(reference: &[f32], actual: &[f32]) -> ComparisonResult {
    let n = reference.len().min(actual.len());
    if n == 0 {
        return ComparisonResult {
            mae: 0.0,
            max_error: 0.0,
            rmse: 0.0,
            cosine_similarity: 1.0,
            num_elements: 0,
        };
    }

    let mut sum_abs_err = 0.0f64;
    let mut sum_sq_err = 0.0f64;
    let mut max_err = 0.0f32;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..n {
        let a = reference[i] as f64;
        let b = actual[i] as f64;
        let err = (a - b).abs();

        sum_abs_err += err;
        sum_sq_err += err * err;
        max_err = max_err.max(err as f32);

        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    let mae = (sum_abs_err / n as f64) as f32;
    let rmse = (sum_sq_err / n as f64).sqrt() as f32;

    let cosine = if norm_a > 1e-12 && norm_b > 1e-12 {
        (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
    } else {
        0.0
    };

    ComparisonResult {
        mae,
        max_error: max_err,
        rmse,
        cosine_similarity: cosine,
        num_elements: n,
    }
}

// ─── Quantization Error Tracker ─────────────────────────────────────────

/// Tracks quantization error accumulation across layers.
///
/// When INT4 quantization is applied, each layer introduces some error.
/// This tracker measures whether error accumulates (bad) or stays bounded
/// (acceptable). If error grows linearly with depth, the model is usable.
/// If it grows exponentially, the quantization is too aggressive.
#[derive(Clone, Debug)]
pub struct QuantizationErrorTracker {
    /// Per-layer MAE.
    pub layer_mae: Vec<f32>,
    /// Per-layer cosine similarity.
    pub layer_cosine: Vec<f32>,
    /// Per-layer RMSE.
    pub layer_rmse: Vec<f32>,
}

impl QuantizationErrorTracker {
    pub fn new() -> Self {
        Self {
            layer_mae: Vec::new(),
            layer_cosine: Vec::new(),
            layer_rmse: Vec::new(),
        }
    }

    /// Record the error at one layer.
    pub fn record_layer(&mut self, result: &ComparisonResult) {
        self.layer_mae.push(result.mae);
        self.layer_cosine.push(result.cosine_similarity);
        self.layer_rmse.push(result.rmse);
    }

    /// Number of layers tracked.
    pub fn num_layers(&self) -> usize {
        self.layer_mae.len()
    }

    /// Average MAE across all layers.
    pub fn mean_mae(&self) -> f32 {
        if self.layer_mae.is_empty() {
            return 0.0;
        }
        self.layer_mae.iter().sum::<f32>() / self.layer_mae.len() as f32
    }

    /// Minimum cosine similarity (worst layer).
    pub fn worst_cosine(&self) -> f32 {
        self.layer_cosine
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
    }

    /// Whether error growth is linear (acceptable) vs exponential (bad).
    ///
    /// Computes the ratio of error at the last layer to the first layer.
    /// Linear growth means ratio ≈ num_layers. Exponential means ratio >> num_layers.
    pub fn error_growth_ratio(&self) -> f32 {
        if self.layer_mae.len() < 2 {
            return 1.0;
        }
        let first = self.layer_mae[0].max(1e-12);
        let last = self.layer_mae[self.layer_mae.len() - 1];
        last / first
    }

    /// Whether quantization error is acceptable.
    pub fn is_acceptable(&self) -> bool {
        self.worst_cosine() > 0.8 && self.error_growth_ratio() < self.num_layers() as f32 * 3.0
    }
}

impl Default for QuantizationErrorTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::DType;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            name: "tiny-ref".into(),
            architecture: "test".into(),
            hidden_dim: 8,
            expert_hidden_dim: 4,
            num_layers: 2,
            num_moe_layers: 2,
            dense_layer_idx: 0,
            num_experts: 4,
            num_active_experts: 2,
            num_heads: 2,
            num_kv_heads: 1,
            max_seq_len: 32,
            vocab_size: 16,
            expert_dtype: DType::FP16,
            shared_dtype: DType::FP16,
            ..Default::default()
        }
    }

    #[test]
    fn test_reference_model_deterministic() {
        let config = tiny_config();
        let model1 = ReferenceModel::new(config.clone(), 42);
        let model2 = ReferenceModel::new(config, 42);

        // Same seed → same weights
        assert_eq!(model1.embeddings[0], model2.embeddings[0]);
        assert_eq!(model1.router_weights[0][0], model2.router_weights[0][0]);
        assert_eq!(model1.up_proj[0][0][0], model2.up_proj[0][0][0]);
    }

    #[test]
    fn test_reference_model_forward() {
        let config = tiny_config();
        let model = ReferenceModel::new(config, 42);

        let embedding = model.embed(0);
        let output = model.forward_moe(embedding);

        // Output should have the right dimension
        assert_eq!(output.len(), 8);

        // Output should be finite
        for &v in &output {
            assert!(v.is_finite(), "Got non-finite value: {}", v);
        }

        // Running again with the same input should produce the same output
        let output2 = model.forward_moe(embedding);
        assert_eq!(output, output2);
    }

    #[test]
    fn test_reference_model_different_seeds() {
        let config = tiny_config();
        let model1 = ReferenceModel::new(config.clone(), 42);
        let model2 = ReferenceModel::new(config, 99);

        // Different seeds → different weights
        assert_ne!(model1.embeddings[0], model2.embeddings[0]);
    }

    #[test]
    fn test_compare_outputs_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let result = compare_outputs(&a, &a);
        assert!(result.is_exact(1e-6));
        assert!((result.cosine_similarity - 1.0).abs() < 1e-6);
        assert_eq!(result.mae, 0.0);
    }

    #[test]
    fn test_compare_outputs_different() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        let result = compare_outputs(&a, &b);

        // Orthogonal vectors → cosine similarity near 0
        assert!(result.cosine_similarity.abs() < 0.1);
        assert!(result.mae > 0.0);
    }

    #[test]
    fn test_compare_outputs_scaled() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; // Scaled version
        let result = compare_outputs(&a, &b);

        // Same direction → cosine similarity near 1
        assert!(result.cosine_similarity > 0.99);
    }

    #[test]
    fn test_quantization_error_tracker() {
        let mut tracker = QuantizationErrorTracker::new();

        // Simulate linear error growth across 4 layers
        for layer in 0..4 {
            let result = ComparisonResult {
                mae: 0.01 * (layer as f32 + 1.0),
                max_error: 0.1,
                rmse: 0.02,
                cosine_similarity: 0.99 - layer as f32 * 0.01,
                num_elements: 64,
            };
            tracker.record_layer(&result);
        }

        assert_eq!(tracker.num_layers(), 4);
        assert!(tracker.mean_mae() > 0.0);
        assert!(tracker.worst_cosine() > 0.9);

        // Linear growth: last/first ≈ 4 (within 3x num_layers = 12)
        assert!(tracker.is_acceptable());
    }

    #[test]
    fn test_quantization_error_tracker_exponential_growth() {
        let mut tracker = QuantizationErrorTracker::new();

        // Simulate exponential error growth (bad!)
        for layer in 0..4 {
            let result = ComparisonResult {
                mae: 0.01 * 10.0f32.powi(layer),
                max_error: 1.0,
                rmse: 0.1,
                cosine_similarity: 0.5, // Poor direction preservation
                num_elements: 64,
            };
            tracker.record_layer(&result);
        }

        // Exponential growth should NOT be acceptable
        assert!(!tracker.is_acceptable());
    }
}
