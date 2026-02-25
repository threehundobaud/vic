//! Domain classifier and activation mode detector.
//!
//! Two components:
//! 1. **DomainClassifier**: Maps embedding vectors to workload domains (e.g. code, math,
//!    creative writing) using cosine-similarity against learned domain centroids.
//! 2. **ActivationModeDetector**: Monitors expert activation patterns over a sliding
//!    window and classifies the current workload as Generalist or Specialist mode
//!    based on activation entropy.

use crate::core::types::ActivationMode;

#[derive(Clone, Debug)]
pub struct DomainPrediction {
    pub domain_id: u32,
    pub domain_name: String,
    pub confidence: f32,
    pub recommended_view: Option<String>,
}

pub struct DomainClassifier {
    centroids: Vec<DomainCentroid>,
}

/// A named domain centroid with an optional recommended view.
struct DomainCentroid {
    domain_id: u32,
    name: String,
    center: Vec<f32>,
    view_name: Option<String>,
}

impl Default for DomainClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainClassifier {
    /// Create a new empty classifier (no domains configured).
    pub fn new() -> Self {
        Self { centroids: vec![] }
    }

    /// Create a classifier from pre-computed domain centroids.
    ///
    /// Each centroid is a tuple of (domain_id, name, center_vector, optional_view_name).
    pub fn from_centroids(centroids: Vec<(u32, String, Vec<f32>, Option<String>)>) -> Self {
        let centroids = centroids
            .into_iter()
            .map(|(id, name, center, view)| DomainCentroid {
                domain_id: id,
                name,
                center,
                view_name: view,
            })
            .collect();
        Self { centroids }
    }

    /// Add a single domain centroid.
    pub fn add_centroid(
        &mut self,
        domain_id: u32,
        name: String,
        center: Vec<f32>,
        view_name: Option<String>,
    ) {
        self.centroids.push(DomainCentroid {
            domain_id,
            name,
            center,
            view_name,
        });
    }

    /// Number of configured domains.
    pub fn num_domains(&self) -> usize {
        self.centroids.len()
    }

    /// Classify from an embedding vector using cosine similarity.
    ///
    /// Returns the nearest domain with a confidence score based on
    /// cosine similarity (mapped to [0, 1] range).
    pub fn classify(&self, embedding: &[f32]) -> DomainPrediction {
        if self.centroids.is_empty() {
            return DomainPrediction {
                domain_id: 0,
                domain_name: "unknown".into(),
                confidence: 0.0,
                recommended_view: None,
            };
        }

        let emb_norm = l2_norm(embedding);
        if emb_norm < 1e-12 {
            // Zero vector — can't classify
            let c = &self.centroids[0];
            return DomainPrediction {
                domain_id: c.domain_id,
                domain_name: c.name.clone(),
                confidence: 0.0,
                recommended_view: c.view_name.clone(),
            };
        }

        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, c) in self.centroids.iter().enumerate() {
            let sim = cosine_similarity(embedding, &c.center, emb_norm);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        let c = &self.centroids[best_idx];

        // Confidence: cosine similarity mapped from [-1, 1] to [0, 1]
        let confidence = (best_sim + 1.0) / 2.0;

        DomainPrediction {
            domain_id: c.domain_id,
            domain_name: c.name.clone(),
            confidence: confidence.clamp(0.0, 1.0),
            recommended_view: c.view_name.clone(),
        }
    }

    /// Classify and return top-k domains with their confidences.
    pub fn classify_top_k(&self, embedding: &[f32], k: usize) -> Vec<DomainPrediction> {
        if self.centroids.is_empty() {
            return vec![DomainPrediction {
                domain_id: 0,
                domain_name: "unknown".into(),
                confidence: 0.0,
                recommended_view: None,
            }];
        }

        let emb_norm = l2_norm(embedding);

        let mut scored: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine_similarity(embedding, &c.center, emb_norm)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        scored
            .into_iter()
            .map(|(idx, sim)| {
                let c = &self.centroids[idx];
                DomainPrediction {
                    domain_id: c.domain_id,
                    domain_name: c.name.clone(),
                    confidence: ((sim + 1.0) / 2.0).clamp(0.0, 1.0),
                    recommended_view: c.view_name.clone(),
                }
            })
            .collect()
    }
}

/// L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity with pre-computed norm for the first vector.
fn cosine_similarity(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let b_norm = l2_norm(b);
    if a_norm < 1e-12 || b_norm < 1e-12 {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (a_norm * b_norm)
}

// ─── Activation Mode Detector ────────────────────────────────────────────

/// Result of mode detection for the current window.
#[derive(Clone, Debug)]
pub struct ModeDetection {
    pub mode: ActivationMode,
    /// Shannon entropy of the activation distribution (bits).
    /// Low entropy → specialist, high entropy → generalist.
    pub entropy: f32,
    /// Entropy threshold used for the decision.
    pub threshold: f32,
    /// Number of unique experts seen in the window.
    pub unique_experts: usize,
    /// Concentration ratio: fraction of activations from top-K experts.
    pub concentration: f32,
    /// Confidence in the mode detection (0.0 = borderline, 1.0 = clear).
    pub confidence: f32,
}

/// Detects whether the current workload is in Generalist or Specialist mode
/// by measuring the Shannon entropy of expert activation frequencies over a
/// sliding window.
///
/// **Key insight**: In specialist mode, the same ~40-60 experts are activated
/// repeatedly (low entropy). In generalist mode, activations are spread across
/// many experts (high entropy). The entropy of the frequency distribution is
/// a single scalar that captures this distinction.
///
/// **Entropy math** (for Kimi K2.5, 384 experts):
/// - Maximum entropy: log2(384) ≈ 8.58 bits (uniform activation)
/// - Specialist mode: ~4-5 bits (40-60 experts dominate)
/// - Generalist mode: ~7-8 bits (200+ experts active)
/// - Default threshold: 6.0 bits (tunable)
///
/// The detector uses exponential moving average (EMA) smoothing to avoid
/// oscillating between modes on every token.
pub struct ActivationModeDetector {
    /// Number of experts in the model.
    num_experts: usize,

    /// Sliding window of recent expert activations.
    /// Each entry is a list of expert IDs activated for one token (summed across layers).
    window: Vec<Vec<u16>>,

    /// Maximum window size (in tokens).
    window_size: usize,

    /// Write pointer into the circular window buffer.
    write_idx: usize,

    /// Total tokens observed (including beyond the window).
    total_tokens: u64,

    /// EMA of entropy (smoothed to prevent mode oscillation).
    ema_entropy: f32,

    /// EMA decay factor (0.0 = no smoothing, 1.0 = ignore new data).
    /// Default: 0.9 (strong smoothing — mode changes are slow and deliberate).
    ema_alpha: f32,

    /// Entropy threshold: below this → Specialist, above → Generalist.
    /// Default: 6.0 bits for 384-expert models.
    entropy_threshold: f32,

    /// Current detected mode.
    current_mode: ActivationMode,

    /// Hysteresis: how many consecutive opposite-mode detections before switching.
    /// Prevents rapid oscillation at the boundary.
    hysteresis_threshold: u32,

    /// Sticky counter: how many consecutive tokens the opposite mode was detected.
    opposite_streak: u32,
}

impl ActivationModeDetector {
    /// Create a new detector.
    ///
    /// - `num_experts`: total experts per layer in the model
    /// - `window_size`: number of recent tokens to consider (e.g., 64-256)
    pub fn new(num_experts: usize, window_size: usize) -> Self {
        // Default threshold: ~70% of max entropy
        let max_entropy = (num_experts as f32).log2();
        let threshold = max_entropy * 0.70;

        Self {
            num_experts,
            window: Vec::with_capacity(window_size),
            window_size,
            write_idx: 0,
            total_tokens: 0,
            ema_entropy: max_entropy * 0.5, // Start neutral
            ema_alpha: 0.9,
            entropy_threshold: threshold,
            current_mode: ActivationMode::Generalist, // Default to generalist (safer)
            hysteresis_threshold: 8,                  // Need 8 consecutive opposite-mode readings
            opposite_streak: 0,
        }
    }

    /// Set the entropy threshold explicitly.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.entropy_threshold = threshold;
    }

    /// Set the EMA smoothing factor.
    pub fn set_ema_alpha(&mut self, alpha: f32) {
        self.ema_alpha = alpha.clamp(0.0, 1.0);
    }

    /// Set hysteresis (number of consecutive opposite readings before mode switch).
    pub fn set_hysteresis(&mut self, count: u32) {
        self.hysteresis_threshold = count;
    }

    /// Record one token's expert activations (combined across all layers).
    ///
    /// This is the hot-path call — should be called once per decoded token
    /// with the union of all expert IDs activated across all MoE layers.
    pub fn record(&mut self, active_experts: &[u16]) {
        self.total_tokens += 1;

        // Add to circular buffer
        if self.window.len() < self.window_size {
            self.window.push(active_experts.to_vec());
        } else {
            self.window[self.write_idx] = active_experts.to_vec();
        }
        self.write_idx = (self.write_idx + 1) % self.window_size;
    }

    /// Detect the current activation mode based on the window.
    ///
    /// Call this periodically (e.g., every 8-16 tokens) rather than every
    /// token, to amortize the histogram computation.
    pub fn detect(&mut self) -> ModeDetection {
        if self.window.is_empty() {
            return ModeDetection {
                mode: self.current_mode,
                entropy: self.ema_entropy,
                threshold: self.entropy_threshold,
                unique_experts: 0,
                concentration: 0.0,
                confidence: 0.0,
            };
        }

        // Build frequency histogram over the window
        let mut counts = vec![0u32; self.num_experts];
        let mut total_activations = 0u32;

        for token_experts in &self.window {
            for &expert in token_experts {
                if (expert as usize) < self.num_experts {
                    counts[expert as usize] += 1;
                    total_activations += 1;
                }
            }
        }

        if total_activations == 0 {
            return ModeDetection {
                mode: self.current_mode,
                entropy: self.ema_entropy,
                threshold: self.entropy_threshold,
                unique_experts: 0,
                concentration: 0.0,
                confidence: 0.0,
            };
        }

        // Compute Shannon entropy: H = -sum(p_i * log2(p_i))
        let total_f = total_activations as f32;
        let mut entropy = 0.0f32;
        let mut unique_experts = 0usize;

        for &count in &counts {
            if count > 0 {
                unique_experts += 1;
                let p = count as f32 / total_f;
                entropy -= p * p.log2();
            }
        }

        // Concentration ratio: fraction of activations from top-32 experts
        let mut sorted_counts = counts.clone();
        sorted_counts.sort_unstable_by(|a, b| b.cmp(a));
        let top_k = 32.min(sorted_counts.len());
        let top_k_sum: u32 = sorted_counts[..top_k].iter().sum();
        let concentration = top_k_sum as f32 / total_f;

        // EMA smoothing
        self.ema_entropy = self.ema_alpha * self.ema_entropy + (1.0 - self.ema_alpha) * entropy;

        // Mode detection with hysteresis
        let raw_mode = if self.ema_entropy < self.entropy_threshold {
            ActivationMode::Specialist
        } else {
            ActivationMode::Generalist
        };

        if raw_mode != self.current_mode {
            self.opposite_streak += 1;
            if self.opposite_streak >= self.hysteresis_threshold {
                self.current_mode = raw_mode;
                self.opposite_streak = 0;
            }
        } else {
            self.opposite_streak = 0;
        }

        // Confidence: how far from the threshold (normalized)
        let max_entropy = (self.num_experts as f32).log2();
        let distance = (self.ema_entropy - self.entropy_threshold).abs();
        let confidence = (distance / (max_entropy * 0.5)).clamp(0.0, 1.0);

        ModeDetection {
            mode: self.current_mode,
            entropy: self.ema_entropy,
            threshold: self.entropy_threshold,
            unique_experts,
            concentration,
            confidence,
        }
    }

    /// Get the current mode without recomputing.
    pub fn current_mode(&self) -> ActivationMode {
        self.current_mode
    }

    /// Get the current smoothed entropy.
    pub fn current_entropy(&self) -> f32 {
        self.ema_entropy
    }

    /// Get the total number of tokens observed.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Force a specific mode (useful for testing or manual override).
    pub fn force_mode(&mut self, mode: ActivationMode) {
        self.current_mode = mode;
        self.opposite_streak = 0;
    }

    /// Get the top-K most frequently activated experts in the current window.
    ///
    /// Used by the specialist pinning logic to determine which experts to lock in T1.
    pub fn top_experts(&self, k: usize) -> Vec<(u16, u32)> {
        if self.window.is_empty() {
            return vec![];
        }

        let mut counts = vec![0u32; self.num_experts];
        for token_experts in &self.window {
            for &expert in token_experts {
                if (expert as usize) < self.num_experts {
                    counts[expert as usize] += 1;
                }
            }
        }

        let mut scored: Vec<(u16, u32)> = counts
            .into_iter()
            .enumerate()
            .filter(|(_, c)| *c > 0)
            .map(|(i, c)| (i as u16, c))
            .collect();

        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(k);
        scored
    }
}
