//! Query Planner — translates router outputs into page-level fetch plans
//! with lookahead prefetching.
//!
//! Database analogy: this is the query optimizer. It takes a "query"
//! (which experts does this token need?) and produces an "execution plan"
//! (which pages to fetch from which tiers, in what order, with what priority).

use crate::core::config::ModelConfig;
use crate::core::error::Result;
use crate::core::types::*;
use crate::index::vector_index::{ActivationProfile, VectorIndex};
use crate::storage::buffer_manager::PageBufferManager;
use crate::storage::format::Vib3File;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing;

/// Resolved plan for one layer's expert computation.
#[derive(Debug)]
pub struct LayerPlan {
    pub layer: u16,
    pub experts: Vec<ExpertPlan>,
    pub all_resident: bool,
}

/// Resolved plan for one expert's pages.
#[derive(Debug)]
pub struct ExpertPlan {
    pub expert_id: u16,
    pub weight: f32,
    pub pages: Vec<ResolvedPage>,
    pub fully_resident: bool,
}

/// A page resolved to a device pointer.
#[derive(Debug)]
pub struct ResolvedPage {
    pub id: PageId,
    pub device_ptr: *mut u8,
    pub row_start: u16,
    pub row_count: u16,
}

unsafe impl Send for ResolvedPage {}
unsafe impl Sync for ResolvedPage {}

/// Planner accuracy tracking.
#[derive(Default)]
pub struct PlannerStats {
    pub plans_created: AtomicU64,
    pub pages_predicted: AtomicU64,
    pub pages_needed: AtomicU64,
    pub prediction_hits: AtomicU64,
    pub prediction_misses: AtomicU64,
}

impl PlannerStats {
    pub fn precision(&self) -> f64 {
        let hits = self.prediction_hits.load(Ordering::Relaxed) as f64;
        let predicted = self.pages_predicted.load(Ordering::Relaxed) as f64;
        if predicted > 0.0 {
            hits / predicted
        } else {
            0.0
        }
    }

    pub fn recall(&self) -> f64 {
        let hits = self.prediction_hits.load(Ordering::Relaxed) as f64;
        let needed = self.pages_needed.load(Ordering::Relaxed) as f64;
        if needed > 0.0 {
            hits / needed
        } else {
            0.0
        }
    }
}

pub struct QueryPlanner {
    buffer_mgr: Arc<PageBufferManager>,
    vector_index: Option<Arc<VectorIndex>>,
    model_file: Arc<Vib3File>,
    model_config: ModelConfig,

    /// Recent hidden states for trajectory prediction.
    trajectory: VecDeque<Vec<f32>>,
    trajectory_max: usize,

    /// Adaptive prefetch multiplier based on accuracy and activation mode.
    /// - Generalist: higher (1.5-2.0x) — wider working set, more aggressive prefetch
    /// - Specialist: lower (0.5-1.0x) — stable working set, pinned pages reduce need
    prefetch_multiplier: f32,

    /// Current activation mode — affects prefetch strategy.
    current_mode: ActivationMode,

    // ── Phase C: Gear-filtered search ────────────────────────────────
    /// Active gear domains for filtered HNSW search.
    /// When non-empty, page retrieval is narrowed to centroids matching
    /// these domain tags. Set from Engine::set_task_context().
    gear_domains: Vec<String>,

    /// Cached vector index prediction for the current token step.
    /// Cleared on each `update_trajectory()` call. Avoids repeating
    /// the same HNSW search + layer iteration 120 times per token.
    cached_prediction: Option<ActivationProfile>,

    pub stats: PlannerStats,
}

impl QueryPlanner {
    pub fn new(
        buffer_mgr: Arc<PageBufferManager>,
        vector_index: Option<Arc<VectorIndex>>,
        model_file: Arc<Vib3File>,
        model_config: ModelConfig,
    ) -> Self {
        Self {
            buffer_mgr,
            vector_index,
            model_file,
            model_config,
            trajectory: VecDeque::with_capacity(32),
            trajectory_max: 32,
            prefetch_multiplier: 1.0,
            current_mode: ActivationMode::Generalist,
            gear_domains: Vec::new(),
            cached_prediction: None,
            stats: PlannerStats::default(),
        }
    }

    /// Map an engine layer index to the storage layer index for expert pages.
    ///
    /// The .vib3 converter stores expert pages at `hf_layer + dense_layer_idx`
    /// (see convert.rs:1519), but the engine iterates layers using the raw HF
    /// layer index. This helper applies the offset so page lookups find the
    /// correct expert data.
    ///
    /// Shared tensors (attention, norms, router) are stored at the raw HF layer
    /// index and do NOT need this adjustment.
    #[inline]
    fn expert_storage_layer(&self, engine_layer: u16) -> u16 {
        engine_layer + self.model_config.dense_layer_idx as u16
    }

    /// Plan computation for one layer given router activations.
    ///
    /// 1. Resolve each active expert to its page list
    /// 2. Check which pages are in T1
    /// 3. Request missing pages from buffer manager
    /// 4. Return resolved device pointers
    ///
    /// When the model is fully resident in T1, this uses a zero-overhead fast
    /// path: a single `HashMap::get()` per page with no locking, no atomics,
    /// and no async suspension.
    pub async fn plan_layer(&self, layer: u16, activation: &ExpertActivation) -> Result<LayerPlan> {
        self.stats.plans_created.fetch_add(1, Ordering::Relaxed);

        // Fast path: model fully resident — bypass DashMap + Mutex entirely.
        if self.buffer_mgr.is_fully_resident() {
            return self.plan_layer_resident(layer, activation);
        }

        let mut experts = Vec::with_capacity(activation.count());
        let mut all_resident = true;

        let storage_layer = self.expert_storage_layer(layer);

        // One-time diagnostic: log the mapping for layer 1
        if self.stats.plans_created.load(Ordering::Relaxed) <= 2 && layer == 1 {
            let first_expert = activation.experts.first().map(|(id, _)| *id).unwrap_or(0);
            let test_pages = self.model_file.pages_for_expert(storage_layer, first_expert);
            tracing::info!(
                "PLAN DIAG: engine_layer={}, storage_layer={}, expert={}, pages={}",
                layer, storage_layer, first_expert, test_pages.len(),
            );
            // Also check what's at the raw engine layer (should be empty)
            let raw_pages = self.model_file.pages_for_expert(layer, first_expert);
            tracing::info!(
                "PLAN DIAG: raw engine_layer={} pages={} (should be 0 for MoE layers)",
                layer, raw_pages.len(),
            );
        }

        for &(expert_id, weight) in &activation.experts {
            let page_entries = self.model_file.pages_for_expert(storage_layer, expert_id);

            let mut pages = Vec::with_capacity(page_entries.len());
            let mut fully_resident = true;

            for entry in page_entries {
                let page_id = entry.page_id();

                // Get page handle (may block if page needs transfer)
                let handle = self.buffer_mgr.get_page(&page_id).await?;

                pages.push(ResolvedPage {
                    id: page_id,
                    device_ptr: handle.device_ptr,
                    row_start: entry.row_start,
                    row_count: entry.row_count,
                });

                if handle.source_tier != Tier::T1Vram {
                    fully_resident = false;
                }
            }

            if !fully_resident {
                all_resident = false;
            }

            experts.push(ExpertPlan {
                expert_id,
                weight,
                pages,
                fully_resident,
            });
        }

        Ok(LayerPlan {
            layer,
            experts,
            all_resident,
        })
    }

    /// Zero-overhead layer planning when all pages are resident in T1.
    ///
    /// Uses the frozen `resident_snapshot` HashMap — a single `HashMap::get()`
    /// per page with no DashMap shard locks, no Mutex on PageSlot, no atomic
    /// tick/stats updates, and no async suspension points.
    fn plan_layer_resident(&self, layer: u16, activation: &ExpertActivation) -> Result<LayerPlan> {
        let mut experts = Vec::with_capacity(activation.count());
        let storage_layer = self.expert_storage_layer(layer);

        for &(expert_id, weight) in &activation.experts {
            let page_entries = self.model_file.pages_for_expert(storage_layer, expert_id);
            let mut pages = Vec::with_capacity(page_entries.len());

            for entry in page_entries {
                let page_id = entry.page_id();

                let handle = self
                    .buffer_mgr
                    .get_page_resident(&page_id)
                    .ok_or(crate::core::error::Error::PageNotFound { page: page_id })?;

                pages.push(ResolvedPage {
                    id: page_id,
                    device_ptr: handle.device_ptr,
                    row_start: entry.row_start,
                    row_count: entry.row_count,
                });
            }

            experts.push(ExpertPlan {
                expert_id,
                weight,
                pages,
                fully_resident: true,
            });
        }

        Ok(LayerPlan {
            layer,
            experts,
            all_resident: true,
        })
    }

    /// Submit lookahead: prefetch pages for a future layer.
    ///
    /// In Specialist mode, prefetch priority is reduced (most pages are pinned).
    /// In Generalist mode, priority stays high and confidence is boosted by the
    /// prefetch multiplier.
    pub fn submit_lookahead(&self, layer: u16, activation: &ExpertActivation) {
        // In Specialist mode, most pages should already be pinned in T1.
        // Only prefetch pages that aren't specialist-pinned.
        let priority = match self.current_mode {
            ActivationMode::Specialist => PrefetchPriority::Low,
            ActivationMode::Generalist => PrefetchPriority::High,
        };

        let base_confidence = 0.9;
        let adjusted_confidence = (base_confidence * self.prefetch_multiplier).clamp(0.0, 1.0);

        let storage_layer = self.expert_storage_layer(layer);

        for &(expert_id, _weight) in &activation.experts {
            let page_entries = self.model_file.pages_for_expert(storage_layer, expert_id);

            for entry in page_entries {
                let page_id = entry.page_id();

                // In Specialist mode, skip prefetch for pages already pinned
                if self.current_mode == ActivationMode::Specialist
                    && self.buffer_mgr.is_specialist_pinned(&page_id)
                {
                    continue;
                }

                self.buffer_mgr.submit_prefetch(PrefetchRequest {
                    page: page_id,
                    source: Tier::T3Nvme,
                    dest: Tier::T1Vram,
                    priority,
                    deadline_tick: 0,
                    confidence: adjusted_confidence,
                });
            }
        }
    }

    /// Update the activation mode and adjust prefetch strategy.
    ///
    /// Called by the engine when the mode detector identifies a transition.
    pub fn set_mode(&mut self, mode: ActivationMode) {
        let previous = self.current_mode;
        self.current_mode = mode;

        self.prefetch_multiplier = match mode {
            // Specialist: stable working set, most pages pinned. Low prefetch.
            ActivationMode::Specialist => 0.5,
            // Generalist: wider working set, need aggressive prefetch.
            ActivationMode::Generalist => 1.5,
        };

        if previous != mode {
            tracing::info!(
                "QueryPlanner mode: {} -> {} (prefetch_multiplier={:.1})",
                previous,
                mode,
                self.prefetch_multiplier,
            );
        }
    }

    /// Get the current activation mode.
    pub fn current_mode(&self) -> ActivationMode {
        self.current_mode
    }

    /// Set gear domain filter for Phase C filtered HNSW search.
    ///
    /// When non-empty, page retrieval will be narrowed to centroids
    /// matching these domain tags. An empty vec disables filtering.
    pub fn set_gear_domains(&mut self, domains: Vec<String>) {
        if !domains.is_empty() {
            tracing::debug!("QueryPlanner gear filter: {:?}", domains);
        }
        self.gear_domains = domains;
    }

    /// Get the current gear domain filter.
    pub fn gear_domains(&self) -> &[String] {
        &self.gear_domains
    }

    /// Feed a hidden state for trajectory tracking.
    pub fn update_trajectory(&mut self, hidden_state: Vec<f32>) {
        if self.trajectory.len() >= self.trajectory_max {
            self.trajectory.pop_front();
        }
        self.trajectory.push_back(hidden_state);
        // Invalidate cached prediction — new embedding means new prediction
        self.cached_prediction = None;
    }

    /// Get or compute the cached vector index prediction for the current token step.
    /// Returns None if no vector index or no trajectory.
    fn ensure_prediction_cached(&mut self) {
        if self.cached_prediction.is_some() {
            return;
        }
        if let Some(vi) = &self.vector_index {
            if let Some(embedding) = self.trajectory.back() {
                self.cached_prediction = Some(vi.predict(embedding));
            }
        }
    }

    /// Speculative prefetch using the vector index and recent trajectory.
    ///
    /// This is the database-index-on-every-query path: after each token,
    /// we use the trajectory (recent hidden states) to predict which pages
    /// will be needed for future tokens and submit prefetch requests.
    ///
    /// Called from Engine::generate_token() after updating the trajectory.
    /// Returns the number of prefetch requests submitted.
    pub fn submit_vector_prefetch(&self, lookahead: usize) -> usize {
        let vi = match &self.vector_index {
            Some(vi) => vi,
            None => return 0,
        };

        if self.trajectory.is_empty() {
            return 0;
        }

        // Build references to recent trajectory states for the vector index
        let recent: Vec<&[f32]> = self.trajectory.iter().map(|v| v.as_slice()).collect();

        // Use the vector index's speculative_prefetch — it extrapolates
        // the trajectory and predicts pages for future tokens
        let requests = vi.speculative_prefetch(&recent, lookahead);
        let count = requests.len();

        if count > 0 {
            // Adjust priorities based on activation mode
            let adjusted: Vec<PrefetchRequest> = requests
                .into_iter()
                .map(|mut req| {
                    // In Specialist mode, skip pages already pinned
                    if self.current_mode == ActivationMode::Specialist
                        && self.buffer_mgr.is_specialist_pinned(&req.page)
                    {
                        req.confidence = 0.0; // will be filtered below
                    }
                    // Scale confidence by prefetch multiplier
                    req.confidence = (req.confidence * self.prefetch_multiplier).clamp(0.0, 1.0);
                    req
                })
                .filter(|req| req.confidence > 0.01) // Drop near-zero confidence
                .collect();

            let submitted = adjusted.len();
            self.buffer_mgr.submit_prefetch_batch(adjusted);

            self.stats
                .pages_predicted
                .fetch_add(submitted as u64, Ordering::Relaxed);

            tracing::debug!(
                "Vector index prefetch: {} requests (lookahead={}, mode={})",
                submitted,
                lookahead,
                self.current_mode,
            );

            submitted
        } else {
            0
        }
    }

    /// Use the vector index to predict pages needed for the current layer
    /// and submit prefetch requests for pages not yet in T1.
    ///
    /// This is called at the start of each MoE layer before plan_layer(),
    /// analogous to an index lookup before a table scan. It uses the latest
    /// hidden state to predict which experts will be activated and pre-warms
    /// their pages.
    ///
    /// Returns the number of prefetch requests submitted.
    pub fn predict_and_prewarm(&mut self, current_layer: u16) -> usize {
        // Ensure cached prediction exists (avoids redundant HNSW search per layer)
        self.ensure_prediction_cached();

        // Extract predicted pages for this layer from cache
        let predicted_pages: Vec<PageId> = match &self.cached_prediction {
            Some(profile) => {
                match profile.layers.iter().find(|lp| lp.layer == current_layer) {
                    Some(lp) => lp.predicted_pages.clone(),
                    None => return 0,
                }
            }
            None => return 0,
        };

        let mut submitted = 0;
        for (idx, page) in predicted_pages.iter().enumerate() {
            // Skip specialist-pinned pages
            if self.current_mode == ActivationMode::Specialist
                && self.buffer_mgr.is_specialist_pinned(page)
            {
                continue;
            }

            let priority = if idx < 4 {
                PrefetchPriority::High
            } else {
                PrefetchPriority::Medium
            };

            let confidence =
                (0.8 * self.prefetch_multiplier / (idx as f32 + 1.0).sqrt()).clamp(0.0, 1.0);

            self.buffer_mgr.submit_prefetch(PrefetchRequest {
                page: *page,
                source: Tier::T3Nvme,
                dest: Tier::T1Vram,
                priority,
                deadline_tick: 0,
                confidence,
            });
            submitted += 1;
        }

        if submitted > 0 {
            self.stats
                .pages_predicted
                .fetch_add(submitted as u64, Ordering::Relaxed);
            tracing::debug!(
                "Vector index prewarm L{}: {} pages predicted, {} submitted",
                current_layer,
                predicted_pages.len(),
                submitted,
            );
        }

        submitted
    }

    /// Submit cross-layer vector index predictions alongside the
    /// same-activation lookahead.
    ///
    /// This extends submit_lookahead() with vector-index-driven predictions
    /// for layers beyond the immediate next layer. The vector index predicts
    /// which experts are likely to activate 2-3 layers ahead based on the
    /// trajectory, enabling deeper prefetch pipelining.
    pub fn submit_cross_layer_prefetch(&mut self, current_layer: u16) {
        // Ensure cached prediction exists (avoids redundant HNSW search per layer)
        self.ensure_prediction_cached();

        let dense_start = self.model_config.dense_layer_idx as u16;
        let num_moe = self.model_config.num_moe_layers as u16;

        // Prefetch pages for layers 2-3 ahead (layer+1 is handled by submit_lookahead)
        for offset in 2..=3u16 {
            let target_layer = current_layer + offset;
            if target_layer >= dense_start + num_moe {
                break;
            }

            // Extract pages for target layer from cached prediction
            let pages: Vec<PageId> = match &self.cached_prediction {
                Some(profile) => {
                    match profile.layers.iter().find(|lp| lp.layer == target_layer) {
                        Some(lp) => lp.predicted_pages.clone(),
                        None => continue,
                    }
                }
                None => return,
            };

            for (idx, page) in pages.iter().enumerate() {
                if self.current_mode == ActivationMode::Specialist
                    && self.buffer_mgr.is_specialist_pinned(page)
                {
                    continue;
                }

                // Lower priority for further-ahead layers
                let priority = if offset == 2 {
                    PrefetchPriority::Medium
                } else {
                    PrefetchPriority::Low
                };

                let confidence = (0.6 * self.prefetch_multiplier
                    / (offset as f32 * (idx as f32 + 1.0).sqrt()))
                .clamp(0.0, 1.0);

                self.buffer_mgr.submit_prefetch(PrefetchRequest {
                    page: *page,
                    source: Tier::T3Nvme,
                    dest: Tier::T1Vram,
                    priority,
                    deadline_tick: 0,
                    confidence,
                });
            }
        }
    }

    /// Whether a vector index is available.
    pub fn has_vector_index(&self) -> bool {
        self.vector_index.is_some()
    }
}
