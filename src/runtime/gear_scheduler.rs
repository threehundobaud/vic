//! Gear-batched request scheduling (Phase E).
//!
//! When multiple concurrent inference requests arrive with gear annotations,
//! the `GearBatchScheduler` groups them by primary gear before dispatching.
//! This maximizes T1 cache reuse: all requests in a batch share the same
//! expert working set, avoiding the page thrashing that occurs when mixed-gear
//! requests compete for T1 slots.
//!
//! ## Strategy
//!
//! ```text
//! Incoming requests:
//!   Request A: gear=code     (Python function generation)
//!   Request B: gear=code     (JavaScript debugging)
//!   Request C: gear=vision   (screenshot analysis)
//!   Request D: gear=code     (SQL query optimization)
//!   Request E: gear=reason   (math problem solving)
//!
//! Gear-batched execution:
//!   Batch 1: {A, B, D} — all code gear
//!     → Load code-domain expert pages once
//!     → T1 hit rate: ~95% (shared working set)
//!
//!   Batch 2: {C} — vision gear
//!     → Swap to vision-domain expert pages
//!
//!   Batch 3: {E} — reason gear
//!     → Swap to reason-domain expert pages
//! ```
//!
//! ## Graceful Degradation
//!
//! Requests without gear annotations are placed in a "general" queue and
//! scheduled via round-robin when no gear-specific requests are pending.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// A pending inference request in the scheduler.
#[derive(Debug)]
pub struct InferenceRequest {
    /// Unique request ID.
    pub id: String,
    /// Primary gear (None = unclassified).
    pub gear: Option<String>,
    /// When this request was enqueued.
    pub enqueued_at: Instant,
    /// The prompt text (or token IDs — opaque to the scheduler).
    pub prompt: String,
    /// Sampling parameters (serialized or typed — opaque here).
    pub params: RequestParams,
}

/// Minimal sampling params carried with the request.
#[derive(Debug, Clone)]
pub struct RequestParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl Default for RequestParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.9,
            max_tokens: 4096,
        }
    }
}

/// Gear-batched request scheduler.
///
/// Groups concurrent requests by primary gear. Prefers extending the current
/// gear's batch (no page swap) over switching to a different gear.
pub struct GearBatchScheduler {
    /// Pending requests grouped by primary gear.
    /// Key "general" is used for requests without gear annotations.
    gear_queues: HashMap<String, VecDeque<InferenceRequest>>,

    /// Current active gear (the gear whose pages are in T1).
    active_gear: Option<String>,

    /// Maximum batch size (requests per dispatch).
    max_batch_size: usize,

    /// Minimum batch size before scheduling a gear switch.
    /// If the current gear has fewer pending requests than this,
    /// and another gear has more, we switch gears.
    /// Reserved for future smarter scheduling policies.
    _min_batch_for_switch: usize,

    /// Maximum wait time before scheduling an undersized batch.
    /// Prevents starvation of low-volume gears.
    max_wait: Duration,

    /// Total requests enqueued (lifetime counter).
    total_enqueued: u64,

    /// Total batches dispatched (lifetime counter).
    total_dispatched: u64,
}

impl Default for GearBatchScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl GearBatchScheduler {
    /// Create a new scheduler with default settings.
    pub fn new() -> Self {
        Self {
            gear_queues: HashMap::new(),
            active_gear: None,
            max_batch_size: 8,
            _min_batch_for_switch: 2,
            max_wait: Duration::from_millis(100),
            total_enqueued: 0,
            total_dispatched: 0,
        }
    }

    /// Create with custom settings.
    pub fn with_config(
        max_batch_size: usize,
        min_batch_for_switch: usize,
        max_wait_ms: u64,
    ) -> Self {
        Self {
            max_batch_size,
            _min_batch_for_switch: min_batch_for_switch,
            max_wait: Duration::from_millis(max_wait_ms),
            ..Self::new()
        }
    }

    /// Enqueue a new inference request.
    pub fn enqueue(&mut self, request: InferenceRequest) {
        let gear_key = request
            .gear
            .clone()
            .unwrap_or_else(|| "general".to_string());

        self.gear_queues
            .entry(gear_key)
            .or_insert_with(VecDeque::new)
            .push_back(request);

        self.total_enqueued += 1;
    }

    /// Get the next batch of requests to process.
    ///
    /// Strategy:
    /// 1. Prefer extending the current gear's batch (no page swap needed).
    /// 2. If current gear is empty, pick the gear with the most pending requests.
    /// 3. If no requests are pending, return an empty batch.
    /// 4. Enforce max_wait: if any request has been waiting too long, include it
    ///    even if it means a gear switch.
    pub fn next_batch(&mut self) -> Vec<InferenceRequest> {
        // Check for starvation first: any request waiting too long?
        let now = Instant::now();
        let starved_gear = self.find_starved_gear(now);

        // Strategy 1: Extend current gear's batch
        if starved_gear.is_none() {
            if let Some(ref gear) = self.active_gear {
                if let Some(queue) = self.gear_queues.get_mut(gear) {
                    if !queue.is_empty() {
                        let batch = Self::drain_batch(queue, self.max_batch_size);
                        if !batch.is_empty() {
                            self.total_dispatched += 1;
                            return batch;
                        }
                    }
                }
            }
        }

        // Strategy 2: Pick the gear with the most pending requests
        // (or the starved gear if there is one)
        let target_gear = starved_gear.or_else(|| {
            self.gear_queues
                .iter()
                .filter(|(_, q)| !q.is_empty())
                .max_by_key(|(_, q)| q.len())
                .map(|(gear, _)| gear.clone())
        });

        if let Some(gear) = target_gear {
            self.active_gear = Some(gear.clone());
            if let Some(queue) = self.gear_queues.get_mut(&gear) {
                let batch = Self::drain_batch(queue, self.max_batch_size);
                if !batch.is_empty() {
                    self.total_dispatched += 1;
                    return batch;
                }
            }
        }

        vec![]
    }

    /// Total pending requests across all gears.
    pub fn pending_count(&self) -> usize {
        self.gear_queues.values().map(|q| q.len()).sum()
    }

    /// Pending requests per gear.
    pub fn pending_by_gear(&self) -> HashMap<String, usize> {
        self.gear_queues
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(gear, q)| (gear.clone(), q.len()))
            .collect()
    }

    /// Current active gear.
    pub fn active_gear(&self) -> Option<&str> {
        self.active_gear.as_deref()
    }

    /// Lifetime stats.
    pub fn stats(&self) -> (u64, u64) {
        (self.total_enqueued, self.total_dispatched)
    }

    // ── Internal ─────────────────────────────────────────────────────

    /// Drain up to max_batch_size requests from a queue.
    fn drain_batch(
        queue: &mut VecDeque<InferenceRequest>,
        max_batch: usize,
    ) -> Vec<InferenceRequest> {
        let n = queue.len().min(max_batch);
        queue.drain(..n).collect()
    }

    /// Find a gear with a starved request (waiting longer than max_wait).
    fn find_starved_gear(&self, now: Instant) -> Option<String> {
        for (gear, queue) in &self.gear_queues {
            if let Some(front) = queue.front() {
                if now.duration_since(front.enqueued_at) >= self.max_wait {
                    return Some(gear.clone());
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: &str, gear: Option<&str>) -> InferenceRequest {
        InferenceRequest {
            id: id.to_string(),
            gear: gear.map(|s| s.to_string()),
            enqueued_at: Instant::now(),
            prompt: format!("prompt for {}", id),
            params: RequestParams::default(),
        }
    }

    #[test]
    fn test_basic_scheduling() {
        let mut sched = GearBatchScheduler::new();

        sched.enqueue(make_request("a", Some("code")));
        sched.enqueue(make_request("b", Some("code")));
        sched.enqueue(make_request("c", Some("vision")));
        sched.enqueue(make_request("d", Some("code")));

        assert_eq!(sched.pending_count(), 4);

        // First batch should pick the gear with the most requests (code: 3)
        let batch1 = sched.next_batch();
        assert_eq!(batch1.len(), 3);
        assert!(batch1.iter().all(|r| r.gear.as_deref() == Some("code")));
        assert_eq!(sched.active_gear(), Some("code"));

        // Next batch should be vision (only remaining)
        let batch2 = sched.next_batch();
        assert_eq!(batch2.len(), 1);
        assert_eq!(batch2[0].gear.as_deref(), Some("vision"));

        // Nothing left
        assert_eq!(sched.pending_count(), 0);
        assert!(sched.next_batch().is_empty());
    }

    #[test]
    fn test_prefer_current_gear() {
        let mut sched = GearBatchScheduler::new();

        // Set active gear to "code"
        sched.enqueue(make_request("a", Some("code")));
        let _ = sched.next_batch(); // activates "code"

        // Now add mixed requests
        sched.enqueue(make_request("b", Some("code")));
        sched.enqueue(make_request("c", Some("vision")));
        sched.enqueue(make_request("d", Some("vision")));
        sched.enqueue(make_request("e", Some("code")));

        // Should prefer code (current active gear) even though vision has 2
        let batch = sched.next_batch();
        assert!(batch.iter().all(|r| r.gear.as_deref() == Some("code")));
    }

    #[test]
    fn test_general_queue() {
        let mut sched = GearBatchScheduler::new();

        sched.enqueue(make_request("a", None));
        sched.enqueue(make_request("b", None));

        let batch = sched.next_batch();
        assert_eq!(batch.len(), 2);
        // Unclassified requests go to "general" queue
        assert_eq!(sched.active_gear(), Some("general"));
    }

    #[test]
    fn test_batch_size_limit() {
        let mut sched = GearBatchScheduler::with_config(2, 1, 100);

        for i in 0..5 {
            sched.enqueue(make_request(&format!("r{}", i), Some("code")));
        }

        let batch = sched.next_batch();
        assert_eq!(batch.len(), 2); // Limited by max_batch_size
        assert_eq!(sched.pending_count(), 3);
    }
}
