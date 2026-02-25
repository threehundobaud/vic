//! Coactivation graph: which experts tend to fire together.

use crate::storage::format::CoactivationEntry;
use std::collections::HashMap;

pub struct CoactivationGraph {
    /// (layer, expert) → Vec<(neighbor_expert, correlation)>
    adjacency: HashMap<u32, Vec<(u16, f32)>>,
}

impl CoactivationGraph {
    pub fn build(entries: &[CoactivationEntry]) -> Self {
        let mut adjacency: HashMap<u32, Vec<(u16, f32)>> = HashMap::new();

        for entry in entries {
            let key_a = (entry.layer as u32) << 16 | entry.expert_a as u32;
            let key_b = (entry.layer as u32) << 16 | entry.expert_b as u32;

            adjacency
                .entry(key_a)
                .or_default()
                .push((entry.expert_b, entry.correlation));
            adjacency
                .entry(key_b)
                .or_default()
                .push((entry.expert_a, entry.correlation));
        }

        // Sort each adjacency list by correlation descending
        for list in adjacency.values_mut() {
            list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }

        Self { adjacency }
    }

    /// Get experts that frequently co-activate with the given expert.
    pub fn neighbors(&self, layer: u16, expert: u16, min_correlation: f32) -> Vec<(u16, f32)> {
        let key = (layer as u32) << 16 | expert as u32;
        self.adjacency
            .get(&key)
            .map(|list| {
                list.iter()
                    .filter(|(_, corr)| *corr >= min_correlation)
                    .copied()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Given a set of active experts, find all likely co-activated experts.
    pub fn neighborhood(
        &self,
        layer: u16,
        active: &[u16],
        min_correlation: f32,
        max_results: usize,
    ) -> Vec<(u16, f32)> {
        let mut seen: HashMap<u16, f32> = HashMap::new();

        for &expert in active {
            for (neighbor, corr) in self.neighbors(layer, expert, min_correlation) {
                let entry = seen.entry(neighbor).or_insert(0.0);
                *entry = entry.max(corr);
            }
        }

        // Remove already-active experts
        for &expert in active {
            seen.remove(&expert);
        }

        let mut results: Vec<(u16, f32)> = seen.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(max_results);
        results
    }
}
