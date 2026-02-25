//! # vib3 — Weight-Indexed Inference Engine for MoE Models
//!
//! vib3 is a storage-engine-first inference runtime for large Mixture-of-Experts
//! language models. It treats expert weight matrices as **indexed database tables**
//! rather than monolithic tensors, enabling page-level access with predictive
//! prefetching across a three-tier storage hierarchy (VRAM → RAM → NVMe).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │        Python API (PyO3)        │
//! ├─────────────────────────────────┤
//! │      Runtime / Query Planner    │  ← Async orchestration
//! ├─────────────────────────────────┤
//! │    Page Buffer Manager          │  ← Three-tier page lifecycle
//! ├─────────────────────────────────┤
//! │    Storage Engine / io_uring    │  ← Async page I/O
//! ├─────────────────────────────────┤
//! │    Vector Index                 │  ← Predictive page lookup
//! ├─────────────────────────────────┤
//! │    CUDA Kernel Launcher         │  ← Thin unsafe FFI boundary
//! └─────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```no_run
//! use vib3::Engine;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut engine = Engine::from_path("model.vib3").await?;
//!     let result = engine.generate("Write a Python function that...").await?;
//!     println!("{}", result.text);
//!     println!("{:.1} tokens/s", result.tokens_per_second);
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod compute;
pub mod core;
pub mod index;
pub mod registry;
pub mod runtime;
pub mod storage;
pub mod validation;

// Re-export primary types
pub use crate::core::config::{GearConfig, ModelConfig};
pub use crate::core::error::{Error, Result};
pub use crate::core::types::*;
pub use crate::runtime::engine::Engine;
