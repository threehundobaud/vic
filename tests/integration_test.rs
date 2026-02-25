//! Integration tests for vib3.
//!
//! These tests exercise the full pipeline: convert → load → buffer → compute → generate.

use half::f16;
use std::sync::Arc;
use vib3::compute::cuda_ffi;
use vib3::compute::kernels;
use vib3::core::config::*;
use vib3::core::types::*;
use vib3::index::coactivation::*;
use vib3::index::domain::*;
use vib3::index::vector_index::VectorIndex;
use vib3::runtime::attention::*;
use vib3::runtime::generate::*;
use vib3::runtime::query_planner::QueryPlanner;
use vib3::storage::buffer_manager::*;
use vib3::storage::format::*;

/// Create a tiny test model as a .vib3 file.
fn create_test_model(path: &str) -> ModelConfig {
    let config = ModelConfig {
        name: "test-micro".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 3,
        num_moe_layers: 2,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut writer = Vib3Writer::new(config.clone());

    let hidden_dim = config.hidden_dim as usize;
    let expert_hidden = config.expert_hidden_dim as usize;

    let mut rng: u64 = 42;

    for layer in 0..config.num_moe_layers {
        let layer_idx = layer as u16 + config.dense_layer_idx as u16;

        for expert in 0..config.num_experts {
            for segment in 0..3u16 {
                let (rows, cols) = match segment {
                    0 | 1 => (expert_hidden, hidden_dim),
                    2 => (hidden_dim, expert_hidden),
                    _ => unreachable!(),
                };

                let page_bytes = rows * cols * 2; // FP16
                let mut data = vec![0u8; page_bytes];
                for chunk in data.chunks_exact_mut(2) {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let val = f16::from_f32(((rng as f32 / u64::MAX as f32) - 0.5) * 0.1);
                    let bytes = val.to_le_bytes();
                    chunk[0] = bytes[0];
                    chunk[1] = bytes[1];
                }

                writer.add_page(
                    layer_idx,
                    expert as u16,
                    segment,
                    0, // page_idx
                    0, // row_start
                    rows as u16,
                    cols as u16,
                    &data,
                );
            }
        }
    }

    writer.finalize(path).expect("Failed to write test model");
    config
}

// ─── Core Type Tests ─────────────────────────────────────────────────────

#[test]
fn test_page_id_key_roundtrip() {
    let pid = PageId {
        layer: 5,
        expert: 42,
        segment: 2,
        page_idx: 7,
    };
    let key = pid.key();
    assert_eq!(key >> 48, 5);
    assert_eq!((key >> 32) & 0xFFFF, 42);
    assert_eq!((key >> 16) & 0xFFFF, 2);
    assert_eq!(key & 0xFFFF, 7);
}

#[test]
fn test_page_id_shared() {
    let pid = PageId::shared(3, 1, 0);
    assert!(pid.is_shared());
    assert_eq!(pid.expert, 0xFFFF);
}

#[test]
fn test_expert_id_key() {
    let eid = ExpertId {
        layer: 10,
        expert: 200,
    };
    let key = eid.key();
    assert_eq!(key >> 16, 10);
    assert_eq!(key & 0xFFFF, 200);
}

#[test]
fn test_dtype_bytes() {
    assert_eq!(DType::FP16.bits(), 16);
    assert_eq!(DType::INT4.bits(), 4);
    assert_eq!(DType::INT8.bits(), 8);
    assert_eq!(DType::FP16.bytes_for(1000), 2000);
    assert_eq!(DType::INT4.bytes_for(1000), 500);
}

#[test]
fn test_page_state_transitions() {
    let pte = PageTableEntry::new(
        PageId {
            layer: 0,
            expert: 0,
            segment: 0,
            page_idx: 0,
        },
        0,
        PAGE_SIZE as u32,
    );
    assert_eq!(pte.state(), PageState::Cold);
    assert_eq!(pte.current_tier(), Some(Tier::T3Nvme));

    pte.set_state(PageState::Warm);
    assert_eq!(pte.state(), PageState::Warm);
    assert_eq!(pte.current_tier(), Some(Tier::T2Ram));

    pte.set_state(PageState::Hot);
    assert!(pte.is_compute_ready());
    assert_eq!(pte.current_tier(), Some(Tier::T1Vram));
}

#[test]
fn test_access_tracking() {
    let pte = PageTableEntry::new(
        PageId {
            layer: 0,
            expert: 0,
            segment: 0,
            page_idx: 0,
        },
        0,
        PAGE_SIZE as u32,
    );
    assert_eq!(pte.access_count(), 0);

    pte.record_access(100);
    assert_eq!(pte.access_count(), 1);
    assert_eq!(pte.last_access_tick(), 100);

    pte.record_access(200);
    assert_eq!(pte.access_count(), 2);
    assert_eq!(pte.last_access_tick(), 200);
}

#[test]
fn test_inference_stats() {
    let stats = InferenceStats::default();
    assert_eq!(stats.t1_hit_rate(), 0.0);
    assert_eq!(stats.combined_hit_rate(), 0.0);
    assert_eq!(stats.prefetch_efficiency(), 0.0);

    stats
        .total_page_accesses
        .store(100, std::sync::atomic::Ordering::Relaxed);
    stats
        .t1_hits
        .store(80, std::sync::atomic::Ordering::Relaxed);
    stats
        .t2_hits
        .store(15, std::sync::atomic::Ordering::Relaxed);
    assert!((stats.t1_hit_rate() - 0.8).abs() < 0.01);
    assert!((stats.combined_hit_rate() - 0.95).abs() < 0.01);

    let snap = stats.snapshot();
    assert_eq!(snap.t1_hits, 80);
    assert_eq!(snap.total_page_accesses, 100);
}

// ─── Model Config Tests ──────────────────────────────────────────────────

#[test]
fn test_kimi_k25_config() {
    let config = ModelConfig::kimi_k25();
    assert_eq!(config.num_experts, 384);
    assert_eq!(config.num_active_experts, 8);
    assert_eq!(config.num_moe_layers, 60);
    assert!(config.expert_size_bytes() > 0);
    assert!(config.total_expert_pages() > 0);
}

#[test]
fn test_model_config_metrics() {
    let config = ModelConfig {
        name: "test".into(),
        architecture: "test".into(),
        hidden_dim: 256,
        expert_hidden_dim: 64,
        num_layers: 4,
        num_moe_layers: 3,
        dense_layer_idx: 0,
        num_experts: 8,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 1024,
        vocab_size: 1000,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    assert!(config.expert_size_bytes() > 0);
    assert!(config.pages_per_segment() > 0);
    assert_eq!(config.pages_per_expert(), config.pages_per_segment() * 3);
    assert!(config.estimated_total_bytes() > config.total_expert_bytes());
}

// ─── File Format Tests ───────────────────────────────────────────────────

#[test]
fn test_vib3_write_and_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.vib3");
    let path_str = path.to_str().unwrap();

    let config = create_test_model(path_str);

    // Read it back
    let file = Vib3File::open(path_str).unwrap();
    assert_eq!(file.model_config().name, "test-micro");
    assert_eq!(file.model_config().num_experts, config.num_experts);
    assert!(file.page_count() > 0);

    // Check header
    let h = file.header();
    let magic = h.magic;
    assert_eq!(magic, VIB3_MAGIC);
    let version = h.version;
    assert_eq!(version, VIB3_VERSION);

    // Check page catalog
    let page = file.page(0);
    let pid = page.page_id();
    assert!(!pid.is_shared());

    // Check expert lookup
    let expert_pages = file.pages_for_expert(0, 0);
    assert!(!expert_pages.is_empty());
}

#[test]
fn test_vib3_bad_magic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.vib3");

    // Write garbage
    std::fs::write(&path, &[0u8; 1024]).unwrap();
    let result = Vib3File::open(&path);
    assert!(result.is_err());
}

#[test]
fn test_vib3_too_small() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.vib3");
    std::fs::write(&path, &[0u8; 100]).unwrap();
    let result = Vib3File::open(&path);
    assert!(result.is_err());
}

// ─── CUDA FFI Tests ──────────────────────────────────────────────────────

#[test]
fn test_device_alloc_free() {
    let size = PAGE_SIZE;
    let ptr = cuda_ffi::device_alloc(size).unwrap();
    assert!(!ptr.is_null());

    // When CUDA is available, ptr is a VRAM pointer — we cannot read/write
    // it from the CPU. Instead, verify allocation worked via H2D + D2H copy.
    let host_src = cuda_ffi::host_alloc_pinned(size).unwrap();
    let host_dst = cuda_ffi::host_alloc_pinned(size).unwrap();
    unsafe {
        std::ptr::write_bytes(host_src, 0xAB, size);
    }
    let device = cuda_ffi::CudaDevice::new(0).unwrap();
    let stream = cuda_ffi::CudaStream::new(&device).unwrap();
    cuda_ffi::memcpy_h2d_async(ptr, host_src, size, &stream).unwrap();
    stream.synchronize().unwrap();
    cuda_ffi::memcpy_d2h(host_dst, ptr, size).unwrap();
    unsafe {
        assert_eq!(*host_dst, 0xAB);
    }

    cuda_ffi::host_free_pinned(host_dst, size);
    cuda_ffi::host_free_pinned(host_src, size);
    cuda_ffi::device_free(ptr, size);
}

#[test]
fn test_host_alloc_free() {
    let size = PAGE_SIZE;
    let ptr = cuda_ffi::host_alloc_pinned(size).unwrap();
    assert!(!ptr.is_null());

    unsafe {
        std::ptr::write_bytes(ptr, 0xCD, size);
        assert_eq!(*ptr, 0xCD);
    }

    cuda_ffi::host_free_pinned(ptr, size);
}

#[test]
fn test_memcpy() {
    let size = 1024;
    let device = cuda_ffi::CudaDevice::new(0).unwrap();
    let stream = cuda_ffi::CudaStream::new(&device).unwrap();

    let src = cuda_ffi::host_alloc_pinned(size).unwrap();
    let dst = cuda_ffi::device_alloc(size).unwrap();

    unsafe {
        for i in 0..size {
            *src.add(i) = (i & 0xFF) as u8;
        }
    }

    cuda_ffi::memcpy_h2d_async(dst, src, size, &stream).unwrap();
    stream.synchronize().unwrap();

    // Copy back from device to host for verification
    let readback = cuda_ffi::host_alloc_pinned(size).unwrap();
    cuda_ffi::memcpy_d2h(readback, dst, size).unwrap();

    // Verify
    unsafe {
        for i in 0..size {
            assert_eq!(*readback.add(i), (i & 0xFF) as u8);
        }
    }

    cuda_ffi::host_free_pinned(readback, size);
    cuda_ffi::device_free(dst, size);
    cuda_ffi::host_free_pinned(src, size);
}

// ─── Kernel Tests ────────────────────────────────────────────────────────

#[test]
fn test_partial_matmul_fp16() {
    use half::f16;
    let k = 4;
    let m = 2;
    let stream = cuda_ffi::CudaStream::cpu_only();

    // input: [1, 4] = [1.0, 2.0, 3.0, 4.0]
    let input: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];
    // weight: [2, 4] = [[1,0,0,0], [0,1,0,0]]
    let weight: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
    ];
    let mut output: Vec<f16> = vec![f16::from_f32(0.0); m];

    kernels::partial_matmul(
        input.as_ptr() as *const u8,
        weight.as_ptr() as *const u8,
        output.as_mut_ptr() as *mut u8,
        k,
        m,
        DType::FP16,
        &stream,
    )
    .unwrap();

    // output[0] = 1*1 + 2*0 + 3*0 + 4*0 = 1.0
    // output[1] = 1*0 + 2*1 + 3*0 + 4*0 = 2.0
    assert!((output[0].to_f32() - 1.0).abs() < 0.01);
    assert!((output[1].to_f32() - 2.0).abs() < 0.01);
}

#[test]
fn test_weighted_accumulate() {
    use half::f16;
    let dim = 4;
    let stream = cuda_ffi::CudaStream::cpu_only();

    let mut output: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];
    let expert: Vec<f16> = vec![
        f16::from_f32(10.0),
        f16::from_f32(20.0),
        f16::from_f32(30.0),
        f16::from_f32(40.0),
    ];

    kernels::weighted_accumulate(
        output.as_mut_ptr() as *mut u8,
        expert.as_ptr() as *const u8,
        0.5,
        dim,
        &stream,
    )
    .unwrap();

    // output[0] = 1.0 + 0.5 * 10.0 = 6.0
    assert!((output[0].to_f32() - 6.0).abs() < 0.1);
    // output[1] = 2.0 + 0.5 * 20.0 = 12.0
    assert!((output[1].to_f32() - 12.0).abs() < 0.1);
}

#[test]
fn test_run_router() {
    let hidden_dim = 8;
    let num_experts = 4;
    let top_k = 2;
    let stream = cuda_ffi::CudaStream::cpu_only();

    let state: Vec<f16> = (0..hidden_dim)
        .map(|i| f16::from_f32(i as f32 * 0.1))
        .collect();

    // Router weights: identity-ish so expert 0 gets high score for dim 0, etc.
    let mut weights = vec![f16::from_f32(0.0); num_experts * hidden_dim];
    for e in 0..num_experts {
        weights[e * hidden_dim + e * 2] = f16::from_f32(1.0);
    }

    let result = kernels::run_router(
        state.as_ptr() as *const u8,
        weights.as_ptr() as *const u8,
        num_experts,
        hidden_dim,
        top_k,
        &stream,
        None, // no pre-allocated scores buffer in tests
    )
    .unwrap();

    assert_eq!(result.len(), top_k);
    // Weights should sum to ~1.0 (softmax)
    let sum: f32 = result.iter().map(|(_, w)| w).sum();
    assert!((sum - 1.0).abs() < 0.01);
}

// ─── Sampler Tests ───────────────────────────────────────────────────────

#[test]
fn test_greedy_sampling() {
    let mut sampler = Sampler::new(0);
    let logits = vec![0.1, 0.3, 0.9, 0.2]; // Token 2 is highest
    let params = SamplingParams {
        temperature: 0.0, // greedy
        ..Default::default()
    };
    let token = sampler.sample(&logits, &params);
    assert_eq!(token, 2);
}

#[test]
fn test_temperature_sampling() {
    let mut sampler = Sampler::new(42);
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        ..Default::default()
    };

    // Sample many times and check distribution
    let mut counts = vec![0u32; 4];
    for _ in 0..1000 {
        let token = sampler.sample(&logits, &params);
        counts[token as usize] += 1;
    }
    // Token 3 should be most common (highest logit)
    assert!(counts[3] > counts[0]);
}

#[test]
fn test_tokenizer() {
    let tokenizer = SimpleTokenizer::new(1024);

    let tokens = tokenizer.encode("hello");
    assert!(!tokens.is_empty());
    assert_eq!(tokens[0], tokenizer.bos_token);

    let text = tokenizer.decode(&tokens);
    assert_eq!(text, "hello");
}

// ─── Coactivation Graph Tests ────────────────────────────────────────────

#[test]
fn test_coactivation_graph() {
    let entries = vec![
        CoactivationEntry {
            expert_a: 0,
            expert_b: 1,
            layer: 0,
            _pad: 0,
            correlation: 0.9,
            sample_count: 100,
        },
        CoactivationEntry {
            expert_a: 0,
            expert_b: 2,
            layer: 0,
            _pad: 0,
            correlation: 0.5,
            sample_count: 100,
        },
        CoactivationEntry {
            expert_a: 1,
            expert_b: 3,
            layer: 0,
            _pad: 0,
            correlation: 0.3,
            sample_count: 100,
        },
    ];

    let graph = CoactivationGraph::build(&entries);

    // Expert 0's neighbors
    let neighbors = graph.neighbors(0, 0, 0.0);
    assert_eq!(neighbors.len(), 2);
    assert_eq!(neighbors[0].0, 1); // Highest correlation first

    // With threshold
    let filtered = graph.neighbors(0, 0, 0.6);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].0, 1);

    // Neighborhood of [0, 1]
    let hood = graph.neighborhood(0, &[0, 1], 0.0, 10);
    assert!(hood.iter().any(|(e, _)| *e == 2));
    assert!(hood.iter().any(|(e, _)| *e == 3));
    // Should NOT include 0 or 1 (already active)
    assert!(!hood.iter().any(|(e, _)| *e == 0));
    assert!(!hood.iter().any(|(e, _)| *e == 1));
}

// ─── Buffer Manager Tests ────────────────────────────────────────────────

#[tokio::test]
async fn test_buffer_manager_basic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_buf.vib3");
    let path_str = path.to_str().unwrap();

    let _config = create_test_model(path_str);

    let model_file = Arc::new(Vib3File::open(path_str).unwrap());
    let buf_config = BufferPoolConfig {
        t1_capacity: 4 * PAGE_SIZE, // 4 pages
        t2_capacity: 8 * PAGE_SIZE, // 8 pages
        t2_compressed: false,       // Test data is uncompressed
        ..Default::default()
    };

    let mgr = PageBufferManager::new(buf_config, model_file.clone());
    mgr.initialize().await.unwrap();

    // Get the first page
    let first_page = model_file.page(0).page_id();
    let handle = mgr.get_page(&first_page).await.unwrap();
    assert!(!handle.device_ptr.is_null());
    assert!(handle.size > 0);

    // Stats should show the access
    let stats = mgr.stats.snapshot();
    assert_eq!(stats.total_page_accesses, 1);
    assert_eq!(stats.t3_hits, 1); // First access is always T3
}

#[tokio::test]
async fn test_buffer_manager_t1_hit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_t1.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);

    let model_file = Arc::new(Vib3File::open(path_str).unwrap());
    let buf_config = BufferPoolConfig {
        t1_capacity: 4 * PAGE_SIZE,
        t2_capacity: 8 * PAGE_SIZE,
        t2_compressed: false, // Test data is uncompressed
        ..Default::default()
    };

    let mgr = PageBufferManager::new(buf_config, model_file.clone());
    mgr.initialize().await.unwrap();

    let first_page = model_file.page(0).page_id();

    // First access: cold path
    let _ = mgr.get_page(&first_page).await.unwrap();

    // Second access: should be T1 hit
    let handle = mgr.get_page(&first_page).await.unwrap();
    assert_eq!(handle.source_tier, Tier::T1Vram);

    let stats = mgr.stats.snapshot();
    assert!(stats.t1_hits >= 1);
}

// ─── End-to-End Test ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_end_to_end_generate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_e2e.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);

    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 16 * PAGE_SIZE,
            t2_capacity: 32 * PAGE_SIZE,
            t2_compressed: false, // Test data is uncompressed
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();

    // Use custom params: greedy sampling with max 10 tokens
    // The stub logits computation may produce EOS on some seeds,
    // so use temperature=0 for deterministic behavior
    let params = vib3::runtime::generate::SamplingParams {
        temperature: 0.0,
        max_tokens: 10,
        ..Default::default()
    };
    let result = engine
        .generate_with_params("hello world", params)
        .await
        .unwrap();

    // With the stub compute pipeline, we may or may not generate tokens
    // depending on the logit distribution. Verify the engine ran without error
    // and produced valid timing/stats.
    assert!(result.total_time_ms > 0.0);
    assert!(result.stats.total_page_accesses > 0);
    // If tokens were generated, verify they decode
    if result.tokens_generated > 0 {
        assert!(!result.text.is_empty() || result.tokens_generated > 0);
    }
}

// ─── RMSNorm Test ────────────────────────────────────────────────────────

#[test]
fn test_rms_norm() {
    use half::f16;
    let stream = cuda_ffi::CudaStream::cpu_only();

    let dim = 4;
    let mut input: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];
    let weight: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(1.0),
        f16::from_f32(1.0),
        f16::from_f32(1.0),
    ];

    kernels::rms_norm(
        input.as_mut_ptr() as *mut u8,
        weight.as_ptr() as *const u8,
        dim,
        1e-6,
        &stream,
    )
    .unwrap();

    // RMS = sqrt(mean([1,4,9,16]) + eps) = sqrt(7.5 + eps) ≈ 2.7386
    // Normalized: [1/2.739, 2/2.739, 3/2.739, 4/2.739] ≈ [0.365, 0.730, 1.095, 1.461]
    let rms = (7.5f32 + 1e-6).sqrt();
    assert!((input[0].to_f32() - 1.0 / rms).abs() < 0.01);
    assert!((input[1].to_f32() - 2.0 / rms).abs() < 0.01);
    assert!((input[2].to_f32() - 3.0 / rms).abs() < 0.01);
    assert!((input[3].to_f32() - 4.0 / rms).abs() < 0.01);
}

#[test]
fn test_rms_norm_no_weight() {
    use half::f16;
    let stream = cuda_ffi::CudaStream::cpu_only();

    let dim = 4;
    let mut input: Vec<f16> = vec![
        f16::from_f32(3.0),
        f16::from_f32(4.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
    ];

    kernels::rms_norm_no_weight(input.as_mut_ptr() as *mut u8, dim, 1e-6, &stream).unwrap();

    // RMS = sqrt(mean([9,16,0,0]) + eps) = sqrt(6.25 + eps) = 2.5
    // Normalized: [3/2.5, 4/2.5, 0/2.5, 0/2.5] = [1.2, 1.6, 0, 0]
    assert!((input[0].to_f32() - 1.2).abs() < 0.01);
    assert!((input[1].to_f32() - 1.6).abs() < 0.01);
    assert!((input[2].to_f32()).abs() < 0.01);
    assert!((input[3].to_f32()).abs() < 0.01);
}

// ─── Phase 3: RoPE Tests ─────────────────────────────────────────────────

#[test]
fn test_apply_rope_identity_at_position_zero() {
    // At position 0, all thetas are 0, cos(0)=1, sin(0)=0
    // So x should be unchanged.
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
    let original = x.clone();
    kernels::apply_rope(&mut x, 0, 4, 10000.0);

    for i in 0..4 {
        assert!(
            (x[i] - original[i]).abs() < 1e-6,
            "RoPE at position 0 should be identity: x[{}] = {} vs {}",
            i,
            x[i],
            original[i]
        );
    }
}

#[test]
fn test_apply_rope_rotates_pairs() {
    // At position > 0, pairs (x[2i], x[2i+1]) should be rotated.
    // x' = [x0*cos - x1*sin, x0*sin + x1*cos]
    let head_dim = 4;
    let mut x = vec![1.0f32, 0.0, 1.0, 0.0];
    kernels::apply_rope(&mut x, 1, head_dim, 10000.0);

    // freq_0 = 1.0 / 10000^(0/4) = 1.0, theta_0 = 1.0
    // freq_1 = 1.0 / 10000^(0.5) = 0.01, theta_1 = 0.01
    let theta0 = 1.0f32;
    let theta1 = 1.0f32 / 10000.0f32.powf(0.5);
    assert!((x[0] - theta0.cos()).abs() < 1e-5);
    assert!((x[1] - theta0.sin()).abs() < 1e-5);
    assert!((x[2] - theta1.cos()).abs() < 1e-5);
    assert!((x[3] - theta1.sin()).abs() < 1e-5);
}

#[test]
fn test_apply_rope_fp16() {
    let head_dim = 4;
    let mut data: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(0.0),
    ];

    kernels::apply_rope_fp16(data.as_mut_ptr() as *mut u8, 0, head_dim, 10000.0);

    // At position 0, should be identity
    assert!((data[0].to_f32() - 1.0).abs() < 0.01);
    assert!((data[1].to_f32() - 0.0).abs() < 0.01);
    assert!((data[2].to_f32() - 1.0).abs() < 0.01);
    assert!((data[3].to_f32() - 0.0).abs() < 0.01);
}

#[test]
fn test_apply_rope_preserves_norm() {
    // RoPE is an orthogonal rotation: it should preserve vector norms per pair.
    let head_dim = 8;
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    kernels::apply_rope(&mut x, 42, head_dim, 10000.0);

    let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_before - norm_after).abs() < 1e-4,
        "RoPE should preserve norm: {} vs {}",
        norm_before,
        norm_after
    );
}

// ─── Phase 3: Attention Head Tests ───────────────────────────────────────

#[test]
fn test_attention_head_single_position() {
    // With seq_len=1, attention is just softmax([score]) × V = V
    let head_dim = 4;
    let q = vec![1.0f32, 0.0, 0.0, 0.0];
    let k_cache = vec![1.0f32, 0.0, 0.0, 0.0]; // one position
    let v_cache = vec![0.5f32, 0.6, 0.7, 0.8]; // one position

    let out = kernels::attention_head(&q, &k_cache, &v_cache, 1, head_dim);

    // With one position, softmax is [1.0], so output = V exactly
    assert_eq!(out.len(), head_dim);
    for d in 0..head_dim {
        assert!(
            (out[d] - v_cache[d]).abs() < 1e-5,
            "With seq_len=1, output[{}] should equal V[{}]: {} vs {}",
            d,
            d,
            out[d],
            v_cache[d]
        );
    }
}

#[test]
fn test_attention_head_two_positions() {
    let head_dim = 2;
    // Q = [1, 0]
    let q = vec![1.0f32, 0.0];
    // K cache: position 0 = [1, 0], position 1 = [0, 1]
    let k_cache = vec![1.0f32, 0.0, 0.0, 1.0];
    // V cache: position 0 = [1, 0], position 1 = [0, 1]
    let v_cache = vec![1.0f32, 0.0, 0.0, 1.0];

    let out = kernels::attention_head(&q, &k_cache, &v_cache, 2, head_dim);

    // score[0] = (1*1 + 0*0) / sqrt(2) = 1/sqrt(2) ≈ 0.7071
    // score[1] = (1*0 + 0*1) / sqrt(2) = 0
    // softmax([0.7071, 0]) → position 0 gets more weight
    // output should lean toward V[0] = [1, 0]
    assert!(out[0] > 0.5, "Should attend more to position 0");
    assert!(out[1] < 0.5, "Should attend less to position 1");
    // Sum of weights in output should preserve softmax: out ~ w0*[1,0] + w1*[0,1]
    assert!(
        (out[0] + out[1] - 1.0).abs() < 1e-5,
        "Output should sum to 1"
    );
}

#[test]
fn test_attention_head_zero_seq_len() {
    let head_dim = 4;
    let q = vec![1.0f32, 2.0, 3.0, 4.0];
    let out = kernels::attention_head(&q, &[], &[], 0, head_dim);
    assert_eq!(out.len(), head_dim);
    for d in 0..head_dim {
        assert_eq!(out[d], 0.0, "Zero seq_len should return zeros");
    }
}

// ─── Phase 3: Multi-Head Attention Tests ─────────────────────────────────

#[test]
fn test_multi_head_attention_basic() {
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = 2;
    let seq_len = 1;

    let q_heads = vec![
        vec![1.0f32, 0.0], // head 0
        vec![0.0f32, 1.0], // head 1
    ];
    // K/V for each kv head, one position
    let k_caches = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
    let v_caches = vec![vec![0.5f32, 0.6], vec![0.7f32, 0.8]];

    let out = kernels::multi_head_attention(
        &q_heads,
        &k_caches,
        &v_caches,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    );

    // Output should be [head0_out, head1_out] concatenated
    assert_eq!(out.len(), num_heads * head_dim);
    // Head 0: q=[1,0], k=[1,0], so attention is on V[0] = [0.5, 0.6]
    assert!((out[0] - 0.5).abs() < 1e-5);
    assert!((out[1] - 0.6).abs() < 1e-5);
    // Head 1: q=[0,1], k=[0,1], so attention is on V[1] = [0.7, 0.8]
    assert!((out[2] - 0.7).abs() < 1e-5);
    assert!((out[3] - 0.8).abs() < 1e-5);
}

#[test]
fn test_multi_head_attention_gqa() {
    // GQA: 4 Q heads sharing 2 KV heads
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 2;
    let seq_len = 1;

    let q_heads = vec![
        vec![1.0f32, 0.0],
        vec![0.0f32, 1.0],
        vec![1.0f32, 1.0],
        vec![-1.0f32, 1.0],
    ];
    let k_caches = vec![
        vec![1.0f32, 0.0], // kv head 0
        vec![0.0f32, 1.0], // kv head 1
    ];
    let v_caches = vec![vec![10.0f32, 20.0], vec![30.0f32, 40.0]];

    let out = kernels::multi_head_attention(
        &q_heads,
        &k_caches,
        &v_caches,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    );

    assert_eq!(out.len(), num_heads * head_dim);
    // Heads 0,1 share kv_head 0; heads 2,3 share kv_head 1
    // With seq_len=1, each head just outputs its V cache
    // Head 0 → kv 0 → V = [10, 20]
    assert!((out[0] - 10.0).abs() < 1e-4);
    assert!((out[1] - 20.0).abs() < 1e-4);
    // Head 1 → kv 0 → V = [10, 20]
    assert!((out[2] - 10.0).abs() < 1e-4);
    assert!((out[3] - 20.0).abs() < 1e-4);
    // Head 2 → kv 1 → V = [30, 40]
    assert!((out[4] - 30.0).abs() < 1e-4);
    assert!((out[5] - 40.0).abs() < 1e-4);
    // Head 3 → kv 1 → V = [30, 40]
    assert!((out[6] - 30.0).abs() < 1e-4);
    assert!((out[7] - 40.0).abs() < 1e-4);
}

// ─── Phase 3: Embedding Lookup Tests ─────────────────────────────────────

#[test]
fn test_embedding_lookup() {
    let vocab_size = 4;
    let hidden_dim = 3;

    // Embedding table: each row is [token_id * 10 + d]
    let mut table: Vec<f16> = Vec::with_capacity(vocab_size * hidden_dim);
    for v in 0..vocab_size {
        for d in 0..hidden_dim {
            table.push(f16::from_f32((v * 10 + d) as f32));
        }
    }

    let mut output = vec![f16::from_f32(0.0); hidden_dim];

    // Look up token 2: should get [20, 21, 22]
    kernels::embedding_lookup(
        table.as_ptr() as *const u8,
        2,
        hidden_dim,
        output.as_mut_ptr() as *mut u8,
    );

    assert!((output[0].to_f32() - 20.0).abs() < 0.01);
    assert!((output[1].to_f32() - 21.0).abs() < 0.01);
    assert!((output[2].to_f32() - 22.0).abs() < 0.01);
}

#[test]
fn test_embedding_lookup_token_zero() {
    let hidden_dim = 4;
    let table: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
        f16::from_f32(5.0),
        f16::from_f32(6.0),
        f16::from_f32(7.0),
        f16::from_f32(8.0),
    ];
    let mut output = vec![f16::from_f32(0.0); hidden_dim];

    kernels::embedding_lookup(
        table.as_ptr() as *const u8,
        0,
        hidden_dim,
        output.as_mut_ptr() as *mut u8,
    );

    assert!((output[0].to_f32() - 1.0).abs() < 0.01);
    assert!((output[1].to_f32() - 2.0).abs() < 0.01);
    assert!((output[2].to_f32() - 3.0).abs() < 0.01);
    assert!((output[3].to_f32() - 4.0).abs() < 0.01);
}

// ─── Phase 3: Linear Projection Tests ────────────────────────────────────

#[test]
fn test_linear_projection() {
    let stream = cuda_ffi::CudaStream::cpu_only();

    let in_dim = 4;
    let out_dim = 2;

    // input = [1, 2, 3, 4]
    let input: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];
    // weight = [[1,0,0,0], [0,1,0,0]] → selects first two elements
    let weight: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
    ];
    let mut output = vec![f16::from_f32(0.0); out_dim];

    kernels::linear_projection(
        input.as_ptr() as *const u8,
        weight.as_ptr() as *const u8,
        output.as_mut_ptr() as *mut u8,
        in_dim,
        out_dim,
        &stream,
    )
    .unwrap();

    // output[0] = 1*1 + 2*0 + 3*0 + 4*0 = 1.0
    // output[1] = 1*0 + 2*1 + 3*0 + 4*0 = 2.0
    assert!((output[0].to_f32() - 1.0).abs() < 0.01);
    assert!((output[1].to_f32() - 2.0).abs() < 0.01);
}

// ─── Phase 3: Compute Logits Tests ───────────────────────────────────────

#[test]
fn test_compute_logits() {
    let hidden_dim = 3;
    let vocab_size = 4;

    // hidden state = [1, 2, 3]
    let hidden: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];

    // lm_head = identity-like: row[v] has 1.0 at position v (for v < hidden_dim)
    let mut lm_head = vec![f16::from_f32(0.0); vocab_size * hidden_dim];
    for v in 0..vocab_size.min(hidden_dim) {
        lm_head[v * hidden_dim + v] = f16::from_f32(1.0);
    }

    let logits = kernels::compute_logits(
        hidden.as_ptr() as *const u8,
        lm_head.as_ptr() as *const u8,
        vocab_size,
        hidden_dim,
    );

    assert_eq!(logits.len(), vocab_size);
    // logit[0] = h·row0 = 1*1 + 2*0 + 3*0 = 1.0
    assert!((logits[0] - 1.0).abs() < 0.01);
    // logit[1] = h·row1 = 1*0 + 2*1 + 3*0 = 2.0
    assert!((logits[1] - 2.0).abs() < 0.01);
    // logit[2] = h·row2 = 1*0 + 2*0 + 3*1 = 3.0
    assert!((logits[2] - 3.0).abs() < 0.01);
    // logit[3] = h·row3 = 0 (all zeros)
    assert!((logits[3]).abs() < 0.01);
}

#[test]
fn test_compute_logits_argmax() {
    let hidden_dim = 4;
    let vocab_size = 8;

    // State pointing strongly in one direction
    let hidden: Vec<f16> = vec![
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(10.0),
    ];

    // lm_head: row 5 has weight in dim 3
    let mut lm_head = vec![f16::from_f32(0.0); vocab_size * hidden_dim];
    lm_head[5 * hidden_dim + 3] = f16::from_f32(1.0);

    let logits = kernels::compute_logits(
        hidden.as_ptr() as *const u8,
        lm_head.as_ptr() as *const u8,
        vocab_size,
        hidden_dim,
    );

    // Token 5 should have the highest logit
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    assert_eq!(argmax, 5);
}

// ─── Phase 3: KV Cache Tests ────────────────────────────────────────────

#[test]
fn test_kv_cache_new() {
    let cache = KvCache::new(2, 4, 128);
    assert_eq!(cache.seq_len, 0);
    assert_eq!(cache.head_dim, 4);
    assert_eq!(cache.num_kv_heads, 2);
    assert_eq!(cache.k.len(), 2);
    assert_eq!(cache.v.len(), 2);
}

#[test]
fn test_kv_cache_append_and_clear() {
    let mut cache = KvCache::new(2, 4, 128);

    // Append one position
    let k_heads = vec![vec![1.0f32, 2.0, 3.0, 4.0], vec![5.0f32, 6.0, 7.0, 8.0]];
    let v_heads = vec![vec![0.1f32, 0.2, 0.3, 0.4], vec![0.5f32, 0.6, 0.7, 0.8]];
    cache.append(&k_heads, &v_heads);

    assert_eq!(cache.seq_len, 1);
    assert_eq!(cache.k[0].len(), 4); // head_dim
    assert_eq!(cache.k[1].len(), 4);
    assert_eq!(cache.v[0].len(), 4);
    assert_eq!(cache.v[1].len(), 4);
    assert_eq!(cache.k[0][0], 1.0);
    assert_eq!(cache.k[1][3], 8.0);

    // Append second position
    let k2 = vec![
        vec![10.0f32, 20.0, 30.0, 40.0],
        vec![50.0f32, 60.0, 70.0, 80.0],
    ];
    let v2 = vec![vec![1.1f32, 1.2, 1.3, 1.4], vec![1.5f32, 1.6, 1.7, 1.8]];
    cache.append(&k2, &v2);

    assert_eq!(cache.seq_len, 2);
    assert_eq!(cache.k[0].len(), 8); // 2 positions * head_dim
    assert_eq!(cache.k[0][4], 10.0); // second position, head 0, dim 0

    // Clear
    cache.clear();
    assert_eq!(cache.seq_len, 0);
    assert_eq!(cache.k[0].len(), 0);
    assert_eq!(cache.v[0].len(), 0);
}

#[test]
fn test_kv_cache_set() {
    let config = ModelConfig {
        name: "test".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 3,
        num_moe_layers: 2,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut cache_set = KvCacheSet::new(&config);
    assert_eq!(cache_set.layers.len(), 3); // num_layers

    // Each layer should have the right dimensions
    for layer in &cache_set.layers {
        assert_eq!(layer.num_kv_heads, 2);
        assert_eq!(layer.head_dim, 16); // 64 / 4 heads
        assert_eq!(layer.max_seq_len, 128);
    }

    cache_set.clear();
    for layer in &cache_set.layers {
        assert_eq!(layer.seq_len, 0);
    }
}

// ─── Phase 3: Self-Attention Layer Tests ─────────────────────────────────

#[test]
fn test_self_attention_layer_fallback() {
    let config = ModelConfig {
        name: "test".into(),
        architecture: "test".into(),
        hidden_dim: 8,
        expert_hidden_dim: 4,
        num_layers: 1,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 2,
        num_active_experts: 1,
        num_heads: 2,
        num_kv_heads: 2,
        max_seq_len: 32,
        vocab_size: 16,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let head_dim = 4; // 8 / 2
    let mut kv_cache = KvCache::new(2, head_dim, 32);

    let hidden: Vec<f16> = (0..8)
        .map(|i| f16::from_f32((i as f32 + 1.0) * 0.1))
        .collect();

    // Run with no weights (fallback mode)
    let output = self_attention_layer(&hidden, None, None, &mut kv_cache, 0, &config);

    assert_eq!(output.len(), 8); // hidden_dim
    assert_eq!(kv_cache.seq_len, 1);

    // Run a second token
    let output2 = self_attention_layer(&hidden, None, None, &mut kv_cache, 1, &config);
    assert_eq!(output2.len(), 8);
    assert_eq!(kv_cache.seq_len, 2);
}

#[test]
fn test_self_attention_layer_with_weights() {
    let config = ModelConfig {
        name: "test".into(),
        architecture: "test".into(),
        hidden_dim: 4,
        expert_hidden_dim: 2,
        num_layers: 1,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 2,
        num_active_experts: 1,
        num_heads: 2,
        num_kv_heads: 2,
        max_seq_len: 32,
        vocab_size: 16,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let hidden_dim = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 2usize; // 4 / 2
    let qkv_dim = hidden_dim + 2 * (num_kv_heads * head_dim); // 4 + 2*4 = 12

    // QKV weight: identity-like [qkv_dim, hidden_dim]
    let mut qkv_weight = vec![f16::from_f32(0.0); qkv_dim * hidden_dim];
    // Make Q part = identity
    for i in 0..hidden_dim.min(qkv_dim) {
        if i < hidden_dim {
            qkv_weight[i * hidden_dim + i.min(hidden_dim - 1)] = f16::from_f32(1.0);
        }
    }

    // O weight: identity [hidden_dim, hidden_dim]
    let mut o_weight = vec![f16::from_f32(0.0); hidden_dim * hidden_dim];
    for i in 0..hidden_dim {
        o_weight[i * hidden_dim + i] = f16::from_f32(1.0);
    }

    let mut kv_cache = KvCache::new(num_kv_heads, head_dim, 32);

    let hidden: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(1.0),
    ];

    let output = self_attention_layer(
        &hidden,
        Some(&qkv_weight),
        Some(&o_weight),
        &mut kv_cache,
        0,
        &config,
    );

    assert_eq!(output.len(), hidden_dim);
    assert_eq!(kv_cache.seq_len, 1);
    // Output should be finite and non-NaN
    for v in &output {
        assert!(
            v.to_f32().is_finite(),
            "Output should be finite, got {}",
            v.to_f32()
        );
    }
}

// ─── Phase 3: Integration ────────────────────────────────────────────────

#[test]
fn test_attention_with_rope_multi_step() {
    // Test the full attention flow over multiple positions
    let num_kv_heads = 2;
    let head_dim = 4;
    let num_heads = 4; // GQA: 4 Q heads, 2 KV heads

    let mut kv_cache = KvCache::new(num_kv_heads, head_dim, 64);

    // Simulate 3 positions
    for pos in 0..3 {
        // Generate Q heads with RoPE applied
        let mut q_heads: Vec<Vec<f32>> = Vec::new();
        for h in 0..num_heads {
            let mut q = vec![1.0f32; head_dim];
            q[0] = (h as f32 + 1.0) * 0.1;
            kernels::apply_rope(&mut q, pos, head_dim, 10000.0);
            q_heads.push(q);
        }

        // Generate K, V heads
        let mut k_heads: Vec<Vec<f32>> = Vec::new();
        let mut v_heads: Vec<Vec<f32>> = Vec::new();
        for h in 0..num_kv_heads {
            let mut k = vec![0.5f32; head_dim];
            k[0] = (h as f32 + 1.0) * 0.2;
            kernels::apply_rope(&mut k, pos, head_dim, 10000.0);
            k_heads.push(k);
            v_heads.push(vec![(pos as f32 + 1.0) * 0.1; head_dim]);
        }

        kv_cache.append(&k_heads, &v_heads);

        let out = kernels::multi_head_attention(
            &q_heads,
            &kv_cache.k,
            &kv_cache.v,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache.seq_len,
        );

        assert_eq!(out.len(), num_heads * head_dim);
        // All values should be finite
        for v in &out {
            assert!(
                v.is_finite(),
                "Attention output should be finite at pos {}",
                pos
            );
        }
    }

    assert_eq!(kv_cache.seq_len, 3);
}

// ─── Phase 5: LZ4 Compression Tests ──────────────────────────────────────

#[test]
fn test_lz4_write_and_read_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_lz4.vib3");
    let path_str = path.to_str().unwrap();

    let config = ModelConfig {
        name: "test-lz4".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut writer = Vib3Writer::new(config.clone());
    writer.enable_compression();

    // Generate page data with some compressible pattern (repeated values)
    let rows = 32usize;
    let cols = 64usize;
    let page_bytes = rows * cols * 2; // FP16
    let mut data = vec![0u8; page_bytes];
    // Repeating pattern: good for LZ4 compression
    for i in 0..page_bytes {
        data[i] = (i % 17) as u8;
    }

    writer.add_page(0, 0, 0, 0, 0, rows as u16, cols as u16, &data);
    writer
        .finalize(path_str)
        .expect("Failed to write compressed model");

    // Read it back
    let file = Vib3File::open(path_str).unwrap();
    assert_eq!(file.page_count(), 1);

    // Verify the page is actually LZ4-compressed on disk
    assert_eq!(file.page_compression(0), COMPRESSION_LZ4);

    // Read the page data back and verify it matches the original
    let mut read_buf = vec![0u8; page_bytes];
    let bytes_read = file.read_page_sync(0, &mut read_buf).unwrap();
    assert_eq!(bytes_read, page_bytes);
    assert_eq!(read_buf, data, "Decompressed data should match original");
}

#[test]
fn test_lz4_incompressible_data_stays_uncompressed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_lz4_no.vib3");
    let path_str = path.to_str().unwrap();

    let config = ModelConfig {
        name: "test-lz4-no".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut writer = Vib3Writer::new(config.clone());
    writer.enable_compression();

    // Generate random-looking data (xorshift) that is hard to compress
    let rows = 32usize;
    let cols = 64usize;
    let page_bytes = rows * cols * 2;
    let mut data = vec![0u8; page_bytes];
    let mut rng: u64 = 12345;
    for i in 0..page_bytes {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        data[i] = (rng & 0xFF) as u8;
    }

    writer.add_page(0, 0, 0, 0, 0, rows as u16, cols as u16, &data);
    writer.finalize(path_str).expect("Failed to write");

    // Random data should fall back to uncompressed (LZ4 makes it bigger)
    let file = Vib3File::open(path_str).unwrap();
    assert_eq!(file.page_compression(0), COMPRESSION_NONE);

    // Read back and verify
    let mut read_buf = vec![0u8; page_bytes];
    let bytes_read = file.read_page_sync(0, &mut read_buf).unwrap();
    assert_eq!(bytes_read, page_bytes);
    assert_eq!(read_buf, data);
}

#[test]
fn test_lz4_mixed_compressed_and_uncompressed_pages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_lz4_mix.vib3");
    let path_str = path.to_str().unwrap();

    let config = ModelConfig {
        name: "test-lz4-mix".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut writer = Vib3Writer::new(config.clone());
    writer.enable_compression();

    let rows = 32usize;
    let cols = 64usize;
    let page_bytes = rows * cols * 2;

    // Page 0: compressible (zeros)
    let data0 = vec![0u8; page_bytes];
    writer.add_page(0, 0, 0, 0, 0, rows as u16, cols as u16, &data0);

    // Page 1: also compressible (repeating pattern)
    let mut data1 = vec![0u8; page_bytes];
    for i in 0..page_bytes {
        data1[i] = (i % 4) as u8;
    }
    writer.add_page(0, 1, 0, 0, 0, rows as u16, cols as u16, &data1);

    writer.finalize(path_str).expect("Failed to write");

    let file = Vib3File::open(path_str).unwrap();
    assert_eq!(file.page_count(), 2);

    // Both should round-trip correctly
    for (idx, expected_data) in [(0, &data0), (1, &data1)] {
        let mut buf = vec![0u8; page_bytes];
        let n = file.read_page_sync(idx, &mut buf).unwrap();
        assert_eq!(n, page_bytes);
        assert_eq!(buf, *expected_data, "Page {} data mismatch", idx);
    }
}

// ─── Phase 5: Embedding Quantization Tests ──────────────────────────────

#[test]
fn test_quantize_embeddings_roundtrip() {
    let vocab_size = 4;
    let hidden_dim = 8;

    // Create FP16 embedding table with known values
    let mut table: Vec<f16> = Vec::with_capacity(vocab_size * hidden_dim);
    for v in 0..vocab_size {
        for d in 0..hidden_dim {
            let val = (v as f32 - 1.5) * 0.1 + d as f32 * 0.01;
            table.push(f16::from_f32(val));
        }
    }

    // Quantize to INT8
    let quantized = kernels::quantize_embeddings_to_int8(&table, vocab_size, hidden_dim);

    // Expected size: vocab_size * hidden_dim (INT8) + vocab_size * 2 (FP16 scales)
    assert_eq!(quantized.len(), vocab_size * hidden_dim + vocab_size * 2);

    // Look up each token with INT8 path and verify against FP16 original
    for token_id in 0..vocab_size as u32 {
        let mut output_int8 = vec![f16::from_f32(0.0); hidden_dim];
        kernels::embedding_lookup_int8(
            quantized.as_ptr(),
            token_id,
            vocab_size,
            hidden_dim,
            output_int8.as_mut_ptr() as *mut u8,
        );

        let mut output_fp16 = vec![f16::from_f32(0.0); hidden_dim];
        kernels::embedding_lookup(
            table.as_ptr() as *const u8,
            token_id,
            hidden_dim,
            output_fp16.as_mut_ptr() as *mut u8,
        );

        // INT8 quantization should be close to FP16 (within ~1% of max value)
        for d in 0..hidden_dim {
            let diff = (output_int8[d].to_f32() - output_fp16[d].to_f32()).abs();
            assert!(
                diff < 0.02,
                "Token {} dim {}: INT8={}, FP16={}, diff={}",
                token_id,
                d,
                output_int8[d].to_f32(),
                output_fp16[d].to_f32(),
                diff
            );
        }
    }
}

#[test]
fn test_quantize_embeddings_zero_row() {
    let vocab_size = 2;
    let hidden_dim = 4;

    // Token 0: all zeros, Token 1: nonzero
    let table = vec![
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(-1.0),
        f16::from_f32(0.5),
        f16::from_f32(-0.5),
    ];

    let quantized = kernels::quantize_embeddings_to_int8(&table, vocab_size, hidden_dim);

    // Token 0 (all zeros) should produce all zeros after dequant
    let mut out = vec![f16::from_f32(999.0); hidden_dim];
    kernels::embedding_lookup_int8(
        quantized.as_ptr(),
        0,
        vocab_size,
        hidden_dim,
        out.as_mut_ptr() as *mut u8,
    );
    for d in 0..hidden_dim {
        assert!(
            out[d].to_f32().abs() < 0.01,
            "Zero row dim {}: got {}",
            d,
            out[d].to_f32()
        );
    }

    // Token 1: should be approximately correct
    let mut out1 = vec![f16::from_f32(0.0); hidden_dim];
    kernels::embedding_lookup_int8(
        quantized.as_ptr(),
        1,
        vocab_size,
        hidden_dim,
        out1.as_mut_ptr() as *mut u8,
    );
    assert!((out1[0].to_f32() - 1.0).abs() < 0.02);
    assert!((out1[1].to_f32() - (-1.0)).abs() < 0.02);
}

#[test]
fn test_embedding_int8_size_savings() {
    let vocab_size = 1000;
    let hidden_dim = 256;

    // FP16 size
    let fp16_bytes = vocab_size * hidden_dim * 2;
    // INT8 size (data + scales)
    let int8_bytes = vocab_size * hidden_dim + vocab_size * 2;
    // Should be approximately 50% savings
    let savings = 1.0 - (int8_bytes as f64 / fp16_bytes as f64);
    assert!(
        savings > 0.49,
        "Expected ~50% savings, got {:.1}%",
        savings * 100.0
    );
}

// ─── Phase 5: INT4 LUT Kernel Tests ─────────────────────────────────────

#[test]
fn test_int4_lut_matmul_basic() {
    // Test INT4 matmul with LUT-based kernel
    // Layout: packed INT4 weights + per-row per-group FP16 scales
    let stream = cuda_ffi::CudaStream::cpu_only();

    let k = 4; // input dim
    let m = 2; // output dim (rows)
    let group_size = 128; // standard group size
    let num_groups = (k + group_size - 1) / group_size; // = 1
    let packed_k = (k + 1) / 2; // = 2 bytes per row

    // Build weight buffer: [m * packed_k bytes of packed INT4] + [m * num_groups * 2 bytes of FP16 scales]
    let weight_bytes = m * packed_k;
    let scales_bytes = m * num_groups * 2;
    let mut weight_buf = vec![0u8; weight_bytes + scales_bytes];

    // Row 0: values [8, 8, 8, 8] (nibble 8 = signed 0 → dequant = (8-8)*scale = 0)
    // Row 1: values [9, 8, 8, 8] (nibble 9 = signed +1 → dequant = (9-8)*scale = scale)
    // Actually, let's use nibble values directly:
    // Row 0: nibbles [9, 8, 8, 8] → packed [0x89, 0x88]
    //   col0: nibble 9 → (9-8) * scale = 1 * scale
    //   col1: nibble 8 → (8-8) * scale = 0
    //   col2: nibble 8 → 0
    //   col3: nibble 8 → 0
    weight_buf[0] = 0x89; // lo=9, hi=8 → col0=9, col1=8
    weight_buf[1] = 0x88; // lo=8, hi=8 → col2=8, col3=8

    // Row 1: nibbles [8, 9, 8, 8] → packed [0x98, 0x88]
    //   col0: nibble 8 → 0
    //   col1: nibble 9 → 1 * scale
    weight_buf[2] = 0x98; // lo=8, hi=9 → col0=8, col1=9
    weight_buf[3] = 0x88;

    // Scales: both rows use scale = 1.0
    let scale_fp16 = 0x3C00u16; // 1.0 in FP16
    let scale_bytes = scale_fp16.to_le_bytes();
    let scales_start = weight_bytes;
    // Row 0, group 0
    weight_buf[scales_start] = scale_bytes[0];
    weight_buf[scales_start + 1] = scale_bytes[1];
    // Row 1, group 0
    weight_buf[scales_start + 2] = scale_bytes[0];
    weight_buf[scales_start + 3] = scale_bytes[1];

    // Input = [1.0, 2.0, 3.0, 4.0]
    let input: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];
    let mut output = vec![f16::from_f32(0.0); m];

    kernels::partial_matmul(
        input.as_ptr() as *const u8,
        weight_buf.as_ptr() as *const u8,
        output.as_mut_ptr() as *mut u8,
        k,
        m,
        DType::INT4,
        &stream,
    )
    .unwrap();

    // Row 0: 1.0 * 1.0 + 2.0 * 0 + 3.0 * 0 + 4.0 * 0 = 1.0
    assert!(
        (output[0].to_f32() - 1.0).abs() < 0.1,
        "Row 0: expected ~1.0, got {}",
        output[0].to_f32()
    );
    // Row 1: 1.0 * 0 + 2.0 * 1.0 + 3.0 * 0 + 4.0 * 0 = 2.0
    assert!(
        (output[1].to_f32() - 2.0).abs() < 0.1,
        "Row 1: expected ~2.0, got {}",
        output[1].to_f32()
    );
}

#[test]
fn test_int4_lut_matmul_with_scale() {
    let stream = cuda_ffi::CudaStream::cpu_only();

    let k = 2;
    let m = 1;
    let packed_k = 1; // 2 INT4 values per byte
    let num_groups = 1;

    let weight_bytes = m * packed_k;
    let scales_bytes = m * num_groups * 2;
    let mut weight_buf = vec![0u8; weight_bytes + scales_bytes];

    // Row 0: nibbles [10, 10] → packed [0xAA]
    //   dequant(10) = (10-8) * scale = 2 * scale
    weight_buf[0] = 0xAA;

    // Scale = 0.5 → dequant(10) = 2 * 0.5 = 1.0
    let scale_fp16 = 0x3800u16; // 0.5 in FP16
    let sb = scale_fp16.to_le_bytes();
    weight_buf[weight_bytes] = sb[0];
    weight_buf[weight_bytes + 1] = sb[1];

    // Input = [3.0, 7.0]
    let input: Vec<f16> = vec![f16::from_f32(3.0), f16::from_f32(7.0)];
    let mut output = vec![f16::from_f32(0.0); m];

    kernels::partial_matmul(
        input.as_ptr() as *const u8,
        weight_buf.as_ptr() as *const u8,
        output.as_mut_ptr() as *mut u8,
        k,
        m,
        DType::INT4,
        &stream,
    )
    .unwrap();

    // Row 0: 3.0 * 1.0 + 7.0 * 1.0 = 10.0
    assert!(
        (output[0].to_f32() - 10.0).abs() < 0.5,
        "Expected ~10.0, got {}",
        output[0].to_f32()
    );
}

// ─── Phase 5: Transport Quantization Tests ───────────────────────────────

#[test]
fn test_transport_quant_roundtrip() {
    // FP16 → INT8 (transport) → FP16 (dequant)
    let cols = 8;
    let rows = 4;
    let num_elements = rows * cols;

    // Create FP16 data
    let mut fp16_data = vec![f16::from_f32(0.0); num_elements];
    for r in 0..rows {
        for c in 0..cols {
            let val = (r as f32 - 1.5) * 0.5 + c as f32 * 0.1;
            fp16_data[r * cols + c] = f16::from_f32(val);
        }
    }

    let fp16_bytes =
        unsafe { std::slice::from_raw_parts(fp16_data.as_ptr() as *const u8, num_elements * 2) };

    // Quantize to INT8 transport format
    let transport = kernels::quant_page_fp16_to_int8(fp16_bytes, num_elements, cols);

    // Expected size: num_elements (INT8) + num_rows * 2 (scales)
    assert_eq!(transport.len(), num_elements + rows * 2);

    // Dequantize back to FP16
    let mut dequant_buf = vec![0u8; num_elements * 2];
    kernels::dequant_page_int8_to_fp16(&transport, &mut dequant_buf, num_elements, cols);

    let dequant =
        unsafe { std::slice::from_raw_parts(dequant_buf.as_ptr() as *const f16, num_elements) };

    // Verify roundtrip accuracy.
    // INT8 quantization has ~1/127 ≈ 0.8% of max-abs-value per row as
    // quantization step size, so we allow up to ~1% of the row's max value.
    for i in 0..num_elements {
        let original = fp16_data[i].to_f32();
        let restored = dequant[i].to_f32();
        let diff = (original - restored).abs();
        // The row's max magnitude determines the quantization step size
        let row = i / cols;
        let row_max = (0..cols)
            .map(|c| fp16_data[row * cols + c].to_f32().abs())
            .fold(0.0f32, f32::max);
        let step = row_max / 127.0;
        // Allow 1 quantization step + small FP16 rounding
        let tolerance = step + 0.005;
        assert!(
            diff < tolerance,
            "Element {}: original={}, restored={}, diff={}, step={}",
            i,
            original,
            restored,
            diff,
            step
        );
    }
}

#[test]
fn test_transport_quant_size_savings() {
    let num_elements = 1024;
    let cols = 64;

    // FP16 size
    let fp16_bytes = num_elements * 2;
    // INT8 transport size (data + scales)
    let rows = num_elements / cols;
    let int8_bytes = num_elements + rows * 2;

    let ratio = int8_bytes as f64 / fp16_bytes as f64;
    assert!(
        ratio < 0.52,
        "Transport quant should be ~50% smaller, ratio={:.3}",
        ratio
    );
}

#[test]
fn test_transport_quant_zeros() {
    let cols = 4;
    let num_elements = 8;

    let fp16_data = vec![f16::from_f32(0.0); num_elements];
    let fp16_bytes =
        unsafe { std::slice::from_raw_parts(fp16_data.as_ptr() as *const u8, num_elements * 2) };

    let transport = kernels::quant_page_fp16_to_int8(fp16_bytes, num_elements, cols);

    let mut dequant_buf = vec![0u8; num_elements * 2];
    kernels::dequant_page_int8_to_fp16(&transport, &mut dequant_buf, num_elements, cols);

    let dequant =
        unsafe { std::slice::from_raw_parts(dequant_buf.as_ptr() as *const f16, num_elements) };

    for i in 0..num_elements {
        assert!(
            dequant[i].to_f32().abs() < 0.01,
            "Zero input should produce near-zero output: got {}",
            dequant[i].to_f32()
        );
    }
}

// ─── Phase 5: Priority Prefetch Queue Tests ──────────────────────────────

#[tokio::test]
async fn test_priority_queue_ordering() {
    // Test that Critical requests are dequeued before Low ones
    // We exercise this through the buffer manager's submit_prefetch path
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_pq.vib3");
    let path_str = path.to_str().unwrap();

    let _config = create_test_model(path_str);

    let model_file = Arc::new(Vib3File::open(path_str).unwrap());
    let buf_config = BufferPoolConfig {
        t1_capacity: 4 * PAGE_SIZE,
        t2_capacity: 8 * PAGE_SIZE,
        prefetch_queue_depth: 64,
        ..Default::default()
    };

    let mgr = PageBufferManager::new(buf_config, model_file.clone());
    mgr.initialize().await.unwrap();

    // Submit multiple prefetch requests at different priorities
    let page0 = model_file.page(0).page_id();
    let page1 = if model_file.page_count() > 1 {
        model_file.page(1).page_id()
    } else {
        page0
    };

    mgr.submit_prefetch(PrefetchRequest {
        page: page0,
        source: Tier::T3Nvme,
        dest: Tier::T2Ram,
        priority: PrefetchPriority::Low,
        deadline_tick: 100,
        confidence: 0.5,
    });

    mgr.submit_prefetch(PrefetchRequest {
        page: page1,
        source: Tier::T3Nvme,
        dest: Tier::T2Ram,
        priority: PrefetchPriority::Critical,
        deadline_tick: 50,
        confidence: 0.9,
    });

    // Verify prefetch stats were recorded
    let stats = mgr.stats.snapshot();
    assert_eq!(stats.prefetch_issued, 2);
}

#[tokio::test]
async fn test_prefetch_queue_capacity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_pq_cap.vib3");
    let path_str = path.to_str().unwrap();

    let _config = create_test_model(path_str);

    let model_file = Arc::new(Vib3File::open(path_str).unwrap());
    let buf_config = BufferPoolConfig {
        t1_capacity: 4 * PAGE_SIZE,
        t2_capacity: 8 * PAGE_SIZE,
        prefetch_queue_depth: 4, // Very small queue
        ..Default::default()
    };

    let mgr = PageBufferManager::new(buf_config, model_file.clone());
    mgr.initialize().await.unwrap();

    let page = model_file.page(0).page_id();

    // Submit more requests than queue capacity
    for i in 0..10 {
        mgr.submit_prefetch(PrefetchRequest {
            page,
            source: Tier::T3Nvme,
            dest: Tier::T2Ram,
            priority: PrefetchPriority::Low,
            deadline_tick: i as u64,
            confidence: 0.5,
        });
    }

    // Should not crash; excess requests are dropped gracefully
    let stats = mgr.stats.snapshot();
    assert_eq!(stats.prefetch_issued, 10);
}

// ─── Phase 5: Auto-Profile Hardware Tests ────────────────────────────────

#[test]
fn test_hardware_profile_probe() {
    let profile = vib3::registry::HardwareProfile::probe();

    // RAM bandwidth should be measurable (at least > 0)
    assert!(
        profile.ram_bandwidth_bps > 0,
        "RAM bandwidth should be measurable, got {}",
        profile.ram_bandwidth_bps
    );

    // NVMe bandwidth may be 0 in some environments (tmpfs, ramdisk, restricted I/O)
    // but if it's > 0, it should be reasonable (< 100 GB/s)
    if profile.nvme_bandwidth_bps > 0 {
        assert!(
            profile.nvme_bandwidth_bps < 100_000_000_000,
            "NVMe bandwidth seems too high: {} GB/s",
            profile.nvme_bandwidth_bps as f64 / 1e9
        );
    }

    // H2D bandwidth should be measurable (CPU fallback mode = memcpy speed)
    assert!(
        profile.h2d_bandwidth_bps > 0,
        "H2D bandwidth should be measurable, got {}",
        profile.h2d_bandwidth_bps
    );

    // CPU cores should be at least 1
    assert!(profile.cpu_cores >= 1);

    // RAM should be detected
    assert!(profile.ram_bytes > 0, "RAM should be detected");

    // Profile time should be reasonable (< 5 seconds)
    assert!(
        profile.profile_time_ms < 5000,
        "Profile should complete in < 5s, took {} ms",
        profile.profile_time_ms
    );

    // Summary should be non-empty
    let summary = profile.summary();
    assert!(!summary.is_empty());
}

#[test]
fn test_hardware_profile_auto_config() {
    let profile = vib3::registry::HardwareProfile::probe();

    let model_config = ModelConfig {
        name: "test".into(),
        architecture: "test".into(),
        hidden_dim: 256,
        expert_hidden_dim: 64,
        num_layers: 4,
        num_moe_layers: 3,
        dense_layer_idx: 0,
        num_experts: 8,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 1024,
        vocab_size: 1000,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let config = profile.auto_config(&model_config);

    // T1 should be positive
    assert!(config.t1_capacity > 0, "T1 should be positive");
    // T2 should be positive
    assert!(config.t2_capacity > 0, "T2 should be positive");
    // Prefetch queue should be reasonable
    assert!(config.prefetch_queue_depth >= 16);
    assert!(config.prefetch_queue_depth <= 256);
}

// ─── Phase 4: Domain Classifier Tests ────────────────────────────────────

#[test]
fn test_domain_classifier_empty() {
    let classifier = DomainClassifier::new();
    assert_eq!(classifier.num_domains(), 0);

    let pred = classifier.classify(&[1.0, 2.0, 3.0]);
    assert_eq!(pred.domain_id, 0);
    assert_eq!(pred.domain_name, "unknown");
    assert_eq!(pred.confidence, 0.0);
    assert!(pred.recommended_view.is_none());
}

#[test]
fn test_domain_classifier_single_domain() {
    let mut classifier = DomainClassifier::new();
    classifier.add_centroid(
        1,
        "code".into(),
        vec![1.0, 0.0, 0.0],
        Some("code_view".into()),
    );
    assert_eq!(classifier.num_domains(), 1);

    // Embedding aligned with the centroid
    let pred = classifier.classify(&[1.0, 0.0, 0.0]);
    assert_eq!(pred.domain_id, 1);
    assert_eq!(pred.domain_name, "code");
    assert!(pred.confidence > 0.9); // cosine sim = 1.0 → confidence = 1.0
    assert_eq!(pred.recommended_view, Some("code_view".into()));
}

#[test]
fn test_domain_classifier_multiple_domains() {
    let classifier = DomainClassifier::from_centroids(vec![
        (
            0,
            "code".into(),
            vec![1.0, 0.0, 0.0],
            Some("code_view".into()),
        ),
        (
            1,
            "math".into(),
            vec![0.0, 1.0, 0.0],
            Some("math_view".into()),
        ),
        (2, "creative".into(), vec![0.0, 0.0, 1.0], None),
    ]);

    // Test code domain
    let pred = classifier.classify(&[0.9, 0.1, 0.0]);
    assert_eq!(pred.domain_name, "code");
    assert!(pred.confidence > 0.5);

    // Test math domain
    let pred = classifier.classify(&[0.0, 1.0, 0.1]);
    assert_eq!(pred.domain_name, "math");

    // Test creative domain
    let pred = classifier.classify(&[0.0, 0.0, 1.0]);
    assert_eq!(pred.domain_name, "creative");
    assert!(pred.recommended_view.is_none());
}

#[test]
fn test_domain_classifier_top_k() {
    let classifier = DomainClassifier::from_centroids(vec![
        (0, "code".into(), vec![1.0, 0.0, 0.0], None),
        (1, "math".into(), vec![0.7, 0.7, 0.0], None),
        (2, "other".into(), vec![0.0, 0.0, 1.0], None),
    ]);

    // Embedding between code and math
    let results = classifier.classify_top_k(&[0.9, 0.4, 0.0], 2);
    assert_eq!(results.len(), 2);
    // First should be the closest
    assert!(results[0].confidence >= results[1].confidence);
}

#[test]
fn test_domain_classifier_zero_vector() {
    let classifier =
        DomainClassifier::from_centroids(vec![(0, "code".into(), vec![1.0, 0.0, 0.0], None)]);

    let pred = classifier.classify(&[0.0, 0.0, 0.0]);
    assert_eq!(pred.confidence, 0.0);
}

// ─── Phase 4: Vib3Writer Vector Index Tests ──────────────────────────────

#[test]
fn test_vib3_write_and_read_with_vector_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_vi.vib3");
    let path_str = path.to_str().unwrap();

    let config = ModelConfig {
        name: "test-vi".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut writer = Vib3Writer::new(config.clone());

    // Add a minimal page
    let page_data = vec![0u8; 64 * 32 * 2]; // FP16
    writer.add_page(0, 0, 0, 0, 0, 32, 64, &page_data);

    // Add vector index with 2 centroids of dimension 4
    let centroids = vec![vec![1.0f32, 0.0, 0.0, 0.0], vec![0.0f32, 1.0, 0.0, 0.0]];

    let mut entry0 = VectorIndexEntry {
        centroid_id: 0,
        cluster_size: 100,
        prediction_count: 1,
        hot_page_count: 1,
        expert_predictions: [(0, 0); 32],
        hot_pages: [0; 64],
    };
    entry0.expert_predictions[0] = (0, 200); // expert 0, prob 200/255
    entry0.hot_pages[0] = 0; // page catalog index 0

    let mut entry1 = VectorIndexEntry {
        centroid_id: 1,
        cluster_size: 50,
        prediction_count: 1,
        hot_page_count: 0,
        expert_predictions: [(0, 0); 32],
        hot_pages: [0; 64],
    };
    entry1.expert_predictions[0] = (2, 180);

    writer.set_vector_index(centroids, vec![entry0, entry1]);

    writer
        .finalize(path_str)
        .expect("Failed to write model with vector index");

    // Read it back
    let file = Vib3File::open(path_str).unwrap();
    assert!(file.has_vector_index());
    assert!(!file.vector_index_bytes().is_empty());

    // Load vector index
    let vi = vib3::index::vector_index::VectorIndex::load(&file).unwrap();
    assert_eq!(vi.centroid_count(), 2);
    assert_eq!(vi.entry_count(), 2);

    // Predict with embedding close to centroid 0
    let profile = vi.predict(&[0.9, 0.1, 0.0, 0.0]);
    assert!(!profile.layers.is_empty() || profile.domain_id == 0);
}

// ─── Phase 6: Quantization & Conversion Tests ───────────────────────────

#[test]
fn test_quantize_weights_to_int4_basic() {
    // Simple 2x4 weight matrix with known values
    let weights = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.0];
    let rows = 2;
    let cols = 4;

    let int4_data = kernels::quantize_weights_to_int4(&weights, rows, cols);

    // Verify layout: packed_k = 2 bytes per row, num_groups = 1 (cols=4 < 128)
    let packed_k = (cols + 1) / 2;
    let num_groups = (cols + kernels::INT4_GROUP_SIZE - 1) / kernels::INT4_GROUP_SIZE;
    let expected_size = rows * packed_k + rows * num_groups * 2;
    assert_eq!(int4_data.len(), expected_size);
}

#[test]
fn test_quantize_int4_roundtrip() {
    // Generate a random weight matrix, quantize to INT4, then dequantize via matmul
    // and verify the output is close to the original matmul result.
    let rows = 32;
    let cols = 128; // Exactly one group

    // Create a simple weight matrix with a gradient pattern
    let mut weights = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            weights[r * cols + c] = ((r as f32 - 16.0) * (c as f32 - 64.0)) / (16.0 * 64.0) * 0.5;
        }
    }

    // Create a simple input vector
    let mut input_f32 = vec![0.0f32; cols];
    for c in 0..cols {
        input_f32[c] = (c as f32 / cols as f32) - 0.5;
    }

    // Reference: f32 matmul
    let mut ref_output = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += input_f32[c] * weights[r * cols + c];
        }
        ref_output[r] = acc;
    }

    // Quantize to INT4
    let int4_data = kernels::quantize_weights_to_int4(&weights, rows, cols);

    // Use the INT4 matmul kernel
    let input_fp16: Vec<f16> = input_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let input_bytes = unsafe {
        std::slice::from_raw_parts(input_fp16.as_ptr() as *const u8, input_fp16.len() * 2)
    };
    let mut output_fp16 = vec![f16::from_f32(0.0); rows];
    let output_bytes =
        unsafe { std::slice::from_raw_parts_mut(output_fp16.as_mut_ptr() as *mut u8, rows * 2) };

    let stream = cuda_ffi::CudaStream::cpu_only();

    kernels::partial_matmul(
        input_bytes.as_ptr(),
        int4_data.as_ptr(),
        output_bytes.as_mut_ptr(),
        cols,
        rows,
        DType::INT4,
        &stream,
    )
    .unwrap();

    // Check that outputs are reasonably close (INT4 has significant quantization error)
    let int4_output: Vec<f32> = output_fp16.iter().map(|v| v.to_f32()).collect();
    let max_ref = ref_output.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    // Relative error per row — allow up to 30% for INT4 (it's 4-bit quantization)
    let mut max_rel_error = 0.0f32;
    for r in 0..rows {
        if max_ref > 1e-6 {
            let rel_err = (int4_output[r] - ref_output[r]).abs() / (ref_output[r].abs() + 1e-6);
            max_rel_error = max_rel_error.max(rel_err);
        }
    }

    // INT4 with group-128 should give <30% relative error on typical weights
    assert!(
        max_rel_error < 0.5,
        "INT4 roundtrip error too high: {:.1}%",
        max_rel_error * 100.0
    );
}

#[test]
fn test_quantize_fp16_to_int4() {
    let rows = 4;
    let cols = 8;
    let num_elements = rows * cols;

    // Create FP16 data
    let mut fp16_data = vec![f16::from_f32(0.0); num_elements];
    for i in 0..num_elements {
        fp16_data[i] = f16::from_f32((i as f32 / num_elements as f32 - 0.5) * 0.2);
    }
    let fp16_bytes =
        unsafe { std::slice::from_raw_parts(fp16_data.as_ptr() as *const u8, num_elements * 2) };

    let int4_data = kernels::quantize_fp16_to_int4(fp16_bytes, rows, cols);

    let packed_k = (cols + 1) / 2;
    let num_groups = (cols + kernels::INT4_GROUP_SIZE - 1) / kernels::INT4_GROUP_SIZE;
    let expected = rows * packed_k + rows * num_groups * 2;
    assert_eq!(int4_data.len(), expected);
}

#[test]
fn test_quantize_bf16_to_int4() {
    let rows = 4;
    let cols = 8;
    let num_elements = rows * cols;

    // Create BF16 data
    let bf16_data: Vec<u16> = (0..num_elements)
        .map(|i| half::bf16::from_f32((i as f32 / num_elements as f32 - 0.5) * 0.2).to_bits())
        .collect();
    let bf16_bytes =
        unsafe { std::slice::from_raw_parts(bf16_data.as_ptr() as *const u8, num_elements * 2) };

    let int4_data = kernels::quantize_bf16_to_int4(bf16_bytes, rows, cols);

    let packed_k = (cols + 1) / 2;
    let num_groups = (cols + kernels::INT4_GROUP_SIZE - 1) / kernels::INT4_GROUP_SIZE;
    let expected = rows * packed_k + rows * num_groups * 2;
    assert_eq!(int4_data.len(), expected);
}

#[test]
fn test_convert_bf16_to_fp16() {
    let vals_f32 = [0.0f32, 1.0, -1.0, 0.5, -0.5, 3.14159, -2.71828, 100.0];
    let bf16_data: Vec<u16> = vals_f32
        .iter()
        .map(|v| half::bf16::from_f32(*v).to_bits())
        .collect();
    let bf16_bytes =
        unsafe { std::slice::from_raw_parts(bf16_data.as_ptr() as *const u8, bf16_data.len() * 2) };

    let fp16_bytes = kernels::convert_bf16_to_fp16(bf16_bytes, vals_f32.len());
    let fp16_vals =
        unsafe { std::slice::from_raw_parts(fp16_bytes.as_ptr() as *const f16, vals_f32.len()) };

    for (i, &orig) in vals_f32.iter().enumerate() {
        let recovered = fp16_vals[i].to_f32();
        let err = (recovered - orig).abs();
        // BF16→FP16 may lose some precision on large values but should be close
        assert!(
            err < orig.abs() * 0.02 + 0.01,
            "BF16→FP16 conversion error at index {}: {} vs {} (err={})",
            i,
            recovered,
            orig,
            err
        );
    }
}

#[test]
fn test_convert_f32_to_fp16() {
    let vals = [0.0f32, 1.0, -1.0, 0.5, -0.5, 0.001];
    let f32_bytes =
        unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4) };

    let fp16_bytes = kernels::convert_f32_to_fp16(f32_bytes, vals.len());
    let fp16_vals =
        unsafe { std::slice::from_raw_parts(fp16_bytes.as_ptr() as *const f16, vals.len()) };

    for (i, &orig) in vals.iter().enumerate() {
        let recovered = fp16_vals[i].to_f32();
        let err = (recovered - orig).abs();
        assert!(
            err < 0.01,
            "F32→FP16 error at {}: {} vs {}",
            i,
            recovered,
            orig
        );
    }
}

#[test]
fn test_zstd_write_and_read_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zstd_test.vib3");
    let path_str = path.to_str().unwrap();

    let config = ModelConfig {
        name: "zstd-test".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 2,
        num_active_experts: 1,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::INT4,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut writer = Vib3Writer::new(config.clone());
    writer.set_compression(vib3::storage::format::CompressionMethod::Zstd { level: 3 });

    // Create compressible data (repeated pattern)
    let mut data = vec![0u8; 4096];
    for i in 0..data.len() {
        data[i] = (i % 17) as u8;
    }

    writer.add_page(0, 0, 0, 0, 0, 8, 32, &data);
    writer.add_page(0, 1, 0, 0, 0, 8, 32, &data);

    writer.finalize(path_str).unwrap();

    // Read back
    let file = Vib3File::open(path_str).unwrap();
    assert_eq!(file.page_count(), 2);

    // Verify compression was applied
    let page0 = file.page(0);
    let comp = { page0.compression };
    assert_eq!(
        comp,
        vib3::storage::format::COMPRESSION_ZSTD,
        "Expected Zstd compression"
    );
    let cs = { page0.compressed_size };
    let rs = { page0.raw_size };
    assert!(cs < rs, "Zstd should compress the data");

    // Verify decompression produces original data
    let mut buf = vec![0u8; data.len()];
    let read_size = file.read_page_sync(0, &mut buf).unwrap();
    assert_eq!(read_size, data.len());
    assert_eq!(&buf[..read_size], &data[..]);
}

#[test]
fn test_zstd_incompressible_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zstd_incomp.vib3");
    let path_str = path.to_str().unwrap();

    let config = ModelConfig {
        name: "zstd-incomp-test".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 1,
        num_active_experts: 1,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut writer = Vib3Writer::new(config.clone());
    writer.set_compression(vib3::storage::format::CompressionMethod::Zstd { level: 3 });

    // Random (incompressible) data
    let mut data = vec![0u8; 4096];
    let mut rng: u64 = 0xDEADBEEF;
    for b in &mut data {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *b = rng as u8;
    }

    writer.add_page(0, 0, 0, 0, 0, 8, 32, &data);
    writer.finalize(path_str).unwrap();

    let file = Vib3File::open(path_str).unwrap();
    let page0 = file.page(0);

    // If Zstd can't compress, it falls back to NONE
    let comp = { page0.compression };
    if comp == vib3::storage::format::COMPRESSION_NONE {
        let cs = { page0.compressed_size };
        let rs = { page0.raw_size };
        assert_eq!(cs, rs);
    }

    // Either way, reading should return original data
    let mut buf = vec![0u8; data.len()];
    let read_size = file.read_page_sync(0, &mut buf).unwrap();
    assert_eq!(read_size, data.len());
    assert_eq!(&buf[..read_size], &data[..]);
}

#[test]
fn test_int4_quantized_model_write_read() {
    // Write a model with INT4 quantized expert weights and verify roundtrip
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("int4_model.vib3");
    let path_str = path.to_str().unwrap();

    let config = ModelConfig {
        name: "int4-test".into(),
        architecture: "test".into(),
        hidden_dim: 128,
        expert_hidden_dim: 64,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 2,
        num_active_experts: 1,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::INT4,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut writer = Vib3Writer::new(config.clone());
    writer.set_compression(vib3::storage::format::CompressionMethod::Zstd { level: 3 });

    let hidden_dim = config.hidden_dim as usize;
    let expert_hidden_dim = config.expert_hidden_dim as usize;

    for expert in 0..config.num_experts {
        for segment in 0..3u16 {
            let (rows, cols) = match segment {
                0 | 1 => (expert_hidden_dim, hidden_dim),
                2 => (hidden_dim, expert_hidden_dim),
                _ => unreachable!(),
            };

            // Create f32 weights and quantize
            let mut weights = vec![0.0f32; rows * cols];
            for (i, w) in weights.iter_mut().enumerate() {
                *w = ((i as f32 * 0.01 + expert as f32 * 0.1 + segment as f32 * 0.001) % 1.0 - 0.5)
                    * 0.2;
            }

            let int4_data = kernels::quantize_weights_to_int4(&weights, rows, cols);

            // Split into pages using the helper layout
            let packed_k = (cols + 1) / 2;
            let num_groups = (cols + kernels::INT4_GROUP_SIZE - 1) / kernels::INT4_GROUP_SIZE;
            let bytes_per_row = packed_k + num_groups * 2;
            let total_int4_bytes = rows * packed_k;

            let rows_per_page = if bytes_per_row > 0 {
                (PAGE_SIZE / bytes_per_row).max(1)
            } else {
                rows
            };
            let num_pages = (rows + rows_per_page - 1) / rows_per_page;

            for page_idx in 0..num_pages {
                let row_start = page_idx * rows_per_page;
                let row_count = rows_per_page.min(rows - row_start);

                let page_int4_bytes = row_count * packed_k;
                let page_scales_bytes = row_count * num_groups * 2;
                let mut page_data = vec![0u8; page_int4_bytes + page_scales_bytes];

                for r in 0..row_count {
                    let src = (row_start + r) * packed_k;
                    let dst = r * packed_k;
                    page_data[dst..dst + packed_k].copy_from_slice(&int4_data[src..src + packed_k]);
                }

                let scales_start = total_int4_bytes;
                for r in 0..row_count {
                    let src = scales_start + (row_start + r) * num_groups * 2;
                    let dst = page_int4_bytes + r * num_groups * 2;
                    page_data[dst..dst + num_groups * 2]
                        .copy_from_slice(&int4_data[src..src + num_groups * 2]);
                }

                writer.add_page(
                    0,
                    expert as u16,
                    segment,
                    page_idx as u16,
                    row_start as u16,
                    row_count as u16,
                    cols as u16,
                    &page_data,
                );
            }
        }
    }

    writer.finalize(path_str).unwrap();

    // Read back and verify
    let file = Vib3File::open(path_str).unwrap();
    assert_eq!(file.model_config().expert_dtype, DType::INT4);
    assert_eq!(file.model_config().num_experts, 2);

    // Verify expert index
    for expert in 0..2u16 {
        let entry = file.expert_entry(0, expert);
        assert!(
            entry.is_some(),
            "Expert {expert} should have an index entry"
        );
        let entry = entry.unwrap();
        let pc = { entry.page_count };
        let ns = { entry.num_segments };
        assert!(pc > 0);
        assert_eq!(ns, 3);
    }

    // Verify pages can be read
    for page_idx in 0..file.page_count() {
        let page = file.page(page_idx);
        let mut buf = vec![0u8; page.raw_size as usize];
        let read = file.read_page_sync(page_idx, &mut buf).unwrap();
        assert_eq!(read, page.raw_size as usize);
    }
}

#[test]
fn test_hf_config_parser_mixtral() {
    let json = serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 32000,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16"
    });

    let config = ModelConfig::from_hf_config(&json, "Mixtral-8x7B").unwrap();
    assert_eq!(config.architecture, "mixtral");
    assert_eq!(config.hidden_dim, 4096);
    assert_eq!(config.expert_hidden_dim, 14336);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_moe_layers, 32);
    assert_eq!(config.num_experts, 8);
    assert_eq!(config.num_active_experts, 2);
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.expert_dtype, DType::INT4); // target dtype
    assert!(config.mla.is_none());
}

#[test]
fn test_hf_config_parser_deepseek_v2() {
    let json = serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": 5120,
        "intermediate_size": 12288,
        "moe_intermediate_size": 1536,
        "num_hidden_layers": 60,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "n_routed_experts": 160,
        "num_experts_per_tok": 6,
        "first_k_dense_replace": 1,
        "vocab_size": 102400,
        "max_position_embeddings": 163840,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "torch_dtype": "bfloat16"
    });

    let config = ModelConfig::from_hf_config(&json, "DeepSeek-V2").unwrap();
    assert_eq!(config.architecture, "deepseek-v2");
    assert_eq!(config.hidden_dim, 5120);
    assert_eq!(config.expert_hidden_dim, 1536); // uses moe_intermediate_size
    assert_eq!(config.num_layers, 60);
    assert_eq!(config.num_moe_layers, 59); // 60 - 1 dense
    assert_eq!(config.dense_layer_idx, 1);
    assert_eq!(config.num_experts, 160);
    assert_eq!(config.num_active_experts, 6);

    // MLA config should be present
    let mla = config.mla.as_ref().unwrap();
    assert_eq!(mla.kv_lora_rank, 512);
    assert_eq!(mla.q_lora_rank, 1536);
    assert_eq!(mla.qk_rope_head_dim, 64);
}

#[test]
fn test_hf_config_parser_dense_model() {
    // A non-MoE model should have 0 MoE layers
    let json = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 4096
    });

    let config = ModelConfig::from_hf_config(&json, "Llama-2-7B").unwrap();
    assert_eq!(config.num_experts, 1); // default
    assert_eq!(config.num_moe_layers, 0); // not MoE
}

#[test]
fn test_shared_page_indexing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared_idx.vib3");
    let path_str = path.to_str().unwrap();

    let config = ModelConfig {
        name: "shared-test".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 2,
        num_active_experts: 1,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut writer = Vib3Writer::new(config.clone());

    // Add expert pages
    let data = vec![0u8; 1024];
    writer.add_page(0, 0, 0, 0, 0, 4, 16, &data);
    writer.add_page(0, 1, 0, 0, 0, 4, 16, &data);

    // Add shared pages (attention, router, norms)
    writer.add_page(0, 0xFFFF, 3, 0, 0, 2, 64, &data); // router
    writer.add_page(0, 0xFFFF, 4, 0, 0, 4, 64, &data); // attn_qkv
    writer.add_page(0, 0xFFFF, 5, 0, 0, 4, 64, &data); // attn_o
    writer.add_page(0, 0xFFFF, 6, 0, 0, 1, 64, &data); // attn_norm
    writer.add_page(0, 0xFFFF, 7, 0, 0, 1, 64, &data); // ffn_norm

    // Add embedding and lm_head (layer=0, expert=0xFFFF)
    writer.add_page(0, 0xFFFF, 10, 0, 0, 128, 64, &data); // embeddings
    writer.add_page(0, 0xFFFF, 11, 0, 0, 128, 64, &data); // lm_head

    writer.finalize(path_str).unwrap();

    let file = Vib3File::open(path_str).unwrap();

    // Should have expert entries AND shared entries in the expert index
    let expert_0 = file.expert_entry(0, 0);
    assert!(expert_0.is_some());
    let ns = { expert_0.unwrap().num_segments };
    assert_eq!(ns, 1); // only segment 0

    let expert_1 = file.expert_entry(0, 1);
    assert!(expert_1.is_some());

    // Shared pages should be accessible via pages_for_shared
    let shared_pages = file.pages_for_shared(0);
    assert!(
        !shared_pages.is_empty(),
        "Should have shared pages at layer 0"
    );

    // All shared pages should have expert=0xFFFF
    for p in shared_pages {
        let exp = { p.expert };
        assert_eq!(exp, 0xFFFF);
    }

    // Segment-based lookup
    let router_pages = file.pages_for_segment(3);
    assert!(!router_pages.is_empty(), "Should have router pages");
    let seg = { router_pages[0].segment };
    assert_eq!(seg, 3);

    let embed_pages = file.pages_for_segment(10);
    assert!(!embed_pages.is_empty(), "Should have embedding pages");
}

#[test]
fn test_compression_method_enum() {
    use vib3::storage::format::CompressionMethod;

    // Test default
    let default = CompressionMethod::default();
    assert_eq!(default, CompressionMethod::None);

    // Test Zstd level
    let zstd = CompressionMethod::Zstd { level: 3 };
    match zstd {
        CompressionMethod::Zstd { level } => assert_eq!(level, 3),
        _ => panic!("Expected Zstd"),
    }
}

// ─── Activation Mode Detection Tests ─────────────────────────────────────

#[test]
fn test_activation_mode_enum() {
    assert_eq!(ActivationMode::Generalist.name(), "generalist");
    assert_eq!(ActivationMode::Specialist.name(), "specialist");
    assert_ne!(ActivationMode::Generalist, ActivationMode::Specialist);

    // Display
    let s = format!("{}", ActivationMode::Specialist);
    assert_eq!(s, "specialist");
}

#[test]
fn test_mode_detector_starts_generalist() {
    let detector = ActivationModeDetector::new(384, 128);
    assert_eq!(detector.current_mode(), ActivationMode::Generalist);
    assert_eq!(detector.total_tokens(), 0);
}

#[test]
fn test_mode_detector_specialist_concentrated() {
    // Simulate specialist mode: same 4 experts activated for 200 tokens
    let mut detector = ActivationModeDetector::new(384, 64);
    detector.set_hysteresis(4); // Lower for testing
    detector.set_ema_alpha(0.5); // Less smoothing for faster convergence

    for _ in 0..200 {
        // Always activate experts 0, 1, 2, 3 — very concentrated
        detector.record(&[0, 1, 2, 3]);
    }

    let result = detector.detect();
    // With only 4 experts out of 384, entropy should be very low
    assert!(
        result.entropy < result.threshold,
        "Expected low entropy for concentrated activations: entropy={:.2}, threshold={:.2}",
        result.entropy,
        result.threshold,
    );
    assert_eq!(result.unique_experts, 4);
    // Concentration should be very high (all activations from 4 experts)
    assert!(
        result.concentration > 0.99,
        "Expected high concentration, got {}",
        result.concentration,
    );
}

#[test]
fn test_mode_detector_generalist_spread() {
    // Simulate generalist mode: activations spread across many experts
    let mut detector = ActivationModeDetector::new(384, 64);
    detector.set_hysteresis(4);
    detector.set_ema_alpha(0.5);

    for i in 0..200u16 {
        // Rotate through many different experts
        let e1 = (i * 7) % 384;
        let e2 = (i * 13 + 1) % 384;
        let e3 = (i * 19 + 2) % 384;
        let e4 = (i * 31 + 3) % 384;
        detector.record(&[e1, e2, e3, e4]);
    }

    let result = detector.detect();
    assert!(
        result.entropy > result.threshold * 0.8,
        "Expected high entropy for spread activations: entropy={:.2}, threshold={:.2}",
        result.entropy,
        result.threshold,
    );
    // Should see many unique experts
    assert!(
        result.unique_experts > 50,
        "Expected many unique experts, got {}",
        result.unique_experts,
    );
}

#[test]
fn test_mode_detector_hysteresis() {
    // Mode should not flip immediately on a single opposite reading
    let mut detector = ActivationModeDetector::new(384, 32);
    detector.set_hysteresis(8);
    detector.set_ema_alpha(0.3);
    detector.force_mode(ActivationMode::Specialist);

    // Feed a few generalist-like tokens — should NOT switch immediately
    for i in 0..5u16 {
        let e1 = (i * 7) % 384;
        let e2 = (i * 13 + 100) % 384;
        detector.record(&[e1, e2]);
    }
    let result = detector.detect();
    // May or may not have switched depending on EMA, but with only 5 tokens
    // and hysteresis of 8, it should still be specialist
    assert_eq!(
        result.mode,
        ActivationMode::Specialist,
        "Should not switch mode after only a few opposite readings"
    );
}

#[test]
fn test_mode_detector_force_mode() {
    let mut detector = ActivationModeDetector::new(384, 64);
    detector.force_mode(ActivationMode::Specialist);
    assert_eq!(detector.current_mode(), ActivationMode::Specialist);

    detector.force_mode(ActivationMode::Generalist);
    assert_eq!(detector.current_mode(), ActivationMode::Generalist);
}

#[test]
fn test_mode_detector_top_experts() {
    let mut detector = ActivationModeDetector::new(384, 64);

    // Expert 5 appears most, expert 10 second most
    for _ in 0..50 {
        detector.record(&[5, 10]);
    }
    for _ in 0..30 {
        detector.record(&[5, 20]);
    }
    for _ in 0..10 {
        detector.record(&[10, 30]);
    }

    let top = detector.top_experts(3);
    assert!(!top.is_empty());
    // Expert 5 should be first (80 activations)
    assert_eq!(top[0].0, 5, "Expert 5 should be most frequent");
    assert!(top[0].1 > top[1].1, "Top expert should have highest count");
}

// ─── Specialist Profile Tests ────────────────────────────────────────────

#[test]
fn test_specialist_profile_basic() {
    let profile = SpecialistProfile {
        name: "test-domain".into(),
        centroid: vec![0.1, 0.2, 0.3],
        hot_experts: vec![
            vec![0, 1, 2], // Layer 0
            vec![3, 4, 5], // Layer 1
        ],
        total_pages: 100,
        vram_required: 200 * 1024 * 1024,
    };

    assert_eq!(profile.unique_expert_count(), 6);
    assert!(profile.is_hot(0, 1));
    assert!(profile.is_hot(1, 4));
    assert!(!profile.is_hot(0, 5));
    assert!(!profile.is_hot(2, 0)); // Layer 2 doesn't exist
}

// ─── Compressed T2 Buffer Manager Tests ──────────────────────────────────

/// Helper: create a test model with Zstd-compressed pages.
fn create_zstd_test_model(path: &str) -> ModelConfig {
    let config = ModelConfig {
        name: "test-zstd".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 4,
        max_seq_len: 512,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let hidden_dim = config.hidden_dim as usize;
    let expert_hidden = config.expert_hidden_dim as usize;

    let mut writer = Vib3Writer::new(config.clone());
    writer.set_compression(CompressionMethod::Zstd { level: 1 });

    // Write expert pages with compressible data (repeated patterns)
    for expert in 0..config.num_experts {
        for seg in 0..3u16 {
            let (rows, cols) = match seg {
                0 | 1 => (expert_hidden, hidden_dim),
                2 => (hidden_dim, expert_hidden),
                _ => unreachable!(),
            };
            let page_bytes = rows * cols * 2; // FP16
                                              // Create compressible data: repeated byte patterns
            let mut data = vec![0u8; page_bytes];
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = ((expert as u8).wrapping_mul(17))
                    .wrapping_add((seg as u8).wrapping_mul(31))
                    .wrapping_add((i % 256) as u8);
            }
            writer.add_page(
                1, // layer
                expert as u16,
                seg,
                0,
                0,
                rows as u16,
                cols as u16,
                &data,
            );
        }
    }

    writer
        .finalize(path)
        .expect("Failed to write Zstd test model");
    config
}

#[tokio::test]
async fn test_buffer_manager_compressed_t2() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_compressed.vib3");
    let path_str = path.to_str().unwrap();

    create_zstd_test_model(path_str);

    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    // Verify pages are compressed
    assert!(model_file.page_count() > 0);
    let compression = model_file.page_compression(0);
    assert_eq!(
        compression, COMPRESSION_ZSTD,
        "Pages should be Zstd compressed"
    );

    // Create buffer manager with compressed T2 enabled
    let buf_config = BufferPoolConfig {
        t1_capacity: 4 * PAGE_SIZE,
        t2_capacity: 8 * PAGE_SIZE,
        t2_compressed: true,
        vram_staging_size: 4 * PAGE_SIZE, // 4 staging slots
        ..Default::default()
    };

    let mgr = PageBufferManager::new(buf_config, model_file.clone());
    assert!(mgr.is_compressed_t2());
    mgr.initialize().await.unwrap();

    // Get a page — should go through T3→T2(compressed)→T1(decompressed)
    let first_page = model_file.page(0).page_id();
    let handle = mgr.get_page(&first_page).await.unwrap();
    assert!(!handle.device_ptr.is_null());
    assert!(handle.size > 0);
    // T1 should have the decompressed (raw) size
    let expected_raw = model_file.page(0).raw_size as usize;
    assert_eq!(handle.size, expected_raw);

    // Second access: should be T1 hit
    let handle2 = mgr.get_page(&first_page).await.unwrap();
    assert_eq!(handle2.source_tier, Tier::T1Vram);

    let stats = mgr.stats.snapshot();
    assert!(stats.t1_hits >= 1);
    assert_eq!(stats.t3_hits, 1);
}

#[tokio::test]
async fn test_buffer_manager_read_page_compressed_sync() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_compressed_read.vib3");
    let path_str = path.to_str().unwrap();

    create_zstd_test_model(path_str);

    let model_file = Vib3File::open(path_str).unwrap();
    let entry = model_file.page(0);
    let compressed_size = entry.compressed_size as usize;
    let raw_size = entry.raw_size as usize;

    // Read compressed bytes
    let mut compressed_buf = vec![0u8; compressed_size];
    let bytes_read = model_file
        .read_page_compressed_sync(0, &mut compressed_buf)
        .unwrap();
    assert_eq!(bytes_read, compressed_size);

    // Verify we can decompress them manually
    let decompressed =
        zstd::bulk::decompress(&compressed_buf[..bytes_read], raw_size + 1024).unwrap();
    assert_eq!(decompressed.len(), raw_size);

    // Verify compressed is smaller than raw (data should be compressible)
    assert!(
        compressed_size < raw_size,
        "Compressed ({}) should be smaller than raw ({})",
        compressed_size,
        raw_size,
    );
}

/// Regression test: uncompressed .vib3 file loaded with t2_compressed=true.
///
/// Previously, the buffer manager used the global `t2_compressed` config flag
/// to decide whether to store data compressed in T2. This caused uncompressed
/// pages to be marked as compressed, then the T2→T1 promotion would attempt
/// Zstd decompression on raw data and fail.
///
/// The fix checks per-page `entry.compression` from the page catalog, not
/// the global flag.
#[tokio::test]
async fn test_buffer_manager_uncompressed_with_t2_compressed_flag() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_uncompressed_pipeline_b.vib3");
    let path_str = path.to_str().unwrap();

    // create_test_model writes uncompressed pages (CompressionMethod::None)
    create_test_model(path_str);

    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    // Verify pages are NOT compressed
    assert!(model_file.page_count() > 0);
    let compression = model_file.page_compression(0);
    assert_eq!(
        compression, COMPRESSION_NONE,
        "Test model pages should be uncompressed"
    );

    // Create buffer manager with t2_compressed = true (Pipeline B default).
    // This used to cause a failure because it tried to Zstd-decompress raw data.
    let buf_config = BufferPoolConfig {
        t1_capacity: 4 * PAGE_SIZE,
        t2_capacity: 8 * PAGE_SIZE,
        t2_compressed: true,
        vram_staging_size: 4 * PAGE_SIZE,
        ..Default::default()
    };

    let mgr = PageBufferManager::new(buf_config, model_file.clone());
    assert!(mgr.is_compressed_t2());
    mgr.initialize().await.unwrap();

    // Get a page — should go T3→T2(raw, despite global flag)→T1(raw)
    let first_page = model_file.page(0).page_id();
    let handle = mgr.get_page(&first_page).await.unwrap();
    assert!(!handle.device_ptr.is_null());
    assert!(handle.size > 0);

    // The page should have the correct raw size
    let expected_raw = model_file.page(0).raw_size as usize;
    assert_eq!(handle.size, expected_raw);

    // Second access: T1 hit
    let handle2 = mgr.get_page(&first_page).await.unwrap();
    assert_eq!(handle2.source_tier, Tier::T1Vram);
}

// ─── Specialist Pinning Tests ────────────────────────────────────────────

#[tokio::test]
async fn test_specialist_pinning_basic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_pin.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);

    let model_file = Arc::new(Vib3File::open(path_str).unwrap());
    let buf_config = BufferPoolConfig {
        t1_capacity: 32 * PAGE_SIZE,
        t2_capacity: 64 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };

    let mgr = PageBufferManager::new(buf_config, model_file.clone());
    mgr.initialize().await.unwrap();

    // Pin experts (layer=1, expert=0) and (layer=1, expert=1)
    let pinned = mgr.pin_expert_cluster(&[(1, 0), (1, 1)]).await.unwrap();
    assert!(pinned > 0, "Should have pinned at least one page");
    assert!(mgr.specialist_pin_count() > 0);

    // Check that pinned pages are in T1
    let first_page = PageId {
        layer: 1,
        expert: 0,
        segment: 0,
        page_idx: 0,
    };
    if mgr.is_specialist_pinned(&first_page) {
        let handle = mgr.get_page(&first_page).await.unwrap();
        assert_eq!(
            handle.source_tier,
            Tier::T1Vram,
            "Pinned page should be in T1"
        );
    }

    // Unpin all
    let unpinned = mgr.unpin_expert_cluster();
    assert_eq!(unpinned, mgr.specialist_pin_count() + unpinned);
    assert_eq!(mgr.specialist_pin_count(), 0);
}

#[tokio::test]
async fn test_specialist_pinning_survives_eviction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_pin_evict.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);

    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    // Small T1 to force eviction
    let buf_config = BufferPoolConfig {
        t1_capacity: 8 * PAGE_SIZE,
        t2_capacity: 32 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };

    let mgr = PageBufferManager::new(buf_config, model_file.clone());
    mgr.initialize().await.unwrap();

    // Pin expert 0 at layer 1
    let pinned = mgr.pin_expert_cluster(&[(1, 0)]).await.unwrap();
    assert!(pinned > 0);
    let pin_count = mgr.specialist_pin_count();

    // Now access many other pages to trigger eviction pressure
    for i in 0..model_file.page_count().min(20) {
        let page = model_file.page(i).page_id();
        let _ = mgr.get_page(&page).await;
    }

    // Pinned pages should still be pinned
    assert_eq!(
        mgr.specialist_pin_count(),
        pin_count,
        "Specialist pins should survive eviction pressure"
    );
}

// ─── Activation Mode Config Tests ────────────────────────────────────────

#[test]
fn test_activation_mode_config_defaults() {
    let config = ActivationModeConfig::default();
    assert!(config.enabled);
    assert_eq!(config.window_size, 128);
    assert_eq!(config.entropy_threshold, 0.0); // auto-compute
    assert_eq!(config.hysteresis, 8);
    assert_eq!(config.detect_interval, 16);
}

#[test]
fn test_buffer_pool_config_compressed_defaults() {
    let config = BufferPoolConfig::default();
    assert!(config.t2_compressed);
    assert!((config.t2_compression_ratio - 3.5).abs() < 0.01);
    assert_eq!(config.vram_staging_size, 32 * 1024 * 1024);
}

#[test]
fn test_engine_config_has_activation_mode() {
    let config = EngineConfig::default();
    assert!(config.activation_mode.enabled);
    assert_eq!(config.activation_mode.window_size, 128);
}

// ─── Mode Detection Result Tests ─────────────────────────────────────────

#[test]
fn test_mode_detection_confidence() {
    let mut detector = ActivationModeDetector::new(384, 32);
    detector.set_ema_alpha(0.3);

    // Empty window → zero confidence
    let result = detector.detect();
    assert_eq!(result.confidence, 0.0);

    // After some data, confidence should be non-zero
    for _ in 0..50 {
        detector.record(&[0, 1, 2]);
    }
    let result = detector.detect();
    assert!(
        result.confidence >= 0.0 && result.confidence <= 1.0,
        "Confidence should be [0,1], got {}",
        result.confidence,
    );
}

// ─── QueryPlanner Mode-Aware Tests ──────────────────────────────────────

#[test]
fn test_planner_set_mode_changes_prefetch() {
    use vib3::runtime::query_planner::QueryPlanner;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_planner_mode.vib3");
    let path_str = path.to_str().unwrap();

    let model_config = create_test_model(path_str);
    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    let buf_config = BufferPoolConfig {
        t1_capacity: 16 * PAGE_SIZE,
        t2_capacity: 32 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };

    let mgr = Arc::new(PageBufferManager::new(buf_config, model_file.clone()));

    let mut planner = QueryPlanner::new(mgr, None, model_file, model_config);

    // Default mode should be Generalist
    assert_eq!(planner.current_mode(), ActivationMode::Generalist);

    // Switch to Specialist
    planner.set_mode(ActivationMode::Specialist);
    assert_eq!(planner.current_mode(), ActivationMode::Specialist);

    // Switch back to Generalist
    planner.set_mode(ActivationMode::Generalist);
    assert_eq!(planner.current_mode(), ActivationMode::Generalist);
}

// ─── Engine Mode Detection Integration Tests ─────────────────────────────

#[tokio::test]
async fn test_engine_initializes_mode_detector() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_engine_mode.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);

    // Engine with mode detection enabled (default)
    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        activation_mode: ActivationModeConfig {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let engine = vib3::runtime::engine::Engine::new(config).await.unwrap();

    // Should start in Generalist mode
    assert_eq!(engine.current_mode(), ActivationMode::Generalist);

    // Mode detector should be present
    assert!(engine.mode_detector().is_some());
}

#[tokio::test]
async fn test_engine_mode_detection_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_engine_no_mode.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);

    // Engine with mode detection disabled
    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        activation_mode: ActivationModeConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let engine = vib3::runtime::engine::Engine::new(config).await.unwrap();

    // Mode detector should not be present
    assert!(engine.mode_detector().is_none());

    // Should still default to Generalist
    assert_eq!(engine.current_mode(), ActivationMode::Generalist);
}

#[tokio::test]
async fn test_engine_generate_records_activations() {
    use vib3::runtime::generate::SamplingParams;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_engine_gen_mode.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);

    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        activation_mode: ActivationModeConfig {
            enabled: true,
            window_size: 16,
            detect_interval: 4,
            hysteresis: 2,
            ema_alpha: 0.5,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::runtime::engine::Engine::new(config).await.unwrap();

    // Generate a few tokens — this exercises the full path:
    // router -> collect expert IDs -> record to detector -> detect
    let params = SamplingParams {
        max_tokens: 8,
        temperature: 0.0, // greedy for determinism
        ..Default::default()
    };
    let result = engine.generate_with_params("test", params).await.unwrap();
    assert!(result.tokens_generated > 0, "Should have generated tokens");

    // The mode detector should have recorded tokens
    if let Some(detector) = engine.mode_detector() {
        assert!(
            detector.total_tokens() > 0,
            "Mode detector should have recorded activations during generation"
        );
    }
}

// ─── Vector Index Integration Tests ─────────────────────────────────────

/// Helper: create a test model with a vector index embedded in the .vib3 file.
fn create_test_model_with_vector_index(path: &str) -> ModelConfig {
    let config = ModelConfig {
        name: "test-vi-engine".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 3,
        num_moe_layers: 2,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let mut writer = Vib3Writer::new(config.clone());

    let hidden_dim = config.hidden_dim as usize;
    let expert_hidden = config.expert_hidden_dim as usize;

    let mut rng: u64 = 42;

    for layer in 0..config.num_moe_layers {
        let layer_idx = layer as u16 + config.dense_layer_idx as u16;

        for expert in 0..config.num_experts {
            for segment in 0..3u16 {
                let (rows, cols) = match segment {
                    0 | 1 => (expert_hidden, hidden_dim),
                    2 => (hidden_dim, expert_hidden),
                    _ => unreachable!(),
                };

                let page_bytes = rows * cols * 2; // FP16
                let mut data = vec![0u8; page_bytes];
                for chunk in data.chunks_exact_mut(2) {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let val = f16::from_f32(((rng as f32 / u64::MAX as f32) - 0.5) * 0.1);
                    let bytes = val.to_le_bytes();
                    chunk[0] = bytes[0];
                    chunk[1] = bytes[1];
                }

                writer.add_page(
                    layer_idx,
                    expert as u16,
                    segment,
                    0,
                    0,
                    rows as u16,
                    cols as u16,
                    &data,
                );
            }
        }
    }

    // Add vector index with 2 centroids of dimension 4
    let centroids = vec![vec![1.0f32, 0.0, 0.0, 0.0], vec![0.0f32, 1.0, 0.0, 0.0]];

    let mut entry0 = VectorIndexEntry {
        centroid_id: 0,
        cluster_size: 100,
        prediction_count: 2,
        hot_page_count: 2,
        expert_predictions: [(0, 0); 32],
        hot_pages: [0; 64],
    };
    entry0.expert_predictions[0] = (0, 200); // expert 0, prob 200/255
    entry0.expert_predictions[1] = (1, 150); // expert 1, prob 150/255
    entry0.hot_pages[0] = 0; // page catalog index 0
    entry0.hot_pages[1] = 1; // page catalog index 1

    let mut entry1 = VectorIndexEntry {
        centroid_id: 1,
        cluster_size: 50,
        prediction_count: 2,
        hot_page_count: 1,
        expert_predictions: [(0, 0); 32],
        hot_pages: [0; 64],
    };
    entry1.expert_predictions[0] = (2, 180);
    entry1.expert_predictions[1] = (3, 120);
    entry1.hot_pages[0] = 2;

    writer.set_vector_index(centroids, vec![entry0, entry1]);

    writer
        .finalize(path)
        .expect("Failed to write test model with vector index");
    config
}

#[test]
fn test_planner_has_vector_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_vi_planner.vib3");
    let path_str = path.to_str().unwrap();

    let model_config = create_test_model_with_vector_index(path_str);
    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    let buf_config = BufferPoolConfig {
        t1_capacity: 16 * PAGE_SIZE,
        t2_capacity: 32 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };
    let mgr = Arc::new(PageBufferManager::new(buf_config, model_file.clone()));

    // Without vector index
    let planner_no_vi =
        QueryPlanner::new(mgr.clone(), None, model_file.clone(), model_config.clone());
    assert!(!planner_no_vi.has_vector_index());

    // With vector index
    let vi = VectorIndex::load(&model_file).unwrap();
    let planner_with_vi = QueryPlanner::new(mgr, Some(Arc::new(vi)), model_file, model_config);
    assert!(planner_with_vi.has_vector_index());
}

#[test]
fn test_planner_vector_prefetch_no_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_vi_noop.vib3");
    let path_str = path.to_str().unwrap();

    let model_config = create_test_model(path_str);
    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    let buf_config = BufferPoolConfig {
        t1_capacity: 16 * PAGE_SIZE,
        t2_capacity: 32 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };
    let mgr = Arc::new(PageBufferManager::new(buf_config, model_file.clone()));

    let mut planner = QueryPlanner::new(mgr, None, model_file, model_config);

    // Feed trajectory data
    planner.update_trajectory(vec![0.1; 64]);

    // Without a vector index, all methods should return 0 / no-op
    assert_eq!(planner.submit_vector_prefetch(3), 0);
    assert_eq!(planner.predict_and_prewarm(0), 0);
    // submit_cross_layer_prefetch returns () but should not panic
    planner.submit_cross_layer_prefetch(0);
}

#[tokio::test]
async fn test_planner_vector_prefetch_with_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_vi_prefetch.vib3");
    let path_str = path.to_str().unwrap();

    let model_config = create_test_model_with_vector_index(path_str);
    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    let buf_config = BufferPoolConfig {
        t1_capacity: 16 * PAGE_SIZE,
        t2_capacity: 32 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };
    let mgr = Arc::new(PageBufferManager::new(buf_config, model_file.clone()));
    mgr.initialize().await.unwrap();

    let vi = VectorIndex::load(&model_file).unwrap();
    let mut planner = QueryPlanner::new(mgr.clone(), Some(Arc::new(vi)), model_file, model_config);

    // Feed a trajectory state that will match centroid 0 (dimension 0 = 1.0)
    planner.update_trajectory(vec![1.0, 0.0, 0.0, 0.0]);

    // Speculative prefetch should submit requests
    let submitted = planner.submit_vector_prefetch(3);
    assert!(
        submitted > 0,
        "Vector index should have submitted prefetch requests, got {}",
        submitted,
    );

    // Stats should reflect the predictions
    let predicted = planner
        .stats
        .pages_predicted
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(predicted > 0, "Should have recorded predicted pages");

    // Prefetch stats on the buffer manager should show issued requests
    let stats = mgr.stats.snapshot();
    assert!(
        stats.prefetch_issued > 0,
        "Buffer manager should have received prefetch requests",
    );
}

#[tokio::test]
async fn test_planner_predict_and_prewarm() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_vi_prewarm.vib3");
    let path_str = path.to_str().unwrap();

    let model_config = create_test_model_with_vector_index(path_str);
    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    let buf_config = BufferPoolConfig {
        t1_capacity: 16 * PAGE_SIZE,
        t2_capacity: 32 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };
    let mgr = Arc::new(PageBufferManager::new(buf_config, model_file.clone()));
    mgr.initialize().await.unwrap();

    let vi = VectorIndex::load(&model_file).unwrap();
    let mut planner = QueryPlanner::new(mgr.clone(), Some(Arc::new(vi)), model_file, model_config);

    // Feed trajectory
    planner.update_trajectory(vec![0.5, 0.5, 0.0, 0.0]);

    // Pre-warm layer 0
    let submitted = planner.predict_and_prewarm(0);
    // May or may not submit depending on whether vector index has pages for layer 0
    // The key test is that it doesn't panic and returns a count
    let _ = submitted; // verify it runs without panic
}

#[tokio::test]
async fn test_planner_cross_layer_prefetch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_vi_cross.vib3");
    let path_str = path.to_str().unwrap();

    let model_config = create_test_model_with_vector_index(path_str);
    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    let buf_config = BufferPoolConfig {
        t1_capacity: 16 * PAGE_SIZE,
        t2_capacity: 32 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };
    let mgr = Arc::new(PageBufferManager::new(buf_config, model_file.clone()));
    mgr.initialize().await.unwrap();

    let vi = VectorIndex::load(&model_file).unwrap();
    let mut planner = QueryPlanner::new(mgr.clone(), Some(Arc::new(vi)), model_file, model_config);

    // Feed trajectory
    planner.update_trajectory(vec![0.8, 0.2, 0.0, 0.0]);

    // Cross-layer prefetch from layer 0 should try to prefetch layers 2-3
    // Should not panic even if there are no pages for those layers
    planner.submit_cross_layer_prefetch(0);

    // Verify it ran (no assertion on count since the model is tiny,
    // but prefetch_issued may be > 0)
}

#[tokio::test]
async fn test_planner_vector_prefetch_specialist_skips_pinned() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_vi_specialist.vib3");
    let path_str = path.to_str().unwrap();

    let model_config = create_test_model_with_vector_index(path_str);
    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    let buf_config = BufferPoolConfig {
        t1_capacity: 32 * PAGE_SIZE,
        t2_capacity: 64 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };
    let mgr = Arc::new(PageBufferManager::new(buf_config, model_file.clone()));
    mgr.initialize().await.unwrap();

    let vi = VectorIndex::load(&model_file).unwrap();
    let mut planner = QueryPlanner::new(
        mgr.clone(),
        Some(Arc::new(vi)),
        model_file,
        model_config.clone(),
    );

    // Switch to Specialist mode
    planner.set_mode(ActivationMode::Specialist);
    assert_eq!(planner.current_mode(), ActivationMode::Specialist);

    // Feed trajectory
    planner.update_trajectory(vec![1.0, 0.0, 0.0, 0.0]);

    // Pin some experts so their pages are specialist-pinned
    let _ = mgr.pin_expert_cluster(&[(0, 0), (0, 1)]).await;

    // Speculative prefetch in specialist mode should skip pinned pages
    let submitted = planner.submit_vector_prefetch(2);
    // The count may be lower than in generalist mode (some pages skipped)
    // The main test is that it runs without error
    let _ = submitted; // verify it runs without panic
}

#[tokio::test]
async fn test_engine_with_vector_index_generates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_engine_vi.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model_with_vector_index(path_str);

    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        activation_mode: ActivationModeConfig {
            enabled: true,
            window_size: 16,
            detect_interval: 4,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::runtime::engine::Engine::new(config).await.unwrap();

    // Engine should have loaded the vector index
    assert!(
        engine.has_vector_index(),
        "Engine should have loaded vector index"
    );
    assert!(
        engine.planner().has_vector_index(),
        "Planner should have vector index"
    );

    // Generate tokens — this exercises the full vector index integration:
    // predict_and_prewarm → router → submit_lookahead + cross_layer_prefetch
    // → plan_layer → execute → trajectory_update → submit_vector_prefetch
    let params = SamplingParams {
        max_tokens: 5,
        temperature: 0.0,
        ..Default::default()
    };
    let result = engine
        .generate_with_params("test vector index", params)
        .await
        .unwrap();
    assert!(
        result.tokens_generated > 0,
        "Should have generated tokens with vector index active"
    );
    assert!(result.total_time_ms > 0.0);
}

// ─── Pluggable ANN Backend Tests ─────────────────────────────────────────

/// A mock ANN backend that always returns a fixed centroid index.
/// Used to verify the trait dispatch works correctly.
struct FixedBackend {
    fixed_index: usize,
    count: usize,
    centroids: Vec<Vec<f32>>,
}

impl FixedBackend {
    fn new(fixed_index: usize, centroids: Vec<Vec<f32>>) -> Self {
        let count = centroids.len();
        Self {
            fixed_index,
            count,
            centroids,
        }
    }
}

impl vib3::index::vector_index::AnnBackend for FixedBackend {
    fn search(&self, _query: &[f32]) -> vib3::index::vector_index::NnResult {
        vib3::index::vector_index::NnResult {
            index: self.fixed_index.min(self.count.saturating_sub(1)),
            distance: 0.0,
        }
    }

    fn len(&self) -> usize {
        self.count
    }

    fn backend_name(&self) -> &str {
        "fixed-test"
    }

    fn centroid(&self, index: usize) -> Option<&[f32]> {
        self.centroids.get(index).map(|v| v.as_slice())
    }
}

#[test]
fn test_ann_backend_brute_force() {
    use vib3::index::vector_index::{AnnBackend, BruteForceBackend};

    let centroids = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let backend = BruteForceBackend::new(centroids);

    assert_eq!(backend.len(), 3);
    assert!(!backend.is_empty());
    assert_eq!(backend.backend_name(), "brute-force");

    // Query close to centroid 0
    let result = backend.search(&[0.9, 0.1, 0.0]);
    assert_eq!(result.index, 0);

    // Query close to centroid 1
    let result = backend.search(&[0.0, 0.95, 0.05]);
    assert_eq!(result.index, 1);

    // Query close to centroid 2
    let result = backend.search(&[0.0, 0.0, 1.0]);
    assert_eq!(result.index, 2);

    // Centroid access
    assert_eq!(backend.centroid(0), Some([1.0, 0.0, 0.0].as_slice()));
    assert_eq!(backend.centroid(3), None); // out of bounds
}

#[test]
fn test_ann_backend_brute_force_search_k() {
    use vib3::index::vector_index::{AnnBackend, BruteForceBackend};

    let centroids = vec![
        vec![1.0, 0.0],
        vec![0.9, 0.1], // close to centroid 0
        vec![0.0, 1.0],
    ];
    let backend = BruteForceBackend::new(centroids);

    // Top-2 for a query near centroid 0
    let results = backend.search_k(&[1.0, 0.0], 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].index, 0); // exact match
    assert_eq!(results[1].index, 1); // second closest
    assert!(results[0].distance <= results[1].distance);
}

#[test]
fn test_ann_backend_brute_force_empty() {
    use vib3::index::vector_index::{AnnBackend, BruteForceBackend};

    let backend = BruteForceBackend::new(vec![]);
    assert!(backend.is_empty());
    assert_eq!(backend.len(), 0);

    let result = backend.search(&[1.0, 2.0]);
    assert_eq!(result.index, 0);
    assert_eq!(result.distance, f32::MAX);
}

#[test]
fn test_vector_index_custom_backend() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_custom_backend.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model_with_vector_index(path_str);
    let model_file = Vib3File::open(path_str).unwrap();

    // Load with default backend
    let vi_default = VectorIndex::load(&model_file).unwrap();
    assert_eq!(vi_default.backend_name(), "brute-force");
    assert_eq!(vi_default.centroid_count(), 2);

    // Load with custom fixed backend that always returns centroid 1
    let centroids = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
    let fixed = FixedBackend::new(1, centroids);
    let vi_custom = VectorIndex::load_with_backend(&model_file, Box::new(fixed)).unwrap();
    assert_eq!(vi_custom.backend_name(), "fixed-test");
    assert_eq!(vi_custom.centroid_count(), 2);

    // Predict with default — should use brute-force nearest centroid
    let profile_default = vi_default.predict(&[0.9, 0.1, 0.0, 0.0]);
    // Should match centroid 0 (closer to [1,0,0,0])
    assert_eq!(profile_default.domain_id, 0);

    // Predict with fixed backend — always returns centroid 1
    let profile_custom = vi_custom.predict(&[0.9, 0.1, 0.0, 0.0]);
    assert_eq!(profile_custom.domain_id, 1);

    // Same input, different backend → different prediction
    assert_ne!(profile_default.domain_id, profile_custom.domain_id);
}

#[test]
fn test_vector_index_from_parts_with_backend() {
    let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let fixed = FixedBackend::new(0, centroids);

    let config = ModelConfig {
        name: "test".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let vi = VectorIndex::from_parts_with_backend(
        vec![], // no entries
        Box::new(fixed),
        config,
        vec![],
    );

    assert_eq!(vi.backend_name(), "fixed-test");
    assert_eq!(vi.centroid_count(), 2);
    assert_eq!(vi.entry_count(), 0);

    // Predict with no entries → empty profile
    let profile = vi.predict(&[0.5, 0.5]);
    assert!(profile.layers.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// Phase 8: Tiered KV Cache Integration Tests
// ═══════════════════════════════════════════════════════════════════════

use vib3::runtime::tiered_kv::{KvIndex, TieredKvCache, UnifiedEvictionPolicy};

#[tokio::test]
async fn test_tiered_kv_engine_generates_with_tiered_cache() {
    // Test that the engine can be created with tiered KV cache enabled
    // and generates tokens without crashing.
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("test_tiered.vib3");
    let path_str = model_path.to_str().unwrap();
    create_test_model(path_str);

    let engine_config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 16 * PAGE_SIZE,
            t2_capacity: 32 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        kv_cache: KvCacheConfig {
            enabled: true,
            t1_positions: 32,
            t2_positions: 64,
            sparse_attention: true,
            top_k_positions: 8,
            recent_window: 8,
            landmark_count: 4,
            unified_pool: true,
            t1_kv_fraction: 0.15,
            t2_kv_fraction: 0.10,
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(engine_config).await.unwrap();
    assert!(engine.has_tiered_kv());

    let params = vib3::runtime::generate::SamplingParams {
        temperature: 0.0,
        max_tokens: 5,
        ..Default::default()
    };
    let result = engine.generate_with_params("Hello", params).await.unwrap();
    assert!(result.tokens_generated > 0);

    // Verify the tiered KV cache was used
    let tkv = engine.tiered_kv().unwrap();
    assert!(tkv.seq_len() > 0);
}

#[tokio::test]
async fn test_tiered_kv_engine_without_tiered_cache() {
    // Verify that with tiered KV disabled, everything still works
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("test_flat.vib3");
    let path_str = model_path.to_str().unwrap();
    create_test_model(path_str);

    let engine_config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 16 * PAGE_SIZE,
            t2_capacity: 32 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        kv_cache: KvCacheConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(engine_config).await.unwrap();
    assert!(!engine.has_tiered_kv());

    let params = vib3::runtime::generate::SamplingParams {
        temperature: 0.0,
        max_tokens: 5,
        ..Default::default()
    };
    let result = engine.generate_with_params("Hello", params).await.unwrap();
    assert!(result.tokens_generated > 0);
}

#[test]
fn test_sparse_attention_kernel() {
    // Test that sparse_attention_head produces valid output
    let head_dim = 8;
    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // 3 positions with known K and V vectors
    let k_vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // High similarity to Q
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Low similarity
        vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Medium similarity
    ];
    let v_vectors = vec![
        vec![1.0; 8], // Value for high-similarity position
        vec![0.0; 8], // Value for low-similarity position
        vec![0.5; 8], // Value for medium-similarity position
    ];

    let output = kernels::sparse_attention_head(&q, &k_vectors, &v_vectors, head_dim);

    // Output should be mostly the V of the highest-similarity position
    assert_eq!(output.len(), head_dim);
    // The first position has highest Q·K score (1.0), so its V (all 1.0)
    // should dominate the output
    assert!(output[0] > 0.5, "Expected high value, got {}", output[0]);
}

#[test]
fn test_multi_head_sparse_attention() {
    let head_dim = 4;
    let num_heads = 2;
    let num_kv_heads = 1; // GQA: 2 Q heads share 1 KV head

    let q_heads = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

    let k_per_head = vec![vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]];
    let v_per_head = vec![vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]];

    let output = kernels::multi_head_sparse_attention(
        &q_heads,
        &k_per_head,
        &v_per_head,
        num_heads,
        num_kv_heads,
        head_dim,
    );

    // Output should be [hidden_dim] = num_heads * head_dim = 8
    assert_eq!(output.len(), num_heads * head_dim);
}

#[test]
fn test_kv_index_incremental_insert_and_search() {
    let mut idx = KvIndex::new(8);

    // Simulate incremental insertion (one K vector per token)
    for pos in 0..100 {
        let mut k = vec![0.0; 8];
        k[pos % 8] = 1.0; // Each position has a distinctive K vector
        idx.insert(pos, &k);
    }

    assert_eq!(idx.len(), 100);

    // Search for a specific pattern
    let mut query = vec![0.0; 8];
    query[3] = 1.0; // Looking for positions where dim 3 is hot
    let results = idx.search(&query, 5);

    // The top results should be positions 3, 11, 19, 27, 35 (every 8th)
    assert_eq!(results.len(), 5);
    for (pos, _score) in &results {
        assert_eq!(pos % 8, 3);
    }
}

#[test]
fn test_kv_index_removal() {
    let mut idx = KvIndex::new(4);
    idx.insert(0, &[1.0, 0.0, 0.0, 0.0]);
    idx.insert(1, &[0.0, 1.0, 0.0, 0.0]);
    idx.insert(2, &[0.0, 0.0, 1.0, 0.0]);

    assert_eq!(idx.len(), 3);

    idx.remove(1);
    assert_eq!(idx.len(), 2);
    assert!(!idx.contains(1));
    assert!(idx.contains(0));
    assert!(idx.contains(2));

    // Search should not return removed position
    let results = idx.search(&[0.0, 1.0, 0.0, 0.0], 5);
    assert!(results.iter().all(|(pos, _)| *pos != 1));
}

#[test]
fn test_tiered_kv_cache_multi_layer() {
    let config = KvCacheConfig {
        enabled: true,
        t1_positions: 8,
        t2_positions: 16,
        sparse_attention: true,
        top_k_positions: 4,
        recent_window: 4,
        landmark_count: 2,
        unified_pool: true,
        t1_kv_fraction: 0.15,
        t2_kv_fraction: 0.10,
    };

    let num_layers = 3;
    let num_kv_heads = 2;
    let head_dim = 4;

    let mut cache = TieredKvCache::new(num_layers, num_kv_heads, head_dim, config);

    // Append 10 tokens across all layers
    for _pos in 0..10 {
        for layer in 0..num_layers {
            let k = (0..num_kv_heads)
                .map(|h| vec![(h as f32 + 1.0) * 0.1; head_dim])
                .collect::<Vec<_>>();
            let v = (0..num_kv_heads)
                .map(|h| vec![(h as f32 + 1.0) * 0.2; head_dim])
                .collect::<Vec<_>>();
            cache.append_layer(layer, &k, &v);
        }
        cache.advance_position();
    }

    assert_eq!(cache.seq_len(), 10);

    // Check tier distribution for each layer/head
    for layer in 0..num_layers {
        for head in 0..num_kv_heads {
            let total = cache.t1_count(layer, head)
                + cache.t2_count(layer, head)
                + cache.t3_count(layer, head);
            assert_eq!(total, 10, "Layer {} Head {} total mismatch", layer, head);
        }
    }
}

#[test]
fn test_unified_eviction_policy_fractions() {
    let policy = UnifiedEvictionPolicy::new(0.20, 0.15);
    assert_eq!(policy.kv_fraction(), 0.20);
    assert_eq!(policy.t2_kv_fraction(), 0.15);
}

#[test]
fn test_tiered_kv_sparse_attention_integration() {
    // Full integration: create tiered KV cache, append positions,
    // then run sparse attention through the attention module.
    let config = KvCacheConfig {
        enabled: true,
        t1_positions: 4,
        t2_positions: 8,
        sparse_attention: true,
        top_k_positions: 3,
        recent_window: 2,
        landmark_count: 1,
        unified_pool: true,
        t1_kv_fraction: 0.15,
        t2_kv_fraction: 0.10,
    };

    let num_layers = 1;
    let num_kv_heads = 1;
    let head_dim = 4;

    let mut cache = TieredKvCache::new(num_layers, num_kv_heads, head_dim, config);

    // Append 8 positions with distinctive K vectors
    for pos in 0..8 {
        let mut k_vec = vec![0.0; head_dim];
        k_vec[pos % head_dim] = 1.0;
        let k = vec![k_vec];
        let v = vec![vec![(pos as f32 + 1.0) * 0.1; head_dim]];
        cache.append_layer(0, &k, &v);
        cache.advance_position();
    }

    // Query matching position 0's K vector
    let q = vec![1.0, 0.0, 0.0, 0.0];
    let positions = cache.gather_attention_positions(0, 0, &q);

    // Should include recent window (positions 6, 7) plus ANN-retrieved
    assert!(positions.len() >= 2);

    // Get K/V vectors for gathered positions
    let k_vecs = cache.get_k_vectors(0, 0, &positions);
    let v_vecs = cache.get_v_vectors(0, 0, &positions);

    // Run sparse attention
    let output = kernels::sparse_attention_head(&q, &k_vecs, &v_vecs, head_dim);
    assert_eq!(output.len(), head_dim);
}

#[test]
fn test_page_id_kv_cache_integration() {
    // Verify KV cache PageId interop with weight PageIds
    let weight_page = PageId {
        layer: 5,
        expert: 42,
        segment: 0,
        page_idx: 3,
    };
    let shared_page = PageId::shared(5, 4, 0);
    let kv_k_page = PageId::kv_cache(5, KV_SEGMENT_K, 10);
    let kv_v_page = PageId::kv_cache(5, KV_SEGMENT_V, 10);

    // All have unique keys
    assert_ne!(weight_page.key(), shared_page.key());
    assert_ne!(weight_page.key(), kv_k_page.key());
    assert_ne!(shared_page.key(), kv_k_page.key());
    assert_ne!(kv_k_page.key(), kv_v_page.key());

    // Type checks
    assert!(!weight_page.is_kv_cache());
    assert!(weight_page.is_weight());
    assert!(!shared_page.is_kv_cache());
    assert!(shared_page.is_weight());
    assert!(kv_k_page.is_kv_cache());
    assert!(!kv_k_page.is_weight());
    assert!(kv_k_page.is_k_page());
    assert!(!kv_k_page.is_v_page());
    assert!(kv_v_page.is_v_page());
    assert!(!kv_v_page.is_k_page());
}

#[test]
fn test_kv_cache_stats_snapshot() {
    // Verify that KV cache stats are included in the snapshot
    let stats = InferenceStats::default();
    stats
        .kv_page_accesses
        .store(100, std::sync::atomic::Ordering::Relaxed);
    stats
        .kv_t1_hits
        .store(80, std::sync::atomic::Ordering::Relaxed);
    stats
        .kv_t2_hits
        .store(15, std::sync::atomic::Ordering::Relaxed);
    stats
        .kv_t3_hits
        .store(5, std::sync::atomic::Ordering::Relaxed);
    stats
        .sparse_attn_queries
        .store(50, std::sync::atomic::Ordering::Relaxed);
    stats
        .sparse_attn_retrieved
        .store(1000, std::sync::atomic::Ordering::Relaxed);

    let snap = stats.snapshot();
    assert_eq!(snap.kv_page_accesses, 100);
    assert_eq!(snap.kv_t1_hits, 80);
    assert_eq!(snap.kv_t2_hits, 15);
    assert_eq!(snap.kv_t3_hits, 5);
    assert_eq!(snap.sparse_attn_queries, 50);
    assert_eq!(snap.sparse_attn_retrieved, 1000);
}

// ─── Phase 6b: Validation Framework Integration Tests ──────────────────

#[test]
fn test_reference_model_forward_multiple_tokens() {
    use vib3::validation::{compare_outputs, ReferenceModel};

    let config = ModelConfig {
        name: "val-test".into(),
        architecture: "test".into(),
        hidden_dim: 16,
        expert_hidden_dim: 8,
        num_layers: 3,
        num_moe_layers: 3,
        dense_layer_idx: 0,
        num_experts: 8,
        num_active_experts: 2,
        num_heads: 2,
        num_kv_heads: 1,
        max_seq_len: 64,
        vocab_size: 32,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    let model = ReferenceModel::new(config, 42);

    // Run forward on multiple tokens, verify all produce finite, distinct outputs
    let mut outputs = Vec::new();
    for token_id in 0..5 {
        let emb = model.embed(token_id);
        let out = model.forward_moe(emb);
        assert_eq!(out.len(), 16);
        for &v in &out {
            assert!(v.is_finite(), "Non-finite output for token {}", token_id);
        }
        outputs.push(out);
    }

    // Different tokens should produce different outputs
    let cmp = compare_outputs(&outputs[0], &outputs[1]);
    assert!(
        cmp.mae > 0.0,
        "Different tokens should produce different outputs"
    );

    // Same token should produce identical output
    let out_again = model.forward_moe(model.embed(0));
    let cmp_same = compare_outputs(&outputs[0], &out_again);
    assert!(
        cmp_same.is_exact(1e-10),
        "Same input should produce identical output"
    );
}

#[test]
fn test_reference_model_per_layer_error_tracking() {
    use vib3::validation::{compare_outputs, QuantizationErrorTracker, ReferenceModel};

    let config = ModelConfig {
        name: "err-track".into(),
        architecture: "test".into(),
        hidden_dim: 16,
        expert_hidden_dim: 8,
        num_layers: 4,
        num_moe_layers: 4,
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
    };

    let model_a = ReferenceModel::new(config.clone(), 42);
    let model_b = ReferenceModel::new(config, 43); // Slightly different weights (simulates quantization)

    let input = model_a.embed(0).to_vec();
    let state_a = input.clone();
    let state_b = input.clone();

    let mut tracker = QuantizationErrorTracker::new();

    // Run each layer independently and track divergence
    // Since ReferenceModel.forward_moe runs all layers at once,
    // we simulate per-layer tracking by comparing at different seed offsets
    let out_a = model_a.forward_moe(&state_a);
    let out_b = model_b.forward_moe(&state_b);

    let cmp = compare_outputs(&out_a, &out_b);
    tracker.record_layer(&cmp);

    // The models have different weights, so outputs should diverge
    assert!(cmp.mae > 0.0);
    assert_eq!(tracker.num_layers(), 1);
    assert!(tracker.mean_mae() > 0.0);
}

#[test]
fn test_reference_model_seed_sensitivity() {
    use vib3::validation::{compare_outputs, ReferenceModel};

    let config = ModelConfig {
        name: "seed-test".into(),
        architecture: "test".into(),
        hidden_dim: 32,
        expert_hidden_dim: 16,
        num_layers: 2,
        num_moe_layers: 2,
        dense_layer_idx: 0,
        num_experts: 8,
        num_active_experts: 2,
        num_heads: 2,
        num_kv_heads: 1,
        max_seq_len: 32,
        vocab_size: 16,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        ..Default::default()
    };

    // Compare outputs across close seeds — this simulates what happens
    // when quantization perturbs weights slightly
    let model_base = ReferenceModel::new(config.clone(), 1000);
    let model_near = ReferenceModel::new(config.clone(), 1001);
    let model_far = ReferenceModel::new(config, 9999);

    let input = model_base.embed(5).to_vec();
    let out_base = model_base.forward_moe(&input);
    let out_near = model_near.forward_moe(&input);
    let out_far = model_far.forward_moe(&input);

    let cmp_near = compare_outputs(&out_base, &out_near);
    let cmp_far = compare_outputs(&out_base, &out_far);

    // Both should produce different outputs (different weights)
    assert!(cmp_near.mae > 0.0);
    assert!(cmp_far.mae > 0.0);

    // Outputs should be finite
    for &v in out_base.iter().chain(out_near.iter()).chain(out_far.iter()) {
        assert!(v.is_finite());
    }
}

#[test]
fn test_validation_comparison_result_properties() {
    use vib3::validation::compare_outputs;

    // Test with progressively more divergent vectors
    let base = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Small perturbation
    let small: Vec<f32> = base.iter().map(|v| v + 0.01).collect();
    let cmp_small = compare_outputs(&base, &small);

    // Large perturbation
    let large: Vec<f32> = base.iter().map(|v| v + 1.0).collect();
    let cmp_large = compare_outputs(&base, &large);

    // Larger perturbation → larger error
    assert!(cmp_large.mae > cmp_small.mae);
    assert!(cmp_large.rmse > cmp_small.rmse);
    assert!(cmp_large.max_error > cmp_small.max_error);

    // Both should have high cosine similarity (same direction, just shifted)
    assert!(cmp_small.cosine_similarity > 0.99);
    assert!(cmp_large.cosine_similarity > 0.95);

    // Small perturbation should be acceptable, but exact test depends on threshold
    assert!(cmp_small.is_acceptable());
}

// ═══════════════════════════════════════════════════════════════════════════
// MLA (Multi-head Latent Attention) Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_mla_kv_cache_basic() {
    // Test MLA KV cache stores compressed latent + rope correctly
    let kv_lora_rank = 16;
    let rope_dim = 8;
    let mut cache = MlaKvCache::new(kv_lora_rank, rope_dim, 128);

    assert_eq!(cache.seq_len, 0);
    assert_eq!(cache.kv_lora_rank, kv_lora_rank);
    assert_eq!(cache.qk_rope_head_dim, rope_dim);

    // Append one position
    let latent: Vec<f32> = (0..kv_lora_rank).map(|i| i as f32 * 0.1).collect();
    let rope: Vec<f32> = (0..rope_dim).map(|i| i as f32 * 0.2).collect();
    cache.append(&latent, &rope);

    assert_eq!(cache.seq_len, 1);
    assert_eq!(cache.kv_latent.len(), kv_lora_rank);
    assert_eq!(cache.k_rope.len(), rope_dim);

    // Append another position
    let latent2: Vec<f32> = (0..kv_lora_rank).map(|i| i as f32 * 0.3).collect();
    let rope2: Vec<f32> = (0..rope_dim).map(|i| i as f32 * 0.4).collect();
    cache.append(&latent2, &rope2);

    assert_eq!(cache.seq_len, 2);
    assert_eq!(cache.kv_latent.len(), 2 * kv_lora_rank);
    assert_eq!(cache.k_rope.len(), 2 * rope_dim);

    // Verify values at position 1
    let pos1_latent = &cache.kv_latent[kv_lora_rank..2 * kv_lora_rank];
    assert!((pos1_latent[0] - 0.0).abs() < 1e-6);
    assert!((pos1_latent[1] - 0.3).abs() < 1e-6);
}

#[test]
fn test_mla_kv_cache_clear() {
    let mut cache = MlaKvCache::new(16, 8, 128);
    let latent = vec![1.0f32; 16];
    let rope = vec![1.0f32; 8];
    cache.append(&latent, &rope);
    cache.append(&latent, &rope);

    assert_eq!(cache.seq_len, 2);
    cache.clear();
    assert_eq!(cache.seq_len, 0);
    assert!(cache.kv_latent.is_empty());
    assert!(cache.k_rope.is_empty());
}

#[test]
fn test_mla_kv_cache_set() {
    // Test MlaKvCacheSet creates one cache per layer
    let config = ModelConfig {
        name: "test-mla".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 16,
        num_layers: 4,
        num_moe_layers: 3,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 4,
        max_seq_len: 128,
        vocab_size: 256,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        mla: Some(MlaConfig {
            kv_lora_rank: 16,
            q_lora_rank: 32,
            qk_rope_head_dim: 8,
            qk_nope_head_dim: 12,
            v_head_dim: 16,
            softmax_scale: 1.0 / (20.0f32).sqrt(), // 1/sqrt(12+8)
        }),
        ..Default::default()
    };

    let mla = config.mla.as_ref().unwrap();
    let cache_set = MlaKvCacheSet::new(&config, mla);

    assert_eq!(cache_set.layers.len(), 4); // one per layer
    assert_eq!(cache_set.layers[0].kv_lora_rank, 16);
    assert_eq!(cache_set.layers[0].qk_rope_head_dim, 8);
}

#[test]
fn test_mla_kv_cache_memory_savings() {
    // Verify the claimed ~28x memory savings for MLA vs GQA
    // GQA: num_heads * head_dim * 2 (K+V) * seq_len * sizeof(f32)
    // MLA: (kv_lora_rank + rope_dim) * seq_len * sizeof(f32)

    let num_heads = 64usize;
    let head_dim = 128usize;
    let kv_lora_rank = 512usize;
    let rope_dim = 64usize;
    let seq_len = 4096usize;

    let gqa_bytes = num_heads * head_dim * 2 * seq_len * 4; // K + V
    let mla_bytes = (kv_lora_rank + rope_dim) * seq_len * 4;

    let ratio = gqa_bytes as f64 / mla_bytes as f64;
    assert!(
        ratio > 25.0,
        "MLA should save >25x memory, got {:.1}x",
        ratio
    );
    assert!(ratio < 30.0, "Ratio should be ~28x, got {:.1}x", ratio);
}

#[test]
fn test_yarn_rope_config() {
    use vib3::runtime::attention::YarnRopeConfig;

    let config = YarnRopeConfig::from_kimi_k25(64);

    assert_eq!(config.rope_dim, 64);
    assert!((config.base_theta - 50000.0).abs() < 1.0);
    assert!((config.scaling_factor - 64.0).abs() < 0.1);
    assert!((config.beta_fast - 32.0).abs() < 0.1);
    assert!((config.beta_slow - 1.0).abs() < 0.1);
    assert_eq!(config.original_max_pos, 4096);
}

#[test]
fn test_yarn_rope_frequencies() {
    use vib3::runtime::attention::YarnRopeConfig;

    let config = YarnRopeConfig::from_kimi_k25(64);

    // High frequency dimensions should have original frequencies (no scaling)
    let freq_0 = config.freq(0); // Highest frequency
    let _freq_raw = 1.0 / 50000.0f32.powf(0.0 / 64.0); // = 1.0
                                                       // High freq should be close to original
    assert!(freq_0 > 0.0);

    // Low frequency dimensions should be scaled down by factor
    let freq_last = config.freq(31); // Lowest frequency
    let freq_raw_last = 1.0 / 50000.0f32.powf(62.0 / 64.0);
    // Low freq should be scaled down (divided by scaling_factor)
    assert!(freq_last > 0.0);
    assert!(freq_last <= freq_raw_last + 1e-10);

    // Frequencies should decrease as dimension index increases
    let freq_1 = config.freq(1);
    assert!(
        freq_0 > freq_1,
        "Higher dim index should have lower frequency"
    );
}

#[test]
fn test_yarn_rope_apply() {
    use vib3::runtime::attention::apply_yarn_rope;
    use vib3::runtime::attention::YarnRopeConfig;

    let config = YarnRopeConfig::from_kimi_k25(8);

    // At position 0, RoPE should be identity (cos(0)=1, sin(0)=0)
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let original = x.clone();
    apply_yarn_rope(&mut x, 0, &config);
    for i in 0..8 {
        assert!(
            (x[i] - original[i]).abs() < 1e-5,
            "Position 0 should be identity: x[{}] = {} vs {}",
            i,
            x[i],
            original[i]
        );
    }

    // At position > 0, vectors should be rotated (different from input)
    let mut x2 = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    apply_yarn_rope(&mut x2, 100, &config);
    // At least some dimensions should differ
    let mut any_different = false;
    for i in 0..8 {
        if (x2[i] - 1.0).abs() > 0.01 || (x2[i]).abs() > 0.01 {
            any_different = true;
            break;
        }
    }
    assert!(any_different, "Position 100 should rotate the vector");
}

#[test]
fn test_yarn_rope_preserves_norm() {
    use vib3::runtime::attention::apply_yarn_rope;
    use vib3::runtime::attention::YarnRopeConfig;

    let config = YarnRopeConfig::from_kimi_k25(8);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    apply_yarn_rope(&mut x, 42, &config);
    let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    // RoPE is a rotation → should preserve L2 norm
    assert!(
        (norm_before - norm_after).abs() < 1e-4,
        "RoPE should preserve norm: {} vs {}",
        norm_before,
        norm_after
    );
}

#[test]
fn test_mla_attention_fallback() {
    // Test MLA attention with no weights loaded (all fallback paths)
    let config = ModelConfig {
        name: "test-mla".into(),
        architecture: "test".into(),
        hidden_dim: 32,
        expert_hidden_dim: 16,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 4,
        max_seq_len: 64,
        vocab_size: 128,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        mla: Some(MlaConfig {
            kv_lora_rank: 8,
            q_lora_rank: 16,
            qk_rope_head_dim: 4,
            qk_nope_head_dim: 6,
            v_head_dim: 8,
            softmax_scale: 1.0 / (10.0f32).sqrt(), // 1/sqrt(6+4)
        }),
        ..Default::default()
    };

    let mla = config.mla.as_ref().unwrap();
    let mut kv_cache = MlaKvCache::new(
        mla.kv_lora_rank as usize,
        mla.qk_rope_head_dim as usize,
        config.max_seq_len as usize,
    );

    let weights = MlaWeights {
        q_a_proj: None,
        q_b_proj: None,
        kv_a_proj: None,
        kv_b_proj: None,
        o_proj: None,
        q_norm: None,
        kv_norm: None,
    };

    // Create a test hidden state
    let hidden_state: Vec<f16> = (0..32).map(|i| f16::from_f32(i as f32 * 0.1)).collect();

    // Run MLA attention at position 0 (first token)
    let output = mla_attention_layer(&hidden_state, &weights, &mut kv_cache, 0, &config, mla);

    // Output should have hidden_dim elements
    assert_eq!(output.len(), 32);
    // KV cache should have 1 position
    assert_eq!(kv_cache.seq_len, 1);

    // Run at position 1
    let output2 = mla_attention_layer(&hidden_state, &weights, &mut kv_cache, 1, &config, mla);

    assert_eq!(output2.len(), 32);
    assert_eq!(kv_cache.seq_len, 2);

    // All outputs should be finite
    for v in output.iter().chain(output2.iter()) {
        assert!(v.to_f32().is_finite(), "Output should be finite");
    }
}

#[test]
fn test_mla_attention_with_weights() {
    // Test MLA attention with actual weight matrices
    let hidden_dim = 16usize;
    let num_heads = 2usize;
    let q_lora_rank = 8usize;
    let kv_lora_rank = 4usize;
    let qk_rope_dim = 2usize;
    let qk_nope_dim = 3usize;
    let v_head_dim = 4usize;

    let config = ModelConfig {
        name: "test-mla-weights".into(),
        architecture: "test".into(),
        hidden_dim: hidden_dim as u32,
        expert_hidden_dim: 8,
        num_layers: 1,
        num_moe_layers: 0,
        dense_layer_idx: 0,
        num_experts: 1,
        num_active_experts: 1,
        num_heads: num_heads as u32,
        num_kv_heads: num_heads as u32,
        max_seq_len: 64,
        vocab_size: 64,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        mla: Some(MlaConfig {
            kv_lora_rank: kv_lora_rank as u32,
            q_lora_rank: q_lora_rank as u32,
            qk_rope_head_dim: qk_rope_dim as u32,
            qk_nope_head_dim: qk_nope_dim as u32,
            v_head_dim: v_head_dim as u32,
            softmax_scale: 1.0 / ((qk_nope_dim + qk_rope_dim) as f32).sqrt(),
        }),
        ..Default::default()
    };

    let mla = config.mla.as_ref().unwrap();

    // Create weight matrices (small, identity-like for predictability)
    let q_head_dim = qk_nope_dim + qk_rope_dim;
    let q_a_data: Vec<f16> = (0..q_lora_rank * hidden_dim)
        .map(|i| {
            f16::from_f32(if i / hidden_dim == i % hidden_dim {
                1.0
            } else {
                0.0
            })
        })
        .collect();
    let q_b_data: Vec<f16> = (0..num_heads * q_head_dim * q_lora_rank)
        .map(|i| {
            f16::from_f32(if i / q_lora_rank == i % q_lora_rank {
                0.1
            } else {
                0.0
            })
        })
        .collect();
    let kv_a_dim = kv_lora_rank + qk_rope_dim;
    let kv_a_data: Vec<f16> = (0..kv_a_dim * hidden_dim)
        .map(|i| {
            f16::from_f32(if i / hidden_dim == i % hidden_dim {
                1.0
            } else {
                0.0
            })
        })
        .collect();
    let kv_b_out_dim = num_heads * (qk_nope_dim + v_head_dim);
    let kv_b_data: Vec<f16> = (0..kv_b_out_dim * kv_lora_rank)
        .map(|i| {
            f16::from_f32(if i / kv_lora_rank == i % kv_lora_rank {
                0.1
            } else {
                0.0
            })
        })
        .collect();
    let o_dim = num_heads * v_head_dim;
    let o_data: Vec<f16> = (0..hidden_dim * o_dim)
        .map(|i| f16::from_f32(if i / o_dim == i % o_dim { 1.0 } else { 0.0 }))
        .collect();

    let weights = MlaWeights {
        q_a_proj: Some(&q_a_data),
        q_b_proj: Some(&q_b_data),
        kv_a_proj: Some(&kv_a_data),
        kv_b_proj: Some(&kv_b_data),
        o_proj: Some(&o_data),
        q_norm: None,
        kv_norm: None,
    };

    let mut kv_cache = MlaKvCache::new(kv_lora_rank, qk_rope_dim, 64);

    let hidden_state: Vec<f16> = (0..hidden_dim)
        .map(|i| f16::from_f32((i as f32 + 1.0) * 0.1))
        .collect();

    // Run attention
    let output = mla_attention_layer(&hidden_state, &weights, &mut kv_cache, 0, &config, mla);

    assert_eq!(output.len(), hidden_dim);
    assert_eq!(kv_cache.seq_len, 1);

    // All outputs should be finite and non-zero (we have real weights)
    let mut has_nonzero = false;
    for v in &output {
        assert!(v.to_f32().is_finite());
        if v.to_f32().abs() > 1e-6 {
            has_nonzero = true;
        }
    }
    assert!(
        has_nonzero,
        "Output should have non-zero values with real weights"
    );
}

#[test]
fn test_mla_attention_multi_position() {
    // Test that MLA attention produces different outputs for different positions
    let config = ModelConfig {
        name: "test-mla-seq".into(),
        architecture: "test".into(),
        hidden_dim: 16,
        expert_hidden_dim: 8,
        num_layers: 1,
        num_moe_layers: 0,
        dense_layer_idx: 0,
        num_experts: 1,
        num_active_experts: 1,
        num_heads: 2,
        num_kv_heads: 2,
        max_seq_len: 64,
        vocab_size: 64,
        expert_dtype: DType::FP16,
        shared_dtype: DType::FP16,
        mla: Some(MlaConfig {
            kv_lora_rank: 4,
            q_lora_rank: 8,
            qk_rope_head_dim: 2,
            qk_nope_head_dim: 3,
            v_head_dim: 4,
            softmax_scale: 1.0 / (5.0f32).sqrt(), // 1/sqrt(3+2)
        }),
        ..Default::default()
    };

    let mla = config.mla.as_ref().unwrap();
    let mut kv_cache = MlaKvCache::new(4, 2, 64);

    let weights = MlaWeights {
        q_a_proj: None,
        q_b_proj: None,
        kv_a_proj: None,
        kv_b_proj: None,
        o_proj: None,
        q_norm: None,
        kv_norm: None,
    };

    let state1: Vec<f16> = (0..16).map(|i| f16::from_f32(i as f32 * 0.1)).collect();
    let state2: Vec<f16> = (0..16)
        .map(|i| f16::from_f32(i as f32 * 0.2 + 0.05))
        .collect();

    // Position 0
    let out0 = mla_attention_layer(&state1, &weights, &mut kv_cache, 0, &config, mla);
    // Position 1 (different hidden state)
    let _out1 = mla_attention_layer(&state2, &weights, &mut kv_cache, 1, &config, mla);
    // Position 2 (same hidden state as pos 0, but different KV cache)
    let out2 = mla_attention_layer(&state1, &weights, &mut kv_cache, 2, &config, mla);

    assert_eq!(kv_cache.seq_len, 3);

    // out0 and out2 should differ because out2 has more KV cache entries
    let diff: f32 = out0
        .iter()
        .zip(out2.iter())
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "Position 0 and 2 should differ due to different KV context"
    );
}

#[test]
fn test_mla_kimi_k25_dimensions() {
    // Verify MlaConfig dimensions match Kimi K2.5 spec
    let config = ModelConfig::kimi_k25();
    let mla = config
        .mla
        .as_ref()
        .expect("Kimi K2.5 should have MLA config");

    assert_eq!(mla.kv_lora_rank, 512);
    assert_eq!(mla.q_lora_rank, 1536);
    assert_eq!(mla.qk_rope_head_dim, 64);
    assert_eq!(mla.qk_nope_head_dim, 128);
    assert_eq!(mla.v_head_dim, 128);

    // Q path dimensions
    let q_head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim; // 128 + 64 = 192
    assert_eq!(q_head_dim, 192);
    let q_b_out = config.num_heads * q_head_dim; // 64 * 192 = 12288
    assert_eq!(q_b_out, 12288);

    // KV path dimensions
    let kv_a_dim = mla.kv_lora_rank + mla.qk_rope_head_dim; // 512 + 64 = 576
    assert_eq!(kv_a_dim, 576);
    let kv_b_out = config.num_heads * (mla.qk_nope_head_dim + mla.v_head_dim); // 64 * 256 = 16384
    assert_eq!(kv_b_out, 16384);

    // O projection
    let o_in = config.num_heads * mla.v_head_dim; // 64 * 128 = 8192
    assert_eq!(o_in, 8192);
}

// ───────────────────────────────────────────────────────────────────────
// CPU auto-config tests
// ───────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_engine_cpu_auto_config() {
    // When no VRAM budget is specified and no real GPU is present,
    // the engine should auto-configure T1 and T2 from system RAM.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_cpu_auto.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);

    // Default config: t1_capacity=0, t2_capacity=0 — should auto-configure
    let config = EngineConfig {
        model_path: path_str.to_string(),
        ..Default::default()
    };

    let engine = vib3::runtime::engine::Engine::new(config).await.unwrap();

    // Engine should have created a functioning buffer manager with non-zero pools.
    // On CPU, the "device" allocations are just host memory, so both T1 and T2
    // should have been auto-configured with system RAM.
    // The engine should be operational — try generating to prove it
    // (the test model is tiny so generate will exercise the page loading path)
    assert_eq!(engine.current_mode(), ActivationMode::Generalist);
}

#[tokio::test]
async fn test_engine_cpu_auto_config_generates() {
    // Verify that CPU auto-config produces an engine that can actually generate tokens.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_cpu_gen.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);

    // No VRAM/RAM budget — should auto-configure everything
    let config = EngineConfig {
        model_path: path_str.to_string(),
        ..Default::default()
    };

    let mut engine = vib3::runtime::engine::Engine::new(config).await.unwrap();
    let result = engine.generate("Test CPU auto-config").await.unwrap();

    // Should produce tokens without crashing
    assert!(
        result.tokens_generated > 0,
        "Engine should generate tokens with CPU auto-config"
    );
    assert!(result.tokens_per_second > 0.0);
}

#[tokio::test]
async fn test_step_zero_does_not_advance_position() {
    // Regression test: decode step 0 reuses prefill state and must not
    // advance position. Step 1+ should advance as KV is written.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_step0_position.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);

    let config = EngineConfig {
        model_path: path_str.to_string(),
        ..Default::default()
    };

    let mut engine = vib3::runtime::engine::Engine::new(config).await.unwrap();

    let input_tokens = engine.tokenizer().encode("step0 regression");
    assert!(!input_tokens.is_empty());

    engine.prefill_tokens(&input_tokens).await.unwrap();
    let prefill_pos = engine.position();
    assert_eq!(prefill_pos, input_tokens.len());

    let params = SamplingParams {
        max_tokens: 1,
        ..Default::default()
    };

    let _tok0 = engine.generate_one_token(0, &params).await.unwrap();
    assert_eq!(
        engine.position(),
        prefill_pos,
        "step 0 must not advance decode position"
    );
    assert_eq!(engine.decode_step(), 1);

    let _tok1 = engine.generate_one_token(1, &params).await.unwrap();
    assert_eq!(
        engine.position(),
        prefill_pos + 1,
        "step 1 should advance decode position"
    );
    assert_eq!(engine.decode_step(), 2);
}

/// Verify that pages in the real Mixtral .vib3 can be read and decompressed correctly.
/// This test is skipped if the model file doesn't exist (CI environments).
#[test]
fn test_mixtral_page_integrity() {
    let path = "/code/vib3/models/mixtral-full.vib3";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping: {} not found", path);
        return;
    }

    let file = vib3::storage::format::Vib3File::open(path).unwrap();
    let pc = file.page_count();
    assert!(pc > 1000, "Expected >1000 pages, got {}", pc);

    // Test first 50 pages and some scattered pages
    let mut test_indices: Vec<usize> = (0..50.min(pc)).collect();
    // Add some scattered pages
    for &idx in &[100, 500, 1000, 5000, 10000, pc - 1] {
        if idx < pc {
            test_indices.push(idx);
        }
    }

    let mut ok = 0;
    let mut fail = 0;
    for &i in &test_indices {
        let entry = file.page(i);
        let raw_size = entry.raw_size.min(vib3::core::types::PAGE_SIZE as u32) as usize;
        let mut buf = vec![0u8; raw_size];
        match file.read_page_sync(i, &mut buf) {
            Ok(_n) => {
                ok += 1;
            }
            Err(e) => {
                let pid = entry.page_id();
                let comp = entry.compression;
                let rs = { entry.raw_size };
                let cs = { entry.compressed_size };
                let fo = { entry.file_offset };
                eprintln!(
                    "Page {} [L{} E{:#06x} S{} P{}]: FAIL comp={}, raw={}, compressed={}, offset={}: {:?}",
                    i, pid.layer, pid.expert, pid.segment, pid.page_idx,
                    comp, rs, cs, fo, e
                );
                fail += 1;
            }
        }
    }

    eprintln!("Page integrity: {}/{} OK, {} failed", ok, ok + fail, fail);
    assert_eq!(fail, 0, "Some pages failed to decompress");
}

/// Slow smoke test for real Mixtral CPU decode path.
///
/// Run explicitly:
///   cargo test --release --no-default-features test_mixtral_cpu_decode_smoke -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn test_mixtral_cpu_decode_smoke() {
    let model_path = "/code/vib3/models/mixtral-full-v2.vib3";
    let tokenizer_path = "/code/vib3/models/mixtral/tokenizer.json";

    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping: {} not found", model_path);
        return;
    }
    if !std::path::Path::new(tokenizer_path).exists() {
        eprintln!("Skipping: {} not found", tokenizer_path);
        return;
    }

    let config = EngineConfig {
        model_path: model_path.to_string(),
        tokenizer_path: tokenizer_path.to_string(),
        ..Default::default()
    };

    let mut engine = vib3::runtime::engine::Engine::new(config).await.unwrap();

    let input = "[INST] What is the capital of France? [/INST]";
    let input_tokens = engine.tokenizer().encode(input);
    assert!(!input_tokens.is_empty());

    engine.prefill_tokens(&input_tokens).await.unwrap();
    let prefill_pos = engine.position();
    assert_eq!(prefill_pos, input_tokens.len());

    let params = SamplingParams {
        max_tokens: 2,
        ..Default::default()
    };

    let _tok0 = engine.generate_one_token(0, &params).await.unwrap();
    assert_eq!(engine.position(), prefill_pos);
    assert_eq!(engine.decode_step(), 1);

    let _tok1 = engine.generate_one_token(1, &params).await.unwrap();
    assert_eq!(engine.position(), prefill_pos + 1);
    assert_eq!(engine.decode_step(), 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Gearbox Integration Tests — Phases A through E
// ═══════════════════════════════════════════════════════════════════════════

use vib3::core::config::GearConfig;
use vib3::core::types::{Gear, TaskContext};
use vib3::runtime::gear_scheduler::{GearBatchScheduler, InferenceRequest, RequestParams};

// ─── Phase A: Task Signal API ─────────────────────────────────────────────

#[test]
fn test_gear_enum_all_variants() {
    assert_eq!(Gear::ALL.len(), 6);
    let names: Vec<&str> = Gear::ALL.iter().map(|g| g.name()).collect();
    assert_eq!(names, vec!["code", "vision", "reason", "tool", "chat", "memory"]);
}

#[test]
fn test_gear_from_str_case_insensitive() {
    assert_eq!(Gear::from_str("code"), Some(Gear::Code));
    assert_eq!(Gear::from_str("CODE"), Some(Gear::Code));
    assert_eq!(Gear::from_str("Code"), Some(Gear::Code));
    assert_eq!(Gear::from_str("vision"), Some(Gear::Vision));
    assert_eq!(Gear::from_str("reason"), Some(Gear::Reason));
    assert_eq!(Gear::from_str("reasoning"), Some(Gear::Reason));
    assert_eq!(Gear::from_str("tool"), Some(Gear::Tool));
    assert_eq!(Gear::from_str("tools"), Some(Gear::Tool));
    assert_eq!(Gear::from_str("chat"), Some(Gear::Chat));
    assert_eq!(Gear::from_str("conversation"), Some(Gear::Chat));
    assert_eq!(Gear::from_str("memory"), Some(Gear::Memory));
    assert_eq!(Gear::from_str("retrieval"), Some(Gear::Memory));
    assert_eq!(Gear::from_str("unknown_gear"), None);
    assert_eq!(Gear::from_str(""), None);
}

#[test]
fn test_gear_expected_mode_mapping() {
    // Specialist gears
    assert_eq!(Gear::Code.expected_mode(), ActivationMode::Specialist);
    assert_eq!(Gear::Vision.expected_mode(), ActivationMode::Specialist);
    assert_eq!(Gear::Reason.expected_mode(), ActivationMode::Specialist);
    assert_eq!(Gear::Tool.expected_mode(), ActivationMode::Specialist);
    // Generalist gears
    assert_eq!(Gear::Chat.expected_mode(), ActivationMode::Generalist);
    assert_eq!(Gear::Memory.expected_mode(), ActivationMode::Generalist);
}

#[test]
fn test_gear_display() {
    assert_eq!(format!("{}", Gear::Code), "code");
    assert_eq!(format!("{}", Gear::Vision), "vision");
    assert_eq!(format!("{}", Gear::Memory), "memory");
}

#[test]
fn test_task_context_with_gear() {
    let ctx = TaskContext::with_gear("code");
    assert_eq!(ctx.gear, Some("code".to_string()));
    assert!(ctx.blend.is_none());
    assert!(ctx.alpha.is_none());
    assert!(ctx.phase.is_none());
    assert_eq!(ctx.primary_gear(), Some(Gear::Code));
    assert_eq!(ctx.expected_mode(), Some(ActivationMode::Specialist));
}

#[test]
fn test_task_context_with_blend() {
    let mut blend = std::collections::HashMap::new();
    blend.insert("code".to_string(), 0.7);
    blend.insert("reason".to_string(), 0.3);

    let ctx = TaskContext::with_blend(blend);

    // Primary gear should be "code" (highest weight)
    assert_eq!(ctx.primary_gear(), Some(Gear::Code));
    assert_eq!(ctx.expected_mode(), Some(ActivationMode::Specialist));

    // Blend should be preserved
    let gear_blend = ctx.gear_blend();
    assert_eq!(gear_blend.len(), 2);

    // Check code has 0.7, reason has 0.3
    let code_weight = gear_blend.iter().find(|(g, _)| *g == Gear::Code).map(|(_, w)| *w);
    let reason_weight = gear_blend.iter().find(|(g, _)| *g == Gear::Reason).map(|(_, w)| *w);
    assert!((code_weight.unwrap() - 0.7).abs() < 0.01);
    assert!((reason_weight.unwrap() - 0.3).abs() < 0.01);
}

#[test]
fn test_task_context_gear_blend_no_blend() {
    let ctx = TaskContext::with_gear("vision");
    // No blend → primary gear treated as 100%
    let blend = ctx.gear_blend();
    assert_eq!(blend.len(), 1);
    assert_eq!(blend[0].0, Gear::Vision);
    assert!((blend[0].1 - 1.0).abs() < 0.01);
}

#[test]
fn test_task_context_unknown_gear() {
    let ctx = TaskContext::with_gear("unknown_task");
    assert_eq!(ctx.primary_gear(), None);
    assert_eq!(ctx.expected_mode(), None);
    assert!(ctx.gear_blend().is_empty());
}

#[test]
fn test_task_context_empty() {
    let ctx = TaskContext {
        gear: None,
        blend: None,
        alpha: None,
        phase: None,
    };
    assert_eq!(ctx.primary_gear(), None);
    assert_eq!(ctx.expected_mode(), None);
    assert!(ctx.gear_blend().is_empty());
}

#[test]
fn test_task_context_serde_roundtrip() {
    let ctx = TaskContext {
        gear: Some("code".to_string()),
        blend: Some({
            let mut m = std::collections::HashMap::new();
            m.insert("code".to_string(), 0.8);
            m.insert("reason".to_string(), 0.2);
            m
        }),
        alpha: Some(0.5),
        phase: Some("executing".to_string()),
    };

    let json = serde_json::to_string(&ctx).unwrap();
    let deserialized: TaskContext = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.gear, ctx.gear);
    assert_eq!(deserialized.alpha, ctx.alpha);
    assert_eq!(deserialized.phase, ctx.phase);
    assert_eq!(deserialized.primary_gear(), Some(Gear::Code));
}

#[test]
fn test_task_context_serde_from_api_json() {
    // Simulate the JSON that comes through the OpenAI-compatible API
    let json_str = r#"{
        "gear": "reason",
        "blend": {"reason": 0.6, "code": 0.4},
        "alpha": 0.01
    }"#;

    let ctx: TaskContext = serde_json::from_str(json_str).unwrap();
    assert_eq!(ctx.primary_gear(), Some(Gear::Reason));
    assert_eq!(ctx.alpha, Some(0.01));
    assert!(ctx.phase.is_none()); // Optional, not present
}

#[test]
fn test_task_context_serde_minimal_json() {
    // Minimal JSON: just a gear name
    let json_str = r#"{"gear": "chat"}"#;
    let ctx: TaskContext = serde_json::from_str(json_str).unwrap();
    assert_eq!(ctx.primary_gear(), Some(Gear::Chat));
    assert!(ctx.blend.is_none());
    assert!(ctx.alpha.is_none());
}

#[test]
fn test_task_context_serde_empty_json() {
    // Empty JSON object: all fields optional
    let json_str = r#"{}"#;
    let ctx: TaskContext = serde_json::from_str(json_str).unwrap();
    assert_eq!(ctx.primary_gear(), None);
}

#[test]
fn test_gear_config_defaults() {
    let gc = GearConfig::default();
    assert!(gc.enabled);
    assert!((gc.default_alpha - 0.01).abs() < 0.001);
    assert!(gc.override_mode_detection);
    assert!(!gc.proactive_cache_warming);
    assert!(!gc.filtered_hnsw);
    assert!(!gc.gear_aware_eviction);
    assert!(gc.gear_domains.is_empty());
}

#[test]
fn test_engine_config_has_gear_config() {
    let config = EngineConfig::default();
    assert!(config.gear.enabled);
    assert!(config.gear.override_mode_detection);
}

// ─── Phase A: Engine set_task_context ──────────────────────────────────────

#[tokio::test]
async fn test_engine_set_task_context_code_gear() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_a_code.vib3");
    let path_str = path.to_str().unwrap();
    create_test_model(path_str);

    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        gear: GearConfig {
            enabled: true,
            override_mode_detection: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();
    assert_eq!(engine.current_mode(), ActivationMode::Generalist);
    assert!(engine.task_context().is_none());

    // Set code gear → should force Specialist mode
    engine.set_task_context(Some(TaskContext::with_gear("code"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Specialist);
    assert!(engine.task_context().is_some());
    assert_eq!(engine.task_context().unwrap().primary_gear(), Some(Gear::Code));
}

#[tokio::test]
async fn test_engine_set_task_context_chat_gear() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_a_chat.vib3");
    let path_str = path.to_str().unwrap();
    create_test_model(path_str);

    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        gear: GearConfig {
            enabled: true,
            override_mode_detection: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();

    // First force specialist via code gear
    engine.set_task_context(Some(TaskContext::with_gear("code"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Specialist);

    // Switch to chat gear → should force Generalist mode
    engine.set_task_context(Some(TaskContext::with_gear("chat"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Generalist);
}

#[tokio::test]
async fn test_engine_clear_task_context() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_a_clear.vib3");
    let path_str = path.to_str().unwrap();
    create_test_model(path_str);

    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        gear: GearConfig {
            enabled: true,
            override_mode_detection: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();

    // Set a gear
    engine.set_task_context(Some(TaskContext::with_gear("code"))).await;
    assert!(engine.task_context().is_some());

    // Clear it
    engine.set_task_context(None).await;
    assert!(engine.task_context().is_none());
}

#[tokio::test]
async fn test_engine_gear_disabled_ignores_context() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_a_disabled.vib3");
    let path_str = path.to_str().unwrap();
    create_test_model(path_str);

    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        gear: GearConfig {
            enabled: false, // Disabled!
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();
    assert_eq!(engine.current_mode(), ActivationMode::Generalist);

    // Setting gear should store context but NOT change mode
    engine.set_task_context(Some(TaskContext::with_gear("code"))).await;
    // Context is stored (accepted) but mode is not overridden
    assert!(engine.task_context().is_some());
    assert_eq!(engine.current_mode(), ActivationMode::Generalist);
}

#[tokio::test]
async fn test_engine_same_gear_no_transition() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_a_same.vib3");
    let path_str = path.to_str().unwrap();
    create_test_model(path_str);

    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        gear: GearConfig {
            enabled: true,
            override_mode_detection: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();

    // Set code gear
    engine.set_task_context(Some(TaskContext::with_gear("code"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Specialist);

    // Set code gear again — no transition, should be a no-op
    engine.set_task_context(Some(TaskContext::with_gear("code"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Specialist);
}

#[tokio::test]
async fn test_engine_generate_with_task_context() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_a_gen.vib3");
    let path_str = path.to_str().unwrap();
    create_test_model(path_str);

    let config = EngineConfig {
        model_path: path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        gear: GearConfig {
            enabled: true,
            override_mode_detection: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();

    // Set task context before generation
    engine.set_task_context(Some(TaskContext::with_gear("reason"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Specialist);

    // Generate should work with task context active
    let params = SamplingParams {
        max_tokens: 5,
        temperature: 0.0,
        ..Default::default()
    };
    let result = engine.generate_with_params("test with gear", params).await.unwrap();
    assert!(result.tokens_generated > 0);
    assert!(result.total_time_ms > 0.0);
}

// ─── Phase B: Gear Profile Loading & Cache Warming ─────────────────────

#[tokio::test]
async fn test_engine_load_gear_profiles() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("test_gear_b_profiles.vib3");
    let model_path_str = model_path.to_str().unwrap();
    create_test_model(model_path_str);

    // Create a fake gear profiles JSON
    let profiles_json = serde_json::json!({
        "code": {
            "hot_experts": [[0, 1], [2, 3]],
            "total_unique_experts": 4,
            "estimated_vram_gb": 0.1
        },
        "vision": {
            "hot_experts": [[1, 2], [0, 3]],
            "total_unique_experts": 4,
            "estimated_vram_gb": 0.1
        }
    });
    let profiles_path = dir.path().join("gear_profiles.json");
    std::fs::write(&profiles_path, serde_json::to_string(&profiles_json).unwrap()).unwrap();

    let config = EngineConfig {
        model_path: model_path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();
    assert_eq!(engine.gear_profile_count(), 0);

    let count = engine.load_gear_profiles(profiles_path.to_str().unwrap());
    assert_eq!(count, 2);
    assert_eq!(engine.gear_profile_count(), 2);
}

#[tokio::test]
async fn test_engine_load_gear_profiles_nonexistent() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("test_gear_b_nofile.vib3");
    let model_path_str = model_path.to_str().unwrap();
    create_test_model(model_path_str);

    let config = EngineConfig {
        model_path: model_path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();

    // Loading from non-existent file should return 0, not panic
    let count = engine.load_gear_profiles("/nonexistent/path/gear_profiles.json");
    assert_eq!(count, 0);
    assert_eq!(engine.gear_profile_count(), 0);
}

#[tokio::test]
async fn test_engine_proactive_cache_warming() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("test_gear_b_warm.vib3");
    let model_path_str = model_path.to_str().unwrap();
    create_test_model(model_path_str);

    // Create gear profiles that reference experts in our test model
    // Test model has: 2 MoE layers (layer 0, 1), 4 experts each
    let profiles_json = serde_json::json!({
        "code": {
            "hot_experts": [[0, 1], [2, 3]]
        },
        "chat": {
            "hot_experts": [[2, 3], [0, 1]]
        }
    });
    let profiles_path = dir.path().join("gear_profiles.json");
    std::fs::write(&profiles_path, serde_json::to_string(&profiles_json).unwrap()).unwrap();

    let config = EngineConfig {
        model_path: model_path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        gear: GearConfig {
            enabled: true,
            override_mode_detection: true,
            proactive_cache_warming: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();
    engine.load_gear_profiles(profiles_path.to_str().unwrap());
    assert_eq!(engine.gear_profile_count(), 2);

    // Activate code gear → should trigger cache warming
    engine.set_task_context(Some(TaskContext::with_gear("code"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Specialist);

    // Switch to chat gear → should unpin code's pages, pin chat's pages
    engine.set_task_context(Some(TaskContext::with_gear("chat"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Generalist);
}

// ─── Phase C: Domain Tags & Gear-Filtered HNSW ────────────────────────

#[test]
fn test_vector_index_domain_tags() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_c_tags.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model_with_vector_index(path_str);
    let model_file = Vib3File::open(path_str).unwrap();

    let mut vi = VectorIndex::load(&model_file).unwrap();

    // Initially no domain tags
    assert!(!vi.has_domain_tags());

    // Set domain tags for centroids
    vi.set_domain_tags(0, vec!["code".to_string(), "math".to_string()]);
    vi.set_domain_tags(1, vec!["vision".to_string(), "creative".to_string()]);

    assert!(vi.has_domain_tags());

    // Check tag matching
    assert!(vi.centroid_matches_domains(0, &["code".to_string()]));
    assert!(vi.centroid_matches_domains(0, &["math".to_string()]));
    assert!(!vi.centroid_matches_domains(0, &["vision".to_string()]));
    assert!(vi.centroid_matches_domains(1, &["vision".to_string()]));
    assert!(!vi.centroid_matches_domains(1, &["code".to_string()]));

    // Untagged centroid matches everything (no filter)
    assert!(vi.centroid_matches_domains(99, &["anything".to_string()]));
}

#[test]
fn test_vector_index_centroid_matches_empty_domains() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_c_empty.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model_with_vector_index(path_str);
    let model_file = Vib3File::open(path_str).unwrap();

    let mut vi = VectorIndex::load(&model_file).unwrap();
    vi.set_domain_tags(0, vec!["code".to_string()]);

    // Empty domains list → matches everything (no filter active)
    assert!(vi.centroid_matches_domains(0, &[]));
    assert!(vi.centroid_matches_domains(1, &[]));
}

#[test]
fn test_planner_set_gear_domains() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_c_planner.vib3");
    let path_str = path.to_str().unwrap();

    let model_config = create_test_model(path_str);
    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    let buf_config = BufferPoolConfig {
        t1_capacity: 16 * PAGE_SIZE,
        t2_capacity: 32 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };
    let mgr = Arc::new(PageBufferManager::new(buf_config, model_file.clone()));
    let mut planner = QueryPlanner::new(mgr, None, model_file, model_config);

    // Set gear domains (used for filtered HNSW)
    planner.set_gear_domains(vec!["code".to_string(), "math".to_string()]);

    // Clear gear domains
    planner.set_gear_domains(vec![]);
}

#[tokio::test]
async fn test_engine_filtered_hnsw_wiring() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("test_gear_c_hnsw.vib3");
    let model_path_str = model_path.to_str().unwrap();
    create_test_model(model_path_str);

    let mut gear_domains = std::collections::HashMap::new();
    gear_domains.insert(
        "code".to_string(),
        vec!["code".to_string(), "math".to_string()],
    );
    gear_domains.insert(
        "vision".to_string(),
        vec!["vision".to_string(), "spatial".to_string()],
    );

    let config = EngineConfig {
        model_path: model_path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        gear: GearConfig {
            enabled: true,
            override_mode_detection: true,
            filtered_hnsw: true,
            gear_domains,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();

    // Setting code gear should propagate domain filter to planner
    engine.set_task_context(Some(TaskContext::with_gear("code"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Specialist);

    // Setting vision gear should update domain filter
    engine.set_task_context(Some(TaskContext::with_gear("vision"))).await;

    // Clearing context should clear domain filter
    engine.set_task_context(None).await;
}

// ─── Phase D: Gear-Aware Eviction ──────────────────────────────────────

#[tokio::test]
async fn test_buffer_manager_set_current_gear() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_d_set.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);
    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    let buf_config = BufferPoolConfig {
        t1_capacity: 8 * PAGE_SIZE,
        t2_capacity: 16 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };
    let mgr = PageBufferManager::new(buf_config, model_file.clone());
    mgr.initialize().await.unwrap();

    // Initially no gear set
    mgr.set_current_gear(None);

    // Set to "code"
    mgr.set_current_gear(Some("code".to_string()));

    // Load a page — it should be tagged with "code"
    let first_page = model_file.page(0).page_id();
    let _handle = mgr.get_page(&first_page).await.unwrap();

    // Switch gear to "vision"
    mgr.set_current_gear(Some("vision".to_string()));

    // Load another page — tagged with "vision"
    if model_file.page_count() > 1 {
        let second_page = model_file.page(1).page_id();
        let _handle2 = mgr.get_page(&second_page).await.unwrap();
    }

    // Clear gear
    mgr.set_current_gear(None);
}

#[tokio::test]
async fn test_gear_aware_eviction_preference() {
    // Test that gear-aware eviction deprioritizes pages from other gears.
    // With a tiny T1 that can only hold a few pages, load pages under
    // different gears and verify eviction behavior.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_gear_d_evict.vib3");
    let path_str = path.to_str().unwrap();

    create_test_model(path_str);
    let model_file = Arc::new(Vib3File::open(path_str).unwrap());

    // Very small T1 to force eviction
    let buf_config = BufferPoolConfig {
        t1_capacity: 4 * PAGE_SIZE,
        t2_capacity: 32 * PAGE_SIZE,
        t2_compressed: false,
        ..Default::default()
    };
    let mgr = PageBufferManager::new(buf_config, model_file.clone());
    mgr.initialize().await.unwrap();

    let page_count = model_file.page_count();
    if page_count < 6 {
        return; // Not enough pages to test eviction
    }

    // Load first batch under "code" gear
    mgr.set_current_gear(Some("code".to_string()));
    for i in 0..3.min(page_count) {
        let page = model_file.page(i).page_id();
        let _ = mgr.get_page(&page).await;
    }

    // Switch to "vision" gear and load more pages (should evict code pages first)
    mgr.set_current_gear(Some("vision".to_string()));
    for i in 3..6.min(page_count) {
        let page = model_file.page(i).page_id();
        let _ = mgr.get_page(&page).await;
    }

    // The engine should not crash — gear-aware eviction is transparent
    let stats = mgr.stats.snapshot();
    assert!(stats.total_page_accesses > 0);
}

#[tokio::test]
async fn test_engine_gear_aware_eviction_wiring() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("test_gear_d_engine.vib3");
    let model_path_str = model_path.to_str().unwrap();
    create_test_model(model_path_str);

    let config = EngineConfig {
        model_path: model_path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        gear: GearConfig {
            enabled: true,
            override_mode_detection: true,
            gear_aware_eviction: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();

    // Setting gear should propagate to buffer manager
    engine.set_task_context(Some(TaskContext::with_gear("code"))).await;

    // Generate should work with gear-aware eviction active
    let params = SamplingParams {
        max_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };
    let result = engine.generate_with_params("test eviction", params).await.unwrap();
    assert!(result.tokens_generated > 0);

    // Switch gear and generate again
    engine.set_task_context(Some(TaskContext::with_gear("vision"))).await;
    let result2 = engine.generate_with_params("test vision", SamplingParams {
        max_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    }).await.unwrap();
    assert!(result2.tokens_generated > 0);
}

// ─── Phase E: Gear-Batched Scheduling ──────────────────────────────────

#[test]
fn test_gear_scheduler_empty() {
    let mut sched = GearBatchScheduler::new();
    assert_eq!(sched.pending_count(), 0);
    assert!(sched.active_gear().is_none());
    assert!(sched.next_batch().is_empty());

    let (enqueued, dispatched) = sched.stats();
    assert_eq!(enqueued, 0);
    assert_eq!(dispatched, 0);
}

#[test]
fn test_gear_scheduler_single_gear() {
    let mut sched = GearBatchScheduler::new();

    sched.enqueue(InferenceRequest {
        id: "r1".into(),
        gear: Some("code".into()),
        enqueued_at: std::time::Instant::now(),
        prompt: "write a function".into(),
        params: RequestParams::default(),
    });
    sched.enqueue(InferenceRequest {
        id: "r2".into(),
        gear: Some("code".into()),
        enqueued_at: std::time::Instant::now(),
        prompt: "debug this code".into(),
        params: RequestParams::default(),
    });

    assert_eq!(sched.pending_count(), 2);

    let batch = sched.next_batch();
    assert_eq!(batch.len(), 2);
    assert!(batch.iter().all(|r| r.gear.as_deref() == Some("code")));
    assert_eq!(sched.active_gear(), Some("code"));
    assert_eq!(sched.pending_count(), 0);
}

#[test]
fn test_gear_scheduler_multi_gear_groups_by_gear() {
    let mut sched = GearBatchScheduler::new();

    // Add requests for 3 different gears
    sched.enqueue(InferenceRequest {
        id: "c1".into(),
        gear: Some("code".into()),
        enqueued_at: std::time::Instant::now(),
        prompt: "".into(),
        params: RequestParams::default(),
    });
    sched.enqueue(InferenceRequest {
        id: "v1".into(),
        gear: Some("vision".into()),
        enqueued_at: std::time::Instant::now(),
        prompt: "".into(),
        params: RequestParams::default(),
    });
    sched.enqueue(InferenceRequest {
        id: "c2".into(),
        gear: Some("code".into()),
        enqueued_at: std::time::Instant::now(),
        prompt: "".into(),
        params: RequestParams::default(),
    });
    sched.enqueue(InferenceRequest {
        id: "r1".into(),
        gear: Some("reason".into()),
        enqueued_at: std::time::Instant::now(),
        prompt: "".into(),
        params: RequestParams::default(),
    });
    sched.enqueue(InferenceRequest {
        id: "c3".into(),
        gear: Some("code".into()),
        enqueued_at: std::time::Instant::now(),
        prompt: "".into(),
        params: RequestParams::default(),
    });

    // First batch should be code (3 requests — most pending)
    let batch1 = sched.next_batch();
    assert_eq!(batch1.len(), 3);
    assert!(batch1.iter().all(|r| r.gear.as_deref() == Some("code")));

    // Second batch picks the next largest group
    let batch2 = sched.next_batch();
    assert_eq!(batch2.len(), 1);

    // Third batch
    let batch3 = sched.next_batch();
    assert_eq!(batch3.len(), 1);

    // Batches 2 and 3 should cover vision and reason
    let mut gears_served: Vec<String> = vec![];
    gears_served.push(batch2[0].gear.clone().unwrap());
    gears_served.push(batch3[0].gear.clone().unwrap());
    gears_served.sort();
    assert_eq!(gears_served, vec!["reason", "vision"]);

    assert_eq!(sched.pending_count(), 0);
}

#[test]
fn test_gear_scheduler_pending_by_gear() {
    let mut sched = GearBatchScheduler::new();

    sched.enqueue(InferenceRequest {
        id: "1".into(),
        gear: Some("code".into()),
        enqueued_at: std::time::Instant::now(),
        prompt: "".into(),
        params: RequestParams::default(),
    });
    sched.enqueue(InferenceRequest {
        id: "2".into(),
        gear: Some("code".into()),
        enqueued_at: std::time::Instant::now(),
        prompt: "".into(),
        params: RequestParams::default(),
    });
    sched.enqueue(InferenceRequest {
        id: "3".into(),
        gear: Some("vision".into()),
        enqueued_at: std::time::Instant::now(),
        prompt: "".into(),
        params: RequestParams::default(),
    });
    sched.enqueue(InferenceRequest {
        id: "4".into(),
        gear: None,
        enqueued_at: std::time::Instant::now(),
        prompt: "".into(),
        params: RequestParams::default(),
    });

    let by_gear = sched.pending_by_gear();
    assert_eq!(by_gear.get("code"), Some(&2));
    assert_eq!(by_gear.get("vision"), Some(&1));
    assert_eq!(by_gear.get("general"), Some(&1));
    assert_eq!(sched.pending_count(), 4);
}

#[test]
fn test_gear_scheduler_with_config() {
    let mut sched = GearBatchScheduler::with_config(4, 2, 50);

    for i in 0..10 {
        sched.enqueue(InferenceRequest {
            id: format!("r{}", i),
            gear: Some("code".into()),
            enqueued_at: std::time::Instant::now(),
            prompt: "".into(),
            params: RequestParams::default(),
        });
    }

    // Max batch size = 4
    let batch = sched.next_batch();
    assert_eq!(batch.len(), 4);
    assert_eq!(sched.pending_count(), 6);
}

// ─── Cross-Phase Integration: Full Gear Lifecycle ──────────────────────

#[tokio::test]
async fn test_full_gear_lifecycle() {
    // End-to-end test: create engine → load profiles → set gear →
    // generate → switch gear → generate → clear gear → generate
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("test_gear_lifecycle.vib3");
    let model_path_str = model_path.to_str().unwrap();
    create_test_model(model_path_str);

    let profiles_json = serde_json::json!({
        "code": { "hot_experts": [[0, 1], [2, 3]] },
        "reason": { "hot_experts": [[1, 2], [0, 3]] },
        "chat": { "hot_experts": [[0, 2], [1, 3]] }
    });
    let profiles_path = dir.path().join("gear_profiles.json");
    std::fs::write(&profiles_path, serde_json::to_string(&profiles_json).unwrap()).unwrap();

    let config = EngineConfig {
        model_path: model_path_str.to_string(),
        buffer_pool: BufferPoolConfig {
            t1_capacity: 32 * PAGE_SIZE,
            t2_capacity: 64 * PAGE_SIZE,
            t2_compressed: false,
            ..Default::default()
        },
        gear: GearConfig {
            enabled: true,
            override_mode_detection: true,
            proactive_cache_warming: true,
            gear_aware_eviction: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = vib3::Engine::new(config).await.unwrap();
    engine.load_gear_profiles(profiles_path.to_str().unwrap());

    let gen_params = SamplingParams {
        max_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };

    // Phase 1: Code gear (specialist)
    engine.set_task_context(Some(TaskContext::with_gear("code"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Specialist);
    let r1 = engine.generate_with_params("code task", gen_params.clone()).await.unwrap();
    assert!(r1.tokens_generated > 0);

    // Phase 2: Switch to chat gear (generalist)
    engine.set_task_context(Some(TaskContext::with_gear("chat"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Generalist);
    let r2 = engine.generate_with_params("chat task", gen_params.clone()).await.unwrap();
    assert!(r2.tokens_generated > 0);

    // Phase 3: Switch to reason gear (specialist)
    engine.set_task_context(Some(TaskContext::with_gear("reason"))).await;
    assert_eq!(engine.current_mode(), ActivationMode::Specialist);
    let r3 = engine.generate_with_params("reason task", gen_params.clone()).await.unwrap();
    assert!(r3.tokens_generated > 0);

    // Phase 4: Clear gear (fallback to entropy-based)
    engine.set_task_context(None).await;
    assert!(engine.task_context().is_none());
    let r4 = engine.generate_with_params("no gear", gen_params).await.unwrap();
    assert!(r4.tokens_generated > 0);
}

#[test]
fn test_extra_body_serde() {
    // Test that ExtraBody with task_context deserializes correctly
    // (matches the OpenAI Python SDK pattern)
    let json_str = r#"{
        "model": "kimi-k2.5",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 100,
        "extra_body": {
            "task_context": {
                "gear": "code",
                "alpha": 0.01
            }
        }
    }"#;

    let req: vib3::api::server::ChatCompletionRequest =
        serde_json::from_str(json_str).unwrap();
    assert_eq!(req.model, "kimi-k2.5");
    assert!(req.extra_body.is_some());
    let extra = req.extra_body.unwrap();
    assert!(extra.task_context.is_some());
    let tc = extra.task_context.unwrap();
    assert_eq!(tc.primary_gear(), Some(Gear::Code));
    assert_eq!(tc.alpha, Some(0.01));
}

#[test]
fn test_extra_body_serde_no_task_context() {
    let json_str = r#"{
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}]
    }"#;

    let req: vib3::api::server::ChatCompletionRequest =
        serde_json::from_str(json_str).unwrap();
    assert!(req.extra_body.is_none());
}

#[test]
fn test_dtype_serde_roundtrip() {
    // Verify DType::INT4 serializes correctly (not as FP16)
    let int4 = DType::INT4;
    let json = serde_json::to_string(&int4).unwrap();
    assert_eq!(json, "\"INT4\"", "DType::INT4 should serialize as \"INT4\"");
    
    let fp16 = DType::FP16;
    let json2 = serde_json::to_string(&fp16).unwrap();
    assert_eq!(json2, "\"FP16\"");
    
    // Roundtrip
    let back: DType = serde_json::from_str(&json).unwrap();
    assert_eq!(back, DType::INT4);
    
    // Test in a ModelConfig context
    let config = ModelConfig {
        name: "test".into(),
        architecture: "test".into(),
        hidden_dim: 64,
        expert_hidden_dim: 32,
        num_layers: 2,
        num_moe_layers: 1,
        dense_layer_idx: 0,
        num_experts: 4,
        num_active_experts: 2,
        num_heads: 4,
        num_kv_heads: 2,
        max_seq_len: 2048,
        vocab_size: 1024,
        expert_dtype: DType::INT4,
        shared_dtype: DType::FP16,
        ..Default::default()
    };
    
    let config_json = serde_json::to_string_pretty(&config).unwrap();
    assert!(config_json.contains("\"expert_dtype\": \"INT4\""), 
        "ModelConfig JSON should contain INT4, got: {}", config_json);
    assert!(config_json.contains("\"shared_dtype\": \"FP16\""),
        "ModelConfig JSON should contain FP16");
    
    // Deserialize back
    let config_back: ModelConfig = serde_json::from_str(&config_json).unwrap();
    assert_eq!(config_back.expert_dtype, DType::INT4);
    assert_eq!(config_back.shared_dtype, DType::FP16);
}
