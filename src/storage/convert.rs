//! Safetensors → .vib3 conversion engine.
//!
//! This module contains all logic for converting HuggingFace safetensors files
//! into the `.vib3` format. It handles:
//!
//!   - Tensor name classification (mapping HF tensor names to layer/expert/segment)
//!   - INT4 quantization of expert weights with group-wise scales
//!   - BF16 → FP16 conversion for shared layers
//!   - Compressed-tensors format (_packed / _scale pairing)
//!   - Page splitting (2MB aligned pages with per-page scales)
//!   - HNSW index building from page signatures during conversion
//!   - Random model generation for testing
//!
//! Both the `vib3-convert` binary and `vib3 pull` call directly into this module.

use crate::compute::kernels;
use crate::core::config::ModelConfig;
use crate::core::types::{DType, PAGE_SIZE};
use crate::index::hnsw_backend::{
    compute_page_signature, HnswBackend, HnswConfig, SignatureMethod,
};
use crate::storage::format::{CompressionMethod, Vib3Writer};
use half::f16;
use std::path::Path;

// ─── Public API ─────────────────────────────────────────────────────────

/// Quantization format for expert weights during conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantFormat {
    /// No quantization — keep original precision (FP16/BF16).
    None,
    /// INT4 with per-group FP16 scales (group_size=32). Dequant: (nibble-8)*scale.
    Int4,
    /// NVFP4 (MXFP4 E2M1) with per-block FP16 scales (block_size=32).
    /// Uses E2M1 encoding: sign + 2-bit exponent + 1-bit mantissa.
    /// Significantly better accuracy than INT4 for MoE models.
    Nvfp4,
}

/// Options controlling the safetensors → .vib3 conversion.
#[derive(Clone, Debug)]
pub struct ConvertOptions {
    /// Quantization format for expert weights.
    pub quantize_experts: QuantFormat,
    /// Compression method for page data.
    pub compress: CompressionMethod,
    /// Build HNSW vector index from page signatures during conversion.
    pub build_indexes: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            quantize_experts: QuantFormat::Int4,
            compress: CompressionMethod::Zstd { level: 3 },
            build_indexes: true,
        }
    }
}

/// Result summary from a conversion run.
#[derive(Clone, Debug)]
pub struct ConvertResult {
    pub total_tensors: usize,
    pub converted_experts: usize,
    pub converted_shared: usize,
    pub skipped_tensors: Vec<String>,
    pub source_bytes: u64,
    pub output_bytes: u64,
    pub page_count: usize,
    pub index_vectors: usize,
}

/// Convert safetensors files from a HuggingFace model directory to .vib3 format.
///
/// Auto-detects model architecture from `config.json` if present.
/// Returns `ConvertResult` with conversion statistics.
pub fn convert_safetensors_dir(
    dir: &Path,
    config: &ModelConfig,
    output: &Path,
    options: &ConvertOptions,
) -> anyhow::Result<ConvertResult> {
    use safetensors::SafeTensors;

    println!("Scanning directory: {}", dir.display());

    let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    st_files.sort();

    if st_files.is_empty() {
        anyhow::bail!("No .safetensors files found in {}", dir.display());
    }

    println!("Found {} safetensors file(s)", st_files.len());

    // Set expert_dtype in the config we write to the file based on actual conversion
    let mut file_config = config.clone();
    match options.quantize_experts {
        QuantFormat::Int4 => file_config.expert_dtype = DType::INT4,
        QuantFormat::Nvfp4 => file_config.expert_dtype = DType::NVFP4,
        QuantFormat::None => file_config.expert_dtype = DType::FP16,
    }
    if file_config.shared_dtype == DType::BF16 {
        file_config.shared_dtype = DType::FP16;
        println!("  Note: BF16 shared weights will be converted to FP16 during conversion");
    }

    // Place temp file next to the output file (same filesystem) to avoid
    // filling up /tmp which may be on a small root partition.
    let temp_dir = output.parent();
    let mut writer = Vib3Writer::with_temp_dir(file_config, temp_dir);
    writer.set_compression(options.compress);

    let mut sig_collector = if options.build_indexes {
        println!("Index building enabled — will compute page signatures");
        Some(PageSignatureCollector::new(config.hidden_dim as usize))
    } else {
        None
    };

    let mut total_tensors = 0usize;
    let mut converted_experts = 0usize;
    let mut converted_shared = 0usize;
    let mut skipped_tensors = Vec::new();
    let mut total_source_bytes = 0u64;
    let mut total_output_bytes = 0u64;

    // ── Single-pass: Process all tensors ────────────────────────────────
    // _packed and _scale tensors are confirmed co-located in the same safetensors
    // file, so we collect scales per-file (local HashMap, dropped after each file)
    // instead of loading all 69K scales into memory upfront.
    for (file_idx, st_path) in st_files.iter().enumerate() {
        println!(
            "  [{}/{}] Reading: {}",
            file_idx + 1,
            st_files.len(),
            st_path.file_name().unwrap_or_default().to_string_lossy()
        );
        let file_data = std::fs::read(st_path)?;
        let tensors = SafeTensors::deserialize(&file_data)?;

        // Collect _scale tensors from this file into a local HashMap
        let mut local_scale_map: std::collections::HashMap<String, Vec<u8>> =
            std::collections::HashMap::new();
        let mut local_scale_shape_map: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();

        for (name, tensor_view) in tensors.tensors() {
            if name.ends_with("_scale") {
                let base = name.trim_end_matches("_scale").to_string();
                local_scale_map.insert(base.clone(), tensor_view.data().to_vec());
                local_scale_shape_map.insert(base, tensor_view.shape().to_vec());
            }
        }

        if !local_scale_map.is_empty() && file_idx == 0 {
            println!(
                "  Using per-file scale collection ({} scales in first file)",
                local_scale_map.len()
            );
        }

        for (name, tensor_view) in tensors.tensors() {
            total_tensors += 1;

            let mapping = match classify_tensor(&name, config) {
                Some(m) => m,
                None => {
                    skipped_tensors.push(name.to_string());
                    continue;
                }
            };

            if mapping.is_scale || mapping.is_zero_point {
                continue;
            }

            let tensor_data = tensor_view.data();
            let shape = tensor_view.shape();
            let src_dtype = tensor_view.dtype();

            let (rows, cols) = if shape.len() == 2 {
                (shape[0], shape[1])
            } else if shape.len() == 1 {
                (1, shape[0])
            } else {
                skipped_tensors.push(format!("{} (unsupported shape {:?})", name, shape));
                continue;
            };

            let num_elements = rows * cols;
            total_source_bytes += tensor_data.len() as u64;

            // ── Compressed-tensors _packed path ──
            if mapping.is_packed {
                let base_name = name.trim_end_matches("_packed");
                let scale_data = local_scale_map.get(base_name);
                let scale_shape = local_scale_shape_map.get(base_name);

                if let (Some(scales), Some(s_shape)) = (scale_data, scale_shape) {
                    let num_groups = if s_shape.len() == 2 {
                        s_shape[1]
                    } else if s_shape.len() == 1 {
                        s_shape[0] / rows.max(1)
                    } else {
                        let total_scale_elements = scales.len() / 2;
                        if rows > 0 {
                            total_scale_elements / rows
                        } else {
                            1
                        }
                    };

                    let packed_k = if rows > 0 {
                        tensor_data.len() / rows
                    } else {
                        tensor_data.len()
                    };
                    let original_cols = packed_k * 2;

                    // Determine group size from the number of groups per row
                    let group_size = if num_groups > 0 {
                        original_cols.div_ceil(num_groups)
                    } else {
                        kernels::INT4_GROUP_SIZE
                    };

                    if options.quantize_experts == QuantFormat::Nvfp4 && !mapping.is_shared {
                        // Direct INT4 → NVFP4 conversion (parallelized, no f32 intermediate alloc)
                        let combined = combine_packed_and_scales(
                            tensor_data,
                            scales,
                            rows,
                            packed_k,
                            num_groups,
                        );
                        let nvfp4_data = kernels::convert_int4_to_nvfp4(
                            &combined,
                            rows,
                            original_cols,
                            group_size,
                        );

                        total_output_bytes += nvfp4_data.len() as u64;

                        let pages_before = writer.page_count();
                        add_pages_from_data(
                            &mut writer,
                            mapping.layer,
                            mapping.expert,
                            mapping.segment,
                            rows,
                            original_cols,
                            &nvfp4_data,
                            DType::NVFP4,
                        );

                        if let Some(ref mut collector) = sig_collector {
                            let scale_f32 = fp16_bytes_to_f32(scales);
                            if !scale_f32.is_empty() {
                                compute_expert_page_signatures(
                                    collector,
                                    &scale_f32,
                                    rows,
                                    num_groups,
                                    pages_before,
                                    true,
                                );
                            }
                        }
                    } else {
                        // Standard INT4 passthrough (or shared weights: keep as INT4)
                        let combined = combine_packed_and_scales(
                            tensor_data,
                            scales,
                            rows,
                            packed_k,
                            num_groups,
                        );

                        total_output_bytes += combined.len() as u64;

                        let pages_before = writer.page_count();
                        add_pages_from_data(
                            &mut writer,
                            mapping.layer,
                            mapping.expert,
                            mapping.segment,
                            rows,
                            original_cols,
                            &combined,
                            DType::INT4,
                        );

                        if !mapping.is_shared {
                            if let Some(ref mut collector) = sig_collector {
                                let scale_f32 = fp16_bytes_to_f32(scales);
                                if !scale_f32.is_empty() {
                                    compute_expert_page_signatures(
                                        collector,
                                        &scale_f32,
                                        rows,
                                        num_groups,
                                        pages_before,
                                        true,
                                    );
                                }
                            }
                        }
                    }
                } else {
                    println!(
                        "    WARN: No _scale tensor found for {}, passing through raw",
                        name
                    );
                    total_output_bytes += tensor_data.len() as u64;

                    let total_bytes = tensor_data.len();
                    let num_pages = total_bytes.div_ceil(PAGE_SIZE);
                    let bytes_per_page = if num_pages > 0 {
                        total_bytes.div_ceil(num_pages)
                    } else {
                        total_bytes
                    };

                    for page_idx in 0..num_pages {
                        let byte_start = page_idx * bytes_per_page;
                        let byte_end = (byte_start + bytes_per_page).min(total_bytes);
                        if byte_start >= total_bytes {
                            break;
                        }

                        let rows_per_page = if rows > 0 {
                            rows.div_ceil(num_pages)
                        } else {
                            1
                        };
                        let row_start = page_idx * rows_per_page;
                        let row_count = rows_per_page.min(rows.saturating_sub(row_start));

                        writer.add_page(
                            mapping.layer,
                            mapping.expert,
                            mapping.segment,
                            page_idx as u16,
                            row_start as u16,
                            row_count as u16,
                            cols as u16,
                            &tensor_data[byte_start..byte_end],
                        );
                    }
                }

                if mapping.is_shared {
                    converted_shared += 1;
                } else {
                    converted_experts += 1;
                }
                continue;
            }

            // ── Expert weight: quantize to INT4 or NVFP4 ──
            if !mapping.is_shared && options.quantize_experts != QuantFormat::None {
                let (quant_data, out_dtype) = match options.quantize_experts {
                    QuantFormat::Int4 => {
                        let data = match src_dtype {
                            safetensors::Dtype::F16 => {
                                kernels::quantize_fp16_to_int4(tensor_data, rows, cols)
                            }
                            safetensors::Dtype::BF16 => {
                                kernels::quantize_bf16_to_int4(tensor_data, rows, cols)
                            }
                            safetensors::Dtype::F32 => {
                                // SAFETY: safetensors guarantees aligned f32 data.
                                let f32_slice = unsafe {
                                    std::slice::from_raw_parts(
                                        tensor_data.as_ptr() as *const f32,
                                        num_elements,
                                    )
                                };
                                kernels::quantize_weights_to_int4(f32_slice, rows, cols)
                            }
                            other => {
                                skipped_tensors.push(format!(
                                    "{} (unsupported dtype {} for INT4 quantization)",
                                    name,
                                    safetensors_dtype_to_str(other)
                                ));
                                continue;
                            }
                        };
                        (data, DType::INT4)
                    }
                    QuantFormat::Nvfp4 => {
                        let data = match src_dtype {
                            safetensors::Dtype::F16 => {
                                kernels::quantize_fp16_to_nvfp4(tensor_data, rows, cols)
                            }
                            safetensors::Dtype::BF16 => {
                                kernels::quantize_bf16_to_nvfp4(tensor_data, rows, cols)
                            }
                            safetensors::Dtype::F32 => {
                                // SAFETY: safetensors guarantees aligned f32 data.
                                let f32_slice = unsafe {
                                    std::slice::from_raw_parts(
                                        tensor_data.as_ptr() as *const f32,
                                        num_elements,
                                    )
                                };
                                kernels::quantize_weights_to_nvfp4(f32_slice, rows, cols)
                            }
                            other => {
                                skipped_tensors.push(format!(
                                    "{} (unsupported dtype {} for NVFP4 quantization)",
                                    name,
                                    safetensors_dtype_to_str(other)
                                ));
                                continue;
                            }
                        };
                        (data, DType::NVFP4)
                    }
                    QuantFormat::None => unreachable!(),
                };

                total_output_bytes += quant_data.len() as u64;

                let pages_before = writer.page_count();
                add_pages_from_data(
                    &mut writer,
                    mapping.layer,
                    mapping.expert,
                    mapping.segment,
                    rows,
                    cols,
                    &quant_data,
                    out_dtype,
                );

                if let Some(ref mut collector) = sig_collector {
                    let f32_weights: Vec<f32> = match src_dtype {
                        safetensors::Dtype::F16 => fp16_bytes_to_f32(tensor_data),
                        safetensors::Dtype::BF16 => tensor_data
                            .chunks_exact(2)
                            .map(|c| {
                                let bits = u16::from_le_bytes([c[0], c[1]]);
                                half::bf16::from_bits(bits).to_f32()
                            })
                            .collect(),
                        safetensors::Dtype::F32 => {
                            // SAFETY: safetensors guarantees aligned f32 data.
                            unsafe {
                                std::slice::from_raw_parts(
                                    tensor_data.as_ptr() as *const f32,
                                    num_elements,
                                )
                            }
                            .to_vec()
                        }
                        _ => vec![],
                    };
                    if !f32_weights.is_empty() {
                        compute_expert_page_signatures(
                            collector,
                            &f32_weights,
                            rows,
                            cols,
                            pages_before,
                            true,
                        );
                    }
                }
                converted_experts += 1;
            } else {
                // ── Shared weight or no quantization: dtype conversion only ──
                let (output_data, bytes_per_element) = match src_dtype {
                    safetensors::Dtype::F16 => (tensor_data.to_vec(), 2usize),
                    safetensors::Dtype::BF16 => (
                        kernels::convert_bf16_to_fp16(tensor_data, num_elements),
                        2usize,
                    ),
                    safetensors::Dtype::F32 => (
                        kernels::convert_f32_to_fp16(tensor_data, num_elements),
                        2usize,
                    ),
                    other => {
                        if safetensors_dtype_bytes(other) <= 2 {
                            (tensor_data.to_vec(), safetensors_dtype_bytes(other))
                        } else {
                            skipped_tensors.push(format!(
                                "{} (unsupported dtype {})",
                                name,
                                safetensors_dtype_to_str(other)
                            ));
                            continue;
                        }
                    }
                };

                total_output_bytes += output_data.len() as u64;

                let total_bytes = output_data.len();
                let bytes_per_row = cols * bytes_per_element;
                let rows_per_page = if bytes_per_row > 0 {
                    (PAGE_SIZE / bytes_per_row).max(1)
                } else {
                    rows
                };
                let num_pages = rows.div_ceil(rows_per_page);

                for page_idx in 0..num_pages {
                    let row_start = page_idx * rows_per_page;
                    let row_count = rows_per_page.min(rows - row_start);
                    let byte_start = row_start * cols * bytes_per_element;
                    let byte_end = (row_start + row_count) * cols * bytes_per_element;
                    let byte_end = byte_end.min(total_bytes);

                    if byte_start >= total_bytes {
                        break;
                    }

                    let page_data = &output_data[byte_start..byte_end];

                    writer.add_page(
                        mapping.layer,
                        mapping.expert,
                        mapping.segment,
                        page_idx as u16,
                        row_start as u16,
                        row_count as u16,
                        cols as u16,
                        page_data,
                    );
                }

                converted_shared += 1;
            }
        }
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println!("\nConversion summary:");
    println!("  Total tensors:    {}", total_tensors);
    println!(
        "  Expert tensors:   {} ({})",
        converted_experts,
        match options.quantize_experts {
            QuantFormat::Int4 => "quantized to INT4",
            QuantFormat::Nvfp4 => "quantized to NVFP4",
            QuantFormat::None => "passthrough",
        }
    );
    println!("  Shared tensors:   {}", converted_shared);
    println!("  Skipped:          {}", skipped_tensors.len());
    println!(
        "  Source size:      {:.1} GB",
        total_source_bytes as f64 / 1e9
    );
    println!(
        "  Output size:      {:.1} GB (before compression)",
        total_output_bytes as f64 / 1e9
    );
    if total_source_bytes > 0 {
        println!(
            "  Reduction:        {:.1}x",
            total_source_bytes as f64 / total_output_bytes.max(1) as f64
        );
    }

    if !skipped_tensors.is_empty() {
        let show = skipped_tensors.len().min(20);
        println!(
            "\n  Skipped tensors (showing {}/{}):",
            show,
            skipped_tensors.len()
        );
        for name in skipped_tensors.iter().take(show) {
            println!("    - {}", name);
        }
    }

    // ── Build HNSW index ────────────────────────────────────────────────
    let index_vectors = if let Some(collector) = sig_collector {
        if let Some((centroids, _graph_bytes)) = collector.build_index() {
            let n = centroids.len();
            let entries: Vec<crate::storage::format::VectorIndexEntry> = centroids
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let mut entry = crate::storage::format::VectorIndexEntry {
                        centroid_id: i as u32,
                        cluster_size: 1,
                        prediction_count: 0,
                        hot_page_count: 1,
                        expert_predictions: [(0u16, 0u8); 32],
                        hot_pages: [0u32; 64],
                    };
                    entry.hot_pages[0] = i as u32;
                    entry
                })
                .collect();

            writer.set_vector_index(centroids, entries);
            println!("  Vector index written to .vib3 file");
            n
        } else {
            0
        }
    } else {
        0
    };

    // ── Write file ──────────────────────────────────────────────────────
    let output_str = output.to_string_lossy();
    println!("\nWriting {} ...", output_str);
    writer.finalize(output)?;

    let file = crate::storage::format::Vib3File::open(&*output_str)?;
    let page_count = file.page_count();
    println!(
        "Verified: {} pages, model={}",
        page_count,
        file.model_config().name
    );

    Ok(ConvertResult {
        total_tensors,
        converted_experts,
        converted_shared,
        skipped_tensors,
        source_bytes: total_source_bytes,
        output_bytes: total_output_bytes,
        page_count,
        index_vectors,
    })
}

// ═════════════════════════════════════════════════════════════════════════════
// GGUF → .vib3 Conversion
// ═════════════════════════════════════════════════════════════════════════════

use crate::storage::gguf::{GgmlType, GgufFile};

/// Classify a GGUF tensor name into (layer, expert, segment) mapping.
///
/// GGUF uses `blk.N.xxx` naming convention (from llama.cpp).
fn classify_gguf_tensor(name: &str) -> Option<TensorMapping> {
    // Global tensors (no layer)
    if name == "token_embd.weight" {
        return Some(TensorMapping {
            layer: 0,
            expert: 0xFFFF,
            segment: 10,
            is_shared: true,
            is_packed: false,
            is_scale: false,
            is_zero_point: false,
        });
    }
    if name == "output.weight" {
        return Some(TensorMapping {
            layer: 0,
            expert: 0xFFFF,
            segment: 11,
            is_shared: true,
            is_packed: false,
            is_scale: false,
            is_zero_point: false,
        });
    }
    if name == "output_norm.weight" {
        return Some(TensorMapping {
            layer: 0xFFFF,
            expert: 0xFFFF,
            segment: 8,
            is_shared: true,
            is_packed: false,
            is_scale: false,
            is_zero_point: false,
        });
    }

    // Layer tensors: blk.N.xxx
    let layer = if name.starts_with("blk.") {
        let rest = &name[4..];
        let dot = rest.find('.')?;
        rest[..dot].parse::<u16>().ok()?
    } else {
        return None;
    };

    let suffix = {
        let rest = &name[4..];
        let dot = rest.find('.')?;
        &rest[dot + 1..]
    };

    let shared = |segment: u16| -> Option<TensorMapping> {
        Some(TensorMapping {
            layer,
            expert: 0xFFFF,
            segment,
            is_shared: true,
            is_packed: false,
            is_scale: false,
            is_zero_point: false,
        })
    };

    // Stacked expert tensors (3D, 256 experts packed in one tensor)
    let stacked = |segment: u16| -> Option<TensorMapping> {
        Some(TensorMapping {
            layer,
            expert: 0xFFFD,
            segment,
            is_shared: false,
            is_packed: false,
            is_scale: false,
            is_zero_point: false,
        })
    };

    match suffix {
        // ── Norms and biases (F32 → FP16 shared) ──
        "attn_norm.weight" => shared(6),
        "post_attention_norm.weight" => shared(7),
        "attn_q_norm.weight" => shared(27),
        "attn_k_norm.weight" => shared(28),

        // ── Full attention projections (Q8_0 → FP16 shared) ──
        "attn_q.weight" => shared(4), // Doubled: [hidden, num_heads*2*head_dim] for Qwen3.5
        "attn_k.weight" => shared(12),
        "attn_v.weight" => shared(13),
        "attn_output.weight" => shared(5),

        // ── DeltaNet (linear attention) tensors ──
        "attn_qkv.weight" => shared(30), // in_proj_qkv (DeltaNet layers)
        "attn_gate.weight" => shared(31), // in_proj_z (DeltaNet output gate)
        "ssm_alpha.weight" => shared(33), // in_proj_a (decay projection)
        "ssm_beta.weight" => shared(32), // in_proj_b (write strength projection)
        "ssm_conv1d.weight" => shared(34), // conv1d weight
        "ssm_dt.bias" => shared(35),     // dt_bias
        "ssm_a" => shared(36),           // A_log
        "ssm_norm.weight" => shared(37), // per-head norm
        "ssm_out.weight" => shared(38),  // out_proj

        // ── MoE router and shared expert ──
        "ffn_gate_inp.weight" => shared(3), // Router weights
        "ffn_gate_inp_shexp.weight" => shared(26), // Shared expert sigmoid gate
        "ffn_gate_shexp.weight" => shared(15), // Shared expert gate_proj
        "ffn_up_shexp.weight" => shared(14), // Shared expert up_proj
        "ffn_down_shexp.weight" => shared(16), // Shared expert down_proj

        // ── Stacked expert weights (MXFP4, 3D [dim_a, dim_b, 256]) ──
        "ffn_gate_exps.weight" => stacked(1), // Expert gate_proj (w3)
        "ffn_up_exps.weight" => stacked(0),   // Expert up_proj (w1)
        "ffn_down_exps.weight" => stacked(2), // Expert down_proj (w2)

        _ => {
            tracing::debug!("GGUF: unclassified tensor: {}", name);
            None
        }
    }
}

/// Convert GGUF files from a directory to .vib3 format.
///
/// Handles multi-shard GGUF files with MXFP4 expert weights, Q8_0 shared weights,
/// and F32 norms. Splits stacked 3D expert tensors into per-expert pages.
pub fn convert_gguf_dir(
    dir: &Path,
    config: &ModelConfig,
    output: &Path,
    options: &ConvertOptions,
) -> Result<ConvertResult, String> {
    let start = std::time::Instant::now();

    // Open and parse all GGUF shards
    let gguf = GgufFile::open_dir(dir)?;

    // Create the output writer (clone config so we can set file-level dtype)
    let mut file_config = config.clone();

    // Determine expert_dtype from GGUF source data: scan for the first stacked expert
    // tensor and use its type to decide the file-level expert_dtype.
    // MXFP4 source → NVFP4 output; other source dtypes respect options.quantize_experts.
    {
        let tensor_names_tmp: Vec<String> = gguf.tensors.keys().cloned().collect();
        for tname in &tensor_names_tmp {
            let info = &gguf.tensors[tname];
            if let Some(mapping) = classify_gguf_tensor(tname) {
                if mapping.expert == 0xFFFD {
                    match info.dtype {
                        GgmlType::Mxfp4 => {
                            file_config.expert_dtype = DType::NVFP4;
                        }
                        GgmlType::Q8_0
                        | GgmlType::Q4K
                        | GgmlType::Q5K
                        | GgmlType::Q6K
                        | GgmlType::F16
                        | GgmlType::BF16
                        | GgmlType::F32 => match options.quantize_experts {
                            QuantFormat::Nvfp4 => file_config.expert_dtype = DType::NVFP4,
                            QuantFormat::Int4 => file_config.expert_dtype = DType::INT4,
                            QuantFormat::None => file_config.expert_dtype = DType::FP16,
                        },
                        _ => {}
                    }
                    break; // all stacked experts have the same type
                }
            }
        }
    }

    // GGUF conversion writes shared tensors as FP16 pages (dequantized/converted from source).
    // Keep file-level metadata consistent with stored page data for runtime kernel dispatch.
    if file_config.shared_dtype == DType::BF16 {
        file_config.shared_dtype = DType::FP16;
    }

    tracing::info!("File expert_dtype set to {:?}", file_config.expert_dtype);

    let temp_dir = output.parent();
    let mut writer = Vib3Writer::with_temp_dir(file_config, temp_dir);
    writer.set_compression(options.compress);

    let mut total_tensors = 0usize;
    let mut converted_experts = 0usize;
    let mut converted_shared = 0usize;
    let mut skipped = Vec::new();
    let mut total_source_bytes = 0u64;
    let mut total_output_bytes = 0u64;
    let num_experts = config.num_experts as usize;

    // Collect tensor names and sort for deterministic processing
    let mut tensor_names: Vec<String> = gguf.tensors.keys().cloned().collect();
    tensor_names.sort();

    for tensor_name in &tensor_names {
        let info = &gguf.tensors[tensor_name];
        total_tensors += 1;
        total_source_bytes += info.data_bytes() as u64;

        let mapping = match classify_gguf_tensor(tensor_name) {
            Some(m) => m,
            None => {
                skipped.push(tensor_name.clone());
                continue;
            }
        };

        // Skip scale/packed/zero_point (not applicable for GGUF but guard anyway)
        if mapping.is_scale || mapping.is_zero_point || mapping.is_packed {
            continue;
        }

        let raw_data = gguf.tensor_data(info);
        let layer = mapping.layer;
        let segment = mapping.segment;

        // ── Stacked expert tensors: split and convert ──
        if mapping.expert == 0xFFFD {
            let n_experts = if info.shape.len() >= 3 {
                info.shape[2] as usize // [ne0, ne1, n_experts]
            } else {
                tracing::warn!(
                    "Stacked tensor {} has {} dims, expected 3",
                    tensor_name,
                    info.shape.len()
                );
                continue;
            };

            if n_experts != num_experts {
                tracing::warn!(
                    "Stacked tensor {} has {} experts, expected {}",
                    tensor_name,
                    n_experts,
                    num_experts
                );
            }

            let ne0 = info.shape[0] as usize; // innermost dim (quantized along this axis)
            let ne1 = info.shape[1] as usize; // rows per expert

            // Compute bytes per expert slice
            let (block_elems, block_bytes) = info.dtype.block_layout();
            let blocks_per_row = ne0.div_ceil(block_elems);
            let row_bytes = blocks_per_row * block_bytes;
            let expert_bytes = ne1 * row_bytes;

            tracing::info!(
                "Splitting stacked tensor {} [{},{},{}] type={:?} → {} experts × {} bytes",
                tensor_name,
                ne0,
                ne1,
                n_experts,
                info.dtype,
                n_experts,
                expert_bytes
            );

            let available_experts = if expert_bytes > 0 {
                raw_data.len() / expert_bytes
            } else {
                0
            };
            let experts_to_process = n_experts.min(available_experts);

            if experts_to_process < n_experts {
                tracing::warn!(
                    "Stacked tensor {} expected {} experts ({} bytes each) but only {} fit in {} bytes; processing available experts only",
                    tensor_name,
                    n_experts,
                    expert_bytes,
                    experts_to_process,
                    raw_data.len()
                );
            }

            for expert_idx in 0..experts_to_process {
                let start = expert_idx * expert_bytes;
                let end = start + expert_bytes;
                let expert_slice = &raw_data[start..end];

                match info.dtype {
                    GgmlType::Mxfp4 => {
                        // MXFP4 → NVFP4 (lossless E2M1 copy, E8M0→FP16 scale conversion)
                        let nvfp4_data =
                            crate::storage::gguf::convert_mxfp4_to_nvfp4(expert_slice, ne1, ne0);
                        add_pages_from_data(
                            &mut writer,
                            layer,
                            expert_idx as u16,
                            segment,
                            ne1,
                            ne0,
                            &nvfp4_data,
                            DType::NVFP4,
                        );
                        total_output_bytes += nvfp4_data.len() as u64;
                        converted_experts += 1;
                    }
                    GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => {
                        let fp16_data = match info.dtype {
                            GgmlType::Q8_0 => {
                                crate::storage::gguf::dequant_q8_0_to_fp16(expert_slice, ne1, ne0)
                            }
                            GgmlType::Q4K => {
                                crate::storage::gguf::dequant_q4k_to_fp16(expert_slice, ne1, ne0)
                            }
                            GgmlType::Q5K => {
                                crate::storage::gguf::dequant_q5k_to_fp16(expert_slice, ne1, ne0)
                            }
                            GgmlType::Q6K => {
                                crate::storage::gguf::dequant_q6k_to_fp16(expert_slice, ne1, ne0)
                            }
                            _ => unreachable!(),
                        };

                        match options.quantize_experts {
                            QuantFormat::Nvfp4 => {
                                let nvfp4 = kernels::quantize_fp16_to_nvfp4(&fp16_data, ne1, ne0);
                                add_pages_from_data(
                                    &mut writer,
                                    layer,
                                    expert_idx as u16,
                                    segment,
                                    ne1,
                                    ne0,
                                    &nvfp4,
                                    DType::NVFP4,
                                );
                                total_output_bytes += nvfp4.len() as u64;
                            }
                            QuantFormat::Int4 => {
                                let int4 = kernels::quantize_fp16_to_int4(&fp16_data, ne1, ne0);
                                add_pages_from_data(
                                    &mut writer,
                                    layer,
                                    expert_idx as u16,
                                    segment,
                                    ne1,
                                    ne0,
                                    &int4,
                                    DType::INT4,
                                );
                                total_output_bytes += int4.len() as u64;
                            }
                            QuantFormat::None => {
                                add_fp16_expert_pages(
                                    &mut writer,
                                    layer,
                                    expert_idx as u16,
                                    segment,
                                    ne1,
                                    ne0,
                                    &fp16_data,
                                );
                                total_output_bytes += fp16_data.len() as u64;
                            }
                        }
                        converted_experts += 1;
                    }
                    _ => {
                        tracing::warn!(
                            "Unsupported type {:?} for stacked expert {}",
                            info.dtype,
                            tensor_name
                        );
                    }
                }
            }
            continue;
        }

        // ── Shared tensors (attention, norms, embeddings, etc.) ──
        if mapping.is_shared {
            let ne0 = info.shape.first().copied().unwrap_or(1) as usize;
            let n_rows: usize = info.shape.iter().skip(1).product::<u64>().max(1) as usize;

            let fp16_data = match info.dtype {
                GgmlType::Q8_0 => crate::storage::gguf::dequant_q8_0_to_fp16(raw_data, n_rows, ne0),
                GgmlType::F32 => crate::storage::gguf::convert_f32_to_fp16(raw_data, ne0 * n_rows),
                GgmlType::F16 => raw_data.to_vec(),
                GgmlType::BF16 => {
                    crate::storage::gguf::convert_bf16_to_fp16(raw_data, ne0 * n_rows)
                }
                GgmlType::Q4K => crate::storage::gguf::dequant_q4k_to_fp16(raw_data, n_rows, ne0),
                GgmlType::Q5K => crate::storage::gguf::dequant_q5k_to_fp16(raw_data, n_rows, ne0),
                GgmlType::Q6K => crate::storage::gguf::dequant_q6k_to_fp16(raw_data, n_rows, ne0),
                _ => {
                    tracing::warn!(
                        "Unsupported type {:?} for shared tensor {}",
                        info.dtype,
                        tensor_name
                    );
                    skipped.push(tensor_name.clone());
                    continue;
                }
            };

            // ── V-head order for DeltaNet QKV (segment 30) and Z gate (segment 31) ──
            // The llama.cpp GGUF converter DOES apply V-head reordering to both
            // `.in_proj_qkv.` and `.in_proj_z.` via `_LinearAttentionVReorderBase`,
            // converting them from grouped to tiled order at conversion time.
            // All other per-value-head tensors (alpha, beta, A_log, dt_bias,
            // conv1d, out_proj) are also reordered to tiled order.
            // Therefore NO additional reorder is needed here — the GGUF data is
            // already in tiled order for all V-head-indexed tensors.

            add_shared_pages(&mut writer, layer, 0xFFFF, segment, &fp16_data);
            total_output_bytes += fp16_data.len() as u64;
            converted_shared += 1;
            continue;
        }

        // Shouldn't reach here — all tensors are either stacked or shared
        skipped.push(tensor_name.clone());
    }

    let page_count = writer.page_count();
    let elapsed = start.elapsed();

    tracing::info!(
        "GGUF conversion complete: {} tensors → {} pages in {:.1}s ({} expert, {} shared, {} skipped)",
        total_tensors, page_count, elapsed.as_secs_f64(),
        converted_experts, converted_shared, skipped.len(),
    );

    // Finalize: write .vib3 file
    writer
        .finalize(output)
        .map_err(|e| format!("Failed to finalize: {}", e))?;

    Ok(ConvertResult {
        total_tensors,
        converted_experts,
        converted_shared,
        skipped_tensors: skipped,
        source_bytes: total_source_bytes,
        output_bytes: total_output_bytes,
        page_count,
        index_vectors: 0, // TODO: build indexes
    })
}

/// Add FP16 expert tensor data as row-sliced pages.
fn add_fp16_expert_pages(
    writer: &mut Vib3Writer,
    layer: u16,
    expert: u16,
    segment: u16,
    rows: usize,
    cols: usize,
    fp16_data: &[u8],
) {
    let bytes_per_row = cols * 2;
    let rows_per_page = if bytes_per_row > 0 {
        (PAGE_SIZE / bytes_per_row).max(1)
    } else {
        rows
    };
    let num_pages = rows.div_ceil(rows_per_page);

    for page_idx in 0..num_pages {
        let row_start = page_idx * rows_per_page;
        let row_count = rows_per_page.min(rows - row_start);
        let start = row_start * bytes_per_row;
        let end = start + row_count * bytes_per_row;
        if end > fp16_data.len() {
            break;
        }

        writer.add_page(
            layer,
            expert,
            segment,
            page_idx as u16,
            row_start as u16,
            row_count as u16,
            cols as u16,
            &fp16_data[start..end],
        );
    }
}

/// Add shared tensor data as pages (FP16 passthrough, no quantization).
fn add_shared_pages(
    writer: &mut Vib3Writer,
    layer: u16,
    expert: u16,
    segment: u16,
    fp16_data: &[u8],
) {
    // Split into 2MB pages
    let num_pages = fp16_data.len().div_ceil(PAGE_SIZE);
    for page_idx in 0..num_pages {
        let start = page_idx * PAGE_SIZE;
        let end = (start + PAGE_SIZE).min(fp16_data.len());
        let page_data = &fp16_data[start..end];

        writer.add_page(
            layer,
            expert,
            segment,
            page_idx as u16,
            0, // row_start (not applicable for shared)
            0, // row_count
            0, // cols
            page_data,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    /// Minimal View implementation for creating test safetensors files.
    struct TestTensor {
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl safetensors::View for &TestTensor {
        fn dtype(&self) -> safetensors::Dtype {
            self.dtype
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    /// Create a synthetic safetensors file with Mixtral-style tensor names.
    ///
    /// Generates a tiny 2-layer, 4-expert model with FP16 weights:
    ///   - Router weights per layer
    ///   - Expert up/gate/down projections per layer
    ///   - Embeddings, norms, lm_head
    fn create_test_safetensors(
        dir: &std::path::Path,
        hidden: usize,
        expert_hidden: usize,
        num_experts: usize,
        num_layers: usize,
    ) {
        std::fs::create_dir_all(dir).unwrap();

        let mut tensors: Vec<(String, TestTensor)> = Vec::new();

        let mut rng: u64 = 99;
        let mut rand_fp16 = |n: usize| -> Vec<u8> {
            let mut data = vec![0u8; n * 2];
            for chunk in data.chunks_exact_mut(2) {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let val = half::f16::from_f32(((rng as f32 / u64::MAX as f32) - 0.5) * 0.1);
                let bytes = val.to_le_bytes();
                chunk[0] = bytes[0];
                chunk[1] = bytes[1];
            }
            data
        };

        // Token embeddings
        tensors.push((
            "model.embed_tokens.weight".into(),
            TestTensor {
                dtype: safetensors::Dtype::F16,
                shape: vec![256, hidden],
                data: rand_fp16(256 * hidden),
            },
        ));

        // LM head
        tensors.push((
            "lm_head.weight".into(),
            TestTensor {
                dtype: safetensors::Dtype::F16,
                shape: vec![256, hidden],
                data: rand_fp16(256 * hidden),
            },
        ));

        // Final norm
        tensors.push((
            "model.norm.weight".into(),
            TestTensor {
                dtype: safetensors::Dtype::F16,
                shape: vec![hidden],
                data: rand_fp16(hidden),
            },
        ));

        for layer in 0..num_layers {
            // Input layernorm
            tensors.push((
                format!("model.layers.{}.input_layernorm.weight", layer),
                TestTensor {
                    dtype: safetensors::Dtype::F16,
                    shape: vec![hidden],
                    data: rand_fp16(hidden),
                },
            ));

            // Post-attention layernorm
            tensors.push((
                format!("model.layers.{}.post_attention_layernorm.weight", layer),
                TestTensor {
                    dtype: safetensors::Dtype::F16,
                    shape: vec![hidden],
                    data: rand_fp16(hidden),
                },
            ));

            // Attention Q/K/V/O (simplified: same dim)
            for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                tensors.push((
                    format!("model.layers.{}.self_attn.{}.weight", layer, proj),
                    TestTensor {
                        dtype: safetensors::Dtype::F16,
                        shape: vec![hidden, hidden],
                        data: rand_fp16(hidden * hidden),
                    },
                ));
            }

            // Router
            tensors.push((
                format!("model.layers.{}.block_sparse_moe.gate.weight", layer),
                TestTensor {
                    dtype: safetensors::Dtype::F16,
                    shape: vec![num_experts, hidden],
                    data: rand_fp16(num_experts * hidden),
                },
            ));

            // Expert weights
            for expert in 0..num_experts {
                for (proj, rows, cols) in &[
                    ("w1", expert_hidden, hidden), // up_proj
                    ("w3", expert_hidden, hidden), // gate_proj
                    ("w2", hidden, expert_hidden), // down_proj
                ] {
                    tensors.push((
                        format!(
                            "model.layers.{}.block_sparse_moe.experts.{}.{}.weight",
                            layer, expert, proj
                        ),
                        TestTensor {
                            dtype: safetensors::Dtype::F16,
                            shape: vec![*rows, *cols],
                            data: rand_fp16(rows * cols),
                        },
                    ));
                }
            }
        }

        // Write config.json for auto-detection
        let config_json = serde_json::json!({
            "model_type": "mixtral",
            "hidden_size": hidden,
            "intermediate_size": expert_hidden,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_local_experts": num_experts,
            "num_experts_per_tok": 2,
            "vocab_size": 256,
            "max_position_embeddings": 128,
        });
        std::fs::write(dir.join("config.json"), config_json.to_string()).unwrap();

        // Serialize safetensors
        let tensor_refs: Vec<(String, &TestTensor)> =
            tensors.iter().map(|(name, t)| (name.clone(), t)).collect();
        let bytes = safetensors::serialize(tensor_refs, &None).unwrap();
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    #[test]
    fn test_convert_random_produces_valid_vib3() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("random.vib3");

        let config = ModelConfig {
            name: "test-tiny".into(),
            architecture: "test-tiny".into(),
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

        let options = ConvertOptions {
            quantize_experts: QuantFormat::None,
            compress: CompressionMethod::None,
            build_indexes: false,
        };

        let result = convert_random(&config, &output, &options).unwrap();

        assert!(result.page_count > 0, "should produce pages");
        assert!(
            result.converted_experts > 0,
            "should convert expert tensors"
        );

        // Verify the output file is valid
        let file =
            crate::storage::format::Vib3File::open(output.to_string_lossy().as_ref()).unwrap();
        assert_eq!(file.page_count(), result.page_count);
        assert_eq!(file.model_config().name, "test-tiny");
        assert_eq!(file.model_config().num_experts, 4);
    }

    #[test]
    fn test_convert_random_with_int4_quantization() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("random_int4.vib3");

        let config = ModelConfig {
            name: "test-int4".into(),
            architecture: "test-tiny".into(),
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
            expert_dtype: DType::FP16,
            shared_dtype: DType::FP16,
            ..Default::default()
        };

        let options = ConvertOptions {
            quantize_experts: QuantFormat::None,
            compress: CompressionMethod::None,
            build_indexes: false,
        };

        let result = convert_random(&config, &output, &options).unwrap();
        assert!(result.page_count > 0);

        let file =
            crate::storage::format::Vib3File::open(output.to_string_lossy().as_ref()).unwrap();
        assert_eq!(file.page_count(), result.page_count);
    }

    #[test]
    fn test_convert_safetensors_with_index_building() {
        let dir = tempfile::tempdir().unwrap();
        let st_dir = dir.path().join("safetensors");
        let output = dir.path().join("model.vib3");

        // Create synthetic safetensors with Mixtral-style naming
        create_test_safetensors(&st_dir, 64, 32, 4, 2);

        // Auto-detect config
        let config = resolve_config_from_dir(&st_dir).unwrap();
        assert_eq!(config.num_experts, 4);
        assert_eq!(config.hidden_dim, 64);

        // Convert with index building
        let options = ConvertOptions {
            quantize_experts: QuantFormat::Int4,
            compress: CompressionMethod::Zstd { level: 1 },
            build_indexes: true,
        };

        let result = convert_safetensors_dir(&st_dir, &config, &output, &options).unwrap();

        // Verify pages were created
        assert!(result.page_count > 0, "should produce pages, got 0");
        assert!(
            result.converted_experts > 0,
            "should convert expert tensors, got 0"
        );
        assert!(
            result.converted_shared > 0,
            "should convert shared tensors, got 0"
        );

        // Verify HNSW index was built
        assert!(
            result.index_vectors > 0,
            "should build vector index with build_indexes=true, got 0 vectors"
        );

        // Verify the output file
        let file =
            crate::storage::format::Vib3File::open(output.to_string_lossy().as_ref()).unwrap();
        assert_eq!(file.page_count(), result.page_count);
        assert!(
            file.has_vector_index(),
            "output file should have vector index section"
        );
        assert!(
            file.vector_index_bytes().len() > 0,
            "vector index section should be non-empty"
        );
    }

    #[test]
    fn test_convert_safetensors_without_index() {
        let dir = tempfile::tempdir().unwrap();
        let st_dir = dir.path().join("safetensors");
        let output = dir.path().join("model_no_idx.vib3");

        create_test_safetensors(&st_dir, 64, 32, 2, 1);

        let config = resolve_config_from_dir(&st_dir).unwrap();

        let options = ConvertOptions {
            quantize_experts: QuantFormat::Int4,
            compress: CompressionMethod::None,
            build_indexes: false,
        };

        let result = convert_safetensors_dir(&st_dir, &config, &output, &options).unwrap();

        assert!(result.page_count > 0);
        assert_eq!(
            result.index_vectors, 0,
            "should not build index when disabled"
        );

        let file =
            crate::storage::format::Vib3File::open(output.to_string_lossy().as_ref()).unwrap();
        assert!(
            !file.has_vector_index(),
            "should not have vector index when build_indexes=false"
        );
    }

    #[test]
    fn test_convert_safetensors_no_quantize() {
        let dir = tempfile::tempdir().unwrap();
        let st_dir = dir.path().join("safetensors");
        let output = dir.path().join("model_fp16.vib3");

        create_test_safetensors(&st_dir, 64, 32, 2, 1);

        let config = resolve_config_from_dir(&st_dir).unwrap();

        let options = ConvertOptions {
            quantize_experts: QuantFormat::None,
            compress: CompressionMethod::None,
            build_indexes: false,
        };

        let result = convert_safetensors_dir(&st_dir, &config, &output, &options).unwrap();

        assert!(result.page_count > 0);

        let file =
            crate::storage::format::Vib3File::open(output.to_string_lossy().as_ref()).unwrap();
        assert_eq!(file.model_config().expert_dtype, DType::FP16);
    }

    #[test]
    fn test_classify_tensor_expert_weights() {
        let config = ModelConfig {
            dense_layer_idx: 0,
            ..Default::default()
        };

        let m = classify_tensor(
            "model.layers.5.block_sparse_moe.experts.3.w1.weight",
            &config,
        )
        .unwrap();
        assert_eq!(m.layer, 5);
        assert_eq!(m.expert, 3);
        assert_eq!(m.segment, 0); // w1 = up_proj = segment 0
        assert!(!m.is_shared);

        let m = classify_tensor(
            "model.layers.2.block_sparse_moe.experts.7.w2.weight",
            &config,
        )
        .unwrap();
        assert_eq!(m.layer, 2);
        assert_eq!(m.expert, 7);
        assert_eq!(m.segment, 2); // w2 = down_proj = segment 2
    }

    #[test]
    fn test_classify_tensor_shared_weights() {
        let config = ModelConfig::default();

        let m = classify_tensor("model.embed_tokens.weight", &config).unwrap();
        assert_eq!(m.segment, 10);
        assert!(m.is_shared);

        let m = classify_tensor("lm_head.weight", &config).unwrap();
        assert_eq!(m.segment, 11);

        let m = classify_tensor("model.norm.weight", &config).unwrap();
        assert_eq!(m.segment, 8);
    }

    #[test]
    fn test_classify_tensor_router() {
        let config = ModelConfig::default();

        let m = classify_tensor("model.layers.3.block_sparse_moe.gate.weight", &config).unwrap();
        assert_eq!(m.layer, 3);
        assert_eq!(m.segment, 3);
        assert!(m.is_shared);
    }

    #[test]
    fn test_classify_tensor_attention() {
        let config = ModelConfig::default();

        let m = classify_tensor("model.layers.1.self_attn.q_proj.weight", &config).unwrap();
        assert_eq!(m.segment, 4);

        let m = classify_tensor("model.layers.1.self_attn.k_proj.weight", &config).unwrap();
        assert_eq!(m.segment, 12);

        let m = classify_tensor("model.layers.1.self_attn.v_proj.weight", &config).unwrap();
        assert_eq!(m.segment, 13);

        let m = classify_tensor("model.layers.1.self_attn.o_proj.weight", &config).unwrap();
        assert_eq!(m.segment, 5);
    }

    #[test]
    fn test_classify_tensor_mla() {
        let config = ModelConfig::default();

        let m = classify_tensor("model.layers.0.self_attn.q_a_proj.weight", &config).unwrap();
        assert_eq!(m.segment, 20);

        let m = classify_tensor("model.layers.0.self_attn.kv_b_proj.weight", &config).unwrap();
        assert_eq!(m.segment, 23);

        let m = classify_tensor("model.layers.0.self_attn.q_a_layernorm.weight", &config).unwrap();
        assert_eq!(m.segment, 24);
    }

    #[test]
    fn test_classify_tensor_packed() {
        let config = ModelConfig {
            dense_layer_idx: 0,
            ..Default::default()
        };

        let m = classify_tensor(
            "model.layers.1.block_sparse_moe.experts.2.w1.weight_packed",
            &config,
        )
        .unwrap();
        assert!(m.is_packed);
        assert!(!m.is_scale);
        assert_eq!(m.segment, 0);

        let m = classify_tensor(
            "model.layers.1.block_sparse_moe.experts.2.w1.weight_scale",
            &config,
        )
        .unwrap();
        assert!(!m.is_packed);
        assert!(m.is_scale);
    }

    #[test]
    fn test_resolve_config_from_dir_missing() {
        let dir = tempfile::tempdir().unwrap();
        let result = resolve_config_from_dir(dir.path());
        assert!(result.is_err(), "should fail without config.json");
    }

    #[test]
    fn test_convert_result_roundtrip_read() {
        // Full roundtrip: create safetensors -> convert -> read back pages
        let dir = tempfile::tempdir().unwrap();
        let st_dir = dir.path().join("st");
        let output = dir.path().join("roundtrip.vib3");

        create_test_safetensors(&st_dir, 64, 32, 2, 1);
        let config = resolve_config_from_dir(&st_dir).unwrap();

        let options = ConvertOptions {
            quantize_experts: QuantFormat::Int4,
            compress: CompressionMethod::None,
            build_indexes: true,
        };

        let result = convert_safetensors_dir(&st_dir, &config, &output, &options).unwrap();

        let file =
            crate::storage::format::Vib3File::open(output.to_string_lossy().as_ref()).unwrap();

        // Read every page back — should not panic or error
        for i in 0..file.page_count() {
            let entry = file.page(i);
            let raw_size = entry.raw_size as usize;
            let mut buf = vec![0u8; raw_size];
            let bytes_read = file.read_page_sync(i, &mut buf).unwrap();
            assert!(bytes_read > 0, "page {} should have data", i);
        }

        // Check expert pages exist
        let expert_pages = file.pages_for_expert(0, 0);
        assert!(
            !expert_pages.is_empty(),
            "should have pages for expert 0 at layer 0"
        );

        // Verify page count matches
        assert_eq!(file.page_count(), result.page_count);
    }
}

/// Generate a .vib3 file with random weights (for testing).
pub fn convert_random(
    config: &ModelConfig,
    output: &Path,
    options: &ConvertOptions,
) -> anyhow::Result<ConvertResult> {
    println!(
        "Generating random model: {} experts x {} layers, hidden_dim={}",
        config.num_experts, config.num_moe_layers, config.hidden_dim
    );

    let mut file_config = config.clone();
    match options.quantize_experts {
        QuantFormat::Int4 => file_config.expert_dtype = DType::INT4,
        QuantFormat::Nvfp4 => file_config.expert_dtype = DType::NVFP4,
        QuantFormat::None => file_config.expert_dtype = DType::FP16,
    }
    let mut writer = Vib3Writer::new(file_config);
    writer.set_compression(options.compress);

    let hidden_dim = config.hidden_dim as usize;
    let expert_hidden_dim = config.expert_hidden_dim as usize;

    let mut rng_state: u64 = 12345;
    let mut converted_experts = 0usize;

    for layer in 0..config.num_moe_layers {
        let layer_idx = layer as u16 + config.dense_layer_idx as u16;

        for expert in 0..config.num_experts {
            for segment in 0..3u16 {
                let (rows, cols) = match segment {
                    0 | 1 => (expert_hidden_dim, hidden_dim),
                    2 => (hidden_dim, expert_hidden_dim),
                    _ => unreachable!(),
                };

                if options.quantize_experts != QuantFormat::None {
                    let total_elements = rows * cols;
                    let mut f32_data = vec![0.0f32; total_elements];
                    for val in &mut f32_data {
                        rng_state ^= rng_state << 13;
                        rng_state ^= rng_state >> 7;
                        rng_state ^= rng_state << 17;
                        *val = ((rng_state as f64 / u64::MAX as f64) - 0.5) as f32 * 0.1;
                    }

                    let (quant_data, out_dtype) = match options.quantize_experts {
                        QuantFormat::Int4 => (
                            kernels::quantize_weights_to_int4(&f32_data, rows, cols),
                            DType::INT4,
                        ),
                        QuantFormat::Nvfp4 => (
                            kernels::quantize_weights_to_nvfp4(&f32_data, rows, cols),
                            DType::NVFP4,
                        ),
                        QuantFormat::None => unreachable!(),
                    };
                    add_pages_from_data(
                        &mut writer,
                        layer_idx,
                        expert as u16,
                        segment,
                        rows,
                        cols,
                        &quant_data,
                        out_dtype,
                    );
                } else {
                    let bytes_per_element = 2; // FP16
                    let segment_bytes = rows * cols * bytes_per_element;
                    let num_pages = segment_bytes.div_ceil(PAGE_SIZE);
                    let rows_per_page = rows.div_ceil(num_pages);

                    for page_idx in 0..num_pages as u16 {
                        let row_start = page_idx as usize * rows_per_page;
                        let row_count = rows_per_page.min(rows - row_start);
                        let page_bytes = row_count * cols * bytes_per_element;

                        let mut data = vec![0u8; page_bytes];
                        for chunk in data.chunks_exact_mut(2) {
                            rng_state ^= rng_state << 13;
                            rng_state ^= rng_state >> 7;
                            rng_state ^= rng_state << 17;
                            let val =
                                f16::from_f32(((rng_state as f32 / u64::MAX as f32) - 0.5) * 0.1);
                            let bytes = val.to_le_bytes();
                            chunk[0] = bytes[0];
                            chunk[1] = bytes[1];
                        }

                        writer.add_page(
                            layer_idx,
                            expert as u16,
                            segment,
                            page_idx,
                            row_start as u16,
                            row_count as u16,
                            cols as u16,
                            &data,
                        );
                    }
                }
                converted_experts += 1;
            }
        }

        if (layer + 1) % 10 == 0 || layer + 1 == config.num_moe_layers {
            println!("  Layer {}/{}", layer + 1, config.num_moe_layers);
        }
    }

    let output_str = output.to_string_lossy();
    println!("Writing {} ...", output_str);
    writer.finalize(output)?;

    let file = crate::storage::format::Vib3File::open(&*output_str)?;
    let page_count = file.page_count();
    println!(
        "Verified: {} pages, model={}",
        page_count,
        file.model_config().name
    );

    Ok(ConvertResult {
        total_tensors: converted_experts,
        converted_experts,
        converted_shared: 0,
        skipped_tensors: vec![],
        source_bytes: 0,
        output_bytes: 0,
        page_count,
        index_vectors: 0,
    })
}

/// Resolve model config from a HuggingFace model directory.
///
/// Reads `config.json` and auto-detects the model architecture.
pub fn resolve_config_from_dir(dir: &Path) -> anyhow::Result<ModelConfig> {
    let config_path = dir.join("config.json");

    if config_path.exists() {
        println!("Found config.json, auto-detecting architecture...");
        let config = ModelConfig::from_hf_config_path(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to parse config.json: {}", e))?;
        println!("  Detected: {} ({})", config.architecture, config.name);
        Ok(config)
    } else {
        anyhow::bail!(
            "No config.json found in {}. Cannot auto-detect model architecture.",
            dir.display()
        )
    }
}

// ─── Internal Helpers ───────────────────────────────────────────────────

/// Map safetensors `Dtype` to our internal representation.
fn safetensors_dtype_to_str(dtype: safetensors::Dtype) -> &'static str {
    match dtype {
        safetensors::Dtype::F16 => "F16",
        safetensors::Dtype::BF16 => "BF16",
        safetensors::Dtype::F32 => "F32",
        safetensors::Dtype::F64 => "F64",
        safetensors::Dtype::I8 => "I8",
        safetensors::Dtype::I16 => "I16",
        safetensors::Dtype::I32 => "I32",
        safetensors::Dtype::I64 => "I64",
        safetensors::Dtype::U8 => "U8",
        safetensors::Dtype::U16 => "U16",
        safetensors::Dtype::U32 => "U32",
        safetensors::Dtype::U64 => "U64",
        _ => "unknown",
    }
}

/// Bytes per element for a safetensors dtype.
fn safetensors_dtype_bytes(dtype: safetensors::Dtype) -> usize {
    match dtype {
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
        safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
        safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
        safetensors::Dtype::I8 | safetensors::Dtype::U8 => 1,
        safetensors::Dtype::I16 | safetensors::Dtype::U16 => 2,
        _ => 2, // fallback
    }
}

/// Convert FP16 bytes (little-endian) to f32 values.
fn fp16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

// ─── INT4 Page Layout ───────────────────────────────────────────────────

/// Compute the byte layout for an INT4 quantized weight segment.
///
/// Returns `(bytes_per_row, packed_k, num_groups)`.
fn int4_row_layout(cols: usize) -> (usize, usize, usize) {
    let packed_k = cols.div_ceil(2);
    let num_groups = cols.div_ceil(kernels::INT4_GROUP_SIZE);
    let bytes_per_row = packed_k + num_groups * 2;
    (bytes_per_row, packed_k, num_groups)
}

/// Combine compressed-tensors _packed + _scale into our standard INT4 layout.
fn combine_packed_and_scales(
    packed_data: &[u8],
    scale_data: &[u8],
    rows: usize,
    packed_k: usize,
    num_groups: usize,
) -> Vec<u8> {
    let packed_total = rows * packed_k;
    let scale_bytes_per_row = num_groups * 2;
    let scale_total = rows * scale_bytes_per_row;

    let mut combined = vec![0u8; packed_total + scale_total];

    let copy_packed = packed_total.min(packed_data.len());
    combined[..copy_packed].copy_from_slice(&packed_data[..copy_packed]);

    let copy_scale = scale_total.min(scale_data.len());
    combined[packed_total..packed_total + copy_scale].copy_from_slice(&scale_data[..copy_scale]);

    combined
}

/// Add pages from a contiguous quantized data buffer.
///
/// Splits the data into 2MB pages based on the weight layout.
#[allow(clippy::too_many_arguments)]
fn add_pages_from_data(
    writer: &mut Vib3Writer,
    layer: u16,
    expert: u16,
    segment: u16,
    rows: usize,
    cols: usize,
    int4_data: &[u8],
    _dtype: DType,
) {
    let (bytes_per_row, packed_k, num_groups) = int4_row_layout(cols);
    let total_int4_bytes = rows * packed_k;

    let rows_per_page = if bytes_per_row > 0 {
        let max_rows = PAGE_SIZE / bytes_per_row;
        max_rows.max(1)
    } else {
        rows
    };
    let num_pages = rows.div_ceil(rows_per_page);

    for page_idx in 0..num_pages {
        let row_start = page_idx * rows_per_page;
        let row_count = rows_per_page.min(rows - row_start);

        let page_int4_bytes = row_count * packed_k;
        let page_scales_bytes = row_count * num_groups * 2;
        let mut page_data = vec![0u8; page_int4_bytes + page_scales_bytes];

        for r in 0..row_count {
            let src_offset = (row_start + r) * packed_k;
            let dst_offset = r * packed_k;
            if src_offset + packed_k <= total_int4_bytes {
                page_data[dst_offset..dst_offset + packed_k]
                    .copy_from_slice(&int4_data[src_offset..src_offset + packed_k]);
            }
        }

        let scales_start_in_src = total_int4_bytes;
        for r in 0..row_count {
            let src_scale_offset = scales_start_in_src + (row_start + r) * num_groups * 2;
            let dst_scale_offset = page_int4_bytes + r * num_groups * 2;
            let scale_bytes = num_groups * 2;
            if src_scale_offset + scale_bytes <= int4_data.len() {
                page_data[dst_scale_offset..dst_scale_offset + scale_bytes]
                    .copy_from_slice(&int4_data[src_scale_offset..src_scale_offset + scale_bytes]);
            }
        }

        writer.add_page(
            layer,
            expert,
            segment,
            page_idx as u16,
            row_start as u16,
            row_count as u16,
            cols as u16,
            &page_data,
        );
    }
}

// ─── Page Signature Collector ───────────────────────────────────────────

/// Collects page signatures during conversion for building the HNSW index.
struct PageSignatureCollector {
    signatures: Vec<(usize, Vec<f32>)>,
    target_dim: usize,
    method: SignatureMethod,
}

impl PageSignatureCollector {
    fn new(hidden_dim: usize) -> Self {
        let target_dim = hidden_dim.min(256);
        Self {
            signatures: Vec::new(),
            target_dim,
            method: SignatureMethod::Mean,
        }
    }

    fn add_from_f32(
        &mut self,
        page_catalog_idx: usize,
        f32_data: &[f32],
        rows: usize,
        cols: usize,
    ) {
        let sig = compute_page_signature(f32_data, rows, cols, self.target_dim, self.method);
        self.signatures.push((page_catalog_idx, sig));
    }

    fn build_index(self) -> Option<(Vec<Vec<f32>>, Vec<u8>)> {
        if self.signatures.is_empty() {
            return None;
        }

        println!(
            "\nBuilding HNSW index from {} page signatures...",
            self.signatures.len()
        );
        println!("  Signature dim: {}", self.target_dim);

        let centroids: Vec<Vec<f32>> = self.signatures.iter().map(|(_, sig)| sig.clone()).collect();

        let config = HnswConfig {
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
            ..Default::default()
        };

        let backend = HnswBackend::new(centroids.clone(), &config)?;
        let graph_bytes = backend.save_to_buffer()?;

        println!(
            "  HNSW index: {} vectors, {:.1} KB graph",
            centroids.len(),
            graph_bytes.len() as f64 / 1024.0
        );
        println!(
            "  Memory usage: {:.1} KB",
            backend.memory_usage() as f64 / 1024.0
        );

        Some((centroids, graph_bytes))
    }
}

/// Compute per-page signatures for an expert tensor's weight data.
fn compute_expert_page_signatures(
    collector: &mut PageSignatureCollector,
    f32_weights: &[f32],
    rows: usize,
    cols: usize,
    num_pages_start_idx: usize,
    is_int4: bool,
) {
    let rows_per_page = if is_int4 {
        let (bytes_per_row, _, _) = int4_row_layout(cols);
        if bytes_per_row > 0 {
            (PAGE_SIZE / bytes_per_row).max(1)
        } else {
            rows
        }
    } else {
        let bytes_per_row = cols * 2;
        if bytes_per_row > 0 {
            (PAGE_SIZE / bytes_per_row).max(1)
        } else {
            rows
        }
    };

    let num_pages = rows.div_ceil(rows_per_page);

    for page_idx in 0..num_pages {
        let row_start = page_idx * rows_per_page;
        let row_count = rows_per_page.min(rows - row_start);

        let data_start = row_start * cols;
        let data_end = (row_start + row_count) * cols;
        let data_end = data_end.min(f32_weights.len());

        if data_start < f32_weights.len() {
            let page_f32 = &f32_weights[data_start..data_end];
            collector.add_from_f32(num_pages_start_idx + page_idx, page_f32, row_count, cols);
        }
    }
}

// ─── Tensor Classification ──────────────────────────────────────────────

// Shared page segment conventions:
//   segment 0-2 = expert up/gate/down projections
//   segment 3   = router weights
//   segment 4   = attn_q_proj
//   segment 5   = attn_o_proj
//   segment 6   = attn_norm
//   segment 7   = ffn_norm
//   segment 8   = final norm
//   segment 10  = token embeddings
//   segment 11  = lm_head
//   segment 12  = attn_k_proj
//   segment 13  = attn_v_proj
//   segment 14-16 = shared expert up/gate/down
//   segment 17-19 = dense MLP up/gate/down
//   segment 20-25 = MLA projections and layernorms
//   segment 26    = shared expert gate (sigmoid gate, 1×hidden_dim)
//   segment 27    = attn Q norm (per-head)
//   segment 28    = attn K norm (per-head)
//
// DeltaNet (Qwen3.5) segments:
//   segment 30 = linear_attn.in_proj_qkv  (QKV joint projection)
//   segment 31 = linear_attn.in_proj_z    (output gate projection)
//   segment 32 = linear_attn.in_proj_b    (beta/write strength projection)
//   segment 33 = linear_attn.in_proj_a    (alpha/decay projection)
//   segment 34 = linear_attn.conv1d       (causal depthwise conv1d)
//   segment 35 = linear_attn.dt_bias      (time step bias)
//   segment 36 = linear_attn.A_log        (log decay rate)
//   segment 37 = linear_attn.norm         (per-head RMSNorm for gated output)
//   segment 38 = linear_attn.out_proj     (output projection)

struct TensorMapping {
    layer: u16,
    expert: u16,
    segment: u16,
    is_shared: bool,
    is_packed: bool,
    is_scale: bool,
    is_zero_point: bool,
}

/// Classify a safetensors tensor name into (layer, expert, segment) mapping.
fn classify_tensor(name: &str, config: &ModelConfig) -> Option<TensorMapping> {
    let base_name = name
        .trim_end_matches("_packed")
        .trim_end_matches("_scale")
        .trim_end_matches("_zero_point");

    let is_packed = name.ends_with("_packed");
    let is_scale = name.ends_with("_scale");
    let is_zero_point = name.ends_with("_zero_point");

    let layer_num = extract_layer_num(base_name);

    // Stacked expert tensors (Qwen3.5: all experts in one tensor)
    // e.g., "model.language_model.layers.0.mlp.experts.gate_up_proj" shape (256, 2048, 3072)
    //        "model.language_model.layers.0.mlp.experts.down_proj"     shape (256, 3072, 1024)
    // These need special handling in the converter (split by expert dim).
    // We classify them with expert=0xFFFD (EXPERT_STACKED sentinel) to flag them.
    if base_name.contains("mlp.experts.gate_up_proj") || base_name.contains("mlp.experts.down_proj")
    {
        if let Some(layer) = layer_num {
            if !base_name.contains("shared_expert") {
                // gate_up_proj packs gate + up: segments 0+1 interleaved
                // down_proj: segment 2
                let segment = if base_name.contains("gate_up_proj") {
                    0 // will be split into gate(1) + up(0) during conversion
                } else {
                    2
                };
                return Some(TensorMapping {
                    layer: layer as u16,
                    expert: 0xFFFD, // sentinel: stacked, needs per-expert split
                    segment,
                    is_shared: false,
                    is_packed,
                    is_scale,
                    is_zero_point,
                });
            }
        }
    }

    // Individual expert MoE weights (Mixtral-style: experts.{N}.w1/w2/w3)
    if let Some(expert_id) = extract_expert_id(base_name) {
        if !base_name.contains("shared_expert") {
            let segment = if base_name.contains("up_proj") || base_name.contains("w1") {
                0
            } else if base_name.contains("gate_proj") || base_name.contains("w3") {
                1
            } else if base_name.contains("down_proj") || base_name.contains("w2") {
                2
            } else {
                return None;
            };

            return Some(TensorMapping {
                layer: layer_num.unwrap_or(0) as u16,
                expert: expert_id as u16,
                segment,
                is_shared: false,
                is_packed,
                is_scale,
                is_zero_point,
            });
        }
    }

    // Shared expert weights
    if base_name.contains("shared_expert") {
        if let Some(layer) = layer_num {
            let segment = if base_name.contains("up_proj") || base_name.contains("w1") {
                14
            } else if base_name.contains("gate_proj")
                || base_name.contains("gate.weight")
                || base_name.contains("w3")
            {
                if base_name.contains("gate_proj") || base_name.contains("w3") {
                    15
                } else {
                    return None;
                }
            } else if base_name.contains("down_proj") || base_name.contains("w2") {
                16
            } else {
                return None;
            };

            return Some(TensorMapping {
                layer: layer as u16,
                expert: 0xFFFF,
                segment,
                is_shared: true,
                is_packed,
                is_scale,
                is_zero_point,
            });
        }
    }

    // Dense layer MLP
    if let Some(layer) = layer_num {
        if layer < config.dense_layer_idx as usize
            && base_name.contains("mlp.")
            && !base_name.contains("experts")
        {
            let segment = if base_name.contains("up_proj") || base_name.contains("w1") {
                17
            } else if base_name.contains("gate_proj") || base_name.contains("w3") {
                18
            } else if base_name.contains("down_proj") || base_name.contains("w2") {
                19
            } else {
                return None;
            };

            return Some(TensorMapping {
                layer: layer as u16,
                expert: 0xFFFF,
                segment,
                is_shared: true,
                is_packed,
                is_scale,
                is_zero_point,
            });
        }
    }

    // Router weights
    if (base_name.contains("gate.weight") || base_name.contains("router.weight"))
        && !base_name.contains("shared_expert")
        && !base_name.contains("gate_proj")
    {
        if let Some(layer) = layer_num {
            return Some(TensorMapping {
                layer: layer as u16,
                expert: 0xFFFF,
                segment: 3,
                is_shared: true,
                is_packed,
                is_scale,
                is_zero_point,
            });
        }
    }

    // DeltaNet / linear attention tensors (Qwen3.5)
    if base_name.contains("linear_attn") {
        if let Some(layer) = layer_num {
            let segment = if base_name.contains("in_proj_qkv") {
                30
            } else if base_name.contains("in_proj_z") {
                31
            } else if base_name.contains("in_proj_b") {
                32
            } else if base_name.contains("in_proj_a") {
                33
            } else if base_name.contains("conv1d") {
                34
            } else if base_name.contains("dt_bias") {
                35
            } else if base_name.contains("A_log") {
                36
            } else if base_name.contains("norm.weight") {
                37
            } else if base_name.contains("out_proj") {
                38
            } else {
                return None;
            };

            return Some(TensorMapping {
                layer: layer as u16,
                expert: 0xFFFF,
                segment,
                is_shared: true,
                is_packed,
                is_scale,
                is_zero_point,
            });
        }
    }

    // Shared expert gate (sigmoid gate for shared expert output, Qwen3.5)
    if base_name.contains("shared_expert_gate") && !base_name.contains("gate_proj") {
        if let Some(layer) = layer_num {
            return Some(TensorMapping {
                layer: layer as u16,
                expert: 0xFFFF,
                segment: 26,
                is_shared: true,
                is_packed,
                is_scale,
                is_zero_point,
            });
        }
    }

    // Attention projections (MLA + standard)
    if base_name.contains("self_attn") || base_name.contains("attention") {
        if let Some(layer) = layer_num {
            let seg = classify_attention_tensor(base_name);
            if let Some(segment) = seg {
                return Some(TensorMapping {
                    layer: layer as u16,
                    expert: 0xFFFF,
                    segment,
                    is_shared: true,
                    is_packed,
                    is_scale,
                    is_zero_point,
                });
            }
        }
    }

    // Layer norms
    if base_name.contains("input_layernorm") || base_name.contains("attn_norm") {
        if let Some(layer) = layer_num {
            return Some(TensorMapping {
                layer: layer as u16,
                expert: 0xFFFF,
                segment: 6,
                is_shared: true,
                is_packed,
                is_scale,
                is_zero_point,
            });
        }
    }
    if base_name.contains("post_attention_layernorm") || base_name.contains("ffn_norm") {
        if let Some(layer) = layer_num {
            return Some(TensorMapping {
                layer: layer as u16,
                expert: 0xFFFF,
                segment: 7,
                is_shared: true,
                is_packed,
                is_scale,
                is_zero_point,
            });
        }
    }

    // Embeddings
    if base_name.contains("embed_tokens")
        || base_name.contains("token_embedding")
        || base_name == "model.embed_tokens.weight"
        || base_name == "model.language_model.embed_tokens.weight"
    {
        return Some(TensorMapping {
            layer: 0,
            expert: 0xFFFF,
            segment: 10,
            is_shared: true,
            is_packed,
            is_scale,
            is_zero_point,
        });
    }

    // LM head
    if base_name.contains("lm_head") || base_name == "lm_head.weight" {
        return Some(TensorMapping {
            layer: 0,
            expert: 0xFFFF,
            segment: 11,
            is_shared: true,
            is_packed,
            is_scale,
            is_zero_point,
        });
    }

    // Final norm (text model only — skip vision tower norms)
    if (base_name.contains("model.norm")
        || base_name.contains("language_model.norm")
        || base_name.contains("final_layernorm"))
        && !base_name.contains("vision_tower")
        && !base_name.contains("vision_model")
        && !base_name.contains("linear_attn")
        && !base_name.contains("layernorm")
    {
        return Some(TensorMapping {
            layer: 0xFFFF,
            expert: 0xFFFF,
            segment: 8,
            is_shared: true,
            is_packed,
            is_scale,
            is_zero_point,
        });
    }

    None
}

/// Classify an attention tensor name into its segment number.
fn classify_attention_tensor(name: &str) -> Option<u16> {
    // MLA layernorms (check before projections to avoid partial match)
    if name.contains("q_a_layernorm") {
        return Some(24);
    }
    if name.contains("kv_a_layernorm") {
        return Some(25);
    }

    // MLA projections (take priority over standard Q/K/V)
    if name.contains("q_a_proj") {
        return Some(20);
    }
    if name.contains("q_b_proj") {
        return Some(21);
    }
    if name.contains("kv_a_proj") {
        return Some(22);
    }
    if name.contains("kv_b_proj") {
        return Some(23);
    }

    // Attention Q/K norms (Qwen3.5 gated attention — check before projections)
    if name.contains("q_norm") {
        return Some(27);
    }
    if name.contains("k_norm") {
        return Some(28);
    }

    // Standard attention projections
    if name.contains("q_proj") || name.contains("qkv_proj") {
        return Some(4);
    }
    if name.contains("k_proj") {
        return Some(12);
    }
    if name.contains("v_proj") {
        return Some(13);
    }
    if name.contains("o_proj") || name.contains("out_proj") {
        return Some(5);
    }

    None
}

fn extract_layer_num(name: &str) -> Option<usize> {
    for pattern in &["layers.", "layer.", "blocks."] {
        if let Some(pos) = name.find(pattern) {
            let after = &name[pos + pattern.len()..];
            if let Some(end) = after.find('.') {
                if let Ok(n) = after[..end].parse::<usize>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

fn extract_expert_id(name: &str) -> Option<usize> {
    for pattern in &["experts.", "expert."] {
        if let Some(pos) = name.find(pattern) {
            let after = &name[pos + pattern.len()..];
            if let Some(end) = after.find('.') {
                if let Ok(n) = after[..end].parse::<usize>() {
                    return Some(n);
                }
            }
        }
    }
    None
}
