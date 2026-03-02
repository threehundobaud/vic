//! `vib3-convert` — CLI wrapper for safetensors → .vib3 conversion.
//!
//! All conversion logic lives in `vib3::storage::convert`. This binary
//! just parses CLI arguments and calls into the library.
//!
//! Usage:
//!   vib3-convert --model /path/to/model --output model.vib3
//!   vib3-convert --model /path/to/model --output model.vib3 --compress zstd
//!   vib3-convert --model random --output test.vib3 --arch test-tiny
//!   vib3-convert --model random --output test.vib3 --arch test-tiny --quantize int4

use clap::Parser;
use std::path::PathBuf;
use vib3::compute::kernels;
use vib3::core::config::ModelConfig;
use vib3::core::types::DType;
use vib3::storage::convert::{
    convert_gguf_dir, convert_random, convert_safetensors_dir, ConvertOptions, QuantFormat,
};
use vib3::storage::format::CompressionMethod;

#[derive(Parser)]
#[command(name = "vib3-convert", about = "Convert models to .vib3 format")]
struct Args {
    /// Input model path (safetensors/HF directory) or "random" for test weights
    #[arg(long)]
    model: String,

    /// Output .vib3 file path
    #[arg(long)]
    output: String,

    /// Model architecture (auto-detected from config.json if available)
    #[arg(long, default_value = "auto")]
    arch: String,

    /// Number of experts (for test-tiny arch)
    #[arg(long, default_value = "8")]
    num_experts: u32,

    /// Number of MoE layers (for test-tiny arch)
    #[arg(long, default_value = "4")]
    num_layers: u32,

    /// Hidden dimension (for test-tiny arch)
    #[arg(long, default_value = "256")]
    hidden_dim: u32,

    /// Expert hidden dimension (for test-tiny arch)
    #[arg(long, default_value = "64")]
    expert_hidden_dim: u32,

    /// Quantize expert weights: none, int4, nvfp4 (default: int4 for real models, none for random)
    #[arg(long, default_value = "auto")]
    quantize: String,

    /// Compression: none, lz4, zstd (default: zstd for quantized, none for random)
    #[arg(long, default_value = "auto")]
    compress: String,

    /// Zstd compression level (1-19, default 3)
    #[arg(long, default_value = "3")]
    zstd_level: i32,

    /// Build vector index from page signatures
    #[arg(long)]
    build_indexes: bool,

    /// Build materialized views
    #[arg(long)]
    build_views: bool,

    /// Verbose
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(if args.verbose {
            "vib3=trace"
        } else {
            "vib3=info"
        })
        .init();

    println!("vib3-convert v0.2.0");
    println!("Input:  {}", args.model);
    println!("Output: {}", args.output);

    let output = PathBuf::from(&args.output);

    match args.model.as_str() {
        "random" => {
            let config = resolve_config_manual(&args);
            println!("Arch:   {}", config.architecture);
            let quantize = match args.quantize.as_str() {
                "int4" => QuantFormat::Int4,
                "nvfp4" => QuantFormat::Nvfp4,
                "none" => QuantFormat::None,
                _ => QuantFormat::None, // auto for random = no quantization
            };
            let options = ConvertOptions {
                quantize_experts: quantize,
                compress: resolve_compression(&args, quantize != QuantFormat::None),
                build_indexes: args.build_indexes,
            };
            convert_random(&config, &output, &options)?;
        }
        path => {
            let p = std::path::Path::new(path);
            let (config, dir, is_gguf) = if p.is_dir() {
                // Check if directory contains .gguf files
                let has_gguf = std::fs::read_dir(p)
                    .map(|entries| {
                        entries
                            .filter_map(|e| e.ok())
                            .any(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
                    })
                    .unwrap_or(false);

                if has_gguf {
                    let config = resolve_config_for_gguf(p, &args)?;
                    (config, p.to_path_buf(), true)
                } else {
                    let config = resolve_config_from_dir(p, &args)?;
                    (config, p.to_path_buf(), false)
                }
            } else if path.ends_with(".gguf") {
                let dir = p.parent().unwrap_or(std::path::Path::new("."));
                let config = resolve_config_for_gguf(dir, &args)?;
                (config, dir.to_path_buf(), true)
            } else if path.ends_with(".safetensors") {
                let dir = p.parent().unwrap_or(std::path::Path::new("."));
                let config = resolve_config_from_dir(dir, &args)?;
                (config, dir.to_path_buf(), false)
            } else {
                anyhow::bail!(
                    "Unsupported input format. Provide a .safetensors file, .gguf file, HF directory, or 'random'"
                );
            };

            println!("Arch:   {} ({})", config.architecture, config.name);
            println!(
                "Model:  {} layers, {} experts (top-{}), hidden={}, expert_hidden={}",
                config.num_layers,
                config.num_experts,
                config.num_active_experts,
                config.hidden_dim,
                config.expert_hidden_dim
            );
            if is_gguf {
                println!("Format: GGUF");
            }

            let quantize = match args.quantize.as_str() {
                "int4" => QuantFormat::Int4,
                "nvfp4" => QuantFormat::Nvfp4,
                "none" => QuantFormat::None,
                _ => {
                    if is_gguf {
                        QuantFormat::None // GGUF MXFP4 → NVFP4 is already handled internally
                    } else {
                        QuantFormat::Int4 // auto for safetensors = quantize to INT4
                    }
                }
            };
            let options = ConvertOptions {
                quantize_experts: quantize,
                compress: resolve_compression(&args, quantize != QuantFormat::None),
                build_indexes: args.build_indexes,
            };

            println!(
                "Quantize experts: {}",
                match quantize {
                    QuantFormat::Int4 => format!("INT4 (group-{})", kernels::INT4_GROUP_SIZE),
                    QuantFormat::Nvfp4 =>
                        format!("NVFP4/E2M1 (block-{})", kernels::NVFP4_BLOCK_SIZE),
                    QuantFormat::None => "none (passthrough/native)".to_string(),
                }
            );
            println!("Compression:      {:?}", options.compress);

            if is_gguf {
                convert_gguf_dir(&dir, &config, &output, &options)
                    .map_err(|e| anyhow::anyhow!(e))?;
            } else {
                convert_safetensors_dir(&dir, &config, &output, &options)?;
            }
        }
    }

    println!("Done!");
    Ok(())
}

// ─── CLI-specific helpers ───────────────────────────────────────────────

fn resolve_compression(args: &Args, quantize: bool) -> CompressionMethod {
    match args.compress.as_str() {
        "none" => CompressionMethod::None,
        "lz4" => CompressionMethod::Lz4,
        "zstd" => CompressionMethod::Zstd {
            level: args.zstd_level,
        },
        _ => {
            if quantize {
                CompressionMethod::Zstd {
                    level: args.zstd_level,
                }
            } else {
                CompressionMethod::None
            }
        }
    }
}

fn resolve_config_manual(args: &Args) -> ModelConfig {
    match args.arch.as_str() {
        "kimi-k2.5" => ModelConfig::kimi_k25(),
        _ => ModelConfig {
            name: "test-tiny".into(),
            architecture: "test-tiny".into(),
            hidden_dim: args.hidden_dim,
            expert_hidden_dim: args.expert_hidden_dim,
            num_layers: args.num_layers + 1,
            num_moe_layers: args.num_layers,
            dense_layer_idx: 0,
            num_experts: args.num_experts,
            num_active_experts: 2.min(args.num_experts),
            num_heads: 4,
            num_kv_heads: 2,
            max_seq_len: 2048,
            vocab_size: 1024,
            expert_dtype: DType::FP16,
            shared_dtype: DType::FP16,
            ..Default::default()
        },
    }
}

fn resolve_config_from_dir(dir: &std::path::Path, args: &Args) -> anyhow::Result<ModelConfig> {
    let config_path = dir.join("config.json");

    if config_path.exists() && (args.arch == "auto" || args.arch.is_empty()) {
        // Delegate to library auto-detect
        vib3::storage::convert::resolve_config_from_dir(dir)
    } else {
        match args.arch.as_str() {
            "kimi-k2.5" => Ok(ModelConfig::kimi_k25()),
            "auto" => {
                anyhow::bail!(
                    "No config.json found in {} and no --arch specified. \
                     Use --arch kimi-k2.5 or --arch test-tiny.",
                    dir.display()
                );
            }
            _ => Ok(resolve_config_manual(args)),
        }
    }
}

/// Resolve model config for GGUF input.
///
/// Tries to read architecture from GGUF metadata (`general.architecture`),
/// then falls back to `--arch` flag. For Qwen3.5-122B we use our hardcoded
/// config since GGUF metadata may not have all the details we need.
fn resolve_config_for_gguf(dir: &std::path::Path, args: &Args) -> anyhow::Result<ModelConfig> {
    // If --arch is explicitly specified, use that
    if args.arch != "auto" && !args.arch.is_empty() {
        return match args.arch.as_str() {
            "qwen3.5-122b" | "qwen35-122b" | "qwen3_5_moe" => Ok(ModelConfig::qwen35_122b()),
            "qwen3.5-35b" | "qwen35-35b" => Ok(ModelConfig::qwen35_35b()),
            "kimi-k2.5" => Ok(ModelConfig::kimi_k25()),
            other => anyhow::bail!("Unknown architecture '{}' for GGUF conversion", other),
        };
    }

    // Try to auto-detect from GGUF metadata
    let gguf = vib3::storage::gguf::GgufFile::open_dir(dir)
        .map_err(|e| anyhow::anyhow!("Failed to open GGUF: {}", e))?;

    if let Some(arch) = gguf.get_metadata("general.architecture") {
        let arch_str = format!("{:?}", arch);
        println!("GGUF architecture: {}", arch_str);

        // Match known architectures
        if arch_str.contains("qwen3_5_moe")
            || arch_str.contains("qwen3moe")
            || arch_str.contains("qwen35moe")
        {
            // Determine architecture prefix for metadata keys
            let arch_prefix = if arch_str.contains("qwen35moe") {
                "qwen35moe"
            } else if arch_str.contains("qwen3_5_moe") {
                "qwen3_5_moe"
            } else {
                "qwen3moe"
            };

            // Distinguish 122B vs 35B by embedding_length (hidden_dim)
            let emb_key = format!("{}.embedding_length", arch_prefix);
            if let Some(emb_len) = gguf.get_metadata(&emb_key).and_then(|v| v.as_u32()) {
                if emb_len <= 2048 {
                    println!("Auto-detected Qwen3.5-35B-A3B (hidden_dim={})", emb_len);
                    return Ok(ModelConfig::qwen35_35b());
                } else {
                    println!("Auto-detected Qwen3.5-122B-A10B (hidden_dim={})", emb_len);
                    return Ok(ModelConfig::qwen35_122b());
                }
            }
            // Fallback: check block_count
            let blk_key = format!("{}.block_count", arch_prefix);
            if let Some(n_layers) = gguf.get_metadata(&blk_key).and_then(|v| v.as_u32()) {
                if n_layers <= 40 {
                    println!("Auto-detected Qwen3.5-35B-A3B (layers={})", n_layers);
                    return Ok(ModelConfig::qwen35_35b());
                } else {
                    println!("Auto-detected Qwen3.5-122B-A10B (layers={})", n_layers);
                    return Ok(ModelConfig::qwen35_122b());
                }
            }
            // Final fallback: default to 122B
            return Ok(ModelConfig::qwen35_122b());
        }
    }

    // Check file names as fallback heuristic
    let has_qwen35 = std::fs::read_dir(dir)
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                let name = e.file_name().to_string_lossy().to_lowercase();
                name.contains("qwen3.5") || name.contains("qwen3_5")
            })
        })
        .unwrap_or(false);

    if has_qwen35 {
        // Try to distinguish by filename
        let has_35b = std::fs::read_dir(dir)
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    let name = e.file_name().to_string_lossy().to_lowercase();
                    name.contains("35b")
                })
            })
            .unwrap_or(false);
        if has_35b {
            println!("Auto-detected Qwen3.5-35B-A3B from filename");
            return Ok(ModelConfig::qwen35_35b());
        }
        println!("Auto-detected Qwen3.5 from filename");
        return Ok(ModelConfig::qwen35_122b());
    }

    anyhow::bail!(
        "Could not auto-detect model architecture from GGUF metadata in {}. \
         Use --arch qwen3.5-122b, --arch qwen3.5-35b, or similar.",
        dir.display()
    )
}
