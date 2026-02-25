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
    convert_random, convert_safetensors_dir, ConvertOptions, QuantFormat,
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
            let (config, dir) = if p.is_dir() {
                let config = resolve_config_from_dir(p, &args)?;
                (config, p.to_path_buf())
            } else if path.ends_with(".safetensors") {
                let dir = p.parent().unwrap_or(std::path::Path::new("."));
                let config = resolve_config_from_dir(dir, &args)?;
                (config, dir.to_path_buf())
            } else {
                anyhow::bail!(
                    "Unsupported input format. Provide a .safetensors file, HF directory, or 'random'"
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

            let quantize = match args.quantize.as_str() {
                "int4" => QuantFormat::Int4,
                "nvfp4" => QuantFormat::Nvfp4,
                "none" => QuantFormat::None,
                _ => QuantFormat::Int4, // auto for real models = quantize to INT4
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
                    QuantFormat::None => "none (passthrough)".to_string(),
                }
            );
            println!("Compression:      {:?}", options.compress);

            convert_safetensors_dir(&dir, &config, &output, &options)?;
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
