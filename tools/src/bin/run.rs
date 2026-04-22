//! `vib3` — the main CLI entry point.
//!
//! Usage:
//!   vib3 run kimi-k2.5              # Download + convert + run (like ollama)
//!   vib3 run kimi-k2.5 --variant fp8
//!   vib3 pull kimi-k2.5             # Download + convert only
//!   vib3 list                       # List local models
//!   vib3 rm kimi-k2.5               # Delete a model
//!   vib3 serve kimi-k2.5            # API server mode
//!   vib3 info kimi-k2.5             # Show model info + hardware check
//!   vib3 hw                         # Show detected hardware
//!   vib3 convert --model /path --output model.vib3

use clap::{Parser, Subcommand};
use std::io::Write;
use vib3::core::config::EngineConfig;
use vib3::registry::format_bytes;
use vib3::registry::*;
use vib3::storage::convert::{
    convert_random, convert_safetensors_dir, resolve_config_from_dir, ConvertOptions, QuantFormat,
};
use vib3::Engine;

#[derive(Parser)]
#[command(
    name = "vib3",
    version,
    about = "Weight-indexed inference engine for MoE models",
    long_about = "Run frontier-class MoE models on a single GPU.\n\n\
                  vib3 run kimi-k2.5    — download and run interactively\n\
                  vib3 serve kimi-k2.5  — start an API server"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download (if needed) and run a model interactively
    Run {
        /// Model name (e.g., kimi-k2.5)
        model: String,
        /// Quantization variant (e.g., int4, fp8)
        #[arg(long)]
        variant: Option<String>,
        /// Override VRAM budget (MB)
        #[arg(long)]
        vram: Option<usize>,
        /// Override RAM budget (MB)
        #[arg(long)]
        ram: Option<usize>,
        /// Maximum tokens to generate per response
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy / deterministic)
        #[arg(long, default_value = "0.0")]
        temperature: f32,
        /// Top-k sampling cutoff (used when temperature > 0)
        #[arg(long, default_value = "50")]
        top_k: usize,
        /// Top-p (nucleus) sampling cutoff (used when temperature > 0)
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        /// Skip hardware check
        #[arg(long)]
        no_check: bool,
        /// Path to tokenizer.json (auto-detected if not specified)
        #[arg(long)]
        tokenizer: Option<String>,
        /// Wrap user input in the model's chat template (role markers +
        /// assistant prefix) before tokenizing. When off (the default),
        /// prompts are sent raw — continue-the-text style. Chat mode is
        /// what produces real "ask a question / get an answer" behaviour
        /// from instruction-tuned checkpoints. Still experimental on K2.6
        /// (the Kimi-style role markers currently trigger a downstream
        /// NaN in the MLA pipeline on some token sequences — under
        /// investigation; see roadmap §4).
        #[arg(long)]
        chat: bool,
        /// Repetition penalty (1.0 = off). 1.05 is a mild default that
        /// keeps greedy decoding from locking into single-token loops.
        /// Applied BEFORE temperature / top-k / top-p. Bump to 1.2+ if
        /// the bare-prompt "masked masked" spiral shows up.
        #[arg(long, default_value = "1.05")]
        rep_penalty: f32,
    },

    /// Start an OpenAI-compatible API server
    Serve {
        model: String,
        #[arg(long)]
        variant: Option<String>,
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Path to tokenizer.json (auto-detected if not specified)
        #[arg(long)]
        tokenizer: Option<String>,
    },

    /// Download a model without running it
    Pull {
        model: String,
        #[arg(long)]
        variant: Option<String>,
    },

    /// List locally available models
    List,

    /// Delete a local model
    Rm { model: String },

    /// Show model info and hardware compatibility
    Info { model: String },

    /// Show detected hardware
    Hw,

    /// Convert a model to .vib3 format
    Convert {
        /// Input model path or "random"
        #[arg(long)]
        model: String,
        /// Output .vib3 file path
        #[arg(long)]
        output: String,
        /// Model architecture (auto-detected from config.json if available)
        #[arg(long, default_value = "auto")]
        arch: String,
        /// Build HNSW vector index
        #[arg(long)]
        build_indexes: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("VIB3_LOG").unwrap_or_else(|_| "vib3=info".into()))
        .with_target(false)
        .init();

    match cli.command {
        Commands::Run {
            model,
            variant,
            vram,
            ram,
            max_tokens,
            temperature,
            top_k,
            top_p,
            no_check,
            tokenizer,
            chat,
            rep_penalty,
        } => {
            cmd_run(
                &model,
                variant.as_deref(),
                vram,
                ram,
                max_tokens,
                temperature,
                top_k,
                top_p,
                no_check,
                tokenizer.as_deref(),
                chat,
                rep_penalty,
            )
            .await
        }

        Commands::Serve {
            model,
            variant,
            port,
            tokenizer,
        } => cmd_serve(&model, variant.as_deref(), port, tokenizer.as_deref()).await,

        Commands::Pull { model, variant } => cmd_pull(&model, variant.as_deref()).await,

        Commands::List => cmd_list(),

        Commands::Rm { model } => cmd_rm(&model),

        Commands::Info { model } => cmd_info(&model).await,

        Commands::Hw => cmd_hw(),

        Commands::Convert {
            model,
            output,
            arch,
            build_indexes,
        } => cmd_convert(&model, &output, &arch, build_indexes),
    }
}

// ─── Commands ────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn cmd_run(
    model: &str,
    variant: Option<&str>,
    vram_override: Option<usize>,
    ram_override: Option<usize>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    no_check: bool,
    tokenizer_path: Option<&str>,
    chat: bool,
    rep_penalty: f32,
) -> anyhow::Result<()> {
    print_banner();

    // 1. Detect hardware
    let hw = HardwareInfo::detect();
    println!("{}\n", hw.summary());

    // 2. Ensure model is downloaded + converted
    let model_path = ensure_model(model, variant).await?;

    // 3. Hardware check
    if !no_check {
        if let Ok(file) =
            vib3::storage::format::Vib3File::open(model_path.to_string_lossy().as_ref())
        {
            let mc = file.model_config();
            let total_pages = file.page_count();
            let model_size_gb =
                (total_pages * vib3::core::types::PAGE_SIZE) as u32 / (1024 * 1024 * 1024);
            let requirements = HardwareRequirements {
                min_vram_gb: 4,
                rec_vram_gb: (model_size_gb / 3).max(8),
                min_ram_gb: (model_size_gb / 2).max(8),
                rec_ram_gb: model_size_gb.max(16),
                min_disk_gb: model_size_gb + 10,
                nvme_required: model_size_gb > 50,
            };
            let warnings = hw.meets_requirements(&requirements);
            if !warnings.is_empty() {
                println!("Hardware warnings for {}:", mc.name);
                for w in &warnings {
                    println!("  ! {}", w);
                }
                println!();
            }
        }
    }

    // 4. Auto-configure
    let mut config = EngineConfig {
        model_path: model_path.to_string_lossy().to_string(),
        tokenizer_path: tokenizer_path.unwrap_or("").to_string(),
        api_port: 0,
        ..Default::default()
    };

    if let Some(vram) = vram_override {
        config.buffer_pool.t1_capacity = vram * 1024 * 1024;
    }
    if let Some(ram) = ram_override {
        config.buffer_pool.t2_capacity = ram * 1024 * 1024;
    }

    // 5. Initialize engine
    println!("Loading {}...", model);
    let mut engine = Engine::new(config).await?;
    let mc = engine.model_config();
    println!(
        "Ready: {} — {} experts x {} layers\n",
        mc.name, mc.num_experts, mc.num_moe_layers
    );

    // 6. Interactive loop.
    //
    // Default is raw (continue-the-text) mode with a small repetition
    // penalty so the greedy path can't lock into a single-token loop.
    // `--chat` opts into the model's chat template (role markers +
    // special stop tokens). Still experimental on K2.6 — the Kimi-style
    // markers trigger an MLA-side NaN on some token sequences pending
    // diagnosis; use `--chat` with care until that lands.
    let tok = engine.tokenizer();
    let chat_supported = tok.token_to_id("<|im_start|>").is_some()
        || tok.token_to_id("<|im_system|>").is_some();
    let use_chat = chat && chat_supported;
    let stop_tokens = if use_chat {
        tok.stop_token_ids()
    } else {
        Vec::new()
    };
    if use_chat {
        println!(
            "Chat mode (arch={}, stop tokens={:?}, rep_penalty={}).",
            mc.architecture, stop_tokens, rep_penalty
        );
    } else if chat && !chat_supported {
        println!("--chat requested but tokenizer has no chat-template markers; falling back to raw mode.");
    } else {
        println!(
            "Raw mode (rep_penalty={}). Pass --chat for instruction-format wrapping.",
            rep_penalty
        );
    }
    println!("Type your prompt (Ctrl+D to exit):\n");

    loop {
        print!(">>> ");
        std::io::stdout().flush()?;

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let params = vib3::runtime::generate::SamplingParams {
            max_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty: rep_penalty,
            stop_tokens: stop_tokens.clone(),
            ..Default::default()
        };

        // Route through encode_chat when chat templating is on — it builds
        // the token stream with raw special-token IDs injected directly,
        // which is the only reliable way to get chat markers into models
        // whose `tokenizer.json` lists them in the BPE vocab but NOT in
        // `added_tokens` (ByteLevel pre-tokenizer would otherwise break
        // `<|im_system|>` into 14 ASCII byte tokens).
        let result = if use_chat {
            let prompt_tokens = engine.tokenizer().encode_chat(input, None);
            engine.generate_with_tokens(&prompt_tokens, params).await?
        } else {
            engine.generate_with_params(input, params).await?
        };

        println!("{}\n", result.text);
        println!(
            "({} tokens, {:.1} t/s, TTFT {:.0}ms, T1 hit {:.0}%)\n",
            result.tokens_generated,
            result.tokens_per_second,
            result.time_to_first_token_ms,
            result.stats.t1_hits as f64 / result.stats.total_page_accesses.max(1) as f64 * 100.0,
        );
    }

    println!("\nBye!");
    Ok(())
}

async fn cmd_serve(
    model: &str,
    variant: Option<&str>,
    port: u16,
    tokenizer_path: Option<&str>,
) -> anyhow::Result<()> {
    print_banner();

    let model_path = ensure_model(model, variant).await?;

    let config = EngineConfig {
        model_path: model_path.to_string_lossy().to_string(),
        tokenizer_path: tokenizer_path.unwrap_or("").to_string(),
        api_port: port,
        ..Default::default()
    };

    let engine = Engine::new(config).await?;
    let model_name = engine.model_config().name.clone();
    println!("Model: {}", model_name);
    println!("API: http://localhost:{}/v1/chat/completions\n", port);

    let state = std::sync::Arc::new(vib3::api::server::AppState {
        model_name,
        engine: tokio::sync::Mutex::new(engine),
    });
    let app = vib3::api::server::create_router_with_engine(state);
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn cmd_pull(model: &str, variant: Option<&str>) -> anyhow::Result<()> {
    println!("Pulling {}...", model);
    let path = ensure_model(model, variant).await?;
    println!("Model ready: {}", path.display());
    Ok(())
}

fn cmd_list() -> anyhow::Result<()> {
    let store = ModelStore::default()?;
    let models = store.list();

    if models.is_empty() {
        println!("No models downloaded yet.");
        println!("Run: vib3 pull kimi-k2.5");
        return Ok(());
    }

    println!("{:<20} {:<10} {:<10} STATUS", "MODEL", "VARIANT", "SIZE");
    println!("{}", "-".repeat(60));

    for m in &models {
        let size = format_bytes(m.size_bytes);
        let status = if m.download_complete {
            "ready"
        } else {
            "partial"
        };
        println!("{:<20} {:<10} {:<10} {}", m.name, m.variant, size, status);
    }

    println!("\nTotal: {}", format_bytes(store.total_size()));

    Ok(())
}

fn cmd_rm(model: &str) -> anyhow::Result<()> {
    let store = ModelStore::default()?;
    store.delete(model)?;
    println!("Deleted {}", model);
    Ok(())
}

async fn cmd_info(model: &str) -> anyhow::Result<()> {
    let store = ModelStore::default()?;

    if let Some(local) = store.find(model, None) {
        println!("Model:    {}", local.name);
        println!("Variant:  {}", local.variant);
        println!("Size:     {}", format_bytes(local.size_bytes));
        println!("Path:     {}", local.path.display());
        println!(
            "Status:   {}",
            if local.download_complete {
                "ready"
            } else {
                "partial"
            }
        );
    } else {
        println!("Model '{}' not found locally.", model);
        println!("Run: vib3 pull {}", model);
    }

    println!("\n--- Hardware ---");
    let hw = HardwareInfo::detect();
    println!("{}", hw.summary());

    Ok(())
}

fn cmd_hw() -> anyhow::Result<()> {
    let hw = HardwareInfo::detect();
    println!("=== Hardware Detection ===\n");
    println!("{}", hw.summary());

    println!("\nNVMe Devices:");
    for dev in &hw.nvme_paths {
        println!(
            "  {} - {} total, {} free, NVMe={}",
            dev.path,
            format_bytes(dev.size_bytes),
            format_bytes(dev.free_bytes),
            dev.is_nvme,
        );
    }

    Ok(())
}

fn cmd_convert(model: &str, output: &str, arch: &str, build_indexes: bool) -> anyhow::Result<()> {
    use vib3::core::config::ModelConfig;
    use vib3::core::types::DType;

    let output_path = std::path::PathBuf::from(output);

    if model == "random" {
        let config = match arch {
            "kimi-k2.5" => ModelConfig::kimi_k25(),
            _ => ModelConfig {
                name: "test-tiny".into(),
                architecture: "test-tiny".into(),
                hidden_dim: 256,
                expert_hidden_dim: 64,
                num_layers: 5,
                num_moe_layers: 4,
                dense_layer_idx: 0,
                num_experts: 8,
                num_active_experts: 2,
                num_heads: 4,
                num_kv_heads: 2,
                max_seq_len: 2048,
                vocab_size: 1024,
                expert_dtype: DType::FP16,
                shared_dtype: DType::FP16,
                ..Default::default()
            },
        };
        let options = ConvertOptions {
            quantize_experts: QuantFormat::None,
            compress: vib3::storage::format::CompressionMethod::None,
            build_indexes,
        };
        convert_random(&config, &output_path, &options)?;
    } else {
        let dir = std::path::Path::new(model);
        if !dir.exists() {
            anyhow::bail!("Path does not exist: {}", model);
        }

        let config = if arch == "auto" || arch.is_empty() {
            resolve_config_from_dir(dir)?
        } else {
            match arch {
                "kimi-k2.5" => ModelConfig::kimi_k25(),
                other => anyhow::bail!(
                    "Unknown architecture: {}. Use 'auto', 'kimi-k2.5', or provide a config.json.",
                    other
                ),
            }
        };

        let options = ConvertOptions {
            quantize_experts: QuantFormat::Int4,
            compress: vib3::storage::format::CompressionMethod::Zstd { level: 3 },
            build_indexes,
        };
        convert_safetensors_dir(dir, &config, &output_path, &options)?;
    }

    println!("Done!");
    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Ensure a model is available locally as a .vib3 file.
///
/// Pipeline: check local store -> check filesystem -> download from HF -> convert -> register.
async fn ensure_model(model: &str, variant: Option<&str>) -> anyhow::Result<std::path::PathBuf> {
    let store = ModelStore::default()?;

    // 1. Check local store for a ready .vib3 file
    if let Some(local) = store.find(model, variant) {
        if local.download_complete {
            return Ok(local.path);
        }
        println!("Resuming partial download...");
    }

    // 2. Check if user provided a direct file/directory path
    let path = std::path::PathBuf::from(model);
    if path.exists() {
        return Ok(path);
    }

    // 3. Download from HuggingFace for known models
    if let Some((repo_id, _format)) = vib3::registry::hf_repo_for_model(model) {
        println!("Downloading from HuggingFace: {}\n", repo_id);

        let hf = vib3::registry::HfDownloader::new();

        if !hf.has_token() {
            println!("Tip: Set HF_TOKEN env var if the model is gated.\n");
        }

        let model_dir = store.model_dir(model);
        let safetensors_dir = model_dir.join("safetensors");

        let _paths = hf
            .download_model(
                repo_id,
                &safetensors_dir,
                Some(Box::new(|filename, downloaded, total| {
                    let pct = if total > 0 {
                        downloaded as f64 / total as f64 * 100.0
                    } else {
                        0.0
                    };
                    print!(
                        "\r  {} - {:.1}% ({}/{})",
                        filename,
                        pct,
                        format_bytes(downloaded),
                        format_bytes(total),
                    );
                    std::io::stdout().flush().ok();
                })),
            )
            .await?;

        println!("\n\nDownload complete.\n");

        // 4. Auto-convert safetensors -> .vib3
        let vib3_path = model_dir.join(format!("{}.vib3", model));
        println!("Converting to .vib3 format...\n");

        let config = resolve_config_from_dir(&safetensors_dir)?;

        let options = ConvertOptions::default();
        convert_safetensors_dir(&safetensors_dir, &config, &vib3_path, &options)?;

        println!("\nConversion complete: {}\n", vib3_path.display());

        // 5. Register in local store
        let vib3_size = std::fs::metadata(&vib3_path).map(|m| m.len()).unwrap_or(0);
        let local = vib3::registry::LocalModel {
            name: model.to_string(),
            variant: "default".to_string(),
            path: vib3_path.clone(),
            size_bytes: vib3_size,
            hash: String::new(),
            download_complete: true,
            chunks_downloaded: 1,
            total_chunks: 1,
        };
        if let Err(e) = store.mark_complete(&local) {
            eprintln!("Warning: failed to update model manifest: {}", e);
        }

        return Ok(vib3_path);
    }

    // Fall back to vib3 registry (currently fictional)
    println!("Model '{}' not found locally. Downloading...\n", model);
    let downloader = ModelDownloader::new(store);

    let manifest = match downloader.fetch_manifest().await {
        Ok(m) => m,
        Err(_) => {
            return Err(anyhow::anyhow!(
                "Model '{}' not found.\n\n\
                 Known models: kimi-k2.5, mixtral\n\n\
                 Or provide a direct path:\n\
                   vib3 run /path/to/model.vib3\n\n\
                 Or download manually:\n\
                   pip install huggingface_hub\n\
                   huggingface-cli download moonshotai/Kimi-K2.5 --local-dir ./kimi-k2.5\n\
                   vib3 convert --model ./kimi-k2.5 --output kimi.vib3",
                model,
            ));
        }
    };

    let model_manifest = manifest
        .models
        .iter()
        .find(|m| m.name == model)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Model '{}' not found in registry.\nAvailable: {}",
                model,
                manifest
                    .models
                    .iter()
                    .map(|m| m.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;

    let variant = variant.unwrap_or(&model_manifest.default_variant);

    let path = downloader
        .download(
            model_manifest,
            variant,
            Some(Box::new(|progress| {
                print!("\r{}", progress.bar(40));
                std::io::stdout().flush().ok();
            })),
        )
        .await?;

    println!("\n");
    Ok(path)
}

fn print_banner() {
    eprintln!(
"\x1b[38;5;75m\
 \u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}
 \u{2588}\u{2591}\u{2592}\u{2593}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2593}\u{2592}\u{2591}\u{2588}
 \u{2588}\u{2591}\u{2592}\u{2593}                                                                     \u{2593}\u{2592}\u{2591}\u{2588}
 \u{2588}\u{2591}\u{2592}\u{2593}         \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}   \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}    \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}      \u{2584}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2584}               \u{2593}\u{2592}\u{2591}\u{2588}
 \u{2588}\u{2591}\u{2592}\u{2593}          \u{2580}\u{2588}\u{2588}\u{2588}\u{2588}\u{2584} \u{2584}\u{2588}\u{2588}\u{2588}\u{2588}\u{2580}       \u{2588}\u{2588}\u{2588}\u{2588}       \u{2588}\u{2588}\u{2588}\u{2588}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2588}\u{2588}\u{2588}\u{2588}\u{2584}             \u{2593}\u{2592}\u{2591}\u{2588}
 \u{2588}\u{2591}\u{2592}\u{2593}            \u{2580}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2580}         \u{2588}\u{2588}\u{2588}\u{2588}       \u{2588}\u{2588}\u{2588}\u{2588}        \u{2580}\u{2580}\u{2580}             \u{2593}\u{2592}\u{2591}\u{2588}
 \u{2588}\u{2591}\u{2592}\u{2593}              \u{2580}\u{2588}\u{2588}\u{2588}\u{2580}           \u{2588}\u{2588}\u{2588}\u{2588}       \u{2588}\u{2588}\u{2588}\u{2588}\u{2584}       \u{2584}\u{2584}\u{2584}             \u{2593}\u{2592}\u{2591}\u{2588}
 \u{2588}\u{2591}\u{2592}\u{2593}               \u{2580}\u{2588}\u{2580}          \u{2584}\u{2584}\u{2588}\u{2588}\u{2588}\u{2588}\u{2584}\u{2584}      \u{2580}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2580}             \u{2593}\u{2592}\u{2591}\u{2588}
 \u{2588}\u{2591}\u{2592}\u{2593}                \u{2580}            \u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}         \u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}               \u{2593}\u{2592}\u{2591}\u{2588}
 \u{2588}\u{2591}\u{2592}\u{2593}                                                                     \u{2593}\u{2592}\u{2591}\u{2588}
 \u{2588}\u{2591}\u{2592}\u{2593}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2593}\u{2592}\u{2591}\u{2588}
 \u{2588}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2588}
 \u{2588}                                                                           \u{2588}
 \u{2588}     Vector Inference Core: Cheating hardware limits since 1982.           \u{2588}
 \u{2588}                                                                           \u{2588}
 \u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\u{2584}\u{2580}\
\x1b[0m");
}
