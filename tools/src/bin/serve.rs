use clap::Parser;
use vib3::core::config::EngineConfig;
use vib3::Engine;

#[derive(Parser)]
#[command(name = "vib3-serve", about = "vib3 inference server")]
struct Args {
    /// Path to .vib3 model file
    #[arg(long)]
    model: String,

    /// API port
    #[arg(long, default_value = "8080")]
    port: u16,

    /// CUDA device
    #[arg(long, default_value = "0")]
    device: i32,

    /// VRAM budget (MB), 0 = auto
    #[arg(long, default_value = "0")]
    t1_budget: usize,

    /// RAM budget (MB), 0 = auto
    #[arg(long, default_value = "0")]
    t2_budget: usize,

    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer: Option<String>,

    /// Path to gear_profiles.json (Gearbox integration)
    #[arg(long)]
    gear_profiles: Option<String>,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(if args.verbose {
            "vib3=trace"
        } else {
            "vib3=info"
        })
        .init();

    println!(
        r#"
 ██╗   ██╗██╗██████╗ ██████╗
 ██║   ██║██║██╔══██╗╚════██╗
 ██║   ██║██║██████╔╝ █████╔╝
 ╚██╗ ██╔╝██║██╔══██╗ ╚═══██╗
  ╚████╔╝ ██║██████╔╝██████╔╝
   ╚═══╝  ╚═╝╚═════╝ ╚═════╝
  Weight-Indexed Inference Engine
"#
    );

    let mut config = EngineConfig {
        model_path: args.model,
        tokenizer_path: args.tokenizer.unwrap_or_default(),
        api_port: args.port,
        ..Default::default()
    };

    // Wire CLI budget overrides into buffer pool config (MB → bytes)
    if args.t1_budget > 0 {
        config.buffer_pool.t1_capacity = args.t1_budget * 1024 * 1024;
    }
    if args.t2_budget > 0 {
        config.buffer_pool.t2_capacity = args.t2_budget * 1024 * 1024;
    }

    let mut engine = Engine::new(config).await?;
    let mc = engine.model_config().clone();

    // Load gear profiles for Gearbox integration (Phase B)
    if let Some(ref gear_path) = args.gear_profiles {
        let count = engine.load_gear_profiles(gear_path);
        if count > 0 {
            println!("Loaded {} gear profiles from {}", count, gear_path);
        }
    }

    // Load e_score_correction_bias for sigmoid routing (DeepSeek-V3/Kimi K2.5)
    // Auto-detect from well-known paths
    let bias_paths = [
        "/e_score_correction_bias.bin",
        "/model/e_score_correction_bias.bin",
    ];
    for bias_path in &bias_paths {
        let p = std::path::Path::new(bias_path);
        if p.exists() {
            match engine.load_e_score_correction_bias(p) {
                Ok(()) => {
                    println!("Loaded e_score_correction_bias from {}", bias_path);
                    break;
                }
                Err(e) => {
                    eprintln!("Warning: failed to load e_score_correction_bias from {}: {}", bias_path, e);
                }
            }
        }
    }

    println!(
        "Model: {} — {} experts × {} MoE layers",
        mc.name, mc.num_experts, mc.num_moe_layers
    );
    println!("Listening on http://localhost:{}", args.port);
    println!("POST /v1/chat/completions");

    // Run API server with engine connected
    let state = std::sync::Arc::new(vib3::api::server::AppState {
        model_name: mc.name.clone(),
        engine: tokio::sync::Mutex::new(engine),
    });
    let app = vib3::api::server::create_router_with_engine(state);
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", args.port)).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
