use clap::Parser;
use vib3::Engine;

#[derive(Parser)]
#[command(name = "vib3-bench", about = "Benchmark vib3 inference")]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "5")]
    runs: usize,
    #[arg(long, default_value = "256")]
    gen_tokens: usize,
    #[arg(long)]
    detailed: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter("vib3=info")
        .init();

    println!("vib3-bench v0.1.0");
    println!("Model: {}", args.model);
    println!("Runs: {}, Gen tokens: {}\n", args.runs, args.gen_tokens);

    let mut engine = Engine::from_path(&args.model).await?;

    let prompts = [
        "Write a Python binary search tree implementation.",
        "Explain the proof of the Fundamental Theorem of Calculus.",
        "Describe the key events of the French Revolution.",
    ];

    let mut tps_results = Vec::new();

    for run in 0..args.runs {
        let prompt = prompts[run % prompts.len()];
        print!("Run {}/{}... ", run + 1, args.runs);

        let result = engine.generate(prompt).await?;

        println!(
            "{:.1} t/s, TTFT {:.1}ms, T1 hit {:.1}%",
            result.tokens_per_second,
            result.time_to_first_token_ms,
            result.stats.t1_hits as f64 / result.stats.total_page_accesses.max(1) as f64 * 100.0,
        );

        tps_results.push(result.tokens_per_second);
    }

    if !tps_results.is_empty() {
        let avg: f64 = tps_results.iter().sum::<f64>() / tps_results.len() as f64;
        let max = tps_results.iter().cloned().fold(f64::MIN, f64::max);
        let min = tps_results.iter().cloned().fold(f64::MAX, f64::min);
        println!("\n═══ Results ═══");
        println!("  t/s: avg={avg:.1} min={min:.1} max={max:.1}");
    }

    Ok(())
}
