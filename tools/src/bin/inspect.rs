//! `vib3-inspect` — Inspect .vib3 model files for debugging.

use clap::Parser;
use vib3::core::types::PAGE_SIZE;
use vib3::storage::format::Vib3File;

#[derive(Parser)]
#[command(name = "vib3-inspect", about = "Inspect .vib3 model files")]
struct Args {
    /// Path to .vib3 file
    path: String,

    /// Show page catalog entries
    #[arg(long)]
    pages: bool,

    /// Show shared tensor segments
    #[arg(long)]
    shared: bool,

    /// Filter by layer
    #[arg(long)]
    layer: Option<u16>,

    /// Filter by segment
    #[arg(long)]
    segment: Option<u16>,

    /// Verify page decompression
    #[arg(long)]
    verify: bool,

    /// Sample N pages to verify
    #[arg(long, default_value = "10")]
    sample: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Opening: {}", args.path);
    let file = Vib3File::open(&args.path)?;

    let config = file.model_config();
    println!("\n═══ Model Configuration ═══");
    println!("Name:            {}", config.name);
    println!("Architecture:    {}", config.architecture);
    println!("Hidden dim:      {}", config.hidden_dim);
    println!("Expert hidden:   {}", config.expert_hidden_dim);
    println!(
        "Layers:          {} ({} MoE, dense starts at {})",
        config.num_layers, config.num_moe_layers, config.dense_layer_idx
    );
    println!(
        "Experts:         {} (top-{})",
        config.num_experts, config.num_active_experts
    );
    println!(
        "Heads:           {} Q, {} KV",
        config.num_heads, config.num_kv_heads
    );
    println!("Vocab size:      {}", config.vocab_size);
    println!("Max seq len:     {}", config.max_seq_len);
    println!("Expert dtype:    {:?}", config.expert_dtype);
    println!("Shared dtype:    {:?}", config.shared_dtype);

    // Critical config values for inference quality
    println!("\n═══ Inference-Critical Values ═══");
    println!(
        "rope_theta:      {} ({})",
        config.rope_theta,
        if config.rope_theta == 1000000.0 {
            "✓ Mixtral standard"
        } else if config.rope_theta == 10000.0 {
            "⚠ DEFAULT - may be wrong!"
        } else {
            "custom"
        }
    );
    println!("rms_norm_eps:    {}", config.rms_norm_eps);
    println!("scoring_func:    {}", config.scoring_func);
    if config.scoring_func == "sigmoid" {
        println!("routed_scaling:  {}", config.routed_scaling_factor);
    }
    println!("shared_experts:  {}", config.num_shared_experts);

    if let Some(ref mla) = config.mla {
        println!("\n═══ MLA Config ═══");
        println!("kv_lora_rank:    {}", mla.kv_lora_rank);
        println!("q_lora_rank:     {}", mla.q_lora_rank);
        println!("qk_rope_dim:     {}", mla.qk_rope_head_dim);
        println!("qk_nope_dim:     {}", mla.qk_nope_head_dim);
        println!("v_head_dim:      {}", mla.v_head_dim);
    }

    let header = file.header();
    // Copy packed struct fields to avoid unaligned references
    let page_catalog_count = header.page_catalog_count;
    let page_catalog_size = header.page_catalog_size;
    let vector_index_size = header.vector_index_size;
    let coactivation_size = header.coactivation_size;

    println!("\n═══ File Structure ═══");
    println!("Pages:           {}", file.page_count());
    println!(
        "Page catalog:    {} entries ({} MB)",
        page_catalog_count,
        page_catalog_size as f64 / 1e6
    );
    println!("Vector index:    {} bytes", vector_index_size);
    println!("Coactivation:    {} bytes", coactivation_size);

    // Expert index summary
    println!("\n═══ Expert Index ═══");
    let mut expert_count = 0usize;
    let mut shared_count = 0usize;
    for layer in 0..config.num_layers as u16 {
        if let Some(_entry) = file.expert_entry(layer, 0xFFFF) {
            shared_count += 1;
        }
        for expert in 0..config.num_experts as u16 {
            if file.expert_entry(layer, expert).is_some() {
                expert_count += 1;
            }
        }
    }
    println!("Expert entries:  {}", expert_count);
    println!("Shared entries:  {}", shared_count);

    // Page catalog inspection
    if args.pages {
        println!("\n═══ Page Catalog (first 20) ═══");
        println!(
            "{:<6} {:<6} {:<6} {:<6} {:<8} {:<8} {:<6} {:<6}",
            "LAYER", "EXPERT", "SEG", "IDX", "RAW", "COMP", "ROWS", "TYPE"
        );
        println!("{}", "─".repeat(70));

        let mut shown = 0;
        for page in file.page_catalog().iter() {
            // Copy packed struct fields to avoid unaligned references
            let layer = page.layer;
            let expert = page.expert;
            let segment = page.segment;
            let page_idx = page.page_idx;
            let raw_size = page.raw_size;
            let compressed_size = page.compressed_size;
            let row_count = page.row_count;
            let compression = page.compression;

            if let (Some(l), Some(s)) = (args.layer, args.segment) {
                if layer != l || segment != s {
                    continue;
                }
            } else if let Some(l) = args.layer {
                if layer != l {
                    continue;
                }
            } else if let Some(s) = args.segment {
                if segment != s {
                    continue;
                }
            }

            let expert_hex = if expert == 0xFFFF {
                "SHARED".to_string()
            } else {
                format!("{:#06x}", expert)
            };

            let comp_type = match compression {
                0 => "NONE",
                1 => "LZ4",
                2 => "ZSTD",
                _ => "???",
            };

            println!(
                "{:<6} {:<6} {:<6} {:<6} {:<8} {:<8} {:<6} {:<6}",
                layer,
                expert_hex,
                segment,
                page_idx,
                raw_size,
                compressed_size,
                row_count,
                comp_type,
            );

            shown += 1;
            if shown >= 20 && args.layer.is_none() && args.segment.is_none() {
                println!("... ({} more pages)", file.page_count() - 20);
                break;
            }
        }
    }

    // Shared segment summary
    if args.shared {
        println!("\n═══ Shared Segments ═══");
        for layer in 0..config.num_layers as u16 {
            let shared_pages = file.pages_for_shared(layer);
            if shared_pages.is_empty() {
                continue;
            }

            // Group by segment
            let mut segments = std::collections::BTreeMap::new();
            for page in shared_pages {
                segments
                    .entry(page.segment)
                    .and_modify(|e: &mut (u32, u32)| {
                        e.0 += 1;
                        e.1 += page.raw_size;
                    })
                    .or_insert((1u32, page.raw_size));
            }

            println!("Layer {} shared:", layer);
            for (seg, (count, total_bytes)) in segments {
                let seg_name = match seg {
                    3 => "router",
                    4 => "q_proj",
                    5 => "o_proj",
                    6 => "attn_norm",
                    7 => "ffn_norm",
                    10 => "embed",
                    11 => "lm_head",
                    12 => "k_proj",
                    13 => "v_proj",
                    _ => "???",
                };
                println!(
                    "  Segment {:2} ({}): {} pages, {} bytes",
                    seg, seg_name, count, total_bytes
                );
            }
        }
    }

    // Page verification
    if args.verify {
        println!("\n═══ Page Verification ═══");

        let total_pages = file.page_count();
        let sample_size = args.sample.min(total_pages);

        // Sample pages evenly across the catalog
        let step = (total_pages / sample_size).max(1);
        let mut tested = 0usize;
        let mut failed = 0usize;

        for i in (0..total_pages).step_by(step) {
            let page = file.page(i);
            let raw_size = page.raw_size as usize;
            let mut buf = vec![0u8; raw_size];

            match file.read_page_sync(i, &mut buf) {
                Ok(n) => {
                    if n != raw_size {
                        println!(
                            "  Page {}: size mismatch (got {}, expected {})",
                            i, n, raw_size
                        );
                        failed += 1;
                    }
                }
                Err(e) => {
                    println!("  Page {}: FAILED - {:?}", i, e);
                    failed += 1;
                }
            }
            tested += 1;
        }

        if failed == 0 {
            println!("  ✓ All {} sampled pages verified OK", tested);
        } else {
            println!("  ✗ {}/{} pages failed", failed, tested);
        }
    }

    // Check for potential issues
    println!("\n═══ Potential Issues ═══");
    let mut issues = Vec::new();

    // Check rope_theta
    if config.architecture.contains("mixtral") && config.rope_theta != 1000000.0 {
        issues.push(format!(
            "rope_theta={} but Mixtral expects 1000000.0",
            config.rope_theta
        ));
    }

    // Check for missing attention weights
    let has_q = !file.pages_for_shared_segment(0, 4).is_empty();
    let has_o = !file.pages_for_shared_segment(0, 5).is_empty();
    if !has_q {
        issues.push("No Q projection weights found (segment 4)".to_string());
    }
    if !has_o {
        issues.push("No O projection weights found (segment 5)".to_string());
    }

    // Check for missing embeddings
    let has_embed = !file.pages_for_shared_segment(0, 10).is_empty();
    let has_lm_head = !file.pages_for_shared_segment(0, 11).is_empty();
    if !has_embed {
        issues.push("No embedding weights found (segment 10)".to_string());
    }
    if !has_lm_head {
        issues.push("No lm_head weights found (segment 11)".to_string());
    }

    // Check for missing final norm
    let has_final_norm = !file.pages_for_shared_segment(0xFFFF, 8).is_empty();
    if !has_final_norm {
        issues.push("No final norm weights found (layer=0xFFFF, segment 8)".to_string());
    }

    if issues.is_empty() {
        println!("  ✓ No issues detected");
    } else {
        for issue in &issues {
            println!("  ⚠ {}", issue);
        }
    }

    println!("\n═══ Memory Estimates ═══");
    let expert_size = config.expert_size_bytes();
    let total_expert_bytes = config.total_expert_bytes();
    let total_pages = config.total_expert_pages();
    println!("Expert size:     {} KB", expert_size / 1024);
    println!(
        "Total experts:   {} MB ({} pages)",
        total_expert_bytes / (1024 * 1024),
        total_pages
    );
    println!(
        "Estimated total: {} MB",
        config.estimated_total_bytes() / (1024 * 1024)
    );

    println!("\n═══ Page Size ═══");
    println!("Page size:       {} MB", PAGE_SIZE / (1024 * 1024));

    Ok(())
}
