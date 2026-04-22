//! Build script: compile CUDA kernels and link into the Rust binary.
//!
//! If nvcc is not available, we skip CUDA compilation entirely and
//! the runtime will use CPU fallback kernels.

fn main() {
    println!("cargo:rerun-if-changed=cuda/src/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/src/cutlass_mla.cu");
    println!("cargo:rerun-if-changed=build.rs");
    // Re-run if the CUDA toolkit or its discovery inputs change so a later
    // `apt install cuda-toolkit-*` takes effect without requiring `cargo clean`.
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=PATH");

    // Only compile CUDA if the feature is enabled and nvcc is available
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        if find_nvcc().is_some() {
            compile_cuda();
        } else {
            println!("cargo:warning=nvcc not found, building without CUDA kernels");
            println!("cargo:warning=GPU compute will use CPU fallback implementations");
        }
    }
}

fn find_nvcc() -> Option<std::path::PathBuf> {
    // Check common locations
    let candidates = [
        std::env::var("CUDA_PATH")
            .ok()
            .map(|p| std::path::PathBuf::from(p).join("bin/nvcc")),
        Some(std::path::PathBuf::from("/usr/local/cuda/bin/nvcc")),
        Some(std::path::PathBuf::from("/usr/bin/nvcc")),
    ];

    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // Try PATH
    if let Ok(output) = std::process::Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(std::path::PathBuf::from(path));
            }
        }
    }

    None
}

fn compile_cuda() {
    // Regular kernels (compiled once with standard flags).
    let cuda_files = ["cuda/src/kernels.cu"];
    // CUTLASS MLA wrapper — needs CUTLASS include paths and C++17.
    // Only builds if the full CUTLASS tree is present (see third_party/cutlass).
    let cutlass_dir = std::path::Path::new("third_party/cutlass");
    let cutlass_mla_src = std::path::Path::new("cuda/src/cutlass_mla.cu");
    let build_cutlass_mla = cutlass_dir.exists()
        && cutlass_dir.join("include/cutlass/cutlass.h").exists()
        && cutlass_dir
            .join("examples/77_blackwell_fmha/device/sm100_mla.hpp")
            .exists()
        && cutlass_mla_src.exists();

    // Target architectures:
    //   sm_89  = Ada Lovelace (RTX 4090)
    //   sm_90  = Hopper (H100/H200)
    //   sm_100  = Blackwell DC      (B200, GB200)
    //   sm_120  = Blackwell client  (RTX 5090)
    //   sm_120a = Blackwell Workstation (RTX PRO 6000) with block-scaled MMA
    //             — REFERENCE ARCH for vib3. Required for native NVFP4 Tensor
    //             Core MMA via mma.sync.aligned.kind::mxf4. Needs CUDA 12.8+.
    // The last entry also embeds PTX (compute_120a) for forward compatibility
    // with future architectures that can JIT-compile from PTX.
    let arch_flags = [
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_120a,code=[sm_120a,compute_120a]",
    ];
    // CUTLASS MLA kernel only targets Blackwell (TMA+WGMMA specialized).
    let cutlass_arch_flags = [
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_120a,code=[sm_120a,compute_120a]",
    ];

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let mut obj_paths: Vec<String> = Vec::new();

    for cuda_file in &cuda_files {
        let stem = std::path::Path::new(cuda_file)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();
        let obj_path = format!("{}/{}.o", out_dir, stem);

        let mut cmd = std::process::Command::new("nvcc");
        cmd.arg("-c")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-Xcompiler=-fPIC");

        for flag in &arch_flags {
            cmd.arg(flag);
        }

        cmd.arg("-o").arg(&obj_path).arg(cuda_file);

        let status = cmd.status().expect("Failed to run nvcc");
        if !status.success() {
            panic!("nvcc compilation failed for {}", cuda_file);
        }
        obj_paths.push(obj_path);
    }

    if build_cutlass_mla {
        println!(
            "cargo:warning=Building CUTLASS MLA kernel from {}",
            cutlass_dir.display()
        );
        let cutlass_include = cutlass_dir.join("include");
        let cutlass_tools_util = cutlass_dir.join("tools/util/include");
        let cutlass_fmha_dir = cutlass_dir.join("examples/77_blackwell_fmha");
        let cutlass_fmha_common = cutlass_dir.join("examples/77_blackwell_fmha/common");
        let cutlass_fmha_collective = cutlass_dir.join("examples/77_blackwell_fmha/collective");
        // 77_blackwell_fmha's kernels include "gather_tensor.hpp" from
        // examples/common (the cross-example shared dir).
        let cutlass_examples_common = cutlass_dir.join("examples/common");

        let obj_path = format!("{}/cutlass_mla.o", out_dir);
        let mut cmd = std::process::Command::new("nvcc");
        cmd.arg("-c")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-std=c++17")
            .arg("-Xcompiler=-fPIC")
            .arg("-Xcompiler=-Wno-deprecated-declarations")
            // CUTLASS template machinery generates enormous error messages;
            // silence the most noisy warnings so we can see the real ones.
            .arg("-Xcudafe=--diag_suppress=177") // set-but-not-used
            .arg("-Xcudafe=--diag_suppress=550")
            .arg("-Xcudafe=--diag_suppress=174") // expression has no effect
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg(format!("-I{}", cutlass_include.display()))
            .arg(format!("-I{}", cutlass_tools_util.display()))
            .arg(format!("-I{}", cutlass_fmha_dir.display()))
            .arg(format!("-I{}", cutlass_fmha_common.display()))
            .arg(format!("-I{}", cutlass_fmha_collective.display()))
            .arg(format!("-I{}", cutlass_examples_common.display()));

        for flag in &cutlass_arch_flags {
            cmd.arg(flag);
        }

        cmd.arg("-o").arg(&obj_path).arg(cutlass_mla_src);

        let status = cmd.status().expect("Failed to run nvcc for CUTLASS MLA");
        if !status.success() {
            panic!("nvcc compilation failed for cuda/src/cutlass_mla.cu");
        }
        obj_paths.push(obj_path);
        println!("cargo:rustc-cfg=has_cutlass_mla");
    }

    // Create combined static library from all object files.
    let lib_path = format!("{}/libvib3_cuda.a", out_dir);
    let mut ar_cmd = std::process::Command::new("ar");
    ar_cmd.args(["rcs", &lib_path]);
    for o in &obj_paths {
        ar_cmd.arg(o);
    }
    let ar_status = ar_cmd.status().expect("Failed to run ar");
    if !ar_status.success() {
        panic!("ar failed");
    }

    // Link
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=vib3_cuda");
    println!("cargo:rustc-cfg=has_cuda_kernels");

    // Link CUDA runtime
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=cudart");
}
