//! Build script: compile CUDA kernels and link into the Rust binary.
//!
//! If nvcc is not available, we skip CUDA compilation entirely and
//! the runtime will use CPU fallback kernels.

fn main() {
    println!("cargo:rerun-if-changed=cuda/src/kernels.cu");

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
    let cuda_files = ["cuda/src/kernels.cu"];

    // Target architectures:
    //   sm_89  = Ada Lovelace (RTX 4090)
    //   sm_90  = Hopper (H100/H200)
    //   sm_100 = Blackwell (RTX 5090, B200)
    //   sm_120 = Blackwell Ultra (RTX PRO 6000)
    // The last entry also embeds PTX (compute_120) for forward compatibility
    // with future architectures that can JIT-compile from PTX.
    let arch_flags = [
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_120,code=[sm_120,compute_120]",
    ];

    let out_dir = std::env::var("OUT_DIR").unwrap();

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

        // Create static library
        let lib_path = format!("{}/libvib3_cuda.a", out_dir);
        let ar_status = std::process::Command::new("ar")
            .args(["rcs", &lib_path, &obj_path])
            .status()
            .expect("Failed to run ar");
        if !ar_status.success() {
            panic!("ar failed");
        }
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
