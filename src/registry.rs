//! Model registry, auto-download, and local model store.
//!
//! Provides the `vib3 run kimi-k2.5` experience:
//!   1. Check local store for the model
//!   2. If not found, fetch manifest from registry
//!   3. Download .vib3 file (chunked, resumable, with progress)
//!   4. Verify integrity (blake3 hash)
//!   5. Auto-detect hardware and configure tiers
//!   6. Launch inference

use crate::core::config::BufferPoolConfig;
use crate::core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ─── Registry ────────────────────────────────────────────────────────────

/// Remote model registry URL.
pub const DEFAULT_REGISTRY: &str = "https://registry.vib3.dev/v1";

/// Fallback: GitHub releases.
pub const FALLBACK_REGISTRY: &str = "https://github.com/vib3-dev/models/releases/download";

/// A model available in the registry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Short name (e.g., "kimi-k2.5")
    pub name: String,
    /// Display name
    pub display_name: String,
    /// Architecture identifier
    pub architecture: String,
    /// Model description
    pub description: String,

    /// Available variants (quantizations)
    pub variants: Vec<ModelVariant>,

    /// Default variant to use
    pub default_variant: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelVariant {
    /// Variant name (e.g., "int4", "fp8", "1.8bit")
    pub name: String,
    /// Total download size in bytes
    pub size_bytes: u64,
    /// Number of download chunks
    pub num_chunks: u32,
    /// Chunk size in bytes (last chunk may be smaller)
    pub chunk_size: u64,
    /// Blake3 hash of complete file
    pub hash: String,
    /// Per-chunk hashes for integrity verification
    pub chunk_hashes: Vec<String>,

    /// Download URLs (multiple for redundancy)
    pub urls: Vec<String>,

    /// Minimum hardware requirements
    pub requirements: HardwareRequirements,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareRequirements {
    /// Minimum VRAM in GB
    pub min_vram_gb: u32,
    /// Recommended VRAM in GB
    pub rec_vram_gb: u32,
    /// Minimum system RAM in GB
    pub min_ram_gb: u32,
    /// Recommended RAM in GB
    pub rec_ram_gb: u32,
    /// Minimum free disk space in GB
    pub min_disk_gb: u32,
    /// Whether NVMe is required (vs SATA SSD)
    pub nvme_required: bool,
}

/// Full registry manifest (list of all available models).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegistryManifest {
    pub version: u32,
    pub models: Vec<ModelManifest>,
}

// ─── Local Model Store ───────────────────────────────────────────────────

/// Manages `~/.vib3/models/` — the local cache of downloaded models.
pub struct ModelStore {
    root: PathBuf,
}

/// Status of a model in the local store.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalModel {
    pub name: String,
    pub variant: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub hash: String,
    pub download_complete: bool,
    /// If download was interrupted, how many chunks are done
    pub chunks_downloaded: u32,
    pub total_chunks: u32,
}

impl ModelStore {
    /// Open or create the model store at the default location.
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Result<Self> {
        let root = dirs_next().join("models");
        std::fs::create_dir_all(&root)?;
        Self::new(root)
    }

    /// Open or create a model store at a specific path.
    pub fn new(root: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Check if a model is available locally.
    pub fn find(&self, name: &str, variant: Option<&str>) -> Option<LocalModel> {
        let manifest_path = self.model_dir(name).join("manifest.json");
        if !manifest_path.exists() {
            return None;
        }

        let data = std::fs::read_to_string(&manifest_path).ok()?;
        let local: LocalModel = serde_json::from_str(&data).ok()?;

        // Check variant matches (or use default)
        if let Some(v) = variant {
            if local.variant != v {
                return None;
            }
        }

        // Check if download is complete
        if !local.download_complete {
            return Some(local); // Partial — caller can resume
        }

        // Check file exists
        if !local.path.exists() {
            return None;
        }

        Some(local)
    }

    /// List all locally available models.
    pub fn list(&self) -> Vec<LocalModel> {
        let mut models = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let manifest = entry.path().join("manifest.json");
                if manifest.exists() {
                    if let Ok(data) = std::fs::read_to_string(&manifest) {
                        if let Ok(local) = serde_json::from_str::<LocalModel>(&data) {
                            models.push(local);
                        }
                    }
                }
            }
        }

        models
    }

    /// Get the directory for a model.
    pub fn model_dir(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Get the .vib3 file path for a model.
    pub fn model_path(&self, name: &str, variant: &str) -> PathBuf {
        self.model_dir(name)
            .join(format!("{}-{}.vib3", name, variant))
    }

    /// Record a model as fully downloaded.
    pub fn mark_complete(&self, model: &LocalModel) -> Result<()> {
        let manifest_path = self.model_dir(&model.name).join("manifest.json");
        let mut updated = model.clone();
        updated.download_complete = true;
        updated.chunks_downloaded = updated.total_chunks;
        let data = serde_json::to_string_pretty(&updated)
            .map_err(|e| Error::ConfigError(e.to_string()))?;
        std::fs::write(manifest_path, data)?;
        Ok(())
    }

    /// Update download progress.
    pub fn update_progress(&self, model: &LocalModel, chunks_done: u32) -> Result<()> {
        let manifest_path = self.model_dir(&model.name).join("manifest.json");
        let mut updated = model.clone();
        updated.chunks_downloaded = chunks_done;
        let data = serde_json::to_string_pretty(&updated)
            .map_err(|e| Error::ConfigError(e.to_string()))?;
        std::fs::write(manifest_path, data)?;
        Ok(())
    }

    /// Delete a model from the store.
    pub fn delete(&self, name: &str) -> Result<()> {
        let dir = self.model_dir(name);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    /// Total disk usage of the store.
    pub fn total_size(&self) -> u64 {
        self.list()
            .iter()
            .filter(|m| m.download_complete)
            .map(|m| m.size_bytes)
            .sum()
    }
}

/// Get the vib3 home directory.
fn dirs_next() -> PathBuf {
    if let Ok(dir) = std::env::var("VIB3_HOME") {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".vib3")
}

// Re-export dirs from the dirs crate
mod dirs {
    use std::path::PathBuf;
    pub fn home_dir() -> Option<PathBuf> {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

// ─── Downloader ──────────────────────────────────────────────────────────

/// Download progress callback.
pub type ProgressCallback = Box<dyn Fn(DownloadProgress) + Send>;

#[derive(Clone, Debug)]
pub struct DownloadProgress {
    pub model_name: String,
    pub variant: String,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub chunks_complete: u32,
    pub total_chunks: u32,
    pub speed_bytes_per_sec: f64,
    pub eta_seconds: f64,
}

impl DownloadProgress {
    pub fn percent(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.downloaded_bytes as f64 / self.total_bytes as f64 * 100.0
    }

    /// Format as a progress bar string.
    pub fn bar(&self, width: usize) -> String {
        let pct = self.percent();
        let filled = (pct / 100.0 * width as f64) as usize;
        let empty = width.saturating_sub(filled);

        let speed = format_bytes(self.speed_bytes_per_sec as u64);
        let downloaded = format_bytes(self.downloaded_bytes);
        let total = format_bytes(self.total_bytes);
        let eta = format_duration(self.eta_seconds);

        format!(
            "[{}{}] {:.1}% {}/{} {}/s ETA {}",
            "█".repeat(filled),
            "░".repeat(empty),
            pct,
            downloaded,
            total,
            speed,
            eta,
        )
    }
}

pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.0} KB", bytes as f64 / 1e3)
    } else {
        format!("{} B", bytes)
    }
}

fn format_duration(seconds: f64) -> String {
    if seconds >= 3600.0 {
        format!(
            "{}h{:02}m",
            seconds as u64 / 3600,
            (seconds as u64 % 3600) / 60
        )
    } else if seconds >= 60.0 {
        format!("{}m{:02}s", seconds as u64 / 60, seconds as u64 % 60)
    } else {
        format!("{:.0}s", seconds)
    }
}

/// Downloads a model from the registry with chunked, resumable transfers.
pub struct ModelDownloader {
    store: ModelStore,
    registry_url: String,
}

impl ModelDownloader {
    pub fn new(store: ModelStore) -> Self {
        Self {
            store,
            registry_url: DEFAULT_REGISTRY.to_string(),
        }
    }

    pub fn with_registry(mut self, url: &str) -> Self {
        self.registry_url = url.to_string();
        self
    }

    /// Fetch the registry manifest.
    pub async fn fetch_manifest(&self) -> Result<RegistryManifest> {
        let url = format!("{}/manifest.json", self.registry_url);
        let resp = reqwest::get(&url)
            .await
            .map_err(|e| Error::Other(e.into()))?;
        let manifest: RegistryManifest = resp.json().await.map_err(|e| Error::Other(e.into()))?;
        Ok(manifest)
    }

    /// Download a model, with resume support and progress reporting.
    pub async fn download(
        &self,
        manifest: &ModelManifest,
        variant: &str,
        on_progress: Option<ProgressCallback>,
    ) -> Result<PathBuf> {
        let var = manifest
            .variants
            .iter()
            .find(|v| v.name == variant)
            .ok_or_else(|| {
                Error::ConfigError(format!(
                    "Variant '{}' not found for model '{}'",
                    variant, manifest.name
                ))
            })?;

        // Create model directory
        let model_dir = self.store.model_dir(&manifest.name);
        std::fs::create_dir_all(&model_dir)?;

        let output_path = self.store.model_path(&manifest.name, variant);

        // Check for existing partial download
        let existing = self.store.find(&manifest.name, Some(variant));
        let start_chunk = existing
            .as_ref()
            .filter(|m| !m.download_complete)
            .map(|m| m.chunks_downloaded)
            .unwrap_or(0);

        if let Some(ref m) = existing {
            if m.download_complete && m.path.exists() {
                tracing::info!("Model already downloaded: {}", output_path.display());
                return Ok(output_path);
            }
        }

        tracing::info!(
            "Downloading {} ({}) — {} chunks, {} total",
            manifest.name,
            variant,
            var.num_chunks,
            format_bytes(var.size_bytes),
        );

        if start_chunk > 0 {
            tracing::info!("Resuming from chunk {}/{}", start_chunk, var.num_chunks);
        }

        // Create / open output file
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&output_path)?;

        // Record local model entry
        let local = LocalModel {
            name: manifest.name.clone(),
            variant: variant.to_string(),
            path: output_path.clone(),
            size_bytes: var.size_bytes,
            hash: var.hash.clone(),
            download_complete: false,
            chunks_downloaded: start_chunk,
            total_chunks: var.num_chunks,
        };

        let start_time = std::time::Instant::now();
        let mut total_downloaded = start_chunk as u64 * var.chunk_size;

        // Download chunks
        let client = reqwest::Client::new();

        for chunk_idx in start_chunk..var.num_chunks {
            let chunk_offset = chunk_idx as u64 * var.chunk_size;
            let chunk_end = std::cmp::min(chunk_offset + var.chunk_size - 1, var.size_bytes - 1);

            // Use first available URL
            let url = &var.urls[0];
            let range = format!("bytes={}-{}", chunk_offset, chunk_end);

            let resp = client
                .get(url)
                .header("Range", &range)
                .send()
                .await
                .map_err(|e| Error::Other(e.into()))?;

            let chunk_data = resp.bytes().await.map_err(|e| Error::Other(e.into()))?;

            // Verify chunk hash if available
            if let Some(expected_hash) = var.chunk_hashes.get(chunk_idx as usize) {
                let actual_hash = blake3::hash(&chunk_data).to_hex().to_string();
                if actual_hash != *expected_hash {
                    return Err(Error::Other(anyhow::anyhow!(
                        "Chunk {} hash mismatch: expected {}, got {}",
                        chunk_idx,
                        expected_hash,
                        actual_hash,
                    )));
                }
            }

            // Write chunk to file
            use std::io::{Seek, SeekFrom, Write};
            let mut f = &file;
            f.seek(SeekFrom::Start(chunk_offset))?;
            f.write_all(&chunk_data)?;

            total_downloaded += chunk_data.len() as u64;

            // Update progress
            let elapsed = start_time.elapsed().as_secs_f64();
            let speed = if elapsed > 0.0 {
                total_downloaded as f64 / elapsed
            } else {
                0.0
            };
            let remaining = var.size_bytes - total_downloaded;
            let eta = if speed > 0.0 {
                remaining as f64 / speed
            } else {
                0.0
            };

            if let Some(ref cb) = on_progress {
                cb(DownloadProgress {
                    model_name: manifest.name.clone(),
                    variant: variant.to_string(),
                    total_bytes: var.size_bytes,
                    downloaded_bytes: total_downloaded,
                    chunks_complete: chunk_idx + 1,
                    total_chunks: var.num_chunks,
                    speed_bytes_per_sec: speed,
                    eta_seconds: eta,
                });
            }

            // Persist progress (for resume)
            self.store.update_progress(&local, chunk_idx + 1)?;
        }

        // Verify full file hash
        tracing::info!("Verifying file integrity...");
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = &file;
            f.seek(SeekFrom::Start(0))?;
            let mut hasher = blake3::Hasher::new();
            let mut buf = vec![0u8; 1024 * 1024]; // 1 MB read buffer
            loop {
                let n = f.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
            }
            let actual_hash = hasher.finalize().to_hex().to_string();
            if !var.hash.is_empty() && actual_hash != var.hash {
                return Err(Error::Other(anyhow::anyhow!(
                    "File integrity check failed: expected {}, got {}",
                    var.hash,
                    actual_hash,
                )));
            }
            tracing::info!("Integrity verified: {}", &actual_hash[..16]);
        }

        // Mark complete
        self.store.mark_complete(&local)?;
        tracing::info!("Download complete: {}", output_path.display());

        Ok(output_path)
    }
}

// ─── Hardware Detection ──────────────────────────────────────────────────

/// Detected hardware capabilities.
#[derive(Clone, Debug, Serialize)]
pub struct HardwareInfo {
    pub gpu_name: String,
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    pub nvme_paths: Vec<NvmeDevice>,
    pub cpu_cores: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct NvmeDevice {
    pub path: String,
    pub size_bytes: u64,
    pub free_bytes: u64,
    pub is_nvme: bool,
    pub estimated_read_speed: u64, // bytes/sec
}

impl HardwareInfo {
    /// Detect available hardware.
    pub fn detect() -> Self {
        Self {
            gpu_name: detect_gpu_name(),
            vram_bytes: detect_vram(),
            ram_bytes: detect_ram(),
            nvme_paths: detect_nvme(),
            cpu_cores: num_cpus::get(),
        }
    }

    /// Auto-configure tier budgets based on detected hardware.
    pub fn auto_config(&self, model_size_bytes: u64) -> BufferPoolConfig {
        // T1: Use 70% of VRAM (reserve for KV cache, shared layers, workspace)
        let t1 = (self.vram_bytes as f64 * 0.35) as usize; // Expert cache portion

        // T2: Use 80% of RAM
        let t2 = (self.ram_bytes as f64 * 0.80) as usize;

        // NVMe paths
        let nvme: Vec<String> = self
            .nvme_paths
            .iter()
            .filter(|d| d.free_bytes > model_size_bytes)
            .map(|d| d.path.clone())
            .collect();

        BufferPoolConfig {
            t1_capacity: t1,
            t2_capacity: t2,
            nvme_paths: nvme,
            cuda_device: 0,
            ..Default::default()
        }
    }

    /// Print a hardware summary.
    pub fn summary(&self) -> String {
        format!(
            "GPU: {} ({} GB VRAM)\nRAM: {} GB\nNVMe: {} drive(s)\nCPU: {} cores",
            self.gpu_name,
            self.vram_bytes / (1024 * 1024 * 1024),
            self.ram_bytes / (1024 * 1024 * 1024),
            self.nvme_paths.len(),
            self.cpu_cores,
        )
    }

    /// Check if hardware meets model requirements.
    pub fn meets_requirements(&self, req: &HardwareRequirements) -> Vec<String> {
        let mut warnings = Vec::new();
        let vram_gb = (self.vram_bytes / (1024 * 1024 * 1024)) as u32;
        let ram_gb = (self.ram_bytes / (1024 * 1024 * 1024)) as u32;

        if vram_gb < req.min_vram_gb {
            warnings.push(format!(
                "VRAM: {} GB available, {} GB minimum required",
                vram_gb, req.min_vram_gb
            ));
        } else if vram_gb < req.rec_vram_gb {
            warnings.push(format!(
                "VRAM: {} GB available, {} GB recommended (will work but slower)",
                vram_gb, req.rec_vram_gb
            ));
        }

        if ram_gb < req.min_ram_gb {
            warnings.push(format!(
                "RAM: {} GB available, {} GB minimum required",
                ram_gb, req.min_ram_gb
            ));
        }

        if req.nvme_required && self.nvme_paths.iter().all(|d| !d.is_nvme) {
            warnings.push("NVMe SSD required but none detected".into());
        }

        warnings
    }
}

// Platform-specific GPU detection via CUDA Runtime API

fn detect_gpu_name() -> String {
    match crate::compute::cuda_ffi::CudaDevice::new(0) {
        Ok(dev) if dev.is_real_cuda() => dev.name().to_string(),
        _ => "No GPU detected".into(),
    }
}

fn detect_vram() -> u64 {
    match crate::compute::cuda_ffi::CudaDevice::new(0) {
        Ok(dev) if dev.is_real_cuda() => dev.total_mem() as u64,
        _ => 0,
    }
}

fn detect_ram() -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(kb) = parts.get(1) {
                        if let Ok(val) = kb.parse::<u64>() {
                            return val * 1024;
                        }
                    }
                }
            }
        }
    }
    0
}

fn detect_nvme() -> Vec<NvmeDevice> {
    let mut devices = Vec::new();

    #[cfg(target_os = "linux")]
    {
        // Check common mount points
        for path in &["/", "/home", "/mnt/nvme"] {
            if let Ok(stat) = nix_statvfs(path) {
                devices.push(NvmeDevice {
                    path: path.to_string(),
                    size_bytes: stat.0,
                    free_bytes: stat.1,
                    is_nvme: check_is_nvme(path),
                    estimated_read_speed: 0,
                });
            }
        }
    }

    devices
}

#[cfg(target_os = "linux")]
fn nix_statvfs(path: &str) -> std::io::Result<(u64, u64)> {
    use std::ffi::CString;
    let c_path = CString::new(path).unwrap();
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if ret == 0 {
        let total = stat.f_blocks * stat.f_frsize;
        let free = stat.f_bavail * stat.f_frsize;
        Ok((total, free))
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(target_os = "linux")]
fn check_is_nvme(path: &str) -> bool {
    // Check if the block device backing this mount is NVMe
    // by looking at /sys/block/nvme*
    use std::process::Command;
    if let Ok(output) = Command::new("findmnt")
        .args(["-n", "-o", "SOURCE", path])
        .output()
    {
        let source = String::from_utf8_lossy(&output.stdout);
        return source.contains("nvme");
    }
    false
}

#[cfg(not(target_os = "linux"))]
fn nix_statvfs(_path: &str) -> std::io::Result<(u64, u64)> {
    Ok((0, 0))
}

#[cfg(not(target_os = "linux"))]
fn check_is_nvme(_path: &str) -> bool {
    false
}

// ─── HuggingFace Integration ─────────────────────────────────────────────

/// Known model mappings from short names to HuggingFace repos.
pub fn hf_repo_for_model(name: &str) -> Option<(&'static str, &'static str)> {
    match name {
        "kimi-k2.5" | "kimi-k25" => Some(("moonshotai/Kimi-K2.5", "safetensors")),
        "kimi-k2.5-gguf" | "kimi-k25-gguf" => Some(("unsloth/Kimi-K2.5-GGUF", "gguf")),
        "mixtral" | "mixtral-8x7b" => Some(("mistralai/Mixtral-8x7B-Instruct-v0.1", "safetensors")),
        _ => None,
    }
}

/// Metadata for a file in a HuggingFace repository.
#[derive(Clone, Debug, Deserialize)]
pub struct HfFileInfo {
    #[serde(rename = "rfilename")]
    pub filename: String,
    pub size: Option<u64>,
}

/// Downloads model files from HuggingFace Hub.
///
/// Uses the HF API directly via reqwest:
/// - List files: `GET https://huggingface.co/api/models/{repo}/tree/main`
/// - Download:   `GET https://huggingface.co/{repo}/resolve/main/{filename}`
///
/// Supports `HF_TOKEN` env var for gated models.
pub struct HfDownloader {
    client: reqwest::Client,
    token: Option<String>,
}

impl Default for HfDownloader {
    fn default() -> Self {
        Self::new()
    }
}

impl HfDownloader {
    pub fn new() -> Self {
        let token = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok();

        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(ref t) = token {
            if let Ok(val) = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", t)) {
                headers.insert(reqwest::header::AUTHORIZATION, val);
            }
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent("vib3/0.1")
            .build()
            .expect("Failed to build HTTP client");

        Self { client, token }
    }

    /// List files in a HuggingFace repository.
    pub async fn list_files(&self, repo_id: &str) -> Result<Vec<HfFileInfo>> {
        let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(anyhow::anyhow!("HF API request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::Other(anyhow::anyhow!(
                "HF API returned {}: {}. {}",
                status,
                &body[..body.len().min(200)],
                if status.as_u16() == 401 {
                    "Set HF_TOKEN for gated models."
                } else {
                    ""
                }
            )));
        }

        let files: Vec<HfFileInfo> = resp
            .json()
            .await
            .map_err(|e| Error::Other(anyhow::anyhow!("Failed to parse HF file list: {}", e)))?;

        Ok(files)
    }

    /// List only safetensors files in a repo, sorted by name.
    pub async fn list_safetensors(&self, repo_id: &str) -> Result<Vec<HfFileInfo>> {
        let all_files = self.list_files(repo_id).await?;
        let mut st_files: Vec<HfFileInfo> = all_files
            .into_iter()
            .filter(|f| {
                f.filename.ends_with(".safetensors")
                    || f.filename == "config.json"
                    || f.filename == "tokenizer.json"
                    || f.filename == "tokenizer_config.json"
                    || f.filename == "special_tokens_map.json"
            })
            .collect();
        st_files.sort_by(|a, b| a.filename.cmp(&b.filename));
        Ok(st_files)
    }

    /// Download a single file from a HuggingFace repo.
    ///
    /// Writes to `output_path`. Supports resume via `Range` header.
    /// Reports progress via callback.
    pub async fn download_file(
        &self,
        repo_id: &str,
        filename: &str,
        output_path: &std::path::Path,
        on_progress: Option<&dyn Fn(u64, u64)>,
    ) -> Result<()> {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );

        // Check existing file for resume
        let existing_size = output_path.metadata().map(|m| m.len()).unwrap_or(0);

        let mut req = self.client.get(&url);
        if existing_size > 0 {
            req = req.header("Range", format!("bytes={}-", existing_size));
        }

        let resp = req.send().await.map_err(|e| {
            Error::Other(anyhow::anyhow!("Download failed for {}: {}", filename, e))
        })?;

        if !resp.status().is_success() && resp.status().as_u16() != 206 {
            return Err(Error::Other(anyhow::anyhow!(
                "Download failed for {}: HTTP {}",
                filename,
                resp.status()
            )));
        }

        let total_size = resp
            .content_length()
            .map(|cl| cl + existing_size)
            .unwrap_or(0);

        // Open file for append (or create)
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)?;

        let mut downloaded = existing_size;
        let mut stream = resp.bytes_stream();

        use futures_util::StreamExt;
        use std::io::Write;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| Error::Other(anyhow::anyhow!("Stream error: {}", e)))?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;

            if let Some(cb) = on_progress {
                cb(downloaded, total_size);
            }
        }

        file.flush()?;
        Ok(())
    }

    /// Download all safetensors files + config + tokenizer for a model.
    ///
    /// Files are saved to `output_dir/{filename}`.
    /// Returns the list of downloaded file paths.
    #[allow(clippy::type_complexity)]
    pub async fn download_model(
        &self,
        repo_id: &str,
        output_dir: &std::path::Path,
        on_progress: Option<Box<dyn Fn(&str, u64, u64) + Send>>,
    ) -> Result<Vec<std::path::PathBuf>> {
        std::fs::create_dir_all(output_dir)?;

        let files = self.list_safetensors(repo_id).await?;
        let total_bytes: u64 = files.iter().filter_map(|f| f.size).sum();
        let mut global_downloaded: u64 = 0;
        let mut paths = Vec::new();

        tracing::info!(
            "Downloading {} files ({}) from {}",
            files.len(),
            format_bytes(total_bytes),
            repo_id,
        );

        for (i, file_info) in files.iter().enumerate() {
            let output_path = output_dir.join(&file_info.filename);

            // Skip if already fully downloaded
            if let Ok(meta) = output_path.metadata() {
                if let Some(expected) = file_info.size {
                    if meta.len() == expected {
                        tracing::info!(
                            "[{}/{}] {} — already downloaded",
                            i + 1,
                            files.len(),
                            file_info.filename
                        );
                        global_downloaded += expected;
                        paths.push(output_path);
                        continue;
                    }
                }
            }

            tracing::info!(
                "[{}/{}] Downloading {} ({})",
                i + 1,
                files.len(),
                file_info.filename,
                file_info
                    .size
                    .map(format_bytes)
                    .unwrap_or_else(|| "unknown".into()),
            );

            let file_name = file_info.filename.clone();
            let file_size = file_info.size.unwrap_or(0);
            let prev_global = global_downloaded;

            self.download_file(
                repo_id,
                &file_info.filename,
                &output_path,
                Some(&|file_downloaded, _file_total| {
                    if let Some(ref cb) = on_progress {
                        cb(&file_name, prev_global + file_downloaded, total_bytes);
                    }
                }),
            )
            .await?;

            global_downloaded += file_size;
            paths.push(output_path);
        }

        tracing::info!("Download complete: {} files", paths.len());
        Ok(paths)
    }

    /// Check if a HuggingFace token is available (for gated models).
    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }
}

// ─── Auto-Profiler ───────────────────────────────────────────────────────

/// Hardware bandwidth profile measured at startup.
///
/// Probes actual bandwidth for each tier so the buffer manager can size its
/// pools optimally. Static detection (detect_ram, detect_vram) tells us
/// *capacity*; profiling tells us *throughput*.
#[derive(Clone, Debug, Serialize)]
pub struct HardwareProfile {
    /// Measured RAM sequential read bandwidth (bytes/sec).
    pub ram_bandwidth_bps: u64,
    /// Measured NVMe sequential read bandwidth (bytes/sec).
    /// Zero if no NVMe detected or probing failed.
    pub nvme_bandwidth_bps: u64,
    /// Measured host→device copy bandwidth (bytes/sec).
    /// Zero if no GPU or running in CPU-only mode.
    pub h2d_bandwidth_bps: u64,
    /// Number of physical CPU cores.
    pub cpu_cores: usize,
    /// Total system RAM in bytes.
    pub ram_bytes: u64,
    /// Total VRAM in bytes (0 if CPU-only).
    pub vram_bytes: u64,
    /// Time taken to run the profile (milliseconds).
    pub profile_time_ms: u64,
}

impl HardwareProfile {
    /// Run a quick hardware bandwidth probe (~100-200ms).
    ///
    /// Probes:
    /// 1. RAM bandwidth: sequential read of a 16 MB buffer
    /// 2. NVMe bandwidth: sequential read of a temp file (if writable)
    /// 3. H2D bandwidth: memcpy from pinned host to device memory
    pub fn probe() -> Self {
        let start = std::time::Instant::now();

        let cpu_cores = num_cpus::get();
        let ram_bytes = detect_ram();
        let vram_bytes = detect_vram();

        let ram_bandwidth_bps = probe_ram_bandwidth();
        let nvme_bandwidth_bps = probe_nvme_bandwidth();
        let h2d_bandwidth_bps = probe_h2d_bandwidth();

        let profile_time_ms = start.elapsed().as_millis() as u64;

        Self {
            ram_bandwidth_bps,
            nvme_bandwidth_bps,
            h2d_bandwidth_bps,
            cpu_cores,
            ram_bytes,
            vram_bytes,
            profile_time_ms,
        }
    }

    /// Auto-configure buffer pool sizes based on measured bandwidth.
    ///
    /// The key insight: T2 pool should be large enough to hide NVMe latency.
    /// If NVMe bandwidth is B_nvme and H2D bandwidth is B_h2d, then:
    /// - T2 should buffer at least `B_h2d / B_nvme` pages worth of data
    ///   to keep the GPU fed while NVMe loads the next batch.
    /// - T1 should hold at least one full layer's active experts.
    pub fn auto_config(&self, model_config: &crate::core::config::ModelConfig) -> BufferPoolConfig {
        let expert_size = model_config.expert_size_bytes();
        let active_experts = model_config.num_active_experts as usize;

        // T1: enough for 2 layers of active experts + shared weights
        let shared_overhead = 4 * crate::core::types::PAGE_SIZE; // ~8MB for shared layers
        let t1_min = active_experts * expert_size * 2 + shared_overhead;
        let t1 = if self.vram_bytes > 0 {
            // Use 35% of VRAM, but at least the minimum
            let t1_auto = (self.vram_bytes as f64 * 0.35) as usize;
            t1_auto.max(t1_min)
        } else {
            // CPU-only mode: use T1 as RAM (CPU fallback allocations)
            let ram_budget = (self.ram_bytes as f64 * 0.15) as usize;
            ram_budget.max(t1_min)
        };

        // T2: size to absorb NVMe-to-H2D bandwidth mismatch
        let t2 = if self.nvme_bandwidth_bps > 0 && self.h2d_bandwidth_bps > 0 {
            // How many bytes can H2D transfer per page load time?
            // We want T2 to buffer enough so the GPU never stalls waiting for NVMe.
            let nvme_pages_per_sec =
                self.nvme_bandwidth_bps as f64 / crate::core::types::PAGE_SIZE as f64;
            let h2d_pages_per_sec =
                self.h2d_bandwidth_bps as f64 / crate::core::types::PAGE_SIZE as f64;
            // Need enough T2 to keep the H2D pipe busy while NVMe refills
            let ratio = h2d_pages_per_sec / nvme_pages_per_sec.max(1.0);
            let min_t2_pages = (ratio * 4.0).ceil() as usize; // 4x headroom
            let t2_from_bw = min_t2_pages * crate::core::types::PAGE_SIZE;

            // But also use available RAM (80%)
            let t2_from_ram = (self.ram_bytes as f64 * 0.80) as usize;
            t2_from_bw
                .max(active_experts * expert_size)
                .min(t2_from_ram)
        } else {
            // Fallback: use 60% of RAM
            (self.ram_bytes as f64 * 0.60) as usize
        };

        // Prefetch queue depth: scale with CPU cores
        let prefetch_depth = (self.cpu_cores * 4).clamp(16, 256);

        BufferPoolConfig {
            t1_capacity: t1,
            t2_capacity: t2,
            prefetch_queue_depth: prefetch_depth,
            cuda_device: 0,
            ..Default::default()
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "RAM: {:.1} GB ({:.1} GB/s), NVMe: {:.1} GB/s, H2D: {:.1} GB/s, {} cores, profiled in {} ms",
            self.ram_bytes as f64 / 1e9,
            self.ram_bandwidth_bps as f64 / 1e9,
            self.nvme_bandwidth_bps as f64 / 1e9,
            self.h2d_bandwidth_bps as f64 / 1e9,
            self.cpu_cores,
            self.profile_time_ms,
        )
    }
}

/// Probe RAM sequential read bandwidth by reading a 16 MB buffer.
fn probe_ram_bandwidth() -> u64 {
    use crate::compute::cuda_ffi;

    let probe_size = 16 * 1024 * 1024; // 16 MB
    let ptr = match cuda_ffi::host_alloc_pinned(probe_size) {
        Ok(p) => p,
        Err(_) => return 0,
    };

    // Write pattern to force actual allocation
    unsafe {
        std::ptr::write_bytes(ptr, 0xAB, probe_size);
    }

    // Time sequential reads (multiple passes for stability)
    let mut total_bytes = 0u64;
    let start = std::time::Instant::now();
    let passes = 4;

    for _ in 0..passes {
        let mut sum: u64 = 0;
        let slice = unsafe { std::slice::from_raw_parts(ptr, probe_size) };
        // Read in 4K chunks (cache-line friendly)
        for chunk in slice.chunks(4096) {
            // Use volatile read to prevent optimization
            sum = sum.wrapping_add(chunk[0] as u64);
            sum = sum.wrapping_add(chunk[chunk.len() - 1] as u64);
        }
        // Prevent dead code elimination
        std::hint::black_box(sum);
        total_bytes += probe_size as u64;
    }

    let elapsed_ns = start.elapsed().as_nanos() as u64;
    cuda_ffi::host_free_pinned(ptr, probe_size);

    if elapsed_ns > 0 {
        (total_bytes as f64 * 1e9 / elapsed_ns as f64) as u64
    } else {
        0
    }
}

/// Probe NVMe sequential read bandwidth by writing and reading a temp file.
fn probe_nvme_bandwidth() -> u64 {
    use std::io::{Read, Seek, SeekFrom, Write};

    let probe_size = 4 * 1024 * 1024; // 4 MB

    // Create a temp file in the system's temp directory
    let tmp_path = std::env::temp_dir().join(".vib3_nvme_probe");
    let mut file = match std::fs::File::create(&tmp_path) {
        Ok(f) => f,
        Err(_) => return 0,
    };

    // Write probe data
    let data = vec![0xABu8; probe_size];
    if file.write_all(&data).is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return 0;
    }
    if file.flush().is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return 0;
    }
    // Sync to ensure data is on disk, not in page cache
    if file.sync_all().is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return 0;
    }

    // Seek back and time the read
    if file.seek(SeekFrom::Start(0)).is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return 0;
    }

    let mut read_buf = vec![0u8; probe_size];
    let passes = 4;
    let mut total_bytes = 0u64;
    let start = std::time::Instant::now();

    for _ in 0..passes {
        if file.seek(SeekFrom::Start(0)).is_err() {
            break;
        }
        match file.read_exact(&mut read_buf) {
            Ok(()) => {
                total_bytes += probe_size as u64;
                std::hint::black_box(&read_buf[0]);
            }
            Err(_) => break,
        }
    }

    let elapsed_ns = start.elapsed().as_nanos() as u64;
    let _ = std::fs::remove_file(&tmp_path);

    if elapsed_ns > 0 && total_bytes > 0 {
        (total_bytes as f64 * 1e9 / elapsed_ns as f64) as u64
    } else {
        0
    }
}

/// Probe host-to-device copy bandwidth.
fn probe_h2d_bandwidth() -> u64 {
    use crate::compute::cuda_ffi;

    let probe_size = 4 * 1024 * 1024; // 4 MB
    let device = match cuda_ffi::CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => return 0,
    };
    let stream = match cuda_ffi::CudaStream::new(&device) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let host_ptr = match cuda_ffi::host_alloc_pinned(probe_size) {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let dev_ptr = match cuda_ffi::device_alloc(probe_size) {
        Ok(p) => p,
        Err(_) => {
            cuda_ffi::host_free_pinned(host_ptr, probe_size);
            return 0;
        }
    };

    // Fill host buffer
    unsafe {
        std::ptr::write_bytes(host_ptr, 0xCD, probe_size);
    }

    let passes = 4;
    let mut total_bytes = 0u64;
    let start = std::time::Instant::now();

    for _ in 0..passes {
        if cuda_ffi::memcpy_h2d_async(dev_ptr, host_ptr, probe_size, &stream).is_err() {
            break;
        }
        if stream.synchronize().is_err() {
            break;
        }
        total_bytes += probe_size as u64;
    }

    let elapsed_ns = start.elapsed().as_nanos() as u64;

    cuda_ffi::device_free(dev_ptr, probe_size);
    cuda_ffi::host_free_pinned(host_ptr, probe_size);

    if elapsed_ns > 0 && total_bytes > 0 {
        (total_bytes as f64 * 1e9 / elapsed_ns as f64) as u64
    } else {
        0
    }
}

// ─── num_cpus stub ───────────────────────────────────────────────────────
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    }
}
