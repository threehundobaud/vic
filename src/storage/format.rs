//! The `.vib3` model file format.
//!
//! A page-aligned, indexed weight file that treats expert matrices as
//! database tables with B-tree-style indexes for page-level lookup.
//!
//! ## Layout
//!
//! ```text
//! ┌───────────────────┐  offset 0
//! │   File Header     │  (512 bytes, fixed)
//! ├───────────────────┤
//! │  Model Metadata   │  (variable, JSON)
//! ├───────────────────┤
//! │  Page Catalog     │  (fixed-size array of PageCatalogEntry)
//! ├───────────────────┤
//! │  Expert Index     │  (expert_id → page ranges)
//! ├───────────────────┤
//! │  Vector Index     │  (embedding → expert activation profiles)
//! ├───────────────────┤
//! │  Coactivation     │  (expert pair correlations)
//! ├───────────────────┤
//! │  View Catalog     │  (materialized view definitions)
//! ├───────────────────┤
//! │  Page Data        │  (2MB-aligned weight pages)
//! └───────────────────┘
//! ```

use crate::core::config::ModelConfig;
use crate::core::error::{Error, Result};
use crate::core::types::{PageId, PAGE_SIZE};
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

// ─── Magic & Constants ───────────────────────────────────────────────────

pub const VIB3_MAGIC: u64 = 0x3342_4956_0000_0001; // "VIB3\0\0\0\1"
pub const VIB3_VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 512;

/// Compression types for page data.
pub const COMPRESSION_NONE: u8 = 0;
pub const COMPRESSION_LZ4: u8 = 1;
pub const COMPRESSION_ZSTD: u8 = 2;

// ─── On-disk Structures (repr(C), Pod for zero-copy) ─────────────────────

/// File header. Exactly 512 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct Vib3Header {
    pub magic: u64,
    pub version: u32,
    pub flags: u32,

    // Section offsets
    pub metadata_offset: u64,
    pub metadata_size: u64,

    pub page_catalog_offset: u64,
    pub page_catalog_size: u64,
    pub page_catalog_count: u32,

    pub expert_index_offset: u64,
    pub expert_index_size: u64,

    pub vector_index_offset: u64,
    pub vector_index_size: u64,

    pub coactivation_offset: u64,
    pub coactivation_size: u64,

    pub view_catalog_offset: u64,
    pub view_catalog_size: u64,
    pub view_count: u32,

    pub page_data_offset: u64,
    pub page_data_size: u64,

    // Quick-access model summary
    pub num_layers: u32,
    pub num_experts: u32,
    pub num_active_experts: u32,
    pub hidden_dim: u32,
    pub expert_hidden_dim: u32,
    pub expert_dtype: u8,
    pub shared_dtype: u8,

    pub _reserved: [u8; 354], // Pad to 512
}

// SAFETY: All fields are plain data, no pointers or references.
unsafe impl Zeroable for Vib3Header {}
unsafe impl Pod for Vib3Header {}

const _: () = assert!(std::mem::size_of::<Vib3Header>() == HEADER_SIZE);

/// One entry in the page catalog. 48 bytes, fixed-size for O(1) lookup.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct PageCatalogEntry {
    pub layer: u16,
    pub expert: u16,
    pub segment: u16,
    pub page_idx: u16,

    pub file_offset: u64,
    pub compressed_size: u32,
    pub raw_size: u32,

    pub row_start: u16,
    pub row_count: u16,
    pub col_start: u16,
    pub col_count: u16,

    pub mean_activation: f32,
    pub access_frequency: f32,
    pub coactivation_cluster: u16,

    pub compression: u8,
    pub flags: u8,
    pub _pad: [u8; 4], // Pad to 48 bytes
}

unsafe impl Zeroable for PageCatalogEntry {}
unsafe impl Pod for PageCatalogEntry {}

const _: () = assert!(std::mem::size_of::<PageCatalogEntry>() == 48);

impl PageCatalogEntry {
    pub fn page_id(&self) -> PageId {
        PageId {
            layer: self.layer,
            expert: self.expert,
            segment: self.segment,
            page_idx: self.page_idx,
        }
    }
}

/// Maps (layer, expert) → page range in the catalog.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct ExpertIndexEntry {
    pub layer: u16,
    pub expert: u16,
    pub first_page_idx: u32,
    pub page_count: u32,
    pub total_bytes: u32,
    pub num_segments: u16,
    pub _reserved: u16,
}

unsafe impl Zeroable for ExpertIndexEntry {}
unsafe impl Pod for ExpertIndexEntry {}

/// Prediction entry in the vector index.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct VectorIndexEntry {
    pub centroid_id: u32,
    pub cluster_size: u16,
    pub prediction_count: u8,
    pub hot_page_count: u8,

    /// Top predicted experts: (expert_id, probability_u8)
    pub expert_predictions: [(u16, u8); 32],

    /// Hot page catalog indices.
    pub hot_pages: [u32; 64],
}

unsafe impl Zeroable for VectorIndexEntry {}
unsafe impl Pod for VectorIndexEntry {}

/// Coactivation between two experts.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct CoactivationEntry {
    pub expert_a: u16,
    pub expert_b: u16,
    pub layer: u16,
    pub _pad: u16,
    pub correlation: f32,
    pub sample_count: u32,
}

unsafe impl Zeroable for CoactivationEntry {}
unsafe impl Pod for CoactivationEntry {}

/// Materialized view header.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct MaterializedViewHeader {
    pub name: [u8; 64],
    pub page_count: u32,
    pub data_offset: u64,
    pub data_size: u64,
    pub expected_coverage: f32,
}

unsafe impl Zeroable for MaterializedViewHeader {}
unsafe impl Pod for MaterializedViewHeader {}

impl MaterializedViewHeader {
    pub fn name_str(&self) -> &str {
        let end = self
            .name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.name.len());
        std::str::from_utf8(&self.name[..end]).unwrap_or("<invalid>")
    }
}

// ─── File Reader ─────────────────────────────────────────────────────────

/// Memory-mapped reader for .vib3 files.
///
/// Uses `mmap` for zero-copy access to the file's metadata sections.
/// Page data is read via io_uring / pread for async I/O.
pub struct Vib3File {
    #[allow(dead_code)]
    file: File,
    mmap: Mmap,
    header: Vib3Header,
    model_config: ModelConfig,

    // Parsed indexes (references into mmap)
    page_catalog: Vec<PageCatalogEntry>,
    expert_index: Vec<ExpertIndexEntry>,
    expert_map: HashMap<u32, usize>, // (layer<<16|expert) → index into expert_index
    view_headers: Vec<MaterializedViewHeader>,
    view_name_map: HashMap<String, usize>,
}

impl Vib3File {
    /// Open a .vib3 file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;

        // SAFETY: We only read from the mmap; the file is opened read-only.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(Error::InvalidFormat {
                reason: "File too small for header".into(),
            });
        }

        // Parse header
        let header: Vib3Header = *bytemuck::from_bytes(&mmap[..HEADER_SIZE]);

        let magic = header.magic;
        if magic != VIB3_MAGIC {
            return Err(Error::InvalidFormat {
                reason: format!("Bad magic: expected {:#x}, got {:#x}", VIB3_MAGIC, magic),
            });
        }

        let version = header.version;
        if version != VIB3_VERSION {
            return Err(Error::InvalidFormat {
                reason: format!("Unsupported version: {}", version),
            });
        }

        // Parse model metadata (JSON)
        let meta_start = header.metadata_offset as usize;
        let meta_end = meta_start + header.metadata_size as usize;
        if meta_end > mmap.len() {
            return Err(Error::InvalidFormat {
                reason: "Metadata section extends past end of file".into(),
            });
        }
        let mut model_config: ModelConfig = serde_json::from_slice(&mmap[meta_start..meta_end])
            .map_err(|e| Error::InvalidFormat {
                reason: format!("Invalid metadata JSON: {e}"),
            })?;
        model_config.fixup_defaults();

        // Parse page catalog
        let catalog_start = header.page_catalog_offset as usize;
        let catalog_bytes =
            header.page_catalog_count as usize * std::mem::size_of::<PageCatalogEntry>();
        let catalog_end = catalog_start + catalog_bytes;
        if catalog_end > mmap.len() {
            return Err(Error::InvalidFormat {
                reason: "Page catalog extends past end of file".into(),
            });
        }
        let page_catalog: Vec<PageCatalogEntry> =
            bytemuck::cast_slice(&mmap[catalog_start..catalog_end]).to_vec();

        // Parse expert index
        let ei_start = header.expert_index_offset as usize;
        let ei_end = ei_start + header.expert_index_size as usize;
        let expert_index: Vec<ExpertIndexEntry> = if ei_end <= mmap.len() && ei_end > ei_start {
            bytemuck::cast_slice(&mmap[ei_start..ei_end]).to_vec()
        } else {
            vec![]
        };

        let mut expert_map = HashMap::new();
        for (i, entry) in expert_index.iter().enumerate() {
            let key = (entry.layer as u32) << 16 | entry.expert as u32;
            expert_map.insert(key, i);
        }

        // Parse view catalog
        let vc_start = header.view_catalog_offset as usize;
        let vc_bytes = header.view_count as usize * std::mem::size_of::<MaterializedViewHeader>();
        let vc_end = vc_start + vc_bytes;
        let view_headers: Vec<MaterializedViewHeader> =
            if header.view_count > 0 && vc_end <= mmap.len() {
                bytemuck::cast_slice(&mmap[vc_start..vc_end]).to_vec()
            } else {
                vec![]
            };

        let mut view_name_map = HashMap::new();
        for (i, vh) in view_headers.iter().enumerate() {
            view_name_map.insert(vh.name_str().to_string(), i);
        }

        Ok(Self {
            file,
            mmap,
            header,
            model_config,
            page_catalog,
            expert_index,
            expert_map,
            view_headers,
            view_name_map,
        })
    }

    // ── Accessors ────────────────────────────────────────────────────

    pub fn header(&self) -> &Vib3Header {
        &self.header
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    pub fn page_count(&self) -> usize {
        self.page_catalog.len()
    }

    pub fn page(&self, idx: usize) -> &PageCatalogEntry {
        &self.page_catalog[idx]
    }

    pub fn page_catalog(&self) -> &[PageCatalogEntry] {
        &self.page_catalog
    }

    // ── Expert Index Lookups ─────────────────────────────────────────

    pub fn expert_entry(&self, layer: u16, expert: u16) -> Option<&ExpertIndexEntry> {
        let key = (layer as u32) << 16 | expert as u32;
        self.expert_map.get(&key).map(|&i| &self.expert_index[i])
    }

    /// Get all page catalog entries for a specific expert.
    pub fn pages_for_expert(&self, layer: u16, expert: u16) -> &[PageCatalogEntry] {
        if let Some(entry) = self.expert_entry(layer, expert) {
            let start = entry.first_page_idx as usize;
            let end = start + entry.page_count as usize;
            if end <= self.page_catalog.len() {
                &self.page_catalog[start..end]
            } else {
                &[]
            }
        } else {
            &[]
        }
    }

    /// Get all page catalog entries for shared layers at a specific layer.
    ///
    /// Shared layers use `expert = 0xFFFF`. This includes attention weights,
    /// norms, router weights, embeddings, and lm_head.
    pub fn pages_for_shared(&self, layer: u16) -> &[PageCatalogEntry] {
        self.pages_for_expert(layer, 0xFFFF)
    }

    /// Get all page catalog entries for a specific shared segment across all layers.
    ///
    /// Returns pages matching `expert == 0xFFFF` and the given segment number.
    /// Useful for finding all embedding pages (segment 10) or all router pages (segment 3).
    pub fn pages_for_segment(&self, segment: u16) -> Vec<&PageCatalogEntry> {
        self.page_catalog
            .iter()
            .filter(|p| p.expert == 0xFFFF && p.segment == segment)
            .collect()
    }

    /// Get all page catalog entries for a specific (layer, segment) pair of shared weights.
    ///
    /// Returns pages sorted by page_idx (ascending), which corresponds to row order.
    /// This is the primary method for loading multi-page shared tensors like
    /// attention Q/K/V/O projections, embeddings, and lm_head.
    pub fn pages_for_shared_segment(
        &self,
        layer: u16,
        segment: u16,
    ) -> Vec<(usize, &PageCatalogEntry)> {
        // Search within the shared pages for this layer
        let shared_pages = self.pages_for_shared(layer);

        // Find entries matching the requested segment, along with their catalog indices
        let mut results: Vec<(usize, &PageCatalogEntry)> = Vec::new();

        if shared_pages.is_empty() {
            return results;
        }

        // The shared_pages slice starts at an offset in the catalog.
        // We need to compute the actual catalog index for each entry.
        if let Some(entry) = self.expert_entry(layer, 0xFFFF) {
            let base_idx = entry.first_page_idx as usize;
            for (i, page) in shared_pages.iter().enumerate() {
                if page.segment == segment {
                    results.push((base_idx + i, page));
                }
            }
        }

        // Sort by page_idx for correct row ordering
        results.sort_by_key(|(_, p)| p.page_idx);
        results
    }

    // ── Vector Index ─────────────────────────────────────────────────

    /// Access the raw vector index data for the VectorIndex to parse.
    pub fn vector_index_bytes(&self) -> &[u8] {
        let start = self.header.vector_index_offset as usize;
        let end = start + self.header.vector_index_size as usize;
        if end <= self.mmap.len() {
            &self.mmap[start..end]
        } else {
            &[]
        }
    }

    pub fn has_vector_index(&self) -> bool {
        self.header.vector_index_size > 0
    }

    // ── Coactivation ─────────────────────────────────────────────────

    pub fn coactivation_bytes(&self) -> &[u8] {
        let start = self.header.coactivation_offset as usize;
        let end = start + self.header.coactivation_size as usize;
        if end <= self.mmap.len() {
            &self.mmap[start..end]
        } else {
            &[]
        }
    }

    // ── Materialized Views ───────────────────────────────────────────

    pub fn view_count(&self) -> usize {
        self.view_headers.len()
    }

    pub fn view(&self, idx: usize) -> &MaterializedViewHeader {
        &self.view_headers[idx]
    }

    pub fn view_by_name(&self, name: &str) -> Option<&MaterializedViewHeader> {
        self.view_name_map.get(name).map(|&i| &self.view_headers[i])
    }

    // ── Page I/O ─────────────────────────────────────────────────────

    /// Get the file offset and size for a page (for async I/O).
    pub fn page_location(&self, page_idx: usize) -> (u64, u32) {
        let entry = &self.page_catalog[page_idx];
        (entry.file_offset, entry.compressed_size)
    }

    /// Get the underlying file descriptor for io_uring submissions.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Read a page synchronously into a buffer (fallback path).
    ///
    /// Handles LZ4 decompression transparently: reads `compressed_size` bytes
    /// from disk, then decompresses into `buf` if `compression == LZ4`.
    /// Returns the raw (decompressed) size written to `buf`.
    pub fn read_page_sync(&self, page_idx: usize, buf: &mut [u8]) -> Result<usize> {
        let entry = &self.page_catalog[page_idx];
        let offset = { entry.file_offset } as usize;
        let compressed_size = { entry.compressed_size } as usize;
        let raw_size = { entry.raw_size } as usize;
        let compression = entry.compression;

        if buf.len() < raw_size {
            return Err(Error::InvalidFormat {
                reason: format!(
                    "Buffer too small: {} < {} for page {}",
                    buf.len(),
                    raw_size,
                    page_idx
                ),
            });
        }

        // Read from mmap (thread-safe, no seek races)
        let data_end = offset + compressed_size;
        if data_end > self.mmap.len() {
            return Err(Error::InvalidFormat {
                reason: format!(
                    "Page {} data extends past EOF: offset={}, size={}, file_len={}",
                    page_idx,
                    offset,
                    compressed_size,
                    self.mmap.len()
                ),
            });
        }
        let on_disk = &self.mmap[offset..data_end];

        if compression == COMPRESSION_LZ4 {
            let decompressed =
                lz4_flex::decompress_size_prepended(on_disk).map_err(|e| Error::InvalidFormat {
                    reason: format!("LZ4 decompression failed for page {}: {}", page_idx, e),
                })?;

            let copy_len = decompressed.len().min(buf.len());
            buf[..copy_len].copy_from_slice(&decompressed[..copy_len]);
            Ok(copy_len)
        } else if compression == COMPRESSION_ZSTD {
            let decompressed = zstd::bulk::decompress(on_disk, raw_size + 1024).map_err(|e| {
                Error::InvalidFormat {
                    reason: format!("Zstd decompression failed for page {}: {}", page_idx, e),
                }
            })?;

            let copy_len = decompressed.len().min(buf.len());
            buf[..copy_len].copy_from_slice(&decompressed[..copy_len]);
            Ok(copy_len)
        } else {
            // Uncompressed: copy directly from mmap
            let read_size = compressed_size.min(buf.len());
            buf[..read_size].copy_from_slice(&on_disk[..read_size]);
            Ok(read_size)
        }
    }

    /// Read a page's compressed bytes directly (no decompression).
    ///
    /// Used by Pipeline B: the compressed bytes are stored in T2 as-is,
    /// then DMA'd to VRAM where the GPU Decompression Engine handles
    /// decompression at 600 GB/s.
    ///
    /// For uncompressed pages, this is identical to reading raw data.
    pub fn read_page_compressed_sync(&self, page_idx: usize, buf: &mut [u8]) -> Result<usize> {
        let entry = &self.page_catalog[page_idx];
        let offset = { entry.file_offset } as usize;
        let compressed_size = { entry.compressed_size } as usize;

        if buf.len() < compressed_size {
            return Err(Error::InvalidFormat {
                reason: format!(
                    "Buffer too small for compressed data: {} < {} for page {}",
                    buf.len(),
                    compressed_size,
                    page_idx
                ),
            });
        }

        // Read from mmap (thread-safe, no seek races)
        let data_end = offset + compressed_size;
        if data_end > self.mmap.len() {
            return Err(Error::InvalidFormat {
                reason: format!(
                    "Page {} compressed data extends past EOF: offset={}, size={}, file_len={}",
                    page_idx,
                    offset,
                    compressed_size,
                    self.mmap.len()
                ),
            });
        }
        buf[..compressed_size].copy_from_slice(&self.mmap[offset..data_end]);
        Ok(compressed_size)
    }

    /// Check if a specific page is compressed.
    pub fn page_compression(&self, page_idx: usize) -> u8 {
        self.page_catalog[page_idx].compression
    }
}

// ─── File Writer ─────────────────────────────────────────────────────────

/// Compression method for page data in the writer.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CompressionMethod {
    /// No compression.
    #[default]
    None,
    /// LZ4 block compression (~5 GB/s decode, ~2x ratio on weight data).
    Lz4,
    /// Zstd compression (~3.5x ratio on INT4 data, nvCOMP HW decode on Blackwell).
    /// Level 1-3 recommended for model conversion (speed vs ratio tradeoff).
    Zstd { level: i32 },
}

/// Builder for creating .vib3 files.
///
/// Usage:
/// ```ignore
/// let mut writer = Vib3Writer::new(model_config);
/// writer.set_compression(CompressionMethod::Zstd { level: 3 });
/// for (layer, expert, segment, data) in weights {
///     writer.add_page(layer, expert, segment, page_idx, row_start, row_count, &data);
/// }
/// writer.finalize("output.vib3")?;
/// ```
pub struct Vib3Writer {
    model_config: ModelConfig,
    /// Page catalog entries only (48 bytes each) — page data is in temp_file.
    pages: Vec<PageCatalogEntry>,
    /// Byte offset of each page's data within the temp file.
    page_temp_offsets: Vec<u64>,
    /// Byte length of each page's compressed data in the temp file.
    page_temp_sizes: Vec<usize>,
    expert_pages: HashMap<u32, Vec<usize>>, // (layer<<16|expert) -> page indices
    coactivation: Vec<CoactivationEntry>,
    views: Vec<MaterializedViewHeader>,
    /// Vector index centroids and entries
    vector_index_centroids: Vec<Vec<f32>>,
    vector_index_entries: Vec<VectorIndexEntry>,
    /// Compression method for page data
    compression: CompressionMethod,
    /// Temp file for streaming page data to disk instead of buffering in RAM.
    temp_file: Option<std::io::BufWriter<std::fs::File>>,
    temp_path: Option<std::path::PathBuf>,
    temp_offset: u64,
}

impl Vib3Writer {
    pub fn new(model_config: ModelConfig) -> Self {
        Self::with_temp_dir(model_config, None)
    }

    /// Create a new writer, placing the temp file in the specified directory.
    /// If `temp_dir` is `None`, falls back to the system temp directory.
    pub fn with_temp_dir(model_config: ModelConfig, temp_dir: Option<&Path>) -> Self {
        // Create a temp file for streaming page data
        // Use a unique counter + pid to avoid collisions when multiple writers run in parallel
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);

        let temp_dir = temp_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(std::env::temp_dir);
        let temp_path = temp_dir.join(format!("vib3_pages_{}_{}.tmp", std::process::id(), id));
        let temp_file = std::fs::File::create(&temp_path).ok().map(|f| {
            std::io::BufWriter::with_capacity(8 * 1024 * 1024, f) // 8MB buffer
        });
        if temp_file.is_none() {
            eprintln!(
                "Warning: could not create temp file at {:?}, will buffer pages in memory",
                temp_path
            );
        }

        Self {
            model_config,
            pages: Vec::new(),
            page_temp_offsets: Vec::new(),
            page_temp_sizes: Vec::new(),
            expert_pages: HashMap::new(),
            coactivation: Vec::new(),
            views: Vec::new(),
            vector_index_centroids: Vec::new(),
            vector_index_entries: Vec::new(),
            compression: CompressionMethod::None,
            temp_file,
            temp_path: Some(temp_path),
            temp_offset: 0,
        }
    }

    /// Number of pages currently added.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Enable LZ4 compression for page data (backward compat).
    ///
    /// Compressed pages load 2-3x faster from NVMe since the bottleneck
    /// is I/O bandwidth, not CPU. LZ4 decompresses at ~5 GB/s per core.
    pub fn enable_compression(&mut self) {
        self.compression = CompressionMethod::Lz4;
    }

    /// Set the compression method for page data.
    ///
    /// - `Lz4`: ~2x ratio, ~5 GB/s CPU decode. Good for FP16 weights.
    /// - `Zstd { level }`: ~3-5x ratio on INT4 data, ~1-2 GB/s CPU decode.
    ///   On Blackwell GPUs, nvCOMP can decode Zstd at 600 GB/s via the
    ///   hardware Decompression Engine, making this the preferred format.
    pub fn set_compression(&mut self, method: CompressionMethod) {
        self.compression = method;
    }

    /// Compress a data block according to the configured method.
    fn compress_block(&self, data: &[u8]) -> (Vec<u8>, u32, u8) {
        if data.len() <= 64 {
            return (data.to_vec(), data.len() as u32, COMPRESSION_NONE);
        }

        match self.compression {
            CompressionMethod::None => (data.to_vec(), data.len() as u32, COMPRESSION_NONE),
            CompressionMethod::Lz4 => {
                let compressed = lz4_flex::compress_prepend_size(data);
                if compressed.len() < data.len() {
                    let len = compressed.len() as u32;
                    (compressed, len, COMPRESSION_LZ4)
                } else {
                    (data.to_vec(), data.len() as u32, COMPRESSION_NONE)
                }
            }
            CompressionMethod::Zstd { level } => match zstd::bulk::compress(data, level) {
                Ok(compressed) if compressed.len() < data.len() => {
                    let len = compressed.len() as u32;
                    (compressed, len, COMPRESSION_ZSTD)
                }
                _ => (data.to_vec(), data.len() as u32, COMPRESSION_NONE),
            },
        }
    }

    /// Add a page of weight data.
    ///
    /// If compression is enabled, the page is compressed. The catalog
    /// entry records both `compressed_size` and `raw_size` so the loader
    /// knows how many bytes to read and how large the decompressed buffer is.
    #[allow(clippy::too_many_arguments)]
    pub fn add_page(
        &mut self,
        layer: u16,
        expert: u16,
        segment: u16,
        page_idx: u16,
        row_start: u16,
        row_count: u16,
        col_count: u16,
        data: &[u8],
    ) {
        let idx = self.pages.len();
        let key = (layer as u32) << 16 | expert as u32;
        self.expert_pages.entry(key).or_default().push(idx);

        let raw_size = data.len() as u32;
        let (stored_data, compressed_size, compression) = self.compress_block(data);

        let entry = PageCatalogEntry {
            layer,
            expert,
            segment,
            page_idx,
            file_offset: 0, // Will be set during finalize
            compressed_size,
            raw_size,
            row_start,
            row_count,
            col_start: 0,
            col_count,
            mean_activation: 0.0,
            access_frequency: 0.0,
            coactivation_cluster: 0,
            compression,
            flags: 0,
            _pad: [0u8; 4],
        };

        // Stream page data to temp file instead of buffering in RAM
        if let Some(ref mut tf) = self.temp_file {
            use std::io::Write;
            let offset = self.temp_offset;
            tf.write_all(&stored_data)
                .expect("failed to write page to temp file");
            self.page_temp_offsets.push(offset);
            self.page_temp_sizes.push(stored_data.len());
            self.temp_offset += stored_data.len() as u64;
        } else {
            // Fallback: buffer in memory (original behavior, for small models/tests)
            self.page_temp_offsets.push(0);
            self.page_temp_sizes.push(stored_data.len());
        }

        self.pages.push(entry);
    }

    /// Add a coactivation entry.
    pub fn add_coactivation(
        &mut self,
        layer: u16,
        expert_a: u16,
        expert_b: u16,
        correlation: f32,
        sample_count: u32,
    ) {
        self.coactivation.push(CoactivationEntry {
            expert_a,
            expert_b,
            layer,
            _pad: 0,
            correlation,
            sample_count,
        });
    }

    /// Set the vector index data.
    ///
    /// Centroids: list of centroid vectors (all same dimension).
    /// Entries: one VectorIndexEntry per centroid, containing expert predictions
    /// and hot page lists.
    pub fn set_vector_index(&mut self, centroids: Vec<Vec<f32>>, entries: Vec<VectorIndexEntry>) {
        self.vector_index_centroids = centroids;
        self.vector_index_entries = entries;
    }

    /// Finalize and write the .vib3 file.
    pub fn finalize(&mut self, path: impl AsRef<Path>) -> Result<()> {
        use std::io::{Read, Seek, Write};

        // Flush and close the temp file writer, then reopen for reading
        let has_temp = self.temp_file.is_some();
        if let Some(ref mut tf) = self.temp_file {
            tf.flush()
                .map_err(|e| Error::ConversionError(format!("Failed to flush temp: {e}")))?;
        }
        // Drop the BufWriter + File to ensure all data is synced to disk
        self.temp_file = None;

        let mut temp_reader = if has_temp {
            let tp = self.temp_path.as_ref().unwrap();
            eprintln!(
                "  Reopening temp file: {:?} (size: {} bytes)",
                tp, self.temp_offset
            );
            Some(std::io::BufReader::with_capacity(
                8 * 1024 * 1024,
                std::fs::File::open(tp).map_err(|e| {
                    Error::ConversionError(format!("Failed to reopen temp file {:?}: {e}", tp))
                })?,
            ))
        } else {
            None
        };

        let mut file = std::fs::File::create(path.as_ref())?;

        // 1. Serialize model config as JSON
        let metadata_json = serde_json::to_vec_pretty(&self.model_config)
            .map_err(|e| Error::ConversionError(format!("Failed to serialize config: {e}")))?;

        // 2. Build expert index (sorted by layer, expert)
        let mut expert_index: Vec<ExpertIndexEntry> = Vec::new();
        let mut sorted_keys: Vec<u32> = self.expert_pages.keys().copied().collect();
        sorted_keys.sort();

        // Re-index pages: sort by (layer, expert, segment, page_idx)
        let mut sorted_page_indices: Vec<usize> = (0..self.pages.len()).collect();
        sorted_page_indices.sort_by(|&a, &b| {
            let pa = &self.pages[a];
            let pb = &self.pages[b];
            (pa.layer, pa.expert, pa.segment, pa.page_idx).cmp(&(
                pb.layer,
                pb.expert,
                pb.segment,
                pb.page_idx,
            ))
        });

        // Build sorted catalog entries and temp file read plan
        let sorted_entries: Vec<PageCatalogEntry> = sorted_page_indices
            .iter()
            .map(|&old_idx| self.pages[old_idx])
            .collect();
        let sorted_temp_offsets: Vec<u64> = sorted_page_indices
            .iter()
            .map(|&old_idx| self.page_temp_offsets[old_idx])
            .collect();
        let sorted_temp_sizes: Vec<usize> = sorted_page_indices
            .iter()
            .map(|&old_idx| self.page_temp_sizes[old_idx])
            .collect();

        // Rebuild expert_pages based on sorted order
        let mut expert_pages_sorted: HashMap<u32, Vec<usize>> = HashMap::new();
        for (new_idx, entry) in sorted_entries.iter().enumerate() {
            let key = (entry.layer as u32) << 16 | entry.expert as u32;
            expert_pages_sorted.entry(key).or_default().push(new_idx);
        }

        // Build expert index entries
        for &key in &sorted_keys {
            let empty = Vec::new();
            let indices = expert_pages_sorted.get(&key).unwrap_or(&empty);
            if indices.is_empty() {
                continue;
            }
            let first = indices[0];
            let count = indices.len();
            let total_bytes: u32 = indices.iter().map(|&i| sorted_entries[i].raw_size).sum();

            // Count distinct segments for this expert
            let mut segments: std::collections::HashSet<u16> = std::collections::HashSet::new();
            for &i in indices {
                segments.insert(sorted_entries[i].segment);
            }

            expert_index.push(ExpertIndexEntry {
                layer: (key >> 16) as u16,
                expert: (key & 0xFFFF) as u16,
                first_page_idx: first as u32,
                page_count: count as u32,
                total_bytes,
                num_segments: segments.len() as u16,
                _reserved: 0,
            });
        }

        let num_pages = sorted_entries.len();

        // 3. Calculate section offsets
        let metadata_offset = HEADER_SIZE as u64;
        let metadata_size = metadata_json.len() as u64;

        let catalog_offset = metadata_offset + metadata_size;
        let catalog_entry_size = std::mem::size_of::<PageCatalogEntry>();
        let catalog_size = num_pages as u64 * catalog_entry_size as u64;

        let ei_offset = catalog_offset + catalog_size;
        let ei_entry_size = std::mem::size_of::<ExpertIndexEntry>();
        let ei_size = expert_index.len() as u64 * ei_entry_size as u64;

        let vi_offset = ei_offset + ei_size;
        let vi_size = if !self.vector_index_centroids.is_empty()
            && !self.vector_index_entries.is_empty()
        {
            let centroid_dim = self.vector_index_centroids[0].len();
            let centroid_count = self.vector_index_centroids.len();
            // Header: [u32 centroid_count, u32 centroid_dim]
            // + centroid data: [f32; centroid_count * centroid_dim]
            // + entries: [VectorIndexEntry; entry_count]
            8u64 + (centroid_count * centroid_dim * 4) as u64
                + (self.vector_index_entries.len() * std::mem::size_of::<VectorIndexEntry>()) as u64
        } else {
            0u64
        };

        let coact_offset = vi_offset + vi_size;
        let coact_entry_size = std::mem::size_of::<CoactivationEntry>();
        let coact_size = self.coactivation.len() as u64 * coact_entry_size as u64;

        let vc_offset = coact_offset + coact_size;
        let vc_entry_size = std::mem::size_of::<MaterializedViewHeader>();
        let vc_size = self.views.len() as u64 * vc_entry_size as u64;

        // Align page data to PAGE_SIZE
        let raw_page_data_offset = vc_offset + vc_size;
        let page_data_offset = raw_page_data_offset.div_ceil(PAGE_SIZE as u64) * PAGE_SIZE as u64;

        // 4. Calculate page file offsets from sorted temp sizes
        let mut page_file_offsets: Vec<u64> = Vec::with_capacity(num_pages);
        let mut current_offset = page_data_offset;
        for i in 0..num_pages {
            page_file_offsets.push(current_offset);
            // Align each page to 4096 bytes for direct I/O
            let aligned_size = (sorted_temp_sizes[i] as u64).div_ceil(4096) * 4096;
            current_offset += aligned_size;
        }
        let page_data_size = current_offset - page_data_offset;

        // Update catalog entries with final file offsets
        let mut final_entries = sorted_entries;
        for (i, entry) in final_entries.iter_mut().enumerate() {
            entry.file_offset = page_file_offsets[i];
        }

        // 5. Build header
        let header = Vib3Header {
            magic: VIB3_MAGIC,
            version: VIB3_VERSION,
            flags: 0,
            metadata_offset,
            metadata_size,
            page_catalog_offset: catalog_offset,
            page_catalog_size: catalog_size,
            page_catalog_count: num_pages as u32,
            expert_index_offset: ei_offset,
            expert_index_size: ei_size,
            vector_index_offset: vi_offset,
            vector_index_size: vi_size,
            coactivation_offset: coact_offset,
            coactivation_size: coact_size,
            view_catalog_offset: vc_offset,
            view_catalog_size: vc_size,
            view_count: self.views.len() as u32,
            page_data_offset,
            page_data_size,
            num_layers: self.model_config.num_layers,
            num_experts: self.model_config.num_experts,
            num_active_experts: self.model_config.num_active_experts,
            hidden_dim: self.model_config.hidden_dim,
            expert_hidden_dim: self.model_config.expert_hidden_dim,
            expert_dtype: self.model_config.expert_dtype as u8,
            shared_dtype: self.model_config.shared_dtype as u8,
            _reserved: [0u8; 354],
        };

        // 6. Write everything
        // Header
        file.write_all(bytemuck::bytes_of(&header))?;

        // Metadata JSON
        file.write_all(&metadata_json)?;

        // Page catalog
        for entry in &final_entries {
            file.write_all(bytemuck::bytes_of(entry))?;
        }

        // Expert index
        for entry in &expert_index {
            file.write_all(bytemuck::bytes_of(entry))?;
        }

        // Vector index
        if !self.vector_index_centroids.is_empty() && !self.vector_index_entries.is_empty() {
            let centroid_count = self.vector_index_centroids.len() as u32;
            let centroid_dim = self.vector_index_centroids[0].len() as u32;
            file.write_all(&centroid_count.to_le_bytes())?;
            file.write_all(&centroid_dim.to_le_bytes())?;

            for centroid in &self.vector_index_centroids {
                for &val in centroid {
                    file.write_all(&val.to_le_bytes())?;
                }
            }

            for entry in &self.vector_index_entries {
                file.write_all(bytemuck::bytes_of(entry))?;
            }
        }

        // Coactivation table
        for entry in &self.coactivation {
            file.write_all(bytemuck::bytes_of(entry))?;
        }

        // View catalog
        for view in &self.views {
            file.write_all(bytemuck::bytes_of(view))?;
        }

        // Padding to page_data_offset
        let current_pos = HEADER_SIZE as u64
            + metadata_size
            + catalog_size
            + ei_size
            + vi_size
            + coact_size
            + vc_size;
        let padding = (page_data_offset - current_pos) as usize;
        if padding > 0 {
            file.write_all(&vec![0u8; padding])?;
        }

        // Page data: stream from temp file in sorted order (with alignment padding)
        let zero_buf = vec![0u8; 4096]; // reusable padding buffer
        if let Some(ref mut reader) = temp_reader {
            // Streaming path: read page data from temp file
            let mut read_buf = vec![0u8; PAGE_SIZE]; // reusable read buffer (2MB)
            for i in 0..num_pages {
                let temp_off = sorted_temp_offsets[i];
                let temp_sz = sorted_temp_sizes[i];

                // Seek to this page's data in the temp file
                reader
                    .seek(std::io::SeekFrom::Start(temp_off))
                    .map_err(|e| {
                        Error::ConversionError(format!(
                            "Failed to seek temp file to offset {} for page {}: {}",
                            temp_off, i, e
                        ))
                    })?;

                // Read page data in chunks and write to output
                let mut remaining = temp_sz;
                while remaining > 0 {
                    let to_read = remaining.min(read_buf.len());
                    reader.read_exact(&mut read_buf[..to_read]).map_err(|e| {
                        Error::ConversionError(format!(
                            "Failed to read {} bytes from temp file for page {}: {}",
                            to_read, i, e
                        ))
                    })?;
                    file.write_all(&read_buf[..to_read])?;
                    remaining -= to_read;
                }

                // Write alignment padding
                let aligned_size = temp_sz.div_ceil(4096) * 4096;
                let page_padding = aligned_size - temp_sz;
                if page_padding > 0 {
                    file.write_all(&zero_buf[..page_padding])?;
                }
            }
        } else {
            // No temp file — this shouldn't happen for large models, but handle gracefully.
            // All data was "buffered" but we have no actual data (the fallback path in add_page
            // doesn't store data). This path exists for API completeness but will produce
            // an empty page data section.
            eprintln!("Warning: no temp file available, page data section will be empty");
        }

        file.flush()?;

        // Clean up temp file
        if let Some(ref tp) = self.temp_path {
            let _ = std::fs::remove_file(tp);
        }

        tracing::info!(
            "Wrote .vib3 file: {} pages, {} experts, {:.1} MB",
            num_pages,
            expert_index.len(),
            (page_data_offset + page_data_size) as f64 / (1024.0 * 1024.0),
        );

        Ok(())
    }
}
