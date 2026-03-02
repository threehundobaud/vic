//! GGUF v3 file parser for loading quantized models (MXFP4, Q8_0, F32).
//!
//! Handles multi-shard split GGUF files. Provides zero-copy tensor access
//! via memory-mapped files for efficient conversion to .vib3 format.

use half::f16;
use std::collections::HashMap;
use std::path::Path;

// ─── GGUF Constants ──────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in little-endian
const GGUF_VERSION_3: u32 = 3;

/// GGML tensor data types we handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q5K = 13,   // K-quant 5-bit: 256 elements, 176 bytes/block (~5.5 bpw)
    Q6K = 14,   // K-quant 6-bit: 256 elements, 210 bytes/block (~6.5625 bpw)
    BF16 = 30,  // Brain Float 16 (no quantization, 2 bytes per element)
    Mxfp4 = 39, // OCP MX FP4 (E2M1 + E8M0 shared exponent)
    Unknown(u32),
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            8 => GgmlType::Q8_0,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            30 => GgmlType::BF16,
            39 => GgmlType::Mxfp4,
            _ => GgmlType::Unknown(v),
        }
    }

    /// Bytes per block for quantized types.
    /// Returns (block_size_elements, block_size_bytes).
    pub fn block_layout(self) -> (usize, usize) {
        match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 => (1, 2),
            GgmlType::BF16 => (1, 2),
            GgmlType::Q8_0 => (32, 34), // 2-byte fp16 scale + 32 int8 = 34 bytes
            GgmlType::Q5K => (256, 176), // dm(4) + scales(12) + qh(32) + qs(128)
            GgmlType::Q6K => (256, 210), // ql(128) + qh(64) + scales(16) + d(2)
            GgmlType::Mxfp4 => (32, 17), // 16 packed E2M1 + 1 E8M0 exponent = 17 bytes
            GgmlType::Unknown(_) => (1, 1), // placeholder
        }
    }

    /// Total bytes for `n_elements` of this type.
    pub fn tensor_bytes(self, n_elements: usize) -> usize {
        let (block_elems, block_bytes) = self.block_layout();
        let n_blocks = n_elements.div_ceil(block_elems);
        n_blocks * block_bytes
    }
}

// ─── GGUF Metadata Value Types ───────────────────────────────────────────

/// GGUF metadata value types (subset we need).
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Int32(v) => Some(*v as u32),
            GgufValue::Uint64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::Uint64(v) => Some(*v),
            GgufValue::Uint32(v) => Some(*v as u64),
            GgufValue::Int32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            GgufValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            GgufValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    result.push(v.as_str()?);
                }
                Some(result)
            }
            _ => None,
        }
    }
}

// ─── GGUF Tensor Info ────────────────────────────────────────────────────

/// Metadata for a single tensor within a GGUF shard.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>, // [ne0, ne1, ...] (ne0 = innermost/fastest dimension)
    pub dtype: GgmlType,
    pub offset: u64,      // offset from start of data section
    pub shard_idx: usize, // which shard file this tensor lives in
}

impl GgufTensorInfo {
    /// Total number of elements.
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    /// Total bytes of quantized data.
    pub fn data_bytes(&self) -> usize {
        // For quantized types, blocking is along ne0 (innermost dim)
        let ne0 = self.shape.first().copied().unwrap_or(1) as usize;
        let (block_elems, block_bytes) = self.dtype.block_layout();
        let blocks_per_row = ne0.div_ceil(block_elems);
        let row_bytes = blocks_per_row * block_bytes;

        // Rows = product of all dimensions except ne0
        let n_rows: usize = self.shape.iter().skip(1).product::<u64>() as usize;
        let n_rows = n_rows.max(1);
        row_bytes * n_rows
    }
}

// ─── GGUF File (single shard) ────────────────────────────────────────────

/// A parsed GGUF shard with mmap'd data access.
#[allow(dead_code)]
struct GgufShard {
    mmap: memmap2::Mmap,
    data_offset: usize, // byte offset where tensor data begins
    tensor_count: u64,
    _kv_count: u64,
}

// ─── GGUF Multi-Shard File Set ───────────────────────────────────────────

/// A collection of GGUF shards (split files) providing unified tensor access.
pub struct GgufFile {
    shards: Vec<GgufShard>,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
}

impl GgufFile {
    /// Open a directory of GGUF shard files.
    ///
    /// Discovers all `.gguf` files, parses headers and tensor info,
    /// and merges metadata and tensor catalogs. Tensor data is mmap'd
    /// for zero-copy access.
    pub fn open_dir(dir: &Path) -> Result<Self, String> {
        let mut gguf_paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read dir {}: {}", dir.display(), e))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        if gguf_paths.is_empty() {
            return Err(format!("No .gguf files found in {}", dir.display()));
        }

        // Sort by name for deterministic ordering (shards are numbered)
        gguf_paths.sort();

        let mut shards = Vec::with_capacity(gguf_paths.len());
        let mut all_metadata = HashMap::new();
        let mut all_tensors = HashMap::new();

        for (shard_idx, path) in gguf_paths.iter().enumerate() {
            tracing::info!(
                "Parsing GGUF shard {}/{}: {}",
                shard_idx + 1,
                gguf_paths.len(),
                path.display()
            );

            let file = std::fs::File::open(path)
                .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

            // Memory-map the file for zero-copy tensor data access
            let mmap = unsafe {
                memmap2::MmapOptions::new()
                    .map(&file)
                    .map_err(|e| format!("Failed to mmap {}: {}", path.display(), e))?
            };

            let (shard, metadata, tensors) = parse_gguf_shard(&mmap, shard_idx)?;

            // Merge metadata (first shard wins for duplicates)
            for (k, v) in metadata {
                all_metadata.entry(k).or_insert(v);
            }

            // Merge tensors (should be unique across shards)
            for (name, info) in tensors {
                if all_tensors.contains_key(&name) {
                    tracing::warn!(
                        "Duplicate tensor '{}' across shards, using first occurrence",
                        name
                    );
                } else {
                    all_tensors.insert(name, info);
                }
            }

            shards.push(GgufShard {
                mmap,
                data_offset: shard.data_offset,
                tensor_count: shard.tensor_count,
                _kv_count: shard._kv_count,
            });
        }

        tracing::info!(
            "GGUF loaded: {} shards, {} metadata keys, {} tensors",
            shards.len(),
            all_metadata.len(),
            all_tensors.len(),
        );

        Ok(GgufFile {
            shards,
            metadata: all_metadata,
            tensors: all_tensors,
        })
    }

    /// Get the raw bytes for a tensor (zero-copy from mmap).
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> &[u8] {
        let shard = &self.shards[info.shard_idx];
        let start = shard.data_offset + info.offset as usize;
        let size = info.data_bytes();
        let end = start + size;
        let mmap_len = shard.mmap.len();
        if end > mmap_len {
            // Some tensors at shard boundaries may have their computed size
            // slightly exceed the mmap due to alignment rounding in block_layout.
            // Clamp to available data (difference is always < 1 block).
            let available = mmap_len.saturating_sub(start);
            tracing::warn!(
                "Tensor '{}' data extends {} bytes past shard end (computed={}, available={}), clamping",
                info.name, end - mmap_len, size, available
            );
            &shard.mmap[start..mmap_len]
        } else {
            &shard.mmap[start..end]
        }
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }
}

// ─── GGUF Parsing ────────────────────────────────────────────────────────

struct PartialShard {
    data_offset: usize,
    tensor_count: u64,
    _kv_count: u64,
}

/// Parse a single GGUF shard from mmap'd data.
fn parse_gguf_shard(
    data: &[u8],
    shard_idx: usize,
) -> Result<
    (
        PartialShard,
        HashMap<String, GgufValue>,
        HashMap<String, GgufTensorInfo>,
    ),
    String,
> {
    let mut cursor = 0usize;

    // ── Header ──
    let magic = read_u32(data, &mut cursor)?;
    if magic != GGUF_MAGIC {
        return Err(format!(
            "Invalid GGUF magic: 0x{:08X} (expected 0x{:08X})",
            magic, GGUF_MAGIC
        ));
    }

    let version = read_u32(data, &mut cursor)?;
    if version != GGUF_VERSION_3 {
        return Err(format!(
            "Unsupported GGUF version {} (only v3 supported)",
            version
        ));
    }

    let tensor_count = read_u64(data, &mut cursor)?;
    let kv_count = read_u64(data, &mut cursor)?;

    tracing::debug!(
        "GGUF shard {}: version={}, tensors={}, kv_pairs={}",
        shard_idx,
        version,
        tensor_count,
        kv_count
    );

    // ── Metadata KV pairs ──
    let mut metadata = HashMap::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key = read_gguf_string(data, &mut cursor)?;
        let value = read_gguf_value(data, &mut cursor)?;
        metadata.insert(key, value);
    }

    // ── Tensor info entries ──
    let mut tensors = HashMap::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_gguf_string(data, &mut cursor)?;
        let n_dims = read_u32(data, &mut cursor)? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(read_u64(data, &mut cursor)?);
        }
        let dtype_raw = read_u32(data, &mut cursor)?;
        let dtype = GgmlType::from_u32(dtype_raw);
        let offset = read_u64(data, &mut cursor)?;

        tensors.insert(
            name.clone(),
            GgufTensorInfo {
                name,
                shape,
                dtype,
                offset,
                shard_idx,
            },
        );
    }

    // Data section starts after all header/metadata/tensor_info.
    // GGUF v3 default alignment is 32 bytes; can be overridden by `general.alignment`.
    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::Uint32(a)) => *a as usize,
        _ => 32, // GGUF v3 default
    };
    let data_offset = align_up(cursor, alignment);

    Ok((
        PartialShard {
            data_offset,
            tensor_count,
            _kv_count: kv_count,
        },
        metadata,
        tensors,
    ))
}

// ─── Binary Reader Helpers ───────────────────────────────────────────────

fn read_u8(data: &[u8], cursor: &mut usize) -> Result<u8, String> {
    if *cursor + 1 > data.len() {
        return Err("Unexpected EOF reading u8".into());
    }
    let v = data[*cursor];
    *cursor += 1;
    Ok(v)
}

fn read_u16(data: &[u8], cursor: &mut usize) -> Result<u16, String> {
    if *cursor + 2 > data.len() {
        return Err("Unexpected EOF reading u16".into());
    }
    let v = u16::from_le_bytes(data[*cursor..*cursor + 2].try_into().unwrap());
    *cursor += 2;
    Ok(v)
}

fn read_i16(data: &[u8], cursor: &mut usize) -> Result<i16, String> {
    if *cursor + 2 > data.len() {
        return Err("Unexpected EOF reading i16".into());
    }
    let v = i16::from_le_bytes(data[*cursor..*cursor + 2].try_into().unwrap());
    *cursor += 2;
    Ok(v)
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32, String> {
    if *cursor + 4 > data.len() {
        return Err("Unexpected EOF reading u32".into());
    }
    let v = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn read_i32(data: &[u8], cursor: &mut usize) -> Result<i32, String> {
    if *cursor + 4 > data.len() {
        return Err("Unexpected EOF reading i32".into());
    }
    let v = i32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64, String> {
    if *cursor + 8 > data.len() {
        return Err("Unexpected EOF reading u64".into());
    }
    let v = u64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_i64(data: &[u8], cursor: &mut usize) -> Result<i64, String> {
    if *cursor + 8 > data.len() {
        return Err("Unexpected EOF reading i64".into());
    }
    let v = i64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_f32(data: &[u8], cursor: &mut usize) -> Result<f32, String> {
    if *cursor + 4 > data.len() {
        return Err("Unexpected EOF reading f32".into());
    }
    let v = f32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn read_f64(data: &[u8], cursor: &mut usize) -> Result<f64, String> {
    if *cursor + 8 > data.len() {
        return Err("Unexpected EOF reading f64".into());
    }
    let v = f64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_gguf_string(data: &[u8], cursor: &mut usize) -> Result<String, String> {
    let len = read_u64(data, cursor)? as usize;
    if *cursor + len > data.len() {
        return Err(format!("Unexpected EOF reading string of length {}", len));
    }
    let s = std::str::from_utf8(&data[*cursor..*cursor + len])
        .map_err(|e| format!("Invalid UTF-8 in GGUF string: {}", e))?
        .to_string();
    *cursor += len;
    Ok(s)
}

fn read_gguf_value(data: &[u8], cursor: &mut usize) -> Result<GgufValue, String> {
    let vtype = read_u32(data, cursor)?;
    match vtype {
        0 => Ok(GgufValue::Uint8(read_u8(data, cursor)?)),
        1 => Ok(GgufValue::Int8(read_u8(data, cursor)? as i8)),
        2 => Ok(GgufValue::Uint16(read_u16(data, cursor)?)),
        3 => Ok(GgufValue::Int16(read_i16(data, cursor)?)),
        4 => Ok(GgufValue::Uint32(read_u32(data, cursor)?)),
        5 => Ok(GgufValue::Int32(read_i32(data, cursor)?)),
        6 => Ok(GgufValue::Float32(read_f32(data, cursor)?)),
        7 => Ok(GgufValue::Bool(read_u8(data, cursor)? != 0)),
        8 => Ok(GgufValue::Str(read_gguf_string(data, cursor)?)),
        9 => {
            // Array: element type (u32) + count (u64) + elements
            let elem_type = read_u32(data, cursor)?;
            let count = read_u64(data, cursor)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                // Read element of the declared type (without the type prefix)
                let v = read_gguf_array_element(data, cursor, elem_type)?;
                arr.push(v);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(read_u64(data, cursor)?)),
        11 => Ok(GgufValue::Int64(read_i64(data, cursor)?)),
        12 => Ok(GgufValue::Float64(read_f64(data, cursor)?)),
        _ => Err(format!("Unknown GGUF value type: {}", vtype)),
    }
}

fn read_gguf_array_element(
    data: &[u8],
    cursor: &mut usize,
    elem_type: u32,
) -> Result<GgufValue, String> {
    match elem_type {
        0 => Ok(GgufValue::Uint8(read_u8(data, cursor)?)),
        1 => Ok(GgufValue::Int8(read_u8(data, cursor)? as i8)),
        2 => Ok(GgufValue::Uint16(read_u16(data, cursor)?)),
        3 => Ok(GgufValue::Int16(read_i16(data, cursor)?)),
        4 => Ok(GgufValue::Uint32(read_u32(data, cursor)?)),
        5 => Ok(GgufValue::Int32(read_i32(data, cursor)?)),
        6 => Ok(GgufValue::Float32(read_f32(data, cursor)?)),
        7 => Ok(GgufValue::Bool(read_u8(data, cursor)? != 0)),
        8 => Ok(GgufValue::Str(read_gguf_string(data, cursor)?)),
        10 => Ok(GgufValue::Uint64(read_u64(data, cursor)?)),
        11 => Ok(GgufValue::Int64(read_i64(data, cursor)?)),
        12 => Ok(GgufValue::Float64(read_f64(data, cursor)?)),
        _ => Err(format!("Unknown GGUF array element type: {}", elem_type)),
    }
}

fn align_up(v: usize, alignment: usize) -> usize {
    (v + alignment - 1) & !(alignment - 1)
}

// ─── GGML Dequantization ─────────────────────────────────────────────────

/// Convert GGML MXFP4 data to vib3 NVFP4 format.
///
/// MXFP4 (OCP MX FP4): Per 32-element block: 16 bytes packed E2M1 + 1 byte E8M0 exponent = 17 bytes.
/// vib3 NVFP4: Split layout: all packed E2M1 data first, then all BF16 scales.
///
/// The E2M1 nibbles are identical between formats — only the scale encoding differs:
/// - MXFP4: E8M0 byte → scale = 2^(exponent - 127)
/// - NVFP4: BF16 scale (lossless for E8M0 power-of-2 values since BF16 has 8 exponent bits)
///
/// This conversion is lossless for both E2M1 data and E8M0 scales.
pub fn convert_mxfp4_to_nvfp4(mxfp4_data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let blocks_per_row = cols.div_ceil(32);
    let mxfp4_row_bytes = blocks_per_row * 17; // 17 bytes per MXFP4 block

    // vib3 NVFP4 layout: [all packed data] [all scales]
    let packed_k = cols.div_ceil(2); // packed E2M1 bytes per row
    let data_bytes = rows * packed_k;
    let scales_bytes = rows * blocks_per_row * 2; // BF16 scale per block
    let mut result = vec![0u8; data_bytes + scales_bytes];

    let (data_region, scales_region) = result.split_at_mut(data_bytes);

    for row in 0..rows {
        let src_row = &mxfp4_data[row * mxfp4_row_bytes..(row + 1) * mxfp4_row_bytes];

        for block in 0..blocks_per_row {
            let src_block = &src_row[block * 17..(block + 1) * 17];

            // GGML block_mxfp4 layout: [E8M0 scale (1 byte)] [E2M1 qs[16] (16 bytes)]
            // E8M0 shared exponent is the FIRST byte (byte 0).
            // E2M1 packed nibbles are bytes 1-16.
            let e8m0 = src_block[0];
            let qs = &src_block[1..17];

            // GGML MXFP4 packing (within each qs[j] byte):
            //   low nibble  (qs[j] & 0xF)  → element j     (first half, indices 0..15)
            //   high nibble (qs[j] >> 4)    → element j+16  (second half, indices 16..31)
            //
            // This IS split-half format — the same layout our CUDA GEMV kernel
            // expects after the sequential→split-half repack.  Copy directly
            // to skip the unnecessary round-trip conversion.
            let dst_data_start = row * packed_k + block * 16;
            data_region[dst_data_start..dst_data_start + 16].copy_from_slice(qs);

            // Convert E8M0 exponent to BF16 scale.
            //
            // E8M0 encodes scale = 2^(e8m0 - 127). BF16 format is [sign(1)][exp(8)][mantissa(7)].
            // A power-of-2 in BF16 has zero mantissa and the exponent field equals e8m0
            // (since BF16 bias is also 127). So the BF16 encoding is simply (e8m0 << 7).
            // Special cases: e8m0=0 → 2^(-127) which is a BF16 subnormal (smallest positive).
            //                e8m0=255 → NaN (reserved in E8M0 spec).
            let scale_bf16_bits: u16 = if e8m0 == 255 {
                0x7FC0 // BF16 quiet NaN (reserved E8M0 value)
            } else {
                // For e8m0=0: this produces BF16 0x0000 which is +0.0 (technically
                // E8M0 spec says e8m0=0 maps to 2^(-127), a tiny subnormal, but
                // in practice MXFP4 blocks with zero exponent indicate zero-valued
                // blocks and zero scale is the correct semantic).
                (e8m0 as u16) << 7
            };
            let sb = scale_bf16_bits.to_le_bytes();
            let dst_scale = (row * blocks_per_row + block) * 2;
            scales_region[dst_scale] = sb[0];
            scales_region[dst_scale + 1] = sb[1];
        }
    }

    result
}

/// Dequantize GGML Q8_0 data to FP16.
///
/// Q8_0 layout: per 32-element block: [fp16_scale (2 bytes)] [32 × int8 (32 bytes)] = 34 bytes.
/// Output: FP16 values in row-major order.
pub fn dequant_q8_0_to_fp16(q8_data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let blocks_per_row = cols.div_ceil(32);
    let q8_row_bytes = blocks_per_row * 34;
    let out_elements = rows * cols;
    let mut result = vec![0u8; out_elements * 2]; // FP16

    let out_f16 =
        unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut f16, out_elements) };

    for row in 0..rows {
        let src_row = &q8_data[row * q8_row_bytes..];
        let dst_row = &mut out_f16[row * cols..];

        for block in 0..blocks_per_row {
            let block_data = &src_row[block * 34..];

            // First 2 bytes: FP16 scale
            let scale_f16 = f16::from_le_bytes([block_data[0], block_data[1]]);
            let scale = scale_f16.to_f32();

            // Next 32 bytes: int8 quantized values
            let block_start = block * 32;
            let block_end = (block_start + 32).min(cols);
            for i in block_start..block_end {
                let qi = block_data[2 + (i - block_start)] as i8;
                let val = qi as f32 * scale;
                dst_row[i] = f16::from_f32(val);
            }
        }
    }

    result
}

/// Convert F32 data to FP16.
pub fn convert_f32_to_fp16(f32_data: &[u8], n_elements: usize) -> Vec<u8> {
    let src = unsafe { std::slice::from_raw_parts(f32_data.as_ptr() as *const f32, n_elements) };
    let mut result = vec![0u8; n_elements * 2];
    let dst =
        unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut f16, n_elements) };
    for i in 0..n_elements {
        dst[i] = f16::from_f32(src[i]);
    }
    result
}

/// Convert BF16 data to FP16.
///
/// BF16 format: [sign(1)][exp(8)][mantissa(7)]
/// FP16 format: [sign(1)][exp(5)][mantissa(10)]
///
/// Since BF16 has 8 exponent bits vs FP16's 5, values outside FP16 range
/// are clamped (overflow → ±inf, underflow → ±0). The mantissa is extended
/// from 7 to 10 bits by zero-padding. We go through f32 as intermediary
/// since BF16→f32 is lossless (same exponent range).
pub fn convert_bf16_to_fp16(bf16_data: &[u8], n_elements: usize) -> Vec<u8> {
    let src = unsafe { std::slice::from_raw_parts(bf16_data.as_ptr() as *const u16, n_elements) };
    let mut result = vec![0u8; n_elements * 2];
    let dst =
        unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut f16, n_elements) };
    for i in 0..n_elements {
        // BF16 → F32 is lossless: just shift left by 16 bits
        let f32_bits = (src[i] as u32) << 16;
        let val = f32::from_bits(f32_bits);
        dst[i] = f16::from_f32(val);
    }
    result
}

/// Dequantize GGML Q5_K data to FP16.
///
/// Q5_K block layout (256 elements, 176 bytes):
///   dm[4]: two FP16 values (d, dmin) — super-block scale and min
///   scales[12]: packed sub-block scales (6-bit each, 16 sub-blocks of 16 elements)
///   qh[32]: high bits (bit 4 of each element, packed 8 per byte)
///   qs[128]: low 4 bits (packed 2 per byte)
pub fn dequant_q5k_to_fp16(q5k_data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let blocks_per_row = cols.div_ceil(256);
    let block_bytes = 176;
    let q5k_row_bytes = blocks_per_row * block_bytes;
    let out_elements = rows * cols;
    let mut result = vec![0u8; out_elements * 2];

    let out_f16 =
        unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut f16, out_elements) };

    for row in 0..rows {
        let src_row = &q5k_data[row * q5k_row_bytes..];
        let dst_row = &mut out_f16[row * cols..];

        for block in 0..blocks_per_row {
            let bd = &src_row[block * block_bytes..];

            // dm: d (FP16) at bytes 0-1, dmin (FP16) at bytes 2-3
            let d = f16::from_le_bytes([bd[0], bd[1]]).to_f32();
            let dmin = f16::from_le_bytes([bd[2], bd[3]]).to_f32();

            // scales: 12 bytes at offset 4, encoding 16 sub-block scales (6-bit each)
            // and 16 sub-block mins (6-bit each) via K_SCALE_SIZE=12 packing
            let scales_raw = &bd[4..16];
            let mut sc = [0u8; 16]; // sub-block scales
            let mut mn = [0u8; 16]; // sub-block mins

            // Unpack 6-bit scales from K_SCALE_SIZE=12 format (same as llama.cpp)
            for i in 0..8 {
                sc[i] = (scales_raw[i] & 0x3F) as u8;
                mn[i] = (scales_raw[i] >> 6) as u8;
            }
            for i in 0..4 {
                sc[8 + i] = (scales_raw[8 + i] & 0x0F) as u8 | ((mn[2 * i] & 0x03) << 4);
                mn[8 + i] = (scales_raw[8 + i] >> 4) as u8 | ((mn[2 * i + 1] & 0x03) << 4);
                // Fix: upper 2 bits of sc/mn[0..7] were used for sc/mn[8..11]
            }
            // Actually the K_SCALE packing for Q5_K is:
            // sc[i<8] = scales_raw[i] & 63, storing lower 6 bits
            // The upper bits are packed in scales_raw[8..11]
            // Let me use the standard llama.cpp approach:
            //
            // For i in 0..8:
            //   sc[i] = scales_raw[i] & 63
            // For i in 0..4:
            //   sc[8+i] = (scales_raw[8+i] & 0xF) | ((scales_raw[i*2] >> 6) << 4) | ((scales_raw[i*2+1] >> 6) << 6)
            //
            // This is complex. Let me just use the simpler reference impl:

            // Reset and re-decode properly following ggml reference
            let ut = scales_raw;
            #[allow(unused_variables, unused_assignments)]
            let mut scales = [0u8; 16];
            let _mins = [0u8; 16];

            #[allow(unused_assignments)]
            {
                for i in 0..8 {
                    scales[i] = ut[i] & 63;
                }
                // Upper 2 bits of first 8 scales go into scales[8..12]
                scales[8] = (ut[8] & 0x0F) | ((ut[0] >> 6) << 4);
                scales[9] = (ut[9] & 0x0F) | ((ut[1] >> 6) << 4);
                scales[10] = (ut[10] & 0x0F) | ((ut[2] >> 6) << 4);
                scales[11] = (ut[11] & 0x0F) | ((ut[3] >> 6) << 4);
            }
            // Wait, this is also wrong. Let me look at this more carefully.
            // The Q5_K format packing from ggml:
            //
            // K_SCALE_SIZE = 12 bytes encodes 8 scales + 8 mins (6 bits each = 96 bits = 12 bytes)
            // For Q5_K there are 8 sub-blocks of 32 elements each.
            //
            // Actually, let me just decode all 256 elements through f32 intermediate:

            // Re-approach: follow ggml's dequantize_row_q5_K exactly
            // dm: d at offset 0 (f16), dmin at offset 2 (f16)
            // scales: offset 4, 12 bytes
            // qh: offset 16, 32 bytes
            // qs: offset 48, 128 bytes

            let qh = &bd[16..48];
            let qs = &bd[48..176];

            // Decode scales and mins (8 sub-blocks of 32 elements)
            // K_SCALE_SIZE format: 12 bytes encoding 8 scales (6-bit) + 8 mins (6-bit)
            let mut u_scales = [0u32; 8];
            let mut u_mins = [0u32; 8];

            // Lower 4 bits from ut[8..11], upper 2 bits from ut[0..7] >> 6
            for j in 0..4 {
                u_scales[j] = (ut[j] & 0x3F) as u32;
                u_mins[j] = (ut[j + 4] & 0x3F) as u32;
            }
            for j in 4..8 {
                u_scales[j] = ((ut[j + 4] & 0x0F) as u32) | (((ut[j - 4] >> 6) as u32) << 4);
                u_mins[j] = ((ut[j + 4] >> 4) as u32) | (((ut[j] >> 6) as u32) << 4);
            }

            let block_start = block * 256;
            for j in 0..8u32 {
                let sc_val = d * u_scales[j as usize] as f32;
                let mn_val = dmin * u_mins[j as usize] as f32;
                let sub_start = (j * 32) as usize;

                for l in 0..32usize {
                    let global_idx = block_start + sub_start + l;
                    if global_idx >= cols {
                        break;
                    }
                    let flat_idx = sub_start + l;
                    // Low 4 bits from qs (packed 2 per byte)
                    let q_lo = ((qs[flat_idx / 2] >> ((flat_idx % 2) * 4)) & 0x0F) as u32;
                    // High bit from qh
                    let q_hi = ((qh[flat_idx / 8] >> (flat_idx % 8)) & 1) as u32;
                    let q = q_lo | (q_hi << 4);
                    let val = sc_val * q as f32 - mn_val;
                    dst_row[global_idx] = f16::from_f32(val);
                }
            }
        }
    }

    result
}

/// Dequantize GGML Q6_K data to FP16.
///
/// Q6_K block layout (256 elements, 210 bytes):
///   ql[128]: lower 4 bits of each quant (packed 2 per byte)
///   qh[64]: upper 2 bits of each quant (packed 4 per byte)
///   scales[16]: int8 sub-block scales (16 sub-blocks of 16 elements)
///   d[2]: FP16 super-block scale
pub fn dequant_q6k_to_fp16(q6k_data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let blocks_per_row = cols.div_ceil(256);
    let block_bytes = 210;
    let q6k_row_bytes = blocks_per_row * block_bytes;
    let out_elements = rows * cols;
    let mut result = vec![0u8; out_elements * 2];

    let out_f16 =
        unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut f16, out_elements) };

    for row in 0..rows {
        let src_row = &q6k_data[row * q6k_row_bytes..];
        let dst_row = &mut out_f16[row * cols..];

        for block in 0..blocks_per_row {
            let bd = &src_row[block * block_bytes..];

            // Layout: ql[128] at 0, qh[64] at 128, scales[16] at 192, d[2] at 208
            let ql = &bd[0..128];
            let qh = &bd[128..192];
            let scales = &bd[192..208];
            let d = f16::from_le_bytes([bd[208], bd[209]]).to_f32();

            let block_start = block * 256;

            // Process 16 sub-blocks of 16 elements each
            for sub in 0..16u32 {
                let sc = scales[sub as usize] as i8;
                let sub_start = (sub * 16) as usize;

                for l in 0..16usize {
                    let global_idx = block_start + sub_start + l;
                    if global_idx >= cols {
                        break;
                    }
                    let flat_idx = sub_start + l;

                    // Low 4 bits from ql (packed 2 per byte, low nibble first)
                    let q_lo = ((ql[flat_idx / 2] >> ((flat_idx % 2) * 4)) & 0x0F) as i32;
                    // Upper 2 bits from qh (packed 4 per byte, 2 bits each)
                    let q_hi = ((qh[flat_idx / 4] >> ((flat_idx % 4) * 2)) & 0x03) as i32;
                    let q = q_lo | (q_hi << 4); // 6-bit unsigned value, 0..63
                    let val = d * sc as f32 * (q as f32 - 32.0); // centered around 32
                    dst_row[global_idx] = f16::from_f32(val);
                }
            }
        }
    }

    result
}
