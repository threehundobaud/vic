//! Async I/O engine for NVMe page reads.
//!
//! Uses io_uring on Linux for kernel-bypass async reads.
//! Falls back to tokio::fs for other platforms.

use crate::core::error::Result;
use std::os::unix::io::AsRawFd;

/// Configuration for the I/O engine.
pub struct IoConfig {
    /// io_uring submission queue depth.
    pub queue_depth: u32,
    /// Whether to use direct I/O (O_DIRECT) for NVMe bypass.
    pub direct_io: bool,
    /// Number of I/O worker threads.
    pub num_workers: usize,
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            queue_depth: 64,
            direct_io: true,
            num_workers: 2,
        }
    }
}

/// Completion token for an async read.
#[derive(Debug)]
pub struct ReadCompletion {
    pub page_idx: usize,
    pub bytes_read: usize,
    pub buffer_ptr: *mut u8,
}

unsafe impl Send for ReadCompletion {}

/// Async I/O engine.
///
/// Wraps io_uring for high-throughput page reads from NVMe.
pub struct IoEngine {
    #[allow(dead_code)]
    config: IoConfig,
    // io_uring instance (platform-specific)
    #[cfg(target_os = "linux")]
    ring: Option<io_uring::IoUring>,
}

impl IoEngine {
    pub fn new(config: IoConfig) -> Result<Self> {
        #[cfg(target_os = "linux")]
        let ring = {
            let ring = io_uring::IoUring::builder()
                .setup_sqpoll(1000) // Kernel-side polling for low latency
                .build(config.queue_depth)
                .ok(); // Fall back to non-SQPOLL if not available
            ring.or_else(|| io_uring::IoUring::new(config.queue_depth).ok())
        };

        Ok(Self {
            config,
            #[cfg(target_os = "linux")]
            ring,
        })
    }

    /// Submit an async read request.
    ///
    /// Reads `size` bytes from `file` at `offset` into `buffer`.
    /// The read completes asynchronously; call `poll_completions` to check.
    #[cfg(target_os = "linux")]
    pub fn submit_read(
        &mut self,
        file: &std::fs::File,
        offset: u64,
        buffer: *mut u8,
        size: u32,
        user_data: u64,
    ) -> Result<()> {
        use io_uring::opcode;
        use io_uring::types;

        if let Some(ref mut ring) = self.ring {
            let fd = types::Fd(file.as_raw_fd());
            let read_op = opcode::Read::new(fd, buffer, size)
                .offset(offset)
                .build()
                .user_data(user_data);

            // SAFETY: The buffer must remain valid until the read completes.
            // The caller (buffer manager) guarantees this by holding the slot lock.
            unsafe {
                ring.submission()
                    .push(&read_op)
                    .map_err(|_| std::io::Error::other("io_uring submission queue full"))?;
            }

            ring.submit()?;
        }

        Ok(())
    }

    /// Poll for completed reads.
    #[cfg(target_os = "linux")]
    pub fn poll_completions(&mut self) -> Vec<(u64, i32)> {
        let mut completions = Vec::new();

        if let Some(ref mut ring) = self.ring {
            let cq = ring.completion();
            for entry in cq {
                completions.push((entry.user_data(), entry.result()));
            }
        }

        completions
    }

    /// Submit a batch of reads and wait for all to complete.
    ///
    /// Uses io_uring for batched submission on Linux when available,
    /// otherwise falls back to sequential reads.
    pub async fn read_batch(
        &mut self,
        file: &std::fs::File,
        reads: &[(u64, *mut u8, u32)], // (offset, buffer, size)
    ) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            if self.ring.is_some() {
                return self.read_batch_uring(file, reads);
            }
        }

        // Sequential fallback (non-Linux or no io_uring)
        self.read_batch_sequential(file, reads)
    }

    /// Sequential fallback for read_batch.
    fn read_batch_sequential(
        &self,
        file: &std::fs::File,
        reads: &[(u64, *mut u8, u32)],
    ) -> Result<()> {
        use std::io::{Read, Seek, SeekFrom};

        for &(offset, buffer, size) in reads {
            let mut f = file;
            f.seek(SeekFrom::Start(offset))?;
            let buf = unsafe { std::slice::from_raw_parts_mut(buffer, size as usize) };
            f.read_exact(buf)?;
        }

        Ok(())
    }

    /// io_uring batched read implementation.
    #[cfg(target_os = "linux")]
    fn read_batch_uring(
        &mut self,
        file: &std::fs::File,
        reads: &[(u64, *mut u8, u32)],
    ) -> Result<()> {
        use io_uring::opcode;
        use io_uring::types;

        let ring = self.ring.as_mut().unwrap();
        let fd = types::Fd(file.as_raw_fd());

        // Submit all reads, handling SQ full by submitting in chunks
        let mut submitted = 0;
        for (i, &(offset, buffer, size)) in reads.iter().enumerate() {
            let read_op = opcode::Read::new(fd, buffer, size)
                .offset(offset)
                .build()
                .user_data(i as u64);

            // SAFETY: buffers must remain valid until reads complete.
            // The caller guarantees this.
            let push_result = unsafe { ring.submission().push(&read_op) };

            match push_result {
                Ok(()) => {
                    submitted += 1;
                }
                Err(_) => {
                    // SQ full — submit and drain completions, then retry
                    ring.submit_and_wait(1)?;
                    {
                        let cq = ring.completion();
                        for entry in cq {
                            if entry.result() < 0 {
                                return Err(
                                    std::io::Error::from_raw_os_error(-entry.result()).into()
                                );
                            }
                        }
                    }
                    // Retry this entry after draining
                    unsafe {
                        ring.submission()
                            .push(&read_op)
                            .map_err(|_| std::io::Error::other("io_uring SQ full after drain"))?;
                    }
                    submitted += 1;
                }
            }
        }

        // Submit all pending
        ring.submit()?;

        // Wait for all completions
        let mut completed = 0;
        while completed < submitted {
            ring.submit_and_wait(1)?;
            {
                let cq = ring.completion();
                for entry in cq {
                    if entry.result() < 0 {
                        return Err(std::io::Error::from_raw_os_error(-entry.result()).into());
                    }
                    completed += 1;
                }
            }
        }

        Ok(())
    }

    /// Submit an async read request (non-Linux stub).
    #[cfg(not(target_os = "linux"))]
    pub fn submit_read(
        &mut self,
        _file: &std::fs::File,
        _offset: u64,
        _buffer: *mut u8,
        _size: u32,
        _user_data: u64,
    ) -> Result<()> {
        Err(crate::core::error::Error::Other(anyhow::anyhow!(
            "io_uring not available on this platform"
        )))
    }

    /// Poll for completed reads (non-Linux stub).
    #[cfg(not(target_os = "linux"))]
    pub fn poll_completions(&mut self) -> Vec<(u64, i32)> {
        Vec::new()
    }

    /// Check if io_uring is available.
    pub fn has_uring(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.ring.is_some()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}
