//! Generation loop — token sampling, streaming output, and KV cache management.

use std::collections::HashSet;

use half::f16;

/// Sampling parameters for text generation.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            max_tokens: 4096,
            stop_tokens: vec![],
        }
    }
}

/// Simple token sampler with temperature, top-k, and top-p.
pub struct Sampler {
    rng_state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng_state: seed.wrapping_add(1), // Avoid 0 seed
        }
    }

    /// Sample a token from logits using the given parameters.
    pub fn sample(&mut self, logits: &[f32], params: &SamplingParams, recent_tokens: &[u32]) -> u32 {
        let vocab_size = logits.len();
        if vocab_size == 0 {
            return 0;
        }

        let mut probs = logits.to_vec();

        // Apply repetition penalty before temperature/top-k/top-p.
        // GPT-style rule: if logit > 0 divide by penalty, else multiply.
        if params.repetition_penalty > 1.0 && !recent_tokens.is_empty() {
            let mut seen = HashSet::new();
            for &token_id in recent_tokens {
                let idx = token_id as usize;
                if idx >= vocab_size || !seen.insert(idx) {
                    continue;
                }
                let logit = probs[idx];
                probs[idx] = if logit > 0.0 {
                    logit / params.repetition_penalty
                } else {
                    logit * params.repetition_penalty
                };
            }
        }

        // Apply temperature
        if params.temperature > 0.0 && params.temperature != 1.0 {
            for p in &mut probs {
                *p /= params.temperature;
            }
        }

        // If temperature is 0, do greedy (argmax)
        if params.temperature <= 0.0 {
            return probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
        }

        // Softmax
        let max_logit = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for p in &mut probs {
            *p = (*p - max_logit).exp();
        }
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        // Top-K filtering
        if params.top_k > 0 && params.top_k < vocab_size {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let threshold = indexed[params.top_k.min(indexed.len()) - 1].1;
            for p in probs.iter_mut() {
                if *p < threshold {
                    *p = 0.0;
                }
            }

            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs {
                    *p /= sum;
                }
            }
        }

        // Top-P (nucleus) filtering
        if params.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cumsum = 0.0f32;
            let mut cutoff_idx = indexed.len();
            for (i, (_, p)) in indexed.iter().enumerate() {
                cumsum += p;
                if cumsum > params.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            let threshold = if cutoff_idx < indexed.len() {
                indexed[cutoff_idx].1
            } else {
                0.0
            };

            for p in &mut probs {
                if *p < threshold {
                    *p = 0.0;
                }
            }

            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs {
                    *p /= sum;
                }
            }
        }

        // Sample from distribution
        let r = self.random_f32();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return i as u32;
            }
        }

        // Fallback: return last non-zero
        (vocab_size - 1) as u32
    }

    /// Simple xorshift64 PRNG.
    fn random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f32) / (u64::MAX as f32)
    }
}

/// Convert FP16 logits buffer to f32 slice for sampling.
pub fn fp16_logits_to_f32(logits_ptr: *const u8, vocab_size: usize) -> Vec<f32> {
    let logits = unsafe { std::slice::from_raw_parts(logits_ptr as *const f16, vocab_size) };
    logits.iter().map(|x| x.to_f32()).collect()
}

/// Unified tokenizer that dispatches to either `SimpleTokenizer` or `Vib3Tokenizer`.
///
/// The engine creates this at startup: if a `tokenizer.json` is found, it uses
/// the real tokenizer; otherwise it falls back to the byte-level stub.
#[allow(clippy::large_enum_variant)]
pub enum TokenizerWrapper {
    Simple(SimpleTokenizer),
    Real(Vib3Tokenizer),
}

impl TokenizerWrapper {
    /// Try to load a real tokenizer, falling back to SimpleTokenizer.
    pub fn load(tokenizer_path: &str, model_path: &str, vocab_size: u32) -> Self {
        // 1. Try explicit tokenizer path
        if !tokenizer_path.is_empty() {
            let path = std::path::Path::new(tokenizer_path);
            if path.is_file() {
                match Vib3Tokenizer::from_file(path) {
                    Ok(t) => return TokenizerWrapper::Real(t),
                    Err(e) => {
                        tracing::warn!("Failed to load tokenizer from {}: {}", tokenizer_path, e)
                    }
                }
            } else if path.is_dir() {
                match Vib3Tokenizer::from_model_dir(path) {
                    Ok(t) => return TokenizerWrapper::Real(t),
                    Err(e) => {
                        tracing::warn!("Failed to load tokenizer from {}: {}", tokenizer_path, e)
                    }
                }
            }
        }

        // 2. Try adjacent to model file
        let model_dir = std::path::Path::new(model_path).parent();
        if let Some(dir) = model_dir {
            let tokenizer_json = dir.join("tokenizer.json");
            if tokenizer_json.exists() {
                match Vib3Tokenizer::from_file(&tokenizer_json) {
                    Ok(t) => return TokenizerWrapper::Real(t),
                    Err(e) => tracing::warn!(
                        "Failed to load tokenizer from {}: {}",
                        tokenizer_json.display(),
                        e
                    ),
                }
            }

            // Also check for a "safetensors" subdirectory (common after HF download)
            let safetensors_tokenizer = dir.join("safetensors").join("tokenizer.json");
            if safetensors_tokenizer.exists() {
                match Vib3Tokenizer::from_file(&safetensors_tokenizer) {
                    Ok(t) => return TokenizerWrapper::Real(t),
                    Err(e) => tracing::warn!("Failed to load tokenizer: {}", e),
                }
            }
        }

        // 3. Fall back to simple tokenizer
        tracing::info!("Using byte-level tokenizer (no tokenizer.json found)");
        TokenizerWrapper::Simple(SimpleTokenizer::new(vocab_size))
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            TokenizerWrapper::Simple(t) => t.encode(text),
            TokenizerWrapper::Real(t) => t.encode(text),
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        match self {
            TokenizerWrapper::Simple(t) => t.decode(tokens),
            TokenizerWrapper::Real(t) => t.decode(tokens),
        }
    }

    pub fn is_eos(&self, token: u32) -> bool {
        match self {
            TokenizerWrapper::Simple(t) => t.is_eos(token),
            TokenizerWrapper::Real(t) => t.is_eos(token),
        }
    }

    pub fn is_real(&self) -> bool {
        matches!(self, TokenizerWrapper::Real(_))
    }

    /// Encode a chat conversation using a tokenizer-aware chat template.
    ///
    /// Chooses the template based on special tokens present in the loaded
    /// tokenizer:
    /// - Qwen ChatML: `<|im_start|>...<|im_end|>`
    /// - Kimi: `<|im_system|>...<|im_middle|>...<|im_end|>`
    ///
    /// For the simple tokenizer, this falls back to naive encoding.
    pub fn encode_chat(&self, user_message: &str, system_prompt: Option<&str>) -> Vec<u32> {
        match self {
            TokenizerWrapper::Simple(t) => {
                // Fallback: encode as "system: ... user: ..."
                let mut text = String::new();
                if let Some(sys) = system_prompt {
                    text.push_str(&format!("system: {}\n", sys));
                }
                text.push_str(&format!("user: {}", user_message));
                t.encode(&text)
            }
            TokenizerWrapper::Real(t) => t.encode_chat(user_message, system_prompt, false),
        }
    }

    /// Get all token IDs that should stop generation.
    ///
    /// Returns EOS token and any model-specific stop tokens (e.g. `<|im_end|>`).
    pub fn stop_token_ids(&self) -> Vec<u32> {
        match self {
            TokenizerWrapper::Simple(t) => vec![t.eos_token],
            TokenizerWrapper::Real(t) => {
                let mut stops = vec![t.eos_token_id];
                // Add <|im_end|> as a stop token for chat models
                if let Some(im_end) = t.inner.token_to_id("<|im_end|>") {
                    stops.push(im_end);
                }
                // Add [EOT] as stop token
                if let Some(eot) = t.inner.token_to_id("[EOT]") {
                    stops.push(eot);
                }
                // Note: PAD (163839) is NOT a stop token — the model may generate it
                // during warmup. We let the caller decide via max_tokens.
                stops
            }
        }
    }
}

/// Simple byte-pair encoding tokenizer stub.
/// This provides a minimal tokenizer for testing. In production, use
/// `Vib3Tokenizer::from_file()` to load a real HuggingFace tokenizer.
pub struct SimpleTokenizer {
    /// EOS token ID
    pub eos_token: u32,
    /// BOS token ID
    pub bos_token: u32,
    /// Vocab size
    pub vocab_size: u32,
}

impl SimpleTokenizer {
    pub fn new(vocab_size: u32) -> Self {
        Self {
            eos_token: 2,
            bos_token: 1,
            vocab_size,
        }
    }

    /// Encode text to token IDs.
    /// This is a simple byte-level tokenizer for testing purposes.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos_token];
        // Simple byte-level encoding: each byte -> token ID offset by 256
        for byte in text.bytes() {
            let token = 256 + byte as u32;
            if token < self.vocab_size {
                tokens.push(token);
            }
        }
        tokens
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &token in tokens {
            if token == self.bos_token || token == self.eos_token {
                continue;
            }
            if (256..256 + 128).contains(&token) {
                bytes.push((token - 256) as u8);
            } else if token >= 256 + 128 {
                // Extended tokens — just output the ID as text
                let s = format!("[{}]", token);
                bytes.extend_from_slice(s.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn is_eos(&self, token: u32) -> bool {
        token == self.eos_token
    }
}

// ─── Real Tokenizer (HuggingFace tokenizers) ────────────────────────────

/// Production tokenizer backed by HuggingFace's `tokenizers` library.
///
/// Loads from `tokenizer.json` (standard HF format) and supports model-aware
/// chat templating (Qwen ChatML and Kimi-style templates).
pub struct Vib3Tokenizer {
    inner: tokenizers::Tokenizer,
    /// BOS token ID (Kimi K2.5: 163584)
    pub bos_token_id: u32,
    /// EOS token ID (Kimi K2.5: 163585)
    pub eos_token_id: u32,
    /// Pad token ID (Kimi K2.5: 163839)
    pub pad_token_id: u32,
    /// Vocab size
    pub vocab_size: u32,
    /// Think opening tag ID (for thinking mode).
    /// Used during generation to detect when the model enters/exits thinking.
    #[allow(dead_code)]
    think_token_id: Option<u32>,
}

impl Vib3Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> crate::core::error::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref()).map_err(|e| {
            crate::core::error::Error::ConfigError(format!(
                "Failed to load tokenizer from {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;

        let vocab_size = inner.get_vocab_size(true) as u32;

        // Look up special token IDs
        let bos_token_id = inner
            .token_to_id("<|begin_of_text|>")
            .or_else(|| inner.token_to_id("<s>"))
            .unwrap_or(163584); // Kimi K2.5 default

        let eos_token_id = inner
            .token_to_id("<|end_of_text|>")
            .or_else(|| inner.token_to_id("<|endoftext|>")) // Qwen3.5
            .or_else(|| inner.token_to_id("</s>"))
            .unwrap_or(163585); // Kimi K2.5 default

        let pad_token_id = inner.token_to_id("<|pad|>").unwrap_or(163839); // Kimi K2.5 default

        let think_token_id = inner.token_to_id("<think>");

        tracing::info!(
            "Tokenizer loaded: vocab_size={}, bos={}, eos={}, pad={}",
            vocab_size,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        );

        Ok(Self {
            inner,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            vocab_size,
            think_token_id,
        })
    }

    /// Load from a model directory (looks for `tokenizer.json`).
    pub fn from_model_dir(dir: impl AsRef<std::path::Path>) -> crate::core::error::Result<Self> {
        let path = dir.as_ref().join("tokenizer.json");
        if !path.exists() {
            return Err(crate::core::error::Error::ConfigError(format!(
                "tokenizer.json not found in {}",
                dir.as_ref().display()
            )));
        }
        Self::from_file(path)
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Enable tokenizer-defined special token handling (e.g., BOS/EOS when
        // appropriate for the model family).
        match self.inner.encode(text, true) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(e) => {
                tracing::warn!("Tokenizer encode failed: {}", e);
                vec![]
            }
        }
    }

    /// Encode with a tokenizer-aware chat template.
    ///
    /// Prefers Qwen ChatML (`<|im_start|>`) when available, otherwise falls
    /// back to Kimi-style tags (`<|im_system|>`, `<|im_middle|>`).
    pub fn encode_chat(
        &self,
        user_message: &str,
        system_prompt: Option<&str>,
        thinking: bool,
    ) -> Vec<u32> {
        let has_qwen_chatml = self.inner.token_to_id("<|im_start|>").is_some()
            && self.inner.token_to_id("<|im_end|>").is_some();

        let system = system_prompt.unwrap_or("You are a helpful assistant.");

        let template = if has_qwen_chatml {
            if thinking {
                format!(
                    "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                    system, user_message
                )
            } else {
                format!(
                    "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    system, user_message
                )
            }
        } else if thinking {
            format!(
                "<|im_system|>system<|im_middle|>{}<|im_end|>\n<|im_user|>user<|im_middle|>{}<|im_end|>\n<|im_assistant|>assistant<|im_middle|><think>",
                system, user_message
            )
        } else {
            format!(
                "<|im_system|>system<|im_middle|>{}<|im_end|>\n<|im_user|>user<|im_middle|>{}<|im_end|>\n<|im_assistant|>assistant<|im_middle|>",
                system, user_message
            )
        };

        self.encode(&template)
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        match self.inner.decode(tokens, true) {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("Tokenizer decode failed: {}", e);
                format!("[decode error: {}]", e)
            }
        }
    }

    /// Decode a single token to its string representation.
    pub fn decode_token(&self, token: u32) -> String {
        self.decode(&[token])
    }

    /// Check if a token is an EOS token.
    pub fn is_eos(&self, token: u32) -> bool {
        token == self.eos_token_id
    }

    /// Check if a token is the im_end special token.
    pub fn is_im_end(&self, token: u32) -> bool {
        if let Some(im_end) = self.inner.token_to_id("<|im_end|>") {
            token == im_end
        } else {
            false
        }
    }

    /// Get the ID for a specific token string, if it exists.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Encode with the Qwen3.5 chat template (thinking mode).
    ///
    /// Wraps the user message in the standard Qwen3.5 format:
    /// ```text
    /// <|im_start|>system
    /// You are a helpful assistant.<|im_end|>
    /// <|im_start|>user
    /// {user_message}<|im_end|>
    /// <|im_start|>assistant
    /// <think>
    /// ```
    pub fn encode_chat_qwen35(&self, user_message: &str, system_prompt: Option<&str>) -> Vec<u32> {
        let sys = system_prompt.unwrap_or("You are a helpful assistant.");
        let template = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            sys, user_message
        );
        self.encode(&template)
    }

    /// Return stop token IDs for chat mode (model-specific).
    ///
    /// For Qwen3.5: <|im_end|> (248046) and <|endoftext|> (248044).
    pub fn chat_stop_tokens(&self) -> Vec<u32> {
        let mut stops = Vec::new();
        if let Some(id) = self.inner.token_to_id("<|im_end|>") {
            stops.push(id);
        }
        if let Some(id) = self.inner.token_to_id("<|endoftext|>") {
            stops.push(id);
        }
        stops
    }
}
