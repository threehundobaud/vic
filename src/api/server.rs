//! OpenAI-compatible HTTP API server with streaming support.

use crate::core::types::TaskContext;
use crate::runtime::engine::Engine;
use crate::runtime::generate::SamplingParams;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Shared application state.
pub struct AppState {
    pub engine: Mutex<Engine>,
    pub model_name: String,
}

// Compile-time assertion: AppState must be Send + Sync for axum handlers.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AppState>();
};

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,

    /// Extension body for vib3-specific features.
    /// Matches the OpenAI Python SDK pattern: `extra_body={"task_context": {...}}`
    #[serde(default)]
    pub extra_body: Option<ExtraBody>,
}

/// Extension fields for vib3-specific features.
///
/// Passed via the OpenAI-compatible API as top-level `extra_body` field.
/// The OpenAI Python SDK merges `extra_body` into the request JSON,
/// so these fields also work when sent as top-level keys.
#[derive(Deserialize, Default)]
pub struct ExtraBody {
    /// Optional task context from Clank's Gearbox or an external classifier.
    /// Drives gear-based cache warming, mode detection, and HNSW filtering.
    #[serde(default)]
    pub task_context: Option<TaskContext>,
}

fn default_max_tokens() -> usize {
    4096
}
fn default_temperature() -> f32 {
    1.0
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming chunk response.
#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Model info response.
#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Create the router without engine state (for standalone use).
pub fn create_router() -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions_standalone))
        .route("/v1/models", get(models_standalone))
        .route("/health", get(health))
}

/// Create the router with engine state.
pub fn create_router_with_engine(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(state)
}

/// OpenAI-compatible error response.
#[derive(Serialize)]
struct ApiError {
    error: ApiErrorBody,
}

#[derive(Serialize)]
struct ApiErrorBody {
    message: String,
    r#type: String,
    code: Option<String>,
}

/// Chat completions handler (with engine state).
///
/// Uses `tokio::spawn` to move the engine interaction into a spawned
/// task. This provides a compile-time Send proof for the future,
/// which axum requires for handlers. The engine mutex guard (which
/// wraps device pointers) is confined to the spawned task's future.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let is_streaming = req.stream;
    let params = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p.unwrap_or(0.9),
        max_tokens: req.max_tokens,
        ..Default::default()
    };

    // Extract task context from extra_body (Gearbox integration — Phase A)
    let task_context = req
        .extra_body
        .as_ref()
        .and_then(|eb| eb.task_context.clone());

    // Extract system prompt and user message for chat template (owned for tokio::spawn)
    let system_prompt: Option<String> = req
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone());

    let user_message: String = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    // Build the raw prompt text (used for both chat-template and fallback encoding)
    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Whether to use chat template encoding (if we have a real tokenizer)
    let use_chat_template = !user_message.is_empty();
    let user_msg_for_spawn = user_message.clone();
    let sys_prompt_for_spawn = system_prompt.clone();

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let request_id = format!("chatcmpl-vib3-{}", created);

    if is_streaming {
        let model_name = state.model_name.clone();
        let request_id_clone = request_id.clone();

        // Run inference in a spawned task (provides Send proof for engine futures)
        let token_texts = {
            let state = state.clone();
            let task_ctx = task_context.clone();
            tokio::spawn(async move {
                let mut engine = state.engine.lock().await;

                // Apply task context before inference (Phase A/B)
                engine.set_task_context(task_ctx).await;

                // Encode with chat template if available, otherwise raw
                let input_tokens = if use_chat_template && engine.tokenizer().is_real() {
                    engine.tokenizer().encode_chat(
                        &user_msg_for_spawn,
                        sys_prompt_for_spawn.as_deref(),
                    )
                } else {
                    engine.tokenizer().encode(&prompt)
                };

                // Collect model-aware stop tokens
                let stop_ids = engine.tokenizer().stop_token_ids();

                let mut texts: Vec<String> = Vec::new();

                if let Err(e) = engine.prefill_tokens(&input_tokens).await {
                    texts.push(format!("[Error: {}]", e));
                    return texts;
                }

                tracing::info!("Input tokens ({}): {:?}", input_tokens.len(),
                    &input_tokens[..input_tokens.len().min(20)]);

                let max_tokens = params.max_tokens;
                let mut token_ids = Vec::new();
                for step in 0..max_tokens {
                    match engine.generate_one_token(step, &params).await {
                        Ok(token_id) => {
                            tracing::info!("Generated token[{}]: id={}", step, token_id);
                            token_ids.push(token_id);
                            if stop_ids.contains(&token_id)
                                || params.stop_tokens.contains(&token_id)
                            {
                                break;
                            }
                            texts.push(engine.tokenizer().decode(&[token_id]));
                        }
                        Err(_) => break,
                    }
                }
                tracing::info!("All generated token IDs: {:?}", token_ids);
                texts
            })
            .await
            .unwrap_or_default()
        };

        // Build SSE events from collected tokens
        let mut events = Vec::new();

        events.push(Ok::<_, std::convert::Infallible>(
            Event::default().data(
                serde_json::to_string(&ChatCompletionChunk {
                    id: request_id_clone.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: Some("assistant".into()),
                            content: None,
                        },
                        finish_reason: None,
                    }],
                })
                .unwrap_or_default(),
            ),
        ));

        for token_text in &token_texts {
            events.push(Ok(Event::default().data(
                serde_json::to_string(&ChatCompletionChunk {
                    id: request_id_clone.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(token_text.clone()),
                        },
                        finish_reason: None,
                    }],
                })
                .unwrap_or_default(),
            )));
        }

        events.push(Ok(Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: request_id_clone.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
            })
            .unwrap_or_default(),
        )));

        events.push(Ok(Event::default().data("[DONE]")));

        let event_stream = stream::iter(events);
        Sse::new(event_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        // Non-streaming: run inference in spawned task
        let result = {
            let state = state.clone();
            let model_name = state.model_name.clone();
            let task_ctx = task_context;
            tokio::spawn(async move {
                let mut engine = state.engine.lock().await;

                // Apply task context before inference (Phase A/B)
                engine.set_task_context(task_ctx).await;

                // Encode with chat template if available
                let input_tokens = if use_chat_template && engine.tokenizer().is_real() {
                    engine.tokenizer().encode_chat(
                        &user_message,
                        system_prompt.as_deref(),
                    )
                } else {
                    engine.tokenizer().encode(&prompt)
                };

                // Collect model-aware stop tokens
                let stop_ids = engine.tokenizer().stop_token_ids();

                let prompt_len = input_tokens.len();
                let start = std::time::Instant::now();

                // Prefill
                if let Err(e) = engine.prefill_tokens(&input_tokens).await {
                    return Err(e);
                }

                // Generate
                let mut generated = Vec::new();
                let max_tokens = params.max_tokens;
                for step in 0..max_tokens {
                    match engine.generate_one_token(step, &params).await {
                        Ok(token_id) => {
                            if stop_ids.contains(&token_id)
                                || params.stop_tokens.contains(&token_id)
                            {
                                break;
                            }
                            generated.push(token_id);
                        }
                        Err(e) => return Err(e),
                    }
                }

                let text = engine.tokenizer().decode(&generated);
                let total_ms = start.elapsed().as_secs_f64() * 1000.0;
                Ok((text, prompt_len, generated.len(), model_name, total_ms))
            })
            .await
        };

        match result {
            Ok(Ok((text, prompt_tokens, tokens_generated, model_name, _total_ms))) => {
                Json(ChatCompletionResponse {
                    id: request_id,
                    object: "chat.completion".into(),
                    created,
                    model: model_name,
                    choices: vec![Choice {
                        index: 0,
                        message: Message {
                            role: "assistant".into(),
                            content: text,
                        },
                        finish_reason: "stop".into(),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens: tokens_generated,
                        total_tokens: prompt_tokens + tokens_generated,
                    },
                })
                .into_response()
            }
            Ok(Err(e)) => {
                let error_resp = ApiError {
                    error: ApiErrorBody {
                        message: format!("{}", e),
                        r#type: "server_error".into(),
                        code: Some("internal_error".into()),
                    },
                };
                (StatusCode::INTERNAL_SERVER_ERROR, Json(error_resp)).into_response()
            }
            Err(e) => {
                let error_resp = ApiError {
                    error: ApiErrorBody {
                        message: format!("Task panicked: {}", e),
                        r#type: "server_error".into(),
                        code: Some("internal_error".into()),
                    },
                };
                (StatusCode::INTERNAL_SERVER_ERROR, Json(error_resp)).into_response()
            }
        }
    }
}

async fn models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".into(),
            owned_by: "vib3".into(),
        }],
    })
}

// Standalone handlers (no engine state) — return proper HTTP errors.

async fn chat_completions_standalone(
    Json(_req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let error_resp = ApiError {
        error: ApiErrorBody {
            message: "No engine loaded. Start the server with a model path to enable inference."
                .into(),
            r#type: "invalid_request_error".into(),
            code: Some("engine_not_loaded".into()),
        },
    };
    (StatusCode::SERVICE_UNAVAILABLE, Json(error_resp)).into_response()
}

async fn models_standalone() -> axum::response::Response {
    let error_resp = ApiError {
        error: ApiErrorBody {
            message: "No model loaded. Start the server with a model path.".into(),
            r#type: "invalid_request_error".into(),
            code: Some("no_model".into()),
        },
    };
    (StatusCode::SERVICE_UNAVAILABLE, Json(error_resp)).into_response()
}

async fn health() -> &'static str {
    "ok"
}
