// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Buffered codec for Gemini `generateContent` request and response JSON.

use std::collections::{BTreeMap, VecDeque};

use serde_json::{Map, Value, json};

use crate::codecs::common::{provider_extensions, text_from_blocks};
use crate::codecs::{
    DecodedRequest, DecodedResponse, EncodedRequest, EncodedResponse, FormatCodec,
};
use crate::diagnostic::TranslationDiagnostic;
use crate::error::{Result, TranslationError};
use crate::format::FormatId;
use crate::llm::{
    AggLlmResponse, ContentBlock, FileSource, ImageSource, InstructionBlock, LlmRequest,
    MediaSource, Message, OutputParams, ProviderExtensions, ResponseOutput, Role, SamplingParams,
    StopReason, ToolCall, ToolChoice, ToolDefinition, ToolResult, Usage,
};
use crate::policy::{DeterministicIdPolicy, TranslationPolicy};
use crate::util::{
    capture_request_preservation, capture_response_preservation, embed_preservation,
    exact_preserved_request, exact_preserved_response, json_string, object, push_lossy, stable_id,
    validate_request_capabilities,
};

// Gemini has no `WireFormat` variant; the codec is addressed by this registry key.
const GEMINI_FORMAT: &str = "gemini_generate_content";

// `generationConfig` fields mapped onto the IR. The rest of the object rides in
// `extensions["generationConfig"]` so a Gemini-to-Gemini hop keeps it.
const MAPPED_GENERATION_CONFIG: [&str; 6] = [
    "temperature",
    "topP",
    "topK",
    "maxOutputTokens",
    "responseSchema",
    "responseJsonSchema",
];

// Top-level request fields replayed from extensions. Only `store` can come from another
// provider, and OpenAI gives it the same meaning.
const REPLAYED_REQUEST_FIELDS: [&str; 5] = [
    "cachedContent",
    "labels",
    "safetySettings",
    "serviceTier",
    "store",
];

/// Format codec for Gemini `generateContent` payloads.
pub struct GeminiGenerateContentCodec;

impl FormatCodec for GeminiGenerateContentCodec {
    fn format(&self) -> FormatId {
        FormatId::new(GEMINI_FORMAT)
    }

    fn decode_request(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedRequest> {
        let body = object(body, "$")?;
        let mut diagnostics = Vec::new();
        let mut generation = match body.get("generationConfig") {
            Some(config) => object(config, "$.generationConfig")?.clone(),
            None => Map::new(),
        };
        let response_format = decode_gemini_response_format(&generation);
        let mut request = LlmRequest {
            model: body
                .get("model")
                .and_then(Value::as_str)
                .filter(|model| !model.is_empty())
                .map(ToOwned::to_owned),
            sampling: SamplingParams {
                temperature: generation.get("temperature").and_then(Value::as_f64),
                top_p: generation.get("topP").and_then(Value::as_f64),
                top_k: generation.get("topK").and_then(Value::as_i64),
            },
            output: OutputParams {
                max_output_tokens: generation.get("maxOutputTokens").and_then(Value::as_u64),
                response_format,
            },
            preservation: capture_request_preservation(
                GEMINI_FORMAT,
                &Value::Object(body.clone()),
                policy,
            ),
            ..LlmRequest::default()
        };
        if let Some(system) = body.get("systemInstruction") {
            // A role on systemInstruction carries no meaning, so it is ignored.
            let content = object(system, "$.systemInstruction")?
                .get("parts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .map(|text| ContentBlock::Text {
                    text: text.to_string(),
                })
                .collect::<Vec<_>>();
            if !content.is_empty() {
                request.instructions.push(InstructionBlock {
                    role: Role::System,
                    content,
                });
            }
        }
        if let Some(contents) = body.get("contents") {
            let contents = contents
                .as_array()
                .ok_or_else(|| TranslationError::InvalidType {
                    path: "$.contents".to_string(),
                    expected: "array",
                })?;
            let mut ids = GeminiToolIds::default();
            for (index, content) in contents.iter().enumerate() {
                let content = object(content, &format!("$.contents[{index}]"))?;
                // Gemini defines only `user` and `model`; an absent role means `user`.
                let role = match content.get("role").and_then(Value::as_str) {
                    None | Some("user") => Role::User,
                    Some("model") => Role::Assistant,
                    Some(other) => {
                        return Err(TranslationError::InvalidValue {
                            path: format!("$.contents[{index}].role"),
                            message: format!(
                                "Invalid value: {other:?}. Supported content roles are user, model."
                            ),
                        });
                    }
                };
                let content = decode_gemini_parts(
                    content.get("parts"),
                    &format!("$.contents[{index}].parts"),
                    &mut ids,
                    policy,
                )?;
                request.messages.push(Message { role, content });
            }
        }
        request.tools = decode_gemini_tools(body.get("tools"), &mut diagnostics, policy)?;
        request.tool_choice = body.get("toolConfig").and_then(decode_gemini_tool_choice);
        request.extensions.fields = provider_extensions(
            body,
            &[
                "model",
                "contents",
                "systemInstruction",
                "tools",
                "toolConfig",
                "generationConfig",
            ],
        );
        for key in MAPPED_GENERATION_CONFIG {
            generation.remove(key);
        }
        if request.output.response_format.is_some() {
            generation.remove("responseMimeType");
        }
        if !generation.is_empty() {
            request
                .extensions
                .fields
                .insert("generationConfig".to_string(), Value::Object(generation));
        }
        Ok(DecodedRequest {
            request,
            diagnostics,
        })
    }

    fn encode_request(
        &self,
        request: &LlmRequest,
        policy: &TranslationPolicy,
    ) -> Result<EncodedRequest> {
        if let Some(body) = exact_preserved_request(&request.preservation, GEMINI_FORMAT, policy) {
            return Ok(EncodedRequest {
                body,
                diagnostics: Vec::new(),
            });
        }
        let mut diagnostics = Vec::new();
        validate_request_capabilities(request, &mut diagnostics, policy)?;
        let mut body = Map::new();
        if let Some(model) = &request.model {
            body.insert("model".to_string(), Value::String(model.clone()));
        }
        let system_parts = request
            .instructions
            .iter()
            .flat_map(|instruction| instruction.content.iter())
            .filter_map(|block| match block {
                ContentBlock::Text { text } | ContentBlock::Refusal { text } => {
                    Some(json!({"text": text}))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if !system_parts.is_empty() {
            body.insert(
                "systemInstruction".to_string(),
                json!({"parts": system_parts}),
            );
        }

        // Gemini keys function responses by name, so resolve each result's call id to its name.
        let names = request
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .filter_map(|block| match block {
                ContentBlock::ToolCall(call) => Some((call.id.as_str(), call.name.as_str())),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let mut contents: Vec<Value> = Vec::new();
        for message in &request.messages {
            let role = if message.role == Role::Assistant {
                "model"
            } else {
                "user"
            };
            let parts = encode_gemini_parts(&message.content, &names, &mut diagnostics, policy)?;
            if parts.is_empty() {
                continue;
            }
            // Gemini wants every response to a parallel call turn in one content, and other
            // formats split tool results into one message each, so same-role turns are merged.
            match contents.last_mut() {
                Some(last) if last["role"] == role => {
                    if let Some(existing) = last["parts"].as_array_mut() {
                        existing.extend(parts);
                    }
                }
                _ => contents.push(json!({"role": role, "parts": parts})),
            }
        }
        body.insert("contents".to_string(), Value::Array(contents));

        if !request.tools.is_empty() {
            body.insert("tools".to_string(), encode_gemini_tools(&request.tools));
            if let Some(choice) = &request.tool_choice
                && let Some(config) = encode_gemini_tool_config(choice, &mut diagnostics, policy)?
            {
                body.insert("toolConfig".to_string(), config);
            }
        }

        let mut generation = request
            .extensions
            .fields
            .get("generationConfig")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        if let Some(value) = request.sampling.temperature {
            generation.insert("temperature".to_string(), json!(value));
        }
        if let Some(value) = request.sampling.top_p {
            generation.insert("topP".to_string(), json!(value));
        }
        if let Some(value) = request.sampling.top_k {
            generation.insert("topK".to_string(), json!(value));
        }
        if let Some(value) = request.output.max_output_tokens {
            generation.insert("maxOutputTokens".to_string(), json!(value));
        }
        if let Some(format) = &request.output.response_format {
            encode_gemini_response_format(format, &mut generation, &mut diagnostics, policy)?;
        }
        if !generation.is_empty() {
            body.insert("generationConfig".to_string(), Value::Object(generation));
        }
        for field in REPLAYED_REQUEST_FIELDS {
            if let Some(value) = request.extensions.fields.get(field) {
                body.insert(field.to_string(), value.clone());
            }
        }

        let body = embed_preservation(Value::Object(body), &request.preservation, policy);
        Ok(EncodedRequest { body, diagnostics })
    }

    fn decode_response(&self, body: &Value, policy: &TranslationPolicy) -> Result<DecodedResponse> {
        let body = object(body, "$")?;
        let candidate = body
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
            .and_then(Value::as_object);
        let mut content = decode_gemini_parts(
            candidate
                .and_then(|candidate| candidate.get("content"))
                .and_then(|content| content.get("parts")),
            "$.candidates[0].content.parts",
            &mut GeminiToolIds::default(),
            policy,
        )?;
        let has_tool_call = content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolCall(_)));
        let stop_reason = match candidate {
            Some(candidate) => gemini_stop_reason(
                candidate.get("finishReason").and_then(Value::as_str),
                has_tool_call,
            ),
            // A prompt blocked before generation returns no candidates, only a block reason.
            None if body
                .get("promptFeedback")
                .and_then(|feedback| feedback.get("blockReason"))
                .is_some() =>
            {
                StopReason::ContentFilter
            }
            None => StopReason::EndTurn,
        };
        if content.is_empty() {
            content.push(ContentBlock::Text {
                text: String::new(),
            });
        }
        let response = AggLlmResponse {
            id: body
                .get("responseId")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            model: body
                .get("modelVersion")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            outputs: vec![ResponseOutput {
                role: Role::Assistant,
                content,
                url_citations: Vec::new(),
                stop_reason: Some(stop_reason),
            }],
            usage: decode_gemini_usage(body.get("usageMetadata")),
            extensions: ProviderExtensions {
                fields: provider_extensions(
                    body,
                    &["candidates", "usageMetadata", "responseId", "modelVersion"],
                ),
            },
            preservation: capture_response_preservation(
                GEMINI_FORMAT,
                &Value::Object(body.clone()),
                policy,
            ),
        };
        Ok(DecodedResponse {
            response,
            diagnostics: Vec::new(),
        })
    }

    fn encode_response(
        &self,
        response: &AggLlmResponse,
        policy: &TranslationPolicy,
    ) -> Result<EncodedResponse> {
        if let Some(body) = exact_preserved_response(&response.preservation, GEMINI_FORMAT, policy)
        {
            return Ok(EncodedResponse {
                body,
                diagnostics: Vec::new(),
            });
        }
        let mut diagnostics = Vec::new();
        let output = response.first_output();
        let parts = match output {
            Some(output) => {
                encode_gemini_parts(&output.content, &BTreeMap::new(), &mut diagnostics, policy)?
            }
            None => Vec::new(),
        };
        let mut body = json!({
            "candidates": [{
                "content": {"role": "model", "parts": parts},
                "finishReason": gemini_finish_reason(output.and_then(|output| output.stop_reason)),
                "index": 0,
            }],
            "usageMetadata": encode_gemini_usage(&response.usage),
        });
        if let Some(id) = &response.id {
            body["responseId"] = json!(id);
        }
        if let Some(model) = &response.model {
            body["modelVersion"] = json!(model);
        }
        Ok(EncodedResponse {
            body: embed_preservation(body, &response.preservation, policy),
            diagnostics,
        })
    }
}

// Pairs Gemini's name-keyed function responses with the IR's id-keyed tool results.
#[derive(Default)]
struct GeminiToolIds {
    counter: usize,
    // Ids issued per function name, oldest first, so parallel calls to one name pair in order.
    pending: BTreeMap<String, VecDeque<String>>,
}

impl GeminiToolIds {
    // Keeps an id Gemini sent and synthesizes one only when it is absent.
    fn call_id(&mut self, id: Option<&str>, name: &str, policy: &TranslationPolicy) -> String {
        self.counter += 1;
        let id = id
            .filter(|id| !id.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| match &policy.deterministic_ids {
                DeterministicIdPolicy::GenerateStable { prefix } => stable_id(prefix, self.counter),
                DeterministicIdPolicy::Preserve => String::new(),
            });
        self.pending
            .entry(name.to_string())
            .or_default()
            .push_back(id.clone());
        id
    }

    // Uses the response's own id when present, else the oldest unanswered call to that name.
    fn result_id(&mut self, id: Option<&str>, name: &str) -> String {
        let queue = self.pending.get_mut(name);
        if let Some(id) = id.filter(|id| !id.is_empty()) {
            if let Some(queue) = queue
                && let Some(position) = queue.iter().position(|pending| pending == id)
            {
                queue.remove(position);
            }
            return id.to_string();
        }
        queue
            .and_then(VecDeque::pop_front)
            .unwrap_or_else(|| name.to_string())
    }
}

// Decodes a Gemini `parts` array into IR content blocks.
fn decode_gemini_parts(
    parts: Option<&Value>,
    path: &str,
    ids: &mut GeminiToolIds,
    policy: &TranslationPolicy,
) -> Result<Vec<ContentBlock>> {
    let Some(parts) = parts else {
        return Ok(Vec::new());
    };
    let parts = parts
        .as_array()
        .ok_or_else(|| TranslationError::InvalidType {
            path: path.to_string(),
            expected: "array",
        })?;
    Ok(parts
        .iter()
        .map(|part| match part.as_object() {
            Some(part) => decode_gemini_part(part, ids, policy),
            None => gemini_unknown(part.clone()),
        })
        .collect())
}

// Decodes one Gemini part. `thought` and `thoughtSignature` are siblings of the data field,
// not part kinds of their own.
fn decode_gemini_part(
    part: &Map<String, Value>,
    ids: &mut GeminiToolIds,
    policy: &TranslationPolicy,
) -> ContentBlock {
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        if part.get("thought").and_then(Value::as_bool) == Some(true) {
            return ContentBlock::Reasoning {
                text: text.to_string(),
                signature: part
                    .get("thoughtSignature")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                details: Vec::new(),
            };
        }
        return ContentBlock::Text {
            text: text.to_string(),
        };
    }
    if let Some(call) = part.get("functionCall").and_then(Value::as_object) {
        let name = gemini_function_name(call);
        return ContentBlock::ToolCall(ToolCall {
            id: ids.call_id(call.get("id").and_then(Value::as_str), &name, policy),
            arguments: call.get("args").cloned().unwrap_or_else(|| json!({})),
            name,
        });
    }
    if let Some(result) = part.get("functionResponse").and_then(Value::as_object) {
        let name = gemini_function_name(result);
        return ContentBlock::ToolResult(ToolResult {
            tool_call_id: ids.result_id(result.get("id").and_then(Value::as_str), &name),
            content: vec![ContentBlock::Text {
                text: result.get("response").map(json_string).unwrap_or_default(),
            }],
            is_error: None,
        });
    }
    if let Some(blob) = part.get("inlineData").and_then(Value::as_object) {
        let media_type = blob
            .get("mimeType")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let data = blob
            .get("data")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        return match media_kind(media_type.as_deref()) {
            Some("image") => ContentBlock::Image {
                source: ImageSource::Base64 { media_type, data },
            },
            Some("audio") => ContentBlock::Audio {
                source: MediaSource::Base64 { media_type, data },
            },
            Some("video") => ContentBlock::Video {
                source: MediaSource::Base64 { media_type, data },
            },
            // Documents travel as data URIs in the IR, which keeps their MIME type.
            _ => ContentBlock::File {
                source: FileSource::FileData {
                    data: match media_type {
                        Some(media_type) => format!("data:{media_type};base64,{data}"),
                        None => data,
                    },
                    filename: blob
                        .get("displayName")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                },
            },
        };
    }
    if let Some(file) = part.get("fileData").and_then(Value::as_object)
        && let Some(url) = file.get("fileUri").and_then(Value::as_str)
    {
        let media_type = file
            .get("mimeType")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let url = url.to_string();
        match media_kind(media_type.as_deref()) {
            Some("image") => {
                return ContentBlock::Image {
                    source: ImageSource::Url { url, detail: None },
                };
            }
            Some("audio") => {
                return ContentBlock::Audio {
                    source: MediaSource::Url { url, media_type },
                };
            }
            Some("video") => {
                return ContentBlock::Video {
                    source: MediaSource::Url { url, media_type },
                };
            }
            _ => {}
        }
    }
    gemini_unknown(Value::Object(part.clone()))
}

// Encodes IR content blocks as Gemini parts. `names` maps tool-call ids to function names.
fn encode_gemini_parts(
    content: &[ContentBlock],
    names: &BTreeMap<&str, &str>,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<Value>> {
    let mut parts = Vec::new();
    for block in content {
        let part = match block {
            ContentBlock::Text { text } | ContentBlock::Refusal { text } => json!({"text": text}),
            ContentBlock::Reasoning {
                text, signature, ..
            } => {
                if text.is_empty() && signature.is_none() {
                    continue;
                }
                let mut part = json!({"text": text, "thought": true});
                if let Some(signature) = signature {
                    part["thoughtSignature"] = json!(signature);
                }
                part
            }
            ContentBlock::ToolCall(call) => {
                let mut function_call =
                    json!({"name": call.name, "args": gemini_args(&call.arguments)});
                if let Some(id) = gemini_wire_id(&call.id, policy) {
                    function_call["id"] = json!(id);
                }
                json!({"functionCall": function_call})
            }
            ContentBlock::ToolResult(result) => {
                if result.content.iter().any(|block| {
                    !matches!(
                        block,
                        ContentBlock::Text { .. } | ContentBlock::Refusal { .. }
                    )
                }) {
                    push_lossy(
                        diagnostics,
                        policy,
                        "Gemini function responses carry only text here; non-text tool-result content was dropped",
                    )?;
                }
                let name = names
                    .get(result.tool_call_id.as_str())
                    .copied()
                    .unwrap_or(result.tool_call_id.as_str());
                let mut function_response =
                    json!({"name": name, "response": gemini_function_response(result)});
                if let Some(id) = gemini_wire_id(&result.tool_call_id, policy) {
                    function_response["id"] = json!(id);
                }
                json!({"functionResponse": function_response})
            }
            ContentBlock::Image {
                source: ImageSource::Url { url, .. },
            } => gemini_url_part(url, None),
            ContentBlock::Image {
                source: ImageSource::Base64 { media_type, data },
            }
            | ContentBlock::Audio {
                source: MediaSource::Base64 { media_type, data },
            }
            | ContentBlock::Video {
                source: MediaSource::Base64 { media_type, data },
            } => gemini_inline_data(media_type.as_deref(), data, None),
            ContentBlock::Audio {
                source: MediaSource::Url { url, media_type },
            }
            | ContentBlock::Video {
                source: MediaSource::Url { url, media_type },
            } => gemini_url_part(url, media_type.as_deref()),
            ContentBlock::File {
                source: FileSource::FileData { data, filename },
            } if split_data_uri(data).is_some() => {
                let (media_type, payload) = split_data_uri(data).unwrap_or_default();
                gemini_inline_data(Some(media_type), payload, filename.as_deref())
            }
            ContentBlock::Unknown { provider, raw } if provider.as_str() == GEMINI_FORMAT => {
                raw.clone()
            }
            other => {
                push_lossy(
                    diagnostics,
                    policy,
                    format!(
                        "Gemini parts cannot carry this {} content; it was dropped",
                        content_kind(other)
                    ),
                )?;
                continue;
            }
        };
        parts.push(part);
    }
    Ok(parts)
}

// Reads the required function name from a `functionCall` or `functionResponse`.
fn gemini_function_name(object: &Map<String, Value>) -> String {
    object
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

// Wraps a part the IR has no shape for, so a Gemini-to-Gemini hop can replay it.
fn gemini_unknown(raw: Value) -> ContentBlock {
    ContentBlock::Unknown {
        provider: FormatId::new(GEMINI_FORMAT),
        raw,
    }
}

// Returns the top-level MIME type, such as `image` for `image/png`.
fn media_kind(media_type: Option<&str>) -> Option<&str> {
    media_type
        .and_then(|media_type| media_type.split_once('/'))
        .map(|(kind, _)| kind)
}

// Splits `data:<media type>[;<parameter>...];base64,<payload>`.
fn split_data_uri(url: &str) -> Option<(&str, &str)> {
    let (metadata, data) = url.strip_prefix("data:")?.split_once(',')?;
    let parameters = metadata.strip_suffix(";base64")?;
    let media_type = parameters
        .split_once(';')
        .map_or(parameters, |(media_type, _)| media_type);
    (!media_type.is_empty()).then_some((media_type, data))
}

// Builds an `inlineData` part.
fn gemini_inline_data(media_type: Option<&str>, data: &str, display_name: Option<&str>) -> Value {
    let mut blob = json!({"data": data});
    if let Some(media_type) = media_type {
        blob["mimeType"] = json!(media_type);
    }
    if let Some(display_name) = display_name {
        blob["displayName"] = json!(display_name);
    }
    json!({"inlineData": blob})
}

// Inlines a base64 data URI and references any other URL through `fileData`.
fn gemini_url_part(url: &str, media_type: Option<&str>) -> Value {
    if let Some((media_type, data)) = split_data_uri(url) {
        return gemini_inline_data(Some(media_type), data, None);
    }
    let mut file = json!({"fileUri": url});
    if let Some(media_type) = media_type {
        file["mimeType"] = json!(media_type);
    }
    json!({"fileData": file})
}

// Names a content block for a lossy-conversion diagnostic.
fn content_kind(block: &ContentBlock) -> &'static str {
    match block {
        ContentBlock::Image { .. } => "image",
        ContentBlock::Audio { .. } => "audio",
        ContentBlock::Video { .. } => "video",
        ContentBlock::File { .. } => "file",
        _ => "provider-specific",
    }
}

// Returns the id to put on the Gemini wire. A synthesized id was never issued by any provider,
// so it is withheld and Gemini pairs the call and its response by name and order instead.
fn gemini_wire_id<'a>(id: &'a str, policy: &TranslationPolicy) -> Option<&'a str> {
    // Recognizes this policy's stable-id shape; a provider id of that exact shape
    // is also withheld, which Gemini tolerates by falling back to name order.
    let synthesized = match &policy.deterministic_ids {
        DeterministicIdPolicy::GenerateStable { prefix } => id
            .strip_prefix(prefix.as_str())
            .and_then(|rest| rest.strip_prefix('_'))
            .is_some_and(|counter| {
                counter.len() >= 8 && counter.bytes().all(|byte| byte.is_ascii_digit())
            }),
        DeterministicIdPolicy::Preserve => false,
    };
    (!id.is_empty() && !synthesized).then_some(id)
}

// Gemini `args` must be an object, while OpenAI formats may carry a JSON string.
fn gemini_args(arguments: &Value) -> Value {
    match arguments {
        Value::Object(_) => arguments.clone(),
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(object)) => Value::Object(object),
            _ => json!({"raw": text}),
        },
        Value::Null => json!({}),
        other => json!({"value": other}),
    }
}

// Gemini `response` must be an object. JSON object text is sent as-is; anything else is
// wrapped under the `output` or `error` key Google documents for function results.
fn gemini_function_response(result: &ToolResult) -> Value {
    let text = text_from_blocks(&result.content, "\n");
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(object)) => Value::Object(object),
        _ if result.is_error == Some(true) => json!({"error": text}),
        _ => json!({"output": text}),
    }
}

// Decodes function declarations. Built-in tools such as `googleSearch` have no IR shape.
fn decode_gemini_tools(
    tools: Option<&Value>,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Vec<ToolDefinition>> {
    let mut definitions = Vec::new();
    for tool in tools.and_then(Value::as_array).into_iter().flatten() {
        let Some(declarations) = tool.get("functionDeclarations").and_then(Value::as_array) else {
            push_lossy(
                diagnostics,
                policy,
                "Gemini built-in tools have no normalized form; the tool was dropped",
            )?;
            continue;
        };
        for declaration in declarations {
            definitions.push(ToolDefinition {
                name: declaration
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                description: declaration
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                parameters: declaration
                    .get("parametersJsonSchema")
                    .cloned()
                    .or_else(|| declaration.get("parameters").map(json_schema_from_openapi))
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                strict: None,
            });
        }
    }
    Ok(definitions)
}

// Encodes tool definitions. `parametersJsonSchema` takes full JSON Schema, which OpenAI and
// Anthropic schemas are; `parameters` rejects keywords such as `additionalProperties`.
fn encode_gemini_tools(tools: &[ToolDefinition]) -> Value {
    let declarations = tools
        .iter()
        .map(|tool| {
            let mut declaration = json!({"name": tool.name});
            if let Some(description) = &tool.description {
                declaration["description"] = json!(description);
            }
            if !tool.parameters.is_null() {
                declaration["parametersJsonSchema"] = tool.parameters.clone();
            }
            declaration
        })
        .collect::<Vec<_>>();
    json!([{"functionDeclarations": declarations}])
}

// Converts a Gemini OpenAPI schema to JSON Schema by lower-casing its type names.
fn json_schema_from_openapi(schema: &Value) -> Value {
    match schema {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| {
                    let value = match value {
                        Value::String(name) if key == "type" => {
                            Value::String(name.to_ascii_lowercase())
                        }
                        other => json_schema_from_openapi(other),
                    };
                    (key.clone(), value)
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(json_schema_from_openapi).collect()),
        other => other.clone(),
    }
}

// Maps `toolConfig.functionCallingConfig` to a tool choice.
fn decode_gemini_tool_choice(config: &Value) -> Option<ToolChoice> {
    let calling = config.get("functionCallingConfig")?;
    let allowed = calling
        .get("allowedFunctionNames")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    Some(
        match (calling.get("mode").and_then(Value::as_str), allowed) {
            (Some("AUTO") | None, _) => ToolChoice::Auto,
            (Some("NONE"), _) => ToolChoice::None,
            (Some("ANY"), []) => ToolChoice::Required,
            (Some("ANY"), [name]) if name.is_string() => ToolChoice::Tool {
                name: name.as_str().unwrap_or_default().to_string(),
            },
            _ => ToolChoice::Raw(config.clone()),
        },
    )
}

// Maps a tool choice to `toolConfig`.
fn encode_gemini_tool_config(
    choice: &ToolChoice,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<Option<Value>> {
    let calling = match choice {
        ToolChoice::Auto => json!({"mode": "AUTO"}),
        ToolChoice::Required => json!({"mode": "ANY"}),
        ToolChoice::None => json!({"mode": "NONE"}),
        ToolChoice::Tool { name } => json!({"mode": "ANY", "allowedFunctionNames": [name]}),
        ToolChoice::Raw(value) if value.get("functionCallingConfig").is_some() => {
            return Ok(Some(value.clone()));
        }
        ToolChoice::Raw(_) => {
            push_lossy(
                diagnostics,
                policy,
                "tool choice has no Gemini functionCallingConfig equivalent; it was dropped",
            )?;
            return Ok(None);
        }
    };
    Ok(Some(json!({"functionCallingConfig": calling})))
}

// Maps Gemini structured output to the OpenAI-shaped `response_format` the IR carries.
fn decode_gemini_response_format(generation: &Map<String, Value>) -> Option<Value> {
    let schema = generation.get("responseJsonSchema").cloned().or_else(|| {
        generation
            .get("responseSchema")
            .map(json_schema_from_openapi)
    });
    if let Some(schema) = schema {
        // Gemini does not name the schema; the neutral shape needs a name.
        return Some(json!({
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }));
    }
    (generation.get("responseMimeType").and_then(Value::as_str) == Some("application/json"))
        .then(|| json!({"type": "json_object"}))
}

// Writes an OpenAI-shaped `response_format` into `generationConfig`.
fn encode_gemini_response_format(
    format: &Value,
    generation: &mut Map<String, Value>,
    diagnostics: &mut Vec<TranslationDiagnostic>,
    policy: &TranslationPolicy,
) -> Result<()> {
    match format.get("type").and_then(Value::as_str) {
        Some("json_schema") => {
            let Some(schema) = format
                .get("json_schema")
                .and_then(|json_schema| json_schema.get("schema"))
            else {
                return push_lossy(
                    diagnostics,
                    policy,
                    "json_schema response format has no schema; the requested format was dropped",
                );
            };
            generation.insert("responseMimeType".to_string(), json!("application/json"));
            generation.insert("responseJsonSchema".to_string(), schema.clone());
        }
        Some("json_object") => {
            generation.insert("responseMimeType".to_string(), json!("application/json"));
        }
        Some("text") => {}
        _ => {
            return push_lossy(
                diagnostics,
                policy,
                "response format has no Gemini equivalent; the requested format was dropped",
            );
        }
    }
    Ok(())
}

// Maps a Gemini finish reason. The explicit reason is read before any function call in the
// candidate, so an aborted or truncated call is not reported as a clean tool-use stop.
fn gemini_stop_reason(reason: Option<&str>, has_tool_call: bool) -> StopReason {
    match reason {
        Some("STOP") | None if has_tool_call => StopReason::ToolUse,
        Some("STOP") | None => StopReason::EndTurn,
        Some("MAX_TOKENS") => StopReason::MaxTokens,
        Some(
            "SAFETY"
            | "RECITATION"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT"
            | "SPII"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT"
            | "IMAGE_RECITATION",
        ) => StopReason::ContentFilter,
        Some(_) => StopReason::Unknown,
    }
}

// Maps a normalized stop reason to a Gemini finish reason.
fn gemini_finish_reason(reason: Option<StopReason>) -> &'static str {
    match reason {
        None | Some(StopReason::EndTurn | StopReason::ToolUse) => "STOP",
        Some(StopReason::MaxTokens) => "MAX_TOKENS",
        Some(StopReason::ContentFilter) => "SAFETY",
        Some(StopReason::Error | StopReason::Unknown) => "OTHER",
    }
}

// Normalizes `usageMetadata`. `promptTokenCount` includes cached tokens, while the IR's
// input count excludes them.
fn decode_gemini_usage(value: Option<&Value>) -> Usage {
    let Some(value) = value.and_then(Value::as_object) else {
        return Usage::default();
    };
    let count = |key: &str| value.get(key).and_then(Value::as_u64);
    let cached = count("cachedContentTokenCount");
    Usage {
        input_tokens: count("promptTokenCount")
            .map(|tokens| tokens.saturating_sub(cached.unwrap_or(0))),
        cache: Usage::cache_details(cached, None),
        output_tokens: count("candidatesTokenCount"),
        total_tokens: count("totalTokenCount"),
        reasoning_tokens: count("thoughtsTokenCount"),
    }
}

// Encodes normalized usage as `usageMetadata`.
fn encode_gemini_usage(usage: &Usage) -> Value {
    let cached = usage.cached_input_tokens();
    let prompt = usage.input_tokens.unwrap_or(0)
        + cached.unwrap_or(0)
        + usage.cache_creation_input_tokens().unwrap_or(0);
    let candidates = usage.output_tokens.unwrap_or(0);
    let mut value = json!({
        "promptTokenCount": prompt,
        "candidatesTokenCount": candidates,
        "totalTokenCount": usage
            .total_tokens
            .unwrap_or(prompt + candidates + usage.reasoning_tokens.unwrap_or(0)),
    });
    if let Some(cached) = cached {
        value["cachedContentTokenCount"] = json!(cached);
    }
    if let Some(reasoning) = usage.reasoning_tokens {
        value["thoughtsTokenCount"] = json!(reasoning);
    }
    value
}
