// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for buffered translation into and out of Gemini `generateContent`.

pub mod common;

use pretty_assertions::assert_eq;
use serde_json::{Value, json};
use switchyard_translation::{
    DeterministicIdPolicy, TranslationEngine, TranslationError, TranslationPolicy, WireFormat,
};

use common::normalized_policy;

type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const GEMINI: &str = "gemini_generate_content";

// Builds the normalized policy that keeps missing ids empty instead of synthesizing them.
fn preserve_ids_policy() -> TranslationPolicy {
    TranslationPolicy {
        deterministic_ids: DeterministicIdPolicy::Preserve,
        ..normalized_policy()
    }
}

// Builds a one-call Gemini response with no call id.
fn gemini_call_response(name: &str) -> Value {
    json!({"candidates": [{
        "content": {"role": "model", "parts": [{"functionCall": {"name": name, "args": {}}}]},
        "finishReason": "STOP"
    }]})
}

// Verifies each function response keeps its own call's name when every Gemini turn reuses an id.
#[test]
fn gemini_function_responses_keep_their_names_across_turns() -> TestResult {
    let engine = TranslationEngine::default();
    for (label, policy) in [
        ("stable ids", normalized_policy()),
        ("preserved ids", preserve_ids_policy()),
    ] {
        let mut messages = vec![json!({"role": "user", "content": "weather, then time"})];
        for (name, output) in [("get_weather", "72F"), ("get_time", "noon")] {
            let chat = engine
                .translate_response(
                    GEMINI,
                    WireFormat::OpenAiChat,
                    &gemini_call_response(name),
                    &policy,
                )?
                .body;
            let assistant = chat["choices"][0]["message"].clone();
            let id = assistant["tool_calls"][0]["id"].clone();
            messages.push(assistant);
            messages.push(json!({"role": "tool", "tool_call_id": id, "content": output}));
        }

        let gemini = engine
            .translate_request(
                WireFormat::OpenAiChat,
                GEMINI,
                &json!({"model": "m", "messages": messages}),
                &policy,
            )?
            .body;

        let responses = gemini["contents"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|content| content["parts"].as_array().into_iter().flatten())
            .filter_map(|part| part.get("functionResponse"))
            .map(|response| (response["name"].clone(), response["response"].clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            responses,
            vec![
                (json!("get_weather"), json!({"output": "72F"})),
                (json!("get_time"), json!({"output": "noon"})),
            ],
            "{label}"
        );
    }
    Ok(())
}

// Verifies a failed tool result reaches Gemini under the `error` key whatever its content.
#[test]
fn failed_tool_results_reach_gemini_as_errors() -> TestResult {
    let engine = TranslationEngine::default();
    for (content, expected) in [
        (json!("not found"), json!({"error": "not found"})),
        (json!("{\"code\":404}"), json!({"error": {"code": 404}})),
    ] {
        let body = json!({
            "model": "m",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
                ]},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": "toolu_1",
                    "is_error": true, "content": content
                }]}
            ]
        });

        let gemini = engine
            .translate_request(
                WireFormat::AnthropicMessages,
                GEMINI,
                &body,
                &normalized_policy(),
            )?
            .body;

        assert_eq!(
            gemini["contents"][2]["parts"][0]["functionResponse"]["response"],
            expected
        );
        let anthropic = engine
            .translate_request(
                GEMINI,
                WireFormat::AnthropicMessages,
                &gemini,
                &normalized_policy(),
            )?
            .body;
        assert_eq!(
            anthropic["messages"][2]["content"][0]["is_error"], true,
            "{content}"
        );
    }
    Ok(())
}

// Verifies Gemini thought tokens count as output tokens in OpenAI Chat, and back.
#[test]
fn gemini_usage_counts_thoughts_as_completion_tokens() -> TestResult {
    let engine = TranslationEngine::default();
    let gemini = json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 20,
            "totalTokenCount": 35
        }
    });
    let chat_usage = json!({
        "prompt_tokens": 10,
        "completion_tokens": 25,
        "total_tokens": 35,
        "completion_tokens_details": {"reasoning_tokens": 20}
    });

    let chat = engine
        .translate_response(
            GEMINI,
            WireFormat::OpenAiChat,
            &gemini,
            &normalized_policy(),
        )?
        .body;
    assert_eq!(chat["usage"], chat_usage);

    let back = engine
        .translate_response(WireFormat::OpenAiChat, GEMINI, &chat, &normalized_policy())?
        .body;
    assert_eq!(back["usageMetadata"], gemini["usageMetadata"]);
    Ok(())
}

// Pins the known gap: a function-call signature survives only exact same-format replay.
// It is not carried through the IR, because Anthropic and OpenAI upstreams reject a foreign
// signature in history.
#[test]
fn gemini_function_call_thought_signature_needs_exact_preservation() -> TestResult {
    let engine = TranslationEngine::default();
    let request = json!({"contents": [
        {"role": "user", "parts": [{"text": "go"}]},
        {"role": "model", "parts": [
            {"functionCall": {"name": "f", "args": {}}, "thoughtSignature": "SIG"}
        ]},
        {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {"output": "ok"}}}]}
    ]});
    for (label, policy, expected) in [
        (
            "exact",
            TranslationPolicy::default(),
            request["contents"][1].clone(),
        ),
        (
            "normalized",
            normalized_policy(),
            json!({"role": "model", "parts": [{"functionCall": {"name": "f", "args": {}}}]}),
        ),
    ] {
        let gemini = engine
            .translate_request(GEMINI, GEMINI, &request, &policy)?
            .body;
        assert_eq!(gemini["contents"][1], expected, "{label}");
    }
    Ok(())
}

// Verifies a function response with no matching call does not send its name to Gemini as an id.
#[test]
fn orphan_gemini_function_response_gets_no_wire_id() -> TestResult {
    let engine = TranslationEngine::default();
    let request = json!({"contents": [{"role": "user", "parts": [
        {"functionResponse": {"name": "get_weather", "response": {"output": "ok"}}}
    ]}]});

    let chat = engine
        .translate_request(
            GEMINI,
            WireFormat::OpenAiChat,
            &request,
            &normalized_policy(),
        )?
        .body;
    let gemini = engine
        .translate_request(WireFormat::OpenAiChat, GEMINI, &chat, &normalized_policy())?
        .body;

    assert_eq!(gemini["contents"], request["contents"]);
    Ok(())
}

// Verifies a non-JSON response MIME type with a schema is kept on a Gemini-to-Gemini hop.
#[test]
fn gemini_enum_response_mime_type_survives_a_normalized_hop() -> TestResult {
    let engine = TranslationEngine::default();
    let request = json!({
        "contents": [{"role": "user", "parts": [{"text": "pick"}]}],
        "generationConfig": {
            "responseMimeType": "text/x.enum",
            "responseSchema": {"type": "STRING", "enum": ["a", "b"]}
        }
    });

    let gemini = engine
        .translate_request(GEMINI, GEMINI, &request, &normalized_policy())?
        .body;

    assert_eq!(
        gemini["generationConfig"]["responseMimeType"],
        "text/x.enum"
    );
    Ok(())
}

// Verifies stop sequences map between Gemini and the other formats.
#[test]
fn stop_sequences_map_to_and_from_gemini() -> TestResult {
    let engine = TranslationEngine::default();
    let gemini = json!({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"stopSequences": ["END"]}
    });
    for (target, field) in [
        ("openai_chat", "stop"),
        ("anthropic_messages", "stop_sequences"),
        (GEMINI, "generationConfig"),
    ] {
        let body = engine
            .translate_request(GEMINI, target, &gemini, &normalized_policy())?
            .body;
        let expected = if target == GEMINI {
            json!({"stopSequences": ["END"]})
        } else {
            json!(["END"])
        };
        assert_eq!(body[field], expected, "Gemini -> {target}");
    }

    for (source, body) in [
        (
            "openai_chat",
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stop": "END"}),
        ),
        (
            "anthropic_messages",
            json!({
                "model": "m", "max_tokens": 64, "stop_sequences": ["END"],
                "messages": [{"role": "user", "content": "hi"}]
            }),
        ),
    ] {
        let gemini = engine
            .translate_request(source, GEMINI, &body, &normalized_policy())?
            .body;
        assert_eq!(
            gemini["generationConfig"]["stopSequences"],
            json!(["END"]),
            "{source} -> Gemini"
        );
    }
    Ok(())
}

// Verifies a request with no conversation turns left is rejected instead of sent to Gemini.
#[test]
fn request_without_conversation_turns_is_rejected_for_gemini() {
    let engine = TranslationEngine::default();
    let body = json!({"model": "m", "messages": [{"role": "system", "content": "s"}]});

    let result =
        engine.translate_request(WireFormat::OpenAiChat, GEMINI, &body, &normalized_policy());

    assert!(
        matches!(result, Err(TranslationError::InvalidValue { ref path, .. }) if path == "$.contents"),
        "{result:?}"
    );
}

// Verifies requests from each format encode Gemini's structured output and tool config.
#[test]
fn requests_encode_gemini_structured_output_and_tool_config() -> TestResult {
    let engine = TranslationEngine::default();
    let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
    let tool =
        json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}});
    let cases = [
        (
            "openai_chat",
            json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [tool],
                "tool_choice": "required",
                "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}}
            }),
            json!({"functionCallingConfig": {"mode": "ANY"}}),
            json!({"responseMimeType": "application/json", "responseJsonSchema": schema}),
        ),
        (
            "openai_responses",
            json!({
                "model": "m",
                "input": "hi",
                "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
                "tool_choice": {"type": "function", "name": "f"},
                "text": {"format": {"type": "json_object"}}
            }),
            json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["f"]}}),
            json!({"responseMimeType": "application/json"}),
        ),
        (
            "anthropic_messages",
            json!({
                "model": "m",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "f", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "none"}
            }),
            json!({"functionCallingConfig": {"mode": "NONE"}}),
            json!({"maxOutputTokens": 64}),
        ),
    ];
    for (source, body, tool_config, generation_config) in cases {
        let gemini = engine
            .translate_request(source, GEMINI, &body, &normalized_policy())?
            .body;
        assert_eq!(gemini["toolConfig"], tool_config, "{source}");
        assert_eq!(gemini["generationConfig"], generation_config, "{source}");
    }
    Ok(())
}

// Verifies responses from each format encode Gemini finish reasons, usage and parts.
#[test]
fn responses_encode_gemini_candidates_and_usage() -> TestResult {
    let engine = TranslationEngine::default();
    let cases = [
        (
            "openai_chat",
            json!({
                "id": "c1",
                "model": "m",
                "choices": [{"index": 0, "finish_reason": "length",
                    "message": {"role": "assistant", "content": "cut"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            }),
            json!([{"text": "cut"}]),
            "MAX_TOKENS",
            json!({"promptTokenCount": 7, "candidatesTokenCount": 3, "totalTokenCount": 10}),
        ),
        (
            "openai_responses",
            json!({
                "id": "r1",
                "object": "response",
                "model": "m",
                "status": "completed",
                "output": [{"type": "function_call", "call_id": "call_9", "name": "f", "arguments": "{\"x\":1}"}],
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
            }),
            json!([{"functionCall": {"name": "f", "args": {"x": 1}, "id": "call_9"}}]),
            "STOP",
            json!({"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6}),
        ),
        (
            "anthropic_messages",
            json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "cache_read_input_tokens": 3, "output_tokens": 2}
            }),
            json!([{"text": "hi"}]),
            "STOP",
            json!({
                "promptTokenCount": 8,
                "cachedContentTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 10
            }),
        ),
    ];
    for (source, body, parts, finish_reason, usage) in cases {
        let gemini = engine
            .translate_response(source, GEMINI, &body, &normalized_policy())?
            .body;
        let candidate = &gemini["candidates"][0];
        assert_eq!(candidate["content"]["parts"], parts, "{source}");
        assert_eq!(candidate["finishReason"], finish_reason, "{source}");
        assert_eq!(gemini["usageMetadata"], usage, "{source}");
    }

    // A synthesized call id stays out of the Gemini wire body.
    let chat = engine
        .translate_response(
            GEMINI,
            WireFormat::OpenAiChat,
            &gemini_call_response("f"),
            &normalized_policy(),
        )?
        .body;
    let gemini = engine
        .translate_response(WireFormat::OpenAiChat, GEMINI, &chat, &normalized_policy())?
        .body;
    assert_eq!(
        gemini["candidates"][0]["content"]["parts"],
        json!([{"functionCall": {"name": "f", "args": {}}}])
    );
    Ok(())
}
