# Context-Window Handling

When an upstream rejects a request because the prompt exceeds the model's
context window, Switchyard calls the remaining targets on the same route in
configured order, stopping when one answers or every target has been tried.
Fallback applies only to the current request. An overflow is not remembered
across turns, so the route may select and try the same target again on the next
request.

## What counts as an overflow

Switchyard treats an upstream reply as a context overflow only when the status
is HTTP 400 **and** the body identifies a context-length error: an `error.code`
of `context_length_exceeded`, or a message containing a phrase such as
`maximum context length` or `prompt is too long`. An overflow reported any other
way — HTTP 413 or 422, for example — is not recognized as one. The request
fails on the spot with the upstream's status code, and the upstream's body is
returned inside Switchyard's error message.

Streaming requests are covered as well. Some engines reject an over-limit
streaming request with HTTP 200 and deliver the error as the first stream
event; when that first event carries a message matching the phrases above, the
call fails as an overflow and falls through before anything reaches the client.
An error event arriving after content has streamed is not retried — delivered
output cannot be taken back — and surfaces to the client as before.

## Targets that cannot serve the request

The same fallback applies when an upstream answers HTTP 400 and the message says
the model cannot take this input at all, such as `does not support image input`
or `is not a multimodal model`. The phrase list is short on purpose: any other 400
fails on the spot as before. The runtime log line for the fallback gives the reason
`capability`; the routing log does not record reasons. A rejecting target is not
remembered, so it is tried again on the next request. If every target rejects, the
caller gets the last target's error.

This covers the candidate list for the answer call, as on `random` and
`stage_router` routes. Calls made during routing still stop on the 400. That
includes the efficient-tier call on `llm_classifier` routes with
`mode = "escalation"` and on `vgr` routes, even when the capable tier could serve
the request. Hosts using the C interface run their own answer loop. Rejections of
unsupported tool schemas are not recognized.

`vision` is a route key, so `vision = false` rejects images for the whole route.
It cannot mark one target of a route as text-only.

## Configuration

There is no eviction key. A route falls through whenever it has more than one
target, so the route below needs nothing beyond its two tiers:

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.strong]
id = "openai/gpt-4o"
llm_client = "openrouter"

[targets.weak]
id = "openai/gpt-4o-mini"
llm_client = "openrouter"

[routes.stage]
id = "switchyard/stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 0.5
```

The response body, `x-model-router-selected-model` header, usage metrics, and
routing log name the candidate that actually served the request, including when
the client falls through from the algorithm's first choice.

## When every target overflows

Once no untried target is left — including a single-target `passthrough` route,
which has no alternative from the start — the request fails with HTTP 400 and an
error `code` of `context_length_exceeded`.
