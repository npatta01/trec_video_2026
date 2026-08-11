from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from trec_video_2026.vqa_frontier_map import EVENT_MAP_SCHEMA
from trec_video_2026.vqa_frontier_map_reka import (
    AUTHENTICATION_ERROR_MESSAGE,
    HTTP_TIMEOUT_SECONDS,
    MAX_RESPONSE_BYTES,
    REKA_CHAT_URL,
    REKA_MODEL_ID,
    REKA_MODELS_URL,
    RekaTransport,
    create_reka_transport,
    preflight_reka_transport,
)
from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    ProviderAuthenticationError,
    ProviderError,
    ProviderProtocolError,
)


def valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "question_interpretation": {
            "meaning": "Identify what changes after the person opens the box.",
            "required_modalities": ["visual"],
            "required_evidence": "The box before and after opening.",
        },
        "events": [
            {
                "event_id": "E1",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "keyframe_seconds": [1.0, 3.5],
                "scene": "A closed box is on a table.",
                "entities": ["person", "box", "table"],
                "actions": ["person approaches box"],
                "relations": ["box on table"],
                "state_before": "box closed",
                "state_change": "person opens box",
                "state_after": "box open",
                "visual_observations": ["lid moves from closed to open"],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "question_relevance": "direct",
                "confidence": 0.95,
                "gaps": [],
            }
        ],
        "support_spans": [
            {
                "event_ids": ["E1"],
                "start_seconds": 2.5,
                "end_seconds": 4.0,
                "modalities": ["visual"],
                "claim": "The box is open after the action.",
            }
        ],
        "top1_answer": "The box is open.",
        "support_explanation": "E1 shows the lid opening and the final open state.",
        "could_be_wrong_because": [
            "The object inside the box is not clearly visible."
        ],
    }


class FakeResponse:
    def __init__(self, payload: object, *, status: int = 200) -> None:
        self.status = status
        self.body = json.dumps(payload).encode("utf-8")
        self.read_sizes: list[int] = []
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.body if size < 0 else self.body[:size]

    def close(self) -> None:
        self.closed = True


class BoundedResponse:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.read_sizes: list[int] = []
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.body if size < 0 else self.body[:size]

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []
        self.timeouts: list[float] = []

    def open(self, request: Any, *, timeout: float) -> object:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)


def _response_payload(*, content: str | None = None) -> dict[str, object]:
    return {
        "id": "reka-request-1",
        "model": REKA_MODEL_ID,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content or json.dumps(valid_payload()),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 123,
            "completion_tokens": 321,
            "total_tokens": 444,
        },
    }


def _deterministic_seed(body: dict[str, object]) -> int:
    seed_input = {key: value for key, value in body.items() if key != "seed"}
    encoded = json.dumps(
        seed_input,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:16], 16) & 0x7FFF_FFFF


def test_reka_posts_one_native_video_request_and_parses_usage(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")
    opener = FakeOpener([FakeResponse(_response_payload())])
    transport = RekaTransport("test-secret", opener=opener)

    upload = transport.upload(video, mime_type="video/mp4")
    result = transport.generate(
        model=REKA_MODEL_ID,
        parts=[{"file_id": upload.file_id, "mime_type": "video/mp4"}, "frozen prompt"],
        schema=EVENT_MAP_SCHEMA,
    )

    assert isinstance(result, GenerationResult)
    assert result.payload == valid_payload()
    assert result.usage.input_tokens == 123
    assert result.usage.output_tokens == 321
    assert result.usage.total_tokens == 444
    assert len(opener.requests) == 1
    request = opener.requests[0]
    assert request.full_url == REKA_CHAT_URL
    assert request.get_method() == "POST"
    assert request.get_header("X-api-key") == "test-secret"
    assert opener.timeouts == [HTTP_TIMEOUT_SECONDS]

    body = json.loads(request.data)
    assert body["model"] == REKA_MODEL_ID
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 16_384
    assert body["stream"] is False
    assert body["seed"] == _deterministic_seed(body)
    content = body["messages"][0]["content"]
    assert [item["type"] for item in content] == ["video_url", "text"]
    assert content[0]["video_url"] == (
        "data:video/mp4;base64," + base64.b64encode(b"video-bytes").decode("ascii")
    )
    assert content[1]["text"].startswith("frozen prompt")
    assert json.dumps(
        EVENT_MAP_SCHEMA, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) in content[1]["text"]


def test_reka_rejects_invalid_json_without_repair_or_retry() -> None:
    opener = FakeOpener([FakeResponse(_response_payload(content="not-json"))])
    transport = RekaTransport("test-secret", opener=opener)

    with pytest.raises(ProviderProtocolError, match="valid JSON"):
        transport.generate(
            model=REKA_MODEL_ID,
            parts=["frozen prompt"],
            schema={"type": "object"},
        )

    assert len(opener.requests) == 1


def test_reka_locally_validates_response_schema() -> None:
    invalid = valid_payload()
    invalid.pop("top1_answer")
    opener = FakeOpener(
        [FakeResponse(_response_payload(content=json.dumps(invalid)))]
    )
    transport = RekaTransport("test-secret", opener=opener)

    with pytest.raises(ProviderProtocolError, match="schema"):
        transport.generate(
            model=REKA_MODEL_ID,
            parts=["frozen prompt"],
            schema=EVENT_MAP_SCHEMA,
        )

    assert len(opener.requests) == 1


def test_reka_bounds_response_reads_and_provider_errors() -> None:
    secret = "test-secret"
    response = BoundedResponse(b"x" * (MAX_RESPONSE_BYTES + 1))
    opener = FakeOpener([response])
    transport = RekaTransport(secret, opener=opener)

    with pytest.raises(ProviderProtocolError) as raised:
        transport.generate(
            model=REKA_MODEL_ID,
            parts=["frozen prompt"],
            schema={"type": "object"},
        )

    assert response.read_sizes == [MAX_RESPONSE_BYTES + 1]
    assert response.closed is True
    assert secret not in str(raised.value)
    assert len(str(raised.value)) <= 256


def test_reka_repr_and_http_errors_redact_credentials() -> None:
    secret = "reka-test-secret"
    opener = FakeOpener([FakeResponse({"error": {"message": secret}} , status=500)])
    transport = RekaTransport(secret, opener=opener)

    assert secret not in repr(transport)
    with pytest.raises(ProviderError) as raised:
        transport.generate(
            model=REKA_MODEL_ID,
            parts=["frozen prompt"],
            schema={"type": "object"},
        )
    assert secret not in str(raised.value)
    assert len(str(raised.value)) <= 256


def test_missing_reka_credentials_fail_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REKA_API_KEY", raising=False)
    opener = FakeOpener([])

    with pytest.raises(ProviderAuthenticationError) as raised:
        create_reka_transport(opener=opener)

    assert str(raised.value) == AUTHENTICATION_ERROR_MESSAGE
    assert opener.requests == []


def test_reka_preflight_checks_model_catalog() -> None:
    opener = FakeOpener(
        [
            FakeResponse(
                {
                    "data": [{"id": REKA_MODEL_ID}],
                }
            )
        ]
    )
    transport = RekaTransport("test-secret", opener=opener)

    result = transport.preflight(model_id=REKA_MODEL_ID)

    assert result == {"authenticated": True, "model_id": REKA_MODEL_ID}
    assert len(opener.requests) == 1
    assert opener.requests[0].full_url == REKA_MODELS_URL
    assert opener.requests[0].get_method() == "GET"


def test_preflight_factory_reads_only_reka_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REKA_API_KEY", "test-secret")
    opener = FakeOpener(
        [FakeResponse({"data": [{"id": REKA_MODEL_ID}]})]
    )

    result = preflight_reka_transport(opener=opener)

    assert result == {"authenticated": True, "model_id": REKA_MODEL_ID}
    assert len(opener.requests) == 1
