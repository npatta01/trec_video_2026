from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

import trec_video_2026.vqa_submission_openrouter as openrouter
from trec_video_2026.vqa_submission_gemini import (
    ProviderError,
    ProviderResponseEvidenceError,
    ProviderProtocolError,
    ProviderAuthenticationError,
    TokenUsage,
)


MODEL_ID = "gemini-3.1-pro-preview"
MODEL_SLUG = "google/gemini-3.1-pro-preview"


class FakeResponse:
    def __init__(self, payload: object, *, status: int = 200) -> None:
        self.status = status
        self._payload = json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True


class BoundedReadResponse:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.body if size < 0 else self.body[:size]

    def close(self) -> None:
        pass


class FakeOpener:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    def open(self, request: Any, *, timeout: float) -> FakeResponse:
        self.requests.append(request)
        assert timeout > 0
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)


def _success_response(content: str = '{"answer":"ok"}') -> FakeResponse:
    return FakeResponse(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 7,
                "total_tokens": 28,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        }
    )


def _transport(monkeypatch: pytest.MonkeyPatch, opener: FakeOpener) -> Any:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-secret")
    return openrouter.create_openrouter_transport(opener=opener)


def test_generate_posts_exact_multimodal_request_and_parses_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    image = tmp_path / "frame.jpg"
    audio = tmp_path / "speech.wav"
    video.write_bytes(b"video-bytes")
    image.write_bytes(b"image-bytes")
    audio.write_bytes(b"audio-bytes")
    opener = FakeOpener([_success_response()])
    transport = _transport(monkeypatch, opener)
    video_record = transport.upload(video, mime_type="video/mp4")
    image_record = transport.upload(image, mime_type="image/jpeg")
    audio_record = transport.upload(audio, mime_type="audio/wav")
    schema = {"type": "object", "additionalProperties": False}
    parts = (
        {"file_id": video_record.file_id, "mime_type": "video/mp4", "role": "native-media"},
        {"file_id": image_record.file_id, "mime_type": "image/jpeg", "role": "frame-0"},
        {"file_id": audio_record.file_id, "mime_type": "audio/wav", "role": "audio"},
        "What happened?",
    )

    result = transport.generate(model=MODEL_ID, parts=parts, schema=schema)

    request = opener.requests[0]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_header("Authorization")
    body = json.loads(request.data)
    assert body["model"] == MODEL_SLUG
    assert body["stream"] is False
    assert body["max_tokens"] >= 8192
    assert body["reasoning"] == {"effort": "high", "exclude": True}
    assert body["provider"] == {
        "zdr": True,
        "data_collection": "deny",
        "require_parameters": True,
    }
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == schema
    content = body["messages"]
    assert len(content) == 1
    multimodal = content[0]["content"]
    assert [item["type"] for item in multimodal] == [
        "video_url",
        "image_url",
        "input_audio",
        "text",
    ]
    assert multimodal[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert multimodal[1]["image_url"]["url"].endswith(
        base64.b64encode(b"image-bytes").decode("ascii")
    )
    assert multimodal[2]["input_audio"] == {
        "data": base64.b64encode(b"audio-bytes").decode("ascii"),
        "format": "wav",
    }
    assert multimodal[3] == {"type": "text", "text": "What happened?"}
    assert result.payload == {"answer": "ok"}
    assert result.raw_response == '{"answer":"ok"}'
    assert result.usage == TokenUsage(
        input_tokens=21,
        output_tokens=7,
        total_tokens=28,
        cached_input_tokens=3,
    )
    assert "temperature" not in body
    assert "order" not in body["provider"]
    assert "allow_fallbacks" not in body["provider"]

    second_opener = FakeOpener([_success_response()])
    second_transport = _transport(monkeypatch, second_opener)
    second_records = [
        second_transport.upload(path, mime_type=mime)
        for path, mime in (
            (video, "video/mp4"),
            (image, "image/jpeg"),
            (audio, "audio/wav"),
        )
    ]
    second_parts = tuple(
        {"file_id": record.file_id, "mime_type": mime, "role": role}
        for record, mime, role in zip(
            second_records,
            ("video/mp4", "image/jpeg", "audio/wav"),
            ("native-media", "frame-0", "audio"),
            strict=True,
        )
    ) + ("What happened?",)
    second_transport.generate(model=MODEL_ID, parts=second_parts, schema=schema)
    assert json.loads(second_opener.requests[0].data)["seed"] == body["seed"]


def test_frontier_policy_pins_route_and_disables_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = FakeOpener(
        [
            FakeResponse(
                {
                    "id": "gen-frontier-test",
                    "provider": "Google AI Studio",
                    "model": MODEL_SLUG,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"answer":"ok"}',
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 21,
                        "completion_tokens": 7,
                        "total_tokens": 28,
                        "cost": 0.0125,
                        "secret": "must-not-escape",
                    },
                    "untrusted": "must-not-escape",
                }
            )
        ]
    )
    transport = openrouter.OpenRouterTransport(
        "test-secret",
        opener=opener,
        policy=openrouter.OpenRouterRequestPolicy(
            max_output_tokens=16384,
            temperature=0.1,
            provider_order=("Google AI Studio",),
            allow_fallbacks=False,
        ),
    )

    result = transport.generate(
        model=MODEL_ID,
        parts=("Describe the clip.",),
        schema={"type": "object", "additionalProperties": False},
    )

    body = json.loads(opener.requests[-1].data)
    assert body["max_tokens"] == 16384
    assert body["temperature"] == 0.1
    assert body["provider"]["order"] == ["Google AI Studio"]
    assert body["provider"]["allow_fallbacks"] is False
    assert result.provider_metadata == {
        "provider": "Google AI Studio",
        "model": MODEL_SLUG,
        "request_id": "gen-frontier-test",
        "reported_cost_usd": 0.0125,
    }


def test_prompt_json_policy_omits_native_response_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = FakeOpener([_success_response()])
    transport = openrouter.OpenRouterTransport(
        "test-secret",
        opener=opener,
        policy=openrouter.OpenRouterRequestPolicy(
            allow_fallbacks=False,
            use_structured_outputs=False,
        ),
    )

    transport.generate(
        model=MODEL_ID,
        parts=("Return strict JSON.",),
        schema={"type": "object", "additionalProperties": False},
    )

    body = json.loads(opener.requests[0].data)
    assert "response_format" not in body
    assert body["provider"]["allow_fallbacks"] is False


def test_prompt_json_policy_preserves_non_json_text_for_local_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = FakeOpener([_success_response("```json\n{\"answer\":\"ok\"}\n```")])
    transport = openrouter.OpenRouterTransport(
        "test-secret",
        opener=opener,
        policy=openrouter.OpenRouterRequestPolicy(use_structured_outputs=False),
    )

    result = transport.generate(model=MODEL_ID, parts=("JSON",), schema={})

    assert result.raw_response == "```json\n{\"answer\":\"ok\"}\n```"
    assert result.payload == result.raw_response
    assert result.usage.total_tokens == 28


def test_null_optional_prompt_token_details_are_treated_as_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _success_response()
    payload = json.loads(response._payload)
    payload["usage"]["prompt_tokens_details"] = None
    opener = FakeOpener([FakeResponse(payload)])
    transport = _transport(monkeypatch, opener)

    result = transport.generate(model=MODEL_ID, parts=("JSON",), schema={})

    assert result.usage.cached_input_tokens == 0


def test_paid_response_content_is_preserved_when_usage_is_invalid() -> None:
    response = _success_response('{"answer":"preserve me"}')
    payload = json.loads(response._payload)
    payload["usage"] = "invalid"
    transport = openrouter.OpenRouterTransport(
        "test-secret",
        opener=FakeOpener([FakeResponse(payload)]),
        policy=openrouter.OpenRouterRequestPolicy(use_structured_outputs=False),
    )

    with pytest.raises(ProviderResponseEvidenceError) as caught:
        transport.generate(model=MODEL_ID, parts=("JSON",), schema={})

    evidence = caught.value.generation_result
    assert evidence.raw_response == '{"answer":"preserve me"}'
    assert evidence.response_sha256
    assert evidence.usage == TokenUsage(input_tokens=0, output_tokens=0)


@pytest.mark.parametrize(
    ("finish_reason", "content", "expected_empty"),
    [
        ("length", '{"partial":true}', False),
        ("stop", "", True),
        ("stop", "   ", True),
        ("stop", [], True),
        ("stop", [{"type": "text", "text": "  "}], True),
        ("stop", [{"type": "text"}], False),
        ("stop", None, None),
    ],
)
def test_evidence_preserves_bounded_finish_and_content_flags(
    finish_reason: str,
    content: object,
    expected_empty: bool | None,
) -> None:
    response = FakeResponse(
        {
            "id": "safe-evidence-id",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "cost": 0.01,
                "secret": "must-not-escape",
            },
            "response_body_secret": "must-not-escape",
        }
    )
    transport = openrouter.OpenRouterTransport(
        "test-secret",
        opener=FakeOpener([response]),
        policy=openrouter.OpenRouterRequestPolicy(use_structured_outputs=False),
    )

    with pytest.raises(ProviderResponseEvidenceError) as caught:
        transport.generate(model=MODEL_ID, parts=("JSON",), schema={})

    evidence = caught.value.generation_result
    assert evidence.provider_metadata["finish_reason"] == finish_reason
    if expected_empty is None:
        assert evidence.provider_metadata.get("content_empty") is not True
    else:
        assert evidence.provider_metadata["content_empty"] is expected_empty
    assert "choices" not in evidence.provider_metadata
    assert "response_body_secret" not in evidence.provider_metadata
    assert "secret" not in evidence.provider_metadata


def test_deterministic_seed_is_signed_31_bit_and_tracks_visible_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {"type": "object", "additionalProperties": False}
    seeds: list[int] = []
    for parts in (
        ("Describe the clip.",),
        ("Describe the clip.",),
        ("Describe the changed clip.",),
    ):
        opener = FakeOpener([_success_response()])
        transport = _transport(monkeypatch, opener)

        transport.generate(model=MODEL_ID, parts=parts, schema=schema)

        seed = json.loads(opener.requests[0].data)["seed"]
        assert isinstance(seed, int)
        seeds.append(seed)

    assert all(0 <= seed <= 2_147_483_647 for seed in seeds)
    assert seeds[0] == seeds[1]
    assert seeds[0] != seeds[2]


def test_response_read_is_bounded_and_oversized_body_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = BoundedReadResponse(b"x" * (openrouter.MAX_RESPONSE_BYTES + 1))
    transport = _transport(monkeypatch, FakeOpener([response]))

    with pytest.raises(ProviderProtocolError, match="size limit"):
        transport.generate(model=MODEL_ID, parts=("hello",), schema={})

    assert response.read_sizes == [openrouter.MAX_RESPONSE_BYTES + 1]


def test_oversized_serialized_request_is_rejected_before_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = FakeOpener([])
    transport = _transport(monkeypatch, opener)

    with pytest.raises(ProviderProtocolError, match="size limit"):
        transport.generate(
            model=MODEL_ID,
            parts=("x" * 30_000_000,),
            schema={},
        )

    assert opener.requests == []


def test_upload_is_local_and_delete_cleans_the_opaque_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"private-media")
    opener = FakeOpener([])
    transport = _transport(monkeypatch, opener)

    record = transport.upload(media, mime_type="video/mp4")

    assert record.uri is None
    assert record.file_id.startswith("local-sha256-")
    assert str(record) .find("private-media") == -1
    assert opener.requests == []
    transport.delete(record.file_id)
    transport.delete("local-sha256-missing")
    with pytest.raises(ProviderProtocolError, match="file identifier"):
        transport.generate(
            model=MODEL_ID,
            parts=({"file_id": record.file_id, "mime_type": "video/mp4", "role": "native-media"},),
            schema={},
        )


@pytest.mark.parametrize(
    ("mime_type", "payload"),
    [("application/pdf", b"not-media"), ("image/jpeg", b"too-large")],
)
def test_upload_rejects_unsupported_mime_and_bounded_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mime_type: str,
    payload: bytes,
) -> None:
    path = tmp_path / "media.bin"
    path.write_bytes(payload)
    transport = _transport(monkeypatch, FakeOpener([]))
    if mime_type == "image/jpeg":
        monkeypatch.setattr(openrouter, "MAX_UPLOAD_BYTES", 1)

    with pytest.raises(ProviderProtocolError):
        transport.upload(path, mime_type=mime_type)


def test_generate_rejects_duplicate_file_ids_and_malformed_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    opener = FakeOpener(
        [
            FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "[]"},
                        }
                    ],
                    "usage": {},
                }
            )
        ]
    )
    transport = _transport(monkeypatch, opener)
    record = transport.upload(media, mime_type="video/mp4")
    duplicate = {"file_id": record.file_id, "mime_type": "video/mp4", "role": "native-media"}

    with pytest.raises(ProviderProtocolError, match="duplicate"):
        transport.generate(model=MODEL_ID, parts=(duplicate, duplicate), schema={})
    with pytest.raises(ProviderProtocolError, match="JSON object"):
        transport.generate(model=MODEL_ID, parts=(duplicate,), schema={})


def test_distinct_identical_frames_have_distinct_ids_and_generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "frame-1.jpg"
    second = tmp_path / "frame-2.jpg"
    first.write_bytes(b"same-image")
    second.write_bytes(b"same-image")
    opener = FakeOpener([_success_response()])
    transport = _transport(monkeypatch, opener)

    first_record = transport.upload(first, mime_type="image/jpeg")
    second_record = transport.upload(second, mime_type="image/jpeg")

    assert first_record.file_id != second_record.file_id
    result = transport.generate(
        model=MODEL_ID,
        parts=(
            {
                "file_id": first_record.file_id,
                "mime_type": "image/jpeg",
                "role": "frame-0",
            },
            {
                "file_id": second_record.file_id,
                "mime_type": "image/jpeg",
                "role": "frame-1",
            },
            "Describe both frames.",
        ),
        schema={},
    )

    assert result.payload == {"answer": "ok"}
    body = json.loads(opener.requests[0].data)
    assert [item["type"] for item in body["messages"][0]["content"]] == [
        "image_url",
        "image_url",
        "text",
    ]


def test_http_and_provider_failures_are_bounded_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "router-test-secret"
    opener = FakeOpener(
        [FakeResponse({"error": {"message": f"{secret} https://bad.invalid/?key={secret}"}})]
    )
    transport = _transport(monkeypatch, opener)

    with pytest.raises(ProviderError) as raised:
        transport.generate(model=MODEL_ID, parts=("hello",), schema={})

    message = str(raised.value)
    assert secret not in message
    assert "bad.invalid" not in message
    assert len(message) <= 256


def test_missing_authentication_has_one_fixed_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ProviderAuthenticationError) as raised:
        openrouter.create_openrouter_transport(opener=FakeOpener([]))

    assert str(raised.value) == openrouter.AUTHENTICATION_ERROR_MESSAGE


def test_preflight_gets_models_without_posting_prompt_or_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = FakeOpener(
        [
            FakeResponse(
                {
                    "data": [
                        {
                            "id": MODEL_SLUG,
                            "architecture": {
                                "input_modalities": ["text", "image", "video", "audio"]
                            },
                            "supported_parameters": ["response_format"],
                        }
                    ]
                }
            )
        ]
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-secret")

    result = openrouter.preflight_openrouter_transport(
        model_id=MODEL_ID, opener=opener
    )

    assert result == {
        "authenticated": True,
        "model_id": MODEL_ID,
        "input_modalities": ["audio", "image", "text", "video"],
        "structured_outputs": True,
    }
    assert len(opener.requests) == 1
    request = opener.requests[0]
    assert request.full_url == "https://openrouter.ai/api/v1/models/user"
    assert request.get_method() == "GET"
    assert request.data is None
