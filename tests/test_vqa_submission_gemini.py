from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from trec_video_2026.vqa_gate import canonical_request_sha256
from trec_video_2026.vqa_submission import (
    PASS_IDS,
    AnswerRequest,
    VerifierCandidate,
    VerifierRequest,
    public_candidate_id,
)
from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    GeminiSubmissionClient,
    ProviderAuthenticationError,
    ProviderError,
    ProviderProtocolError,
    TokenUsage,
    UploadedFileRecord,
)


MODEL_ID = "gemini-3.1-pro-preview"


def _valid_answer_payload() -> dict[str, Any]:
    return {"answer": "four times"}


def _valid_timeline_payload() -> dict[str, Any]:
    span = {
        "span_id": "span-1",
        "start_seconds": 1.0,
        "end_seconds": 2.5,
        "modalities": ["visual", "speech"],
        "description": "A person enters the room.",
    }
    return {"events": [span], "support_spans": [span]}


class FakeTransport:
    def __init__(
        self,
        *,
        answer_payload: dict[str, Any] | None = None,
        sequence: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.answer_payload = answer_payload
        self.sequence = sequence
        self.error = error
        self.uploaded_ids: list[str] = []
        self.deleted_ids: list[str] = []
        self.generate_count = 0
        self.parts_by_call: list[tuple[object, ...]] = []
        self.schemas: list[dict[str, object]] = []

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        file_id = f"file-{len(self.uploaded_ids) + 1}"
        self.uploaded_ids.append(file_id)
        return UploadedFileRecord(
            file_id=file_id,
            mime_type=mime_type,
            uri=f"https://upload.invalid/{file_id}",
        )

    def generate(
        self,
        *,
        model: str,
        parts: tuple[object, ...],
        schema: dict[str, object],
    ) -> GenerationResult:
        assert model == MODEL_ID
        self.generate_count += 1
        self.parts_by_call.append(parts)
        self.schemas.append(schema)
        if self.error is not None:
            raise self.error
        if self.sequence is not None:
            payload = self.sequence[self.generate_count - 1]
        else:
            assert self.answer_payload is not None
            payload = self.answer_payload
        raw_response = json.dumps(payload, sort_keys=True).encode("utf-8")
        return GenerationResult(
            payload=payload,
            raw_response=raw_response,
            usage=TokenUsage(input_tokens=321, output_tokens=17),
        )

    def delete(self, file_id: str) -> None:
        self.deleted_ids.append(file_id)


def _answer_request(tmp_path: Path) -> AnswerRequest:
    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"video bytes")
    return AnswerRequest(
        run_id="run-1",
        condition_id="direct-av",
        video_id="video-1",
        question="How many times does the person wave?",
        pass_id="direct-av",
        frame_paths=(),
        frame_timestamps=(),
        transcript=None,
        media_path=media_path,
        audio_path=None,
        audio_mode="audio-on",
        evidence_policy="Return one concise supported response.",
    )


def _timeline_request(tmp_path: Path) -> dict[str, object]:
    media_path = tmp_path / "timeline.mp4"
    media_path.write_bytes(b"timeline bytes")
    return {
        "run_id": "run-1",
        "video_id": "video-1",
        "question": "What happens?",
        "media_path": str(media_path),
        "frame_paths": [],
        "frame_timestamps": [],
        "transcript": "[00:01.00] A person enters.",
        "duration_seconds": 5.0,
    }


def _verifier_request() -> VerifierRequest:
    candidates = tuple(
        VerifierCandidate(
            public_id=public_candidate_id("run-1", "video-1", pass_id, 7),
            candidate_text=f"candidate {index}",
            pass_id=pass_id,
            support_span_ids=("span-1",),
            confidence=0.8,
            source_request_sha256="a" * 64,
        )
        for index, pass_id in enumerate(PASS_IDS)
    )
    return VerifierRequest(
        run_id="run-1",
        video_id="video-1",
        question="What happens?",
        frame_paths=(),
        frame_timestamps=(),
        transcript="[00:01.00] A person enters.",
        timeline_payload={"events": [], "support_spans": []},
        candidates=candidates,
        seed=7,
    )


def _valid_verifier_payload(request: VerifierRequest) -> dict[str, Any]:
    public_ids = tuple(candidate.public_id for candidate in request.candidates)
    return {
        "ranked_public_ids": list(public_ids),
        "selected_public_id": public_ids[0],
        "support_by_public_id": {public_id: "span-1" for public_id in public_ids},
        "status_by_public_id": {public_id: "supported" for public_id in public_ids},
    }


def test_client_uploads_generates_records_usage_and_deletes(tmp_path: Path) -> None:
    request = _answer_request(tmp_path)
    transport = FakeTransport(answer_payload=_valid_answer_payload())

    result = GeminiSubmissionClient(transport, model_id=MODEL_ID).answer(request)

    assert result.answer == "four times"
    assert transport.deleted_ids == transport.uploaded_ids
    assert result.usage.input_tokens == 321
    assert result.usage.output_tokens == 17
    assert result.attempt_count == 1
    assert result.model_id == MODEL_ID
    assert result.request_sha256 == canonical_request_sha256(request.to_model_payload())
    assert len(result.response_sha256) == 64
    assert result.elapsed_seconds >= 0
    assert result.uploaded_file_deletion_status == {
        file_id: True for file_id in transport.uploaded_ids
    }


def test_client_deletes_upload_after_generation_failure(tmp_path: Path) -> None:
    transport = FakeTransport(
        error=RuntimeError("provider failed; Authorization: Bearer super-secret-token")
    )

    with pytest.raises(ProviderError, match="provider failed") as raised:
        GeminiSubmissionClient(transport, model_id=MODEL_ID).answer(
            _answer_request(tmp_path)
        )

    assert transport.deleted_ids == transport.uploaded_ids
    assert "super-secret-token" not in str(raised.value)
    assert "Authorization" not in str(raised.value)
    assert len(str(raised.value)) <= 256


def test_client_deletes_upload_after_protocol_failure(tmp_path: Path) -> None:
    transport = FakeTransport(answer_payload={"bad": True})

    with pytest.raises(ProviderProtocolError):
        GeminiSubmissionClient(transport, model_id=MODEL_ID).answer(
            _answer_request(tmp_path)
        )

    assert transport.deleted_ids == transport.uploaded_ids


def test_invalid_primary_output_gets_one_fixed_schema_repair(tmp_path: Path) -> None:
    transport = FakeTransport(
        sequence=[{"bad": True}, _valid_timeline_payload()]
    )

    result = GeminiSubmissionClient(transport, model_id=MODEL_ID).timeline(
        _timeline_request(tmp_path)
    )

    assert result.attempt_count == 2
    assert transport.generate_count == 2
    assert transport.deleted_ids == transport.uploaded_ids
    assert result.usage.input_tokens == 642
    assert "Repair the JSON object" in str(transport.parts_by_call[1][-1])
    assert '{"bad":true}' in str(transport.parts_by_call[1][-1])


def test_second_invalid_output_fails_without_third_call(tmp_path: Path) -> None:
    transport = FakeTransport(sequence=[{"bad": True}, {"still_bad": True}])

    with pytest.raises(ProviderProtocolError):
        GeminiSubmissionClient(transport, model_id=MODEL_ID).timeline(
            _timeline_request(tmp_path)
        )

    assert transport.generate_count == 2
    assert transport.deleted_ids == transport.uploaded_ids


def test_verifier_validates_current_complete_decision_schema() -> None:
    request = _verifier_request()
    transport = FakeTransport(answer_payload=_valid_verifier_payload(request))

    result = GeminiSubmissionClient(transport, model_id=MODEL_ID).verifier(request)

    assert result.decision.ranked_pass_ids == PASS_IDS
    assert result.decision.selected_pass_id == PASS_IDS[0]
    assert set(result.decision.status_by_pass or {}) == set(PASS_IDS)
    assert transport.deleted_ids == transport.uploaded_ids == []
    assert transport.schemas[0]["additionalProperties"] is False


def test_authentication_error_is_lazy_and_single_sanitized_message(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google", None)

    with pytest.raises(
        ProviderAuthenticationError,
        match="Gemini API or Vertex authentication is not configured",
    ) as raised:
        from trec_video_2026.vqa_submission_gemini import create_google_transport

        create_google_transport()

    assert str(raised.value) == (
        "Gemini API or Vertex authentication is not configured"
    )


def test_raw_response_hash_is_sha256_of_transport_bytes(tmp_path: Path) -> None:
    request = _answer_request(tmp_path)
    transport = FakeTransport(answer_payload=_valid_answer_payload())

    result = GeminiSubmissionClient(transport, model_id=MODEL_ID).answer(request)

    expected = hashlib.sha256(b'{"answer": "four times"}').hexdigest()
    assert result.response_sha256 == expected
