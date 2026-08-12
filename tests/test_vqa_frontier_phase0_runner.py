from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pytest

from trec_video_2026.vqa_frontier_phase0_runner import (
    AmbiguousStartedAttemptError,
    CostGateError,
    PilotGateError,
    authorize_screen,
    freeze_pilot,
    freeze_screen,
    authorize_remaining,
    prepare_run,
    project_remaining_cost,
    run_pilot,
    run_screen_remaining,
    run_screen_row,
    status,
)
from trec_video_2026.vqa_submission import canonical_sha256
from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    ProviderError,
    TokenUsage,
    UploadedFileRecord,
)


SENTINEL_QUESTION = "sentinel private question"


def test_screen_row_dispatches_exactly_three_calls_with_one_neutral_map(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(tmp_path, pilot_authorized=True)
    transport = RecordingTransport()

    summary = run_screen_row(
        prepared,
        ordinal=1,
        transport_factory=lambda _condition: transport,
    )

    assert summary.passed == 3
    assert [call.condition_id for call in transport.calls] == [
        "neutral_map",
        "direct_answer",
        "map_guided_answer",
    ]
    assert SENTINEL_QUESTION not in transport.calls[0].prompt
    assert SENTINEL_QUESTION in transport.calls[1].prompt
    assert SENTINEL_QUESTION in transport.calls[2].prompt
    assert '"event_id":"E1"' in canonicalize(transport.calls[2].prompt)


def test_screen_cannot_run_without_usable_pilot_audit(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path, pilot_authorized=False)

    with pytest.raises(PilotGateError, match="pilot audit authorization"):
        run_screen_row(
            prepared,
            ordinal=1,
            transport_factory=lambda _condition: RecordingTransport(),
        )


def test_cost_gate_stops_before_twenty_five_dollars(tmp_path: Path) -> None:
    prepared = prepared_fixture(
        tmp_path,
        cumulative_cost=24.90,
        pilot_authorized=True,
    )
    transport = RecordingTransport(next_cost=0.11)

    with pytest.raises(CostGateError, match="USD 25"):
        run_screen_row(
            prepared,
            ordinal=1,
            transport_factory=lambda _condition: transport,
        )

    assert transport.call_count == 0


@pytest.mark.parametrize("failure", ["transport", "rate_limit", "empty", "truncated"])
def test_one_identical_retry_for_only_approved_failures(
    tmp_path: Path,
    failure: str,
) -> None:
    transport = FailOnceTransport(failure)

    run_pilot(prepared_fixture(tmp_path), transport)

    assert transport.request_hashes[0] == transport.request_hashes[1]
    assert len(transport.request_hashes) == 2


def test_completed_screen_row_resumes_without_provider_calls(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path, pilot_authorized=True)
    transport = RecordingTransport()

    first = run_screen_row(
        prepared,
        ordinal=1,
        transport_factory=lambda _condition: transport,
    )
    second = run_screen_row(
        prepared,
        ordinal=1,
        transport_factory=lambda _condition: transport,
    )

    assert first.passed == 3
    assert second.skipped == 3
    assert transport.call_count == 3


def test_started_attempt_without_terminal_record_blocks_resume(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path)
    transport = RecordingTransport()
    run_pilot(prepared, transport)
    attempts_path = prepared.path / "attempts.jsonl"
    started = json.loads(attempts_path.read_text(encoding="utf-8").splitlines()[0])
    started["status"] = "started"
    started["attempt_index"] = 99
    with attempts_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(started) + "\n")

    with pytest.raises(AmbiguousStartedAttemptError, match="started attempt"):
        run_pilot(prepared, transport)


def test_first_row_cost_gate_binds_all_result_hashes_and_screen_freezes(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(tmp_path, pilot_authorized=True)
    run_screen_row(
        prepared,
        ordinal=1,
        transport_factory=lambda _condition: RecordingTransport(),
    )
    projection = project_remaining_cost(prepared)
    assert projection["conditions"]["direct_answer"]["observed_calls"] == 1
    gate = authorize_remaining(prepared)
    assert set(gate["first_row_result_hashes"]) == {
        "neutral_map",
        "direct_answer",
        "map_guided_answer",
    }

    remaining = run_screen_remaining(
        prepared,
        transport_factory=lambda _condition: RecordingTransport(),
    )
    assert remaining.passed == 18
    freeze = freeze_screen(prepared)
    assert freeze["status"] == "frozen"
    assert status(prepared)["terminal_conditions"] == 22


def test_failed_validation_preserves_raw_body_without_repair_or_retry(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(tmp_path)
    transport = InvalidJsonTransport()

    summary = run_pilot(prepared, transport)

    assert summary.failed == 1
    assert transport.call_count == 1
    raw = prepared.path / "raw" / "neutral_map" / "pilot-video.json"
    assert json.loads(raw.read_text(encoding="utf-8"))["raw_response"] == "not-json"
    result = json.loads(
        (prepared.path / "results" / "neutral_map.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert result["failure_code"] == "invalid_json"


def test_every_completed_condition_deletes_transport_upload(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path, pilot_authorized=True)
    transport = RecordingTransport()

    run_screen_row(
        prepared,
        ordinal=1,
        transport_factory=lambda _condition: transport,
    )

    assert len(transport.deleted_file_ids) == 3


def test_request_hashes_bind_delivery_schema_and_guided_map_only(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path, pilot_authorized=True)
    run_screen_row(
        prepared,
        ordinal=1,
        transport_factory=lambda _condition: RecordingTransport(),
    )
    requests = [
        json.loads(line)
        for line in (prepared.path / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_condition = {request["condition_id"]: request for request in requests if request["phase"] == "screen"}
    assert by_condition["neutral_map"]["delivery_sha256"]
    assert by_condition["neutral_map"]["schema_sha256"] != by_condition["direct_answer"]["schema_sha256"]
    assert by_condition["direct_answer"]["max_output_tokens"] == 4096
    assert by_condition["map_guided_answer"]["max_output_tokens"] == 4096
    assert by_condition["direct_answer"]["neutral_map_sha256"] is None
    assert by_condition["map_guided_answer"]["neutral_map_sha256"]
    for request in by_condition.values():
        expected = dict(request)
        del expected["request_sha256"]
        assert request["request_sha256"] == canonical_sha256(expected)


def test_failed_retry_cost_is_included_in_observed_condition_cost(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(tmp_path)
    transport = FailOnceTransport("transport")
    run_pilot(prepared, transport)

    projection = project_remaining_cost(prepared)

    assert projection["conditions"]["neutral_map"]["observed_calls"] == 1
    assert projection["conditions"]["neutral_map"]["observed_cost_usd"] > 0.0003


def canonicalize(prompt: str) -> str:
    return prompt.replace(" ", "")


def prepared_fixture(
    tmp_path: Path,
    *,
    pilot_authorized: bool = False,
    cumulative_cost: float = 0.0,
):
    source = tmp_path / "source.webm"
    _write_fixture_media(source)
    media_bytes = source.read_bytes()
    media_sha256 = hashlib.sha256(media_bytes).hexdigest()
    rows: list[dict[str, Any]] = []
    rows.append(_panel_row(1, "pilot", "pilot-video", media_sha256, len(media_bytes)))
    for ordinal in range(1, 8):
        rows.append(
            _panel_row(
                ordinal + 1,
                "screen",
                f"screen-video-{ordinal}",
                media_sha256,
                len(media_bytes),
                question=SENTINEL_QUESTION,
            )
        )
    panel = {
        "schema_version": 1,
        "run_id": "vqa-frontier-phase0-20260812",
        "selection_policy": "synthetic-offline-fixture",
        "reference_policy": "not-read-not-serialized",
        "conditions": ["neutral_map", "direct_answer", "map_guided_answer"],
        "examples": rows,
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel), encoding="utf-8")
    prepared = prepare_run(panel_path, tmp_path / "run")
    fixture_root = prepared.path / "fixture-media"
    fixture_root.mkdir()
    for row in rows:
        shutil.copyfile(source, fixture_root / f"{row['video_id']}.webm")

    if pilot_authorized:
        pilot_transport = RecordingTransport(
            reported_cost_usd=cumulative_cost if cumulative_cost else None
        )
        run_pilot(prepared, pilot_transport)
        freeze_pilot(prepared)
        audit_dir = tmp_path / "pilot-audit"
        audit_dir.mkdir()
        (audit_dir / "freeze.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "frozen",
                    "bindings": {"neutral_map_decision": "usable_for_screen"},
                }
            ),
            encoding="utf-8",
        )
        authorize_screen(prepared, audit_dir / "freeze.json")
    return prepared


def _panel_row(
    ordinal: int,
    phase: str,
    video_id: str,
    media_sha256: str,
    size_bytes: int,
    *,
    question: str = "Pilot question is not sent to the neutral map.",
) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "phase": phase,
        "video_id": video_id,
        "question": question,
        "category": "temporal",
        "evidence_topology": "single",
        "selection_rationale": "complete fixture",
        "media_relative_path": f"fixture/{video_id}.webm",
        "media_sha256": media_sha256,
        "size_bytes": size_bytes,
        "duration_seconds": 2.0,
        "has_audio": True,
        "has_video": True,
        "condition_ids": ["neutral_map"]
        if phase == "pilot"
        else ["neutral_map", "direct_answer", "map_guided_answer"],
    }


def _write_fixture_media(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x32:r=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=8000",
            "-t",
            "2",
            "-c:v",
            "libvpx-vp9",
            "-c:a",
            "libopus",
            "-b:v",
            "80k",
            "-b:a",
            "16k",
            str(path),
        ],
        check=True,
    )


def _neutral_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "perceived_duration_seconds": 2.0,
        "global_description": "A blue fixture frame.",
        "entities": [],
        "events": [
            {
                "event_id": "E1",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "keyframe_seconds": [0.5],
                "description": "The fixture is visible.",
                "entity_ids": [],
                "actions": ["remain visible"],
                "relations": [],
                "state_before": "visible",
                "state_change": "none",
                "state_after": "visible",
                "visual_observations": ["blue frame"],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "confidence": 0.9,
                "uncertainties": [],
            }
        ],
        "global_uncertainties": [],
    }


def _answer_payload(*, guided: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "top1_answer": "The fixture remains visible.",
        "claims": [
            {
                "claim_id": "C1",
                "text": "The fixture remains visible.",
                "supporting_spans": [{"start_seconds": 0.0, "end_seconds": 1.0}],
                "supporting_modalities": ["video"],
                "map_event_ids": ["E1"] if guided else [],
            }
        ],
        "could_be_wrong_because": [],
    }


class RecordingTransport:
    def __init__(
        self,
        *,
        next_cost: float = 0.01,
        reported_cost_usd: float | None = None,
    ) -> None:
        self.next_cost = next_cost
        self.reported_cost_usd = reported_cost_usd
        self.calls: list[object] = []
        self.call_count = 0
        self.deleted_file_ids: list[str] = []

    provider_name = "openrouter"

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        return UploadedFileRecord(
            file_id=f"fixture-upload-{self.call_count + 1}",
            mime_type=mime_type,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def delete(self, file_id: str) -> None:
        self.deleted_file_ids.append(file_id)

    def estimate_cost_usd(self, provider: str, row: object) -> float:
        return self.next_cost

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: object,
    ) -> GenerationResult:
        prompt = str(parts[1])
        if "frontier-neutral-map-v1" in prompt:
            condition_id = "neutral_map"
            payload = _neutral_payload()
        elif "Neutral map: supplied" in prompt:
            condition_id = "map_guided_answer"
            payload = _answer_payload(guided=True)
        else:
            condition_id = "direct_answer"
            payload = _answer_payload(guided=False)
        self.call_count += 1
        self.calls.append(
            Call(condition_id=condition_id, prompt=prompt, schema=schema)
        )
        metadata: dict[str, object] = {
            "provider": "fixture-route",
            "model": "google/gemini-3.1-pro-preview",
            "request_id": f"fixture-{self.call_count}",
        }
        if self.reported_cost_usd is not None:
            metadata["reported_cost_usd"] = self.reported_cost_usd
        return GenerationResult(
            payload=payload,
            raw_response=json.dumps(payload, separators=(",", ":")),
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            provider_metadata=metadata,
        )


class FailOnceTransport:
    provider_name = "openrouter"

    def __init__(self, failure: str) -> None:
        self.failure = failure
        self.request_hashes: list[str] = []
        self.call_count = 0

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        return UploadedFileRecord(
            file_id=f"fail-once-upload-{self.call_count + 1}",
            mime_type=mime_type,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def delete(self, file_id: str) -> None:
        return None

    def estimate_cost_usd(self, provider: str, row: object) -> float:
        return 0.01

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: object,
    ) -> GenerationResult:
        self.call_count += 1
        self.request_hashes.append(
            canonical_sha256(
                {
                    "model": model,
                    "prompt": parts[1],
                    "schema": schema,
                }
            )
        )
        if self.call_count == 1:
            if self.failure == "empty":
                return GenerationResult(
                    payload="",
                    raw_response="",
                    usage=TokenUsage(input_tokens=10, output_tokens=0),
                    provider_metadata={"provider": "fixture-route", "model": "google/gemini-3.1-pro-preview"},
                )
            error = ProviderError(
                "truncated response" if self.failure == "truncated" else self.failure,
                failure_code="protocol_failure" if self.failure == "truncated" else "provider_request",
            )
            if self.failure == "rate_limit":
                error.status_code = 429
            raise error
        payload = _neutral_payload()
        return GenerationResult(
            payload=payload,
            raw_response=json.dumps(payload, separators=(",", ":")),
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            provider_metadata={
                "provider": "fixture-route",
                "model": "google/gemini-3.1-pro-preview",
            },
        )


class InvalidJsonTransport(RecordingTransport):
    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: object,
    ) -> GenerationResult:
        self.call_count += 1
        self.calls.append(
            Call(condition_id="neutral_map", prompt=str(parts[1]), schema=schema)
        )
        return GenerationResult(
            payload="not-json",
            raw_response="not-json",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            provider_metadata={
                "provider": "fixture-route",
                "model": "google/gemini-3.1-pro-preview",
            },
        )


@dataclass(frozen=True)
class Call:
    condition_id: str
    prompt: str
    schema: object
