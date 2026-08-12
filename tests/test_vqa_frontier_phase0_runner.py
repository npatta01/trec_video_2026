from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pytest

from trec_video_2026.vqa_frontier_phase0_runner import (
    ANSWER_POLICY,
    AmbiguousStartedAttemptError,
    CostGateError,
    MAP_POLICY,
    RequestBindingError,
    Phase0RunnerError,
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
from trec_video_2026.vqa_frontier_phase0_panel import (
    CATEGORIES,
    CONDITIONS,
    PHASE0_RUN_ID,
    QWEN_CONDITION_ID,
    REFERENCE_POLICY,
    SELECTION_POLICY,
)
from trec_video_2026.vqa_experiments.temporal_bottleneck import freeze_json_artifact
from trec_video_2026.vqa_submission import canonical_sha256
from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    ProviderAuthenticationError,
    ProviderError,
    ProviderProtocolError,
    ProviderResponseEvidenceError,
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
        assert request["request_sha256"] == request["request_binding_sha256"]
        assert request["request_sha256"] == canonical_sha256(request["request_binding"])


def test_failed_retry_cost_is_included_in_observed_condition_cost(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(tmp_path)
    transport = FailOnceTransport("transport")
    run_pilot(prepared, transport)

    projection = project_remaining_cost(prepared)

    assert projection["conditions"]["neutral_map"]["observed_calls"] == 1
    assert projection["conditions"]["neutral_map"]["observed_cost_usd"] > 0.0003


def test_concurrent_claim_allows_one_dispatch_and_blocks_the_other(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(tmp_path)
    first = BlockingTransport()
    second = RecordingTransport()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(run_pilot, prepared, first)
        assert first.started.wait(timeout=5)
        second_future = pool.submit(run_pilot, prepared, second)
        with pytest.raises(AmbiguousStartedAttemptError, match="claim"):
            second_future.result(timeout=5)
        first.release.set()
        assert first_future.result(timeout=5).passed == 1

    assert first.call_count == 1
    assert second.call_count == 0


def test_crash_window_claim_without_result_blocks_resume(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path)
    transport = RecordingTransport()
    run_pilot(prepared, transport)
    result_path = prepared.path / "results" / "neutral_map.jsonl"
    result_path.write_text("", encoding="utf-8")
    request = json.loads(
        (prepared.path / "requests.jsonl").read_text(encoding="utf-8").strip()
    )
    claim_path = prepared.path / "request-claims" / f"{request['request_sha256']}.json"
    claim_path.parent.mkdir(exist_ok=True)
    claim_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "claimed",
                "run_id": prepared.run_id,
                "request_sha256": request["request_sha256"],
                "request_binding_sha256": request["request_binding_sha256"],
                "request_binding": request["request_binding"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AmbiguousStartedAttemptError, match="claim"):
        run_pilot(prepared, RecordingTransport())


def test_unknown_cost_is_terminal_and_blocks_projection_and_resume(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(tmp_path)
    summary = run_pilot(prepared, UnknownCostTransport())

    assert summary.failed == 1
    result = json.loads(
        (prepared.path / "results" / "neutral_map.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert result["failure_code"] == "cost_unknown"
    assert result["cost_known"] is False
    assert result["cost_usd"] is None
    assert status(prepared)["status"] == "blocked-cost-unknown"
    with pytest.raises(CostGateError, match="cost_unknown"):
        project_remaining_cost(prepared)
    with pytest.raises(CostGateError, match="cost_unknown"):
        run_pilot(prepared, RecordingTransport())


def test_success_response_with_delete_failure_is_terminal_cleanup_failure(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(tmp_path)
    transport = DeleteFailureTransport()

    summary = run_pilot(prepared, transport)

    assert summary.failed == 1
    assert transport.call_count == 1
    result = json.loads(
        (prepared.path / "results" / "neutral_map.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert result["failure_code"] == "cleanup_failure"
    assert result["raw_response_sha256"]
    assert result["cost_known"] is True


@pytest.mark.parametrize(
    ("failure", "expected_retry"),
    [
        ("finish_length", True),
        ("value_error", False),
        ("os_error", False),
        ("protocol", False),
        ("auth", False),
        ("http_400", False),
        ("http_500", True),
    ],
)
def test_retry_classifier_only_retries_explicit_approved_cases(
    tmp_path: Path,
    failure: str,
    expected_retry: bool,
) -> None:
    transport = ClassifiedFailureTransport(failure)

    run_pilot(prepared_fixture(tmp_path), transport)

    assert transport.call_count == (2 if expected_retry else 1)


def test_duplicate_terminal_result_is_rejected_by_status_reader(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path)
    run_pilot(prepared, RecordingTransport())
    result_path = prepared.path / "results" / "neutral_map.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8").strip())
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")

    with pytest.raises(RequestBindingError, match="multiple terminal"):
        status(prepared)


@pytest.mark.parametrize("field", ["temperature", "response_format_policy"])
def test_request_binding_policy_tamper_blocks_resume(
    tmp_path: Path,
    field: str,
) -> None:
    prepared = prepared_fixture(tmp_path)
    run_pilot(prepared, RecordingTransport())
    request_path = prepared.path / "requests.jsonl"
    request = json.loads(request_path.read_text(encoding="utf-8").strip())
    request[field] = 99 if field == "temperature" else "wrong"
    request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")

    with pytest.raises(RequestBindingError):
        run_pilot(prepared, RecordingTransport())


@pytest.mark.parametrize("kind", ["provider", "policy", "mime", "delete"])
def test_transport_boundary_rejects_before_billable_generation(
    tmp_path: Path,
    kind: str,
) -> None:
    prepared = prepared_fixture(tmp_path)
    transport = BoundaryTransport(kind)

    if kind == "mime":
        run_pilot(prepared, transport)
    else:
        with pytest.raises(Phase0RunnerError):
            run_pilot(prepared, transport)

    assert transport.upload_count == (1 if kind == "mime" else 0)
    assert transport.call_count == 0


@pytest.mark.parametrize("mutation", ["run_id", "boolean", "category", "qwen"])
def test_panel_parser_rejects_tampered_frozen_manifest(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "panel.json"
    source.write_text(json.dumps(valid_panel_payload()), encoding="utf-8")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if mutation == "run_id":
        payload["run_id"] = "wrong"
    elif mutation == "boolean":
        payload["examples"][0]["has_audio"] = 1
    elif mutation == "category":
        payload["examples"][1]["category"] = "counting"
    else:
        payload["examples"][1].pop("qwen_condition_id")
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        prepare_run(source, tmp_path / "run")


def test_freeze_revalidates_retained_pilot_audit_hashes(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path, pilot_authorized=True)
    audit_path = prepared.path / "pilot-audit-freeze.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["artifact_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(PilotGateError, match="audit"):
        run_screen_row(
            prepared,
            ordinal=1,
            transport_factory=lambda _condition: RecordingTransport(),
        )


def test_cost_gate_rejects_forged_projection_snapshot(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path, pilot_authorized=True)
    run_screen_row(
        prepared,
        ordinal=1,
        transport_factory=lambda _condition: RecordingTransport(),
    )
    authorize_remaining(prepared)
    gate_path = prepared.path / "cost-gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["projection_snapshot"]["projected_total_usd"] = 0.0
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(CostGateError, match="snapshot"):
        run_screen_row(
            prepared,
            ordinal=2,
            transport_factory=lambda _condition: RecordingTransport(),
        )


def test_result_binding_projection_cannot_be_bypassed_by_rehashing_result(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(tmp_path)
    run_pilot(prepared, RecordingTransport())
    result_path = prepared.path / "results" / "neutral_map.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8").strip())
    result["temperature"] = 42.0
    unsigned = dict(result)
    del unsigned["result_sha256"]
    result["result_sha256"] = canonical_sha256(unsigned)
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    with pytest.raises(RequestBindingError, match="identity"):
        status(prepared)


def test_upload_failure_is_terminal_without_delete_or_retry(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path)
    transport = UploadFailureTransport()

    summary = run_pilot(prepared, transport)

    assert summary.failed == 1
    assert transport.upload_count == 1
    assert transport.delete_count == 0
    assert transport.call_count == 0
    result = json.loads(
        (prepared.path / "results" / "neutral_map.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert result["failure_code"] == "filesystem_error"


def test_retry_reserve_stops_before_second_call_when_first_failure_reaches_cap(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(tmp_path)
    transport = NearCapRetryTransport()

    summary = run_pilot(prepared, transport)

    assert summary.failed == 1
    assert transport.call_count == 1
    result = json.loads(
        (prepared.path / "results" / "neutral_map.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert result["failure_code"] == "cost_gate"
    assert result["cost_usd"] == pytest.approx(24.99)


def test_known_cost_at_cap_records_cap_state_and_blocks_new_condition(
    tmp_path: Path,
) -> None:
    prepared = prepared_fixture(
        tmp_path,
        pilot_authorized=True,
        cumulative_cost=25.0,
    )

    assert status(prepared)["status"] == "blocked-cost-cap"
    assert json.loads(
        (prepared.path / "cost-state.json").read_text(encoding="utf-8")
    )["state"] == "cap_reached"
    with pytest.raises(CostGateError, match="USD 25"):
        run_screen_row(
            prepared,
            ordinal=1,
            transport_factory=lambda _condition: RecordingTransport(),
        )


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
    panel = valid_panel_payload(rows)
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
        audit = {
            "schema_version": 1,
            "run_id": prepared.run_id,
            "panel_sha256": prepared.panel_sha256,
            "phase": "pilot",
            "decision": "usable_for_screen",
        }
        audit_path = audit_dir / "audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        freeze = freeze_json_artifact(
            audit_path,
            bindings={
                "run_id": prepared.run_id,
                "panel_sha256": prepared.panel_sha256,
                "phase": "pilot",
                "decision": "usable_for_screen",
            },
        )
        (audit_dir / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
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
    row: dict[str, Any] = {
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
    if phase == "screen":
        row["qwen_condition_id"] = QWEN_CONDITION_ID
        row["qwen_status"] = "passed"
    return row


def valid_panel_payload(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if rows is None:
        digest = "a" * 64
        rows = [
            _panel_row(1, "pilot", "pilot-video", digest, 1),
            *[
                _panel_row(
                    index + 2,
                    "screen",
                    f"screen-video-{index + 1}",
                    digest,
                    1,
                    question=SENTINEL_QUESTION,
                )
                for index in range(7)
            ],
        ]
    for index, row in enumerate(rows):
        if row["phase"] == "screen":
            row["category"] = CATEGORIES[index - 1]
            row["qwen_condition_id"] = QWEN_CONDITION_ID
            row["qwen_status"] = "passed"
    ids = [row["video_id"] for row in rows]
    return {
        "schema_version": 1,
        "run_id": PHASE0_RUN_ID,
        "selection_policy": SELECTION_POLICY,
        "reference_policy": REFERENCE_POLICY,
        "conditions": list(CONDITIONS),
        "pilot_count": 1,
        "screen_count": 7,
        "bindings": {
            "source_panel_sha256": "b" * 64,
            "qwen_predictions_sha256": "c" * 64,
            "selected_ordered_ids_sha256": canonical_sha256(ids),
            "source_media_sha256": canonical_sha256(
                [
                    {"video_id": row["video_id"], "source_sha256": row["media_sha256"]}
                    for row in rows
                ]
            ),
            "selection_policy": SELECTION_POLICY,
        },
        "examples": rows,
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
        self.policy_identity = {
            "neutral_map": MAP_POLICY,
            "direct_answer": ANSWER_POLICY,
            "map_guided_answer": ANSWER_POLICY,
        }

    provider_name = "openrouter"
    response_format_policy = "prompt-json-local-validation-v3-exact-fence"

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
    response_format_policy = "prompt-json-local-validation-v3-exact-fence"

    def __init__(self, failure: str) -> None:
        self.failure = failure
        self.request_hashes: list[str] = []
        self.call_count = 0
        self.policy_identity = {
            "neutral_map": MAP_POLICY,
            "direct_answer": ANSWER_POLICY,
            "map_guided_answer": ANSWER_POLICY,
        }

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
                    provider_metadata={
                        "provider": "fixture-route",
                        "model": "google/gemini-3.1-pro-preview",
                        "reported_cost_usd": 0.01,
                    },
                )
            if self.failure == "truncated":
                evidence = GenerationResult(
                    payload="",
                    raw_response='{"finish_reason":"length"}',
                    usage=TokenUsage(input_tokens=10, output_tokens=1),
                    provider_metadata={
                        "provider": "fixture-route",
                        "model": "google/gemini-3.1-pro-preview",
                        "reported_cost_usd": 0.01,
                    },
                )
                raise ProviderResponseEvidenceError(
                    "provider returned bounded evidence",
                    generation_result=evidence,
                )
            error = ProviderError(
                self.failure,
                failure_code="provider_request",
            )
            error.reported_cost_usd = 0.01
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


class BlockingTransport(RecordingTransport):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: object,
    ) -> GenerationResult:
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().generate(model=model, parts=parts, schema=schema)


class UnknownCostTransport(RecordingTransport):
    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: object,
    ) -> GenerationResult:
        payload = _neutral_payload()
        self.call_count += 1
        return GenerationResult(
            payload=payload,
            raw_response=json.dumps(payload, separators=(",", ":")),
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            provider_metadata={
                "provider": "fixture-route",
                "model": "google/gemini-3.1-pro-preview",
            },
        )


class DeleteFailureTransport(RecordingTransport):
    def delete(self, file_id: str) -> None:
        raise OSError("fixture delete failed")


class ClassifiedFailureTransport(RecordingTransport):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: object,
    ) -> GenerationResult:
        self.call_count += 1
        if self.call_count == 1:
            if self.failure == "finish_length":
                evidence = GenerationResult(
                    payload="",
                    raw_response='{"finish_reason":"length"}',
                    usage=TokenUsage(input_tokens=10, output_tokens=1),
                    provider_metadata={
                        "provider": "fixture-route",
                        "model": "google/gemini-3.1-pro-preview",
                        "reported_cost_usd": 0.01,
                    },
                )
                raise ProviderResponseEvidenceError(
                    "provider returned bounded evidence",
                    generation_result=evidence,
                )
            if self.failure == "value_error":
                raise ValueError("fixture programming error")
            if self.failure == "os_error":
                raise OSError("fixture filesystem error")
            if self.failure == "protocol":
                raise ProviderProtocolError("fixture schema protocol error")
            if self.failure == "auth":
                raise ProviderAuthenticationError("fixture auth error")
            error = ProviderError("fixture http error", failure_code="provider_request")
            error.status_code = 400 if self.failure == "http_400" else 500
            error.reported_cost_usd = 0.01
            raise error
        result = super().generate(model=model, parts=parts, schema=schema)
        self.call_count -= 1
        return result


class BoundaryTransport(RecordingTransport):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.upload_count = 0
        if kind == "provider":
            self.provider_name = "wrong-provider"
        elif kind == "policy":
            self.response_format_policy = "wrong-policy"
        elif kind == "delete":
            self.delete = None  # type: ignore[assignment]

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        self.upload_count += 1
        record = super().upload(path, mime_type=mime_type)
        if self.kind == "mime":
            return UploadedFileRecord(
                file_id=record.file_id,
                mime_type="video/mp4",
                sha256=record.sha256,
            )
        return record


class UploadFailureTransport(RecordingTransport):
    def __init__(self) -> None:
        super().__init__()
        self.upload_count = 0
        self.delete_count = 0

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        self.upload_count += 1
        raise OSError("fixture upload failed")

    def delete(self, file_id: str) -> None:
        self.delete_count += 1


class NearCapRetryTransport(RecordingTransport):
    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: object,
    ) -> GenerationResult:
        self.call_count += 1
        error = ProviderError("fixture transport failure", failure_code="provider_request")
        error.reported_cost_usd = 24.99
        raise error


@dataclass(frozen=True)
class Call:
    condition_id: str
    prompt: str
    schema: object
