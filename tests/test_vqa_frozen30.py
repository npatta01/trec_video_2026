from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

import trec_video_2026.vqa_frozen30_workflow as frozen_workflow
from trec_video_2026.vqa_frozen30 import (
    EXPECTED_FROZEN_COUNT,
    FrozenExecution,
    SerialBenchmarkLock,
    append_attempt_rows,
    append_launch_decision,
    audit_frozen_requests,
    evidence_identity_sha256,
    pending_frozen_requests,
    select_complete_frozen_rows,
    validate_decision_bindings,
    validate_launch_decision,
    validate_frozen_attempt,
)
from trec_video_2026.vqa_gate import canonical_request_sha256
from trec_video_2026.vqa_frozen30_workflow import (
    assert_condition_transition,
    attach_expected_runtime_image,
    audit_whisper_requests,
    build_parser,
    checkpoint_chunks,
    freeze_selected_predictions,
    validate_transcript_sources,
)


FROZEN_IDS = tuple(f"video-{index:02d}" for index in range(EXPECTED_FROZEN_COUNT))


def make_request(tmp_path: Path, video_id: str) -> dict:
    frame = tmp_path / f"{video_id}.png"
    frame.write_bytes(video_id.encode())
    media = tmp_path / f"{video_id}.mkv"
    media.write_bytes((video_id + "-media").encode())
    return {
        "run_id": "native-audio-benchmark-v1",
        "condition_id": "qwen35-4b-24f-frozen30",
        "video_id": video_id,
        "question": "What happens?",
        "frame_paths": [str(frame.resolve())],
        "frame_timestamps": [0.5],
        "frame_descriptions": None,
        "transcript": None,
        "media_path": str(media.resolve()),
        "audio_path": None,
        "audio_mode": None,
    }


def execution() -> FrozenExecution:
    return FrozenExecution(
        condition_id="qwen35-4b-24f-frozen30",
        model_id="Qwen/Qwen3.5-4B",
        model_revision="a" * 40,
        runtime_image="local/image@sha256:" + "b" * 64,
        evidence_manifest_sha256="c" * 64,
        device="cuda:0 (ROCm)",
        precision="bfloat16",
    )


def passed_attempt(request: dict, launch_sha256: str) -> dict:
    return {
        "run_id": request["run_id"],
        "condition_id": request["condition_id"],
        "video_id": request["video_id"],
        "status": "passed",
        "answer": "A jump",
        "request": request,
        "request_sha256": canonical_request_sha256(request),
        "launch_decision_sha256": launch_sha256,
        "evidence_identity_sha256": evidence_identity_sha256(request),
        "evidence_manifest_sha256": execution().evidence_manifest_sha256,
        "model_id": execution().model_id,
        "model_revision": execution().model_revision,
        "runtime_image": execution().runtime_image,
        "device": execution().device,
        "precision": execution().precision,
        "elapsed_seconds": 1.0,
        "peak_memory_bytes": 1,
        "orchestration_chunk_id": "test-chunk",
        "outer_chunk_elapsed_seconds": 2.0,
        "outer_condition_elapsed_seconds": 2.0,
        "runtime": "test",
        "error_type": None,
        "error_message": None,
    }


def test_request_audit_requires_exact_ordered_frozen_panel_and_no_labels(
    tmp_path: Path,
) -> None:
    requests = [make_request(tmp_path, video_id) for video_id in FROZEN_IDS]

    summary = audit_frozen_requests(
        requests,
        expected_video_ids=FROZEN_IDS,
        condition_id=execution().condition_id,
    )

    assert summary["row_count"] == EXPECTED_FROZEN_COUNT
    assert summary["video_ids"] == list(FROZEN_IDS)
    assert len(summary["request_set_sha256"]) == 64

    leaked = [dict(row) for row in requests]
    leaked[0]["metadata"] = {"correct_answer": "A jump"}
    with pytest.raises(ValueError, match="label|unrecognized"):
        audit_frozen_requests(
            leaked,
            expected_video_ids=FROZEN_IDS,
            condition_id=execution().condition_id,
        )


def test_launch_decision_is_append_only_and_precedes_attempts(tmp_path: Path) -> None:
    requests = [make_request(tmp_path, video_id) for video_id in FROZEN_IDS]
    audit = audit_frozen_requests(
        requests,
        expected_video_ids=FROZEN_IDS,
        condition_id=execution().condition_id,
    )
    ledger = tmp_path / "promotion-decisions.jsonl"
    attempts = tmp_path / "attempts.jsonl"
    decision = {
        "schema_version": 1,
        "manifest_version": "trecvid-native-audio-2026-07-21-v1",
        "gate_selection_sha256": "c" * 64,
        "condition_id": execution().condition_id,
        "decision": "promote",
        "rationale": "Leading controlled sampled-frame baseline.",
        "recorded_at_utc": "2026-07-23T20:00:00Z",
        "request_set_sha256": audit["request_set_sha256"],
    }

    recorded = append_launch_decision(
        ledger,
        decision,
        attempts_path=attempts,
    )

    assert ledger.read_text().count("\n") == 1
    assert recorded["launch_decision_sha256"]
    validate_launch_decision(recorded)
    with pytest.raises(ValueError, match="checksum"):
        validate_launch_decision({**recorded, "rationale": "changed"})
    with pytest.raises(ValueError, match="already recorded"):
        append_launch_decision(ledger, decision, attempts_path=attempts)

    attempts.write_text(
        json.dumps({"condition_id": "another-condition"}) + "\n",
        encoding="utf-8",
    )
    second = {**decision, "condition_id": "another-condition"}
    with pytest.raises(ValueError, match="before its first attempt"):
        append_launch_decision(ledger, second, attempts_path=attempts)

    third = {**decision, "condition_id": "third-condition"}
    assert (
        append_launch_decision(
            ledger,
            third,
            attempts_path=attempts,
        )["condition_id"]
        == "third-condition"
    )


def test_resume_requires_exact_request_evidence_provenance_and_decision(
    tmp_path: Path,
) -> None:
    requests = [make_request(tmp_path, video_id) for video_id in FROZEN_IDS]
    launch_sha256 = "d" * 64
    good = passed_attempt(requests[0], launch_sha256)
    later_failure = {
        **good,
        "status": "failed",
        "answer": None,
        "error_type": "RunnerProcessError",
    }
    invalid_revision = {
        **passed_attempt(requests[1], launch_sha256),
        "model_revision": "e" * 40,
    }

    pending = pending_frozen_requests(
        requests,
        [good, later_failure, invalid_revision],
        execution=execution(),
        launch_decision_sha256=launch_sha256,
    )

    assert requests[0] not in pending
    assert requests[1] in pending
    assert len(pending) == EXPECTED_FROZEN_COUNT - 1
    assert not validate_frozen_attempt(
        requests[0],
        {**good, "peak_memory_bytes": None},
        execution=execution(),
        launch_decision_sha256=launch_sha256,
    )

    changed_frame = Path(requests[0]["frame_paths"][0])
    changed_frame.write_bytes(b"mutated")
    assert not validate_frozen_attempt(
        requests[0],
        good,
        execution=execution(),
        launch_decision_sha256=launch_sha256,
    )


def test_complete_selection_requires_all_expected_rows(tmp_path: Path) -> None:
    requests = [make_request(tmp_path, video_id) for video_id in FROZEN_IDS]
    launch_sha256 = "f" * 64
    attempts = [passed_attempt(request, launch_sha256) for request in requests]

    selected = select_complete_frozen_rows(
        requests,
        attempts,
        execution=execution(),
        launch_decision_sha256=launch_sha256,
    )

    assert len(selected) == EXPECTED_FROZEN_COUNT
    with pytest.raises(ValueError, match="expected 30"):
        select_complete_frozen_rows(
            requests,
            attempts[:-1],
            execution=execution(),
            launch_decision_sha256=launch_sha256,
        )


def test_evidence_identity_binds_transcript_source_and_file_bytes(
    tmp_path: Path,
) -> None:
    request = make_request(tmp_path, FROZEN_IDS[0])
    request["transcript"] = "[00:00:00.000] hello"
    request["transcript_source"] = {
        "attempt_sha256": "a" * 64,
        "quality": "accepted",
    }

    requests = [make_request(tmp_path, video_id) for video_id in FROZEN_IDS]
    requests[0] = request
    with pytest.raises(ValueError, match="unrecognized"):
        audit_frozen_requests(
            requests,
            expected_video_ids=FROZEN_IDS,
            condition_id=execution().condition_id,
        )

    first = evidence_identity_sha256(request)
    request["transcript_source"]["attempt_sha256"] = "b" * 64
    assert evidence_identity_sha256(request) != first


def test_frozen_promotion_config_records_decisions_before_launch() -> None:
    payload = json.loads(
        Path("configs/vqa/native_audio_frozen30_promotions.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == 1
    assert payload["manifest_version"] == "trecvid-native-audio-2026-07-21-v1"
    assert payload["gate_selection_sha256"] == (
        "099dedfa4d74aa7ed1c135a013c8b342a62e422a6692934507c94839a2803f69"
    )
    assert payload["recorded_before_any_frozen30_attempt"] is True
    assert payload["recorded_at_utc"] == "2026-07-24T00:28:27Z"
    assert [item["condition_id"] for item in payload["promoted_conditions"]] == [
        "qwen35-4b-12f-frozen30",
        "qwen35-4b-24f-frozen30",
        "minicpmo45-native-video-audio-off-frozen30",
        "qwen35-4b-12f-whisper-large-v3-frozen30",
    ]
    assert all(item["decision"] == "promote" for item in payload["promoted_conditions"])
    assert len(payload["stopped_conditions"]) >= 8


def test_v2_launch_config_marks_transcript_as_exploratory_not_promoted() -> None:
    payload = json.loads(
        Path("configs/vqa/native_audio_frozen30_v2_decisions.json").read_text(
            encoding="utf-8"
        )
    )
    decisions = payload["condition_decisions"]

    assert [item["decision"] for item in decisions[:3]] == ["promote"] * 3
    transcript = decisions[3]
    assert transcript["condition_id"] == frozen_workflow.TRANSCRIPT_CONDITION
    assert transcript["decision"] == "explore"
    assert transcript["promotion_path"] == "predeclared-exploratory-diagnostic"
    assert "missing visual-control gate" in transcript["rationale"]
    assert "METEOR -0.1032" in transcript["rationale"]
    assert "F1 -0.0552" in transcript["rationale"]


def test_launch_timestamps_require_config_before_authorization_and_now() -> None:
    now = datetime(2026, 7, 24, 4, 20, tzinfo=UTC)

    frozen_workflow.validate_launch_timestamp_order(
        "2026-07-24T04:10:00Z",
        "2026-07-24T04:15:00Z",
        now=now,
    )
    with pytest.raises(ValueError, match="before authorization"):
        frozen_workflow.validate_launch_timestamp_order(
            "2026-07-24T04:15:00Z",
            "2026-07-24T04:15:00Z",
            now=now,
        )
    with pytest.raises(ValueError, match="future"):
        frozen_workflow.validate_launch_timestamp_order(
            "2026-07-24T04:30:00Z",
            None,
            now=now,
        )


def test_exploratory_decision_can_be_authorized_but_is_not_promoted(
    tmp_path: Path,
) -> None:
    decision = {
        "schema_version": 1,
        "manifest_version": "manifest",
        "gate_selection_sha256": "a" * 64,
        "condition_id": "exploration",
        "decision": "explore",
        "rationale": "Predeclared exploratory measurement only.",
        "recorded_at_utc": "2026-07-24T00:00:00Z",
        "request_set_sha256": "b" * 64,
    }

    recorded = append_launch_decision(
        tmp_path / "decisions.jsonl",
        decision,
        attempts_path=tmp_path / "attempts.jsonl",
    )

    assert recorded["decision"] == "explore"


def test_attempt_append_is_fsynced_jsonl_and_rejects_identity_mismatch(
    tmp_path: Path,
) -> None:
    request = make_request(tmp_path, FROZEN_IDS[0])
    row = passed_attempt(request, "d" * 64)
    ledger = tmp_path / "attempts.jsonl"

    append_attempt_rows(ledger, [row])
    append_attempt_rows(ledger, [{**row, "status": "failed", "answer": None}])

    assert [json.loads(line)["status"] for line in ledger.read_text().splitlines()] == [
        "passed",
        "failed",
    ]
    with pytest.raises(ValueError, match="identity"):
        append_attempt_rows(ledger, [{**row, "video_id": "different"}])


def test_serial_lock_prevents_overlapping_conditions(tmp_path: Path) -> None:
    path = tmp_path / "serial.lock"

    with SerialBenchmarkLock(path, condition_id="first"):
        with pytest.raises(RuntimeError, match="already running"):
            with SerialBenchmarkLock(path, condition_id="second"):
                pass

    with SerialBenchmarkLock(path, condition_id="second"):
        assert json.loads(path.read_text())["condition_id"] == "second"


def test_frozen_workflow_exposes_separate_non_gpu_and_gpu_phases() -> None:
    parser = build_parser()

    assert parser.parse_args(["prepare"]).action == "prepare"
    assert parser.parse_args(["status"]).action == "status"
    assert parser.parse_args(["run-whisper"]).action == "run-whisper"
    assert (
        parser.parse_args(["authorize", "qwen35-4b-12f-frozen30"]).condition
        == "qwen35-4b-12f-frozen30"
    )
    assert (
        parser.parse_args(["run", "qwen35-4b-24f-frozen30"]).condition
        == "qwen35-4b-24f-frozen30"
    )
    assert parser.parse_args(["score"]).action == "score"


def test_transcript_sources_are_bound_outside_runner_request(tmp_path: Path) -> None:
    path = tmp_path / "transcript-sources.json"
    sources = [
        {
            "video_id": "video-01",
            "attempt_sha256": "a" * 64,
            "quality": {"status": "accepted"},
        }
    ]
    path.write_text(json.dumps(sources), encoding="utf-8")
    artifact_sha256 = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    evidence_sha256 = canonical_request_sha256({"transcript_sources": sources})

    validate_transcript_sources(
        path,
        expected_artifact_sha256=artifact_sha256,
        expected_evidence_sha256=evidence_sha256,
    )

    sources[0]["attempt_sha256"] = "b" * 64
    path.write_text(json.dumps(sources), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|selections"):
        validate_transcript_sources(
            path,
            expected_artifact_sha256=artifact_sha256,
            expected_evidence_sha256=evidence_sha256,
        )


def test_whisper_request_audit_requires_exact_order_schema_and_preflight_hash() -> None:
    requests = [
        {
            "run_id": frozen_workflow.RUN_ID,
            "condition_id": "whisper-large-v3-bf16-frozen30",
            "video_id": video_id,
            "audio_path": f"/audio/{video_id}.wav",
        }
        for video_id in FROZEN_IDS
    ]
    request_sha256 = canonical_request_sha256({"requests": requests})

    audit_whisper_requests(
        requests,
        expected_video_ids=FROZEN_IDS,
        expected_sha256=request_sha256,
    )
    with pytest.raises(ValueError, match="ordered"):
        audit_whisper_requests(
            list(reversed(requests)),
            expected_video_ids=FROZEN_IDS,
            expected_sha256=request_sha256,
        )
    leaked = [dict(row) for row in requests]
    leaked[0]["reference_answer"] = "secret"
    with pytest.raises(ValueError, match="field"):
        audit_whisper_requests(
            leaked,
            expected_video_ids=FROZEN_IDS,
            expected_sha256=request_sha256,
        )


def test_whisper_wrapper_binds_expected_runtime_image_and_command() -> None:
    image = "local/image@sha256:" + "a" * 64
    assert (
        attach_expected_runtime_image({"status": "passed"}, image)["runtime_image"]
        == image
    )
    with pytest.raises(ValueError, match="runtime image"):
        attach_expected_runtime_image(
            {"runtime_image": "local/other@sha256:" + "b" * 64},
            image,
        )


def test_condition_authorize_and_run_order_is_exact() -> None:
    order = ("first", "second", "third", "transcript")

    assert_condition_transition(
        "first", order=order, decisions=(), completed_counts={}, action="authorize"
    )
    with pytest.raises(ValueError, match="prior condition"):
        assert_condition_transition(
            "second",
            order=order,
            decisions=({"condition_id": "first"},),
            completed_counts={"first": 0},
            action="authorize",
        )
    with pytest.raises(ValueError, match="authorized"):
        assert_condition_transition(
            "second",
            order=order,
            decisions=({"condition_id": "first"},),
            completed_counts={"first": 30},
            action="run",
        )
    assert_condition_transition(
        "first",
        order=order,
        decisions=({"condition_id": "first"},),
        completed_counts={},
        action="run",
    )


def test_checkpoint_chunks_stop_at_exact_3_10_20_boundaries() -> None:
    assert checkpoint_chunks(30) == (3, 7, 10, 10)


def test_selected_prediction_hash_is_persisted_before_reference_loading(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.jsonl"
    freeze = tmp_path / "selected-freeze.json"
    rows = [{"condition_id": "condition", "video_id": "video", "answer": "yes"}]

    digest = freeze_selected_predictions(selected, freeze, rows)

    assert json.loads(freeze.read_text()) == {
        "selected_prediction_sha256": digest,
        "selected_rows": 1,
    }
    assert digest == __import__("hashlib").sha256(selected.read_bytes()).hexdigest()


def test_decision_bindings_revalidate_every_current_field() -> None:
    expected = {
        "manifest_version": "manifest-v1",
        "gate_selection_sha256": "a" * 64,
        "condition_id": "condition",
        "request_set_sha256": "b" * 64,
        "evidence_manifest_sha256": "c" * 64,
        "model_id": "model",
        "model_revision": "d" * 40,
        "runtime_image": "image@sha256:" + "e" * 64,
    }

    validate_decision_bindings(expected, expected)
    for key in expected:
        with pytest.raises(ValueError, match=key):
            validate_decision_bindings({**expected, key: "stale"}, expected)


def test_prepare_does_not_rewrite_authorized_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decisions = tmp_path / "promotion-decisions.jsonl"
    decisions.write_text('{"condition_id":"first"}\n', encoding="utf-8")
    monkeypatch.setattr(frozen_workflow, "DECISIONS", decisions)
    monkeypatch.setattr(
        frozen_workflow, "_validate_all_current_decisions", lambda: [{}]
    )

    def fail_write(*_args: object, **_kwargs: object) -> None:
        pytest.fail("prepare rewrote authorized state")

    monkeypatch.setattr(frozen_workflow, "_atomic_write", fail_write)

    frozen_workflow.prepare()


def test_whisper_timeout_uses_named_cleanup_and_appends_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    audio = run / "audio.wav"
    audio.write_bytes(b"audio")
    requests = [
        {
            "run_id": frozen_workflow.RUN_ID,
            "condition_id": frozen_workflow.ASR_CONDITION,
            "video_id": video_id,
            "audio_path": str(audio),
        }
        for video_id in FROZEN_IDS
    ]
    request_path = run / "whisper-requests.json"
    request_path.write_text(json.dumps(requests), encoding="utf-8")
    image = "local/image@sha256:" + "a" * 64
    (run / "preflight.json").write_text(
        json.dumps(
            {
                "whisper": {
                    "runtime_image": image,
                    "request_set_sha256": canonical_request_sha256(
                        {"requests": requests}
                    ),
                    "requests_file_sha256": __import__("hashlib")
                    .sha256(request_path.read_bytes())
                    .hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    attempts = run / "whisper-attempts.jsonl"
    invocations: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        invocations.append(command)
        if command[:2] == ["podman", "run"]:
            raise subprocess.TimeoutExpired(command, 900)
        if command[:3] in (
            ["podman", "stop", "--time"],
            ["podman", "rm", "--force"],
        ):
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] == ["podman", "container", "exists"]:
            return subprocess.CompletedProcess(command, 1, "", "")
        raise AssertionError(command)

    monkeypatch.setattr(frozen_workflow, "RUN", run)
    monkeypatch.setattr(frozen_workflow, "ASR_ATTEMPTS", attempts)
    monkeypatch.setattr(frozen_workflow, "_expected_ids", lambda: FROZEN_IDS)
    monkeypatch.setattr(
        frozen_workflow, "audit_no_competing_benchmark_containers", lambda: None
    )
    monkeypatch.setattr(
        frozen_workflow,
        "_whisper_command",
        lambda: ["podman", "run", "--rm", image, "runner"],
    )
    monkeypatch.setattr(frozen_workflow.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="failed validation"):
        frozen_workflow.run_whisper()

    row = json.loads(attempts.read_text().splitlines()[0])
    assert "--name" in invocations[0]
    assert row["status"] == "timed_out"
    assert row["runtime_image"] == image
    assert row["container_name"]
    assert row["timeout_cleanup"]["verified"] is True
    assert row["timeout_cleanup"]["container_absent"] is True


def test_frozen_workflow_uses_tracked_label_free_manifest() -> None:
    assert frozen_workflow.MANIFEST_PATH.name.endswith("_inference.json")
    payload = json.loads(frozen_workflow.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert all(
        "annotation" not in row
        and "reference" not in json.dumps(row).lower()
        and "correct_answer" not in row
        for row in payload["examples"]
    )


def test_explicit_v2_run_identity_uses_separate_directory_and_request_id() -> None:
    frozen_workflow.configure_run("native-audio-benchmark-v2")

    assert frozen_workflow.RUN.name == "native-audio-benchmark-v2"
    assert frozen_workflow.RUN_ID == "native-audio-benchmark-v2"
    assert frozen_workflow.ATTEMPTS.parent == frozen_workflow.RUN

    frozen_workflow.configure_run("native-audio-benchmark-v2")


def test_v1_and_v2_share_one_host_global_benchmark_lock() -> None:
    frozen_workflow.configure_run("native-audio-benchmark-v1")
    v1_lock = frozen_workflow.benchmark_lock_path()
    frozen_workflow.configure_run("native-audio-benchmark-v2")
    v2_lock = frozen_workflow.benchmark_lock_path()

    assert v1_lock == v2_lock
    assert v1_lock.parent == frozen_workflow.DATA.resolve() / "vqa/benchmarks"
    assert not v1_lock.is_relative_to(frozen_workflow.RUN)


def test_cli_defaults_to_v2_and_v1_is_status_only() -> None:
    parser = build_parser()

    assert parser.parse_args(["status"]).run_id == "native-audio-benchmark-v2"
    with pytest.raises(ValueError, match="read-only"):
        frozen_workflow.validate_cli_run_access("native-audio-benchmark-v1", "prepare")
    frozen_workflow.validate_cli_run_access("native-audio-benchmark-v1", "status")


@pytest.mark.parametrize(
    ("mutation", "args"),
    [
        ("prepare", ()),
        ("authorize", ("condition",)),
        ("run_condition", ("condition",)),
        ("run_whisper", ()),
        ("build_transcript_requests", ()),
        ("score", ()),
    ],
)
def test_every_mutating_public_workflow_rejects_imported_v1_calls(
    mutation: str,
    args: tuple[object, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(frozen_workflow, "RUN_ID", "native-audio-benchmark-v1")

    with pytest.raises(ValueError, match="read-only"):
        getattr(frozen_workflow, mutation)(*args)


def test_full_current_evidence_manifest_changes_with_any_evidence_byte(
    tmp_path: Path,
) -> None:
    requests = [make_request(tmp_path, video_id) for video_id in FROZEN_IDS]
    before = frozen_workflow.current_evidence_manifest_sha256(requests)

    Path(requests[-1]["media_path"]).write_bytes(b"changed")

    assert frozen_workflow.current_evidence_manifest_sha256(requests) != before


def test_reusable_pass_binds_full_evidence_manifest(tmp_path: Path) -> None:
    request = make_request(tmp_path, FROZEN_IDS[0])
    launch_sha256 = "d" * 64
    good = passed_attempt(request, launch_sha256)

    assert validate_frozen_attempt(
        request,
        good,
        execution=execution(),
        launch_decision_sha256=launch_sha256,
    )
    assert not validate_frozen_attempt(
        request,
        {**good, "evidence_manifest_sha256": "0" * 64},
        execution=execution(),
        launch_decision_sha256=launch_sha256,
    )


def test_competing_podman_benchmark_audit_fails_closed() -> None:
    completed = subprocess.CompletedProcess(
        ["podman", "ps"],
        0,
        json.dumps(
            [
                {
                    "Names": ["trec-vqa-other"],
                    "Command": "python -m trec_video_2026.vqa_transformers_runner",
                }
            ]
        ),
        "",
    )

    with pytest.raises(RuntimeError, match="competing"):
        frozen_workflow.audit_no_competing_benchmark_containers(
            run_command=lambda *_args, **_kwargs: completed,
            kfd_command=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                ["fuser"], 1, "", ""
            ),
        )
    with pytest.raises(RuntimeError, match="audit"):
        frozen_workflow.audit_no_competing_benchmark_containers(
            run_command=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                ["podman", "ps"], 125, "", "unavailable"
            ),
            kfd_command=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                ["fuser"], 1, "", ""
            ),
        )


def test_competing_audit_rejects_live_kfd_pid_and_generic_gpu_container() -> None:
    empty_podman = subprocess.CompletedProcess(["podman", "ps"], 0, "[]", "")
    with pytest.raises(RuntimeError, match="KFD"):
        frozen_workflow.audit_no_competing_benchmark_containers(
            run_command=lambda *_args, **_kwargs: empty_podman,
            kfd_command=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                ["fuser"], 0, "1234 5678", ""
            ),
        )

    containers = subprocess.CompletedProcess(
        ["podman", "ps"],
        0,
        json.dumps([{"Id": "abc", "Names": ["generic-worker"], "Command": "python"}]),
        "",
    )

    def podman(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        if command[:2] == ["podman", "ps"]:
            return containers
        assert command == ["podman", "inspect", "abc"]
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                [
                    {
                        "HostConfig": {
                            "Devices": [
                                {
                                    "PathOnHost": "/dev/kfd",
                                    "PathInContainer": "/dev/kfd",
                                }
                            ]
                        }
                    }
                ]
            ),
            "",
        )

    with pytest.raises(RuntimeError, match="GPU container"):
        frozen_workflow.audit_no_competing_benchmark_containers(
            run_command=podman,
            kfd_command=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                ["fuser"], 1, "", ""
            ),
        )


def test_condition_wall_time_requires_measured_unique_outer_chunks() -> None:
    attempts = [
        {
            "orchestration_chunk_id": "chunk-a",
            "outer_chunk_elapsed_seconds": 10.0,
        },
        {
            "orchestration_chunk_id": "chunk-a",
            "outer_chunk_elapsed_seconds": 10.0,
        },
        {
            "orchestration_chunk_id": "chunk-b",
            "outer_chunk_elapsed_seconds": 20.0,
        },
    ]

    assert frozen_workflow.measured_condition_wall_seconds(attempts) == 30.0
    with pytest.raises(ValueError, match="unmeasurable"):
        frozen_workflow.measured_condition_wall_seconds([{"elapsed_seconds": 1.0}])


def test_whisper_nonzero_exit_appends_structured_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    audio = run / "audio.wav"
    audio.write_bytes(b"audio")
    requests = [
        {
            "run_id": frozen_workflow.RUN_ID,
            "condition_id": frozen_workflow.ASR_CONDITION,
            "video_id": video_id,
            "audio_path": str(audio),
        }
        for video_id in FROZEN_IDS
    ]
    request_path = run / "whisper-requests.json"
    request_path.write_text(json.dumps(requests), encoding="utf-8")
    image = "local/image@sha256:" + "a" * 64
    (run / "preflight.json").write_text(
        json.dumps(
            {
                "whisper": {
                    "runtime_image": image,
                    "request_set_sha256": canonical_request_sha256(
                        {"requests": requests}
                    ),
                    "requests_file_sha256": __import__("hashlib")
                    .sha256(request_path.read_bytes())
                    .hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    attempts = run / "whisper-attempts.jsonl"

    monkeypatch.setattr(frozen_workflow, "RUN", run)
    monkeypatch.setattr(frozen_workflow, "ASR_ATTEMPTS", attempts)
    monkeypatch.setattr(frozen_workflow, "_expected_ids", lambda: FROZEN_IDS)
    monkeypatch.setattr(
        frozen_workflow, "audit_no_competing_benchmark_containers", lambda: None
    )
    monkeypatch.setattr(
        frozen_workflow,
        "_whisper_command",
        lambda: ["podman", "run", "--rm", image, "runner"],
    )
    monkeypatch.setattr(
        frozen_workflow.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["podman", "run"], 7, "", "runner crashed"
        ),
    )

    with pytest.raises(RuntimeError, match="failed validation"):
        frozen_workflow.run_whisper()

    row = json.loads(attempts.read_text().splitlines()[0])
    assert row["status"] == "failed"
    assert row["error_type"] == "RunnerProcessError"
    assert "runner crashed" in row["error_message"]


def test_whisper_command_mounts_only_source_current_audio_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    data = tmp_path / "data"
    run = data / "vqa/benchmarks/runs/native-audio-benchmark-v2"
    cache = tmp_path / "cache"
    for path in (root / "src", run / "audio", cache):
        path.mkdir(parents=True)
    image = "local/image@sha256:" + "a" * 64
    (run / "preflight.json").write_text(
        json.dumps({"whisper": {"runtime_image": image}}), encoding="utf-8"
    )
    monkeypatch.setattr(frozen_workflow, "ROOT", root)
    monkeypatch.setattr(frozen_workflow, "DATA", data)
    monkeypatch.setattr(frozen_workflow, "RUN", run)
    monkeypatch.setattr(frozen_workflow, "HF_CACHE", cache)

    command = frozen_workflow._whisper_command()

    source = (root / "src").resolve()
    audio = (run / "audio").resolve()
    joined = " ".join(command)
    assert f"{source}:{source}:ro" in command
    assert f"{audio}:{audio}:ro" in command
    assert f"{cache.resolve()}:/root/.cache/huggingface:rw" in command
    assert f"{root.resolve()}:{root.resolve()}:ro" not in command
    assert f"{data.resolve()}:{data.resolve()}:ro" not in command
    assert str((data / "vqa/benchmarks").resolve()) not in command
    assert "native-audio-benchmark-v1" not in joined
    assert "annotations" not in joined
    assert "reports" not in joined
    assert command[command.index("-w") + 1] == str(source)
    assert f"PYTHONPATH={source}" in command


def test_scorer_runtime_validation_checks_live_image_version_and_resource_bytes(
    tmp_path: Path,
) -> None:
    punkt = tmp_path / "punkt_tab/english"
    punkt.mkdir(parents=True)
    (punkt / "a.tab").write_bytes(b"one")
    (punkt / "nested").mkdir()
    (punkt / "nested/b.txt").write_bytes(b"two")
    wordnet = tmp_path / "corpora/wordnet.zip"
    wordnet.parent.mkdir()
    wordnet.write_bytes(b"wordnet")

    class Data:
        @staticmethod
        def find(name: str) -> Path:
            return {
                "tokenizers/punkt_tab/english": punkt,
                "corpora/wordnet.zip": wordnet,
            }[name]

    nltk_module = type("Nltk", (), {"__version__": "3.9.2", "data": Data()})()
    image = frozen_workflow.SCORER_IMAGE
    expected_punkt = frozen_workflow.canonical_resource_tree_sha256(punkt)
    expected_wordnet = frozen_workflow.canonical_resource_tree_sha256(wordnet)

    observed = frozen_workflow.validate_scorer_runtime(
        nltk_module,
        {"VQA_RUNTIME_IMAGE": image},
        expected_punkt_sha256=expected_punkt,
        expected_wordnet_sha256=expected_wordnet,
    )

    assert observed == {
        "runtime_image": image,
        "nltk_version": "3.9.2",
        "punkt_tab_english_tree_sha256": expected_punkt,
        "wordnet_zip_sha256": expected_wordnet,
    }
    with pytest.raises(RuntimeError, match="image"):
        frozen_workflow.validate_scorer_runtime(
            nltk_module,
            {"VQA_RUNTIME_IMAGE": "wrong"},
            expected_punkt_sha256=expected_punkt,
            expected_wordnet_sha256=expected_wordnet,
        )
    (punkt / "a.tab").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="punkt"):
        frozen_workflow.validate_scorer_runtime(
            nltk_module,
            {"VQA_RUNTIME_IMAGE": image},
            expected_punkt_sha256=expected_punkt,
            expected_wordnet_sha256=expected_wordnet,
        )
