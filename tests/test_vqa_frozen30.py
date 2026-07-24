from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    validate_launch_decision,
    validate_frozen_attempt,
)
from trec_video_2026.vqa_gate import canonical_request_sha256
from trec_video_2026.vqa_frozen30_workflow import (
    build_parser,
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
        "model_id": execution().model_id,
        "model_revision": execution().model_revision,
        "runtime_image": execution().runtime_image,
        "device": execution().device,
        "precision": execution().precision,
        "elapsed_seconds": 1.0,
        "peak_memory_bytes": 1,
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
