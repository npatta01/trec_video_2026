from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import trec_video_2026.vqa_qwen_transcript_answer as answer_module
from trec_video_2026.vqa_qwen_transcript_answer import (
    ANSWER_MODEL_ID,
    ANSWER_MODEL_REVISION,
    QWEN_CONDITION,
    RUNTIME_IMAGE,
    WHISPER_CONDITION,
    build_answer_command,
    build_answer_requests,
    freeze_answer_predictions,
    prepare_answer_test,
    record_answer_attempt,
    verify_answer_inputs,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _inputs() -> tuple[dict, dict, list[dict], dict]:
    selection_rows = [
        {
            "video_id": f"video-{index:02d}",
            "audio_sha256": _sha(f"audio-{index}"),
            "source_order_index": index,
        }
        for index in range(12)
    ]
    selection = {
        "selection_kind": "qwen_aligned_english_fresh_speech_panel",
        "panel_size": 12,
        "rows": selection_rows,
    }
    selected_examples = [
        {
            "video_id": row["video_id"],
            "question": f"Why did event {index} happen?",
        }
        for index, row in enumerate(selection_rows)
    ]
    filler_examples = [
        {
            "video_id": f"filler-{index:03d}",
            "question": "What is visible?",
        }
        for index in range(475 - len(selected_examples))
    ]
    manifest = {"examples": [*selected_examples, *filler_examples]}
    qwen_results = [
        {
            "video_id": row["video_id"],
            "status": "passed",
            "source_audio_sha256": row["audio_sha256"],
            "text": f"Qwen transcript {index}",
            "answer": None,
        }
        for index, row in enumerate(selection_rows)
    ]
    whisper = {
        "results": [
            {
                "video_id": row["video_id"],
                "status": "passed",
                "source_audio_sha256": row["audio_sha256"],
                "text": f"Whisper transcript {index}",
            }
            for index, row in enumerate(selection_rows)
        ]
    }
    return selection, manifest, qwen_results, whisper


def test_build_answer_requests_creates_two_label_free_matched_arms() -> None:
    selection, manifest, qwen_results, whisper = _inputs()

    requests = build_answer_requests(
        selection=selection,
        manifest=manifest,
        qwen_results=qwen_results,
        whisper_payload=whisper,
    )

    assert len(requests) == 24
    assert {row["condition_id"] for row in requests} == {
        QWEN_CONDITION,
        WHISPER_CONDITION,
    }
    assert all(row["frame_paths"] == [] for row in requests)
    assert all(row["frame_timestamps"] == [] for row in requests)
    assert [row["condition_id"] for row in requests[:4]] == [
        QWEN_CONDITION,
        WHISPER_CONDITION,
        QWEN_CONDITION,
        WHISPER_CONDITION,
    ]
    assert [row["video_id"] for row in requests[:4]] == [
        "video-00",
        "video-00",
        "video-01",
        "video-01",
    ]
    rendered = json.dumps(requests, sort_keys=True).casefold()
    for forbidden in (
        '"answer"',
        '"correct_answer"',
        '"incorrect_answers"',
        '"reference"',
        '"references"',
    ):
        assert forbidden not in rendered
    for request in requests:
        live = {
            key: value
            for key, value in request.items()
            if key not in {"canonical_request", "request_sha256"}
        }
        canonical = json.dumps(
            live,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assert request["canonical_request"] == canonical
        assert request["request_sha256"] == hashlib.sha256(
            canonical.encode()
        ).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    (
        "short-selection",
        "duplicate-selection",
        "short-manifest",
        "missing-qwen",
        "empty-qwen",
        "qwen-audio-drift",
        "missing-whisper",
        "empty-whisper",
        "whisper-audio-drift",
    ),
)
def test_build_answer_requests_rejects_unmatched_or_invalid_inputs(
    mutation: str,
) -> None:
    selection, manifest, qwen_results, whisper = _inputs()
    if mutation == "short-selection":
        selection["rows"].pop()
    elif mutation == "duplicate-selection":
        selection["rows"][1]["video_id"] = selection["rows"][0]["video_id"]
    elif mutation == "short-manifest":
        manifest["examples"].pop()
    elif mutation == "missing-qwen":
        qwen_results.pop()
    elif mutation == "empty-qwen":
        qwen_results[0]["text"] = " "
    elif mutation == "qwen-audio-drift":
        qwen_results[0]["source_audio_sha256"] = "0" * 64
    elif mutation == "missing-whisper":
        whisper["results"].pop()
    elif mutation == "empty-whisper":
        whisper["results"][0]["text"] = ""
    elif mutation == "whisper-audio-drift":
        whisper["results"][0]["source_audio_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        build_answer_requests(
            selection=selection,
            manifest=manifest,
            qwen_results=qwen_results,
            whisper_payload=whisper,
        )


def test_build_answer_requests_uses_raw_whisper_text_even_when_gate_rejected() -> None:
    selection, manifest, qwen_results, whisper = _inputs()
    whisper["results"][0]["original_quality"] = {
        "status": "rejected-timestamp-overrun"
    }

    requests = build_answer_requests(
        selection=selection,
        manifest=manifest,
        qwen_results=qwen_results,
        whisper_payload=whisper,
    )

    whisper_request = next(
        row
        for row in requests
        if row["video_id"] == "video-00"
        and row["condition_id"] == WHISPER_CONDITION
    )
    assert whisper_request["transcript"] == "Whisper transcript 0"


def _write_preparation_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    selection, manifest, qwen_results, whisper = _inputs()
    source_root = tmp_path / "sources"
    stage_root = source_root / "stage05f"
    canonical_root = source_root / "canonical"
    stage_root.mkdir(parents=True)
    canonical_root.mkdir(parents=True)
    paths = {
        "selection": stage_root / "selection.json",
        "qwen_results": stage_root / "raw-results.json",
        "prediction_freeze": stage_root / "predictions-freeze.json",
        "manifest": canonical_root / "manifest.json",
        "whisper": canonical_root / "whisper.json",
        "whisper_checksums": canonical_root / "SHA256SUMS",
        "runner": source_root / "vqa_transformers_runner.py",
    }
    paths["selection"].write_text(json.dumps(selection), encoding="utf-8")
    paths["qwen_results"].write_text(json.dumps(qwen_results), encoding="utf-8")
    paths["prediction_freeze"].write_text(
        json.dumps({"decision": "ready_for_quality_test"}),
        encoding="utf-8",
    )
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    paths["whisper"].write_text(json.dumps(whisper), encoding="utf-8")
    paths["whisper_checksums"].write_text(
        f"{hashlib.sha256(paths['whisper'].read_bytes()).hexdigest()}  whisper.json\n",
        encoding="utf-8",
    )
    paths["runner"].write_text("# frozen runner\n", encoding="utf-8")
    monkeypatch.setattr(answer_module, "STAGE05F_ROOT", stage_root)
    monkeypatch.setattr(answer_module, "STAGE05F_SELECTION_PATH", paths["selection"])
    monkeypatch.setattr(
        answer_module,
        "STAGE05F_QWEN_RESULTS_PATH",
        paths["qwen_results"],
    )
    monkeypatch.setattr(
        answer_module,
        "STAGE05F_PREDICTION_FREEZE_PATH",
        paths["prediction_freeze"],
    )
    monkeypatch.setattr(answer_module, "MANIFEST_PATH", paths["manifest"])
    monkeypatch.setattr(answer_module, "WHISPER_PATH", paths["whisper"])
    monkeypatch.setattr(
        answer_module,
        "WHISPER_CHECKSUM_PATH",
        paths["whisper_checksums"],
    )
    monkeypatch.setattr(answer_module, "RUNNER_PATH", paths["runner"])
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir()
    monkeypatch.setattr(answer_module, "HUB_CACHE", hub_cache)
    monkeypatch.setattr(answer_module, "REPOSITORY_ROOT", tmp_path.resolve())
    monkeypatch.setattr(answer_module, "CANONICAL_DATA_ROOT", source_root.resolve())
    monkeypatch.setattr(
        answer_module,
        "verify_stage05f_root",
        lambda: {
            "status": "scored_postfreeze",
            "decision": "ready_for_quality_test",
        },
    )
    return paths


def test_prepare_answer_test_freezes_exact_label_closed_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_preparation_sources(tmp_path, monkeypatch)
    root = tmp_path / "private" / "answer-stage"

    freeze = prepare_answer_test(root, prepared_at="2026-08-16T12:00:00Z")

    assert sorted(path.name for path in root.iterdir()) == [
        "inputs-freeze.json",
        "protocol.json",
        "requests.json",
    ]
    protocol = json.loads((root / "protocol.json").read_text())
    requests = json.loads((root / "requests.json").read_text())
    assert protocol["status"] == "inputs_frozen"
    assert protocol["answer_model_id"] == ANSWER_MODEL_ID
    assert protocol["answer_model_revision"] == ANSWER_MODEL_REVISION
    assert protocol["runtime_image"] == RUNTIME_IMAGE
    assert protocol["request_count"] == 24
    assert protocol["label_state"] == "closed"
    assert len(requests["requests"]) == 24
    assert freeze["status"] == "inputs_frozen"
    assert set(freeze["source_files"]) == set(paths)
    assert verify_answer_inputs(root)["status"] == "inputs_frozen"
    with pytest.raises(FileExistsError):
        prepare_answer_test(root, prepared_at="2026-08-16T12:00:01Z")


@pytest.mark.parametrize(
    "source_name",
    (
        "selection",
        "qwen_results",
        "prediction_freeze",
        "manifest",
        "whisper",
        "whisper_checksums",
        "runner",
    ),
)
def test_verify_answer_inputs_rejects_any_frozen_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
) -> None:
    paths = _write_preparation_sources(tmp_path, monkeypatch)
    root = tmp_path / "private" / "answer-stage"
    prepare_answer_test(root, prepared_at="2026-08-16T12:00:00Z")

    paths[source_name].write_bytes(paths[source_name].read_bytes() + b" ")

    with pytest.raises(ValueError, match="hash|source|freeze"):
        verify_answer_inputs(root)


def _prepared_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "answer-stage",
) -> Path:
    _write_preparation_sources(tmp_path, monkeypatch)
    root = tmp_path / "private" / name
    prepare_answer_test(root, prepared_at="2026-08-16T12:00:00Z")
    return root


def _passed_answer_results(root: Path) -> list[dict]:
    requests = json.loads((root / "requests.json").read_text())["requests"]
    return [
        {
            "run_id": request["run_id"],
            "condition_id": request["condition_id"],
            "video_id": request["video_id"],
            "status": "passed",
            "answer": f"Concise answer {index}",
            "model_id": ANSWER_MODEL_ID,
            "model_revision": ANSWER_MODEL_REVISION,
            "runtime_image": RUNTIME_IMAGE,
            "answer_policy": "current",
            "elapsed_seconds": 0.5,
            "peak_memory_bytes": 100 + index,
            "error_type": None,
            "error_message": None,
        }
        for index, request in enumerate(requests)
    ]


def test_build_answer_command_is_offline_pinned_and_text_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, monkeypatch)

    command = build_answer_command(root)

    assert command[:2] == ["podman", "run"]
    assert "-i" in command
    assert "--read-only" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--tty" not in command and "-t" not in command
    assert RUNTIME_IMAGE in command
    assert "trec_video_2026.vqa_transformers_runner" in command
    assert command[command.index("--model-id") + 1] == ANSWER_MODEL_ID
    assert command[command.index("--revision") + 1] == ANSWER_MODEL_REVISION
    assert command[command.index("--max-new-tokens") + 1] == "32"
    assert command[command.index("--answer-policy") + 1] == "current"
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command


def test_record_and_freeze_require_exactly_24_nonempty_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, monkeypatch)
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_passed_answer_results(root)),
            stderr="model loaded",
        )

    terminal = record_answer_attempt(root, runner=fake_run)
    freeze = freeze_answer_predictions(root)

    assert terminal["status"] == "completed"
    assert terminal["exit_code"] == 0
    assert len(calls) == 1
    assert calls[0][1]["text"] is True
    assert calls[0][1]["capture_output"] is True
    assert len(json.loads(calls[0][1]["input"])) == 24
    validation = json.loads((root / "validated-results.json").read_text())
    assert validation["status"] == "ready_for_scoring"
    assert validation["completion_count"] == 24
    assert freeze["decision"] == "ready_for_scoring"
    assert set(freeze["artifacts"]) == {
        "command",
        "raw-results",
        "stderr",
        "terminal",
        "validated-results",
    }
    with pytest.raises(ValueError, match="attempt|already|exists"):
        record_answer_attempt(root, runner=fake_run)
    with pytest.raises(ValueError, match="freeze|exists|overwrite"):
        freeze_answer_predictions(root)


@pytest.mark.parametrize("failure", ("short", "empty", "wrong-model"))
def test_incomplete_or_drifted_answers_freeze_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = _prepared_root(tmp_path, monkeypatch, name=f"failed-{failure}")
    results = _passed_answer_results(root)
    if failure == "short":
        results.pop()
    elif failure == "empty":
        results[0]["answer"] = " "
    else:
        results[0]["model_revision"] = "wrong"

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(results),
            stderr="",
        )

    record_answer_attempt(root, runner=fake_run)

    validation = json.loads((root / "validated-results.json").read_text())
    assert validation["status"] == "not_ready"
    assert freeze_answer_predictions(root)["decision"] == "not_ready"


def test_record_rechecks_frozen_sources_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_preparation_sources(tmp_path, monkeypatch)
    root = tmp_path / "private" / "source-drift"
    prepare_answer_test(root, prepared_at="2026-08-16T12:00:00Z")
    paths["manifest"].write_bytes(paths["manifest"].read_bytes() + b" ")
    called = False

    def must_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("mutated inputs must not execute")

    with pytest.raises(ValueError, match="hash|source|freeze"):
        record_answer_attempt(root, runner=must_not_run)
    assert called is False
    assert not (root / "command.json").exists()


def test_launch_exception_preserves_terminal_not_ready_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, monkeypatch, name="launch-failure")

    def launch_failure(*args, **kwargs):
        raise FileNotFoundError("podman missing")

    terminal = record_answer_attempt(root, runner=launch_failure)

    assert terminal["status"] == "failed"
    assert terminal["exit_code"] == 127
    assert sorted(path.name for path in root.iterdir()) == [
        "command.json",
        "inputs-freeze.json",
        "protocol.json",
        "raw-results.json",
        "requests.json",
        "stderr.txt",
        "terminal.json",
        "validated-results.json",
    ]
    assert freeze_answer_predictions(root)["decision"] == "not_ready"


def test_freeze_rejects_raw_result_mutation_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepared_root(tmp_path, monkeypatch, name="raw-mutation")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_passed_answer_results(root)),
            stderr="",
        )

    record_answer_attempt(root, runner=fake_run)
    raw_path = root / "raw-results.json"
    raw_path.write_text(raw_path.read_text() + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="result|validation|hash"):
        freeze_answer_predictions(root)
    assert not (root / "predictions-freeze.json").exists()
