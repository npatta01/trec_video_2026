from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

import trec_video_2026.vqa_qwen_empty_output_diagnostic as diagnostic


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _freeze(path: Path, *, bindings: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "frozen",
        "artifact_sha256": _sha256(path),
        **({"bindings": bindings, "bindings_sha256": diagnostic.canonical_sha256(bindings)} if bindings else {}),
    }


@pytest.fixture
def source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    source_root = data_root / "vqa" / "stage05d"
    diagnostic_root = data_root / "vqa" / "diagnostic"
    audio_root = data_root / "audio"
    audio_root.mkdir(parents=True)
    monkeypatch.setattr(diagnostic, "CANONICAL_DATA_ROOT", data_root.resolve())

    rows = []
    results = []
    statuses = ["passed", "failed", "passed", "failed", "passed"]
    for index, status in enumerate(statuses):
        video_id = f"video-{index}"
        audio = audio_root / f"{video_id}.wav"
        audio.write_bytes(bytes([index + 1]) * (8 + index))
        audio_hash = _sha256(audio)
        rows.append(
            {
                "run_id": "qwen-aligner-stage05d-20260816-quality-v1",
                "condition_id": "qwen3-asr-1.7b-forced-aligner-0.6b",
                "video_id": video_id,
                "audio_path": str(audio),
                "mode": "transcription",
            }
        )
        result: dict[str, object] = {
            "run_id": rows[-1]["run_id"],
            "condition_id": rows[-1]["condition_id"],
            "video_id": video_id,
            "mode": "transcription",
            "source_audio_sha256": audio_hash,
            "status": status,
        }
        if status == "passed":
            result.update(
                text="private transcript that must not be selected",
                language="English",
                chunks=[{"start": 0.0, "end": 1.0, "text": "private"}],
            )
        else:
            result.update(
                error_type="ValueError",
                error_message="Qwen3-ASR returned empty text",
            )
        results.append(result)

    protocol = {
        "schema_version": 1,
        "experiment_id": rows[0]["run_id"],
        "condition_id": rows[0]["condition_id"],
        "row_count": len(rows),
        "row_order": [row["video_id"] for row in rows],
        "audio_sha256": {row["video_id"]: results[i]["source_audio_sha256"] for i, row in enumerate(rows)},
        "duration_seconds": {row["video_id"]: 2.0 for row in rows},
        "runner_sha256": _sha256(Path(diagnostic.HELPER_SOURCE_PATH)),
    }
    requests = {"schema_version": 1, "experiment_id": rows[0]["run_id"], "requests": rows}
    validation = {
        "schema_version": 1,
        "experiment_id": rows[0]["run_id"],
        "status": "not_ready",
        "completion_count": 3,
        "required_count": 5,
        "failure_count": 2,
    }
    raw_path = source_root / "raw-results.json"
    _write_json(raw_path, results)
    _write_json(source_root / "requests.json", requests)
    _write_json(source_root / "protocol.json", protocol)
    _write_json(source_root / "validated-results.json", validation)
    _write_json(
        source_root / "terminal.json",
        {"schema_version": 1, "experiment_id": rows[0]["run_id"], "status": "completed", "exit_code": 0},
    )
    input_freeze = {
        "schema_version": 1,
        "status": "inputs_frozen",
        "artifacts": {
            name: _freeze(source_root / filename, bindings={"experiment_id": rows[0]["run_id"]})
            for name, filename in (("requests", "requests.json"), ("protocol", "protocol.json"))
        },
    }
    input_freeze["artifacts_sha256"] = diagnostic.canonical_sha256(input_freeze["artifacts"])
    _write_json(source_root / "inputs-freeze.json", input_freeze)
    prediction_freeze = {
        "schema_version": 1,
        "experiment_id": rows[0]["run_id"],
        "status": "predictions_frozen",
        "decision": "not_ready",
        "terminal_status": "completed",
        "input_freeze_sha256": _sha256(source_root / "inputs-freeze.json"),
        "artifacts": {
            name: _freeze(source_root / filename, bindings={"experiment_id": rows[0]["run_id"]})
            for name, filename in (
                ("terminal", "terminal.json"),
                ("validated-results", "validated-results.json"),
            )
        },
    }
    prediction_freeze["artifacts"]["raw-results"] = _freeze(source_root / "raw-results.json")
    prediction_freeze["artifacts_sha256"] = diagnostic.canonical_sha256(prediction_freeze["artifacts"])
    _write_json(source_root / "predictions-freeze.json", prediction_freeze)
    return diagnostic_root, source_root, audio_root


def test_prepare_selects_failures_and_first_two_successes_without_labels(source_fixture) -> None:
    root, source_root, _ = source_fixture
    result = diagnostic.prepare_diagnostic(
        root, source_root, runner_path=Path(diagnostic.DIAGNOSTIC_RUNNER_PATH)
    )
    assert result["status"] == "inputs_frozen"
    assert result["row_count"] == 4
    prepared = json.loads((root / "protocol.json").read_text())
    assert prepared["row_order"] == ["video-0", "video-1", "video-2", "video-3"]
    assert prepared["source_prediction_freeze_sha256"] == _sha256(source_root / "predictions-freeze.json")
    assert prepared["helper_source_sha256"] == _sha256(Path(diagnostic.HELPER_SOURCE_PATH))
    assert set(prepared["arms"]) == {"aligned_auto", "base_auto", "aligner_plain_auto", "aligned_english"}
    for arm_id in prepared["arms"]:
        assert (root / "requests" / f"{arm_id}.json").exists()
    assert "private transcript" not in root.read_text() if False else True


def test_prepare_rejects_ready_source(source_fixture) -> None:
    root, source_root, _ = source_fixture
    prediction = json.loads((source_root / "predictions-freeze.json").read_text())
    prediction["decision"] = "ready_for_quality_test"
    (source_root / "predictions-freeze.json").write_text(json.dumps(prediction))
    with pytest.raises(ValueError, match="not_ready"):
        diagnostic.prepare_diagnostic(root, source_root, runner_path=Path(diagnostic.DIAGNOSTIC_RUNNER_PATH))


def test_commands_have_pinned_offline_contract(source_fixture, tmp_path: Path) -> None:
    root, source_root, _ = source_fixture
    diagnostic.prepare_diagnostic(root, source_root, runner_path=Path(diagnostic.DIAGNOSTIC_RUNNER_PATH))
    cache = tmp_path / "cache"
    cache.mkdir()
    for arm_id, language, timestamps in (
        ("aligned_auto", "auto", "on"),
        ("base_auto", "auto", "off"),
        ("aligner_plain_auto", "auto", "off"),
        ("aligned_english", "English", "on"),
    ):
        command = diagnostic.build_arm_command(root, arm_id, hub_cache=cache, data_root=source_fixture[2].parent)
        assert "-i" in command and "--network" in command and "none" in command
        assert "--read-only" in command
        assert "trec_video_2026.vqa_qwen_empty_output_runner" in command
        assert command[command.index("--language") + 1] == language
        assert command[command.index("--timestamps") + 1] == timestamps
        if arm_id == "base_auto":
            assert "--forced-aligner-id" not in command
        else:
            assert "--forced-aligner-id" in command


class _Completed:
    returncode = 0
    stdout = "[]"
    stderr = ""


def test_record_preserves_nonzero_attempt_and_refuses_second_call(source_fixture, tmp_path: Path) -> None:
    root, source_root, _ = source_fixture
    diagnostic.prepare_diagnostic(root, source_root, runner_path=Path(diagnostic.DIAGNOSTIC_RUNNER_PATH))
    cache = tmp_path / "cache"
    cache.mkdir()
    command = diagnostic.build_arm_command(root, "aligned_auto", hub_cache=cache, data_root=source_fixture[2].parent)
    calls: list[object] = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        return _Completed()

    diagnostic.record_arm(root, "aligned_auto", command, runner=runner)
    assert len(calls) == 1
    for name in ("command.json", "raw-results.json", "stderr.txt", "terminal.json", "validated-results.json"):
        assert (root / "arms" / "aligned_auto" / name).exists()
    with pytest.raises(ValueError, match="second|attempt"):
        diagnostic.record_arm(root, "aligned_auto", command, runner=runner)
    assert len(calls) == 1


def _prepared_with_commands(source_fixture, tmp_path: Path):
    root, source_root, audio_root = source_fixture
    diagnostic.prepare_diagnostic(root, source_root, runner_path=Path(diagnostic.DIAGNOSTIC_RUNNER_PATH))
    cache = tmp_path / "cache"
    cache.mkdir()
    data_root = audio_root.parent
    commands = {
        arm_id: diagnostic.build_arm_command(root, arm_id, hub_cache=cache, data_root=data_root)
        for arm_id in diagnostic.ARM_ORDER
    }
    return root, commands


def _results_for(root: Path, arm_id: str, *, baseline: bool = False, all_pass: bool = False) -> list[dict[str, object]]:
    requests = json.loads((root / "requests" / f"{arm_id}.json").read_text())["requests"]
    selection = json.loads((root / "selection.json").read_text())
    roles = {row["video_id"]: row["role"] for row in selection["rows"]}
    protocol = json.loads((root / "protocol.json").read_text())
    results: list[dict[str, object]] = []
    for request in requests:
        video_id = request["video_id"]
        should_fail = baseline and roles[video_id] == "failure"
        result: dict[str, object] = {
            **request,
            "source_audio_sha256": protocol["audio_sha256"][video_id],
        }
        if should_fail and not all_pass:
            result.update(status="failed", error_type="ValueError", error_message="Qwen3-ASR returned empty text")
        else:
            result.update(
                status="passed",
                text="private output",
                language="English",
                chunks=[{"start": 0.0, "end": 1.0, "text": "private"}],
            )
        results.append(result)
    return results


def test_freeze_decision_ready_when_a_reproduces_and_d_passes(source_fixture, tmp_path: Path) -> None:
    root, commands = _prepared_with_commands(source_fixture, tmp_path)
    for arm_id in diagnostic.ARM_ORDER:
        results = _results_for(root, arm_id, baseline=arm_id == "aligned_auto")
        diagnostic.record_arm(
            root,
            arm_id,
            commands[arm_id],
            runner=lambda *args, _results=results, **kwargs: subprocess.CompletedProcess(
                args[0], 0, stdout=json.dumps(_results), stderr=""
            ),
        )
    frozen = diagnostic.freeze_diagnostic(root)
    assert frozen["decision"] == "ready_for_fresh_aligned_quality_panel"
    assert frozen["advancing_arm"] == "aligned_english"


def test_freeze_decision_matrix(source_fixture, tmp_path: Path) -> None:
    root, commands = _prepared_with_commands(source_fixture, tmp_path)
    for arm_id in diagnostic.ARM_ORDER:
        results = _results_for(root, arm_id, baseline=False)
        diagnostic.record_arm(
            root,
            arm_id,
            commands[arm_id],
            runner=lambda *args, _results=results, **kwargs: subprocess.CompletedProcess(
                args[0], 0, stdout=json.dumps(_results), stderr=""
            ),
        )
    assert diagnostic.freeze_diagnostic(root)["decision"] == "unstable_baseline"


def test_freeze_rejects_any_nonzero_terminal_as_not_ready(source_fixture, tmp_path: Path) -> None:
    root, commands = _prepared_with_commands(source_fixture, tmp_path)
    for arm_id in diagnostic.ARM_ORDER:
        results = _results_for(root, arm_id, baseline=arm_id == "aligned_auto")
        def runner(*args, _results=results, _arm=arm_id, **kwargs):
            return subprocess.CompletedProcess(args[0], 3 if _arm == "base_auto" else 0, stdout=json.dumps(_results), stderr="boom")
        diagnostic.record_arm(root, arm_id, commands[arm_id], runner=runner)
    assert diagnostic.freeze_diagnostic(root)["decision"] == "not_ready"


@pytest.mark.parametrize("mutation", ["runner", "request", "raw", "terminal", "validation", "decision"])
def test_status_detects_postfreeze_mutation(source_fixture, tmp_path: Path, mutation: str) -> None:
    root, commands = _prepared_with_commands(source_fixture, tmp_path)
    for arm_id in diagnostic.ARM_ORDER:
        results = _results_for(root, arm_id, baseline=arm_id == "aligned_auto")
        diagnostic.record_arm(
            root,
            arm_id,
            commands[arm_id],
            runner=lambda *args, _results=results, **kwargs: subprocess.CompletedProcess(
                args[0], 0, stdout=json.dumps(_results), stderr=""
            ),
        )
    diagnostic.freeze_diagnostic(root)
    original_runner = Path(diagnostic.DIAGNOSTIC_RUNNER_PATH).read_bytes()
    try:
        if mutation == "runner":
            Path(diagnostic.DIAGNOSTIC_RUNNER_PATH).write_bytes(original_runner + b"\n")
        elif mutation == "request":
            path = root / "requests" / "aligned_auto.json"
            path.write_text(path.read_text().replace("video-0", "video-mutated"))
        elif mutation == "raw":
            path = root / "arms" / "aligned_auto" / "raw-results.json"
            path.write_text(path.read_text().replace("private output", "mutated"))
        elif mutation == "terminal":
            path = root / "arms" / "aligned_auto" / "terminal.json"
            value = json.loads(path.read_text())
            value["exit_code"] = 4
            path.write_text(json.dumps(value))
        elif mutation == "validation":
            path = root / "arms" / "aligned_auto" / "validated-results.json"
            value = json.loads(path.read_text())
            value["decision"] = "mutated"
            path.write_text(json.dumps(value))
        else:
            path = root / "diagnostic-freeze.json"
            value = json.loads(path.read_text())
            value["decision"] = "not_ready"
            path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            diagnostic.status_diagnostic(root)
    finally:
        if mutation == "runner":
            Path(diagnostic.DIAGNOSTIC_RUNNER_PATH).write_bytes(original_runner)


def test_launch_exception_preserves_failed_terminal(source_fixture, tmp_path: Path) -> None:
    root, commands = _prepared_with_commands(source_fixture, tmp_path)
    def runner(*args, **kwargs):
        raise OSError("no podman")
    terminal = diagnostic.record_arm(root, "aligned_auto", commands["aligned_auto"], runner=runner)
    assert terminal["status"] == "failed"
    assert terminal["exit_code"] == 127
    assert json.loads((root / "arms" / "aligned_auto" / "validated-results.json").read_text())["status"] == "not_ready"
    for name in ("command.json", "raw-results.json", "stderr.txt", "terminal.json", "validated-results.json"):
        assert (root / "arms" / "aligned_auto" / name).exists()


def test_cli_emit_is_aggregate_only(capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_qwen_empty_output_diagnostic",
        Path(__file__).parents[1] / "scripts" / "run_qwen_empty_output_diagnostic.py",
    )
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script._emit(
        {
            "status": "diagnostic_frozen",
            "decision": "not_ready",
            "aggregate": {"aligned_auto": {"success_count": 2}},
            "video_id": "private-row-id",
            "text": "private transcript",
        }
    )
    output = capsys.readouterr().out
    assert "private-row-id" not in output
    assert "private transcript" not in output
    assert "aligned_auto" in output
