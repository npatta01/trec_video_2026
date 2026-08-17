from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import trec_video_2026.vqa_qwen_aligner_canary as canary_module
from trec_video_2026.vqa_qwen_aligner_canary import (
    EXPERIMENT_ID,
    build_podman_command,
    freeze_predictions,
    prepare_canary,
    record_attempt,
    score_postfreeze,
    select_quality_panel,
    select_panel,
    status_canary,
    validate_results,
)


GEMINI_CONDITION = "gemini-3.1-pro-native-av-4fps"


@pytest.fixture(autouse=True)
def _bind_test_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(canary_module, "CANONICAL_DATA_ROOT", tmp_path.resolve())


def _write_audio(root: Path, video_id: str, payload: bytes) -> Path:
    path = root / f"{video_id}.wav"
    path.write_bytes(payload)
    return path


def test_select_panel_chooses_unique_fresh_target_and_best_fresh_control(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "video_id": "target",
            "question_stratum": "audio_or_text",
            "semantic_composite": 0.1,
        },
        {
            "video_id": "control-low",
            "question_stratum": "audio_or_text",
            "semantic_composite": 0.7,
        },
        {
            "video_id": "control-high",
            "question_stratum": "audio_or_text",
            "semantic_composite": 0.9,
        },
        {
            "video_id": "reused-control",
            "question_stratum": "audio_or_text",
            "semantic_composite": 1.0,
        },
    ]
    scores = {"rows": {GEMINI_CONDITION: rows}}
    maps = [
        {"video_id": "target", "status": "passed", "map": "OVERVIEW | target"},
        {
            "video_id": "control-low",
            "status": "passed",
            "map": "OVERVIEW | control\nSPEECH | words",
        },
        {
            "video_id": "control-high",
            "status": "passed",
            "map": "OVERVIEW | control\nSPEECH | words",
        },
        {
            "video_id": "reused-control",
            "status": "passed",
            "map": "OVERVIEW | control\nSPEECH | words",
        },
    ]
    transcripts = {
        "results": [
            {
                "video_id": "target",
                "status": "passed",
                "source_language": "english",
                "source_audio_duration_seconds": 30.0,
                "transcript_included": False,
                "quality": {"status": "rejected-timestamp-overrun"},
            },
            *[
                {
                    "video_id": video_id,
                    "status": "passed",
                    "source_language": "english",
                    "source_audio_duration_seconds": 15.0,
                    "transcript_included": True,
                    "quality": {"status": "accepted"},
                }
                for video_id in ("control-low", "control-high", "reused-control")
            ],
        ]
    }
    payloads = {
        video_id: f"audio:{video_id}".encode()
        for video_id in ("target", "control-low", "control-high", "reused-control")
    }
    for video_id, payload in payloads.items():
        _write_audio(tmp_path, video_id, payload)

    selected = select_panel(
        scores=scores,
        maps=maps,
        transcripts=transcripts,
        prior_qwen_ids={"reused-control"},
        reviewed_failure_ids=set(),
        audio_root=tmp_path,
        selected_at="2026-08-15T12:00:00Z",
        source_hashes={
            "scores_sha256": "1" * 64,
            "maps_sha256": "2" * 64,
            "transcripts_sha256": "3" * 64,
            "prior_qwen_sha256": "4" * 64,
        },
    )

    assert set(selected) == {
        "schema_version",
        "experiment_id",
        "selected_at",
        "selection_policy",
        "rows",
        "source_hashes",
    }
    assert [(row["role"], row["video_id"]) for row in selected["rows"]] == [
        ("target", "target"),
        ("control", "control-high"),
    ]
    assert selected["rows"][0] == {
        "role": "target",
        "video_id": "target",
        "audio_path": str((tmp_path / "target.wav").resolve()),
        "audio_sha256": hashlib.sha256(payloads["target"]).hexdigest(),
        "duration_seconds": 30.0,
        "selection_provenance": (
            "audio_or_text+missing_speech_record+english+"
            "whisper_timestamp_overrun+fresh_qwen"
        ),
    }
    assert selected["rows"][1]["audio_sha256"] == hashlib.sha256(
        payloads["control-high"]
    ).hexdigest()
    forbidden = ("question", "answer", "transcript", "reference", "metric", "map")
    rendered = repr(selected).lower()
    assert not any(f"'{key}'" in rendered for key in forbidden)


def _literal_selection(tmp_path: Path) -> dict:
    rows = []
    for role, video_id, duration in (
        ("target", "target", 30.0),
        ("control", "control", 15.0),
    ):
        payload = f"audio:{video_id}".encode()
        path = _write_audio(tmp_path, video_id, payload).resolve()
        rows.append(
            {
                "role": role,
                "video_id": video_id,
                "audio_path": str(path),
                "audio_sha256": hashlib.sha256(payload).hexdigest(),
                "duration_seconds": duration,
                "selection_provenance": f"private-{role}-policy",
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "selected_at": "2026-08-15T12:00:00Z",
        "selection_policy": "private deterministic policy",
        "rows": rows,
        "source_hashes": {
            "scores_sha256": "1" * 64,
            "maps_sha256": "2" * 64,
            "transcripts_sha256": "3" * 64,
            "prior_qwen_sha256": "4" * 64,
        },
    }


def test_prepare_canary_exclusively_freezes_label_free_inputs(tmp_path: Path) -> None:
    root = tmp_path / "run"

    prepared = prepare_canary(
        root,
        _literal_selection(tmp_path),
        runner_path=canary_module.RUNNER_PATH,
    )

    assert prepared["status"] == "inputs_frozen"
    assert sorted(path.name for path in root.iterdir()) == [
        "inputs-freeze.json",
        "protocol.json",
        "requests.json",
        "selection.json",
    ]
    requests = json.loads((root / "requests.json").read_text(encoding="utf-8"))
    assert [row["video_id"] for row in requests["requests"]] == [
        "target",
        "control",
    ]
    assert set(requests["requests"][0]) == {
        "run_id",
        "condition_id",
        "video_id",
        "audio_path",
        "mode",
    }
    assert status_canary(root)["status"] == "inputs_frozen"
    with pytest.raises(ValueError, match="exists|overwrite"):
        prepare_canary(
            root,
            _literal_selection(tmp_path),
            runner_path=canary_module.RUNNER_PATH,
        )


def test_status_rejects_mutation_after_input_freeze(tmp_path: Path) -> None:
    root = tmp_path / "run"
    prepare_canary(
        root,
        _literal_selection(tmp_path),
        runner_path=canary_module.RUNNER_PATH,
    )
    requests_path = root / "requests.json"
    requests = json.loads(requests_path.read_text(encoding="utf-8"))
    requests["requests"].reverse()
    requests_path.write_text(json.dumps(requests), encoding="utf-8")

    with pytest.raises(ValueError, match="hash|freeze|artifact"):
        status_canary(root)


def test_podman_command_is_offline_digest_pinned_and_read_only(tmp_path: Path) -> None:
    root = _prepared_root(tmp_path)
    cache = tmp_path / "hub"
    data_root = tmp_path
    cache.mkdir()

    command = build_podman_command(root, hub_cache=cache, data_root=data_root)

    rendered = " ".join(command)
    assert command[:3] == ["podman", "run", "--rm"]
    assert "-i" in command
    assert command.index("-i") < command.index(
        "localhost/trec-vqa-audio-rocm@sha256:"
        "faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234"
    )
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "HF_HOME=/root/.cache/huggingface" in command
    assert "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub" in command
    assert "/dev/kfd" in command
    assert "/dev/dri" in command
    assert "keep-groups" in command
    assert "label=disable" in command
    assert (
        "localhost/trec-vqa-audio-rocm@sha256:"
        "faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234"
        in command
    )
    assert f"{cache.resolve()}:/root/.cache/huggingface/hub:ro" in command
    assert f"{data_root.resolve()}:{data_root.resolve()}:ro" in command
    assert "/opt/venv/bin/python" in command
    assert "trec_video_2026.vqa_audio_runner" in command
    assert "Qwen/Qwen3-ASR-1.7B" in command
    assert "7278e1e70fe206f11671096ffdd38061171dd6e5" in command
    assert "Qwen/Qwen3-ForcedAligner-0.6B" in command
    assert "c7cbfc2048c462b0d63a45797104fc9db3ad62b7" in command
    assert "HF_TOKEN" not in rendered
    assert "reference" not in rendered.casefold()


def _prepared_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    prepare_canary(
        root,
        _literal_selection(tmp_path),
        runner_path=canary_module.RUNNER_PATH,
    )
    return root


def _passed_results(root: Path) -> list[dict]:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    requests = json.loads((root / "requests.json").read_text(encoding="utf-8"))
    results = []
    for request in requests["requests"]:
        video_id = request["video_id"]
        duration = protocol["duration_seconds"][video_id]
        results.append(
            {
                "run_id": EXPERIMENT_ID,
                "condition_id": protocol["condition_id"],
                "video_id": video_id,
                "status": "passed",
                "mode": "transcription",
                "source_audio_sha256": protocol["audio_sha256"][video_id],
                "text": f"spoken phrase for {video_id}",
                "translated_text": None,
                "answer": None,
                "language": "English",
                "timestamped_text": f"[00:00] spoken phrase for {video_id}",
                "chunks": [
                    {
                        "start": 0.0,
                        "end": min(1.25, duration),
                        "text": f"spoken phrase for {video_id}",
                    }
                ],
                "quality": {"status": "accepted"},
                "elapsed_seconds": 2.5,
                "model_id": protocol["asr_model_id"],
                "model_revision": protocol["asr_revision"],
                "forced_aligner_model_id": protocol["forced_aligner_model_id"],
                "forced_aligner_revision": protocol["forced_aligner_revision"],
                "runtime": "transformers 4.57.6; torch 2.10.0+rocm",
                "runtime_image": protocol["runtime_image"],
                "device": "cuda:0 (ROCm)",
                "precision": "bfloat16",
                "peak_memory_bytes": 4_500_000_000,
                "error_type": None,
                "error_message": None,
                "model_load_seconds": 4.0,
            }
        )
    return results


def test_validate_results_requires_two_ordered_bounded_aligned_rows(
    tmp_path: Path,
) -> None:
    root = _prepared_root(tmp_path)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    requests = json.loads((root / "requests.json").read_text(encoding="utf-8"))
    results = _passed_results(root)

    summary = validate_results(protocol, requests, results)

    assert summary["status"] == "ready_for_quality_test"
    assert summary["completion_count"] == 2
    assert summary["valid_alignment_count"] == 2
    assert summary["paid_compute_usd"] == 0.0
    assert summary["max_peak_memory_bytes"] == 4_500_000_000

    results[0]["chunks"][0]["end"] = protocol["duration_seconds"]["target"] + 0.1
    with pytest.raises(ValueError, match="duration|timestamp"):
        validate_results(protocol, requests, results)


def test_validate_results_preserves_terminal_failure_as_not_ready(
    tmp_path: Path,
) -> None:
    root = _prepared_root(tmp_path)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    requests = json.loads((root / "requests.json").read_text(encoding="utf-8"))
    results = _passed_results(root)
    results[0].update(
        status="failed",
        text=None,
        language=None,
        timestamped_text=None,
        chunks=[],
        quality=None,
        source_audio_sha256=None,
        error_type="RuntimeError",
        error_message="load failed",
        peak_memory_bytes=0,
    )

    summary = validate_results(protocol, requests, results)

    assert summary["status"] == "not_ready"
    assert summary["completion_count"] == 1
    assert summary["valid_alignment_count"] == 1
    assert summary["failure_count"] == 1


def test_record_is_single_attempt_and_prediction_freeze_precedes_scoring(
    tmp_path: Path,
) -> None:
    root = _prepared_root(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_passed_results(root)),
            stderr=(
                f"private runtime log {tmp_path / 'target.wav'} "
                "hf_abcdefghijklmnopqrstuvwxyz123456\n"
            ),
        )

    command = [
        "podman",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--read-only",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HOME=/root/.cache/huggingface",
        "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub",
        (
            "localhost/trec-vqa-audio-rocm@sha256:"
            "faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234"
        ),
        "7278e1e70fe206f11671096ffdd38061171dd6e5",
        "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
    ]
    terminal = record_attempt(root, command, runner=fake_run)

    assert terminal["status"] == "completed"
    assert terminal["attempt_index"] == 1
    assert terminal["paid_compute_usd"] == 0.0
    assert len(calls) == 1
    assert json.loads(calls[0][1]["input"])[0]["video_id"] == "target"
    assert json.loads((root / "command.json").read_text(encoding="utf-8"))[
        "command_sha256"
    ] == terminal["command_sha256"]
    sanitized_stderr = (root / "stderr.txt").read_text(encoding="utf-8")
    assert str(tmp_path / "target.wav") not in sanitized_stderr
    assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in sanitized_stderr
    with pytest.raises(ValueError, match="attempt|overwrite|output"):
        record_attempt(root, command, runner=fake_run)
    with pytest.raises(ValueError, match="prediction freeze"):
        score_postfreeze(
            root,
            references={"target": "spoken phrase", "control": "spoken phrase"},
            whisper={"target": {"quality_status": "rejected-timestamp-overrun"}},
        )

    frozen = freeze_predictions(root)
    assert frozen["status"] == "predictions_frozen"
    scored = score_postfreeze(
        root,
        references={"target": "spoken phrase", "control": "spoken phrase"},
        whisper={
            "target": {
                "quality_status": "rejected-timestamp-overrun",
                "text": "spoken phrase",
            },
            "control": {"quality_status": "accepted", "text": "unrelated"},
        },
    )
    assert scored["status"] == "scored_postfreeze"
    assert scored["official_phrase_present_count"] == 2
    assert scored["whisper_official_phrase_present_count"] == 1
    assert status_canary(root)["status"] == "scored_postfreeze"

    raw_path = root / "raw-results.json"
    raw_path.write_text(raw_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|freeze|artifact"):
        status_canary(root)


def test_record_rejects_podman_without_interactive_stdin_before_execution(
    tmp_path: Path,
) -> None:
    root = _prepared_root(tmp_path)
    called = False

    def must_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid command must not execute")

    with pytest.raises(ValueError, match="interactive|stdin"):
        record_attempt(
            root,
            [
                "podman",
                "run",
                "--rm",
                (
                    "localhost/trec-vqa-audio-rocm@sha256:"
                    "faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234"
                ),
            ],
            runner=must_not_run,
        )
    assert called is False
    assert sorted(path.name for path in root.iterdir()) == [
        "inputs-freeze.json",
        "protocol.json",
        "requests.json",
        "selection.json",
    ]


def test_nonzero_exit_cannot_freeze_ready_decision(tmp_path: Path) -> None:
    root = _prepared_root(tmp_path)

    def failed_after_output(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(_passed_results(root)),
            stderr="terminal failure\n",
        )

    command = build_podman_command(root, hub_cache=tmp_path, data_root=tmp_path)
    record_attempt(root, command, runner=failed_after_output)
    validation = json.loads(
        (root / "validated-results.json").read_text(encoding="utf-8")
    )
    assert validation["completion_count"] == 2
    assert validation["status"] == "not_ready"
    assert "NonzeroExitCode" in validation["failure_types"]
    assert freeze_predictions(root)["decision"] == "not_ready"
    with pytest.raises(ValueError, match="not ready|ready_for_quality_test"):
        score_postfreeze(
            root,
            references={"target": "phrase", "control": "phrase"},
            whisper={
                "target": {"text": "phrase"},
                "control": {"text": "phrase"},
            },
        )
    assert not (root / "postfreeze-score.json").exists()


def test_scoring_rejects_freeze_validation_decision_mismatch(tmp_path: Path) -> None:
    root = _prepared_root(tmp_path)

    def failed_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(_passed_results(root)),
            stderr="terminal failure\n",
        )

    command = build_podman_command(root, hub_cache=tmp_path, data_root=tmp_path)
    record_attempt(root, command, runner=failed_run)
    freeze = freeze_predictions(root)
    assert freeze["decision"] == "not_ready"

    freeze["decision"] = "ready_for_quality_test"
    (root / "predictions-freeze.json").write_text(
        json.dumps(freeze), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match validation"):
        score_postfreeze(
            root,
            references={"target": "phrase", "control": "phrase"},
            whisper={
                "target": {"text": "phrase"},
                "control": {"text": "phrase"},
            },
        )
    assert not (root / "postfreeze-score.json").exists()


def test_launch_exception_is_preserved_as_terminal_failed_attempt(tmp_path: Path) -> None:
    root = _prepared_root(tmp_path)

    def launch_failure(*args, **kwargs):
        raise FileNotFoundError("podman executable missing")

    command = build_podman_command(root, hub_cache=tmp_path, data_root=tmp_path)
    terminal = record_attempt(root, command, runner=launch_failure)

    assert terminal["status"] == "failed"
    assert terminal["exit_code"] == 127
    assert (root / "raw-results.json").read_text(encoding="utf-8") == ""
    validation = json.loads(
        (root / "validated-results.json").read_text(encoding="utf-8")
    )
    assert validation["status"] == "not_ready"
    assert validation["failure_types"] == ["FileNotFoundError"]
    assert freeze_predictions(root)["decision"] == "not_ready"


def test_runner_binding_rejects_wrong_source_and_postfreeze_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong = tmp_path / "wrong.py"
    wrong.write_text("# unrelated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runner source"):
        prepare_canary(
            tmp_path / "wrong-run",
            _literal_selection(tmp_path),
            runner_path=wrong,
        )

    runner = tmp_path / "vqa_audio_runner.py"
    runner.write_bytes(canary_module.RUNNER_PATH.read_bytes())
    monkeypatch.setattr(canary_module, "RUNNER_PATH", runner.resolve())
    root = tmp_path / "bound-run"
    prepare_canary(root, _literal_selection(tmp_path), runner_path=runner)
    runner.write_text("# mutated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runner.*hash|hash.*runner"):
        build_podman_command(root, hub_cache=tmp_path, data_root=tmp_path)
    with pytest.raises(ValueError, match="runner.*hash|hash.*runner"):
        status_canary(root)


def test_selection_rejects_noncanonical_audio_root(tmp_path: Path) -> None:
    wrong_root = tmp_path.parent
    with pytest.raises(ValueError, match="canonical data root"):
        select_panel(
            scores={"rows": {GEMINI_CONDITION: []}},
            maps=[],
            transcripts={"results": []},
            prior_qwen_ids=set(),
            reviewed_failure_ids=set(),
            audio_root=wrong_root,
            selected_at="2026-08-15T12:00:00Z",
            source_hashes={"source_sha256": "1" * 64},
        )


def test_fresh_stage05b_experiment_id_isolated_from_frozen_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alternate_id = "qwen-aligner-stage05b-20260815-v1"
    selection = _literal_selection(tmp_path)
    selection["experiment_id"] = alternate_id
    monkeypatch.setattr(canary_module, "EXPERIMENT_ID", alternate_id)
    root = tmp_path / "stage05b"
    prepared = prepare_canary(
        root,
        selection,
        runner_path=canary_module.RUNNER_PATH,
    )

    assert prepared["experiment_id"] == alternate_id
    assert status_canary(root)["experiment_id"] == alternate_id


def test_quality_panel_selects_exactly_eight_fresh_english_rows(
    tmp_path: Path,
) -> None:
    candidates = []
    for index in range(10):
        video_id = f"speech-{index:02d}"
        _write_audio(tmp_path, video_id, f"audio:{video_id}".encode())
        candidates.append(
            {
                "video_id": video_id,
                "source_language": "en",
                "whisper_status": "accepted",
                "duration_seconds": 5.0,
            }
        )

    selected = select_quality_panel(
        candidates=candidates,
        prior_qwen_ids={"speech-00", "speech-01"},
        audio_root=tmp_path,
        selected_at="2026-08-16T12:00:00Z",
        source_hashes={"manifest_sha256": "1" * 64},
        panel_size=8,
    )

    assert selected["selection_kind"] == "qwen_aligner_quality_panel"
    assert selected["panel_size"] == 8
    assert [row["video_id"] for row in selected["rows"]] == [
        f"speech-{index:02d}" for index in range(2, 10)
    ]
    assert all(row["role"] == "speech_quality" for row in selected["rows"])
    assert not {"speech-00", "speech-01"}.intersection(
        row["video_id"] for row in selected["rows"]
    )


def test_quality_panel_validation_accepts_eight_rows_and_rejects_short_results(
    tmp_path: Path,
) -> None:
    selection = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "selection_kind": "qwen_aligner_quality_panel",
        "panel_size": 8,
        "selected_at": "2026-08-16T12:00:00Z",
        "selection_policy": "fresh English exact-speech panel",
        "rows": [],
        "source_hashes": {"manifest_sha256": "1" * 64},
    }
    for index in range(8):
        video_id = f"speech-{index:02d}"
        payload = f"audio:{video_id}".encode()
        path = _write_audio(tmp_path, video_id, payload).resolve()
        selection["rows"].append(
            {
                "role": "speech_quality",
                "video_id": video_id,
                "audio_path": str(path),
                "audio_sha256": hashlib.sha256(payload).hexdigest(),
                "duration_seconds": 5.0,
                "selection_provenance": "fresh-English-exact-speech",
            }
        )

    root = tmp_path / "quality"
    prepare_canary(
        root,
        selection,
        runner_path=canary_module.RUNNER_PATH,
    )
    requests = json.loads((root / "requests.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["row_count"] == 8
    assert len(requests["requests"]) == 8

    with pytest.raises(ValueError, match="exactly 8 rows"):
        validate_results(protocol, requests, [{"video_id": "speech-00"}])


def test_quality_panel_launch_failure_preserves_eight_row_denominator(
    tmp_path: Path,
) -> None:
    selection = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "selection_kind": "qwen_aligner_quality_panel",
        "panel_size": 8,
        "selected_at": "2026-08-16T12:00:00Z",
        "selection_policy": "fresh English exact-speech panel",
        "rows": [],
        "source_hashes": {"manifest_sha256": "1" * 64},
    }
    for index in range(8):
        video_id = f"speech-{index:02d}"
        payload = f"audio:{video_id}".encode()
        path = _write_audio(tmp_path, video_id, payload).resolve()
        selection["rows"].append(
            {
                "role": "speech_quality",
                "video_id": video_id,
                "audio_path": str(path),
                "audio_sha256": hashlib.sha256(payload).hexdigest(),
                "duration_seconds": 5.0,
                "selection_provenance": "fresh-English-exact-speech",
            }
        )

    root = tmp_path / "quality-launch-failure"
    prepare_canary(root, selection, runner_path=canary_module.RUNNER_PATH)

    def malformed_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="not-json", stderr="")

    command = build_podman_command(root, hub_cache=tmp_path, data_root=tmp_path)
    record_attempt(root, command, runner=malformed_runner)
    validation = json.loads((root / "validated-results.json").read_text())
    assert validation["required_count"] == 8
    assert validation["failure_count"] == 8
