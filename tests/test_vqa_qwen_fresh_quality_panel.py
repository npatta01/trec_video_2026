from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import wave
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import scripts.run_qwen_aligner_canary as canary_cli  # noqa: E402
import trec_video_2026.vqa_qwen_aligner_canary as canary
from trec_video_2026.vqa_qwen_aligner_canary import (
    build_podman_command,
    build_qwen_asr_exposure_index,
    freeze_predictions,
    prepare_canary,
    record_attempt,
    score_postfreeze,
    select_fresh_english_quality_panel,
)


EXPERIMENT_ID = "qwen-aligner-stage05f-20260816-aligned-english-12-v1"


@pytest.fixture(autouse=True)
def _bind_test_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = (tmp_path / "data").resolve()
    data_root.mkdir()
    monkeypatch.setattr(canary, "CANONICAL_DATA_ROOT", data_root)
    monkeypatch.setattr(canary, "EXPERIMENT_ID", EXPERIMENT_ID)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(root: Path, video_id: str, *, seconds: int = 1) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{video_id}.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8_000)
        handle.writeframes(b"\x00\x00" * 8_000 * seconds)
    return path


def _manifest_rows() -> list[dict[str, str]]:
    return [
        {"video_id": "used", "question": "What did the person say?"},
        {
            "video_id": "r0",
            "question": "What did she say while singing?",
        },
        {"video_id": "c0", "question": "Who is talking?"},
        {"video_id": "v0", "question": "What song is she singing?"},
        {"video_id": "r1", "question": "What did she tell him?"},
        {"video_id": "visible", "question": "What does the screen say?"},
        {"video_id": "c1", "question": "Why are they speaking?"},
        {"video_id": "r2", "question": "What did he ask?"},
        {"video_id": "v1", "question": "Why did he shout?"},
        {"video_id": "r3", "question": "What answer did she announce?"},
        {"video_id": "c2", "question": "What are they talking about?"},
        {"video_id": "v2", "question": "What did the crowd yell?"},
        {"video_id": "r4", "question": "What did she mention?"},
        {"video_id": "r5", "question": "How did he respond?"},
        {"video_id": "oracle", "question": "What did he reply?"},
    ]


def _build_index(data_root: Path) -> tuple[dict, Path, Path]:
    prior = data_root / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3-ASR-1.7B",
                "rows": [{"video_id": "used", "text": "private output"}],
            }
        ),
        encoding="utf-8",
    )
    oracle = data_root / "oracle-manifest.json"
    oracle.write_text(
        json.dumps({"rows": [{"video_id": "oracle", "question": "private"}]}),
        encoding="utf-8",
    )
    index = build_qwen_asr_exposure_index(
        source_paths=[prior],
        oracle_manifest=oracle,
        built_at="2026-08-16T00:00:00Z",
    )
    return index, prior, oracle


def _select(data_root: Path) -> tuple[dict, dict, Path, Path, Path]:
    index, prior, oracle = _build_index(data_root)
    manifest = {"examples": _manifest_rows()}
    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audio_root = data_root / "audio"
    for row in manifest["examples"]:
        _write_wav(audio_root, row["video_id"])
    selection = select_fresh_english_quality_panel(
        manifest=manifest,
        exposure_index=index,
        audio_root=audio_root,
        selected_at="2026-08-16T00:00:01Z",
        source_hashes={
            "manifest_sha256": _sha256(manifest_path),
            "exposure_index_sha256": canary.canonical_sha256(index),
        },
    )
    return selection, index, prior, oracle, audio_root


def test_exposure_index_and_selector_freeze_exact_private_safe_panel(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    selection, index, prior, oracle, _ = _select(data_root)

    assert index["markers"] == [
        "Qwen/Qwen3-ASR-1.7B",
        "qwen3_asr",
        "qwen3-asr",
    ]
    assert index["excluded_video_ids"] == ["oracle", "used"]
    assert index["sources"] == [
        {
            "path": str(prior.resolve()),
            "sha256": _sha256(prior),
            "matched_markers": ["Qwen/Qwen3-ASR-1.7B"],
            "excluded_video_id_count": 1,
        }
    ]
    assert index["oracle_manifest"] == {
        "path": str(oracle.resolve()),
        "sha256": _sha256(oracle),
        "excluded_video_id_count": 1,
    }
    assert Counter(row["speech_stratum"] for row in selection["rows"]) == {
        "reported_speech": 6,
        "conversation": 3,
        "vocalization": 3,
    }
    assert [row["video_id"] for row in selection["rows"]] == [
        "r0",
        "c0",
        "v0",
        "r1",
        "c1",
        "r2",
        "v1",
        "r3",
        "c2",
        "v2",
        "r4",
        "r5",
    ]
    assert selection["rows"][0]["speech_stratum"] == "reported_speech"
    assert selection["rows"][0]["source_order_index"] == 1
    assert selection["panel_size"] == 12
    assert selection["stratum_counts"] == {
        "reported_speech": 6,
        "conversation": 3,
        "vocalization": 3,
    }
    rendered = json.dumps(selection, sort_keys=True).casefold()
    for forbidden in ("question", "answer", "transcript", "reference", "metric"):
        assert f'"{forbidden}"' not in rendered


def test_exposure_index_rejects_noncanonical_or_duplicate_sources(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    _, prior, oracle = _build_index(data_root)
    outside = tmp_path / "outside.json"
    outside.write_text('{"model_id":"qwen3_asr","video_id":"outside"}')

    with pytest.raises(ValueError, match="canonical data root"):
        build_qwen_asr_exposure_index(
            source_paths=[outside],
            oracle_manifest=oracle,
            built_at="2026-08-16T00:00:00Z",
        )
    with pytest.raises(ValueError, match="unique"):
        build_qwen_asr_exposure_index(
            source_paths=[prior, prior],
            oracle_manifest=oracle,
            built_at="2026-08-16T00:00:00Z",
        )


def test_selector_rejects_exposure_source_mutation(tmp_path: Path) -> None:
    data_root = (tmp_path / "data").resolve()
    index, prior, _ = _build_index(data_root)
    prior.write_text('{"model_id":"qwen3_asr","video_id":"mutated"}')

    with pytest.raises(ValueError, match="exposure.*hash|hash.*exposure"):
        select_fresh_english_quality_panel(
            manifest={"examples": _manifest_rows()},
            exposure_index=index,
            audio_root=data_root,
            selected_at="2026-08-16T00:00:01Z",
            source_hashes={"manifest_sha256": "1" * 64},
        )


def test_selector_rejects_missing_audio_and_insufficient_stratum(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    index, _, _ = _build_index(data_root)
    audio_root = data_root / "audio"
    for row in _manifest_rows():
        if row["video_id"] != "r5":
            _write_wav(audio_root, row["video_id"])

    with pytest.raises(ValueError, match="audio.*missing|missing.*audio"):
        select_fresh_english_quality_panel(
            manifest={"examples": _manifest_rows()},
            exposure_index=index,
            audio_root=audio_root,
            selected_at="2026-08-16T00:00:01Z",
            source_hashes={"manifest_sha256": "1" * 64},
        )

    sparse = [
        row
        for row in _manifest_rows()
        if row["video_id"] not in {"c2", "v2"}
    ]
    with pytest.raises(ValueError, match="quota|strata|stratum"):
        select_fresh_english_quality_panel(
            manifest={"examples": sparse},
            exposure_index=index,
            audio_root=audio_root,
            selected_at="2026-08-16T00:00:01Z",
            source_hashes={"manifest_sha256": "1" * 64},
        )


def _old_selection(data_root: Path) -> dict:
    rows = []
    for role in ("target", "control"):
        path = _write_wav(data_root, role)
        rows.append(
            {
                "role": role,
                "video_id": role,
                "audio_path": str(path.resolve()),
                "audio_sha256": _sha256(path),
                "duration_seconds": 1.0,
                "selection_provenance": f"old-{role}",
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "selected_at": "2026-08-16T00:00:00Z",
        "selection_policy": "old two-row test",
        "rows": rows,
        "source_hashes": {"source_sha256": "1" * 64},
    }


def test_prepare_and_command_bind_forced_english_dedicated_runner(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    selection, _, _, _, _ = _select(data_root)
    root = data_root / "run"
    prepare_canary(root, selection, runner_path=canary.DEDICATED_RUNNER_PATH)
    protocol = json.loads((root / "protocol.json").read_text())
    requests = json.loads((root / "requests.json").read_text())
    cache = data_root / "hub"
    cache.mkdir()

    command = build_podman_command(root, hub_cache=cache, data_root=data_root)

    assert protocol["condition_id"] == (
        "qwen3-asr-1.7b-forced-aligner-0.6b-english"
    )
    assert protocol["runtime_contract"]["language"] == "English"
    assert protocol["runtime_contract"]["return_time_stamps"] is True
    assert protocol["runner_path"] == str(canary.DEDICATED_RUNNER_PATH)
    assert protocol["helper_path"] == str(canary.RUNNER_PATH)
    assert protocol["controller_path"] == str(canary.CONTROLLER_PATH)
    assert len(requests["requests"]) == 12
    assert set(requests["requests"][0]) == {
        "run_id",
        "condition_id",
        "video_id",
        "audio_path",
        "mode",
    }
    assert command.count("trec_video_2026.vqa_qwen_empty_output_runner") == 1
    assert command[command.index("--language") + 1] == "English"
    assert command[command.index("--timestamps") + 1] == "on"


def test_old_selection_keeps_original_runner_surface(tmp_path: Path) -> None:
    data_root = (tmp_path / "data").resolve()
    root = data_root / "old-run"
    prepare_canary(root, _old_selection(data_root), runner_path=canary.RUNNER_PATH)

    command = build_podman_command(root, hub_cache=data_root, data_root=data_root)

    assert "trec_video_2026.vqa_audio_runner" in command
    assert "trec_video_2026.vqa_qwen_empty_output_runner" not in command
    assert "--language" not in command
    assert "--timestamps" not in command


@pytest.mark.parametrize(
    ("binding_name", "protocol_key"),
    [
        ("DEDICATED_RUNNER_PATH", "runner"),
        ("RUNNER_PATH", "helper"),
        ("CONTROLLER_PATH", "controller"),
    ],
)
def test_new_command_rejects_bound_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding_name: str,
    protocol_key: str,
) -> None:
    data_root = (tmp_path / "data").resolve()
    bound_files = {}
    for name in ("DEDICATED_RUNNER_PATH", "RUNNER_PATH", "CONTROLLER_PATH"):
        source = getattr(canary, name)
        copied = data_root / f"{name.casefold()}.py"
        copied.write_bytes(source.read_bytes())
        copied = copied.resolve()
        monkeypatch.setattr(canary, name, copied)
        bound_files[name] = copied
    selection, _, _, _, _ = _select(data_root)
    root = data_root / f"mutated-{protocol_key}"
    prepare_canary(
        root,
        selection,
        runner_path=canary.DEDICATED_RUNNER_PATH,
    )
    bound_files[binding_name].write_text("# mutated\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"{protocol_key}.*hash|hash.*{protocol_key}"):
        build_podman_command(root, hub_cache=data_root, data_root=data_root)


def test_record_rejects_any_forced_english_command_drift_before_execution(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    selection, _, _, _, _ = _select(data_root)
    root = data_root / "command-drift"
    prepare_canary(root, selection, runner_path=canary.DEDICATED_RUNNER_PATH)
    command = build_podman_command(root, hub_cache=data_root, data_root=data_root)
    command.extend(["--language", "English"])
    called = False

    def must_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("drifted command must not execute")

    with pytest.raises(ValueError, match="exact forced-English contract"):
        record_attempt(root, command, runner=must_not_run)
    assert called is False
    assert not (root / "command.json").exists()


def _passed_results(root: Path) -> list[dict]:
    protocol = json.loads((root / "protocol.json").read_text())
    requests = json.loads((root / "requests.json").read_text())
    rows = []
    for index, request in enumerate(requests["requests"]):
        video_id = request["video_id"]
        qwen_text = "official phrase" if index < 6 else "different words"
        rows.append(
            {
                "run_id": EXPERIMENT_ID,
                "condition_id": protocol["condition_id"],
                "video_id": video_id,
                "status": "passed",
                "mode": "transcription",
                "source_audio_sha256": protocol["audio_sha256"][video_id],
                "text": qwen_text,
                "translated_text": None,
                "answer": None,
                "language": "English",
                "timestamped_text": f"[00:00] {qwen_text}",
                "chunks": [{"start": 0.0, "end": 0.5, "text": qwen_text}],
                "quality": {"status": "accepted"},
                "elapsed_seconds": 1.0,
                "model_id": protocol["asr_model_id"],
                "model_revision": protocol["asr_revision"],
                "forced_aligner_model_id": protocol["forced_aligner_model_id"],
                "forced_aligner_revision": protocol["forced_aligner_revision"],
                "runtime": "transformers test; torch test",
                "runtime_image": protocol["runtime_image"],
                "device": "cuda:0 (ROCm)",
                "precision": "bfloat16",
                "peak_memory_bytes": 1,
                "error_type": None,
                "error_message": None,
                "model_load_seconds": 1.0,
            }
        )
    return rows


def test_score_adds_paired_cells_and_directional_decision(tmp_path: Path) -> None:
    data_root = (tmp_path / "data").resolve()
    selection, _, _, _, _ = _select(data_root)
    root = data_root / "score-run"
    prepare_canary(root, selection, runner_path=canary.DEDICATED_RUNNER_PATH)
    command = build_podman_command(root, hub_cache=data_root, data_root=data_root)

    def fake_run(command_value, **kwargs):
        return subprocess.CompletedProcess(
            command_value,
            0,
            stdout=json.dumps(_passed_results(root)),
            stderr="",
        )

    record_attempt(root, command, runner=fake_run)
    freeze_predictions(root)
    ids = [row["video_id"] for row in selection["rows"]]
    references = {video_id: "official phrase" for video_id in ids}
    whisper = {
        video_id: {
            "text": "official phrase" if 3 <= index < 9 else None,
            "quality_status": "accepted" if 3 <= index < 9 else "rejected",
        }
        for index, video_id in enumerate(ids)
    }

    score = score_postfreeze(root, references=references, whisper=whisper)

    assert score["official_phrase_present_count"] == 6
    assert score["whisper_official_phrase_present_count"] == 6
    assert score["both_present_count"] == 3
    assert score["qwen_only_present_count"] == 3
    assert score["whisper_only_present_count"] == 3
    assert score["neither_present_count"] == 3
    assert score["directional_decision"] == "no_directional_difference"
    assert sum(
        score[key]
        for key in (
            "both_present_count",
            "qwen_only_present_count",
            "whisper_only_present_count",
            "neither_present_count",
        )
    ) == 12


def test_fresh_cli_commands_create_exclusive_index_and_selection(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    _, prior, oracle = _build_index(data_root)
    index_path = data_root / "exposure-index.json"

    assert (
        canary_cli.main(
            [
                "fresh-exposure-index",
                "--source",
                str(prior),
                "--oracle-manifest",
                str(oracle),
                "--output",
                str(index_path),
                "--built-at",
                "2026-08-16T00:00:00Z",
                "--experiment-id",
                EXPERIMENT_ID,
            ]
        )
        == 0
    )
    frozen_index = index_path.read_bytes()
    assert canary_cli.main(
        [
            "fresh-exposure-index",
            "--source",
            str(prior),
            "--oracle-manifest",
            str(oracle),
            "--output",
            str(index_path),
            "--built-at",
            "2026-08-16T00:00:00Z",
            "--experiment-id",
            EXPERIMENT_ID,
        ]
    ) == 1
    assert index_path.read_bytes() == frozen_index

    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(json.dumps({"examples": _manifest_rows()}))
    audio_root = data_root / "audio"
    for row in _manifest_rows():
        _write_wav(audio_root, row["video_id"])
    selection_path = data_root / "selection.json"
    assert (
        canary_cli.main(
            [
                "fresh-quality-select",
                "--manifest",
                str(manifest_path),
                "--exposure-index",
                str(index_path),
                "--audio-root",
                str(audio_root),
                "--output",
                str(selection_path),
                "--selected-at",
                "2026-08-16T00:00:01Z",
                "--experiment-id",
                EXPERIMENT_ID,
            ]
        )
        == 0
    )
    selected = json.loads(selection_path.read_text())
    assert selected["panel_size"] == 12
    assert selected["source_hashes"] == {
        "exposure_index_sha256": _sha256(index_path),
        "manifest_sha256": _sha256(manifest_path),
    }
