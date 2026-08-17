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
    monkeypatch.setattr(
        canary,
        "FRESH_MANIFEST_PATH",
        data_root / "canonical-pool" / "manifest.json",
        raising=False,
    )
    monkeypatch.setattr(
        canary,
        "FRESH_AUDIO_ROOT",
        data_root / "audio",
        raising=False,
    )
    monkeypatch.setattr(
        canary,
        "FRESH_EXPOSURE_INDEX_PATH",
        data_root / "canonical-selection" / "exposure-index.json",
        raising=False,
    )
    monkeypatch.setattr(
        canary,
        "FRESH_SELECTION_PATH",
        data_root / "canonical-selection" / "selection.json",
        raising=False,
    )
    monkeypatch.setattr(
        canary,
        "FRESH_ORACLE_MANIFEST_PATH",
        data_root / "oracle-speech-10" / "manifest.json",
        raising=False,
    )
    monkeypatch.setattr(
        canary,
        "FRESH_ANNOTATIONS_DIR",
        data_root / "annotations",
        raising=False,
    )
    monkeypatch.setattr(
        canary,
        "FRESH_WHISPER_PATH",
        data_root / "canonical-pool" / "whisper.json",
        raising=False,
    )
    monkeypatch.setattr(
        canary,
        "FRESH_WHISPER_CHECKSUM_PATH",
        data_root / "canonical-pool" / "SHA256SUMS",
        raising=False,
    )
    monkeypatch.setattr(canary, "FRESH_SCAN_ROOT", data_root, raising=False)
    monkeypatch.setattr(
        canary,
        "FRESH_PREPARATION_ONLY_ROOTS",
        (
            data_root / "archives" / "selection-pre-review-455e7ac",
            data_root / "archives" / "run-pre-review-455e7ac",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        canary,
        "PRE_REVIEW_CONTROLLER_REVISION",
        "455e7acb6ad70e7014143077117bc0274c72b3c7",
        raising=False,
    )


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
    decisive = [
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
    fillers = [
        {
            "video_id": f"filler-{index:03d}",
            "question": "What object is visible in the scene?",
        }
        for index in range(475 - len(decisive))
    ]
    return [*decisive, *fillers]


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
    oracle = canary.FRESH_ORACLE_MANIFEST_PATH
    oracle.parent.mkdir(parents=True, exist_ok=True)
    oracle.write_text(
        json.dumps({"rows": [{"video_id": "oracle", "question": "private"}]}),
        encoding="utf-8",
    )
    _write_pre_review_archives(data_root)
    index = build_qwen_asr_exposure_index(
        source_paths=[prior],
        oracle_manifest=oracle,
        built_at="2026-08-16T00:00:00Z",
    )
    return index, prior, oracle


def _write_pre_review_archives(data_root: Path) -> None:
    selected = [
        ("r0", "reported_speech", 1),
        ("c0", "conversation", 2),
        ("v0", "vocalization", 3),
        ("r1", "reported_speech", 4),
        ("c1", "conversation", 6),
        ("r2", "reported_speech", 7),
        ("v1", "vocalization", 8),
        ("r3", "reported_speech", 9),
        ("c2", "conversation", 10),
        ("v2", "vocalization", 11),
        ("r4", "reported_speech", 12),
        ("r5", "reported_speech", 13),
    ]
    rows = []
    for video_id, stratum, source_index in selected:
        audio_path = _write_wav(canary.FRESH_AUDIO_ROOT, video_id)
        rows.append(
            {
                "role": "speech_quality",
                "speech_stratum": stratum,
                "video_id": video_id,
                "audio_path": str(audio_path.resolve()),
                "audio_sha256": _sha256(audio_path),
                "duration_seconds": 1.0,
                "source_order_index": source_index,
                "selection_provenance": "fresh_qwen3_asr",
            }
        )
    selection_payload = {"rows": rows}
    selection_root, run_root = canary.FRESH_PREPARATION_ONLY_ROOTS
    selection_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    (selection_root / "exposure-index.json").write_text(
        json.dumps({"model_id": "qwen3_asr"}), encoding="utf-8"
    )
    (selection_root / "selection.json").write_text(
        json.dumps(selection_payload), encoding="utf-8"
    )
    (run_root / "selection.json").write_text(
        json.dumps(selection_payload), encoding="utf-8"
    )
    (run_root / "requests.json").write_text(
        json.dumps(
            {
                "condition_id": "qwen3-asr-preparation",
                "requests": [{"video_id": video_id} for video_id, _, _ in selected],
            }
        ),
        encoding="utf-8",
    )
    (run_root / "protocol.json").write_text(
        json.dumps({"model_id": "Qwen/Qwen3-ASR-1.7B"}), encoding="utf-8"
    )
    (run_root / "inputs-freeze.json").write_text(
        json.dumps({"status": "inputs_frozen"}), encoding="utf-8"
    )


def _select(data_root: Path) -> tuple[dict, dict, Path, Path, Path]:
    index, prior, oracle = _build_index(data_root)
    manifest = {"examples": _manifest_rows()}
    manifest_path = canary.FRESH_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    index_path = canary.FRESH_EXPOSURE_INDEX_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    audio_root = data_root / "audio"
    for row in manifest["examples"][:15]:
        _write_wav(audio_root, row["video_id"])
    selection = select_fresh_english_quality_panel(
        manifest=manifest,
        exposure_index=index,
        audio_root=audio_root,
        selected_at="2026-08-16T00:00:01Z",
        source_hashes={
            "manifest_sha256": _sha256(manifest_path),
            "exposure_index_sha256": _sha256(index_path),
        },
    )
    canary.FRESH_SELECTION_PATH.write_text(
        json.dumps(selection), encoding="utf-8"
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
    assert index["freshness_definition"] == "never_previously_submitted_to_qwen3_asr"
    assert len(index["preparation_only_dispositions"]) == 1
    disposition = index["preparation_only_dispositions"][0]
    assert disposition["disposition"] == "preparation_only_never_submitted"
    assert disposition["attempt_artifacts_present"] == []
    assert len(disposition["prepared_video_ids"]) == 12
    assert not set(disposition["prepared_video_ids"]).intersection(
        index["excluded_video_ids"]
    )
    marker_paths = {record["path"] for record in index["marker_matches"]}
    classified_paths = {
        record["path"] for record in index["sources"]
    } | {
        record["path"]
        for record in index["marker_matches"]
        if record["disposition"] == "preparation_only_never_submitted"
    }
    assert marker_paths == classified_paths
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


def test_exposure_census_rejects_unclassified_marker_file(tmp_path: Path) -> None:
    data_root = (tmp_path / "data").resolve()
    _, prior, oracle = _build_index(data_root)
    unexpected = data_root / "unexpected.json"
    unexpected.write_text('{"model_id":"qwen3_asr","video_id":"new"}')

    with pytest.raises(ValueError, match="unclassified marker|complete marker census"):
        build_qwen_asr_exposure_index(
            source_paths=[prior],
            oracle_manifest=oracle,
            built_at="2026-08-16T00:00:00Z",
        )


def test_exposure_census_ignores_nonmarker_symlink_but_rejects_marker_symlink(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    _, prior, oracle = _build_index(data_root)
    nonmarker = data_root / "nonmarker.json"
    nonmarker.write_text('{"model_id":"unrelated"}')
    (data_root / "nonmarker-alias.json").symlink_to(nonmarker)

    build_qwen_asr_exposure_index(
        source_paths=[prior],
        oracle_manifest=oracle,
        built_at="2026-08-16T00:00:00Z",
    )

    (data_root / "marker-alias.json").symlink_to(prior)
    with pytest.raises(ValueError, match="marker-bearing symlink"):
        build_qwen_asr_exposure_index(
            source_paths=[prior],
            oracle_manifest=oracle,
            built_at="2026-08-16T00:00:00Z",
        )


def test_preparation_disposition_rejects_attempt_or_inventory_mutation(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    index, _, _ = _build_index(data_root)
    run_root = canary.FRESH_PREPARATION_ONLY_ROOTS[1]
    (run_root / "command.json").write_text('{"status":"command_frozen"}')

    with pytest.raises(ValueError, match="attempt|inventory|preparation"):
        canary._verify_qwen_asr_exposure_index(index)


def test_preparation_disposition_rejects_nested_duplicate_basename(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    _, prior, oracle = _build_index(data_root)
    nested = canary.FRESH_PREPARATION_ONLY_ROOTS[1] / "nested"
    nested.mkdir()
    (nested / "protocol.json").write_text("{}")

    with pytest.raises(ValueError, match="inventory|unexpected"):
        build_qwen_asr_exposure_index(
            source_paths=[prior],
            oracle_manifest=oracle,
            built_at="2026-08-16T00:00:00Z",
        )


def test_preparation_identity_in_real_exposure_stops_census(tmp_path: Path) -> None:
    data_root = (tmp_path / "data").resolve()
    _, prior, oracle = _build_index(data_root)
    prior.write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3-ASR-1.7B",
                "rows": [{"video_id": "used"}, {"video_id": "r0"}],
            }
        )
    )

    with pytest.raises(ValueError, match="prepared.*exposure|exposure.*prepared"):
        build_qwen_asr_exposure_index(
            source_paths=[prior],
            oracle_manifest=oracle,
            built_at="2026-08-16T00:00:00Z",
        )


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
    alternate_oracle = data_root / "alternate-oracle.json"
    alternate_oracle.write_text('{"rows":[{"video_id":"oracle"}]}')
    with pytest.raises(ValueError, match="exact canonical oracle"):
        build_qwen_asr_exposure_index(
            source_paths=[prior],
            oracle_manifest=alternate_oracle,
            built_at="2026-08-16T00:00:00Z",
        )


def test_selector_rejects_self_consistent_alternate_oracle_index(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    index, _, _ = _build_index(data_root)
    alternate_oracle = data_root / "alternate-oracle.json"
    alternate_oracle.write_text('{"rows":[{"video_id":"oracle"}]}')
    index["oracle_manifest"] = {
        "path": str(alternate_oracle.resolve()),
        "sha256": _sha256(alternate_oracle),
        "excluded_video_id_count": 1,
    }
    canary.FRESH_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="exact canonical oracle"):
        select_fresh_english_quality_panel(
            manifest={"examples": _manifest_rows()},
            exposure_index=index,
            audio_root=canary.FRESH_AUDIO_ROOT,
            selected_at="2026-08-16T00:00:01Z",
            source_hashes={
                "manifest_sha256": "1" * 64,
                "exposure_index_sha256": "2" * 64,
            },
        )


def test_selector_rejects_exposure_source_mutation(tmp_path: Path) -> None:
    data_root = (tmp_path / "data").resolve()
    index, prior, _ = _build_index(data_root)
    canary.FRESH_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    prior.write_text('{"model_id":"qwen3_asr","video_id":"mutated"}')

    with pytest.raises(
        ValueError,
        match="exposure.*hash|hash.*exposure|marker census",
    ):
        select_fresh_english_quality_panel(
            manifest={"examples": _manifest_rows()},
            exposure_index=index,
            audio_root=canary.FRESH_AUDIO_ROOT,
            selected_at="2026-08-16T00:00:01Z",
            source_hashes={"manifest_sha256": "1" * 64},
        )


@pytest.mark.parametrize("mutation", ["omitted_id", "forged_markers", "wrong_count"])
def test_selector_rejects_forged_exposure_index_derived_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    data_root = (tmp_path / "data").resolve()
    index, _, _ = _build_index(data_root)
    canary.FRESH_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    if mutation == "omitted_id":
        index["excluded_video_ids"].remove("used")
    elif mutation == "forged_markers":
        index["sources"][0]["matched_markers"] = ["qwen3_asr"]
    else:
        index["sources"][0]["excluded_video_id_count"] = 999

    with pytest.raises(ValueError, match="derived|marker|count|union"):
        select_fresh_english_quality_panel(
            manifest={"examples": _manifest_rows()},
            exposure_index=index,
            audio_root=canary.FRESH_AUDIO_ROOT,
            selected_at="2026-08-16T00:00:01Z",
            source_hashes={
                "manifest_sha256": "1" * 64,
                "exposure_index_sha256": "2" * 64,
            },
        )


def test_selector_rejects_missing_audio_and_insufficient_stratum(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    index, _, _ = _build_index(data_root)
    canary.FRESH_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    audio_root = data_root / "audio"
    manifest = {"examples": _manifest_rows()}
    canary.FRESH_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    canary.FRESH_MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")
    canary.FRESH_EXPOSURE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    canary.FRESH_EXPOSURE_INDEX_PATH.write_text(
        json.dumps(index), encoding="utf-8"
    )
    source_hashes = {
        "manifest_sha256": _sha256(canary.FRESH_MANIFEST_PATH),
        "exposure_index_sha256": _sha256(canary.FRESH_EXPOSURE_INDEX_PATH),
    }
    for row in _manifest_rows()[:15]:
        if row["video_id"] != "r5":
            _write_wav(audio_root, row["video_id"])
    (audio_root / "r5.wav").unlink()

    with pytest.raises(ValueError, match="audio.*missing|missing.*audio"):
        select_fresh_english_quality_panel(
            manifest=manifest,
            exposure_index=index,
            audio_root=audio_root,
            selected_at="2026-08-16T00:00:01Z",
            source_hashes=source_hashes,
        )

    sparse = [
        {
            **row,
            "question": "What object is visible in the scene?",
        }
        if row["video_id"] in {"c2", "v2"}
        else row
        for row in _manifest_rows()
    ]
    sparse_manifest = {"examples": sparse}
    canary.FRESH_MANIFEST_PATH.write_text(
        json.dumps(sparse_manifest), encoding="utf-8"
    )
    sparse_source_hashes = {
        **source_hashes,
        "manifest_sha256": _sha256(canary.FRESH_MANIFEST_PATH),
    }
    with pytest.raises(ValueError, match="quota|strata|stratum"):
        select_fresh_english_quality_panel(
            manifest=sparse_manifest,
            exposure_index=index,
            audio_root=audio_root,
            selected_at="2026-08-16T00:00:01Z",
            source_hashes=sparse_source_hashes,
        )


def test_selector_rejects_truncated_pool(tmp_path: Path) -> None:
    data_root = (tmp_path / "data").resolve()
    index, _, _ = _build_index(data_root)
    canary.FRESH_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="475"):
        select_fresh_english_quality_panel(
            manifest={"examples": _manifest_rows()[:15]},
            exposure_index=index,
            audio_root=canary.FRESH_AUDIO_ROOT,
            selected_at="2026-08-16T00:00:01Z",
            source_hashes={
                "manifest_sha256": "1" * 64,
                "exposure_index_sha256": "2" * 64,
            },
        )


def test_selector_rejects_preparation_replay_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    index, _, _ = _build_index(data_root)
    rows = _manifest_rows()
    rows[1]["question"] = "What object is visible in the scene?"
    rows[15]["question"] = "What did the person say?"
    manifest = {"examples": rows}
    canary.FRESH_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    canary.FRESH_MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")
    canary.FRESH_EXPOSURE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    canary.FRESH_EXPOSURE_INDEX_PATH.write_text(json.dumps(index), encoding="utf-8")
    for row in rows:
        _write_wav(canary.FRESH_AUDIO_ROOT, row["video_id"])

    with pytest.raises(ValueError, match="replay fingerprint"):
        select_fresh_english_quality_panel(
            manifest=manifest,
            exposure_index=index,
            audio_root=canary.FRESH_AUDIO_ROOT,
            selected_at="2026-08-16T00:00:01Z",
            source_hashes={
                "manifest_sha256": _sha256(canary.FRESH_MANIFEST_PATH),
                "exposure_index_sha256": _sha256(
                    canary.FRESH_EXPOSURE_INDEX_PATH
                ),
            },
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


def test_prepare_rejects_missing_source_binding_or_forged_selection(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    selection, _, _, _, _ = _select(data_root)
    missing_binding = json.loads(json.dumps(selection))
    del missing_binding["source_hashes"]["exposure_index_sha256"]
    with pytest.raises(ValueError, match="source bindings|source hashes"):
        prepare_canary(
            data_root / "missing-binding",
            missing_binding,
            runner_path=canary.DEDICATED_RUNNER_PATH,
        )

    forged = json.loads(json.dumps(selection))
    forged["rows"][0]["selection_provenance"] = "handcrafted"
    with pytest.raises(ValueError, match="canonical derived selection"):
        prepare_canary(
            data_root / "forged-selection",
            forged,
            runner_path=canary.DEDICATED_RUNNER_PATH,
        )


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


def test_build_and_record_reject_marker_census_delta_before_execution(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    selection, _, _, _, _ = _select(data_root)
    root = data_root / "census-delta"
    prepare_canary(root, selection, runner_path=canary.DEDICATED_RUNNER_PATH)
    unexpected = data_root / "late-marker.json"
    unexpected.write_text('{"model_id":"qwen3_asr","video_id":"late"}')

    with pytest.raises(ValueError, match="census delta|unclassified marker"):
        build_podman_command(root, hub_cache=data_root, data_root=data_root)

    unexpected.unlink()
    command = build_podman_command(root, hub_cache=data_root, data_root=data_root)
    unexpected.write_text('{"model_id":"qwen3_asr","video_id":"later"}')
    called = False

    def must_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("changed census must not execute")

    with pytest.raises(ValueError, match="census delta|unclassified marker"):
        record_attempt(root, command, runner=must_not_run)
    assert called is False
    assert not (root / "command.json").exists()


@pytest.mark.parametrize("mutation", ["index", "manifest", "audio"])
def test_build_rejects_post_prepare_source_or_audio_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    data_root = (tmp_path / "data").resolve()
    selection, _, _, _, _ = _select(data_root)
    root = data_root / f"source-mutation-{mutation}"
    prepare_canary(root, selection, runner_path=canary.DEDICATED_RUNNER_PATH)
    if mutation == "index":
        payload = json.loads(canary.FRESH_EXPOSURE_INDEX_PATH.read_text())
        canary.FRESH_EXPOSURE_INDEX_PATH.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    elif mutation == "manifest":
        payload = json.loads(canary.FRESH_MANIFEST_PATH.read_text())
        canary.FRESH_MANIFEST_PATH.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    else:
        audio_path = Path(selection["rows"][0]["audio_path"])
        audio_path.write_bytes(audio_path.read_bytes() + b"\x00")

    with pytest.raises(ValueError, match="source hash|audio SHA|duration"):
        build_podman_command(root, hub_cache=data_root, data_root=data_root)


def test_record_rechecks_audio_binding_before_command_freeze(tmp_path: Path) -> None:
    data_root = (tmp_path / "data").resolve()
    selection, _, _, _, _ = _select(data_root)
    root = data_root / "record-audio-mutation"
    prepare_canary(root, selection, runner_path=canary.DEDICATED_RUNNER_PATH)
    command = build_podman_command(root, hub_cache=data_root, data_root=data_root)
    audio_path = Path(selection["rows"][0]["audio_path"])
    audio_path.write_bytes(audio_path.read_bytes() + b"\x00")
    called = False

    def must_not_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("mutated audio must not execute")

    with pytest.raises(ValueError, match="audio SHA|duration"):
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


def _write_canonical_score_sources(
    ids: list[str],
    *,
    whisper_text_by_index: dict[int, str | None],
    rejected_indexes: set[int] | None = None,
) -> None:
    rejected_indexes = rejected_indexes or set()
    canary.FRESH_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for video_id in ids:
        (canary.FRESH_ANNOTATIONS_DIR / f"{video_id}.json").write_text(
            json.dumps({"correct_answer": "official phrase"}),
            encoding="utf-8",
        )
    canary.FRESH_WHISPER_PATH.parent.mkdir(parents=True, exist_ok=True)
    whisper_rows = []
    for index, video_id in enumerate(ids):
        whisper_rows.append(
            {
                "video_id": video_id,
                "text": whisper_text_by_index.get(index),
                "original_quality": {
                    "status": "rejected-test"
                    if index in rejected_indexes
                    else "accepted"
                },
            }
        )
    canary.FRESH_WHISPER_PATH.write_text(
        json.dumps({"results": whisper_rows}), encoding="utf-8"
    )
    relative = canary.FRESH_WHISPER_PATH.relative_to(
        canary.FRESH_WHISPER_CHECKSUM_PATH.parent
    )
    canary.FRESH_WHISPER_CHECKSUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    canary.FRESH_WHISPER_CHECKSUM_PATH.write_text(
        f"{_sha256(canary.FRESH_WHISPER_PATH)}  ./{relative}\n",
        encoding="utf-8",
    )


def _ready_fresh_score_root(data_root: Path, name: str) -> tuple[Path, list[str]]:
    selection, _, _, _, _ = _select(data_root)
    root = data_root / name
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
    return root, [row["video_id"] for row in selection["rows"]]


def test_fresh_score_inputs_reject_vqa_answers_before_loading_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = (tmp_path / "data").resolve()
    root, ids = _ready_fresh_score_root(data_root, "invalid-score-inputs")
    _write_canonical_score_sources(ids, whisper_text_by_index={})
    original_load = canary_cli._load_json

    def reject_annotation_read(path: Path):
        if path.parent == canary.FRESH_ANNOTATIONS_DIR:
            raise AssertionError("VQA answer labels were opened")
        return original_load(path)

    monkeypatch.setattr(canary_cli, "_load_json", reject_annotation_read)

    with pytest.raises(
        ValueError,
        match="VQA answers are not transcript references",
    ):
        canary_cli._score_inputs(
            root,
            canary.FRESH_ANNOTATIONS_DIR,
            canary.FRESH_WHISPER_PATH,
        )


def test_fresh_direct_scorer_rejects_vqa_answers_as_transcript_references(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    root, ids = _ready_fresh_score_root(data_root, "invalid-direct-score")
    _write_canonical_score_sources(ids, whisper_text_by_index={})
    references = {video_id: "because he received some news" for video_id in ids}
    whisper_payload = json.loads(canary.FRESH_WHISPER_PATH.read_text())
    whisper = canary._fresh_whisper_comparison(whisper_payload, ids)
    relative = canary.FRESH_WHISPER_PATH.relative_to(
        canary.FRESH_WHISPER_CHECKSUM_PATH.parent
    ).as_posix()
    source_bindings = {
        "annotations_dir": str(canary.FRESH_ANNOTATIONS_DIR.resolve()),
        "annotation_files_sha256": {
            video_id: _sha256(canary.FRESH_ANNOTATIONS_DIR / f"{video_id}.json")
            for video_id in ids
        },
        "whisper_path": str(canary.FRESH_WHISPER_PATH.resolve()),
        "whisper_sha256": _sha256(canary.FRESH_WHISPER_PATH),
        "whisper_checksum_path": str(
            canary.FRESH_WHISPER_CHECKSUM_PATH.resolve()
        ),
        "whisper_checksum_sha256": _sha256(
            canary.FRESH_WHISPER_CHECKSUM_PATH
        ),
        "whisper_checksum_relative_path": relative,
    }

    with pytest.raises(
        ValueError,
        match="VQA answers are not transcript references",
    ):
        score_postfreeze(
            root,
            references=references,
            whisper=whisper,
            source_bindings=source_bindings,
        )
    assert not (root / "postfreeze-score.json").exists()


def test_fresh_score_inputs_reject_noncanonical_comparator_paths(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    selection, _, _, _, _ = _select(data_root)
    root = data_root / "score-path-run"
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
    _write_canonical_score_sources(ids, whisper_text_by_index={})
    wrong_annotations = data_root / "wrong-annotations"
    wrong_annotations.mkdir()
    wrong_whisper = data_root / "wrong-whisper.json"
    wrong_whisper.write_bytes(canary.FRESH_WHISPER_PATH.read_bytes())

    with pytest.raises(
        ValueError,
        match="VQA answers are not transcript references",
    ):
        canary_cli._score_inputs(
            root,
            wrong_annotations,
            canary.FRESH_WHISPER_PATH,
        )
    with pytest.raises(
        ValueError,
        match="VQA answers are not transcript references",
    ):
        canary_cli._score_inputs(
            root,
            canary.FRESH_ANNOTATIONS_DIR,
            wrong_whisper,
        )


def test_phrase_containment_requires_exact_token_boundaries() -> None:
    assert canary._normalized_phrase_present("he", "the") is False
    assert canary._normalized_phrase_present("he", "the he spoke") is True
    assert canary._normalized_phrase_present("two words", "two other words") is False
    assert canary._normalized_phrase_present("two words", "said two words now") is True


def test_fresh_row_failure_freezes_not_ready_before_score_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = (tmp_path / "data").resolve()
    selection, _, _, _, _ = _select(data_root)
    root = data_root / "failed-row-run"
    prepare_canary(root, selection, runner_path=canary.DEDICATED_RUNNER_PATH)
    command = build_podman_command(root, hub_cache=data_root, data_root=data_root)
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
        error_message="synthetic failure",
        peak_memory_bytes=0,
    )

    def fake_run(command_value, **kwargs):
        return subprocess.CompletedProcess(
            command_value,
            0,
            stdout=json.dumps(results),
            stderr="",
        )

    record_attempt(root, command, runner=fake_run)
    freeze = freeze_predictions(root)
    assert freeze["decision"] == "not_ready"

    def must_not_load(path: Path):
        raise AssertionError("score inputs were opened")

    monkeypatch.setattr(canary_cli, "_load_json", must_not_load)
    with pytest.raises(ValueError, match="not ready|ready_for_quality_test"):
        canary_cli._score_inputs(
            root,
            canary.FRESH_ANNOTATIONS_DIR,
            canary.FRESH_WHISPER_PATH,
        )
    assert not (root / "postfreeze-score.json").exists()


def test_fresh_cli_commands_create_exclusive_index_and_selection(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    _, prior, oracle = _build_index(data_root)
    index_path = canary.FRESH_EXPOSURE_INDEX_PATH

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

    manifest_path = canary.FRESH_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"examples": _manifest_rows()}))
    audio_root = data_root / "audio"
    for row in _manifest_rows()[:15]:
        _write_wav(audio_root, row["video_id"])
    selection_path = canary.FRESH_SELECTION_PATH
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


def test_fresh_quality_cli_rejects_noncanonical_manifest_path(
    tmp_path: Path,
) -> None:
    data_root = (tmp_path / "data").resolve()
    index, _, _ = _build_index(data_root)
    index_path = canary.FRESH_EXPOSURE_INDEX_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    canonical_manifest = canary.FRESH_MANIFEST_PATH
    canonical_manifest.parent.mkdir(parents=True, exist_ok=True)
    canonical_manifest.write_text(json.dumps({"examples": _manifest_rows()}))
    wrong_manifest = data_root / "substitute-manifest.json"
    wrong_manifest.write_bytes(canonical_manifest.read_bytes())
    audio_root = data_root / "audio"
    for row in _manifest_rows()[:15]:
        _write_wav(audio_root, row["video_id"])

    assert (
        canary_cli.main(
            [
                "fresh-quality-select",
                "--manifest",
                str(wrong_manifest),
                "--exposure-index",
                str(index_path),
                "--audio-root",
                str(audio_root),
                "--output",
                str(canary.FRESH_SELECTION_PATH),
                "--experiment-id",
                EXPERIMENT_ID,
            ]
        )
        == 1
    )
    assert not canary.FRESH_SELECTION_PATH.exists()
