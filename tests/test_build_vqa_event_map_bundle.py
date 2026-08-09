from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_vqa_event_map_bundle.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("vqa_event_map_bundle", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load event-map bundle builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def fixture_sources(tmp_path: Path) -> dict[str, Path]:
    media = tmp_path / "media"
    media.mkdir(parents=True)
    examples = []
    evidence_rows = []
    for index in range(8):
        video_id = f"video-{index:02d}"
        questions = [
            "What does the woman say?",
            "What sound is heard?",
            "What color is written on the sign?",
            "What happens after the person sits?",
            "How many people are visible?",
            "Why does the man leave?",
            "What is shown?",
            "What happens next?",
        ]
        raw = f"media-{index}".encode()
        path = media / f"{video_id}.mkv"
        path.write_bytes(raw)
        examples.append({"video_id": video_id, "question": questions[index]})
        evidence_rows.append(
            {
                "video_id": video_id,
                "question": questions[index],
                "media_path": f"videos/{video_id}.mkv",
                "media_sha256": hashlib.sha256(raw).hexdigest(),
                "duration_seconds": 10.0 + index,
            }
        )
    manifest = tmp_path / "manifest.json"
    evidence = tmp_path / "evidence.json"
    write_json(
        manifest,
        {
            "schema_version": 1,
            "label_policy": "label-free-inference-input",
            "example_count": len(examples),
            "examples": examples,
        },
    )
    write_json(
        evidence,
        {
            "schema_version": 1,
            "label_policy": "label-free-evidence",
            "canonical_id_digest": hashlib.sha256(
                ("\n".join(row["video_id"] for row in examples) + "\n").encode()
            ).hexdigest(),
            "examples": evidence_rows,
        },
    )
    return {"manifest": manifest, "evidence": evidence, "media_root": media}


def test_build_bundle_emits_exact_controller_slices_and_private_freeze_identity(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    paths = fixture_sources(tmp_path)
    output = tmp_path / "bundle"
    summary = builder.build_bundle(
        **paths,
        output_dir=output,
        run_id="event-map-test",
        expected_count=8,
        expected_ids_sha256=None,
    )

    controller = json.loads((output / "event-map-controller.json").read_text())
    protocol = json.loads((output / "event-map-protocol.json").read_text())
    assert summary["row_count"] == 8
    assert controller["label_policy"] == "label-free-controller"
    assert controller["panel"]["row_count"] == 8
    assert controller["panel"]["shard_counts"] == [2, 2, 2, 2]
    assert controller["panel"]["canary_count"] == 4
    assert controller["panel"]["ids_sha256"] == summary["ids_sha256"]
    assert protocol["conditions"] == ["v-map", "av-map"]
    assert protocol["decoder"]["max_output_tokens"] <= 768
    assert protocol["decoder"]["temperature"] == 0.0
    assert protocol["decoder"]["enable_thinking"] is False
    assert len(protocol["decoder"]["prompt_sha256"]) == 64
    assert len(protocol["decoder"]["retry_prompt_sha256"]) == 64
    assert protocol["decoder"]["measured_duration_in_prompt"] is True
    assert protocol["video_sampling"] == {
        "fps": 1,
        "max_frames": 64,
        "video_pruning_rate": 0.5,
        "kv_cache_dtype": "fp8",
        "reasoning_parser": "nemotron_v3",
    }
    assert len(protocol["inference_module_sha256"]) == 64
    assert protocol["canary"]["paired"] is True
    assert protocol["canary"]["row_count"] == 4
    assert protocol["request_files"]["event-map-canary.jsonl"]["row_count"] == 8
    assert protocol["request_files"]["v-map-chunk-00.jsonl"]["row_count"] == 2
    assert protocol["request_files"]["av-map-chunk-03.jsonl"]["row_count"] == 2
    for row in controller["rows"]:
        assert set(row) == {
            "video_id",
            "question",
            "question_stratum",
            "media_path",
            "media_sha256",
            "duration_seconds",
        }
    encoded = json.dumps(controller).lower()
    assert "reference" not in encoded
    assert "metric" not in encoded
    assert "correct_answer" not in encoded
    assert (output / "SHA256SUMS").is_file()
    assert summary["protocol_sha256"] == protocol["protocol_sha256"]


def test_build_bundle_matches_verified_475_identity_and_shards(tmp_path: Path) -> None:
    builder = load_builder()
    assert builder.EXPECTED_COUNT == 475
    assert builder.EXPECTED_IDS_SHA256 == (
        "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
    )
    rows = [
        {"video_id": f"{index:03d}", "question": "What is shown?"}
        for index in range(475)
    ]
    assert [len(chunk) for chunk in builder.partition_rows(rows, chunk_count=4)] == [
        119,
        119,
        119,
        118,
    ]


def test_builder_rejects_labels_answers_metrics_and_existing_output(tmp_path: Path) -> None:
    builder = load_builder()
    paths = fixture_sources(tmp_path)
    manifest = json.loads(paths["manifest"].read_text())
    manifest["examples"][0]["reference"] = "leak"
    write_json(paths["manifest"], manifest)
    with pytest.raises(ValueError, match="label|reference"):
        builder.build_bundle(
            **paths,
            output_dir=tmp_path / "bad",
            run_id="event-map-test",
            expected_count=8,
            expected_ids_sha256=None,
        )

    paths = fixture_sources(tmp_path / "clean")
    output = tmp_path / "clean-output"
    output.mkdir()
    (output / "event-map-protocol.json").write_text("existing\n")
    with pytest.raises(FileExistsError):
        builder.build_bundle(
            **paths,
            output_dir=output,
            run_id="event-map-test",
            expected_count=8,
            expected_ids_sha256=None,
        )


def test_builder_rejects_hash_duration_and_question_mismatches(tmp_path: Path) -> None:
    builder = load_builder()
    paths = fixture_sources(tmp_path)
    evidence = json.loads(paths["evidence"].read_text())
    evidence["examples"][0]["media_sha256"] = "0" * 64
    write_json(paths["evidence"], evidence)
    with pytest.raises(ValueError, match="hash"):
        builder.build_bundle(
            **paths,
            output_dir=tmp_path / "hash-bad",
            run_id="event-map-test",
            expected_count=8,
            expected_ids_sha256=None,
        )

    paths = fixture_sources(tmp_path / "question")
    evidence = json.loads(paths["evidence"].read_text())
    evidence["examples"][0]["question"] = "different"
    write_json(paths["evidence"], evidence)
    with pytest.raises(ValueError, match="question"):
        builder.build_bundle(
            **paths,
            output_dir=tmp_path / "question-bad",
            run_id="event-map-test",
            expected_count=8,
            expected_ids_sha256=None,
        )
