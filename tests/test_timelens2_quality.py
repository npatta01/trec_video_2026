"""Contracts for the blinded TimeLens2 selector quality gate."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.timelens2_quality import (
    QualityEvidenceStore,
    build_matched_recipes,
    component_coverage,
    evaluate_selector_coverage,
    extract_recipe_frames,
    freeze_quality_inference,
    prepare_quality_requests,
    quality_decision,
    run_quality_inference,
)
from trec_video_2026.vqa_experiments.timelens2_selector import (
    build_normalized_interval_result,
    parse_raw_interval_result,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_source(
    tmp_path: Path,
    *,
    mutation: str | None = None,
) -> tuple[Path, Path, str]:
    dataset_root = tmp_path / "data"
    media_root = dataset_root / "vqa/2025/training/videos_original"
    media_root.mkdir(parents=True)
    examples = []
    for index in range(10):
        video_id = f"synthetic-video-{index:02d}"
        media = f"private-media-{index}".encode()
        (media_root / f"{video_id}.mkv").write_bytes(media)
        examples.append(
            {
                "video_id": video_id,
                "question": f"Where is event {index}?",
                "duration_seconds": float(index + 10),
                "media_sha256": _sha256_bytes(media),
                "anchor_seconds": float(index),
                "anchor_source": "frozen-label-free-panel",
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": "synthetic-temporal-panel",
        "label_policy": "labels excluded",
        "conditions": ["uniform12", "uniform24", "directed12"],
        "runtime_source_sha256": "a" * 64,
        "examples": examples,
    }
    source = tmp_path / "inference-manifest.json"
    source.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    accepted_sha256 = _sha256_bytes(source.read_bytes())

    if mutation == "source_hash":
        accepted_sha256 = "f" * 64
    elif mutation == "duplicate":
        examples[-1]["video_id"] = examples[0]["video_id"]
        source.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        accepted_sha256 = _sha256_bytes(source.read_bytes())
    elif mutation == "media_hash":
        examples[0]["media_sha256"] = "0" * 64
        source.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        accepted_sha256 = _sha256_bytes(source.read_bytes())
    elif mutation == "label":
        examples[0]["correct_answer"] = "must remain private"
        source.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        accepted_sha256 = _sha256_bytes(source.read_bytes())
    return source, dataset_root, accepted_sha256


def test_prepare_quality_requests_binds_exact_frozen_ten_rows(tmp_path: Path) -> None:
    source, dataset_root, source_sha256 = _write_source(tmp_path)
    output = tmp_path / "quality-requests.json"

    prepared = prepare_quality_requests(
        source_path=source,
        dataset_root=dataset_root,
        output_path=output,
        container_media_root=Path("/dstack/input/media"),
        expected_source_sha256=source_sha256,
    )

    assert prepared["schema_version"] == "timelens2-quality-requests-v1"
    assert prepared["source_sha256"] == source_sha256
    assert len(prepared["requests"]) == 10
    assert len({row["video_id"] for row in prepared["requests"]}) == 10
    assert all(
        Path(row["media_path"]).parent == Path("/dstack/input/media")
        for row in prepared["requests"]
    )
    assert json.loads(output.read_text(encoding="utf-8")) == prepared
    encoded = json.dumps(prepared)
    assert "correct_answer" not in encoded
    assert "incorrect_answers" not in encoded


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("source_hash", "frozen ten-row source"),
        ("duplicate", "distinct video identities"),
        ("media_hash", "media hash mismatch"),
        ("label", "label leakage"),
    ],
)
def test_prepare_quality_requests_rejects_mutation(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    source, dataset_root, source_sha256 = _write_source(tmp_path, mutation=mutation)

    with pytest.raises(ValueError, match=message):
        prepare_quality_requests(
            source_path=source,
            dataset_root=dataset_root,
            output_path=tmp_path / "quality-requests.json",
            container_media_root=Path("/dstack/input/media"),
            expected_source_sha256=source_sha256,
        )


def test_build_matched_recipes_uses_no_padding_and_distinct_frames() -> None:
    recipes = build_matched_recipes(
        duration_seconds=12.0,
        normalized_intervals=[[3.0, 6.0]],
    )

    assert recipes["uniform-12"]["timestamps_seconds"] == [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
        5.5,
        6.5,
        7.5,
        8.5,
        9.5,
        10.5,
        11.5,
    ]
    hybrid = recipes["hybrid-12"]
    assert len(hybrid["timestamps_seconds"]) == 12
    assert hybrid["timestamps_seconds"] == sorted(hybrid["timestamps_seconds"])
    assert len(set(hybrid["timestamps_seconds"])) == 12
    assert len(hybrid["global_timestamps_seconds"]) == 6
    assert len(hybrid["local_timestamps_seconds"]) == 6
    assert all(3.0 < value < 6.0 for value in hybrid["local_timestamps_seconds"])
    assert hybrid["interval_padding_seconds"] == 0.0


def test_build_matched_recipes_rejects_global_local_timestamp_collision() -> None:
    with pytest.raises(ValueError, match="distinct timestamps"):
        build_matched_recipes(
            duration_seconds=12.0,
            normalized_intervals=[[0.0, 12.0]],
        )


def _complete_inference_rows() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    requests = []
    raw_results = []
    normalized_results = []
    recipes = []
    frame_manifests = []
    answers = []
    for index in range(10):
        video_id = f"synthetic-video-{index:02d}"
        request_sha256 = f"{index + 1:064x}"
        request = {
            "request_id": f"quality-{index + 1:02d}",
            "video_id": video_id,
            "request_sha256": request_sha256,
        }
        requests.append(request)
        raw = parse_raw_interval_result(
            request_id=request["request_id"],
            video_id=video_id,
            duration_seconds=12.0,
            raw_output="[[3.0, 6.0]]",
            metadata={
                "request_sha256": request_sha256,
                "model_id": "MCG-NJU/TimeLens2-2B",
                "model_revision": "90dd30b3eeff3520a30c830eff2209f4ad788ca9",
                "runtime_image": "docker.io/pytorch/pytorch@sha256:" + "a" * 64,
                "transformers_version": "4.57.6",
                "latency_seconds": 1.0,
                "peak_memory_bytes": 100,
                "device": "NVIDIA A40",
                "precision": "bfloat16",
            },
        )
        raw_results.append(raw)
        normalized_results.append(build_normalized_interval_result(raw))
        recipe = build_matched_recipes(
            duration_seconds=12.0,
            normalized_intervals=[[3.0, 6.0]],
        )
        recipes.append({"video_id": video_id, "conditions": recipe})
        frame_manifests.append(
            {
                "video_id": video_id,
                "conditions": {
                    condition: [f"{offset + index * 24 + 1:064x}" for offset in range(12)]
                    for condition in ("uniform-12", "hybrid-12")
                },
            }
        )
        for condition in ("uniform-12", "hybrid-12"):
            answers.append(
                {
                    "video_id": video_id,
                    "condition_id": condition,
                    "request_sha256": request_sha256,
                    "status": "passed",
                    "answer": f"answer {index} {condition}",
                    "model_id": "Qwen/Qwen3.5-4B",
                    "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                    "device": "cuda:0 (NVIDIA CUDA)",
                    "precision": "bfloat16",
                }
            )
    return requests, raw_results, normalized_results, recipes, frame_manifests, answers


def test_freeze_quality_inference_requires_all_ten_videos_and_twenty_answers(
    tmp_path: Path,
) -> None:
    rows = _complete_inference_rows()

    with pytest.raises(ValueError, match="20 complete answers"):
        freeze_quality_inference(
            requests=rows[0],
            raw_results=rows[1],
            normalized_results=rows[2],
            recipes=rows[3],
            frame_manifests=rows[4],
            answers=rows[5][:-1],
            output_path=tmp_path / "inference-freeze.json",
        )


def test_freeze_quality_inference_permits_labels_only_after_complete_freeze(
    tmp_path: Path,
) -> None:
    rows = _complete_inference_rows()

    freeze = freeze_quality_inference(
        requests=rows[0],
        raw_results=rows[1],
        normalized_results=rows[2],
        recipes=rows[3],
        frame_manifests=rows[4],
        answers=rows[5],
        output_path=tmp_path / "inference-freeze.json",
    )

    assert freeze["status"] == "frozen"
    assert freeze["label_access"] == "permitted_after_verified_freeze"
    assert freeze["completion"] == {
        "videos": {"accepted": 10, "required": 10},
        "answers": {"accepted": 20, "required": 20},
    }
    assert json.loads((tmp_path / "inference-freeze.json").read_text()) == freeze


def test_quality_evidence_store_persists_partial_failure_and_denominator(
    tmp_path: Path,
) -> None:
    store = QualityEvidenceStore(tmp_path)
    store.checkpoint("selector:model-load")
    store.record_selector_attempt(
        {
            "request_id": "quality-01",
            "video_id": "synthetic-video-00",
            "request_sha256": "1" * 64,
            "status": "accepted",
        }
    )
    store.checkpoint("selector:quality-02:inference")
    store.finalize(
        failure={
            "phase": "selector:quality-02:inference",
            "error_type": "RuntimeError",
            "error_message": "CUDA out of memory",
        }
    )

    status = json.loads((tmp_path / "phase-status.json").read_text())
    summary = json.loads((tmp_path / "quality-summary.json").read_text())
    assert status["status"] == "failed"
    assert summary["completion"] == {
        "videos": {"accepted": 1, "attempted": 1, "required": 10},
        "answers": {"accepted": 0, "attempted": 0, "required": 20},
    }
    assert summary["failure"]["error_type"] == "RuntimeError"


def test_extract_recipe_frames_records_exact_timestamps_and_distinct_hashes(
    tmp_path: Path,
) -> None:
    media = tmp_path / "video.mkv"
    media.write_bytes(b"video")
    recipes = {
        "uniform-12": {"timestamps_seconds": [index + 0.5 for index in range(12)]},
        "hybrid-12": {"timestamps_seconds": [index + 0.25 for index in range(12)]},
    }

    def fake_runner(argv: list[str], **kwargs: object) -> None:
        assert kwargs == {"check": True, "capture_output": True}
        timestamp = argv[argv.index("-ss") + 1]
        Path(argv[-1]).write_bytes(f"frame at {timestamp}".encode())

    manifest = extract_recipe_frames(
        video_id="synthetic-video",
        media_path=media,
        recipes=recipes,
        output_dir=tmp_path / "frames",
        ffmpeg_executable="/pinned/ffmpeg",
        runner=fake_runner,
    )

    assert manifest["video_id"] == "synthetic-video"
    assert manifest["requested_unique_timestamps"] == 24
    assert manifest["conditions"]["uniform-12"][0]["timestamp_seconds"] == 0.5
    assert len(manifest["conditions"]["uniform-12"]) == 12
    assert len(manifest["conditions"]["hybrid-12"]) == 12
    assert len({row["sha256"] for row in manifest["frames"]}) == 24
    assert all(Path(row["path"]).is_file() for row in manifest["frames"])


def test_extract_recipe_frames_rejects_duplicate_decoded_images(tmp_path: Path) -> None:
    media = tmp_path / "video.mkv"
    media.write_bytes(b"video")
    recipes = {
        condition: {"timestamps_seconds": [index + offset for index in range(12)]}
        for condition, offset in (("uniform-12", 0.1), ("hybrid-12", 0.2))
    }

    def duplicate_runner(argv: list[str], **kwargs: object) -> None:
        Path(argv[-1]).write_bytes(b"same decoded image")

    with pytest.raises(ValueError, match="distinct decoded frames"):
        extract_recipe_frames(
            video_id="synthetic-video",
            media_path=media,
            recipes=recipes,
            output_dir=tmp_path / "frames",
            ffmpeg_executable="/pinned/ffmpeg",
            runner=duplicate_runner,
        )


def test_quality_inference_has_no_label_or_adjudication_input() -> None:
    parameters = inspect.signature(run_quality_inference).parameters

    assert "labels" not in parameters
    assert "adjudication" not in parameters
    assert "references" not in parameters


def test_component_coverage_uses_preferred_sufficient_set_and_inclusive_bounds() -> None:
    row = {
        "video_id": "synthetic-video",
        "eligibility": "labelable_visual",
        "sufficient_sets": [
            [[1.0, 2.0]],
            [[3.0, 4.0], [7.0, 8.0]],
        ],
        "preferred_set_index": 1,
    }

    coverage = component_coverage(row, [0.5, 3.0, 7.5, 9.0])

    assert coverage == {
        "component_count": 2,
        "covered_component_count": 2,
        "all_components_covered": True,
        "component_coverage": [True, True],
    }


def test_component_coverage_reports_missing_component() -> None:
    row = {
        "video_id": "synthetic-video",
        "eligibility": "labelable_visual",
        "sufficient_sets": [[[1.0, 2.0], [3.0, 4.0]]],
        "preferred_set_index": 0,
    }

    coverage = component_coverage(row, [1.5, 5.0])

    assert coverage["covered_component_count"] == 1
    assert coverage["all_components_covered"] is False
    assert coverage["component_coverage"] == [True, False]


def test_quality_decision_requires_complete_answers_coverage_and_transitions() -> None:
    assert (
        quality_decision(
            completed_rows=10,
            answers_complete=True,
            rows_with_all_components=8,
            improved_rows=2,
            regressed_rows=1,
        )
        == "pass"
    )
    assert (
        quality_decision(
            completed_rows=10,
            answers_complete=True,
            rows_with_all_components=7,
            improved_rows=2,
            regressed_rows=1,
        )
        == "valid-negative"
    )
    assert (
        quality_decision(
            completed_rows=10,
            answers_complete=False,
            rows_with_all_components=10,
            improved_rows=10,
            regressed_rows=0,
        )
        == "incomplete"
    )


def test_selector_coverage_evaluator_refuses_unfrozen_output(tmp_path: Path) -> None:
    adjudication = tmp_path / "adjudicated.json"
    adjudication.write_text(json.dumps({"rows": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="verified selector freeze"):
        evaluate_selector_coverage(
            run_output_dir=tmp_path / "run",
            expected_run_manifest_sha256="0" * 64,
            adjudication_path=adjudication,
            expected_adjudication_sha256=_sha256_bytes(adjudication.read_bytes()),
            output_path=tmp_path / "coverage.json",
        )
