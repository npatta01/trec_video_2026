from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from trec_video_2026.vqa_safety_submission import (
    build_ranked_tail_messages,
    build_test_manifest,
    complete_ranked_candidates,
    extract_uniform_frames,
    finalize_submission,
    build_prediction_row,
    prepare_inference_requests,
    parse_ranked_tail,
    run_nist_validator,
    uniform_timestamps,
    validate_resume_prefix,
    write_new_json,
)
from trec_video_2026.vqa_submission import verify_prediction_freeze
from trec_video_2026.vqa_submission import audit_label_free_payload


def _write_topics(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_manifest_preserves_queries_and_reuses_resolved_media(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    supplied = dataset / "vqa/2026/testing/videos"
    downloaded = dataset / "vqa/2026/testing/videos_original"
    supplied.mkdir(parents=True)
    downloaded.mkdir(parents=True)
    supplied_media = supplied / "8384.webm"
    supplied_media.write_bytes(b"supplied media")
    youtube_media = downloaded / "abcdefghijk.mkv"
    youtube_media.write_bytes(b"youtube media")
    topics = _write_topics(
        dataset / "vqa/2026/testing/topics.json",
        [
            {"Q_ID": 2, "Video_ID": "abcdefghijk", "Question": "Question two?"},
            {"Q_ID": 1, "Video_ID": "8384.webm", "Question": "Question one?"},
            {"Q_ID": 3, "Video_ID": "abcdefghijk", "Question": "Question three?"},
            {"Q_ID": 4, "Video_ID": "missing-id", "Question": "Question four?"},
        ],
    )

    manifest = build_test_manifest(topics, dataset)

    assert manifest["query_count"] == 4
    assert manifest["video_count"] == 3
    assert manifest["available_video_count"] == 2
    assert manifest["unavailable_video_count"] == 1
    assert [row["query_id"] for row in manifest["queries"]] == [2, 1, 3, 4]
    assert manifest["queries"][0]["media"] == (
        "vqa/2026/testing/videos_original/abcdefghijk.mkv"
    )
    assert manifest["queries"][1]["media"] == "vqa/2026/testing/videos/8384.webm"
    assert (
        manifest["queries"][2]["source_sha256"]
        == hashlib.sha256(b"youtube media").hexdigest()
    )
    assert manifest["queries"][3]["availability"] == "unavailable"
    assert manifest["queries"][3]["media"] is None
    audit_label_free_payload(manifest)


def test_manifest_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    topics = _write_topics(
        dataset / "topics.json",
        [
            {"Q_ID": 7, "Video_ID": "first", "Question": "One?"},
            {"Q_ID": 7, "Video_ID": "second", "Question": "Two?"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate query ID"):
        build_test_manifest(topics, dataset)


def test_manifest_rejects_ambiguous_media(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    media_root = dataset / "vqa/2026/testing/videos_original"
    media_root.mkdir(parents=True)
    (media_root / "abcdefghijk.mkv").write_bytes(b"first")
    (media_root / "abcdefghijk.mp4").write_bytes(b"second")
    topics = _write_topics(
        dataset / "topics.json",
        [{"Q_ID": 1, "Video_ID": "abcdefghijk", "Question": "Question?"}],
    )

    with pytest.raises(ValueError, match="multiple local media files"):
        build_test_manifest(topics, dataset)


def test_manifest_rejects_label_bearing_topic_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    topics = _write_topics(
        dataset / "topics.json",
        [
            {
                "Q_ID": 1,
                "Video_ID": "abcdefghijk",
                "Question": "Question?",
                "Answer": "private label",
            }
        ],
    )

    with pytest.raises(ValueError, match="forbidden"):
        build_test_manifest(topics, dataset)


def test_uniform_timestamps_are_twelve_bin_midpoints() -> None:
    assert uniform_timestamps(24.0) == tuple(float(value) for value in range(1, 24, 2))


def test_ranked_tail_prompt_excludes_primary_and_requests_exact_json(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    messages = build_ranked_tail_messages(
        question="What happens next?",
        frame_paths=(frame,),
        frame_timestamps=(0.5,),
        primary_answer="The person leaves",
    )

    instruction = messages[0]["content"][-1]["text"]
    assert "The person leaves" in instruction
    assert "exactly nine" in instruction
    assert 'JSON object with the single key "answers"' in instruction
    serialized = json.dumps(messages).lower()
    assert "reference answer" not in serialized
    assert "ground truth" not in serialized


def test_parse_ranked_tail_accepts_only_ordered_nonempty_strings() -> None:
    text = json.dumps({"answers": [f"candidate {index}" for index in range(1, 10)]})
    assert parse_ranked_tail(text) == tuple(
        f"candidate {index}" for index in range(1, 10)
    )

    with pytest.raises(ValueError, match="exactly nine"):
        parse_ranked_tail(json.dumps({"answers": ["only one"]}))
    with pytest.raises(ValueError, match="JSON object"):
        parse_ranked_tail("not json")


def test_candidate_completion_deduplicates_and_records_fallbacks() -> None:
    tail = json.dumps(
        {
            "answers": [
                "Three.",
                "three",
                "Four",
                "Five",
                "Six",
                "Seven",
                "Eight",
                "Nine",
                "Ten",
            ]
        }
    )

    completion = complete_ranked_candidates(
        primary_answer="Three",
        tail_text=tail,
        question="How many people are visible?",
    )

    assert len(completion.answers) == 10
    assert completion.answers[0] == "Three"
    assert len({answer.casefold().rstrip(".") for answer in completion.answers}) == 10
    assert completion.parsed_tail_count == 7
    assert completion.fallback_count == 2


def _private_predictions(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "safety-run",
        "predictions": [
            {
                "query_id": row["query_id"],
                "video_id": row["video_id"],
                "availability": row["availability"],
                "answers": [
                    {
                        "answer": f"candidate {rank} for {row['query_id']}",
                        "elapsed_seconds": rank / 10,
                        "source": "rank1" if rank == 1 else "ranked-tail",
                    }
                    for rank in range(1, 11)
                ],
            }
            for row in manifest["queries"]
        ],
    }


def test_finalizer_freezes_before_serializing_all_cutoffs(tmp_path: Path) -> None:
    topics = _write_topics(
        tmp_path / "topics.json",
        [
            {"Q_ID": 2, "Video_ID": "missing-two", "Question": "Second?"},
            {"Q_ID": 1, "Video_ID": "missing-one", "Question": "First?"},
        ],
    )
    manifest = build_test_manifest(topics, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    predictions_path = tmp_path / "predictions.json"
    write_new_json(manifest_path, manifest)
    write_new_json(predictions_path, _private_predictions(manifest))

    outputs = finalize_submission(
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        output_dir=tmp_path / "final",
    )

    freeze = json.loads(outputs["freeze"].read_text(encoding="utf-8"))
    verify_prediction_freeze(outputs["predictions"], freeze)
    top10 = json.loads(outputs["top10"].read_text(encoding="utf-8"))
    assert [row["Q_ID"] for row in top10] == [2, 1]
    assert len(top10[0]["Answers"]) == 10
    assert list(top10[0]["Answers"][0]) == ["Rank", "Answer", "Time"]
    assert (
        len(json.loads(outputs["top1"].read_text(encoding="utf-8"))[0]["Answers"]) == 1
    )
    assert (
        len(json.loads(outputs["top5"].read_text(encoding="utf-8"))[0]["Answers"]) == 5
    )

    outputs["predictions"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_sha256"):
        verify_prediction_freeze(outputs["predictions"], freeze)


def test_finalizer_rejects_duplicate_answers_and_wrong_order(tmp_path: Path) -> None:
    topics = _write_topics(
        tmp_path / "topics.json",
        [{"Q_ID": 1, "Video_ID": "missing", "Question": "First?"}],
    )
    manifest = build_test_manifest(topics, tmp_path)
    predictions = _private_predictions(manifest)
    predictions["predictions"][0]["answers"][1]["answer"] = "candidate 1 for 1."
    manifest_path = tmp_path / "manifest.json"
    predictions_path = tmp_path / "predictions.json"
    write_new_json(manifest_path, manifest)
    write_new_json(predictions_path, predictions)

    with pytest.raises(ValueError, match="duplicate normalized answers"):
        finalize_submission(manifest_path, predictions_path, tmp_path / "final")


def test_nist_validator_nonzero_exit_is_propagated(tmp_path: Path) -> None:
    validator = tmp_path / "validator.py"
    validator.write_text("import sys\nprint('validator rejected')\nsys.exit(1)\n")
    topics = _write_topics(tmp_path / "topics.json", [])
    submission = _write_topics(tmp_path / "submission.json", [])

    with pytest.raises(RuntimeError, match="validator rejected"):
        run_nist_validator(
            validator_script=validator,
            topics_path=topics,
            submission_path=submission,
            report_path=tmp_path / "report.json",
            python_executable=sys.executable,
        )


def test_request_preparation_extracts_shared_video_only_once(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    media_root = dataset / "vqa/2026/testing/videos_original"
    media_root.mkdir(parents=True)
    media = media_root / "abcdefghijk.mkv"
    media.write_bytes(b"video")
    topics = _write_topics(
        dataset / "topics.json",
        [
            {"Q_ID": 1, "Video_ID": "abcdefghijk", "Question": "First?"},
            {"Q_ID": 2, "Video_ID": "abcdefghijk", "Question": "Second?"},
            {"Q_ID": 3, "Video_ID": "missing", "Question": "Third?"},
        ],
    )
    manifest = build_test_manifest(topics, dataset)
    calls: list[Path] = []

    def extract_once(
        source: Path, destination: Path
    ) -> tuple[tuple[Path, ...], tuple[float, ...]]:
        calls.append(source)
        destination.mkdir(parents=True)
        frames = tuple(destination / f"frame-{index:02d}.jpg" for index in range(12))
        for frame in frames:
            frame.write_bytes(b"frame")
        return frames, tuple(float(index) for index in range(12))

    requests = prepare_inference_requests(
        manifest,
        dataset_root=dataset,
        frame_root=tmp_path / "frames",
        extract_frames=extract_once,
    )

    assert len(calls) == 1
    assert requests[0]["frame_paths"] == requests[1]["frame_paths"]
    assert requests[2]["frame_paths"] == []
    assert requests[2]["availability"] == "unavailable"


def test_request_preparation_parallelizes_unique_videos_without_reordering(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data"
    media_root = dataset / "vqa/2026/testing/videos_original"
    media_root.mkdir(parents=True)
    for video_id in ("video-one", "video-two"):
        (media_root / f"{video_id}.mkv").write_bytes(video_id.encode())
    topics = _write_topics(
        dataset / "topics.json",
        [
            {"Q_ID": 2, "Video_ID": "video-two", "Question": "Second?"},
            {"Q_ID": 1, "Video_ID": "video-one", "Question": "First?"},
        ],
    )
    manifest = build_test_manifest(topics, dataset)

    def extract(
        source: Path, destination: Path
    ) -> tuple[tuple[Path, ...], tuple[float, ...]]:
        destination.mkdir(parents=True)
        frames = tuple(destination / f"frame-{index:02d}.jpg" for index in range(12))
        for frame in frames:
            frame.write_bytes(source.name.encode())
        return frames, tuple(float(index) for index in range(12))

    requests = prepare_inference_requests(
        manifest,
        dataset_root=dataset,
        frame_root=tmp_path / "frames",
        extract_frames=extract,
        workers=2,
    )

    assert [row["query_id"] for row in requests] == [2, 1]
    assert {Path(row["frame_paths"][0]).parent.name for row in requests} == {
        "video-one",
        "video-two",
    }


def test_prediction_row_keeps_primary_first_and_records_tail_fallbacks() -> None:
    request = {
        "query_id": 9,
        "video_id": "video-9",
        "question": "How many people are visible?",
        "availability": "available",
        "frame_paths": [f"frame-{index}.jpg" for index in range(12)],
        "frame_timestamps": [float(index) for index in range(12)],
    }

    row = build_prediction_row(
        request,
        generate_primary=lambda _: ("Three", 1.25, {"request_sha256": "a" * 64}),
        generate_tail=lambda _request, _primary: (
            json.dumps(
                {"answers": [f"choice {index}" for index in range(8)] + ["Three"]}
            ),
            2.5,
            {"request_sha256": "b" * 64},
        ),
    )

    assert row["answers"][0] == {
        "answer": "Three",
        "elapsed_seconds": 1.25,
        "source": "rank1-uniform12",
    }
    assert len(row["answers"]) == 10
    assert row["parsed_tail_count"] == 8
    assert row["fallback_count"] == 1
    assert row["answers"][-1]["source"] == "deterministic-fallback"
    assert row["rank1_provenance"]["request_sha256"] == "a" * 64


def test_uniform_frame_extraction_writes_twelve_hashed_frames(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=1:r=24",
            "-pix_fmt",
            "yuv420p",
            str(media),
        ],
        check=True,
    )

    frames, timestamps = extract_uniform_frames(media, tmp_path / "frames")

    assert len(frames) == len(timestamps) == 12
    assert all(path.is_file() and path.stat().st_size > 0 for path in frames)
    evidence = json.loads((tmp_path / "frames/evidence.json").read_text())
    assert evidence["source_sha256"] == hashlib.sha256(media.read_bytes()).hexdigest()
    assert len(evidence["frames"]) == 12


def test_resume_prefix_requires_exact_completed_query_order() -> None:
    requests = [
        {"query_id": 1, "video_id": "one"},
        {"query_id": 2, "video_id": "two"},
    ]
    completed = [{"query_id": 1, "video_id": "one", "answers": []}]

    assert validate_resume_prefix(requests, completed) == 1

    completed[0]["video_id"] = "wrong"
    with pytest.raises(ValueError, match="resume prefix"):
        validate_resume_prefix(requests, completed)
