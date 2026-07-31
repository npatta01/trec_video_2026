"""Contract tests for the TimeLens2 compatibility selector adapter."""

import json
import math
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.timelens2_selector import (
    allocate_hamilton,
    build_hybrid_timestamps,
    build_messages,
    build_normalized_interval_result,
    freeze_compatibility_results,
    normalize_intervals,
    parse_raw_interval_result,
    prepare_compatibility_requests,
)

MODEL_REVISION = "90dd30b3eeff3520a30c830eff2209f4ad788ca9"
RUNTIME_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)


def _metadata() -> dict[str, object]:
    return {
        "request_sha256": "a" * 64,
        "model_id": "MCG-NJU/TimeLens2-2B",
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "transformers_version": "4.57.6",
        "latency_seconds": 1.25,
        "peak_memory_bytes": 1234,
        "device": "NVIDIA A100-SXM4-80GB",
        "precision": "bfloat16",
    }


def test_raw_result_preserves_model_order_and_overlaps() -> None:
    raw_output = "[[8, 12], [1.5, 4], [3, 6]]"

    result = parse_raw_interval_result(
        request_id="row-1",
        video_id="video-1",
        duration_seconds=15.0,
        raw_output=raw_output,
        metadata=_metadata(),
    )

    assert result["raw_output"] == raw_output
    assert result["intervals"] == [[8.0, 12.0], [1.5, 4.0], [3.0, 6.0]]
    assert result["duration_seconds"] == 15.0
    assert result["request_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("raw_output", "duration", "message"),
    [
        ("[[1, NaN]]", 10.0, "finite"),
        ("[[4, 3]]", 10.0, "reversed"),
        ("[[-0.1, 3]]", 10.0, "outside"),
        ("[[1, 10.1]]", 10.0, "outside"),
        ("[[true, 3]]", 10.0, "number"),
        ('{"start": 1, "end": 2}', 10.0, "array"),
    ],
)
def test_raw_result_rejects_invalid_intervals(
    raw_output: str, duration: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_raw_interval_result(
            request_id="row-1",
            video_id="video-1",
            duration_seconds=duration,
            raw_output=raw_output,
            metadata=_metadata(),
        )


def test_normalization_is_separate_and_merges_only_sorted_overlaps() -> None:
    raw_intervals = [[8.0, 12.0], [1.5, 4.0], [3.0, 6.0], [7.0, 7.0]]

    normalized = normalize_intervals(raw_intervals)

    assert raw_intervals == [[8.0, 12.0], [1.5, 4.0], [3.0, 6.0], [7.0, 7.0]]
    assert normalized == [[1.5, 6.0], [7.0, 7.0], [8.0, 12.0]]


def test_normalized_result_links_to_raw_output_without_copying_it() -> None:
    raw = parse_raw_interval_result(
        request_id="row-1",
        video_id="video-1",
        duration_seconds=15.0,
        raw_output="[[8, 12], [1.5, 4], [3, 6]]",
        metadata=_metadata(),
    )

    normalized = build_normalized_interval_result(raw)

    assert normalized["schema_version"] == "timelens2-normalized-interval-result-v1"
    assert normalized["intervals"] == [[1.5, 6.0], [8.0, 12.0]]
    assert normalized["raw_output_sha256"] == (
        "51326c9dc63f107b03b828f6ccd8b2bbe90fd3dda102b32939058df4688c9610"
    )
    assert "raw_output" not in normalized


def test_hamilton_allocator_uses_largest_remainders_with_stable_ties() -> None:
    assert allocate_hamilton([5.0, 3.0, 2.0], seats=6) == [3, 2, 1]
    assert allocate_hamilton([1.0, 1.0, 1.0, 1.0], seats=2) == [1, 1, 0, 0]
    assert allocate_hamilton([0.0, 0.0], seats=3) == [2, 1]


def test_hybrid_allocator_returns_six_global_and_six_local_midpoints() -> None:
    allocation = build_hybrid_timestamps(
        duration_seconds=12.0,
        normalized_intervals=[[0.0, 6.0], [9.0, 12.0]],
    )

    assert allocation == {
        "global": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
        "local": [0.75, 2.25, 3.75, 5.25, 9.75, 11.25],
        "local_counts": [4, 2],
        "all": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 0.75, 2.25, 3.75, 5.25, 9.75, 11.25],
    }


def test_prepare_requests_writes_only_label_free_three_video_panel(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    rows = []
    for index in range(3):
        video_id = f"video-{index}"
        media = Path("vqa/videos") / f"{video_id}.mkv"
        annotation = Path("vqa/annotations") / f"{video_id}.json"
        media_path = dataset_root / media
        annotation_path = dataset_root / annotation
        media_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(f"media-{index}".encode())
        annotation_path.write_text(
            json.dumps(
                {
                    "question": f"question {index}",
                    "correct_answer": f"secret {index}",
                    "incorrect_answers": ["also secret"],
                }
            )
        )
        rows.append(
            {
                "video_id": video_id,
                "category": "temporal",
                "media": str(media),
                "annotation": str(annotation),
            }
        )
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"schema_version": 1, "examples": rows}))
    output = tmp_path / "private" / "compatibility-requests.json"

    prepared = prepare_compatibility_requests(
        gate_path=gate,
        dataset_root=dataset_root,
        output_path=output,
        container_media_root=Path("/dstack/input/media"),
    )

    assert prepared["schema_version"] == "timelens2-compatibility-requests-v1"
    assert len(prepared["requests"]) == 3
    assert [row["query"] for row in prepared["requests"]] == [
        "question 0",
        "question 1",
        "question 2",
    ]
    assert all(len(row["request_sha256"]) == 64 for row in prepared["requests"])
    encoded = output.read_text()
    assert "secret" not in encoded
    assert "correct_answer" not in encoded
    assert "incorrect_answers" not in encoded
    assert not any(math.isnan(value) for value in [0.0])


def test_messages_follow_official_video_grounding_prompt_contract() -> None:
    messages = build_messages(Path("/dstack/input/media/video.mkv"), "Find the event")

    assert messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "file:///dstack/input/media/video.mkv",
                    "fps": 2.0,
                    "min_pixels": 32 * 32,
                    "max_pixels": 480 * 480,
                    "total_pixels": 128000 * 32 * 32,
                },
                {
                    "type": "text",
                    "text": (
                        'Given the query: "Find the event", return ALL time spans '
                        "(in seconds) where the query is relevant.\n"
                        "Output format MUST be a JSON array of [start, end] pairs.\n"
                    ),
                },
            ],
        }
    ]


def test_freeze_requires_exact_three_complete_raw_and_normalized_results(
    tmp_path: Path,
) -> None:
    raw_results = []
    normalized_results = []
    for index in range(3):
        raw = parse_raw_interval_result(
            request_id=f"row-{index}",
            video_id=f"video-{index}",
            duration_seconds=15.0,
            raw_output="[[1, 2]]",
            metadata={**_metadata(), "request_sha256": f"{index + 1:064x}"},
        )
        raw_results.append(raw)
        normalized_results.append(build_normalized_interval_result(raw))

    freeze_path = tmp_path / "selector-freeze.json"
    freeze = freeze_compatibility_results(
        raw_results=raw_results,
        normalized_results=normalized_results,
        output_path=freeze_path,
    )

    assert freeze["status"] == "frozen"
    assert freeze["completion"] == {"accepted": 3, "required": 3}
    assert len(freeze["raw_results_sha256"]) == 64
    assert len(freeze["normalized_results_sha256"]) == 64
    assert freeze_path.read_text() == json.dumps(freeze, sort_keys=True) + "\n"

    with pytest.raises(ValueError, match="exactly three"):
        freeze_compatibility_results(
            raw_results=raw_results[:2],
            normalized_results=normalized_results[:2],
            output_path=tmp_path / "incomplete.json",
        )
