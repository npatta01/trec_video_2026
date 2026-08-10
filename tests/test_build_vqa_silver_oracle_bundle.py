from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.qwen3_omni_structured import (
    REPAIR_PROMPT as RUNTIME_REPAIR_PROMPT,
)
from trec_video_2026.vqa_experiments.qwen3_omni_structured import (
    SELECTOR_PROMPT as RUNTIME_SELECTOR_PROMPT,
)

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_vqa_silver_oracle_bundle.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("vqa_silver_bundle", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load silver-oracle bundle builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def fixture_sources(tmp_path: Path) -> dict[str, Path]:
    media_dir = tmp_path / "media"
    annotations_dir = tmp_path / "annotations"
    media_dir.mkdir()
    annotations_dir.mkdir()
    examples = []
    evidence_examples = []
    whisper_results = []
    questions = {
        "video-a": "What happens after the door opens?",
        "video-b": "How many cups are visible?",
    }
    for index, (video_id, question) in enumerate(questions.items()):
        media_bytes = f"media-{video_id}".encode()
        suffix = ".mkv" if index == 0 else ".mp4"
        media_path = media_dir / f"{video_id}{suffix}"
        media_path.write_bytes(media_bytes)
        media_sha = hashlib.sha256(media_bytes).hexdigest()
        examples.append({"video_id": video_id, "question": question})
        evidence_examples.append(
            {
                "video_id": video_id,
                "question": question,
                "duration_seconds": 10.0 + index,
                "media_path": f"videos/{video_id}{suffix}",
                "media_sha256": media_sha,
            }
        )
        transcript = "[00:00:00.000] Hello.\n[00:00:04.000] Goodbye."
        whisper_results.append(
            {
                "video_id": video_id,
                "status": "passed",
                "transcript_included": index == 0,
                "selected_transcript": transcript if index == 0 else None,
            }
        )
        write_json(
            annotations_dir / f"{video_id}.json",
            {
                "question": question,
                "correct_answer": f"reference-{index}",
                "incorrect_answers": [],
                "timestamp": "2026-01-01T00:00:00.000Z",
                "video_url": f"https://example.invalid/{video_id}",
            },
        )
    manifest = tmp_path / "manifest.json"
    evidence = tmp_path / "evidence.json"
    whisper = tmp_path / "whisper.json"
    write_json(
        manifest,
        {
            "schema_version": 1,
            "label_policy": "label-free",
            "example_count": 2,
            "examples": examples,
        },
    )
    write_json(
        evidence,
        {
            "schema_version": 1,
            "example_count": 2,
            "examples": evidence_examples,
        },
    )
    write_json(
        whisper,
        {
            "schema_version": 1,
            "example_count": 2,
            "results": whisper_results,
        },
    )
    return {
        "manifest": manifest,
        "evidence": evidence,
        "whisper": whisper,
        "annotations_dir": annotations_dir,
        "media_dir": media_dir,
    }


def test_build_bundle_binds_private_selector_rows_and_media(tmp_path: Path) -> None:
    builder = load_builder()
    paths = fixture_sources(tmp_path)
    output = tmp_path / "output"

    summary = builder.build_bundle(
        **paths,
        output_dir=output,
        run_id="silver-a1",
        expected_count=2,
        expected_ids_sha256=None,
    )

    rows = [
        json.loads(line)
        for line in (output / "selector-requests.jsonl").read_text().splitlines()
    ]
    media = json.loads((output / "media-manifest.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    assert summary["row_count"] == 2
    assert [row["video_id"] for row in rows] == ["video-a", "video-b"]
    assert rows[0]["reference"] == "reference-0"
    assert rows[0]["video_path"] == "media/video-a.mkv"
    assert rows[1]["video_path"] == "media/video-b.mp4"
    assert rows[0]["whisper_segments"] == [
        {
            "segment_id": 0,
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "source_line": "[00:00:00.000] Hello.",
        },
        {
            "segment_id": 1,
            "start_seconds": 4.0,
            "end_seconds": 10.0,
            "source_line": "[00:00:04.000] Goodbye.",
        },
    ]
    assert rows[1]["whisper_segments"] == []
    assert rows[0]["whisper_dropped_trailing_line_count"] == 0
    assert len(rows[0]["request_sha256"]) == 64
    assert media["files"][0]["sha256"] == rows[0]["media_sha256"]
    assert protocol["panel"]["count"] == 2
    assert protocol["selector"]["primary_seed"] == 42
    assert protocol["selector"]["repair_seed"] == 43
    assert protocol["selector"]["interval_duration_budget_formula"] == (
        "min(video_duration, max(12, min(30, 0.2 * video_duration)))"
    )
    assert "combined interval duration" in protocol["selector"]["prompt"]
    assert "combined interval duration" in protocol["selector"]["repair_prompt"]
    assert summary["protocol_sha256"] == protocol["protocol_sha256"]
    request_files = protocol["request_files"]
    assert request_files["selector-requests.jsonl"]["row_count"] == 2
    assert (
        request_files["selector-requests.jsonl"]["file_sha256"]
        == hashlib.sha256((output / "selector-requests.jsonl").read_bytes()).hexdigest()
    )
    assert request_files["selector-canary.jsonl"]["row_count"] == 2
    assert request_files["selector-chunk-00.jsonl"]["row_count"] == 1
    assert request_files["selector-chunk-01.jsonl"]["row_count"] == 1
    assert request_files["selector-chunk-02.jsonl"]["row_count"] == 0
    assert request_files["selector-chunk-03.jsonl"]["row_count"] == 0


def test_builder_default_revision_is_fresh_b4() -> None:
    builder = load_builder()

    assert builder.RUN_ID == "vqa-silver-oracle-election-20260809-b4"


def test_bundle_and_runtime_prompts_are_identical() -> None:
    builder = load_builder()

    assert builder.SELECTOR_PROMPT == RUNTIME_SELECTOR_PROMPT
    assert builder.REPAIR_PROMPT == RUNTIME_REPAIR_PROMPT


def test_resource_sentinel_selection_is_label_free_and_max_burden() -> None:
    builder = load_builder()
    rows = [
        {
            "video_id": "short-many-segments",
            "duration_seconds": 20.0,
            "whisper_segments": [{} for _ in range(20)],
            "reference": "must not affect selection",
        },
        {
            "video_id": "long-few-segments",
            "duration_seconds": 30.5,
            "whisper_segments": [{}],
            "reference": "different label",
        },
        {
            "video_id": "long-more-segments",
            "duration_seconds": 30.5,
            "whisper_segments": [{}, {}],
            "reference": "another label",
        },
    ]

    selected = builder.select_resource_sentinel(rows)

    assert selected["video_id"] == "long-more-segments"


def test_build_bundle_rejects_question_mismatch_before_writing(tmp_path: Path) -> None:
    builder = load_builder()
    paths = fixture_sources(tmp_path)
    annotation = paths["annotations_dir"] / "video-a.json"
    payload = json.loads(annotation.read_text())
    payload["question"] = "different question"
    write_json(annotation, payload)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="question mismatch"):
        builder.build_bundle(
            **paths,
            output_dir=output,
            run_id="silver-a1",
            expected_count=2,
            expected_ids_sha256=None,
        )

    assert not (output / "selector-requests.jsonl").exists()


def test_build_bundle_rejects_media_hash_mismatch(tmp_path: Path) -> None:
    builder = load_builder()
    paths = fixture_sources(tmp_path)
    (paths["media_dir"] / "video-b.mp4").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="media hash mismatch"):
        builder.build_bundle(
            **paths,
            output_dir=tmp_path / "output",
            run_id="silver-a1",
            expected_count=2,
            expected_ids_sha256=None,
        )


def test_build_bundle_refuses_existing_reserved_output(tmp_path: Path) -> None:
    builder = load_builder()
    paths = fixture_sources(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "protocol.json").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="protocol.json"):
        builder.build_bundle(
            **paths,
            output_dir=output,
            run_id="silver-a1",
            expected_count=2,
            expected_ids_sha256=None,
        )


def test_partition_rows_uses_disjoint_119_119_119_118_chunks() -> None:
    builder = load_builder()
    rows = [{"video_id": f"video-{index:03d}"} for index in range(475)]

    chunks = builder.partition_rows(rows, chunk_count=4)

    assert [len(chunk) for chunk in chunks] == [119, 119, 119, 118]
    assert [row["video_id"] for chunk in chunks for row in chunk] == [
        row["video_id"] for row in rows
    ]


def test_question_strata_and_canary_allocation_are_question_only() -> None:
    builder = load_builder()
    rows = [
        {"video_id": "t1", "question": "What happens after the bell?"},
        {"video_id": "t2", "question": "What changes at the end?"},
        {"video_id": "n1", "question": "How many people appear?"},
        {"video_id": "n2", "question": "What number of cups remain?"},
        {"video_id": "a1", "question": "What does the speaker say?"},
        {"video_id": "a2", "question": "What sound can be heard?"},
        {"video_id": "c1", "question": "Why does the light turn off?"},
        {"video_id": "o1", "question": "What text is written on the sign?"},
        {"video_id": "x1", "question": "Where is the cat?"},
        {"video_id": "x2", "question": "Who enters the room?"},
    ]

    selected = builder.select_canary_rows(rows)

    assert [row["video_id"] for row in selected] == [
        "t1",
        "t2",
        "n1",
        "n2",
        "a1",
        "a2",
        "c1",
        "o1",
        "x1",
        "x2",
    ]
    assert [row["question_stratum"] for row in selected] == [
        "temporal",
        "temporal",
        "counting",
        "counting",
        "audio_text",
        "audio_text",
        "causal",
        "ocr_detail",
        "other",
        "other",
    ]
