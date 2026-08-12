from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

import trec_video_2026.vqa_frontier_map_panel as panel_module
from trec_video_2026.vqa_benchmark import MediaMetadata
from trec_video_2026.vqa_frontier_map_panel import (
    CATEGORIES,
    build_candidate_sheet,
    classify_question,
    validate_panel_manifest,
    verify_canonical_data_root,
)


PROBED_DURATIONS: dict[str, float] = {}


def _load_cli_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts/build_vqa_frontier_map_panel.py"
    spec = importlib.util.spec_from_file_location("build_vqa_frontier_map_panel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _question(category: str, index: int) -> str:
    templates = {
        "temporal": "What happens after the person opens the box?",
        "counting": "How many objects are visible in the room?",
        "causal": "Why does the person leave the room?",
        "speech": "What does the person say to the camera?",
        "sound": "What sound is heard after the door closes?",
        "ocr_detail": "What text is shown on the sign?",
        "other_global": "What is happening in the scene?",
    }
    return f"{templates[category]} Fixture {index}."


def _media_path(data_root: Path, video_id: str) -> Path:
    path = data_root / "vqa/2025/training/videos_original" / f"{video_id}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"synthetic media {video_id}".encode())
    return path


def _row(
    data_root: Path,
    ordinal: int,
    phase: str,
    video_id: str,
    question: str,
    category: str,
    topology: str,
    duration: float,
) -> dict[str, object]:
    media = _media_path(data_root, video_id)
    PROBED_DURATIONS[video_id] = duration
    return {
        "ordinal": ordinal,
        "phase": phase,
        "video_id": video_id,
        "question": question,
        "category": category,
        "evidence_topology": topology,
        "selection_rationale": "synthetic question-only fixture selection",
        "media_relative_path": str(media.relative_to(data_root)),
        "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "size_bytes": media.stat().st_size,
        "duration_seconds": duration,
        "has_audio": True,
        "has_video": True,
    }


@pytest.fixture(autouse=True)
def patch_synthetic_media_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    PROBED_DURATIONS.clear()

    def probe(path: Path) -> MediaMetadata:
        video_id = path.stem
        duration = PROBED_DURATIONS.get(video_id)
        assert duration is not None, f"missing synthetic duration for {video_id}"
        return MediaMetadata(
            duration_seconds=duration,
            has_video=True,
            has_audio=True,
            stream_types=("video", "audio"),
            streams=(),
        )

    monkeypatch.setattr(panel_module, "probe_media", probe)
    yield
    PROBED_DURATIONS.clear()


def make_panel_fixture(tmp_path: Path) -> dict[str, object]:
    annotation_root = tmp_path / "vqa/2025/training/annotations"
    annotation_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    topology_cycle = ("single", "before-after", "multi-moment")
    ordinal = 1
    for category in CATEGORIES:
        for index in range(5):
            video_id = f"v{ordinal:02d}"
            question = _question(category, index)
            (annotation_root / f"{video_id}.json").write_text(
                json.dumps(
                    {
                        "video_url": f"https://example.test/{video_id}",
                        "question": question,
                        "correct_answer": "must never cross the boundary",
                        "incorrect_answers": ["also private"],
                    }
                ),
                encoding="utf-8",
            )
            rows.append(
                _row(
                    tmp_path,
                    ordinal,
                    "evaluation",
                    video_id,
                    question,
                    category,
                    topology_cycle[index % len(topology_cycle)],
                    float(index + 1),
                )
            )
            ordinal += 1

    for category in ("temporal", "sound"):
        video_id = f"v{ordinal:02d}"
        question = _question(category, 99)
        (annotation_root / f"{video_id}.json").write_text(
            json.dumps({"question": question, "reference": "private"}),
            encoding="utf-8",
        )
        rows.append(
            _row(
                tmp_path,
                ordinal,
                "preflight",
                video_id,
                question,
                category,
                "single",
                3.0,
            )
        )
        ordinal += 1

    return {
        "schema_version": 1,
        "run_id": "frontier-video-map-20260811",
        "selection_policy": "question-only-category-purity-duration-topology-v1",
        "reference_policy": "not-read-not-serialized",
        "evaluation_count": 35,
        "preflight_count": 2,
        "examples": rows,
    }


def _write_candidate_annotation(
    data_root: Path,
    *,
    video_id: str,
    question: str,
    duration: float,
    topology: str = "before-after",
) -> dict[str, object]:
    annotation_root = data_root / "vqa/2025/training/annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    (annotation_root / f"{video_id}.json").write_text(
        json.dumps(
            {
                "video_url": f"https://example.test/{video_id}",
                "question": question,
                "correct_answer": "must never cross the boundary",
            }
        ),
        encoding="utf-8",
    )
    return _row(
        data_root,
        0,
        "evaluation",
        video_id,
        question,
        classify_question(question),
        topology,
        duration,
    )


def _set_row_duration(row: dict[str, object], duration: float) -> None:
    video_id = row["video_id"]
    assert isinstance(video_id, str)
    PROBED_DURATIONS[video_id] = duration
    row["duration_seconds"] = duration


def test_sound_rule_does_not_match_heart_substring() -> None:
    assert classify_question("What is the heart made of?") == "other_global"
    assert classify_question("What sound is heard after the door closes?") == "sound"


def test_classification_uses_documented_precedence() -> None:
    assert classify_question("How many things happen after the door closes?") == "counting"
    assert classify_question("Why does the person speak after entering?") == "causal"


def test_panel_requires_five_rows_per_category_and_two_preflight(tmp_path: Path) -> None:
    payload = make_panel_fixture(tmp_path)
    panel = validate_panel_manifest(payload, tmp_path)

    assert len(panel.evaluation_rows) == 35
    assert len(panel.preflight_rows) == 2
    assert Counter(row.category for row in panel.evaluation_rows) == {
        "temporal": 5,
        "counting": 5,
        "causal": 5,
        "speech": 5,
        "sound": 5,
        "ocr_detail": 5,
        "other_global": 5,
    }

    assert {row.ordinal for row in panel.preflight_rows}.isdisjoint(
        row.ordinal for row in panel.evaluation_rows
    )


def test_panel_requires_distinct_preflight_rows(tmp_path: Path) -> None:
    payload = make_panel_fixture(tmp_path)
    examples = payload["examples"]
    assert isinstance(examples, list)
    duplicate = dict(examples[-2])
    duplicate["ordinal"] = 37
    examples[-1] = duplicate

    with pytest.raises(ValueError, match="preflight"):
        validate_panel_manifest(payload, tmp_path)


def test_panel_accepts_exact_30_second_boundary_for_selected_rows(
    tmp_path: Path,
) -> None:
    payload = make_panel_fixture(tmp_path)
    examples = payload["examples"]
    assert isinstance(examples, list)

    evaluation_row = examples[0]
    preflight_row = examples[-2]
    assert isinstance(evaluation_row, dict)
    assert isinstance(preflight_row, dict)
    _set_row_duration(evaluation_row, 30.0)
    _set_row_duration(preflight_row, 30.0)

    panel = validate_panel_manifest(payload, tmp_path)

    assert panel.evaluation_rows[0].duration_seconds == 30.0
    assert panel.preflight_rows[0].duration_seconds == 30.0


def test_panel_rejects_evaluation_rows_longer_than_30_seconds(
    tmp_path: Path,
) -> None:
    payload = make_panel_fixture(tmp_path)
    examples = payload["examples"]
    assert isinstance(examples, list)

    row = examples[0]
    assert isinstance(row, dict)
    _set_row_duration(row, 30.1)

    with pytest.raises(ValueError, match="30\\.0"):
        validate_panel_manifest(payload, tmp_path)


def test_panel_rejects_preflight_rows_longer_than_30_seconds(
    tmp_path: Path,
) -> None:
    payload = make_panel_fixture(tmp_path)
    examples = payload["examples"]
    assert isinstance(examples, list)

    row = examples[-1]
    assert isinstance(row, dict)
    _set_row_duration(row, 30.1)

    with pytest.raises(ValueError, match="30\\.0"):
        validate_panel_manifest(payload, tmp_path)


def test_panel_binds_duration_and_media_identity_to_current_bytes(tmp_path: Path) -> None:
    payload = make_panel_fixture(tmp_path)
    validate_panel_manifest(payload, tmp_path)

    first = payload["examples"][0]
    assert isinstance(first, dict)
    first["media_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        validate_panel_manifest(payload, tmp_path)


def test_panel_requires_three_topologies_per_category(tmp_path: Path) -> None:
    payload = make_panel_fixture(tmp_path)
    examples = payload["examples"]
    assert isinstance(examples, list)
    for row in examples:
        assert isinstance(row, dict)
        if row["phase"] == "evaluation" and row["category"] == "sound":
            row["evidence_topology"] = "single"

    with pytest.raises(ValueError, match="topolog"):
        validate_panel_manifest(payload, tmp_path)


def test_panel_rejects_preflight_reuse_of_evaluation_video_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_panel_fixture(tmp_path)
    examples = payload["examples"]
    assert isinstance(examples, list)

    evaluation_row = examples[0]
    preflight_row = examples[-2]
    assert isinstance(evaluation_row, dict)
    assert isinstance(preflight_row, dict)
    preflight_row["video_id"] = evaluation_row["video_id"]
    preflight_row["media_relative_path"] = evaluation_row["media_relative_path"]
    preflight_row["media_sha256"] = evaluation_row["media_sha256"]
    preflight_row["size_bytes"] = evaluation_row["size_bytes"]
    preflight_row["duration_seconds"] = evaluation_row["duration_seconds"]

    candidate_sheet = build_candidate_sheet(tmp_path)
    candidates = candidate_sheet["candidates"]
    assert isinstance(candidates, list)
    reused_candidate = dict(candidates[0])
    reused_candidate["question"] = preflight_row["question"]
    reused_candidate["category"] = preflight_row["category"]
    candidates.append(reused_candidate)
    monkeypatch.setattr(panel_module, "build_candidate_sheet", lambda data_root: candidate_sheet)

    with pytest.raises(ValueError, match="preflight"):
        validate_panel_manifest(payload, tmp_path)


def test_panel_rejects_preflight_reuse_of_another_preflight_video_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_panel_fixture(tmp_path)
    examples = payload["examples"]
    assert isinstance(examples, list)

    first_preflight = examples[-2]
    second_preflight = examples[-1]
    assert isinstance(first_preflight, dict)
    assert isinstance(second_preflight, dict)
    second_preflight["video_id"] = first_preflight["video_id"]
    second_preflight["media_relative_path"] = first_preflight["media_relative_path"]
    second_preflight["media_sha256"] = first_preflight["media_sha256"]
    second_preflight["size_bytes"] = first_preflight["size_bytes"]
    second_preflight["duration_seconds"] = first_preflight["duration_seconds"]

    candidate_sheet = build_candidate_sheet(tmp_path)
    candidates = candidate_sheet["candidates"]
    assert isinstance(candidates, list)
    reused_candidate = dict(candidates[-2])
    reused_candidate["question"] = second_preflight["question"]
    reused_candidate["category"] = second_preflight["category"]
    candidates.append(reused_candidate)
    monkeypatch.setattr(panel_module, "build_candidate_sheet", lambda data_root: candidate_sheet)

    with pytest.raises(ValueError, match="preflight"):
        validate_panel_manifest(payload, tmp_path)


def test_panel_requires_short_typical_and_long_duration_coverage(
    tmp_path: Path,
) -> None:
    payload = make_panel_fixture(tmp_path)
    examples = payload["examples"]
    assert isinstance(examples, list)
    replacements = [
        _write_candidate_annotation(
            tmp_path,
            video_id=f"v8{index}",
            question=_question("temporal", 200 + index),
            duration=2.1 + (0.1 * index),
            topology=("before-after", "multi-moment", "ordered-sequence")[index % 3],
        )
        for index in range(5)
    ]
    temporal_indexes = [
        index
        for index, row in enumerate(examples)
        if isinstance(row, dict)
        and row["phase"] == "evaluation"
        and row["category"] == "temporal"
    ]
    for index, replacement in zip(temporal_indexes, replacements, strict=True):
        replacement["ordinal"] = examples[index]["ordinal"]
        examples[index] = replacement

    with pytest.raises(ValueError, match="duration"):
        validate_panel_manifest(payload, tmp_path)


def test_panel_rejects_recursive_label_fields(tmp_path: Path) -> None:
    payload = make_panel_fixture(tmp_path)
    payload["review"] = {"nested": {"reference_answer": "private"}}

    with pytest.raises(ValueError, match="label-free"):
        validate_panel_manifest(payload, tmp_path)


def test_candidate_sheet_projects_questions_and_excludes_labels(tmp_path: Path) -> None:
    make_panel_fixture(tmp_path)

    sheet = build_candidate_sheet(tmp_path)

    assert sheet["example_count"] == 37
    candidates = sheet["candidates"]
    assert isinstance(candidates, list)
    candidate_by_id = {
        row["video_id"]: row for row in candidates if isinstance(row, dict)
    }
    first = candidate_by_id["v01"]
    assert set(first) == {
        "video_id",
        "question",
        "category",
        "duration_seconds",
        "evidence_topology_hints",
        "media_relative_path",
        "media_sha256",
        "size_bytes",
    }
    assert all(set(row) == set(first) for row in candidates if isinstance(row, dict))
    assert first["category"] == "temporal"
    assert first["duration_seconds"] == 1.0
    assert first["media_relative_path"] == "vqa/2025/training/videos_original/v01.mp4"
    serialized = json.dumps(sheet)
    assert "annotation_filename" not in serialized
    assert "annotation_sha256" not in serialized
    assert "annotation_size_bytes" not in serialized
    assert "correct_answer" not in serialized
    assert "incorrect_answers" not in serialized
    assert '"reference":' not in serialized
    assert "reference_answer" not in serialized
    assert all("annotation" not in row for row in sheet["candidates"])


def test_candidate_sheet_skips_annotations_without_media(tmp_path: Path) -> None:
    annotation_root = tmp_path / "vqa/2025/training/annotations"
    annotation_root.mkdir(parents=True)

    present_id = "present-video"
    present_question = "What happens after the person opens the box?"
    (annotation_root / f"{present_id}.json").write_text(
        json.dumps(
            {
                "video_url": f"https://example.test/{present_id}",
                "question": present_question,
                "correct_answer": "private answer",
            }
        ),
        encoding="utf-8",
    )
    _media_path(tmp_path, present_id)
    PROBED_DURATIONS[present_id] = 4.0

    missing_id = "missing-video"
    (annotation_root / f"{missing_id}.json").write_text(
        json.dumps(
            {
                "video_url": f"https://example.test/{missing_id}",
                "question": "What is happening in the scene?",
                "correct_answer": "private answer",
            }
        ),
        encoding="utf-8",
    )

    sheet = build_candidate_sheet(tmp_path)

    assert sheet["example_count"] == 1
    assert sheet["missing_media_count"] == 1
    candidates = sheet["candidates"]
    assert isinstance(candidates, list)
    assert [row["video_id"] for row in candidates] == [present_id]
    serialized = json.dumps(sheet)
    assert missing_id not in serialized
    assert f"{missing_id}.json" not in serialized


def test_verify_canonical_data_root_requires_exact_resolved_match(
    tmp_path: Path,
) -> None:
    expected_root = tmp_path / "canonical"
    expected_root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(expected_root, target_is_directory=True)
    other_root = tmp_path / "other"
    other_root.mkdir()

    assert verify_canonical_data_root(alias, expected_root=expected_root) == expected_root.resolve()
    with pytest.raises(ValueError, match="canonical dataset root"):
        verify_canonical_data_root(other_root, expected_root=expected_root)


def test_cli_freeze_writes_canonical_manifest_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    monkeypatch.setattr(cli, "verify_canonical_data_root", lambda path: path.resolve())
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(make_panel_fixture(tmp_path)), encoding="utf-8"
    )
    output = tmp_path / "frozen.json"

    assert cli.main(
        [
            "freeze",
            "--data-root",
            str(tmp_path),
            "--selection",
            str(selection),
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["evaluation_count"] == 35

    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(
            [
                "freeze",
                "--data-root",
                str(tmp_path),
                "--selection",
                str(selection),
                "--output",
                str(output),
            ]
        )


def test_cli_freeze_refuses_symlinked_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli_module()
    monkeypatch.setattr(cli, "verify_canonical_data_root", lambda path: path.resolve())
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(make_panel_fixture(tmp_path)), encoding="utf-8")
    target = tmp_path / "target.json"
    target.write_text("keep", encoding="utf-8")
    output = tmp_path / "frozen.json"
    output.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        cli.main(
            [
                "freeze",
                "--data-root",
                str(tmp_path),
                "--selection",
                str(selection),
                "--output",
                str(output),
            ]
        )
