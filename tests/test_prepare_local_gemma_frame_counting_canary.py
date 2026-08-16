from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/prepare_local_gemma_frame_counting_canary.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_local_gemma", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_indices_are_stride_four_and_endpoint_inclusive() -> None:
    module = _load_module()

    assert module.selected_source_indices(217) == tuple(range(0, 217, 4))
    assert module.selected_source_indices(448) == (*range(0, 448, 4), 447)


def test_frozen_rows_are_label_free_and_limited_to_eligible_questions() -> None:
    module = _load_module()

    assert [(row["video_id"], row["target"]) for row in module.ROWS] == [
        ("RZrkA52TGBA", "balls"),
        ("PgKOzRgYMAI", "a colorful screen with lines"),
    ]
    forbidden = {"reference", "answer", "correct", "panel_role", "expected"}
    assert all(forbidden.isdisjoint(row) for row in module.ROWS)


def test_preparation_binds_the_complete_inference_and_reducer_trust_base() -> None:
    module = _load_module()

    assert set(module.IMPLEMENTATION_PATHS) == {
        "scripts/prepare_local_gemma_frame_counting_canary.py",
        "src/trec_video_2026/vqa_blind_frame_counting.py",
        "src/trec_video_2026/vqa_blind_frame_model.py",
        "src/trec_video_2026/vqa_count_frame_presence.py",
        "src/trec_video_2026/vqa_counting_evidence.py",
        "src/trec_video_2026/vqa_experiments/temporal_bottleneck.py",
        "src/trec_video_2026/vqa_gate.py",
        "src/trec_video_2026/vqa_models.py",
        "src/trec_video_2026/vqa_submission.py",
        "src/trec_video_2026/vqa_transformers_runner.py",
        "pyproject.toml",
        "uv.lock",
    }
    assert module.MEDIA_EXTENSION == ".png"
