from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_gemini_evidence_inspector.py"
MODULE = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("build_gemini_inspector", SCRIPT)
    assert spec and spec.loader
    MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MODULE)


VIDEO_IDS = (
    "FBD7bvFnshE",
    "FKsBrmJ2xSg",
    "IRTTtnSRTyY",
    "IVqpg_1aWXU",
    "JTolYjiErBw",
    "LSz_xBDBPNc",
    "Mj3kccs07fg",
)


def make_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    metric_rows = []
    for index, video_id in enumerate(VIDEO_IDS):
        row = root / "rows" / video_id
        (row / "timeline").mkdir(parents=True)
        (row / "selected-frames").mkdir()
        (row / "delivery" / "delivery-media").mkdir(parents=True)
        complete = {
            "video_id": video_id,
            "category": f"category-{index}",
            "question": f"Question {index}?",
            "direct_answer": f"Direct {index}",
            "timeline_answer": f"Timeline {index}",
            "frame_answer": f"Frame {index}",
            "selected_spans": [
                {"start_seconds": 1.0, "end_seconds": 2.0, "reason": "Relevant"}
            ],
        }
        (row / "complete.json").write_text(json.dumps(complete))
        (row / "timeline" / "events.json").write_text(
            json.dumps(
                [
                    {
                        "start_seconds": 0.0,
                        "end_seconds": 2.0,
                        "description": f"Event {index}",
                    }
                ]
            )
        )
        (row / "selected-frames" / "manifest.json").write_text(
            json.dumps(
                [
                    {
                        "path": "frame-01.jpg",
                        "timestamp_seconds": 1.5,
                        "sha256": "a" * 64,
                        "size_bytes": 10,
                    }
                ]
            )
        )
        (row / "delivery" / "delivery-media" / f"{video_id}.webm").write_bytes(b"video")
        (row / "selected-frames" / "frame-01.jpg").write_bytes(b"frame")
        for condition, prefix in (
            ("direct_answer", "Direct"),
            ("timeline_answer", "Timeline"),
            ("frame_answer", "Frame"),
        ):
            metric_rows.append(
                {
                    "video_id": video_id,
                    "category": f"category-{index}",
                    "condition": condition,
                    "prediction": f"{prefix} {index}",
                    "reference": f"Reference {index}",
                    "metrics": {"semantic_composite": 0.5},
                }
            )
    (root / "postfreeze-metrics.json").write_text(json.dumps({"rows": metric_rows}))
    return root


def test_load_cases_projects_all_evidence_with_portal_urls(tmp_path: Path) -> None:
    assert MODULE is not None, "inspector generator is not implemented"
    cases = MODULE.load_cases(make_run(tmp_path), "/private/canary")

    assert len(cases) == 7
    assert cases[3]["question"] == "Question 3?"
    assert cases[3]["reference"] == "Reference 3"
    assert len(cases[3]["answers"]) == 3
    assert cases[3]["timeline"][0]["description"] == "Event 3"
    assert cases[3]["frames"][0]["url"] == (
        "/private/canary/rows/IVqpg_1aWXU/selected-frames/frame-01.jpg"
    )
    assert cases[3]["video_url"] == (
        "/private/canary/rows/IVqpg_1aWXU/delivery/delivery-media/IVqpg_1aWXU.webm"
    )
    assert {answer["verdict"] for answer in cases[3]["answers"]} <= {
        "good",
        "okay",
        "wrong",
        "ambiguous",
    }


def test_load_cases_rejects_missing_required_artifact(tmp_path: Path) -> None:
    assert MODULE is not None, "inspector generator is not implemented"
    root = make_run(tmp_path)
    (root / "rows" / VIDEO_IDS[0] / "timeline" / "events.json").unlink()

    with pytest.raises(ValueError, match="Missing required artifact"):
        MODULE.load_cases(root, "/private/canary")


def test_render_html_contains_comparison_and_seek_controls(tmp_path: Path) -> None:
    assert MODULE is not None, "inspector generator is not implemented"
    cases = MODULE.load_cases(make_run(tmp_path), "/private/canary")

    rendered = MODULE.render_html(cases)

    assert rendered.count('class="case-choice"') == 7
    assert 'id="native-video"' in rendered
    assert 'id="answer-grid"' in rendered
    assert 'id="synthesis-section"' in rendered
    assert 'id="timeline-list"' in rendered
    assert 'id="frame-grid"' in rendered
    assert "seekVideo" in rendered
    assert "@media (max-width: 760px)" in rendered
    assert "Direct native AV" in rendered
    assert "Timeline + selected frames" in rendered


def test_render_html_escapes_artifact_text_inside_page_data(tmp_path: Path) -> None:
    assert MODULE is not None, "inspector generator is not implemented"
    root = make_run(tmp_path)
    complete_path = root / "rows" / VIDEO_IDS[0] / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["question"] = "Could </script><script>alert(1)</script> run?"
    complete_path.write_text(json.dumps(complete))
    cases = MODULE.load_cases(root, "/private/canary")

    rendered = MODULE.render_html(cases)

    assert "</script><script>alert(1)</script>" not in rendered
    assert (
        "Could \\u003c/script\\u003e\\u003cscript\\u003ealert(1)"
        "\\u003c/script\\u003e run?"
    ) in rendered
