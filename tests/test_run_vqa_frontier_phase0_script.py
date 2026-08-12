from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def test_prepare_preflight_and_status_cli_stdout_is_safe(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(_panel_payload()), encoding="utf-8")
    run_path = tmp_path / "run"

    prepared = _run_cli(
        "prepare",
        "--panel",
        str(panel_path),
        "--output",
        str(run_path),
    )
    assert prepared["status"] == "prepared"
    assert prepared["pilot_count"] == 1
    assert prepared["screen_count"] == 7

    preflight = _run_cli("preflight", "--prepared", str(run_path))
    assert preflight["status"] == "preflight-ready"
    assert preflight["conditions"] == [
        "neutral_map",
        "direct_answer",
        "map_guided_answer",
    ]

    current = _run_cli("status", "--prepared", str(run_path))
    assert current["status"] == "prepared"
    assert current["terminal_conditions"] == 0


def _run_cli(*args: str) -> dict[str, object]:
    script = Path(__file__).parents[1] / "scripts" / "run_vqa_frontier_phase0.py"
    completed = subprocess.run(
        [sys.executable, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for forbidden in (
        "sentinel private question",
        "question:",
        "prompt",
        "top1_answer",
        "raw_response",
        "media_relative_path",
        "fixture/",
        "credential",
        "provider error",
    ):
        assert forbidden not in serialized
    return payload


def _panel_payload() -> dict[str, object]:
    media_sha256 = hashlib.sha256(b"fixture-media").hexdigest()
    rows = [_row(1, "pilot", "pilot-video", media_sha256)]
    rows.extend(
        _row(index + 1, "screen", f"screen-video-{index}", media_sha256)
        for index in range(1, 8)
    )
    return {
        "schema_version": 1,
        "run_id": "vqa-frontier-phase0-20260812",
        "selection_policy": "synthetic-offline-fixture",
        "reference_policy": "not-read-not-serialized",
        "conditions": ["neutral_map", "direct_answer", "map_guided_answer"],
        "examples": rows,
    }


def _row(
    ordinal: int,
    phase: str,
    video_id: str,
    media_sha256: str,
) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "phase": phase,
        "video_id": video_id,
        "question": "sentinel private question",
        "category": "temporal",
        "evidence_topology": "single",
        "selection_rationale": "complete fixture",
        "media_relative_path": f"fixture/{video_id}.webm",
        "media_sha256": media_sha256,
        "size_bytes": len(b"fixture-media"),
        "duration_seconds": 2.0,
        "has_audio": True,
        "has_video": True,
        "condition_ids": ["neutral_map"]
        if phase == "pilot"
        else ["neutral_map", "direct_answer", "map_guided_answer"],
    }
