from __future__ import annotations

import json

from trec_video_2026.vqa_local_gemma_canary_freeze import build_smoke_freeze


def test_fenced_smoke_outputs_freeze_as_invalid_without_repair() -> None:
    preparation = {
        "complete_sha256": "a" * 64,
        "rows": [
            {
                "video_id": "video-1",
                "protocol_request": {
                    "frames": [{"frame_id": "F0"}, {"frame_id": "F1"}]
                },
            }
        ],
    }
    raw_results = [
        {
            "video_id": "video-1",
            "condition_id": "primary",
            "status": "passed",
            "answer": "```json\n"
            + json.dumps(
                [
                    {"frame_id": "F0", "target_state": "absent"},
                    {"frame_id": "F1", "target_state": "present"},
                ]
            )
            + "\n```",
        }
    ]

    frozen = build_smoke_freeze(
        preparation=preparation,
        raw_results=raw_results,
        request_batch_sha256="b" * 64,
        raw_results_sha256="c" * 64,
        source_revision="deadbeef",
    )

    assert frozen["status"] == "stopped_smoke_gate"
    assert frozen["planned_rows"] == 1
    assert frozen["attempted_rows"] == 1
    assert frozen["completed_rows"] == 0
    assert frozen["rows"] == [
        {
            "video_id": "video-1",
            "condition_id": "primary",
            "status": "invalid_json",
            "failure_type": "JSONDecodeError",
        }
    ]
    assert len(frozen["complete_sha256"]) == 64
