"""Freeze the local Gemma smoke decision before any reference join."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from trec_video_2026.vqa_blind_frame_model import parse_state_batch_result
from trec_video_2026.vqa_submission import canonical_sha256


def build_smoke_freeze(
    *,
    preparation: Mapping[str, object],
    raw_results: Sequence[Mapping[str, object]],
    request_batch_sha256: str,
    raw_results_sha256: str,
    source_revision: str,
) -> dict[str, object]:
    """Classify strict smoke outputs without repairing model text."""

    preparation_rows = preparation.get("rows")
    if not isinstance(preparation_rows, list):
        raise TypeError("preparation rows must be an array")
    expected_by_video: dict[str, tuple[str, ...]] = {}
    for row in preparation_rows:
        if not isinstance(row, Mapping):
            raise TypeError("preparation row must be an object")
        protocol = row.get("protocol_request")
        if not isinstance(protocol, Mapping) or not isinstance(
            protocol.get("frames"), list
        ):
            raise TypeError("protocol request frames must be an array")
        expected_by_video[str(row["video_id"])] = tuple(
            str(frame["frame_id"]) for frame in protocol["frames"]
        )
    rows: list[dict[str, object]] = []
    completed = 0
    for result in raw_results:
        video_id = str(result.get("video_id"))
        condition_id = str(result.get("condition_id"))
        if video_id not in expected_by_video or condition_id != "primary":
            raise ValueError("smoke result identity is not in the frozen request")
        try:
            parse_state_batch_result(
                result, expected_frame_ids=expected_by_video[video_id]
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            rows.append(
                {
                    "video_id": video_id,
                    "condition_id": condition_id,
                    "status": "invalid_json",
                    "failure_type": type(error).__name__,
                }
            )
        else:
            completed += 1
            rows.append(
                {
                    "video_id": video_id,
                    "condition_id": condition_id,
                    "status": "completed",
                    "failure_type": None,
                }
            )
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "smoke_passed"
            if completed == len(preparation_rows)
            else "stopped_smoke_gate"
        ),
        "planned_rows": len(preparation_rows),
        "attempted_rows": len(raw_results),
        "completed_rows": completed,
        "request_complete_sha256": preparation["complete_sha256"],
        "request_batch_sha256": request_batch_sha256,
        "raw_results_sha256": raw_results_sha256,
        "source_revision": source_revision,
        "labels_available": False,
        "provider_calls": 0,
        "provider_cost_usd": 0.0,
        "rows": rows,
    }
    return {**unsigned, "complete_sha256": canonical_sha256(unsigned)}
