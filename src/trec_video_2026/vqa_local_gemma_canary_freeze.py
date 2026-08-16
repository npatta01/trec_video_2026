"""Freeze the local Gemma smoke decision before any reference join."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from trec_video_2026.vqa_blind_frame_model import parse_state_batch_result
from trec_video_2026.vqa_submission import canonical_sha256


def _require_hex(value: object, *, length: int, name: str) -> str:
    token = str(value)
    if re.fullmatch(rf"[0-9a-f]{{{length}}}", token) is None:
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return token


def _matches_model_identity(observed: object, expected: str, revision: str) -> bool:
    observed_text = str(observed)
    if observed_text == expected:
        return True
    cached_name = "models--" + expected.replace("/", "--")
    return observed_text.replace("\\", "/").endswith(
        f"/{cached_name}/snapshots/{revision}"
    )


def build_smoke_freeze(
    *,
    preparation: Mapping[str, object],
    raw_results: Sequence[Mapping[str, object]],
    request_batch_sha256: str,
    raw_results_sha256: str,
    source_revision: str,
    expected_run_id: str,
    expected_answer_policy: str,
) -> dict[str, object]:
    """Classify strict smoke outputs without repairing model text."""

    request_batch_sha256 = _require_hex(
        request_batch_sha256, length=64, name="request batch SHA-256"
    )
    raw_results_sha256 = _require_hex(
        raw_results_sha256, length=64, name="raw results SHA-256"
    )
    source_revision = _require_hex(source_revision, length=40, name="source revision")
    request_complete_sha256 = _require_hex(
        preparation.get("complete_sha256"),
        length=64,
        name="request complete SHA-256",
    )
    model_id = str(preparation.get("model_id"))
    model_revision = _require_hex(
        preparation.get("model_revision"), length=40, name="model revision"
    )
    if not model_id or model_id == "None":
        raise ValueError("preparation model identity is required")
    if not expected_run_id or not expected_answer_policy:
        raise ValueError("expected run ID and answer policy are required")

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
        video_id = str(row["video_id"])
        if video_id in expected_by_video:
            raise ValueError("duplicate planned video identity")
        expected_by_video[video_id] = tuple(
            str(frame["frame_id"]) for frame in protocol["frames"]
        )
    rows: list[dict[str, object]] = []
    completed = 0
    seen_video_ids: set[str] = set()
    for result in raw_results:
        video_id = str(result.get("video_id"))
        condition_id = str(result.get("condition_id"))
        if video_id not in expected_by_video or condition_id != "primary":
            raise ValueError("smoke result identity is not in the frozen request")
        if video_id in seen_video_ids:
            raise ValueError("each planned smoke video must appear exactly once")
        seen_video_ids.add(video_id)
        if (
            result.get("run_id") != expected_run_id
            or result.get("model_revision") != model_revision
            or result.get("answer_policy") != expected_answer_policy
            or not _matches_model_identity(
                result.get("model_id"), model_id, model_revision
            )
        ):
            raise ValueError("smoke result condition provenance does not match")
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
    if seen_video_ids != set(expected_by_video):
        raise ValueError("each planned smoke video must appear exactly once")
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
        "request_complete_sha256": request_complete_sha256,
        "request_batch_sha256": request_batch_sha256,
        "raw_results_sha256": raw_results_sha256,
        "source_revision": source_revision,
        "run_id": expected_run_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "answer_policy": expected_answer_policy,
        "labels_available": False,
        "provider_calls": 0,
        "provider_cost_usd": 0.0,
        "rows": rows,
    }
    return {**unsigned, "complete_sha256": canonical_sha256(unsigned)}
