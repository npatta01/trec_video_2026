#!/usr/bin/env python3
"""Run the four-video AV-only plain-text timeline canary.

The driver intentionally keeps the map and selector as two separate calls. It
stores hashes and parsed evidence, never raw completion text or exceptions, and
does not retry malformed generations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

from trec_video_2026.vqa_experiments import nemotron3_omni_event_map as event_map


Post = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("request row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError("request file must not be empty")
    return rows


def _load_canary_rows(path: Path) -> list[dict[str, Any]]:
    """Deduplicate the frozen paired request file to its four AV rows."""

    by_id: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("request video_id must be nonempty text")
        if row.get("condition_id") != "av-map":
            continue
        if video_id in by_id and any(
            by_id[video_id].get(key) != row.get(key)
            for key in ("question", "duration_seconds", "media_path", "media_sha256")
        ):
            raise ValueError("duplicate AV request rows disagree")
        by_id[video_id] = row
    if len(by_id) != 4:
        raise ValueError("canary must contain exactly four AV rows")
    return list(by_id.values())


def _load_normalized_controller(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("normalized controller rows are missing")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("video_id"), str):
            raise ValueError("normalized controller row is invalid")
        video_id = row["video_id"]
        if video_id in by_id:
            raise ValueError("normalized controller has duplicate video IDs")
        if not isinstance(row.get("derived_duration_seconds"), (int, float)):
            raise ValueError("normalized duration is missing")
        if not isinstance(row.get("derived_media_sha256"), str):
            raise ValueError("normalized media hash is missing")
        by_id[video_id] = dict(row)
    return by_id


def _post_json(url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    body = event_map.canonical_json(payload)
    http_request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(http_request, timeout=300.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("vLLM response must be an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failed_row(base: Mapping[str, Any], error_type: str, elapsed: float) -> dict[str, Any]:
    return {
        **dict(base),
        "status": "failed",
        "error_type": error_type,
        "elapsed_seconds": max(0.0, elapsed),
    }


def _failure_detail(kind: str, error: Exception) -> str:
    """Map parser exceptions to a bounded diagnostic taxonomy."""

    text = str(error)
    categories = (
        ("does not include the opening portion", "missing_opening_anchor"),
        ("does not include the ending portion", "missing_ending_anchor"),
        ("event description must be at most 48 words", "description_too_long"),
        ("timestamps must be strictly increasing", "timestamps_not_increasing"),
        ("timestamp is outside clip", "timestamp_out_of_range"),
        ("must end with ANSWER", "missing_answer"),
        ("1–8 event lines", "event_count_invalid"),
        ("unknown event", "unknown_event"),
        ("unique IDs", "duplicate_event"),
        ("SELECT and ANSWER", "selector_shape_invalid"),
    )
    for phrase, category in categories:
        if phrase in text:
            return f"{kind}_{category}"
    return f"{kind}_invalid"


def run_plaintext_canary(
    *,
    requests_path: Path,
    normalized_root: Path,
    normalized_controller_path: Path,
    output_path: Path,
    server_url: str,
    post: Post | None = None,
) -> list[dict[str, Any]]:
    """Run exactly four AV map calls and selectors for parsed maps."""

    rows = _load_canary_rows(requests_path)
    controller = _load_normalized_controller(normalized_controller_path)
    sender = post or _post_json
    results_by_id: dict[str, dict[str, Any]] = {}
    pending_selectors: list[dict[str, Any]] = []
    for row in rows:
        video_id = row["video_id"]
        normalized = controller.get(video_id)
        if normalized is None:
            raise ValueError("canary row is missing from normalized controller")
        media_path = normalized_root / f"{video_id}.mp4"
        if not media_path.is_file() or media_path.is_symlink():
            raise ValueError("normalized canary media is missing")
        duration = float(normalized["derived_duration_seconds"])
        map_request = event_map.build_plaintext_map_request(
            media_path=media_path.resolve(),
            question=row["question"],
            duration_seconds=duration,
            audio_enabled=True,
        )
        started = time.monotonic()
        base: dict[str, Any] = {
            "schema_version": 1,
            "video_id": video_id,
            "question": row["question"],
            "condition_id": "av-map",
            "source_media_sha256": row.get("media_sha256"),
            "normalized_media_sha256": normalized["derived_media_sha256"],
            "normalized_file_sha256": _file_sha256(media_path),
            "map_request_sha256": event_map.canonical_sha256(map_request),
        }
        try:
            map_response = sender(server_url, map_request)
            map_response_sha256 = event_map.canonical_sha256(map_response)
            parsed_map = event_map.parse_plaintext_map(
                map_response, duration_seconds=duration
            )
        except ValueError as error:
            failed = _failed_row(base, "map_parse_failed", time.monotonic() - started)
            failed["failure_detail"] = _failure_detail("map", error)
            results_by_id[video_id] = failed
            continue
        except Exception:
            failed = _failed_row(base, "map_request_failed", time.monotonic() - started)
            failed["failure_detail"] = "map_transport_or_response_invalid"
            results_by_id[video_id] = failed
            continue

        map_text_sha256 = event_map.canonical_sha256(parsed_map["map_text"])
        cells = event_map.derive_anchor_cells(parsed_map["events"], duration_seconds=duration)
        pending_selectors.append(
            {
                "row": row,
                "base": base,
                "started": started,
                "duration": duration,
                "parsed_map": parsed_map,
                "map_response_sha256": map_response_sha256,
                "map_text_sha256": map_text_sha256,
                "cells": cells,
            }
        )

    # Keep the map phase complete and immutable before sending any selector.
    for item in pending_selectors:
        row = item["row"]
        video_id = row["video_id"]
        selector_request = event_map.build_plaintext_selector_request(
            question=row["question"], frozen_map=item["parsed_map"]["map_text"]
        )
        selector_started = time.monotonic()
        selector_base = {
            **item["base"],
            "map_response_sha256": item["map_response_sha256"],
            "map_text_sha256": item["map_text_sha256"],
            "map_answer": item["parsed_map"]["answer"],
            "coverage_warnings": item["parsed_map"]["coverage_warnings"],
            "events": item["parsed_map"]["events"],
            "cells": item["cells"],
            "selector_request_sha256": event_map.canonical_sha256(selector_request),
            "selector_input_sha256": item["map_text_sha256"],
        }
        try:
            selector_response = sender(server_url, selector_request)
            selector_response_sha256 = event_map.canonical_sha256(selector_response)
            parsed_selector = event_map.parse_plaintext_selector(
                selector_response,
                event_ids={
                    event["event_id"] for event in item["parsed_map"]["events"]
                },
            )
            windows = event_map.expand_selected_anchors(
                item["parsed_map"]["events"],
                parsed_selector["selected_event_ids"],
                duration_seconds=item["duration"],
            )
        except ValueError as error:
            failed = _failed_row(
                selector_base,
                "selector_parse_failed",
                time.monotonic() - item["started"],
            )
            failed["failure_detail"] = _failure_detail("selector", error)
            results_by_id[video_id] = failed
            continue
        except Exception:
            failed = _failed_row(
                selector_base,
                "selector_request_failed",
                time.monotonic() - item["started"],
            )
            failed["failure_detail"] = "selector_transport_or_response_invalid"
            results_by_id[video_id] = failed
            continue
        results_by_id[video_id] = (
            {
                **selector_base,
                "selector_response_sha256": selector_response_sha256,
                "selected_event_ids": parsed_selector["selected_event_ids"],
                "selector_answer": parsed_selector["answer"],
                "selected_windows": windows,
                "status": "passed",
                "error_type": None,
                "map_elapsed_seconds": max(0.0, selector_started - item["started"]),
                "selector_elapsed_seconds": max(0.0, time.monotonic() - selector_started),
                "elapsed_seconds": max(0.0, time.monotonic() - item["started"]),
            }
        )

    results = [results_by_id[row["video_id"]] for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        "".join(
            json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
            for result in results
        ).encode("utf-8")
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--normalized-controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_plaintext_canary(
        requests_path=args.requests,
        normalized_root=args.normalized_root,
        normalized_controller_path=args.normalized_controller,
        output_path=args.output,
        server_url=args.server_url,
    )
    print(
        json.dumps(
            {
                "row_count": len(results),
                "passed": sum(row["status"] == "passed" for row in results),
                "failed": sum(row["status"] == "failed" for row in results),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
