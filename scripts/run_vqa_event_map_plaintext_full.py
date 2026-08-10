#!/usr/bin/env python3
"""Run the approved plain-text AV event-map panel over all 475 rows.

The run is deliberately two-phase and resumable:

1. complete and freeze all full-video timeline maps;
2. select one event from each frozen map with a text-only call.

Only hashes, parsed evidence, timestamps, and bounded failure categories are
written. Raw model completions are intentionally not persisted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib import request as urllib_request

from trec_video_2026.vqa_experiments import nemotron3_omni_event_map as event_map


Post = Callable[[str, Mapping[str, object]], Mapping[str, object]]
EXPECTED_COUNT = 475


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return event_map.canonical_sha256(value)


def _post_json(url: str, payload: Mapping[str, object]) -> Mapping[str, object]:
    body = event_map.canonical_json(payload)
    request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=600.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("vLLM response must be an object")
    return value


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return value


def _controller_rows(path: Path) -> list[dict[str, object]]:
    controller = _read_json(path)
    rows = controller.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise ValueError("controller must contain exactly 475 rows")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("controller row must be an object")
        video_id = raw.get("video_id")
        media_path = raw.get("media_path")
        question = raw.get("question")
        duration = raw.get("duration_seconds")
        media_sha256 = raw.get("media_sha256")
        if (
            not isinstance(video_id, str)
            or not video_id
            or video_id in seen
            or not isinstance(media_path, str)
            or not media_path.startswith("media/")
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(duration, (int, float))
            or duration <= 0
            or not isinstance(media_sha256, str)
        ):
            raise ValueError("controller row is malformed")
        seen.add(video_id)
        result.append(dict(raw))
    return sorted(result, key=lambda row: str(row["video_id"]))


def _probe_codecs(path: Path) -> tuple[bool, float]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    streams = payload.get("streams", [])
    codecs = {
        (stream.get("codec_type"), stream.get("codec_name"))
        for stream in streams
        if isinstance(stream, Mapping)
    }
    duration = float(payload["format"]["duration"])
    return ("video", "av1") in codecs, duration


def _normalize_one(
    row: Mapping[str, object], *, source_root: Path, normalized_root: Path
) -> dict[str, object]:
    video_id = str(row["video_id"])
    source = source_root / str(row["media_path"])
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"source media is missing: {video_id}")
    source_sha256 = _sha256_file(source)
    if source_sha256 != row["media_sha256"]:
        raise ValueError(f"source media hash mismatch: {video_id}")
    is_av1, source_duration = _probe_codecs(source)
    if is_av1:
        derived = normalized_root / f"{video_id}.mp4"
        derived.parent.mkdir(parents=True, exist_ok=True)
        if not derived.exists():
            temporary = derived.with_name(f".{video_id}.tmp.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
                    "-i", str(source), "-map", "0:v:0", "-map", "0:a:0", "-map_metadata", "-1",
                    "-map_chapters", "-1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-threads", "1", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-ac", "2", "-movflags", "+faststart", "-shortest", str(temporary),
                ],
                check=True,
            )
            os.replace(temporary, derived)
        event_map.validate_normalized_media(
            source_path=source,
            derived_path=derived,
            expected_source_sha256=source_sha256,
        )
        normalized_path = derived
        normalized_sha256 = _sha256_file(derived)
        normalized_duration = event_map.probe_media(derived)["format"]["duration"]
        normalized_duration = float(normalized_duration)
    else:
        # The model accepts these original containers; retaining them avoids a
        # lossy/re-encoded copy and binds the artifact to the source hash.
        normalized_path = source
        normalized_sha256 = source_sha256
        normalized_duration = source_duration
    return {
        "video_id": video_id,
        "source_media_path": str(source),
        "normalized_media_path": str(normalized_path),
        "source_media_sha256": source_sha256,
        "normalized_media_sha256": normalized_sha256,
        "source_duration_seconds": source_duration,
        "normalized_duration_seconds": normalized_duration,
        "source_codec_normalized": is_av1,
    }


def prepare_normalized_controller(
    *, controller_path: Path, source_root: Path, normalized_root: Path, output_path: Path
) -> list[dict[str, object]]:
    rows = _controller_rows(controller_path)
    def normalize(row: Mapping[str, object]) -> dict[str, object]:
        return _normalize_one(row, source_root=source_root, normalized_root=normalized_root)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        normalized = list(executor.map(normalize, rows))
    payload = {
        "schema_version": 1,
        "source_controller_sha256": _sha256_file(controller_path),
        "rows": normalized,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return normalized


def _normalized_rows(path: Path) -> dict[str, dict[str, object]]:
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise ValueError("normalized controller must contain exactly 475 rows")
    result: dict[str, dict[str, object]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("video_id"), str):
            raise ValueError("normalized controller row is malformed")
        video_id = str(raw["video_id"])
        if video_id in result:
            raise ValueError("normalized controller contains duplicate IDs")
        media_path = raw.get("normalized_media_path")
        if not isinstance(media_path, str) or not Path(media_path).is_file():
            raise ValueError(f"normalized media is missing: {video_id}")
        result[video_id] = dict(raw)
    return result


def _load_jsonl(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("video_id"), str):
                raise ValueError(f"invalid checkpoint row in {path}")
            result[str(row["video_id"])] = row
    return result


def _append(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _failure_detail(kind: str, error: Exception) -> str:
    text = str(error)
    categories = (
        ("opening portion", "missing_opening_anchor"),
        ("ending portion", "missing_ending_anchor"),
        ("at most 48 words", "description_too_long"),
        ("strictly increasing", "timestamps_not_increasing"),
        ("outside clip", "timestamp_out_of_range"),
        ("1–8 event", "event_count_invalid"),
        ("unknown event", "unknown_event"),
        ("unique IDs", "duplicate_event"),
        ("SELECT and ANSWER", "selector_shape_invalid"),
    )
    for phrase, category in categories:
        if phrase in text:
            return f"{kind}_{category}"
    return f"{kind}_invalid"


def _base_row(
    panel: Mapping[str, object], normalized: Mapping[str, object], *, map_request: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "video_id": panel["video_id"],
        "question": panel["question"],
        "question_stratum": panel.get("question_stratum"),
        "condition_id": "av-map",
        "controller_row_sha256": _canonical_sha256(panel),
        "source_media_sha256": normalized["source_media_sha256"],
        "normalized_media_sha256": normalized["normalized_media_sha256"],
        "normalized_file_sha256": _sha256_file(Path(str(normalized["normalized_media_path"]))),
        "source_duration_seconds": normalized["source_duration_seconds"],
        "normalized_duration_seconds": normalized["normalized_duration_seconds"],
        "normalized_media_path": normalized["normalized_media_path"],
        "map_request_sha256": _canonical_sha256(map_request),
    }


def _run_maps(
    *, panels: Sequence[Mapping[str, object]], normalized: Mapping[str, Mapping[str, object]],
    output_path: Path, server_url: str, sender: Post,
) -> dict[str, dict[str, object]]:
    existing = _load_jsonl(output_path)
    for index, panel in enumerate(panels, start=1):
        video_id = str(panel["video_id"])
        if video_id in existing:
            continue
        norm = normalized[video_id]
        media_path = Path(str(norm["normalized_media_path"]))
        duration = float(norm["normalized_duration_seconds"])
        request = event_map.build_plaintext_map_request(
            media_path=media_path,
            question=str(panel["question"]),
            duration_seconds=duration,
            audio_enabled=True,
        )
        base = _base_row(panel, norm, map_request=request)
        started = time.monotonic()
        try:
            response = sender(server_url, request)
            parse_mode = "strict"
            try:
                parsed = event_map.parse_plaintext_map(response, duration_seconds=duration)
            except ValueError:
                parsed = event_map.salvage_plaintext_map(response, duration_seconds=duration)
                parse_mode = "salvaged"
            events = parsed["events"]
            cells = event_map.derive_anchor_cells(events, duration_seconds=duration)
            row = {
                **base,
                "status": "passed",
                "error_type": None,
                "map_response_sha256": _canonical_sha256(response),
                "map_text_sha256": _canonical_sha256(parsed["map_text"]),
                "map_text": parsed["map_text"],
                "map_answer": parsed["answer"],
                "coverage_warnings": parsed["coverage_warnings"],
                "map_parse_mode": parse_mode,
                "map_parse_warnings": parsed.get("parse_warnings", []),
                "events": events,
                "cells": cells,
                "map_elapsed_seconds": time.monotonic() - started,
            }
        except ValueError as error:
            row = {
                **base,
                "status": "failed",
                "error_type": "map_parse_failed",
                "failure_detail": _failure_detail("map", error),
                "map_elapsed_seconds": time.monotonic() - started,
            }
        except Exception:
            row = {
                **base,
                "status": "failed",
                "error_type": "map_request_failed",
                "failure_detail": "map_transport_or_response_invalid",
                "map_elapsed_seconds": time.monotonic() - started,
            }
        _append(output_path, row)
        existing[video_id] = row
        if index % 25 == 0 or index == len(panels):
            print(json.dumps({"phase": "maps", "completed": len(existing), "passed": sum(r.get("status") == "passed" for r in existing.values())}), flush=True)
    if len(existing) != len(panels):
        raise RuntimeError("map phase did not produce one row per panel video")
    return existing


def _run_selectors(
    *, panels: Sequence[Mapping[str, object]], maps: Mapping[str, Mapping[str, object]],
    normalized: Mapping[str, Mapping[str, object]], output_path: Path, checkpoint_path: Path,
    server_url: str, sender: Post, checkpoint_count: int,
) -> dict[str, dict[str, object]]:
    existing = _load_jsonl(output_path)
    for index, panel in enumerate(panels, start=1):
        video_id = str(panel["video_id"])
        if video_id in existing:
            continue
        map_row = maps[video_id]
        started = time.monotonic()
        if map_row.get("status") != "passed":
            row = {**map_row, "phase": "selector", "selector_status": "skipped_map_failure"}
            _append(output_path, row)
            existing[video_id] = row
        else:
            norm = normalized[video_id]
            request = event_map.build_plaintext_selector_request(
                question=str(panel["question"]), frozen_map=str(map_row["map_text"])
                if "map_text" in map_row else _rebuild_map_text(map_row),
            )
            # map_text is not persisted as raw completion; reconstruct the exact
            # selector input from the parsed events and the fixed line format.
            base = {
                **dict(map_row),
                "selector_request_sha256": _canonical_sha256(request),
                "selector_input_sha256": map_row["map_text_sha256"],
                "normalized_media_path": norm["normalized_media_path"],
            }
            try:
                response = sender(server_url, request)
                parsed = event_map.parse_plaintext_selector(
                    response,
                    event_ids={str(event["event_id"]) for event in map_row["events"]},
                )
                windows = event_map.expand_selected_anchors(
                    map_row["events"], parsed["selected_event_ids"],
                    duration_seconds=float(norm["normalized_duration_seconds"]),
                )
                row = {
                    **base,
                    "status": "passed",
                    "error_type": None,
                    "selector_response_sha256": _canonical_sha256(response),
                    "selected_event_ids": parsed["selected_event_ids"],
                    "selector_answer": parsed["answer"],
                    "selected_windows": windows,
                    "selector_elapsed_seconds": time.monotonic() - started,
                }
            except ValueError as error:
                row = {
                    **base,
                    "status": "failed",
                    "error_type": "selector_parse_failed",
                    "failure_detail": _failure_detail("selector", error),
                    "selector_elapsed_seconds": time.monotonic() - started,
                }
            except Exception:
                row = {
                    **base,
                    "status": "failed",
                    "error_type": "selector_request_failed",
                    "failure_detail": "selector_transport_or_response_invalid",
                    "selector_elapsed_seconds": time.monotonic() - started,
                }
            _append(output_path, row)
            existing[video_id] = row
        completed = index
        passed = sum(row.get("status") == "passed" for row in existing.values())
        if completed == checkpoint_count:
            checkpoint = {
                "schema_version": 1,
                "checkpoint": checkpoint_count,
                "completed": completed,
                "passed": passed,
                "required_passed": checkpoint_count - 1,
                "status": "go" if passed >= checkpoint_count - 1 else "stop",
            }
            checkpoint_path.write_text(json.dumps(checkpoint, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"phase": "checkpoint", **checkpoint}), flush=True)
            if passed < checkpoint_count - 1:
                raise SystemExit(3)
        if index % 25 == 0 or index == len(panels):
            print(json.dumps({"phase": "selectors", "completed": completed, "passed": passed}), flush=True)
    return existing


def _rebuild_map_text(row: Mapping[str, object]) -> str:
    lines = [
        f"{event['event_id']}|{float(event['timestamp']):.3f}|{event['description']}"
        for event in row["events"]
    ]
    answer = row.get("map_answer")
    if isinstance(answer, str) and answer:
        lines.append(f"ANSWER|{answer}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--normalized-controller", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--checkpoint-count", type=int, default=32)
    args = parser.parse_args()
    panels = _controller_rows(args.controller)
    normalized = _normalized_rows(args.normalized_controller)
    if set(normalized) != {str(row["video_id"]) for row in panels}:
        raise ValueError("normalized controller IDs do not match panel")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps = _run_maps(
        panels=panels, normalized=normalized, output_path=args.output_dir / "maps.jsonl",
        server_url=args.server_url, sender=_post_json,
    )
    results = _run_selectors(
        panels=panels, maps=maps, normalized=normalized,
        output_path=args.output_dir / "results.jsonl",
        checkpoint_path=args.output_dir / "checkpoint.json",
        server_url=args.server_url, sender=_post_json,
        checkpoint_count=args.checkpoint_count,
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "panel_count": len(panels),
        "map_passed": sum(row.get("status") == "passed" for row in maps.values()),
        "selector_passed": sum(row.get("status") == "passed" for row in results.values()),
        "maps_sha256": _sha256_file(args.output_dir / "maps.jsonl"),
        "results_sha256": _sha256_file(args.output_dir / "results.jsonl"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "terminal-status.json").write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
