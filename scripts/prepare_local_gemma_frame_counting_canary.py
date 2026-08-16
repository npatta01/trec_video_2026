#!/usr/bin/env python3
"""Freeze the label-blind local Gemma structured-frame canary inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from trec_video_2026.vqa_blind_frame_counting import (
    VisibleEpisodeCountSpec,
    build_frame_decision_request,
    frame_decision_request_to_dict,
    parse_visible_episode_count_spec,
    visible_episode_count_spec_to_dict,
)
from trec_video_2026.vqa_blind_frame_model import (
    build_contact_sheet_state_request,
    build_individual_state_request,
)
from trec_video_2026.vqa_submission import canonical_sha256

ROWS = (
    {
        "video_id": "RZrkA52TGBA",
        "question": "How many times balls appear in the video?",
        "target": "balls",
        "decoded_source_frame_count": 217,
    },
    {
        "video_id": "PgKOzRgYMAI",
        "question": "how many times we see a colorful screen with lines in the video?",
        "target": "a colorful screen with lines",
        "decoded_source_frame_count": 448,
    },
)
MODEL_ID = "google/gemma-3n-E4B-it"
MODEL_REVISION = "c1221e9c62e34a43ab7ffacd1be0ea71f126ef10"
RUNTIME_IMAGE = (
    "localhost/trec-vqa-omni-rocm@sha256:"
    "b1f97c47769a8a2f32a5ac9a18cc62d2aab57ed32c66720775407f6213ebf96f"
)
BATCH_SIZE = 11
MEDIA_EXTENSION = ".png"
IMPLEMENTATION_PATHS = (
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
)


def selected_source_indices(decoded_frame_count: int) -> tuple[int, ...]:
    if decoded_frame_count < 1:
        raise ValueError("decoded_frame_count must be positive")
    selected = list(range(0, decoded_frame_count, 4))
    if selected[-1] != decoded_frame_count - 1:
        selected.append(decoded_frame_count - 1)
    return tuple(selected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, payload: object) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing conflicting freeze: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _self_hashed(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def _decoded_pts(source: Path) -> tuple[float, ...]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    return tuple(
        float(frame["best_effort_timestamp_time"]) for frame in payload["frames"]
    )


def _contact_sheet(
    *, output: Path, frames: tuple[tuple[str, float, Path], ...]
) -> None:
    if output.exists():
        return
    tile_width, image_height, label_height = 320, 180, 30
    columns = 4
    rows = (len(frames) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * 210), "white")
    draw = ImageDraw.Draw(canvas)
    for position, (frame_id, timestamp, path) in enumerate(frames):
        with Image.open(path) as image:
            tile = image.convert("RGB")
            tile.thumbnail((tile_width, image_height))
            x = (position % columns) * tile_width
            y = (position // columns) * (image_height + label_height)
            canvas.paste(tile, (x + (tile_width - tile.width) // 2, y))
            draw.text(
                (x + 8, y + image_height + 6),
                f"{frame_id}  {timestamp:.3f}s",
                fill="black",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", compress_level=6)


def _prepare_row(*, data_root: Path, run_root: Path, row: dict[str, object]):
    video_id = str(row["video_id"])
    question = str(row["question"])
    target = str(row["target"])
    source = data_root / "vqa/2025/training/videos_original" / f"{video_id}.mkv"
    source_indices = selected_source_indices(int(row["decoded_source_frame_count"]))
    all_pts = _decoded_pts(source)
    if len(all_pts) != int(row["decoded_source_frame_count"]):
        raise ValueError(f"decoded source frame count changed for {video_id}")
    pts = tuple(all_pts[index] for index in source_indices)
    maximum_gap = max(b - a for a, b in zip(pts, pts[1:], strict=False))
    frame_dir = run_root / "media" / video_id / "frames"
    frame_paths = tuple(sorted(frame_dir.glob(f"*{MEDIA_EXTENSION}")))
    if len(frame_paths) != len(source_indices):
        raise ValueError(f"selected frame count changed for {video_id}")
    frame_ids = tuple(f"F{index}" for index in range(len(frame_paths)))
    frame_records = tuple(zip(frame_ids, pts, frame_paths, strict=True))
    sheet_records = []
    for start in range(0, len(frame_records), BATCH_SIZE):
        batch = frame_records[start : start + BATCH_SIZE]
        sheet_path = (
            run_root
            / "media"
            / video_id
            / "contact-sheets"
            / f"sheet-{start // BATCH_SIZE:02d}{MEDIA_EXTENSION}"
        )
        _contact_sheet(output=sheet_path, frames=batch)
        sheet_records.append((sheet_path, batch))

    decoder_text = "ffmpeg 8.0.1; select=not(mod(n,4))+eq(n,last); fps_mode=vfr; png"
    decoder_path = run_root / "preparation" / "decoder-configuration.txt"
    if not decoder_path.exists():
        decoder_path.write_text(decoder_text + "\n", encoding="utf-8")
    source_sha = _sha256(source)
    spec_payload = {
        "schema_version": 1,
        "question": question,
        "unit": "visible_appearance_episode",
        "scope": "full_video",
        "episode_rule": "maximal contiguous run of present samples",
        "maximum_sample_gap_seconds": maximum_gap,
        "every_decoded_frame": True,
        "cut_policy": "continuous_across_cuts",
        "occluded_identity": "unsupported",
    }
    spec = parse_visible_episode_count_spec(spec_payload)
    if not isinstance(spec, VisibleEpisodeCountSpec) or spec.target != target:
        raise ValueError(f"question router changed for {video_id}")
    inventory = {
        "schema_version": 1,
        "video_id": f"V{source_sha[:16]}",
        "source_sha256": source_sha,
        "source_size_bytes": source.stat().st_size,
        "decoder": {
            "name": "ffmpeg",
            "version": "8.0.1",
            "configuration_sha256": _sha256(decoder_path),
            "configuration_size_bytes": decoder_path.stat().st_size,
        },
        "duration_seconds": pts[-1],
        "decoded_frame_count": len(frame_paths),
        "frames": [
            {
                "frame_id": frame_id,
                "decoded_frame_index": position,
                "timestamp_seconds": timestamp,
                "artifact_sha256": _sha256(path),
                "artifact_size_bytes": path.stat().st_size,
                "is_first_decoded_frame": position == 0,
                "is_last_decoded_frame": position == len(frame_paths) - 1,
            }
            for position, (frame_id, timestamp, path) in enumerate(frame_records)
        ],
    }
    presentations = [
        {
            "presentation_id": f"P{_sha256(path)}",
            "presentation_sha256": _sha256(path),
            "artifact_size_bytes": path.stat().st_size,
            "frame_ids": [frame_id for frame_id, _timestamp, _path in batch],
            "mode": "numbered_contact_sheet",
        }
        for path, batch in sheet_records
    ]
    presentations.extend(
        {
            "presentation_id": f"P{_sha256(path)}",
            "presentation_sha256": _sha256(path),
            "artifact_size_bytes": path.stat().st_size,
            "frame_ids": [frame_id],
            "mode": "individual_frame",
        }
        for frame_id, _timestamp, path in frame_records
    )
    protocol_request = build_frame_decision_request(spec, inventory, presentations)
    primary = build_contact_sheet_state_request(
        target=target,
        sheets=tuple(
            (
                path,
                tuple(frame_id for frame_id, _timestamp, _path in batch),
                tuple(timestamp for _frame_id, timestamp, _path in batch),
            )
            for path, batch in sheet_records
        ),
    )
    challenges = [
        build_individual_state_request(
            target=target,
            frame_ids=tuple(frame_id for frame_id, _timestamp, _path in batch),
            frame_paths=tuple(path for _frame_id, _timestamp, path in batch),
            timestamps_seconds=tuple(
                timestamp for _frame_id, timestamp, _path in batch
            ),
        )
        for start in range(0, len(frame_records), BATCH_SIZE)
        for batch in (frame_records[start : start + BATCH_SIZE],)
    ]
    return {
        "video_id": video_id,
        "question": question,
        "target": target,
        "source_indices": list(source_indices),
        "spec": visible_episode_count_spec_to_dict(spec),
        "protocol_request": frame_decision_request_to_dict(protocol_request),
        "primary_model_request": primary,
        "challenge_model_requests": challenges,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        _prepare_row(data_root=args.data_root, run_root=args.run_root, row=row)
        for row in ROWS
    ]
    request = _self_hashed(
        {
            "schema_version": 1,
            "condition": "local-gemma-3n-structured-frame-v1",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_image": RUNTIME_IMAGE,
            "batch_size": BATCH_SIZE,
            "rows": rows,
        }
    )
    _write_once(args.run_root / "preparation" / "requests.label-blind.json", request)
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    provenance = _self_hashed(
        {
            "schema_version": 1,
            "condition": "local-gemma-3n-structured-frame-v1",
            "request_complete_sha256": request["complete_sha256"],
            "source_revision": revision,
            "implementation_sources": {
                path: _sha256(repository / path) for path in IMPLEMENTATION_PATHS
            },
            "labels_available": False,
            "provider_calls": 0,
            "provider_cost_usd": 0.0,
        }
    )
    _write_once(
        args.run_root / "preparation" / "preparation-provenance.json", provenance
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "row_count": len(rows),
                "complete_sha256": request["complete_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
