#!/usr/bin/env python3
"""Build the nine label-free VQA silver-oracle evidence request cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_experiments.silver_oracle import (
    ANSWER_REQUEST_FIELDS,
    EVIDENCE_CELLS,
    FRAME_VIEWS,
    TRANSCRIPT_VIEWS,
    assert_global_plan_reuse,
    audit_label_free_request,
    audit_request_matrix,
    build_frame_plans,
    build_request_matrix,
    build_transcript_views,
    model_input_sha256,
    request_sha256,
    validate_selector_result,
)

RUN_ID = "vqa-silver-oracle-election-20260809-a1"
EXPECTED_COUNT = 475
CANONICAL_PANEL_ID_DIGEST = (
    "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
)
EXTRACTION_COMMAND_IDENTITY = "build_vqa_silver_evidence.py:verified-decoded-frame"
_SHA256 = re.compile(r"[0-9a-f]{64}")

ContactSheetBuilder = Callable[..., dict[str, object]]
FrameExtractor = Callable[
    [Mapping[str, Any], Mapping[str, Any], str, Path], Mapping[str, Any]
]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error


def _rows_from_payload(
    payload: Any, names: Sequence[str], *, source: str
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = None
        for name in names:
            candidate = payload.get(name)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise ValueError(f"{source} must contain one of {', '.join(names)}")
    else:
        raise ValueError(f"{source} must be a JSON list or object")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{source} rows must be objects")
    return [dict(row) for row in rows]


def _indexed(
    rows: Sequence[Mapping[str, Any]], *, source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{source} row {index} has an invalid video_id")
        if video_id in indexed:
            raise ValueError(f"{source} contains duplicate video_id {video_id}")
        indexed[video_id] = dict(row)
    return indexed


def _canonical_id_digest(video_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(video_ids)) + "\n").encode("utf-8")
    ).hexdigest()


def _validate_frozen_identities(
    *,
    rows: Sequence[Mapping[str, Any]],
    selectors: Sequence[Mapping[str, Any]],
    global_evidence: Sequence[Mapping[str, Any]],
    whisper: Sequence[Mapping[str, Any]],
    baseline_requests: Mapping[str, Mapping[str, Any]],
    expected_count: int,
    expected_panel_id_digest: str,
    expected_selector_protocol_sha256: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Reject stale panel/protocol inputs before media or frames are opened."""

    if not _SHA256.fullmatch(expected_panel_id_digest):
        raise ValueError("expected panel ID digest must be lowercase SHA-256")
    if expected_count == EXPECTED_COUNT and (
        expected_panel_id_digest != CANONICAL_PANEL_ID_DIGEST
    ):
        raise ValueError("the 475-row panel must use the frozen canonical ID digest")
    if not _SHA256.fullmatch(expected_selector_protocol_sha256):
        raise ValueError("expected selector protocol must be lowercase SHA-256")

    panel_by_id = _indexed(rows, source="panel")
    panel_ids = list(panel_by_id)
    if panel_ids != sorted(panel_ids):
        raise ValueError("label-free panel IDs must be in canonical sorted order")
    if _canonical_id_digest(panel_ids) != expected_panel_id_digest:
        raise ValueError("canonical panel ID digest mismatch")

    selector_by_id = _indexed(selectors, source="selector")
    evidence_by_id = _indexed(global_evidence, source="global evidence")
    whisper_by_id = _indexed(whisper, source="Whisper")
    expected_ids = set(panel_by_id)
    if set(selector_by_id) != expected_ids:
        raise ValueError("selector IDs do not exactly match the label-free panel")
    if set(evidence_by_id) != expected_ids:
        raise ValueError("global evidence IDs do not exactly match the panel")
    if set(whisper_by_id) != expected_ids:
        raise ValueError("Whisper IDs do not exactly match the panel")
    if set(baseline_requests) != expected_ids:
        raise ValueError("G/F baseline request IDs do not exactly match the panel")
    for video_id, selector_row in selector_by_id.items():
        if selector_row.get("protocol_sha256") != expected_selector_protocol_sha256:
            raise ValueError(f"{video_id} selector protocol identity mismatch")
    return panel_by_id, selector_by_id, evidence_by_id, whisper_by_id


def _validate_panel_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one label-free row without opening any annotation source."""

    audit_label_free_request(row, location="panel row")
    video_id = row.get("video_id")
    question = row.get("question")
    duration = row.get("duration_seconds")
    media_path = row.get("media_path")
    media_sha256 = row.get("media_sha256")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("panel row video_id must be nonempty text")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{video_id} question must be nonempty text")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration <= 0
    ):
        raise ValueError(f"{video_id} duration_seconds must be positive")
    if (
        not isinstance(row.get("whisper_segment_count"), int)
        or isinstance(row.get("whisper_segment_count"), bool)
        or row["whisper_segment_count"] < 0
    ):
        raise ValueError(f"{video_id} whisper_segment_count must be non-negative")
    if not isinstance(media_path, str) or not media_path:
        raise ValueError(f"{video_id} media_path must be nonempty text")
    if (
        not isinstance(media_sha256, str)
        or len(media_sha256) != 64
        or any(character not in "0123456789abcdef" for character in media_sha256)
    ):
        raise ValueError(f"{video_id} media_sha256 must be lowercase SHA-256")
    resolved_media = Path(media_path).resolve(strict=True)
    if resolved_media.is_symlink() or not resolved_media.is_file():
        raise ValueError(f"{video_id} media must be a regular non-symlink file")
    if _sha256_file(resolved_media) != media_sha256:
        raise ValueError(f"{video_id} source media hash mismatch")
    return {
        **dict(row),
        "video_id": video_id,
        "question": question,
        "duration_seconds": float(duration),
        "media_path": str(resolved_media),
        "media_sha256": media_sha256,
    }


def _validated_selector(
    row: Mapping[str, Any], selector_row: Mapping[str, Any]
) -> dict[str, Any]:
    if "video_id" not in selector_row or "selector" not in selector_row:
        raise ValueError("selector row must contain video_id and selector")
    if selector_row.get("video_id") != row["video_id"]:
        raise ValueError(f"{row['video_id']} selector video_id mismatch")
    # Task 3 result rows carry runtime metadata around the validated terminal
    # object. Do not let that metadata become answer evidence, and fail closed
    # if an unexpected label-bearing field is present.
    for key in selector_row:
        if key in {"video_id", "selector"}:
            continue
        normalized = str(key).casefold()
        if any(
            token in normalized
            for token in (
                "answer",
                "reference",
                "prediction",
                "rationale",
                "confidence",
            )
        ):
            raise ValueError(f"selector row contains forbidden field {key}")
    selector = selector_row.get("selector")
    if not isinstance(selector, Mapping):
        raise TypeError(f"{row['video_id']} selector must be an object")
    return validate_selector_result(selector, row)


def _frame_path(frame: Mapping[str, Any]) -> str | None:
    for key in ("frame_path", "path", "image_path"):
        value = frame.get(key)
        if isinstance(value, str) and value:
            return str(Path(value).resolve(strict=True))
    return None


def _decoded_identity(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Validate identity reported by decoding, without planned-field fallback."""

    missing = [
        field
        for field in (
            "stream_index",
            "frame_index",
            "time_base",
            "actual_pts_seconds",
        )
        if field not in payload
    ]
    if missing:
        raise ValueError(f"{source} lacks actual decoded identity field {missing[0]}")
    stream_index = payload["stream_index"]
    frame_index = payload["frame_index"]
    if (
        isinstance(stream_index, bool)
        or not isinstance(stream_index, int)
        or stream_index < 0
    ):
        raise ValueError(f"{source} has invalid actual decoded stream_index")
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
    ):
        raise ValueError(f"{source} has invalid actual decoded frame_index")
    ordinal = payload.get("decoded_frame_ordinal", frame_index)
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 0
        or ordinal != frame_index
    ):
        raise ValueError(
            f"{source} decoded_frame_ordinal must equal actual frame_index"
        )
    time_base = payload["time_base"]
    if not isinstance(time_base, str) or not time_base.strip():
        raise ValueError(f"{source} has invalid actual decoded time_base")
    try:
        parsed_time_base = Fraction(time_base)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{source} has invalid actual decoded time_base") from error
    if parsed_time_base <= 0:
        raise ValueError(f"{source} has invalid actual decoded time_base")
    raw_pts = payload.get("raw_pts")
    if raw_pts is not None and (
        isinstance(raw_pts, bool) or not isinstance(raw_pts, int)
    ):
        raise ValueError(f"{source} raw_pts must be an integer or null")
    actual_pts_seconds = payload["actual_pts_seconds"]
    if (
        isinstance(actual_pts_seconds, bool)
        or not isinstance(actual_pts_seconds, (int, float))
        or not math.isfinite(float(actual_pts_seconds))
        or float(actual_pts_seconds) < 0
    ):
        raise ValueError(f"{source} has invalid actual decoded PTS")
    return {
        "stream_index": stream_index,
        "frame_index": frame_index,
        "decoded_frame_ordinal": ordinal,
        "time_base": f"{parsed_time_base.numerator}/{parsed_time_base.denominator}",
        "raw_pts": raw_pts,
        "actual_pts_seconds": round(float(actual_pts_seconds), 6),
    }


def _exact_time_base(value: Any) -> tuple[str, Fraction]:
    """Return a lossless rational time-base identity from a PyAV value."""

    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            time_base = Fraction(value.numerator, value.denominator)
        else:
            time_base = Fraction(str(value))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("decoded frame has no exact time_base") from error
    if time_base <= 0:
        raise ValueError("decoded frame has no exact time_base")
    return f"{time_base.numerator}/{time_base.denominator}", time_base


def _materialize_frame(
    row: Mapping[str, Any],
    frame: Mapping[str, Any],
    view: str,
    output_dir: Path,
    *,
    frame_extractor: FrameExtractor | None,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    planned = dict(frame)
    source_path = _frame_path(frame)
    materialization_mode = "decoded"
    if source_path is None:
        extractor = frame_extractor or extract_frame
        target = (
            output_dir
            / row["video_id"]
            / view
            / f"frame-{frame['frame_index']:03d}.png"
        )
        extracted = extractor(row, frame, view, target)
        if not isinstance(extracted, Mapping):
            raise TypeError("frame extractor must return an object")
        identity = _decoded_identity(
            extracted, source="frame extractor actual decoded identity"
        )
        path = _frame_path(extracted)
        if path is None:
            raise ValueError("frame extractor did not return a frame path")
        expected_image_sha256 = extracted.get("image_sha256", extracted.get("sha256"))
        if not isinstance(expected_image_sha256, str) or not _SHA256.fullmatch(
            expected_image_sha256
        ):
            raise ValueError("frame extractor did not return an image SHA-256")
    else:
        materialization_mode = "verified-copy"
        verified_identity = frame.get("verified_decoded_identity")
        if not isinstance(verified_identity, Mapping):
            raise ValueError(
                f"{row['video_id']} {view} frame lacks verified decoded identity"
            )
        identity = _decoded_identity(
            verified_identity, source="verified decoded identity"
        )
        expected_image_sha256 = frame.get("image_sha256", frame.get("sha256"))
        if not isinstance(expected_image_sha256, str) or not _SHA256.fullmatch(
            expected_image_sha256
        ):
            raise ValueError(
                f"{row['video_id']} {view} verified frame lacks image SHA-256"
            )
        source = Path(source_path)
        suffix = source.suffix or ".img"
        target = (
            output_dir
            / row["video_id"]
            / view
            / f"frame-{frame['frame_index']:03d}{suffix}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        path = _frame_path({"frame_path": str(target)})
        if path is None:
            raise AssertionError("copied frame path was not materialized")
    image_path = Path(path)
    if image_path.is_symlink() or not image_path.is_file():
        raise ValueError(f"{row['video_id']} {view} frame is not a regular file")
    if not image_path.is_relative_to(output_dir.resolve()):
        raise ValueError(f"{row['video_id']} {view} frame escapes evidence output")
    image_sha256 = _sha256_file(image_path)
    if image_sha256 != expected_image_sha256:
        raise ValueError(f"{row['video_id']} {view} image hash mismatch")
    source_media_sha256 = planned.get("source_media_sha256")
    if source_media_sha256 != row["media_sha256"]:
        raise ValueError(f"{row['video_id']} {view} source media identity mismatch")
    requested = float(planned["requested_time_seconds"])
    return {
        "frame_path": str(image_path),
        "requested_timestamp_seconds": requested,
        "requested_time_seconds": requested,
        **identity,
        "source_media_sha256": source_media_sha256,
        "image_sha256": image_sha256,
        "sha256": image_sha256,
        "extraction_command_identity": EXTRACTION_COMMAND_IDENTITY,
        "materialization_mode": materialization_mode,
        "runtime_identity": dict(runtime_identity),
        "frame_plan_sha256": planned.get("frame_plan_sha256"),
        "slot_kind": planned["slot_kind"],
        "desired_time_seconds": planned["desired_time_seconds"],
        "construction_fallback": planned.get("construction_fallback"),
    }


def extract_frame(
    row: Mapping[str, Any],
    frame: Mapping[str, Any],
    view: str,
    output_path: Path,
) -> Mapping[str, Any]:
    """Extract the nearest decoded frame when verified evidence has no image path."""

    del view
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            "PyAV is required to extract missing evidence frames"
        ) from error
    media_path = Path(str(row["media_path"])).resolve(strict=True)
    target_seconds = float(frame["actual_pts_seconds"])
    best_image = None
    best_time: float | None = None
    best_identity: dict[str, Any] | None = None
    with av.open(str(media_path)) as container:
        stream = next(stream for stream in container.streams if stream.type == "video")
        for ordinal, decoded in enumerate(container.decode(stream)):
            raw_pts = getattr(decoded, "pts", None)
            time_base_value = getattr(decoded, "time_base", None)
            if time_base_value is None:
                time_base_value = getattr(stream, "time_base", None)
            try:
                time_base, time_base_fraction = _exact_time_base(time_base_value)
            except ValueError:
                continue
            if raw_pts is not None:
                if isinstance(raw_pts, bool) or not isinstance(raw_pts, int):
                    continue
                actual_time = float(raw_pts * time_base_fraction)
            else:
                decoded_time = getattr(decoded, "time", None)
                if decoded_time is None:
                    continue
                actual_time = float(decoded_time)
            if not math.isfinite(actual_time) or actual_time < 0:
                continue
            if best_time is None or abs(actual_time - target_seconds) < abs(
                best_time - target_seconds
            ):
                best_image = decoded.to_image().convert("RGB")
                best_time = actual_time
                best_identity = {
                    "stream_index": stream.index,
                    "frame_index": ordinal,
                    "decoded_frame_ordinal": ordinal,
                    "time_base": time_base,
                    "raw_pts": raw_pts,
                    "actual_pts_seconds": round(actual_time, 6),
                }
    if best_image is None or best_identity is None:
        raise RuntimeError(
            f"could not decode a frame near {target_seconds}s from {media_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_image.save(output_path, format="PNG", optimize=True)
    return {
        "frame_path": str(output_path.resolve()),
        **best_identity,
        "image_sha256": _sha256_file(output_path),
    }


def make_contact_sheet(
    *, video_id: str, view: str, frame_paths: list[str], output_path: Path
) -> dict[str, object]:
    """Render one twelve-frame audit sheet for one row/view."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("Pillow is required to render contact sheets") from error
    columns, rows = 4, 3
    cell_width, cell_height = 320, 250
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(frame_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_width - 8, 220), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(
            image,
            (x + (cell_width - image.width) // 2, y + 28 + (220 - image.height) // 2),
        )
        draw.text((x + 5, y + 5), f"{video_id} {view} {index + 1:02d}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)
    return {
        "path": str(output_path.resolve()),
        "sha256": _sha256_file(output_path),
        "frame_count": len(frame_paths),
    }


def _validated_baseline_request(
    baseline_requests: Mapping[str, Mapping[str, Any]], video_id: str
) -> Mapping[str, Any]:
    request = baseline_requests.get(video_id)
    if not isinstance(request, Mapping):
        raise ValueError(f"{video_id} is missing its verified G/F baseline request")
    audit_label_free_request(request, location=f"baseline[{video_id}]")
    if set(request) != ANSWER_REQUEST_FIELDS:
        raise ValueError(f"{video_id} G/F baseline request schema is not exact")
    return request


def _validated_contact_sheet(
    contact_sheet: Mapping[str, Any], *, output_dir: Path, video_id: str, view: str
) -> dict[str, Any]:
    path = contact_sheet.get("path")
    digest = contact_sheet.get("sha256")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{video_id} {view} contact sheet lacks a path")
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(output_dir.resolve()):
        raise ValueError(f"{video_id} {view} contact sheet escapes output_dir")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{video_id} {view} contact sheet lacks a SHA-256")
    actual_digest = _sha256_file(resolved)
    if actual_digest != digest:
        raise ValueError(f"{video_id} {view} contact sheet hash mismatch")
    return {**dict(contact_sheet), "path": str(resolved), "sha256": actual_digest}


def _bind_verified_global_frames(
    planned_frames: Sequence[Mapping[str, Any]],
    baseline_frames: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach verified baseline image artifacts to the canonical G plan."""

    normalized_baseline: list[dict[str, Any]] = []
    bound: list[dict[str, Any]] = []
    for planned, raw_baseline in zip(planned_frames, baseline_frames, strict=True):
        baseline = dict(raw_baseline)
        if "requested_time_seconds" not in baseline:
            requested = baseline.get("requested_timestamp_seconds")
            if requested is not None:
                baseline["requested_time_seconds"] = requested
        if "actual_pts_seconds" not in baseline:
            actual = baseline.get("selected_frame_timestamp_seconds")
            if actual is not None:
                baseline["actual_pts_seconds"] = actual
        if "frame_path" not in baseline:
            for key in ("path", "image_path"):
                if isinstance(baseline.get(key), str) and baseline[key]:
                    baseline["frame_path"] = baseline[key]
                    break
        if "image_sha256" not in baseline:
            digest = baseline.get("sha256")
            if isinstance(digest, str):
                baseline["image_sha256"] = digest

        supplied_identity = baseline.get("verified_decoded_identity", baseline)
        if not isinstance(supplied_identity, Mapping):
            raise ValueError("verified global frame decoded identity must be an object")
        verified_identity = _decoded_identity(
            supplied_identity, source="verified global frame decoded identity"
        )
        for field in (
            "stream_index",
            "frame_index",
            "time_base",
            "actual_pts_seconds",
        ):
            if field in baseline and verified_identity[field] != baseline[field]:
                raise ValueError(
                    f"verified global frame decoded identity disagrees on {field}"
                )

        candidate = dict(planned)
        for field in ("frame_path", "image_sha256"):
            if field in baseline:
                candidate[field] = baseline[field]
        candidate["verified_decoded_identity"] = verified_identity
        normalized_baseline.append(baseline)
        bound.append(candidate)
    assert_global_plan_reuse(bound, normalized_baseline)
    return bound


def build_evidence_bundle(
    *,
    rows: Sequence[Mapping[str, Any]],
    selectors: Sequence[Mapping[str, Any]],
    global_evidence: Sequence[Mapping[str, Any]],
    whisper: Sequence[Mapping[str, Any]],
    baseline_requests: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    expected_count: int = EXPECTED_COUNT,
    expected_panel_id_digest: str = CANONICAL_PANEL_ID_DIGEST,
    expected_selector_protocol_sha256: str,
    run_id: str,
    model_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    prompt_source_identity: str,
    contact_sheet_builder: ContactSheetBuilder = make_contact_sheet,
    frame_extractor: FrameExtractor | None = None,
) -> dict[str, Any]:
    """Build all nine cells after in-memory, label-free provenance checks."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be nonempty text")
    if not isinstance(model_identity, Mapping) or not model_identity:
        raise ValueError("model_identity must be a nonempty object")
    if not isinstance(runtime_identity, Mapping) or not runtime_identity:
        raise ValueError("runtime_identity must be a nonempty object")
    if not isinstance(prompt_source_identity, str) or not prompt_source_identity:
        raise ValueError("prompt_source_identity must be nonempty text")
    if not rows:
        raise ValueError("rows must be nonempty")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("expected_count must be a positive integer")
    if len(rows) != expected_count:
        raise ValueError(f"panel must contain exactly {expected_count} rows")

    (
        _,
        selector_by_id,
        evidence_by_id,
        whisper_by_id,
    ) = _validate_frozen_identities(
        rows=rows,
        selectors=selectors,
        global_evidence=global_evidence,
        whisper=whisper,
        baseline_requests=baseline_requests,
        expected_count=expected_count,
        expected_panel_id_digest=expected_panel_id_digest,
        expected_selector_protocol_sha256=expected_selector_protocol_sha256,
    )
    for video_id, evidence_row in evidence_by_id.items():
        audit_label_free_request(evidence_row, location=f"global evidence[{video_id}]")
    for video_id, whisper_row in whisper_by_id.items():
        audit_label_free_request(whisper_row, location=f"Whisper[{video_id}]")
    for video_id in sorted(baseline_requests):
        _validated_baseline_request(baseline_requests, video_id)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing evidence output {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = [_validate_panel_row(row) for row in rows]

    requests_by_cell: dict[str, list[dict[str, Any]]] = {
        cell: [] for cell in EVIDENCE_CELLS
    }
    evidence_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in panel:
        video_id = row["video_id"]
        selector = _validated_selector(row, selector_by_id[video_id])
        evidence_row = evidence_by_id[video_id]
        if evidence_row.get("source_media_sha256") != row["media_sha256"]:
            raise ValueError(f"{video_id} global evidence media identity mismatch")
        inventory = evidence_row.get("inventory")
        baseline_frames = evidence_row.get("frames")
        if not isinstance(inventory, list) or not isinstance(baseline_frames, list):
            raise ValueError(
                f"{video_id} global evidence needs inventory and twelve frames"
            )
        if len(baseline_frames) != 12:
            raise ValueError(
                f"{video_id} verified global evidence must contain twelve frames"
            )
        plans = build_frame_plans(row, selector, inventory)
        plans["G"] = _bind_verified_global_frames(plans["G"], baseline_frames)
        transcript_text = whisper_by_id[video_id].get("text")
        transcript_views = build_transcript_views(
            transcript_text,
            selector["whisper_segment_ids"],
            duration_seconds=float(row["duration_seconds"]),
        )

        view_records: dict[str, dict[str, Any]] = {}
        for view in FRAME_VIEWS:
            frames = [
                _materialize_frame(
                    row,
                    frame,
                    view,
                    output_dir / "frames",
                    frame_extractor=frame_extractor,
                    runtime_identity=runtime_identity,
                )
                for frame in plans[view]
            ]
            contact_sheet = contact_sheet_builder(
                video_id=video_id,
                view=view,
                frame_paths=[frame["frame_path"] for frame in frames],
                output_path=output_dir / "contact-sheets" / f"{video_id}-{view}.jpg",
            )
            if not isinstance(contact_sheet, Mapping):
                raise TypeError("contact_sheet_builder must return an object")
            contact_sheet = _validated_contact_sheet(
                contact_sheet,
                output_dir=output_dir,
                video_id=video_id,
                view=view,
            )
            view_records[view] = {
                "frame_count": len(frames),
                "frames": frames,
                "contact_sheet": {
                    **dict(contact_sheet),
                    "video_id": video_id,
                    "view": view,
                    "frame_count": len(frames),
                },
            }

        materialized_plans = {
            view: view_records[view]["frames"] for view in FRAME_VIEWS
        }
        materialized_plans["localized"] = plans["localized"]
        materialized_plans["fallback_reason"] = plans["fallback_reason"]
        matrix = build_request_matrix(
            row=row,
            frame_plans=materialized_plans,
            transcript_views=transcript_views,
            run_id=run_id,
        )
        matrix_audit = audit_request_matrix(matrix)
        held_fixed_identity = _canonical_sha256(
            {
                "row_identity": matrix_audit["held_fixed_identity_sha256"],
                "model_identity": dict(model_identity),
                "runtime_identity": dict(runtime_identity),
                "prompt_source_identity": prompt_source_identity,
            }
        )
        baseline = _validated_baseline_request(baseline_requests, video_id)
        if model_input_sha256(matrix["gf"]) != model_input_sha256(baseline):
            raise ValueError(
                f"{video_id} G/F model-input digest does not reuse baseline"
            )

        cell_records: dict[str, dict[str, Any]] = {}
        for cell in EVIDENCE_CELLS:
            request = matrix[cell]
            transcript_view = TRANSCRIPT_VIEWS["NFS".index(cell[1].upper())]
            cell_record = {
                "request_sha256": request_sha256(request),
                "model_input_sha256": model_input_sha256(request),
                "held_fixed_identity_sha256": held_fixed_identity,
                "transcript": request["transcript"],
                "transcript_source_sha256": transcript_views[transcript_view][
                    "source_sha256"
                ],
                "selected_line_indices": transcript_views[transcript_view][
                    "selected_line_indices"
                ],
            }
            cell_records[cell] = cell_record
            requests_by_cell[cell].append(request)
        evidence_rows.append(
            {
                "video_id": video_id,
                "source_media_sha256": row["media_sha256"],
                "selector": selector,
                "views": view_records,
                "cells": cell_records,
            }
        )
        audits.append(
            {
                "video_id": video_id,
                "cells": {
                    cell: {
                        "leakage": False,
                        "request_sha256": cell_records[cell]["request_sha256"],
                        "model_input_sha256": cell_records[cell]["model_input_sha256"],
                    }
                    for cell in EVIDENCE_CELLS
                },
            }
        )

    for cell in EVIDENCE_CELLS:
        for request in requests_by_cell[cell]:
            audit_label_free_request(request, location=f"request[{cell}]")
        _write_json(output_dir / "requests" / f"{cell}.json", requests_by_cell[cell])

    evidence_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "canonical_id_digest": expected_panel_id_digest,
        "selector_protocol_sha256": expected_selector_protocol_sha256,
        "model_identity": dict(model_identity),
        "runtime_identity": dict(runtime_identity),
        "prompt_source_identity": prompt_source_identity,
        "rows": evidence_rows,
    }
    audit_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "rows": audits,
        "references_loaded": False,
    }
    _write_json(output_dir / "evidence.json", evidence_payload)
    _write_json(output_dir / "leakage-audit.json", audit_payload)
    marker = {
        "schema_version": 1,
        "label_policy": "label-free-evidence",
        "example_count": len(panel),
        "canonical_id_digest": expected_panel_id_digest,
        "checksum_file": "SHA256SUMS",
    }
    _write_json(output_dir / "evidence-marker.json", marker)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "row_count": len(panel),
        "cell_count": len(EVIDENCE_CELLS),
        "cells": list(EVIDENCE_CELLS),
        "request_counts": {
            cell: len(requests_by_cell[cell]) for cell in EVIDENCE_CELLS
        },
        "request_set_sha256": {
            cell: _canonical_sha256(requests_by_cell[cell]) for cell in EVIDENCE_CELLS
        },
        "evidence_sha256": _canonical_sha256(evidence_payload),
        "leakage_audit_sha256": _canonical_sha256(audit_payload),
        "canonical_id_digest": expected_panel_id_digest,
        "protocol_sha256": expected_selector_protocol_sha256,
        "selector_protocol_sha256": expected_selector_protocol_sha256,
    }
    _write_json(output_dir / "summary.json", summary)
    payload_files = sorted(
        path.relative_to(output_dir)
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256_file(output_dir / path)}  {path.as_posix()}\n"
            for path in payload_files
        ),
        encoding="utf-8",
    )
    return summary


def _baseline_mapping(payload: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if all(isinstance(value, Mapping) for value in payload.values()):
            return {str(key): value for key, value in payload.items()}
        rows = payload.get("requests")
        if not isinstance(rows, list):
            raise ValueError("baseline requests must be a mapping or request list")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("baseline requests must be a mapping or request list")
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("video_id"), str):
            raise ValueError("baseline request rows need video_id")
        indexed[row["video_id"]] = row
    return indexed


# Keep the public name parallel with the task's evidence artifact vocabulary.
build_evidence = build_evidence_bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--selectors", type=Path, required=True)
    parser.add_argument("--global-evidence", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--baseline-requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--prompt-source-identity", required=True)
    parser.add_argument("--expected-selector-protocol-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _rows_from_payload(
        _load_json(args.rows), ("examples", "rows"), source="rows"
    )
    selectors = _rows_from_payload(
        _load_json(args.selectors), ("results", "selectors", "rows"), source="selectors"
    )
    global_evidence = _rows_from_payload(
        _load_json(args.global_evidence), ("examples", "rows"), source="global evidence"
    )
    whisper = _rows_from_payload(
        _load_json(args.whisper), ("results", "rows"), source="Whisper"
    )
    baseline = _baseline_mapping(_load_json(args.baseline_requests))
    summary = build_evidence_bundle(
        rows=rows,
        selectors=selectors,
        global_evidence=global_evidence,
        whisper=whisper,
        baseline_requests=baseline,
        output_dir=args.output_dir,
        expected_count=EXPECTED_COUNT,
        expected_panel_id_digest=CANONICAL_PANEL_ID_DIGEST,
        expected_selector_protocol_sha256=(args.expected_selector_protocol_sha256),
        run_id=args.run_id,
        model_identity={"model_id": args.model_id, "revision": args.model_revision},
        runtime_identity={"image": args.runtime_image},
        prompt_source_identity=args.prompt_source_identity,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
