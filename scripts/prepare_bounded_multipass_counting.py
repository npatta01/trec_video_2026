#!/usr/bin/env python3
"""Prepare the private panel and label-blind requests for bounded counting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_counting_evidence import validate_label_blind
from trec_video_2026.vqa_submission import canonical_sha256


REQUIRED_STRATA = {
    "failure_action": 4,
    "failure_static": 3,
    "failure_other_temporal": 1,
    "control_temporal": 2,
    "control_static": 2,
}
IMPLEMENTATION_SOURCE_PATHS = (
    "pyproject.toml",
    "scripts/prepare_bounded_multipass_counting.py",
    "src/trec_video_2026/vqa_bounded_multipass_counting.py",
    "src/trec_video_2026/vqa_counting_evidence.py",
    "src/trec_video_2026/vqa_gate.py",
    "src/trec_video_2026/vqa_submission.py",
    "uv.lock",
)


def load_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=reject_duplicates)


def _routing_rows(routing: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    if routing.get("schema_version") != "bounded-multipass-routing-v1":
        raise ValueError("unsupported routing schema_version")
    values = routing.get("rows")
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise ValueError("routing rows must be an array")
    result: dict[str, Mapping[str, object]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("routing row must be an object")
        if set(value) != {
            "video_id",
            "stratum",
            "supported",
            "question_only_rationale",
        }:
            raise ValueError("routing row fields are invalid")
        video_id = value.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("routing video_id must be nonempty")
        if video_id in result:
            raise ValueError(f"duplicate routing video_id: {video_id}")
        result[video_id] = value
    return result


def _selection_key(video_id: str) -> tuple[str, str]:
    return hashlib.sha256(video_id.encode()).hexdigest(), video_id


def select_panel(
    *,
    rows: Sequence[Mapping[str, object]],
    routing: Mapping[str, object],
    excluded_ids: set[str],
) -> list[dict[str, object]]:
    route_by_id = _routing_rows(routing)
    seen: set[str] = set()
    eligible: dict[str, list[dict[str, object]]] = {
        stratum: [] for stratum in REQUIRED_STRATA
    }
    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("census row video_id must be nonempty")
        if video_id in seen:
            raise ValueError(f"duplicate video_id: {video_id}")
        seen.add(video_id)
        route = route_by_id.get(video_id)
        if (
            video_id in excluded_ids
            or route is None
            or route.get("supported") is not True
        ):
            continue
        stratum = route.get("stratum")
        if stratum not in REQUIRED_STRATA:
            continue
        expected_status = "wrong" if str(stratum).startswith("failure") else "correct"
        if row.get("status") != expected_status:
            raise ValueError(
                f"status does not match routed stratum for {video_id}: "
                f"{row.get('status')} vs {expected_status}"
            )
        selected = dict(row)
        selected["stratum"] = stratum
        selected["question_only_rationale"] = route.get("question_only_rationale")
        eligible[str(stratum)].append(selected)
    panel: list[dict[str, object]] = []
    for stratum, required in REQUIRED_STRATA.items():
        candidates = sorted(
            eligible[stratum], key=lambda row: _selection_key(str(row["video_id"]))
        )
        if len(candidates) < required:
            raise ValueError(
                f"insufficient {stratum} rows: required {required}, got {len(candidates)}"
            )
        panel.extend(candidates[:required])
    counts = Counter(str(row["stratum"]) for row in panel)
    if dict(counts) != REQUIRED_STRATA:
        raise AssertionError("selected panel does not match required strata")
    return panel


def project_label_blind(
    panel: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    requests = [
        {
            "ordinal": row["ordinal"],
            "video_id": row["video_id"],
            "question": row["question"],
        }
        for row in panel
    ]
    validate_label_blind(requests)
    return requests


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_media(
    *,
    panel: Sequence[Mapping[str, object]],
    prior_requests: Mapping[str, object],
) -> list[dict[str, object]]:
    values = prior_requests.get("rows")
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise ValueError("prior requests must contain rows")
    by_id: dict[str, Mapping[str, object]] = {}
    for value in values:
        if not isinstance(value, Mapping) or not isinstance(value.get("video_id"), str):
            raise ValueError("prior request row is invalid")
        video_id = str(value["video_id"])
        if video_id in by_id:
            raise ValueError(f"duplicate media binding for {video_id}")
        by_id[video_id] = value
    result: list[dict[str, object]] = []
    for row in panel:
        video_id = str(row["video_id"])
        binding = by_id.get(video_id)
        if binding is None:
            raise ValueError(f"missing media binding for {video_id}")
        source_path = Path(str(binding.get("source_path")))
        delivery_path = Path(str(binding.get("delivery_path")))
        if not source_path.is_file() or not delivery_path.is_file():
            raise ValueError(f"missing bound media file for {video_id}")
        source_sha256 = _file_sha256(source_path)
        delivery_sha256 = _file_sha256(delivery_path)
        if source_sha256 != binding.get("source_sha256"):
            raise ValueError(f"source_sha256 mismatch for {video_id}")
        if delivery_sha256 != binding.get("delivery_sha256"):
            raise ValueError(f"delivery_sha256 mismatch for {video_id}")
        duration = binding.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, int | float):
            raise ValueError(f"duration_seconds is invalid for {video_id}")
        mime_type = binding.get("delivery_mime_type")
        if not isinstance(mime_type, str) or not mime_type:
            raise ValueError(f"delivery_mime_type is invalid for {video_id}")
        result.append(
            {
                "ordinal": row["ordinal"],
                "video_id": video_id,
                "question": row["question"],
                "source_path": str(source_path),
                "source_sha256": source_sha256,
                "delivery_path": str(delivery_path),
                "delivery_sha256": delivery_sha256,
                "delivery_mime_type": mime_type,
                "duration_seconds": float(duration),
            }
        )
    validate_label_blind(result)
    return result


def _probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(completed.stdout.strip())
    if duration <= 0:
        raise ValueError(f"nonpositive media duration: {path}")
    return duration


def bind_media_from_roots(
    *,
    panel: Sequence[Mapping[str, object]],
    source_root: Path,
    delivery_root: Path,
) -> list[dict[str, object]]:
    prior_rows = []
    for row in panel:
        video_id = str(row["video_id"])
        source_path = source_root / f"{video_id}.mkv"
        media_root = delivery_root / video_id / "delivery-media"
        delivery_candidates = [
            path
            for path in (
                media_root / f"{video_id}.webm",
                media_root / f"{video_id}.mp4",
            )
            if path.is_file()
        ]
        if not source_path.is_file() or len(delivery_candidates) != 1:
            raise ValueError(f"missing bound media file for {video_id}")
        delivery_path = delivery_candidates[0]
        mime_type = "video/webm" if delivery_path.suffix == ".webm" else "video/mp4"
        prior_rows.append(
            {
                "video_id": video_id,
                "source_path": str(source_path),
                "source_sha256": _file_sha256(source_path),
                "delivery_path": str(delivery_path),
                "delivery_sha256": _file_sha256(delivery_path),
                "delivery_mime_type": mime_type,
                "duration_seconds": _probe_duration(delivery_path),
            }
        )
    return bind_media(panel=panel, prior_requests={"rows": prior_rows})


def _render_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_self_hashed_once(
    path: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    normalized = dict(payload)
    normalized.pop("complete_sha256", None)
    normalized["complete_sha256"] = canonical_sha256(normalized)
    rendered = _render_json(normalized)
    if path.exists():
        if path.read_text() != rendered:
            raise FileExistsError(f"refusing to overwrite conflicting freeze: {path}")
        return normalized
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    return normalized


def prepare_experiment(
    *,
    census_path: Path,
    routing_path: Path,
    excluded_ids_path: Path,
    source_root: Path,
    delivery_root: Path,
    output_root: Path,
) -> dict[str, object]:
    census = load_json(census_path)
    routing = load_json(routing_path)
    exclusions = load_json(excluded_ids_path)
    if not isinstance(census, Mapping) or not isinstance(census.get("rows"), list):
        raise ValueError("census must contain rows")
    if not isinstance(routing, Mapping):
        raise ValueError("routing must be an object")
    if not isinstance(exclusions, Mapping) or not isinstance(
        exclusions.get("video_ids"), list
    ):
        raise ValueError("excluded IDs must contain video_ids")
    panel = select_panel(
        rows=census["rows"],
        routing=routing,
        excluded_ids={str(value) for value in exclusions["video_ids"]},
    )
    panel_freeze = write_self_hashed_once(
        output_root / "panel.freeze.json",
        {
            "schema_version": "bounded-multipass-panel-v1",
            "selection_policy": {
                "required_strata": REQUIRED_STRATA,
                "ordering": "sha256(video_id), then video_id",
            },
            "rows": panel,
        },
    )
    requests = write_self_hashed_once(
        output_root / "requests.label-blind.json",
        {
            "schema_version": "bounded-multipass-requests-v1",
            "panel_complete_sha256": panel_freeze["complete_sha256"],
            "rows": bind_media_from_roots(
                panel=panel, source_root=source_root, delivery_root=delivery_root
            ),
        },
    )
    repository = Path(__file__).resolve().parents[1]
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return write_self_hashed_once(
        output_root / "preparation-complete.json",
        {
            "schema_version": "bounded-multipass-preparation-complete-v1",
            "panel_complete_sha256": panel_freeze["complete_sha256"],
            "requests_complete_sha256": requests["complete_sha256"],
            "planned_rows": len(panel),
            "input_files": {
                "census": {
                    "path": str(census_path),
                    "sha256": _file_sha256(census_path),
                },
                "routing": {
                    "path": str(routing_path),
                    "sha256": _file_sha256(routing_path),
                },
                "excluded_ids": {
                    "path": str(excluded_ids_path),
                    "sha256": _file_sha256(excluded_ids_path),
                },
            },
            "source_revision": source_revision,
            "implementation_sources": {
                relative: _file_sha256(repository / relative)
                for relative in IMPLEMENTATION_SOURCE_PATHS
            },
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--excluded-ids", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--delivery-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = prepare_experiment(
        census_path=args.census,
        routing_path=args.routing,
        excluded_ids_path=args.excluded_ids,
        source_root=args.source_root,
        delivery_root=args.delivery_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
