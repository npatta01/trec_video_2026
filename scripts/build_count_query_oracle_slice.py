#!/usr/bin/env python3
"""Freeze a held-out, label-blind oracle slice for the count query engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_counting_evidence import validate_label_blind
from trec_video_2026.vqa_submission import canonical_sha256


def _verify_complete_hash(payload: Mapping[str, object], *, name: str) -> str:
    embedded = payload.get("complete_sha256")
    if not isinstance(embedded, str):
        raise TypeError(f"{name} lacks complete_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "complete_sha256"
    }
    if canonical_sha256(unsigned) != embedded:
        raise ValueError(f"{name} complete_sha256 changed")
    return embedded


def _unique_rows(
    payload: Mapping[str, object], *, name: str
) -> dict[str, dict[str, object]]:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError(f"{name}.rows must be an array")
    rows: dict[str, dict[str, object]] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict) or not isinstance(raw.get("video_id"), str):
            raise TypeError(f"{name} row must contain video_id")
        video_id = raw["video_id"]
        if video_id in rows:
            raise ValueError(f"{name} video IDs must be unique")
        rows[video_id] = raw
    return rows


def _rank(*, role: str, stratum: str, video_id: str) -> str:
    return hashlib.sha256(
        f"count-query-oracle-v1\0{role}\0{stratum}\0{video_id}".encode()
    ).hexdigest()


def _balanced_ids(
    candidates: Sequence[str],
    *,
    role: str,
    count: int,
    semantic_strata: Mapping[str, str],
) -> list[str]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for video_id in candidates:
        stratum = semantic_strata.get(video_id)
        if not isinstance(stratum, str) or not stratum:
            raise ValueError(f"missing semantic stratum for {video_id}")
        buckets[stratum].append(video_id)
    for stratum, video_ids in buckets.items():
        video_ids.sort(
            key=lambda item: (_rank(role=role, stratum=stratum, video_id=item), item)
        )
    selected: list[str] = []
    strata = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            if buckets[stratum] and len(selected) < count:
                selected.append(buckets[stratum].pop(0))
                progressed = True
        if not progressed:
            raise ValueError(f"not enough {role} rows for requested slice")
    return selected


def _with_hash(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def build_oracle_slice(
    *,
    panel: Mapping[str, object],
    requests: Mapping[str, object],
    design_video_ids: Sequence[str],
    semantic_strata: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Select 8 failures and 4 controls, then project label-blind authoring files."""

    panel_hash = _verify_complete_hash(panel, name="panel")
    requests_hash = _verify_complete_hash(requests, name="requests")
    validate_label_blind(requests)
    panel_rows = _unique_rows(panel, name="panel")
    request_rows = _unique_rows(requests, name="requests")
    ordered_ids = [row["video_id"] for row in requests["rows"]]  # type: ignore[index]
    if requests.get("ordered_video_ids_sha256") != canonical_sha256(ordered_ids):
        raise ValueError("requests ordered_video_ids_sha256 changed")
    if len(design_video_ids) != len(set(design_video_ids)):
        raise ValueError("design video IDs must be unique")
    missing_design = sorted(set(design_video_ids) - set(panel_rows))
    if missing_design:
        raise ValueError(f"design video IDs missing from panel: {missing_design}")

    design = set(design_video_ids)
    by_role: dict[str, list[str]] = defaultdict(list)
    for video_id, row in panel_rows.items():
        role = row.get("panel_role")
        if role in {"confirmed_failure", "correct_control"} and video_id not in design:
            if video_id not in request_rows:
                raise ValueError(f"selected candidate missing request row: {video_id}")
            by_role[str(role)].append(video_id)
    selected_ids = _balanced_ids(
        by_role["confirmed_failure"],
        role="confirmed_failure",
        count=8,
        semantic_strata=semantic_strata,
    ) + _balanced_ids(
        by_role["correct_control"],
        role="correct_control",
        count=4,
        semantic_strata=semantic_strata,
    )

    selection_rows = []
    for video_id in selected_ids:
        panel_row = panel_rows[video_id]
        request_row = request_rows[video_id]
        request_hash = canonical_sha256(request_row)
        selection_rows.append(
            {
                **panel_row,
                "semantic_stratum": semantic_strata[video_id],
                "request_sha256": request_hash,
            }
        )
    authoring_rows = []
    scaffold_rows = []
    authoring_ids = sorted(
        selected_ids,
        key=lambda video_id: hashlib.sha256(
            f"count-query-authoring-order-v1\0{video_id}".encode()
        ).hexdigest(),
    )
    for video_id in authoring_ids:
        request_row = request_rows[video_id]
        request_hash = canonical_sha256(request_row)
        authoring_rows.append({**request_row, "request_sha256": request_hash})
        scaffold_rows.append({"video_id": video_id, "query": None, "graph": None})

    selection = _with_hash(
        {
            "schema_version": 1,
            "selection_policy": "balanced-question-semantics-hash-ranked-v1",
            "panel_complete_sha256": panel_hash,
            "requests_complete_sha256": requests_hash,
            "design_video_ids": list(design_video_ids),
            "rows": selection_rows,
        }
    )
    authoring_requests = _with_hash(
        {
            "schema_version": 1,
            "selection_complete_sha256": selection["complete_sha256"],
            "source_requests_complete_sha256": requests_hash,
            "rows": authoring_rows,
        }
    )
    authoring_scaffolds = _with_hash(
        {
            "schema_version": 1,
            "authoring_requests_complete_sha256": authoring_requests["complete_sha256"],
            "rows": scaffold_rows,
        }
    )
    validate_label_blind(authoring_requests)
    validate_label_blind(authoring_scaffolds)
    return {
        "selection": selection,
        "authoring_requests": authoring_requests,
        "authoring_scaffolds": authoring_scaffolds,
    }


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_once_json(path: Path, payload: object) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite conflicting freeze: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--design-ids", type=Path, required=True)
    parser.add_argument("--semantic-strata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    design_payload = _load_json(args.design_ids)
    design_ids = (
        design_payload.get("canary_video_ids")
        if isinstance(design_payload, dict)
        else design_payload
    )
    if not isinstance(design_ids, list) or not all(
        isinstance(item, str) for item in design_ids
    ):
        raise TypeError("design IDs must be an array of strings")
    semantic_strata = _load_json(args.semantic_strata)
    if not isinstance(semantic_strata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in semantic_strata.items()
    ):
        raise TypeError("semantic strata must map video IDs to strings")
    artifacts = build_oracle_slice(
        panel=_load_json(args.panel),
        requests=_load_json(args.requests),
        design_video_ids=design_ids,
        semantic_strata=semantic_strata,
    )
    names = {
        "selection": "selection.freeze.json",
        "authoring_requests": "authoring.requests.json",
        "authoring_scaffolds": "authoring.scaffolds.json",
    }
    for key, name in names.items():
        _write_once_json(args.output_root / name, artifacts[key])
    print(
        json.dumps(
            {
                "row_count": len(artifacts["selection"]["rows"]),
                "selection_complete_sha256": artifacts["selection"]["complete_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
