#!/usr/bin/env python3
"""Execute frozen label-blind count queries over frozen evidence graphs."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_count_query import (
    count_video,
    parse_count_query,
    parse_evidence_graph,
    proof_to_dict,
)
from trec_video_2026.vqa_counting_evidence import validate_label_blind
from trec_video_2026.vqa_submission import canonical_sha256


def _verify_hash(payload: Mapping[str, object], *, name: str) -> str:
    embedded = payload.get("complete_sha256")
    if not isinstance(embedded, str):
        raise TypeError(f"{name} lacks complete_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "complete_sha256"
    }
    if canonical_sha256(unsigned) != embedded:
        raise ValueError(f"{name} complete_sha256 changed")
    return embedded


def _rows_by_id(
    payload: Mapping[str, object], *, name: str, verify_rows: bool = False
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
        if verify_rows:
            _verify_hash(raw, name=f"{name} row {video_id}")
        rows[video_id] = raw
    return rows


def _verify_request_row(row: Mapping[str, object]) -> str:
    embedded = row.get("request_sha256")
    if not isinstance(embedded, str):
        raise TypeError("authoring request row lacks request_sha256")
    unsigned = {key: value for key, value in row.items() if key != "request_sha256"}
    if canonical_sha256(unsigned) != embedded:
        raise ValueError("authoring request row request_sha256 changed")
    return embedded


def execute_oracle_slice(
    *,
    authoring_requests: Mapping[str, object],
    authored_inputs: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate and execute a complete label-blind authored slice."""

    requests_hash = _verify_hash(authoring_requests, name="authoring requests")
    authored_hash = _verify_hash(authored_inputs, name="authored inputs")
    validate_label_blind(authoring_requests)
    validate_label_blind(authored_inputs)
    if authored_inputs.get("authoring_requests_complete_sha256") != requests_hash:
        raise ValueError("authored inputs request-manifest binding changed")
    requests = _rows_by_id(authoring_requests, name="authoring requests")
    authored = _rows_by_id(authored_inputs, name="authored inputs", verify_rows=True)
    if set(requests) != set(authored):
        raise ValueError("authored input IDs must exactly match authoring request IDs")

    frozen_rows: list[dict[str, object]] = []
    for video_id in requests:
        request = requests[video_id]
        authored_row = authored[video_id]
        request_hash = _verify_request_row(request)
        if authored_row.get("request_sha256") != request_hash:
            raise ValueError(f"authored request binding changed for {video_id}")
        if authored_row.get("source_sha256") != request.get("source_sha256"):
            raise ValueError(f"authored source media hash changed for {video_id}")
        query_payload = authored_row.get("query")
        graph_payload = authored_row.get("graph")
        if not isinstance(query_payload, Mapping) or not isinstance(
            graph_payload, Mapping
        ):
            raise TypeError(f"authored query and graph required for {video_id}")
        query = parse_count_query(query_payload)
        graph = parse_evidence_graph(graph_payload)
        if query.query_id != video_id or graph.graph_id != video_id:
            raise ValueError("query and graph IDs must equal video ID")
        if graph.duration_seconds != request.get("duration_seconds"):
            raise ValueError(f"graph duration changed for {video_id}")
        proof = proof_to_dict(count_video(query, graph))
        frozen_row = {
            "video_id": video_id,
            "request_sha256": request_hash,
            "source_sha256": request["source_sha256"],
            "query_complete_sha256": query_payload["complete_sha256"],
            "graph_complete_sha256": graph_payload["complete_sha256"],
            "proof": proof,
        }
        frozen_row["complete_sha256"] = canonical_sha256(frozen_row)
        frozen_rows.append(frozen_row)
    freeze: dict[str, object] = {
        "schema_version": 1,
        "authoring_requests_complete_sha256": requests_hash,
        "authored_inputs_complete_sha256": authored_hash,
        "rows": frozen_rows,
    }
    freeze["complete_sha256"] = canonical_sha256(freeze)
    return freeze


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
    parser.add_argument("--authoring-requests", type=Path, required=True)
    parser.add_argument("--authored-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    freeze = execute_oracle_slice(
        authoring_requests=_load_json(args.authoring_requests),
        authored_inputs=_load_json(args.authored_inputs),
    )
    _write_once_json(args.output, freeze)
    print(
        json.dumps(
            {
                "row_count": len(freeze["rows"]),
                "complete_sha256": freeze["complete_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
