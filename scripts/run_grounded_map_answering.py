#!/usr/bin/env python3
"""Run the issue #35 text-only grounded scene-map answering treatment."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence


MODEL = "gemini-3.1-pro-preview"
EXPECTED_TARGET_COUNT = 12
EXPECTED_CONTROL_COUNT = 12
MAX_ANSWER_WORDS = 24
ANSWER_CONTRACT_VERSION = 2
ROW_ARTIFACT_NAMES = (
    "request.json",
    "raw-response.txt",
    "metadata.json",
    "parsed.json",
    "terminal.json",
)
PROMOTION_RULE = {
    "minimum_target_rescues": 6,
    "maximum_control_regressions": 1,
    "require_all_answer_claims_citation_supported": True,
    "require_no_unnecessary_verification_routes": True,
    "require_changed_row_audit_complete": True,
}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "question_interpretation": {"type": "string"},
        "answer_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["claim", "evidence_ids"],
            },
        },
        "conflicts_or_missing_qualifiers": {
            "type": "array",
            "items": {"type": "string"},
        },
        "needs_verification": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": [
        "question_interpretation",
        "answer_claims",
        "conflicts_or_missing_qualifiers",
        "needs_verification",
        "answer",
    ],
}
CallFunction = Callable[
    [str],
    tuple[str, dict[str, Any]] | tuple[dict[str, Any], str, dict[str, Any]],
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error


def _write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite immutable artifact: {path}") from error


def _write_text_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite immutable artifact: {path}") from error


def answer_prompt(
    question: str,
    evidence: Sequence[Mapping[str, str]],
    *,
    contract_version: int = ANSWER_CONTRACT_VERSION,
) -> str:
    """Build the frozen question-plus-oracle-map-lines treatment prompt."""

    evidence_text = "\n".join(
        f"[{item['evidence_id']}] {item['text']}" for item in evidence
    )
    answer_binding = (
        " One answer_claim must exactly repeat the final answer so the top-1 answer "
        "is directly bound to its evidence IDs."
        if contract_version >= 2
        else ""
    )
    return f"""Answer using only the supplied frozen evidence lines. First interpret exactly what the question asks: temporal relation, causal purpose, count, speech, state, action, or requested qualifier.

For every answer claim, cite all supporting evidence IDs.{answer_binding} Do not add an exact count, quotation, causal mechanism, identity, temporal event, or qualifier that the evidence does not state or logically entail. Record any conflict or missing qualifier. If the evidence cannot support one concise top-1 answer, set needs_verification to true, return no answer claims, and set answer exactly to NEEDS VERIFICATION. Otherwise set needs_verification to false and give one answer of at most {MAX_ANSWER_WORDS} words on one line.

Return the structured JSON fields question_interpretation, answer_claims (claim plus evidence_ids), conflicts_or_missing_qualifiers, needs_verification, and answer.

Question:
{question}

Frozen evidence:
{evidence_text}"""


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def validate_answer(
    payload: Mapping[str, Any],
    evidence_ids: set[str],
    *,
    require_answer_claim_match: bool = True,
) -> dict[str, Any]:
    """Validate the evidence-citing answer and verification-routing contract."""

    required = set(ANSWER_SCHEMA["required"])
    if set(payload) != required:
        raise ValueError("response must contain exactly the required fields")
    interpretation = _nonempty_string(
        payload["question_interpretation"], field="question_interpretation"
    )
    claims_payload = payload["answer_claims"]
    if not isinstance(claims_payload, list):
        raise ValueError("answer_claims must be a list")
    claims: list[dict[str, Any]] = []
    for index, item in enumerate(claims_payload):
        if not isinstance(item, Mapping) or set(item) != {"claim", "evidence_ids"}:
            raise ValueError(f"answer_claims[{index}] has invalid fields")
        claim = _nonempty_string(item["claim"], field=f"answer_claims[{index}].claim")
        citations = item["evidence_ids"]
        if (
            not isinstance(citations, list)
            or not citations
            or any(not isinstance(citation, str) for citation in citations)
            or len(citations) != len(set(citations))
        ):
            raise ValueError(f"answer_claims[{index}] requires unique evidence IDs")
        unknown = set(citations) - evidence_ids
        if unknown:
            raise ValueError(f"unknown evidence ID: {sorted(unknown)}")
        claims.append({"claim": claim, "evidence_ids": citations})
    conflicts_payload = payload["conflicts_or_missing_qualifiers"]
    if not isinstance(conflicts_payload, list):
        raise ValueError("conflicts_or_missing_qualifiers must be a list")
    conflicts = [
        _nonempty_string(value, field="conflicts_or_missing_qualifiers item")
        for value in conflicts_payload
    ]
    needs_verification = payload["needs_verification"]
    if not isinstance(needs_verification, bool):
        raise ValueError("needs_verification must be boolean")
    answer = _nonempty_string(payload["answer"], field="answer")
    if "\n" in answer or "\r" in answer:
        raise ValueError("answer must be one line")
    if len(answer.split()) > MAX_ANSWER_WORDS:
        raise ValueError(f"answer must contain at most {MAX_ANSWER_WORDS} words")
    if needs_verification:
        if answer != "NEEDS VERIFICATION":
            raise ValueError("verification-routed answer must be NEEDS VERIFICATION")
        if claims:
            raise ValueError("verification-routed answer must not contain answer claims")
        if not conflicts:
            raise ValueError("verification-routed answer requires a missing-evidence reason")
    elif not claims:
        raise ValueError("non-verification answer requires at least one cited claim")
    elif answer == "NEEDS VERIFICATION":
        raise ValueError("non-verification answer cannot use verification sentinel")
    elif require_answer_claim_match and answer not in {
        item["claim"] for item in claims
    }:
        raise ValueError("non-verification answer must exactly match one cited claim")
    return {
        "question_interpretation": interpretation,
        "answer_claims": claims,
        "conflicts_or_missing_qualifiers": conflicts,
        "needs_verification": needs_verification,
        "answer": answer,
    }


def _response_metadata(response: Any, elapsed: float, text: str) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    candidates = getattr(response, "candidates", None) or []
    finish_reason = None
    if candidates:
        value = getattr(candidates[0], "finish_reason", None)
        finish_reason = getattr(value, "name", str(value)) if value is not None else None
    return {
        "elapsed_seconds": round(elapsed, 6),
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "response_chars": len(text),
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
        "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
        "total_token_count": getattr(usage, "total_token_count", None),
        "cached_content_token_count": getattr(usage, "cached_content_token_count", None),
        "model_version": getattr(response, "model_version", None),
        "response_id": getattr(response, "response_id", None),
        "finish_reason": finish_reason,
    }


def _call_provider(prompt: str) -> tuple[str, dict[str, Any]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    started = time.monotonic()
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            seed=42,
            candidate_count=1,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_schema=ANSWER_SCHEMA,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=False,
            ),
        ),
    )
    raw = response.text or ""
    return raw, _response_metadata(response, time.monotonic() - started, raw)


def _index_unique(rows: Sequence[Mapping[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{source} contains a row without a video_id")
        if video_id in indexed:
            raise ValueError(f"{source} contains duplicate video_id: {video_id}")
        indexed[video_id] = dict(row)
    return indexed


def _selected_evidence(map_text: str, line_numbers: Any, *, video_id: str) -> list[dict[str, Any]]:
    if (
        not isinstance(line_numbers, list)
        or not line_numbers
        or any(isinstance(value, bool) or not isinstance(value, int) for value in line_numbers)
        or len(line_numbers) != len(set(line_numbers))
    ):
        raise ValueError(f"{video_id} evidence line numbers must be unique integers")
    lines = map_text.splitlines()
    selected: list[dict[str, Any]] = []
    for evidence_index, line_number in enumerate(line_numbers, start=1):
        if line_number < 1 or line_number > len(lines):
            raise ValueError(f"map line {line_number} does not exist for {video_id}")
        text = lines[line_number - 1].strip()
        if not text:
            raise ValueError(f"map line {line_number} is blank for {video_id}")
        selected.append(
            {
                "evidence_id": f"E{evidence_index:02d}",
                "map_line_number": line_number,
                "text": text,
            }
        )
    return selected


def prepare(
    source_root: Path,
    selection_path: Path,
    output: Path,
    *,
    experiment_id: str | None = None,
) -> None:
    """Freeze the audited target/control panel and exact frozen-map evidence."""

    paths = {
        "protocol": source_root / "protocol.json",
        "maps": source_root / "maps.question-blind.json",
        "direct_predictions": source_root / "predictions.direct-answer.json",
        "manual_audit": source_root / "failure-audit-v1" / "manual-review.json",
        "selection": selection_path,
    }
    source_protocol = _load_json(paths["protocol"])
    maps_payload = _load_json(paths["maps"])
    direct_payload = _load_json(paths["direct_predictions"])
    audit_payload = _load_json(paths["manual_audit"])
    selection = _load_json(selection_path)
    if selection.get("expected_target_count") != EXPECTED_TARGET_COUNT:
        raise ValueError(f"selection must declare {EXPECTED_TARGET_COUNT} targets")
    if selection.get("expected_control_count") != EXPECTED_CONTROL_COUNT:
        raise ValueError(f"selection must declare {EXPECTED_CONTROL_COUNT} controls")
    pairs = selection.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_TARGET_COUNT:
        raise ValueError(f"selection requires exactly {EXPECTED_TARGET_COUNT} pairs")

    protocol_by_id = _index_unique(source_protocol.get("rows", []), source="protocol")
    maps_by_id = _index_unique(maps_payload, source="maps")
    direct_by_id = _index_unique(direct_payload, source="direct predictions")
    audit_by_id = _index_unique(audit_payload.get("rows", []), source="manual audit")
    target_ids = [pair.get("target", {}).get("video_id") for pair in pairs]
    control_ids = [pair.get("control", {}).get("video_id") for pair in pairs]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("target video IDs must be unique")
    if len(control_ids) != len(set(control_ids)):
        raise ValueError("control video IDs must be unique")
    if set(target_ids) & set(control_ids):
        raise ValueError("target and control video IDs must be disjoint")
    audited_targets = {
        video_id
        for video_id, row in audit_by_id.items()
        if row.get("verdict") in {"incorrect", "partial"}
        and row.get("map_sufficient") is True
    }
    if set(target_ids) != audited_targets:
        raise ValueError(
            "target set must exactly match audited map-sufficient incorrect/partial rows"
        )

    rows: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    for pair in pairs:
        pair_id = pair.get("pair_id")
        stratum = pair.get("stratum")
        if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
            raise ValueError("pair IDs must be unique non-empty strings")
        pair_ids.add(pair_id)
        if not isinstance(stratum, str) or not stratum:
            raise ValueError(f"pair {pair_id} requires a stratum")
        for role in ("target", "control"):
            selected = pair[role]
            video_id = selected["video_id"]
            if video_id not in protocol_by_id or video_id not in maps_by_id:
                raise ValueError(f"selected video is absent from frozen sources: {video_id}")
            if video_id not in direct_by_id:
                raise ValueError(f"selected video has no frozen direct prediction: {video_id}")
            if maps_by_id[video_id].get("status") != "passed":
                raise ValueError(f"selected map did not pass: {video_id}")
            if direct_by_id[video_id].get("status") != "passed":
                raise ValueError(f"selected direct prediction did not pass: {video_id}")
            source_row = protocol_by_id[video_id]
            rows.append(
                {
                    "video_id": video_id,
                    "pair_id": pair_id,
                    "role": role,
                    "stratum": stratum,
                    "question": source_row["question"],
                    "evidence": _selected_evidence(
                        maps_by_id[video_id]["map"],
                        selected.get("evidence_line_numbers"),
                        video_id=video_id,
                    ),
                }
            )

    protocol = {
        "schema_version": 1,
        "answer_contract_version": ANSWER_CONTRACT_VERSION,
        "experiment_id": experiment_id or selection.get("experiment_id"),
        "source_selection_experiment_id": selection.get("experiment_id"),
        "created_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "condition": "question plus exact frozen question-blind map lines",
        "settings": {
            "temperature": 0.0,
            "seed": 42,
            "thinking_level": "HIGH",
            "candidate_count": 1,
            "retry_count": 0,
        },
        "label_isolation": (
            "panel selection is label-aware on spent training rows; provider receives "
            "only the frozen question and selected frozen-map lines"
        ),
        "excluded_inputs": [
            "reference answers",
            "direct predictions",
            "native video or audio",
            "newly acquired evidence",
            "multiple answer candidates",
        ],
        "panel": {
            "target_count": EXPECTED_TARGET_COUNT,
            "control_count": EXPECTED_CONTROL_COUNT,
            "pair_count": EXPECTED_TARGET_COUNT,
            "validity": "selected mechanism test; not a prevalence or generalization estimate",
        },
        "decision_gate": {
            "required": [
                "meaningful rescue on evidence-present target failures",
                "no more than one supported-control regression",
                "every non-verification answer cites supplied frozen evidence",
                "unsupported claims route to verification rather than invention",
            ]
        },
        "source_artifacts": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "prompt_template_sha256": hashlib.sha256(
            answer_prompt(
                "{QUESTION}",
                [{"evidence_id": "E01", "text": "{EVIDENCE}"}],
                contract_version=ANSWER_CONTRACT_VERSION,
            ).encode()
        ).hexdigest(),
        "response_schema_sha256": hashlib.sha256(
            json.dumps(ANSWER_SCHEMA, sort_keys=True).encode()
        ).hexdigest(),
        "rows": rows,
    }
    _write_once(output / "protocol.json", protocol)


def _verify_protocol_sources(protocol: Mapping[str, Any]) -> None:
    for source in protocol["source_artifacts"].values():
        path = Path(source["path"])
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"frozen source artifact changed: {path}")
    contract_version = int(protocol.get("answer_contract_version", 1))
    prompt_hash = hashlib.sha256(
        answer_prompt(
            "{QUESTION}",
            [{"evidence_id": "E01", "text": "{EVIDENCE}"}],
            contract_version=contract_version,
        ).encode()
    ).hexdigest()
    if prompt_hash != protocol["prompt_template_sha256"]:
        raise ValueError("frozen prompt template changed")
    schema_hash = hashlib.sha256(
        json.dumps(ANSWER_SCHEMA, sort_keys=True).encode()
    ).hexdigest()
    if schema_hash != protocol["response_schema_sha256"]:
        raise ValueError("frozen response schema changed")


def run_row(
    root: Path,
    video_id: str,
    *,
    call_fn: CallFunction | None = None,
) -> None:
    """Run one immutable one-attempt text-only grounded answer call."""

    if (root / "predictions-freeze.json").exists():
        raise ValueError("predictions are already frozen")
    protocol = _load_json(root / "protocol.json")
    _verify_protocol_sources(protocol)
    row = next(
        (item for item in protocol["rows"] if item["video_id"] == video_id), None
    )
    if row is None:
        raise ValueError("video is not in the frozen protocol")
    row_root = root / "rows" / video_id
    terminal_path = row_root / "terminal.json"
    if terminal_path.exists():
        print(json.dumps({"video_id": video_id, "status": "already-terminal"}))
        return
    contract_version = int(protocol.get("answer_contract_version", 1))
    prompt = answer_prompt(
        row["question"], row["evidence"], contract_version=contract_version
    )
    _write_once(
        row_root / "request.json",
        {
            "model": protocol["model"],
            "question": row["question"],
            "evidence": row["evidence"],
            "prompt": prompt,
            "response_schema": ANSWER_SCHEMA,
        },
    )
    provider = call_fn or _call_provider
    try:
        provider_result = provider(prompt)
    except Exception as error:
        _write_once(
            terminal_path,
            {
                "video_id": video_id,
                "status": "provider_error",
                "error_type": type(error).__name__,
            },
        )
        raise
    if len(provider_result) == 3:
        payload, raw, metadata = provider_result
    else:
        raw, metadata = provider_result
        payload = None
    _write_text_once(row_root / "raw-response.txt", raw)
    _write_once(row_root / "metadata.json", metadata)
    evidence_ids = {item["evidence_id"] for item in row["evidence"]}
    try:
        if payload is None:
            payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("provider response must be a JSON object")
        parsed = validate_answer(
            payload,
            evidence_ids,
            require_answer_claim_match=contract_version >= 2,
        )
    except (json.JSONDecodeError, ValueError) as error:
        _write_once(
            terminal_path,
            {
                "video_id": video_id,
                "status": "contract_error",
                "error_type": type(error).__name__,
            },
        )
        raise
    _write_once(row_root / "parsed.json", parsed)
    _write_once(
        terminal_path,
        {
            "video_id": video_id,
            "pair_id": row["pair_id"],
            "role": row["role"],
            "stratum": row["stratum"],
            "question": row["question"],
            "status": "passed",
            **parsed,
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        },
    )
    print(json.dumps({"video_id": video_id, "status": "passed"}))


def freeze_gate(root: Path) -> None:
    """Additively freeze exact issue #35 promotion thresholds before inference."""

    protocol_path = root / "protocol.json"
    protocol = _load_json(protocol_path)
    if (root / "rows").exists() or (root / "predictions-freeze.json").exists():
        raise ValueError("promotion gate must freeze before inference")
    _write_once(
        root / "preinference-gate-freeze.json",
        {
            "schema_version": 1,
            "experiment_id": protocol["experiment_id"],
            "frozen_at": datetime.now(UTC).isoformat(),
            "protocol_sha256": _sha256(protocol_path),
            "rule": PROMOTION_RULE,
        },
    )


def freeze(root: Path) -> None:
    """Freeze all terminal prediction states before references can be loaded."""

    protocol_path = root / "protocol.json"
    protocol = _load_json(protocol_path)
    _verify_protocol_sources(protocol)
    rows: list[dict[str, Any]] = []
    passed_count = 0
    for row in protocol["rows"]:
        terminal = root / "rows" / row["video_id"] / "terminal.json"
        if not terminal.is_file():
            raise ValueError(f"row has no terminal state: {row['video_id']}")
        payload = _load_json(terminal)
        if payload.get("video_id") != row["video_id"]:
            raise ValueError(f"terminal row identity mismatch: {row['video_id']}")
        if payload.get("status") == "passed":
            passed_count += 1
        elif payload.get("status") not in {"provider_error", "contract_error"}:
            raise ValueError(f"unknown terminal status: {row['video_id']}")
        rows.append(
            {
                "video_id": row["video_id"],
                "status": payload["status"],
                "terminal_sha256": _sha256(terminal),
            }
        )
    _write_once(
        root / "predictions-freeze.json",
        {
            "schema_version": 1,
            "experiment_id": protocol["experiment_id"],
            "status": "frozen",
            "frozen_at": datetime.now(UTC).isoformat(),
            "row_count": len(rows),
            "passed_count": passed_count,
            "failed_count": len(rows) - passed_count,
            "protocol_sha256": _sha256(protocol_path),
            "rows": rows,
        },
    )
    freeze_evidence(root)


def freeze_evidence(root: Path) -> None:
    """Additively bind every persisted per-row request/response artifact."""

    prediction_freeze_path = root / "predictions-freeze.json"
    prediction_freeze = _load_json(prediction_freeze_path)
    protocol = _load_json(root / "protocol.json")
    frozen_by_id = {
        row["video_id"]: row for row in prediction_freeze.get("rows", [])
    }
    rows: list[dict[str, Any]] = []
    for protocol_row in protocol["rows"]:
        video_id = protocol_row["video_id"]
        status = frozen_by_id[video_id]["status"]
        row_root = root / "rows" / video_id
        required = {"request.json", "terminal.json"}
        if status == "passed":
            required.update({"raw-response.txt", "metadata.json", "parsed.json"})
        elif status == "contract_error":
            required.update({"raw-response.txt", "metadata.json"})
        missing = [name for name in required if not (row_root / name).is_file()]
        if missing:
            raise ValueError(f"row evidence artifacts missing: {video_id}: {missing}")
        artifacts = {
            name: _sha256(row_root / name)
            for name in ROW_ARTIFACT_NAMES
            if (row_root / name).is_file()
        }
        rows.append({"video_id": video_id, "status": status, "artifacts": artifacts})
    _write_once(
        root / "evidence-freeze.json",
        {
            "schema_version": 1,
            "experiment_id": protocol["experiment_id"],
            "status": "frozen",
            "frozen_at": datetime.now(UTC).isoformat(),
            "prediction_freeze_sha256": _sha256(prediction_freeze_path),
            "rows": rows,
        },
    )


def verify_evidence_freeze(root: Path) -> None:
    """Reject drift in any per-row artifact bound by the evidence sidecar."""

    freeze = _load_json(root / "evidence-freeze.json")
    if freeze.get("prediction_freeze_sha256") != _sha256(
        root / "predictions-freeze.json"
    ):
        raise ValueError("evidence freeze does not bind prediction freeze")
    protocol = _load_json(root / "protocol.json")
    frozen_rows = {row["video_id"]: row for row in freeze.get("rows", [])}
    protocol_ids = {row["video_id"] for row in protocol["rows"]}
    if set(frozen_rows) != protocol_ids:
        raise ValueError("evidence freeze row IDs do not match protocol")
    for video_id, frozen in frozen_rows.items():
        row_root = root / "rows" / video_id
        artifacts = frozen.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"invalid frozen row artifacts: {video_id}")
        current_names = {
            name for name in ROW_ARTIFACT_NAMES if (row_root / name).is_file()
        }
        if current_names != set(artifacts):
            raise ValueError(f"frozen row artifact set changed: {video_id}")
        for name, digest in artifacts.items():
            if _sha256(row_root / name) != digest:
                raise ValueError(f"frozen row artifact changed: {video_id}/{name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--selection", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--experiment-id")
    row_parser = subparsers.add_parser("run-row")
    row_parser.add_argument("--root", type=Path, required=True)
    row_parser.add_argument("--video-id", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    evidence_freeze_parser = subparsers.add_parser("freeze-evidence")
    evidence_freeze_parser.add_argument("--root", type=Path, required=True)
    gate_parser = subparsers.add_parser("freeze-gate")
    gate_parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare(
            args.source_root,
            args.selection,
            args.output,
            experiment_id=args.experiment_id,
        )
    elif args.command == "run-row":
        run_row(args.root, args.video_id)
    elif args.command == "freeze":
        freeze(args.root)
    elif args.command == "freeze-evidence":
        freeze_evidence(args.root)
    else:
        freeze_gate(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
