#!/usr/bin/env python3
"""Run the issue #35 premise-tolerant treatment over frozen map evidence."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_ROOT))

from run_grounded_map_answering import (  # noqa: E402
    MAX_ANSWER_WORDS,
    MODEL,
    _load_json,
    _response_metadata,
    _sha256,
    _write_once,
    _write_text_once,
    freeze_evidence,
    verify_evidence_freeze,
)


EXPECTED_ROLE_COUNTS = {
    "fix_false_abstention": 5,
    "detectable_insufficiency": 2,
    "known_limit": 1,
    "stability_control": 3,
}
EXPECTED_ROWS = sum(EXPECTED_ROLE_COUNTS.values())
PREMISE_STATUSES = {
    "supported",
    "harmless_locator_variance",
    "answer_blocking",
}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "question_interpretation": {"type": "string"},
        "premise_status": {
            "type": "string",
            "enum": sorted(PREMISE_STATUSES),
        },
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
        "premise_status",
        "answer_claims",
        "conflicts_or_missing_qualifiers",
        "needs_verification",
        "answer",
    ],
}
CallFunction = Callable[[str], tuple[str, dict[str, Any]]]


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def answer_prompt(question: str, evidence: Sequence[Mapping[str, str]]) -> str:
    """Build the frozen premise-tolerant question/evidence prompt."""

    evidence_text = "\n".join(
        f"[{item['evidence_id']}] {item['text']}" for item in evidence
    )
    return f"""Answer using only the supplied frozen evidence lines.

Separate two judgments:
1. Resolve the question's locator or premise to one relevant event or entity.
2. Decide whether the requested answer itself is supported.

Set premise_status to supported when the locator is directly supported, harmless_locator_variance when wording or an incidental attribute differs but one relevant event and answer remain clear, or answer_blocking when the mismatch changes the requested answer, leaves multiple plausible referents or successors, or makes the answer contradictory. A harmless locator variance does not by itself require verification; record it under conflicts_or_missing_qualifiers and answer when the requested fact is supported.

Allowed entailments are limited to: timestamped scene order supports before/after; quoted speech supports what that speaker explicitly says, including a stated feeling or reason; ordinary paraphrases and aliases may identify one relevant event; and a directly stated event may answer a broad what-happens question.

Do not infer purpose from mechanics, causal reason from mere co-occurrence, unique-entity count from repeated appearances, or missing content from world knowledge. If the requested answer is absent, ambiguous, contradicted, or has multiple plausible candidates, return no claims, set needs_verification true, and set answer exactly to NEEDS VERIFICATION.

Otherwise cite every answer claim with supplied evidence IDs, set needs_verification false, and give one concise top-1 answer of at most {MAX_ANSWER_WORDS} words on one line. The final answer must exactly repeat one cited claim.

Return only the structured JSON fields question_interpretation, premise_status, answer_claims, conflicts_or_missing_qualifiers, needs_verification, and answer.

Question:
{question}

Frozen evidence:
{evidence_text}"""


def validate_answer(
    payload: Mapping[str, Any], evidence_ids: set[str]
) -> dict[str, Any]:
    """Validate premise status, citations, verification, and exact answer binding."""

    required = set(ANSWER_SCHEMA["required"])
    if set(payload) != required:
        raise ValueError("response must contain exactly the required fields")
    interpretation = _nonempty_string(
        payload["question_interpretation"], field="question_interpretation"
    )
    premise_status = payload["premise_status"]
    if premise_status not in PREMISE_STATUSES:
        raise ValueError("premise_status is invalid")
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
            raise ValueError("verification-routed answer requires a reason")
    else:
        if not claims:
            raise ValueError("non-verification answer requires at least one cited claim")
        if answer == "NEEDS VERIFICATION":
            raise ValueError("non-verification answer cannot use verification sentinel")
        if answer not in {item["claim"] for item in claims}:
            raise ValueError("non-verification answer must exactly match one cited claim")
    return {
        "question_interpretation": interpretation,
        "premise_status": premise_status,
        "answer_claims": claims,
        "conflicts_or_missing_qualifiers": conflicts,
        "needs_verification": needs_verification,
        "answer": answer,
    }


def _index_rows(rows: Any, *, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{source} rows must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{source} row must be an object")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in indexed:
            raise ValueError(f"{source} video IDs must be unique")
        indexed[video_id] = dict(row)
    return indexed


def _validate_panel(panel: Any) -> tuple[list[str], dict[str, list[str]]]:
    if not isinstance(panel, Mapping) or set(panel) != set(EXPECTED_ROLE_COUNTS):
        raise ValueError("panel roles are invalid")
    normalized: dict[str, list[str]] = {}
    for role, expected in EXPECTED_ROLE_COUNTS.items():
        values = panel[role]
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError("panel role counts are invalid")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("panel IDs must be non-empty strings")
        normalized[role] = list(values)
    ids = [video_id for role in EXPECTED_ROLE_COUNTS for video_id in normalized[role]]
    if len(ids) != len(set(ids)):
        raise ValueError("panel IDs must be unique")
    return ids, normalized


def prepare(source_protocol: Path, selection_path: Path, output: Path) -> None:
    """Freeze provider-safe rows and a separate label-bearing gate before calls."""

    source_hash = _sha256(source_protocol)
    selection = _load_json(selection_path)
    if selection.get("source_protocol_sha256") != source_hash:
        raise ValueError("source protocol hash does not match selection")
    ids, panel = _validate_panel(selection.get("panel"))
    source = _load_json(source_protocol)
    source_by_id = _index_rows(source.get("rows"), source="source protocol")
    missing = set(ids) - set(source_by_id)
    if missing:
        raise ValueError(f"panel IDs must match source protocol: {sorted(missing)}")
    rows = [
        {
            "video_id": video_id,
            "question": source_by_id[video_id]["question"],
            "evidence": source_by_id[video_id]["evidence"],
        }
        for video_id in ids
    ]
    protocol = {
        "schema_version": 1,
        "experiment_id": selection["experiment_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "condition": "premise-tolerant answer policy over exact frozen v2 map lines",
        "settings": {
            "temperature": 0.0,
            "seed": 42,
            "thinking_level": "HIGH",
            "candidate_count": 1,
            "retry_count": 0,
        },
        "label_isolation": (
            "provider receives only each frozen question and evidence; panel roles and "
            "gate remain in the separate preinference freeze"
        ),
        "source_artifacts": {
            "v2_protocol": {
                "path": str(source_protocol.resolve()),
                "sha256": source_hash,
            },
            "selection": {
                "path": str(selection_path.resolve()),
                "sha256": _sha256(selection_path),
            },
        },
        "prompt_template_sha256": hashlib.sha256(
            answer_prompt(
                "{QUESTION}",
                [{"evidence_id": "E01", "text": "{EVIDENCE}"}],
            ).encode()
        ).hexdigest(),
        "response_schema_sha256": hashlib.sha256(
            json.dumps(ANSWER_SCHEMA, sort_keys=True).encode()
        ).hexdigest(),
        "rows": rows,
    }
    protocol_path = output / "protocol.json"
    _write_once(protocol_path, protocol)
    _write_once(
        output / "preinference-gate-freeze.json",
        {
            "schema_version": 1,
            "experiment_id": selection["experiment_id"],
            "frozen_at": datetime.now(UTC).isoformat(),
            "protocol_sha256": _sha256(protocol_path),
            "selection_sha256": _sha256(selection_path),
            "source_protocol_sha256": source_hash,
            "panel": panel,
            "gate": selection["gate"],
        },
    )


def _verify_protocol(protocol: Mapping[str, Any]) -> None:
    for source in protocol["source_artifacts"].values():
        path = Path(source["path"])
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"frozen source artifact changed: {path}")
    prompt_hash = hashlib.sha256(
        answer_prompt(
            "{QUESTION}",
            [{"evidence_id": "E01", "text": "{EVIDENCE}"}],
        ).encode()
    ).hexdigest()
    if prompt_hash != protocol["prompt_template_sha256"]:
        raise ValueError("frozen prompt template changed")
    schema_hash = hashlib.sha256(
        json.dumps(ANSWER_SCHEMA, sort_keys=True).encode()
    ).hexdigest()
    if schema_hash != protocol["response_schema_sha256"]:
        raise ValueError("frozen response schema changed")


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


def run_row(
    root: Path,
    video_id: str,
    *,
    call_fn: CallFunction | None = None,
) -> None:
    """Issue at most one immutable provider call for a prepared row."""

    if (root / "predictions-freeze.json").exists():
        raise ValueError("predictions are already frozen")
    protocol = _load_json(root / "protocol.json")
    _verify_protocol(protocol)
    row = next(
        (candidate for candidate in protocol["rows"] if candidate["video_id"] == video_id),
        None,
    )
    if row is None:
        raise ValueError("video is not in the frozen protocol")
    row_root = root / "rows" / video_id
    terminal_path = row_root / "terminal.json"
    if terminal_path.exists():
        print(json.dumps({"video_id": video_id, "status": "already-terminal"}))
        return
    prompt = answer_prompt(row["question"], row["evidence"])
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
        raw, metadata = provider(prompt)
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
    _write_text_once(row_root / "raw-response.txt", raw)
    _write_once(row_root / "metadata.json", metadata)
    try:
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("provider response must be a JSON object")
        parsed = validate_answer(
            payload,
            {item["evidence_id"] for item in row["evidence"]},
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
            "question": row["question"],
            "status": "passed",
            **parsed,
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        },
    )
    print(json.dumps({"video_id": video_id, "status": "passed"}))


def freeze(root: Path) -> None:
    """Freeze all 11 terminal states and every persisted row artifact."""

    protocol_path = root / "protocol.json"
    protocol = _load_json(protocol_path)
    _verify_protocol(protocol)
    rows: list[dict[str, Any]] = []
    passed_count = 0
    for row in protocol["rows"]:
        terminal = root / "rows" / row["video_id"] / "terminal.json"
        if not terminal.is_file():
            raise ValueError(f"row has no terminal state: {row['video_id']}")
        payload = _load_json(terminal)
        if payload.get("video_id") != row["video_id"]:
            raise ValueError(f"terminal row identity mismatch: {row['video_id']}")
        status = payload.get("status")
        if status == "passed":
            passed_count += 1
        elif status not in {"provider_error", "contract_error"}:
            raise ValueError(f"unknown terminal status: {row['video_id']}")
        rows.append(
            {
                "video_id": row["video_id"],
                "status": status,
                "terminal_sha256": _sha256(terminal),
            }
        )
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("prediction panel must contain exactly 11 rows")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-protocol", type=Path, required=True)
    prepare_parser.add_argument("--selection", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    row_parser = subparsers.add_parser("run-row")
    row_parser.add_argument("--root", type=Path, required=True)
    row_parser.add_argument("--video-id", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify-evidence-freeze")
    verify_parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.source_protocol, args.selection, args.output)
    elif args.command == "run-row":
        run_row(args.root, args.video_id)
    elif args.command == "freeze":
        freeze(args.root)
    else:
        verify_evidence_freeze(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
