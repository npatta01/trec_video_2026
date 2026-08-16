#!/usr/bin/env python3
"""Run the frozen single-versus-multi interpretation issue #35 diagnostic."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_ROOT))

from run_grounded_map_answering import (  # noqa: E402
    MODEL,
    _load_json,
    _response_metadata,
    _sha256,
    _write_once,
    _write_text_once,
    freeze_evidence,
    verify_evidence_freeze,
)
from run_premise_tolerant_grounded_answering import (  # noqa: E402
    ANSWER_SCHEMA as SINGLE_SCHEMA,
    validate_answer as validate_single,
)
ARMS = ("single", "multi")
EXPECTED_ROWS = 12
INTERPRETATION_TYPES = (
    "literal_premise",
    "unique_event_paraphrase",
    "conservative_verification",
)
CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretation_type": {"type": "string", "enum": list(INTERPRETATION_TYPES)},
        **SINGLE_SCHEMA["properties"],
    },
    "required": ["interpretation_type", *SINGLE_SCHEMA["required"]],
}
MULTI_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": CANDIDATE_SCHEMA,
        }
    },
    "required": ["candidates"],
}
CallFunction = Callable[[str], tuple[str, dict[str, Any]]]


def normalize_answer(text: str) -> str:
    """Normalize case, whitespace, and punctuation for exact agreement groups."""

    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _evidence_text(evidence: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(f"[{item['evidence_id']}] {item['text']}" for item in evidence)


def single_prompt(question: str, evidence: Sequence[Mapping[str, str]]) -> str:
    return f"""Answer using only the complete supplied frozen scene-map evidence.

First resolve the question to one relevant event or entity. A harmless wording, locator, or alias variance may be accepted only when exactly one event is compatible with the requested relation and answer type. Record that variance under conflicts_or_missing_qualifiers. If the mismatch changes the requested answer, leaves multiple plausible referents, or the requested fact is absent, set premise_status to answer_blocking, needs_verification true, return no claims, and answer exactly NEEDS VERIFICATION.

Do not infer purpose from mechanics, cause from co-occurrence, unique counts, exact speech, musical identity, geographic origin, or omitted actions. Otherwise cite every answer claim using supplied evidence IDs, set needs_verification false, and make the concise final answer exactly repeat one cited claim.

Return only question_interpretation, premise_status, answer_claims, conflicts_or_missing_qualifiers, needs_verification, and answer.

Question:
{question}

Complete frozen evidence:
{_evidence_text(evidence)}"""


def multi_prompt(question: str, evidence: Sequence[Mapping[str, str]]) -> str:
    return f"""Generate exactly three independent answer candidates using only the complete supplied frozen scene-map evidence. Do not select a winner.

Candidate 1 must use interpretation_type literal_premise and interpret the question wording literally. Candidate 2 must use interpretation_type unique_event_paraphrase and may resolve a harmless locator, wording, or entity-alias variance only when exactly one compatible event and answer type exist. Candidate 3 must use interpretation_type conservative_verification and independently test whether the requested fact requires verification.

For every candidate, record question_interpretation, premise_status, cited answer_claims, conflicts_or_missing_qualifiers, needs_verification, and one concise answer. A harmless_locator_variance requires a recorded conflict. An answer_blocking candidate must return no claims, set needs_verification true, and answer exactly NEEDS VERIFICATION. Every non-verification final answer must exactly repeat one cited claim.

Do not infer purpose from mechanics, cause from co-occurrence, unique counts, exact speech, musical identity, geographic origin, or omitted actions. Return only an object with the candidates array.

Question:
{question}

Complete frozen evidence:
{_evidence_text(evidence)}"""


def validate_candidate(payload: Mapping[str, Any], evidence_ids: set[str]) -> dict[str, Any]:
    required = set(CANDIDATE_SCHEMA["required"])
    if set(payload) != required:
        raise ValueError("candidate must contain exactly the required fields")
    kind = payload.get("interpretation_type")
    if kind not in INTERPRETATION_TYPES:
        raise ValueError("candidate interpretation type is invalid")
    answer_fields = {key: payload[key] for key in SINGLE_SCHEMA["required"]}
    return {"interpretation_type": kind, **validate_single(answer_fields, evidence_ids)}


def elect_candidates(payload: Mapping[str, Any], evidence_ids: set[str]) -> dict[str, Any]:
    """Apply the frozen reference-free election rule and preserve its full trace."""

    if set(payload) != {"candidates"} or not isinstance(payload["candidates"], list):
        raise ValueError("multi response must contain only a candidates array")
    candidates = payload["candidates"]
    if len(candidates) != 3 or any(not isinstance(item, Mapping) for item in candidates):
        raise ValueError("multi response requires exactly three candidate objects")
    kinds = [item.get("interpretation_type") for item in candidates]
    if set(kinds) != set(INTERPRETATION_TYPES) or len(kinds) != len(set(kinds)):
        raise ValueError("candidate interpretation types must be exact and unique")

    trace: list[dict[str, Any]] = []
    supported: list[tuple[int, dict[str, Any], str]] = []
    valid_verifications = 0
    for index, raw in enumerate(candidates):
        try:
            candidate = validate_candidate(raw, evidence_ids)
        except ValueError as error:
            trace.append(
                {
                    "candidate_index": index,
                    "interpretation_type": raw.get("interpretation_type"),
                    "valid": False,
                    "rejection_reason": str(error),
                    "raw_candidate": dict(raw),
                }
            )
            continue
        normalized = None
        if candidate["needs_verification"]:
            valid_verifications += 1
        else:
            normalized = normalize_answer(candidate["answer"])
            supported.append((index, candidate, normalized))
        trace.append(
            {
                "candidate_index": index,
                "interpretation_type": candidate["interpretation_type"],
                "valid": True,
                "rejection_reason": None,
                "normalized_answer": normalized,
                "candidate": candidate,
            }
        )

    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, candidate, normalized in supported:
        groups.setdefault(normalized, []).append((index, candidate))
    agreeing = [group for group in groups.values() if len(group) >= 2]
    if len(agreeing) == 1:
        group = agreeing[0]
        chosen_index, chosen = min(
            group,
            key=lambda item: (
                len(item[1]["answer"].split()),
                len(item[1]["answer"]),
                item[0],
            ),
        )
        del chosen_index
        return {
            "candidate_trace": trace,
            "normalized_groups": {key: [item[0] for item in value] for key, value in groups.items()},
            "election_reason": "supported_candidate_agreement",
            "elected_candidate_indices": [item[0] for item in group],
            "needs_verification": False,
            "answer": chosen["answer"],
            "answer_claims": chosen["answer_claims"],
        }
    if len(supported) == 1 and valid_verifications == sum(item["valid"] for item in trace) - 1:
        index, chosen, _ = supported[0]
        return {
            "candidate_trace": trace,
            "normalized_groups": {key: [item[0] for item in value] for key, value in groups.items()},
            "election_reason": "sole_supported_answer",
            "elected_candidate_indices": [index],
            "needs_verification": False,
            "answer": chosen["answer"],
            "answer_claims": chosen["answer_claims"],
        }
    reason = "supported_answer_conflict" if len(groups) > 1 else "no_supported_answer"
    return {
        "candidate_trace": trace,
        "normalized_groups": {key: [item[0] for item in value] for key, value in groups.items()},
        "election_reason": reason,
        "elected_candidate_indices": [],
        "needs_verification": True,
        "answer": "NEEDS VERIFICATION",
        "answer_claims": [],
    }


def _prompt_hash(arm: str) -> str:
    function = single_prompt if arm == "single" else multi_prompt
    return hashlib.sha256(
        function("{QUESTION}", [{"evidence_id": "E01", "text": "{EVIDENCE}"}]).encode()
    ).hexdigest()


def _schema(arm: str) -> dict[str, Any]:
    return SINGLE_SCHEMA if arm == "single" else MULTI_SCHEMA


def prepare_arms(root: Path) -> None:
    """Create immutable, label-free arm protocols from the selection protocol."""

    source_path = root / "protocol.source.json"
    source = _load_json(source_path)
    source_files = {
        "protocol_source": source_path,
        "selection_freeze": root / "selection-freeze.json",
        "preinference_gate_freeze": root / "preinference-gate-freeze.json",
        "selection_integrity": root / "selection-integrity.json",
        "runner": Path(__file__).resolve(),
    }
    integrity = _load_json(root / "selection-integrity.json")
    expected = {
        "protocol_source_sha256": _sha256(source_path),
        "selection_freeze_sha256": _sha256(root / "selection-freeze.json"),
        "preinference_gate_freeze_sha256": _sha256(root / "preinference-gate-freeze.json"),
    }
    if integrity != expected:
        raise ValueError("selection integrity mismatch")
    for arm in ARMS:
        protocol = {
            "schema_version": 1,
            "experiment_id": source["experiment_id"],
            "created_at": datetime.now(UTC).isoformat(),
            "arm": arm,
            "model": MODEL,
            "settings": {
                "temperature": 0.0,
                "seed": 42,
                "thinking_level": "HIGH",
                "candidate_count": 1,
                "retry_count": 0,
            },
            "source_artifacts": {
                name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for name, path in source_files.items()
            },
            "prompt_template_sha256": _prompt_hash(arm),
            "response_schema_sha256": hashlib.sha256(
                json.dumps(_schema(arm), sort_keys=True).encode()
            ).hexdigest(),
            "election_rule": "local_deterministic_v1" if arm == "multi" else None,
            "rows": source["rows"],
        }
        _write_once(root / arm / "protocol.json", protocol)


def _verify_protocol(root: Path, arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown arm")
    protocol = _load_json(root / arm / "protocol.json")
    if protocol.get("arm") != arm or protocol.get("prompt_template_sha256") != _prompt_hash(arm):
        raise ValueError("arm protocol prompt integrity mismatch")
    schema_hash = hashlib.sha256(json.dumps(_schema(arm), sort_keys=True).encode()).hexdigest()
    if protocol.get("response_schema_sha256") != schema_hash:
        raise ValueError("arm protocol schema integrity mismatch")
    for source in protocol.get("source_artifacts", {}).values():
        path = Path(source["path"])
        if not path.is_file() or _sha256(path) != source["sha256"]:
            raise ValueError(f"frozen source artifact changed: {path}")
    return protocol


def _call_provider(prompt: str, schema: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
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
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_schema=dict(schema),
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=False,
            ),
        ),
    )
    raw = response.text or ""
    return raw, _response_metadata(response, time.monotonic() - started, raw)


def run_row(root: Path, arm: str, video_id: str, *, call_fn: CallFunction | None = None) -> None:
    arm_root = root / arm
    if (arm_root / "predictions-freeze.json").exists():
        raise ValueError("predictions are already frozen")
    protocol = _verify_protocol(root, arm)
    row = next((item for item in protocol["rows"] if item["video_id"] == video_id), None)
    if row is None:
        raise ValueError("video is not in the frozen protocol")
    row_root = arm_root / "rows" / video_id
    terminal_path = row_root / "terminal.json"
    if terminal_path.exists():
        print(json.dumps({"arm": arm, "video_id": video_id, "status": "already-terminal"}))
        return
    prompt_fn = single_prompt if arm == "single" else multi_prompt
    prompt = prompt_fn(row["question"], row["evidence"])
    _write_once(
        row_root / "request.json",
        {
            "model": protocol["model"],
            "arm": arm,
            "question": row["question"],
            "evidence": row["evidence"],
            "prompt": prompt,
            "response_schema": _schema(arm),
        },
    )
    provider = call_fn or (lambda value: _call_provider(value, _schema(arm)))
    try:
        raw, metadata = provider(prompt)
    except Exception as error:
        _write_once(
            terminal_path,
            {"video_id": video_id, "arm": arm, "status": "provider_error", "error_type": type(error).__name__},
        )
        raise
    _write_text_once(row_root / "raw-response.txt", raw)
    _write_once(row_root / "metadata.json", metadata)
    try:
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("provider response must be a JSON object")
        evidence_ids = {item["evidence_id"] for item in row["evidence"]}
        if arm == "single":
            parsed = validate_single(payload, evidence_ids)
            emitted = {
                "needs_verification": parsed["needs_verification"],
                "answer": parsed["answer"],
                "answer_claims": parsed["answer_claims"],
            }
        else:
            parsed = elect_candidates(payload, evidence_ids)
            emitted = {
                "needs_verification": parsed["needs_verification"],
                "answer": parsed["answer"],
                "answer_claims": parsed["answer_claims"],
            }
    except (json.JSONDecodeError, ValueError) as error:
        _write_once(
            terminal_path,
            {"video_id": video_id, "arm": arm, "status": "contract_error", "error_type": type(error).__name__},
        )
        raise
    _write_once(row_root / "parsed.json", parsed)
    _write_once(
        terminal_path,
        {
            "video_id": video_id,
            "arm": arm,
            "question": row["question"],
            "status": "passed",
            **emitted,
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        },
    )
    print(json.dumps({"arm": arm, "video_id": video_id, "status": "passed"}))


def freeze_arm(root: Path, arm: str, *, expected_rows: int = EXPECTED_ROWS) -> None:
    arm_root = root / arm
    protocol_path = arm_root / "protocol.json"
    protocol = _verify_protocol(root, arm)
    rows = []
    passed = 0
    for row in protocol["rows"]:
        terminal_path = arm_root / "rows" / row["video_id"] / "terminal.json"
        if not terminal_path.is_file():
            raise ValueError(f"row has no terminal state: {arm}/{row['video_id']}")
        terminal = _load_json(terminal_path)
        status = terminal.get("status")
        if terminal.get("video_id") != row["video_id"] or status not in {"passed", "provider_error", "contract_error"}:
            raise ValueError(f"invalid terminal state: {arm}/{row['video_id']}")
        passed += status == "passed"
        rows.append({"video_id": row["video_id"], "status": status, "terminal_sha256": _sha256(terminal_path)})
    if len(rows) != expected_rows:
        raise ValueError(f"arm requires exactly {expected_rows} rows")
    _write_once(
        arm_root / "predictions-freeze.json",
        {
            "schema_version": 1,
            "experiment_id": protocol["experiment_id"],
            "arm": arm,
            "status": "frozen",
            "frozen_at": datetime.now(UTC).isoformat(),
            "row_count": len(rows),
            "passed_count": passed,
            "failed_count": len(rows) - passed,
            "protocol_sha256": _sha256(protocol_path),
            "preinference_gate_freeze_sha256": _sha256(root / "preinference-gate-freeze.json"),
            "rows": rows,
        },
    )
    freeze_evidence(arm_root)


def freeze_combined(root: Path) -> None:
    for arm in ARMS:
        verify_evidence_freeze(root / arm)
        _verify_protocol(root, arm)
    _write_once(
        root / "experiment-integrity.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "selection_integrity_sha256": _sha256(root / "selection-integrity.json"),
            "preinference_gate_freeze_sha256": _sha256(root / "preinference-gate-freeze.json"),
            "arms": {
                arm: {
                    "protocol_sha256": _sha256(root / arm / "protocol.json"),
                    "predictions_freeze_sha256": _sha256(root / arm / "predictions-freeze.json"),
                    "evidence_freeze_sha256": _sha256(root / arm / "evidence-freeze.json"),
                }
                for arm in ARMS
            },
        },
    )


def verify_combined(root: Path) -> None:
    integrity = _load_json(root / "experiment-integrity.json")
    if integrity.get("selection_integrity_sha256") != _sha256(root / "selection-integrity.json"):
        raise ValueError("combined selection integrity mismatch")
    if integrity.get("preinference_gate_freeze_sha256") != _sha256(root / "preinference-gate-freeze.json"):
        raise ValueError("combined gate integrity mismatch")
    for arm in ARMS:
        _verify_protocol(root, arm)
        verify_evidence_freeze(root / arm)
        expected = {
            "protocol_sha256": _sha256(root / arm / "protocol.json"),
            "predictions_freeze_sha256": _sha256(root / arm / "predictions-freeze.json"),
            "evidence_freeze_sha256": _sha256(root / arm / "evidence-freeze.json"),
        }
        if integrity.get("arms", {}).get(arm) != expected:
            raise ValueError(f"combined arm integrity mismatch: {arm}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare-arms")
    prepare_parser.add_argument("--root", type=Path, required=True)
    row_parser = subparsers.add_parser("run-row")
    row_parser.add_argument("--root", type=Path, required=True)
    row_parser.add_argument("--arm", choices=ARMS, required=True)
    row_parser.add_argument("--video-id", required=True)
    freeze_parser = subparsers.add_parser("freeze-arm")
    freeze_parser.add_argument("--root", type=Path, required=True)
    freeze_parser.add_argument("--arm", choices=ARMS, required=True)
    combined_parser = subparsers.add_parser("freeze-combined")
    combined_parser.add_argument("--root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify-combined")
    verify_parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-arms":
        prepare_arms(args.root)
    elif args.command == "run-row":
        run_row(args.root, args.arm, args.video_id)
    elif args.command == "freeze-arm":
        freeze_arm(args.root, args.arm)
    elif args.command == "freeze-combined":
        freeze_combined(args.root)
    else:
        verify_combined(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
