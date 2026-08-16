#!/usr/bin/env python3
"""Run the frozen, single-answer task-support verifier.

The runner deliberately has no retry or answer-repair path.  Every provider
attempt is persisted as a terminal row, including malformed responses, and the
prediction freeze binds the resulting bytes before any downstream scoring.
"""

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
EXPECTED_ROWS = 12
MAX_ANSWER_WORDS = 24
VERIFICATION_SENTINEL = "NEEDS VERIFICATION"
SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_SETTINGS: dict[str, Any] = {
    "model": MODEL,
    "temperature": 0.0,
    "seed": 42,
    "thinking_level": "HIGH",
    "candidate_count": 1,
    "max_output_tokens": 2048,
    "retry_count": 0,
}
CONFLICT_KINDS = {
    "harmless_alias",
    "missing_fact",
    "ambiguous",
    "contradiction",
    "missing_qualifier",
}
ROW_ARTIFACT_NAMES = (
    "request.json",
    "raw-response.txt",
    "metadata.json",
    "parsed.json",
    "terminal.json",
)

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question_interpretation": {
            "type": "object",
            "properties": {
                "requested_entity": {"type": "string"},
                "requested_fact": {"type": "string"},
                "qualifier": {"type": "string"},
                "answer_unit": {"type": "string"},
            },
            "required": [
                "requested_entity",
                "requested_fact",
                "qualifier",
                "answer_unit",
            ],
        },
        "cited_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string"},
                    "claim": {"type": "string"},
                },
                "required": ["evidence_id", "claim"],
            },
        },
        "conflicts_or_missing_qualifiers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": sorted(CONFLICT_KINDS)},
                    "blocking": {"type": "boolean"},
                    "detail": {"type": "string"},
                },
                "required": ["kind", "blocking", "detail"],
            },
        },
        "needs_verification": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": [
        "question_interpretation",
        "cited_evidence",
        "conflicts_or_missing_qualifiers",
        "needs_verification",
        "answer",
    ],
}

CallFunction = Callable[
    [str],
    tuple[str, dict[str, Any]]
    | tuple[Mapping[str, Any] | None, str, dict[str, Any]],
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


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _evidence_claim(item: Mapping[str, Any], *, field: str) -> str:
    claim = item.get("claim")
    if claim is None:
        # Task 2 uses ``claim``; accepting ``text`` keeps the prompt helper
        # usable with older frozen map projections without changing the output
        # contract.
        claim = item.get("text")
    return _nonempty_string(claim, field=f"{field}.claim")


def build_prompt(
    question: str, evidence: Sequence[Mapping[str, str]]
) -> str:
    """Build the exact label-free question-plus-evidence treatment prompt."""

    question_text = _nonempty_string(question, field="question")
    lines: list[str] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise ValueError(f"evidence[{index}] must be an object")
        evidence_id = _nonempty_string(
            item.get("evidence_id"), field=f"evidence[{index}].evidence_id"
        )
        claim = _evidence_claim(item, field=f"evidence[{index}]")
        lines.append(f"[{evidence_id}] {claim}")
    evidence_text = "\n".join(lines)
    return f"""Answer the question using only the supplied frozen evidence lines.

First identify the requested entity, fact or action, temporal or speech qualifier,
and answer unit. Answer only when one cited evidence interpretation uniquely
supports that complete requested tuple. If the fact, qualifier, or entity is
missing, ambiguous, contradictory, or has more than one plausible answer, route
to verification.

The provider input is label-free: do not use labels, reference answers, direct
predictions, scores, audit verdicts, or reference guessing. Do not infer purpose,
cause, an exact count, speech or an exact quotation, identity, or any other fact
not stated by the evidence. Do not invent a qualifier or use world knowledge.
Return exactly one answer or the exact sentinel {VERIFICATION_SENTINEL}; never
return multiple candidate answers or a list of alternatives. A non-verification
answer must exactly repeat one cited evidence claim and cite its evidence ID.

Return exactly these five top-level fields: question_interpretation, cited_evidence,
conflicts_or_missing_qualifiers, needs_verification, and answer. The
question_interpretation object must contain requested_entity, requested_fact,
qualifier, and answer_unit. If verification is needed, return no cited evidence,
record each conflict as an object with kind, blocking, and detail, set
needs_verification to true, and set answer exactly to {VERIFICATION_SENTINEL}.
For an answered row, every conflict must be a non-blocking harmless_alias
conflict. Otherwise cite at least one evidence line, set needs_verification to
false, and return one concise answer of at most {MAX_ANSWER_WORDS} words.

Question:
{question_text}

Frozen evidence:
{evidence_text}"""


def _schema_sha256() -> str:
    return hashlib.sha256(
        json.dumps(ANSWER_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def provider_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact ordered provider-safe row projection."""

    projection = [
        {
            "video_id": row["video_id"],
            "question": row["question"],
            "evidence": [
                {
                    "evidence_id": evidence["evidence_id"],
                    "claim": evidence["claim"],
                }
                for evidence in row["evidence"]
            ],
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _prompt_template_sha256() -> str:
    return hashlib.sha256(
        build_prompt(
            "{QUESTION}",
            [{"evidence_id": "E01", "claim": "{EVIDENCE}"}],
        ).encode()
    ).hexdigest()


_INTERPRETATION_FIELDS = {
    "requested_entity",
    "requested_fact",
    "qualifier",
    "answer_unit",
}


def validate_answer(payload: Any, evidence_ids: set[str]) -> dict[str, Any]:
    """Validate the single top-1 answer and deterministic verification route."""

    if not isinstance(payload, Mapping):
        raise ValueError("response must be a JSON object")
    required = set(ANSWER_SCHEMA["required"])
    if set(payload) != required:
        raise ValueError("response must contain exactly the required fields")

    interpretation_payload = payload["question_interpretation"]
    if not isinstance(interpretation_payload, Mapping) or set(
        interpretation_payload
    ) != _INTERPRETATION_FIELDS:
        raise ValueError("question_interpretation fields are invalid")
    interpretation = {
        key: _nonempty_string(
            interpretation_payload[key], field=f"question_interpretation.{key}"
        )
        for key in sorted(_INTERPRETATION_FIELDS)
    }

    cited_payload = payload["cited_evidence"]
    if not isinstance(cited_payload, list):
        raise ValueError("cited_evidence must be a list")
    cited: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(cited_payload):
        if not isinstance(item, Mapping) or set(item) != {"evidence_id", "claim"}:
            raise ValueError(f"cited_evidence[{index}] fields are invalid")
        evidence_id = _nonempty_string(
            item["evidence_id"], field=f"cited_evidence[{index}].evidence_id"
        )
        if evidence_id in seen_ids:
            raise ValueError("cited evidence IDs must be unique")
        if evidence_id not in evidence_ids:
            raise ValueError(f"unknown evidence ID: {evidence_id}")
        seen_ids.add(evidence_id)
        cited.append(
            {
                "evidence_id": evidence_id,
                "claim": _nonempty_string(
                    item["claim"], field=f"cited_evidence[{index}].claim"
                ),
            }
        )

    conflicts_payload = payload["conflicts_or_missing_qualifiers"]
    if not isinstance(conflicts_payload, list):
        raise ValueError("conflicts_or_missing_qualifiers must be a list")
    conflicts: list[dict[str, Any]] = []
    for index, item in enumerate(conflicts_payload):
        if not isinstance(item, Mapping) or set(item) != {"kind", "blocking", "detail"}:
            raise ValueError(f"conflict[{index}] fields are invalid")
        kind = item["kind"]
        if kind not in CONFLICT_KINDS:
            raise ValueError(f"conflict[{index}] kind is invalid")
        blocking = item["blocking"]
        if not isinstance(blocking, bool):
            raise ValueError(f"conflict[{index}] blocking must be boolean")
        conflicts.append(
            {
                "kind": kind,
                "blocking": blocking,
                "detail": _nonempty_string(
                    item["detail"], field=f"conflict[{index}].detail"
                ),
            }
        )

    needs_verification = payload["needs_verification"]
    if not isinstance(needs_verification, bool):
        raise ValueError("needs_verification must be boolean")
    answer = _nonempty_string(payload["answer"], field="answer")
    if "\n" in answer or "\r" in answer:
        raise ValueError("answer must be one line")
    if len(answer.split()) > MAX_ANSWER_WORDS:
        raise ValueError(f"answer must contain at most {MAX_ANSWER_WORDS} words")

    if needs_verification:
        if answer != VERIFICATION_SENTINEL:
            raise ValueError(
                "verification-routed answer must be NEEDS VERIFICATION"
            )
        if cited:
            raise ValueError("verification-routed answer must not contain answer claims")
        if not conflicts or not any(item["blocking"] for item in conflicts):
            raise ValueError("verification-routed answer requires a reason")
    else:
        if not cited:
            raise ValueError("non-verification answer requires cited_evidence")
        if answer == VERIFICATION_SENTINEL:
            raise ValueError("non-verification answer cannot use verification sentinel")
        if answer not in {item["claim"] for item in cited}:
            raise ValueError("answer must exactly match one cited claim")
        if any(item["blocking"] for item in conflicts):
            raise ValueError("answered row has a blocking verification conflict")
        if any(item["kind"] != "harmless_alias" for item in conflicts):
            raise ValueError("answered row conflicts must be harmless_alias")

    return {
        "question_interpretation": interpretation,
        "cited_evidence": cited,
        "conflicts_or_missing_qualifiers": conflicts,
        "needs_verification": needs_verification,
        "answer": answer,
    }


def _index_protocol_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("rows"), list):
        raise ValueError("protocol rows must be a list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload["rows"]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"protocol row {index} must be an object")
        if set(raw) != {"video_id", "question", "evidence"}:
            raise ValueError("protocol rows must contain only video_id, question, evidence")
        video_id = _nonempty_string(raw["video_id"], field=f"protocol row {index}.video_id")
        if video_id in seen:
            raise ValueError(f"protocol contains duplicate video_id: {video_id}")
        seen.add(video_id)
        question = _nonempty_string(raw["question"], field=f"protocol row {index}.question")
        evidence_payload = raw["evidence"]
        if not isinstance(evidence_payload, list) or not evidence_payload:
            raise ValueError(f"protocol row {video_id} requires evidence")
        evidence: list[dict[str, str]] = []
        evidence_seen: set[str] = set()
        for evidence_index, item in enumerate(evidence_payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"protocol evidence {video_id}/{evidence_index} must be an object")
            evidence_id = _nonempty_string(
                item.get("evidence_id"),
                field=f"protocol evidence {video_id}/{evidence_index}.evidence_id",
            )
            if evidence_id in evidence_seen:
                raise ValueError(f"protocol evidence IDs must be unique: {video_id}")
            evidence_seen.add(evidence_id)
            evidence.append({"evidence_id": evidence_id, "claim": _evidence_claim(item, field=f"protocol evidence {video_id}/{evidence_index}")})
        rows.append({"video_id": video_id, "question": question, "evidence": evidence})
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"protocol must contain exactly {EXPECTED_ROWS} rows")
    return rows


def _load_inputs(protocol_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    protocol_path = protocol_root / "protocol.json"
    treatment_path = protocol_root / "treatment-freeze.json"
    selection_path = protocol_root / "selection-freeze.json"
    authorization_path = protocol_root / "provider-authorization.json"
    protocol = _load_json(protocol_path)
    rows = _index_protocol_rows(protocol)
    treatment = _load_json(treatment_path)
    selection = _load_json(selection_path)
    authorization = _load_json(authorization_path) if authorization_path.is_file() else None
    if not isinstance(treatment, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("treatment and selection freezes must be JSON objects")
    _validate_freeze_contract(
        protocol,
        rows,
        treatment,
        selection,
        authorization,
        protocol_path=protocol_path,
        treatment_path=treatment_path,
        selection_path=selection_path,
        authorization_path=authorization_path,
    )
    return dict(protocol), rows, dict(treatment), dict(selection)


def _validate_freeze_contract(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    treatment: Mapping[str, Any],
    selection: Mapping[str, Any],
    authorization: Mapping[str, Any] | None,
    *,
    protocol_path: Path,
    treatment_path: Path,
    selection_path: Path,
    authorization_path: Path,
) -> None:
    """Validate the immutable Task 2 -> Task 1 boundary before any call."""

    if authorization is None:
        raise ValueError(f"provider authorization is required: {authorization_path}")
    if authorization.get("status") != "authorized":
        raise ValueError("provider authorization status must be exactly authorized")
    if treatment.get("status") != "frozen":
        raise ValueError("treatment freeze must have exact status=frozen")
    if selection.get("status") != "frozen":
        raise ValueError("selection freeze must have exact status=frozen")
    if treatment.get("runner_sha256") != _sha256(SCRIPT_PATH):
        raise ValueError("treatment freeze runner hash does not match current runner")
    if treatment.get("prompt_template_sha256") != _prompt_template_sha256():
        raise ValueError("treatment freeze prompt hash does not match current prompt")
    if treatment.get("response_schema_sha256") != _schema_sha256():
        raise ValueError("treatment freeze schema hash does not match current schema")
    if treatment.get("settings") != EXPECTED_SETTINGS:
        raise ValueError("treatment freeze settings do not match frozen settings")
    if protocol.get("model") != MODEL:
        raise ValueError("protocol model does not match frozen model")
    if protocol.get("settings") != EXPECTED_SETTINGS:
        raise ValueError("protocol settings do not match frozen settings")
    if protocol.get("prompt_template_sha256") != _prompt_template_sha256():
        raise ValueError("protocol prompt hash does not match current prompt")
    if protocol.get("response_schema_sha256") != _schema_sha256():
        raise ValueError("protocol schema hash does not match current schema")
    if protocol.get("treatment_freeze_sha256") != _sha256(treatment_path):
        raise ValueError("protocol treatment freeze hash does not match current freeze")
    if protocol.get("selection_freeze_sha256") != _sha256(selection_path):
        raise ValueError("protocol selection freeze hash does not match current freeze")
    if selection.get("treatment_freeze_sha256") != _sha256(treatment_path):
        raise ValueError("selection treatment freeze hash does not match current freeze")
    if selection.get("provider_rows_sha256") != provider_rows_sha256(rows):
        raise ValueError("selection provider rows digest does not match protocol rows")
    selected = selection.get("selected_rows")
    if not isinstance(selected, list) or len(selected) != len(rows):
        raise ValueError("selection freeze selected_rows do not match protocol")
    selected_ids: list[str] = []
    for item in selected:
        if isinstance(item, str):
            selected_ids.append(item)
        elif isinstance(item, Mapping) and isinstance(item.get("video_id"), str):
            selected_ids.append(item["video_id"])
        else:
            raise ValueError("selection freeze selected_rows are invalid")
    protocol_ids = [row["video_id"] for row in rows]
    if selected_ids != protocol_ids:
        raise ValueError("selection freeze selected_rows do not match protocol rows")
    selection_protocol_hash = selection.get("protocol_sha256")
    if selection_protocol_hash is not None and selection_protocol_hash != _sha256(protocol_path):
        raise ValueError("selection protocol hash does not match current protocol")
    _validate_authorization_contract(
        authorization,
        protocol,
        rows,
        treatment,
        selection,
        protocol_path=protocol_path,
        treatment_path=treatment_path,
        selection_path=selection_path,
        authorization_path=authorization_path,
    )


def _validate_authorization_contract(
    authorization: Mapping[str, Any] | None,
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    treatment: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    protocol_path: Path,
    treatment_path: Path,
    selection_path: Path,
    authorization_path: Path,
) -> None:
    """Require an independently written final authorization before calls."""

    del protocol, treatment
    if authorization is None:
        raise ValueError(f"provider authorization is required: {authorization_path}")
    if authorization.get("status") != "authorized":
        raise ValueError("provider authorization status must be exactly authorized")
    expected = {
        "treatment_freeze_sha256": _sha256(treatment_path),
        "selection_freeze_sha256": _sha256(selection_path),
        "protocol_sha256": _sha256(protocol_path),
        "provider_rows_sha256": provider_rows_sha256(rows),
    }
    for field, digest in expected.items():
        if authorization.get(field) != digest:
            label = field.replace("_sha256", " hash").replace("provider_rows", "provider rows digest")
            raise ValueError(f"provider authorization {label} does not match current bytes")
    manual_audit_sha256 = selection.get("manual_audit_sha256")
    if not isinstance(manual_audit_sha256, str) or not manual_audit_sha256:
        raise ValueError("selection freeze must bind manual audit SHA-256")
    if authorization.get("manual_audit_sha256") != manual_audit_sha256:
        raise ValueError(
            "provider authorization manual audit SHA-256 does not match selection"
        )
    if authorization.get("promotion_gate") != "PASS":
        raise ValueError("provider authorization promotion gate must be PASS")


def _response_metadata(response: Any, elapsed_seconds: float, raw: str) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    metadata: dict[str, Any] = {
        "elapsed_seconds": elapsed_seconds,
        "response_text_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }
    if usage is not None:
        for source, target in (
            ("prompt_token_count", "prompt_token_count"),
            ("candidates_token_count", "candidates_token_count"),
            ("total_token_count", "total_token_count"),
        ):
            value = getattr(usage, source, None)
            if value is not None:
                metadata[target] = value
    response_id = getattr(response, "response_id", None)
    if response_id:
        metadata["response_id"] = response_id
    return metadata


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


def _provider_result(result: Any) -> tuple[Mapping[str, Any] | None, str, dict[str, Any]]:
    if not isinstance(result, tuple) or len(result) not in {2, 3}:
        raise ValueError("provider call must return (raw, metadata) or (payload, raw, metadata)")
    if len(result) == 2:
        raw, metadata = result
        payload = None
    else:
        payload, raw, metadata = result
    if not isinstance(raw, str):
        raise ValueError("provider raw response must be text")
    if not isinstance(metadata, Mapping):
        raise ValueError("provider metadata must be an object")
    if payload is not None and not isinstance(payload, Mapping):
        raise ValueError("provider parsed payload must be an object")
    return payload, raw, dict(metadata)


def _copy_input_once(path: Path, destination: Path) -> None:
    if path.resolve() == destination.resolve():
        return
    _write_text_once(destination, path.read_text(encoding="utf-8"))


def run(
    protocol_root: Path,
    output_root: Path,
    call: CallFunction | None = None,
) -> None:
    """Run exactly one provider attempt for every selected protocol row."""

    protocol_root = Path(protocol_root)
    output_root = Path(output_root)
    same_root = output_root.resolve() == protocol_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not same_root:
        raise ValueError("output already has artifacts; refusing to overwrite")
    if same_root and (
        (output_root / "rows").exists()
        or (output_root / "predictions-freeze.json").exists()
    ):
        raise ValueError("output already has artifacts; refusing to overwrite")
    protocol, rows, treatment, selection = _load_inputs(protocol_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_input_once(protocol_root / "protocol.json", output_root / "protocol.json")
    _copy_input_once(
        protocol_root / "treatment-freeze.json",
        output_root / "treatment-freeze.json",
    )
    _copy_input_once(
        protocol_root / "selection-freeze.json",
        output_root / "selection-freeze.json",
    )
    _copy_input_once(
        protocol_root / "provider-authorization.json",
        output_root / "provider-authorization.json",
    )
    provider = call or _call_provider
    for row in rows:
        video_id = row["video_id"]
        row_root = output_root / "rows" / video_id
        terminal_path = row_root / "terminal.json"
        prompt = build_prompt(row["question"], row["evidence"])
        _write_once(
            row_root / "request.json",
            {
                "model": protocol.get("model", MODEL),
                "question": row["question"],
                "evidence": row["evidence"],
                "prompt": prompt,
                "response_schema": ANSWER_SCHEMA,
            },
        )
        try:
            payload, raw, metadata = _provider_result(provider(prompt))
        except Exception as error:
            _write_once(
                row_root / "metadata.json",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            _write_once(
                terminal_path,
                {
                    "video_id": video_id,
                    "status": "provider_error",
                    "error_type": type(error).__name__,
                },
            )
            continue
        _write_text_once(row_root / "raw-response.txt", raw)
        _write_once(row_root / "metadata.json", metadata)
        try:
            # The provider's raw bytes are authoritative.  A test seam may
            # return a parsed convenience payload, but it can never bypass
            # malformed JSON or alter what was actually persisted.
            parsed_payload = json.loads(raw)
            parsed = validate_answer(
                parsed_payload,
                {item["evidence_id"] for item in row["evidence"]},
            )
        except json.JSONDecodeError as error:
            _write_once(
                terminal_path,
                {
                    "video_id": video_id,
                    "status": "invalid_json",
                    "error_type": type(error).__name__,
                },
            )
            continue
        except ValueError as error:
            _write_once(
                terminal_path,
                {
                    "video_id": video_id,
                    "status": "contract_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            continue
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


def _verify_freeze_contents(root: Path, freeze: Mapping[str, Any]) -> None:
    protocol_path = root / "protocol.json"
    protocol = _load_json(protocol_path)
    rows = _index_protocol_rows(protocol)
    treatment_path = root / "treatment-freeze.json"
    selection_path = root / "selection-freeze.json"
    treatment = _load_json(treatment_path)
    selection = _load_json(selection_path)
    _validate_freeze_contract(
        protocol,
        rows,
        treatment,
        selection,
        _load_json(root / "provider-authorization.json"),
        protocol_path=protocol_path,
        treatment_path=treatment_path,
        selection_path=selection_path,
        authorization_path=root / "provider-authorization.json",
    )
    if freeze.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("predictions freeze does not bind current protocol")
    expected_freeze_hashes = {
        "treatment_freeze_sha256": treatment_path,
        "selection_freeze_sha256": selection_path,
        "provider_authorization_sha256": root / "provider-authorization.json",
    }
    for key, path in expected_freeze_hashes.items():
        if freeze.get(key) != _sha256(path):
            raise ValueError(f"predictions freeze does not bind current {path.name}")
    frozen_rows = freeze.get("rows")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != len(rows):
        raise ValueError("predictions freeze row count does not match protocol")
    frozen_by_id = {item.get("video_id"): item for item in frozen_rows if isinstance(item, Mapping)}
    if set(frozen_by_id) != {row["video_id"] for row in rows}:
        raise ValueError("predictions freeze row IDs do not match protocol")
    passed = 0
    for row in rows:
        video_id = row["video_id"]
        frozen = frozen_by_id[video_id]
        row_root = root / "rows" / video_id
        status = frozen.get("status")
        if status not in {"passed", "invalid_json", "contract_error", "provider_error"}:
            raise ValueError(f"unknown terminal status: {video_id}")
        terminal_path = row_root / "terminal.json"
        if not terminal_path.is_file():
            raise ValueError(f"row has no terminal state: {video_id}")
        terminal = _load_json(terminal_path)
        if terminal.get("video_id") != video_id or terminal.get("status") != status:
            raise ValueError(f"terminal row identity/status mismatch: {video_id}")
        required = {"request.json", "metadata.json", "terminal.json"}
        if status in {"passed", "invalid_json", "contract_error"}:
            required.add("raw-response.txt")
        if status == "passed":
            required.add("parsed.json")
            validate_answer(
                _load_json(row_root / "parsed.json"),
                {item["evidence_id"] for item in row["evidence"]},
            )
            passed += 1
        artifacts = frozen.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"frozen row artifacts are invalid: {video_id}")
        current_names = {
            name for name in ROW_ARTIFACT_NAMES if (row_root / name).is_file()
        }
        if current_names != set(artifacts):
            raise ValueError(f"frozen row artifact set changed: {video_id}")
        if not required.issubset(current_names):
            raise ValueError(f"row evidence artifacts missing: {video_id}")
        for name, digest in artifacts.items():
            if _sha256(row_root / name) != digest:
                raise ValueError(f"frozen row artifact changed: {video_id}/{name}")
    if freeze.get("passed_count") != passed:
        raise ValueError("predictions freeze passed count is inconsistent")


def verify_predictions_freeze(root: Path) -> None:
    """Freeze all terminal rows once, then reject any subsequent byte drift."""

    root = Path(root)
    protocol_path = root / "protocol.json"
    protocol = _load_json(protocol_path)
    rows = _index_protocol_rows(protocol)
    treatment_path = root / "treatment-freeze.json"
    selection_path = root / "selection-freeze.json"
    treatment = _load_json(treatment_path)
    selection = _load_json(selection_path)
    if not isinstance(treatment, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("treatment and selection freezes must be JSON objects")
    _validate_freeze_contract(
        protocol,
        rows,
        treatment,
        selection,
        _load_json(root / "provider-authorization.json"),
        protocol_path=protocol_path,
        treatment_path=treatment_path,
        selection_path=selection_path,
        authorization_path=root / "provider-authorization.json",
    )
    frozen_rows: list[dict[str, Any]] = []
    passed_count = 0
    for row in rows:
        video_id = row["video_id"]
        terminal_path = root / "rows" / video_id / "terminal.json"
        if not terminal_path.is_file():
            raise ValueError(f"row has no terminal state: {video_id}")
        terminal = _load_json(terminal_path)
        status = terminal.get("status")
        if terminal.get("video_id") != video_id:
            raise ValueError(f"terminal row identity mismatch: {video_id}")
        if status not in {"passed", "invalid_json", "contract_error", "provider_error"}:
            raise ValueError(f"unknown terminal status: {video_id}")
        if status == "passed":
            passed_count += 1
        row_root = root / "rows" / video_id
        required = {"request.json", "metadata.json", "terminal.json"}
        if status in {"passed", "invalid_json", "contract_error"}:
            required.add("raw-response.txt")
        if status == "passed":
            required.add("parsed.json")
        missing = [name for name in required if not (row_root / name).is_file()]
        if missing:
            raise ValueError(f"row evidence artifacts missing: {video_id}: {missing}")
        frozen_rows.append(
            {
                "video_id": video_id,
                "status": status,
                "artifacts": {
                    name: _sha256(row_root / name)
                    for name in ROW_ARTIFACT_NAMES
                    if (row_root / name).is_file()
                },
            }
        )
    prediction_path = root / "predictions-freeze.json"
    if prediction_path.exists():
        freeze = _load_json(prediction_path)
        _verify_freeze_contents(root, freeze)
        return
    _write_once(
        prediction_path,
        {
            "schema_version": 1,
            "experiment_id": protocol.get("experiment_id"),
            "status": "frozen",
            "frozen_at": datetime.now(UTC).isoformat(),
            "row_count": len(rows),
            "completed_count": len(rows),
            "passed_count": passed_count,
            "failed_count": len(rows) - passed_count,
            "protocol_sha256": _sha256(protocol_path),
            "treatment_freeze_sha256": _sha256(treatment_path),
            "selection_freeze_sha256": _sha256(selection_path),
            "provider_authorization_sha256": _sha256(
                root / "provider-authorization.json"
            ),
            "rows": frozen_rows,
        },
    )
    _verify_freeze_contents(root, _load_json(prediction_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol-root", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run(args.protocol_root, args.output)
    else:
        verify_predictions_freeze(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
