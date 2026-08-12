"""One-call, question-blind verifier for a frozen frontier neutral map."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trec_video_2026.vqa_experiments.temporal_bottleneck import (
    freeze_json_artifact,
    verify_json_artifact,
)
from trec_video_2026.vqa_frontier_map_media import (
    DeliveryMedia,
    prepare_delivery_media,
    verify_delivery_media,
)
from trec_video_2026.vqa_frontier_phase0_contract import (
    NEUTRAL_MAP_SCHEMA,
    NeutralVideoMap,
    _provider_schema,
    validate_neutral_map,
)
from trec_video_2026.vqa_frontier_phase0_runner import load_prepared
from trec_video_2026.vqa_frontier_response_store import parse_prompt_json_response
from trec_video_2026.vqa_submission import MODEL_ID, canonical_sha256
from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    ProviderResponseEvidenceError,
    UploadedFileRecord,
)
from trec_video_2026.vqa_submission_openrouter import (
    OpenRouterRequestPolicy,
    OpenRouterTransport,
)


SCHEMA_VERSION = 1
RUNNER_ID = "frontier-boundary-verifier-v1"
PROMPT_VERSION = "frontier-boundary-verifier-prompt-v1"
ABSOLUTE_CAP_USD = 25.0
PROJECTED_CALL_COST_USD = 0.50
CANONICAL_DATA_ROOT = Path("/home/npatta01/data/competitions/trec_video_2026/data")
VERIFIER_POLICY = OpenRouterRequestPolicy(
    max_output_tokens=16384,
    temperature=0.0,
    allow_fallbacks=False,
    use_structured_outputs=True,
)
VERDICTS = frozenset({"confirmed", "corrected", "unusable"})
ACTIONS = frozenset({"confirmed", "adjusted", "split", "merged", "removed", "added"})
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "annotation",
        "annotations",
        "correctanswer",
        "gold",
        "goldanswer",
        "label",
        "labels",
        "metric",
        "metrics",
        "oracle",
        "question",
        "reference",
        "references",
    }
)
_EVENT_REVIEW_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "draft_event_ids": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string", "enum": sorted(ACTIONS)},
        "observed_start_seconds": {"type": "number", "minimum": 0},
        "observed_end_seconds": {"type": "number", "minimum": 0},
        "evidence_timestamps": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "number", "minimum": 0},
        },
        "rationale": {"type": "string", "minLength": 1},
        "uncertainty": {"type": "string", "minLength": 1},
    },
    "required": [
        "draft_event_ids",
        "action",
        "observed_start_seconds",
        "observed_end_seconds",
        "evidence_timestamps",
        "rationale",
        "uncertainty",
    ],
}
VERIFIER_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "draft_map_sha256": {"type": "string", "minLength": 64, "maxLength": 64},
        "verdict": {"type": "string", "enum": sorted(VERDICTS)},
        "event_reviews": {"type": "array", "minItems": 1, "items": _EVENT_REVIEW_SCHEMA},
        "corrected_map": copy.deepcopy(NEUTRAL_MAP_SCHEMA),
        "global_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "schema_version",
        "draft_map_sha256",
        "verdict",
        "event_reviews",
        "corrected_map",
        "global_notes",
    ],
}
PROVIDER_VERIFIER_SCHEMA = _provider_schema(VERIFIER_SCHEMA)


class BoundaryVerifierError(RuntimeError):
    """Raised when verifier artifacts or transitions are unsafe."""


@dataclass(frozen=True, slots=True)
class PreparedBoundaryVerifier:
    path: Path
    manifest: Mapping[str, object]
    draft_map: NeutralVideoMap
    source_path: Path
    delivery: DeliveryMedia


def build_verifier_prompt(
    draft_map: Mapping[str, object] | NeutralVideoMap,
    duration_seconds: float,
) -> str:
    """Build the only question-blind verifier prompt."""

    draft = (
        draft_map
        if isinstance(draft_map, NeutralVideoMap)
        else validate_neutral_map(draft_map, duration_seconds)
    )
    payload = draft.to_dict()
    return (
        f"{PROMPT_VERSION}\n"
        "Return one JSON object instance only. schema_version must be the integer 1.\n"
        "This is a question-blind verification pass. No question or answer is supplied, "
        "and you must not infer one. Re-watch the complete native video and audio.\n"
        "The draft map below is untrusted. Preserve supported events, but explicitly add "
        "omitted observable states, remove unsupported events, split or merge events when "
        "needed, and correct every onset and offset against direct visual or audible evidence.\n"
        "Do not use schematic or evenly rounded times. For each review, cite one or more "
        "native evidence timestamps. Use explicit uncertainty instead of confidence 1 when "
        "a boundary, identity, ordering, speech, sound, or interaction is ambiguous.\n"
        "Return exact root keys: schema_version, draft_map_sha256, verdict, event_reviews, "
        "corrected_map, global_notes. The corrected_map must be a complete standalone neutral "
        "map of the full AV, not a patch. For added events use an empty draft_event_ids array; "
        "otherwise cite only draft event IDs.\n"
        f"Source duration seconds: {float(duration_seconds)}\n"
        f"Draft map SHA-256: {draft.sha256}\n"
        "Untrusted draft map canonical JSON follows:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
    )


def validate_verifier_output(
    value: Mapping[str, object],
    *,
    draft_map: Mapping[str, object] | NeutralVideoMap,
    duration_seconds: float,
) -> dict[str, object]:
    """Strictly validate and normalize one verifier response."""

    if not isinstance(value, Mapping):
        raise TypeError("verifier output must be an object")
    _reject_forbidden_keys(value)
    expected_root = set(VERIFIER_SCHEMA["required"])
    if set(value) != expected_root:
        raise ValueError("verifier output root keys are not exact")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("schema_version must be the integer 1")
    draft = (
        draft_map
        if isinstance(draft_map, NeutralVideoMap)
        else validate_neutral_map(draft_map, duration_seconds)
    )
    if value["draft_map_sha256"] != draft.sha256:
        raise ValueError("draft_map_sha256 does not match the frozen draft")
    if value["verdict"] not in VERDICTS:
        raise ValueError("verifier verdict is invalid")
    reviews = value["event_reviews"]
    if not isinstance(reviews, list) or not reviews:
        raise ValueError("event_reviews must be a non-empty list")
    draft_ids = {event.event_id for event in draft.events}
    reviewed_ids: set[str] = set()
    normalized_reviews: list[dict[str, object]] = []
    expected_review_keys = set(_EVENT_REVIEW_SCHEMA["required"])
    for index, review in enumerate(reviews):
        if not isinstance(review, Mapping) or set(review) != expected_review_keys:
            raise ValueError(f"event_reviews[{index}] keys are not exact")
        action = review["action"]
        if action not in ACTIONS:
            raise ValueError(f"event_reviews[{index}] action is invalid")
        ids = review["draft_event_ids"]
        if not isinstance(ids, list) or not all(isinstance(item, str) and item for item in ids):
            raise ValueError(f"event_reviews[{index}] draft_event_ids are invalid")
        if len(set(ids)) != len(ids) or set(ids) - draft_ids:
            raise ValueError(f"event_reviews[{index}] cites an unknown draft event")
        if action == "added" and ids:
            raise ValueError("added event review must not cite a draft event")
        if action != "added" and not ids:
            raise ValueError("non-added event review must cite a draft event")
        reviewed_ids.update(ids)
        start = _number(review["observed_start_seconds"], "observed_start_seconds")
        end = _number(review["observed_end_seconds"], "observed_end_seconds")
        if start < 0 or end <= start or end > duration_seconds:
            raise ValueError(f"event_reviews[{index}] span is outside the source")
        timestamps = review["evidence_timestamps"]
        if not isinstance(timestamps, list) or not timestamps:
            raise ValueError(f"event_reviews[{index}] evidence_timestamps are empty")
        normalized_timestamps = [
            _number(timestamp, f"event_reviews[{index}] evidence timestamp")
            for timestamp in timestamps
        ]
        if any(timestamp < 0 or timestamp > duration_seconds for timestamp in normalized_timestamps):
            raise ValueError(f"event_reviews[{index}] evidence timestamp is outside the source")
        rationale = _text(review["rationale"], "rationale")
        uncertainty = _text(review["uncertainty"], "uncertainty")
        normalized_reviews.append(
            {
                "draft_event_ids": list(ids),
                "action": action,
                "observed_start_seconds": start,
                "observed_end_seconds": end,
                "evidence_timestamps": normalized_timestamps,
                "rationale": rationale,
                "uncertainty": uncertainty,
            }
        )
    if reviewed_ids != draft_ids:
        raise ValueError("event_reviews must cover every draft event")
    corrected = validate_neutral_map(value["corrected_map"], duration_seconds)
    notes = value["global_notes"]
    if not isinstance(notes, list) or not all(isinstance(note, str) and note.strip() for note in notes):
        raise ValueError("global_notes must contain non-empty strings")
    return {
        "schema_version": 1,
        "draft_map_sha256": draft.sha256,
        "verdict": value["verdict"],
        "event_reviews": normalized_reviews,
        "corrected_map": corrected.to_dict(),
        "global_notes": [note.strip() for note in notes],
    }


def prepare_verifier(
    source_run: Path,
    pilot_audit_freeze: Path,
    video_id: str,
    output: Path,
) -> PreparedBoundaryVerifier:
    """Bind a frozen failed pilot map and native media into a fresh verifier run."""

    source_run = Path(source_run)
    output = Path(output)
    _refuse_nonempty(output)
    prepared = load_prepared(source_run)
    if video_id != prepared.pilot_row.video_id:
        raise ValueError("verifier video must be the frozen pilot video")
    pilot_freeze_path = source_run / "pilot-freeze.json"
    pilot_results_path = source_run / "pilot-results.json"
    pilot_freeze = _read_json(pilot_freeze_path)
    verify_json_artifact(pilot_results_path, dict(pilot_freeze))
    audit_freeze_path = Path(pilot_audit_freeze)
    audit_freeze = _read_json(audit_freeze_path)
    audit_path = audit_freeze_path.parent / "audit.json"
    audit_payload = _read_json(audit_path)
    verify_json_artifact(audit_path, dict(audit_freeze))
    audit_bindings = audit_freeze.get("bindings")
    if (
        not isinstance(audit_bindings, Mapping)
        or audit_bindings.get("decision") != "revise_contract"
        or audit_bindings.get("phase") != "pilot"
        or audit_bindings.get("panel_sha256") != prepared.panel_sha256
        or audit_bindings.get("run_id") != prepared.run_id
    ):
        raise ValueError("verifier requires the matching revise_contract pilot audit")
    if (
        audit_payload.get("prediction_artifact_sha256")
        != _sha256_file(pilot_results_path)
        or audit_payload.get("prediction_freeze_sha256")
        != _sha256_file(pilot_freeze_path)
    ):
        raise ValueError("pilot audit does not bind the supplied prediction freeze")
    pilot_results = _read_json(pilot_results_path)
    prior_phase0_spend_usd = _nonnegative_number(
        pilot_results.get("observed_cost_usd"), "prior_phase0_spend_usd"
    )
    results_value = pilot_results.get("results")
    if not isinstance(results_value, list):
        raise ValueError("frozen pilot aggregate has no result list")
    matches = [
        result
        for result in results_value
        if isinstance(result, Mapping)
        if result.get("phase") == "pilot"
        and result.get("video_id") == video_id
        and result.get("condition_id") == "neutral_map"
        and result.get("status") == "succeeded"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("validated_output"), Mapping):
        raise ValueError("verifier requires one successful frozen pilot map")
    row = prepared.pilot_row
    draft = validate_neutral_map(matches[0]["validated_output"], row.duration_seconds)
    source_path = CANONICAL_DATA_ROOT / row.media_relative_path
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("canonical verifier source media is missing")
    if _sha256_file(source_path) != row.media_sha256 or source_path.stat().st_size != row.size_bytes:
        raise ValueError("canonical verifier source media identity changed")
    output.mkdir(parents=True)
    _write_json_once(output / "draft-map.json", draft.to_dict())
    delivery = prepare_delivery_media(
        run_path=output,
        source_path=source_path,
        video_id=video_id,
        provider="gemini",
    )
    prompt = build_verifier_prompt(draft, row.duration_seconds)
    manifest = {
        "schema_version": 1,
        "runner_id": RUNNER_ID,
        "status": "prepared",
        "run_id": prepared.run_id,
        "panel_sha256": prepared.panel_sha256,
        "video_id": video_id,
        "duration_seconds": row.duration_seconds,
        "source_path": str(source_path.resolve()),
        "source_sha256": row.media_sha256,
        "source_size_bytes": row.size_bytes,
        "source_run": str(source_run.resolve()),
        "source_pilot_artifact_sha256": _sha256_file(pilot_results_path),
        "source_pilot_freeze_sha256": _sha256_file(pilot_freeze_path),
        "source_audit_freeze": str(audit_freeze_path.resolve()),
        "source_audit_freeze_sha256": _sha256_file(audit_freeze_path),
        "source_audit_artifact_sha256": _sha256_file(audit_path),
        "draft_map_sha256": draft.sha256,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": canonical_sha256(prompt),
        "schema_sha256": canonical_sha256(VERIFIER_SCHEMA),
        "provider_schema_sha256": canonical_sha256(PROVIDER_VERIFIER_SCHEMA),
        "model_id": MODEL_ID,
        "max_output_tokens": VERIFIER_POLICY.max_output_tokens,
        "projected_call_cost_usd": PROJECTED_CALL_COST_USD,
        "prior_phase0_spend_usd": prior_phase0_spend_usd,
        "projected_cumulative_cost_usd": _cumulative_cost(
            prior_phase0_spend_usd, PROJECTED_CALL_COST_USD
        ),
        **delivery.request_identity(),
    }
    _write_json_once(output / "manifest.json", manifest)
    (output / "attempts.jsonl").touch(mode=0o600)
    return PreparedBoundaryVerifier(output, manifest, draft, source_path, delivery)


def load_verifier(path: Path) -> PreparedBoundaryVerifier:
    """Load and revalidate a prepared verifier run."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("prepared verifier must be a regular directory")
    manifest = _read_json(root / "manifest.json")
    if manifest.get("runner_id") != RUNNER_ID or manifest.get("status") != "prepared":
        raise ValueError("verifier manifest identity is invalid")
    draft = validate_neutral_map(
        _read_json(root / "draft-map.json"),
        _number(manifest.get("duration_seconds"), "duration_seconds"),
    )
    if draft.sha256 != manifest.get("draft_map_sha256"):
        raise ValueError("verifier draft map identity changed")
    source_path = Path(_text(manifest.get("source_path"), "source_path"))
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("verifier source media is missing")
    if (
        _sha256_file(source_path) != manifest.get("source_sha256")
        or source_path.stat().st_size != manifest.get("source_size_bytes")
    ):
        raise ValueError("verifier source media identity changed")
    source_run = Path(_text(manifest.get("source_run"), "source_run"))
    if _sha256_file(source_run / "pilot-results.json") != manifest.get(
        "source_pilot_artifact_sha256"
    ):
        raise ValueError("source pilot artifact identity changed")
    if _sha256_file(source_run / "pilot-freeze.json") != manifest.get("source_pilot_freeze_sha256"):
        raise ValueError("source pilot freeze identity changed")
    audit_freeze = Path(_text(manifest.get("source_audit_freeze"), "source_audit_freeze"))
    if _sha256_file(audit_freeze) != manifest.get("source_audit_freeze_sha256"):
        raise ValueError("source audit freeze identity changed")
    if _sha256_file(audit_freeze.parent / "audit.json") != manifest.get(
        "source_audit_artifact_sha256"
    ):
        raise ValueError("source audit artifact identity changed")
    delivery = _delivery_from_manifest(root, manifest)
    verify_delivery_media(delivery)
    prompt = build_verifier_prompt(draft, float(manifest["duration_seconds"]))
    expected = {
        "prompt_sha256": canonical_sha256(prompt),
        "schema_sha256": canonical_sha256(VERIFIER_SCHEMA),
        "provider_schema_sha256": canonical_sha256(PROVIDER_VERIFIER_SCHEMA),
        "model_id": MODEL_ID,
        "max_output_tokens": VERIFIER_POLICY.max_output_tokens,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"verifier manifest {key} binding changed")
    if "prior_phase0_spend_usd" in manifest:
        _nonnegative_number(
            manifest["prior_phase0_spend_usd"], "prior_phase0_spend_usd"
        )
    if "projected_cumulative_cost_usd" in manifest:
        prior = _nonnegative_number(
            manifest.get("prior_phase0_spend_usd"), "prior_phase0_spend_usd"
        )
        projected = _nonnegative_number(
            manifest.get("projected_call_cost_usd"), "projected_call_cost_usd"
        )
        if manifest["projected_cumulative_cost_usd"] != _cumulative_cost(
            prior, projected
        ):
            raise ValueError("verifier cumulative cost binding changed")
    return PreparedBoundaryVerifier(root, manifest, draft, source_path, delivery)


def preflight_verifier(prepared: PreparedBoundaryVerifier | Path) -> dict[str, object]:
    prepared = load_verifier(prepared) if isinstance(prepared, (str, Path)) else prepared
    _verify_upstream_chain(prepared)
    prior = _prior_phase0_spend(prepared)
    projected = _projected_call_cost(prepared)
    projected_cumulative = _cumulative_cost(prior, projected)
    if (
        "projected_cumulative_cost_usd" in prepared.manifest
        and prepared.manifest["projected_cumulative_cost_usd"]
        != projected_cumulative
    ):
        raise ValueError("verifier cumulative cost binding changed")
    if projected_cumulative > ABSOLUTE_CAP_USD:
        raise BoundaryVerifierError(
            "projected cumulative verifier cost exceeds the USD 25 cap"
        )
    return {
        "schema_version": 1,
        "status": "preflight-ready",
        "video_id": prepared.manifest["video_id"],
        "model_id": MODEL_ID,
        "source_sha256": prepared.manifest["source_sha256"],
        "draft_map_sha256": prepared.draft_map.sha256,
        "prior_phase0_spend_usd": prior,
        "projected_call_cost_usd": projected,
        "projected_cumulative_cost_usd": projected_cumulative,
        "absolute_cap_usd": ABSOLUTE_CAP_USD,
    }


def run_verifier(
    prepared: PreparedBoundaryVerifier | Path,
    transport: OpenRouterTransport,
) -> dict[str, object]:
    """Execute exactly one terminal verifier attempt without retry."""

    prepared = load_verifier(prepared) if isinstance(prepared, (str, Path)) else prepared
    if (prepared.path / "result.json").exists() or (prepared.path / "raw-response.json").exists():
        raise FileExistsError("verifier already has a terminal attempt")
    preflight_verifier(prepared)
    prior_phase0_spend_usd = _prior_phase0_spend(prepared)
    projected_call_cost_usd = _projected_call_cost(prepared)
    projected_cumulative_cost_usd = _cumulative_cost(
        prior_phase0_spend_usd, projected_call_cost_usd
    )
    if (
        not isinstance(transport, OpenRouterTransport)
        or getattr(transport, "_policy", None) != VERIFIER_POLICY
    ):
        raise BoundaryVerifierError("verifier transport policy does not match")
    prompt = build_verifier_prompt(
        prepared.draft_map, float(prepared.manifest["duration_seconds"])
    )
    request_binding = {
        "schema_version": 1,
        "runner_id": RUNNER_ID,
        "video_id": prepared.manifest["video_id"],
        "source_sha256": prepared.manifest["source_sha256"],
        "delivery_sha256": prepared.delivery.sha256,
        "draft_map_sha256": prepared.draft_map.sha256,
        "prompt_sha256": canonical_sha256(prompt),
        "schema_sha256": canonical_sha256(VERIFIER_SCHEMA),
        "provider_schema_sha256": canonical_sha256(PROVIDER_VERIFIER_SCHEMA),
        "model_id": MODEL_ID,
        "max_output_tokens": VERIFIER_POLICY.max_output_tokens,
        "temperature": 0.0,
        "use_structured_outputs": True,
        "prior_phase0_spend_usd": prior_phase0_spend_usd,
        "projected_call_cost_usd": projected_call_cost_usd,
        "projected_cumulative_cost_usd": projected_cumulative_cost_usd,
        "absolute_cap_usd": ABSOLUTE_CAP_USD,
    }
    request = {**request_binding, "request_sha256": canonical_sha256(request_binding)}
    _write_json_once(prepared.path / "request.json", request)
    _append_jsonl(
        prepared.path / "attempts.jsonl",
        {"schema_version": 1, "status": "started", "request_sha256": request["request_sha256"]},
    )
    uploaded: UploadedFileRecord | None = None
    generated: GenerationResult | None = None
    output: dict[str, object] | None = None
    failure_code: str | None = None
    try:
        uploaded = transport.upload(prepared.delivery.path, mime_type=prepared.delivery.mime_type)
        if uploaded.sha256 != prepared.delivery.sha256:
            raise BoundaryVerifierError("uploaded verifier media identity changed")
        try:
            generated = transport.generate(
                model=MODEL_ID,
                parts=(
                    {"file_id": uploaded.file_id, "mime_type": uploaded.mime_type},
                    prompt,
                ),
                schema=PROVIDER_VERIFIER_SCHEMA,
            )
        except ProviderResponseEvidenceError as error:
            generated = error.generation_result
            failure_code = "provider_protocol"
        if generated is not None:
            _write_raw(prepared.path / "raw-response.json", prepared, generated)
        if failure_code is None and generated is not None:
            observed_model = generated.provider_metadata.get("model")
            observed_provider = generated.provider_metadata.get("provider")
            if observed_model not in {
                MODEL_ID,
                "google/gemini-3.1-pro-preview",
            }:
                raise BoundaryVerifierError("observed verifier model changed")
            if not isinstance(observed_provider, str) or not observed_provider.strip():
                raise BoundaryVerifierError("verifier provider evidence is missing")
            parsed = parse_prompt_json_response(generated.raw_response)
            output = validate_verifier_output(
                parsed,
                draft_map=prepared.draft_map,
                duration_seconds=float(prepared.manifest["duration_seconds"]),
            )
    except Exception as error:
        if generated is not None and not (prepared.path / "raw-response.json").exists():
            _write_raw(prepared.path / "raw-response.json", prepared, generated)
        failure_code = failure_code or (
            "schema_validation" if isinstance(error, (TypeError, ValueError)) else "provider_failure"
        )
    finally:
        if uploaded is not None:
            transport.delete(uploaded.file_id)
    cost = None
    cost_known = False
    input_tokens = 0
    output_tokens = 0
    raw_sha = None
    if generated is not None:
        input_tokens = generated.usage.input_tokens
        output_tokens = generated.usage.output_tokens
        reported = generated.provider_metadata.get("reported_cost_usd")
        if isinstance(reported, (int, float)) and not isinstance(reported, bool):
            cost = float(reported)
            cost_known = math.isfinite(cost) and cost >= 0
        raw_sha = hashlib.sha256(generated.raw_response.encode("utf-8")).hexdigest()
    if failure_code is None and not cost_known:
        failure_code = "provider_evidence"
    cumulative_cost_usd = (
        None
        if not cost_known or cost is None
        else _cumulative_cost(prior_phase0_spend_usd, cost)
    )
    if failure_code is None and cumulative_cost_usd is not None and cumulative_cost_usd > ABSOLUTE_CAP_USD:
        failure_code = "cost_cap_exceeded"
    status = "succeeded" if output is not None and failure_code is None else "failed"
    result: dict[str, object] = {
        "schema_version": 1,
        "runner_id": RUNNER_ID,
        "status": status,
        "failure_code": failure_code,
        "video_id": prepared.manifest["video_id"],
        "request_sha256": request["request_sha256"],
        "source_sha256": prepared.manifest["source_sha256"],
        "draft_map_sha256": prepared.draft_map.sha256,
        "prior_phase0_spend_usd": prior_phase0_spend_usd,
        "projected_call_cost_usd": projected_call_cost_usd,
        "projected_cumulative_cost_usd": projected_cumulative_cost_usd,
        "raw_response_sha256": raw_sha,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_known": cost_known,
        "cost_usd": cost,
        "cumulative_cost_usd": cumulative_cost_usd,
        "provider": None
        if generated is None
        else generated.provider_metadata.get("provider"),
        "observed_model": None
        if generated is None
        else generated.provider_metadata.get("model"),
        "validated_output": output,
    }
    _write_json_once(prepared.path / "result.json", result)
    _append_jsonl(
        prepared.path / "attempts.jsonl",
        {
            "schema_version": 1,
            "status": status,
            "failure_code": failure_code,
            "request_sha256": request["request_sha256"],
            "raw_response_sha256": raw_sha,
            "cost_known": cost_known,
            "cost_usd": cost,
            "prior_phase0_spend_usd": prior_phase0_spend_usd,
            "cumulative_cost_usd": cumulative_cost_usd,
        },
    )
    return result


def freeze_verifier(prepared: PreparedBoundaryVerifier | Path) -> dict[str, object]:
    prepared = load_verifier(prepared) if isinstance(prepared, (str, Path)) else prepared
    _verify_upstream_chain(prepared)
    freeze_path = prepared.path / "freeze.json"
    if freeze_path.exists():
        freeze = _read_json(freeze_path)
        _verify_verifier_freeze(prepared, freeze)
        return dict(freeze)
    result = _read_json(prepared.path / "result.json")
    bindings = _verifier_freeze_bindings(prepared, result)
    freeze = freeze_json_artifact(prepared.path / "result.json", bindings=bindings)
    artifact_chain = _artifact_chain(prepared)
    freeze["artifact_chain"] = artifact_chain
    freeze["artifact_chain_sha256"] = canonical_sha256(artifact_chain)
    _write_json_once(freeze_path, freeze)
    _verify_verifier_freeze(prepared, freeze)
    return freeze


def status_verifier(prepared: PreparedBoundaryVerifier | Path) -> dict[str, object]:
    prepared = load_verifier(prepared) if isinstance(prepared, (str, Path)) else prepared
    _verify_upstream_chain(prepared)
    result_path = prepared.path / "result.json"
    result = _read_json(result_path) if result_path.is_file() else None
    frozen = False
    freeze_path = prepared.path / "freeze.json"
    if freeze_path.is_file():
        _verify_verifier_freeze(prepared, _read_json(freeze_path))
        frozen = True
    return {
        "schema_version": 1,
        "status": "prepared" if result is None else result["status"],
        "video_id": prepared.manifest["video_id"],
        "draft_map_sha256": prepared.draft_map.sha256,
        "terminal_attempts": 0 if result is None else 1,
        "cost_known": False if result is None else result["cost_known"],
        "cost_usd": None if result is None else result["cost_usd"],
        "frozen": frozen,
    }


def _artifact_chain(prepared: PreparedBoundaryVerifier) -> dict[str, str]:
    """Hash every local and upstream artifact needed to reproduce the call."""

    manifest = prepared.manifest
    source_run = Path(_text(manifest.get("source_run"), "source_run"))
    audit_freeze = Path(
        _text(manifest.get("source_audit_freeze"), "source_audit_freeze")
    )
    paths = {
        "manifest": prepared.path / "manifest.json",
        "draft_map": prepared.path / "draft-map.json",
        "request": prepared.path / "request.json",
        "attempts": prepared.path / "attempts.jsonl",
        "raw_response": prepared.path / "raw-response.json",
        "result": prepared.path / "result.json",
        "delivery_media": prepared.delivery.path,
        "delivery_provenance": prepared.delivery.provenance_path,
        "source_media": prepared.source_path,
        "source_pilot_results": source_run / "pilot-results.json",
        "source_pilot_freeze": source_run / "pilot-freeze.json",
        "source_audit": audit_freeze.parent / "audit.json",
        "source_audit_freeze": audit_freeze,
    }
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def _verify_verifier_freeze(
    prepared: PreparedBoundaryVerifier, freeze: Mapping[str, object]
) -> None:
    _verify_upstream_chain(prepared)
    verify_json_artifact(prepared.path / "result.json", dict(freeze))
    result = _read_json(prepared.path / "result.json")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("verifier freeze bindings are missing")
    if bindings.get("status") == "succeeded" and "cumulative_cost_usd" not in bindings:
        expected_legacy = {
            "runner_id": RUNNER_ID,
            "status": "succeeded",
            "video_id": prepared.manifest["video_id"],
            "request_sha256": result["request_sha256"],
            "source_sha256": prepared.manifest["source_sha256"],
            "draft_map_sha256": prepared.draft_map.sha256,
        }
        if dict(bindings) != expected_legacy:
            raise ValueError("legacy verifier freeze bindings changed")
        _validate_legacy_succeeded_result(result)
    elif dict(bindings) != _verifier_freeze_bindings(prepared, result):
        raise ValueError("verifier freeze bindings changed")
    chain = freeze.get("artifact_chain")
    if not isinstance(chain, Mapping):
        raise ValueError("verifier freeze artifact chain is missing")
    if freeze.get("artifact_chain_sha256") != canonical_sha256(chain):
        raise ValueError("verifier freeze artifact chain binding changed")
    if dict(chain) != _artifact_chain(prepared):
        raise ValueError("verifier frozen artifact chain changed")


def _verify_upstream_chain(prepared: PreparedBoundaryVerifier) -> None:
    """Re-prove source prediction and audit freeze relationships."""

    manifest = prepared.manifest
    source_run = Path(_text(manifest.get("source_run"), "source_run"))
    pilot_results_path = source_run / "pilot-results.json"
    pilot_freeze_path = source_run / "pilot-freeze.json"
    pilot_freeze = _read_json(pilot_freeze_path)
    verify_json_artifact(pilot_results_path, dict(pilot_freeze))
    pilot_results = _read_json(pilot_results_path)
    if "prior_phase0_spend_usd" in manifest:
        prior_phase0_spend_usd = _nonnegative_number(
            pilot_results.get("observed_cost_usd"), "prior_phase0_spend_usd"
        )
        if manifest["prior_phase0_spend_usd"] != prior_phase0_spend_usd:
            raise ValueError("prior Phase 0 spend binding changed")
    audit_freeze_path = Path(
        _text(manifest.get("source_audit_freeze"), "source_audit_freeze")
    )
    audit_path = audit_freeze_path.parent / "audit.json"
    audit_freeze = _read_json(audit_freeze_path)
    audit_payload = _read_json(audit_path)
    verify_json_artifact(audit_path, dict(audit_freeze))
    if (
        audit_payload.get("prediction_artifact_sha256")
        != _sha256_file(pilot_results_path)
        or audit_payload.get("prediction_freeze_sha256")
        != _sha256_file(pilot_freeze_path)
    ):
        raise ValueError("pilot audit prediction-freeze relationship changed")


def _delivery_from_manifest(root: Path, manifest: Mapping[str, object]) -> DeliveryMedia:
    video_id = _text(manifest.get("video_id"), "video_id")
    delivery_path = root / "delivery-media" / f"{video_id}.webm"
    if manifest.get("delivery_derived") is False:
        delivery_path = Path(_text(manifest.get("source_path"), "source_path"))
    return DeliveryMedia(
        path=delivery_path,
        provenance_path=root / "delivery-media" / f"gemini-{video_id}.json",
        mime_type=_text(manifest.get("delivery_mime_type"), "delivery_mime_type"),
        sha256=_text(manifest.get("delivery_sha256"), "delivery_sha256"),
        size_bytes=int(manifest["delivery_size_bytes"]),
        source_sha256=_text(manifest.get("source_sha256"), "source_sha256"),
        source_size_bytes=int(manifest["source_size_bytes"]),
        policy_version=_text(manifest.get("delivery_policy_version"), "delivery_policy_version"),
        derived=bool(manifest["delivery_derived"]),
    )


def _write_raw(path: Path, prepared: PreparedBoundaryVerifier, generated: GenerationResult) -> None:
    _write_json_once(
        path,
        {
            "schema_version": 1,
            "video_id": prepared.manifest["video_id"],
            "draft_map_sha256": prepared.draft_map.sha256,
            "raw_response": generated.raw_response,
            "raw_response_sha256": hashlib.sha256(
                generated.raw_response.encode("utf-8")
            ).hexdigest(),
        },
    )


def _reject_forbidden_keys(value: object, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = "".join(character for character in str(key).casefold() if character.isalnum())
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden field at {location}.{key}")
            _reject_forbidden_keys(nested, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_forbidden_keys(nested, f"{location}[{index}]")


def _refuse_nonempty(path: Path) -> None:
    if path.is_symlink():
        raise FileExistsError("refusing symlinked verifier output")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("verifier output is not empty")


def _read_json(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON artifact is missing: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("JSON artifact must contain an object")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError("JSONL row must contain an object")
            rows.append(value)
    return rows


def _write_json_once(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"artifact already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ValueError("refusing symlinked JSONL artifact")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _nonnegative_number(value: object, field: str) -> float:
    result = _number(value, field)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _cumulative_cost(prior: float, projected: float) -> float:
    return round(prior + projected, 6)


def _prior_phase0_spend(prepared: PreparedBoundaryVerifier) -> float:
    return _nonnegative_number(
        prepared.manifest.get("prior_phase0_spend_usd"),
        "prior_phase0_spend_usd",
    )


def _projected_call_cost(prepared: PreparedBoundaryVerifier) -> float:
    return _nonnegative_number(
        prepared.manifest.get("projected_call_cost_usd"),
        "projected_call_cost_usd",
    )


def _validate_legacy_succeeded_result(result: Mapping[str, object]) -> None:
    if (
        result.get("status") != "succeeded"
        or result.get("failure_code") is not None
        or not isinstance(result.get("validated_output"), Mapping)
        or result.get("cost_known") is not True
        or not isinstance(result.get("cost_usd"), (int, float))
        or isinstance(result.get("cost_usd"), bool)
        or not math.isfinite(float(result["cost_usd"]))
        or float(result["cost_usd"]) < 0
        or not isinstance(result.get("provider"), str)
        or not result["provider"].strip()
        or result.get("observed_model")
        not in {MODEL_ID, "google/gemini-3.1-pro-preview"}
    ):
        raise BoundaryVerifierError("legacy succeeded verifier result is not freezeable")


def _verify_raw_response_evidence(
    prepared: PreparedBoundaryVerifier, result: Mapping[str, object]
) -> None:
    raw_sha = result.get("raw_response_sha256")
    if not isinstance(raw_sha, str) or not raw_sha:
        raise BoundaryVerifierError("terminal verifier raw response evidence is missing")
    raw_payload = _read_json(prepared.path / "raw-response.json")
    raw_response = raw_payload.get("raw_response")
    if (
        not isinstance(raw_response, str)
        or hashlib.sha256(raw_response.encode("utf-8")).hexdigest() != raw_sha
        or raw_payload.get("raw_response_sha256") != raw_sha
    ):
        raise BoundaryVerifierError("terminal verifier raw response evidence changed")


def _verifier_freeze_bindings(
    prepared: PreparedBoundaryVerifier, result: Mapping[str, object]
) -> dict[str, object]:
    status = result.get("status")
    if status not in {"succeeded", "failed"}:
        raise BoundaryVerifierError("verifier result is not terminal")
    prior = _prior_phase0_spend(prepared)
    projected = _projected_call_cost(prepared)
    expected_projected_cumulative = _cumulative_cost(prior, projected)
    if (
        result.get("prior_phase0_spend_usd") != prior
        or result.get("projected_call_cost_usd") != projected
        or result.get("projected_cumulative_cost_usd") != expected_projected_cumulative
    ):
        raise BoundaryVerifierError("verifier result cumulative projection binding changed")
    if status == "succeeded":
        _validate_legacy_succeeded_result(result)
        _verify_raw_response_evidence(prepared, result)
    else:
        if result.get("failure_code") not in {
            "provider_failure",
            "provider_protocol",
            "schema_validation",
        }:
            raise BoundaryVerifierError("verifier failure is not freezeable")
        if result.get("validated_output") is not None:
            raise BoundaryVerifierError("failed verifier result has validated output")
        _verify_raw_response_evidence(prepared, result)
        if not isinstance(result.get("provider"), str) or not result["provider"].strip():
            raise BoundaryVerifierError("failed verifier provider evidence is missing")
        if result.get("observed_model") not in {
            MODEL_ID,
            "google/gemini-3.1-pro-preview",
        }:
            raise BoundaryVerifierError("failed verifier model evidence is missing")
    cost = result.get("cost_usd")
    if (
        result.get("cost_known") is not True
        or not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise BoundaryVerifierError("verifier result cost evidence is missing")
    cumulative = _cumulative_cost(prior, float(cost))
    if result.get("cumulative_cost_usd") != cumulative or cumulative > ABSOLUTE_CAP_USD:
        raise BoundaryVerifierError("verifier cumulative cost is not freezeable")
    return {
        "runner_id": RUNNER_ID,
        "status": status,
        "failure_code": result.get("failure_code"),
        "video_id": prepared.manifest["video_id"],
        "request_sha256": result.get("request_sha256"),
        "source_sha256": prepared.manifest["source_sha256"],
        "draft_map_sha256": prepared.draft_map.sha256,
        "raw_response_sha256": result.get("raw_response_sha256"),
        "provider": result.get("provider"),
        "observed_model": result.get("observed_model"),
        "cost_known": result.get("cost_known"),
        "cost_usd": cost,
        "prior_phase0_spend_usd": prior,
        "projected_call_cost_usd": projected,
        "projected_cumulative_cost_usd": expected_projected_cumulative,
        "cumulative_cost_usd": cumulative,
    }


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


__all__ = [
    "ABSOLUTE_CAP_USD",
    "BoundaryVerifierError",
    "PreparedBoundaryVerifier",
    "PROVIDER_VERIFIER_SCHEMA",
    "VERIFIER_POLICY",
    "VERIFIER_SCHEMA",
    "build_verifier_prompt",
    "freeze_verifier",
    "load_verifier",
    "preflight_verifier",
    "prepare_verifier",
    "run_verifier",
    "status_verifier",
    "validate_verifier_output",
]
