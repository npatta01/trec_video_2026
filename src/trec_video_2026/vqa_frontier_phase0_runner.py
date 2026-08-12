"""Immutable, resumable runner for the Frontier VQA Phase 0 screen.

The runner owns only private run state.  It deliberately keeps the provider
boundary small: media delivery, transport generation, response preservation,
and the Task 1 local validators are all delegated to their existing modules.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from trec_video_2026.vqa_frontier_map_media import (
    DeliveryMedia,
    DeliveryMediaError,
    prepare_delivery_media,
    verify_delivery_media,
)
from trec_video_2026.vqa_frontier_phase0_contract import (
    GROUNDED_ANSWER_SCHEMA,
    NEUTRAL_MAP_SCHEMA,
    PROVIDER_GROUNDED_ANSWER_SCHEMA,
    PROVIDER_NEUTRAL_MAP_SCHEMA,
    NeutralVideoMap,
    build_grounded_answer_prompt,
    build_neutral_map_prompt,
    validate_grounded_answer,
    validate_neutral_map,
)
from trec_video_2026.vqa_frontier_phase0_panel import (
    CATEGORIES,
    CONDITIONS,
    PHASE0_RUN_ID,
    PanelRow,
    Phase0Panel,
    QWEN_CONDITION_ID,
    REFERENCE_POLICY,
    SELECTION_POLICY,
)
from trec_video_2026.vqa_frontier_response_store import (
    RawResponseConflictError,
    parse_prompt_json_response,
    preserve_raw_response,
    verify_raw_response,
)
from trec_video_2026.vqa_experiments.temporal_bottleneck import (
    freeze_json_artifact,
    verify_json_artifact,
)
from trec_video_2026.vqa_submission import MODEL_ID, canonical_sha256
from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    ProviderAuthenticationError,
    ProviderError,
    ProviderProtocolError,
    ProviderResponseEvidenceError,
    UploadedFileRecord,
)
from trec_video_2026.vqa_submission_openrouter import (
    OpenRouterRequestPolicy,
    OpenRouterTransport,
)


SCHEMA_VERSION = 1
RUNNER_ID = "frontier-phase0-runner-v1"
PRICE_SNAPSHOT_DATE = "2026-08-12"
ABSOLUTE_CAP_USD = 25.0
OPENROUTER_PROVIDER = "openrouter"
DELIVERY_PROVIDER = "gemini"
MAP_POLICY = OpenRouterRequestPolicy(
    max_output_tokens=16384,
    temperature=0.0,
    allow_fallbacks=False,
    use_structured_outputs=True,
)
ANSWER_POLICY = OpenRouterRequestPolicy(
    max_output_tokens=4096,
    temperature=0.0,
    allow_fallbacks=False,
    use_structured_outputs=True,
)
_MODEL_SLUG = "google/gemini-3.1-pro-preview"
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TERMINAL = frozenset({"succeeded", "failed"})
_RETRYABLE = frozenset({"transport", "rate_limit", "empty", "truncated"})
_CONDITIONS = tuple(CONDITIONS)
_PHASES = ("pilot", "screen")
_DEFAULT_INPUT_TOKENS = 1_000_000
_MAP_OUTPUT_TOKENS = 16384
_ANSWER_OUTPUT_TOKENS = 4096
_RESPONSE_FORMAT_POLICY = "native-json-schema-v1"


class Phase0RunnerError(RuntimeError):
    """Base class for safe, operator-facing runner failures."""


class PilotGateError(Phase0RunnerError):
    """Raised when screen dispatch lacks the frozen pilot authorization."""


class CostGateError(Phase0RunnerError):
    """Raised before a provider call would cross the hard cost cap."""


class CostUnknownError(CostGateError):
    """Raised when provider billing cannot be determined conservatively."""


class AmbiguousStartedAttemptError(Phase0RunnerError):
    """Raised when an earlier attempt has no terminal record."""


class RequestBindingError(Phase0RunnerError):
    """Raised when an existing artifact no longer matches its request."""


class RequestClaimDisposition(str, Enum):
    """Outcome of the permanent claim acquisition race."""

    OWNED = "owned"
    ALREADY_COMPLETED = "already_completed"


class FrontierTransport(Protocol):
    """Provider boundary shared by OpenRouter and complete fixture transports."""

    provider_name: str

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord: ...

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: Mapping[str, object],
    ) -> GenerationResult: ...

    def delete(self, file_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedPhase0Run:
    path: Path
    run_id: str
    panel_sha256: str
    manifest_sha256: str
    panel: Phase0Panel
    pilot_row: PanelRow
    screen_rows: tuple[PanelRow, ...]


@dataclass(frozen=True, slots=True)
class Phase0RunSummary:
    phase: str
    expected: int
    passed: int
    failed: int
    skipped: int
    input_tokens: int = 0
    output_tokens: int = 0
    observed_cost_usd: float | None = 0.0

    @property
    def estimated_cost_usd(self) -> float | None:
        """Compatibility name used by the neighboring frontier runner."""

        return self.observed_cost_usd


def prepare_run(panel: Path, output: Path) -> PreparedPhase0Run:
    """Create a new immutable run directory from a frozen Phase 0 panel."""

    panel = Path(panel)
    output = Path(output)
    if panel.is_symlink() or not panel.is_file():
        raise ValueError("panel must be a regular file")
    payload = _read_json(panel)
    parsed = _parse_panel(payload)
    _refuse_existing_run(output)
    output.mkdir(parents=True, exist_ok=True)
    for directory in (
        "results",
        "raw/neutral_map",
        "raw/direct_answer",
        "raw/map_guided_answer",
        "delivery/gemini",
        "request-claims",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)
    _write_json_once(output / "panel.json", payload)
    panel_sha256 = canonical_sha256(payload)
    manifest = _manifest_payload(parsed, panel_sha256=panel_sha256)
    _write_json_once(output / "run-manifest.json", manifest)
    for relative in (
        "requests.jsonl",
        "attempts.jsonl",
        "results/neutral_map.jsonl",
        "results/direct_answer.jsonl",
        "results/map_guided_answer.jsonl",
    ):
        _create_empty_file(output / relative)
    prepared = _prepared_from_payload(
        output,
        payload,
        panel_sha256=panel_sha256,
        manifest_sha256=_sha256_file(output / "run-manifest.json"),
    )
    _write_terminal_status(prepared)
    return prepared


def load_prepared(path: Path) -> PreparedPhase0Run:
    """Load and revalidate a prepared run without changing its artifacts."""

    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("prepared run must be a regular directory")
    manifest = _read_json(path / "run-manifest.json")
    panel = _read_json(path / "panel.json")
    panel_sha256 = manifest.get("panel_sha256")
    if not isinstance(panel_sha256, str) or canonical_sha256(panel) != panel_sha256:
        raise RequestBindingError("panel identity does not match the prepared manifest")
    expected_manifest_fields = {
        "schema_version": SCHEMA_VERSION,
        "runner_id": RUNNER_ID,
        "panel_sha256": panel_sha256,
        "prompt_sha256": _prompt_sha256s()["neutral_map"],
        "answer_prompt_sha256": _prompt_sha256s()["answer"],
        "neutral_schema_sha256": canonical_sha256(NEUTRAL_MAP_SCHEMA),
        "answer_schema_sha256": canonical_sha256(GROUNDED_ANSWER_SCHEMA),
    }
    for key, expected in expected_manifest_fields.items():
        if manifest.get(key) != expected:
            raise RequestBindingError(f"prepared manifest {key} binding changed")
    return _prepared_from_payload(
        path,
        panel,
        panel_sha256=panel_sha256,
        manifest_sha256=_sha256_file(path / "run-manifest.json"),
    )


def preflight(prepared: PreparedPhase0Run | Path) -> Mapping[str, object]:
    """Check the local prepared-run contract without making a provider call."""

    prepared = _coerce_prepared(prepared)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "preflight-ready",
        "run_id": prepared.run_id,
        "panel_sha256": prepared.panel_sha256,
        "model_id": MODEL_ID,
        "pilot_rows": 1,
        "screen_rows": len(prepared.screen_rows),
        "conditions": list(_CONDITIONS),
    }


def run_pilot(
    prepared: PreparedPhase0Run | Path,
    transport: FrontierTransport,
) -> Phase0RunSummary:
    """Run the single question-blind pilot map, resuming exact completions."""

    prepared = _coerce_prepared(prepared)
    _ensure_cost_known(prepared)
    _fail_on_ambiguous_started_attempts(prepared)
    row = prepared.pilot_row
    summary = _dispatch_condition(
        prepared,
        phase="pilot",
        row=row,
        condition_id="neutral_map",
        transport=transport,
        neutral_map=None,
    )
    result = _summary_from_outcomes("pilot", [summary])
    _write_terminal_status(prepared)
    return result


def authorize_screen(
    prepared: PreparedPhase0Run | Path,
    pilot_audit_freeze: Path,
) -> Mapping[str, object]:
    """Record a usable, frozen pilot audit as the screen authorization."""

    prepared = _coerce_prepared(prepared)
    _ensure_cost_known(prepared)
    pilot_freeze = prepared.path / "pilot-freeze.json"
    if not pilot_freeze.is_file():
        raise PilotGateError("pilot audit authorization requires frozen pilot results")
    _verify_freeze_file(prepared, pilot_freeze)
    pilot_results = _require_complete_phase(prepared, "pilot")
    if pilot_results[0].get("status") != "succeeded":
        raise PilotGateError("pilot audit authorization requires a successful pilot map")
    audit = _read_json(Path(pilot_audit_freeze))
    _verify_audit_freeze(prepared, Path(pilot_audit_freeze), audit)
    decision = _audit_decision(audit)
    if decision != "usable_for_screen":
        raise PilotGateError("pilot audit authorization is not usable_for_screen")
    source_freeze = Path(pilot_audit_freeze)
    retained_root = prepared.path / "pilot-audit"
    _retain_audit_packet_once(source_freeze.parent, retained_root)
    retained_freeze = retained_root / "freeze.json"
    retained_payload = _read_json(retained_freeze)
    _verify_audit_freeze(prepared, retained_freeze, retained_payload)
    pilot_artifact = _read_json(prepared.path / "pilot-results.json")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "authorized",
        "run_id": prepared.run_id,
        "panel_sha256": prepared.panel_sha256,
        "pilot_freeze_sha256": _sha256_file(pilot_freeze),
        "pilot_results_artifact_sha256": _sha256_file(prepared.path / "pilot-results.json"),
        "pilot_result_hashes": pilot_artifact["result_hashes"],
        "pilot_request_hashes": pilot_artifact["request_hashes"],
        "pilot_results_sha256": pilot_artifact["results_sha256"],
        "pilot_audit_freeze_sha256": _sha256_file(Path(pilot_audit_freeze)),
        "pilot_audit_artifact_sha256": audit["artifact_sha256"],
        "pilot_audit_bindings_sha256": audit["bindings_sha256"],
        "pilot_audit_decision": decision,
    }
    path = prepared.path / "pilot-authorization.json"
    if path.exists() or path.is_symlink():
        existing = _read_json(path)
        if existing != payload:
            raise FileExistsError("refusing to overwrite pilot authorization")
    else:
        _write_json_once(path, payload)
    _write_terminal_status(prepared)
    return payload


def run_screen_row(
    prepared: PreparedPhase0Run | Path,
    *,
    ordinal: int,
    transport_factory: Callable[[str], FrontierTransport],
) -> Phase0RunSummary:
    """Run one fresh row in neutral/direct/guided order."""

    prepared = _coerce_prepared(prepared)
    _ensure_cost_known(prepared)
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 1 <= ordinal <= 7:
        raise ValueError("screen ordinal must be between 1 and 7")
    if not (prepared.path / "pilot-authorization.json").is_file():
        raise PilotGateError("screen requires pilot audit authorization")
    _verify_pilot_authorization(prepared)
    if ordinal >= 2:
        gate_path = prepared.path / "cost-gate.json"
        if not gate_path.is_file():
            raise CostGateError("remaining screen requires an authorized cost gate")
        _verify_cost_gate(prepared, _read_json(gate_path))
    _fail_on_ambiguous_started_attempts(prepared)
    row = prepared.screen_rows[ordinal - 1]
    outcomes: list[Mapping[str, object]] = []
    neutral_map: NeutralVideoMap | None = None
    for condition_id in _CONDITIONS:
        transport = transport_factory(condition_id)
        if not hasattr(transport, "upload") or not hasattr(transport, "generate"):
            raise TypeError("transport_factory must return a FrontierTransport")
        outcome = _dispatch_condition(
            prepared,
            phase="screen",
            row=row,
            condition_id=condition_id,
            transport=transport,
            neutral_map=neutral_map if condition_id == "map_guided_answer" else None,
        )
        outcomes.append(outcome)
        if condition_id == "neutral_map" and outcome.get("status") == "succeeded":
            neutral_map = _neutral_map_from_result(outcome, row.duration_seconds)
    result = _summary_from_outcomes("screen", outcomes)
    _write_terminal_status(prepared)
    return result


def run_screen_remaining(
    prepared: PreparedPhase0Run | Path,
    *,
    transport_factory: Callable[[str], FrontierTransport],
) -> Phase0RunSummary:
    """Run ordinals two through seven after the immutable cost gate."""

    prepared = _coerce_prepared(prepared)
    _ensure_cost_known(prepared)
    gate = prepared.path / "cost-gate.json"
    if not gate.is_file():
        raise CostGateError("remaining screen requires an authorized cost gate")
    _verify_cost_gate(prepared, _read_json(gate))
    outcomes: list[Phase0RunSummary] = []
    for ordinal in range(2, len(prepared.screen_rows) + 1):
        outcomes.append(
            run_screen_row(
                prepared,
                ordinal=ordinal,
                transport_factory=transport_factory,
            )
        )
    return _combine_summaries("screen", outcomes)


def project_remaining_cost(
    prepared: PreparedPhase0Run | Path,
    *,
    _screen_limit: int | None = None,
) -> Mapping[str, object]:
    """Project the complete 22-call run from observed per-condition means."""

    prepared = _coerce_prepared(prepared)
    _ensure_cost_known(prepared)
    conditions: dict[str, dict[str, object]] = {}
    observed_total = 0.0
    projected_remaining = 0.0
    for condition_id in _CONDITIONS:
        results = _bound_results(prepared, condition_id)
        if _screen_limit is not None:
            allowed_ids = {
                row.video_id for row in prepared.screen_rows[:_screen_limit]
            }
            results = [
                result
                for result in results
                if result.get("phase") == "pilot"
                or result.get("video_id") in allowed_ids
            ]
        allowed_ids = None
        if _screen_limit is not None:
            allowed_ids = {
                row.video_id for row in prepared.screen_rows[:_screen_limit]
            }
        unrepresented_calls, unrepresented_cost, _ = _unrepresented_terminal_attempts(
            prepared, condition_id, allowed_video_ids=allowed_ids
        )
        observed_cost = sum(
            _number(result.get("cost_usd")) for result in results
        ) + unrepresented_cost
        observed_calls = len(results) + unrepresented_calls
        screen_count = len(prepared.screen_rows)
        expected_calls = 1 + screen_count if condition_id == "neutral_map" else screen_count
        remaining_calls = max(0, expected_calls - observed_calls)
        observed_mean = observed_cost / observed_calls if observed_calls else None
        mean_for_projection = (
            observed_mean
            if observed_mean is not None
            else _conservative_cost_estimate(condition_id)
        )
        projected = mean_for_projection * remaining_calls
        observed_total += observed_cost
        projected_remaining += projected
        conditions[condition_id] = {
            "observed_calls": observed_calls,
            "observed_cost_usd": round(observed_cost, 6),
            "observed_mean_usd": None
            if observed_mean is None
            else round(observed_mean, 6),
            "remaining_calls": remaining_calls,
            "projected_remaining_usd": round(projected, 6),
        }
    projected_total = observed_total + projected_remaining
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared.run_id,
        "panel_sha256": prepared.panel_sha256,
        "price_snapshot_date": PRICE_SNAPSHOT_DATE,
        "conditions": conditions,
        "observed_cost_usd": round(observed_total, 6),
        "projected_remaining_usd": round(projected_remaining, 6),
        "projected_total_usd": round(projected_total, 6),
        "absolute_cap_usd": ABSOLUTE_CAP_USD,
        "projected_total_usd_lt_cap": projected_total < ABSOLUTE_CAP_USD,
    }


def authorize_remaining(prepared: PreparedPhase0Run | Path) -> Mapping[str, object]:
    """Freeze the first-row hashes and the below-cap remaining projection."""

    prepared = _coerce_prepared(prepared)
    _ensure_cost_known(prepared)
    _verify_pilot_authorization(prepared)
    first_row = _first_row_results(prepared)
    if set(first_row) != set(_CONDITIONS):
        raise CostGateError("first screen row is not complete")
    projection = project_remaining_cost(prepared)
    first_row_projection = project_remaining_cost(prepared, _screen_limit=1)
    if projection["projected_total_usd_lt_cap"] is not True:
        raise CostGateError("projected total would reach USD 25")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "authorized",
        "run_id": prepared.run_id,
        "panel_sha256": prepared.panel_sha256,
        "first_row_result_hashes": {
            condition: first_row[condition]["result_sha256"]
            for condition in _CONDITIONS
        },
        "first_row_request_hashes": {
            condition: first_row[condition]["request_sha256"]
            for condition in _CONDITIONS
        },
        "projected_total_usd": first_row_projection["projected_total_usd"],
        "absolute_cap_usd": ABSOLUTE_CAP_USD,
        "projected_total_usd_lt_cap": True,
        "projection_snapshot": first_row_projection,
        "projection_snapshot_sha256": canonical_sha256(first_row_projection),
    }
    path = prepared.path / "cost-gate.json"
    if path.exists() or path.is_symlink():
        existing = _read_json(path)
        if existing != payload:
            raise FileExistsError("refusing to overwrite cost gate")
    else:
        _write_json_once(path, payload)
    _write_terminal_status(prepared)
    return payload


def freeze_pilot(prepared: PreparedPhase0Run | Path) -> Mapping[str, object]:
    """Freeze the complete pilot result set and its request bindings."""

    prepared = _coerce_prepared(prepared)
    _ensure_cost_known(prepared)
    results = _require_complete_phase(prepared, "pilot")
    return _freeze_phase(prepared, "pilot", results)


def freeze_screen(prepared: PreparedPhase0Run | Path) -> Mapping[str, object]:
    """Freeze all 21 screen results after the cost-gated run completes."""

    prepared = _coerce_prepared(prepared)
    _ensure_cost_known(prepared)
    if not (prepared.path / "cost-gate.json").is_file():
        raise CostGateError("screen freeze requires an authorized cost gate")
    pilot_freeze = prepared.path / "pilot-freeze.json"
    if not pilot_freeze.is_file():
        raise PilotGateError("screen freeze requires frozen pilot results")
    _verify_freeze_file(prepared, pilot_freeze)
    _verify_pilot_authorization(prepared)
    _verify_cost_gate(prepared, _read_json(prepared.path / "cost-gate.json"))
    results = _require_complete_phase(prepared, "screen")
    return _freeze_phase(prepared, "screen", results)


def status(prepared: PreparedPhase0Run | Path) -> Mapping[str, object]:
    """Return only safe counts, state names, hashes, and costs."""

    prepared = _coerce_prepared(prepared)
    for freeze_name in ("pilot-freeze.json", "screen-freeze.json"):
        freeze_path = prepared.path / freeze_name
        if freeze_path.is_file() and not freeze_path.is_symlink():
            _verify_freeze_file(prepared, freeze_path)
    counts = {
        condition: {
            "pilot_terminal": sum(
                1
                for result in _bound_results(prepared, condition)
                if result.get("phase") == "pilot"
            ),
            "screen_terminal": sum(
                1
                for result in _bound_results(prepared, condition)
                if result.get("phase") == "screen"
            ),
        }
        for condition in _CONDITIONS
    }
    attempts = _read_jsonl(prepared.path / "attempts.jsonl")
    ambiguous = len(_ambiguous_attempts(prepared, attempts))
    ambiguous_claims = len(_orphan_claims(prepared))
    observed_cost, cost_known = _cost_state(prepared)
    failure_codes: dict[str, int] = {}
    for record in (*attempts, *_all_terminal_results(prepared)):
        code = record.get("failure_code")
        if isinstance(code, str):
            failure_codes[code] = failure_codes.get(code, 0) + 1
    cost_state = None
    cost_state_path = prepared.path / "cost-state.json"
    if cost_state_path.is_file():
        cost_state = _read_json(cost_state_path).get("state")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": _state_name(prepared, counts, ambiguous, ambiguous_claims),
        "run_id": prepared.run_id,
        "panel_sha256": prepared.panel_sha256,
        "counts": counts,
        "terminal_conditions": sum(
            counts[condition]["pilot_terminal"] + counts[condition]["screen_terminal"]
            for condition in _CONDITIONS
        ),
        "failed_attempts": sum(
            attempt.get("status") == "failed"
            for attempt in attempts
            if attempt.get("run_id") == prepared.run_id
        ),
        "ambiguous_started_attempts": ambiguous,
        "ambiguous_request_claims": ambiguous_claims,
        "cost_known": cost_known,
        "cost_state": cost_state,
        "failure_codes": failure_codes,
        "observed_cost_usd": None if observed_cost is None else round(observed_cost, 6),
        "pilot_frozen": (prepared.path / "pilot-freeze.json").is_file(),
        "pilot_authorized": (prepared.path / "pilot-authorization.json").is_file(),
        "cost_gate_authorized": (prepared.path / "cost-gate.json").is_file(),
        "screen_frozen": (prepared.path / "screen-freeze.json").is_file(),
    }


def _dispatch_condition(
    prepared: PreparedPhase0Run,
    *,
    phase: str,
    row: PanelRow,
    condition_id: str,
    transport: FrontierTransport,
    neutral_map: NeutralVideoMap | None,
) -> Mapping[str, object]:
    if condition_id not in _CONDITIONS:
        raise ValueError("unsupported Phase 0 condition")
    if phase not in _PHASES:
        raise ValueError("unsupported Phase 0 phase")
    if phase == "pilot" and condition_id != "neutral_map":
        raise ValueError("pilot permits only neutral_map")
    _ensure_cost_known(prepared)
    _fail_on_ambiguous_started_attempts(prepared)
    prompt, schema, provider_schema, validated_map = _condition_inputs(
        condition_id,
        row,
        neutral_map,
        prepared=prepared,
    )
    delivery = _prepare_delivery(prepared, row)
    request = _request_record(
        prepared,
        phase=phase,
        row=row,
        condition_id=condition_id,
        prompt=prompt,
        schema=schema,
        provider_schema=provider_schema,
        delivery=delivery,
        neutral_map=validated_map,
    )
    _ensure_request_record(prepared, request)
    existing = _existing_logical_result(prepared, phase, row, condition_id)
    if existing is not None:
        if existing.get("request_sha256") != request["request_sha256"] or not _binding_matches(existing, request):
            raise RequestBindingError("completed request binding changed")
        if existing.get("status") != "succeeded":
            return existing
        skipped = dict(existing)
        skipped["status"] = "skipped"
        skipped["input_tokens"] = 0
        skipped["output_tokens"] = 0
        skipped["cost_usd"] = 0.0
        skipped["estimated_cost_usd"] = 0.0
        return skipped
    _validate_transport(transport, condition_id)
    next_cost = _transport_estimate(transport, condition_id, row)
    if _known_cumulative_cost(prepared) + next_cost >= ABSOLUTE_CAP_USD:
        raise CostGateError("next provider call would reach USD 25")
    claim_disposition, completed = _claim_request(prepared, request)
    if claim_disposition is RequestClaimDisposition.ALREADY_COMPLETED:
        if completed is None:
            raise RequestBindingError("completed request claim has no result")
        if completed.get("status") != "succeeded":
            return completed
        skipped = dict(completed)
        skipped["status"] = "skipped"
        skipped["input_tokens"] = 0
        skipped["output_tokens"] = 0
        skipped["cost_usd"] = 0.0
        skipped["estimated_cost_usd"] = 0.0
        return skipped
    total_cost = 0.0
    total_cost_known = True
    total_input = 0
    total_output = 0
    raw_hashes: list[str] = []
    raw_paths: list[str] = []
    for attempt_index in (1, 2):
        if attempt_index == 2 and (
            _known_cumulative_cost(prepared) + total_cost + next_cost
            >= ABSOLUTE_CAP_USD
        ):
            result = _result_record(
                request,
                status="failed",
                output=None,
                estimated_cost_usd=total_cost,
                cost_known=total_cost_known,
                cost_reason="cost_gate",
                input_tokens=total_input,
                output_tokens=total_output,
                raw_hashes=raw_hashes,
                raw_paths=raw_paths,
                failure_code="cost_gate",
            )
            _append_result(prepared, condition_id, result)
            _record_cost_state(prepared)
            return result
        _append_jsonl(
            prepared.path / "attempts.jsonl",
            _attempt_record(
                request,
                status="started",
                attempt_index=attempt_index,
                cost_known=False,
                cost_reason="not_observed",
            ),
        )
        outcome = _execute_attempt(
            prepared,
            condition_id=condition_id,
            row=row,
            transport=transport,
            delivery=delivery,
            prompt=prompt,
            provider_schema=provider_schema,
            validated_map=validated_map,
        )
        generated = outcome["generated"]
        if isinstance(generated, GenerationResult):
            raw_hashes.extend(outcome["raw_hashes"])
            raw_paths.extend(outcome["raw_paths"])
        attempt_input = int(outcome["input_tokens"])
        attempt_output = int(outcome["output_tokens"])
        attempt_cost = outcome["cost_usd"]
        attempt_known = bool(outcome["cost_known"])
        total_input += attempt_input
        total_output += attempt_output
        if attempt_known and total_cost_known:
            total_cost += float(attempt_cost)
        else:
            total_cost_known = False
        error = outcome["error"]
        cleanup_error = outcome["cleanup_error"]
        if cleanup_error is not None:
            failure_code = "cleanup_failure"
        elif not attempt_known:
            failure_code = "cost_unknown"
        elif error is None:
            failure_code = None
        else:
            failure_code = _failure_code(error, generated)
        terminal_status = "succeeded" if error is None and cleanup_error is None else "failed"
        if failure_code is None:
            _append_jsonl(
                prepared.path / "attempts.jsonl",
                _attempt_record(
                    request,
                    status="succeeded",
                    attempt_index=attempt_index,
                    input_tokens=attempt_input,
                    output_tokens=attempt_output,
                    cost_usd=float(attempt_cost),
                    cost_known=attempt_known,
                    cost_reason=outcome["cost_reason"],
                ),
            )
            result = _result_record(
                request,
                status=terminal_status,
                output=outcome["output"],
                estimated_cost_usd=total_cost,
                cost_known=total_cost_known,
                cost_reason=outcome["cost_reason"],
                input_tokens=total_input,
                output_tokens=total_output,
                raw_hashes=raw_hashes,
                raw_paths=raw_paths,
            )
            _append_result(prepared, condition_id, result)
            _record_cost_state(prepared)
            return result
        _append_jsonl(
            prepared.path / "attempts.jsonl",
            _attempt_record(
                request,
                status="failed",
                attempt_index=attempt_index,
                failure_code=failure_code,
                input_tokens=attempt_input,
                output_tokens=attempt_output,
                cost_usd=attempt_cost,
                cost_known=attempt_known,
                cost_reason=outcome["cost_reason"],
            ),
        )
        total_failure = failure_code
        _record_cost_state(prepared)
        if attempt_index == 1 and attempt_known and failure_code in _RETRYABLE:
            continue
        result = _result_record(
            request,
            status="failed",
            output=None,
            estimated_cost_usd=total_cost,
            cost_known=total_cost_known,
            cost_reason=outcome["cost_reason"],
            input_tokens=total_input,
            output_tokens=total_output,
            raw_hashes=raw_hashes,
            raw_paths=raw_paths,
            failure_code=total_failure,
        )
        _append_result(prepared, condition_id, result)
        _record_cost_state(prepared)
        return result
    raise RuntimeError("unreachable retry loop")


def _execute_attempt(
    prepared: PreparedPhase0Run,
    *,
    condition_id: str,
    row: PanelRow,
    transport: FrontierTransport,
    delivery: DeliveryMedia,
    prompt: str,
    provider_schema: Mapping[str, object],
    validated_map: NeutralVideoMap | None,
) -> dict[str, object]:
    uploaded: UploadedFileRecord | None = None
    generated: GenerationResult | None = None
    error: Exception | None = None
    cleanup_error: Exception | None = None
    output: Mapping[str, object] | None = None
    raw_hashes: list[str] = []
    raw_paths: list[str] = []
    try:
        try:
            uploaded = transport.upload(delivery.path, mime_type=delivery.mime_type)
            if (
                not isinstance(uploaded, UploadedFileRecord)
                or uploaded.sha256 != delivery.sha256
                or uploaded.mime_type != delivery.mime_type
            ):
                raise DeliveryMediaError("uploaded delivery identity changed")
        except Exception as upload_error:
            error = upload_error
        if error is None:
            try:
                generated = transport.generate(
                    model=MODEL_ID,
                    parts=(
                        {"file_id": uploaded.file_id, "mime_type": uploaded.mime_type},
                        prompt,
                    ),
                    schema=provider_schema,
                )
                if not isinstance(generated, GenerationResult):
                    raise TypeError("transport returned an invalid generation result")
                raw_path = _preserve_generation(
                    prepared,
                    condition_id=condition_id,
                    video_id=row.video_id,
                    generated=generated,
                )
                raw_hashes.append(generated.response_sha256)
                raw_paths.append(raw_path)
                _validate_provider_model(generated)
                parsed = parse_prompt_json_response(generated.raw_response)
                output = _validate_output(
                    condition_id,
                    parsed,
                    row.duration_seconds,
                    validated_map,
                )
            except ProviderResponseEvidenceError as evidence_error:
                generated = evidence_error.generation_result
                raw_path = _preserve_generation(
                    prepared,
                    condition_id=condition_id,
                    video_id=row.video_id,
                    generated=generated,
                )
                raw_hashes.append(generated.response_sha256)
                raw_paths.append(raw_path)
                error = evidence_error
            except AmbiguousStartedAttemptError:
                raise
            except Exception as generation_error:
                error = generation_error
    finally:
        if uploaded is not None:
            try:
                transport.delete(uploaded.file_id)
            except Exception as delete_error:
                cleanup_error = delete_error
    cost_usd, cost_known, cost_reason = _cost_observation(
        generated=generated,
        error=error,
        provider_reached=uploaded is not None,
    )
    if cleanup_error is not None and not cost_known and generated is None:
        cost_reason = "cleanup_failure_cost_unknown"
    return {
        "generated": generated,
        "error": error,
        "cleanup_error": cleanup_error,
        "output": output,
        "raw_hashes": raw_hashes,
        "raw_paths": raw_paths,
        "input_tokens": 0 if generated is None else generated.usage.input_tokens,
        "output_tokens": 0 if generated is None else generated.usage.output_tokens,
        "cost_usd": cost_usd,
        "cost_known": cost_known,
        "cost_reason": cost_reason,
    }


def _condition_inputs(
    condition_id: str,
    row: PanelRow,
    neutral_map: NeutralVideoMap | None,
    *,
    prepared: PreparedPhase0Run,
) -> tuple[str, Mapping[str, object], Mapping[str, object], NeutralVideoMap | None]:
    if condition_id == "neutral_map":
        if neutral_map is not None:
            raise ValueError("neutral_map condition cannot receive a prior map")
        return (
            build_neutral_map_prompt(row.duration_seconds),
            NEUTRAL_MAP_SCHEMA,
            PROVIDER_NEUTRAL_MAP_SCHEMA,
            None,
        )
    if condition_id == "direct_answer":
        if neutral_map is not None:
            raise ValueError("direct answer cannot receive a neutral map")
        return (
            build_grounded_answer_prompt(row.question, row.duration_seconds, None),
            GROUNDED_ANSWER_SCHEMA,
            PROVIDER_GROUNDED_ANSWER_SCHEMA,
            None,
        )
    if condition_id == "map_guided_answer":
        if neutral_map is None:
            neutral_result = _existing_logical_result(
                prepared, "screen", row, "neutral_map"
            )
            if neutral_result is None or neutral_result.get("status") != "succeeded":
                raise Phase0RunnerError("map-guided answer requires a completed neutral map")
            neutral_map = _neutral_map_from_result(neutral_result, row.duration_seconds)
        return (
            build_grounded_answer_prompt(
                row.question,
                row.duration_seconds,
                neutral_map.to_dict(),
            ),
            GROUNDED_ANSWER_SCHEMA,
            PROVIDER_GROUNDED_ANSWER_SCHEMA,
            neutral_map,
        )
    raise ValueError("unsupported Phase 0 condition")


def _validate_output(
    condition_id: str,
    parsed: object,
    duration_seconds: float,
    neutral_map: NeutralVideoMap | None,
) -> Mapping[str, object]:
    if condition_id == "neutral_map":
        return validate_neutral_map(parsed, duration_seconds).to_dict()
    allowed = None if condition_id == "direct_answer" else {
        event["event_id"]
        for event in neutral_map.to_dict()["events"]
        if isinstance(event, Mapping)
    }
    return validate_grounded_answer(
        parsed,
        duration_seconds,
        allowed_map_event_ids=allowed,
    ).to_dict()


def _request_record(
    prepared: PreparedPhase0Run,
    *,
    phase: str,
    row: PanelRow,
    condition_id: str,
    prompt: str,
    schema: Mapping[str, object],
    provider_schema: Mapping[str, object],
    delivery: DeliveryMedia,
    neutral_map: NeutralVideoMap | None,
) -> dict[str, object]:
    policy = MAP_POLICY if condition_id == "neutral_map" else ANSWER_POLICY
    binding: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared.run_id,
        "phase": phase,
        "panel_sha256": prepared.panel_sha256,
        "source_sha256": row.media_sha256,
        "source_size_bytes": row.size_bytes,
        "video_id": row.video_id,
        "model_id": MODEL_ID,
        "provider": OPENROUTER_PROVIDER,
        "route": "openrouter-pinned-model-no-fallback",
        "condition_id": condition_id,
        "prompt_sha256": canonical_sha256(prompt),
        "schema_sha256": canonical_sha256(schema),
        "provider_schema_sha256": canonical_sha256(provider_schema),
        "response_format_policy": _RESPONSE_FORMAT_POLICY,
        "request_policy": _policy_payload(policy),
        "delivery_provider": DELIVERY_PROVIDER,
        **delivery.request_identity(),
        "delivery_source_sha256": delivery.source_sha256,
        "delivery_source_size_bytes": delivery.source_size_bytes,
        "neutral_map_sha256": None
        if neutral_map is None
        else neutral_map.sha256,
    }
    request_sha256 = canonical_sha256(binding)
    return {
        **binding,
        "request_binding": binding,
        "request_binding_sha256": request_sha256,
        "request_sha256": request_sha256,
        # Keep these flattened fields as an operator-friendly compatibility
        # projection; the nested binding is the authoritative identity.
        **binding["request_policy"],
    }


def _attempt_record(
    request: Mapping[str, object],
    *,
    status: str,
    attempt_index: int,
    failure_code: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    cost_usd: float | None = None,
    cost_known: bool = False,
    cost_reason: str = "not_observed",
) -> dict[str, object]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": request["run_id"],
        "phase": request["phase"],
        "video_id": request["video_id"],
        "condition_id": request["condition_id"],
        "request_sha256": request["request_sha256"],
        "request_binding_sha256": request["request_binding_sha256"],
        "request_binding": request["request_binding"],
        "source_sha256": request["source_sha256"],
        "delivery_sha256": request["delivery_sha256"],
        "model_id": request["model_id"],
        "prompt_sha256": request["prompt_sha256"],
        "schema_sha256": request["schema_sha256"],
        "neutral_map_sha256": request["neutral_map_sha256"],
        "status": status,
        "attempt_index": attempt_index,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": None if not cost_known else round(
            estimated_cost_usd if cost_usd is None else cost_usd, 6
        ),
        "cost_usd": None if not cost_known or cost_usd is None else round(cost_usd, 6),
        "cost_known": cost_known,
        "cost_status": "known" if cost_known else "unknown",
        "cost_reason": cost_reason,
    }
    if failure_code is not None:
        record["failure_code"] = failure_code
    return record


def _result_record(
    request: Mapping[str, object],
    *,
    status: str,
    output: Mapping[str, object] | None,
    estimated_cost_usd: float,
    cost_known: bool,
    cost_reason: str,
    input_tokens: int,
    output_tokens: int,
    raw_hashes: Sequence[str],
    raw_paths: Sequence[str],
    failure_code: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        **dict(request),
        "status": status,
        "validated_output": output,
        "raw_response_sha256": list(raw_hashes),
        "raw_response_paths": list(raw_paths),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": None if not cost_known else round(estimated_cost_usd, 6),
        "cost_usd": None if not cost_known else round(estimated_cost_usd, 6),
        "cost_known": cost_known,
        "cost_status": "known" if cost_known else "unknown",
        "cost_reason": cost_reason,
    }
    if failure_code is not None:
        record["failure_code"] = failure_code
    record["result_sha256"] = canonical_sha256(record)
    return record


def _append_result(
    prepared: PreparedPhase0Run,
    condition_id: str,
    result: Mapping[str, object],
) -> None:
    _append_jsonl(prepared.path / "results" / f"{condition_id}.jsonl", result)


def _preserve_generation(
    prepared: PreparedPhase0Run,
    *,
    condition_id: str,
    video_id: str,
    generated: GenerationResult,
) -> str:
    try:
        preserve_raw_response(
            prepared.path,
            run_id=prepared.run_id,
            condition_id=condition_id,
            video_id=video_id,
            raw_response=generated.raw_response,
        )
        primary = prepared.path / "raw" / condition_id / f"{video_id}.json"
        return primary.relative_to(prepared.path).as_posix()
    except RawResponseConflictError:
        digest = generated.response_sha256
        conflict_root = prepared.path / "raw-conflicts" / condition_id / video_id
        for candidate in sorted(conflict_root.glob("*.json")):
            try:
                payload = _read_json(candidate)
            except ValueError:
                continue
            raw = payload.get("raw_response")
            if isinstance(raw, str) and hashlib.sha256(raw.encode("utf-8")).hexdigest() == digest:
                return candidate.relative_to(prepared.path).as_posix()
        raise AmbiguousStartedAttemptError("raw response conflict could not be located")


def _prepare_delivery(prepared: PreparedPhase0Run, row: PanelRow) -> DeliveryMedia:
    source = _source_path(prepared, row)
    delivery = prepare_delivery_media(
        run_path=prepared.path / "delivery" / DELIVERY_PROVIDER / row.video_id,
        source_path=source,
        video_id=row.video_id,
        provider=DELIVERY_PROVIDER,
    )
    return verify_delivery_media(delivery)


def _source_path(prepared: PreparedPhase0Run, row: PanelRow) -> Path:
    suffix = Path(row.media_relative_path).suffix.lower()
    fixture_root = prepared.path / "fixture-media"
    candidates = [
        fixture_root / f"{row.video_id}{suffix}",
        fixture_root / f"{row.video_id}.webm",
        fixture_root / f"{row.video_id}.mp4",
        fixture_root / f"{row.video_id}.mkv",
    ]
    source: Path | None = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        data_root = Path("/home/npatta01/data/competitions/trec_video_2026/data")
        candidate = (data_root / row.media_relative_path).resolve()
        if candidate.is_relative_to(data_root.resolve()) and candidate.is_file():
            source = candidate
    if source is None or source.is_symlink():
        raise DeliveryMediaError("source media is not available for delivery")
    source = source.resolve(strict=True)
    if _sha256_file(source) != row.media_sha256 or source.stat().st_size != row.size_bytes:
        raise DeliveryMediaError("source media bytes do not match frozen panel identity")
    return source


def _existing_logical_result(
    prepared: PreparedPhase0Run,
    phase: str,
    row: PanelRow,
    condition_id: str,
) -> Mapping[str, object] | None:
    matches = [
        result
        for result in _read_jsonl(prepared.path / "results" / f"{condition_id}.jsonl")
        if result.get("run_id") == prepared.run_id
        and result.get("phase") == phase
        and result.get("video_id") == row.video_id
        and result.get("condition_id") == condition_id
    ]
    if len(matches) > 1:
        raise RequestBindingError("multiple terminal records bind one request")
    if matches and (
        not _result_hash_valid(matches[0]) or not _result_binding_valid(matches[0])
    ):
        raise RequestBindingError("terminal result identity changed")
    return matches[0] if matches else None


def _binding_matches(existing: Mapping[str, object], request: Mapping[str, object]) -> bool:
    return (
        all(existing.get(key) == request.get(key) for key in request)
        and
        existing.get("request_sha256") == request.get("request_sha256")
        and existing.get("request_binding_sha256")
        == request.get("request_binding_sha256")
        and existing.get("request_binding") == request.get("request_binding")
        and canonical_sha256(existing.get("request_binding"))
        == existing.get("request_binding_sha256")
        and canonical_sha256(request.get("request_binding"))
        == request.get("request_binding_sha256")
    )


def _ensure_request_record(
    prepared: PreparedPhase0Run,
    request: Mapping[str, object],
) -> None:
    records = _read_jsonl(prepared.path / "requests.jsonl")
    logical_matches = [
        existing
        for existing in records
        if existing.get("run_id") == request["run_id"]
        and existing.get("phase") == request["phase"]
        and existing.get("video_id") == request["video_id"]
        and existing.get("condition_id") == request["condition_id"]
    ]
    if len(logical_matches) > 1:
        raise RequestBindingError("multiple request records bind one logical request")
    for existing in records:
        if existing.get("request_sha256") == request["request_sha256"]:
            if not _binding_matches(existing, request):
                raise RequestBindingError("request identity changed")
            return
        if (
            existing.get("run_id") == request["run_id"]
            and existing.get("phase") == request["phase"]
            and existing.get("video_id") == request["video_id"]
            and existing.get("condition_id") == request["condition_id"]
        ):
            raise RequestBindingError("request identity changed")
    _append_jsonl(prepared.path / "requests.jsonl", request)


def _claim_request(
    prepared: PreparedPhase0Run,
    request: Mapping[str, object],
) -> tuple[RequestClaimDisposition, Mapping[str, object] | None]:
    path = prepared.path / "request-claims" / f"{request['request_sha256']}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "claimed",
        "run_id": prepared.run_id,
        "phase": request["phase"],
        "video_id": request["video_id"],
        "condition_id": request["condition_id"],
        "request_sha256": request["request_sha256"],
        "request_binding_sha256": request["request_binding_sha256"],
        "request_binding": request["request_binding"],
    }
    try:
        _write_json_once(path, payload)
    except FileExistsError as error:
        existing = _read_json(path)
        if existing == payload:
            completed = _completed_result_for_request(prepared, request)
            if completed is not None:
                return RequestClaimDisposition.ALREADY_COMPLETED, completed
        raise AmbiguousStartedAttemptError(
            "permanent request claim without a valid terminal result requires operator review"
        ) from error
    return RequestClaimDisposition.OWNED, None


def _completed_result_for_request(
    prepared: PreparedPhase0Run,
    request: Mapping[str, object],
) -> Mapping[str, object] | None:
    matches: list[Mapping[str, object]] = []
    for condition in _CONDITIONS:
        for result in _read_jsonl(prepared.path / "results" / f"{condition}.jsonl"):
            if (
                result.get("run_id") == prepared.run_id
                and result.get("request_sha256") == request.get("request_sha256")
                and result.get("status") in _TERMINAL
                and result.get("request_binding_sha256")
                == request.get("request_binding_sha256")
                and result.get("request_binding") == request.get("request_binding")
            ):
                if not _result_hash_valid(result) or not _result_binding_valid(result):
                    raise RequestBindingError("terminal result identity changed")
                matches.append(result)
    if len(matches) > 1:
        raise RequestBindingError("multiple terminal records bind one request")
    return matches[0] if matches else None


def _result_for_request_is_valid(
    prepared: PreparedPhase0Run,
    request: Mapping[str, object],
) -> bool:
    return _completed_result_for_request(prepared, request) is not None


def _fail_on_ambiguous_started_attempts(prepared: PreparedPhase0Run) -> None:
    if _orphan_claims(prepared):
        raise AmbiguousStartedAttemptError(
            "permanent request claim without a valid terminal result requires operator review"
        )
    ambiguous = _ambiguous_attempts(prepared, _read_jsonl(prepared.path / "attempts.jsonl"))
    if ambiguous:
        if any(
            (
                isinstance(attempt.get("request_sha256"), str)
                and (prepared.path / "request-claims" / f"{attempt['request_sha256']}.json").is_file()
                and not _result_for_request_is_valid(
                    prepared,
                    {
                        "request_sha256": attempt["request_sha256"],
                        "request_binding_sha256": attempt.get("request_binding_sha256"),
                        "request_binding": attempt.get("request_binding"),
                    },
                )
            )
            for attempt in ambiguous
        ):
            raise AmbiguousStartedAttemptError(
                "permanent request claim without a valid terminal result requires operator review"
            )
        raise AmbiguousStartedAttemptError("started attempt lacks a terminal record")


def _orphan_claims(prepared: PreparedPhase0Run) -> list[Mapping[str, object]]:
    claims_dir = prepared.path / "request-claims"
    if not claims_dir.is_dir() or claims_dir.is_symlink():
        return []
    orphaned: list[Mapping[str, object]] = []
    for path in sorted(claims_dir.glob("*.json")):
        try:
            claim = _read_json(path)
        except ValueError:
            orphaned.append({"path": path.name})
            continue
        if claim.get("run_id") != prepared.run_id:
            orphaned.append(claim)
            continue
        if not _result_for_request_is_valid(prepared, claim):
            orphaned.append(claim)
    return orphaned


def _ambiguous_attempts(
    prepared: PreparedPhase0Run,
    attempts: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    invalid_bindings = [
        attempt
        for attempt in attempts
        if attempt.get("run_id") == prepared.run_id
        and not _request_binding_record_valid(attempt)
    ]
    terminal_keys = {
        (
            attempt.get("condition_id"),
            attempt.get("phase"),
            attempt.get("video_id"),
            attempt.get("request_sha256"),
            attempt.get("attempt_index"),
        )
        for attempt in attempts
        if attempt.get("run_id") == prepared.run_id and attempt.get("status") in _TERMINAL
    }
    ambiguous = invalid_bindings + [
        attempt
        for attempt in attempts
        if attempt.get("run_id") == prepared.run_id
        and attempt.get("status") == "started"
        and (
            attempt.get("condition_id"),
            attempt.get("phase"),
            attempt.get("video_id"),
            attempt.get("request_sha256"),
            attempt.get("attempt_index"),
        )
        not in terminal_keys
    ]
    terminal_results = [
        result
        for condition in _CONDITIONS
        for result in _read_jsonl(prepared.path / "results" / f"{condition}.jsonl")
        if result.get("run_id") == prepared.run_id
        and result.get("status") in _TERMINAL
        and _result_hash_valid(result)
        and _result_binding_valid(result)
    ]
    for attempt in attempts:
        if attempt.get("run_id") != prepared.run_id or attempt.get("status") not in _TERMINAL:
            continue
        matches = [
            result
            for result in terminal_results
            if result.get("request_sha256") == attempt.get("request_sha256")
        ]
        if len(matches) != 1:
            ambiguous.append(attempt)
    return ambiguous


def _request_binding_record_valid(record: Mapping[str, object]) -> bool:
    binding = record.get("request_binding")
    digest = record.get("request_binding_sha256")
    return (
        isinstance(binding, Mapping)
        and isinstance(digest, str)
        and canonical_sha256(binding) == digest
        and record.get("request_sha256") == digest
        and all(
            key not in record or record.get(key) == value
            for key, value in binding.items()
        )
    )


def _neutral_map_from_result(
    result: Mapping[str, object], duration_seconds: float
) -> NeutralVideoMap:
    output = result.get("validated_output")
    if not isinstance(output, Mapping):
        raise Phase0RunnerError("neutral map result has no validated output")
    return validate_neutral_map(output, duration_seconds)


def _summary_from_outcomes(
    phase: str,
    outcomes: Sequence[Mapping[str, object]],
) -> Phase0RunSummary:
    return Phase0RunSummary(
        phase=phase,
        expected=len(outcomes),
        passed=sum(result.get("status") == "succeeded" for result in outcomes),
        failed=sum(result.get("status") == "failed" for result in outcomes),
        skipped=sum(result.get("status") == "skipped" for result in outcomes),
        input_tokens=sum(_int(result.get("input_tokens", 0)) for result in outcomes),
        output_tokens=sum(_int(result.get("output_tokens", 0)) for result in outcomes),
        observed_cost_usd=(
            None
            if any(result.get("cost_known") is not True for result in outcomes)
            else round(
                sum(_number(result.get("cost_usd", 0.0)) for result in outcomes),
                6,
            )
        ),
    )


def _combine_summaries(
    phase: str,
    summaries: Sequence[Phase0RunSummary],
) -> Phase0RunSummary:
    return Phase0RunSummary(
        phase=phase,
        expected=sum(summary.expected for summary in summaries),
        passed=sum(summary.passed for summary in summaries),
        failed=sum(summary.failed for summary in summaries),
        skipped=sum(summary.skipped for summary in summaries),
        input_tokens=sum(summary.input_tokens for summary in summaries),
        output_tokens=sum(summary.output_tokens for summary in summaries),
        observed_cost_usd=(
            None
            if any(summary.observed_cost_usd is None for summary in summaries)
            else round(sum(summary.observed_cost_usd or 0.0 for summary in summaries), 6)
        ),
    )


def _cost_observation(
    *,
    generated: GenerationResult | None,
    error: Exception | None,
    provider_reached: bool,
) -> tuple[float | None, bool, str]:
    metadata = {} if generated is None else generated.provider_metadata
    reported = metadata.get("reported_cost_usd")
    if reported is None and error is not None:
        reported = getattr(error, "reported_cost_usd", None)
    if isinstance(reported, (int, float)) and not isinstance(reported, bool):
        if math.isfinite(float(reported)) and float(reported) >= 0:
            return float(reported), True, "provider_reported"
    usage = None if generated is None else generated.usage
    if usage is None and error is not None:
        usage = getattr(error, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if (
        isinstance(input_tokens, int)
        and not isinstance(input_tokens, bool)
        and isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
        and (input_tokens > 0 or output_tokens > 0)
    ):
        return (
            (input_tokens * 2.0 + output_tokens * 12.0) / 1_000_000,
            True,
            "token_usage",
        )
    if not provider_reached:
        return 0.0, True, "provider_not_reached"
    return None, False, "missing_provider_cost_and_usage"


def _validate_transport(transport: FrontierTransport, condition_id: str) -> None:
    if getattr(transport, "provider_name", None) != OPENROUTER_PROVIDER:
        raise Phase0RunnerError("transport provider must be openrouter")
    if getattr(transport, "response_format_policy", None) != _RESPONSE_FORMAT_POLICY:
        raise Phase0RunnerError("transport response-format policy is not exact-fence")
    if not all(callable(getattr(transport, name, None)) for name in ("upload", "generate", "delete")):
        raise Phase0RunnerError("transport must expose callable upload/generate/delete")
    expected = MAP_POLICY if condition_id == "neutral_map" else ANSWER_POLICY
    identity = getattr(transport, "policy_identity", None)
    observed = identity.get(condition_id) if isinstance(identity, Mapping) else None
    if observed is None:
        public_policy = getattr(transport, "request_policy", None)
        if public_policy is not None:
            observed = public_policy
    if isinstance(transport, OpenRouterTransport):
        observed = getattr(transport, "_policy", None)
    if observed != expected:
        raise Phase0RunnerError("transport request policy does not match condition")


def _transport_estimate(
    transport: FrontierTransport,
    condition_id: str,
    row: PanelRow,
) -> float:
    estimator = getattr(transport, "estimate_cost_usd", None)
    if callable(estimator):
        try:
            return _number(estimator(OPENROUTER_PROVIDER, row))
        except TypeError:
            return _number(estimator(condition_id, row))
    for name in ("next_cost", "estimated_cost_usd"):
        value = getattr(transport, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return _number(value)
    return _conservative_cost_estimate(condition_id)


def _conservative_cost_estimate(condition_id: str) -> float:
    output_tokens = _MAP_OUTPUT_TOKENS if condition_id == "neutral_map" else _ANSWER_OUTPUT_TOKENS
    return (_DEFAULT_INPUT_TOKENS * 2.0 + output_tokens * 12.0) / 1_000_000


def _failure_code(error: BaseException, generated: GenerationResult | None) -> str:
    if isinstance(error, ProviderAuthenticationError):
        return "authentication"
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return "rate_limit"
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return "transport"
    if isinstance(status_code, int) and 400 <= status_code <= 499:
        return "http_client_error"
    if isinstance(error, ProviderResponseEvidenceError):
        metadata = generated.provider_metadata if generated is not None else {}
        if metadata.get("content_empty") is True:
            return "empty"
        if metadata.get("finish_reason") in {
            "length",
            "max_tokens",
            "max_output_tokens",
            "truncated",
        }:
            return "truncated"
        return "truncated" if _has_truncation_evidence(generated) else "provider_protocol"
    if generated is not None and not generated.raw_response:
        return "empty"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(error, ProviderError):
        code = getattr(error, "failure_code", "")
        if code in {"provider_request", "provider_generation"}:
            return "transport"
        message = str(error).casefold()
        if any(
            marker in message
            for marker in ("timeout", "timed out", "connection reset", "temporarily unavailable")
        ) and not (isinstance(status_code, int) and 400 <= status_code <= 499):
            return "transport"
        if code == "protocol_no_response":
            return "empty"
        return "provider_failure"
    if isinstance(error, ProviderProtocolError):
        return "provider_protocol"
    if isinstance(error, DeliveryMediaError):
        return "delivery_media"
    if isinstance(error, OSError):
        return "filesystem_error"
    if isinstance(error, ValueError):
        return "schema_validation"
    return "internal_error"


def _has_truncation_evidence(generated: GenerationResult | None) -> bool:
    if generated is None:
        return False
    finish_reason = generated.provider_metadata.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason.casefold() in {
        "length",
        "max_tokens",
        "max_output_tokens",
        "truncated",
    }:
        return True
    try:
        payload = json.loads(generated.raw_response)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    for key in ("finish_reason", "finishReason", "finish", "reason"):
        value = payload.get(key)
        if isinstance(value, str) and value.casefold() in {
            "length",
            "max_tokens",
            "max_output_tokens",
            "truncated",
        }:
            return True
    return False


def _validate_provider_model(generated: GenerationResult) -> None:
    observed = generated.provider_metadata.get("model")
    if observed is not None and observed not in {MODEL_ID, _MODEL_SLUG}:
        raise ProviderProtocolError("observed model mismatch")


def _require_complete_phase(
    prepared: PreparedPhase0Run,
    phase: str,
) -> list[Mapping[str, object]]:
    found = _phase_results(prepared, phase)
    if _ambiguous_attempts(prepared, _read_jsonl(prepared.path / "attempts.jsonl")):
        raise AmbiguousStartedAttemptError("started attempt lacks a terminal record")
    return found


def _phase_results(
    prepared: PreparedPhase0Run,
    phase: str,
) -> list[Mapping[str, object]]:
    if phase == "pilot":
        expected = {("pilot", prepared.pilot_row.video_id, "neutral_map")}
    elif phase == "screen":
        expected = {
            ("screen", row.video_id, condition)
            for row in prepared.screen_rows
            for condition in _CONDITIONS
        }
    else:
        raise ValueError("unsupported Phase 0 phase")
    found: dict[tuple[object, object, object], Mapping[str, object]] = {}
    for condition in _CONDITIONS:
        for result in _bound_results(prepared, condition):
            key = (result.get("phase"), result.get("video_id"), condition)
            if key in expected:
                if key in found:
                    raise RequestBindingError("multiple terminal records bind one request")
                found[key] = result
    if set(found) != expected:
        raise Phase0RunnerError(f"{phase} results are incomplete")
    return [
        found[key]
        for key in sorted(found, key=lambda item: (str(item[0]), str(item[1]), str(item[2])))
    ]


def _raw_response_file_records(
    prepared: PreparedPhase0Run,
    results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for result in results:
        raw_paths = result.get("raw_response_paths")
        raw_hashes = result.get("raw_response_sha256")
        if (
            not isinstance(raw_paths, Sequence)
            or isinstance(raw_paths, (str, bytes, bytearray))
            or not isinstance(raw_hashes, Sequence)
            or isinstance(raw_hashes, (str, bytes, bytearray))
            or len(raw_paths) != len(raw_hashes)
        ):
            raise RequestBindingError("raw response paths and hashes do not match")
        condition_id = result.get("condition_id")
        video_id = result.get("video_id")
        if not isinstance(condition_id, str) or not isinstance(video_id, str):
            raise RequestBindingError("raw response identity is invalid")
        for raw_path_value, response_sha256 in zip(raw_paths, raw_hashes):
            if not isinstance(raw_path_value, str) or not isinstance(response_sha256, str):
                raise RequestBindingError("raw response path or hash is invalid")
            raw_path = _safe_raw_response_path(prepared, raw_path_value)
            try:
                verify_raw_response(
                    raw_path,
                    run_id=prepared.run_id,
                    condition_id=condition_id,
                    video_id=video_id,
                    response_sha256=response_sha256,
                )
                file_sha256, size_bytes = _raw_response_file_identity(raw_path)
            except (OSError, ValueError) as error:
                raise RequestBindingError(
                    "preserved raw response integrity changed"
                ) from error
            records.append(
                {
                    "path": raw_path_value,
                    "sha256": file_sha256,
                    "size_bytes": size_bytes,
                }
            )
    return records


def _safe_raw_response_path(prepared: PreparedPhase0Run, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if (
        candidate.is_absolute()
        or not raw_path
        or raw_path != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise RequestBindingError("raw response path is unsafe")
    return prepared.path / candidate


def _raw_response_file_identity(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("raw response path is missing or symlinked")
    return _sha256_file(path), path.stat().st_size


def _freeze_phase(
    prepared: PreparedPhase0Run,
    phase: str,
    results: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    artifact_name = "pilot-results.json" if phase == "pilot" else "screen-results.json"
    freeze_name = "pilot-freeze.json" if phase == "pilot" else "screen-freeze.json"
    artifact_path = prepared.path / artifact_name
    freeze_path = prepared.path / freeze_name
    result_hashes = [str(result["result_sha256"]) for result in results]
    request_hashes = [str(result["request_sha256"]) for result in results]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "runner_id": RUNNER_ID,
        "run_id": prepared.run_id,
        "phase": phase,
        "panel_sha256": prepared.panel_sha256,
        "result_count": len(results),
        "results_sha256": canonical_sha256(list(results)),
        "result_hashes": result_hashes,
        "request_hashes": request_hashes,
        "result_hashes_sha256": canonical_sha256(result_hashes),
        "request_hashes_sha256": canonical_sha256(request_hashes),
        "raw_response_files": _raw_response_file_records(prepared, results),
        "observed_cost_usd": round(
            sum(_number(result.get("estimated_cost_usd", 0.0)) for result in results),
            6,
        ),
        "results": list(results),
    }
    if artifact_path.exists() or artifact_path.is_symlink():
        existing = _read_json(artifact_path)
        legacy_artifact = dict(artifact)
        legacy_artifact.pop("raw_response_files")
        if existing == legacy_artifact:
            artifact = existing
        elif existing != artifact:
            raise FileExistsError(f"refusing to overwrite existing {artifact_name}")
    else:
        _write_json_once(artifact_path, artifact)
    freeze = freeze_json_artifact(
        artifact_path,
        bindings={
            "run_id": prepared.run_id,
            "phase": phase,
            "panel_sha256": prepared.panel_sha256,
            "result_count": len(results),
            "results_sha256": artifact["results_sha256"],
            "result_hashes_sha256": canonical_sha256(result_hashes),
            "request_hashes_sha256": canonical_sha256(request_hashes),
        },
    )
    if freeze_path.exists() or freeze_path.is_symlink():
        existing_freeze = _read_json(freeze_path)
        if existing_freeze != freeze:
            raise FileExistsError(f"refusing to overwrite existing {freeze_name}")
    else:
        _write_json_once(freeze_path, freeze)
    _write_terminal_status(prepared)
    return freeze


def _first_row_results(prepared: PreparedPhase0Run) -> dict[str, Mapping[str, object]]:
    row = prepared.screen_rows[0]
    results: dict[str, Mapping[str, object]] = {}
    for condition in _CONDITIONS:
        result = _existing_logical_result(prepared, "screen", row, condition)
        if result is not None and result.get("status") in _TERMINAL:
            results[condition] = result
    return results


def _bound_results(
    prepared: PreparedPhase0Run,
    condition_id: str,
) -> list[Mapping[str, object]]:
    results = []
    logical: dict[tuple[object, object, object], Mapping[str, object]] = {}
    for result in _read_jsonl(prepared.path / "results" / f"{condition_id}.jsonl"):
        if (
            result.get("run_id") == prepared.run_id
            and result.get("condition_id") == condition_id
            and result.get("status") in _TERMINAL
            and isinstance(result.get("result_sha256"), str)
        ):
            if not _result_hash_valid(result) or not _result_binding_valid(result):
                raise RequestBindingError("terminal result identity changed")
            _raw_response_file_records(prepared, [result])
            key = (result.get("phase"), result.get("video_id"), condition_id)
            if key in logical:
                raise RequestBindingError("multiple terminal records bind one request")
            logical[key] = result
            results.append(result)
    return results


def _all_terminal_results(prepared: PreparedPhase0Run) -> list[Mapping[str, object]]:
    return [
        result
        for condition in _CONDITIONS
        for result in _bound_results(prepared, condition)
    ]


def _result_hash_valid(result: Mapping[str, object]) -> bool:
    declared = result.get("result_sha256")
    if not isinstance(declared, str):
        return False
    unsigned = dict(result)
    del unsigned["result_sha256"]
    return canonical_sha256(unsigned) == declared


def _result_binding_valid(result: Mapping[str, object]) -> bool:
    binding = result.get("request_binding")
    digest = result.get("request_binding_sha256")
    policy = binding.get("request_policy") if isinstance(binding, Mapping) else None
    flattened_policy_valid = isinstance(policy, Mapping) and all(
        result.get(key) == value for key, value in policy.items()
    )
    return (
        isinstance(binding, Mapping)
        and isinstance(digest, str)
        and canonical_sha256(binding) == digest
        and result.get("request_sha256") == digest
        and all(result.get(key) == value for key, value in binding.items())
        and flattened_policy_valid
    )


def _unrepresented_terminal_attempts(
    prepared: PreparedPhase0Run,
    condition_id: str,
    *,
    allowed_video_ids: set[str] | None = None,
) -> tuple[int, float, bool]:
    result_hashes = {
        str(result.get("request_sha256"))
        for result in _bound_results(prepared, condition_id)
    }
    grouped: dict[str, tuple[float, bool]] = {}
    unknown = False
    for attempt in _read_jsonl(prepared.path / "attempts.jsonl"):
        request_sha256 = attempt.get("request_sha256")
        if (
            attempt.get("run_id") == prepared.run_id
            and attempt.get("condition_id") == condition_id
            and (
                allowed_video_ids is None
                or attempt.get("video_id") in allowed_video_ids
                or attempt.get("phase") == "pilot"
            )
            and attempt.get("status") in _TERMINAL
            and isinstance(request_sha256, str)
            and request_sha256 not in result_hashes
        ):
            known = attempt.get("cost_known") is True and attempt.get("cost_usd") is not None
            if not known:
                unknown = True
                grouped.setdefault(request_sha256, (0.0, False))
            else:
                current, current_known = grouped.get(request_sha256, (0.0, True))
                grouped[request_sha256] = (current + _number(attempt["cost_usd"]), current_known)
    return len(grouped), sum(value[0] for value in grouped.values()), unknown


def _cost_state(prepared: PreparedPhase0Run) -> tuple[float | None, bool]:
    total = 0.0
    known = True
    for condition in _CONDITIONS:
        for result in _bound_results(prepared, condition):
            if result.get("cost_known") is not True or result.get("cost_usd") is None:
                known = False
            else:
                total += _number(result["cost_usd"])
        _, attempt_cost, attempts_known = _unrepresented_terminal_attempts(
            prepared, condition
        )
        total += attempt_cost
        known = known and not attempts_known
    return (total if known else None), known


def _known_cumulative_cost(prepared: PreparedPhase0Run) -> float:
    total, known = _cost_state(prepared)
    if not known or total is None:
        raise CostUnknownError("cost_unknown blocks further provider calls")
    return total


def _cumulative_cost(prepared: PreparedPhase0Run) -> float:
    return _known_cumulative_cost(prepared)


def _ensure_cost_known(prepared: PreparedPhase0Run) -> None:
    total, known = _cost_state(prepared)
    if not known or total is None:
        _write_cost_state(prepared, "cost_unknown", None)
        raise CostUnknownError("cost_unknown blocks dispatch, authorization, and freeze")


def _record_cost_state(prepared: PreparedPhase0Run) -> None:
    total, known = _cost_state(prepared)
    if not known:
        _write_cost_state(prepared, "cost_unknown", None)
    elif total is not None and total >= ABSOLUTE_CAP_USD:
        _write_cost_state(prepared, "cap_reached", total)
    else:
        _write_cost_state(prepared, "known", total)


def _write_cost_state(
    prepared: PreparedPhase0Run,
    state: str,
    observed_cost: float | None,
) -> None:
    _write_json_replace(
        prepared.path / "cost-state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": prepared.run_id,
            "panel_sha256": prepared.panel_sha256,
            "state": state,
            "cost_known": state != "cost_unknown",
            "observed_cost_usd": None if observed_cost is None else round(observed_cost, 6),
        },
    )


def _verify_cost_gate(prepared: PreparedPhase0Run, gate: Mapping[str, object]) -> None:
    _ensure_cost_known(prepared)
    if (
        gate.get("status") != "authorized"
        or gate.get("run_id") != prepared.run_id
        or gate.get("panel_sha256") != prepared.panel_sha256
        or gate.get("projected_total_usd_lt_cap") is not True
    ):
        raise CostGateError("cost gate binding is not authorized")
    first = _first_row_results(prepared)
    hashes = gate.get("first_row_result_hashes")
    request_hashes = gate.get("first_row_request_hashes")
    if (
        not isinstance(hashes, Mapping)
        or not isinstance(request_hashes, Mapping)
        or set(hashes) != set(_CONDITIONS)
        or set(request_hashes) != set(_CONDITIONS)
        or any(
            hashes.get(condition) != first.get(condition, {}).get("result_sha256")
            or request_hashes.get(condition)
            != first.get(condition, {}).get("request_sha256")
            for condition in _CONDITIONS
        )
    ):
        raise CostGateError("cost gate first-row result binding changed")
    snapshot = gate.get("projection_snapshot")
    snapshot_sha = gate.get("projection_snapshot_sha256")
    if (
        not isinstance(snapshot, Mapping)
        or not isinstance(snapshot_sha, str)
        or canonical_sha256(snapshot) != snapshot_sha
    ):
        raise CostGateError("cost gate projection snapshot is missing or forged")
    recomputed = project_remaining_cost(prepared, _screen_limit=1)
    if recomputed != snapshot:
        raise CostGateError("cost gate first-row projection binding changed")
    if gate.get("projected_total_usd") != snapshot.get("projected_total_usd"):
        raise CostGateError("cost gate projection total changed")


def _verify_pilot_authorization(prepared: PreparedPhase0Run) -> None:
    path = prepared.path / "pilot-authorization.json"
    payload = _read_json(path)
    if payload.get("status") != "authorized" or payload.get("run_id") != prepared.run_id or payload.get("panel_sha256") != prepared.panel_sha256 or payload.get("pilot_audit_decision") != "usable_for_screen":
        raise PilotGateError("pilot audit authorization binding changed")
    pilot_freeze = prepared.path / "pilot-freeze.json"
    _verify_freeze_file(prepared, pilot_freeze)
    if payload.get("pilot_freeze_sha256") != _sha256_file(pilot_freeze):
        raise PilotGateError("pilot audit authorization is stale")
    pilot_artifact = _read_json(prepared.path / "pilot-results.json")
    if (
        payload.get("pilot_results_artifact_sha256")
        != _sha256_file(prepared.path / "pilot-results.json")
        or payload.get("pilot_result_hashes") != pilot_artifact.get("result_hashes")
        or payload.get("pilot_request_hashes") != pilot_artifact.get("request_hashes")
        or payload.get("pilot_results_sha256") != pilot_artifact.get("results_sha256")
    ):
        raise PilotGateError("pilot results authorization binding changed")
    audit_path = prepared.path / "pilot-audit" / "freeze.json"
    if not audit_path.is_file():
        raise PilotGateError("pilot audit freeze artifact is not retained")
    audit = _read_json(audit_path)
    _verify_audit_freeze(prepared, audit_path, audit)
    if (
        payload.get("pilot_audit_freeze_sha256") != _sha256_file(audit_path)
        or payload.get("pilot_audit_artifact_sha256") != audit.get("artifact_sha256")
        or payload.get("pilot_audit_bindings_sha256") != audit.get("bindings_sha256")
    ):
        raise PilotGateError("pilot audit authorization audit binding changed")


def _verify_freeze_file(prepared: PreparedPhase0Run, path: Path) -> None:
    freeze = _read_json(path)
    phase = "pilot" if path.name == "pilot-freeze.json" else "screen"
    artifact_name = "pilot-results.json" if phase == "pilot" else "screen-results.json"
    artifact_path = path.parent / artifact_name
    try:
        verify_json_artifact(artifact_path, freeze)
    except (TypeError, ValueError) as error:
        raise PilotGateError("frozen results artifact verification failed") from error
    artifact = _read_json(artifact_path)
    expected_results = _phase_results(prepared, phase)
    result_hashes = [str(result["result_sha256"]) for result in expected_results]
    request_hashes = [str(result["request_sha256"]) for result in expected_results]
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("runner_id") != RUNNER_ID
        or artifact.get("run_id") != prepared.run_id
        or artifact.get("panel_sha256") != prepared.panel_sha256
        or artifact.get("phase") != phase
        or artifact.get("results") != expected_results
        or artifact.get("result_count") != len(expected_results)
        or artifact.get("results_sha256") != canonical_sha256(expected_results)
        or artifact.get("result_hashes") != result_hashes
        or artifact.get("request_hashes") != request_hashes
        or artifact.get("result_hashes_sha256") != canonical_sha256(result_hashes)
        or artifact.get("request_hashes_sha256") != canonical_sha256(request_hashes)
    ):
        raise PilotGateError("frozen results artifact binding changed")
    expected_raw_response_files = _raw_response_file_records(prepared, expected_results)
    recorded_raw_response_files = artifact.get("raw_response_files")
    if (
        "raw_response_files" in artifact
        and recorded_raw_response_files != expected_raw_response_files
    ):
        raise PilotGateError("frozen raw response file binding changed")
    expected_bindings = {
        "run_id": prepared.run_id,
        "phase": phase,
        "panel_sha256": prepared.panel_sha256,
        "result_count": len(expected_results),
        "results_sha256": canonical_sha256(expected_results),
        "result_hashes_sha256": canonical_sha256(result_hashes),
        "request_hashes_sha256": canonical_sha256(request_hashes),
    }
    if freeze.get("bindings") != expected_bindings:
        raise PilotGateError("frozen results freeze bindings changed")


def _verify_audit_freeze(
    prepared: PreparedPhase0Run,
    path: Path,
    freeze: Mapping[str, object],
) -> None:
    packet_manifest = path.parent / "packet-manifest.json"
    if packet_manifest.is_symlink() or not packet_manifest.is_file():
        raise PilotGateError(
            "pilot authorization requires a complete Phase 0 audit packet"
        )
    from trec_video_2026.vqa_frontier_phase0_audit import validate_audit_packet

    try:
        validate_audit_packet(path.parent)
    except (TypeError, ValueError) as error:
        raise PilotGateError("pilot audit packet verification failed") from error
    if (
        freeze.get("schema_version") != SCHEMA_VERSION
        or freeze.get("status") != "frozen"
        or not isinstance(freeze.get("artifact_sha256"), str)
        or not _SHA256.fullmatch(str(freeze.get("artifact_sha256")))
        or not isinstance(freeze.get("bindings_sha256"), str)
        or not _SHA256.fullmatch(str(freeze.get("bindings_sha256")))
    ):
        raise PilotGateError("pilot audit authorization requires a frozen audit")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping):
        raise PilotGateError("pilot audit freeze bindings are missing")
    required = {"run_id", "panel_sha256", "phase", "decision"}
    if set(bindings) != required:
        raise PilotGateError("pilot audit freeze core bindings are incomplete")
    if (
        bindings.get("run_id") != prepared.run_id
        or bindings.get("panel_sha256") != prepared.panel_sha256
        or bindings.get("phase") != "pilot"
        or bindings.get("decision") != "usable_for_screen"
    ):
        raise PilotGateError("pilot audit freeze core binding is invalid")
    if not isinstance(_audit_decision(freeze), str):
        raise PilotGateError("pilot audit freeze decision is invalid")
    artifact = path.parent / "audit.json"
    if not artifact.is_file() or artifact.is_symlink():
        raise PilotGateError("pilot audit artifact bound by freeze is missing")
    try:
        verify_json_artifact(artifact, dict(freeze))
    except (TypeError, ValueError) as error:
        raise PilotGateError("pilot audit artifact freeze verification failed") from error
    audit_payload = _read_json(artifact)
    if (
        audit_payload.get("schema_version") != SCHEMA_VERSION
        or audit_payload.get("run_id") != prepared.run_id
        or audit_payload.get("panel_sha256") != prepared.panel_sha256
        or audit_payload.get("phase") != "pilot"
        or audit_payload.get("decision") != "usable_for_screen"
    ):
        raise PilotGateError("pilot audit artifact core binding is invalid")


def _retain_audit_packet_once(source: Path, destination: Path) -> None:
    source_snapshot = _audit_packet_snapshot(source)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise FileExistsError("retained pilot audit packet path is unsafe")
        if _audit_packet_snapshot(destination) != source_snapshot:
            raise FileExistsError("refusing to overwrite retained pilot audit packet")
        return
    shutil.copytree(source, destination, copy_function=shutil.copyfile)
    if _audit_packet_snapshot(destination) != source_snapshot:
        raise PilotGateError("retained pilot audit packet copy changed")


def _audit_packet_snapshot(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise PilotGateError("pilot authorization requires a nonsymlink audit packet")
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PilotGateError(
                "pilot authorization requires a nonsymlink audit packet"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PilotGateError(
                "pilot authorization requires a regular-file audit packet"
            )
        snapshot[path.relative_to(root).as_posix()] = _sha256_file(path)
    return snapshot


def _audit_decision(freeze: Mapping[str, object]) -> object:
    bindings = freeze.get("bindings")
    if isinstance(bindings, Mapping):
        for key in ("neutral_map_decision", "overall_decision", "decision"):
            if key in bindings:
                return bindings[key]
    for key in ("neutral_map_decision", "overall_decision", "decision"):
        if key in freeze:
            return freeze[key]
    return None


def _state_name(
    prepared: PreparedPhase0Run,
    counts: Mapping[str, Mapping[str, int]],
    ambiguous: int,
    ambiguous_claims: int = 0,
) -> str:
    if ambiguous or ambiguous_claims:
        return "blocked-ambiguous-attempt"
    _, cost_known = _cost_state(prepared)
    if not cost_known:
        return "blocked-cost-unknown"
    if (prepared.path / "cost-state.json").is_file():
        cost_state = _read_json(prepared.path / "cost-state.json").get("state")
        if cost_state == "cap_reached":
            return "blocked-cost-cap"
    if (prepared.path / "screen-freeze.json").is_file():
        return "screen-frozen"
    if sum(counts[c]["screen_terminal"] for c in _CONDITIONS) == len(prepared.screen_rows) * len(_CONDITIONS):
        return "screen-complete"
    if (prepared.path / "cost-gate.json").is_file():
        return "remaining-authorized"
    if sum(counts[c]["screen_terminal"] for c in _CONDITIONS) == 3:
        return "first-row-complete"
    if (prepared.path / "pilot-authorization.json").is_file():
        return "pilot-authorized"
    if (prepared.path / "pilot-freeze.json").is_file():
        return "pilot-frozen"
    if counts["neutral_map"]["pilot_terminal"]:
        return "pilot-complete"
    return "prepared"


def _prepared_from_payload(
    path: Path,
    payload: Mapping[str, object],
    *,
    panel_sha256: str,
    manifest_sha256: str,
) -> PreparedPhase0Run:
    panel = _parse_panel(payload)
    return PreparedPhase0Run(
        path=path,
        run_id=panel.run_id,
        panel_sha256=panel_sha256,
        manifest_sha256=manifest_sha256,
        panel=panel,
        pilot_row=panel.pilot_rows[0],
        screen_rows=panel.screen_rows,
    )


def _manifest_payload(panel: Phase0Panel, *, panel_sha256: str) -> dict[str, object]:
    prompt_hashes = _prompt_sha256s()
    return {
        "schema_version": SCHEMA_VERSION,
        "runner_id": RUNNER_ID,
        "run_id": panel.run_id,
        "panel_sha256": panel_sha256,
        "prompt_sha256": prompt_hashes["neutral_map"],
        "answer_prompt_sha256": prompt_hashes["answer"],
        "neutral_schema_sha256": canonical_sha256(NEUTRAL_MAP_SCHEMA),
        "answer_schema_sha256": canonical_sha256(GROUNDED_ANSWER_SCHEMA),
        "provider": OPENROUTER_PROVIDER,
        "model_id": MODEL_ID,
        "map_policy": _policy_payload(MAP_POLICY),
        "answer_policy": _policy_payload(ANSWER_POLICY),
        "conditions": list(_CONDITIONS),
        "pilot_count": len(panel.pilot_rows),
        "screen_count": len(panel.screen_rows),
    }


def _policy_payload(policy: OpenRouterRequestPolicy) -> dict[str, object]:
    return {
        "max_output_tokens": policy.max_output_tokens,
        "temperature": policy.temperature,
        "provider_order": list(policy.provider_order),
        "allow_fallbacks": policy.allow_fallbacks,
        "use_structured_outputs": policy.use_structured_outputs,
    }


def _prompt_sha256s() -> dict[str, str]:
    return {
        "neutral_map": canonical_sha256(build_neutral_map_prompt(1.0).replace("1.0", "{duration}")),
        "answer": canonical_sha256(build_grounded_answer_prompt("{question}", 1.0, None).replace("{question}", "{question}")),
    }


def _parse_panel(payload: Mapping[str, object]) -> Phase0Panel:
    if not isinstance(payload, Mapping):
        raise ValueError("panel must be an object")
    _reject_labels(payload)
    expected_keys = {
        "schema_version",
        "run_id",
        "selection_policy",
        "reference_policy",
        "conditions",
        "pilot_count",
        "screen_count",
        "bindings",
        "examples",
    }
    if set(payload) != expected_keys:
        raise ValueError("panel manifest fields do not match the frozen contract")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("panel schema_version does not match")
    if payload.get("run_id") != PHASE0_RUN_ID:
        raise ValueError("panel run_id does not match")
    if payload.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("panel selection_policy does not match")
    if payload.get("reference_policy") != REFERENCE_POLICY:
        raise ValueError("panel reference_policy does not match")
    if payload.get("conditions") != list(_CONDITIONS):
        raise ValueError("panel conditions do not match")
    if payload.get("pilot_count") != 1 or payload.get("screen_count") != 7:
        raise ValueError("panel counts do not match")
    raw_rows = payload.get("examples")
    if not isinstance(raw_rows, list):
        raise ValueError("panel must contain examples")
    rows = tuple(_parse_row(raw, index) for index, raw in enumerate(raw_rows))
    if len(rows) != 8 or rows[0].phase != "pilot" or any(
        row.phase != "screen" for row in rows[1:]
    ):
        raise ValueError("panel rows must be pilot first followed by seven screen rows")
    if any(row.ordinal <= 0 for row in rows):
        raise ValueError("panel ordinals must be positive")
    if len({row.ordinal for row in rows}) != len(rows):
        raise ValueError("panel ordinals must be unique")
    pilot = (rows[0],)
    screen = rows[1:]
    if len({row.video_id for row in rows}) != len(rows):
        raise ValueError("panel video IDs must be unique")
    if tuple(row.category for row in screen) != CATEGORIES:
        raise ValueError("panel screen categories are not in frozen order")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("panel bindings are required")
    required_bindings = {
        "source_panel_sha256",
        "qwen_predictions_sha256",
        "selected_ordered_ids_sha256",
        "source_media_sha256",
        "selection_policy",
    }
    optional_bindings = {
        "source_panel_file_sha256",
        "qwen_prediction_file_sha256",
    }
    if set(bindings) - required_bindings - optional_bindings:
        raise ValueError("panel contains an unknown binding")
    if set(bindings) & required_bindings != required_bindings:
        raise ValueError("panel is missing a required binding")
    if bindings.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("panel binding selection_policy does not match")
    for field in required_bindings - {"selection_policy"} | optional_bindings & set(bindings):
        if _SHA256.fullmatch(_text(bindings.get(field), field)) is None:
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    expected_ids_sha = canonical_sha256([row.video_id for row in rows])
    expected_media_sha = canonical_sha256(
        [
            {"video_id": row.video_id, "source_sha256": row.media_sha256}
            for row in rows
        ]
    )
    if bindings["selected_ordered_ids_sha256"] != expected_ids_sha:
        raise ValueError("panel ordered-id binding does not match rows")
    if bindings["source_media_sha256"] != expected_media_sha:
        raise ValueError("panel source-media binding does not match rows")
    return Phase0Panel(
        schema_version=SCHEMA_VERSION,
        run_id=PHASE0_RUN_ID,
        selection_policy=SELECTION_POLICY,
        reference_policy=REFERENCE_POLICY,
        source_panel_sha256=str(bindings["source_panel_sha256"]),
        qwen_predictions_sha256=str(bindings["qwen_predictions_sha256"]),
        selected_ordered_ids_sha256=str(bindings["selected_ordered_ids_sha256"]),
        source_media_sha256=str(bindings["source_media_sha256"]),
        rows=(pilot[0], *screen),
        source_panel_file_sha256=bindings.get("source_panel_file_sha256"),
        qwen_prediction_file_sha256=bindings.get("qwen_prediction_file_sha256"),
    )


def _parse_row(raw: object, index: int) -> PanelRow:
    if not isinstance(raw, Mapping):
        raise ValueError(f"examples[{index}] must be an object")
    phase = raw.get("phase")
    required = {
        "ordinal", "phase", "video_id", "question", "category",
        "evidence_topology", "selection_rationale", "media_relative_path",
        "media_sha256", "size_bytes", "duration_seconds", "has_audio",
        "has_video", "condition_ids",
    }
    if phase == "screen":
        required |= {"qwen_condition_id", "qwen_status"}
    if phase == "pilot":
        if "qwen_condition_id" in raw or "qwen_status" in raw:
            raise ValueError("pilot row must not contain Qwen fields")
    if set(raw) != required:
        raise ValueError(f"examples[{index}] fields do not match the frozen contract")
    if phase not in {"pilot", "screen"}:
        raise ValueError("panel row phase must be pilot or screen")
    condition_ids = raw.get("condition_ids")
    expected = ["neutral_map"] if phase == "pilot" else list(_CONDITIONS)
    if condition_ids != expected:
        raise ValueError("panel row condition_ids do not match the phase")
    if phase == "screen" and (
        raw.get("qwen_condition_id") != QWEN_CONDITION_ID
        or raw.get("qwen_status") != "passed"
    ):
        raise ValueError("screen row must bind the passed Qwen baseline")
    video_id = _text(raw.get("video_id"), "video_id")
    if _SAFE_ID.fullmatch(video_id) is None:
        raise ValueError("video_id must be a safe path segment")
    media_path = _text(raw.get("media_relative_path"), "media_relative_path")
    category = _text(raw.get("category"), "category")
    if category not in CATEGORIES:
        raise ValueError("category is not a frozen Phase 0 category")
    duration = _number(raw.get("duration_seconds"))
    if duration <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    media_sha = _text(raw.get("media_sha256"), "media_sha256")
    if _SHA256.fullmatch(media_sha) is None:
        raise ValueError("media_sha256 must be a lowercase SHA-256 digest")
    size = _int(raw.get("size_bytes"))
    if size < 0:
        raise ValueError("size_bytes must be non-negative")
    return PanelRow(
        ordinal=_int(raw.get("ordinal")),
        phase=phase,
        video_id=video_id,
        question=_text(raw.get("question"), "question"),
        category=category,
        evidence_topology=_text(raw.get("evidence_topology"), "evidence_topology"),
        selection_rationale=_text(raw.get("selection_rationale"), "selection_rationale"),
        media_relative_path=media_path,
        media_sha256=media_sha,
        size_bytes=size,
        duration_seconds=duration,
        has_audio=_bool(raw.get("has_audio")),
        has_video=_bool(raw.get("has_video")),
        condition_ids=tuple(condition_ids),
        qwen_condition_id=_text(raw.get("qwen_condition_id"), "qwen_condition_id")
        if phase == "screen" else None,
        qwen_status=_text(raw.get("qwen_status"), "qwen_status")
        if phase == "screen" else None,
    )


def _reject_labels(value: object, location: str = "$") -> None:
    forbidden = {
        "annotation",
        "annotations",
        "answer",
        "answers",
        "correctanswer",
        "expectedanswer",
        "gold",
        "goldanswer",
        "groundtruth",
        "label",
        "labels",
        "oracle",
        "reference",
        "references",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized in forbidden:
                raise ValueError(f"forbidden field at {location}.{key}")
            _reject_labels(nested, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_labels(nested, f"{location}[{index}]")


def _coerce_prepared(prepared: PreparedPhase0Run | Path) -> PreparedPhase0Run:
    return load_prepared(prepared) if isinstance(prepared, (str, Path)) else prepared


def _refuse_existing_run(path: Path) -> None:
    if path.is_symlink():
        raise FileExistsError("refusing symlinked run output")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output already contains run artifacts: {path}")


def _read_json(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON artifact is missing: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("JSON artifact must contain an object")
    return payload


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        return []
    if path.is_symlink():
        raise ValueError("JSONL artifact must not be a symlink")
    records: list[Mapping[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError("JSONL records must be objects")
        records.append(payload)
    return records


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ValueError("refusing symlinked JSONL artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _create_empty_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"artifact already exists: {path.name}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)


def _write_json_once(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"artifact already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"artifact already exists: {path.name}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_replace(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_bytes_once(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"artifact already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"artifact already exists: {path.name}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_terminal_status(prepared: PreparedPhase0Run) -> None:
    _write_json_replace(prepared.path / "terminal-status.json", status(prepared))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("value must be an integer")
    return value


def _bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("value must be a boolean")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("value must be finite and non-negative")
    return result


__all__ = [
    "ABSOLUTE_CAP_USD",
    "ANSWER_POLICY",
    "AmbiguousStartedAttemptError",
    "CostGateError",
    "FrontierTransport",
    "MAP_POLICY",
    "PilotGateError",
    "Phase0RunSummary",
    "Phase0RunnerError",
    "PreparedPhase0Run",
    "RequestBindingError",
    "RequestClaimDisposition",
    "authorize_remaining",
    "authorize_screen",
    "freeze_pilot",
    "freeze_screen",
    "load_prepared",
    "preflight",
    "prepare_run",
    "project_remaining_cost",
    "run_pilot",
    "run_screen_remaining",
    "run_screen_row",
    "status",
]
