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
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    CONDITIONS,
    PHASE0_RUN_ID,
    PanelRow,
    Phase0Panel,
)
from trec_video_2026.vqa_frontier_response_store import (
    RawResponseConflictError,
    parse_prompt_json_response,
    preserve_raw_response,
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
from trec_video_2026.vqa_submission_openrouter import OpenRouterRequestPolicy


SCHEMA_VERSION = 1
RUNNER_ID = "frontier-phase0-runner-v1"
PRICE_SNAPSHOT_DATE = "2026-08-12"
ABSOLUTE_CAP_USD = 25.0
OPENROUTER_PROVIDER = "openrouter"
DELIVERY_PROVIDER = "gemini"
MAP_POLICY = OpenRouterRequestPolicy(
    max_output_tokens=8192,
    temperature=0.0,
    allow_fallbacks=False,
    use_structured_outputs=False,
)
ANSWER_POLICY = OpenRouterRequestPolicy(
    max_output_tokens=4096,
    temperature=0.0,
    allow_fallbacks=False,
    use_structured_outputs=False,
)
_MODEL_SLUG = "google/gemini-3.1-pro-preview"
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TERMINAL = frozenset({"succeeded", "failed"})
_RETRYABLE = frozenset({"transport", "rate_limit", "empty", "truncated"})
_CONDITIONS = tuple(CONDITIONS)
_PHASES = ("pilot", "screen")
_DEFAULT_INPUT_TOKENS = 1_000_000
_MAP_OUTPUT_TOKENS = 8192
_ANSWER_OUTPUT_TOKENS = 4096


class Phase0RunnerError(RuntimeError):
    """Base class for safe, operator-facing runner failures."""


class PilotGateError(Phase0RunnerError):
    """Raised when screen dispatch lacks the frozen pilot authorization."""


class CostGateError(Phase0RunnerError):
    """Raised before a provider call would cross the hard cost cap."""


class AmbiguousStartedAttemptError(Phase0RunnerError):
    """Raised when an earlier attempt has no terminal record."""


class RequestBindingError(Phase0RunnerError):
    """Raised when an existing artifact no longer matches its request."""


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
    observed_cost_usd: float = 0.0

    @property
    def estimated_cost_usd(self) -> float:
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
    pilot_freeze = prepared.path / "pilot-freeze.json"
    if not pilot_freeze.is_file():
        raise PilotGateError("pilot audit authorization requires frozen pilot results")
    _verify_freeze_file(pilot_freeze)
    pilot_results = _require_complete_phase(prepared, "pilot")
    if pilot_results[0].get("status") != "succeeded":
        raise PilotGateError("pilot audit authorization requires a successful pilot map")
    audit = _read_json(Path(pilot_audit_freeze))
    _verify_audit_freeze(prepared, Path(pilot_audit_freeze), audit)
    decision = _audit_decision(audit)
    if decision != "usable_for_screen":
        raise PilotGateError("pilot audit authorization is not usable_for_screen")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "authorized",
        "run_id": prepared.run_id,
        "panel_sha256": prepared.panel_sha256,
        "pilot_freeze_sha256": _sha256_file(pilot_freeze),
        "pilot_audit_freeze_sha256": _sha256_file(Path(pilot_audit_freeze)),
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
) -> Mapping[str, object]:
    """Project the complete 22-call run from observed per-condition means."""

    prepared = _coerce_prepared(prepared)
    conditions: dict[str, dict[str, object]] = {}
    observed_total = 0.0
    projected_remaining = 0.0
    for condition_id in _CONDITIONS:
        results = _bound_results(prepared, condition_id)
        unrepresented_calls, unrepresented_cost = _unrepresented_terminal_attempts(
            prepared, condition_id
        )
        observed_cost = sum(
            _number(result.get("estimated_cost_usd", 0.0)) for result in results
        ) + unrepresented_cost
        observed_calls = len(results) + unrepresented_calls
        expected_calls = 1 + len(prepared.screen_rows) if condition_id == "neutral_map" else len(prepared.screen_rows)
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
    _verify_pilot_authorization(prepared)
    first_row = _first_row_results(prepared)
    if set(first_row) != set(_CONDITIONS):
        raise CostGateError("first screen row is not complete")
    projection = project_remaining_cost(prepared)
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
        "projected_total_usd": projection["projected_total_usd"],
        "absolute_cap_usd": ABSOLUTE_CAP_USD,
        "projected_total_usd_lt_cap": True,
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
    results = _require_complete_phase(prepared, "pilot")
    return _freeze_phase(prepared, "pilot", results)


def freeze_screen(prepared: PreparedPhase0Run | Path) -> Mapping[str, object]:
    """Freeze all 21 screen results after the cost-gated run completes."""

    prepared = _coerce_prepared(prepared)
    if not (prepared.path / "cost-gate.json").is_file():
        raise CostGateError("screen freeze requires an authorized cost gate")
    _verify_cost_gate(prepared, _read_json(prepared.path / "cost-gate.json"))
    results = _require_complete_phase(prepared, "screen")
    return _freeze_phase(prepared, "screen", results)


def status(prepared: PreparedPhase0Run | Path) -> Mapping[str, object]:
    """Return only safe counts, state names, hashes, and costs."""

    prepared = _coerce_prepared(prepared)
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
    return {
        "schema_version": SCHEMA_VERSION,
        "status": _state_name(prepared, counts, ambiguous),
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
        "observed_cost_usd": round(_cumulative_cost(prepared), 6),
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
        skipped["estimated_cost_usd"] = 0.0
        return skipped
    _ensure_request_record(prepared, request)
    next_cost = _transport_estimate(transport, condition_id, row)
    if _cumulative_cost(prepared) + next_cost >= ABSOLUTE_CAP_USD:
        raise CostGateError("next provider call would reach USD 25")
    total_cost = 0.0
    total_input = 0
    total_output = 0
    raw_hashes: list[str] = []
    raw_paths: list[str] = []
    last_failure: str | None = None
    for attempt_index in (1, 2):
        if attempt_index == 2 and (
            _cumulative_cost(prepared) + next_cost >= ABSOLUTE_CAP_USD
        ):
            result = _result_record(
                request,
                status="failed",
                output=None,
                estimated_cost_usd=total_cost,
                input_tokens=total_input,
                output_tokens=total_output,
                raw_hashes=raw_hashes,
                raw_paths=raw_paths,
                failure_code="cost_gate",
            )
            _append_result(prepared, condition_id, result)
            return result
        _append_jsonl(
            prepared.path / "attempts.jsonl",
            _attempt_record(request, status="started", attempt_index=attempt_index),
        )
        uploaded: UploadedFileRecord | None = None
        generated: GenerationResult | None = None
        attempt_cost = 0.0
        attempt_input = 0
        attempt_output = 0
        try:
            uploaded = transport.upload(delivery.path, mime_type=delivery.mime_type)
            if uploaded.sha256 is not None and uploaded.sha256 != delivery.sha256:
                raise DeliveryMediaError("uploaded delivery identity changed")
            generated = transport.generate(
                model=MODEL_ID,
                parts=(
                    {"file_id": uploaded.file_id, "mime_type": uploaded.mime_type},
                    prompt,
                ),
                schema=provider_schema,
            )
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
            attempt_input = generated.usage.input_tokens
            attempt_output = generated.usage.output_tokens
            attempt_cost = _generation_cost(generated)
            total_cost += attempt_cost
            total_input += attempt_input
            total_output += attempt_output
            _append_jsonl(
                prepared.path / "attempts.jsonl",
                _attempt_record(
                    request,
                    status="succeeded",
                    attempt_index=attempt_index,
                    input_tokens=attempt_input,
                    output_tokens=attempt_output,
                    estimated_cost_usd=attempt_cost,
                ),
            )
            result = _result_record(
                request,
                status="succeeded",
                output=output,
                estimated_cost_usd=total_cost,
                input_tokens=total_input,
                output_tokens=total_output,
                raw_hashes=raw_hashes,
                raw_paths=raw_paths,
            )
            _append_result(prepared, condition_id, result)
            return result
        except Exception as error:
            if isinstance(error, (AmbiguousStartedAttemptError, RequestBindingError)):
                raise
            if isinstance(error, ProviderResponseEvidenceError):
                generated = error.generation_result
                try:
                    raw_path = _preserve_generation(
                        prepared,
                        condition_id=condition_id,
                        video_id=row.video_id,
                        generated=generated,
                    )
                    raw_hashes.append(generated.response_sha256)
                    raw_paths.append(raw_path)
                except RawResponseConflictError:
                    raise AmbiguousStartedAttemptError(
                        "raw response conflict requires operator review"
                    ) from error
            if generated is not None:
                attempt_input = generated.usage.input_tokens
                attempt_output = generated.usage.output_tokens
                attempt_cost = _generation_cost(generated)
            elif isinstance(error, DeliveryMediaError):
                attempt_cost = 0.0
            else:
                attempt_cost = next_cost
            total_cost += attempt_cost
            total_input += attempt_input
            total_output += attempt_output
            failure_code = _failure_code(error, generated)
            last_failure = failure_code
            _append_jsonl(
                prepared.path / "attempts.jsonl",
                _attempt_record(
                    request,
                    status="failed",
                    attempt_index=attempt_index,
                    failure_code=failure_code,
                    input_tokens=attempt_input,
                    output_tokens=attempt_output,
                    estimated_cost_usd=attempt_cost,
                ),
            )
            if attempt_index == 1 and failure_code in _RETRYABLE:
                continue
            result = _result_record(
                request,
                status="failed",
                output=None,
                estimated_cost_usd=total_cost,
                input_tokens=total_input,
                output_tokens=total_output,
                raw_hashes=raw_hashes,
                raw_paths=raw_paths,
                failure_code=last_failure,
            )
            _append_result(prepared, condition_id, result)
            return result
        finally:
            if uploaded is not None:
                try:
                    transport.delete(uploaded.file_id)
                except Exception:
                    # Cleanup failure is terminally observable in the attempt
                    # record only when the provider itself returned a result.
                    pass
    raise RuntimeError("unreachable retry loop")


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
    identity: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared.run_id,
        "phase": phase,
        "panel_sha256": prepared.panel_sha256,
        "source_sha256": row.media_sha256,
        "source_size_bytes": row.size_bytes,
        "video_id": row.video_id,
        "model_id": MODEL_ID,
        "provider": OPENROUTER_PROVIDER,
        "condition_id": condition_id,
        "prompt_sha256": canonical_sha256(prompt),
        "schema_sha256": canonical_sha256(schema),
        "provider_schema_sha256": canonical_sha256(provider_schema),
        "response_format_policy": "prompt-json-local-validation-v3-exact-fence",
        "max_output_tokens": policy.max_output_tokens,
        "temperature": policy.temperature,
        "allow_fallbacks": policy.allow_fallbacks,
        "use_structured_outputs": policy.use_structured_outputs,
        "delivery_provider": DELIVERY_PROVIDER,
        **delivery.request_identity(),
        "neutral_map_sha256": None
        if neutral_map is None
        else neutral_map.sha256,
    }
    identity["request_sha256"] = canonical_sha256(identity)
    return identity


def _attempt_record(
    request: Mapping[str, object],
    *,
    status: str,
    attempt_index: int,
    failure_code: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
) -> dict[str, object]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": request["run_id"],
        "phase": request["phase"],
        "video_id": request["video_id"],
        "condition_id": request["condition_id"],
        "request_sha256": request["request_sha256"],
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
        "estimated_cost_usd": round(estimated_cost_usd, 6),
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
        "estimated_cost_usd": round(estimated_cost_usd, 6),
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
    if matches and not _result_hash_valid(matches[0]):
        raise RequestBindingError("terminal result identity changed")
    return matches[0] if matches else None


def _binding_matches(existing: Mapping[str, object], request: Mapping[str, object]) -> bool:
    fields = (
        "request_sha256",
        "run_id",
        "phase",
        "panel_sha256",
        "source_sha256",
        "source_size_bytes",
        "video_id",
        "model_id",
        "provider",
        "condition_id",
        "prompt_sha256",
        "schema_sha256",
        "provider_schema_sha256",
        "delivery_provider",
        "delivery_policy_version",
        "delivery_sha256",
        "delivery_size_bytes",
        "delivery_mime_type",
        "delivery_derived",
        "neutral_map_sha256",
    )
    return all(existing.get(field) == request.get(field) for field in fields)


def _ensure_request_record(
    prepared: PreparedPhase0Run,
    request: Mapping[str, object],
) -> None:
    records = _read_jsonl(prepared.path / "requests.jsonl")
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


def _fail_on_ambiguous_started_attempts(prepared: PreparedPhase0Run) -> None:
    ambiguous = _ambiguous_attempts(prepared, _read_jsonl(prepared.path / "attempts.jsonl"))
    if ambiguous:
        raise AmbiguousStartedAttemptError("started attempt lacks a terminal record")


def _ambiguous_attempts(
    prepared: PreparedPhase0Run,
    attempts: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
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
    return [
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
        observed_cost_usd=round(
            sum(_number(result.get("estimated_cost_usd", 0.0)) for result in outcomes),
            6,
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
        observed_cost_usd=round(
            sum(summary.observed_cost_usd for summary in summaries), 6
        ),
    )


def _generation_cost(generated: GenerationResult) -> float:
    reported = generated.provider_metadata.get("reported_cost_usd")
    if reported is not None:
        return _number(reported)
    return (
        generated.usage.input_tokens * 2.0
        + generated.usage.output_tokens * 12.0
    ) / 1_000_000


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
    message = str(error).casefold()
    if "rate" in message and "limit" in message:
        return "rate_limit"
    if "truncat" in message or "maximum output" in message or "length" in message and "response" in message:
        return "truncated"
    if generated is not None and not generated.raw_response:
        return "empty"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(error, ProviderResponseEvidenceError):
        code = getattr(error, "failure_code", "")
        return "truncated" if code in {"protocol_no_response", "protocol_failure"} and "truncat" in message else "provider_protocol"
    if isinstance(error, ProviderError):
        code = getattr(error, "failure_code", "")
        if code in {"provider_request", "provider_generation", "provider_failure"}:
            return "transport"
        if code == "protocol_no_response":
            return "empty"
        return "provider_failure"
    if isinstance(error, ProviderProtocolError):
        return "schema_validation"
    if isinstance(error, DeliveryMediaError):
        return "delivery_media"
    if isinstance(error, ValueError):
        return "schema_validation"
    return "transport"


def _validate_provider_model(generated: GenerationResult) -> None:
    observed = generated.provider_metadata.get("model")
    if observed is not None and observed not in {MODEL_ID, _MODEL_SLUG}:
        raise ProviderProtocolError("observed model mismatch")


def _require_complete_phase(
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
                found[key] = result
    if set(found) != expected:
        raise Phase0RunnerError(f"{phase} results are incomplete")
    if _ambiguous_attempts(prepared, _read_jsonl(prepared.path / "attempts.jsonl")):
        raise AmbiguousStartedAttemptError("started attempt lacks a terminal record")
    return [found[key] for key in sorted(found, key=lambda item: (str(item[0]), str(item[1]), str(item[2])))]


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
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "runner_id": RUNNER_ID,
        "run_id": prepared.run_id,
        "phase": phase,
        "panel_sha256": prepared.panel_sha256,
        "result_count": len(results),
        "results_sha256": canonical_sha256(list(results)),
        "result_hashes": result_hashes,
        "observed_cost_usd": round(
            sum(_number(result.get("estimated_cost_usd", 0.0)) for result in results),
            6,
        ),
        "results": list(results),
    }
    if artifact_path.exists() or artifact_path.is_symlink():
        existing = _read_json(artifact_path)
        if existing != artifact:
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
    for result in _read_jsonl(prepared.path / "results" / f"{condition_id}.jsonl"):
        if (
            result.get("run_id") == prepared.run_id
            and result.get("condition_id") == condition_id
            and result.get("status") in _TERMINAL
            and isinstance(result.get("result_sha256"), str)
        ):
            if not _result_hash_valid(result):
                raise RequestBindingError("terminal result identity changed")
            results.append(result)
    return results


def _result_hash_valid(result: Mapping[str, object]) -> bool:
    declared = result.get("result_sha256")
    if not isinstance(declared, str):
        return False
    unsigned = dict(result)
    del unsigned["result_sha256"]
    return canonical_sha256(unsigned) == declared


def _unrepresented_terminal_attempts(
    prepared: PreparedPhase0Run,
    condition_id: str,
) -> tuple[int, float]:
    result_hashes = {
        str(result.get("request_sha256"))
        for result in _bound_results(prepared, condition_id)
    }
    grouped: dict[str, float] = {}
    for attempt in _read_jsonl(prepared.path / "attempts.jsonl"):
        request_sha256 = attempt.get("request_sha256")
        if (
            attempt.get("run_id") == prepared.run_id
            and attempt.get("condition_id") == condition_id
            and attempt.get("status") in _TERMINAL
            and isinstance(request_sha256, str)
            and request_sha256 not in result_hashes
        ):
            grouped[request_sha256] = grouped.get(request_sha256, 0.0) + _number(
                attempt.get("estimated_cost_usd", 0.0)
            )
    return len(grouped), sum(grouped.values())


def _cumulative_cost(prepared: PreparedPhase0Run) -> float:
    result_cost = sum(
        _number(result.get("estimated_cost_usd", 0.0))
        for condition in _CONDITIONS
        for result in _bound_results(prepared, condition)
    )
    attempt_cost = sum(
        _unrepresented_terminal_attempts(prepared, condition)[1]
        for condition in _CONDITIONS
    )
    return result_cost + attempt_cost


def _verify_cost_gate(prepared: PreparedPhase0Run, gate: Mapping[str, object]) -> None:
    if (
        gate.get("status") != "authorized"
        or gate.get("run_id") != prepared.run_id
        or gate.get("panel_sha256") != prepared.panel_sha256
        or gate.get("projected_total_usd_lt_cap") is not True
    ):
        raise CostGateError("cost gate binding is not authorized")
    first = _first_row_results(prepared)
    hashes = gate.get("first_row_result_hashes")
    if not isinstance(hashes, Mapping) or any(hashes.get(condition) != first.get(condition, {}).get("result_sha256") for condition in _CONDITIONS):
        raise CostGateError("cost gate first-row result binding changed")


def _verify_pilot_authorization(prepared: PreparedPhase0Run) -> None:
    path = prepared.path / "pilot-authorization.json"
    payload = _read_json(path)
    if payload.get("status") != "authorized" or payload.get("run_id") != prepared.run_id or payload.get("panel_sha256") != prepared.panel_sha256 or payload.get("pilot_audit_decision") != "usable_for_screen":
        raise PilotGateError("pilot audit authorization binding changed")
    if payload.get("pilot_freeze_sha256") != _sha256_file(prepared.path / "pilot-freeze.json"):
        raise PilotGateError("pilot audit authorization is stale")


def _verify_freeze_file(path: Path) -> None:
    freeze = _read_json(path)
    artifact_name = "pilot-results.json" if path.name == "pilot-freeze.json" else "screen-results.json"
    verify_json_artifact(path.parent / artifact_name, freeze)


def _verify_audit_freeze(
    prepared: PreparedPhase0Run,
    path: Path,
    freeze: Mapping[str, object],
) -> None:
    if freeze.get("status") != "frozen":
        raise PilotGateError("pilot audit authorization requires a frozen audit")
    bindings = freeze.get("bindings")
    if isinstance(bindings, Mapping):
        if bindings.get("run_id") is not None and bindings.get("run_id") != prepared.run_id:
            raise PilotGateError("pilot audit freeze run binding is invalid")
        if bindings.get("panel_sha256") is not None and bindings.get("panel_sha256") != prepared.panel_sha256:
            raise PilotGateError("pilot audit freeze panel binding is invalid")
    artifact_sha = freeze.get("artifact_sha256")
    if isinstance(artifact_sha, str):
        artifact = path.parent / "audit.json"
        if artifact.is_file():
            verify_json_artifact(artifact, dict(freeze))
        else:
            raise PilotGateError("pilot audit artifact bound by freeze is missing")


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
) -> str:
    if ambiguous:
        return "blocked-ambiguous-attempt"
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
    raw_rows = payload.get("examples")
    if not isinstance(raw_rows, list):
        raise ValueError("panel must contain examples")
    rows = tuple(_parse_row(raw, index) for index, raw in enumerate(raw_rows))
    pilot = tuple(row for row in rows if row.phase == "pilot")
    screen = tuple(row for row in rows if row.phase == "screen")
    if len(pilot) != 1 or len(screen) != 7:
        raise ValueError("panel must contain one pilot and seven screen rows")
    if [row.ordinal for row in screen] != sorted(row.ordinal for row in screen):
        screen = tuple(sorted(screen, key=lambda row: row.ordinal))
    if len({row.video_id for row in rows}) != len(rows):
        raise ValueError("panel video IDs must be unique")
    if payload.get("conditions") not in (None, list(_CONDITIONS)):
        raise ValueError("panel conditions do not match")
    bindings = payload.get("bindings")
    binding = bindings if isinstance(bindings, Mapping) else {}
    run_id = payload.get("run_id", PHASE0_RUN_ID)
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("panel run_id must be non-empty")
    return Phase0Panel(
        schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        run_id=run_id,
        selection_policy=str(payload.get("selection_policy", "synthetic-offline-fixture")),
        reference_policy=str(payload.get("reference_policy", "not-read-not-serialized")),
        source_panel_sha256=str(binding.get("source_panel_sha256", canonical_sha256(payload))),
        qwen_predictions_sha256=str(binding.get("qwen_predictions_sha256", canonical_sha256([]))),
        selected_ordered_ids_sha256=str(binding.get("selected_ordered_ids_sha256", canonical_sha256([row.video_id for row in rows]))),
        source_media_sha256=str(binding.get("source_media_sha256", canonical_sha256([{"video_id": row.video_id, "source_sha256": row.media_sha256} for row in rows]))),
        rows=(pilot[0], *screen),
        source_panel_file_sha256=None,
        qwen_prediction_file_sha256=None,
    )


def _parse_row(raw: object, index: int) -> PanelRow:
    if not isinstance(raw, Mapping):
        raise ValueError(f"examples[{index}] must be an object")
    phase = raw.get("phase")
    if phase not in {"pilot", "screen"}:
        raise ValueError("panel row phase must be pilot or screen")
    condition_ids = raw.get("condition_ids")
    expected = ["neutral_map"] if phase == "pilot" else list(_CONDITIONS)
    if condition_ids != expected:
        raise ValueError("panel row condition_ids do not match the phase")
    video_id = _text(raw.get("video_id"), "video_id")
    if _SAFE_ID.fullmatch(video_id) is None:
        raise ValueError("video_id must be a safe path segment")
    media_path = _text(raw.get("media_relative_path"), "media_relative_path")
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
        category=_text(raw.get("category"), "category"),
        evidence_topology=_text(raw.get("evidence_topology"), "evidence_topology"),
        selection_rationale=_text(raw.get("selection_rationale"), "selection_rationale"),
        media_relative_path=media_path,
        media_sha256=media_sha,
        size_bytes=size,
        duration_seconds=duration,
        has_audio=bool(raw.get("has_audio")),
        has_video=bool(raw.get("has_video")),
        condition_ids=tuple(condition_ids),
        qwen_condition_id=raw.get("qwen_condition_id") if isinstance(raw.get("qwen_condition_id"), str) else None,
        qwen_status=raw.get("qwen_status") if isinstance(raw.get("qwen_status"), str) else None,
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
