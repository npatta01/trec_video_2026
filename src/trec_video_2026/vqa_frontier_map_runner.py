"""Immutable, resumable offline runner for the frontier VQA event-map panel."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from trec_video_2026.vqa_frontier_map import (
    EVENT_MAP_PROMPT_VERSION,
    EVENT_MAP_SCHEMA,
    build_event_map_prompt,
    validate_event_map,
)
from trec_video_2026.vqa_frontier_map_panel import (
    CATEGORIES,
    PanelRow,
    RUN_ID as PANEL_RUN_ID,
)
from trec_video_2026.vqa_frontier_map_reka import REKA_MODEL_ID
from trec_video_2026.vqa_submission import MODEL_ID, canonical_sha256
from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    ProviderAuthenticationError,
    ProviderError,
    ProviderProtocolError,
    TokenUsage,
    UploadedFileRecord,
)
from trec_video_2026.vqa_experiments.temporal_bottleneck import freeze_json_artifact


PRICE_SNAPSHOT = {
    "gemini_input_per_million": 2.0,
    "gemini_output_per_million": 12.0,
    "reka_input_per_million": 2.0,
    "reka_output_per_million": 6.0,
    "reka_video_per_minute": 0.08,
    "review_threshold_usd": 15.0,
    "absolute_cap_usd": 30.0,
}
PRICE_SNAPSHOT_DATE = "2026-08-11"
SCHEMA_VERSION = 1
RUNNER_ID = "frontier-map-runner-v1"
_PROVIDERS = ("gemini", "reka")
_PHASES = ("preflight", "panel")
_TERMINAL_ATTEMPT_STATUSES = frozenset({"succeeded", "failed"})
_FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "answer",
        "answers",
        "correctanswer",
        "expectedanswer",
        "gold",
        "goldanswer",
        "groundtruth",
        "groundtruthanswer",
        "incorrectanswer",
        "incorrectanswers",
        "label",
        "labels",
        "oracle",
        "reference",
        "references",
        "referenceanswer",
        "referenceanswers",
    }
)


class AmbiguousStartedAttemptError(RuntimeError):
    """Raised when a started attempt has no terminal operator classification."""


class CostGateError(RuntimeError):
    """Raised when preflight or cost gates forbid provider dispatch."""


class FrontierTransport(Protocol):
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
class PreparedFrontierRun:
    path: Path
    run_id: str
    panel_sha256: str
    prompt_sha256: str
    schema_sha256: str
    examples: tuple[PanelRow, ...]
    provider_configs: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class FrontierRunSummary:
    provider: str
    phase: str
    expected: int
    passed: int
    failed: int
    skipped: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class FixtureTransport:
    """Deterministic offline provider double for runner tests and dry runs."""

    def __init__(
        self,
        *,
        provider: str = "openrouter",
        route: str = "fixture-route",
        raw_response: str | None = None,
        input_tokens: int = 100,
        output_tokens: int = 50,
        reported_cost_usd: float | None = None,
    ) -> None:
        if provider not in {"openrouter", "reka", "fixture"}:
            raise ValueError("fixture provider must be openrouter, reka, or fixture")
        self.provider_name = provider
        self.route = route
        self.raw_response = raw_response
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.reported_cost_usd = reported_cost_usd
        self.call_count = 0
        self.requests: list[dict[str, object]] = []

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        return UploadedFileRecord(file_id=f"fixture-{path.name}", mime_type=mime_type)

    def delete(self, file_id: str) -> None:
        return None

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: Mapping[str, object],
    ) -> GenerationResult:
        self.call_count += 1
        self.requests.append(
            {
                "model": model,
                "part_count": len(tuple(parts)),
                "schema_sha256": canonical_sha256(schema),
            }
        )
        raw_response = self.raw_response
        if raw_response is None:
            raw_response = json.dumps(
                _fixture_event_map_payload(), sort_keys=True, separators=(",", ":")
            )
        try:
            payload: object = json.loads(raw_response)
        except json.JSONDecodeError:
            payload = raw_response
        metadata: dict[str, object] = {
            "provider": self.route if self.provider_name == "openrouter" else self.provider_name,
            "model": model,
            "request_id": f"fixture-{self.call_count}",
        }
        if self.reported_cost_usd is not None:
            metadata["reported_cost_usd"] = self.reported_cost_usd
        return GenerationResult(
            payload=payload,
            raw_response=raw_response,
            usage=TokenUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            ),
            provider_metadata=metadata,
        )

    def estimate_cost_usd(self, provider: str, row: PanelRow) -> float:
        if self.reported_cost_usd is not None:
            return float(self.reported_cost_usd)
        return _estimated_cost_usd(
            provider=provider,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            duration_seconds=row.duration_seconds,
            provider_metadata={},
        )


def prepare_fixture_run(
    path: Path,
    *,
    evaluation_rows: int,
    preflight_rows: int,
) -> PreparedFrontierRun:
    """Create a synthetic prepared run without touching real media or providers."""

    if evaluation_rows < 0 or preflight_rows < 0:
        raise ValueError("row counts cannot be negative")
    rows: list[PanelRow] = []
    ordinal = 1
    for index in range(evaluation_rows):
        rows.append(_fixture_row(ordinal, "evaluation", f"eval-{ordinal:03d}", index))
        ordinal += 1
    for index in range(preflight_rows):
        rows.append(
            _fixture_row(ordinal, "preflight", f"preflight-{ordinal:03d}", index)
        )
        ordinal += 1
    panel_payload = _panel_payload(tuple(rows), fixture=True)
    return _write_prepared_run(path, panel_payload, tuple(rows), fixture=True)


def prepare_run(panel: Path, output: Path) -> PreparedFrontierRun:
    """Prepare an immutable offline run from an already frozen panel JSON."""

    payload = json.loads(panel.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("panel must be a JSON object")
    _reject_label_fields(payload)
    raw_rows = payload.get("examples")
    if not isinstance(raw_rows, list):
        raise ValueError("panel must contain examples")
    rows = tuple(
        _row_from_mapping(raw, location=f"examples[{index}]")
        for index, raw in enumerate(raw_rows)
    )
    fixture = payload.get("selection_policy") == "synthetic-offline-fixture"
    return _write_prepared_run(output, dict(payload), rows, fixture=fixture)


def load_prepared(path: Path) -> PreparedFrontierRun:
    """Load a prepared run from disk."""

    manifest = _read_json(path / "run-manifest.json")
    panel = _read_json(path / "panel.json")
    raw_rows = panel.get("examples")
    if not isinstance(raw_rows, list):
        raise ValueError("prepared panel must contain examples")
    rows = tuple(
        _row_from_mapping(raw, location=f"examples[{index}]")
        for index, raw in enumerate(raw_rows)
    )
    configs = manifest.get("provider_configs")
    if not isinstance(configs, Mapping):
        raise ValueError("run manifest provider_configs are invalid")
    return PreparedFrontierRun(
        path=path,
        run_id=_required_text(manifest.get("run_id"), "run_id"),
        panel_sha256=_required_text(manifest.get("panel_sha256"), "panel_sha256"),
        prompt_sha256=_required_text(manifest.get("prompt_sha256"), "prompt_sha256"),
        schema_sha256=_required_text(manifest.get("schema_sha256"), "schema_sha256"),
        examples=rows,
        provider_configs=_freeze_mapping(configs),
    )


def run_phase(
    prepared: PreparedFrontierRun | Path,
    *,
    phase: str,
    provider: str,
    transport: FrontierTransport,
) -> FrontierRunSummary:
    """Run one provider/phase, resuming only fully matching completed rows."""

    prepared = _coerce_prepared(prepared)
    _validate_phase_provider(phase, provider)
    _validate_transport_provider(provider, transport)
    rows = _rows_for_phase(prepared, phase)
    if phase == "panel":
        _enforce_panel_gates(prepared)

    passed = failed = skipped = input_tokens = output_tokens = 0
    estimated_cost = 0.0
    for row in rows:
        _fail_on_ambiguous_started_attempts(prepared, provider=provider, phase=phase)
        route = _expected_route(prepared, provider=provider, phase=phase, transport=transport)
        request = _request_record(
            prepared,
            row=row,
            phase=phase,
            provider=provider,
            route=route,
        )
        completed = _matching_completed_result(prepared, provider, request)
        if completed is not None:
            skipped += 1
            continue
        next_cost = _transport_estimate(transport, provider, row)
        if _cumulative_cost(prepared) + next_cost > PRICE_SNAPSHOT["absolute_cap_usd"]:
            raise CostGateError("absolute cumulative cost cap would be exceeded")
        _ensure_request_record(prepared, request)
        attempt_index = _next_attempt_index(prepared, request)
        started_attempt = _attempt_record(
            request,
            status="started",
            attempt_index=attempt_index,
        )
        _append_jsonl(prepared.path / "attempts.jsonl", started_attempt)
        try:
            generated = transport.generate(
                model=str(request["model_id"]),
                parts=_provider_parts(row),
                schema=EVENT_MAP_SCHEMA,
            )
            raw_payload = _parse_raw_response(generated.raw_response)
            validate_event_map(raw_payload, row.duration_seconds)
        except json.JSONDecodeError:
            failed += 1
            _append_jsonl(
                prepared.path / "attempts.jsonl",
                _attempt_record(
                    request,
                    status="failed",
                    attempt_index=attempt_index,
                    failure_code="invalid_json",
                ),
            )
            continue
        except (ProviderAuthenticationError, ProviderError, ProviderProtocolError, ValueError):
            failed += 1
            _append_jsonl(
                prepared.path / "attempts.jsonl",
                _attempt_record(
                    request,
                    status="failed",
                    attempt_index=attempt_index,
                    failure_code="provider_or_schema_failure",
                ),
            )
            continue
        usage = generated.usage
        cost = _estimated_cost_usd(
            provider=provider,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            duration_seconds=row.duration_seconds,
            provider_metadata=generated.provider_metadata,
        )
        result = _result_record(
            request,
            generated=generated,
            estimated_cost_usd=cost,
        )
        _write_raw_response(prepared, provider, row.video_id, generated.raw_response)
        _append_jsonl(prepared.path / "results" / f"{provider}.jsonl", result)
        _append_jsonl(
            prepared.path / "attempts.jsonl",
            _attempt_record(
                request,
                status="succeeded",
                attempt_index=attempt_index,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                estimated_cost_usd=cost,
            ),
        )
        passed += 1
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        estimated_cost += cost

    _write_terminal_status(prepared)
    return FrontierRunSummary(
        provider=provider,
        phase=phase,
        expected=len(rows),
        passed=passed,
        failed=failed,
        skipped=skipped,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=round(estimated_cost, 6),
    )


def project_cost(prepared: PreparedFrontierRun | Path) -> dict[str, object]:
    """Project panel cost from observed preflight provider usage."""

    prepared = _coerce_prepared(prepared)
    evaluation_rows = _rows_for_phase(prepared, "panel")
    preflight_rows = _rows_for_phase(prepared, "preflight")
    provider_summaries: dict[str, dict[str, object]] = {}
    observed_total = 0.0
    projected_panel_total = 0.0
    for provider in _PROVIDERS:
        preflight_results = [
            result
            for result in _provider_results(prepared, provider)
            if result.get("phase") == "preflight"
            and _result_still_bound_to_prepared(prepared, result)
        ]
        observed = sum(_number(result.get("estimated_cost_usd")) for result in preflight_results)
        observed_total += observed
        observed_duration = sum(
            _number(result.get("duration_seconds")) for result in preflight_results
        )
        panel_duration = sum(row.duration_seconds for row in evaluation_rows)
        if not preflight_results:
            panel_projection = 0.0
        else:
            row_scale = len(evaluation_rows) / len(preflight_results)
            duration_scale = (
                panel_duration / observed_duration if observed_duration > 0 else row_scale
            )
            panel_projection = observed * max(row_scale, duration_scale)
        projected_panel_total += panel_projection
        provider_summaries[provider] = {
            "observed_preflight_usd": round(observed, 6),
            "preflight_results": len(preflight_results),
            "preflight_rows": len(preflight_rows),
            "panel_rows": len(evaluation_rows),
            "panel_projected_usd": round(panel_projection, 6),
        }
    observed_projected = observed_total + projected_panel_total
    return {
        "schema_version": SCHEMA_VERSION,
        "price_snapshot_date": PRICE_SNAPSHOT_DATE,
        "providers": provider_summaries,
        "low_projected_usd": round(observed_projected * 0.75, 6),
        "observed_projected_usd": round(observed_projected, 6),
        "high_projected_usd": round(observed_projected * 1.25, 6),
        "authorized_to_continue": observed_projected
        <= PRICE_SNAPSHOT["review_threshold_usd"],
        "review_threshold_usd": PRICE_SNAPSHOT["review_threshold_usd"],
        "absolute_cap_usd": PRICE_SNAPSHOT["absolute_cap_usd"],
    }


def status(prepared: PreparedFrontierRun | Path) -> dict[str, object]:
    """Return a compact safe status summary for a prepared run."""

    prepared = _coerce_prepared(prepared)
    complete_results: dict[str, dict[str, int]] = {}
    failed_attempts = 0
    for provider in _PROVIDERS:
        complete_results[provider] = {}
        results = [
            result
            for result in _provider_results(prepared, provider)
            if _result_still_bound_to_prepared(prepared, result)
        ]
        for phase in _PHASES:
            complete_results[provider][phase] = sum(
                1 for result in results if result.get("phase") == phase
            )
    for attempt in _attempt_records(prepared):
        if attempt.get("status") == "failed":
            failed_attempts += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared.run_id,
        "complete_results": complete_results,
        "failed_attempts": failed_attempts,
        "ambiguous_started_attempts": len(_ambiguous_started_attempts(prepared)),
        "estimated_cost_usd": round(_cumulative_cost(prepared), 6),
        "frozen": (prepared.path / "freeze.json").is_file(),
    }


def freeze_provider_outputs(prepared: PreparedFrontierRun | Path) -> dict[str, object]:
    """Freeze complete panel provider results and write freeze.json."""

    prepared = _coerce_prepared(prepared)
    _require_complete_panel_outputs(prepared)
    panel_results = {
        provider: [
            result
            for result in _provider_results(prepared, provider)
            if result.get("phase") == "panel"
            and _result_still_bound_to_prepared(prepared, result)
        ]
        for provider in _PROVIDERS
    }
    provider_result_count = sum(len(results) for results in panel_results.values())
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared.run_id,
        "panel_sha256": prepared.panel_sha256,
        "prompt_sha256": prepared.prompt_sha256,
        "schema_sha256": prepared.schema_sha256,
        "provider_result_count": provider_result_count,
        "results_sha256": canonical_sha256(panel_results),
        "providers": {
            provider: {
                "panel_results": len(results),
                "results_sha256": canonical_sha256(results),
            }
            for provider, results in panel_results.items()
        },
    }
    artifact_path = prepared.path / "provider-results.json"
    _write_json_atomically(artifact_path, artifact)
    freeze = freeze_json_artifact(
        artifact_path,
        bindings={
            "panel_sha256": prepared.panel_sha256,
            "prompt_sha256": prepared.prompt_sha256,
            "schema_sha256": prepared.schema_sha256,
            "provider_result_count": provider_result_count,
        },
    )
    _write_json_atomically(prepared.path / "freeze.json", freeze)
    _write_terminal_status(prepared)
    return freeze


def _write_prepared_run(
    path: Path,
    panel_payload: Mapping[str, object],
    rows: tuple[PanelRow, ...],
    *,
    fixture: bool,
) -> PreparedFrontierRun:
    path.mkdir(parents=True, exist_ok=True)
    for directory in ("results", "raw/gemini", "raw/reka"):
        (path / directory).mkdir(parents=True, exist_ok=True)
    panel_sha256 = canonical_sha256(panel_payload)
    prompt_sha256 = canonical_sha256(
        {
            "prompt_version": EVENT_MAP_PROMPT_VERSION,
            "prompt_template_sha256": canonical_sha256(
                build_event_map_prompt("{question}", 30.0)
            ),
        }
    )
    schema_sha256 = canonical_sha256(EVENT_MAP_SCHEMA)
    provider_configs = {
        "gemini": {
            "transport": "openrouter",
            "model_id": MODEL_ID,
            "route_policy": "pin-successful-preflight-route",
        },
        "reka": {
            "transport": "reka",
            "model_id": REKA_MODEL_ID,
            "route_policy": "direct",
        },
    }
    prepared = PreparedFrontierRun(
        path=path,
        run_id=str(panel_payload.get("run_id", PANEL_RUN_ID)),
        panel_sha256=panel_sha256,
        prompt_sha256=prompt_sha256,
        schema_sha256=schema_sha256,
        examples=rows,
        provider_configs=_freeze_mapping(provider_configs),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "runner_id": RUNNER_ID,
        "run_id": prepared.run_id,
        "panel_sha256": panel_sha256,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "fixture_manifest": fixture,
        "price_snapshot_date": PRICE_SNAPSHOT_DATE,
        "provider_configs": provider_configs,
        "example_count": len(rows),
        "examples": [_safe_manifest_row(row) for row in rows],
    }
    _write_json_atomically(path / "panel.json", panel_payload)
    _write_json_atomically(path / "run-manifest.json", manifest)
    _write_terminal_status(prepared)
    return prepared


def _fixture_row(ordinal: int, phase: str, video_id: str, index: int) -> PanelRow:
    category = CATEGORIES[index % len(CATEGORIES)]
    return PanelRow(
        ordinal=ordinal,
        phase=phase,
        video_id=video_id,
        question=f"What happens after the fixture event {ordinal}?",
        category=category,
        evidence_topology="single",
        selection_rationale="synthetic offline fixture",
        media_relative_path=f"synthetic/{video_id}.mp4",
        media_sha256=f"{ordinal:064x}"[-64:],
        size_bytes=1024 + ordinal,
        duration_seconds=30.0,
        has_audio=True,
        has_video=True,
    )


def _panel_payload(rows: tuple[PanelRow, ...], *, fixture: bool) -> dict[str, object]:
    evaluation = [row for row in rows if row.phase == "evaluation"]
    preflight = [row for row in rows if row.phase == "preflight"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{PANEL_RUN_ID}-fixture" if fixture else PANEL_RUN_ID,
        "selection_policy": "synthetic-offline-fixture" if fixture else "frozen-panel",
        "reference_policy": "not-read-not-serialized",
        "evaluation_count": len(evaluation),
        "preflight_count": len(preflight),
        "examples": [asdict(row) for row in rows],
    }


def _fixture_event_map_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "question_interpretation": {
            "meaning": "Identify the fixture event state change.",
            "required_modalities": ["visual"],
            "required_evidence": "The fixture event before and after the change.",
        },
        "events": [
            {
                "event_id": "E1",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "keyframe_seconds": [0.5],
                "scene": "A synthetic fixture scene.",
                "entities": ["fixture object"],
                "actions": ["fixture action occurs"],
                "relations": ["object in scene"],
                "state_before": "fixture before state",
                "state_change": "fixture state changes",
                "state_after": "fixture after state",
                "visual_observations": ["the object changes state"],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "question_relevance": "direct",
                "confidence": 0.9,
                "gaps": [],
            }
        ],
        "support_spans": [
            {
                "event_ids": ["E1"],
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "modalities": ["visual"],
                "claim": "The fixture event supports the answer.",
            }
        ],
        "top1_answer": "The fixture state changed.",
        "support_explanation": "E1 shows the fixture state transition.",
        "could_be_wrong_because": ["Synthetic fixtures are minimal."],
    }


def _safe_manifest_row(row: PanelRow) -> dict[str, object]:
    return {
        "ordinal": row.ordinal,
        "phase": row.phase,
        "video_id": row.video_id,
        "category": row.category,
        "evidence_topology": row.evidence_topology,
        "source_sha256": row.media_sha256,
        "size_bytes": row.size_bytes,
        "duration_seconds": row.duration_seconds,
        "has_audio": row.has_audio,
        "has_video": row.has_video,
    }


def _request_record(
    prepared: PreparedFrontierRun,
    *,
    row: PanelRow,
    phase: str,
    provider: str,
    route: str,
) -> dict[str, object]:
    config = prepared.provider_configs[provider]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared.run_id,
        "runner_id": RUNNER_ID,
        "phase": phase,
        "provider": provider,
        "ordinal": row.ordinal,
        "video_id": row.video_id,
        "category": row.category,
        "evidence_topology": row.evidence_topology,
        "duration_seconds": row.duration_seconds,
        "has_audio": row.has_audio,
        "has_video": row.has_video,
        "source_sha256": row.media_sha256,
        "source_size_bytes": row.size_bytes,
        "panel_sha256": prepared.panel_sha256,
        "prompt_sha256": prepared.prompt_sha256,
        "schema_sha256": prepared.schema_sha256,
        "model_id": config["model_id"],
        "route": route,
        "price_snapshot_date": PRICE_SNAPSHOT_DATE,
    }
    return {**identity, "request_sha256": canonical_sha256(identity)}


def _result_record(
    request: Mapping[str, object],
    *,
    generated: GenerationResult,
    estimated_cost_usd: float,
) -> dict[str, object]:
    usage = generated.usage
    provider_metadata = dict(generated.provider_metadata)
    return {
        **copy.deepcopy(dict(request)),
        "status": "succeeded",
        "response_sha256": generated.response_sha256,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "provider_request_id": provider_metadata.get("request_id"),
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
) -> dict[str, object]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": request["run_id"],
        "phase": request["phase"],
        "provider": request["provider"],
        "video_id": request["video_id"],
        "request_sha256": request["request_sha256"],
        "source_sha256": request["source_sha256"],
        "model_id": request["model_id"],
        "route": request["route"],
        "prompt_sha256": request["prompt_sha256"],
        "schema_sha256": request["schema_sha256"],
        "status": status,
        "attempt_index": attempt_index,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 6),
    }
    if failure_code is not None:
        record["failure_code"] = failure_code
    return record


def _provider_parts(row: PanelRow) -> tuple[object, str]:
    media_identity = {
        "file_id": f"sha256-{row.media_sha256}",
        "mime_type": "video/mp4",
    }
    return (media_identity, build_event_map_prompt(row.question, row.duration_seconds))


def _parse_raw_response(raw_response: bytes | str) -> object:
    if isinstance(raw_response, bytes):
        raw_response = raw_response.decode("utf-8")
    return json.loads(raw_response)


def _write_raw_response(
    prepared: PreparedFrontierRun,
    provider: str,
    video_id: str,
    raw_response: bytes | str,
) -> None:
    if isinstance(raw_response, bytes):
        raw_text = raw_response.decode("utf-8", errors="replace")
    else:
        raw_text = raw_response
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared.run_id,
        "provider": provider,
        "video_id": video_id,
        "raw_response_sha256": canonical_sha256(raw_text),
        "raw_response": raw_text,
    }
    _write_json_atomically(prepared.path / "raw" / provider / f"{video_id}.json", payload)


def _estimated_cost_usd(
    *,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    duration_seconds: float,
    provider_metadata: Mapping[str, object],
) -> float:
    reported = provider_metadata.get("reported_cost_usd")
    if reported is not None:
        return _number(reported)
    if provider == "gemini":
        return (
            input_tokens * PRICE_SNAPSHOT["gemini_input_per_million"]
            + output_tokens * PRICE_SNAPSHOT["gemini_output_per_million"]
        ) / 1_000_000
    if provider == "reka":
        token_cost = (
            input_tokens * PRICE_SNAPSHOT["reka_input_per_million"]
            + output_tokens * PRICE_SNAPSHOT["reka_output_per_million"]
        ) / 1_000_000
        return token_cost + (duration_seconds / 60.0) * PRICE_SNAPSHOT["reka_video_per_minute"]
    raise ValueError("unknown provider")


def _transport_estimate(
    transport: FrontierTransport,
    provider: str,
    row: PanelRow,
) -> float:
    estimator = getattr(transport, "estimate_cost_usd", None)
    if callable(estimator):
        value = estimator(provider, row)
        return _number(value)
    return 0.0


def _expected_route(
    prepared: PreparedFrontierRun,
    *,
    provider: str,
    phase: str,
    transport: FrontierTransport,
) -> str:
    if provider == "reka":
        return "reka-direct"
    route = getattr(transport, "route", None)
    if isinstance(route, str) and route.strip():
        return route.strip()
    if phase == "panel":
        pinned = _pinned_gemini_route(prepared)
        if pinned is not None:
            return pinned
    return "openrouter-unpinned"


def _pinned_gemini_route(prepared: PreparedFrontierRun) -> str | None:
    routes = {
        str(result["route"])
        for result in _provider_results(prepared, "gemini")
        if result.get("phase") == "preflight"
        and isinstance(result.get("route"), str)
        and _result_still_bound_to_prepared(prepared, result)
    }
    if not routes:
        return None
    if len(routes) > 1:
        raise AmbiguousStartedAttemptError("ambiguous gemini preflight route pin")
    return next(iter(routes))


def _matching_completed_result(
    prepared: PreparedFrontierRun,
    provider: str,
    request: Mapping[str, object],
) -> Mapping[str, object] | None:
    for result in _provider_results(prepared, provider):
        if _matches_request_binding(prepared, result, request):
            return result
    return None


def _matches_request_binding(
    prepared: PreparedFrontierRun,
    result: Mapping[str, object],
    request: Mapping[str, object],
) -> bool:
    required = (
        "request_sha256",
        "source_sha256",
        "model_id",
        "route",
        "prompt_sha256",
        "schema_sha256",
        "panel_sha256",
        "provider",
        "phase",
        "video_id",
    )
    return all(result.get(field) == request.get(field) for field in required) and (
        result.get("status") == "succeeded"
        and result.get("run_id") == prepared.run_id
    )


def _result_still_bound_to_prepared(
    prepared: PreparedFrontierRun, result: Mapping[str, object]
) -> bool:
    return (
        result.get("run_id") == prepared.run_id
        and result.get("panel_sha256") == prepared.panel_sha256
        and result.get("prompt_sha256") == prepared.prompt_sha256
        and result.get("schema_sha256") == prepared.schema_sha256
        and result.get("status") == "succeeded"
    )


def _ensure_request_record(
    prepared: PreparedFrontierRun, request: Mapping[str, object]
) -> None:
    if any(
        existing.get("request_sha256") == request["request_sha256"]
        for existing in _request_records(prepared)
    ):
        return
    _append_jsonl(prepared.path / "requests.jsonl", request)


def _next_attempt_index(
    prepared: PreparedFrontierRun, request: Mapping[str, object]
) -> int:
    indexes = [
        int(attempt["attempt_index"])
        for attempt in _attempt_records(prepared)
        if attempt.get("request_sha256") == request["request_sha256"]
        and isinstance(attempt.get("attempt_index"), int)
    ]
    return max(indexes, default=0) + 1


def _fail_on_ambiguous_started_attempts(
    prepared: PreparedFrontierRun,
    *,
    provider: str,
    phase: str,
) -> None:
    ambiguous = [
        attempt
        for attempt in _ambiguous_started_attempts(prepared)
        if attempt.get("provider") == provider and attempt.get("phase") == phase
    ]
    if ambiguous:
        raise AmbiguousStartedAttemptError(
            "lone started attempt requires explicit operator classification"
        )


def _ambiguous_started_attempts(
    prepared: PreparedFrontierRun,
) -> list[Mapping[str, object]]:
    attempts = _attempt_records(prepared)
    terminal_keys = {
        _attempt_key(attempt)
        for attempt in attempts
        if attempt.get("status") in _TERMINAL_ATTEMPT_STATUSES
    }
    return [
        attempt
        for attempt in attempts
        if attempt.get("run_id") == prepared.run_id
        and attempt.get("status") == "started"
        and _attempt_key(attempt) not in terminal_keys
    ]


def _attempt_key(attempt: Mapping[str, object]) -> tuple[object, ...]:
    return (
        attempt.get("provider"),
        attempt.get("phase"),
        attempt.get("video_id"),
        attempt.get("request_sha256"),
        attempt.get("attempt_index"),
    )


def _enforce_panel_gates(prepared: PreparedFrontierRun) -> None:
    preflight_rows = _rows_for_phase(prepared, "preflight")
    if not preflight_rows:
        return
    if not _complete_preflight(prepared):
        raise CostGateError("panel phase requires complete preflight")
    projection = project_cost(prepared)
    if projection["authorized_to_continue"] is not True:
        raise CostGateError("review threshold cost gate requires operator approval")


def _complete_preflight(prepared: PreparedFrontierRun) -> bool:
    preflight_count = len(_rows_for_phase(prepared, "preflight"))
    if preflight_count == 0:
        return True
    for provider in _PROVIDERS:
        results = [
            result
            for result in _provider_results(prepared, provider)
            if result.get("phase") == "preflight"
            and _result_still_bound_to_prepared(prepared, result)
        ]
        if len(results) != preflight_count:
            return False
    return True


def _require_complete_panel_outputs(prepared: PreparedFrontierRun) -> None:
    expected = len(_rows_for_phase(prepared, "panel"))
    for provider in _PROVIDERS:
        results = [
            result
            for result in _provider_results(prepared, provider)
            if result.get("phase") == "panel"
            and _result_still_bound_to_prepared(prepared, result)
        ]
        if len(results) != expected:
            raise ValueError("provider outputs are not complete")


def _cumulative_cost(prepared: PreparedFrontierRun) -> float:
    total = 0.0
    for provider in _PROVIDERS:
        total += sum(
            _number(result.get("estimated_cost_usd"))
            for result in _provider_results(prepared, provider)
            if _result_still_bound_to_prepared(prepared, result)
        )
    return total


def _rows_for_phase(prepared: PreparedFrontierRun, phase: str) -> tuple[PanelRow, ...]:
    row_phase = "evaluation" if phase == "panel" else phase
    return tuple(row for row in prepared.examples if row.phase == row_phase)


def _validate_phase_provider(phase: str, provider: str) -> None:
    if phase not in _PHASES:
        raise ValueError("phase must be preflight or panel")
    if provider not in _PROVIDERS:
        raise ValueError("provider must be gemini or reka")


def _validate_transport_provider(provider: str, transport: FrontierTransport) -> None:
    transport_provider = getattr(transport, "provider_name", "")
    if provider == "gemini":
        if transport_provider not in {"openrouter", "fixture"}:
            raise ValueError("transport provider does not match gemini/openrouter")
        return
    if transport_provider not in {"reka", "fixture"}:
        raise ValueError("transport provider does not match reka")


def _provider_results(
    prepared: PreparedFrontierRun, provider: str
) -> list[Mapping[str, object]]:
    return _read_jsonl(prepared.path / "results" / f"{provider}.jsonl")


def _request_records(prepared: PreparedFrontierRun) -> list[Mapping[str, object]]:
    return _read_jsonl(prepared.path / "requests.jsonl")


def _attempt_records(prepared: PreparedFrontierRun) -> list[Mapping[str, object]]:
    return _read_jsonl(prepared.path / "attempts.jsonl")


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        return []
    records: list[Mapping[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL record must be an object: {path}")
        records.append(payload)
    return records


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())


def _write_terminal_status(prepared: PreparedFrontierRun) -> None:
    payload = status(prepared)
    _write_json_atomically(prepared.path / "terminal-status.json", payload, replace=True)


def _write_json_atomically(
    path: Path,
    payload: Mapping[str, object],
    *,
    replace: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        return
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    if replace:
        temporary.replace(path)
    else:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _row_from_mapping(raw: object, *, location: str) -> PanelRow:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{location} must be an object")
    return PanelRow(
        ordinal=_required_int(raw.get("ordinal"), f"{location}.ordinal"),
        phase=_required_text(raw.get("phase"), f"{location}.phase"),
        video_id=_required_text(raw.get("video_id"), f"{location}.video_id"),
        question=_required_text(raw.get("question"), f"{location}.question"),
        category=_required_text(raw.get("category"), f"{location}.category"),
        evidence_topology=_required_text(
            raw.get("evidence_topology"), f"{location}.evidence_topology"
        ),
        selection_rationale=_required_text(
            raw.get("selection_rationale"), f"{location}.selection_rationale"
        ),
        media_relative_path=_required_text(
            raw.get("media_relative_path"), f"{location}.media_relative_path"
        ),
        media_sha256=_required_text(raw.get("media_sha256"), f"{location}.media_sha256"),
        size_bytes=_required_int(raw.get("size_bytes"), f"{location}.size_bytes"),
        duration_seconds=_required_float(
            raw.get("duration_seconds"), f"{location}.duration_seconds"
        ),
        has_audio=_required_bool(raw.get("has_audio"), f"{location}.has_audio"),
        has_video=_required_bool(raw.get("has_video"), f"{location}.has_video"),
    )


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _required_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("cost value must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("cost value must be finite and non-negative")
    return number


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    frozen: dict[str, Mapping[str, object]] = {}
    for key, nested in value.items():
        if not isinstance(key, str) or not isinstance(nested, Mapping):
            raise ValueError("provider configs must be nested mappings")
        frozen[key] = MappingProxyType(dict(nested))
    return MappingProxyType(frozen)


def _coerce_prepared(prepared: PreparedFrontierRun | Path) -> PreparedFrontierRun:
    if isinstance(prepared, PreparedFrontierRun):
        return prepared
    return load_prepared(Path(prepared))


def _normalized_key(key: object) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _reject_label_fields(value: object, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalized_key(key) in _FORBIDDEN_LABEL_KEYS:
                raise ValueError(f"label-free manifest contains forbidden field at {location}.{key}")
            _reject_label_fields(nested, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_label_fields(nested, location=f"{location}[{index}]")


__all__ = [
    "PRICE_SNAPSHOT",
    "AmbiguousStartedAttemptError",
    "CostGateError",
    "FixtureTransport",
    "FrontierRunSummary",
    "PreparedFrontierRun",
    "freeze_provider_outputs",
    "load_prepared",
    "prepare_fixture_run",
    "prepare_run",
    "project_cost",
    "run_phase",
    "status",
]
