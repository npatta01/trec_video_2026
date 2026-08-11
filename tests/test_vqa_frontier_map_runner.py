from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from trec_video_2026.vqa_frontier_map_runner import (
    PRICE_SNAPSHOT,
    AmbiguousStartedAttemptError,
    CostGateError,
    FixtureTransport,
    PreparedFrontierRun,
    LockContentionError,
    prepare_fixture_run,
    project_cost,
    freeze_provider_outputs,
    prepare_run,
    run_phase,
    status,
)
from trec_video_2026.vqa_frontier_map import build_event_map_prompt
from trec_video_2026.vqa_submission import canonical_sha256
import trec_video_2026.vqa_frontier_map_runner as runner_module
from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    TokenUsage,
    UploadedFileRecord,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _cli_script() -> Path:
    return Path(__file__).parents[1] / "scripts/run_vqa_frontier_map.py"


def _run_cli(*args: str, expect: int = 0) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(_cli_script()), *args],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    assert completed.returncode == expect, completed.stderr
    output = completed.stdout.strip()
    assert output
    payload = json.loads(output)
    serialized = json.dumps(payload, sort_keys=True)
    forbidden = (
        "prompt",
        "question",
        "answer",
        "media_relative_path",
        "synthetic/",
        "raw_response",
        "provider error",
        "not-json",
    )
    assert not any(term in serialized.lower() for term in forbidden)
    return payload


class StrictUploadTransport:
    provider_name = "openrouter"

    def __init__(
        self,
        *,
        provider_name: str = "openrouter",
        observed_route: str = "strict-provider-route",
        input_tokens: int = 100,
        output_tokens: int = 50,
    ) -> None:
        self.provider_name = provider_name
        self.observed_route = observed_route
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.uploaded: dict[str, Path] = {}
        self.upload_paths: list[Path] = []
        self.generated_parts: list[tuple[object, ...]] = []
        self.call_count = 0

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        assert path.is_file(), "runner must upload an actual source file"
        assert path.is_absolute(), "runner should resolve the actual source path"
        file_id = f"registered-{len(self.uploaded) + 1}"
        self.uploaded[file_id] = path
        self.upload_paths.append(path)
        return UploadedFileRecord(file_id=file_id, mime_type=mime_type)

    def delete(self, file_id: str) -> None:
        return None

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: Mapping[str, object],
    ) -> GenerationResult:
        assert len(parts) == 2
        video_part, prompt = parts
        assert isinstance(video_part, Mapping)
        assert video_part.get("file_id") in self.uploaded
        assert isinstance(prompt, str) and prompt.strip()
        self.call_count += 1
        self.generated_parts.append(tuple(parts))
        raw = json.dumps(
            {
                "schema_version": 1,
                "question_interpretation": {
                    "meaning": "Identify the strict fixture event.",
                    "required_modalities": ["visual"],
                    "required_evidence": "The uploaded video and prompt.",
                },
                "events": [
                    {
                        "event_id": "E1",
                        "start_seconds": 0.0,
                        "end_seconds": 1.0,
                        "keyframe_seconds": [0.5],
                        "scene": "Strict upload fixture scene.",
                        "entities": ["fixture"],
                        "actions": ["fixture changes"],
                        "relations": ["fixture in scene"],
                        "state_before": "before",
                        "state_change": "changed",
                        "state_after": "after",
                        "visual_observations": ["fixture changed"],
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
                        "claim": "The uploaded source supports the map.",
                    }
                ],
                "top1_answer": "The fixture changed.",
                "support_explanation": "The uploaded source shows the change.",
                "could_be_wrong_because": ["The fixture is synthetic."],
            },
            sort_keys=True,
        )
        return GenerationResult(
            payload=json.loads(raw),
            raw_response=raw,
            usage=TokenUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            ),
            provider_metadata={
                "provider": self.observed_route,
                "model": model,
                "request_id": f"strict-{self.call_count}",
            },
        )


def test_fixture_run_makes_one_call_per_provider_row_and_freezes(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=2, preflight_rows=2)
    gemini = FixtureTransport(provider="openrouter")
    reka = FixtureTransport(provider="reka")

    run_phase(prepared, phase="preflight", provider="gemini", transport=gemini)
    run_phase(prepared, phase="preflight", provider="reka", transport=reka)
    run_phase(prepared, phase="panel", provider="gemini", transport=gemini)
    run_phase(prepared, phase="panel", provider="reka", transport=reka)
    freeze = freeze_provider_outputs(prepared)

    assert gemini.call_count == 4
    assert reka.call_count == 4
    assert freeze["bindings"]["provider_result_count"] == 4
    assert (prepared.path / "freeze.json").is_file()
    assert status(prepared)["complete_results"] == {
        "gemini": {"preflight": 2, "panel": 2},
        "reka": {"preflight": 2, "panel": 2},
    }


def test_invalid_json_is_terminal_without_content_retry(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=0, preflight_rows=1)
    transport = FixtureTransport(raw_response="not-json")

    summary = run_phase(prepared, phase="preflight", provider="gemini", transport=transport)

    assert summary.failed == 1
    assert transport.call_count == 1
    attempts = _read_jsonl(prepared.path / "attempts.jsonl")
    assert [attempt["status"] for attempt in attempts] == ["started", "failed"]
    assert attempts[-1]["failure_code"] == "invalid_json"
    assert _read_jsonl(prepared.path / "results" / "gemini.jsonl") == []


def test_prepared_run_is_immutable_and_writes_safe_manifest(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=1)

    assert is_dataclass(prepared)
    assert PreparedFrontierRun.__dataclass_params__.frozen
    with pytest.raises(FrozenInstanceError):
        prepared.run_id = "changed"

    manifest = json.loads(
        (prepared.path / "run-manifest.json").read_text(encoding="utf-8")
    )
    serialized = json.dumps(manifest, sort_keys=True)
    assert manifest["panel_sha256"] == prepared.panel_sha256
    assert manifest["prompt_sha256"] == prepared.prompt_sha256
    assert manifest["schema_sha256"] == prepared.schema_sha256
    assert manifest["fixture_manifest"] is True
    assert "/tmp" not in serialized
    assert str(tmp_path) not in serialized
    assert "data:" not in serialized
    assert "answer" not in serialized.lower()
    assert "reference" not in serialized.lower()
    assert "label" not in serialized.lower()


def test_requests_bind_prompt_schema_source_model_route_and_media_identity(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=1)
    run_phase(
        prepared,
        phase="preflight",
        provider="gemini",
        transport=FixtureTransport(provider="openrouter", route="google-vertex"),
    )

    request = _read_jsonl(prepared.path / "requests.jsonl")[0]
    result = _read_jsonl(prepared.path / "results" / "gemini.jsonl")[0]
    row = next(row for row in prepared.examples if row.video_id == request["video_id"])

    assert request["prompt_sha256"] == canonical_sha256(
        build_event_map_prompt(row.question, row.duration_seconds)
    )
    assert request["schema_sha256"] == prepared.schema_sha256
    assert request["panel_sha256"] == prepared.panel_sha256
    assert request["source_sha256"] == prepared.examples[-1].media_sha256
    assert request["model_id"] == "gemini-3.1-pro-preview"
    assert request["route"] == "google-vertex"
    assert "media_relative_path" not in request
    assert "question" not in request
    assert result["request_sha256"] == request["request_sha256"]
    assert result["source_sha256"] == request["source_sha256"]
    assert result["model_id"] == request["model_id"]
    assert result["route"] == request["route"]
    assert result["prompt_sha256"] == request["prompt_sha256"]
    assert result["schema_sha256"] == request["schema_sha256"]


def test_lone_started_attempt_fails_closed_without_rebilling(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=0, preflight_rows=1)
    started = {
        "run_id": prepared.run_id,
        "phase": "preflight",
        "provider": "gemini",
        "video_id": "preflight-001",
        "request_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "model_id": "gemini-3.1-pro-preview",
        "route": "fixture-route",
        "prompt_sha256": prepared.prompt_sha256,
        "schema_sha256": prepared.schema_sha256,
        "status": "started",
        "attempt_index": 1,
    }
    with (prepared.path / "attempts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(started, sort_keys=True) + "\n")
        handle.flush()

    transport = FixtureTransport()
    with pytest.raises(AmbiguousStartedAttemptError, match="operator"):
        run_phase(prepared, phase="preflight", provider="gemini", transport=transport)

    assert transport.call_count == 0


def test_completed_result_reuse_requires_full_binding_match(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=0, preflight_rows=1)
    first_transport = FixtureTransport(input_tokens=100, output_tokens=50)
    first = run_phase(
        prepared, phase="preflight", provider="gemini", transport=first_transport
    )

    second_transport = FixtureTransport(input_tokens=999, output_tokens=999)
    second = run_phase(
        prepared, phase="preflight", provider="gemini", transport=second_transport
    )

    assert first.passed == 1
    assert second.skipped == 1
    assert second_transport.call_count == 0

    result_path = prepared.path / "results" / "gemini.jsonl"
    row = _read_jsonl(result_path)[0]
    row["route"] = "unexpected-route"
    result_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    third_transport = FixtureTransport(route="fixture-route")
    third = run_phase(
        prepared, phase="preflight", provider="gemini", transport=third_transport
    )

    assert third.passed == 1
    assert third_transport.call_count == 1


def test_provider_ledgers_are_separate_and_wrong_transport_provider_is_rejected(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=0, preflight_rows=1)

    with pytest.raises(ValueError, match="provider"):
        run_phase(
            prepared,
            phase="preflight",
            provider="gemini",
            transport=FixtureTransport(provider="reka"),
        )

    run_phase(prepared, phase="preflight", provider="gemini", transport=FixtureTransport())
    run_phase(
        prepared,
        phase="preflight",
        provider="reka",
        transport=FixtureTransport(provider="reka"),
    )

    assert (prepared.path / "results" / "gemini.jsonl").is_file()
    assert (prepared.path / "results" / "reka.jsonl").is_file()
    assert len(_read_jsonl(prepared.path / "results" / "gemini.jsonl")) == 1
    assert len(_read_jsonl(prepared.path / "results" / "reka.jsonl")) == 1


def test_panel_refuses_without_complete_preflight_and_review_cost_gate(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=2)

    with pytest.raises(CostGateError, match="preflight"):
        run_phase(prepared, phase="panel", provider="gemini", transport=FixtureTransport())

    run_phase(prepared, phase="preflight", provider="gemini", transport=FixtureTransport())
    with pytest.raises(CostGateError, match="preflight"):
        run_phase(prepared, phase="panel", provider="gemini", transport=FixtureTransport())

    run_phase(
        prepared,
        phase="preflight",
        provider="reka",
        transport=FixtureTransport(provider="reka", reported_cost_usd=8.0),
    )
    projection = project_cost(prepared)
    assert projection["authorized_to_continue"] is False
    assert projection["observed_projected_usd"] > PRICE_SNAPSHOT["review_threshold_usd"]
    with pytest.raises(CostGateError, match="review"):
        run_phase(prepared, phase="panel", provider="gemini", transport=FixtureTransport())


def test_cumulative_absolute_cost_cap_is_checked_before_dispatch(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=2, preflight_rows=2)
    run_phase(
        prepared,
        phase="preflight",
        provider="gemini",
        transport=FixtureTransport(reported_cost_usd=0.10),
    )
    run_phase(
        prepared,
        phase="preflight",
        provider="reka",
        transport=FixtureTransport(provider="reka", reported_cost_usd=0.10),
    )
    transport = FixtureTransport(
        reported_cost_usd=PRICE_SNAPSHOT["absolute_cap_usd"] + 1.0
    )

    with pytest.raises(CostGateError, match="absolute"):
        run_phase(prepared, phase="panel", provider="gemini", transport=transport)

    assert transport.call_count == 0


def test_project_cost_uses_reported_openrouter_cost_and_reka_video_minutes(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=3, preflight_rows=1)
    run_phase(
        prepared,
        phase="preflight",
        provider="gemini",
        transport=FixtureTransport(provider="openrouter", reported_cost_usd=0.25),
    )
    run_phase(
        prepared,
        phase="preflight",
        provider="reka",
        transport=FixtureTransport(provider="reka", input_tokens=1_000_000, output_tokens=0),
    )

    projection = project_cost(prepared)

    assert projection["price_snapshot_date"] == "2026-08-11"
    assert projection["providers"]["gemini"]["observed_preflight_usd"] == 0.25
    assert projection["providers"]["reka"]["observed_preflight_usd"] == pytest.approx(
        2.0 + (0.5 * PRICE_SNAPSHOT["reka_video_per_minute"])
    )
    assert projection["low_projected_usd"] <= projection["observed_projected_usd"]
    assert projection["high_projected_usd"] >= projection["observed_projected_usd"]


def test_status_and_freeze_require_complete_provider_outputs(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=2)

    with pytest.raises(ValueError, match="complete"):
        freeze_provider_outputs(prepared)

    run_phase(prepared, phase="preflight", provider="gemini", transport=FixtureTransport())
    run_phase(
        prepared,
        phase="preflight",
        provider="reka",
        transport=FixtureTransport(provider="reka"),
    )
    run_phase(prepared, phase="panel", provider="gemini", transport=FixtureTransport())
    run_phase(
        prepared,
        phase="panel",
        provider="reka",
        transport=FixtureTransport(provider="reka"),
    )

    summary = status(prepared)
    freeze = freeze_provider_outputs(prepared)

    assert summary["frozen"] is False
    assert freeze["status"] == "frozen"
    assert freeze["bindings"]["panel_sha256"] == prepared.panel_sha256
    assert freeze["bindings"]["prompt_sha256"] == prepared.prompt_sha256
    assert freeze["bindings"]["schema_sha256"] == prepared.schema_sha256
    assert freeze["bindings"]["provider_result_count"] == 2
    assert status(prepared)["frozen"] is True


def test_real_transport_uploads_actual_source_before_generate_and_artifacts_stay_safe(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=0, preflight_rows=1)
    transport = StrictUploadTransport()

    summary = run_phase(
        prepared, phase="preflight", provider="gemini", transport=transport
    )

    assert summary.passed == 1
    assert transport.call_count == 1
    assert transport.upload_paths[0].name == "preflight-001.mp4"
    request = _read_jsonl(prepared.path / "requests.jsonl")[0]
    serialized_request = json.dumps(request, sort_keys=True)
    assert str(transport.upload_paths[0]) not in serialized_request
    assert "media_relative_path" not in request
    assert "data:" not in serialized_request


def test_terminal_failed_attempt_is_reused_without_rebilling(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=0, preflight_rows=1)
    first = FixtureTransport(raw_response="not-json")
    second = FixtureTransport(raw_response="not-json")

    failed = run_phase(prepared, phase="preflight", provider="gemini", transport=first)
    reused = run_phase(prepared, phase="preflight", provider="gemini", transport=second)

    assert failed.failed == 1
    assert reused.failed == 1
    assert second.call_count == 0
    assert [attempt["status"] for attempt in _read_jsonl(prepared.path / "attempts.jsonl")] == [
        "started",
        "failed",
    ]


def test_request_lock_blocks_reentrant_duplicate_paid_call(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=0, preflight_rows=1)
    lock_error = getattr(runner_module, "LockContentionError", RuntimeError)

    class ReentrantEstimateTransport(StrictUploadTransport):
        def __init__(self) -> None:
            super().__init__()
            self.reentered = False

        def estimate_cost_usd(self, provider: str, row: object) -> float:
            if not self.reentered:
                self.reentered = True
                with pytest.raises(lock_error):
                    run_phase(
                        prepared,
                        phase="preflight",
                        provider="gemini",
                        transport=StrictUploadTransport(),
                    )
            return 0.001

    transport = ReentrantEstimateTransport()

    summary = run_phase(
        prepared, phase="preflight", provider="gemini", transport=transport
    )

    assert summary.passed == 1
    assert transport.call_count == 1
    assert len(_read_jsonl(prepared.path / "results" / "gemini.jsonl")) == 1


def test_request_binds_actual_row_prompt_and_panel_edits_fail_before_reuse(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=0, preflight_rows=2)
    run_phase(prepared, phase="preflight", provider="gemini", transport=FixtureTransport())
    requests = _read_jsonl(prepared.path / "requests.jsonl")

    row_by_video = {row.video_id: row for row in prepared.examples}
    assert requests[0]["prompt_sha256"] == canonical_sha256(
        build_event_map_prompt(
            row_by_video[str(requests[0]["video_id"])].question,
            row_by_video[str(requests[0]["video_id"])].duration_seconds,
        )
    )
    assert requests[0]["prompt_sha256"] != requests[1]["prompt_sha256"]

    panel_path = prepared.path / "panel.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel["examples"][0]["question"] = "What text changed after the fixture event?"
    panel_path.write_text(json.dumps(panel, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="panel"):
        run_phase(prepared.path, phase="preflight", provider="gemini", transport=FixtureTransport())


def test_openrouter_observed_preflight_route_is_recorded_and_cli_pins_panel_route(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path / "prepared", evaluation_rows=1, preflight_rows=2)
    observed_route = "openrouter/google-vertex"
    run_phase(
        prepared,
        phase="preflight",
        provider="gemini",
        transport=StrictUploadTransport(observed_route=observed_route),
    )
    run_phase(
        prepared,
        phase="preflight",
        provider="reka",
        transport=StrictUploadTransport(provider_name="reka", observed_route="reka"),
    )

    result = _read_jsonl(prepared.path / "results" / "gemini.jsonl")[0]
    assert result["route"] == observed_route
    assert result["route"] != "openrouter-unpinned"

    import importlib.util

    spec = importlib.util.spec_from_file_location("run_vqa_frontier_map_cli", _cli_script())
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    constructed: list[object] = []

    def openrouter_factory(*, policy: object) -> StrictUploadTransport:
        constructed.append(policy)
        return StrictUploadTransport(observed_route=observed_route)

    module.create_openrouter_transport = openrouter_factory

    assert module.main(["run", "--prepared", str(prepared.path), "--provider", "gemini"]) == 0
    assert constructed
    assert constructed[0].provider_order == (observed_route,)
    assert "openrouter-unpinned" not in constructed[0].provider_order


def test_real_transport_absent_estimate_and_failed_attempts_count_against_cap(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=0, preflight_rows=1)
    failed_cost = PRICE_SNAPSHOT["absolute_cap_usd"] - 0.000001
    _read_jsonl(prepared.path / "attempts.jsonl")
    with (prepared.path / "attempts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "run_id": prepared.run_id,
                    "phase": "preflight",
                    "provider": "gemini",
                    "video_id": "already-failed",
                    "request_sha256": "c" * 64,
                    "status": "failed",
                    "attempt_index": 1,
                    "estimated_cost_usd": failed_cost,
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    transport = StrictUploadTransport()

    with pytest.raises(CostGateError, match="absolute"):
        run_phase(prepared, phase="preflight", provider="gemini", transport=transport)

    assert transport.call_count == 0


def test_run_wide_budget_lock_serializes_distinct_request_cost_checks(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=0, preflight_rows=1)
    first_cost = PRICE_SNAPSHOT["absolute_cap_usd"] - 0.0005
    second_cost = 0.0005
    observed_started_before_reentry: list[str] = []

    class BudgetTransport(StrictUploadTransport):
        def __init__(
            self,
            *,
            provider_name: str,
            reported_cost_usd: float,
            reenter: bool = False,
        ) -> None:
            super().__init__(provider_name=provider_name)
            self.reported_cost_usd = reported_cost_usd
            self.reenter = reenter

        def estimate_cost_usd(self, provider: str, row: object) -> float:
            return self.reported_cost_usd

        def generate(
            self,
            *,
            model: str,
            parts: Sequence[object],
            schema: Mapping[str, object],
        ) -> GenerationResult:
            if self.reenter:
                attempts = _read_jsonl(prepared.path / "attempts.jsonl")
                observed_started_before_reentry.extend(
                    attempt["status"]
                    for attempt in attempts
                    if attempt["provider"] == "gemini"
                )
                with pytest.raises(LockContentionError):
                    run_phase(
                        prepared,
                        phase="preflight",
                        provider="reka",
                        transport=BudgetTransport(
                            provider_name="reka",
                            reported_cost_usd=second_cost,
                        ),
                    )
            result = super().generate(model=model, parts=parts, schema=schema)
            metadata = dict(result.provider_metadata)
            metadata["reported_cost_usd"] = self.reported_cost_usd
            return GenerationResult(
                payload=result.payload,
                raw_response=result.raw_response,
                usage=result.usage,
                provider_metadata=metadata,
            )

    first = run_phase(
        prepared,
        phase="preflight",
        provider="gemini",
        transport=BudgetTransport(
            provider_name="openrouter",
            reported_cost_usd=first_cost,
            reenter=True,
        ),
    )

    assert first.passed == 1
    assert observed_started_before_reentry == ["started"]
    attempts = _read_jsonl(prepared.path / "attempts.jsonl")
    assert [attempt["status"] for attempt in attempts if attempt["provider"] == "gemini"] == [
        "started",
        "succeeded",
    ]
    with pytest.raises(CostGateError, match="absolute"):
        run_phase(
            prepared,
            phase="preflight",
            provider="reka",
            transport=BudgetTransport(
                provider_name="reka",
                reported_cost_usd=second_cost,
            ),
        )


def test_panel_requires_exact_four_successful_preflight_results(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=1)
    run_phase(prepared, phase="preflight", provider="gemini", transport=FixtureTransport())
    run_phase(
        prepared,
        phase="preflight",
        provider="reka",
        transport=FixtureTransport(provider="reka"),
    )

    with pytest.raises(CostGateError, match="preflight"):
        run_phase(prepared, phase="panel", provider="gemini", transport=FixtureTransport())


def test_freeze_requires_exact_unique_provider_by_evaluation_row_coverage(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=2, preflight_rows=2)
    run_phase(prepared, phase="preflight", provider="gemini", transport=FixtureTransport())
    run_phase(
        prepared,
        phase="preflight",
        provider="reka",
        transport=FixtureTransport(provider="reka"),
    )
    run_phase(prepared, phase="panel", provider="gemini", transport=FixtureTransport())
    run_phase(
        prepared,
        phase="panel",
        provider="reka",
        transport=FixtureTransport(provider="reka"),
    )
    gemini_result = _read_jsonl(prepared.path / "results" / "gemini.jsonl")
    panel_results = [row for row in gemini_result if row["phase"] == "panel"]
    duplicated = [row for row in gemini_result if row["phase"] == "preflight"]
    duplicated.extend([panel_results[0], panel_results[0]])
    (prepared.path / "results" / "gemini.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in duplicated) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="complete"):
        freeze_provider_outputs(prepared)


def test_prepare_refuses_existing_run_directory_and_rejects_unsafe_video_ids(
    tmp_path: Path,
) -> None:
    source = prepare_fixture_run(tmp_path / "source", evaluation_rows=1, preflight_rows=2)
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(
        (source.path / "panel.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    prepare_run(panel_path, output)

    with pytest.raises(FileExistsError, match="existing"):
        prepare_run(panel_path, output)

    unsafe = json.loads(panel_path.read_text(encoding="utf-8"))
    unsafe["examples"][0]["video_id"] = "../evil"
    unsafe_panel = tmp_path / "unsafe-panel.json"
    unsafe_panel.write_text(json.dumps(unsafe, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="video_id"):
        prepare_run(unsafe_panel, tmp_path / "unsafe-output")


def test_cli_fixture_lifecycle_outputs_compact_safe_json(tmp_path: Path) -> None:
    panel = tmp_path / "fixture-panel.json"
    prepared_path = tmp_path / "prepared"
    source_prepared = prepare_fixture_run(
        tmp_path / "source", evaluation_rows=1, preflight_rows=2
    )
    panel.write_text(
        (source_prepared.path / "panel.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    prepared = _run_cli("prepare", "--panel", str(panel), "--output", str(prepared_path))
    assert prepared == {
        "status": "prepared",
        "run_id": "frontier-video-map-20260811-fixture",
        "example_count": 3,
        "fixture_manifest": True,
    }

    assert _run_cli(
        "preflight", "--prepared", str(prepared_path), "--provider", "fixture"
    )["summary"] == {"expected": 4, "failed": 0, "passed": 4, "skipped": 0}
    projection = _run_cli("project-cost", "--prepared", str(prepared_path))
    assert projection["authorized_to_continue"] is True
    assert _run_cli(
        "run", "--prepared", str(prepared_path), "--provider", "fixture"
    )["summary"] == {"expected": 2, "failed": 0, "passed": 2, "skipped": 0}
    status_payload = _run_cli("status", "--prepared", str(prepared_path))
    assert status_payload["frozen"] is False
    freeze = _run_cli("freeze", "--prepared", str(prepared_path))
    assert freeze["status"] == "frozen"
    assert freeze["provider_result_count"] == 2


def test_cli_rejects_fixture_provider_for_nonfixture_manifest(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path / "prepared", evaluation_rows=1, preflight_rows=2)
    manifest_path = prepared.path / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixture_manifest"] = False
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    payload = _run_cli(
        "run",
        "--prepared",
        str(prepared.path),
        "--provider",
        "fixture",
        expect=2,
    )

    assert payload == {"status": "error", "error": "fixture_not_allowed"}


def test_cli_real_provider_construction_paths_are_offline_monkeypatchable(
    tmp_path: Path,
) -> None:
    import importlib.util

    script = _cli_script()
    spec = importlib.util.spec_from_file_location("run_vqa_frontier_map_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prepared = prepare_fixture_run(tmp_path / "prepared", evaluation_rows=1, preflight_rows=2)
    constructed: list[tuple[str, object]] = []

    class OfflineTransport(FixtureTransport):
        def __init__(self, provider: str = "openrouter") -> None:
            super().__init__(provider=provider)

    def openrouter_factory(*, policy: object) -> OfflineTransport:
        constructed.append(("openrouter", policy))
        return OfflineTransport("openrouter")

    def reka_factory() -> OfflineTransport:
        constructed.append(("reka", None))
        return OfflineTransport("reka")

    module.create_openrouter_transport = openrouter_factory
    module.create_reka_transport = reka_factory

    assert module.main(["preflight", "--prepared", str(prepared.path), "--provider", "gemini"]) == 0
    assert module.main(["preflight", "--prepared", str(prepared.path), "--provider", "reka"]) == 0
    assert module.main(["run", "--prepared", str(prepared.path), "--provider", "gemini"]) == 0

    assert constructed[0][0] == "openrouter"
    assert constructed[1] == ("reka", None)
    assert constructed[2][0] == "openrouter"
    policy = constructed[2][1]
    assert policy.provider_order == ("fixture-route",)
    assert policy.allow_fallbacks is False
