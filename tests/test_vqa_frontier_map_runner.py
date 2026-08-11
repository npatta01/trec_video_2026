from __future__ import annotations

import json
import os
import subprocess
import sys
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
    prepare_fixture_run,
    project_cost,
    freeze_provider_outputs,
    run_phase,
    status,
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


def test_fixture_run_makes_one_call_per_provider_row_and_freezes(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=2, preflight_rows=1)
    gemini = FixtureTransport(provider="openrouter")
    reka = FixtureTransport(provider="reka")

    run_phase(prepared, phase="preflight", provider="gemini", transport=gemini)
    run_phase(prepared, phase="preflight", provider="reka", transport=reka)
    run_phase(prepared, phase="panel", provider="gemini", transport=gemini)
    run_phase(prepared, phase="panel", provider="reka", transport=reka)
    freeze = freeze_provider_outputs(prepared)

    assert gemini.call_count == 3
    assert reka.call_count == 3
    assert freeze["bindings"]["provider_result_count"] == 4
    assert (prepared.path / "freeze.json").is_file()
    assert status(prepared)["complete_results"] == {
        "gemini": {"preflight": 1, "panel": 2},
        "reka": {"preflight": 1, "panel": 2},
    }


def test_invalid_json_is_terminal_without_content_retry(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=0)
    transport = FixtureTransport(raw_response="not-json")

    summary = run_phase(prepared, phase="panel", provider="gemini", transport=transport)

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

    assert request["prompt_sha256"] == prepared.prompt_sha256
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
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=0)
    started = {
        "run_id": prepared.run_id,
        "phase": "panel",
        "provider": "gemini",
        "video_id": "eval-001",
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
        run_phase(prepared, phase="panel", provider="gemini", transport=transport)

    assert transport.call_count == 0


def test_completed_result_reuse_requires_full_binding_match(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=0)
    first_transport = FixtureTransport(input_tokens=100, output_tokens=50)
    first = run_phase(
        prepared, phase="panel", provider="gemini", transport=first_transport
    )

    second_transport = FixtureTransport(input_tokens=999, output_tokens=999)
    second = run_phase(
        prepared, phase="panel", provider="gemini", transport=second_transport
    )

    assert first.passed == 1
    assert second.skipped == 1
    assert second_transport.call_count == 0

    result_path = prepared.path / "results" / "gemini.jsonl"
    row = _read_jsonl(result_path)[0]
    row["route"] = "unexpected-route"
    result_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    third_transport = FixtureTransport(route="fixture-route")
    third = run_phase(prepared, phase="panel", provider="gemini", transport=third_transport)

    assert third.passed == 1
    assert third_transport.call_count == 1


def test_provider_ledgers_are_separate_and_wrong_transport_provider_is_rejected(
    tmp_path: Path,
) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=0)

    with pytest.raises(ValueError, match="provider"):
        run_phase(
            prepared,
            phase="panel",
            provider="gemini",
            transport=FixtureTransport(provider="reka"),
        )

    run_phase(prepared, phase="panel", provider="gemini", transport=FixtureTransport())
    run_phase(
        prepared,
        phase="panel",
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
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=2, preflight_rows=1)
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
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=1)

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


def test_cli_fixture_lifecycle_outputs_compact_safe_json(tmp_path: Path) -> None:
    panel = tmp_path / "fixture-panel.json"
    prepared_path = tmp_path / "prepared"
    source_prepared = prepare_fixture_run(tmp_path / "source", evaluation_rows=1, preflight_rows=1)
    panel.write_text(
        (source_prepared.path / "panel.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    prepared = _run_cli("prepare", "--panel", str(panel), "--output", str(prepared_path))
    assert prepared == {
        "status": "prepared",
        "run_id": "frontier-video-map-20260811-fixture",
        "example_count": 2,
        "fixture_manifest": True,
    }

    assert _run_cli(
        "preflight", "--prepared", str(prepared_path), "--provider", "fixture"
    )["summary"] == {"expected": 2, "failed": 0, "passed": 2, "skipped": 0}
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
    prepared = prepare_fixture_run(tmp_path / "prepared", evaluation_rows=1, preflight_rows=0)
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

    prepared = prepare_fixture_run(tmp_path / "prepared", evaluation_rows=1, preflight_rows=0)
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

    assert module.main(["run", "--prepared", str(prepared.path), "--provider", "gemini"]) == 0
    assert module.main(["run", "--prepared", str(prepared.path), "--provider", "reka"]) == 0

    assert constructed[0][0] == "openrouter"
    assert constructed[1] == ("reka", None)
    policy = constructed[0][1]
    assert policy.provider_order == ()
    assert policy.allow_fallbacks is False
