#!/usr/bin/env python3
"""Thin operational CLI for the frontier VQA event-map runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_frontier_map_runner import (
    CostGateError,
    FixtureTransport,
    FrontierRunSummary,
    freeze_provider_outputs,
    freeze_preflight_outputs,
    prepare_run,
    project_cost,
    run_phase,
    status,
)
from trec_video_2026.vqa_submission_openrouter import (
    OpenRouterRequestPolicy,
    create_openrouter_transport,
)
from trec_video_2026.vqa_frontier_map_reka import create_reka_transport


def _write_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _manifest(prepared: Path) -> dict[str, Any]:
    payload = json.loads((prepared / "run-manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid_manifest")
    return payload


def _fixture_allowed(prepared: Path) -> bool:
    return _manifest(prepared).get("fixture_manifest") is True


def _safe_prepare(args: argparse.Namespace) -> dict[str, Any]:
    prepared = prepare_run(Path(args.panel), Path(args.output))
    manifest = _manifest(prepared.path)
    return {
        "status": "prepared",
        "run_id": prepared.run_id,
        "example_count": len(prepared.examples),
        "fixture_manifest": manifest.get("fixture_manifest") is True,
    }


def _summary_payload(summary: FrontierRunSummary) -> dict[str, int]:
    return {
        "expected": summary.expected,
        "failed": summary.failed,
        "passed": summary.passed,
        "skipped": summary.skipped,
    }


def _aggregate_summaries(summaries: list[FrontierRunSummary]) -> dict[str, int]:
    return {
        "expected": sum(summary.expected for summary in summaries),
        "failed": sum(summary.failed for summary in summaries),
        "passed": sum(summary.passed for summary in summaries),
        "skipped": sum(summary.skipped for summary in summaries),
    }


def _pinned_route(prepared: Path) -> tuple[str, ...]:
    result_path = prepared / "results" / "gemini.jsonl"
    if not result_path.exists():
        return ()
    routes: set[str] = set()
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            isinstance(row, dict)
            and row.get("phase") == "preflight"
            and isinstance(row.get("route"), str)
            and row["route"] not in {"openrouter-unpinned", "openrouter-dynamic-preflight"}
        ):
            routes.add(row["route"])
    if len(routes) == 1:
        return (next(iter(routes)),)
    return ()


def _transport(prepared: Path, provider: str) -> Any:
    if provider == "gemini":
        return create_openrouter_transport(
            policy=OpenRouterRequestPolicy(
                provider_order=_pinned_route(prepared),
                allow_fallbacks=False,
                use_structured_outputs=False,
            )
        )
    if provider == "reka":
        return create_reka_transport()
    raise ValueError("unsupported_provider")


def _run_one(
    prepared: Path,
    *,
    phase: str,
    provider: str,
    video_ids: tuple[str, ...] | None = None,
) -> FrontierRunSummary:
    if provider == "fixture":
        raise ValueError("fixture aggregate must be handled separately")
    return run_phase(
        prepared,
        phase=phase,
        provider=provider,
        transport=_transport(prepared, provider),
        stop_on_failure=phase == "preflight",
        video_ids=video_ids,
    )


def _run_fixture(prepared: Path, *, phase: str) -> list[FrontierRunSummary]:
    if not _fixture_allowed(prepared):
        raise PermissionError("fixture_not_allowed")
    return [
        run_phase(
            prepared,
            phase=phase,
            provider="gemini",
            transport=FixtureTransport(provider="openrouter"),
        ),
        run_phase(
            prepared,
            phase=phase,
            provider="reka",
            transport=FixtureTransport(provider="reka"),
        ),
    ]


def _safe_phase(args: argparse.Namespace, *, phase: str) -> dict[str, Any]:
    prepared = Path(args.prepared)
    if args.provider == "fixture":
        summaries = _run_fixture(prepared, phase=phase)
        return {"status": "ok", "summary": _aggregate_summaries(summaries)}
    video_ids = tuple(args.video_id) if args.video_id else None
    summary = _run_one(
        prepared, phase=phase, provider=args.provider, video_ids=video_ids
    )
    return {"status": "ok", "summary": _summary_payload(summary)}


def _safe_project_cost(args: argparse.Namespace) -> dict[str, Any]:
    projection = project_cost(Path(args.prepared))
    return {
        "status": "ok",
        "authorized_to_continue": projection["authorized_to_continue"],
        "low_projected_usd": projection["low_projected_usd"],
        "observed_projected_usd": projection["observed_projected_usd"],
        "high_projected_usd": projection["high_projected_usd"],
        "review_threshold_usd": projection["review_threshold_usd"],
        "absolute_cap_usd": projection["absolute_cap_usd"],
    }


def _safe_status(args: argparse.Namespace) -> dict[str, Any]:
    summary = status(Path(args.prepared))
    return {
        "status": "ok",
        "run_id": summary["run_id"],
        "complete_results": summary["complete_results"],
        "failed_attempts": summary["failed_attempts"],
        "ambiguous_started_attempts": summary["ambiguous_started_attempts"],
        "estimated_cost_usd": summary["estimated_cost_usd"],
        "frozen": summary["frozen"],
    }


def _safe_freeze(args: argparse.Namespace) -> dict[str, Any]:
    freeze = freeze_provider_outputs(Path(args.prepared))
    bindings = freeze.get("bindings", {})
    provider_result_count = (
        bindings.get("provider_result_count") if isinstance(bindings, dict) else None
    )
    return {
        "status": freeze["status"],
        "artifact_sha256": freeze["artifact_sha256"],
        "provider_result_count": provider_result_count,
    }


def _safe_freeze_preflight(args: argparse.Namespace) -> dict[str, Any]:
    freeze = freeze_preflight_outputs(
        Path(args.prepared),
        provider=str(args.provider),
        video_ids=tuple(args.video_id) if args.video_id else None,
    )
    bindings = freeze.get("bindings", {})
    return {
        "status": freeze["status"],
        "artifact_sha256": freeze["artifact_sha256"],
        "provider": bindings.get("provider") if isinstance(bindings, dict) else None,
        "result_count": bindings.get("result_count")
        if isinstance(bindings, dict)
        else None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--panel", required=True)
    prepare.add_argument("--output", required=True)

    for name in ("preflight", "run"):
        command = subcommands.add_parser(name)
        command.add_argument("--prepared", required=True)
        command.add_argument("--provider", choices=("gemini", "reka", "fixture"), required=True)
        command.add_argument("--video-id", action="append")

    project = subcommands.add_parser("project-cost")
    project.add_argument("--prepared", required=True)

    status_command = subcommands.add_parser("status")
    status_command.add_argument("--prepared", required=True)

    freeze = subcommands.add_parser("freeze")
    freeze.add_argument("--prepared", required=True)
    freeze_preflight = subcommands.add_parser("freeze-preflight")
    freeze_preflight.add_argument("--prepared", required=True)
    freeze_preflight.add_argument(
        "--provider", choices=("gemini", "reka"), required=True
    )
    freeze_preflight.add_argument("--video-id", action="append")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            payload = _safe_prepare(args)
        elif args.command == "preflight":
            payload = _safe_phase(args, phase="preflight")
        elif args.command == "project-cost":
            payload = _safe_project_cost(args)
        elif args.command == "run":
            payload = _safe_phase(args, phase="panel")
        elif args.command == "status":
            payload = _safe_status(args)
        elif args.command == "freeze":
            payload = _safe_freeze(args)
        elif args.command == "freeze-preflight":
            payload = _safe_freeze_preflight(args)
        else:
            raise ValueError("unsupported_command")
    except PermissionError:
        _write_json({"status": "error", "error": "fixture_not_allowed"})
        return 2
    except CostGateError:
        _write_json({"status": "error", "error": "cost_gate"})
        return 2
    except Exception:  # noqa: BLE001 - CLI must not leak provider/local error bodies
        _write_json({"status": "error", "error": "operation_failed"})
        return 2
    _write_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
