#!/usr/bin/env python3
"""Prepare and score the four-option VQA evidence-agent experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_experiments.vqa_mc_common import (
    OPTION_IDS,
    parse_ranking,
    score_rankings,
    sha256_file,
)


MAX_COST_USD = 1.50


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact is not an object: {path.name}")
    return value


def _rows(path: Path) -> list[Mapping[str, Any]]:
    rows = _read(path).get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"artifact rows are invalid: {path.name}")
    return list(rows)


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _label_index(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in _rows(path):
        query_id = row.get("query_id")
        answer = row.get("correct_option_id")
        if (
            not isinstance(query_id, str)
            or query_id in labels
            or answer not in OPTION_IDS
        ):
            raise ValueError("label rows are invalid")
        labels[query_id] = str(answer)
    return labels


def _prediction_index(
    path: Path,
    *,
    required_ids: Sequence[str],
    label: str,
    require_passed: bool,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in _rows(path):
        if row.get("variant_kind") != "canonical":
            continue
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or video_id in indexed:
            raise ValueError(f"{label} prediction identity drift")
        if require_passed:
            if row.get("status") != "passed":
                raise ValueError(f"{label} contains an invalid row")
            parse_ranking({"ranking": row.get("ranking")})
        indexed[video_id] = row
    missing = [video_id for video_id in required_ids if video_id not in indexed]
    if missing:
        raise ValueError(f"{label} prediction identity drift")
    return {video_id: indexed[video_id] for video_id in required_ids}


def _valid_ranking(row: Mapping[str, Any] | None) -> list[str] | None:
    if not isinstance(row, Mapping) or row.get("status") != "passed":
        return None
    try:
        return parse_ranking({"ranking": row.get("ranking")})
    except ValueError:
        return None


def _request_payload(path: Path) -> Mapping[str, Any]:
    payload = _read(path)
    rows = payload.get("rows")
    variants = payload.get("variants")
    if not isinstance(rows, list) or not isinstance(variants, list):
        raise ValueError("request bundle is invalid")
    ids = [row.get("video_id") for row in rows if isinstance(row, Mapping)]
    if (
        len(ids) != len(rows)
        or not all(isinstance(value, str) for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("request row identity is invalid")
    canonical = [
        row
        for row in variants
        if isinstance(row, Mapping) and row.get("variant_kind") == "canonical"
    ]
    if [row.get("video_id") for row in canonical] != ids:
        raise ValueError("request canonical variant identity drift")
    return payload


def prepare_canary(
    *,
    requests_path: Path,
    labels_path: Path,
    control_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    payload = _request_payload(requests_path)
    request_rows = list(payload["rows"])
    request_ids = [str(row["video_id"]) for row in request_rows]
    labels_all = _label_index(labels_path)
    if any(video_id not in labels_all for video_id in request_ids):
        raise ValueError("request/label identity drift")
    labels = {video_id: labels_all[video_id] for video_id in request_ids}
    control = _prediction_index(
        control_path,
        required_ids=request_ids,
        label="control",
        require_passed=True,
    )
    rankings = {
        video_id: parse_ranking({"ranking": control[video_id]["ranking"]})
        for video_id in request_ids
    }
    metrics = score_rankings(rankings, labels)
    if (
        len(request_ids) != 20
        or metrics["valid_count"] != 20
        or metrics["top1_correct"] != 15
    ):
        raise ValueError("control is not the frozen 15/20 control")
    miss_ids = [
        video_id
        for video_id in request_ids
        if rankings[video_id][0] != labels[video_id]
    ]
    if len(miss_ids) != 5:
        raise ValueError("frozen control does not contain exactly five misses")
    selected = set(miss_ids)
    output = {
        "schema_version": 1,
        "experiment_id": "vqa-mc-four-option-agent-canary-v1",
        "selection_kind": "frozen_control_misses",
        "source_sha256s": {
            "requests": sha256_file(requests_path),
            "labels": sha256_file(labels_path),
            "frozen_control": sha256_file(control_path),
        },
        "rows": [row for row in request_rows if row["video_id"] in selected],
        "variants": [
            row
            for row in payload["variants"]
            if isinstance(row, Mapping) and row.get("video_id") in selected
        ],
    }
    serialized = json.dumps(output, ensure_ascii=False).casefold()
    forbidden = (
        "correct_option",
        "reference_answer",
        '"label"',
        "ground_truth",
    )
    if any(value in serialized for value in forbidden):
        raise ValueError("canary request bundle contains label data")
    _write_exclusive(output_path, output)
    return {
        "status": "canary_prepared",
        "row_count": len(output["rows"]),
        "output_sha256": sha256_file(output_path),
        "source_sha256s": output["source_sha256s"],
    }


def _transitions(
    before: Mapping[str, Sequence[str] | None],
    after: Mapping[str, Sequence[str] | None],
    labels: Mapping[str, str],
) -> dict[str, int]:
    counts: Counter[str] = Counter(
        {
            "repaired": 0,
            "regressed": 0,
            "unchanged_correct": 0,
            "unchanged_wrong": 0,
        }
    )
    for video_id, correct in labels.items():
        before_correct = before.get(video_id) is not None and before[video_id][0] == correct
        after_correct = after.get(video_id) is not None and after[video_id][0] == correct
        if not before_correct and after_correct:
            counts["repaired"] += 1
        elif before_correct and not after_correct:
            counts["regressed"] += 1
        elif before_correct:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    return dict(counts)


def _safe_artifact(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path is invalid")
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    return path


def _verify_declared_sources(final: Mapping[str, Any], root: Path) -> bool:
    paths = final.get("source_paths")
    hashes = final.get("source_sha256s")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        return False
    if not paths or set(paths) != set(hashes):
        return False
    return all(
        sha256_file(_safe_artifact(root, relative)) == hashes.get(key)
        for key, relative in paths.items()
    )


def _observer_ids(notebook: Mapping[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    rounds = notebook.get("rounds")
    if not isinstance(rounds, list):
        return identifiers
    for entry in rounds:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("observer"), Mapping):
            continue
        for observation in entry["observer"].get("observations", []):
            if isinstance(observation, Mapping) and isinstance(
                observation.get("observation_id"), str
            ):
                identifiers.add(str(observation["observation_id"]))
    return identifiers


def _current_ledger(notebook: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rounds = notebook.get("rounds")
    if isinstance(rounds, list):
        for entry in reversed(rounds):
            if isinstance(entry, Mapping) and isinstance(entry.get("updater"), Mapping):
                ledger = entry["updater"].get("ledger")
                if isinstance(ledger, list):
                    return [row for row in ledger if isinstance(row, Mapping)]
    initializer = notebook.get("initializer")
    if isinstance(initializer, Mapping) and isinstance(initializer.get("ledger"), list):
        return [row for row in initializer["ledger"] if isinstance(row, Mapping)]
    return []


def _ledger_complete(notebook: Mapping[str, Any], known: set[str]) -> bool:
    ledger = _current_ledger(notebook)
    if len(ledger) != 4 or {row.get("option_id") for row in ledger} != set(OPTION_IDS):
        return False
    for row in ledger:
        support = row.get("supporting_evidence_ids", [])
        contradiction = row.get("contradicting_evidence_ids", [])
        if not isinstance(support, list) or not isinstance(contradiction, list):
            return False
        if (set(support) | set(contradiction)) - known:
            return False
    return True


def _audit_complete(root: Path, row: Mapping[str, Any]) -> bool:
    paths = row.get("observer_audit_paths")
    if not isinstance(paths, list) or not paths:
        return False
    for relative in paths:
        audit = _read(_safe_artifact(root, relative))
        if audit != {"passed": True, "matched_value_hashes": []}:
            return False
    return True


def _truncation_answer_count(root: Path) -> int:
    count = 0
    for path in root.glob("rows/*/**/attempt-1.json"):
        first = _read(path)
        if str(first.get("finish_reason", "")).upper() != "MAX_TOKENS":
            continue
        second = path.with_name("attempt-2.json")
        if not second.is_file() or str(_read(second).get("finish_reason", "")).upper() != "STOP":
            count += 1
    return count


def _by_stratum(
    predictions: Mapping[str, Sequence[str] | None],
    labels: Mapping[str, str],
    strata: Mapping[str, str],
) -> dict[str, dict[str, int | float]]:
    return {
        stratum: score_rankings(
            {
                video_id: predictions[video_id]
                for video_id in labels
                if strata[video_id] == stratum
            },
            {
                video_id: labels[video_id]
                for video_id in labels
                if strata[video_id] == stratum
            },
        )
        for stratum in sorted(set(strata.values()))
    }


def score_phase(
    *,
    phase: str,
    requests_path: Path,
    labels_path: Path,
    control_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    if phase not in {"canary", "full"}:
        raise ValueError("phase is invalid")
    payload = _request_payload(requests_path)
    request_rows = list(payload["rows"])
    ids = [str(row["video_id"]) for row in request_rows]
    labels_all = _label_index(labels_path)
    if any(video_id not in labels_all for video_id in ids):
        raise ValueError("request/label identity drift")
    labels = {video_id: labels_all[video_id] for video_id in ids}
    control_index = _prediction_index(
        control_path,
        required_ids=ids,
        label="control",
        require_passed=True,
    )
    root = run_root.resolve(strict=True)
    final_path = root / "final-predictions-freeze.json"
    summary_path = root / "summary.json"
    final_artifact = _read(final_path)
    final_index = _prediction_index(
        final_path,
        required_ids=ids,
        label="final",
        require_passed=False,
    )
    control_rankings = {
        video_id: parse_ranking({"ranking": control_index[video_id]["ranking"]})
        for video_id in ids
    }
    final_rankings = {
        video_id: _valid_ranking(final_index[video_id]) for video_id in ids
    }
    notebooks: dict[str, Mapping[str, Any]] = {}
    mandatory = 0
    blind = 0
    complete_ledgers = 0
    valid_citations = 0
    private_rows: list[dict[str, Any]] = []
    for video_id in ids:
        row = final_index[video_id]
        notebook_path = _safe_artifact(root, row.get("notebook_path"))
        if sha256_file(notebook_path) != row.get("notebook_sha256"):
            raise ValueError("final notebook hash mismatch")
        notebook = _read(notebook_path)
        notebooks[video_id] = notebook
        observations = _observer_ids(notebook)
        mandatory += bool(observations)
        blind += _audit_complete(root, row)
        complete_ledgers += _ledger_complete(notebook, observations)
        judge = row.get("judge")
        citations = (
            judge.get("decisive_evidence_ids", [])
            if isinstance(judge, Mapping)
            else []
        )
        citation_valid = bool(
            isinstance(citations, list)
            and citations
            and set(citations) <= observations
        )
        valid_citations += citation_valid
        private_rows.append(
            {
                "query_id": video_id,
                "correct_option_id": labels[video_id],
                "control_ranking": control_rankings[video_id],
                "final_ranking": final_rankings[video_id],
                "status": row.get("status"),
                "final_source": row.get("final_source"),
                "observer_action_count": row.get("observer_action_count"),
                "citation_valid": citation_valid,
                "notebook_sha256": row.get("notebook_sha256"),
            }
        )
    metrics = {
        "frozen_control": score_rankings(control_rankings, labels),
        "final": score_rankings(final_rankings, labels),
    }
    transitions = _transitions(control_rankings, final_rankings, labels)
    strata = {str(row["video_id"]): str(row["stratum"]) for row in request_rows}
    summary = _read(summary_path)
    source_valid = _verify_declared_sources(final_artifact, root)
    summary_valid = bool(
        summary.get("status") == "completed"
        and summary.get("row_count") == len(ids)
        and summary.get("final_valid_count") == metrics["final"]["valid_count"]
        and summary.get("final_predictions_sha256") == sha256_file(final_path)
    )
    operational_invalid = sum(
        final_index[video_id].get("status") != "passed" for video_id in ids
    )
    integrity = {
        "observer_blind_count": blind,
        "mandatory_observation_count": mandatory,
        "complete_ledger_count": complete_ledgers,
        "citation_valid_count": valid_citations,
        "truncation_answer_count": _truncation_answer_count(root),
        "protocol_identity_valid": source_valid and summary_valid,
        "operational_invalid_count": operational_invalid,
    }
    usage = {
        "observed_cost_usd": summary.get("observed_cost_usd"),
        "formal_cost_usd": summary.get("formal_cost_usd"),
        "combined_reserved_cost_usd": summary.get(
            "combined_reserved_cost_usd", summary.get("formal_cost_usd")
        ),
        "cost_cap_usd": summary.get("max_cost_usd", MAX_COST_USD),
        "provider_request_count": summary.get("provider_request_count"),
        "recovery_call_count": summary.get("recovery_call_count"),
    }
    scored: dict[str, Any] = {
        "schema_version": 1,
        "status": "scored",
        "phase": phase,
        "source_sha256s": {
            "requests": sha256_file(requests_path),
            "labels": sha256_file(labels_path),
            "frozen_control": sha256_file(control_path),
            "final": sha256_file(final_path),
            "summary": sha256_file(summary_path),
            "manifest": sha256_file(root / "run-manifest.json"),
        },
        "metrics": metrics,
        "by_stratum": {
            "frozen_control": _by_stratum(control_rankings, labels, strata),
            "final": _by_stratum(final_rankings, labels, strata),
        },
        "transitions_control_to_final": transitions,
        "path_distribution": dict(
            sorted(
                Counter(
                    f"{final_index[video_id].get('observer_action_count')}_observations"
                    for video_id in ids
                ).items()
            )
        ),
        "integrity": integrity,
        "usage": usage,
        "rows": private_rows,
    }
    scored["decision_gate"] = (
        build_canary_gate(scored) if phase == "canary" else build_full_gate(scored)
    )
    return scored


def build_canary_gate(scored: Mapping[str, Any]) -> dict[str, Any]:
    final = scored["metrics"]["final"]
    transitions = scored["transitions_control_to_final"]
    integrity = scored["integrity"]
    usage = scored["usage"]
    checks = {
        "five_valid": final.get("row_count") == 5 and final.get("valid_count") == 5,
        "mandatory_observation": integrity.get("mandatory_observation_count") == 5,
        "observer_blind": integrity.get("observer_blind_count") == 5,
        "complete_ledgers": integrity.get("complete_ledger_count") == 5,
        "citations_valid": integrity.get("citation_valid_count") == 5,
        "no_truncation_as_answer": integrity.get("truncation_answer_count") == 0,
        "repairs_at_least_one": transitions.get("repaired", 0) >= 1,
        "within_cost_cap": (
            isinstance(usage.get("combined_reserved_cost_usd"), (int, float))
            and not isinstance(usage.get("combined_reserved_cost_usd"), bool)
            and float(usage["combined_reserved_cost_usd"]) <= MAX_COST_USD
        ),
        "protocol_frozen": integrity.get("protocol_identity_valid") is True,
        "no_operational_invalids": integrity.get("operational_invalid_count") == 0,
    }
    return {
        **checks,
        "continue_to_full": all(checks.values()),
        "source_sha256s": dict(scored.get("source_sha256s", {})),
        "canary_manifest_sha256": scored.get("source_sha256s", {}).get("manifest"),
        "canary_final_sha256": scored.get("source_sha256s", {}).get("final"),
    }


def build_full_gate(scored: Mapping[str, Any]) -> dict[str, Any]:
    final = scored["metrics"]["final"]
    control = scored["metrics"]["frozen_control"]
    transitions = scored["transitions_control_to_final"]
    integrity = scored["integrity"]
    usage = scored["usage"]
    checks = {
        "twenty_valid": final.get("row_count") == 20 and final.get("valid_count") == 20,
        "exceeds_frozen_control": (
            final.get("top1_correct", 0) > control.get("top1_correct", 0)
            and control.get("top1_correct") == 15
        ),
        "repairs_exceed_regressions": transitions.get("repaired", 0)
        > transitions.get("regressed", 0),
        "mandatory_observation": integrity.get("mandatory_observation_count") == 20,
        "observer_blind": integrity.get("observer_blind_count") == 20,
        "complete_ledgers": integrity.get("complete_ledger_count") == 20,
        "citations_valid": integrity.get("citation_valid_count") == 20,
        "no_truncation_as_answer": integrity.get("truncation_answer_count") == 0,
        "protocol_frozen": integrity.get("protocol_identity_valid") is True,
        "no_operational_invalids": integrity.get("operational_invalid_count") == 0,
        "within_cost_cap": (
            isinstance(usage.get("combined_reserved_cost_usd"), (int, float))
            and not isinstance(usage.get("combined_reserved_cost_usd"), bool)
            and float(usage["combined_reserved_cost_usd"]) <= MAX_COST_USD
        ),
    }
    return {**checks, "candidate_for_fresh_evaluation": all(checks.values())}


def sanitized_aggregate(scored: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "schema_version",
        "status",
        "phase",
        "source_sha256s",
        "metrics",
        "by_stratum",
        "transitions_control_to_final",
        "path_distribution",
        "integrity",
        "usage",
        "decision_gate",
    )
    return {key: scored[key] for key in allowed if key in scored}


def build_private_audit(scored: Mapping[str, Any]) -> dict[str, Any]:
    rows = scored.get("rows")
    if not isinstance(rows, list):
        raise ValueError("private scored rows are unavailable")
    return {
        "schema_version": 1,
        "status": "private_audit",
        "source_sha256s": dict(scored.get("source_sha256s", {})),
        "rows": list(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="score_vqa_mc_four_option_agent_poc20")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-canary", action="store_true")
    mode.add_argument("--score-canary", action="store_true")
    mode.add_argument("--score-full", action="store_true")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--sanitized-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.prepare_canary:
            result = prepare_canary(
                requests_path=args.requests,
                labels_path=args.labels,
                control_path=args.control,
                output_path=args.output,
            )
        else:
            if args.run_root is None:
                raise ValueError("score mode requires --run-root")
            phase = "canary" if args.score_canary else "full"
            result = score_phase(
                phase=phase,
                requests_path=args.requests,
                labels_path=args.labels,
                control_path=args.control,
                run_root=args.run_root,
            )
            _write_exclusive(args.output, result)
            if args.gate_output is not None:
                _write_exclusive(args.gate_output, result["decision_gate"])
            if args.audit_output is not None:
                _write_exclusive(args.audit_output, build_private_audit(result))
            if args.sanitized_output is not None:
                _write_exclusive(args.sanitized_output, sanitized_aggregate(result))
        print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))
        return 0
    except Exception:  # noqa: BLE001 - CLI must not leak private rows or paths
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
