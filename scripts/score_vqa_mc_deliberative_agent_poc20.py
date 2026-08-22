#!/usr/bin/env python3
"""Score frozen deliberative-agent MC rankings after inference is complete."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_experiments.vqa_mc_poc20 import (
    canonical_json,
    parse_ranking,
    score_rankings,
    sha256_file,
)


MAX_COST_USD = 0.75
OPTION_IDS = frozenset("ABCD")


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


def _prediction_index(
    path: Path, request_ids: Sequence[str], *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in _rows(path):
        if row.get("variant_kind") != "canonical":
            continue
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or video_id not in request_ids:
            raise ValueError(f"{label} prediction identity drift")
        if video_id in indexed:
            raise ValueError(f"duplicate {label} prediction row: {video_id}")
        indexed[video_id] = row
    if list(indexed) != list(request_ids):
        raise ValueError(f"{label} prediction identity drift")
    return indexed


def _valid_ranking(row: Mapping[str, Any] | None) -> list[str] | None:
    if (
        not isinstance(row, Mapping)
        or row.get("status") != "passed"
        or not isinstance(row.get("ranking"), list)
    ):
        return None
    try:
        return parse_ranking({"ranking": row["ranking"]})
    except ValueError:
        return None


def _record_ranking(record: object) -> list[str] | None:
    if not isinstance(record, Mapping) or not isinstance(record.get("ranking"), list):
        return None
    try:
        return parse_ranking({"ranking": record["ranking"]})
    except ValueError:
        return None


def _metric_by_stratum(
    predictions: Mapping[str, Sequence[str] | None],
    labels: Mapping[str, str],
    strata: Mapping[str, str],
) -> dict[str, dict[str, int | float]]:
    return {
        stratum: score_rankings(
            {
                video_id: predictions.get(video_id)
                for video_id in labels
                if strata[video_id] == stratum
            },
            {
                video_id: answer
                for video_id, answer in labels.items()
                if strata[video_id] == stratum
            },
        )
        for stratum in sorted(set(strata.values()))
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
        before_correct = (
            before.get(video_id) is not None and before[video_id][0] == correct
        )
        after_correct = (
            after.get(video_id) is not None and after[video_id][0] == correct
        )
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
    candidate = (root / relative).resolve(strict=True)
    candidate.relative_to(root)
    return candidate


def _observation_ids(notebook: Mapping[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    solvers = notebook.get("solvers")
    rounds = notebook.get("rounds")
    if isinstance(solvers, list):
        for solver in solvers:
            if not isinstance(solver, Mapping):
                continue
            for item in solver.get("observations", []):
                if isinstance(item, Mapping) and isinstance(
                    item.get("observation_id"), str
                ):
                    identifiers.add(str(item["observation_id"]))
    if isinstance(rounds, list):
        for entry in rounds:
            if not isinstance(entry, Mapping):
                continue
            observer = entry.get("observer")
            if not isinstance(observer, Mapping):
                continue
            for item in observer.get("observations", []):
                if isinstance(item, Mapping) and isinstance(
                    item.get("observation_id"), str
                ):
                    identifiers.add(str(item["observation_id"]))
    return identifiers


def _valid_action(action: object, provenance: object) -> bool:
    if not isinstance(action, Mapping) or not isinstance(provenance, Mapping):
        return False
    required_action = {
        "kind",
        "start_seconds",
        "end_seconds",
        "fps",
        "resolution",
    }
    if not required_action <= set(action):
        return False
    for key in ("start_seconds", "end_seconds", "fps", "resolution"):
        if provenance.get(key) != action.get(key):
            return False
    return isinstance(provenance.get("view_id"), str) and isinstance(
        provenance.get("estimated_input_tokens"), int
    )


def _verify_declared_sources(final: Mapping[str, Any], run_root: Path) -> int:
    paths = final.get("source_paths")
    hashes = final.get("source_sha256s")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("final source hash declarations are missing")
    if not paths or set(paths) != set(hashes):
        raise ValueError("final source hash declarations disagree")
    for key, relative in paths.items():
        if sha256_file(_safe_artifact(run_root, relative)) != hashes[key]:
            raise ValueError("source freeze hash mismatch")
    return len(paths)


def _load_notebooks(
    *, final_rows: Mapping[str, Mapping[str, Any]], run_root: Path
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, bool],
    dict[str, bool],
    dict[str, bool],
]:
    notebooks: dict[str, Mapping[str, Any]] = {}
    complete: dict[str, bool] = {}
    provenance_complete: dict[str, bool] = {}
    citation_valid: dict[str, bool] = {}
    for video_id, row in final_rows.items():
        digest = row.get("notebook_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("final notebook hash is missing")
        path = _safe_artifact(run_root, row.get("notebook_path"))
        if sha256_file(path) != digest:
            raise ValueError("final notebook hash mismatch")
        notebook = _read(path)
        notebooks[video_id] = notebook
        solvers = notebook.get("solvers")
        rounds = notebook.get("rounds")
        complete[video_id] = bool(
            notebook.get("video_id") == video_id
            and isinstance(solvers, list)
            and len(solvers) <= 2
            and isinstance(rounds, list)
            and len(rounds) <= 2
        )
        provenance_flags: list[bool] = []
        if isinstance(rounds, list):
            for entry in rounds:
                if isinstance(entry, Mapping) and isinstance(
                    entry.get("observer"), Mapping
                ):
                    provenance_flags.append(
                        _valid_action(entry.get("action"), entry.get("provenance"))
                    )
        provenance_complete[video_id] = all(provenance_flags)
        judge = row.get("judge")
        notebook_judge = notebook.get("judge")
        citations: object = None
        records_agree = False
        if isinstance(judge, Mapping):
            citations = judge.get("decisive_observation_ids")
            records_agree = bool(
                isinstance(notebook_judge, Mapping)
                and notebook_judge.get("record") == judge
            )
        if row.get("final_source") == "judge":
            citation_valid[video_id] = bool(
                records_agree
                and isinstance(citations, list)
                and citations
                and all(isinstance(item, str) for item in citations)
                and set(citations) <= _observation_ids(notebook)
            )
        else:
            citation_valid[video_id] = judge is None
    return notebooks, complete, provenance_complete, citation_valid


def _action_summary(
    ids: Sequence[str],
    notebooks: Mapping[str, Mapping[str, Any]],
    final_rows: Mapping[str, Mapping[str, Any]],
    durations: Mapping[str, float],
) -> dict[str, Any]:
    path_counts: Counter[str] = Counter(
        {"zero_actions": 0, "one_action": 0, "two_actions": 0}
    )
    configurations: Counter[str] = Counter()
    fps: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    stop_points: Counter[str] = Counter()
    interval_fractions: list[float] = []
    for video_id in ids:
        declared_count = final_rows[video_id].get("observer_action_count")
        if declared_count not in {0, 1, 2}:
            raise ValueError("observer action count is invalid")
        rounds = notebooks[video_id].get("rounds")
        entries = rounds if isinstance(rounds, list) else []
        actions = [
            entry["action"]
            for entry in entries
            if isinstance(entry, Mapping)
            and isinstance(entry.get("observer"), Mapping)
            and isinstance(entry.get("action"), Mapping)
        ]
        if len(actions) != declared_count:
            raise ValueError("observer action count disagrees with notebook")
        path_counts[("zero_actions", "one_action", "two_actions")[declared_count]] += 1
        for action in actions:
            kind = str(action["kind"])
            action_fps = float(action["fps"])
            resolution = str(action["resolution"])
            key = f"{kind}|{action_fps:g}fps|{resolution}"
            configurations[key] += 1
            fps[f"{action_fps:g}"] += 1
            resolutions[resolution] += 1
            kinds[kind] += 1
            fraction = (
                float(action["end_seconds"]) - float(action["start_seconds"])
            ) / durations[video_id]
            if not 0 < fraction <= 1.000001:
                raise ValueError("inspection duration fraction is invalid")
            interval_fractions.append(fraction)
        if not entries:
            stop_points["no_valid_planner"] += 1
        else:
            last = entries[-1]
            planner = last.get("planner") if isinstance(last, Mapping) else None
            tool = last.get("tool_result") if isinstance(last, Mapping) else None
            if isinstance(tool, Mapping) and tool.get("authorized") is False:
                stop_points["tool_denied"] += 1
            elif isinstance(planner, Mapping) and planner.get("decision") == "finish_search":
                stop_points[f"round_{last.get('round_index')}_finish"] += 1
            elif declared_count == 2:
                stop_points["two_action_limit"] += 1
            else:
                stop_points["observer_or_parser_stop"] += 1
    return {
        "path_distribution": dict(path_counts),
        "action_configurations": dict(sorted(configurations.items())),
        "distinct_action_configurations": len(configurations),
        "fps": dict(sorted(fps.items())),
        "resolution": dict(sorted(resolutions.items())),
        "kind": dict(sorted(kinds.items())),
        "stop_points": dict(sorted(stop_points.items())),
        "inspected_duration_fraction": {
            "count": len(interval_fractions),
            "mean": (
                round(sum(interval_fractions) / len(interval_fractions), 6)
                if interval_fractions
                else None
            ),
            "maximum": round(max(interval_fractions), 6) if interval_fractions else None,
        },
    }


def score(
    *,
    requests_path: Path,
    labels_path: Path,
    control_path: Path,
    adaptive_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    request_rows = _rows(requests_path)
    label_rows = _rows(labels_path)
    request_ids = [row.get("video_id") for row in request_rows]
    label_ids = [row.get("query_id") for row in label_rows]
    if (
        not all(isinstance(value, str) for value in request_ids)
        or request_ids != label_ids
        or len(set(request_ids)) != len(request_ids)
    ):
        raise ValueError("request/label identity drift")
    ids = [str(value) for value in request_ids]
    labels: dict[str, str] = {}
    for row in label_rows:
        answer = row.get("correct_option_id")
        if answer not in OPTION_IDS:
            raise ValueError("label answer is invalid")
        labels[str(row["query_id"])] = str(answer)
    strata = {str(row["video_id"]): str(row["stratum"]) for row in request_rows}
    durations = {
        str(row["video_id"]): float(row["duration_seconds"]) for row in request_rows
    }

    root = run_root.resolve(strict=True)
    final_path = root / "final-predictions-freeze.json"
    summary_path = root / "summary.json"
    anchor_path = root / "anchor/predictions-freeze.json"
    blind_path = root / "blind/predictions-freeze.json"
    final_artifact = _read(final_path)
    declared_source_count = _verify_declared_sources(final_artifact, root)
    control_index = _prediction_index(control_path, ids, label="control")
    adaptive_index = _prediction_index(adaptive_path, ids, label="adaptive")
    anchor_index = _prediction_index(anchor_path, ids, label="anchor")
    blind_index = _prediction_index(blind_path, ids, label="blind")
    final_index = _prediction_index(final_path, ids, label="final")
    notebooks, notebook_complete, provenance_complete, citation_valid = (
        _load_notebooks(final_rows=final_index, run_root=root)
    )

    last_planner: dict[str, list[str] | None] = {}
    final_judge: dict[str, list[str] | None] = {}
    for video_id in ids:
        rounds = notebooks[video_id].get("rounds")
        planner_records = [
            entry.get("planner")
            for entry in rounds if isinstance(entry, Mapping)
        ] if isinstance(rounds, list) else []
        last_planner[video_id] = next(
            (
                ranking
                for ranking in (
                    _record_ranking(record) for record in reversed(planner_records)
                )
                if ranking is not None
            ),
            None,
        )
        final_judge[video_id] = _record_ranking(final_index[video_id].get("judge"))

    systems = {
        "frozen_control": {
            video_id: _valid_ranking(control_index[video_id]) for video_id in ids
        },
        "adaptive_two_call": {
            video_id: _valid_ranking(adaptive_index[video_id]) for video_id in ids
        },
        "anchor": {
            video_id: _valid_ranking(anchor_index[video_id]) for video_id in ids
        },
        "blind": {
            video_id: _valid_ranking(blind_index[video_id]) for video_id in ids
        },
        "last_planner": last_planner,
        "final_judge": final_judge,
        "final": {
            video_id: _valid_ranking(final_index[video_id]) for video_id in ids
        },
    }
    metrics = {
        name: score_rankings(predictions, labels)
        for name, predictions in systems.items()
    }
    by_stratum = {
        name: _metric_by_stratum(predictions, labels, strata)
        for name, predictions in systems.items()
    }
    transitions_control = _transitions(
        systems["frozen_control"], systems["final"], labels
    )
    transitions_adaptive = _transitions(
        systems["adaptive_two_call"], systems["final"], labels
    )
    known_regression_ids = [
        video_id
        for video_id in ids
        if systems["frozen_control"][video_id] is not None
        and systems["frozen_control"][video_id][0] == labels[video_id]
        and (
            systems["adaptive_two_call"][video_id] is None
            or systems["adaptive_two_call"][video_id][0] != labels[video_id]
        )
    ]

    action_summary = _action_summary(ids, notebooks, final_index, durations)
    summary = _read(summary_path)
    notebook_complete_count = sum(notebook_complete.values())
    provenance_complete_count = sum(provenance_complete.values())
    citation_valid_count = sum(citation_valid.values())
    summary_consistent = bool(
        summary.get("status") == "completed"
        and summary.get("row_count") == len(ids)
        and summary.get("final_valid_count") == metrics["final"]["valid_count"]
        and summary.get("request_count")
        == sum(int(final_index[video_id].get("call_count", -1)) for video_id in ids)
    )
    integrity = {
        "declared_source_count": declared_source_count,
        "notebook_hash_valid_count": len(notebooks),
        "notebook_complete_count": notebook_complete_count,
        "provenance_complete_count": provenance_complete_count,
        "citation_valid_count": citation_valid_count,
        "summary_consistent": summary_consistent,
    }
    structural_integrity = bool(
        declared_source_count > 0
        and notebook_complete_count == len(ids)
        and provenance_complete_count == len(ids)
        and citation_valid_count == len(ids)
        and summary_consistent
    )
    observed_cost = summary.get("observed_cost_usd")
    combined_observed_cost = summary.get(
        "combined_observed_cost_usd", observed_cost
    )
    within_cost_cap = bool(
        isinstance(combined_observed_cost, (int, float))
        and not isinstance(combined_observed_cost, bool)
        and float(combined_observed_cost) <= MAX_COST_USD
    )
    known_hits_preserved = all(
        systems["final"][video_id] is not None
        and systems["final"][video_id][0] == labels[video_id]
        for video_id in known_regression_ids
    )
    decision_gate = {
        "requires_20_valid": metrics["final"]["valid_count"] == len(ids),
        "final_exceeds_control": (
            metrics["final"]["top1_correct"]
            > metrics["frozen_control"]["top1_correct"]
        ),
        "repairs_exceed_regressions": (
            transitions_control["repaired"] > transitions_control["regressed"]
        ),
        "known_regression_hits_preserved": known_hits_preserved,
        "structural_integrity": structural_integrity,
        "adaptive_paths": bool(
            action_summary["path_distribution"]["zero_actions"] >= 1
            and action_summary["distinct_action_configurations"] >= 2
        ),
        "within_cost_cap": within_cost_cap,
    }
    decision_gate["promising_development_result"] = all(decision_gate.values())

    final_source_counts = Counter(
        str(final_index[video_id].get("final_source")) for video_id in ids
    )
    ambiguity_flag_count = sum(
        bool(
            isinstance(final_index[video_id].get("judge"), Mapping)
            and final_index[video_id]["judge"].get("ambiguity_present") is True
        )
        for video_id in ids
    )
    departures_from_both = 0
    departures_with_new_evidence = 0
    for video_id in ids:
        final_ranking = systems["final"][video_id]
        anchor_ranking = systems["anchor"][video_id]
        blind_ranking = systems["blind"][video_id]
        judge = final_index[video_id].get("judge")
        if (
            final_ranking is not None
            and anchor_ranking is not None
            and blind_ranking is not None
            and final_ranking[0] not in {anchor_ranking[0], blind_ranking[0]}
        ):
            departures_from_both += 1
            citations = (
                judge.get("decisive_observation_ids", [])
                if isinstance(judge, Mapping)
                else []
            )
            if any(str(item).startswith("round-") for item in citations):
                departures_with_new_evidence += 1

    private_rows = [
        {
            "query_id": video_id,
            "stratum": strata[video_id],
            "correct_option_id": labels[video_id],
            "frozen_control_ranking": systems["frozen_control"][video_id],
            "adaptive_two_call_ranking": systems["adaptive_two_call"][video_id],
            "anchor_ranking": systems["anchor"][video_id],
            "blind_ranking": systems["blind"][video_id],
            "last_planner_ranking": systems["last_planner"][video_id],
            "final_ranking": systems["final"][video_id],
            "final_source": final_index[video_id].get("final_source"),
            "observer_action_count": final_index[video_id].get(
                "observer_action_count"
            ),
            "call_count": final_index[video_id].get("call_count"),
            "ambiguity_present": bool(
                isinstance(final_index[video_id].get("judge"), Mapping)
                and final_index[video_id]["judge"].get("ambiguity_present") is True
            ),
            "citation_valid": citation_valid[video_id],
            "notebook_sha256": final_index[video_id].get("notebook_sha256"),
        }
        for video_id in ids
    ]
    return {
        "schema_version": 1,
        "status": "scored",
        "source_sha256s": {
            "requests": sha256_file(requests_path),
            "labels": sha256_file(labels_path),
            "frozen_control": sha256_file(control_path),
            "adaptive_two_call": sha256_file(adaptive_path),
            "deliberative_final": sha256_file(final_path),
            "deliberative_summary": sha256_file(summary_path),
        },
        "metrics": metrics,
        "by_stratum": by_stratum,
        "transitions_control_to_final": transitions_control,
        "transitions_adaptive_to_final": transitions_adaptive,
        "known_prior_regression_count": len(known_regression_ids),
        "path_distribution": action_summary.pop("path_distribution"),
        "action_distribution": action_summary,
        "final_source_counts": dict(sorted(final_source_counts.items())),
        "ambiguity_flag_count": ambiguity_flag_count,
        "judge_departures_from_both_solvers": departures_from_both,
        "departures_citing_new_evidence": departures_with_new_evidence,
        "usage": {
            "request_count": summary.get("request_count"),
            "prompt_tokens": summary.get("prompt_tokens"),
            "output_tokens": summary.get("output_tokens"),
            "usage_complete": summary.get("usage_complete"),
            "observed_cost_usd": observed_cost,
            "combined_observed_cost_usd": combined_observed_cost,
            "cost_cap_usd": MAX_COST_USD,
        },
        "integrity": integrity,
        "decision_gate": decision_gate,
        "rows": private_rows,
    }


def sanitized_aggregate(result: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "schema_version",
        "status",
        "source_sha256s",
        "metrics",
        "by_stratum",
        "transitions_control_to_final",
        "transitions_adaptive_to_final",
        "known_prior_regression_count",
        "path_distribution",
        "action_distribution",
        "final_source_counts",
        "ambiguity_flag_count",
        "judge_departures_from_both_solvers",
        "departures_citing_new_evidence",
        "usage",
        "integrity",
        "decision_gate",
    )
    return {key: result[key] for key in allowed}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="score_vqa_mc_deliberative_agent_poc20")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = score(
        requests_path=args.requests,
        labels_path=args.labels,
        control_path=args.control,
        adaptive_path=args.adaptive,
        run_root=args.run_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical_json(result))
    print(json.dumps(sanitized_aggregate(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
