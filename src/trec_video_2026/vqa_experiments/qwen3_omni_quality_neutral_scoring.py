"""Post-hoc semantic diagnostic for the strict-failed neutral-prompt A2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from . import qwen3_omni_quality_scoring as common
from .qwen3_omni_quality_dstack import _condition_passed


ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "qwen3-omni-quality-10-neutral-20260725-a2"
PREDICTIONS_SHA256 = "48f411538ba4e7118a08d9da5d4917f08bbf06f9694211e355c4d80ad678be08"
INSTRUCTION_PROFILE = "neutral"
INSTRUCTION_SHA256 = "44b83ed11a2b9b8a62584302b126fb8eec13b03c55acb57d08b146f41c5e09a6"
RECOVERY_VIDEO_ID = "NkXPPJoZ-h8"
RECOVERY_CONDITION_ID = "frames-native-audio"
RECOVERY_PAIR_INDEX = 3
RECOVERY_RAW_OUTPUT_SHA256 = (
    "0dbb246367b2abdd7e1810d39597acf2cab9326c453e54db3299d07dd719872a"
)
A1_SCORE_ARTIFACT_SHA256 = (
    "a54e8773f8550f8eb30b43131292d201bd1a80a0f06a017cfa44e0ea83634084"
)
SCORER_PATH = (
    "src/trec_video_2026/vqa_experiments/qwen3_omni_quality_neutral_scoring.py"
)
DEFAULT_RUN_DIRECTORY = (
    ROOT / "data/vqa/benchmarks/qwen3-omni-quality-10" / RUN_ID
)
DEFAULT_CONTRACT_PATH = common.DEFAULT_CONTRACT_PATH
DEFAULT_A1_SCORE_PATH = common.DEFAULT_ARTIFACT_PATH
DEFAULT_ARTIFACT_PATH = (
    ROOT
    / "docs/experiments/artifacts/qwen3-omni-quality-10-neutral-a2-score.json"
)

MANUAL_JUDGMENTS: dict[tuple[str, str], dict[str, str]] = {
    ("Mi-NpnCBegw", "frames-only"): {
        "judgment": "C",
        "reason": "Matches the demonstrated final synonym.",
    },
    ("Mi-NpnCBegw", "frames-native-audio"): {
        "judgment": "C",
        "reason": "Matches the demonstrated final synonym.",
    },
    ("NN7G28sBOlY", "frames-only"): {
        "judgment": "W",
        "reason": "Answers a different proposition instead of the requested reply.",
    },
    ("NN7G28sBOlY", "frames-native-audio"): {
        "judgment": "W",
        "reason": "Answers a different proposition instead of the requested reply.",
    },
    ("NUmb2P3Bzu8", "frames-only"): {
        "judgment": "P",
        "reason": "Captures visible intensity but omits the central vocal escalation.",
    },
    ("NUmb2P3Bzu8", "frames-native-audio"): {
        "judgment": "C",
        "reason": "Identifies both vocal escalation and emphatic visible behavior.",
    },
    ("NkXPPJoZ-h8", "frames-only"): {
        "judgment": "P",
        "reason": "Identifies the animal vocalization but not its specified form.",
    },
    ("NkXPPJoZ-h8", "frames-native-audio"): {
        "judgment": "C",
        "reason": "Identifies the source and the scream-like character of the sound.",
    },
    ("IWSPAfCKtV8", "frames-only"): {
        "judgment": "W",
        "reason": "Does not identify the requested end sound.",
    },
    ("IWSPAfCKtV8", "frames-native-audio"): {
        "judgment": "W",
        "reason": "Names a different end sound.",
    },
    ("CH8Dp7EgW4I", "frames-only"): {
        "judgment": "W",
        "reason": "Describes an intervening look rather than the requested action.",
    },
    ("CH8Dp7EgW4I", "frames-native-audio"): {
        "judgment": "P",
        "reason": "Describes dance-like body movement without identifying the dance.",
    },
    ("EtNOkUJd5w8", "frames-only"): {
        "judgment": "W",
        "reason": "Gives an unrelated preceding event.",
    },
    ("EtNOkUJd5w8", "frames-native-audio"): {
        "judgment": "W",
        "reason": "Gives an unrelated preceding event.",
    },
    ("Gp4LVTn3YbY", "frames-only"): {
        "judgment": "P",
        "reason": "Captures crunching and sourness but omits the verbal response.",
    },
    ("Gp4LVTn3YbY", "frames-native-audio"): {
        "judgment": "P",
        "reason": "Captures crunching but omits the sour verbal reaction.",
    },
    ("EMI_anpHII4", "frames-only"): {
        "judgment": "C",
        "reason": "Identifies an ornament type consistent with the target.",
    },
    ("EMI_anpHII4", "frames-native-audio"): {
        "judgment": "C",
        "reason": "Identifies an ornament type consistent with the target.",
    },
    ("LusRurRrmf0", "frames-only"): {
        "judgment": "C",
        "reason": "Numeric answer is semantically equivalent to the target.",
    },
    ("LusRurRrmf0", "frames-native-audio"): {
        "judgment": "C",
        "reason": "Numeric answer is semantically equivalent to the target.",
    },
}

_TERMINAL_FENCE = re.compile(r"(?:^|\n)```json\n(?P<payload>\{.*\})\n```\Z", re.DOTALL)


def _parse_terminal_fenced_json(raw_output: str) -> dict[str, str]:
    if raw_output.count("```json\n") != 1:
        raise ValueError("post-hoc recovery requires one terminal fenced JSON object")
    match = _TERMINAL_FENCE.search(raw_output)
    if match is None:
        raise ValueError("post-hoc recovery requires one terminal fenced JSON object")
    try:
        payload = json.loads(match.group("payload"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "post-hoc recovery requires one terminal fenced JSON object"
        ) from error
    if not (
        isinstance(payload, dict)
        and set(payload) == {"answer", "evidence"}
        and all(isinstance(payload[key], str) and payload[key].strip() for key in payload)
    ):
        raise ValueError("post-hoc recovery requires one terminal fenced JSON object")
    return payload


def _recover_pinned_failure(row: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    raw_output = row.get("raw_output")
    generation = (
        row.get("modality_attestation", {}).get("generation", {})
        if isinstance(row.get("modality_attestation"), Mapping)
        else {}
    )
    raw_hash = (
        hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        if isinstance(raw_output, str)
        else None
    )
    identity_is_exact = (
        row.get("run_id") == RUN_ID
        and row.get("video_id") == RECOVERY_VIDEO_ID
        and row.get("condition_id") == RECOVERY_CONDITION_ID
        and row.get("audio_enabled") is True
        and row.get("pair_index") == RECOVERY_PAIR_INDEX
        and row.get("instruction_profile") == INSTRUCTION_PROFILE
        and row.get("instruction_sha256") == INSTRUCTION_SHA256
        and row.get("status") == "failed"
        and row.get("output_parse_status") == "failed"
        and row.get("failure_stage") == "parsing"
        and row.get("error_type") == "output_parse_error"
        and row.get("input_contract_error") is False
        and generation.get("ended_with_eos") is True
        and generation.get("raw_output_sha256") == RECOVERY_RAW_OUTPUT_SHA256
        and raw_hash == RECOVERY_RAW_OUTPUT_SHA256
    )
    attestation = row.get("modality_attestation")
    if not (
        identity_is_exact
        and isinstance(attestation, Mapping)
        and attestation.get("same_request") is True
        and attestation.get("same_forward_pass") is True
        and attestation.get("generation_call_count") == 1
        and attestation.get("vision_element_count") == 12
        and attestation.get("audio_element_count") == 1
    ):
        raise ValueError("row does not match the pinned A2 recovery boundary")
    parsed = _parse_terminal_fenced_json(raw_output)
    recovery = {
        "policy": "pinned-terminal-fenced-json",
        "raw_output_sha256": RECOVERY_RAW_OUTPUT_SHA256,
        "recovered_json_sha256": common.canonical_sha256(parsed),
        "recovered_answer_sha256": hashlib.sha256(
            parsed["answer"].encode("utf-8")
        ).hexdigest(),
    }
    return parsed, recovery


def _passed_row_is_strict(row: Mapping[str, Any]) -> bool:
    parsed = row.get("parsed_output")
    return (
        isinstance(row, dict)
        and _condition_passed(row)
        and row.get("run_id") == RUN_ID
        and row.get("instruction_profile") == INSTRUCTION_PROFILE
        and row.get("instruction_sha256") == INSTRUCTION_SHA256
        and isinstance(parsed, Mapping)
        and isinstance(parsed.get("answer"), str)
    )


def _require_frozen_a2(
    run_dir: Path, contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions_path = run_dir / "predictions.jsonl"
    summary = common._load_json(run_dir / "summary.json")
    manifest = common._load_json(run_dir / "manifest.json")
    terminal = common._load_json(run_dir / "terminal-status.json")
    dstack = common._load_json(run_dir / "dstack-run.json")
    prediction_hash = common._sha256_file(predictions_path)
    contract_hash = common.canonical_sha256(contract)
    if not (
        prediction_hash == PREDICTIONS_SHA256
        and contract_hash == common.QUALITY_CONTRACT_SHA256
        and summary.get("run_id") == RUN_ID
        and summary.get("rows") == 20
        and summary.get("succeeded") is False
        and summary.get("canary_passed") is True
        and summary.get("predictions_sha256") == PREDICTIONS_SHA256
        and summary.get("quality_contract_sha256") == contract_hash
        and summary.get("instruction_profile") == INSTRUCTION_PROFILE
        and summary.get("instruction_sha256") == INSTRUCTION_SHA256
        and manifest.get("run_id") == RUN_ID
        and manifest.get("inputs", {}).get("quality_contract_sha256") == contract_hash
        and manifest.get("inputs", {}).get("instruction_profile")
        == INSTRUCTION_PROFILE
        and manifest.get("inputs", {}).get("instruction_sha256")
        == INSTRUCTION_SHA256
        and manifest.get("outputs", {}).get("predictions.jsonl")
        == PREDICTIONS_SHA256
        and terminal.get("workload_returncode") == 1
        and dstack.get("run_id") == RUN_ID
        and dstack.get("status") == "failed"
        and dstack.get("exit_status") == 1
        and dstack.get("cost") == 1.7253
        and dstack.get("elapsed_seconds") == 1414.836509
    ):
        raise ValueError("artifacts do not match the frozen strict-failed A2 run")
    rows = common._load_predictions(predictions_path)
    expected = common._expected_pairs(contract)
    observed = [
        (
            row.get("video_id"),
            row.get("condition_id"),
            row.get("audio_enabled"),
            row.get("pair_index"),
        )
        for row in rows
    ]
    if observed != expected:
        raise ValueError("A2 predictions are not complete adjacent contract pairs")
    passed = [row for row in rows if row.get("status") == "passed"]
    failed = [row for row in rows if row.get("status") == "failed"]
    for row in passed:
        if not _passed_row_is_strict(row):
            raise ValueError("A2 passed row violates the frozen condition contract")
    if len(passed) != 19 or len(failed) != 1:
        raise ValueError("A2 must contain exactly 19 strict passes and one failure")
    recovered, recovery = _recover_pinned_failure(failed[0])
    failed[0]["diagnostic_parsed_output"] = recovered
    failed[0]["diagnostic_recovery"] = recovery
    return rows, recovery


def _score_rows(
    rows: list[dict[str, Any]], data_root: Path
) -> list[dict[str, Any]]:
    expected_keys = {
        (row["video_id"], row["condition_id"])
        for row in rows
    }
    if set(MANUAL_JUDGMENTS) != expected_keys:
        raise ValueError("every A2 diagnostic row requires one manual judgment")
    scored: list[dict[str, Any]] = []
    for row in rows:
        parsed = (
            row["parsed_output"]
            if row["status"] == "passed"
            else row["diagnostic_parsed_output"]
        )
        video_id = row["video_id"]
        annotation_path, tracked_path = common._annotation_path(data_root, video_id)
        annotation = common._load_json(annotation_path)
        reference = annotation.get("correct_answer")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"annotation lacks a usable answer for {video_id}")
        answer = parsed["answer"]
        manual = MANUAL_JUDGMENTS[(video_id, row["condition_id"])]
        scored.append(
            {
                "video_id": video_id,
                "operational_tag": common.PANEL_TAGS[video_id],
                "pair_index": row["pair_index"],
                "condition_id": row["condition_id"],
                "audio_enabled": row["audio_enabled"],
                "answer": answer,
                "annotation_path": tracked_path,
                "reference_sha256": hashlib.sha256(
                    reference.encode("utf-8")
                ).hexdigest(),
                "judgment": manual["judgment"],
                "judgment_reason": manual["reason"],
                "normalized_exact_match": common.normalized_exact_match(
                    answer, reference
                ),
                "token_f1": common.token_f1(answer, reference),
                "diagnostic_recovery": row.get("diagnostic_recovery"),
            }
        )
    return scored


def _transition_counts(
    earlier: list[Mapping[str, Any]], later: list[Mapping[str, Any]]
) -> dict[str, int]:
    transitions = Counter(
        f"{left['judgment']}->{right['judgment']}"
        for left, right in zip(earlier, later, strict=True)
    )
    return dict(sorted(transitions.items()))


def _aggregate_delta(
    current: Mapping[str, float | int], prior: Mapping[str, float | int]
) -> dict[str, float | int]:
    return {
        "correct_delta": int(current["correct"]) - int(prior["correct"]),
        "partial_delta": int(current["partial"]) - int(prior["partial"]),
        "wrong_delta": int(current["wrong"]) - int(prior["wrong"]),
        "normalized_exact_match_delta": (
            float(current["normalized_exact_match"])
            - float(prior["normalized_exact_match"])
        ),
        "token_f1_delta": float(current["token_f1"]) - float(prior["token_f1"]),
    }


def score_neutral_diagnostic(
    *,
    data_root: Path,
    run_dir: Path,
    contract_path: Path,
    a1_score_path: Path,
    scorer_source_path: Path,
) -> dict[str, Any]:
    contract = common._load_json(contract_path)
    rows, recovery = _require_frozen_a2(run_dir, contract)
    scored = _score_rows(rows, data_root)
    frames = [row for row in scored if row["condition_id"] == "frames-only"]
    audio = [
        row for row in scored if row["condition_id"] == "frames-native-audio"
    ]
    aggregate = {
        "frames-only": common._aggregate(frames),
        "frames-native-audio": common._aggregate(audio),
        "all_conditions": common._aggregate(scored),
    }
    a1 = common._load_json(a1_score_path)
    if not (
        common._sha256_file(a1_score_path) == A1_SCORE_ARTIFACT_SHA256
        and a1.get("run_id") == common.RUN_ID
        and a1.get("predictions_sha256") == common.PREDICTIONS_SHA256
        and a1.get("quality_contract_sha256") == common.QUALITY_CONTRACT_SHA256
    ):
        raise ValueError("A1 comparison artifact does not match its frozen identity")
    a1_rows = a1["examples"]
    a1_frames = [
        row for row in a1_rows if row["condition_id"] == "frames-only"
    ]
    a1_audio = [
        row for row in a1_rows if row["condition_id"] == "frames-native-audio"
    ]
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "predictions_sha256": PREDICTIONS_SHA256,
        "quality_contract_sha256": common.QUALITY_CONTRACT_SHA256,
        "instruction_profile": INSTRUCTION_PROFILE,
        "instruction_sha256": INSTRUCTION_SHA256,
        "strict_result": {
            "eligible": False,
            "run_succeeded": False,
            "strict_passed_rows": 19,
            "strict_failed_rows": 1,
            "failed_video_id": RECOVERY_VIDEO_ID,
            "failed_condition_id": RECOVERY_CONDITION_ID,
            "failure_type": "output_parse_error",
        },
        "post_hoc_semantic_diagnostic": {
            "eligible_for_strict_gate": False,
            "recovered_rows": 1,
            "recovery": recovery,
            "semantic_conditions_met": True,
            "qualifying_audio_improvements": [
                "NUmb2P3Bzu8",
                "NkXPPJoZ-h8",
                "CH8Dp7EgW4I",
            ],
            "visual_control_regressions": [],
        },
        "runtime": {
            "elapsed_seconds": 1414.836509,
            "cost_usd": 1.7253,
            "cumulative_qwen_cost_usd": 6.3131,
            "next_full_run_worst_case_usd": 10.8131,
            "next_full_run_within_budget": False,
        },
        "reference_hash_contract": (
            "SHA-256 of exact UTF-8 correct_answer text without a trailing newline"
        ),
        "scorer": {
            "source": SCORER_PATH,
            "source_sha256": common._sha256_file(scorer_source_path),
            "primary_metrics": ["normalized_exact_match", "token_f1"],
            "primary_metric_contract": a1["scorer"]["primary_metric_contract"],
        },
        "manual_adjudication": {
            "rubric": "answer-only C/P/W",
            "blinded": False,
            "reviewer_count": 1,
        },
        "examples": scored,
        "aggregate": aggregate,
        "paired_deltas": {
            "audio_minus_frames": {
                "normalized_exact_match": fmean(
                    right["normalized_exact_match"]
                    - left["normalized_exact_match"]
                    for left, right in zip(frames, audio, strict=True)
                ),
                "token_f1": fmean(
                    right["token_f1"] - left["token_f1"]
                    for left, right in zip(frames, audio, strict=True)
                ),
                "judgment_transitions": _transition_counts(frames, audio),
            }
        },
        "comparison_to_a1": {
            "a1_run_id": common.RUN_ID,
            "a1_score_artifact_sha256": A1_SCORE_ARTIFACT_SHA256,
            "frames-only": _aggregate_delta(
                aggregate["frames-only"], a1["aggregate"]["frames-only"]
            ),
            "frames-native-audio": _aggregate_delta(
                aggregate["frames-native-audio"],
                a1["aggregate"]["frames-native-audio"],
            ),
            "all_conditions": _aggregate_delta(
                aggregate["all_conditions"], a1["aggregate"]["all_conditions"]
            ),
            "judgment_transitions": {
                "frames-only": _transition_counts(a1_frames, frames),
                "frames-native-audio": _transition_counts(a1_audio, audio),
            },
        },
    }


def _default_score() -> dict[str, Any]:
    return score_neutral_diagnostic(
        data_root=ROOT / "data",
        run_dir=DEFAULT_RUN_DIRECTORY,
        contract_path=DEFAULT_CONTRACT_PATH,
        a1_score_path=DEFAULT_A1_SCORE_PATH,
        scorer_source_path=Path(__file__),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args()
    result = _default_score()
    if args.verify:
        artifact = common._load_json(args.output)
        if common.canonical_sha256(artifact) != common.canonical_sha256(result):
            raise ValueError("tracked A2 diagnostic does not reproduce")
        print("Qwen3-Omni neutral A2 post-hoc diagnostic reproduces exactly.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote sanitized A2 diagnostic: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
