"""Score the frozen Qwen3-Omni quality-10 run without publishing labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import string
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "qwen3-omni-quality-10-20260725-a1"
PREDICTIONS_SHA256 = "9804197489450f1a31836ff1b80c773261c3917bea350243a5c374c9d52c89a8"
QUALITY_CONTRACT_SHA256 = "df016767b1a072770149fdc06fa47dfe7f17930b6c3145af4e92f55d58f4ddeb"
ANNOTATION_DIRECTORY = Path("vqa/2025/training/annotations")
SCORER_PATH = "src/trec_video_2026/vqa_experiments/qwen3_omni_quality_scoring.py"
DEFAULT_RUN_DIRECTORY = (
    ROOT / "data/vqa/benchmarks/qwen3-omni-quality-10" / RUN_ID
)
DEFAULT_CONTRACT_PATH = ROOT / "data/vqa/benchmarks/qwen3-omni-quality-10/contract.json"
DEFAULT_ARTIFACT_PATH = (
    ROOT / "docs/experiments/artifacts/qwen3-omni-quality-10-score.json"
)

PANEL_TAGS = {
    "Mi-NpnCBegw": "lexical_speech",
    "NN7G28sBOlY": "lexical_speech_prompt_repair",
    "NUmb2P3Bzu8": "prosody",
    "NkXPPJoZ-h8": "non_speech_sound",
    "IWSPAfCKtV8": "endpoint_sound",
    "CH8Dp7EgW4I": "temporal_action",
    "EtNOkUJd5w8": "temporal_action",
    "Gp4LVTn3YbY": "cross_modal_reaction",
    "EMI_anpHII4": "spatial_visual_control",
    "LusRurRrmf0": "counting_visual_control",
}

# Reasons deliberately describe the judged answer relationship without storing a
# correct-answer string from the ignored annotations.
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
        "judgment": "C",
        "reason": "Identifies the central vocal and gestural intensity cues.",
    },
    ("NUmb2P3Bzu8", "frames-native-audio"): {
        "judgment": "C",
        "reason": "Identifies the central vocal and gestural intensity cues.",
    },
    ("NkXPPJoZ-h8", "frames-only"): {
        "judgment": "P",
        "reason": "Identifies the animal vocalization but not its specified form.",
    },
    ("NkXPPJoZ-h8", "frames-native-audio"): {
        "judgment": "P",
        "reason": "Identifies the animal vocalization but not its specified form.",
    },
    ("IWSPAfCKtV8", "frames-only"): {
        "judgment": "W",
        "reason": "Names a different end sound.",
    },
    ("IWSPAfCKtV8", "frames-native-audio"): {
        "judgment": "W",
        "reason": "Names a different end sound.",
    },
    ("CH8Dp7EgW4I", "frames-only"): {
        "judgment": "W",
        "reason": "Describes an intervening response rather than the requested action.",
    },
    ("CH8Dp7EgW4I", "frames-native-audio"): {
        "judgment": "W",
        "reason": "Describes an intervening response rather than the requested action.",
    },
    ("EtNOkUJd5w8", "frames-only"): {
        "judgment": "W",
        "reason": "Gives an unrelated preceding event.",
    },
    ("EtNOkUJd5w8", "frames-native-audio"): {
        "judgment": "W",
        "reason": "Rejects the premise and gives an unrelated sequence.",
    },
    ("Gp4LVTn3YbY", "frames-only"): {
        "judgment": "C",
        "reason": "Includes the requested reaction sequence and verbal response.",
    },
    ("Gp4LVTn3YbY", "frames-native-audio"): {
        "judgment": "P",
        "reason": "Captures chewing and sourness but omits the verbal response.",
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
        "reason": "Numeric answer is equivalent under the scorer rule.",
    },
    ("LusRurRrmf0", "frames-native-audio"): {
        "judgment": "W",
        "reason": "Numeric count conflicts with the target.",
    },
}

_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def canonical_sha256(value: object) -> str:
    """Return the canonical JSON SHA-256 used for frozen JSON identities."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_answer(answer: str) -> str:
    """Match the frozen repository scorer's punctuation/article normalization."""
    lowered = answer.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    tokens = [
        token
        for token in without_punctuation.split()
        if token not in {"a", "an", "the"}
    ]
    return " ".join(tokens)


def _normalize_numeric_equivalent(answer: str) -> str:
    return " ".join(
        _NUMBER_WORDS.get(token, token) for token in normalize_answer(answer).split()
    )


def normalized_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def _token_f1(prediction_tokens: list[str], reference_tokens: list[str]) -> float:
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, reference: str) -> float:
    return _token_f1(
        normalize_answer(prediction).split(), normalize_answer(reference).split()
    )


def numeric_equivalent_exact_match(prediction: str, reference: str) -> float:
    """Return a secondary digit-word-equivalent exact-match diagnostic."""
    return float(
        _normalize_numeric_equivalent(prediction)
        == _normalize_numeric_equivalent(reference)
    )


def numeric_equivalent_token_f1(prediction: str, reference: str) -> float:
    """Return a secondary digit-word-equivalent token-F1 diagnostic."""
    return _token_f1(
        _normalize_numeric_equivalent(prediction).split(),
        _normalize_numeric_equivalent(reference).split(),
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read frozen artifact {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"frozen artifact {path} must be a JSON object")
    return payload


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read frozen predictions {path}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("frozen predictions must contain JSON object records")
    return rows


def _require_complete_frozen_run(
    *,
    run_dir: Path,
    contract: Mapping[str, Any],
    expected_run_id: str,
    expected_predictions_sha256: str,
    expected_contract_sha256: str,
) -> list[dict[str, Any]]:
    predictions_path = run_dir / "predictions.jsonl"
    summary = _load_json(run_dir / "summary.json")
    manifest = _load_json(run_dir / "manifest.json")
    terminal_status = _load_json(run_dir / "terminal-status.json")
    prediction_hash = _sha256_file(predictions_path)
    contract_hash = canonical_sha256(contract)
    if prediction_hash != expected_predictions_sha256:
        raise ValueError("predictions hash does not match the frozen identity")
    if contract_hash != expected_contract_sha256:
        raise ValueError("contract hash does not match the frozen identity")
    if not (
        summary.get("run_id") == expected_run_id
        and summary.get("succeeded") is True
        and summary.get("canary_passed") is True
        and summary.get("predictions_sha256") == prediction_hash
        and summary.get("quality_contract_sha256") == contract_hash
        and manifest.get("run_id") == expected_run_id
        and manifest.get("inputs", {}).get("quality_contract_sha256") == contract_hash
        and manifest.get("outputs", {}).get("predictions.jsonl") == prediction_hash
        and terminal_status.get("workload_returncode") == 0
    ):
        raise ValueError("artifacts do not attest a complete frozen run")
    rows = _load_predictions(predictions_path)
    if summary.get("rows") != len(rows):
        raise ValueError("frozen summary row count does not match predictions")
    return rows


def _expected_pairs(contract: Mapping[str, Any]) -> list[tuple[str, str, bool, int]]:
    examples = contract.get("examples")
    conditions = contract.get("conditions")
    if not isinstance(examples, list) or not isinstance(conditions, list):
        raise ValueError("contract must contain examples and conditions lists")
    expected: list[tuple[str, str, bool, int]] = []
    for pair_index, example in enumerate(examples):
        video_id = example.get("video_id") if isinstance(example, dict) else None
        if not isinstance(video_id, str) or video_id not in PANEL_TAGS:
            raise ValueError("contract contains an unknown panel video")
        for condition in conditions:
            if not isinstance(condition, dict):
                raise ValueError("contract condition must be an object")
            condition_id = condition.get("condition_id")
            audio_enabled = condition.get("audio_enabled")
            if not isinstance(condition_id, str) or not isinstance(audio_enabled, bool):
                raise ValueError("contract condition is malformed")
            expected.append((video_id, condition_id, audio_enabled, pair_index))
    if len(expected) != len(set((video_id, condition_id) for video_id, condition_id, _, _ in expected)):
        raise ValueError("contract contains duplicate paired conditions")
    return expected


def _validate_prediction_pairs(
    rows: list[dict[str, Any]],
    expected: list[tuple[str, str, bool, int]],
    expected_run_id: str,
) -> None:
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
        raise ValueError("predictions are not complete adjacent contract pairs")
    for row in rows:
        parsed_output = row.get("parsed_output")
        if not (
            row.get("run_id") == expected_run_id
            and row.get("status") == "passed"
            and row.get("output_parse_status") == "passed"
            and isinstance(parsed_output, dict)
            and isinstance(parsed_output.get("answer"), str)
        ):
            raise ValueError("prediction does not meet the completed inference contract")


def _validate_manual_judgments(
    manual_judgments: Mapping[tuple[str, str], Mapping[str, str]],
    expected: list[tuple[str, str, bool, int]],
) -> None:
    expected_keys = {(video_id, condition_id) for video_id, condition_id, _, _ in expected}
    if set(manual_judgments) != expected_keys:
        raise ValueError("every frozen prediction requires one manual judgment")
    for key, judgment in manual_judgments.items():
        if not (
            isinstance(judgment, Mapping)
            and judgment.get("judgment") in {"C", "P", "W"}
            and isinstance(judgment.get("reason"), str)
            and judgment["reason"].strip()
        ):
            raise ValueError(f"manual judgment for {key!r} is malformed")


def _annotation_path(data_root: Path, video_id: str) -> tuple[Path, str]:
    relative = ANNOTATION_DIRECTORY / f"{video_id}.json"
    path = (data_root / relative).resolve()
    root = data_root.resolve()
    if root not in path.parents:
        raise ValueError("annotation path escapes data root")
    return path, str(Path("data") / relative)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "correct": sum(row["judgment"] == "C" for row in rows),
        "partial": sum(row["judgment"] == "P" for row in rows),
        "wrong": sum(row["judgment"] == "W" for row in rows),
        "normalized_exact_match": fmean(row["normalized_exact_match"] for row in rows),
        "token_f1": fmean(row["token_f1"] for row in rows),
    }


def score_run(
    *,
    data_root: Path,
    run_dir: Path,
    contract_path: Path,
    expected_run_id: str,
    expected_predictions_sha256: str,
    expected_contract_sha256: str,
    manual_judgments: Mapping[tuple[str, str], Mapping[str, str]],
    scorer_source_path: Path,
) -> dict[str, Any]:
    """Join a complete frozen run to annotations and return a label-free score."""
    contract = _load_json(contract_path)
    rows = _require_complete_frozen_run(
        run_dir=run_dir,
        contract=contract,
        expected_run_id=expected_run_id,
        expected_predictions_sha256=expected_predictions_sha256,
        expected_contract_sha256=expected_contract_sha256,
    )
    expected = _expected_pairs(contract)
    _validate_prediction_pairs(rows, expected, expected_run_id)
    _validate_manual_judgments(manual_judgments, expected)

    scored_rows: list[dict[str, Any]] = []
    for prediction in rows:
        video_id = prediction["video_id"]
        condition_id = prediction["condition_id"]
        annotation_path, tracked_annotation_path = _annotation_path(data_root, video_id)
        annotation = _load_json(annotation_path)
        reference = annotation.get("correct_answer")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"annotation lacks a usable answer for {video_id}")
        answer = prediction["parsed_output"]["answer"]
        manual = manual_judgments[(video_id, condition_id)]
        scored_rows.append(
            {
                "video_id": video_id,
                "operational_tag": PANEL_TAGS[video_id],
                "pair_index": prediction["pair_index"],
                "condition_id": condition_id,
                "audio_enabled": prediction["audio_enabled"],
                "answer": answer,
                "annotation_path": tracked_annotation_path,
                "reference_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
                "judgment": manual["judgment"],
                "judgment_reason": manual["reason"],
                "normalized_exact_match": normalized_exact_match(answer, reference),
                "token_f1": token_f1(answer, reference),
                "secondary_numeric_equivalent_exact_match": (
                    numeric_equivalent_exact_match(answer, reference)
                ),
                "secondary_numeric_equivalent_token_f1": (
                    numeric_equivalent_token_f1(answer, reference)
                ),
            }
        )

    grouped = {
        condition_id: [row for row in scored_rows if row["condition_id"] == condition_id]
        for _, condition_id, _, _ in expected
    }
    frames = grouped["frames-only"]
    audio = grouped["frames-native-audio"]
    result = {
        "schema_version": 1,
        "run_id": expected_run_id,
        "predictions_sha256": expected_predictions_sha256,
        "quality_contract_sha256": expected_contract_sha256,
        "reference_hash_contract": "SHA-256 of exact UTF-8 correct_answer text without a trailing newline",
        "scorer": {
            "source": SCORER_PATH,
            "source_sha256": _sha256_file(scorer_source_path),
            "primary_metrics": ["normalized_exact_match", "token_f1"],
            "primary_metric_contract": {
                "source": "src/trec_video_2026/vqa_evaluation.py",
                "source_commit": "44d59446bcb26df292fb2094aadaf5b24b0b0208",
                "source_sha256": (
                    "0cdd46df8347f3ba6e0986c9da349f6d89b41aa2ecfb15f637570a15f7480712"
                ),
            },
            "secondary_diagnostics": {
                "fields": [
                    "secondary_numeric_equivalent_exact_match",
                    "secondary_numeric_equivalent_token_f1",
                ],
                "numeric_equivalence": {"1": "one"},
            },
        },
        "manual_adjudication": {
            "rubric": "answer-only C/P/W",
            "blinded": False,
            "reviewer_count": 1,
        },
        "examples": scored_rows,
        "aggregate": {
            "frames-only": _aggregate(frames),
            "frames-native-audio": _aggregate(audio),
            "all_conditions": _aggregate(scored_rows),
        },
        "paired_deltas": {
            "audio_minus_frames": {
                "normalized_exact_match": fmean(
                    audio_row["normalized_exact_match"] - frames_row["normalized_exact_match"]
                    for frames_row, audio_row in zip(frames, audio, strict=True)
                ),
                "token_f1": fmean(
                    audio_row["token_f1"] - frames_row["token_f1"]
                    for frames_row, audio_row in zip(frames, audio, strict=True)
                ),
                "judgment_transitions": {
                    f"{frames_row['judgment']}->{audio_row['judgment']}": sum(
                        1
                        for earlier, later in zip(frames, audio, strict=True)
                        if earlier["judgment"] == frames_row["judgment"]
                        and later["judgment"] == audio_row["judgment"]
                    )
                    for frames_row, audio_row in zip(frames, audio, strict=True)
                },
            }
        },
    }
    return result


def _default_score() -> dict[str, Any]:
    return score_run(
        data_root=ROOT / "data",
        run_dir=DEFAULT_RUN_DIRECTORY,
        contract_path=DEFAULT_CONTRACT_PATH,
        expected_run_id=RUN_ID,
        expected_predictions_sha256=PREDICTIONS_SHA256,
        expected_contract_sha256=QUALITY_CONTRACT_SHA256,
        manual_judgments=MANUAL_JUDGMENTS,
        scorer_source_path=Path(__file__),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args()
    result = _default_score()
    if args.verify:
        artifact = _load_json(args.output)
        if canonical_sha256(artifact) != canonical_sha256(result):
            raise ValueError("tracked score artifact does not reproduce from frozen inputs")
        print("Qwen3-Omni quality-10 score reproduces exactly.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote sanitized score artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
