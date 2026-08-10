"""Offline diagnostics and quarantined adapters for the NIST VQA tools."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import shutil
import string
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


LOCAL_DIAGNOSTICS_MODE = "local-diagnostics"
NIST_2025_PARITY_MODE = "nist-2025-parity"
NIST_2026_VALIDATE_AG_MODE = "nist-2026-validate-ag"
SUPPORTED_NIST_MODES = frozenset({NIST_2025_PARITY_MODE, NIST_2026_VALIDATE_AG_MODE})

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SCORER_OUTPUT_NAMES = frozenset({"evaluation_results.csv"})
_SCORER_OUTPUT_SUFFIXES = ("_diagnostic.csv", "_per_response.csv")
_SHELL_ACTIVE_CHARACTERS = frozenset({'"', "`", "$", "\\", "\r", "\n"})


@dataclass(frozen=True)
class NistInvocation:
    """One prepared scorer subprocess with an isolated submission directory."""

    mode: str
    command: tuple[str, ...]
    cwd: Path
    run_dir: Path
    report_path: Path | None
    uses_shell: bool = False


@dataclass(frozen=True)
class NistExecution:
    """Captured, mode-labeled result from an isolated NIST subprocess."""

    mode: str
    returncode: int
    stdout: str
    stderr: str


def _normalize_answer(answer: str) -> str:
    lowered = answer.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def normalized_exact_match(prediction: str, reference: str) -> float:
    """Return SQuAD-style normalized exact match as zero or one."""

    return float(_normalize_answer(prediction) == _normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Return bag-of-token F1 after the same deterministic normalization."""

    prediction_tokens = _normalize_answer(prediction).split()
    reference_tokens = _normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


_SEMANTIC_FIELDS = ("meteor", "bertscore_f1", "sts")


def _finite_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def semantic_composite(metrics: Mapping[str, float]) -> float:
    """Return the analysis-only mean of NIST's three semantic metrics."""

    values = []
    for field in _SEMANTIC_FIELDS:
        if field not in metrics:
            raise ValueError(f"Missing semantic field: {field}")
        values.append(_finite_float(metrics[field], field=field))
    return fmean(values)


def _validate_ndcg_cutoff(k: int | None, size: int) -> int:
    if k is None:
        return size
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= size:
        raise ValueError(f"NDCG cutoff must be an integer between 1 and {size}")
    return k


def _dcg_from_gains(gains: Sequence[float]) -> float:
    return sum(
        (math.pow(2.0, gain) - 1.0) / math.log2(index + 2)
        for index, gain in enumerate(gains)
    )


def ndcg_from_gains(gains: Sequence[float], k: int | None = None) -> float:
    """Return NIST 2025's order-sensitive NDCG for an answer ranking."""

    checked_gains = [
        _finite_float(gain, field="gain") for gain in gains
    ]
    if not checked_gains:
        if k is not None:
            raise ValueError("NDCG cutoff cannot be used with an empty ranking")
        return 0.0
    cutoff = _validate_ndcg_cutoff(k, len(checked_gains))
    if len(checked_gains) == 1:
        return 1.0
    actual = _dcg_from_gains(checked_gains[:cutoff])
    ideal = _dcg_from_gains(sorted(checked_gains, reverse=True)[:cutoff])
    return actual / ideal if ideal else 0.0


def ranked_cutoff_metrics(
    rows: Sequence[Mapping[str, float]], cutoffs: Sequence[int] = (1, 5, 10)
) -> dict[str, Any]:
    """Summarize semantic gains at NIST's ranked answer cutoffs.

    Rows may provide a precomputed ``gain`` or ``semantic_composite``. If
    neither is present, the three declared semantic fields are averaged.
    """

    if not rows:
        raise ValueError("Ranked rows must be non-empty")
    gains = []
    for row in rows:
        if "gain" in row:
            gains.append(_finite_float(row["gain"], field="gain"))
        elif "semantic_composite" in row:
            gains.append(
                _finite_float(row["semantic_composite"], field="semantic_composite")
            )
        else:
            gains.append(semantic_composite(row))

    checked_cutoffs = []
    for cutoff in cutoffs:
        if cutoff in checked_cutoffs:
            raise ValueError(f"Duplicate ranked cutoff: {cutoff}")
        checked_cutoffs.append(_validate_ndcg_cutoff(cutoff, len(gains)))

    result: dict[str, Any] = {"rank_1": gains[0]}
    for cutoff in checked_cutoffs:
        prefix = gains[:cutoff]
        result[f"best@{cutoff}"] = max(prefix)
        result[f"mean@{cutoff}"] = fmean(prefix)
        result[f"ndcg@{cutoff}"] = ndcg_from_gains(gains, cutoff)
    return result


def evaluate_local_answer(
    prediction: str,
    reference: str,
    *,
    semantic_metrics: Mapping[str, float | None] | None = None,
) -> dict[str, str | float | None]:
    """Compute offline diagnostics and preserve explicit missing metric values."""

    semantic_metrics = semantic_metrics or {}
    return {
        "mode": LOCAL_DIAGNOSTICS_MODE,
        "normalized_exact_match": normalized_exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "sts": semantic_metrics.get("sts"),
        "meteor": semantic_metrics.get("meteor"),
        "bertscore_f1": semantic_metrics.get("bertscore_f1"),
        # The historical scorer's NDCG is meaningless for a single answer.
        "ndcg": None,
    }


_PRIMARY_METRICS = ("normalized_exact_match", "token_f1", "meteor")


def evaluate_saved_predictions(
    predictions_path: Path,
    *,
    references: Mapping[str, str],
    strata: Mapping[str, str],
) -> dict[str, Any]:
    """Regenerate answer scores by joining saved label-free predictions to labels.

    Reference answers and strata are deliberately separate inputs. If a saved
    runner request contains any evaluation-only field, regeneration stops
    instead of silently accepting a leakage-contaminated artifact.
    """

    raw_predictions = _load_saved_predictions(predictions_path)
    selected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for prediction in raw_predictions:
        if not isinstance(prediction, Mapping):
            raise ValueError("Each saved prediction must be a JSON object")
        condition_id = _required_prediction_text(prediction, "condition_id")
        video_id = _required_prediction_text(prediction, "video_id")
        request = prediction.get("request")
        if request is not None:
            if not isinstance(request, Mapping):
                raise ValueError("Saved runner request must be a JSON object")
            _validate_label_free_request(
                request,
                condition_id=condition_id,
                video_id=video_id,
            )
        identity = (condition_id, video_id)
        current = selected.get(identity)
        if (
            current is None
            or _is_valid_success(prediction)
            or not _is_valid_success(current)
        ):
            selected[identity] = prediction
    predictions = list(selected.values())

    examples: list[dict[str, Any]] = []
    for prediction in predictions:
        condition_id = _required_prediction_text(prediction, "condition_id")
        video_id = _required_prediction_text(prediction, "video_id")
        if video_id not in references:
            raise ValueError(f"Missing reference for video: {video_id}")
        if video_id not in strata:
            raise ValueError(f"Missing stratum for video: {video_id}")
        answer = prediction.get("answer")
        completed = _is_valid_success(prediction)
        reference = references[video_id]
        if completed:
            raw_semantic = prediction.get("semantic_metrics", {})
            if not isinstance(raw_semantic, Mapping):
                raise ValueError("semantic_metrics must be a JSON object")
            semantic = {
                name: raw_semantic.get(name, prediction.get(name))
                for name in ("sts", "meteor", "bertscore_f1")
            }
            metrics = evaluate_local_answer(
                answer, reference, semantic_metrics=semantic
            )
        else:
            metrics = evaluate_local_answer("", reference)
            for name in _PRIMARY_METRICS:
                metrics[name] = None
        examples.append(
            {
                **dict(prediction),
                "condition_id": condition_id,
                "video_id": video_id,
                "stratum": strata[video_id],
                "reference": reference,
                "completed": completed,
                **{name: metrics[name] for name in _PRIMARY_METRICS},
            }
        )

    conditions = {
        condition_id: _summarize_examples(
            [item for item in examples if item["condition_id"] == condition_id]
        )
        for condition_id in sorted({item["condition_id"] for item in examples})
    }
    overall = _summarize_examples(examples)
    return {
        "examples": examples,
        "aggregate": overall["aggregate"],
        "strata": overall["strata"],
        "conditions": conditions,
    }


def _load_saved_predictions(predictions_path: Path) -> list[Any]:
    try:
        text = predictions_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("Saved predictions are missing or invalid JSON") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise ValueError("Saved predictions are missing or invalid JSON") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            "Saved predictions must be a non-empty JSON list or JSONL file"
        )
    return payload


def summarize_paired_conditions(
    scored_examples: Sequence[Mapping[str, Any]],
    *,
    baseline_condition_id: str,
    treatment_condition_id: str,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize treatment-minus-baseline changes on aligned examples."""

    if baseline_condition_id == treatment_condition_id:
        raise ValueError("Baseline and treatment condition IDs must be different")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for example in scored_examples:
        key = (str(example.get("condition_id", "")), str(example.get("video_id", "")))
        if key in indexed:
            raise ValueError(f"Duplicate scored example: {key[0]}/{key[1]}")
        indexed[key] = example
    video_ids = sorted(
        video_id
        for condition_id, video_id in indexed
        if condition_id == baseline_condition_id
        and (treatment_condition_id, video_id) in indexed
    )
    pairs = [
        (
            indexed[(baseline_condition_id, video_id)],
            indexed[(treatment_condition_id, video_id)],
        )
        for video_id in video_ids
        if indexed[(baseline_condition_id, video_id)].get("completed")
        and indexed[(treatment_condition_id, video_id)].get("completed")
    ]
    if not pairs:
        raise ValueError("Paired conditions have no aligned completed examples")
    if any(left.get("stratum") != right.get("stratum") for left, right in pairs):
        raise ValueError("Paired examples must have matching strata")

    summary = _paired_group(
        pairs,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    summary.update(
        {
            "baseline_condition_id": baseline_condition_id,
            "treatment_condition_id": treatment_condition_id,
            "strata": {
                stratum: _paired_group(
                    [pair for pair in pairs if pair[0]["stratum"] == stratum],
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=bootstrap_seed,
                    confidence=confidence,
                )
                for stratum in sorted({str(pair[0]["stratum"]) for pair in pairs})
            },
        }
    )
    return summary


def _required_prediction_text(prediction: Mapping[str, Any], key: str) -> str:
    value = prediction.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Saved prediction {key} must be a non-empty string")
    return value.strip()


def _is_valid_success(prediction: Mapping[str, Any]) -> bool:
    answer = prediction.get("answer")
    return (
        prediction.get("status") == "passed"
        and isinstance(answer, str)
        and bool(answer.strip())
    )


def _validate_label_free_request(
    request: Mapping[str, Any], *, condition_id: str, video_id: str
) -> None:
    from trec_video_2026.vqa_gate import RunnerRequest

    canonical_fields = frozenset(field.name for field in fields(RunnerRequest))
    unknown = sorted(str(key) for key in request if key not in canonical_fields)
    if unknown:
        raise ValueError(
            f"Runner request contains unrecognized fields: {', '.join(unknown)}"
        )
    for key, value in request.items():
        _reject_nested_request_mapping(value, path=str(key))
    for field, expected in (("condition_id", condition_id), ("video_id", video_id)):
        if field in request and request[field] != expected:
            raise ValueError(f"Attempt/request identity mismatch for {field}")

    legacy_fields = frozenset({"run_id", "condition_id", "video_id", "question"})
    request_fields = frozenset(request)
    if request_fields <= legacy_fields:
        _validate_legacy_runner_request(request)
        return
    missing = sorted(canonical_fields - {"audio_mode"} - request_fields)
    if missing:
        raise ValueError(
            "Invalid canonical RunnerRequest; missing fields: " + ", ".join(missing)
        )
    _validate_canonical_runner_request(request, RunnerRequest)


def _validate_legacy_runner_request(request: Mapping[str, Any]) -> None:
    for field in ("run_id", "condition_id", "video_id", "question"):
        if field in request and not isinstance(request[field], str):
            raise ValueError(f"Invalid legacy RunnerRequest field type: {field}")
    for field in ("run_id", "condition_id", "video_id"):
        if field in request and not request[field]:
            raise ValueError(f"Invalid legacy RunnerRequest field value: {field}")
    if "question" in request and not request["question"].strip():
        raise ValueError("Invalid legacy RunnerRequest field value: question")


def _validate_canonical_runner_request(
    request: Mapping[str, Any], request_type
) -> None:
    try:
        for field in ("run_id", "condition_id", "video_id", "question"):
            if not isinstance(request[field], str):
                raise TypeError(f"{field} must be a string")
        frame_paths = request["frame_paths"]
        frame_timestamps = request["frame_timestamps"]
        frame_descriptions = request["frame_descriptions"]
        if not isinstance(frame_paths, list) or any(
            not isinstance(value, str) for value in frame_paths
        ):
            raise TypeError("frame_paths must be a list of strings")
        if not isinstance(frame_timestamps, list) or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in frame_timestamps
        ):
            raise TypeError("frame_timestamps must be a list of numbers")
        if frame_descriptions is not None and (
            not isinstance(frame_descriptions, list)
            or any(not isinstance(value, str) for value in frame_descriptions)
        ):
            raise TypeError("frame_descriptions must be null or a list of strings")
        for field in ("transcript", "media_path", "audio_path"):
            if request[field] is not None and not isinstance(request[field], str):
                raise TypeError(f"{field} must be null or a string")
        audio_mode = request.get("audio_mode")
        if audio_mode is not None and not isinstance(audio_mode, str):
            raise TypeError("audio_mode must be null or a string")
        request_type(
            run_id=request["run_id"],
            condition_id=request["condition_id"],
            video_id=request["video_id"],
            question=request["question"],
            frame_paths=tuple(Path(value) for value in frame_paths),
            frame_timestamps=tuple(frame_timestamps),
            frame_descriptions=(
                tuple(frame_descriptions) if frame_descriptions is not None else None
            ),
            transcript=request["transcript"],
            media_path=(
                Path(request["media_path"])
                if request["media_path"] is not None
                else None
            ),
            audio_path=(
                Path(request["audio_path"])
                if request["audio_path"] is not None
                else None
            ),
            audio_mode=audio_mode,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid canonical RunnerRequest: {error}") from error


def _reject_nested_request_mapping(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        raise ValueError(f"Runner request contains a nested mapping at {path}")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_nested_request_mapping(item, path=f"{path}[{index}]")


def _summarize_examples(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "aggregate": _aggregate(examples),
        "strata": {
            stratum: _aggregate(
                [item for item in examples if str(item["stratum"]) == stratum]
            )
            for stratum in sorted({str(item["stratum"]) for item in examples})
        },
    }


def _aggregate(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed_count = sum(bool(item.get("completed")) for item in examples)
    result: dict[str, Any] = {
        "example_count": len(examples),
        "completed_count": completed_count,
        "completion_rate": completed_count / len(examples) if examples else 0.0,
    }
    for metric in _PRIMARY_METRICS:
        values = [
            float(item[metric]) for item in examples if item.get(metric) is not None
        ]
        result[metric] = fmean(values) if values else None
    return result


def _paired_group(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    deltas = {
        metric: [
            float(treatment[metric]) - float(baseline[metric])
            for baseline, treatment in pairs
            if baseline.get(metric) is not None and treatment.get(metric) is not None
        ]
        for metric in _PRIMARY_METRICS
    }
    return {
        "paired_count": len(pairs),
        "exact_match_wins": sum(
            baseline["normalized_exact_match"] == 0.0
            and treatment["normalized_exact_match"] == 1.0
            for baseline, treatment in pairs
        ),
        "exact_match_losses": sum(
            baseline["normalized_exact_match"] == 1.0
            and treatment["normalized_exact_match"] == 0.0
            for baseline, treatment in pairs
        ),
        "metric_deltas": {
            metric: fmean(values) if values else None
            for metric, values in deltas.items()
        },
        "bootstrap_intervals": {
            metric: _bootstrap_mean_interval(
                values,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
                confidence=confidence,
            )
            if values
            else None
            for metric, values in deltas.items()
        },
    }


def _bootstrap_mean_interval(
    values: Sequence[float], *, iterations: int, seed: int, confidence: float
) -> dict[str, Any]:
    random_source = random.Random(seed)
    sample_size = len(values)
    means = sorted(
        fmean(random_source.choices(values, k=sample_size)) for _ in range(iterations)
    )
    tail = (1.0 - confidence) / 2.0
    lower_index = min(iterations - 1, int(tail * iterations))
    upper_index = min(iterations - 1, max(0, int((1.0 - tail) * iterations)))
    return {
        "lower": means[lower_index],
        "upper": means[upper_index],
        "confidence": confidence,
        "iterations": iterations,
        "seed": seed,
        "exploratory": True,
    }


def nist_style_semantic_metrics(
    predictions: list[str],
    references: list[str],
    *,
    word_tokenize,
    meteor_score,
    bert_score,
) -> list[dict[str, float | None]]:
    """Reproduce NIST's local METEOR/BERTScore calls without unsafe STS I/O.

    This intentionally omits the historical scorer's plaintext-HTTP STS call
    and its meaningless single-response NDCG value.
    """

    if not predictions or len(predictions) != len(references):
        raise ValueError("Predictions and references must be non-empty and aligned")
    normalized_predictions = [_nist_normalize(value) for value in predictions]
    normalized_references = [_nist_normalize(value) for value in references]
    meteor_values = [
        float(meteor_score([word_tokenize(reference)], word_tokenize(prediction)))
        for prediction, reference in zip(
            normalized_predictions, normalized_references, strict=True
        )
    ]
    _, _, bert_f1 = bert_score(normalized_predictions, normalized_references, lang="en")
    bert_values = [float(value) for value in bert_f1.tolist()]
    return [
        {
            "meteor": meteor,
            "bertscore_f1": bert,
            "sts": None,
            "ndcg": None,
        }
        for meteor, bert in zip(meteor_values, bert_values, strict=True)
    ]


def _nist_normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def discover_submission_csvs(directory: Path) -> list[Path]:
    """Find candidate CSVs without rediscovering historical scorer outputs."""

    if not directory.is_dir():
        raise ValueError(f"CSV directory does not exist: {directory}")
    candidates = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        if _is_scorer_output(path.name):
            continue
        candidates.append(path)
    return sorted(candidates)


def prepare_nist_invocation(
    mode: str,
    *,
    scorer_dir: Path,
    references_path: Path,
    run_path: Path,
    isolated_dir: Path,
) -> NistInvocation:
    """Stage one run and build an argv-only invocation for a pinned NIST checkout."""

    if mode not in SUPPORTED_NIST_MODES:
        raise ValueError(f"Unsupported NIST mode: {mode}")
    scorer_dir = scorer_dir.resolve()
    references_path = references_path.resolve()
    run_path = run_path.resolve()
    isolated_dir = isolated_dir.resolve()
    _require_file(references_path, "references")
    _require_file(run_path, "run")
    if not scorer_dir.is_dir():
        raise ValueError(f"Scorer directory does not exist: {scorer_dir}")
    if isolated_dir.exists() and any(isolated_dir.iterdir()):
        raise ValueError(f"Isolation directory must be empty: {isolated_dir}")
    isolated_dir.mkdir(parents=True, exist_ok=True)

    if mode == NIST_2025_PARITY_MODE:
        if _is_scorer_output(run_path.name):
            raise ValueError(f"Submission looks like scorer output: {run_path.name}")
        _reject_shell_active_answers(references_path, "reference")
        _reject_shell_active_answers(run_path, "submission")
        script = scorer_dir / "eval.vqa.ag.py"
        _require_file(script, "NIST 2025 evaluator")
        run_dir = isolated_dir / "runs"
        run_dir.mkdir()
        staged_run = run_dir / run_path.name
        shutil.copyfile(run_path, staged_run)
        return NistInvocation(
            mode=mode,
            command=(
                sys.executable,
                str(script),
                str(references_path),
                str(run_dir),
            ),
            cwd=scorer_dir,
            run_dir=run_dir,
            report_path=None,
        )

    script = scorer_dir / "validate.ag.run.py"
    _require_file(script, "NIST 2026 validator")
    run_dir = isolated_dir / "run"
    run_dir.mkdir()
    staged_run = run_dir / run_path.name
    shutil.copyfile(run_path, staged_run)
    report_path = isolated_dir / "validation-report.json"
    return NistInvocation(
        mode=mode,
        command=(
            sys.executable,
            str(script),
            str(references_path),
            str(staged_run),
            "--report-json",
            str(report_path),
        ),
        cwd=scorer_dir,
        run_dir=run_dir,
        report_path=report_path,
    )


def execute_nist_invocation(
    invocation: NistInvocation, *, timeout_seconds: float
) -> NistExecution:
    """Execute prepared argv with a credential-free environment and no shell."""

    if invocation.uses_shell:
        raise ValueError("NIST invocations must never use a shell")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    sandbox_home = invocation.run_dir.parent / "home"
    sandbox_home.mkdir(exist_ok=True)
    allowed_environment = {
        name: value
        for name in (
            "PATH",
            "LANG",
            "LC_ALL",
            "TZ",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "REQUESTS_CA_BUNDLE",
        )
        if (value := os.environ.get(name)) is not None
    }
    allowed_environment["HOME"] = str(sandbox_home)
    completed = subprocess.run(
        list(invocation.command),
        cwd=invocation.cwd,
        env=allowed_environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
        shell=False,
    )
    return NistExecution(
        mode=invocation.mode,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"{description.capitalize()} file does not exist: {path}")


def _is_scorer_output(filename: str) -> bool:
    return filename in _SCORER_OUTPUT_NAMES or filename.endswith(
        _SCORER_OUTPUT_SUFFIXES
    )


def _reject_shell_active_answers(path: Path, description: str) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) < 4:
                raise ValueError(
                    f"Malformed {description} CSV row {row_number}: expected four fields"
                )
            answer = row[3]
            if any(character in answer for character in _SHELL_ACTIVE_CHARACTERS):
                raise ValueError(
                    f"Answer text on {description} row {row_number} is unsafe for "
                    "the upstream scorer"
                )
