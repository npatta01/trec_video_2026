"""Offline diagnostics and quarantined adapters for the NIST VQA tools."""

from __future__ import annotations

import csv
import os
import re
import shutil
import string
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


LOCAL_DIAGNOSTICS_MODE = "local-diagnostics"
NIST_2025_PARITY_MODE = "nist-2025-parity"
NIST_2026_VALIDATE_AG_MODE = "nist-2026-validate-ag"
SUPPORTED_NIST_MODES = frozenset(
    {NIST_2025_PARITY_MODE, NIST_2026_VALIDATE_AG_MODE}
)

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
        float(
            meteor_score(
                [word_tokenize(reference)], word_tokenize(prediction)
            )
        )
        for prediction, reference in zip(
            normalized_predictions, normalized_references, strict=True
        )
    ]
    _, _, bert_f1 = bert_score(
        normalized_predictions, normalized_references, lang="en"
    )
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
