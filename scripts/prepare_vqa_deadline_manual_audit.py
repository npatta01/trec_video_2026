"""Prepare a private five-row VQA deadline inspection packet.

This command is deliberately a post-freeze boundary.  It reads only the two
label-free prediction freezes and panel/media metadata until both raw freezes
pass their model-specific shape checks.  Reference JSON is opened in one helper,
called only after the private media packet has been assembled.

The command is a local diagnostic and packet builder, not an official TREC
submission or scorer.  Official AG validation remains a separate invocation of
``validate.ag.run.py`` from the NIST VQA repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from trec_video_2026.vqa_deadline_canary import (
    load_panel,
    validate_ranked_submission,
)


DEFAULT_PANEL = Path("configs/vqa/deadline_canary_5.json")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_REFERENCE_ROOT = Path("data/vqa/2025/training/annotations")
DEFAULT_OUTPUT_ROOT = Path(
    "data/vqa/benchmarks/deadline-submission-canary-5-20260816/manual-audit"
)
_READY_DECISIONS = frozenset(
    {"ready", "ready_for_scoring", "quality_ready", "success"}
)
_PAYLOAD_KEYS = ("predictions", "submission", "rows", "records", "results")
_PREDICTION_FILENAMES = (
    "submission.json",
    "submission.jsonl",
    "predictions.json",
    "predictions.jsonl",
    "answers.json",
    "answers.jsonl",
    "predictions-freeze.json",
)
_LABEL_FIELDS = frozenset(
    {
        "answer",
        "correct_answer",
        "reference",
        "ground_truth",
        "groundtruth",
        "annotation",
        "annotations",
    }
)
_QWEN_CONDITIONS = {
    "video-plus-transcript": "qwen_transcript",
    "native-video": "qwen_no_transcript",
}


class FreezeValidationError(ValueError):
    """Raised when a prediction freeze cannot be used for the audit."""


@dataclass(frozen=True)
class FreezeResult:
    """A verified, label-free raw freeze and its immutable source hash."""

    path: Path
    sha256: str
    rows: tuple[dict[str, Any], ...]
    answers: dict[str, dict[str, str]]
    status: str | None
    decision: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    body = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _read_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise FreezeValidationError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
        return records
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FreezeValidationError(f"invalid JSON freeze: {path}") from error


def _resolve_freeze_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FreezeValidationError(f"prediction freeze is missing: {path}")
    for name in _PREDICTION_FILENAMES:
        candidate = path / name
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise FreezeValidationError(
        f"no prediction freeze found under directory: {path}"
    )


def _extract_rows(payload: Any, *, source: Path) -> tuple[list[dict[str, Any]], str | None, str | None]:
    status: str | None = None
    decision: str | None = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        artifact_hash = payload.get("_artifact_sha256")
        if artifact_hash is not None:
            body = dict(payload)
            body.pop("_artifact_sha256", None)
            if artifact_hash != _canonical_sha256(body):
                raise FreezeValidationError("prediction freeze artifact hash mismatch")
        raw_status = payload.get("status")
        raw_decision = payload.get("decision")
        if raw_status is not None and not isinstance(raw_status, str):
            raise FreezeValidationError("freeze status must be a string")
        if raw_decision is not None and not isinstance(raw_decision, str):
            raise FreezeValidationError("freeze decision must be a string")
        status = raw_status
        decision = raw_decision
        if status is not None and status not in {
            "predictions_frozen",
            "raw_frozen",
            "frozen",
            "complete",
        }:
            raise FreezeValidationError(f"freeze status is not complete: {status}")
        if decision is not None and decision not in _READY_DECISIONS:
            raise FreezeValidationError(f"freeze decision is not ready: {decision}")
        rows = None
        for key in _PAYLOAD_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise FreezeValidationError(
                f"freeze has no rows (expected one of {_PAYLOAD_KEYS}): {source}"
            )
    else:
        raise FreezeValidationError("prediction freeze must be a JSON object or array")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise FreezeValidationError("prediction freeze rows must be objects")
    if isinstance(payload, Mapping):
        row_count = payload.get("row_count")
        if row_count is not None and row_count != len(rows):
            raise FreezeValidationError("prediction freeze row_count does not match rows")
    return [dict(row) for row in rows], status, decision


def _normalise_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Accept official keys and the equivalent lower-case runner keys."""

    if "Q_ID" in row:
        result = dict(row)
    else:
        q_id = row.get("q_id", row.get("query_id"))
        video_id = row.get("video_id", row.get("Video_ID"))
        raw_answers = row.get("answers", row.get("ranked_answers"))
        if q_id is None or video_id is None or raw_answers is None:
            return dict(row)
        if not isinstance(raw_answers, list):
            return dict(row)
        answers = []
        for index, answer in enumerate(raw_answers, start=1):
            if not isinstance(answer, Mapping):
                return dict(row)
            answers.append(
                {
                    "Rank": answer.get("Rank", answer.get("rank", index)),
                    "Answer": answer.get("Answer", answer.get("text", answer.get("answer"))),
                    "Time": answer.get("Time", answer.get("time", answer.get("elapsed", 0.0))),
                }
            )
        result = {"Q_ID": q_id, "Video_ID": video_id, "Answers": answers}
        if "question" in row:
            result["Question"] = row["question"]
    return result


def _text_answer(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("answer", "text", "response", "value"):
            answer = value.get(key)
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
    return None


def _condition_answer(row: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    for key in aliases:
        if key in row:
            answer = _text_answer(row[key])
            if answer is not None:
                return answer
    for container_key in ("answers", "conditions", "answer_conditions"):
        container = row.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key in aliases:
            if key in container:
                answer = _text_answer(container[key])
                if answer is not None:
                    return answer
    return None


def _raw_answers(row: Mapping[str, Any], *, model: str) -> tuple[str, dict[str, str]]:
    video_id = row.get("video_id", row.get("Video_ID"))
    if not isinstance(video_id, str) or not video_id.strip():
        raise FreezeValidationError("raw prediction row has no video_id")
    forbidden = _LABEL_FIELDS - {"answer"}
    if forbidden.intersection(row):
        raise FreezeValidationError("prediction freeze contains a label field")
    if model == "gemini":
        answer = _condition_answer(row, ("answer", "gemini_answer", "rank_1"))
        if answer is None:
            raise FreezeValidationError("Gemini raw row has no non-empty answer")
        return video_id, {"gemini": answer}
    if model != "qwen":
        raise ValueError(f"unknown raw freeze model: {model}")
    condition_id = row.get("condition_id")
    answer = _text_answer(row.get("answer"))
    if condition_id not in _QWEN_CONDITIONS:
        raise FreezeValidationError(
            "Qwen raw row has an unexpected condition_id"
        )
    if row.get("status") != "passed" or answer is None:
        raise FreezeValidationError(
            "Qwen condition must be passed with a non-empty answer"
        )
    return video_id, {_QWEN_CONDITIONS[condition_id]: answer}


def validate_prediction_freeze(
    path: Path,
    *,
    expected_video_ids: set[str],
    model: str,
) -> FreezeResult:
    """Validate one raw freeze without opening references.

    Gemini has one row per panel video. Qwen has two flat rows per video, one
    for each declared condition.
    """

    source = _resolve_freeze_file(path)
    payload = _read_json_or_jsonl(source)
    rows, status, decision = _extract_rows(payload, source=source)
    expected_row_count = len(expected_video_ids) * (2 if model == "qwen" else 1)
    if len(rows) != expected_row_count:
        raise FreezeValidationError(
            f"raw {model} freeze must contain exactly {expected_row_count} rows"
        )
    answers: dict[str, dict[str, str]] = {}
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        video_id, extracted = _raw_answers(row, model=model)
        if model == "gemini":
            if video_id in answers:
                raise FreezeValidationError(
                    f"duplicate raw prediction video_id: {video_id}"
                )
            answers[video_id] = extracted
        else:
            existing = answers.setdefault(video_id, {})
            duplicate_conditions = set(existing).intersection(extracted)
            if duplicate_conditions:
                raise FreezeValidationError(
                    f"duplicate Qwen condition for video_id: {video_id}"
                )
            existing.update(extracted)
        normalized_rows.append(dict(row))
    if set(answers) != expected_video_ids:
        raise FreezeValidationError(f"raw {model} freeze video IDs do not match the panel")
    if model == "qwen":
        expected_conditions = set(_QWEN_CONDITIONS.values())
        incomplete = [
            video_id
            for video_id, values in answers.items()
            if set(values) != expected_conditions
        ]
        if incomplete:
            raise FreezeValidationError(
                "Qwen freeze is missing one condition for: " + ", ".join(sorted(incomplete))
            )
    return FreezeResult(
        path=source,
        sha256=_sha256(source),
        rows=tuple(normalized_rows),
        answers=answers,
        status=status,
        decision=decision,
    )


def validate_official_submission(
    path: Path,
    *,
    expected_query_ids: set[int],
    expected_video_ids: set[str],
) -> FreezeResult:
    """Validate an optional ten-answer organizer-shape canary separately."""

    source = _resolve_freeze_file(path)
    rows, status, decision = _extract_rows(_read_json_or_jsonl(source), source=source)
    normalized = [_normalise_row(row) for row in rows]
    try:
        validate_ranked_submission(normalized, expected_query_ids=expected_query_ids)
    except (TypeError, ValueError) as error:
        raise FreezeValidationError(str(error)) from error
    video_ids = {row["Video_ID"] for row in normalized}
    if video_ids != expected_video_ids:
        raise FreezeValidationError("official canary video IDs do not match the panel")
    return FreezeResult(
        path=source,
        sha256=_sha256(source),
        rows=tuple(normalized),
        answers={},
        status=status,
        decision=decision,
    )


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_contact_sheet(
    source: Path,
    destination: Path,
    *,
    duration_seconds: float,
    frame_count: int,
    ffmpeg: str,
) -> None:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = max(0.05, frame_count / max(float(duration_seconds), 0.1))
    columns = min(3, frame_count)
    rows = math.ceil(frame_count / columns)
    vf = f"fps={fps:.8f},scale=480:-2,tile={columns}x{rows}:padding=4:margin=4"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        vf,
        "-frames:v",
        "1",
        str(destination),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise RuntimeError(f"could not execute {ffmpeg!r} for {source}") from error
    if completed.returncode != 0 or not destination.is_file():
        detail = (completed.stderr or "").strip()
        raise RuntimeError(f"contact sheet failed for {source}: {detail}")


def assemble_private_packet(
    panel_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    frame_count: int = 6,
    ffmpeg: str = "ffmpeg",
    contact_sheet_builder=_make_contact_sheet,
) -> dict[str, Any]:
    """Create media-only evidence before any reference is loaded."""

    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    sheets: list[dict[str, Any]] = []
    for row in panel_rows:
        source = Path(str(row["source_path"])).resolve(strict=True)
        video_id = str(row["video_id"])
        destination = output_root / "contact-sheets" / f"{video_id}.jpg"
        contact_sheet_builder(
            source,
            destination,
            duration_seconds=float(row["duration_seconds"]),
            frame_count=frame_count,
            ffmpeg=ffmpeg,
        )
        sheets.append(
            {
                "video_id": video_id,
                "stratum": row["stratum"],
                "source_path": str(source),
                "source_sha256": row["source_sha256"],
                "contact_sheet": str(destination.relative_to(output_root)),
                "contact_sheet_sha256": _sha256(destination),
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "references_loaded": False,
        "contact_sheets": sheets,
    }
    _json_dump(output_root / "packet-manifest.json", manifest)
    return manifest


def _load_references(
    reference_root: Path, panel_rows: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    """Read exactly the five post-freeze reference records."""

    reference_root = reference_root.expanduser().resolve(strict=True)
    references: dict[str, str] = {}
    for row in panel_rows:
        video_id = str(row["video_id"])
        path = reference_root / f"{video_id}.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"reference is not a regular file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"reference must be a JSON object: {path}")
        answer = payload.get("correct_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"reference correct_answer is missing: {path}")
        references[video_id] = answer.strip()
    return references


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _normalize(prediction).split()
    reference_tokens = _normalize(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _local_metrics(
    answers: Mapping[str, str], references: Mapping[str, str]
) -> dict[str, Any]:
    f1_values = []
    exact_values = []
    for video_id, answer in answers.items():
        reference = references[video_id]
        f1_values.append(_token_f1(answer, reference))
        exact_values.append(float(_normalize(answer) == _normalize(reference)))
    return {
        "mode": "local-lexical-diagnostics",
        "note": "Token F1 and normalized exact match are local diagnostics; they are not the official 2026 semantic scores.",
        "mean_token_f1": fmean(f1_values),
        "mean_normalized_exact_match": fmean(exact_values),
        "rows": {
            video_id: {
                "token_f1": _token_f1(answer, references[video_id]),
                "normalized_exact_match": float(
                    _normalize(answer) == _normalize(references[video_id])
                ),
            }
            for video_id, answer in sorted(answers.items())
        },
    }


def _write_manual_audit(
    output_root: Path,
    panel_rows: Sequence[Mapping[str, Any]],
    references: Mapping[str, str],
    gemini: FreezeResult,
    qwen: FreezeResult,
) -> None:
    audit_rows = []
    markdown = [
        "# Deadline VQA five-row manual audit",
        "",
        "This is a private, post-freeze inspection template. Mark correctness independently; the local metrics are diagnostic only.",
        "",
        "| Stratum | Video | Question | Reference | Gemini answer | Qwen + transcript | Qwen - transcript | Gemini correct? | Qwen correct? | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in panel_rows:
        video_id = str(row["video_id"])
        reference = references[video_id]
        gemini_answer = gemini.answers[video_id]["gemini"]
        qwen_transcript = qwen.answers[video_id]["qwen_transcript"]
        qwen_no_transcript = qwen.answers[video_id]["qwen_no_transcript"]
        audit_rows.append(
            {
                "video_id": video_id,
                "stratum": row["stratum"],
                "question": row["question"],
                "reference": reference,
                "gemini_answer": gemini_answer,
                "qwen_transcript_answer": qwen_transcript,
                "qwen_no_transcript_answer": qwen_no_transcript,
                "gemini_correct": None,
                "qwen_correct": None,
                "notes": "",
            }
        )
        markdown.append(
            "| "
            + " | ".join(
                value.replace("|", "\\|").replace("\n", " ")
                for value in (
                    str(row["stratum"]),
                    video_id,
                    str(row["question"]),
                    reference,
                    gemini_answer,
                    qwen_transcript,
                    qwen_no_transcript,
                    "",
                    "",
                    "",
                )
            )
            + " |"
        )
    _json_dump(output_root / "manual-audit.json", {"rows": audit_rows})
    (output_root / "manual-audit.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    _json_dump(
        output_root / "metrics.json",
        {
            "schema_version": 1,
            "gemini": _local_metrics(
                {video_id: values["gemini"] for video_id, values in gemini.answers.items()},
                references,
            ),
            "qwen_transcript": _local_metrics(
                {
                    video_id: values["qwen_transcript"]
                    for video_id, values in qwen.answers.items()
                },
                references,
            ),
            "qwen_no_transcript": _local_metrics(
                {
                    video_id: values["qwen_no_transcript"]
                    for video_id, values in qwen.answers.items()
                },
                references,
            ),
        },
    )


def prepare_manual_audit(
    *,
    panel_path: Path = DEFAULT_PANEL,
    data_root: Path = DEFAULT_DATA_ROOT,
    gemini_freeze: Path,
    qwen_freeze: Path,
    official_canary: Path | None = None,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    frame_count: int = 6,
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    """Validate two freezes, build media evidence, then bind references."""

    panel_rows = load_panel(panel_path, data_root=data_root)
    panel_video_ids = {str(row["video_id"]) for row in panel_rows}

    gemini = validate_prediction_freeze(
        gemini_freeze,
        expected_video_ids=panel_video_ids,
        model="gemini",
    )
    qwen = validate_prediction_freeze(
        qwen_freeze,
        expected_video_ids=panel_video_ids,
        model="qwen",
    )

    official_result: FreezeResult | None = None
    if official_canary is not None:
        # The local canary assigns Q_ID 1..5 only for structural validation.
        # It is never represented as an official test submission.
        official_result = validate_official_submission(
            official_canary,
            expected_query_ids=set(range(1, len(panel_rows) + 1)),
            expected_video_ids=panel_video_ids,
        )

    # This is the explicit boundary: no reference path is opened before this
    # media-only packet exists and both prediction hashes are recorded.
    packet = assemble_private_packet(
        panel_rows,
        output_root,
        frame_count=frame_count,
        ffmpeg=ffmpeg,
    )
    references = _load_references(reference_root, panel_rows)
    _write_manual_audit(output_root, panel_rows, references, gemini, qwen)
    packet["references_loaded"] = True
    packet["prediction_freezes"] = {
        "gemini": {"path": str(gemini.path), "sha256": gemini.sha256},
        "qwen": {"path": str(qwen.path), "sha256": qwen.sha256},
    }
    if official_result is not None:
        packet["official_canary_structure"] = {
            "path": str(official_result.path),
            "sha256": official_result.sha256,
            "query_ids_are_local_fixture_only": True,
        }
    _json_dump(output_root / "packet-manifest.json", packet)
    return packet


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--gemini-freeze", type=Path, required=True)
    parser.add_argument("--qwen-freeze", type=Path, required=True)
    parser.add_argument(
        "--official-canary",
        type=Path,
        help="optional already-assembled five-row ten-answer JSON for shape parity",
    )
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frame-count", type=int, default=6)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare_manual_audit(
            panel_path=args.panel,
            data_root=args.data_root,
            gemini_freeze=args.gemini_freeze,
            qwen_freeze=args.qwen_freeze,
            official_canary=args.official_canary,
            reference_root=args.references,
            output_root=args.output_root,
            frame_count=args.frame_count,
            ffmpeg=args.ffmpeg,
        )
    except (FreezeValidationError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"prepare_vqa_deadline_manual_audit: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
