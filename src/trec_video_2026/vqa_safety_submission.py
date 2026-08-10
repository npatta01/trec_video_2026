"""Contracts for the private Qwen3.5 TRECVID 2026 safety submission."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from trec_video_2026.vqa_models import build_controlled_messages, classify_question_type
from trec_video_2026.vqa_submission import (
    audit_label_free_payload,
    freeze_predictions,
)


SUPPORTED_MEDIA_SUFFIXES = (".mkv", ".mp4", ".webm", ".mov")


@dataclass(frozen=True)
class CandidateCompletion:
    answers: tuple[str, ...]
    parsed_tail_count: int
    repair_count: int
    fallback_count: int


ExtractFrames = Callable[[Path, Path], tuple[tuple[Path, ...], tuple[float, ...]]]
PrimaryGenerator = Callable[[dict[str, Any]], tuple[str, float, dict[str, Any]]]
TailGenerator = Callable[[dict[str, Any], str], tuple[str, float, dict[str, Any]]]
RepairGenerator = Callable[
    [dict[str, Any], tuple[str, ...], int], tuple[str, float, dict[str, Any]]
]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_candidates(dataset_root: Path, video_id: str) -> list[Path]:
    testing_root = dataset_root / "vqa/2026/testing"
    roots = (testing_root / "videos", testing_root / "videos_original")
    video_path = Path(video_id)
    names = (
        (video_path.name,)
        if video_path.suffix.lower() in SUPPORTED_MEDIA_SUFFIXES
        else tuple(f"{video_id}{suffix}" for suffix in SUPPORTED_MEDIA_SUFFIXES)
    )
    return sorted(
        {root / name for root in roots for name in names if (root / name).is_file()},
        key=lambda path: path.as_posix(),
    )


def locate_test_media(dataset_root: Path, video_id: str) -> Path | None:
    """Resolve one official test video without guessing among duplicates."""

    candidates = _media_candidates(dataset_root, video_id)
    if len(candidates) > 1:
        raise ValueError(f"multiple local media files for {video_id}: {candidates}")
    return candidates[0] if candidates else None


def uniform_timestamps(
    duration_seconds: float, frame_count: int = 12
) -> tuple[float, ...]:
    """Return deterministic bin-midpoint timestamps for the verified baseline."""

    if not isinstance(duration_seconds, (int, float)) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise ValueError("frame_count must be a positive integer")
    return tuple(
        round((index + 0.5) * float(duration_seconds) / frame_count, 6)
        for index in range(frame_count)
    )


def _last_decodable_video_timestamp(ffprobe: str, media: Path) -> float:
    """Return the final decoded video PTS when container duration runs longer."""

    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    timestamps = []
    for line in completed.stdout.splitlines():
        try:
            timestamp = float(line.strip())
        except ValueError:
            continue
        if math.isfinite(timestamp) and timestamp >= 0:
            timestamps.append(timestamp)
    if not timestamps:
        raise RuntimeError(f"ffprobe found no decodable video timestamps: {media}")
    return max(timestamps)


def extract_uniform_frames(
    media_path: Path | str, destination: Path | str
) -> tuple[tuple[Path, ...], tuple[float, ...]]:
    """Materialize the verified uniform-12 image evidence once per video."""

    import shutil

    media = Path(media_path).resolve(strict=True)
    output = Path(destination)
    evidence_path = output / "evidence.json"
    source_digest = sha256_path(media)
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("source_sha256") != source_digest:
            raise ValueError("existing frame evidence source hash mismatch")
        records = evidence.get("frames")
        if not isinstance(records, list) or len(records) != 12:
            raise ValueError("existing frame evidence is incomplete")
        frames = tuple(output / record["path"] for record in records)
        timestamps = tuple(float(record["timestamp_seconds"]) for record in records)
        for frame, record in zip(frames, records, strict=True):
            if not frame.is_file() or sha256_path(frame) != record.get("sha256"):
                raise ValueError("existing frame evidence hash mismatch")
        return frames, timestamps
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"incomplete frame evidence directory exists: {output}")

    ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required for frame extraction")
    duration_result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        duration = float(duration_result.stdout.strip())
    except ValueError as error:
        raise ValueError(f"cannot determine media duration: {media}") from error
    timestamps = uniform_timestamps(duration)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        records: list[dict[str, Any]] = []
        last_video_timestamp: float | None = None
        for index, timestamp in enumerate(timestamps, start=1):
            frame = stage / f"frame-{index:02d}.png"
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(media),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-y",
                str(frame),
            ]
            subprocess.run(command, check=True, capture_output=True)
            if not frame.is_file() or frame.stat().st_size == 0:
                if last_video_timestamp is None:
                    last_video_timestamp = _last_decodable_video_timestamp(
                        ffprobe, media
                    )
                command[5] = f"{min(timestamp, last_video_timestamp):.6f}"
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                )
            if not frame.is_file() or frame.stat().st_size == 0:
                raise RuntimeError(
                    f"ffmpeg produced no frame for {media} at {timestamp}"
                )
            records.append(
                {
                    "path": frame.name,
                    "timestamp_seconds": timestamp,
                    "sha256": sha256_path(frame),
                }
            )
        write_new_json(
            stage / "evidence.json",
            {
                "schema_version": 1,
                "source_sha256": source_digest,
                "source_size_bytes": media.stat().st_size,
                "duration_seconds": duration,
                "frame_count": 12,
                "frames": records,
            },
        )
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return (
        tuple(output / record["path"] for record in records),
        timestamps,
    )


def build_ranked_tail_messages(
    *,
    question: str,
    frame_paths: tuple[Path, ...],
    frame_timestamps: tuple[float, ...],
    primary_answer: str,
) -> list[dict[str, Any]]:
    """Build the second-pass request for nine ordered alternative answers."""

    primary = " ".join(primary_answer.split())
    if not primary:
        raise ValueError("primary_answer cannot be empty")
    messages = build_controlled_messages(
        question=question,
        frame_paths=frame_paths,
        frame_timestamps=frame_timestamps,
        transcript=None,
        answer_policy="current",
    )
    instruction = (
        "Generate exactly nine additional concise plausible answers, ordered from "
        "most to least likely. Each must give a genuinely different core hypothesis; "
        "paraphrases and repeated strings are forbidden. Each must be semantically "
        "distinct from the primary "
        f"response {json.dumps(primary, ensure_ascii=False)} and from every other "
        'candidate. Return only a JSON object with the single key "answers" whose '
        "value is an array of exactly nine non-empty strings."
    )
    messages[0]["content"][-1] = {"type": "text", "text": instruction}
    audit_label_free_payload(messages)
    return messages


def build_ranked_repair_messages(
    *,
    question: str,
    frame_paths: tuple[Path, ...],
    frame_timestamps: tuple[float, ...],
    excluded_answers: tuple[str, ...],
    missing_count: int,
) -> list[dict[str, Any]]:
    """Build a bounded repair request for missing distinct tail candidates."""

    number_words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }
    if missing_count not in number_words:
        raise ValueError("missing_count must be between one and nine")
    if not excluded_answers or any(not value.strip() for value in excluded_answers):
        raise ValueError("excluded_answers must contain non-empty strings")
    messages = build_controlled_messages(
        question=question,
        frame_paths=frame_paths,
        frame_timestamps=frame_timestamps,
        transcript=None,
        answer_policy="current",
    )
    instruction = (
        f"Generate exactly {number_words[missing_count]} additional concise plausible "
        "answers, ordered from most to least likely. Each answer must give a genuinely "
        "different core hypothesis about the event, entity, action, object, state, or "
        "setting. Paraphrases are forbidden. Copying any excluded response is a failure. "
        "The answers must be semantically distinct from one another and from every "
        "excluded response in this JSON array: "
        f"{json.dumps(excluded_answers, ensure_ascii=False)}. Return only a JSON object "
        'with the single key "answers" whose value is an array of exactly '
        f"{number_words[missing_count]} non-empty strings."
    )
    messages[0]["content"][-1] = {"type": "text", "text": instruction}
    audit_label_free_payload(messages)
    return messages


def parse_ranked_tail(text: str) -> tuple[str, ...]:
    """Parse the exact ranked-tail JSON contract without extracting loose text."""

    if not isinstance(text, str):
        raise TypeError("ranked tail must be text")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("ranked tail must be a JSON object") from error
    if not isinstance(payload, dict) or set(payload) != {"answers"}:
        raise ValueError("ranked tail must be a JSON object with only answers")
    answers = payload["answers"]
    if not isinstance(answers, list) or len(answers) != 9:
        raise ValueError("ranked tail must contain exactly nine answers")
    normalized: list[str] = []
    for answer in answers:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("ranked tail answers must be non-empty strings")
        normalized.append(" ".join(answer.split()))
    return tuple(normalized)


def _parse_ranked_repair(text: str, expected_count: int) -> tuple[str, ...]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("ranked repair must be a JSON object") from error
    if not isinstance(payload, dict) or set(payload) != {"answers"}:
        raise ValueError("ranked repair must be a JSON object with only answers")
    answers = payload["answers"]
    if not isinstance(answers, list) or len(answers) != expected_count:
        raise ValueError(f"ranked repair must contain exactly {expected_count} answers")
    if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ValueError("ranked repair answers must be non-empty strings")
    return tuple(" ".join(answer.split()) for answer in answers)


def _answer_key(answer: str) -> str:
    normalized = answer.strip().casefold()
    while normalized and unicodedata.category(normalized[-1]).startswith("P"):
        normalized = normalized[:-1].rstrip()
    return normalized


_GENERAL_FALLBACKS = (
    "The video does not clearly show the answer",
    "It cannot be determined from the visible evidence",
    "The relevant event is unclear",
    "There is not enough visible information",
    "The action is not clearly identifiable",
    "The scene is ambiguous",
    "No definite answer is visible",
    "The requested detail cannot be seen",
    "The video evidence is inconclusive",
    "The answer is uncertain",
)


def _fallback_candidates(question: str) -> tuple[str, ...]:
    question_type = classify_question_type(question)
    if question_type == "count":
        return tuple(str(value) for value in range(0, 21)) + _GENERAL_FALLBACKS
    if question_type == "visible-text":
        return (
            "The text is not legible",
            "No readable words are visible",
            "The writing cannot be determined",
        ) + _GENERAL_FALLBACKS
    if question_type == "speech":
        return (
            "The speech is not intelligible",
            "No clear words can be heard",
            "The spoken answer cannot be determined",
        ) + _GENERAL_FALLBACKS
    return _GENERAL_FALLBACKS


def complete_ranked_candidates(
    *,
    primary_answer: str,
    tail_text: str,
    question: str,
    repair_text: str | None = None,
) -> CandidateCompletion:
    """Build ten unique candidates and record any deterministic tail fallback."""

    primary = " ".join(primary_answer.split())
    if not primary:
        raise ValueError("primary_answer cannot be empty")
    answers = [primary]
    seen = {_answer_key(primary)}
    try:
        parsed = parse_ranked_tail(tail_text)
    except ValueError:
        parsed = ()
    parsed_tail_count = 0
    for answer in parsed:
        key = _answer_key(answer)
        if key and key not in seen:
            answers.append(answer)
            seen.add(key)
            parsed_tail_count += 1
        if len(answers) == 10:
            break
    repair_count = 0
    missing_count = 10 - len(answers)
    if repair_text is not None and missing_count:
        try:
            repaired = _parse_ranked_repair(repair_text, missing_count)
        except ValueError:
            repaired = ()
        for answer in repaired:
            key = _answer_key(answer)
            if key and key not in seen:
                answers.append(answer)
                seen.add(key)
                repair_count += 1
            if len(answers) == 10:
                break
    for answer in _fallback_candidates(question):
        if len(answers) == 10:
            break
        key = _answer_key(answer)
        if key and key not in seen:
            answers.append(answer)
            seen.add(key)
    if len(answers) != 10:
        raise ValueError("could not complete exactly ten unique candidates")
    return CandidateCompletion(
        answers=tuple(answers),
        parsed_tail_count=parsed_tail_count,
        repair_count=repair_count,
        fallback_count=9 - parsed_tail_count - repair_count,
    )


def prepare_inference_requests(
    manifest: dict[str, Any],
    *,
    dataset_root: Path | str,
    frame_root: Path | str,
    extract_frames: ExtractFrames,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Materialize one shared uniform-12 evidence set per available video."""

    queries = manifest.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("manifest queries must be a non-empty list")
    if workers <= 0:
        raise ValueError("workers must be positive")
    root = Path(dataset_root).resolve(strict=True)
    frames_root = Path(frame_root)
    media_by_video: dict[str, Path] = {}
    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise TypeError(f"manifest query {index} must be an object")
        video_id = query.get("video_id")
        availability = query.get("availability")
        if not isinstance(video_id, str):
            raise ValueError(f"manifest query {index} has an invalid video ID")
        if availability == "available" and video_id not in media_by_video:
            media_value = query.get("media")
            if not isinstance(media_value, str):
                raise ValueError("available query is missing media")
            media = (root / media_value).resolve(strict=True)
            if not media.is_relative_to(root):
                raise ValueError("manifest media escapes dataset root")
            if sha256_path(media) != query.get("source_sha256"):
                raise ValueError("manifest media hash mismatch")
            if media.stat().st_size != query.get("source_size_bytes"):
                raise ValueError("manifest media size mismatch")
            media_by_video[video_id] = media
        elif availability not in {"available", "unavailable"}:
            raise ValueError("query availability must be available or unavailable")

    def extract(
        item: tuple[str, Path],
    ) -> tuple[str, tuple[tuple[Path, ...], tuple[float, ...]]]:
        video_id, media = item
        return video_id, extract_frames(media, frames_root / video_id)

    if workers == 1:
        extracted = map(extract, media_by_video.items())
        cached = dict(extracted)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            cached = dict(executor.map(extract, media_by_video.items()))

    requests: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        video_id = query.get("video_id")
        availability = query.get("availability")
        frames: tuple[Path, ...] = ()
        timestamps: tuple[float, ...] = ()
        if availability == "available":
            frames, timestamps = cached[video_id]
            if len(frames) != 12 or len(timestamps) != 12:
                raise ValueError("uniform evidence must contain exactly twelve frames")
        requests.append(
            {
                "query_id": query["query_id"],
                "video_id": video_id,
                "question": query["question"],
                "availability": availability,
                "frame_paths": [str(path.resolve()) for path in frames],
                "frame_timestamps": list(timestamps),
                "source_sha256": query.get("source_sha256"),
            }
        )
    audit_label_free_payload(requests)
    return requests


def build_prediction_row(
    request: dict[str, Any],
    *,
    generate_primary: PrimaryGenerator,
    generate_tail: TailGenerator,
    generate_repair: RepairGenerator | None = None,
) -> dict[str, Any]:
    """Execute the two-pass safety policy for one prepared query."""

    primary, primary_elapsed, primary_provenance = generate_primary(request)
    if not isinstance(primary_provenance, dict):
        raise TypeError("primary provenance must be an object")
    tail_text, tail_elapsed, tail_provenance = generate_tail(request, primary)
    if not isinstance(tail_provenance, dict):
        raise TypeError("tail provenance must be an object")
    for name, elapsed in (("primary", primary_elapsed), ("tail", tail_elapsed)):
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError(f"{name} elapsed time must be finite and non-negative")
    completion = complete_ranked_candidates(
        primary_answer=primary,
        tail_text=tail_text,
        question=request["question"],
    )
    repair_text: str | None = None
    repair_elapsed = 0.0
    repair_provenance: dict[str, Any] | None = None
    if completion.fallback_count and generate_repair is not None:
        excluded = completion.answers[: 1 + completion.parsed_tail_count]
        repair_text, repair_elapsed, repair_provenance = generate_repair(
            request, excluded, completion.fallback_count
        )
        if not isinstance(repair_provenance, dict):
            raise TypeError("repair provenance must be an object")
        if (
            isinstance(repair_elapsed, bool)
            or not isinstance(repair_elapsed, (int, float))
            or not math.isfinite(repair_elapsed)
            or repair_elapsed < 0
        ):
            raise ValueError("repair elapsed time must be finite and non-negative")
        completion = complete_ranked_candidates(
            primary_answer=primary,
            tail_text=tail_text,
            question=request["question"],
            repair_text=repair_text,
        )
    answers: list[dict[str, Any]] = []
    for index, answer in enumerate(completion.answers):
        if index == 0:
            elapsed = float(primary_elapsed)
            source = "rank1-uniform12"
        elif index <= completion.parsed_tail_count:
            elapsed = float(tail_elapsed)
            source = "ranked-tail"
        elif index <= completion.parsed_tail_count + completion.repair_count:
            elapsed = float(repair_elapsed)
            source = "ranked-repair"
        else:
            elapsed = 0.0
            source = "deterministic-fallback"
        answers.append({"answer": answer, "elapsed_seconds": elapsed, "source": source})
    return {
        "query_id": request["query_id"],
        "video_id": request["video_id"],
        "availability": request["availability"],
        "answers": answers,
        "parsed_tail_count": completion.parsed_tail_count,
        "repair_count": completion.repair_count,
        "fallback_count": completion.fallback_count,
        "tail_raw_text": tail_text,
        "repair_raw_text": repair_text,
        "rank1_provenance": primary_provenance,
        "tail_provenance": tail_provenance,
        "repair_provenance": repair_provenance,
    }


def validate_resume_prefix(
    requests: list[dict[str, Any]], completed: list[dict[str, Any]]
) -> int:
    """Return the next row index after validating an exact completed prefix."""

    if len(completed) > len(requests):
        raise ValueError("resume prefix is longer than the request list")
    for index, (request, prediction) in enumerate(
        zip(requests, completed, strict=False)
    ):
        if prediction.get("query_id") != request.get("query_id") or prediction.get(
            "video_id"
        ) != request.get("video_id"):
            raise ValueError(f"resume prefix mismatch at row {index}")
    return len(completed)


def _validate_prediction_rows(
    manifest: dict[str, Any], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    queries = manifest.get("queries")
    predictions = payload.get("predictions")
    if not isinstance(queries, list) or not isinstance(predictions, list):
        raise ValueError("manifest and prediction rows must be lists")
    if len(predictions) != len(queries):
        raise ValueError("prediction count does not match manifest")
    validated: list[dict[str, Any]] = []
    for index, (query, prediction) in enumerate(zip(queries, predictions, strict=True)):
        if not isinstance(prediction, dict):
            raise TypeError(f"prediction {index} must be an object")
        if prediction.get("query_id") != query.get("query_id"):
            raise ValueError("prediction query IDs/order do not match manifest")
        if prediction.get("video_id") != query.get("video_id"):
            raise ValueError("prediction video IDs/order do not match manifest")
        answers = prediction.get("answers")
        if not isinstance(answers, list) or len(answers) != 10:
            raise ValueError("each prediction must contain exactly ten answers")
        seen: set[str] = set()
        official_answers: list[dict[str, Any]] = []
        for rank, candidate in enumerate(answers, start=1):
            if not isinstance(candidate, dict):
                raise TypeError("prediction answers must be objects")
            answer = candidate.get("answer")
            elapsed = candidate.get("elapsed_seconds")
            source = candidate.get("source")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("prediction answers must be non-empty strings")
            key = _answer_key(answer)
            if key in seen:
                raise ValueError("duplicate normalized answers are not allowed")
            seen.add(key)
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError("prediction time must be finite and non-negative")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("prediction source must be a non-empty string")
            official_answers.append(
                {"Rank": rank, "Answer": answer.strip(), "Time": float(elapsed)}
            )
        validated.append(
            {
                "Q_ID": query["query_id"],
                "Video_ID": query["video_id"],
                "Answers": official_answers,
            }
        )
    return validated


def finalize_submission(
    manifest_path: Path | str,
    predictions_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Freeze private predictions, then create immutable AG submission views."""

    manifest_file = Path(manifest_path).resolve(strict=True)
    prediction_file = Path(predictions_path).resolve(strict=True)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(predictions, dict):
        raise ValueError("manifest and predictions must be JSON objects")
    official = _validate_prediction_rows(manifest, predictions)
    run_id = predictions.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("predictions must declare a run_id")

    destination = Path(output_dir)
    paths = {
        "predictions": destination / "ranked-predictions.json",
        "freeze": destination / "freeze.json",
        "top1": destination / "submission-top1.json",
        "top5": destination / "submission-top5.json",
        "top10": destination / "submission-top10.json",
        "checksums": destination / "SHA256SUMS",
    }
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    destination.mkdir(parents=True, exist_ok=True)
    bindings = {
        "run_id": run_id,
        "query_count": len(official),
        "manifest_sha256": sha256_path(manifest_file),
        "topics_sha256": manifest.get("topics_sha256"),
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }
    write_new_json(paths["predictions"], {**predictions, **bindings})
    freeze_predictions(
        paths["predictions"],
        bindings=bindings,
        output_path=paths["freeze"],
    )
    for cutoff, key in ((1, "top1"), (5, "top5"), (10, "top10")):
        write_new_json(
            paths[key],
            [{**row, "Answers": row["Answers"][:cutoff]} for row in official],
            sort_keys=False,
        )
    checksum_lines = [
        f"{sha256_path(paths[key])}  {paths[key].name}"
        for key in ("predictions", "freeze", "top1", "top5", "top10")
    ]
    with paths["checksums"].open("x", encoding="utf-8") as handle:
        handle.write("\n".join(checksum_lines) + "\n")
    return paths


def run_nist_validator(
    *,
    validator_script: Path | str,
    topics_path: Path | str,
    submission_path: Path | str,
    report_path: Path | str,
    python_executable: str = sys.executable,
) -> subprocess.CompletedProcess[str]:
    """Run the pinned NIST AG validator and propagate a rejected submission."""

    report = Path(report_path)
    if report.exists() or report.is_symlink():
        raise FileExistsError(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(Path(validator_script).resolve(strict=True)),
        str(Path(topics_path).resolve(strict=True)),
        str(Path(submission_path).resolve(strict=True)),
        "--report-json",
        str(report),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        raise RuntimeError(
            f"NIST validator exited {completed.returncode}: {detail or 'no output'}"
        )
    return completed


def build_test_manifest(
    topics_path: Path | str, dataset_root: Path | str
) -> dict[str, Any]:
    """Build an ordered, label-free manifest from official 2026 AG topics."""

    topics_file = Path(topics_path).resolve(strict=True)
    root = Path(dataset_root).resolve(strict=True)
    topics = json.loads(topics_file.read_text(encoding="utf-8"))
    audit_label_free_payload(topics)
    if not isinstance(topics, list) or not topics:
        raise ValueError("topics must be a non-empty list")

    seen_query_ids: set[int] = set()
    video_availability: dict[str, bool] = {}
    queries: list[dict[str, Any]] = []
    for index, topic in enumerate(topics):
        if not isinstance(topic, dict):
            raise TypeError(f"topic {index} must be an object")
        if set(topic) != {"Q_ID", "Video_ID", "Question"}:
            raise ValueError(f"topic {index} has unexpected or forbidden fields")
        query_id = topic["Q_ID"]
        video_id = topic["Video_ID"]
        question = topic["Question"]
        if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id < 1:
            raise ValueError(f"topic {index} has an invalid query ID")
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query ID: {query_id}")
        seen_query_ids.add(query_id)
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"topic {index} has an invalid video ID")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"topic {index} has an invalid question")

        media = locate_test_media(root, video_id)
        available = media is not None
        video_availability.setdefault(video_id, available)
        if video_availability[video_id] != available:
            raise ValueError(f"inconsistent media availability for {video_id}")
        queries.append(
            {
                "query_id": query_id,
                "video_id": video_id,
                "question": question.strip(),
                "availability": "available" if available else "unavailable",
                "media": media.relative_to(root).as_posix() if media else None,
                "source_sha256": sha256_path(media) if media else None,
                "source_size_bytes": media.stat().st_size if media else None,
            }
        )

    available_count = sum(video_availability.values())
    manifest = {
        "schema_version": 1,
        "manifest_version": "trecvid-2026-ag-qwen35-safety-v1",
        "data_boundary": "label-free-inference-input",
        "topics_sha256": sha256_path(topics_file),
        "query_count": len(queries),
        "video_count": len(video_availability),
        "available_video_count": available_count,
        "unavailable_video_count": len(video_availability) - available_count,
        "queries": queries,
    }
    audit_label_free_payload(manifest)
    return manifest


def write_new_json(path: Path | str, payload: Any, *, sort_keys: bool = True) -> None:
    """Atomically create a JSON artifact and refuse overwrite."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CandidateCompletion",
    "SUPPORTED_MEDIA_SUFFIXES",
    "build_ranked_tail_messages",
    "build_prediction_row",
    "build_test_manifest",
    "complete_ranked_candidates",
    "extract_uniform_frames",
    "finalize_submission",
    "locate_test_media",
    "parse_ranked_tail",
    "prepare_inference_requests",
    "run_nist_validator",
    "sha256_path",
    "uniform_timestamps",
    "validate_resume_prefix",
    "write_new_json",
]
