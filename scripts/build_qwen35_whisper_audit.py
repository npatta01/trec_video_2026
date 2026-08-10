#!/usr/bin/env python3
"""Score a frozen Qwen frame/transcript pair and build a private audit index.

The remote run is label-free.  This script is deliberately post-run: it
verifies the downloaded output checksum manifest before reading the local VQA
annotations, then joins labels only in the private audit output.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import parse_qs, urlparse


FRAMES_ONLY = "qwen35-4b-12f-frames-only"
FRAMES_WHISPER = "qwen35-4b-12f-whisper-large-v3"
SUPPORTED_MEDIA = (".mkv", ".mp4", ".webm", ".mov")
CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_checksums(output_root: Path) -> dict[str, str]:
    checksum_path = output_root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ValueError(f"Missing checksum manifest: {checksum_path}")
    verified: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"Invalid checksum line: {line!r}")
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe checksum path: {raw_path}")
        path = output_root.joinpath(*relative.parts)
        if not path.is_file():
            raise ValueError(f"Checksum target is missing: {raw_path}")
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"Checksum mismatch for {raw_path}")
        verified[relative.as_posix()] = digest
    if not verified:
        raise ValueError("Checksum manifest is empty")
    return verified


def load_inference_manifest(path: Path) -> tuple[dict[str, dict], list[str]]:
    payload = read_json(path)
    if payload.get("label_policy") != "label-free-inference-input":
        raise ValueError("Inference manifest is not label-free")
    examples = payload.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Inference manifest has no examples")
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for example in examples:
        if set(example) != {"video_id", "question"}:
            raise ValueError("Inference manifest rows must contain only video_id/question")
        video_id = example["video_id"]
        question = example["question"]
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("Invalid video ID in inference manifest")
        if video_id in by_id:
            raise ValueError(f"Duplicate video ID: {video_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid question for {video_id}")
        by_id[video_id] = {"video_id": video_id, "question": question}
        order.append(video_id)
    return by_id, order


def load_references(
    annotation_root: Path, inference: dict[str, dict], order: list[str]
) -> dict[str, dict[str, str]]:
    references: dict[str, dict[str, str]] = {}
    for video_id in order:
        path = annotation_root / f"{video_id}.json"
        if not path.is_file():
            raise ValueError(f"Missing annotation for {video_id}: {path}")
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Annotation is not an object: {path}")
        video_url = payload.get("video_url")
        if not isinstance(video_url, str):
            raise ValueError(f"Annotation URL is missing: {path}")
        query_ids = parse_qs(urlparse(video_url).query).get("v", [])
        if query_ids != [video_id]:
            raise ValueError(f"Annotation video identity mismatch: {path}")
        question = payload.get("question")
        answer = payload.get("correct_answer")
        if question != inference[video_id]["question"]:
            raise ValueError(f"Question mismatch for {video_id}")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Missing correct answer for {video_id}")
        references[video_id] = {
            "question": question,
            "reference_answer": answer,
            "annotation_sha256": sha256(path),
        }
    return references


def load_predictions(path: Path, expected_condition: str, order: list[str]) -> dict[str, dict]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Predictions must be a list: {path}")
    by_id: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Prediction row is not an object: {path}")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or video_id not in order:
            raise ValueError(f"Prediction has unexpected video ID: {video_id}")
        if row.get("condition_id") != expected_condition:
            raise ValueError(f"Prediction condition mismatch for {video_id}")
        if video_id in by_id:
            raise ValueError(f"Duplicate prediction for {expected_condition}/{video_id}")
        if row.get("status") != "passed":
            raise ValueError(f"Prediction did not pass for {expected_condition}/{video_id}")
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Prediction answer is empty for {expected_condition}/{video_id}")
        by_id[video_id] = row
    if list(by_id) != order:
        raise ValueError(f"Prediction order/coverage mismatch for {expected_condition}")
    return by_id


def load_evidence(path: Path, order: list[str]) -> dict[str, dict]:
    payload = read_json(path)
    examples = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(examples, list) or len(examples) != len(order):
        raise ValueError("Frame evidence coverage mismatch")
    by_id: dict[str, dict] = {}
    for row in examples:
        video_id = row.get("video_id") if isinstance(row, dict) else None
        if not isinstance(video_id, str) or video_id not in order or video_id in by_id:
            raise ValueError(f"Invalid or duplicate frame evidence row: {video_id}")
        frames = row.get("frames")
        if not isinstance(frames, list) or len(frames) != 12:
            raise ValueError(f"Expected 12 frames for {video_id}")
        for frame in frames:
            if not isinstance(frame.get("selected_frame_timestamp_seconds"), (int, float)):
                raise ValueError(f"Missing selected frame PTS for {video_id}")
            if not isinstance(frame.get("sha256"), str):
                raise ValueError(f"Missing frame hash for {video_id}")
        by_id[video_id] = row
    if set(by_id) != set(order):
        raise ValueError("Frame evidence IDs do not match inference IDs")
    return by_id


def load_whisper(path: Path, order: list[str]) -> dict[str, dict]:
    payload = read_json(path)
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(order):
        raise ValueError("Whisper coverage mismatch")
    by_id: dict[str, dict] = {}
    for row in rows:
        video_id = row.get("video_id") if isinstance(row, dict) else None
        if not isinstance(video_id, str) or video_id not in order or video_id in by_id:
            raise ValueError(f"Invalid or duplicate Whisper row: {video_id}")
        by_id[video_id] = row
    if set(by_id) != set(order):
        raise ValueError("Whisper IDs do not match inference IDs")
    return by_id


def lexical_meteor(prediction: str, reference: str) -> float:
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    class LexicalWordNet:
        @staticmethod
        def synsets(_word: str) -> tuple[object, ...]:
            return ()

    return float(
        meteor_score(
            [word_tokenize(reference, preserve_line=True)],
            word_tokenize(prediction, preserve_line=True),
            wordnet=LexicalWordNet(),
        )
    )


def score_pair(
    frames: dict[str, dict],
    whisper: dict[str, dict],
    references: dict[str, dict[str, str]],
    order: list[str],
) -> tuple[list[dict], dict]:
    from trec_video_2026.vqa_evaluation import (
        normalized_exact_match,
        summarize_paired_conditions,
        token_f1,
    )

    scored: list[dict] = []
    for video_id in order:
        reference = references[video_id]["reference_answer"]
        for condition, predictions in (
            (FRAMES_ONLY, frames),
            (FRAMES_WHISPER, whisper),
        ):
            answer = predictions[video_id]["answer"]
            scored.append(
                {
                    "condition_id": condition,
                    "video_id": video_id,
                    "stratum": "vqa-2025-training-full-panel",
                    "completed": True,
                    "normalized_exact_match": normalized_exact_match(answer, reference),
                    "token_f1": token_f1(answer, reference),
                    "meteor": lexical_meteor(answer, reference),
                }
            )
    paired = summarize_paired_conditions(
        scored,
        baseline_condition_id=FRAMES_ONLY,
        treatment_condition_id=FRAMES_WHISPER,
        bootstrap_iterations=2000,
        bootstrap_seed=20260803,
    )
    return scored, paired


def media_path(data_root: Path, video_id: str) -> Path:
    root = data_root / "vqa" / "2025" / "training" / "videos_original"
    matches = [root / f"{video_id}{suffix}" for suffix in SUPPORTED_MEDIA if (root / f"{video_id}{suffix}").is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one source media file for {video_id}, found {matches}")
    return matches[0]


def build_rows(
    frames: dict[str, dict],
    whisper_predictions: dict[str, dict],
    whisper_evidence: dict[str, dict],
    evidence: dict[str, dict],
    references: dict[str, dict[str, str]],
    scored: list[dict],
    order: list[str],
    data_root: Path,
) -> list[dict]:
    score_by_key = {(row["condition_id"], row["video_id"]): row for row in scored}
    rows = []
    for video_id in order:
        evidence_row = evidence[video_id]
        whisper_row = whisper_evidence[video_id]
        frames_score = score_by_key[(FRAMES_ONLY, video_id)]
        whisper_score = score_by_key[(FRAMES_WHISPER, video_id)]
        wrong_conditions = [
            condition
            for condition, score in (
                (FRAMES_ONLY, frames_score),
                (FRAMES_WHISPER, whisper_score),
            )
            if score["normalized_exact_match"] == 0.0
        ]
        audio = evidence_row.get("audio", {})
        quality = whisper_row.get("quality", {})
        rows.append(
            {
                "video_id": video_id,
                "question": references[video_id]["question"],
                "reference_answer": references[video_id]["reference_answer"],
                "frames_only_answer": frames[video_id]["answer"],
                "frames_whisper_answer": whisper_predictions[video_id]["answer"],
                "wrong_by_normalized_exact_match": bool(wrong_conditions),
                "wrong_conditions": wrong_conditions,
                "scores": {
                    "frames_only": {
                        key: frames_score[key]
                        for key in ("normalized_exact_match", "token_f1", "meteor")
                    },
                    "frames_whisper": {
                        key: whisper_score[key]
                        for key in ("normalized_exact_match", "token_f1", "meteor")
                    },
                },
                "frame_evidence": {
                    "contact_sheet": f"frame-contact-sheets/{evidence_row['contact_sheet']}",
                    "contact_sheet_sha256": evidence_row["contact_sheet_sha256"],
                    "duration_seconds": evidence_row["duration_seconds"],
                    "requested_timestamps_seconds": [
                        frame["requested_timestamp_seconds"] for frame in evidence_row["frames"]
                    ],
                    "selected_frame_timestamps_seconds": [
                        frame["selected_frame_timestamp_seconds"] for frame in evidence_row["frames"]
                    ],
                    "frame_sha256": [frame["sha256"] for frame in evidence_row["frames"]],
                },
                "audio_evidence": {
                    "status": audio.get("status"),
                    "sha256": audio.get("sha256"),
                    "duration_seconds": audio.get("duration_seconds"),
                    "source_media": str(media_path(data_root, video_id)),
                },
                "whisper_evidence": {
                    "status": whisper_row.get("status"),
                    "text": whisper_row.get("text"),
                    "timestamped_text": whisper_row.get("timestamped_text"),
                    "quality_status": quality.get("status"),
                    "quality": quality,
                },
                "answerability_review": {
                    "status": "pending_manual_review",
                    "visual_evidence_available": True,
                    "audio_evidence_available": audio.get("status") == "extracted",
                    "review_scope": "Inspect the contact sheet and source audio/transcript before assigning present/absent/uncertain.",
                },
            }
        )
    return rows


def render_html(path: Path, payload: dict) -> None:
    rows = payload["rows"]
    body = []
    for row in rows:
        if not row["wrong_by_normalized_exact_match"]:
            continue
        evidence = row["frame_evidence"]
        body.append(
            "<tr>"
            f"<td><code>{html.escape(row['video_id'])}</code></td>"
            f"<td>{html.escape(row['question'])}</td>"
            f"<td>{html.escape(row['reference_answer'])}</td>"
            f"<td>{html.escape(row['frames_only_answer'])}</td>"
            f"<td>{html.escape(row['frames_whisper_answer'])}</td>"
            f"<td>{html.escape(', '.join(row['wrong_conditions']))}</td>"
            f"<td><a href=\"{html.escape(evidence['contact_sheet'])}\">frames</a><br>"
            f"<small>PTS: {html.escape(', '.join(f'{value:.3f}' for value in evidence['selected_frame_timestamps_seconds']))}</small></td>"
            f"<td>{html.escape(str(row['whisper_evidence']['text'] or ''))}</td>"
            "</tr>"
        )
    document = """<!doctype html>
<meta charset="utf-8">
<title>Qwen3.5-4B Whisper frame/audio audit</title>
<style>
body { font: 14px system-ui, sans-serif; margin: 2rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: .45rem; vertical-align: top; }
th { position: sticky; top: 0; background: #f4f4f4; }
img { max-width: 640px; }
small { color: #555; }
</style>
<h1>Qwen3.5-4B Whisper frame/audio audit</h1>
<p>Rows below are those with a normalized exact-match error in at least one
condition. This is a private, label-bearing audit index. The answerability
column remains pending until manual evidence review.</p>
<table><thead><tr>
<th>Video</th><th>Question</th><th>Reference</th><th>Frames only</th>
<th>Frames + Whisper</th><th>Wrong conditions</th><th>Selected frames / PTS</th>
<th>Whisper transcript</th>
</tr></thead><tbody>
""" + "\n".join(body) + """
</tbody></table>
"""
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remote-output-uri", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve(strict=True)
    verified_files = verify_checksums(output_root)
    inference, order = load_inference_manifest(args.inference_manifest.resolve(strict=True))
    references = load_references(args.annotation_root.resolve(strict=True), inference, order)
    frames = load_predictions(output_root / "predictions.frames-only.json", FRAMES_ONLY, order)
    whisper_predictions = load_predictions(output_root / "predictions.frames-whisper.json", FRAMES_WHISPER, order)
    evidence = load_evidence(output_root / "frame-evidence.json", order)
    whisper = load_whisper(output_root / "whisper-transcripts.json", order)
    scored, paired = score_pair(frames, whisper_predictions, references, order)
    rows = build_rows(
        frames,
        whisper_predictions,
        whisper,
        evidence,
        references,
        scored,
        order,
        args.data_root.resolve(strict=True),
    )
    wrong_rows = [row for row in rows if row["wrong_by_normalized_exact_match"]]
    condition_metrics = {}
    for condition in (FRAMES_ONLY, FRAMES_WHISPER):
        condition_rows = [row for row in scored if row["condition_id"] == condition]
        condition_metrics[condition] = {
            metric: sum(row[metric] for row in condition_rows) / len(condition_rows)
            for metric in ("normalized_exact_match", "token_f1", "meteor")
        }
    transitions = {
        "both_exactly_correct": sum(not row["wrong_by_normalized_exact_match"] for row in rows),
        "frames_only_only_wrong": sum(
            row["wrong_conditions"] == [FRAMES_ONLY] for row in rows
        ),
        "whisper_only_wrong": sum(
            row["wrong_conditions"] == [FRAMES_WHISPER] for row in rows
        ),
        "both_wrong": sum(len(row["wrong_conditions"]) == 2 for row in rows),
    }
    private_payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "remote_output_uri": args.remote_output_uri,
        "checksum_verified": True,
        "checksum_file_sha256": sha256(output_root / "SHA256SUMS"),
        "label_policy": "labels joined locally after remote prediction freeze",
        "frame_policy": "12 uniform temporal midpoint requests; selected FFmpeg PTS recorded",
        "condition_ids": [FRAMES_ONLY, FRAMES_WHISPER],
        "row_count": len(rows),
        "wrong_row_count": len(wrong_rows),
        "transition_counts": transitions,
        "condition_metrics": condition_metrics,
        "paired_metrics": paired,
        "rows": rows,
    }
    write_json(args.audit_output, private_payload)
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "remote_output_uri": args.remote_output_uri,
        "checksum_verified": True,
        "checksum_file_sha256": sha256(output_root / "SHA256SUMS"),
        "verified_remote_file_count": len(verified_files),
        "label_policy": "labels joined locally after remote prediction freeze",
        "panel": {
            "annotation_files": 481,
            "local_media_rows": 475,
            "inference_rows": len(rows),
        },
        "conditions": [FRAMES_ONLY, FRAMES_WHISPER],
        "condition_metrics": condition_metrics,
        "frame_artifact": {
            "evidence_file": "frame-evidence.json",
            "contact_sheet_directory": "frame-contact-sheets/",
            "frames_per_video": 12,
            "selected_pts_recorded": True,
            "per_frame_sha256_recorded": True,
        },
        "paired_metrics": paired,
        "transition_counts": transitions,
        "wrong_row_audit": {
            "row_count": len(wrong_rows),
            "private_audit_json": str(args.audit_output),
            "private_audit_html": str(args.html_output),
            "answerability_review_status": "pending_manual_review",
        },
    }
    write_json(args.summary_output, summary)
    render_html(args.html_output, private_payload)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
