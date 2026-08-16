#!/usr/bin/env python3
"""Prepare and execute the immutable local Qwen forced-aligner canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import trec_video_2026.vqa_qwen_aligner_canary as canary_module
from trec_video_2026.vqa_qwen_aligner_canary import (
    build_podman_command,
    freeze_predictions,
    prepare_canary,
    record_attempt,
    require_quality_scoring_ready,
    score_postfreeze,
    select_quality_panel,
    select_panel,
    status_canary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_qwen_aligner_canary")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_experiment_id(command: argparse.ArgumentParser) -> None:
        command.add_argument("--experiment-id")

    select = commands.add_parser("select")
    select.add_argument("--scores", required=True, type=Path)
    select.add_argument("--maps", required=True, type=Path)
    select.add_argument("--transcripts", required=True, type=Path)
    select.add_argument("--prior-qwen", required=True, action="append", type=Path)
    select.add_argument("--review", required=True, type=Path)
    select.add_argument("--audio-root", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--selected-at")
    add_experiment_id(select)

    quality_select = commands.add_parser("quality-select")
    quality_select.add_argument("--oracle-transcripts", required=True, type=Path)
    quality_select.add_argument("--manifest", required=True, type=Path)
    quality_select.add_argument(
        "--prior-qwen", required=True, action="append", type=Path
    )
    quality_select.add_argument("--audio-root", required=True, type=Path)
    quality_select.add_argument("--output", required=True, type=Path)
    quality_select.add_argument("--selected-at")
    quality_select.add_argument("--panel-size", type=int, default=8)
    add_experiment_id(quality_select)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--root", required=True, type=Path)
    prepare.add_argument("--selection", required=True, type=Path)
    prepare.add_argument("--runner", required=True, type=Path)
    add_experiment_id(prepare)

    for name in ("command", "record"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--hub-cache", required=True, type=Path)
        command.add_argument("--data-root", required=True, type=Path)
        add_experiment_id(command)

    for name in ("freeze", "status"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True, type=Path)
        add_experiment_id(command)

    score = commands.add_parser("score")
    score.add_argument("--root", required=True, type=Path)
    score.add_argument("--annotations-dir", required=True, type=Path)
    score.add_argument("--whisper", required=True, type=Path)
    add_experiment_id(score)
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _video_ids(path: Path) -> set[str]:
    raw = path.read_text(encoding="utf-8")
    try:
        payloads = [json.loads(raw)]
    except json.JSONDecodeError:
        payloads = [json.loads(line) for line in raw.splitlines() if line.strip()]
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            video_id = value.get("video_id")
            if isinstance(video_id, str) and video_id:
                found.add(video_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payloads)
    return found


def _reviewed_ids(path: Path) -> set[str]:
    payload = _load_json(path)
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("review rows are missing")
    return {
        str(row["video_id"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("video_id"), str)
    }


def _audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
    except (OSError, wave.Error) as exc:
        raise ValueError(f"cannot read WAV metadata: {path.name}") from exc
    if frame_rate <= 0 or frame_count <= 0:
        raise ValueError(f"WAV duration is invalid: {path.name}")
    return frame_count / frame_rate


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def _score_inputs(
    root: Path, annotations_dir: Path, whisper_path: Path
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    status = status_canary(root)
    if status["status"] != "predictions_frozen":
        raise ValueError("prediction freeze must verify before reading score inputs")
    if status.get("decision") != "ready_for_quality_test":
        raise ValueError("predictions are not ready for quality scoring")
    require_quality_scoring_ready(root)
    results = _load_json(root / "raw-results.json")
    if not isinstance(results, list):
        raise ValueError("raw results are not an array")
    ids = [row.get("video_id") for row in results if isinstance(row, dict)]
    if not ids or len(ids) != len(results) or any(
        not isinstance(video_id, str) for video_id in ids
    ):
        raise ValueError("raw result IDs are invalid")
    references: dict[str, str] = {}
    for video_id in ids:
        annotation = _load_json(annotations_dir / f"{video_id}.json")
        answer = annotation.get("correct_answer") if isinstance(annotation, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("annotation answer is invalid")
        references[video_id] = answer
    whisper_payload = _load_json(whisper_path)
    if isinstance(whisper_payload, dict):
        whisper_rows = whisper_payload.get("results")
    elif isinstance(whisper_payload, list):
        whisper_rows = whisper_payload
    else:
        whisper_rows = None
    if not isinstance(whisper_rows, list):
        raise ValueError("Whisper results are missing")
    by_id = {
        row.get("video_id"): row
        for row in whisper_rows
        if isinstance(row, dict) and isinstance(row.get("video_id"), str)
    }
    comparison = {}
    for video_id in ids:
        row = by_id.get(video_id)
        if isinstance(row, dict):
            quality = row.get("original_quality") or row.get("quality")
            text = row.get("text")
            if not isinstance(text, str):
                original = row.get("original_language_transcript")
                if isinstance(original, dict):
                    text = original.get("text")
        else:
            quality = None
            text = None
        comparison[video_id] = {
            "quality_status": quality.get("status") if isinstance(quality, dict) else None,
            "text": text,
        }
    return references, comparison


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.experiment_id is not None:
            canary_module.configure_experiment_id(args.experiment_id)
        if args.command == "select":
            prior_ids: set[str] = set()
            for path in args.prior_qwen:
                prior_ids.update(_video_ids(path))
            sources = [
                args.scores,
                args.maps,
                args.transcripts,
                *args.prior_qwen,
                args.review,
            ]
            selection = select_panel(
                scores=_load_json(args.scores),
                maps=_load_json(args.maps),
                transcripts=_load_json(args.transcripts),
                prior_qwen_ids=prior_ids,
                reviewed_failure_ids=_reviewed_ids(args.review),
                audio_root=args.audio_root,
                selected_at=args.selected_at
                or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                source_hashes={f"source_{index}_sha256": _sha256(path) for index, path in enumerate(sources)},
            )
            _exclusive_json(args.output, selection)
            _print({"status": "selected", "row_count": len(selection["rows"])})
            return 0
        if args.command == "quality-select":
            manifest = _load_json(args.manifest)
            examples = manifest.get("examples") if isinstance(manifest, dict) else None
            if not isinstance(examples, list):
                raise ValueError("manifest examples are missing")
            oracle = _load_json(args.oracle_transcripts)
            if not isinstance(oracle, list):
                raise ValueError("oracle transcript rows are missing")
            oracle_by_id = {
                row.get("video_id"): row
                for row in oracle
                if isinstance(row, dict) and isinstance(row.get("video_id"), str)
            }
            candidates: list[dict[str, Any]] = []
            for example in examples:
                if not isinstance(example, dict):
                    raise ValueError("manifest example is invalid")
                video_id = example.get("video_id")
                row = oracle_by_id.get(video_id)
                if not isinstance(video_id, str) or not isinstance(row, dict):
                    continue
                audio_path = (args.audio_root / f"{video_id}.wav").resolve()
                candidates.append(
                    {
                        "video_id": video_id,
                        "source_language": row.get("source_language"),
                        "whisper_status": (
                            row.get("original_quality", {}).get("status")
                            if isinstance(row.get("original_quality"), dict)
                            else None
                        ),
                        "duration_seconds": _audio_duration(audio_path),
                    }
                )
            prior_ids: set[str] = set()
            for path in args.prior_qwen:
                prior_ids.update(_video_ids(path))
            sources = [args.manifest, args.oracle_transcripts, *args.prior_qwen]
            selection = select_quality_panel(
                candidates=candidates,
                prior_qwen_ids=prior_ids,
                audio_root=args.audio_root,
                selected_at=args.selected_at
                or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                source_hashes={
                    f"source_{index}_sha256": _sha256(path)
                    for index, path in enumerate(sources)
                },
                panel_size=args.panel_size,
            )
            _exclusive_json(args.output, selection)
            _print({"status": "selected", "row_count": len(selection["rows"])})
            return 0
        if args.command == "prepare":
            freeze = prepare_canary(
                args.root,
                _load_json(args.selection),
                runner_path=args.runner,
            )
            _print({"status": freeze["status"], "artifacts_sha256": freeze["artifacts_sha256"]})
            return 0
        if args.command in {"command", "record"}:
            command = build_podman_command(
                args.root,
                hub_cache=args.hub_cache,
                data_root=args.data_root,
            )
            if args.command == "command":
                _print({"status": "command_ready", "argv": command})
            else:
                terminal = record_attempt(args.root, command)
                _print({"status": terminal["status"], "attempt_index": 1})
            return 0
        if args.command == "freeze":
            freeze = freeze_predictions(args.root)
            _print({"status": freeze["status"], "decision": freeze["decision"]})
            return 0
        if args.command == "score":
            references, whisper = _score_inputs(
                args.root, args.annotations_dir, args.whisper
            )
            score = score_postfreeze(
                args.root, references=references, whisper=whisper
            )
            _print(
                {
                    "status": score["status"],
                    "operational_decision": score["operational_decision"],
                }
            )
            return 0
        if args.command == "status":
            _print(status_canary(args.root))
            return 0
        raise ValueError("unknown command")
    except Exception:
        print('{"status":"error","error":"validation failed"}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
