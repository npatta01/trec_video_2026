#!/usr/bin/env python3
"""Prepare and score private Nemotron frozen-30 grading artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from trec_video_2026.vqa_experiments.nemotron3_omni_frozen30 import (
    build_blind_grading_bundle,
    derive_operational_evidence,
    reconcile_frozen30_grades,
    score_replacement,
    validate_private_grader_assignments,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def require_private_output_directory(path: Path) -> Path:
    """Reject raw grading outputs anywhere inside the Git repository."""

    resolved = path.expanduser().resolve()
    if resolved == REPOSITORY_ROOT or resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("private output directory must be outside the repository")
    existing = resolved
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    result = subprocess.run(
        [
            "git",
            "-C",
            str(existing),
            "rev-parse",
            "--is-inside-work-tree",
            "--is-inside-git-dir",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and "true" in result.stdout.splitlines():
        raise ValueError(
            "private output directory must be outside every Git repository"
        )
    return resolved


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qwen-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-checksums", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-media-root", type=Path, required=True)
    parser.add_argument("--normalized-media-root", type=Path, required=True)
    parser.add_argument("--audit-bundle-root", type=Path, required=True)
    parser.add_argument("--input-bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grader-id", action="append", required=True)
    parser.add_argument("--shuffle-seed", action="append", type=int, required=True)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify, blindly grade, and score Nemotron frozen-30 privately"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", parents=[_common_parser()])
    prepare.set_defaults(handler=_prepare)
    score = subparsers.add_parser("score", parents=[_common_parser()])
    score.add_argument("--grader-submission", action="append", type=Path, required=True)
    score.add_argument("--grader-assignments", type=Path, required=True)
    score.add_argument("--adjudication", type=Path)
    score.set_defaults(handler=_score)
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_loader(path: Path):
    def load(keys: tuple[str, ...]) -> list[dict[str, str]]:
        value = _load_json(path)
        rows = value.get("examples") if isinstance(value, dict) else value
        if not isinstance(rows, list):
            raise ValueError("references must be a list or contain examples")
        by_id = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("reference examples must be objects")
            video_id = row.get("row_key", row.get("video_id"))
            reference = row.get("reference", row.get("correct_answer"))
            by_id[video_id] = {
                "row_key": video_id,
                "category": row.get("category"),
                "question": row.get("question"),
                "reference": reference,
            }
        return [by_id[key] for key in keys if key in by_id]

    return load


def _bundle(args: argparse.Namespace):
    if len(args.grader_id) != 2 or len(args.shuffle_seed) != 2:
        raise ValueError("exactly two grader IDs and two shuffle seeds are required")
    return build_blind_grading_bundle(
        qwen_predictions_path=args.qwen_predictions,
        candidate_predictions_path=args.candidate_predictions,
        candidate_manifest_path=args.candidate_manifest,
        candidate_checksums_path=args.candidate_checksums,
        source_media_root=args.source_media_root,
        normalized_media_root=args.normalized_media_root,
        audit_bundle_root=args.audit_bundle_root,
        input_bundle_root=args.input_bundle_root,
        reference_loader=_reference_loader(args.references),
        grader_ids=tuple(args.grader_id),
        shuffle_seeds=tuple(args.shuffle_seed),
        source_manifest_path=args.source_manifest,
    )


def _write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prepare(args: argparse.Namespace) -> int:
    output_dir = require_private_output_directory(args.output_dir)
    bundle = _bundle(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, packet in enumerate(bundle.packets.values(), start=1):
        _write_json(output_dir / f"blind-grading-packet-{index}.json", packet)
    _write_json(
        output_dir / "private-grader-assignments.json",
        bundle._grader_assignment_artifact,
    )
    return 0


def _meteor(prediction: str, reference: str) -> float:
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    class _LexicalWordNet:
        @staticmethod
        def synsets(_word: str) -> tuple[object, ...]:
            return ()

    # Frozen short answers are single-line strings; preserve_line keeps this
    # scorer deterministic without downloading mutable NLTK sentence data. The
    # empty WordNet adapter likewise keeps METEOR lexical and self-contained.
    return float(
        meteor_score(
            [word_tokenize(reference, preserve_line=True)],
            word_tokenize(prediction, preserve_line=True),
            wordnet=_LexicalWordNet(),
        )
    )


def _score(args: argparse.Namespace) -> int:
    output_dir = require_private_output_directory(args.output_dir)
    bundle = _bundle(args)
    validate_private_grader_assignments(bundle, _load_json(args.grader_assignments))
    submissions = [_load_json(path) for path in args.grader_submission]
    if len(submissions) != 2:
        raise ValueError("exactly two grader submissions are required")
    adjudication = _load_json(args.adjudication) if args.adjudication else None
    resolution = reconcile_frozen30_grades(
        bundle, submissions, adjudication=adjudication
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if resolution.status == "adjudication_required":
        _write_json(
            output_dir / "disagreement-only-adjudication.json",
            resolution.adjudication_packet,
        )
        return 2
    report = score_replacement(
        bundle,
        resolution,
        meteor_fn=_meteor,
        operational=derive_operational_evidence(
            candidate_predictions_path=args.candidate_predictions,
            candidate_manifest_path=args.candidate_manifest,
            audit_bundle_root=args.audit_bundle_root,
        ),
    )
    _write_json(output_dir / "private-score-report.json", report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
