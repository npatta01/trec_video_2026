"""Build a label-free panel for the oracle-transcript diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

DEFAULT_SELECTED_IDS = (
    "LKiaCf1QZFw",
    "Ie-KCtcZ7uc",
    "J1CS47pBX_M",
    "Mi-NpnCBegw",
    "NN7G28sBOlY",
    "HD2_qedLUTE",
    "LSz_xBDBPNc",
    "NW3Mhr76ClM",
    "NrB4LPosWBk",
    "QgzFuzktegE",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_for(video_dir: Path, video_id: str) -> Path:
    candidates = sorted(
        path
        for path in video_dir.glob(f"{video_id}.*")
        if path.is_file() and not path.name.endswith(".info.json")
    )
    if not candidates:
        raise ValueError(f"missing media for video ID {video_id}")
    if len(candidates) > 1:
        rendered = ", ".join(path.name for path in candidates)
        raise ValueError(f"ambiguous media for video ID {video_id}: {rendered}")
    return candidates[0]


def build_oracle_panel(
    annotation_dir: Path,
    video_dir: Path,
    selected_ids: Sequence[str],
) -> dict:
    """Return a deterministic manifest containing questions but no answer labels."""
    selected = list(selected_ids)
    if len(selected) != 10:
        raise ValueError(
            f"oracle panel requires exactly 10 IDs, received {len(selected)}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate video IDs are not allowed")

    examples = []
    for video_id in selected:
        media_path = _media_for(video_dir, video_id)
        annotation_path = annotation_dir / f"{video_id}.json"
        if not annotation_path.is_file():
            raise ValueError(f"missing annotation for video ID {video_id}")

        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        question = annotation.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"annotation for {video_id} lacks a question")

        language_guess = annotation.get(
            "language_guess", annotation.get("source_language", "und")
        )
        if not isinstance(language_guess, str) or not language_guess.strip():
            language_guess = "und"

        examples.append(
            {
                "video_id": video_id,
                "question": question.strip(),
                "media_path": str(media_path),
                "media_sha256": _sha256(media_path),
                "language_guess": language_guess.strip(),
                "operational_tag": "speech_content",
            }
        )

    return {
        "schema_version": 1,
        "panel_id": "oracle-speech-10",
        "label_access": "forbidden",
        "examples": examples,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selected-id",
        action="append",
        dest="selected_ids",
        help="Video ID to select; repeat exactly ten times",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected_ids = args.selected_ids or DEFAULT_SELECTED_IDS
    manifest = build_oracle_panel(args.annotations, args.videos, selected_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest['examples'])} label-free examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
