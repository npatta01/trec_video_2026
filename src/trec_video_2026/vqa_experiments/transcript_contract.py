"""Validate independently reviewed, label-blind oracle transcript records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_TEXT = ("source_text", "english_translation")
_REQUIRED_FIELDS = (
    "video_id",
    "source_language",
    "segments",
    "source_audio_sha256",
    "transcriber",
    "reviewer",
    "clip_duration_seconds",
)


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def validate_oracle_transcript(record: dict, media_sha256: str) -> None:
    """Raise ``ValueError`` unless a record meets the oracle transcript contract."""
    for field in _REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"missing required field: {field}")
    for field in ("video_id", "source_language"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string")

    if not _SHA256.fullmatch(media_sha256):
        raise ValueError("expected media SHA-256 is invalid")
    if record["source_audio_sha256"] != media_sha256:
        raise ValueError("source_audio_sha256 does not match frozen media")

    if any(
        not isinstance(record.get(field), str) or not record[field].strip()
        for field in _REQUIRED_TEXT
    ):
        raise ValueError("transcript text and translation must both be nonempty")

    if record.get("transcription_scope") != "verbatim_full_clip":
        raise ValueError(
            "answer-only paraphrases are forbidden; transcribe the full clip"
        )
    if record.get("label_access") != "none":
        raise ValueError("oracle transcripts must be label-blind")
    if record.get("review_status") != "accepted":
        raise ValueError("independent review must be accepted")

    transcriber = record["transcriber"]
    reviewer = record["reviewer"]
    if not isinstance(transcriber, str) or not transcriber.strip():
        raise ValueError("transcriber must be identified")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer must be identified")
    if _normalized(transcriber) == _normalized(reviewer):
        raise ValueError("transcriber and reviewer must be distinct")

    duration = record["clip_duration_seconds"]
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        raise TypeError("clip_duration_seconds must be numeric")
    if duration <= 0:
        raise ValueError("clip_duration_seconds must be positive")

    segments = record["segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError("timestamped segments are required")

    previous_end = 0.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise TypeError(f"segment {index} must be an object")
        start = segment.get("start_seconds")
        end = segment.get("end_seconds")
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            raise TypeError(f"segment {index} start timestamp must be numeric")
        if not isinstance(end, (int, float)) or isinstance(end, bool):
            raise TypeError(f"segment {index} end timestamp must be numeric")
        if start < 0 or end <= start or end > duration:
            raise ValueError(f"segment {index} timestamp is outside clip duration")
        if start < previous_end:
            raise ValueError("segments must be ordered and non-overlapping")
        previous_end = end
        for field in _REQUIRED_TEXT:
            value = segment.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"segment {index} has empty {field}")

    joined_source = " ".join(segment["source_text"] for segment in segments)
    if _normalized(joined_source) != _normalized(record["source_text"]):
        raise ValueError("source transcript does not match timestamped segments")
    joined_translation = " ".join(
        segment["english_translation"] for segment in segments
    )
    if _normalized(joined_translation) != _normalized(record["english_translation"]):
        raise ValueError("translation does not match timestamped segments")


def _record_sha256(record: dict) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    examples = manifest.get("examples", [])
    expected = {
        row["video_id"]: row["media_sha256"]
        for row in examples
        if isinstance(row, dict)
    }

    accepted = []
    errors = []
    if not args.transcripts.is_dir():
        errors.append(f"transcript directory does not exist: {args.transcripts}")
    else:
        files = {path.stem: path for path in args.transcripts.glob("*.json")}
        for video_id, media_sha256 in expected.items():
            path = files.get(video_id)
            if path is None:
                errors.append(f"{video_id}: missing transcript")
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("video_id") != video_id:
                    raise ValueError("video_id does not match transcript filename")
                validate_oracle_transcript(record, media_sha256)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(f"{video_id}: {exc}")
                continue
            accepted.append(
                {
                    "video_id": video_id,
                    "record_sha256": _record_sha256(record),
                    "review_status": "accepted",
                }
            )
        for unexpected in sorted(set(files) - set(expected)):
            errors.append(f"{unexpected}: transcript not present in manifest")

    status = (
        "accepted" if len(accepted) == len(expected) == 10 and not errors else "blocked"
    )
    report = {
        "schema_version": 1,
        "panel_id": manifest.get("panel_id"),
        "status": status,
        "accepted_count": len(accepted),
        "expected_count": len(expected),
        "accepted": accepted,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"transcript review {status}: {len(accepted)}/{len(expected)} accepted; "
        f"wrote {args.output}"
    )
    return 0 if status == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
