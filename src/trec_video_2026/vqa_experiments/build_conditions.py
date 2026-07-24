"""Build hash-bound requests for the three oracle-transcript conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .transcript_contract import validate_oracle_transcript

MODEL_ID = "Qwen/Qwen3.5-4B"
FRAME_COUNT = 12
SEED = 20260724
IMAGE_RESOLUTION = {"max_pixels": 1003520, "min_pixels": 50176}
DECODING = {
    "do_sample": False,
    "max_new_tokens": 96,
    "temperature": 0.0,
    "top_p": 1.0,
}
PROMPT_TEMPLATE = """Use only the supplied frame evidence and optional transcript.
Answer the question concisely. Return one JSON object with exactly two string
fields: "answer" and "evidence_quote". If a transcript is supplied, quote the
smallest span that supports the answer; otherwise use an empty evidence_quote.
Question: {question}
"""
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_LABEL_KEYS = {
    "answer",
    "correct_answer",
    "incorrect_answers",
    "reference_answer",
    "references",
    "prediction",
    "predictions",
}
_FORBIDDEN_PREDICTION_KEYS = {
    "correct_answer",
    "incorrect_answers",
    "reference_answer",
    "references",
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _reject_keys(value: object, forbidden: set[str], location: str) -> None:
    if isinstance(value, dict):
        leaked = forbidden.intersection(value)
        if leaked:
            raise ValueError(
                f"label leakage in {location}: forbidden key {min(leaked)}"
            )
        for key, child in value.items():
            _reject_keys(child, forbidden, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_keys(child, forbidden, f"{location}[{index}]")


def _reject_label_keys(value: object, location: str = "input") -> None:
    _reject_keys(value, _FORBIDDEN_LABEL_KEYS, location)


def group_by_video(requests: Sequence[dict]) -> dict[str, list[dict]]:
    """Group matched requests by their frozen video ID."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        grouped[request["video_id"]].append(request)
    return dict(grouped)


def _validated_frames(frames: object, video_id: str) -> tuple[list[str], list[str]]:
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise ValueError(f"{video_id} must have exactly 12 frames")
    paths = []
    hashes = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise TypeError(f"{video_id} frame {index} must be an object")
        path = frame.get("path")
        sha256 = frame.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{video_id} frame {index} lacks a path")
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            raise ValueError(f"{video_id} frame {index} has invalid SHA-256")
        paths.append(path)
        hashes.append(sha256)
    if len(set(hashes)) != FRAME_COUNT:
        raise ValueError(f"{video_id} frame SHA-256 values must be unique")
    return paths, hashes


def _automatic_text(record: object, video_id: str) -> str:
    if not isinstance(record, dict):
        raise TypeError(f"{video_id} automatic transcript must be an object")
    if record.get("model_id") != "openai/whisper-large-v3":
        raise ValueError(f"{video_id} automatic transcript must use Whisper large-v3")
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{video_id} automatic transcript text is empty")
    return text.strip()


def _oracle_text(record: object, video_id: str, media_sha256: str) -> str:
    if not isinstance(record, dict):
        raise TypeError(f"{video_id} oracle transcript must be an object")
    if record.get("video_id") != video_id:
        raise ValueError(f"{video_id} oracle transcript has mismatched video_id")
    validate_oracle_transcript(record, media_sha256)
    return (
        f"Source ({record['source_language']}): {record['source_text']}\n"
        f"English: {record['english_translation']}"
    )


def _request(payload: dict) -> dict:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return payload | {
        "canonical_request": canonical,
        "request_sha256": _sha256_text(canonical),
    }


def build_matched_conditions(
    manifest: dict,
    frames: dict,
    automatic: dict,
    oracle: dict,
) -> list[dict]:
    """Return three label-free requests per example with matched visual settings."""
    _reject_label_keys(manifest, "manifest")
    _reject_label_keys(automatic, "automatic transcripts")
    _reject_label_keys(oracle, "oracle transcripts")

    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("manifest examples must be a nonempty list")

    requests = []
    seen_ids = set()
    for example in examples:
        video_id = example.get("video_id")
        question = example.get("question")
        media_sha256 = example.get("media_sha256")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("manifest example lacks video_id")
        if video_id in seen_ids:
            raise ValueError(f"duplicate video_id in manifest: {video_id}")
        seen_ids.add(video_id)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{video_id} lacks a question")
        if not isinstance(media_sha256, str) or not _SHA256.fullmatch(media_sha256):
            raise ValueError(f"{video_id} has invalid media_sha256")

        frame_paths, frame_sha256s = _validated_frames(frames.get(video_id), video_id)
        automatic_text = _automatic_text(automatic.get(video_id), video_id)
        oracle_text = _oracle_text(oracle.get(video_id), video_id, media_sha256)
        rendered_prompt = PROMPT_TEMPLATE.format(question=question.strip())
        prompt_sha256 = _sha256_text(rendered_prompt)

        for condition, transcript in (
            ("frames_only", None),
            ("frames_automatic_transcript", automatic_text),
            ("frames_oracle_transcript", oracle_text),
        ):
            requests.append(
                _request(
                    {
                        "panel_id": manifest.get("panel_id"),
                        "video_id": video_id,
                        "condition": condition,
                        "model_id": MODEL_ID,
                        "media_sha256": media_sha256,
                        "frame_paths": frame_paths,
                        "frame_sha256s": frame_sha256s,
                        "frame_count": FRAME_COUNT,
                        "image_resolution": IMAGE_RESOLUTION,
                        "prompt": rendered_prompt,
                        "prompt_sha256": prompt_sha256,
                        "transcript": transcript,
                        "seed": SEED,
                        "decoding": DECODING,
                        "output_schema": {
                            "answer": "string",
                            "evidence_quote": "string",
                        },
                        "score_field": "answer",
                    }
                )
            )

    return requests


def freeze_predictions(requests: Sequence[dict], predictions: Sequence[dict]) -> dict:
    """Validate and hash a complete prediction set before any labels are loaded."""
    expected_conditions = {
        "frames_only",
        "frames_automatic_transcript",
        "frames_oracle_transcript",
    }
    request_index = {}
    for request in requests:
        key = (request.get("video_id"), request.get("condition"))
        if key in request_index:
            raise ValueError(f"duplicate request pair: {key}")
        request_index[key] = request
    if not request_index:
        raise ValueError("cannot freeze an empty request set")
    for video_id, rows in group_by_video(requests).items():
        conditions = {row.get("condition") for row in rows}
        if conditions != expected_conditions:
            raise ValueError(f"{video_id} does not have all three matched conditions")

    prediction_index = {}
    for prediction in predictions:
        _reject_keys(
            prediction,
            _FORBIDDEN_PREDICTION_KEYS,
            "prediction label",
        )
        key = (prediction.get("video_id"), prediction.get("condition"))
        if key in prediction_index:
            raise ValueError(f"duplicate prediction pair: {key}")
        request = request_index.get(key)
        if request is None:
            raise ValueError(f"prediction pair is not present in requests: {key}")
        if prediction.get("request_sha256") != request.get("request_sha256"):
            raise ValueError(f"prediction request_sha256 mismatch for {key}")
        if prediction.get("status") != "success":
            raise ValueError(f"prediction did not succeed for {key}")
        output = prediction.get("output")
        if not isinstance(output, dict):
            raise TypeError(f"prediction output must be an object for {key}")
        if set(output) != {"answer", "evidence_quote"}:
            raise ValueError(f"prediction output schema mismatch for {key}")
        if any(not isinstance(output[field], str) for field in output):
            raise ValueError(f"prediction output fields must be strings for {key}")
        prediction_index[key] = prediction

    if set(prediction_index) != set(request_index):
        missing = sorted(set(request_index) - set(prediction_index))
        raise ValueError(f"prediction set is incomplete; missing {missing}")

    ordered_predictions = [
        prediction_index[(request["video_id"], request["condition"])]
        for request in requests
    ]
    canonical_requests = json.dumps(
        [request["request_sha256"] for request in requests],
        separators=(",", ":"),
    )
    canonical_predictions = json.dumps(
        ordered_predictions,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": 1,
        "status": "frozen",
        "label_access": "forbidden_until_after_freeze",
        "row_count": len(ordered_predictions),
        "requests_sha256": _sha256_text(canonical_requests),
        "predictions_sha256": _sha256_text(canonical_predictions),
    }


def _load_transcript_directory(path: Path) -> dict:
    return {
        record_path.stem: json.loads(record_path.read_text(encoding="utf-8"))
        for record_path in sorted(path.glob("*.json"))
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--oracle-transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requests = build_matched_conditions(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.frames.read_text(encoding="utf-8")),
        json.loads(args.automatic.read_text(encoding="utf-8")),
        _load_transcript_directory(args.oracle_transcripts),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(requests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(requests)} matched requests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
