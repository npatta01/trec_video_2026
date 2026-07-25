"""Build hash-bound requests for the three oracle-transcript conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .automatic_transcript import (
    select_automatic_transcript_text,
    validate_automatic_transcript,
    validate_dataset_recording_prompt_reference,
    validate_human_caption_reference,
)
from .transcript_contract import validate_oracle_transcript

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RUNTIME_IMAGE = (
    "localhost/trec-vqa-gate-rocm"
    "@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5"
)
FRAME_COUNT = 12
SEED = 20260724
IMAGE_RESOLUTION = {"max_pixels": 512 * 512, "min_pixels": 256 * 256}
DECODING = {
    "do_sample": False,
    "max_new_tokens": 32,
    "temperature": 0.0,
    "top_p": 1.0,
}
PROMPT_TEMPLATE = """Use the chronological frame evidence and optional timestamped
audio transcript. Question: {question}
Answer with one short phrase or sentence. Return only the answer; do not explain
your reasoning.
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


def _validated_frames(
    frames: object, video_id: str
) -> tuple[list[str], list[str], list[float]]:
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise ValueError(f"{video_id} must have exactly 12 frames")
    paths = []
    hashes = []
    timestamps = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise TypeError(f"{video_id} frame {index} must be an object")
        path = frame.get("path")
        sha256 = frame.get("sha256")
        timestamp = frame.get("timestamp_seconds")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{video_id} frame {index} lacks a path")
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            raise ValueError(f"{video_id} frame {index} has invalid SHA-256")
        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or timestamp < 0
        ):
            raise ValueError(f"{video_id} frame {index} has invalid timestamp")
        paths.append(path)
        hashes.append(sha256)
        timestamps.append(float(timestamp))
    if len(set(hashes)) != FRAME_COUNT:
        raise ValueError(f"{video_id} frame SHA-256 values must be unique")
    if timestamps != sorted(timestamps) or len(set(timestamps)) != FRAME_COUNT:
        raise ValueError(f"{video_id} frame timestamps must be unique and ordered")
    return paths, hashes, timestamps


def _automatic_text(record: object, video_id: str, media_sha256: str) -> str:
    if not isinstance(record, dict):
        raise TypeError(f"{video_id} automatic transcript must be an object")
    if record.get("video_id") != video_id:
        raise ValueError(f"{video_id} automatic transcript has mismatched video_id")
    validate_automatic_transcript(record, media_sha256)
    return select_automatic_transcript_text(record)


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


def _dataset_reference_text(record: object, video_id: str, media_sha256: str) -> str:
    if not isinstance(record, dict):
        raise TypeError(f"{video_id} dataset reference must be an object")
    if record.get("video_id") != video_id:
        raise ValueError(f"{video_id} dataset reference has mismatched video_id")
    validate_dataset_recording_prompt_reference(record, media_sha256)
    return record["text"].strip()


def _human_caption_text(record: object, video_id: str, media_sha256: str) -> str:
    if not isinstance(record, dict):
        raise TypeError(f"{video_id} human-caption reference must be an object")
    if record.get("video_id") != video_id:
        raise ValueError(f"{video_id} human-caption reference has mismatched video_id")
    validate_human_caption_reference(record, media_sha256)
    return record["text"].strip()


def _request(payload: dict) -> dict:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return payload | {
        "canonical_request": canonical,
        "request_sha256": _sha256_text(canonical),
    }


def _validate_request_identity(request: dict) -> None:
    canonical = request.get("canonical_request")
    request_sha256 = request.get("request_sha256")
    if not isinstance(canonical, str):
        raise ValueError("canonical_request must be a string")
    live_payload = {
        key: value
        for key, value in request.items()
        if key not in {"canonical_request", "request_sha256"}
    }
    expected_canonical = json.dumps(live_payload, sort_keys=True, separators=(",", ":"))
    if canonical != expected_canonical:
        raise ValueError("canonical_request does not match the live request payload")
    if request_sha256 != _sha256_text(canonical):
        raise ValueError("request_sha256 does not match canonical_request")


def _build_conditions(
    manifest: dict,
    frames: dict,
    automatic: dict,
    oracle: dict | None,
    dataset_reference: dict | None = None,
    human_caption_reference: dict | None = None,
) -> list[dict]:
    optional_tiers = (oracle, dataset_reference, human_caption_reference)
    if sum(tier is not None for tier in optional_tiers) > 1:
        raise ValueError("reference transcript tiers are mutually exclusive")
    _reject_label_keys(manifest, "manifest")
    _reject_label_keys(automatic, "automatic transcripts")
    if oracle is not None:
        _reject_label_keys(oracle, "oracle transcripts")
    if dataset_reference is not None:
        _reject_label_keys(dataset_reference, "dataset references")
    if human_caption_reference is not None:
        _reject_label_keys(human_caption_reference, "human-caption references")

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

        frame_paths, frame_sha256s, frame_timestamps = _validated_frames(
            frames.get(video_id), video_id
        )
        automatic_text = _automatic_text(
            automatic.get(video_id), video_id, media_sha256
        )
        rendered_prompt = PROMPT_TEMPLATE.format(question=question.strip())
        prompt_sha256 = _sha256_text(rendered_prompt)

        conditions = [
            ("frames_only", None),
            ("frames_automatic_transcript", automatic_text),
        ]
        if oracle is not None:
            conditions.append(
                (
                    "frames_oracle_transcript",
                    _oracle_text(oracle.get(video_id), video_id, media_sha256),
                )
            )
        if dataset_reference is not None:
            conditions.append(
                (
                    "frames_dataset_reference",
                    _dataset_reference_text(
                        dataset_reference.get(video_id), video_id, media_sha256
                    ),
                )
            )
        if human_caption_reference is not None:
            conditions.append(
                (
                    "frames_human_caption_reference",
                    _human_caption_text(
                        human_caption_reference.get(video_id),
                        video_id,
                        media_sha256,
                    ),
                )
            )
        for condition, transcript in conditions:
            requests.append(
                _request(
                    {
                        "run_id": f"{manifest.get('panel_id')}-matched-v1",
                        "panel_id": manifest.get("panel_id"),
                        "video_id": video_id,
                        "condition": condition,
                        "condition_id": condition,
                        "question": question.strip(),
                        "model_id": MODEL_ID,
                        "model_revision": MODEL_REVISION,
                        "runtime_image": RUNTIME_IMAGE,
                        "media_path": example.get("media_path"),
                        "audio_path": None,
                        "audio_mode": None,
                        "media_sha256": media_sha256,
                        "frame_paths": frame_paths,
                        "frame_sha256s": frame_sha256s,
                        "frame_timestamps": frame_timestamps,
                        "frame_count": FRAME_COUNT,
                        "image_resolution": IMAGE_RESOLUTION,
                        "prompt": rendered_prompt,
                        "prompt_sha256": prompt_sha256,
                        "transcript": transcript,
                        "seed": SEED,
                        "decoding": DECODING,
                        "output_schema": {"answer": "string"},
                        "score_field": "answer",
                    }
                )
            )

    return requests


def build_automatic_ablation(
    manifest: dict,
    frames: dict,
    automatic: dict,
) -> list[dict]:
    """Return frames-only and matched automatic-transcript requests."""
    return _build_conditions(manifest, frames, automatic, None)


def build_matched_conditions(
    manifest: dict,
    frames: dict,
    automatic: dict,
    oracle: dict,
) -> list[dict]:
    """Return three label-free requests per example with matched visual settings."""
    return _build_conditions(manifest, frames, automatic, oracle)


def build_dataset_reference_conditions(
    manifest: dict,
    frames: dict,
    automatic: dict,
    dataset_reference: dict,
) -> list[dict]:
    """Return three matched conditions for a controlled-corpus reference."""
    return _build_conditions(
        manifest,
        frames,
        automatic,
        oracle=None,
        dataset_reference=dataset_reference,
    )


def build_human_caption_reference_conditions(
    manifest: dict,
    frames: dict,
    automatic: dict,
    human_caption_reference: dict,
) -> list[dict]:
    """Return three matched conditions for verified uploader-caption evidence."""
    return _build_conditions(
        manifest,
        frames,
        automatic,
        oracle=None,
        human_caption_reference=human_caption_reference,
    )


def normalize_runner_predictions(
    requests: Sequence[dict], raw_rows: Sequence[dict]
) -> list[dict]:
    """Bind runner outputs to requests and validate immutable provenance."""
    for request in requests:
        _validate_request_identity(request)
    request_index = {
        (request["video_id"], request["condition_id"]): request for request in requests
    }
    if len(request_index) != len(requests):
        raise ValueError("requests contain duplicate video/condition pairs")
    normalized = {}
    for raw in raw_rows:
        _reject_keys(raw, _FORBIDDEN_PREDICTION_KEYS, "raw runner result")
        key = (raw.get("video_id"), raw.get("condition_id"))
        if key in normalized:
            raise ValueError(f"duplicate runner result pair: {key}")
        request = request_index.get(key)
        if request is None:
            raise ValueError(f"runner result pair is not present in requests: {key}")
        for field in (
            "run_id",
            "model_id",
            "model_revision",
            "runtime_image",
            "request_sha256",
        ):
            if raw.get(field) != request.get(field):
                raise ValueError(f"{field} mismatch for {key}")
        if raw.get("status") != "passed":
            raise ValueError(f"runner result did not pass for {key}")
        answer = raw.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"runner answer must be nonempty for {key}")
        if raw.get("device") != "cuda:0 (ROCm)":
            raise ValueError(f"device attestation mismatch for {key}")
        if raw.get("precision") != "bfloat16":
            raise ValueError(f"precision attestation mismatch for {key}")
        normalized[key] = {
            "video_id": request["video_id"],
            "condition": request["condition"],
            "request_sha256": request["request_sha256"],
            "status": "success",
            "output": {"answer": answer.strip(), "evidence_quote": ""},
            "runner_provenance": {
                field: raw.get(field)
                for field in (
                    "model_id",
                    "model_revision",
                    "runtime",
                    "runtime_image",
                    "device",
                    "precision",
                    "runtime_attestation",
                    "peak_memory_bytes",
                    "request_sha256",
                )
            },
        }
    if set(normalized) != set(request_index):
        missing = sorted(set(request_index) - set(normalized))
        raise ValueError(f"runner result set is incomplete; missing {missing}")
    return [
        normalized[(request["video_id"], request["condition_id"])]
        for request in requests
    ]


def freeze_predictions(requests: Sequence[dict], predictions: Sequence[dict]) -> dict:
    """Validate and hash a complete prediction set before any labels are loaded."""
    allowed_condition_sets = {
        frozenset({"frames_only", "frames_automatic_transcript"}),
        frozenset(
            {
                "frames_only",
                "frames_automatic_transcript",
                "frames_oracle_transcript",
            }
        ),
        frozenset(
            {
                "frames_only",
                "frames_automatic_transcript",
                "frames_dataset_reference",
            }
        ),
        frozenset(
            {
                "frames_only",
                "frames_automatic_transcript",
                "frames_human_caption_reference",
            }
        ),
    }
    expected_conditions: frozenset[str] | None = None
    request_index = {}
    for request in requests:
        _validate_request_identity(request)
        key = (request.get("video_id"), request.get("condition"))
        if key in request_index:
            raise ValueError(f"duplicate request pair: {key}")
        request_index[key] = request
    if not request_index:
        raise ValueError("cannot freeze an empty request set")
    for video_id, rows in group_by_video(requests).items():
        conditions = frozenset(row.get("condition") for row in rows)
        if conditions not in allowed_condition_sets:
            raise ValueError(f"{video_id} has an unsupported condition set")
        if expected_conditions is None:
            expected_conditions = conditions
        if conditions != expected_conditions:
            raise ValueError(f"{video_id} does not have the matched condition set")

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


def verify_prediction_freeze(
    requests: Sequence[dict],
    predictions: Sequence[dict],
    freeze: dict,
) -> None:
    """Reject request, prediction, or freeze mutation before label access."""
    if not isinstance(freeze, dict):
        raise TypeError("freeze must be an object")
    expected = freeze_predictions(requests, predictions)
    if freeze != expected:
        raise ValueError("prediction freeze does not match live requests/predictions")


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
