"""Frozen request and result contracts for joint vision/audio probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PANEL_ID = "omni-compatibility-3"
REQUIRED_ROLES = ("lexical_speech", "prosody", "non_speech_sound")
HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class PanelSelection:
    video_id: str
    operational_tag: str
    media_path: str
    audio_path: str
    frame_dir: str
    frame_timestamps: tuple[float, ...]


DEFAULT_PANEL = (
    PanelSelection(
        video_id="NN7G28sBOlY",
        operational_tag="lexical_speech",
        media_path="vqa/2025/training/videos_original/NN7G28sBOlY.mkv",
        audio_path=(
            "vqa/benchmarks/runs/native-audio-benchmark-v2/audio/NN7G28sBOlY.wav"
        ),
        frame_dir=(
            "vqa/benchmarks/evidence/"
            "58ccb3f134e01202658b5495112cc1985207b4e12c225a262dc82ed47531861a"
        ),
        frame_timestamps=(
            1.1545,
            3.4635,
            5.7725,
            8.0815,
            10.3905,
            12.6995,
            15.0085,
            17.3175,
            19.6265,
            21.9355,
            24.2445,
            26.5535,
        ),
    ),
    PanelSelection(
        video_id="NUmb2P3Bzu8",
        operational_tag="prosody",
        media_path="vqa/2025/training/videos_original/NUmb2P3Bzu8.mkv",
        audio_path=(
            "vqa/benchmarks/runs/native-audio-benchmark-v2/audio/NUmb2P3Bzu8.wav"
        ),
        frame_dir=(
            "vqa/benchmarks/evidence/"
            "a3a26cce16ff883114ce4ec6f589fd9409917345048ead7ecc68568208ff22ee"
        ),
        frame_timestamps=(
            1.241167,
            3.7235,
            6.205833,
            8.688167,
            11.1705,
            13.652833,
            16.135167,
            18.6175,
            21.099833,
            23.582167,
            26.0645,
            28.546833,
        ),
    ),
    PanelSelection(
        video_id="NkXPPJoZ-h8",
        operational_tag="non_speech_sound",
        media_path="vqa/2025/training/videos_original/NkXPPJoZ-h8.mkv",
        audio_path=(
            "vqa/benchmarks/runs/native-audio-benchmark-v2/audio/NkXPPJoZ-h8.wav"
        ),
        frame_dir=(
            "vqa/benchmarks/evidence/"
            "8a065fc8f32273d1f81b58cf8773fda8790d3953f8223c02dee3951f5a53d5b4"
        ),
        frame_timestamps=(
            0.568667,
            1.706,
            2.843333,
            3.980667,
            5.118,
            6.255333,
            7.392667,
            8.53,
            9.667333,
            10.804667,
            11.942,
            13.079333,
        ),
    ),
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and set(value).issubset(HEX_DIGITS)
    )


def _is_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and set(value).issubset(HEX_DIGITS)
    )


def _resolve_relative(data_root: Path, value: str, field: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must be a safe data-root-relative path")
    return data_root.joinpath(*relative.parts)


def build_compatibility_contract(
    data_root: Path,
    selections: Sequence[PanelSelection] = DEFAULT_PANEL,
) -> dict[str, Any]:
    """Build a deterministic, answer-free contract bound to source evidence."""

    data_root = data_root.resolve()
    examples: list[dict[str, Any]] = []
    for selection in selections:
        annotation_path = (
            data_root / "vqa/2025/training/annotations" / f"{selection.video_id}.json"
        )
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        question = annotation.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{selection.video_id}: question is required")

        media_path = _resolve_relative(data_root, selection.media_path, "media_path")
        audio_path = _resolve_relative(data_root, selection.audio_path, "audio_path")
        frame_dir = _resolve_relative(data_root, selection.frame_dir, "frame_dir")
        frame_paths = [frame_dir / f"frame-{index:02d}.png" for index in range(1, 13)]
        examples.append(
            {
                "video_id": selection.video_id,
                "operational_tag": selection.operational_tag,
                "question": question.strip(),
                "media_path": selection.media_path,
                "media_sha256": _sha256_file(media_path),
                "audio_path": selection.audio_path,
                "audio_sha256": _sha256_file(audio_path),
                "frame_paths": [
                    str(path.relative_to(data_root)) for path in frame_paths
                ],
                "frame_sha256s": [_sha256_file(path) for path in frame_paths],
                "frame_timestamps": list(selection.frame_timestamps),
            }
        )

    contract = {
        "schema_version": 1,
        "panel_id": PANEL_ID,
        "evidence_policy": (
            "Each request supplies all 12 frozen sampled frames and the native "
            "audio waveform to one model generation call."
        ),
        "examples": examples,
    }
    validate_contract(contract, data_root, verify_files=True)
    return contract


def validate_contract(
    contract: dict[str, Any],
    data_root: Path,
    *,
    verify_files: bool = False,
) -> None:
    if contract.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if contract.get("panel_id") != PANEL_ID:
        raise ValueError(f"panel_id must be {PANEL_ID}")
    examples = contract.get("examples")
    if not isinstance(examples, list) or len(examples) != 3:
        raise ValueError("examples must contain exactly three rows")
    roles = tuple(row.get("operational_tag") for row in examples)
    if roles != REQUIRED_ROLES:
        raise ValueError(f"operational_tag roles must be {REQUIRED_ROLES}")
    video_ids = [row.get("video_id") for row in examples]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("video_id values must be unique")

    data_root = data_root.resolve()
    for row in examples:
        video_id = row.get("video_id", "<unknown>")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"{video_id}: question is required")
        for field in ("media_sha256", "audio_sha256"):
            if not _is_sha256(row.get(field)):
                raise ValueError(f"{video_id}: {field} must be a SHA-256")
        frame_paths = row.get("frame_paths")
        frame_hashes = row.get("frame_sha256s")
        timestamps = row.get("frame_timestamps")
        if not isinstance(frame_paths, list) or len(frame_paths) != 12:
            raise ValueError(f"{video_id}: frame_paths must contain 12 paths")
        if (
            not isinstance(frame_hashes, list)
            or len(frame_hashes) != 12
            or not all(_is_sha256(item) for item in frame_hashes)
        ):
            raise ValueError(
                f"{video_id}: frame_sha256s must contain 12 SHA-256 values"
            )
        if not isinstance(timestamps, list) or len(timestamps) != 12:
            raise ValueError(f"{video_id}: frame_timestamps must contain 12 values")

        path_hash_pairs = [
            ("media_path", row["media_path"], row["media_sha256"]),
            ("audio_path", row["audio_path"], row["audio_sha256"]),
            *[
                ("frame_paths", path, digest)
                for path, digest in zip(frame_paths, frame_hashes, strict=True)
            ],
        ]
        for field, relative, expected_hash in path_hash_pairs:
            path = _resolve_relative(data_root, relative, field)
            if verify_files:
                if not path.is_file():
                    raise ValueError(f"{video_id}: missing {field}: {relative}")
                if _sha256_file(path) != expected_hash:
                    raise ValueError(
                        f"{video_id}: {field} hash does not match {relative}"
                    )


def contract_sha256(contract: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(contract)).hexdigest()


def request_sha256(example: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(example)).hexdigest()


def parse_prediction(raw_output: str) -> dict[str, str]:
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json\n"):
                text = text.lstrip()[5:]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("prediction must be a JSON object") from error
    if not isinstance(parsed, dict):
        raise ValueError("prediction must be a JSON object")  # noqa: TRY004
    answer = parsed.get("answer")
    evidence = parsed.get("evidence")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("prediction answer must be non-empty text")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("prediction evidence must be non-empty text")
    return {"answer": answer.strip(), "evidence": evidence.strip()}


def validate_prediction(
    prediction: dict[str, Any], expected_example: dict[str, Any]
) -> None:
    required = (
        "schema_version",
        "run_id",
        "video_id",
        "operational_tag",
        "request_sha256",
        "model_id",
        "model_revision",
        "runtime_image_digest",
        "device",
        "precision",
        "elapsed_seconds",
        "peak_memory_bytes",
        "status",
        "raw_output",
        "output_parse_status",
        "input_contract_error",
        "modality_attestation",
    )
    for field in required:
        if field not in prediction:
            raise ValueError(f"{field} is required")
    if prediction["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for field in ("video_id", "operational_tag"):
        if prediction[field] != expected_example[field]:
            raise ValueError(f"{field} does not match the request")
    if prediction["request_sha256"] != request_sha256(expected_example):
        raise ValueError("request_sha256 does not match the request")
    if not _is_revision(prediction["model_revision"]):
        raise ValueError("model_revision must be a 40- or 64-character revision")
    digest = prediction["runtime_image_digest"]
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or not _is_sha256(digest.removeprefix("sha256:"))
    ):
        raise ValueError("runtime_image_digest must be sha256:<64 hex>")
    for field in ("model_id", "device", "precision", "run_id"):
        if not isinstance(prediction[field], str) or not prediction[field]:
            raise ValueError(f"{field} must be non-empty text")
    if not isinstance(prediction["elapsed_seconds"], (int, float)):
        raise ValueError("elapsed_seconds must be numeric")  # noqa: TRY004
    if (
        not isinstance(prediction["peak_memory_bytes"], int)
        or prediction["peak_memory_bytes"] < 0
    ):
        raise ValueError("peak_memory_bytes must be a non-negative integer")

    if prediction["status"] == "passed":
        if prediction["output_parse_status"] != "passed":
            raise ValueError("passed prediction must have parsed output")
        if prediction.get("parsed_output") != parse_prediction(
            prediction["raw_output"]
        ):
            raise ValueError("parsed_output does not match raw_output")
        if prediction["input_contract_error"] is not False:
            raise ValueError("passed prediction cannot have an input contract error")
        attestation = prediction["modality_attestation"]
        if not attestation.get("same_request"):
            raise ValueError("vision and audio must be in the same request")
        if not attestation.get("same_forward_pass"):
            raise ValueError("vision and audio must be in the same forward pass")
        if attestation.get("generation_call_count") != 1:
            raise ValueError("exactly one joint generation call is required")
        if attestation.get("vision_element_count", 0) != 12:
            raise ValueError("vision_element_count must be 12")
        if attestation.get("audio_element_count", 0) != 1:
            raise ValueError("audio_element_count must be 1")
        for field in ("vision_input_sha256", "audio_input_sha256"):
            if not _is_sha256(attestation.get(field)):
                raise ValueError(f"{field} must attest the processed input")
        generation_keys = set(attestation.get("generation_input_keys", []))
        vision_keys = set(attestation.get("vision_input_keys", []))
        audio_keys = set(attestation.get("audio_input_keys", []))
        if not vision_keys or not vision_keys.issubset(generation_keys):
            raise ValueError("vision inputs must be present in generation inputs")
        if not audio_keys or not audio_keys.issubset(generation_keys):
            raise ValueError("audio inputs must be present in generation inputs")


def promotion_decision(
    contract: dict[str, Any], predictions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    expected_by_role = {row["operational_tag"]: row for row in contract["examples"]}
    rows_by_role = {row.get("operational_tag"): row for row in predictions}
    failed_roles: list[str] = []
    reasons: dict[str, str] = {}
    for role in REQUIRED_ROLES:
        prediction = rows_by_role.get(role)
        if prediction is None:
            failed_roles.append(role)
            reasons[role] = "missing result"
            continue
        try:
            validate_prediction(prediction, expected_by_role[role])
        except ValueError as error:
            failed_roles.append(role)
            reasons[role] = str(error)
    return {
        "decision": "promote" if not failed_roles else "stop",
        "failed_roles": failed_roles,
        "reasons": reasons,
    }


def write_contract(contract: dict[str, Any], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = contract_sha256(contract)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the three-row joint vision/audio compatibility panel."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = build_compatibility_contract(args.data_root)
    digest = write_contract(contract, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "contract_sha256": digest,
                "examples": len(contract["examples"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
