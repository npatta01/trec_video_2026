"""Label-blind native audio-video pilot for NVIDIA Nemotron 3 Nano Omni."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib import request as urllib_request

MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
MODEL_REVISION = "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
RUNTIME_IMAGE = (
    "docker.io/vllm/vllm-openai:v0.20.0@"
    "sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
)
RUNTIME_IMAGE_DIGEST = (
    "sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
)
TARGET_ROWS = (
    ("IRTTtnSRTyY", "Non-speech sound and music", "target"),
    ("IWSPAfCKtV8", "Non-speech sound and music", "target"),
    ("MdMKHhFFqJQ", "Non-speech sound and music", "target"),
    ("OzNdzgQi1CA", "Non-speech sound and music", "target"),
    ("EMI_anpHII4", "Visual controls", "control"),
    ("LusRurRrmf0", "Visual controls", "control"),
)
CONDITIONS = (
    {"condition_id": "audio-off", "audio_enabled": False},
    {"condition_id": "audio-on", "audio_enabled": True},
)
FORBIDDEN_LABEL_KEYS = frozenset(
    {"annotation", "correct_answer", "distractor", "distractors", "reference"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_media(data_root: Path, media_stem: object) -> tuple[str, Path]:
    if not isinstance(media_stem, str) or not media_stem:
        raise ValueError("media must be a non-empty data-root-relative path")
    relative = PurePosixPath(f"{media_stem}.mkv")
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "vqa":
        raise ValueError("media must be a safe vqa-relative path")
    data_root = data_root.resolve()
    resolved = data_root.joinpath(*relative.parts).resolve(strict=True)
    try:
        resolved.relative_to(data_root)
    except ValueError as error:
        raise ValueError("media escapes data_root") from error
    if not resolved.is_file():
        raise ValueError("media must resolve to a file")
    return relative.as_posix(), resolved


def _panel_rows(panel: object) -> dict[str, dict[str, Any]]:
    if not isinstance(panel, dict) or not isinstance(panel.get("examples"), list):
        raise TypeError("failure panel must contain an examples list")
    rows: dict[str, dict[str, Any]] = {}
    for row in panel["examples"]:
        if not isinstance(row, dict):
            raise TypeError("failure-panel rows must be objects")
        video_id = row.get("video_id")
        if isinstance(video_id, str):
            if video_id in rows:
                raise ValueError(f"duplicate video_id: {video_id}")
            rows[video_id] = row
    return rows


def build_label_free_contract(*, panel_path: Path, data_root: Path) -> dict[str, Any]:
    """Freeze the six predeclared rows without exposing annotations to inference."""
    panel_bytes = panel_path.read_bytes()
    rows = _panel_rows(json.loads(panel_bytes))
    examples = []
    for video_id, expected_category, role in TARGET_ROWS:
        if video_id not in rows:
            raise ValueError(f"failure panel is missing {video_id}")
        source = rows[video_id]
        if source.get("category") != expected_category:
            raise ValueError(f"category mismatch for {video_id}")
        question = source.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question is missing for {video_id}")
        media_path, resolved_media = _resolve_media(data_root, source.get("media"))
        examples.append(
            {
                "video_id": video_id,
                "category": expected_category,
                "role": role,
                "question": question.strip(),
                "media_path": media_path,
                "media_sha256": _sha256_file(resolved_media),
            }
        )
    contract = {
        "schema_version": 1,
        "mode": "pilot",
        "panel_id": "nemotron3-omni-focused-6",
        "source_panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conditions": [dict(condition) for condition in CONDITIONS],
        "examples": examples,
    }
    encoded = json.dumps(contract).lower()
    if any(key in encoded for key in FORBIDDEN_LABEL_KEYS):
        raise ValueError("label-bearing field leaked into inference contract")
    return contract


def build_canary_contract(media_paths: list[Path]) -> dict[str, Any]:
    """Build a public synthetic three-request native audio-video contract."""
    if len(media_paths) != 3:
        raise ValueError("canary requires exactly three media files")
    examples = []
    for index, media_path in enumerate(media_paths, start=1):
        if not media_path.is_file():
            raise ValueError(f"canary media is missing: {media_path}")
        examples.append(
            {
                "video_id": f"public-synthetic-av-{index}",
                "category": "public synthetic compatibility",
                "role": "compatibility",
                "question": "What color is the screen while the tone plays?",
                "media_path": media_path.name,
                "media_sha256": _sha256_file(media_path),
            }
        )
    return {
        "schema_version": 1,
        "mode": "canary",
        "panel_id": "nemotron3-omni-public-av-canary-3",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conditions": [{"condition_id": "audio-on", "audio_enabled": True}],
        "examples": examples,
    }


def build_chat_request(
    *, media_path: Path, question: str, audio_enabled: bool
) -> dict[str, Any]:
    """Build one greedy non-thinking native video request for vLLM."""
    if not media_path.is_absolute():
        raise ValueError("media_path must be absolute")
    if not question.strip():
        raise ValueError("question must be non-empty")
    prompt = (
        f"{question.strip()}\nAnswer in at most 12 words. "
        "Return only the answer, with no analysis or explanation."
    )
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": media_path.as_uri()},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32,
        "stream": False,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"use_audio_in_video": audio_enabled},
    }


def parse_short_answer(response: object) -> str:
    """Return only a valid short answer and refuse any exposed reasoning."""
    try:
        choices = response["choices"]  # type: ignore[index]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("response must contain exactly one choice")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("finish_reason must be stop")
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("response omitted choices[0].message.content") from error
    reasoning = message.get("reasoning_content")
    if reasoning not in (None, ""):
        raise ValueError("reasoning content is forbidden")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("answer must be a non-empty string")
    answer = content.strip()
    lowered = answer.lower()
    if "<think" in lowered or "</think" in lowered:
        raise ValueError("thinking markup is forbidden")
    if len(answer.split()) > 12:
        raise ValueError("answer must contain at most 12 words")
    return answer


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _contract_sha256(contract: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(contract)).hexdigest()


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = _canonical_json(payload)
    http_request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(http_request, timeout=600) as response:
        parsed = json.loads(response.read())
    if not isinstance(parsed, dict):
        raise TypeError("server response must be an object")
    return parsed


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _resolve_contract_media(media_root: Path, example: dict[str, Any]) -> Path:
    value = example.get("media_path")
    if not isinstance(value, str) or not value:
        raise ValueError("contract media_path must be non-empty")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("contract media_path must be relative")
    media_root = media_root.resolve()
    candidate = media_root.joinpath(*relative.parts).resolve(strict=True)
    try:
        candidate.relative_to(media_root)
    except ValueError as error:
        raise ValueError("contract media_path escapes media_root") from error
    if _sha256_file(candidate) != example.get("media_sha256"):
        raise ValueError(f"media hash mismatch for {example.get('video_id')}")
    return candidate


def _validate_run_contract(
    contract: dict[str, Any],
) -> tuple[str, list[Any], list[Any]]:
    mode = contract.get("mode")
    if mode not in {"canary", "pilot"}:
        raise ValueError("contract mode must be canary or pilot")
    if (
        contract.get("model_id") != MODEL_ID
        or contract.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("contract model identity mismatch")
    examples = contract.get("examples")
    conditions = contract.get("conditions")
    if not isinstance(examples, list) or not isinstance(conditions, list):
        raise TypeError("contract examples and conditions must be lists")
    if mode == "pilot":
        actual_rows = [
            (row.get("video_id"), row.get("category"), row.get("role"))
            for row in examples
            if isinstance(row, dict)
        ]
        if actual_rows != list(TARGET_ROWS):
            raise ValueError("pilot row identity mismatch")
        if conditions != [dict(condition) for condition in CONDITIONS]:
            raise ValueError("pilot condition identity mismatch")
    else:
        expected_ids = [f"public-synthetic-av-{index}" for index in range(1, 4)]
        actual_ids = [row.get("video_id") for row in examples if isinstance(row, dict)]
        if actual_ids != expected_ids:
            raise ValueError("canary row identity mismatch")
        if conditions != [{"condition_id": "audio-on", "audio_enabled": True}]:
            raise ValueError("canary condition identity mismatch")
    return mode, examples, conditions


def validate_contract_inputs(
    *, contract_path: Path, media_root: Path, marker_path: Path | None = None
) -> dict[str, Any]:
    """Verify the exact label-free contract, marker, and all media before loading."""
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    if not isinstance(contract, dict):
        raise TypeError("contract must be an object")
    mode, examples, conditions = _validate_run_contract(contract)
    if mode == "pilot":
        if marker_path is None:
            raise ValueError("pilot input marker is required")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict):
            raise ValueError("input marker must be an object")
        if marker.get("schema_version") != 1 or marker.get("input_count") != 7:
            raise ValueError("input marker identity mismatch")
        actual_hash = hashlib.sha256(contract_bytes).hexdigest()
        if marker.get("contract_sha256") != actual_hash:
            raise ValueError("input marker contract hash mismatch")
    elif marker_path is not None:
        raise ValueError("canary must not use a private input marker")
    for example in examples:
        if not isinstance(example, dict):
            raise TypeError("contract examples must be objects")
        _resolve_contract_media(media_root, example)
    return {
        "mode": mode,
        "media_files": len(examples),
        "requests": len(examples) * len(conditions),
    }


def run_probe(
    *,
    contract: dict[str, Any],
    media_root: Path,
    output_dir: Path,
    run_id: str,
    request: Any,
) -> dict[str, Any]:
    """Run one foreground, label-blind request for every declared condition."""
    mode, examples, conditions = _validate_run_contract(contract)
    expected = len(examples) * len(conditions)
    if (mode, len(examples), expected) not in {("canary", 3, 3), ("pilot", 6, 12)}:
        raise ValueError("contract request denominator is not predeclared")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    contract_path = output_dir / "contract.json"
    for path in (predictions_path, summary_path, manifest_path, contract_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    _write_json_exclusive(contract_path, contract)

    completed = 0
    with predictions_path.open("x", encoding="utf-8") as handle:
        for pair_index, example in enumerate(examples):
            if not isinstance(example, dict):
                raise TypeError("contract examples must be objects")
            media_path = _resolve_contract_media(media_root, example)
            for condition in conditions:
                if not isinstance(condition, dict):
                    raise TypeError("contract conditions must be objects")
                condition_id = condition.get("condition_id")
                audio_enabled = condition.get("audio_enabled")
                if not isinstance(condition_id, str) or not isinstance(
                    audio_enabled, bool
                ):
                    raise TypeError("condition identity is invalid")
                payload = build_chat_request(
                    media_path=media_path,
                    question=example["question"],
                    audio_enabled=audio_enabled,
                )
                started = time.monotonic()
                record = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "mode": mode,
                    "pair_index": pair_index,
                    "video_id": example["video_id"],
                    "category": example["category"],
                    "role": example["role"],
                    "condition_id": condition_id,
                    "audio_enabled": audio_enabled,
                    "request_sha256": hashlib.sha256(
                        _canonical_json(payload)
                    ).hexdigest(),
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
                }
                try:
                    answer = parse_short_answer(request(payload))
                # Every request failure must become a durable denominator record.
                except Exception as error:  # noqa: BLE001
                    record.update(
                        {
                            "status": "failed",
                            "answer": "",
                            "error_type": type(error).__name__,
                            "error_message": "request_or_response_validation_failed",
                        }
                    )
                else:
                    record.update(
                        {
                            "status": "passed",
                            "answer": answer,
                            "answer_sha256": hashlib.sha256(
                                answer.encode()
                            ).hexdigest(),
                            "error_type": None,
                            "error_message": None,
                        }
                    )
                    completed += 1
                record["elapsed_seconds"] = time.monotonic() - started
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    predictions_sha256 = _sha256_file(predictions_path)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "completion": {"completed": completed, "expected": expected},
        "frozen": True,
        "reference_answers_available": False,
        "predictions_sha256": predictions_sha256,
    }
    _write_json_exclusive(summary_path, summary)
    _write_json_exclusive(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "mode": mode,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_image": RUNTIME_IMAGE,
            "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
            "contract_sha256": _contract_sha256(contract),
            "contract_file_sha256": _sha256_file(contract_path),
            "predictions_sha256": predictions_sha256,
            "reference_answers_available": False,
        },
    )
    return summary


def _main_build_contract(args: argparse.Namespace) -> int:
    contract = build_label_free_contract(
        panel_path=args.panel,
        data_root=args.data_root,
    )
    json.dump(contract, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _main_build_canary(args: argparse.Namespace) -> int:
    contract = build_canary_contract(args.media)
    json.dump(contract, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _main_run(args: argparse.Namespace) -> int:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    summary = run_probe(
        contract=contract,
        media_root=args.media_root,
        output_dir=args.output_dir,
        run_id=args.run_id,
        request=lambda payload: _post_json(args.server_url, payload),
    )
    return int(summary["completion"]["completed"] != summary["completion"]["expected"])


def _main_validate_inputs(args: argparse.Namespace) -> int:
    result = validate_contract_inputs(
        contract_path=args.contract,
        media_root=args.media_root,
        marker_path=args.marker,
    )
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-contract")
    build.add_argument("--panel", type=Path, required=True)
    build.add_argument("--data-root", type=Path, required=True)
    build.set_defaults(handler=_main_build_contract)
    canary = subparsers.add_parser("build-canary-contract")
    canary.add_argument("--media", type=Path, action="append", required=True)
    canary.set_defaults(handler=_main_build_canary)
    validate = subparsers.add_parser("validate-inputs")
    validate.add_argument("--contract", type=Path, required=True)
    validate.add_argument("--media-root", type=Path, required=True)
    validate.add_argument("--marker", type=Path)
    validate.set_defaults(handler=_main_validate_inputs)
    run = subparsers.add_parser("run")
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--media-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    run.set_defaults(handler=_main_run)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
