"""Pinned native-video Qwen3-Omni runner for numeric silver-oracle selection.

The selector is reference-assisted by design, but its persisted output is limited
to bounded time intervals and accepted-Whisper segment IDs. Generated reasoning,
answers, and raw model text are never written to the result artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .silver_oracle import protocol_sha256, validate_selector_result

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
MODEL_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
RUNTIME_IMAGE_DIGEST = (
    "sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)
RUNTIME_IMAGE = f"docker.io/qwenllm/qwen3-omni@{RUNTIME_IMAGE_DIGEST}"
SELECTOR_PROMPT = (
    "You are selecting evidence for a video question. You may use the provided "
    "reference answer only to find where the supporting evidence occurs. Return "
    "one terminal JSON object and no prose. The object must have exactly: status "
    "(localized or non_localizable), intervals (zero to two sorted non-overlapping "
    "objects with start_seconds and end_seconds), and whisper_segment_ids (zero "
    "to three sorted unique integer IDs). Do not return an answer, caption, "
    "rationale, confidence, hint, or rewritten transcript."
)
REPAIR_PROMPT = (
    "Your previous response did not match the required JSON schema. Re-evaluate "
    "the same evidence and return only one valid terminal JSON object with exactly "
    "status, intervals, and whisper_segment_ids. Do not add prose or any other key."
)
PRIMARY_SEED = 42
REPAIR_SEED = 43
VIDEO_FPS = 2.0
MAX_NEW_TOKENS = 2048
SAMPLING_TEMPERATURE = 0.6
SAMPLING_TOP_P = 0.95
SAMPLING_TOP_K = 20
MIN_GPU_MEMORY_BYTES = 80 * 1024**3
EXPECTED_EOS_TOKEN_ID = 151645
EXPECTED_PAD_TOKEN_ID = 151643
EXPECTED_EOS_TOKEN = "<|im_end|>"
EXPECTED_PAD_TOKEN = "<|endoftext|>"
NATIVE_VIDEO_INPUT_KEYS = frozenset({"pixel_values_videos", "video_grid_thw"})
NATIVE_AUDIO_INPUT_KEYS = frozenset({"input_features", "feature_attention_mask"})
REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "video_id",
        "video_path",
        "media_sha256",
        "duration_seconds",
        "question",
        "question_stratum",
        "reference",
        "whisper_segments",
        "whisper_dropped_trailing_line_count",
        "whisper_sha256",
        "request_sha256",
    }
)
SEGMENT_FIELDS = frozenset(
    {"segment_id", "start_seconds", "end_seconds", "source_line"}
)
QUESTION_STRATA = frozenset(
    {"temporal", "counting", "audio_text", "causal", "ocr_detail", "other"}
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_TERMINAL_FENCE = re.compile(
    r"(?:\A|\n)```(?:json)?[ \t]*\r?\n(?P<payload>.*?)\r?\n```[ \t]*\Z",
    re.DOTALL | re.IGNORECASE,
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _safe_media_path(input_root: Path, relative: str, video_id: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or pure.parts[:1] != ("media",)
        or len(pure.parts) != 2
        or pure.stem != video_id
        or pure.suffix.casefold() not in {".mkv", ".mp4", ".webm"}
    ):
        raise ValueError("video_path is unsafe or does not match video_id")
    root = input_root.resolve(strict=True)
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("selector media must be a regular non-symlink file")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("selector media escapes input_root")
    return resolved


def validate_selector_request(
    request: Mapping[str, Any],
    input_root: Path,
    *,
    verify_media: bool = True,
) -> dict[str, Any]:
    """Fail closed on request identity, label binding, media, and transcript IDs."""

    if not isinstance(request, Mapping) or set(request) != REQUEST_FIELDS:
        raise ValueError("selector request fields do not match the frozen schema")
    if request.get("schema_version") != 1:
        raise ValueError("selector request schema_version must be 1")
    expected_request_sha = request.get("request_sha256")
    if not isinstance(expected_request_sha, str) or _HEX_SHA256.fullmatch(
        expected_request_sha
    ) is None:
        raise ValueError("request_sha256 must be lowercase SHA-256")
    unhashed = {key: value for key, value in request.items() if key != "request_sha256"}
    if _canonical_sha256(unhashed) != expected_request_sha:
        raise ValueError("selector request SHA-256 mismatch")

    run_id = _nonempty_text(request.get("run_id"), "run_id")
    video_id = _nonempty_text(request.get("video_id"), "video_id")
    question = _nonempty_text(request.get("question"), "question")
    reference = _nonempty_text(request.get("reference"), "reference")
    question_stratum = request.get("question_stratum")
    if question_stratum not in QUESTION_STRATA:
        raise ValueError("question_stratum is not recognized")
    duration = _finite_number(request.get("duration_seconds"), "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    relative_media = _nonempty_text(request.get("video_path"), "video_path")
    media_path = _safe_media_path(input_root, relative_media, video_id)
    media_sha = request.get("media_sha256")
    if not isinstance(media_sha, str) or _HEX_SHA256.fullmatch(media_sha) is None:
        raise ValueError("media_sha256 must be lowercase SHA-256")
    if verify_media and _sha256_file(media_path) != media_sha:
        raise ValueError("selector media SHA-256 mismatch")

    raw_segments = request.get("whisper_segments")
    if not isinstance(raw_segments, list):
        raise ValueError("whisper_segments must be a list")
    normalized_segments: list[dict[str, Any]] = []
    previous_end: float | None = None
    for expected_id, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping) or set(segment) != SEGMENT_FIELDS:
            raise ValueError("Whisper segment fields do not match the frozen schema")
        segment_id = segment.get("segment_id")
        if isinstance(segment_id, bool) or segment_id != expected_id:
            raise ValueError("Whisper segment IDs must be contiguous from zero")
        start = _finite_number(segment.get("start_seconds"), "segment start_seconds")
        end = _finite_number(segment.get("end_seconds"), "segment end_seconds")
        source_line = _nonempty_text(segment.get("source_line"), "segment source_line")
        if start < 0 or end <= start or end > duration:
            raise ValueError("Whisper segment is outside video duration")
        if previous_end is not None and start < previous_end:
            raise ValueError("Whisper segments overlap or are not sorted")
        previous_end = end
        normalized_segments.append(
            {
                "segment_id": segment_id,
                "start_seconds": start,
                "end_seconds": end,
                "source_line": source_line,
            }
        )
    dropped = request.get("whisper_dropped_trailing_line_count")
    if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped < 0:
        raise ValueError("whisper_dropped_trailing_line_count must be non-negative")
    whisper_sha = request.get("whisper_sha256")
    if not isinstance(whisper_sha, str) or _HEX_SHA256.fullmatch(whisper_sha) is None:
        raise ValueError("whisper_sha256 must be lowercase SHA-256")

    return {
        **dict(request),
        "run_id": run_id,
        "video_id": video_id,
        "question": question,
        "reference": reference,
        "duration_seconds": duration,
        "whisper_segments": normalized_segments,
        "whisper_segment_count": len(normalized_segments),
        "media_path": media_path,
    }


def _selector_text(normalized: Mapping[str, Any], *, repair: bool) -> str:
    transcript = normalized["whisper_segments"]
    if transcript:
        transcript_text = "\n".join(
            "segment_id={segment_id} | {start_seconds:.6f}-{end_seconds:.6f} | "
            "{source_line}".format(**segment)
            for segment in transcript
        )
    else:
        transcript_text = "(none)"
    repair_text = f"\n\n{REPAIR_PROMPT}" if repair else ""
    return (
        f"{SELECTOR_PROMPT}{repair_text}\n\n"
        f"Video duration: {normalized['duration_seconds']:.6f} seconds\n"
        f"Question: {normalized['question']}\n"
        f"Reference answer (localization only): {normalized['reference']}\n"
        f"Indexed accepted Whisper transcript:\n{transcript_text}"
    )


def build_selector_messages(
    request: Mapping[str, Any],
    input_root: Path,
    *,
    repair: bool = False,
    verify_media: bool = True,
) -> list[dict[str, Any]]:
    """Build the official native-video-plus-embedded-audio Qwen message."""

    if not isinstance(repair, bool):
        raise TypeError("repair must be a bool")
    normalized = validate_selector_request(
        request, input_root, verify_media=verify_media
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(normalized["media_path"]),
                    "fps": VIDEO_FPS,
                },
                {"type": "text", "text": _selector_text(normalized, repair=repair)},
            ],
        }
    ]


def extract_terminal_selector_json(
    raw_output: str, request: Mapping[str, Any]
) -> dict[str, object]:
    """Extract one terminal object after hidden/visible Thinking text."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("selector output must contain a terminal JSON object")
    text = raw_output.strip()
    fence = _TERMINAL_FENCE.search(text)
    if fence is not None:
        payload_text = fence.group("payload").strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as error:
            raise ValueError("selector terminal fence does not contain JSON") from error
        return validate_selector_result(payload, _selector_row_context(request))

    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if text[end:].strip():
            continue
        return validate_selector_result(payload, _selector_row_context(request))
    raise ValueError("selector output must end with one terminal JSON object")


def _selector_row_context(request: Mapping[str, Any]) -> dict[str, Any]:
    raw_segments = request.get("whisper_segments")
    if not isinstance(raw_segments, list):
        raise ValueError("selector request whisper_segments must be a list")
    return {
        "duration_seconds": request.get("duration_seconds"),
        "whisper_segment_count": len(raw_segments),
    }


GenerationFunction = Callable[
    [list[dict[str, Any]], int], str | tuple[str, Mapping[str, Any]]
]


def run_selector_with_retry(
    request: Mapping[str, Any],
    input_root: Path,
    generate: GenerationFunction,
    *,
    verify_media: bool = True,
) -> dict[str, Any]:
    """Run one fixed primary attempt and at most one schema-only repair."""

    normalized = validate_selector_request(
        request, input_root, verify_media=verify_media
    )
    base: dict[str, Any] = {
        "schema_version": 1,
        "run_id": normalized["run_id"],
        "video_id": normalized["video_id"],
        "request_sha256": normalized["request_sha256"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
    }
    attempts: list[dict[str, Any]] = []
    for attempt_index, (seed, repair) in enumerate(
        ((PRIMARY_SEED, False), (REPAIR_SEED, True)), start=1
    ):
        messages = build_selector_messages(
            request, input_root, repair=repair, verify_media=False
        )
        started = time.perf_counter()
        try:
            generated = generate(messages, seed)
        except Exception as error:  # noqa: BLE001 - persisted as bounded provenance
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "seed": seed,
                    "repair": repair,
                    "status": "failed",
                    "failure_stage": "generation",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "elapsed_seconds": time.perf_counter() - started,
                    "raw_output_sha256": None,
                }
            )
            return {
                **base,
                "status": "failed",
                "failure_stage": "generation",
                "selector": None,
                "selector_attempt_count": len(attempts),
                "repair_used": repair,
                "attempts": attempts,
            }
        if isinstance(generated, tuple):
            raw_output, generation_metadata = generated
            generation_metadata = dict(generation_metadata)
        else:
            raw_output = generated
            generation_metadata = {}
        if not isinstance(raw_output, str):
            raise TypeError("selector generator must return text")
        attempt = {
            "attempt_index": attempt_index,
            "seed": seed,
            "repair": repair,
            "elapsed_seconds": time.perf_counter() - started,
            "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
            "generation": generation_metadata,
        }
        try:
            selector = extract_terminal_selector_json(raw_output, request)
        except (TypeError, ValueError) as error:
            attempts.append(
                {
                    **attempt,
                    "status": "failed",
                    "failure_stage": "schema",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            if not repair:
                continue
            return {
                **base,
                "status": "failed",
                "failure_stage": "schema",
                "selector": None,
                "selector_attempt_count": len(attempts),
                "repair_used": True,
                "attempts": attempts,
            }
        attempts.append(
            {
                **attempt,
                "status": "passed",
                "failure_stage": None,
                "error_type": None,
                "error_message": None,
            }
        )
        return {
            **base,
            "status": "passed",
            "failure_stage": None,
            "selector": selector,
            "selector_attempt_count": len(attempts),
            "repair_used": repair,
            "attempts": attempts,
        }
    raise AssertionError("selector retry loop is exhaustive")


def _runtime_versions(torch: Any, transformers: Any) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("accelerate", "flash-attn", "qwen-omni-utils"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "packages": packages,
    }


def generation_special_token_ids(processor: Any) -> dict[str, int]:
    """Resolve and attest the explicit termination IDs for the pinned model."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor has no tokenizer")
    expected = {
        "eos_token_id": (EXPECTED_EOS_TOKEN_ID, EXPECTED_EOS_TOKEN),
        "pad_token_id": (EXPECTED_PAD_TOKEN_ID, EXPECTED_PAD_TOKEN),
    }
    result: dict[str, int] = {}
    for attribute, (expected_id, expected_token) in expected.items():
        value = getattr(tokenizer, attribute, None)
        if type(value) is not int or value != expected_id:
            raise ValueError(f"tokenizer {attribute} must be {expected_id}")
        token = tokenizer.convert_ids_to_tokens(value)
        if token != expected_token:
            raise ValueError(
                f"tokenizer {attribute} must resolve to {expected_token!r}"
            )
        result[attribute] = value
    return result


def _input_bytes(value: Any) -> bytes:
    candidate = value
    for method in ("detach", "cpu", "contiguous"):
        function = getattr(candidate, method, None)
        if function is not None:
            candidate = function()
    numpy = getattr(candidate, "numpy", None)
    if numpy is not None:
        candidate = numpy()
    tobytes = getattr(candidate, "tobytes", None)
    if tobytes is None:
        raise TypeError(f"cannot attest input of type {type(value).__name__}")
    return tobytes()


def _inputs_sha256(inputs: Mapping[str, Any], keys: frozenset[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_input_bytes(inputs[key]))
    return digest.hexdigest()


@contextlib.contextmanager
def _capture_first_forward(model: Any):
    captures: list[set[str]] = []

    def hook(_module: Any, _args: Any, kwargs: Mapping[str, Any]) -> None:
        if not captures:
            captures.append({key for key, value in kwargs.items() if value is not None})

    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield captures
    finally:
        handle.remove()


def attest_native_video_inputs(
    processor_inputs: Mapping[str, Any], *, first_forward_input_keys: set[str]
) -> dict[str, Any]:
    """Prove one native video and its audio reached the same first forward."""

    available = {key for key, value in processor_inputs.items() if value is not None}
    required = NATIVE_VIDEO_INPUT_KEYS | NATIVE_AUDIO_INPUT_KEYS
    missing = required - available
    if missing:
        raise ValueError(f"processor omitted native modality inputs: {sorted(missing)}")
    missing_forward = required - first_forward_input_keys
    if missing_forward:
        raise ValueError(
            f"first model forward omitted native modality inputs: {sorted(missing_forward)}"
        )
    video_shape = getattr(processor_inputs["video_grid_thw"], "shape", ())
    if not video_shape or int(video_shape[0]) != 1:
        raise ValueError("processed selector request must contain exactly one video")
    audio_shape = getattr(processor_inputs["input_features"], "shape", ())
    if not audio_shape or int(audio_shape[0]) != 1:
        raise ValueError("processed selector request must contain exactly one audio input")
    return {
        "same_first_forward": True,
        "generation_call_count": 1,
        "generation_input_keys": sorted(available),
        "first_forward_input_keys": sorted(first_forward_input_keys),
        "native_video_input_keys": sorted(NATIVE_VIDEO_INPUT_KEYS),
        "native_audio_input_keys": sorted(NATIVE_AUDIO_INPUT_KEYS),
        "native_video_element_count": 1,
        "native_audio_element_count": 1,
        "native_video_input_sha256": _inputs_sha256(
            processor_inputs, NATIVE_VIDEO_INPUT_KEYS
        ),
        "native_audio_input_sha256": _inputs_sha256(
            processor_inputs, NATIVE_AUDIO_INPUT_KEYS
        ),
    }


def _load_runtime() -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from qwen_omni_utils import process_mm_info
    from transformers import (
        Qwen3OmniMoeProcessor,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("an NVIDIA CUDA BF16 accelerator is required")
    properties = torch.cuda.get_device_properties(0)
    total_memory = int(properties.total_memory)
    if total_memory <= MIN_GPU_MEMORY_BYTES:
        raise RuntimeError("GPU has insufficient memory headroom for native video")
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    ).eval()
    runtime = {
        "device": f"cuda:0 (NVIDIA CUDA; {properties.name})",
        "precision": "bfloat16",
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": total_memory,
        "runtime_versions": _runtime_versions(torch, transformers),
    }
    return torch, processor, model, process_mm_info, runtime


def _runtime_generator(
    torch: Any,
    processor: Any,
    model: Any,
    process_mm_info: Any,
) -> GenerationFunction:
    def generate(messages: list[dict[str, Any]], seed: int) -> tuple[str, Mapping[str, Any]]:
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=True
        )
        if videos is None:
            raise ValueError("native video preprocessing returned no video")
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        processor_inputs = dict(inputs)
        input_length = int(inputs["input_ids"].shape[1])
        input_keys = sorted(key for key, value in inputs.items() if value is not None)
        special_token_ids = generation_special_token_ids(processor)
        inputs = inputs.to(next(model.parameters()).device).to(model.dtype)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), _capture_first_forward(model) as captures:
            generated = model.generate(
                **inputs,
                use_audio_in_video=True,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=SAMPLING_TEMPERATURE,
                top_p=SAMPLING_TOP_P,
                top_k=SAMPLING_TOP_K,
                **special_token_ids,
            )
        if not captures:
            raise RuntimeError("model.generate did not invoke the model forward")
        modality_attestation = attest_native_video_inputs(
            processor_inputs, first_forward_input_keys=captures[0]
        )
        sequences = getattr(generated, "sequences", generated)
        if isinstance(sequences, tuple):
            sequences = sequences[0]
        response_tokens = sequences[:, input_length:]
        if int(response_tokens.shape[1]) <= 0:
            raise RuntimeError("Qwen3-Omni generated no response tokens")
        final_generated_token_id = int(response_tokens[0, -1].item())
        ended_with_eos = final_generated_token_id == special_token_ids["eos_token_id"]
        if not ended_with_eos:
            raise RuntimeError(
                "Qwen3-Omni generation did not terminate with the pinned EOS token"
            )
        decoded = processor.batch_decode(
            response_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) != 1:
            raise ValueError("Qwen3-Omni returned an unexpected result count")
        return decoded[0], {
            "use_audio_in_video": True,
            "video_fps": VIDEO_FPS,
            "input_keys": input_keys,
            "input_token_count": input_length,
            "generated_token_count": int(response_tokens.shape[1]),
            "final_generated_token_id": final_generated_token_id,
            "ended_with_eos": ended_with_eos,
            "eos_token": EXPECTED_EOS_TOKEN,
            "pad_token": EXPECTED_PAD_TOKEN,
            **special_token_ids,
            "modality_attestation": modality_attestation,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }

    return generate


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row {index} must be an object")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_protocol(protocol: Mapping[str, Any], requests: Sequence[Mapping[str, Any]]) -> str:
    """Bind runtime constants to the frozen controller protocol and request set."""

    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be an object")
    declared_sha = protocol.get("protocol_sha256")
    unhashed = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    actual_sha = protocol_sha256(unhashed)
    if declared_sha != actual_sha:
        raise ValueError("protocol SHA-256 mismatch")
    selector = protocol.get("selector")
    panel = protocol.get("panel")
    if not isinstance(selector, Mapping) or not isinstance(panel, Mapping):
        raise ValueError("protocol selector and panel must be objects")
    expected = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "primary_seed": PRIMARY_SEED,
        "repair_seed": REPAIR_SEED,
        "max_retries": 1,
        "prompt": SELECTOR_PROMPT,
        "repair_prompt": REPAIR_PROMPT,
    }
    for key, value in expected.items():
        if selector.get(key) != value:
            raise ValueError(f"protocol selector {key} mismatch")
    if panel.get("count") != 475:
        raise ValueError("protocol panel must contain 475 annotated rows")
    if len(requests) not in {10, 118, 119, 475}:
        raise ValueError("request file is not a frozen canary, chunk, or full panel")
    run_id = protocol.get("run_id")
    if any(row.get("run_id") != run_id for row in requests):
        raise ValueError("request run_id does not match protocol")
    ids = [row.get("video_id") for row in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("request video IDs must be unique")
    return actual_sha


def run_requests(
    *,
    requests_path: Path,
    protocol_path: Path,
    input_root: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """Run one canary or disjoint full-panel chunk with durable row checkpoints."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = _load_jsonl(requests_path)
    protocol = _load_json(protocol_path)
    protocol_digest = validate_protocol(protocol, requests)
    normalized = [validate_selector_request(row, input_root) for row in requests]
    preflight = {
        "schema_version": 1,
        "run_id": run_id,
        "controller_run_id": protocol["run_id"],
        "protocol_sha256": protocol_digest,
        "request_file_sha256": _sha256_file(requests_path),
        "request_count": len(requests),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "native_video_requested": True,
        "embedded_audio_requested": True,
        "video_fps": VIDEO_FPS,
    }
    _write_json(output_dir / "preflight.json", preflight)
    load_started = time.perf_counter()
    torch, processor, model, process_mm_info, runtime = _load_runtime()
    model_load_seconds = time.perf_counter() - load_started
    generator = _runtime_generator(torch, processor, model, process_mm_info)
    results: list[dict[str, Any]] = []
    for request, normalized_request in zip(requests, normalized, strict=True):
        torch.cuda.empty_cache()
        record = run_selector_with_retry(
            request, input_root, generator, verify_media=False
        )
        record.update(
            {
                "execution_run_id": run_id,
                "protocol_sha256": protocol_digest,
                "device": runtime["device"],
                "precision": runtime["precision"],
                "gpu_name": runtime["gpu_name"],
                "gpu_total_memory_bytes": runtime["gpu_total_memory_bytes"],
                "runtime_versions": runtime["runtime_versions"],
                "model_load_seconds": model_load_seconds,
                "duration_seconds": normalized_request["duration_seconds"],
                "whisper_segment_count": normalized_request[
                    "whisper_segment_count"
                ],
            }
        )
        results.append(record)
        _write_jsonl(output_dir / "selector-results.jsonl", results)
    passed = sum(row["status"] == "passed" for row in results)
    schema_failures = sum(row.get("failure_stage") == "schema" for row in results)
    summary = {
        **preflight,
        **runtime,
        "model_load_seconds": model_load_seconds,
        "rows": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "schema_failures": schema_failures,
        "repair_count": sum(row["repair_used"] for row in results),
        "elapsed_generation_seconds": sum(
            attempt["elapsed_seconds"]
            for row in results
            for attempt in row["attempts"]
        ),
        "results_sha256": _sha256_file(output_dir / "selector-results.jsonl"),
        "status": "passed" if passed == len(results) else "failed",
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_requests(
            requests_path=args.requests,
            protocol_path=args.protocol,
            input_root=args.input_root,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    except Exception as error:  # noqa: BLE001 - detached task emits bounded failure
        print(
            json.dumps(
                {
                    "status": "failed",
                    "failure_stage": "fatal",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
