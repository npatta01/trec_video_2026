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
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .silver_oracle import (
    INTERVAL_DURATION_BUDGET_FORMULA,
    protocol_sha256,
    selector_interval_budget,
    validate_selector_result,
)

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
    "(localized or non_localizable), intervals, and whisper_segment_ids. Return "
    "no more than two intervals; they must be sorted, non-overlapping, inside the "
    "listed video range, and their combined interval duration must not exceed the "
    "listed limit. Return no more than three sorted unique Whisper segment IDs "
    "inside the listed ID range. For non_localizable, return both arrays empty. "
    'Valid localized JSON shape (example times only; do not copy them): {"status":'
    '"localized","intervals":[{"start_seconds":0.0,"end_seconds":1.0}],'
    '"whisper_segment_ids":[]}. '
    "Do not return an answer, caption, rationale, confidence, hint, rewritten "
    "transcript, or any additional key."
)
REPAIR_PROMPT = (
    "Your previous response did not match the required JSON schema. Re-evaluate "
    "the same evidence and return only one valid terminal JSON object with exactly "
    "status, intervals, and whisper_segment_ids. Return no more than two intervals "
    "and obey the listed range, ordering, non-overlap, and combined interval "
    "duration limit. Return no more than three Whisper segment IDs and obey the "
    "listed ID range, uniqueness, and ordering. Valid localized JSON shape "
    '(example times only; do not copy them): {"status":"localized","intervals":'
    '[{"start_seconds":0.0,"end_seconds":1.0}],"whisper_segment_ids":[]}. '
    "Do not add prose or any other key."
)
PRIMARY_SEED = 42
REPAIR_SEED = 43
VIDEO_FPS = 2.0
MAX_NEW_TOKENS = 2048
SAMPLING_TEMPERATURE = 0.6
SAMPLING_TOP_P = 0.95
SAMPLING_TOP_K = 20
MIN_GPU_MEMORY_BYTES = 78_700_000_000
MIN_FREE_MEMORY_BYTES = 6 * 1024**3
MAX_PEAK_ALLOCATED_FRACTION = 0.90
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
CANONICAL_PANEL_IDS_SHA256 = (
    "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
)
EXPECTED_REQUEST_FILENAMES = frozenset(
    {
        "selector-canary.jsonl",
        "selector-chunk-00.jsonl",
        "selector-chunk-01.jsonl",
        "selector-chunk-02.jsonl",
        "selector-chunk-03.jsonl",
        "selector-requests.jsonl",
    }
)
EXPECTED_REQUEST_ROW_COUNTS = {
    "selector-canary.jsonl": 11,
    "selector-chunk-00.jsonl": 119,
    "selector-chunk-01.jsonl": 119,
    "selector-chunk-02.jsonl": 119,
    "selector-chunk-03.jsonl": 118,
    "selector-requests.jsonl": 475,
}


_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX_REVISION = re.compile(r"[0-9a-f]{40}")
_GPU_FAMILY = re.compile(r"(?<![A-Z0-9])(?:A|H|L|T|V)\d{2,4}(?![A-Z0-9])")
_ZERO_UUID = uuid.UUID(int=0)
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


def _ordered_ids_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    ids = [row.get("video_id") for row in rows]
    if any(not isinstance(video_id, str) or not video_id for video_id in ids):
        raise ValueError("request rows must have nonempty video IDs")
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


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
    if (
        not isinstance(expected_request_sha, str)
        or _HEX_SHA256.fullmatch(expected_request_sha) is None
    ):
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
    duration = float(normalized["duration_seconds"])
    interval_budget = selector_interval_budget(duration)
    segment_count = int(normalized["whisper_segment_count"])
    segment_range = f"0 through {segment_count - 1}" if segment_count else "(none)"
    return (
        f"{SELECTOR_PROMPT}{repair_text}\n\n"
        f"Video duration: {duration:.6f} seconds\n"
        "Valid interval range: 0.000000 <= start_seconds < end_seconds <= "
        f"{duration:.6f}.\n"
        f"Combined interval duration limit: {interval_budget:.6f} seconds.\n"
        f"Whisper segment ID range: {segment_range}.\n"
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


def validate_generation_memory_headroom(
    *, total_memory: int, peak_allocated: int, peak_reserved: int
) -> dict[str, int]:
    """Enforce and report the frozen H100 generation-memory safety gates."""

    if peak_allocated > int(total_memory * MAX_PEAK_ALLOCATED_FRACTION):
        raise RuntimeError("peak allocated GPU memory exceeded the 90% gate")
    reserved_margin = total_memory - peak_reserved
    if reserved_margin < MIN_FREE_MEMORY_BYTES:
        raise RuntimeError("peak reserved GPU memory violated the 6 GiB free gate")
    return {
        "gpu_total_memory_bytes": total_memory,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "reserved_memory_margin_bytes": reserved_margin,
    }


def validate_h100_gpu_identity(gpu_name: Any) -> str:
    """Accept a single genuine H100 family name and reject mixed accelerators."""

    if not isinstance(gpu_name, str) or not gpu_name.strip():
        raise RuntimeError("runtime GPU identity must be a single NVIDIA H100")
    name = gpu_name.strip()
    upper = name.upper()
    if not (upper == "H100" or upper.startswith("NVIDIA H100")):
        raise RuntimeError("runtime GPU identity must be a single NVIDIA H100")
    families = _GPU_FAMILY.findall(upper)
    if families != ["H100"]:
        raise RuntimeError("runtime GPU identity must be a single NVIDIA H100")
    return name


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
        raise ValueError(
            "processed selector request must contain exactly one audio input"
        )
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
    gpu_name = validate_h100_gpu_identity(properties.name)
    total_memory = int(properties.total_memory)
    if total_memory < MIN_GPU_MEMORY_BYTES:
        raise RuntimeError("GPU has insufficient memory headroom for native video")
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    ).eval()
    runtime = {
        "device": f"cuda:0 (NVIDIA CUDA; {gpu_name})",
        "precision": "bfloat16",
        "gpu_name": gpu_name,
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
    def generate(
        messages: list[dict[str, Any]], seed: int
    ) -> tuple[str, Mapping[str, Any]]:
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
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
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        peak_memory_allocated = int(torch.cuda.max_memory_allocated())
        peak_memory_reserved = int(torch.cuda.max_memory_reserved())
        memory_headroom = validate_generation_memory_headroom(
            total_memory=total_memory,
            peak_allocated=peak_memory_allocated,
            peak_reserved=peak_memory_reserved,
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
            **memory_headroom,
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


def validate_protocol(
    protocol: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    *,
    requests_filename: str,
    request_file_sha256: str,
) -> str:
    """Bind runtime constants to the frozen controller protocol and request set."""

    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be an object")
    declared_sha = protocol.get("protocol_sha256")
    unhashed = {
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    }
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
        "max_intervals": 2,
        "max_whisper_segments": 3,
        "interval_duration_budget_formula": INTERVAL_DURATION_BUDGET_FORMULA,
    }
    for key, value in expected.items():
        if selector.get(key) != value:
            raise ValueError(f"protocol selector {key} mismatch")
    if panel.get("count") != 475:
        raise ValueError("protocol panel must contain 475 annotated rows")
    if panel.get("ids_sha256") != CANONICAL_PANEL_IDS_SHA256:
        raise ValueError("protocol canonical panel ID digest mismatch")
    if panel.get("canary_count") != 11:
        raise ValueError("protocol canary row count must be 11")
    if panel.get("stratified_canary_count") != 10:
        raise ValueError("protocol stratified canary row count must be 10")
    if panel.get("resource_sentinel_count") != 1:
        raise ValueError("protocol resource sentinel count must be 1")
    sentinel_hash = panel.get("resource_sentinel_id_sha256")
    if (
        not isinstance(sentinel_hash, str)
        or _HEX_SHA256.fullmatch(sentinel_hash) is None
    ):
        raise ValueError("protocol resource sentinel hash is invalid")
    if panel.get("chunk_counts") != [119, 119, 119, 118]:
        raise ValueError("protocol chunk counts mismatch")
    request_files = protocol.get("request_files")
    if not isinstance(request_files, Mapping) or set(request_files) != set(
        EXPECTED_REQUEST_FILENAMES
    ):
        raise ValueError("protocol request file identity set mismatch")
    if requests_filename not in EXPECTED_REQUEST_FILENAMES:
        raise ValueError("request filename is not frozen")
    identity = request_files.get(requests_filename)
    if not isinstance(identity, Mapping):
        raise ValueError("request filename is not bound by protocol")
    if identity.get("row_count") != EXPECTED_REQUEST_ROW_COUNTS[requests_filename]:
        raise ValueError("request file row count is not frozen")
    if identity.get("row_count") != len(requests):
        raise ValueError("request file row count mismatch")
    if identity.get("ids_sha256") != _ordered_ids_sha256(requests):
        raise ValueError("request file ID digest mismatch")
    if identity.get("file_sha256") != request_file_sha256:
        raise ValueError("request file SHA-256 mismatch")
    expected_sentinel_rows = [
        row
        for row in requests
        if isinstance(row.get("video_id"), str)
        and hashlib.sha256(row["video_id"].encode("utf-8")).hexdigest() == sentinel_hash
    ]
    if (
        requests_filename == "selector-canary.jsonl"
        and len(expected_sentinel_rows) != 1
    ):
        raise ValueError("request canary resource sentinel mismatch")
    if requests_filename == "selector-requests.jsonl" and (
        _ordered_ids_sha256(requests) != CANONICAL_PANEL_IDS_SHA256
    ):
        raise ValueError("full request file canonical panel ID digest mismatch")
    run_id = protocol.get("run_id")
    if any(row.get("run_id") != run_id for row in requests):
        raise ValueError("request run_id does not match protocol")
    ids = [row.get("video_id") for row in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("request video IDs must be unique")
    return actual_sha


def _validate_execution_attempt_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("execution attempt ID must be a nonzero canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError(
            "execution attempt ID must be a nonzero canonical UUID"
        ) from error
    if parsed == _ZERO_UUID or str(parsed) != value:
        raise ValueError("execution attempt ID must be a nonzero canonical UUID")
    return value


def _validate_source_identity(
    source_bundle_sha256: Any, submitted_code_revision: Any
) -> tuple[str, str]:
    if (
        not isinstance(source_bundle_sha256, str)
        or _HEX_SHA256.fullmatch(source_bundle_sha256) is None
        or source_bundle_sha256 == "0" * 64
    ):
        raise ValueError("source bundle SHA-256 must be a nonzero lowercase digest")
    if (
        not isinstance(submitted_code_revision, str)
        or _HEX_REVISION.fullmatch(submitted_code_revision) is None
        or submitted_code_revision == "0" * 40
    ):
        raise ValueError("submitted code revision must be a nonzero lowercase revision")
    return source_bundle_sha256, submitted_code_revision


def _validate_remote_output_uri(value: Any, execution_attempt_id: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("hf://buckets/")
        or not value.endswith(f"/{execution_attempt_id}")
        or "/selector/" not in value
        or any(character in value for character in "\r\n?#")
    ):
        raise ValueError("remote output URI must bind the execution attempt ID")
    return value


def _prepare_runtime_preflight(
    *,
    requests_path: Path,
    protocol_path: Path,
    input_root: Path,
    run_id: str,
    execution_attempt_id: str,
    remote_output_uri: str,
    source_bundle_sha256: str,
    submitted_code_revision: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("execution run ID must be nonempty")
    attempt_id = _validate_execution_attempt_id(execution_attempt_id)
    output_uri = _validate_remote_output_uri(remote_output_uri, attempt_id)
    source_digest, revision = _validate_source_identity(
        source_bundle_sha256, submitted_code_revision
    )
    requests = _load_jsonl(requests_path)
    protocol = _load_json(protocol_path)
    request_file_sha256 = _sha256_file(requests_path)
    protocol_digest = validate_protocol(
        protocol,
        requests,
        requests_filename=requests_path.name,
        request_file_sha256=request_file_sha256,
    )
    normalized = [validate_selector_request(row, input_root) for row in requests]
    preflight = {
        "schema_version": 1,
        "run_id": run_id,
        "controller_run_id": protocol["run_id"],
        "execution_attempt_id": attempt_id,
        "remote_output_uri": output_uri,
        "protocol_sha256": protocol_digest,
        "requests_filename": requests_path.name,
        "request_file_sha256": request_file_sha256,
        "request_ids_sha256": _ordered_ids_sha256(requests),
        "request_count": len(requests),
        "panel_ids_sha256": protocol["panel"]["ids_sha256"],
        "resource_sentinel_id_sha256": protocol["panel"]["resource_sentinel_id_sha256"],
        "source_bundle_sha256": source_digest,
        "submitted_code_revision": revision,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "native_video_requested": True,
        "embedded_audio_requested": True,
        "video_fps": VIDEO_FPS,
    }
    return preflight, requests, normalized


def _expected_preflight_marker(preflight: Mapping[str, Any]) -> dict[str, Any]:
    marker_without_hash = {**preflight, "status": "passed"}
    return {
        **marker_without_hash,
        "marker_sha256": _canonical_sha256(marker_without_hash),
    }


def _create_or_validate_preflight_marker(
    marker_path: Path,
    preflight: Mapping[str, Any],
    *,
    create: bool,
) -> dict[str, Any]:
    expected = _expected_preflight_marker(preflight)
    if marker_path.is_symlink():
        raise ValueError("runtime preflight marker must be a regular file")
    if not marker_path.exists():
        if not create:
            raise FileNotFoundError("authenticated runtime preflight marker is absent")
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with marker_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    expected,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
        except FileExistsError:
            pass
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("runtime preflight marker must be a regular file")
    actual = _load_json(marker_path)
    if actual != expected:
        raise ValueError("authenticated runtime preflight marker identity mismatch")
    return expected


def authenticate_runtime_preflight_marker(
    *,
    requests_path: Path,
    protocol_path: Path,
    input_root: Path,
    marker_path: Path,
    run_id: str,
    execution_attempt_id: str,
    remote_output_uri: str,
    source_bundle_sha256: str,
    submitted_code_revision: str,
    create: bool,
) -> dict[str, Any]:
    """Create or verify the exact marker after protocol, request, and media checks."""

    preflight, _, _ = _prepare_runtime_preflight(
        requests_path=requests_path,
        protocol_path=protocol_path,
        input_root=input_root,
        run_id=run_id,
        execution_attempt_id=execution_attempt_id,
        remote_output_uri=remote_output_uri,
        source_bundle_sha256=source_bundle_sha256,
        submitted_code_revision=submitted_code_revision,
    )
    return _create_or_validate_preflight_marker(marker_path, preflight, create=create)


def run_requests(
    *,
    requests_path: Path,
    protocol_path: Path,
    input_root: Path,
    output_dir: Path,
    run_id: str,
    execution_attempt_id: str,
    remote_output_uri: str,
    source_bundle_sha256: str,
    submitted_code_revision: str,
    preflight_marker_path: Path,
) -> dict[str, Any]:
    """Run one canary or disjoint full-panel chunk with durable row checkpoints."""

    preflight, requests, normalized = _prepare_runtime_preflight(
        requests_path=requests_path,
        protocol_path=protocol_path,
        input_root=input_root,
        run_id=run_id,
        execution_attempt_id=execution_attempt_id,
        remote_output_uri=remote_output_uri,
        source_bundle_sha256=source_bundle_sha256,
        submitted_code_revision=submitted_code_revision,
    )
    _create_or_validate_preflight_marker(preflight_marker_path, preflight, create=False)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
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
                "controller_run_id": preflight["controller_run_id"],
                "execution_attempt_id": preflight["execution_attempt_id"],
                "remote_output_uri": preflight["remote_output_uri"],
                "protocol_sha256": preflight["protocol_sha256"],
                "requests_filename": preflight["requests_filename"],
                "request_file_sha256": preflight["request_file_sha256"],
                "request_ids_sha256": preflight["request_ids_sha256"],
                "panel_ids_sha256": preflight["panel_ids_sha256"],
                "resource_sentinel_id_sha256": preflight["resource_sentinel_id_sha256"],
                "source_bundle_sha256": preflight["source_bundle_sha256"],
                "submitted_code_revision": preflight["submitted_code_revision"],
                "model_id": preflight["model_id"],
                "model_revision": preflight["model_revision"],
                "runtime_image": preflight["runtime_image"],
                "device": runtime["device"],
                "precision": runtime["precision"],
                "gpu_name": runtime["gpu_name"],
                "gpu_total_memory_bytes": runtime["gpu_total_memory_bytes"],
                "runtime_versions": runtime["runtime_versions"],
                "model_load_seconds": model_load_seconds,
                "duration_seconds": normalized_request["duration_seconds"],
                "whisper_segment_count": normalized_request["whisper_segment_count"],
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
            attempt["elapsed_seconds"] for row in results for attempt in row["attempts"]
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execution-attempt-id", required=True)
    parser.add_argument("--remote-output-uri", required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--submitted-code-revision", required=True)
    parser.add_argument("--preflight-marker", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--create-preflight-marker", action="store_true")
    mode.add_argument("--verify-preflight-marker", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        identity_args = {
            "requests_path": args.requests,
            "protocol_path": args.protocol,
            "input_root": args.input_root,
            "run_id": args.run_id,
            "execution_attempt_id": args.execution_attempt_id,
            "remote_output_uri": args.remote_output_uri,
            "source_bundle_sha256": args.source_bundle_sha256,
            "submitted_code_revision": args.submitted_code_revision,
        }
        if args.create_preflight_marker or args.verify_preflight_marker:
            marker = authenticate_runtime_preflight_marker(
                **identity_args,
                marker_path=args.preflight_marker,
                create=args.create_preflight_marker,
            )
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "execution_attempt_id": marker["execution_attempt_id"],
                        "marker_sha256": marker["marker_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.output_dir is None:
            raise ValueError("--output-dir is required for selector execution")
        summary = run_requests(
            **identity_args,
            output_dir=args.output_dir,
            preflight_marker_path=args.preflight_marker,
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
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
