"""Label-isolated helpers for the VideoChat3-4B CUDA pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any

from trec_video_2026.vqa_models import normalize_generated_answer


FRAME_COUNT = 12
SUPPORTED_FRAME_COUNTS = (12, 24)
GATE_ROW_IDS = ("BDKkVYmWqdM", "O60MtMuSpf8", "Fk_j6re_Uiw")
PILOT_ROW_IDS = (
    "CH8Dp7EgW4I",
    "EtNOkUJd5w8",
    "Gp4LVTn3YbY",
    "KgOvCPOX-Ms",
    "EMI_anpHII4",
    "LusRurRrmf0",
)
MATCHED_GATE_ROW_IDS = ("CH8Dp7EgW4I", "EMI_anpHII4")
FORBIDDEN_LABEL_KEYS = (
    "answer",
    "annotation",
    "reference",
    "label",
    "grade",
    "gold",
    "ground_truth",
    "score",
)
REQUEST_SOURCE_KEYS = frozenset(
    {
        "run_id",
        "condition_id",
        "video_id",
        "question",
        "frame_paths",
        "frame_timestamps",
        "frame_descriptions",
        "transcript",
        "evidence_cache_key",
    }
)
SHORT_ANSWER_INSTRUCTION = (
    "Answer with one short phrase or sentence. Return only the answer; "
    "do not explain your reasoning."
)
SEED = 20260724
MAX_NEW_TOKENS = 32
INPUT_FLOW_TIMESTAMPS = (
    0.125,
    0.625,
    1.375,
    2.25,
    3.5,
    5.0,
    6.75,
    8.75,
    11.0,
    13.5,
    16.25,
    19.25,
)
INPUT_FLOW_PROMPT_TIMESTAMPS = (1.1875, 6.125, 15.125)
GATE_REQUESTS_SHA256 = (
    "1f476fb3a0f7f5cac3bd3a790348bc6a3eed91c16311e79909f48f8c9760a772"
)
PILOT_REQUESTS_SHA256 = (
    "8b5cf944553ac000795b837090799f07041c8b86cc3ed9a997dc52a9ed4941c4"
)
PILOT_24_REQUESTS_SHA256 = (
    "d13a1135184d92988af28df0fcda47a6d0debef268a8861cfd4ced3898a9518b"
)


def _declared_pilot_sha256(frame_count: int) -> str:
    declared = {
        12: PILOT_REQUESTS_SHA256,
        24: PILOT_24_REQUESTS_SHA256,
    }
    try:
        return declared[frame_count]
    except KeyError as error:
        raise ValueError(f"unsupported frame count: {frame_count}") from error


def _reject_label_fields(value: Any, *, location: str = "request") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(forbidden in lowered for forbidden in FORBIDDEN_LABEL_KEYS):
                raise ValueError(f"forbidden label-bearing field at {location}.{key}")
            _reject_label_fields(nested, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_label_fields(nested, location=f"{location}[{index}]")


def _reject_unexpected_request_fields(request: dict[str, Any]) -> None:
    unexpected = sorted(set(request) - REQUEST_SOURCE_KEYS)
    if unexpected:
        raise ValueError(f"unexpected request fields: {', '.join(unexpected)}")


def _validate_request(
    request: dict[str, Any], *, frame_count: int = FRAME_COUNT
) -> None:
    _reject_label_fields(request)
    _reject_unexpected_request_fields(request)
    if not str(request.get("question", "")).strip():
        raise ValueError("request question must be non-empty")
    paths = request.get("frame_paths")
    timestamps = request.get("frame_timestamps")
    if frame_count not in SUPPORTED_FRAME_COUNTS:
        raise ValueError(f"unsupported frame count: {frame_count}")
    if not isinstance(paths, list) or len(paths) != frame_count:
        raise ValueError(f"request must contain exactly {frame_count} frame paths")
    if not isinstance(timestamps, list) or len(timestamps) != frame_count:
        raise ValueError(f"request must contain exactly {frame_count} timestamps")
    numeric_timestamps = [float(value) for value in timestamps]
    if any(not math.isfinite(value) or value < 0 for value in numeric_timestamps):
        raise ValueError("frame timestamps must be finite and non-negative")
    if numeric_timestamps != sorted(numeric_timestamps):
        raise ValueError("frame timestamps must be chronological")
    if len(set(numeric_timestamps)) != frame_count:
        raise ValueError("frame timestamps must be distinct")


def select_label_free_requests(
    source: Path,
    *,
    run_id: str,
    condition_id: str,
    frame_count: int = FRAME_COUNT,
) -> list[dict[str, Any]]:
    """Project the declared six rows without exposing reference fields."""

    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("request source must be a JSON array")
    by_id: dict[str, dict[str, Any]] = {}
    for request in payload:
        if not isinstance(request, dict):
            raise TypeError("each request must be an object")
        _reject_label_fields(request)
        _reject_unexpected_request_fields(request)
        video_id = str(request.get("video_id", ""))
        if video_id in PILOT_ROW_IDS:
            if video_id in by_id:
                raise ValueError(f"duplicate request for {video_id}")
            by_id[video_id] = request
    missing = [video_id for video_id in PILOT_ROW_IDS if video_id not in by_id]
    if missing:
        raise ValueError(f"missing declared pilot rows: {', '.join(missing)}")
    selected = []
    for video_id in PILOT_ROW_IDS:
        source_request = by_id[video_id]
        request = {
            "run_id": run_id,
            "condition_id": condition_id,
            "video_id": source_request["video_id"],
            "question": source_request.get("question"),
            "frame_paths": source_request.get("frame_paths"),
            "frame_timestamps": source_request.get("frame_timestamps"),
            "frame_descriptions": source_request.get("frame_descriptions"),
            "transcript": source_request.get("transcript"),
        }
        if "evidence_cache_key" in source_request:
            request["evidence_cache_key"] = source_request["evidence_cache_key"]
        _validate_request(request, frame_count=frame_count)
        selected.append(request)
    return selected


def select_gate_requests(
    attempts: Path,
    *,
    run_id: str,
    condition_id: str,
    frame_count: int = FRAME_COUNT,
) -> list[dict[str, Any]]:
    """Project the three declared gate requests from prior label-free requests."""

    by_id: dict[str, dict[str, Any]] = {}
    for line in attempts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        request = record.get("request") if isinstance(record, dict) else None
        if not isinstance(request, dict):
            continue
        video_id = str(request.get("video_id", ""))
        if (
            video_id not in GATE_ROW_IDS
            or video_id in by_id
            or request.get("transcript") is not None
            or len(request.get("frame_paths", ())) != frame_count
        ):
            continue
        projected = {
            "run_id": run_id,
            "condition_id": condition_id,
            "video_id": video_id,
            "question": request.get("question"),
            "frame_paths": [
                f"/dstack/input/evidence/{video_id}/{Path(path).name}"
                for path in request["frame_paths"]
            ],
            "frame_timestamps": request.get("frame_timestamps"),
            "frame_descriptions": request.get("frame_descriptions"),
            "transcript": None,
        }
        _validate_request(projected, frame_count=frame_count)
        by_id[video_id] = projected
    missing = [video_id for video_id in GATE_ROW_IDS if video_id not in by_id]
    if missing:
        raise ValueError(f"missing declared gate rows: {', '.join(missing)}")
    return [by_id[video_id] for video_id in GATE_ROW_IDS]


def build_exact_video_metadata(
    metadata_base: type[Any],
    *,
    processor_metadata_base: type[Any] | None = None,
    timestamps: tuple[float, ...],
    width: int,
    height: int,
) -> Any:
    """Build model-native metadata that preserves exact cached timestamps.

    VideoChat3's pinned processor accepts Transformers ``VideoMetadata``
    instances alongside a 4-D frame tensor. Its stock timestamp property
    assumes constant-rate source frames. The cached timestamps are authoritative
    and rounded independently, so this narrow subclass retains integer frame
    indices while returning those exact values to the model processor.
    """

    if not timestamps or any(
        not math.isfinite(value) or value < 0 for value in timestamps
    ):
        raise ValueError("timestamps must be finite and non-negative")
    if tuple(sorted(timestamps)) != timestamps or len(set(timestamps)) != len(
        timestamps
    ):
        raise ValueError("timestamps must be distinct and chronological")

    secondary_base = processor_metadata_base or object

    class ExactTimestampMetadata(  # type: ignore[misc, valid-type]
        metadata_base,
        secondary_base,
    ):
        def __init__(self) -> None:
            super().__init__(
                total_num_frames=len(timestamps),
                fps=1.0,
                width=width,
                height=height,
                duration=float(len(timestamps)),
                video_backend="frozen-frame-tensor",
                frames_indices=list(range(len(timestamps))),
            )
            if processor_metadata_base is not None:
                self.video_start_time = 0.0
                self.clip_start_time = None
                self.clip_end_time = None

        @property
        def timestamps(self) -> list[float]:
            return list(timestamps)

    return ExactTimestampMetadata()


def build_videochat3_messages(question: str) -> list[dict[str, Any]]:
    """Return one video message using the project's controlled prompt."""

    question = question.strip()
    if not question:
        raise ValueError("question must be non-empty")
    return [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {
                    "type": "text",
                    "text": f"Question: {question}\n{SHORT_ANSWER_INSTRUCTION}",
                },
            ],
        }
    ]


def prepare_videochat3_inputs(
    *,
    processor: Any,
    metadata_base: type[Any],
    processor_metadata_base: type[Any] | None = None,
    question: str,
    video_tensor: Any,
    timestamps: tuple[float, ...],
    width: int,
    height: int,
    model_device: Any,
    model_dtype: Any,
) -> tuple[Any, Any]:
    """Prepare one frozen tensor video without processor-side resampling."""

    metadata = build_exact_video_metadata(
        metadata_base,
        processor_metadata_base=processor_metadata_base,
        timestamps=timestamps,
        width=width,
        height=height,
    )
    messages = build_videochat3_messages(question)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    video_processor = processor.video_processor
    video_kwargs = {
        name: getattr(video_processor, name, None)
        for name in video_processor.valid_kwargs.__annotations__
    }
    size = video_kwargs.get("size")
    if size is not None and not isinstance(size, (int, list, tuple, dict)):
        video_kwargs["size"] = {
            name: value for name, value in vars(size).items() if value is not None
        }
    video_kwargs.update(
        video_metadata=[metadata],
        do_sample_frames=False,
        do_convert_rgb=True,
        return_metadata=True,
        return_tensors="pt",
    )
    inputs = processor(
        text=text,
        videos=[video_tensor],
        **video_kwargs,
    )
    inputs = inputs.to(model_device)
    inputs = inputs.to(model_dtype)
    return inputs, metadata


def attest_videochat3_input_flow(
    *,
    processor: Any,
    inputs: Any,
    source_metadata: Any,
    expected_timestamps: tuple[float, ...],
    expected_prompt_timestamps: tuple[float, ...],
) -> dict[str, Any]:
    """Verify the pinned processor retained exact metadata through conversion."""

    processed_batch = inputs.get("video_metadata")
    if not isinstance(processed_batch, list) or len(processed_batch) != 1:
        raise RuntimeError("processor did not return exactly one video metadata item")
    processed_metadata = processed_batch[0]
    if not isinstance(processed_metadata, type(source_metadata)):
        raise RuntimeError("processor replaced the exact timestamp metadata type")
    preserved_identity = processed_metadata is source_metadata
    processed_timestamps = tuple(float(value) for value in processed_metadata.timestamps)
    if processed_timestamps != expected_timestamps:
        raise RuntimeError("processor changed the exact frame timestamps")
    prompt_timestamps = tuple(
        float(value)
        for value in processor._calculate_timestamps(
            processed_metadata,
            processor.video_processor.temporal_merge_size,
        )
    )
    if prompt_timestamps != expected_prompt_timestamps:
        raise RuntimeError("processor changed the exact merged prompt timestamps")
    return {
        "processor_preserved_metadata_identity": preserved_identity,
        "processor_preserved_exact_timestamps": True,
        "processor_metadata_timestamps": list(processed_timestamps),
        "processor_prompt_timestamps": list(prompt_timestamps),
    }


def _videochat3_prepare_inputs_for_generation(
    self: Any,
    input_ids: Any,
    past_key_values: Any = None,
    inputs_embeds: Any = None,
    pixel_values: Any = None,
    image_grid_thw: Any = None,
    pixel_values_videos: Any = None,
    video_grid_thw: Any = None,
    attention_mask: Any = None,
    cache_position: Any = None,
    logits_to_keep: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Bridge VideoChat3 remote code to the pinned Transformers generation API."""

    model_inputs = super(type(self), self).prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        **kwargs,
    )
    model_inputs.setdefault("inputs_embeds", None)
    if past_key_values is not None or (
        cache_position is not None and cache_position[0] != 0
    ):
        model_inputs["pixel_values"] = None
        model_inputs["pixel_values_videos"] = None
    return model_inputs


def install_videochat3_generation_compatibility(model: Any) -> None:
    """Install the narrow remote-code compatibility bridge once per model."""

    if getattr(model, "_videochat3_generation_compatibility", False):
        return
    model.prepare_inputs_for_generation = MethodType(
        _videochat3_prepare_inputs_for_generation,
        model,
    )
    vision_tower = getattr(getattr(model, "model", None), "vision_tower", None)
    encoder = getattr(vision_tower, "encoder", None)
    for block in getattr(encoder, "blocks", ()):
        block.attn_impl = "sdpa"
    model._videochat3_generation_compatibility = True


def tie_videochat3_output_embeddings(model: Any) -> dict[str, Any]:
    """Replace an untrained output head with the checkpointed token embeddings.

    The pinned VideoChat3 checkpoint omits ``lm_head.weight`` while its outer
    config disables automatic weight tying. Validate the two module contracts
    before making the intended shared-parameter relationship explicit.
    """

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("model must expose input and output embeddings")
    input_weight = input_embeddings.weight
    output_weight = output_embeddings.weight
    input_shape = tuple(input_weight.shape)
    output_shape = tuple(output_weight.shape)
    if input_shape != output_shape:
        raise ValueError(
            f"input/output embedding shape mismatch: {input_shape} != {output_shape}"
        )
    if input_weight.dtype != output_weight.dtype:
        raise ValueError("input/output embedding dtype mismatch")
    if input_weight.device != output_weight.device:
        raise ValueError("input/output embedding device mismatch")
    was_already_tied = input_weight is output_weight
    output_embeddings.weight = input_weight
    if output_embeddings.weight is not input_weight:
        raise RuntimeError("output embedding assignment did not preserve identity")
    return {
        "was_already_tied": was_already_tied,
        "is_tied": True,
        "shape": list(input_shape),
        "dtype": str(input_weight.dtype).removeprefix("torch."),
        "device": str(input_weight.device),
    }


def generate_videochat3_answer(processor: Any, model: Any, inputs: Any) -> str:
    """Generate one deterministic short answer from prepared VideoChat3 inputs."""

    input_length = inputs["input_ids"].shape[-1]
    install_videochat3_generation_compatibility(model)
    generation_inputs = {
        name: value for name, value in inputs.items() if name != "video_metadata"
    }
    generation_inputs.setdefault("inputs_embeds", None)
    generated = model.generate(
        **generation_inputs,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    answer_ids = [output[input_length:] for output in generated]
    text = processor.tokenizer.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    answer = normalize_generated_answer(text)
    if not answer:
        raise ValueError("model returned an empty answer")
    return answer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_requests(
    path: Path,
    *,
    expected_sha256: str,
    expected_row_ids: tuple[str, ...],
    manifest_name: str,
) -> tuple[list[dict[str, Any]], str]:
    """Verify a complete frozen request manifest before parsing its rows."""

    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError(f"{manifest_name} expected SHA-256 must be lowercase hex")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{manifest_name} manifest SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise TypeError(f"{manifest_name} manifest must be a JSON array of objects")
    actual_row_ids = tuple(str(row.get("video_id", "")) for row in payload)
    if actual_row_ids != expected_row_ids:
        raise ValueError(f"{manifest_name} request identity/order mismatch")
    return payload, actual_sha256


def verify_input_manifests(args: argparse.Namespace) -> dict[str, Any]:
    """Return sanitized provenance after verifying both complete manifests."""

    _validate_declared_manifest_hashes(args)
    gate = verify_frozen_manifest(
        args,
        manifest_name="gate",
        declared_sha256=GATE_REQUESTS_SHA256,
    )
    pilot = verify_frozen_manifest(
        args,
        manifest_name="pilot",
        declared_sha256=_declared_pilot_sha256(
            getattr(args, "frame_count", FRAME_COUNT)
        ),
    )
    return {
        "schema_version": "videochat3-input-manifests-v1",
        "verified": True,
        "input_manifest_sha256": {
            "gate_requests": gate["input_manifest_sha256"],
            "pilot_requests": pilot["input_manifest_sha256"],
        },
        "row_ids": {
            "gate": list(GATE_ROW_IDS),
            "pilot": list(PILOT_ROW_IDS),
        },
    }


def verify_frozen_manifest(
    args: argparse.Namespace,
    *,
    manifest_name: str,
    declared_sha256: str,
) -> dict[str, Any]:
    """Verify one complete frozen manifest before model loading."""

    expected_rows = {
        "gate": GATE_ROW_IDS,
        "pilot": PILOT_ROW_IDS,
    }
    if manifest_name not in expected_rows:
        raise ValueError(f"unknown manifest: {manifest_name}")
    path = getattr(args, f"{manifest_name}_requests")
    expected_sha256 = getattr(args, f"{manifest_name}_requests_sha256")
    if expected_sha256 != declared_sha256:
        raise ValueError(f"{manifest_name} manifest SHA-256 pin mismatch")
    _, actual_sha256 = load_frozen_requests(
        path,
        expected_sha256=expected_sha256,
        expected_row_ids=expected_rows[manifest_name],
        manifest_name=manifest_name,
    )
    return {
        "schema_version": "videochat3-input-manifest-v1",
        "verified": True,
        "manifest_name": manifest_name,
        "input_manifest_sha256": actual_sha256,
        "row_ids": list(expected_rows[manifest_name]),
    }


def verify_frozen_evidence(
    args: argparse.Namespace,
    *,
    manifest_name: str,
    declared_sha256: str,
) -> dict[str, Any]:
    """Verify one manifest and every exact cached frame before model loading."""

    provenance = verify_frozen_manifest(
        args,
        manifest_name=manifest_name,
        declared_sha256=declared_sha256,
    )
    path = getattr(args, f"{manifest_name}_requests")
    expected_sha256 = getattr(args, f"{manifest_name}_requests_sha256")
    rows, _ = load_frozen_requests(
        path,
        expected_sha256=expected_sha256,
        expected_row_ids=tuple(provenance["row_ids"]),
        manifest_name=manifest_name,
    )
    provenance["evidence_verified"] = True
    provenance["evidence"] = [
        validate_cached_request(
            row, frame_count=getattr(args, "frame_count", FRAME_COUNT)
        )
        for row in rows
    ]
    return provenance


def _validate_declared_manifest_hashes(args: argparse.Namespace) -> None:
    if args.gate_requests_sha256 != GATE_REQUESTS_SHA256:
        raise ValueError("gate manifest SHA-256 pin mismatch")
    if args.pilot_requests_sha256 != _declared_pilot_sha256(
        getattr(args, "frame_count", FRAME_COUNT)
    ):
        raise ValueError("pilot manifest SHA-256 pin mismatch")


def validate_cached_request(
    request: dict[str, Any], *, frame_count: int = FRAME_COUNT
) -> dict[str, Any]:
    """Verify one request against its exact cached-frame evidence record."""

    _validate_request(request, frame_count=frame_count)
    paths = [Path(value) for value in request["frame_paths"]]
    parent = paths[0].parent
    if any(path.parent != parent for path in paths):
        raise ValueError(f"{request['video_id']}: frame paths cross evidence roots")
    expected_names = [f"frame-{index:02d}.png" for index in range(1, frame_count + 1)]
    if [path.name for path in paths] != expected_names:
        raise ValueError(
            f"{request['video_id']}: frame indices are not the frozen 01..{frame_count:02d} order"
        )
    evidence_path = parent / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    _reject_label_fields(evidence, location="evidence")
    declared_cache_key = request.get("evidence_cache_key")
    if frame_count == 24 and not isinstance(declared_cache_key, str):
        raise ValueError(f"{request['video_id']}: missing evidence cache-key binding")
    if declared_cache_key is not None and (
        declared_cache_key != parent.name
        or declared_cache_key != evidence.get("cache_key")
    ):
        raise ValueError(f"{request['video_id']}: evidence cache-key binding mismatch")
    if evidence.get("parameters", {}).get("frame_count") != frame_count:
        raise ValueError(f"{request['video_id']}: cache frame-count policy differs")
    if evidence.get("timestamps_seconds") != request["frame_timestamps"]:
        raise ValueError(f"{request['video_id']}: request timestamps differ from cache")
    frames = evidence.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise ValueError(
            f"{request['video_id']}: cache must contain exactly {frame_count} frames"
        )
    hashes = []
    for path, frame, expected_name in zip(paths, frames, expected_names, strict=True):
        if frame.get("path") != expected_name:
            raise ValueError(f"{request['video_id']}: cache frame order differs")
        actual = _sha256(path)
        if actual != frame.get("sha256"):
            raise ValueError(f"{request['video_id']}: frame checksum mismatch")
        hashes.append(actual)
    return {
        "video_id": request["video_id"],
        "cache_key": evidence.get("cache_key"),
        "frame_indices": list(range(frame_count)),
        "frame_names": expected_names,
        "frame_timestamps": request["frame_timestamps"],
        "frame_sha256s": hashes,
        "evidence_sha256": _sha256(evidence_path),
    }


def _load_video_tensor(paths: list[str], image_module: Any, torch_module: Any) -> Any:
    from torchvision.transforms.v2 import functional as vision_functional

    frames = []
    size = None
    for value in paths:
        with image_module.open(value) as image:
            rgb = image.convert("RGB")
            if size is None:
                size = rgb.size
            elif rgb.size != size:
                raise ValueError(
                    "all cached frames in a tensor video must have equal size"
                )
            frames.append(vision_functional.pil_to_tensor(rgb))
    return torch_module.stack(frames, dim=0)


def _set_determinism(torch_module: Any) -> None:
    random.seed(SEED)
    torch_module.manual_seed(SEED)
    torch_module.cuda.manual_seed_all(SEED)
    if hasattr(torch_module, "use_deterministic_algorithms"):
        torch_module.use_deterministic_algorithms(True, warn_only=True)


def _load_videochat3_processor(
    model_dir: Path,
) -> tuple[Any, type[Any], type[Any]]:
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor_package = processor.video_processor.__class__.__module__.rsplit(
        ".", 1
    )[0]
    processor_metadata_base = importlib.import_module(
        f"{processor_package}.videochat3_utils"
    ).VideoChat3VideoMetadata
    return processor, VideoMetadata, processor_metadata_base


def _load_videochat3(
    model_dir: Path, torch_module: Any
) -> tuple[Any, Any, type[Any], type[Any], dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    processor, metadata_base, processor_metadata_base = (
        _load_videochat3_processor(model_dir)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_module.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    output_head_tie = tie_videochat3_output_embeddings(model)
    return (
        processor,
        model,
        metadata_base,
        processor_metadata_base,
        output_head_tie,
    )


def _videochat3_rows(
    requests: list[dict[str, Any]],
    *,
    processor: Any,
    model: Any,
    metadata_base: type[Any],
    processor_metadata_base: type[Any],
    image_module: Any,
    torch_module: Any,
    frame_count: int = FRAME_COUNT,
) -> list[dict[str, Any]]:
    results = []
    for request in requests:
        evidence = validate_cached_request(request, frame_count=frame_count)
        started = time.monotonic()
        video = _load_video_tensor(request["frame_paths"], image_module, torch_module)
        height, width = int(video.shape[-2]), int(video.shape[-1])
        inputs, metadata = prepare_videochat3_inputs(
            processor=processor,
            metadata_base=metadata_base,
            processor_metadata_base=processor_metadata_base,
            question=request["question"],
            video_tensor=video,
            timestamps=tuple(float(value) for value in request["frame_timestamps"]),
            width=width,
            height=height,
            model_device=model.device,
            model_dtype=model.dtype,
        )
        answer = generate_videochat3_answer(processor, model, inputs)
        results.append(
            {
                "run_id": request["run_id"],
                "condition_id": request["condition_id"],
                "video_id": request["video_id"],
                "status": "passed",
                "answer": answer,
                "elapsed_seconds": time.monotonic() - started,
                "input_form": f"single-{frame_count}-frame-tensor-video",
                "processor_resampling": False,
                "metadata_frame_indices": list(metadata.frames_indices),
                "metadata_timestamps": list(metadata.timestamps),
                "evidence": evidence,
            }
        )
    return results


def _load_qwen(model_dir: Path, torch_module: Any) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        min_pixels=256 * 256,
        max_pixels=512 * 512,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch_module.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()
    return processor, model


def _qwen_rows(
    requests: list[dict[str, Any]],
    *,
    processor: Any,
    model: Any,
    torch_module: Any,
    frame_count: int = FRAME_COUNT,
) -> list[dict[str, Any]]:
    from trec_video_2026.vqa_models import build_controlled_messages

    results = []
    for source_request in requests:
        request = {
            **source_request,
            "condition_id": f"qwen35-4b-{frame_count}f-matched-control",
        }
        evidence = validate_cached_request(request, frame_count=frame_count)
        started = time.monotonic()
        messages = build_controlled_messages(
            question=request["question"],
            frame_paths=tuple(Path(value) for value in request["frame_paths"]),
            frame_timestamps=tuple(
                float(value) for value in request["frame_timestamps"]
            ),
            frame_descriptions=(
                tuple(request["frame_descriptions"])
                if request.get("frame_descriptions") is not None
                else None
            ),
            transcript=None,
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        input_length = inputs["input_ids"].shape[-1]
        inputs = inputs.to(model.device)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        answer = normalize_generated_answer(
            processor.decode(generated[0][input_length:], skip_special_tokens=True)
        )
        if not answer:
            raise ValueError("Qwen returned an empty answer")
        results.append(
            {
                "run_id": request["run_id"],
                "condition_id": request["condition_id"],
                "video_id": request["video_id"],
                "status": "passed",
                "answer": answer,
                "elapsed_seconds": time.monotonic() - started,
                "input_form": f"{frame_count}-ordered-image-messages",
                "evidence": evidence,
            }
        )
    return results


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    path.chmod(0o444)


def _base_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "videochat3-pilot-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "frozen": True,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "frame_count": getattr(args, "frame_count", FRAME_COUNT),
        "runtime_image": args.runtime_image,
        "videochat3": {
            "model_id": args.videochat_model_id,
            "revision": args.videochat_revision,
        },
    }


def run_input_flow_canary(args: argparse.Namespace) -> dict[str, Any]:
    """Exercise the pinned processor's exact metadata flow without model weights."""

    import torch

    if getattr(torch.version, "hip", None):
        raise RuntimeError("ROCm runtime is not permitted")
    if not getattr(torch.version, "cuda", None) or not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA runtime is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA accelerator does not support bfloat16")
    processor, metadata_base, processor_metadata_base = (
        _load_videochat3_processor(args.videochat_model_dir)
    )
    synthetic = torch.zeros((FRAME_COUNT, 3, 32, 32), dtype=torch.uint8)
    inputs, metadata = prepare_videochat3_inputs(
        processor=processor,
        metadata_base=metadata_base,
        processor_metadata_base=processor_metadata_base,
        question="Which colors become brighter over time?",
        video_tensor=synthetic,
        timestamps=INPUT_FLOW_TIMESTAMPS,
        width=32,
        height=32,
        model_device="cuda:0",
        model_dtype=torch.bfloat16,
    )
    flow = attest_videochat3_input_flow(
        processor=processor,
        inputs=inputs,
        source_metadata=metadata,
        expected_timestamps=INPUT_FLOW_TIMESTAMPS,
        expected_prompt_timestamps=INPUT_FLOW_PROMPT_TIMESTAMPS,
    )
    pixel_values = inputs["pixel_values_videos"]
    pixel_device = str(pixel_values.device)
    if not pixel_device.startswith("cuda"):
        raise RuntimeError("processed video tensor is not on CUDA")
    if pixel_values.dtype != torch.bfloat16:
        raise RuntimeError("processed video tensor is not bfloat16")
    input_ids = inputs["input_ids"][0]
    if hasattr(input_ids, "detach"):
        input_ids = input_ids.detach().cpu().tolist()
    decoded_prompt = processor.tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    prompt_time_labels = [
        f"<{value:.1f} seconds>" for value in INPUT_FLOW_PROMPT_TIMESTAMPS
    ]
    if any(label not in decoded_prompt for label in prompt_time_labels):
        raise RuntimeError("tokenized prompt omitted an exact merged timestamp label")
    return {
        "schema_version": "videochat3-input-flow-canary-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "synthetic-input-flow-canary",
        "videochat3": {
            "model_id": args.videochat_model_id,
            "revision": args.videochat_revision,
        },
        "runtime_image": args.runtime_image,
        "model_loaded": False,
        "private_media_loaded": False,
        "input_form": "single-12-frame-synthetic-tensor-video",
        "input_timestamps": list(INPUT_FLOW_TIMESTAMPS),
        "prompt_time_labels": prompt_time_labels,
        **flow,
        "runtime_attestation": {
            "accelerator_backend": "nvidia-cuda",
            "device": pixel_device,
            "precision": "bfloat16",
            "torch_cuda": str(torch.version.cuda),
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
    }


def _reset_cuda_memory_tracking(torch_module: Any) -> None:
    torch_module.cuda.reset_peak_memory_stats()


def _cuda_peak_memory_bytes(torch_module: Any) -> int:
    return int(torch_module.cuda.max_memory_allocated())


def run_canary(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from PIL import Image

    _set_determinism(torch)
    _reset_cuda_memory_tracking(torch)
    (
        processor,
        model,
        metadata_base,
        processor_metadata_base,
        output_head_tie,
    ) = _load_videochat3(args.videochat_model_dir, torch)
    attestation = validate_cuda_bfloat16(model, torch)
    frame_count = getattr(args, "frame_count", FRAME_COUNT)
    synthetic = torch.zeros((frame_count, 3, 224, 224), dtype=torch.uint8)
    for index in range(frame_count):
        synthetic[index, index % 3] = index * 10
    timestamps = tuple(0.25 + index * 0.5 for index in range(frame_count))
    inputs, metadata = prepare_videochat3_inputs(
        processor=processor,
        metadata_base=metadata_base,
        processor_metadata_base=processor_metadata_base,
        question="Which colors become brighter over time?",
        video_tensor=synthetic,
        timestamps=timestamps,
        width=224,
        height=224,
        model_device=model.device,
        model_dtype=model.dtype,
    )
    answer = generate_videochat3_answer(processor, model, inputs)
    record = _base_record(args)
    record.update(
        {
            "mode": "no-private-media-canary",
            "completion": "1/1",
            "answer": answer,
            "input_form": f"single-{frame_count}-frame-synthetic-tensor-video",
            "metadata_frame_indices": list(metadata.frames_indices),
            "metadata_timestamps": list(metadata.timestamps),
            "cuda_peak_memory_allocated_bytes": _cuda_peak_memory_bytes(torch),
            "runtime_attestation": attestation,
            "output_head_tie": output_head_tie,
            "pil_version": getattr(Image, "__version__", None),
        }
    )
    return record


def gate_answers_are_valid(rows: list[dict[str, Any]]) -> bool:
    """Require exactly three completed, nonempty, single-line short answers."""

    return len(rows) == len(GATE_ROW_IDS) and all(
        row.get("status") == "passed"
        and isinstance(row.get("answer"), str)
        and bool(row["answer"].strip())
        and "\n" not in row["answer"]
        and len(row["answer"].split()) <= 64
        for row in rows
    )


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    if args.gate_requests_sha256 != GATE_REQUESTS_SHA256:
        raise ValueError("gate manifest SHA-256 pin mismatch")
    gate_requests, gate_manifest_sha256 = load_frozen_requests(
        args.gate_requests,
        expected_sha256=args.gate_requests_sha256,
        expected_row_ids=GATE_ROW_IDS,
        manifest_name="gate",
    )

    import torch
    from PIL import Image

    _set_determinism(torch)
    (
        processor,
        model,
        metadata_base,
        processor_metadata_base,
        output_head_tie,
    ) = _load_videochat3(args.videochat_model_dir, torch)
    videochat_attestation = validate_cuda_bfloat16(model, torch)
    gate = _videochat3_rows(
        gate_requests,
        processor=processor,
        model=model,
        metadata_base=metadata_base,
        processor_metadata_base=processor_metadata_base,
        image_module=Image,
        torch_module=torch,
        frame_count=getattr(args, "frame_count", FRAME_COUNT),
    )
    record = _base_record(args)
    record.update(
        {
            "mode": "compatibility-gate",
            "gate_passed": gate_answers_are_valid(gate),
            "gate_completion": f"{len(gate)}/{len(GATE_ROW_IDS)}",
            "gate_predictions": gate,
            "runtime_attestation": {"videochat3": videochat_attestation},
            "output_head_tie": output_head_tie,
            "references_loaded": False,
            "grades_loaded": False,
            "input_manifest_sha256": {"gate_requests": gate_manifest_sha256},
        }
    )
    return record


def _matched_answers_are_valid(
    rows: list[dict[str, Any]], expected_ids: tuple[str, ...]
) -> bool:
    return [str(row.get("video_id", "")) for row in rows] == list(expected_ids) and all(
        row.get("status") == "passed"
        and isinstance(row.get("answer"), str)
        and bool(row["answer"].strip())
        and "\n" not in row["answer"]
        and len(row["answer"].split()) <= 64
        for row in rows
    )


def _run_matched_rows(
    args: argparse.Namespace,
    *,
    selected_ids: tuple[str, ...],
    mode: str,
) -> dict[str, Any]:
    frame_count = getattr(args, "frame_count", FRAME_COUNT)
    declared_sha256 = _declared_pilot_sha256(frame_count)
    if args.pilot_requests_sha256 != declared_sha256:
        raise ValueError("pilot manifest SHA-256 pin mismatch")
    pilot_requests, pilot_manifest_sha256 = load_frozen_requests(
        args.pilot_requests,
        expected_sha256=args.pilot_requests_sha256,
        expected_row_ids=PILOT_ROW_IDS,
        manifest_name="pilot",
    )
    evidence_preflight = [
        validate_cached_request(request, frame_count=frame_count)
        for request in pilot_requests
    ]
    by_id = {str(request["video_id"]): request for request in pilot_requests}
    selected_requests = [by_id[video_id] for video_id in selected_ids]

    import torch
    from PIL import Image

    _set_determinism(torch)
    (
        processor,
        model,
        metadata_base,
        processor_metadata_base,
        output_head_tie,
    ) = _load_videochat3(args.videochat_model_dir, torch)
    videochat_attestation = validate_cuda_bfloat16(model, torch)
    treatment = _videochat3_rows(
        selected_requests,
        processor=processor,
        model=model,
        metadata_base=metadata_base,
        processor_metadata_base=processor_metadata_base,
        image_module=Image,
        torch_module=torch,
        frame_count=frame_count,
    )
    if not _matched_answers_are_valid(treatment, selected_ids):
        raise RuntimeError("VideoChat3 matched rows did not complete with valid answers")
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    _set_determinism(torch)
    qwen_processor, qwen_model = _load_qwen(args.qwen_model_dir, torch)
    qwen_attestation = validate_cuda_bfloat16(qwen_model, torch)
    control = _qwen_rows(
        selected_requests,
        processor=qwen_processor,
        model=qwen_model,
        torch_module=torch,
        frame_count=frame_count,
    )
    if not _matched_answers_are_valid(control, selected_ids):
        raise RuntimeError("Qwen matched rows did not complete with valid answers")
    count = len(selected_ids)
    record = _base_record(args)
    record.update(
        {
            "mode": mode,
            "treatment_predictions": treatment,
            "control_predictions": control,
            "runtime_attestation": {
                "videochat3": videochat_attestation,
                "qwen35": qwen_attestation,
            },
            "output_head_tie": output_head_tie,
            "qwen": {
                "model_id": args.qwen_model_id,
                "revision": args.qwen_revision,
            },
            "references_loaded": False,
            "grades_loaded": False,
            "input_manifest_sha256": {
                "pilot_requests": pilot_manifest_sha256,
            },
            "input_evidence_preflight": evidence_preflight,
        }
    )
    if mode == "matched-two-gate":
        record.update(
            {
                "gate_passed": True,
                "gate_completion": f"{count}/{count}+{count}/{count}",
            }
        )
    else:
        record["pilot_completion"] = f"{count}/{count}+{count}/{count}"
    return record


def run_matched_gate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the frozen target/control operational pair through both models."""

    if getattr(args, "frame_count", FRAME_COUNT) != 24:
        raise ValueError("matched gate is declared only for the 24-frame condition")
    return _run_matched_rows(
        args,
        selected_ids=MATCHED_GATE_ROW_IDS,
        mode="matched-two-gate",
    )


def run_matched_pilot(args: argparse.Namespace) -> dict[str, Any]:
    return _run_matched_rows(
        args,
        selected_ids=PILOT_ROW_IDS,
        mode="matched-six",
    )


def prepare_local_inputs(args: argparse.Namespace) -> dict[str, Any]:
    args.prepare_output_dir.mkdir(parents=True, exist_ok=True)
    gate = select_gate_requests(
        args.source_gate_attempts,
        run_id=args.gate_run_id,
        condition_id="videochat3-compatibility-gate",
    )
    pilot = select_label_free_requests(
        args.source_pilot_requests,
        run_id=args.pilot_run_id,
        condition_id=f"videochat3-4b-{getattr(args, 'frame_count', FRAME_COUNT)}f",
        frame_count=getattr(args, "frame_count", FRAME_COUNT),
    )
    outputs = {
        "gate-requests.json": gate,
        "pilot-requests.json": pilot,
    }
    hashes = {}
    for name, rows in outputs.items():
        path = args.prepare_output_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen input: {path}")
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hashes[name] = _sha256(path)
    return hashes


def prepare_24_frame_manifest(
    *,
    source_requests: Path,
    dataset_root: Path,
    evidence_root: Path,
    output: Path,
    run_id: str,
    remote_evidence_root: Path,
) -> dict[str, Any]:
    """Freeze six label-free requests against exact cached 24-frame evidence."""

    from trec_video_2026.vqa_benchmark import (
        EvidenceParameters,
        _resolved_media,
        cache_key,
        load_cached_evidence,
        source_fingerprint,
    )

    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen input: {output}")
    source_rows = select_label_free_requests(
        source_requests,
        run_id=run_id,
        condition_id="videochat3-4b-12f-source",
    )
    parameters = EvidenceParameters(frame_count=24)
    frozen_rows = []
    cache_keys = []
    total_frame_bytes = 0
    for source_row in source_rows:
        video_id = str(source_row["video_id"])
        media = _resolved_media(
            dataset_root,
            f"vqa/2025/training/videos_original/{video_id}",
        )
        fingerprint = source_fingerprint(media)
        evidence_key = cache_key(fingerprint, parameters)
        evidence = load_cached_evidence(evidence_root / evidence_key, evidence_key)
        row = {
            **source_row,
            "run_id": run_id,
            "condition_id": "videochat3-4b-24f",
            "frame_paths": [
                str(remote_evidence_root / evidence_key / frame.path.name)
                for frame in evidence.frames
            ],
            "frame_timestamps": list(evidence.timestamps_seconds),
            "frame_descriptions": None,
            "evidence_cache_key": evidence_key,
        }
        _validate_request(row, frame_count=24)
        frozen_rows.append(row)
        cache_keys.append(evidence_key)
        total_frame_bytes += sum(frame.path.stat().st_size for frame in evidence.frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(frozen_rows, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    output.chmod(0o444)
    return {
        "manifest_sha256": _sha256(output),
        "evidence_cache_keys": cache_keys,
        "frame_count": 24,
        "row_count": len(frozen_rows),
        "total_frame_bytes": total_frame_bytes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "prepare-local",
            "prepare-24-local",
            "verify-inputs",
            "verify-gate",
            "verify-pilot",
            "input-flow-canary",
            "canary",
            "gate",
            "matched-gate",
            "matched-pilot",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-image")
    parser.add_argument("--videochat-model-dir", type=Path)
    parser.add_argument("--videochat-model-id")
    parser.add_argument("--videochat-revision")
    parser.add_argument("--qwen-model-dir", type=Path)
    parser.add_argument("--qwen-model-id")
    parser.add_argument("--qwen-revision")
    parser.add_argument("--gate-requests", type=Path)
    parser.add_argument("--pilot-requests", type=Path)
    parser.add_argument("--gate-requests-sha256")
    parser.add_argument("--pilot-requests-sha256")
    parser.add_argument("--source-gate-attempts", type=Path)
    parser.add_argument("--source-pilot-requests", type=Path)
    parser.add_argument("--prepare-output-dir", type=Path)
    parser.add_argument("--gate-run-id")
    parser.add_argument("--pilot-run-id")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--remote-evidence-root", type=Path)
    parser.add_argument(
        "--frame-count",
        type=int,
        choices=SUPPORTED_FRAME_COUNTS,
        default=FRAME_COUNT,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "prepare-local":
        hashes = prepare_local_inputs(args)
        print(json.dumps(hashes, sort_keys=True))
        return 0
    if args.mode == "prepare-24-local":
        if args.output is None:
            raise ValueError("--output is required for prepare-24-local mode")
        result = prepare_24_frame_manifest(
            source_requests=args.source_pilot_requests,
            dataset_root=args.dataset_root,
            evidence_root=args.evidence_root,
            output=args.output,
            run_id=args.pilot_run_id,
            remote_evidence_root=args.remote_evidence_root,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.mode == "verify-inputs":
        if args.output is None:
            raise ValueError("--output is required for verify-inputs mode")
        provenance = verify_input_manifests(args)
        _write_exclusive_json(args.output, provenance)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "verified_input_manifest_sha256": provenance[
                        "input_manifest_sha256"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.mode in {"verify-gate", "verify-pilot"}:
        if args.output is None:
            raise ValueError("--output is required for manifest verification")
        manifest_name = args.mode.removeprefix("verify-")
        declared_sha256 = (
            GATE_REQUESTS_SHA256
            if manifest_name == "gate"
            else _declared_pilot_sha256(args.frame_count)
        )
        provenance = verify_frozen_evidence(
            args,
            manifest_name=manifest_name,
            declared_sha256=declared_sha256,
        )
        _write_exclusive_json(args.output, provenance)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "manifest_name": manifest_name,
                    "verified_input_manifest_sha256": provenance[
                        "input_manifest_sha256"
                    ],
                    "evidence_verified": provenance["evidence_verified"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.output is None:
        raise ValueError("--output is required for inference modes")
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen predictions: {args.output}"
        )
    if args.mode == "input-flow-canary":
        record = run_input_flow_canary(args)
    elif args.mode == "canary":
        record = run_canary(args)
    elif args.mode == "gate":
        record = run_gate(args)
    elif args.mode == "matched-gate":
        record = run_matched_gate(args)
    else:
        record = run_matched_pilot(args)
    _write_exclusive_json(args.output, record)
    terminal_output = {"output": str(args.output), "sha256": _sha256(args.output)}
    if "input_manifest_sha256" in record:
        terminal_output["verified_input_manifest_sha256"] = record[
            "input_manifest_sha256"
        ]
    print(json.dumps(terminal_output, sort_keys=True))
    if args.mode in {"gate", "matched-gate"} and not record["gate_passed"]:
        return 3
    return 0


def validate_cuda_bfloat16(model: Any, torch_module: Any) -> dict[str, Any]:
    """Attest NVIDIA CUDA BF16 placement without any fallback."""

    if getattr(torch_module.version, "hip", None):
        raise RuntimeError("ROCm runtime is not permitted")
    if not getattr(torch_module.version, "cuda", None):
        raise RuntimeError("NVIDIA CUDA runtime is unavailable")
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA accelerator is unavailable")
    if not torch_module.cuda.is_bf16_supported():
        raise RuntimeError("CUDA accelerator does not support bfloat16")
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    if not parameters:
        raise RuntimeError("model has no parameters to attest")
    if any(str(parameter.device) != "cuda:0" for parameter in parameters):
        raise RuntimeError("model parameters are not all on cuda:0")
    if any(parameter.dtype != torch_module.bfloat16 for parameter in parameters):
        raise RuntimeError("model parameters are not all bfloat16")
    if any(str(buffer.device) != "cuda:0" for buffer in buffers):
        raise RuntimeError("model buffers are not all on cuda:0")
    return {
        "accelerator_backend": "nvidia-cuda",
        "device": "cuda:0",
        "precision": "bfloat16",
        "torch_cuda": str(torch_module.version.cuda),
        "cuda_device_name": torch_module.cuda.get_device_name(0),
        "parameter_tensor_count": len(parameters),
        "buffer_tensor_count": len(buffers),
    }


if __name__ == "__main__":
    sys.exit(main())
