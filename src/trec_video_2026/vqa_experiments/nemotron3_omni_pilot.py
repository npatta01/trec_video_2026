"""Label-blind native audio-video pilot for NVIDIA Nemotron 3 Nano Omni."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib import request as urllib_request

from trec_video_2026.vqa_benchmark import MediaMetadata, probe_media

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
EXPANDED_ROWS = (
    ("NN7G28sBOlY", "Dialogue and speech", "target"),
    ("NUmb2P3Bzu8", "Dialogue and speech", "target"),
    ("IRTTtnSRTyY", "Non-speech sound and music", "target"),
    ("IWSPAfCKtV8", "Non-speech sound and music", "target"),
    ("MdMKHhFFqJQ", "Non-speech sound and music", "target"),
    ("OzNdzgQi1CA", "Non-speech sound and music", "target"),
    ("CH8Dp7EgW4I", "Temporal and action", "target"),
    ("EtNOkUJd5w8", "Temporal and action", "target"),
    ("Gp4LVTn3YbY", "Temporal and action", "target"),
    ("KgOvCPOX-Ms", "Temporal and action", "target"),
    ("EMI_anpHII4", "Visual controls", "control"),
    ("LusRurRrmf0", "Visual controls", "control"),
)
FROZEN30_ROWS = (
    ("LKiaCf1QZFw", "Dialogue and speech", "target"),
    ("Ie-KCtcZ7uc", "Dialogue and speech", "target"),
    ("J1CS47pBX_M", "Dialogue and speech", "target"),
    ("Mi-NpnCBegw", "Dialogue and speech", "target"),
    ("NN7G28sBOlY", "Dialogue and speech", "target"),
    ("NUmb2P3Bzu8", "Dialogue and speech", "target"),
    ("IRTTtnSRTyY", "Non-speech sound and music", "target"),
    ("IWSPAfCKtV8", "Non-speech sound and music", "target"),
    ("Jc_vZjKwtvg", "Non-speech sound and music", "target"),
    ("MdMKHhFFqJQ", "Non-speech sound and music", "target"),
    ("NkXPPJoZ-h8", "Non-speech sound and music", "target"),
    ("OzNdzgQi1CA", "Non-speech sound and music", "target"),
    ("CH8Dp7EgW4I", "Temporal and action", "target"),
    ("EtNOkUJd5w8", "Temporal and action", "target"),
    ("Gp4LVTn3YbY", "Temporal and action", "target"),
    ("KgOvCPOX-Ms", "Temporal and action", "target"),
    ("PcDuz7j9LL8", "Temporal and action", "target"),
    ("C4j3Pp79KSg", "Temporal and action", "target"),
    ("Hw0bRvn1ZIo", "Spatial, OCR, and color", "target"),
    ("J-sG9vePDmI", "Spatial, OCR, and color", "target"),
    ("Jc2BdtyAF48", "Spatial, OCR, and color", "target"),
    ("HyOKrz1C4zU", "Spatial, OCR, and color", "target"),
    ("Msbpqv5aOxk", "Spatial, OCR, and color", "target"),
    ("JOClNGH6yL8", "Spatial, OCR, and color", "target"),
    ("EMI_anpHII4", "Visual controls", "control"),
    ("IGL7j9-JRG4", "Visual controls", "control"),
    ("JSOh_pKFq84", "Visual controls", "control"),
    ("QFljXc4WpFA", "Visual controls", "control"),
    ("PiIrinu42DQ", "Visual controls", "control"),
    ("LusRurRrmf0", "Visual controls", "control"),
)
FROZEN30_INFERENCE_MANIFEST_SHA256 = (
    "12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499"
)
CONDITIONS = (
    {"condition_id": "audio-off", "audio_enabled": False},
    {"condition_id": "audio-on", "audio_enabled": True},
)
CODEC_SENTINEL_VIDEO_IDS = (
    "IWSPAfCKtV8",
    "OzNdzgQi1CA",
    "EMI_anpHII4",
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


def _resolve_frozen30_media(data_root: Path, media_stem: object) -> tuple[str, Path]:
    """Resolve exactly one pinned frozen-30 source container without guessing."""
    if not isinstance(media_stem, str) or not media_stem:
        raise ValueError("media must be a non-empty data-root-relative path")
    data_root = data_root.resolve()
    candidates: list[tuple[PurePosixPath, Path]] = []
    for suffix in (".mkv", ".mp4"):
        relative = PurePosixPath(f"{media_stem}{suffix}")
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "vqa"
        ):
            raise ValueError("media must be a safe vqa-relative path")
        candidate = data_root.joinpath(*relative.parts)
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(data_root)
            except ValueError as error:
                raise ValueError("media escapes data_root") from error
            candidates.append((relative, resolved))
    if len(candidates) != 1:
        raise ValueError(f"exactly one source media is required: {media_stem}")
    relative, resolved = candidates[0]
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


def _contains_label_bearing_field(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_LABEL_KEYS:
                return True
            if _contains_label_bearing_field(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_label_bearing_field(item) for item in value)
    return False


def _build_label_free_contract(
    *,
    panel_path: Path,
    data_root: Path,
    rows_to_freeze: tuple[tuple[str, str, str], ...],
    mode: str,
    panel_id: str,
) -> dict[str, Any]:
    """Freeze predeclared rows without exposing annotations to inference."""
    panel_bytes = panel_path.read_bytes()
    rows = _panel_rows(json.loads(panel_bytes))
    examples = []
    for video_id, expected_category, role in rows_to_freeze:
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
        "mode": mode,
        "panel_id": panel_id,
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


def build_label_free_contract(*, panel_path: Path, data_root: Path) -> dict[str, Any]:
    """Freeze the historical six predeclared rows without exposing labels."""
    return _build_label_free_contract(
        panel_path=panel_path,
        data_root=data_root,
        rows_to_freeze=TARGET_ROWS,
        mode="pilot",
        panel_id="nemotron3-omni-focused-6",
    )


def build_expanded_contract(*, panel_path: Path, data_root: Path) -> dict[str, Any]:
    """Freeze the predeclared 12-row panel without exposing labels."""
    return _build_label_free_contract(
        panel_path=panel_path,
        data_root=data_root,
        rows_to_freeze=EXPANDED_ROWS,
        mode="expanded-pilot",
        panel_id="nemotron3-omni-expanded-12",
    )


def build_frozen30_contract(*, panel_path: Path, data_root: Path) -> dict[str, Any]:
    """Freeze the pinned label-free 30-row source manifest for paired inference."""
    panel_bytes = panel_path.read_bytes()
    panel = json.loads(panel_bytes)
    if _contains_label_bearing_field(panel):
        raise ValueError("label-bearing field leaked into frozen30 source manifest")
    if hashlib.sha256(panel_bytes).hexdigest() != FROZEN30_INFERENCE_MANIFEST_SHA256:
        raise ValueError("frozen30 source manifest SHA-256 mismatch")
    if not isinstance(panel, dict) or not isinstance(panel.get("examples"), list):
        raise TypeError("frozen30 source manifest must contain an examples list")
    source_examples = panel["examples"]
    actual_rows = [
        (row.get("video_id"), row.get("category"))
        for row in source_examples
        if isinstance(row, dict)
    ]
    expected_rows = [
        (video_id, category) for video_id, category, _role in FROZEN30_ROWS
    ]
    if actual_rows != expected_rows or len(source_examples) != len(FROZEN30_ROWS):
        raise ValueError("frozen30-paired row identity mismatch")

    examples = []
    for source, (video_id, category, role) in zip(
        source_examples, FROZEN30_ROWS, strict=True
    ):
        if not isinstance(source, dict):
            raise TypeError("frozen30 source rows must be objects")
        question = source.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question is missing for {video_id}")
        media_path, resolved_media = _resolve_frozen30_media(
            data_root, source.get("media")
        )
        examples.append(
            {
                "video_id": video_id,
                "category": category,
                "role": role,
                "question": question.strip(),
                "media_path": media_path,
                "media_sha256": _sha256_file(resolved_media),
            }
        )
    return {
        "schema_version": 1,
        "mode": "frozen30-paired",
        "panel_id": "nemotron3-omni-frozen30",
        "source_manifest_sha256": FROZEN30_INFERENCE_MANIFEST_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conditions": [dict(condition) for condition in CONDITIONS],
        "examples": examples,
    }


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


def build_normalized_contract(
    *,
    source_contract: dict[str, Any],
    source_media_root: Path,
    normalized_media_root: Path,
    mode: str,
    media_probe: Callable[[Path], MediaMetadata] = probe_media,
    decode_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Bind a frozen pilot contract to verified H.264/AAC-derived MP4 bytes."""
    if mode not in {"pilot", "codec-sentinel", "expanded-pilot", "frozen30-paired"}:
        raise ValueError(
            "normalized contract mode must be pilot, codec-sentinel, expanded-pilot, or frozen30-paired"
        )
    source_mode, source_examples, _conditions = _validate_run_contract(source_contract)
    allowed_modes = {
        "pilot": {"pilot", "codec-sentinel"},
        "expanded-pilot": {"expanded-pilot"},
        "frozen30-paired": {"frozen30-paired"},
    }
    if mode not in allowed_modes.get(source_mode, set()):
        raise ValueError(
            "normalized media requires the matching frozen source contract"
        )

    normalized_media_root = normalized_media_root.resolve()
    normalized_examples = []
    for source in source_examples:
        if not isinstance(source, dict):
            raise TypeError("contract examples must be objects")
        source_path = _resolve_contract_media(source_media_root, source)
        video_id = source["video_id"]
        relative = PurePosixPath(f"{video_id}.mp4")
        candidate = normalized_media_root.joinpath(*relative.parts)
        if not candidate.is_file():
            raise ValueError(f"normalized media is missing: {video_id}")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(normalized_media_root)
        except ValueError as error:
            raise ValueError(f"normalized media escapes root: {video_id}") from error
        normalized_example = {
            "video_id": video_id,
            "category": source["category"],
            "role": source["role"],
            "question": source["question"],
            "source_media_path": source["media_path"],
            "source_media_sha256": source["media_sha256"],
            "media_path": relative.as_posix(),
            "derived_media_sha256": _sha256_file(resolved),
        }
        if mode in {"expanded-pilot", "frozen30-paired"}:
            source_duration, derived_duration = _validate_normalized_media(
                source_path=source_path,
                derived_path=resolved,
                media_probe=media_probe,
                decode_runner=decode_runner,
            )
            normalized_example.update(
                {
                    "source_duration_seconds": source_duration,
                    "derived_duration_seconds": derived_duration,
                }
            )
        normalized_examples.append(normalized_example)

    conditions = [dict(condition) for condition in CONDITIONS]
    panel_id = source_contract.get("panel_id")
    if mode == "codec-sentinel":
        normalized_examples = [
            example
            for video_id in CODEC_SENTINEL_VIDEO_IDS
            for example in normalized_examples
            if example["video_id"] == video_id
        ]
        conditions = [{"condition_id": "audio-on", "audio_enabled": True}]
        panel_id = "nemotron3-omni-codec-sentinel-3"

    contract = {
        "schema_version": 1,
        "mode": mode,
        "panel_id": panel_id,
        "source_contract_sha256": _contract_sha256(source_contract),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "media_derivation": {
            "container": "mp4",
            "video_codec": "h264",
            "audio_codec": "aac",
        },
        "conditions": conditions,
        "examples": normalized_examples,
    }
    if mode == "frozen30-paired":
        contract["source_manifest_sha256"] = source_contract.get(
            "source_manifest_sha256"
        )
    else:
        contract["source_panel_sha256"] = source_contract.get("source_panel_sha256")
    encoded = json.dumps(contract).lower()
    if any(key in encoded for key in FORBIDDEN_LABEL_KEYS):
        raise ValueError("label-bearing field leaked into inference contract")
    _validate_run_contract(contract)
    return contract


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


_REASONING_FIELD_NAMES = frozenset(
    {"analysis", "reasoning", "reasoning_content", "reasoning_details"}
)


def _is_empty_reasoning_payload(value: object) -> bool:
    """Return whether a designated reasoning field carries no reasoning text."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return not value or all(
            _is_empty_reasoning_payload(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return not value or all(_is_empty_reasoning_payload(item) for item in value)
    return False


def _contains_reasoning_field(value: object) -> bool:
    """Return whether an API envelope exposes a reasoning-designated field."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if (
                isinstance(key, str)
                and key.lower().replace("-", "_") in _REASONING_FIELD_NAMES
                and not _is_empty_reasoning_payload(nested)
            ):
                return True
            if _contains_reasoning_field(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_reasoning_field(nested) for nested in value)
    return False


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
    if _contains_reasoning_field(response):
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
    expected_hash = example.get("derived_media_sha256", example.get("media_sha256"))
    if _sha256_file(candidate) != expected_hash:
        raise ValueError(f"media hash mismatch for {example.get('video_id')}")
    return candidate


def _codec_names(metadata: MediaMetadata, codec_type: str) -> list[str]:
    return sorted(
        stream.codec_name
        for stream in metadata.streams
        if stream.codec_type == codec_type
    )


def _decode_stream(
    *, path: Path, stream_specifier: str, decode_runner: Callable[..., Any]
) -> None:
    try:
        decode_runner(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                stream_specifier,
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"media decode failed for {path.name}") from error


def _validate_normalized_media(
    *,
    source_path: Path | None,
    derived_path: Path,
    media_probe: Callable[[Path], MediaMetadata],
    decode_runner: Callable[..., Any],
    source_duration_seconds: float | None = None,
) -> tuple[float, float]:
    """Require one decodable H.264/AAC derivation with bounded duration drift."""
    try:
        if source_path is not None:
            source_duration_seconds = media_probe(source_path).duration_seconds
        derived_metadata = media_probe(derived_path)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as error:
        raise ValueError(f"media probe failed for {derived_path.name}") from error
    if (
        not isinstance(source_duration_seconds, (int, float))
        or isinstance(source_duration_seconds, bool)
        or source_duration_seconds <= 0
        or derived_metadata.duration_seconds <= 0
    ):
        raise ValueError(
            f"normalized media duration is invalid for {derived_path.name}"
        )
    video_codecs = _codec_names(derived_metadata, "video")
    audio_codecs = _codec_names(derived_metadata, "audio")
    if video_codecs != ["h264"]:
        raise ValueError(f"normalized video codec mismatch: {video_codecs}")
    if audio_codecs != ["aac"]:
        raise ValueError(f"normalized audio codec mismatch: {audio_codecs}")
    if abs(source_duration_seconds - derived_metadata.duration_seconds) > 0.5:
        raise ValueError(
            "normalized duration drift: "
            f"source={source_duration_seconds}, derived={derived_metadata.duration_seconds}"
        )
    _decode_stream(
        path=derived_path, stream_specifier="0:v:0", decode_runner=decode_runner
    )
    _decode_stream(
        path=derived_path, stream_specifier="0:a:0", decode_runner=decode_runner
    )
    return float(source_duration_seconds), derived_metadata.duration_seconds


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_run_contract(
    contract: dict[str, Any],
) -> tuple[str, list[Any], list[Any]]:
    if _contains_label_bearing_field(contract):
        raise ValueError("label-bearing field leaked into inference contract")
    mode = contract.get("mode")
    if mode not in {
        "canary",
        "pilot",
        "codec-sentinel",
        "expanded-pilot",
        "frozen30-paired",
    }:
        raise ValueError(
            "contract mode must be canary, pilot, codec-sentinel, expanded-pilot, or frozen30-paired"
        )
    if (
        contract.get("model_id") != MODEL_ID
        or contract.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("contract model identity mismatch")
    examples = contract.get("examples")
    conditions = contract.get("conditions")
    if not isinstance(examples, list) or not isinstance(conditions, list):
        raise TypeError("contract examples and conditions must be lists")
    derivation = contract.get("media_derivation")
    if mode in {"expanded-pilot", "frozen30-paired"}:
        expected_fields = {
            "schema_version",
            "mode",
            "panel_id",
            "model_id",
            "model_revision",
            "conditions",
            "examples",
        }
        source_anchor = "source_panel_sha256"
        if mode == "frozen30-paired":
            source_anchor = "source_manifest_sha256"
        expected_fields.add(source_anchor)
        if derivation is not None:
            expected_fields.update({"source_contract_sha256", "media_derivation"})
        if set(contract) != expected_fields:
            raise ValueError(f"{mode} contract schema mismatch")
        if not _is_sha256(contract.get(source_anchor)):
            raise ValueError(f"{mode} source manifest SHA-256 is invalid")
        if (
            mode == "frozen30-paired"
            and contract.get("source_manifest_sha256")
            != FROZEN30_INFERENCE_MANIFEST_SHA256
        ):
            raise ValueError("frozen30-paired source manifest SHA-256 mismatch")
    if mode in {"pilot", "codec-sentinel", "expanded-pilot", "frozen30-paired"}:
        expected_rows = list(TARGET_ROWS)
        expected_conditions = [dict(condition) for condition in CONDITIONS]
        row_error = "pilot row identity mismatch"
        condition_error = "pilot condition identity mismatch"
        if mode == "expanded-pilot":
            expected_rows = list(EXPANDED_ROWS)
            row_error = "expanded-pilot row identity mismatch"
            condition_error = "expanded-pilot condition identity mismatch"
        if mode == "frozen30-paired":
            expected_rows = list(FROZEN30_ROWS)
            row_error = "frozen30-paired row identity mismatch"
            condition_error = "frozen30-paired condition identity mismatch"
        if mode == "codec-sentinel":
            by_id = {row[0]: row for row in TARGET_ROWS}
            expected_rows = [by_id[video_id] for video_id in CODEC_SENTINEL_VIDEO_IDS]
            expected_conditions = [{"condition_id": "audio-on", "audio_enabled": True}]
            row_error = "codec-sentinel row identity mismatch"
            condition_error = "codec-sentinel condition identity mismatch"
        actual_rows = [
            (row.get("video_id"), row.get("category"), row.get("role"))
            for row in examples
            if isinstance(row, dict)
        ]
        if actual_rows != expected_rows:
            raise ValueError(row_error)
        if conditions != expected_conditions:
            raise ValueError(condition_error)
    else:
        expected_ids = [f"public-synthetic-av-{index}" for index in range(1, 4)]
        actual_ids = [row.get("video_id") for row in examples if isinstance(row, dict)]
        if actual_ids != expected_ids:
            raise ValueError("canary row identity mismatch")
        if conditions != [{"condition_id": "audio-on", "audio_enabled": True}]:
            raise ValueError("canary condition identity mismatch")

    if derivation is not None:
        if derivation != {
            "container": "mp4",
            "video_codec": "h264",
            "audio_codec": "aac",
        }:
            raise ValueError("normalized media derivation identity mismatch")
        if not _is_sha256(contract.get("source_contract_sha256")):
            raise ValueError("normalized source contract hash is invalid")
        for example in examples:
            if not isinstance(example, dict):
                raise TypeError("contract examples must be objects")
            video_id = example.get("video_id")
            if (
                example.get("media_path") != f"{video_id}.mp4"
                or (
                    mode != "frozen30-paired"
                    and not str(example.get("source_media_path", "")).endswith(".mkv")
                )
                or (
                    mode == "frozen30-paired"
                    and not str(example.get("source_media_path", "")).endswith(
                        (".mkv", ".mp4")
                    )
                )
                or not _is_sha256(example.get("source_media_sha256"))
                or not _is_sha256(example.get("derived_media_sha256"))
            ):
                raise ValueError(f"normalized media identity mismatch for {video_id}")
            if mode in {"expanded-pilot", "frozen30-paired"} and (
                not isinstance(example.get("source_duration_seconds"), (int, float))
                or isinstance(example.get("source_duration_seconds"), bool)
                or example["source_duration_seconds"] <= 0
                or not isinstance(example.get("derived_duration_seconds"), (int, float))
                or isinstance(example.get("derived_duration_seconds"), bool)
                or example["derived_duration_seconds"] <= 0
            ):
                raise ValueError(
                    f"normalized media duration identity mismatch for {video_id}"
                )
    elif mode == "codec-sentinel":
        raise ValueError("codec-sentinel requires normalized media")
    else:
        for example in examples:
            if not isinstance(example, dict):
                raise TypeError("contract examples must be objects")
            if not _is_sha256(example.get("media_sha256")):
                raise ValueError(
                    f"source media hash is invalid for {example.get('video_id')}"
                )
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
    is_normalized = contract.get("media_derivation") is not None
    if mode in {"pilot", "expanded-pilot", "frozen30-paired"} and not is_normalized:
        if marker_path is None:
            raise ValueError("pilot input marker is required")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict):
            raise ValueError("input marker must be an object")
        if (
            marker.get("schema_version") != 1
            or marker.get("input_count") != len(examples) + 1
        ):
            raise ValueError("input marker identity mismatch")
        actual_hash = hashlib.sha256(contract_bytes).hexdigest()
        if marker.get("contract_sha256") != actual_hash:
            raise ValueError("input marker contract hash mismatch")
    elif marker_path is not None:
        raise ValueError(f"{mode} derived inputs must not use a private input marker")
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
    media_probe: Callable[[Path], MediaMetadata] = probe_media,
    decode_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run one foreground, label-blind request for every declared condition."""
    mode, examples, conditions = _validate_run_contract(contract)
    expected = len(examples) * len(conditions)
    if (mode, len(examples), expected) not in {
        ("canary", 3, 3),
        ("pilot", 6, 12),
        ("codec-sentinel", 3, 3),
        ("expanded-pilot", 12, 24),
        ("frozen30-paired", 30, 60),
    }:
        raise ValueError("contract request denominator is not predeclared")

    if mode in {"expanded-pilot", "frozen30-paired"}:
        for example in examples:
            if not isinstance(example, dict):
                raise TypeError("contract examples must be objects")
            derived_path = _resolve_contract_media(media_root, example)
            _validate_normalized_media(
                source_path=None,
                derived_path=derived_path,
                source_duration_seconds=example["source_duration_seconds"],
                media_probe=media_probe,
                decode_runner=decode_runner,
            )

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
            "frozen": True,
            "completion": {"completed": completed, "expected": expected},
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


def _main_build_expanded_contract(args: argparse.Namespace) -> int:
    contract = build_expanded_contract(
        panel_path=args.panel,
        data_root=args.data_root,
    )
    json.dump(contract, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _main_build_frozen30_contract(args: argparse.Namespace) -> int:
    contract = build_frozen30_contract(
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


def _main_build_normalized(args: argparse.Namespace) -> int:
    source_contract = json.loads(args.source_contract.read_text(encoding="utf-8"))
    if not isinstance(source_contract, dict):
        raise TypeError("source contract must be an object")
    contract = build_normalized_contract(
        source_contract=source_contract,
        source_media_root=args.source_media_root,
        normalized_media_root=args.normalized_media_root,
        mode=args.mode,
    )
    _write_json_exclusive(args.output, contract)
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
    expanded = subparsers.add_parser("build-expanded-contract")
    expanded.add_argument("--panel", type=Path, required=True)
    expanded.add_argument("--data-root", type=Path, required=True)
    expanded.set_defaults(handler=_main_build_expanded_contract)
    frozen30 = subparsers.add_parser("build-frozen30-contract")
    frozen30.add_argument("--panel", type=Path, required=True)
    frozen30.add_argument("--data-root", type=Path, required=True)
    frozen30.set_defaults(handler=_main_build_frozen30_contract)
    canary = subparsers.add_parser("build-canary-contract")
    canary.add_argument("--media", type=Path, action="append", required=True)
    canary.set_defaults(handler=_main_build_canary)
    normalized = subparsers.add_parser("build-normalized-contract")
    normalized.add_argument("--source-contract", type=Path, required=True)
    normalized.add_argument("--source-media-root", type=Path, required=True)
    normalized.add_argument("--normalized-media-root", type=Path, required=True)
    normalized.add_argument(
        "--mode",
        choices=("pilot", "codec-sentinel", "expanded-pilot", "frozen30-paired"),
        required=True,
    )
    normalized.add_argument("--output", type=Path, required=True)
    normalized.set_defaults(handler=_main_build_normalized)
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
