"""Transport-injected Gemini execution for the label-blind VQA submission path."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import unquote

from trec_video_2026.vqa_gate import canonical_request_sha256
from trec_video_2026.vqa_submission import (
    PASS_IDS,
    SUPPORT_STATUSES,
    AnswerRequest,
    TimelineSpan,
    VerifierDecision,
    VerifierRequest,
)
from trec_video_2026.vqa_submission_evidence import (
    ValidatedTimeline,
    validate_timeline,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_ERROR_LENGTH = 256
_FAILURE_CODES = frozenset(
    {
        "provider_request",
        "provider_upload",
        "provider_generation",
        "provider_failure",
        "protocol_upload_record",
        "protocol_media_mime",
        "protocol_schema",
        "protocol_no_response",
        "protocol_failure",
        "cleanup_delete",
        "auth_vertex_unsupported",
    }
)
_ORPHAN_UPLOAD_SLOT_RE = re.compile(r"orphan-upload-[1-9][0-9]*\Z")
_ORPHAN_UPLOAD_STATUSES = frozenset({"succeeded", "failed"})
_MIME_TYPES = {
    ".avi": "video/x-msvideo",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".webp": "image/webp",
    ".webm": "video/webm",
}
_ALLOWED_MODALITIES = frozenset(
    {"visual", "speech", "non_speech_audio", "visible_text"}
)
_TIMELINE_SPAN_FIELDS = frozenset(
    {"span_id", "start_seconds", "end_seconds", "modalities", "description"}
)
SCHEMA_REPAIR_INSTRUCTION = (
    "Return JSON only and conform exactly to the supplied response schema."
)
SAFE_ERROR_MARKER = "[REDACTED]"
AUTHENTICATION_ERROR_MESSAGE = "Gemini Developer API authentication is not configured"
VERTEX_UNSUPPORTED_MESSAGE = (
    "Vertex AI configuration is unsupported; use the paid Gemini Developer API"
)
_SAFE_ERROR_WORDS = frozenset(
    {
        "a",
        "an",
        "api",
        "bad",
        "busy",
        "by",
        "cancelled",
        "connection",
        "configured",
        "deadline",
        "denied",
        "error",
        "exceeded",
        "failed",
        "file",
        "for",
        "gateway",
        "generation",
        "internal",
        "invalid",
        "limit",
        "media",
        "no",
        "not",
        "of",
        "provider",
        "protocol",
        "rate",
        "request",
        "response",
        "schema",
        "server",
        "service",
        "temporarily",
        "the",
        "timed",
        "timeout",
        "to",
        "unavailable",
        "unknown",
        "upload",
        "validation",
    }
)
_SAFE_ERROR_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .,'/_-]{0,159}\Z")
_SENSITIVE_ERROR_RE = re.compile(
    r"(?is)(?:authorization|proxy-authorization|x-goog-api-key|x-api-key|"
    r"api[_ -]?key|access[_ -]?token|refresh[_ -]?token|bearer|token|secret|"
    r"password|credential)"
)


def _schema_span() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "span_id": {"type": "string", "minLength": 1},
            "start_seconds": {"type": "number", "minimum": 0},
            "end_seconds": {"type": "number", "exclusiveMinimum": 0},
            "modalities": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(_ALLOWED_MODALITIES)},
                "minItems": 1,
                "uniqueItems": True,
            },
            "description": {"type": "string", "minLength": 1, "maxLength": 512},
        },
        "required": sorted(_TIMELINE_SPAN_FIELDS),
        "additionalProperties": False,
    }


ANSWER_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string", "minLength": 1}},
    "required": ["answer"],
    "additionalProperties": False,
}

TIMELINE_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "events": {"type": "array", "items": _schema_span()},
        "support_spans": {"type": "array", "items": _schema_span()},
        "schema_version": {"type": "integer", "const": 1},
        "duration_seconds": {"type": "number", "minimum": 1},
    },
    "required": [
        "schema_version",
        "duration_seconds",
        "events",
        "support_spans",
    ],
    "additionalProperties": False,
}

VERIFIER_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "ranked_public_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": len(PASS_IDS),
            "maxItems": len(PASS_IDS),
            "uniqueItems": True,
        },
        "selected_public_id": {"type": "string", "minLength": 1},
        "support_by_public_id": {
            "type": "object",
            "additionalProperties": {"type": "string", "minLength": 1},
        },
        "status_by_public_id": {
            "type": "object",
            "additionalProperties": {"type": "string", "enum": list(SUPPORT_STATUSES)},
        },
    },
    "required": [
        "ranked_public_ids",
        "selected_public_id",
        "support_by_public_id",
        "status_by_public_id",
    ],
    "additionalProperties": False,
}


class GeminiTransport(Protocol):
    """The small provider boundary used by the adapter and its offline tests."""

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord: ...

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: Mapping[str, object],
    ) -> GenerationResult: ...

    def delete(self, file_id: str) -> None: ...


class GeminiError(RuntimeError):
    """Base class for bounded adapter failures."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str = "provider_request",
        uploaded_file_deletion_status: Mapping[str, bool] | None = None,
        orphan_upload_deletion_status: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = _normalize_failure_code(failure_code, self)
        self.failure_record: FailureRecord | None = None
        self.uploaded_file_deletion_status = MappingProxyType(
            dict(uploaded_file_deletion_status or {})
        )
        self.orphan_upload_deletion_status = MappingProxyType(
            _safe_orphan_upload_status(orphan_upload_deletion_status)
        )


class ProviderError(GeminiError):
    """The provider transport failed before a valid response was available."""


class ProviderProtocolError(GeminiError):
    """The provider returned JSON that did not satisfy the local contract."""


class ProviderAuthenticationError(GeminiError):
    """Gemini's SDK could not be constructed with its standard authentication."""


def _normalize_failure_code(code: object, error: GeminiError) -> str:
    if isinstance(code, str) and code in _FAILURE_CODES:
        return code
    if isinstance(error, ProviderProtocolError):
        return "protocol_failure"
    return "provider_failure"


def _safe_orphan_upload_status(
    statuses: Mapping[str, str] | None,
) -> dict[str, str]:
    if not isinstance(statuses, Mapping):
        return {}
    safe_statuses: dict[str, str] = {}
    for slot, status in statuses.items():
        if (
            isinstance(slot, str)
            and _ORPHAN_UPLOAD_SLOT_RE.fullmatch(slot) is not None
            and isinstance(status, str)
            and status in _ORPHAN_UPLOAD_STATUSES
        ):
            safe_statuses[slot] = status
    return safe_statuses


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider token counts aggregated across the primary and repair calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int | None = None
    cached_input_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        total = self.total_tokens
        if total is None:
            total = self.input_tokens + self.output_tokens
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("total_tokens must be a non-negative integer")
        object.__setattr__(self, "total_tokens", total)

    def __add__(self, other: object) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """Safe, finalized provenance attached to provider/protocol failures."""

    failure_code: str
    request_sha256: str
    model_id: str
    attempt_count: int
    usage: TokenUsage
    elapsed_seconds: float
    response_sha256: str | None
    uploaded_file_deletion_status: Mapping[str, bool | str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.failure_code, str)
            or self.failure_code not in _FAILURE_CODES
        ):
            raise ValueError("failure_code is not allowed")
        if not isinstance(self.request_sha256, str) or _SHA256_RE.fullmatch(
            self.request_sha256
        ) is None:
            raise ValueError("request_sha256 must be a lowercase 64-hex SHA-256")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if (
            isinstance(self.attempt_count, bool)
            or not isinstance(self.attempt_count, int)
            or self.attempt_count < 0
        ):
            raise ValueError("attempt_count must be a non-negative integer")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be a finite non-negative number")
        if self.response_sha256 is not None and _SHA256_RE.fullmatch(
            self.response_sha256
        ) is None:
            raise ValueError("response_sha256 must be a lowercase 64-hex SHA-256")
        statuses = dict(self.uploaded_file_deletion_status)
        for slot, status in statuses.items():
            if isinstance(status, bool):
                continue
            if (
                isinstance(slot, str)
                and _ORPHAN_UPLOAD_SLOT_RE.fullmatch(slot) is not None
                and isinstance(status, str)
                and status in _ORPHAN_UPLOAD_STATUSES
            ):
                continue
            raise TypeError("deletion statuses must be safe booleans or orphan statuses")
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "uploaded_file_deletion_status", MappingProxyType(statuses))

    def to_dict(self) -> dict[str, object]:
        return {
            "failure_code": self.failure_code,
            "request_sha256": self.request_sha256,
            "model_id": self.model_id,
            "attempt_count": self.attempt_count,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
                "cached_input_tokens": self.usage.cached_input_tokens,
            },
            "elapsed_seconds": self.elapsed_seconds,
            "response_sha256": self.response_sha256,
            "uploaded_file_deletion_status": dict(
                self.uploaded_file_deletion_status
            ),
        }


Usage = TokenUsage


@dataclass(frozen=True, slots=True)
class UploadedFileRecord:
    """Provider-neutral identity for one uploaded evidence file."""

    file_id: str
    mime_type: str
    uri: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.file_id, str) or not self.file_id.strip():
            raise ValueError("file_id must be a non-empty string")
        if not isinstance(self.mime_type, str) or not self.mime_type.strip():
            raise ValueError("mime_type must be a non-empty string")
        if self.uri is not None and (
            not isinstance(self.uri, str) or not self.uri.strip()
        ):
            raise ValueError("uri must be a non-empty string when supplied")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Provider response before the adapter applies a local response schema."""

    payload: object
    raw_response: bytes | str
    usage: TokenUsage

    def __post_init__(self) -> None:
        if not isinstance(self.raw_response, (bytes, str)):
            raise TypeError("raw_response must be bytes or text")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")

    @property
    def raw_bytes(self) -> bytes:
        if isinstance(self.raw_response, bytes):
            return self.raw_response
        return self.raw_response.encode("utf-8")

    @property
    def response_sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()


TimelineOutput = ValidatedTimeline


@dataclass(frozen=True, slots=True)
class _ResultMetadata:
    model_id: str
    request_sha256: str
    response_sha256: str
    elapsed_seconds: float
    usage: TokenUsage
    attempt_count: int
    uploaded_files: tuple[UploadedFileRecord, ...]
    uploaded_file_deletion_status: Mapping[str, bool]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        for field_name in ("request_sha256", "response_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a lowercase 64-hex SHA-256")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be a finite non-negative number")
        if (
            isinstance(self.attempt_count, bool)
            or not isinstance(self.attempt_count, int)
            or self.attempt_count < 1
        ):
            raise ValueError("attempt_count must be positive")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        files = tuple(self.uploaded_files)
        if not all(isinstance(item, UploadedFileRecord) for item in files):
            raise TypeError("uploaded_files must contain UploadedFileRecord values")
        statuses = dict(self.uploaded_file_deletion_status)
        if set(statuses) != {item.file_id for item in files}:
            raise ValueError("deletion status must cover every uploaded file")
        if any(not isinstance(status, bool) for status in statuses.values()):
            raise TypeError("deletion statuses must be booleans")
        object.__setattr__(self, "uploaded_files", files)
        object.__setattr__(self, "uploaded_file_deletion_status", MappingProxyType(statuses))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))

    @property
    def raw_response_sha256(self) -> str:
        return self.response_sha256

    @property
    def deletion_status(self) -> Mapping[str, bool]:
        return self.uploaded_file_deletion_status

    @property
    def uploaded_deletion_status(self) -> Mapping[str, bool]:
        return self.uploaded_file_deletion_status


@dataclass(frozen=True, slots=True)
class AnswerResult(_ResultMetadata):
    answer: str

    def __post_init__(self) -> None:
        _ResultMetadata.__post_init__(self)
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be a non-empty string")


@dataclass(frozen=True, slots=True)
class TimelineResult(_ResultMetadata):
    timeline: ValidatedTimeline

    def __post_init__(self) -> None:
        _ResultMetadata.__post_init__(self)
        if not isinstance(self.timeline, ValidatedTimeline):
            raise TypeError("timeline must be ValidatedTimeline")

    @property
    def events(self) -> tuple[TimelineSpan, ...]:
        return self.timeline.events

    @property
    def support_spans(self) -> tuple[TimelineSpan, ...]:
        return self.timeline.support_spans


@dataclass(frozen=True, slots=True)
class VerifierResult(_ResultMetadata):
    decision: VerifierDecision

    def __post_init__(self) -> None:
        _ResultMetadata.__post_init__(self)
        if not isinstance(self.decision, VerifierDecision):
            raise TypeError("decision must be VerifierDecision")

    @property
    def ranked_pass_ids(self) -> tuple[str, ...]:
        return self.decision.ranked_pass_ids


def _bounded_message(error: BaseException) -> str:
    """Return only a conservative generic diagnostic or the fixed marker."""

    message = str(error).replace("\x00", " ").strip()
    if (
        not message
        or len(message) > _MAX_ERROR_LENGTH
        or "\n" in message
        or "\r" in message
        or "%" in message
        or "://" in message
        or "{" in message
        or "}" in message
        or "[" in message
        or "]" in message
        or "(" in message
        or ")" in message
        or _SENSITIVE_ERROR_RE.search(unquote(unquote(message)))
    ):
        return SAFE_ERROR_MARKER
    if _SAFE_ERROR_RE.fullmatch(message) is None:
        return SAFE_ERROR_MARKER
    words = re.findall(r"[A-Za-z]+", message.lower())
    if any(word not in _SAFE_ERROR_WORDS for word in words):
        return SAFE_ERROR_MARKER
    return message


def _reconstruct_transport_error(error: GeminiError) -> GeminiError:
    error_type = (
        ProviderProtocolError
        if isinstance(error, ProviderProtocolError)
        else ProviderError
    )
    failure_code = (
        "protocol_failure"
        if error_type is ProviderProtocolError
        else "provider_failure"
    )
    return error_type(
        f"{type(error).__name__}: {_bounded_message(error)}",
        failure_code=failure_code,
        orphan_upload_deletion_status=_safe_orphan_upload_status(
            getattr(error, "orphan_upload_deletion_status", None)
        ),
    )


def _transport_error(operation: str, error: BaseException) -> ProviderError:
    return ProviderError(
        f"{type(error).__name__}: {_bounded_message(error)}",
        failure_code=(
            f"provider_{operation}"
            if operation in {"upload", "generation"}
            else "provider_request"
        ),
    )


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProviderProtocolError(
            f"Provider JSON was not serializable: {_bounded_message(error)}"
        ) from None


def _request_payload(request: object) -> dict[str, Any]:
    if isinstance(request, Mapping):
        payload = copy.deepcopy(dict(request))
    else:
        to_model_payload = getattr(request, "to_model_payload", None)
        if not callable(to_model_payload):
            raise TypeError("request must be a mapping or a model request contract")
        payload = to_model_payload()
        if not isinstance(payload, Mapping):
            raise TypeError("request.to_model_payload() must return a mapping")
        payload = copy.deepcopy(dict(payload))
    return payload


def _request_hash(request: object, payload: Mapping[str, Any]) -> str:
    declared_hash = getattr(request, "request_sha256", None)
    computed_hash = canonical_request_sha256(payload)
    if declared_hash is not None and declared_hash != computed_hash:
        raise ValueError("request_sha256 does not match the model-visible request")
    return computed_hash


def _path_from_value(value: object, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field} must be a path")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"{field} must be a regular file")
    return path


def _request_files(request: object, payload: Mapping[str, Any]) -> tuple[tuple[str, Path, str], ...]:
    files: list[tuple[str, Path, str]] = []
    seen: set[Path] = set()

    def add(role: str, value: object, mime_type: str | None = None) -> None:
        path = _path_from_value(value, role)
        resolved = path.resolve()
        if resolved in seen:
            return
        resolved_mime_type = mime_type or _MIME_TYPES.get(path.suffix.lower())
        if resolved_mime_type is None:
            raise ProviderProtocolError(
                "Unsupported media MIME type",
                failure_code="protocol_media_mime",
            )
        seen.add(resolved)
        files.append((role, path, resolved_mime_type))

    for role, field_name in (("native-media", "media_path"), ("audio", "audio_path")):
        value = getattr(request, field_name, payload.get(field_name))
        if value is not None:
            add(role, value)

    values = getattr(request, "frame_paths", payload.get("frame_paths", ()))
    if values is None:
        values = ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("frame_paths must be a sequence")
    for index, value in enumerate(values):
        add(f"frame-{index}", value)
    return tuple(files)


def _provider_payload(
    payload: Mapping[str, Any], uploads: Sequence[tuple[str, UploadedFileRecord]]
) -> dict[str, Any]:
    """Replace local paths with provider identities before composing the prompt."""

    sanitized = copy.deepcopy(dict(payload))
    for key in ("media_path", "audio_path", "frame_paths"):
        sanitized.pop(key, None)
    provenance = sanitized.get("frame_provenance")
    if isinstance(provenance, list):
        sanitized["frame_provenance"] = [
            {key: value for key, value in item.items() if key != "frame_path"}
            if isinstance(item, Mapping)
            else item
            for item in provenance
        ]
    sanitized["uploaded_files"] = [
        {
            "role": role,
            "file_id": record.file_id,
            "mime_type": record.mime_type,
        }
        for role, record in uploads
    ]
    return sanitized


def _parts(
    payload: Mapping[str, Any], uploads: Sequence[tuple[str, UploadedFileRecord]]
) -> tuple[object, ...]:
    file_parts = tuple(
        {
            "file_id": record.file_id,
            "file_uri": record.uri,
            "mime_type": record.mime_type,
            "role": role,
        }
        for role, record in uploads
    )
    prompt = json.dumps(
        {
            "request": _provider_payload(payload, uploads),
            "instruction": "Return only the JSON object required by the response schema.",
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (*file_parts, prompt)


def _parse_answer(payload: object) -> str:
    if not isinstance(payload, Mapping) or set(payload) != {"answer"}:
        raise ValueError("answer response must contain exactly the answer field")
    answer = payload["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    return answer.strip()


def _parse_span(payload: object, field: str) -> TimelineSpan:
    if not isinstance(payload, Mapping) or set(payload) != _TIMELINE_SPAN_FIELDS:
        raise ValueError(f"{field} must contain exactly the timeline span fields")
    modalities = payload["modalities"]
    if (
        isinstance(modalities, (str, bytes))
        or not isinstance(modalities, Sequence)
        or not modalities
        or any(item not in _ALLOWED_MODALITIES for item in modalities)
        or len(set(modalities)) != len(modalities)
    ):
        raise ValueError(f"{field}.modalities is invalid")
    description = payload["description"]
    if not isinstance(description, str) or not description.strip() or len(description) > 512:
        raise ValueError(f"{field}.description is invalid")
    span = TimelineSpan(
        span_id=payload["span_id"],
        start_seconds=payload["start_seconds"],
        end_seconds=payload["end_seconds"],
        modalities=tuple(modalities),
        description=description,
    )
    if span.start_seconds >= span.end_seconds:
        raise ValueError(f"{field} must have start before end")
    return span


def _parse_timeline(
    payload: object, *, duration_seconds: float
) -> ValidatedTimeline:
    if not isinstance(payload, Mapping):
        raise ValueError("timeline response must be an object")
    keys = set(payload)
    required = {"schema_version", "duration_seconds", "events", "support_spans"}
    if keys != required:
        raise ValueError("timeline response must have the exact schema")
    if payload["schema_version"] != 1 or isinstance(
        payload["schema_version"], bool
    ):
        raise ValueError("timeline schema_version must be 1")
    model_duration = payload["duration_seconds"]
    if (
        isinstance(model_duration, bool)
        or not isinstance(model_duration, (int, float))
        or not math.isfinite(float(model_duration))
        or not math.isclose(
            float(model_duration), duration_seconds, rel_tol=0.0, abs_tol=1e-6
        )
    ):
        raise ValueError("timeline duration_seconds does not match source duration")
    events = payload["events"]
    support_spans = payload["support_spans"]
    if (
        isinstance(events, (str, bytes))
        or not isinstance(events, Sequence)
        or isinstance(support_spans, (str, bytes))
        or not isinstance(support_spans, Sequence)
    ):
        raise ValueError("timeline events and support_spans must be arrays")
    normalized_payload = {
        "schema_version": 1,
        "duration_seconds": float(model_duration),
        "events": [
            _parse_span(item, f"events[{index}]").to_dict()
            for index, item in enumerate(events)
        ],
        "support_spans": [
            _parse_span(item, f"support_spans[{index}]").to_dict()
            for index, item in enumerate(support_spans)
        ],
    }
    return validate_timeline(
        normalized_payload,
        duration_seconds=duration_seconds,
        allow_fallback=False,
    )


def _request_duration(request: object) -> float:
    payload = _request_payload(request)
    value = getattr(request, "duration_seconds", payload.get("duration_seconds"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeline request requires numeric duration_seconds")
    duration = float(value)
    if not math.isfinite(duration) or duration < 1.0:
        raise ValueError("timeline request duration_seconds must be at least one")
    return duration


def _parse_verifier(payload: object, request: VerifierRequest) -> VerifierDecision:
    if not isinstance(payload, Mapping):
        raise ValueError("verifier response must be an object")
    expected_keys = {
        "ranked_public_ids",
        "selected_public_id",
        "support_by_public_id",
        "status_by_public_id",
    }
    if set(payload) != expected_keys:
        raise ValueError("verifier response must contain the complete decision schema")
    ranked = payload["ranked_public_ids"]
    support = payload["support_by_public_id"]
    status = payload["status_by_public_id"]
    selected = payload["selected_public_id"]
    if (
        isinstance(ranked, (str, bytes))
        or not isinstance(ranked, Sequence)
        or not all(isinstance(item, str) for item in ranked)
        or not isinstance(selected, str)
        or not isinstance(support, Mapping)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in support.items())
        or not isinstance(status, Mapping)
        or not all(isinstance(key, str) and value in SUPPORT_STATUSES for key, value in status.items())
    ):
        raise ValueError("verifier response contains invalid field types")
    return request.map_public_ranking(
        tuple(ranked),
        support_by_public_id=dict(support),
        status_by_public_id=dict(status),
        selected_public_id=selected,
    )


class GeminiSubmissionClient:
    """Execute answer, timeline, and verifier requests over an injected transport."""

    def __init__(self, transport: GeminiTransport, *, model_id: str) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        self._transport = transport
        self.model_id = model_id

    def answer(self, request: AnswerRequest) -> AnswerResult:
        if not isinstance(request, AnswerRequest):
            raise TypeError("answer requires an AnswerRequest")
        answer, metadata = self._execute(request, ANSWER_RESPONSE_SCHEMA, _parse_answer)
        return AnswerResult(answer=answer, **metadata)

    def timeline(self, request: object) -> TimelineResult:
        duration_seconds = _request_duration(request)
        timeline, metadata = self._execute(
            request,
            TIMELINE_RESPONSE_SCHEMA,
            lambda payload: _parse_timeline(
                payload, duration_seconds=duration_seconds
            ),
        )
        return TimelineResult(timeline=timeline, **metadata)

    def verifier(self, request: VerifierRequest) -> VerifierResult:
        if not isinstance(request, VerifierRequest):
            raise TypeError("verifier requires a VerifierRequest")
        decision, metadata = self._execute(
            request,
            VERIFIER_RESPONSE_SCHEMA,
            lambda payload: _parse_verifier(payload, request),
        )
        return VerifierResult(decision=decision, **metadata)

    def verify(self, request: VerifierRequest) -> VerifierResult:
        return self.verifier(request)

    def _execute(
        self,
        request: object,
        schema: Mapping[str, object],
        parser: Any,
    ) -> tuple[Any, dict[str, object]]:
        started = time.monotonic()
        payload = _request_payload(request)
        request_hash = _request_hash(request, payload)
        uploaded: list[tuple[str, UploadedFileRecord]] = []
        deletion_status: dict[str, bool] = {}
        usage = TokenUsage()
        attempts = 0
        response_sha256: str | None = None
        parsed: Any = None
        failure: GeminiError | None = None

        try:
            for role, path, mime_type in _request_files(request, payload):
                try:
                    record = self._transport.upload(path, mime_type=mime_type)
                except GeminiError as error:
                    raise _reconstruct_transport_error(error) from None
                except Exception as error:  # noqa: BLE001 - provider boundary
                    raise _transport_error("upload", error) from None
                if not isinstance(record, UploadedFileRecord):
                    raise ProviderProtocolError(
                        "Gemini upload returned an invalid file record",
                        failure_code="protocol_upload_record",
                    )
                uploaded.append((role, record))

            primary_parts = _parts(payload, uploaded)
            attempts += 1
            generated = self._generate(primary_parts, schema)
            usage += generated.usage
            response_sha256 = generated.response_sha256
            try:
                parsed = parser(generated.payload)
            except Exception:  # noqa: BLE001 - local schema boundary
                repaired_parts = (*primary_parts[:-1], SCHEMA_REPAIR_INSTRUCTION)
                attempts += 1
                repaired = self._generate(repaired_parts, schema)
                usage += repaired.usage
                response_sha256 = repaired.response_sha256
                try:
                    parsed = parser(repaired.payload)
                except Exception:  # noqa: BLE001 - schema boundary
                    raise ProviderProtocolError(
                        "Provider JSON failed local schema validation after one repair",
                        failure_code="protocol_schema",
                    ) from None
        except GeminiError as error:
            failure = error
        except Exception as error:  # noqa: BLE001 - provider boundary
            failure = _transport_error("request", error)
        finally:
            for _role, record in reversed(uploaded):
                try:
                    self._transport.delete(record.file_id)
                except Exception:  # noqa: BLE001 - preserve the primary failure
                    deletion_status[record.file_id] = False
                else:
                    deletion_status[record.file_id] = True

        if failure is None and any(not deleted for deleted in deletion_status.values()):
            failure = ProviderError(
                "Gemini uploaded-file deletion failed",
                failure_code="cleanup_delete",
            )
        if failure is not None:
            failure.__context__ = None
            failure.__cause__ = None
            failure.uploaded_file_deletion_status = MappingProxyType(
                dict(deletion_status)
            )
            failure_status = dict(deletion_status)
            failure_status.update(failure.orphan_upload_deletion_status)
            failure.failure_record = FailureRecord(
                failure_code=failure.failure_code,
                request_sha256=request_hash,
                model_id=self.model_id,
                attempt_count=attempts,
                usage=usage,
                elapsed_seconds=time.monotonic() - started,
                response_sha256=response_sha256,
                uploaded_file_deletion_status=failure_status,
            )
            raise failure
        if attempts < 1 or parsed is None:
            failure = ProviderProtocolError(
                "Gemini produced no validated response",
                failure_code="protocol_no_response",
            )
            failure.failure_record = FailureRecord(
                failure_code=failure.failure_code,
                request_sha256=request_hash,
                model_id=self.model_id,
                attempt_count=attempts,
                usage=usage,
                elapsed_seconds=time.monotonic() - started,
                response_sha256=response_sha256,
                uploaded_file_deletion_status=deletion_status,
            )
            raise failure

        metadata = {
            "model_id": self.model_id,
            "request_sha256": request_hash,
            "response_sha256": response_sha256,
            "elapsed_seconds": time.monotonic() - started,
            "usage": usage,
            "attempt_count": attempts,
            "uploaded_files": tuple(record for _role, record in uploaded),
            "uploaded_file_deletion_status": deletion_status,
        }
        return parsed, metadata

    def _generate(
        self, parts: Sequence[object], schema: Mapping[str, object]
    ) -> GenerationResult:
        try:
            result = self._transport.generate(
                model=self.model_id,
                parts=tuple(parts),
                schema=schema,
            )
        except GeminiError as error:
            raise _reconstruct_transport_error(error) from None
        except Exception as error:  # noqa: BLE001 - provider boundary
            raise _transport_error("generation", error) from None
        if not isinstance(result, GenerationResult):
            raise ProviderProtocolError(
                "Gemini transport returned an invalid generation result"
            )
        return result


class _GoogleGenaiTransport:
    """Small google-genai implementation kept behind the lazy factory."""

    def __init__(self, client: object) -> None:
        self._client = client
        self._orphan_upload_count = 0

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        uploaded = self._client.files.upload(
            file=str(path), config={"mime_type": mime_type}
        )
        try:
            file_id = _provider_file_id(uploaded)
            uri = _provider_file_uri(uploaded)
            return UploadedFileRecord(
                file_id=file_id, mime_type=mime_type, uri=uri
            )
        except Exception:
            orphan_id = _provider_file_id_or_none(uploaded)
            orphan_status: dict[str, str] = {}
            if orphan_id is not None:
                self._orphan_upload_count += 1
                orphan_slot = f"orphan-upload-{self._orphan_upload_count}"
                try:
                    self._client.files.delete(name=orphan_id)
                except Exception:
                    orphan_status[orphan_slot] = "failed"
                else:
                    orphan_status[orphan_slot] = "succeeded"
            raise ProviderProtocolError(
                "Gemini upload response did not contain a valid file record",
                failure_code="protocol_upload_record",
                orphan_upload_deletion_status=orphan_status,
            ) from None

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: Mapping[str, object],
    ) -> GenerationResult:
        response = self._client.models.generate_content(
            model=model,
            contents=[self._as_google_part(part) for part in parts],
            config={
                "response_mime_type": "application/json",
                "response_schema": dict(schema),
            },
        )
        raw_response = response.text
        payload = json.loads(raw_response)
        metadata = getattr(response, "usage_metadata", None)
        usage = TokenUsage(
            input_tokens=_provider_count(metadata, "prompt_token_count"),
            output_tokens=_provider_count(metadata, "candidates_token_count"),
            total_tokens=_provider_count_or_none(metadata, "total_token_count"),
            cached_input_tokens=_provider_count(metadata, "cached_content_token_count"),
        )
        return GenerationResult(
            payload=payload,
            raw_response=raw_response,
            usage=usage,
        )

    def delete(self, file_id: str) -> None:
        self._client.files.delete(name=file_id)

    @staticmethod
    def _as_google_part(part: object) -> object:
        if not isinstance(part, Mapping) or "file_uri" not in part:
            return part
        try:
            from google.genai import types

            return types.Part.from_uri(
                file_uri=part["file_uri"], mime_type=part["mime_type"]
            )
        except ImportError:
            return part


def _provider_file_id_or_none(uploaded: object) -> str | None:
    try:
        if isinstance(uploaded, Mapping):
            value = uploaded.get("name", uploaded.get("file_id"))
        else:
            value = getattr(uploaded, "name", None)
            if value is None:
                value = getattr(uploaded, "file_id", None)
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def _provider_file_id(uploaded: object) -> str:
    file_id = _provider_file_id_or_none(uploaded)
    if file_id is None:
        raise ValueError("missing provider file identifier")
    return file_id


def _provider_file_uri(uploaded: object) -> str | None:
    if isinstance(uploaded, Mapping):
        uri = uploaded.get("uri")
    else:
        uri = getattr(uploaded, "uri", None)
    if uri is not None and (not isinstance(uri, str) or not uri):
        raise ValueError("invalid provider file URI")
    return uri


def _provider_count(metadata: object, field_name: str) -> int:
    value = _provider_count_or_none(metadata, field_name)
    return 0 if value is None else value


def _provider_count_or_none(metadata: object, field_name: str) -> int | None:
    value = getattr(metadata, field_name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def create_google_transport() -> GeminiTransport:
    """Create the standard SDK client without importing or inspecting credentials."""

    try:
        from google import genai
    except Exception:
        raise ProviderAuthenticationError(AUTHENTICATION_ERROR_MESSAGE) from None
    try:
        client = genai.Client()
    except Exception:
        raise ProviderAuthenticationError(AUTHENTICATION_ERROR_MESSAGE) from None
    if _is_vertex_configured(client):
        raise ProviderAuthenticationError(
            VERTEX_UNSUPPORTED_MESSAGE,
            failure_code="auth_vertex_unsupported",
        )
    return _GoogleGenaiTransport(client)


def preflight_google_transport() -> GeminiTransport:
    """Construct the Developer API transport without uploading any media."""

    return create_google_transport()


def _is_vertex_configured(client: object) -> bool:
    for candidate in (client, getattr(client, "_api_client", None)):
        if candidate is None:
            continue
        for field_name in ("vertexai", "vertex_ai", "use_vertexai"):
            try:
                if getattr(candidate, field_name, False) is True:
                    return True
            except Exception:
                continue
    return False


__all__ = [
    "ANSWER_RESPONSE_SCHEMA",
    "TIMELINE_RESPONSE_SCHEMA",
    "VERIFIER_RESPONSE_SCHEMA",
    "AnswerResult",
    "GenerationResult",
    "GeminiError",
    "GeminiSubmissionClient",
    "GeminiTransport",
    "FailureRecord",
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderProtocolError",
    "TimelineOutput",
    "TimelineResult",
    "TokenUsage",
    "Usage",
    "UploadedFileRecord",
    "VerifierResult",
    "AUTHENTICATION_ERROR_MESSAGE",
    "SAFE_ERROR_MARKER",
    "SCHEMA_REPAIR_INSTRUCTION",
    "VERTEX_UNSUPPORTED_MESSAGE",
    "create_google_transport",
    "preflight_google_transport",
]
