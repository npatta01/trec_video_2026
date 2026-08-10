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

from trec_video_2026.vqa_gate import canonical_request_sha256
from trec_video_2026.vqa_submission import (
    PASS_IDS,
    SUPPORT_STATUSES,
    AnswerRequest,
    TimelineSpan,
    VerifierDecision,
    VerifierRequest,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_ERROR_LENGTH = 256
_MIME_TYPES = {
    ".avi": "video/x-msvideo",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".webm": "video/webm",
}
_ALLOWED_MODALITIES = frozenset(
    {"visual", "speech", "non_speech_audio", "visible_text"}
)
_TIMELINE_SPAN_FIELDS = frozenset(
    {"span_id", "start_seconds", "end_seconds", "modalities", "description"}
)
_REPAIR_PREFIX = (
    "Repair the JSON object to conform exactly to the supplied response schema. "
    "Return JSON only. Parsed invalid object: "
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
    "required": ["events", "support_spans"],
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
        uploaded_file_deletion_status: Mapping[str, bool] | None = None,
    ) -> None:
        super().__init__(message)
        self.uploaded_file_deletion_status = MappingProxyType(
            dict(uploaded_file_deletion_status or {})
        )


class ProviderError(GeminiError):
    """The provider transport failed before a valid response was available."""


class ProviderProtocolError(GeminiError):
    """The provider returned JSON that did not satisfy the local contract."""


class ProviderAuthenticationError(GeminiError):
    """Gemini's SDK could not be constructed with its standard authentication."""


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


@dataclass(frozen=True, slots=True)
class TimelineOutput:
    """Locally validated timeline JSON returned by Gemini."""

    events: tuple[TimelineSpan, ...]
    support_spans: tuple[TimelineSpan, ...]
    schema_version: int | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        events = tuple(self.events)
        support_spans = tuple(self.support_spans)
        if not all(isinstance(item, TimelineSpan) for item in events + support_spans):
            raise TypeError("timeline output must contain TimelineSpan values")
        if len({span.span_id for span in events}) != len(events):
            raise ValueError("timeline event IDs must be unique")
        if len({span.span_id for span in support_spans}) != len(support_spans):
            raise ValueError("timeline support span IDs must be unique")
        if self.schema_version is not None and self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.duration_seconds is not None and (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or self.duration_seconds < 1
        ):
            raise ValueError("duration_seconds must be a finite number of at least 1")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "support_spans", support_spans)
        if self.duration_seconds is not None:
            object.__setattr__(self, "duration_seconds", float(self.duration_seconds))

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "events": [event.to_dict() for event in self.events],
            "support_spans": [span.to_dict() for span in self.support_spans],
        }
        if self.schema_version is not None:
            payload["schema_version"] = self.schema_version
        if self.duration_seconds is not None:
            payload["duration_seconds"] = self.duration_seconds
        return payload


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
        if isinstance(self.attempt_count, bool) or self.attempt_count < 1:
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
    timeline: TimelineOutput

    def __post_init__(self) -> None:
        _ResultMetadata.__post_init__(self)
        if not isinstance(self.timeline, TimelineOutput):
            raise TypeError("timeline must be TimelineOutput")

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
    """Keep provider diagnostics useful without echoing secret-bearing context."""

    message = str(error).replace("\x00", " ").strip()
    message = re.sub(
        r"(?is)(authorization|x-goog-api-key|api[-_ ]?key)\s*[:=]\s*"
        r"(?:bearer\s+)?[^\s,;]+",
        "[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?is)(bearer|token|secret|password|credential)\s*[:=]\s*[^\s,;]+",
        "[REDACTED]",
        message,
    )
    message = re.sub(r"(?i)\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]", message)
    message = re.sub(
        r"(?i)\b(client|session|headers?)\s*=\s*[^\s,;]+",
        "[REDACTED]",
        message,
    )
    message = " ".join(message.split())
    if not message:
        message = "no provider diagnostic"
    return message[:_MAX_ERROR_LENGTH]


def _transport_error(operation: str, error: BaseException) -> ProviderError:
    return ProviderError(
        f"Gemini {operation} failed: {type(error).__name__}: {_bounded_message(error)}"
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
        seen.add(resolved)
        files.append((role, path, mime_type or _MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")))

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
        add(f"frame-{index}", value, "image/jpeg")
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


def _parse_timeline(payload: object) -> TimelineOutput:
    if not isinstance(payload, Mapping):
        raise ValueError("timeline response must be an object")
    keys = set(payload)
    allowed = {"events", "support_spans", "schema_version", "duration_seconds"}
    if not {"events", "support_spans"}.issubset(keys) or not keys <= allowed:
        raise ValueError("timeline response has an incomplete or unknown schema")
    events = payload["events"]
    support_spans = payload["support_spans"]
    if (
        isinstance(events, (str, bytes))
        or not isinstance(events, Sequence)
        or isinstance(support_spans, (str, bytes))
        or not isinstance(support_spans, Sequence)
    ):
        raise ValueError("timeline events and support_spans must be arrays")
    return TimelineOutput(
        events=tuple(_parse_span(item, f"events[{index}]") for index, item in enumerate(events)),
        support_spans=tuple(
            _parse_span(item, f"support_spans[{index}]")
            for index, item in enumerate(support_spans)
        ),
        schema_version=payload.get("schema_version"),
        duration_seconds=payload.get("duration_seconds"),
    )


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
        timeline, metadata = self._execute(
            request, TIMELINE_RESPONSE_SCHEMA, _parse_timeline
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
        response_sha256 = hashlib.sha256(b"").hexdigest()
        parsed: Any = None
        failure: GeminiError | None = None

        try:
            for role, path, mime_type in _request_files(request, payload):
                try:
                    record = self._transport.upload(path, mime_type=mime_type)
                except Exception as error:  # noqa: BLE001 - provider boundary
                    raise _transport_error("upload", error) from None
                if not isinstance(record, UploadedFileRecord):
                    raise ProviderProtocolError(
                        "Gemini upload returned an invalid file record"
                    )
                uploaded.append((role, record))

            primary_parts = _parts(payload, uploaded)
            generated = self._generate(primary_parts, schema)
            attempts = 1
            usage += generated.usage
            response_sha256 = generated.response_sha256
            try:
                parsed = parser(generated.payload)
            except Exception:  # noqa: BLE001 - local schema boundary
                repair_prompt = _REPAIR_PREFIX + _json_bytes(generated.payload).decode(
                    "utf-8"
                )
                repaired_parts = (*primary_parts[:-1], repair_prompt)
                repaired = self._generate(repaired_parts, schema)
                attempts = 2
                usage += repaired.usage
                response_sha256 = repaired.response_sha256
                try:
                    parsed = parser(repaired.payload)
                except Exception as repair_error:  # noqa: BLE001 - schema boundary
                    raise ProviderProtocolError(
                        "Provider JSON failed local schema validation after one repair: "
                        f"{type(repair_error).__name__}: {_bounded_message(repair_error)}"
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

        if failure is not None:
            failure.uploaded_file_deletion_status = MappingProxyType(
                dict(deletion_status)
            )
            raise failure
        if any(not deleted for deleted in deletion_status.values()):
            raise ProviderError(
                "Gemini uploaded-file deletion failed",
                uploaded_file_deletion_status=deletion_status,
            )
        if attempts < 1 or parsed is None:
            raise ProviderProtocolError("Gemini produced no validated response")

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
        except GeminiError:
            raise
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

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        uploaded = self._client.files.upload(
            file=str(path), config={"mime_type": mime_type}
        )
        file_id = getattr(uploaded, "name", None)
        if not isinstance(file_id, str) or not file_id:
            raise RuntimeError("Gemini upload did not return a file name")
        uri = getattr(uploaded, "uri", None)
        return UploadedFileRecord(file_id=file_id, mime_type=mime_type, uri=uri)

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
        raise ProviderAuthenticationError(
            "Gemini API or Vertex authentication is not configured"
        ) from None
    try:
        client = genai.Client()
    except Exception:
        raise ProviderAuthenticationError(
            "Gemini API or Vertex authentication is not configured"
        ) from None
    return _GoogleGenaiTransport(client)


__all__ = [
    "ANSWER_RESPONSE_SCHEMA",
    "TIMELINE_RESPONSE_SCHEMA",
    "VERIFIER_RESPONSE_SCHEMA",
    "AnswerResult",
    "GenerationResult",
    "GeminiError",
    "GeminiSubmissionClient",
    "GeminiTransport",
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderProtocolError",
    "TimelineOutput",
    "TimelineResult",
    "TokenUsage",
    "Usage",
    "UploadedFileRecord",
    "VerifierResult",
    "create_google_transport",
]
