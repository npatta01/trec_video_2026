"""Bounded, credential-injected Reka Core transport for the frontier map."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import stat
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from trec_video_2026.vqa_frontier_map import EVENT_MAP_SCHEMA
from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    GeminiTransport,
    ProviderAuthenticationError,
    ProviderError,
    ProviderProtocolError,
    TokenUsage,
    UploadedFileRecord,
)


REKA_MODEL_ID = "reka-core"
REKA_CHAT_URL = "https://api.reka.ai/v1/chat/completions"
REKA_MODELS_URL = "https://api.reka.ai/v1/models"
AUTHENTICATION_ERROR_MESSAGE = "Reka API authentication is not configured"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_TOKENS = 16384
HTTP_TIMEOUT_SECONDS = 180.0

_ALLOWED_MIME_TYPES = frozenset(
    {
        "video/mp4",
        "video/mpeg",
        "video/quicktime",
        "video/webm",
        "video/x-matroska",
        "video/x-msvideo",
    }
)
_INVALID_GENERATION_SHAPE = (
    "Reka request must contain exactly one video followed by one non-empty prompt"
)


class _Opener(Protocol):
    def open(self, request: urllib.request.Request, *, timeout: float) -> Any: ...


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ProviderProtocolError(
            "Reka request contains unsupported JSON values"
        ) from None


def _api_key() -> str:
    value = os.environ.get("REKA_API_KEY")
    if not isinstance(value, str) or not value.strip():
        raise ProviderAuthenticationError(AUTHENTICATION_ERROR_MESSAGE)
    return value


def _status(response: object) -> int:
    value = getattr(response, "status", None)
    if value is None:
        getcode = getattr(response, "getcode", None)
        if callable(getcode):
            value = getcode()
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderError("Reka response did not include a valid status")
    return value


def _response_json(body: bytes) -> object:
    if len(body) > MAX_RESPONSE_BYTES:
        raise ProviderProtocolError("Reka response exceeded the size limit")
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProviderProtocolError("Reka response was not valid JSON") from None


def _token_count(value: object, *, required: bool) -> int | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderProtocolError("Reka token usage was invalid")
    return value


def _schema_type_matches(value: object, schema_type: object) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return False


def _matches_schema(value: object, schema: Mapping[str, object]) -> bool:
    schema_type = schema.get("type")
    if not isinstance(schema_type, str) or not _schema_type_matches(value, schema_type):
        return False
    if "const" in schema and value != schema["const"]:
        return False
    enum = schema.get("enum")
    if enum is not None and (
        not isinstance(enum, list) or not any(value == candidate for candidate in enum)
    ):
        return False

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            return False
        if any(
            not isinstance(name, str) or name not in value for name in required
        ):
            return False
        if schema.get("additionalProperties") is False and any(
            name not in properties for name in value
        ):
            return False
        return all(
            name not in value
            or not isinstance(child_schema, Mapping)
            or _matches_schema(value[name], child_schema)
            for name, child_schema in properties.items()
        )

    if schema_type == "array":
        items = schema.get("items")
        if not isinstance(items, Mapping):
            return False
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return False
        if isinstance(max_items, int) and len(value) > max_items:
            return False
        if schema.get("uniqueItems") is True:
            try:
                if len({_json_bytes(item) for item in value}) != len(value):
                    return False
            except ProviderProtocolError:
                return False
        return all(_matches_schema(item, items) for item in value)

    if schema_type == "string":
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        return not (
            isinstance(min_length, int) and len(value) < min_length
            or isinstance(max_length, int) and len(value) > max_length
        )

    if schema_type in {"number", "integer"}:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            return False
        if isinstance(maximum, (int, float)) and value > maximum:
            return False
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            return False
        if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
            return False
    return True


def _validate_response_payload(
    payload: object, schema: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ProviderProtocolError("Reka message must be a JSON object")
    if not _matches_schema(payload, schema):
        raise ProviderProtocolError("Reka message did not match the response schema")
    return payload


def _schema_prompt(prompt: object) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProviderProtocolError("Reka text prompt was invalid")
    compact_schema = json.dumps(
        EVENT_MAP_SCHEMA,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{prompt}\n\nReturn only JSON matching this schema:\n{compact_schema}"


class RekaTransport:
    """Implement the provider-neutral GeminiTransport over Reka Chat."""

    provider_name = "reka"

    def __init__(
        self,
        api_key: str,
        *,
        opener: _Opener | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ProviderAuthenticationError(AUTHENTICATION_ERROR_MESSAGE)
        self._api_key = api_key
        self._opener = opener
        self._uploads: dict[str, tuple[str, str]] = {}

    def __repr__(self) -> str:
        return "RekaTransport(provider='reka', model='reka-core')"

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        if not isinstance(path, Path):
            path = Path(path)
        if not isinstance(mime_type, str) or mime_type not in _ALLOWED_MIME_TYPES:
            raise ProviderProtocolError("Reka media MIME type is not allowed")
        try:
            if path.is_symlink():
                raise ProviderProtocolError("Reka media must be a regular file")
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ProviderProtocolError("Reka media must be a regular file")
            if metadata.st_size > MAX_UPLOAD_BYTES:
                raise ProviderProtocolError("Reka media exceeds the size limit")
            digest = hashlib.sha256()
            data = bytearray()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    data.extend(chunk)
                    if len(data) > MAX_UPLOAD_BYTES:
                        raise ProviderProtocolError("Reka media exceeds the size limit")
                    digest.update(chunk)
        except ProviderProtocolError:
            raise
        except (OSError, ValueError):
            raise ProviderProtocolError("Reka media could not be read") from None
        path_identity = hashlib.sha256(
            f"{path.resolve()}\0{mime_type}".encode("utf-8")
        ).hexdigest()[:16]
        file_id = f"local-sha256-{digest.hexdigest()}-{path_identity}"
        data_uri = (
            f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
        )
        self._uploads[file_id] = (data_uri, mime_type)
        return UploadedFileRecord(file_id=file_id, mime_type=mime_type)

    def delete(self, file_id: str) -> None:
        if isinstance(file_id, str):
            self._uploads.pop(file_id, None)

    def _open(self, request: urllib.request.Request) -> tuple[int, bytes]:
        try:
            if self._opener is None:
                response = urllib.request.urlopen(  # noqa: S310 - fixed HTTPS URLs
                    request, timeout=HTTP_TIMEOUT_SECONDS
                )
            else:
                response = self._opener.open(request, timeout=HTTP_TIMEOUT_SECONDS)
            try:
                status = _status(response)
                body = response.read(MAX_RESPONSE_BYTES + 1)
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except (ProviderError, ProviderProtocolError):
            raise
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                raise ProviderAuthenticationError("Reka authentication failed") from None
            raise ProviderError("Reka request failed") from None
        except (OSError, TimeoutError, Exception):
            raise ProviderError("Reka request failed") from None
        if not isinstance(body, bytes):
            raise ProviderProtocolError("Reka response body was invalid")
        return status, body

    def _request_json(self, request: urllib.request.Request) -> object:
        status, body = self._open(request)
        if status in {401, 403}:
            raise ProviderAuthenticationError("Reka authentication failed")
        if status < 200 or status >= 300:
            raise ProviderError("Reka request failed")
        payload = _response_json(body)
        if isinstance(payload, Mapping) and "error" in payload:
            raise ProviderError("Reka provider returned an error")
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "X-Api-Key": self._api_key,
            "Content-Type": "application/json",
        }

    def _part_content(
        self, part: object, seen_file_ids: set[str]
    ) -> dict[str, object]:
        if isinstance(part, str):
            return {"type": "text", "text": _schema_prompt(part)}
        if isinstance(part, Mapping) and "file_id" not in part:
            text = part.get("text")
            if set(part) == {"text"} and isinstance(text, str):
                return {"type": "text", "text": _schema_prompt(text)}
        if not isinstance(part, Mapping):
            raise ProviderProtocolError("Reka request part was invalid")
        file_id = part.get("file_id")
        mime_type = part.get("mime_type")
        if not isinstance(file_id, str) or not file_id:
            raise ProviderProtocolError("Reka file identifier was invalid")
        if file_id in seen_file_ids:
            raise ProviderProtocolError("Reka request contains a duplicate file")
        stored = self._uploads.get(file_id)
        if stored is None:
            raise ProviderProtocolError("Reka file identifier is unknown")
        data_uri, stored_mime_type = stored
        if mime_type != stored_mime_type:
            raise ProviderProtocolError("Reka file MIME type did not match")
        seen_file_ids.add(file_id)
        return {"type": "video_url", "video_url": data_uri}

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: Mapping[str, object],
    ) -> GenerationResult:
        if model != REKA_MODEL_ID:
            raise ProviderProtocolError("Reka model is not supported")
        if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
            raise ProviderProtocolError("Reka request parts were invalid")
        if not isinstance(schema, Mapping):
            raise ProviderProtocolError("Reka response schema was invalid")
        if len(parts) != 2:
            raise ProviderProtocolError(_INVALID_GENERATION_SHAPE)
        video_part, prompt_part = parts
        if (
            not isinstance(video_part, Mapping)
            or "file_id" not in video_part
            or not isinstance(prompt_part, str)
            or not prompt_part.strip()
        ):
            raise ProviderProtocolError(_INVALID_GENERATION_SHAPE)
        seen_file_ids: set[str] = set()
        content = [
            self._part_content(video_part, seen_file_ids),
            self._part_content(prompt_part, seen_file_ids),
        ]
        if content[0]["type"] != "video_url":
            raise ProviderProtocolError(_INVALID_GENERATION_SHAPE)
        payload: dict[str, object] = {
            "model": REKA_MODEL_ID,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        }
        seed = int(hashlib.sha256(_json_bytes(payload)).hexdigest()[:16], 16) & 0x7FFF_FFFF
        payload["seed"] = seed
        request = urllib.request.Request(
            REKA_CHAT_URL,
            data=_json_bytes(payload),
            headers=self._headers(),
            method="POST",
        )
        response = self._request_json(request)
        if not isinstance(response, Mapping):
            raise ProviderProtocolError("Reka response envelope was invalid")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ProviderProtocolError("Reka response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ProviderProtocolError("Reka response was not a successful message")
        if choice.get("finish_reason") == "error":
            raise ProviderError("Reka provider returned a generation error")
        if choice.get("finish_reason") != "stop":
            raise ProviderProtocolError("Reka response was not a successful message")
        message = choice.get("message")
        if (
            not isinstance(message, Mapping)
            or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ProviderProtocolError("Reka response message was invalid")
        raw_response = message["content"]
        try:
            parsed = json.loads(raw_response)
        except (TypeError, json.JSONDecodeError):
            raise ProviderProtocolError("Reka message was not valid JSON") from None
        payload_result = _validate_response_payload(parsed, schema)
        usage_payload = response.get("usage")
        if not isinstance(usage_payload, Mapping):
            raise ProviderProtocolError("Reka token usage was invalid")
        input_tokens = _token_count(
            usage_payload.get("prompt_tokens", usage_payload.get("input_tokens")),
            required=True,
        )
        output_tokens = _token_count(
            usage_payload.get("completion_tokens", usage_payload.get("output_tokens")),
            required=True,
        )
        total_tokens = _token_count(usage_payload.get("total_tokens"), required=False)
        return GenerationResult(
            payload=payload_result,
            raw_response=raw_response,
            usage=TokenUsage(
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
                total_tokens=total_tokens,
            ),
            provider_metadata={"provider": "reka", "model": REKA_MODEL_ID},
        )

    def preflight(self, *, model_id: str = REKA_MODEL_ID) -> dict[str, object]:
        if model_id != REKA_MODEL_ID:
            raise ProviderProtocolError("Reka model is not supported")
        request = urllib.request.Request(
            REKA_MODELS_URL,
            headers=self._headers(),
            method="GET",
        )
        response = self._request_json(request)
        records: object = response.get("data") if isinstance(response, Mapping) else response
        if not isinstance(records, list):
            raise ProviderProtocolError("Reka model catalog was invalid")
        if not any(
            isinstance(record, Mapping) and record.get("id") == REKA_MODEL_ID
            for record in records
        ):
            raise ProviderProtocolError("Reka model was not in the catalog")
        return {"authenticated": True, "model_id": model_id}


def create_reka_transport(*, opener: _Opener | None = None) -> GeminiTransport:
    """Create a Reka transport from only REKA_API_KEY."""

    return RekaTransport(_api_key(), opener=opener)


def preflight_reka_transport(
    *, model_id: str = REKA_MODEL_ID, opener: _Opener | None = None
) -> dict[str, object]:
    """Perform the authenticated, no-media model-catalog check."""

    return RekaTransport(_api_key(), opener=opener).preflight(model_id=model_id)


__all__ = [
    "AUTHENTICATION_ERROR_MESSAGE",
    "HTTP_TIMEOUT_SECONDS",
    "MAX_OUTPUT_TOKENS",
    "MAX_RESPONSE_BYTES",
    "MAX_UPLOAD_BYTES",
    "REKA_CHAT_URL",
    "REKA_MODEL_ID",
    "REKA_MODELS_URL",
    "RekaTransport",
    "create_reka_transport",
    "preflight_reka_transport",
]
