"""Thin, credential-injected OpenRouter transport for the VQA submission path."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from types import MappingProxyType

from trec_video_2026.vqa_submission import MODEL_ID
from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    GeminiTransport,
    ProviderAuthenticationError,
    ProviderError,
    ProviderProtocolError,
    ProviderResponseEvidenceError,
    TokenUsage,
    UploadedFileRecord,
)


OPENROUTER_MODEL_ID = "google/gemini-3.1-pro-preview"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models/user"
AUTHENTICATION_ERROR_MESSAGE = "OpenRouter API authentication is not configured"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = 30_000_000
MAX_OUTPUT_TOKENS = 32768
HTTP_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class OpenRouterRequestPolicy:
    """Optional request controls for one pinned OpenRouter generation."""

    max_output_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float | None = None
    provider_order: tuple[str, ...] = ()
    allow_fallbacks: bool | None = None
    use_structured_outputs: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number")
        if (
            not isinstance(self.provider_order, tuple)
            or not all(isinstance(provider, str) and provider for provider in self.provider_order)
        ):
            raise ValueError("provider_order must contain non-empty provider names")
        if self.allow_fallbacks is not None and not isinstance(
            self.allow_fallbacks, bool
        ):
            raise TypeError("allow_fallbacks must be a boolean or None")
        if not isinstance(self.use_structured_outputs, bool):
            raise TypeError("use_structured_outputs must be a boolean")

_MODEL_MAP = {MODEL_ID: OPENROUTER_MODEL_ID}
_ALLOWED_MIME_TYPES = frozenset(
    {
        "video/mp4",
        "video/mpeg",
        "video/quicktime",
        "video/webm",
        "video/x-matroska",
        "video/x-msvideo",
        "image/jpeg",
        "image/png",
        "image/webp",
        "audio/mp4",
        "audio/mpeg",
        "audio/wav",
    }
)
_AUDIO_FORMATS = {
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
}


class _Opener(Protocol):
    def open(self, request: urllib.request.Request, *, timeout: float) -> Any: ...


class OpenRouterRequestError(ProviderError):
    """A bounded HTTP failure with no provider body retained."""

    def __init__(self, status_code: int) -> None:
        super().__init__("OpenRouter request failed", failure_code="provider_request")
        self.status_code = status_code


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
            "OpenRouter request contains unsupported JSON values"
        ) from None


def _preserved_response_text(response: object) -> str:
    """Prefer assistant content, otherwise preserve the bounded response envelope."""

    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and len(choices) == 1:
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, Mapping) else None
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list) and content:
                text_blocks = [
                    item.get("text")
                    for item in content
                    if isinstance(item, Mapping)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ]
                if len(text_blocks) == len(content):
                    return "".join(text_blocks)
    return json.dumps(
        response,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _preserved_body_text(body: bytes) -> str:
    """Preserve the bounded HTTP body even when its JSON envelope is malformed."""

    return body.decode("utf-8", errors="replace")


def _model_slug(model: object) -> str:
    if not isinstance(model, str) or model not in _MODEL_MAP:
        raise ProviderProtocolError("OpenRouter model is not supported")
    return _MODEL_MAP[model]


def _api_key() -> str:
    value = os.environ.get("OPENROUTER_API_KEY")
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
        raise ProviderError("OpenRouter response did not include a valid status")
    return value


def _response_json(body: bytes) -> object:
    if len(body) > MAX_RESPONSE_BYTES:
        raise ProviderProtocolError("OpenRouter response exceeded the size limit")
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProviderProtocolError("OpenRouter response was not valid JSON") from None


def _token_count(value: object, *, required: bool) -> int | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderProtocolError("OpenRouter token usage was invalid")
    return value


class OpenRouterTransport:
    """Implement the provider-neutral GeminiTransport over OpenRouter chat."""

    provider_name = "openrouter"

    def __init__(
        self,
        api_key: str,
        *,
        opener: _Opener | None = None,
        policy: OpenRouterRequestPolicy | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ProviderAuthenticationError(AUTHENTICATION_ERROR_MESSAGE)
        if policy is not None and not isinstance(policy, OpenRouterRequestPolicy):
            raise TypeError("policy must be an OpenRouterRequestPolicy or None")
        self._api_key = api_key
        self._opener = opener
        self._policy = policy or OpenRouterRequestPolicy()
        self._uploads: dict[str, tuple[str, str, str]] = {}

    def __repr__(self) -> str:
        return "OpenRouterTransport(provider='openrouter')"

    @property
    def response_format_policy(self) -> str:
        if self._policy.use_structured_outputs:
            return "native-json-schema-v1"
        return "prompt-json-local-validation-v3-exact-fence"

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        if not isinstance(path, Path):
            path = Path(path)
        if not isinstance(mime_type, str) or mime_type not in _ALLOWED_MIME_TYPES:
            raise ProviderProtocolError("OpenRouter media MIME type is not allowed")
        try:
            if path.is_symlink():
                raise ProviderProtocolError("OpenRouter media must be a regular file")
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ProviderProtocolError("OpenRouter media must be a regular file")
            if metadata.st_size > MAX_UPLOAD_BYTES:
                raise ProviderProtocolError("OpenRouter media exceeds the size limit")
            digest = hashlib.sha256()
            data = bytearray()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    data.extend(chunk)
                    if len(data) > MAX_UPLOAD_BYTES:
                        raise ProviderProtocolError(
                            "OpenRouter media exceeds the size limit"
                        )
                    digest.update(chunk)
        except ProviderProtocolError:
            raise
        except (OSError, ValueError):
            raise ProviderProtocolError("OpenRouter media could not be read") from None
        path_identity = hashlib.sha256(
            f"{path.resolve()}\0{mime_type}".encode("utf-8")
        ).hexdigest()[:16]
        file_id = f"local-sha256-{digest.hexdigest()}-{path_identity}"
        data_uri = (
            f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"
        )
        self._uploads[file_id] = (data_uri, mime_type, digest.hexdigest())
        return UploadedFileRecord(
            file_id=file_id,
            mime_type=mime_type,
            sha256=digest.hexdigest(),
        )

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
                raise ProviderAuthenticationError(
                    "OpenRouter authentication failed"
                ) from None
            raise OpenRouterRequestError(error.code) from None
        except (OSError, TimeoutError, Exception):
            raise ProviderError("OpenRouter request failed") from None
        if not isinstance(body, bytes):
            raise ProviderProtocolError("OpenRouter response body was invalid")
        return status, body

    def _request_json(self, request: urllib.request.Request) -> object:
        status, body = self._open(request)
        if status in {401, 403}:
            raise ProviderAuthenticationError("OpenRouter authentication failed")
        if status < 200 or status >= 300:
            raise OpenRouterRequestError(status)
        payload = _response_json(body)
        if isinstance(payload, Mapping) and "error" in payload:
            raise ProviderError("OpenRouter provider returned an error")
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _part_content(
        self, part: object, seen_file_ids: set[str]
    ) -> tuple[dict[str, object], dict[str, object]]:
        if isinstance(part, str):
            return {"type": "text", "text": part}, {"type": "text", "text": part}
        if isinstance(part, Mapping) and "file_id" not in part:
            text = part.get("text")
            if set(part) == {"text"} and isinstance(text, str):
                return {"type": "text", "text": text}, {"type": "text", "text": text}
        if not isinstance(part, Mapping):
            raise ProviderProtocolError("OpenRouter request part was invalid")
        file_id = part.get("file_id")
        mime_type = part.get("mime_type")
        if not isinstance(file_id, str) or not file_id:
            raise ProviderProtocolError("OpenRouter file identifier was invalid")
        if file_id in seen_file_ids:
            raise ProviderProtocolError("OpenRouter request contains a duplicate file")
        stored = self._uploads.get(file_id)
        if stored is None:
            raise ProviderProtocolError("OpenRouter file identifier is unknown")
        data_uri, stored_mime_type, stored_sha256 = stored
        if mime_type != stored_mime_type:
            raise ProviderProtocolError("OpenRouter file MIME type did not match")
        seen_file_ids.add(file_id)
        encoded = data_uri.rsplit(",", 1)[-1]
        if mime_type.startswith("video/"):
            rendered = {
                "type": "video_url",
                "video_url": {"url": data_uri},
            }
        elif mime_type.startswith("image/"):
            rendered = {
                "type": "image_url",
                "image_url": {"url": data_uri},
            }
        elif mime_type.startswith("audio/"):
            rendered = {
                "type": "input_audio",
                "input_audio": {
                    "data": encoded,
                    "format": _AUDIO_FORMATS[mime_type],
                },
            }
        else:
            raise ProviderProtocolError("OpenRouter media MIME type is not allowed")
        identity = {
            "type": "file",
            "mime_type": mime_type,
            "sha256": stored_sha256,
        }
        return rendered, identity

    @staticmethod
    def _provider_metadata(response: Mapping[str, object]) -> dict[str, object]:
        metadata: dict[str, object] = {}
        for key in ("provider", "model"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                metadata[key] = value
        request_id = response.get("request_id")
        if request_id is None:
            request_id = response.get("id")
        if isinstance(request_id, str) and request_id.strip():
            metadata["request_id"] = request_id
        usage = response.get("usage")
        if isinstance(usage, Mapping) and "cost" in usage:
            cost = usage.get("cost")
            if cost is not None:
                if (
                    isinstance(cost, bool)
                    or not isinstance(cost, (int, float))
                    or not math.isfinite(float(cost))
                    or cost < 0
                ):
                    raise ProviderProtocolError(
                        "OpenRouter reported cost was invalid"
                    )
                metadata["reported_cost_usd"] = float(cost)
        return metadata

    def generate(
        self,
        *,
        model: str,
        parts: Sequence[object],
        schema: Mapping[str, object],
    ) -> GenerationResult:
        model_slug = _model_slug(model)
        if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
            raise ProviderProtocolError("OpenRouter request parts were invalid")
        if not isinstance(schema, Mapping):
            raise ProviderProtocolError("OpenRouter response schema was invalid")
        seen_file_ids: set[str] = set()
        content: list[dict[str, object]] = []
        identities: list[dict[str, object]] = []
        for part in parts:
            rendered, identity = self._part_content(part, seen_file_ids)
            content.append(rendered)
            identities.append(identity)
        payload = {
            "model": model_slug,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "max_tokens": self._policy.max_output_tokens,
            "reasoning": {"effort": "high", "exclude": True},
            "provider": {
                "zdr": True,
                "data_collection": "deny",
                "require_parameters": True,
            },
        }
        if self._policy.use_structured_outputs:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "vqa_response",
                    "strict": True,
                    "schema": dict(schema),
                },
            }
        if self._policy.temperature is not None:
            payload["temperature"] = self._policy.temperature
        if self._policy.provider_order:
            payload["provider"]["order"] = list(self._policy.provider_order)
        if self._policy.allow_fallbacks is not None:
            payload["provider"]["allow_fallbacks"] = self._policy.allow_fallbacks
        seed_input = dict(payload)
        seed_input["messages"] = [{"role": "user", "content": identities}]
        seed = int(hashlib.sha256(_json_bytes(seed_input)).hexdigest()[:16], 16) & 0x7FFF_FFFF
        payload["seed"] = seed
        request_bytes = _json_bytes(payload)
        if len(request_bytes) > MAX_REQUEST_BYTES:
            raise ProviderProtocolError("OpenRouter request exceeds the size limit")
        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=request_bytes,
            headers=self._headers(),
            method="POST",
        )
        status, body = self._open(request)
        if status in {401, 403}:
            raise ProviderAuthenticationError("OpenRouter authentication failed")
        if status < 200 or status >= 300:
            raise OpenRouterRequestError(status)
        try:
            response = _response_json(body)
            if isinstance(response, Mapping) and "error" in response:
                raise ProviderError("OpenRouter provider returned an error")
            return self._generation_result_from_response(response)
        except ProviderProtocolError as error:
            raw_response = (
                _preserved_response_text(response)
                if "response" in locals()
                else _preserved_body_text(body)
            )
            evidence = GenerationResult(
                payload={},
                raw_response=raw_response,
                usage=TokenUsage(input_tokens=0, output_tokens=0),
                provider_metadata=self._safe_provider_metadata(
                    response if "response" in locals() else None
                ),
            )
            # GenerationResult intentionally rejects arbitrary provider keys.
            # These two bounded evidence flags are safe, structured facts needed
            # by the immutable runner's retry classifier; they never contain the
            # provider response body or untrusted metadata.
            evidence_metadata = self._safe_evidence_metadata(
                response if "response" in locals() else None
            )
            object.__setattr__(
                evidence,
                "provider_metadata",
                MappingProxyType(evidence_metadata),
            )
            raise ProviderResponseEvidenceError(
                str(error),
                generation_result=evidence,
            ) from error

    def _generation_result_from_response(self, response: object) -> GenerationResult:
        if not isinstance(response, Mapping):
            raise ProviderProtocolError("OpenRouter response envelope was invalid")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ProviderProtocolError("OpenRouter response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ProviderProtocolError("OpenRouter response was not a successful message")
        if choice.get("finish_reason") == "error":
            raise ProviderError("OpenRouter provider returned a generation error")
        if choice.get("finish_reason") != "stop":
            raise ProviderProtocolError("OpenRouter response was not a successful message")
        message = choice.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise ProviderProtocolError("OpenRouter response message was invalid")
        content_value = message.get("content")
        if isinstance(content_value, str):
            raw_response = content_value
        elif isinstance(content_value, list) and content_value:
            text_blocks = [
                item.get("text")
                for item in content_value
                if isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ]
            if len(text_blocks) != len(content_value):
                raise ProviderProtocolError("OpenRouter response message was invalid")
            raw_response = "".join(text_blocks)
        else:
            raise ProviderProtocolError("OpenRouter response message was invalid")
        if not raw_response.strip():
            raise ProviderProtocolError("OpenRouter response message was invalid")
        try:
            parsed = json.loads(raw_response)
        except (TypeError, json.JSONDecodeError):
            if self._policy.use_structured_outputs:
                raise ProviderProtocolError(
                    "OpenRouter message was not valid JSON"
                ) from None
            parsed = raw_response
        if self._policy.use_structured_outputs and not isinstance(parsed, Mapping):
            raise ProviderProtocolError("OpenRouter message must be a JSON object")
        usage_payload = response.get("usage")
        if not isinstance(usage_payload, Mapping):
            raise ProviderProtocolError("OpenRouter token usage was invalid")
        input_tokens = _token_count(usage_payload.get("prompt_tokens"), required=True)
        output_tokens = _token_count(
            usage_payload.get("completion_tokens"), required=True
        )
        total_tokens = _token_count(usage_payload.get("total_tokens"), required=False)
        details = usage_payload.get("prompt_tokens_details") or {}
        if not isinstance(details, Mapping):
            raise ProviderProtocolError("OpenRouter token usage was invalid")
        cached_tokens = _token_count(details.get("cached_tokens", 0), required=True)
        return GenerationResult(
            payload=dict(parsed) if isinstance(parsed, Mapping) else parsed,
            raw_response=raw_response,
            usage=TokenUsage(
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
                total_tokens=total_tokens,
                cached_input_tokens=cached_tokens or 0,
            ),
            provider_metadata=self._provider_metadata(response),
        )

    @staticmethod
    def _safe_provider_metadata(response: object) -> dict[str, object]:
        if not isinstance(response, Mapping):
            return {}
        metadata: dict[str, object] = {}
        for key in ("provider", "model"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                metadata[key] = value
        request_id = response.get("request_id")
        if request_id is None:
            request_id = response.get("id")
        if isinstance(request_id, str) and request_id.strip():
            metadata["request_id"] = request_id
        usage = response.get("usage")
        cost = usage.get("cost") if isinstance(usage, Mapping) else None
        if (
            not isinstance(cost, bool)
            and isinstance(cost, (int, float))
            and math.isfinite(float(cost))
            and cost >= 0
        ):
            metadata["reported_cost_usd"] = float(cost)
        return metadata

    @classmethod
    def _safe_evidence_metadata(cls, response: object) -> dict[str, object]:
        metadata = cls._safe_provider_metadata(response)
        if not isinstance(response, Mapping):
            return metadata
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if isinstance(choice, Mapping):
            finish_reason = choice.get("finish_reason")
            if (
                isinstance(finish_reason, str)
                and 0 < len(finish_reason) <= 64
                and finish_reason.isascii()
                and all(character.isalnum() or character in "_.-" for character in finish_reason)
            ):
                metadata["finish_reason"] = finish_reason
            message = choice.get("message")
            if isinstance(message, Mapping) and "content" in message:
                content = message["content"]
                if isinstance(content, str):
                    metadata["content_empty"] = not bool(content.strip())
                elif isinstance(content, list):
                    valid_text_blocks = all(
                        isinstance(item, Mapping)
                        and item.get("type") == "text"
                        and isinstance(item.get("text"), str)
                        for item in content
                    )
                    if valid_text_blocks:
                        metadata["content_empty"] = not bool(
                            "".join(item["text"] for item in content).strip()
                        )
                    else:
                        metadata["content_empty"] = False
        return metadata

    def preflight(self, *, model_id: str) -> dict[str, object]:
        model_slug = _model_slug(model_id)
        request = urllib.request.Request(
            OPENROUTER_MODELS_URL,
            headers=self._headers(),
            method="GET",
        )
        response = self._request_json(request)
        if not isinstance(response, Mapping) or not isinstance(response.get("data"), list):
            raise ProviderProtocolError("OpenRouter model catalog was invalid")
        model_record = next(
            (
                item
                for item in response["data"]
                if isinstance(item, Mapping)
                and item.get("id") in {model_slug, model_id}
            ),
            None,
        )
        if not isinstance(model_record, Mapping):
            raise ProviderProtocolError("OpenRouter model was not in the catalog")
        architecture = model_record.get("architecture")
        modalities = model_record.get("input_modalities")
        if modalities is None and isinstance(architecture, Mapping):
            modalities = architecture.get("input_modalities")
        if (
            not isinstance(modalities, Sequence)
            or isinstance(modalities, (str, bytes))
            or not all(isinstance(item, str) and item for item in modalities)
        ):
            raise ProviderProtocolError("OpenRouter model modalities were invalid")
        supported = model_record.get("supported_parameters", [])
        if not isinstance(supported, Sequence) or isinstance(supported, (str, bytes)) or not all(
            isinstance(item, str) for item in supported
        ):
            raise ProviderProtocolError("OpenRouter model capabilities were invalid")
        input_modalities = {item.lower() for item in modalities}
        if not {"audio", "image", "text", "video"}.issubset(input_modalities):
            raise ProviderProtocolError("OpenRouter model lacks required modalities")
        supported_lower = {item.lower() for item in supported}
        if not {"response_format", "structured_outputs"}.intersection(supported_lower):
            raise ProviderProtocolError("OpenRouter model lacks structured output support")
        return {
            "authenticated": True,
            "model_id": model_id,
            "input_modalities": sorted(input_modalities),
            "structured_outputs": True,
        }


def create_openrouter_transport(
    *,
    opener: _Opener | None = None,
    policy: OpenRouterRequestPolicy | None = None,
) -> GeminiTransport:
    """Create an OpenRouter transport from only OPENROUTER_API_KEY."""

    return OpenRouterTransport(_api_key(), opener=opener, policy=policy)


def preflight_openrouter_transport(
    *, model_id: str = MODEL_ID, opener: _Opener | None = None
) -> dict[str, object]:
    """Perform the authenticated, no-media model-catalog check."""

    return OpenRouterTransport(_api_key(), opener=opener).preflight(model_id=model_id)


__all__ = [
    "AUTHENTICATION_ERROR_MESSAGE",
    "MAX_OUTPUT_TOKENS",
    "MAX_UPLOAD_BYTES",
    "MODEL_ID",
    "OPENROUTER_CHAT_URL",
    "OPENROUTER_MODEL_ID",
    "OPENROUTER_MODELS_URL",
    "OpenRouterRequestPolicy",
    "OpenRouterTransport",
    "create_openrouter_transport",
    "preflight_openrouter_transport",
]
