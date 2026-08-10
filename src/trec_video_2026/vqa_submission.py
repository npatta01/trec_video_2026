"""Immutable host-side contracts for the label-blind VQA submission path."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trec_video_2026.vqa_experiments.temporal_bottleneck import (
    freeze_json_artifact,
    verify_json_artifact,
)
from trec_video_2026.vqa_gate import canonical_request_sha256


SCHEMA_VERSION = 1
MODEL_ID = "gemini-3.1-pro-preview"
PASS_IDS = (
    "direct-av",
    "direct-av-alternate",
    "global-transcript",
    "global-local-timeline",
    "question-specialist",
    "local-only",
    "audio-first",
    "scene-boundary",
    "detail-temporal-route",
    "conservative-synthesis",
)
FALLBACK_PASS_PRECEDENCE = (
    "global-local-timeline",
    "direct-av",
    "question-specialist",
    "detail-temporal-route",
    "global-transcript",
    "local-only",
    "scene-boundary",
    "audio-first",
    "direct-av-alternate",
    "conservative-synthesis",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_LABEL_TERMS = (
    "answer",
    "reference",
    "annotation",
    "label",
    "metric",
    "score",
    "gold",
    "oracle",
)
_KEY_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[^A-Za-z0-9]+")
_SAFE_PROVENANCE_TOKEN = re.compile(r"[A-Za-z0-9_.:@-]{1,256}\Z")
_SAFE_PROVENANCE_DEVICE = re.compile(r"[A-Za-z0-9_.:() -]{1,64}\Z")
_SAFE_PROVENANCE_FIELDS = frozenset(
    {
        "device",
        "model_id",
        "model_revision",
        "precision",
        "provider",
        "request_sha256",
        "response_sha256",
        "runtime",
        "runtime_identity",
        "sdk_version",
        "status",
    }
)
_SAFE_PROVENANCE_STATUSES = frozenset(
    {"failed", "passed", "timed_out", "unsupported"}
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw(nested) for nested in value]
    return copy.deepcopy(value)


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using the submission contract's canonical UTF-8 form."""

    canonical_value = _thaw(value)
    if isinstance(canonical_value, Mapping):
        # Keep request hashing aligned with the existing gate implementation.
        return canonical_request_sha256(canonical_value)
    return hashlib.sha256(_canonical_json(canonical_value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("mapping keys must be strings")
            frozen[key] = _freeze(nested)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(nested) for nested in value)
    if isinstance(value, (set, frozenset)):
        raise TypeError("sets are not JSON serializable")
    return copy.deepcopy(value)


def _as_tuple(value: Any, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be a sequence") from exc


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if nonnegative and number < 0:
        raise ValueError(f"{field} cannot be negative")
    return number


def _validate_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256")
    return value


def _validate_text_tuple(value: Any, field: str) -> tuple[str, ...]:
    values = _as_tuple(value, field)
    for index, item in enumerate(values):
        _nonempty_text(item, f"{field}[{index}]")
    return values  # type: ignore[return-value]


def _validate_runtime_provenance(
    value: Mapping[str, object],
) -> Mapping[str, object]:
    if not value:
        raise ValueError("runtime_provenance must not be empty")
    validated: dict[str, object] = {}
    for key, raw_value in value.items():
        if key not in _SAFE_PROVENANCE_FIELDS:
            raise ValueError(
                f"runtime_provenance contains unsupported field: {key}"
            )
        if not isinstance(raw_value, str):
            raise TypeError(f"runtime_provenance[{key}] must be a string")
        if key in {"request_sha256", "response_sha256"}:
            validated[key] = _validate_sha256(
                raw_value, f"runtime_provenance[{key}]"
            )
        elif key == "model_id":
            if raw_value != MODEL_ID:
                raise ValueError(
                    f"runtime_provenance[{key}] must be {MODEL_ID}"
                )
            validated[key] = raw_value
        elif key == "status":
            if raw_value not in _SAFE_PROVENANCE_STATUSES:
                raise ValueError(
                    f"runtime_provenance[{key}] has an unsupported status"
                )
            validated[key] = raw_value
        else:
            pattern = (
                _SAFE_PROVENANCE_DEVICE
                if key == "device"
                else _SAFE_PROVENANCE_TOKEN
            )
            if pattern.fullmatch(raw_value) is None:
                raise ValueError(
                    f"runtime_provenance[{key}] must be a safe scalar token"
                )
            validated[key] = raw_value
    return MappingProxyType(validated)


def _validate_verifier_contract(
    ranked_pass_ids: Sequence[str], support_by_pass: Mapping[str, str]
) -> None:
    if (
        len(ranked_pass_ids) != len(PASS_IDS)
        or len(set(ranked_pass_ids)) != len(ranked_pass_ids)
        or set(ranked_pass_ids) != set(PASS_IDS)
    ):
        raise ValueError("ranked_pass_ids must be a complete permutation")
    if not set(support_by_pass).issubset(set(ranked_pass_ids)):
        raise ValueError("support_by_pass contains an unknown ranked pass")
    for pass_id, support in support_by_pass.items():
        if not isinstance(pass_id, str) or not isinstance(support, str):
            raise TypeError("support_by_pass must map pass IDs to strings")


def _forbidden_term(text: str) -> str | None:
    normalized = " ".join(
        token.casefold() for token in _KEY_BOUNDARY.split(text) if token
    )
    for term in _FORBIDDEN_LABEL_TERMS:
        if term in normalized:
            return term
    return None


def _is_path_key(key: str) -> bool:
    normalized = _KEY_BOUNDARY.sub("_", key).casefold().strip("_")
    return (
        normalized in {"path", "paths", "file", "files", "directory", "url"}
        or normalized.endswith(("_path", "_paths", "_file", "_files", "_url"))
    )


def _audit(value: Any, *, location: str, path_context: bool = False) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            forbidden = _forbidden_term(key_text)
            if forbidden is not None:
                raise ValueError(
                    f"label leakage in {location}.{key_text}: forbidden term {forbidden}"
                )
            _audit(
                nested,
                location=f"{location}.{key_text}",
                path_context=path_context or _is_path_key(key_text),
            )
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _audit(
                nested,
                location=f"{location}[{index}]",
                path_context=path_context,
            )
        return
    if isinstance(value, Path):
        value = str(value)
        path_context = True
    if isinstance(value, str) and (path_context or "/" in value or "\\" in value):
        for component in re.split(r"[/\\]+", value):
            forbidden = _forbidden_term(component)
            if forbidden is not None:
                raise ValueError(
                    f"label leakage in {location}: forbidden path component {forbidden}"
                )


def audit_label_free_payload(payload: Any) -> None:
    """Reject label-bearing keys and path components recursively."""

    _audit(payload, location="payload")


@dataclass(frozen=True)
class TimelineSpan:
    span_id: str
    start_seconds: float
    end_seconds: float
    modalities: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        _nonempty_text(self.span_id, "span_id")
        start = _finite_number(self.start_seconds, "start_seconds", nonnegative=True)
        end = _finite_number(self.end_seconds, "end_seconds", nonnegative=True)
        if end < start:
            raise ValueError("end_seconds cannot precede start_seconds")
        modalities = _validate_text_tuple(self.modalities, "modalities")
        if len(set(modalities)) != len(modalities):
            raise ValueError("modalities must be unique")
        _nonempty_text(self.description, "description")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "modalities", modalities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "modalities": list(self.modalities),
            "description": self.description,
        }


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    protocol_version: str
    run_id: str
    ordered_video_ids: tuple[str, ...]
    panel_sha256: str
    model_id: str
    runtime_identity: str
    evidence_policy_sha256: str
    pass_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        _nonempty_text(self.protocol_version, "protocol_version")
        _nonempty_text(self.run_id, "run_id")
        video_ids = _validate_text_tuple(self.ordered_video_ids, "ordered_video_ids")
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("ordered_video_ids must be unique")
        _validate_sha256(self.panel_sha256, "panel_sha256")
        if self.model_id != MODEL_ID:
            raise ValueError(f"model_id must be {MODEL_ID}")
        _nonempty_text(self.runtime_identity, "runtime_identity")
        _validate_sha256(self.evidence_policy_sha256, "evidence_policy_sha256")
        pass_ids = _validate_text_tuple(self.pass_ids, "pass_ids")
        if pass_ids != PASS_IDS:
            raise ValueError("pass_ids must contain the exact ten submission passes")
        object.__setattr__(self, "ordered_video_ids", video_ids)
        object.__setattr__(self, "pass_ids", pass_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "run_id": self.run_id,
            "ordered_video_ids": list(self.ordered_video_ids),
            "panel_sha256": self.panel_sha256,
            "model_id": self.model_id,
            "runtime_identity": self.runtime_identity,
            "evidence_policy_sha256": self.evidence_policy_sha256,
            "pass_ids": list(self.pass_ids),
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class CandidateRecord:
    video_id: str
    pass_id: str
    answer: str
    elapsed_seconds: float
    confidence: float
    support_span_ids: tuple[str, ...]
    request_sha256: str
    runtime_provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        _nonempty_text(self.video_id, "video_id")
        if self.pass_id not in PASS_IDS:
            raise ValueError("pass_id must be one of the exact ten submission passes")
        _nonempty_text(self.answer, "answer")
        if isinstance(self.elapsed_seconds, bool) or not isinstance(
            self.elapsed_seconds, (int, float)
        ):
            raise TypeError("elapsed_seconds must be numeric")
        elapsed = _finite_number(
            self.elapsed_seconds, "elapsed_seconds", nonnegative=True
        )
        confidence = _finite_number(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        support_span_ids = _validate_text_tuple(
            self.support_span_ids, "support_span_ids"
        )
        if len(set(support_span_ids)) != len(support_span_ids):
            raise ValueError("support_span_ids must be unique")
        _validate_sha256(self.request_sha256, "request_sha256")
        if not isinstance(self.runtime_provenance, Mapping):
            raise TypeError("runtime_provenance must be a mapping")
        frozen_provenance = _validate_runtime_provenance(self.runtime_provenance)
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "support_span_ids", support_span_ids)
        object.__setattr__(self, "runtime_provenance", frozen_provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "pass_id": self.pass_id,
            "answer": self.answer,
            "elapsed_seconds": self.elapsed_seconds,
            "confidence": self.confidence,
            "support_span_ids": list(self.support_span_ids),
            "request_sha256": self.request_sha256,
            "runtime_provenance": _thaw(self.runtime_provenance),
        }


@dataclass(frozen=True)
class VerifierDecision:
    ranked_pass_ids: tuple[str, ...]
    support_by_pass: Mapping[str, str]

    def __post_init__(self) -> None:
        ranked_pass_ids = _validate_text_tuple(
            self.ranked_pass_ids, "ranked_pass_ids"
        )
        if not isinstance(self.support_by_pass, Mapping):
            raise TypeError("support_by_pass must be a mapping")
        support_by_pass = dict(self.support_by_pass)
        for pass_id, support in support_by_pass.items():
            if not isinstance(pass_id, str) or not isinstance(support, str):
                raise TypeError("support_by_pass must map pass IDs to strings")
        _validate_verifier_contract(ranked_pass_ids, support_by_pass)
        object.__setattr__(self, "ranked_pass_ids", ranked_pass_ids)
        object.__setattr__(
            self, "support_by_pass", MappingProxyType(copy.deepcopy(support_by_pass))
        )

    def to_dict(self) -> dict[str, Any]:
        _validate_verifier_contract(self.ranked_pass_ids, self.support_by_pass)
        return {
            "ranked_pass_ids": list(self.ranked_pass_ids),
            "support_by_pass": dict(self.support_by_pass),
        }


@dataclass(frozen=True)
class RankedPrediction:
    query_id: int
    video_id: str
    candidates: tuple[CandidateRecord, ...]
    ranked_pass_ids: tuple[str, ...]
    ranking_source: str
    fallback_status: str | None

    def __post_init__(self) -> None:
        if isinstance(self.query_id, bool) or not isinstance(self.query_id, int):
            raise TypeError("query_id must be an integer")
        if self.query_id < 0:
            raise ValueError("query_id cannot be negative")
        _nonempty_text(self.video_id, "video_id")
        candidates = _as_tuple(self.candidates, "candidates")
        if len(candidates) != len(PASS_IDS):
            raise ValueError("RankedPrediction requires exactly ten candidates")
        if not all(isinstance(candidate, CandidateRecord) for candidate in candidates):
            raise TypeError("candidates must contain CandidateRecord values")
        if any(candidate.video_id != self.video_id for candidate in candidates):
            raise ValueError("candidate video_id does not match prediction video_id")
        candidate_pass_ids = tuple(candidate.pass_id for candidate in candidates)
        if len(set(candidate_pass_ids)) != len(candidate_pass_ids):
            raise ValueError("candidates must contain unique pass IDs")
        ranked_pass_ids = _validate_text_tuple(
            self.ranked_pass_ids, "ranked_pass_ids"
        )
        if len(ranked_pass_ids) != len(PASS_IDS) or set(ranked_pass_ids) != set(
            candidate_pass_ids
        ):
            raise ValueError("ranked_pass_ids must be a complete permutation")
        _nonempty_text(self.ranking_source, "ranking_source")
        if self.fallback_status is not None:
            _nonempty_text(self.fallback_status, "fallback_status")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "ranked_pass_ids", ranked_pass_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "video_id": self.video_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "ranked_pass_ids": list(self.ranked_pass_ids),
            "ranking_source": self.ranking_source,
            "fallback_status": self.fallback_status,
        }


def _normalized_answer(answer: str) -> str:
    normalized = answer.strip().casefold()
    while normalized:
        last = normalized[-1]
        if not unicodedata.category(last).startswith("P"):
            break
        normalized = normalized[:-1].rstrip()
    return normalized


def deduplicate_candidates(
    candidates: Sequence[CandidateRecord],
) -> tuple[CandidateRecord, ...]:
    """Keep the first candidate for each video and normalized answer."""

    seen: set[tuple[str, str]] = set()
    result: list[CandidateRecord] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, CandidateRecord):
            raise TypeError(f"candidate {index} must be a CandidateRecord")
        key = (candidate.video_id, _normalized_answer(candidate.answer))
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return tuple(result)


def _candidate_by_pass(
    candidates: Sequence[CandidateRecord],
) -> dict[str, CandidateRecord]:
    if not candidates:
        raise ValueError("candidates cannot be empty")
    by_pass: dict[str, CandidateRecord] = {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, CandidateRecord):
            raise TypeError(f"candidate {index} must be a CandidateRecord")
        if candidate.pass_id in by_pass:
            raise ValueError("candidates must contain unique pass IDs")
        by_pass[candidate.pass_id] = candidate
    normalized_answers = {
        _normalized_answer(candidate.answer) for candidate in by_pass.values()
    }
    if len(normalized_answers) != len(by_pass):
        raise ValueError("candidates contain duplicate normalized answers")
    return by_pass


def apply_verifier_ranking(
    candidates: Sequence[CandidateRecord],
    decision: VerifierDecision | None,
) -> tuple[CandidateRecord, ...]:
    """Apply a complete reference-blind ranking or deterministic fallback."""

    by_pass = _candidate_by_pass(candidates)
    if decision is None:
        return tuple(
            by_pass[pass_id]
            for pass_id in FALLBACK_PASS_PRECEDENCE
            if pass_id in by_pass
        )
    if not isinstance(decision, VerifierDecision):
        raise TypeError("decision must be a VerifierDecision or None")
    ranked = decision.ranked_pass_ids
    if len(ranked) != len(by_pass) or set(ranked) != set(by_pass):
        raise ValueError("ranked_pass_ids must be a complete permutation")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked_pass_ids must be a complete permutation")
    if not set(decision.support_by_pass).issubset(set(by_pass)):
        raise ValueError("support_by_pass contains an unknown candidate")
    return tuple(by_pass[pass_id] for pass_id in ranked)


def freeze_predictions(
    path: str | Path,
    *,
    bindings: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze prediction bytes and optionally write a freeze sidecar once."""

    freeze = freeze_json_artifact(path, bindings=dict(bindings))
    if output_path is not None:
        sidecar = Path(output_path)
        if sidecar.exists():
            raise ValueError(f"refusing to overwrite existing freeze output: {sidecar}")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sidecar.open("x", encoding="utf-8") as handle:
                json.dump(
                    freeze,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
        except FileExistsError as exc:
            raise ValueError(
                f"refusing to overwrite existing freeze output: {sidecar}"
            ) from exc
    return freeze


def verify_prediction_freeze(
    path: str | Path,
    freeze_record: Mapping[str, Any],
) -> None:
    """Verify prediction bytes against a previously returned freeze record."""

    verify_json_artifact(path, dict(freeze_record))


def serialize_ag_submission(
    predictions: Sequence[RankedPrediction],
    *,
    cutoff: int,
) -> list[dict[str, Any]]:
    """Serialize a frozen ranking prefix using the official AG JSON schema."""

    if isinstance(cutoff, bool) or cutoff not in {1, 5, 10}:
        raise ValueError("cutoff must be one of {1, 5, 10}")
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise TypeError("predictions must be a sequence")

    result: list[dict[str, Any]] = []
    seen_query_ids: set[int] = set()
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, RankedPrediction):
            raise TypeError(f"prediction {index} must be a RankedPrediction")
        if prediction.query_id in seen_query_ids:
            raise ValueError("query_id values must be unique")
        seen_query_ids.add(prediction.query_id)

        candidates_by_pass = {
            candidate.pass_id: candidate for candidate in prediction.candidates
        }
        normalized_answers: set[str] = set()
        for candidate in prediction.candidates:
            if (
                not math.isfinite(candidate.elapsed_seconds)
                or candidate.elapsed_seconds < 0
            ):
                raise ValueError("candidate time must be finite and non-negative")
            normalized = _normalized_answer(candidate.answer)
            if normalized in normalized_answers:
                raise ValueError("duplicate normalized answers are not allowed")
            normalized_answers.add(normalized)

        answers: list[dict[str, Any]] = []
        for rank, pass_id in enumerate(prediction.ranked_pass_ids[:cutoff], start=1):
            candidate = candidates_by_pass.get(pass_id)
            if candidate is None:
                raise ValueError("ranked_pass_ids must be a complete permutation")
            answers.append(
                {
                    "Rank": rank,
                    "Answer": candidate.answer.strip(),
                    "Time": candidate.elapsed_seconds,
                }
            )
        if len(answers) != cutoff or [answer["Rank"] for answer in answers] != list(
            range(1, cutoff + 1)
        ):
            raise ValueError("ranks must be contiguous starting at one")
        result.append(
            {
                "Q_ID": prediction.query_id,
                "Video_ID": prediction.video_id,
                "Answers": answers,
            }
        )
    return result


__all__ = [
    "FALLBACK_PASS_PRECEDENCE",
    "MODEL_ID",
    "PASS_IDS",
    "SCHEMA_VERSION",
    "CandidateRecord",
    "RankedPrediction",
    "RunManifest",
    "TimelineSpan",
    "VerifierDecision",
    "apply_verifier_ranking",
    "audit_label_free_payload",
    "canonical_sha256",
    "deduplicate_candidates",
    "freeze_predictions",
    "serialize_ag_submission",
    "verify_prediction_freeze",
]
