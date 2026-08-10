"""Immutable host-side contracts for the label-blind VQA submission path."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
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
SUPPORT_STATUSES = ("supported", "contradicted", "uncertain")
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
_FILE_HASH_CHUNK_BYTES = 1024 * 1024
_CANDIDATE_LEAKAGE_PATTERNS = (
    re.compile(r"\breference\s+answer\b", re.IGNORECASE),
    re.compile(r"\bgold\s+answer\b", re.IGNORECASE),
    re.compile(r"\bmetric\s*=", re.IGNORECASE),
    re.compile(r"\bscore\s*=", re.IGNORECASE),
    re.compile(
        r"(?:^|[\{\[\(,])\s*[\"']?"
        r"(?:annotation|label|reference|gold|metric|score)"
        r"[\"']?\s*[:=]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:annotation|label)[_\s-]*(?:payload|record|field|data)\b",
        re.IGNORECASE,
    ),
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


def _require_regular_file(path: Path, field: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{field} must be an existing regular file: {path}")
    return path


def _sha256_file(path: Path, field: str) -> str:
    path = _require_regular_file(path, field)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_FILE_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"unable to read {field}: {path}") from exc
    return digest.hexdigest()


def _verify_file_hash(path: Path, declared: Any, field: str) -> str:
    actual = _sha256_file(path, field)
    if declared is not None:
        _validate_sha256(declared, field)
        if declared != actual:
            raise ValueError(f"{field} does not match current file bytes: {path}")
    return actual


def _validate_candidate_text(text: Any) -> str:
    _nonempty_text(text, "candidate text")
    for pattern in _CANDIDATE_LEAKAGE_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                "candidate text contains forbidden label or annotation marker"
            )
    return text


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
            raise ValueError(f"runtime_provenance contains unsupported field: {key}")
        if not isinstance(raw_value, str):
            raise TypeError(f"runtime_provenance[{key}] must be a string")
        if key in {"request_sha256", "response_sha256"}:
            validated[key] = _validate_sha256(raw_value, f"runtime_provenance[{key}]")
        elif key == "model_id":
            if raw_value != MODEL_ID:
                raise ValueError(f"runtime_provenance[{key}] must be {MODEL_ID}")
            validated[key] = raw_value
        elif key == "status":
            if raw_value not in _SAFE_PROVENANCE_STATUSES:
                raise ValueError(f"runtime_provenance[{key}] has an unsupported status")
            validated[key] = raw_value
        else:
            pattern = (
                _SAFE_PROVENANCE_DEVICE if key == "device" else _SAFE_PROVENANCE_TOKEN
            )
            if pattern.fullmatch(raw_value) is None:
                raise ValueError(
                    f"runtime_provenance[{key}] must be a safe scalar token"
                )
            validated[key] = raw_value
    return MappingProxyType(validated)


def _validate_verifier_contract(
    ranked_pass_ids: Sequence[str],
    support_by_pass: Mapping[str, str],
    selected_pass_id: str | None,
    status_by_pass: Mapping[str, str] | None,
) -> None:
    if (
        len(ranked_pass_ids) != len(PASS_IDS)
        or len(set(ranked_pass_ids)) != len(ranked_pass_ids)
        or set(ranked_pass_ids) != set(PASS_IDS)
    ):
        raise ValueError("ranked_pass_ids must be a complete permutation")
    if selected_pass_id != ranked_pass_ids[0]:
        raise ValueError("selected_pass_id must equal the ranked first pass")
    if status_by_pass is None or set(status_by_pass) != set(PASS_IDS):
        raise ValueError("status_by_pass must cover every pass exactly")
    if any(status not in SUPPORT_STATUSES for status in status_by_pass.values()):
        raise ValueError("status_by_pass contains an unsupported status")
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
    return normalized in {
        "path",
        "paths",
        "file",
        "files",
        "directory",
        "url",
    } or normalized.endswith(("_path", "_paths", "_file", "_files", "_url"))


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
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != SCHEMA_VERSION
        ):
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
    selected_pass_id: str | None = None
    status_by_pass: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        ranked_pass_ids = _validate_text_tuple(self.ranked_pass_ids, "ranked_pass_ids")
        if not isinstance(self.support_by_pass, Mapping):
            raise TypeError("support_by_pass must be a mapping")
        support_by_pass = dict(self.support_by_pass)
        for pass_id, support in support_by_pass.items():
            if not isinstance(pass_id, str) or not isinstance(support, str):
                raise TypeError("support_by_pass must map pass IDs to strings")
        selected_pass_id = self.selected_pass_id
        status_by_pass = self.status_by_pass
        if status_by_pass is not None:
            status_by_pass = dict(status_by_pass)
            if any(
                not isinstance(pass_id, str) or not isinstance(status, str)
                for pass_id, status in status_by_pass.items()
            ):
                raise TypeError("status_by_pass must map pass IDs to statuses")
        _validate_verifier_contract(
            ranked_pass_ids,
            support_by_pass,
            selected_pass_id,
            status_by_pass,
        )
        object.__setattr__(self, "ranked_pass_ids", ranked_pass_ids)
        object.__setattr__(
            self, "support_by_pass", MappingProxyType(copy.deepcopy(support_by_pass))
        )
        object.__setattr__(self, "selected_pass_id", selected_pass_id)
        object.__setattr__(
            self,
            "status_by_pass",
            MappingProxyType(copy.deepcopy(status_by_pass)),
        )

    @property
    def support_status_by_pass(self) -> Mapping[str, str]:
        if self.status_by_pass is None:
            raise ValueError("status_by_pass is required")
        return self.status_by_pass

    def to_dict(self) -> dict[str, Any]:
        _validate_verifier_contract(
            self.ranked_pass_ids,
            self.support_by_pass,
            self.selected_pass_id,
            self.status_by_pass,
        )
        return {
            "ranked_pass_ids": list(self.ranked_pass_ids),
            "support_by_pass": dict(self.support_by_pass),
            "selected_pass_id": self.selected_pass_id,
            "status_by_pass": dict(self.support_status_by_pass),
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
        ranked_pass_ids = _validate_text_tuple(self.ranked_pass_ids, "ranked_pass_ids")
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


_QUESTION_ROUTE_PATTERNS = (
    (
        "ocr-detail",
        (
            r"\b(?:text|writing|written|letter|word|read|transcribe)\b",
            r"\b(?:sign|poster|banner|screen|label)\s+(?:say|show|read)",
            r"\bwhat\s+is\s+on\b",
            r"\bcolou?r\b",
        ),
    ),
    (
        "speech",
        (
            r"\b(?:say|said|saying|speak|spoken|speech|conversation)\b",
            r"\bwhat\s+is\s+being\s+said\b",
        ),
    ),
    (
        "count",
        (
            r"^how\s+many\b",
            r"^how\s+much\b",
            r"\b(?:number|count)\s+of\b",
        ),
    ),
    (
        "temporal",
        (
            r"\b(?:before|after|next|first|last|then|when|while|during)\b",
            r"\b(?:happens?|happened|occurs?)\b",
            r"\bby\s+the\s+end\b",
        ),
    ),
    (
        "causal",
        (
            r"^why\b",
            r"\bwhat\s+(?:causes?|caused|leads?)\b",
            r"^how\s+can\b",
            r"\bhow\s+does\b.*\bwork\b",
        ),
    ),
)
_QUESTION_ROUTES = frozenset(
    {route for route, _patterns in _QUESTION_ROUTE_PATTERNS} | {"other"}
)
_BACKFILL_ATTEMPTS = frozenset({1, 2})


def route_question(question: str) -> str:
    """Route a question using only deterministic text rules."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("question cannot be empty")
    for route, patterns in _QUESTION_ROUTE_PATTERNS:
        if any(
            re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns
        ):
            return route
    return "other"


def _row_value(row: Any, *names: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                return row[name]
        return default
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    return default


def _as_optional_path(value: Any, field: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field} must be a path or None")
    path = Path(value)
    if not str(path):
        raise ValueError(f"{field} cannot be empty")
    return path


def _timeline_model_payload(timeline: Any) -> dict[str, Any]:
    if not all(
        hasattr(timeline, field)
        for field in ("duration_seconds", "events", "support_spans")
    ):
        raise TypeError("timeline must be a ValidatedTimeline")
    payload = {
        "duration_seconds": timeline.duration_seconds,
        "events": [event.to_dict() for event in timeline.events],
        "support_spans": [span.to_dict() for span in timeline.support_spans],
    }
    if getattr(timeline, "fallback_reason", None) is not None:
        payload["fallback_reason"] = timeline.fallback_reason
    audit_label_free_payload(payload)
    return payload


def _frame_path(frame: Any) -> Path:
    path = getattr(frame, "frame_path", None)
    if path is None:
        raise ValueError("materialized evidence frame is missing frame_path")
    return Path(path)


def _frame_provenance(frames: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "frame_path": str(_frame_path(frame)),
            "requested_time_seconds": frame.requested_time_seconds,
            "actual_pts_seconds": frame.actual_pts_seconds,
            "stream_index": frame.stream_index,
            "frame_index": frame.frame_index,
            "time_base": frame.time_base,
            "source_media_sha256": frame.source_media_sha256,
            "image_sha256": frame.image_sha256,
            "support_span_id": frame.support_span_id,
        }
        for frame in frames
    )


def _source_span_ids(evidence_plan: Any) -> tuple[str, ...]:
    return tuple(span.span_id for span in evidence_plan.support_spans)


def _frame_values(frames: Sequence[Any]) -> tuple[tuple[Path, ...], tuple[float, ...]]:
    return (
        tuple(_frame_path(frame) for frame in frames),
        tuple(frame.requested_time_seconds for frame in frames),
    )


def _transcript_for_scope(
    full_transcript: str | None,
    overlap_transcript: str | None,
    *,
    include_overlap: bool,
) -> str | None:
    if full_transcript is None:
        if include_overlap and overlap_transcript is not None:
            return "Overlapping support-span transcript:\n" + overlap_transcript
        return None
    if (
        include_overlap
        and overlap_transcript is not None
        and overlap_transcript != full_transcript
    ):
        return (
            "Full timestamped transcript:\n"
            + full_transcript
            + "\n\nOverlapping support-span transcript:\n"
            + overlap_transcript
        )
    return full_transcript


def _context(row: Any, timeline: Any, evidence_plan: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        audit_label_free_payload(row)
    if not all(
        hasattr(evidence_plan, field)
        for field in ("duration_seconds", "global_frames", "local_frames", "all_frames")
    ):
        raise TypeError("evidence_plan must be an EvidencePlan")
    timeline_payload = _timeline_model_payload(timeline)
    if not math.isclose(
        float(timeline.duration_seconds),
        float(evidence_plan.duration_seconds),
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise ValueError("timeline and evidence plan durations must match")
    video_id = _row_value(row, "video_id", "Video_ID")
    question = _row_value(row, "question", "Question")
    run_id = _row_value(row, "run_id", "Run_ID", default="vqa-submission")
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("row video_id must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("row question must be a non-empty string")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("row run_id must be a non-empty string")
    media_path = _as_optional_path(
        _row_value(row, "media_path", "media", "video_path"), "media_path"
    )
    audio_path = _as_optional_path(_row_value(row, "audio_path", "audio"), "audio_path")
    overlap_transcript = (
        evidence_plan.transcript.text.strip()
        if evidence_plan.transcript and evidence_plan.transcript.text.strip()
        else None
    )
    full_transcript = _row_value(
        row,
        "full_transcript",
        "timestamped_whisper",
        "whisper_text",
    )
    if full_transcript is not None:
        _nonempty_text(full_transcript, "full_transcript")
        full_transcript = full_transcript.strip()
    frames = tuple(evidence_plan.all_frames)
    row_source_sha256 = _row_value(
        row, "media_sha256", "source_media_sha256"
    )
    if row_source_sha256 is not None:
        _validate_sha256(row_source_sha256, "media_sha256")
    row_audio_sha256 = _row_value(row, "audio_sha256", "source_audio_sha256")
    if row_audio_sha256 is not None:
        _validate_sha256(row_audio_sha256, "audio_sha256")
    return {
        "run_id": run_id,
        "video_id": video_id,
        "question": " ".join(question.split()),
        "media_path": media_path,
        "audio_path": audio_path,
        "question_route": route_question(question),
        "timeline_payload": timeline_payload,
        "full_transcript": full_transcript,
        "overlap_transcript": overlap_transcript,
        "source_span_ids": _source_span_ids(evidence_plan),
        "all_frames": frames,
        "media_sha256": row_source_sha256,
        "audio_sha256": row_audio_sha256,
    }


def _scene_boundary_frames(all_frames: Sequence[Any], timeline: Any) -> tuple[Any, ...]:
    selected: dict[tuple[int, int], Any] = {}
    for frame in all_frames:
        if frame.slot_kind == "global":
            selected[frame.decoded_identity] = frame
    for event in timeline.events:
        before = [
            frame
            for frame in all_frames
            if frame.actual_pts_seconds <= event.start_seconds
        ]
        after = [
            frame
            for frame in all_frames
            if frame.actual_pts_seconds >= event.start_seconds
        ]
        if before:
            frame = max(
                before,
                key=lambda item: (item.actual_pts_seconds, item.decoded_identity),
            )
            selected[frame.decoded_identity] = frame
        if after:
            frame = min(
                after, key=lambda item: (item.actual_pts_seconds, item.decoded_identity)
            )
            selected[frame.decoded_identity] = frame
    return tuple(
        sorted(
            selected.values(),
            key=lambda frame: (
                frame.requested_time_seconds,
                frame.actual_pts_seconds,
                frame.stream_index,
                frame.frame_index,
            ),
        )
    )


def _candidate_text(candidate: Any) -> str:
    if isinstance(candidate, CandidateRecord):
        return _validate_candidate_text(candidate.answer)
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("candidate text must be a non-empty string")
    return _validate_candidate_text(candidate)


def _build_answer_request(
    context: Mapping[str, Any],
    *,
    pass_id: str,
    frames: Sequence[Any],
    transcript: str | None,
    timeline_payload: Mapping[str, Any] | None,
    instruction: str,
    audio_mode: str | None = None,
    include_media: bool = False,
    candidate_texts: Sequence[str] = (),
    attempt: int | None = None,
    excluded_normalized_answers: Sequence[str] = (),
) -> AnswerRequest:
    frame_paths, frame_timestamps = _frame_values(frames)
    return AnswerRequest(
        run_id=context["run_id"],
        condition_id=pass_id,
        video_id=context["video_id"],
        question=context["question"],
        pass_id=pass_id,
        frame_paths=frame_paths,
        frame_timestamps=frame_timestamps,
        transcript=transcript,
        media_path=context["media_path"] if include_media else None,
        audio_path=context["audio_path"] if audio_mode == "audio-on" else None,
        frame_descriptions=tuple(
            f"Frame at {frame.requested_time_seconds:.3f} seconds "
            f"(decoded PTS {frame.actual_pts_seconds:.3f}):"
            for frame in frames
        ),
        audio_mode=audio_mode,
        source_span_ids=context["source_span_ids"],
        question_route=context["question_route"],
        evidence_policy=instruction,
        timeline_payload=timeline_payload,
        frame_provenance=_frame_provenance(frames),
        media_bytes_sha256=context["media_sha256"],
        audio_bytes_sha256=context["audio_sha256"],
        candidate_texts=tuple(candidate_texts),
        attempt=attempt,
        excluded_normalized_answers=tuple(excluded_normalized_answers),
    )


_PASS_INSTRUCTIONS = {
    "direct-av": "Use the complete native video and audio. Return one concise supported response.",
    "direct-av-alternate": "Use the complete native video and audio. Reconsider ambiguity and return one distinct supported response.",
    "global-transcript": "Use the chronological global frames and timestamped transcript to answer concisely.",
    "global-local-timeline": "Use the complete global and local evidence, transcript, and frozen timeline. Answer concisely.",
    "question-specialist": "Use the union evidence and apply the deterministic question specialist route before answering concisely.",
    "local-only": "Use only dense support-span frames and overlapping transcript text. Avoid unsupported specificity.",
    "audio-first": "Prioritize relevant speech. Distinguish the full timestamped transcript from the overlapping support-span transcript and check global visual context.",
    "scene-boundary": "Compare frames around detected event boundaries with global context and answer concisely.",
    "detail-temporal-route": "Use the deterministic route-specific evidence view and answer only what it supports.",
    "conservative-synthesis": "Synthesize only from the supplied candidate texts and union evidence; add no unsupported detail.",
}


@dataclass(frozen=True)
class AnswerRequest:
    """One deterministic, label-free model input for an answer pass."""

    run_id: str
    condition_id: str
    video_id: str
    question: str
    pass_id: str
    frame_paths: tuple[Path, ...]
    frame_timestamps: tuple[float, ...]
    transcript: str | None
    media_path: Path | None
    audio_path: Path | None = None
    frame_descriptions: tuple[str, ...] | None = None
    audio_mode: str | None = None
    source_span_ids: tuple[str, ...] = ()
    question_route: str = "other"
    evidence_policy: str = ""
    timeline_payload: Mapping[str, Any] | None = None
    frame_provenance: tuple[Mapping[str, Any], ...] = ()
    media_bytes_sha256: str | None = None
    audio_bytes_sha256: str | None = None
    candidate_texts: tuple[str, ...] = ()
    candidate_pass_ids: tuple[str, ...] = ()
    attempt: int | None = None
    excluded_normalized_answers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("run_id", "condition_id", "video_id", "question", "pass_id"):
            _nonempty_text(getattr(self, field_name), field_name)
        if self.pass_id not in PASS_IDS and self.pass_id not in {
            "backfill-1",
            "backfill-2",
        }:
            raise ValueError(
                "pass_id must be one of the ten passes or a bounded backfill"
            )
        paths = tuple(Path(path) for path in self.frame_paths)
        timestamps = tuple(
            _finite_number(value, f"frame_timestamps[{index}]", nonnegative=True)
            for index, value in enumerate(self.frame_timestamps)
        )
        if len(paths) != len(timestamps):
            raise ValueError("Frame paths and timestamps must have equal length")
        if self.frame_descriptions is not None:
            descriptions = _validate_text_tuple(
                self.frame_descriptions, "frame_descriptions"
            )
            if len(descriptions) != len(paths):
                raise ValueError("Frame descriptions and paths must have equal length")
            object.__setattr__(self, "frame_descriptions", descriptions)
        if self.transcript is not None:
            _nonempty_text(self.transcript, "transcript")
        if self.audio_mode not in {None, "audio-off", "audio-on"}:
            raise ValueError("audio_mode must be audio-off or audio-on")
        if self.question_route not in _QUESTION_ROUTES:
            raise ValueError("question_route is unsupported")
        source_span_ids = _validate_text_tuple(self.source_span_ids, "source_span_ids")
        if len(set(source_span_ids)) != len(source_span_ids):
            raise ValueError("source_span_ids must be unique")
        candidate_texts = tuple(_candidate_text(text) for text in self.candidate_texts)
        if len(candidate_texts) > 9:
            raise ValueError(
                "conservative-synthesis accepts at most nine candidate texts"
            )
        if self.candidate_pass_ids:
            raise ValueError("model requests cannot contain private candidate pass IDs")
        if self.attempt is not None and self.attempt not in _BACKFILL_ATTEMPTS:
            raise ValueError("backfill attempt must be exactly 1 or 2")
        excluded = _validate_text_tuple(
            self.excluded_normalized_answers, "excluded_normalized_answers"
        )
        if tuple(sorted(set(excluded))) != excluded:
            raise ValueError("excluded_normalized_answers must be sorted and unique")
        if self.media_bytes_sha256 is not None:
            _validate_sha256(self.media_bytes_sha256, "media_bytes_sha256")
        if self.audio_bytes_sha256 is not None:
            _validate_sha256(self.audio_bytes_sha256, "audio_bytes_sha256")
        if self.timeline_payload is not None:
            audit_label_free_payload(self.timeline_payload)
        provenance = tuple(_freeze(item) for item in self.frame_provenance)
        for item in provenance:
            if not isinstance(item, Mapping):
                raise TypeError("frame_provenance must contain mappings")
        object.__setattr__(self, "frame_paths", paths)
        object.__setattr__(self, "frame_timestamps", timestamps)
        object.__setattr__(self, "source_span_ids", source_span_ids)
        object.__setattr__(self, "candidate_texts", candidate_texts)
        object.__setattr__(self, "excluded_normalized_answers", excluded)
        object.__setattr__(self, "frame_provenance", provenance)
        if self.media_path is not None:
            object.__setattr__(self, "media_path", Path(self.media_path))
        if self.audio_path is not None:
            object.__setattr__(self, "audio_path", Path(self.audio_path))
        if self.pass_id in {"direct-av", "direct-av-alternate"}:
            if self.media_path is None:
                raise ValueError("direct AV passes require native media")
        if self.pass_id == "audio-first" and self.audio_path is None:
            raise ValueError("audio-first requires a separate audio evidence file")
        if len(provenance) != len(paths):
            raise ValueError("frame_provenance and frame_paths must have equal length")
        self._model_visible_file_records()

    def to_model_payload(self) -> dict[str, Any]:
        self._model_visible_file_records()
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "video_id": self.video_id,
            "question": self.question,
            "frame_paths": [str(path.resolve()) for path in self.frame_paths],
            "frame_timestamps": list(self.frame_timestamps),
            "frame_descriptions": (
                list(self.frame_descriptions)
                if self.frame_descriptions is not None
                else None
            ),
            "transcript": self.transcript,
            "media_path": str(self.media_path.resolve()) if self.media_path else None,
            "audio_path": str(self.audio_path.resolve()) if self.audio_path else None,
            "audio_mode": self.audio_mode,
            "source_span_ids": list(self.source_span_ids),
            "question_route": self.question_route,
            "evidence_policy": self.evidence_policy,
            "instruction": self.evidence_policy,
            "frame_provenance": _thaw(self.frame_provenance),
        }
        if self.timeline_payload is not None:
            payload["timeline"] = _thaw(self.timeline_payload)
        if self.candidate_texts:
            payload["candidate_texts"] = list(self.candidate_texts)
        if self.excluded_normalized_answers:
            payload["excluded_normalized_texts"] = list(
                self.excluded_normalized_answers
            )
        audit_label_free_payload(payload)
        return payload

    @property
    def request_sha256(self) -> str:
        self._model_visible_file_records()
        return canonical_sha256(self.to_model_payload())

    @property
    def model_visible_bytes_sha256(self) -> str:
        return canonical_sha256(
            {
                "payload": self.to_model_payload(),
                "files": self._model_visible_file_records(),
            }
        )

    def _model_visible_file_records(self) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        if self.media_path is not None:
            records.append(
                {
                    "role": "native-media",
                    "path": str(self.media_path.resolve()),
                    "sha256": _verify_file_hash(
                        self.media_path,
                        self.media_bytes_sha256,
                        "media_sha256",
                    ),
                }
            )
        if self.audio_path is not None:
            records.append(
                {
                    "role": "audio",
                    "path": str(self.audio_path.resolve()),
                    "sha256": _verify_file_hash(
                        self.audio_path,
                        self.audio_bytes_sha256,
                        "audio_sha256",
                    ),
                }
            )
        for index, (path, provenance) in enumerate(
            zip(self.frame_paths, self.frame_provenance, strict=True)
        ):
            declared_path = provenance.get("frame_path")
            if declared_path is not None and Path(declared_path).resolve() != path.resolve():
                raise ValueError("frame provenance path does not match frame path")
            records.append(
                {
                    "role": f"frame-{index}",
                    "path": str(path.resolve()),
                    "sha256": _verify_file_hash(
                        path,
                        provenance.get("image_sha256"),
                        "image_sha256",
                    ),
                }
            )
        return records

    @property
    def model_visible_byte_sha256(self) -> str:
        return self.model_visible_bytes_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "video_id": self.video_id,
            "pass_id": self.pass_id,
            "request_sha256": self.request_sha256,
            "model_visible_bytes_sha256": self.model_visible_bytes_sha256,
            "source_span_ids": list(self.source_span_ids),
            "attempt": self.attempt,
            "model_payload": self.to_model_payload(),
        }


def build_answer_requests(
    row: Any,
    timeline: Any,
    evidence_plan: Any,
    *,
    candidates: Sequence[Any] = (),
) -> dict[str, AnswerRequest]:
    """Build the exact ten deterministic label-free answer views."""

    context = _context(row, timeline, evidence_plan)
    global_frames = tuple(evidence_plan.global_frames)
    local_frames = tuple(evidence_plan.local_frames)
    all_frames = tuple(evidence_plan.all_frames)
    route = context["question_route"]
    route_frames = all_frames
    full_transcript = context["full_transcript"]
    overlap_transcript = context["overlap_transcript"]
    union_transcript = _transcript_for_scope(
        full_transcript, overlap_transcript, include_overlap=True
    )
    boundary_frames = _scene_boundary_frames(all_frames, timeline)
    views: dict[
        str, tuple[Sequence[Any], str | None, Mapping[str, Any] | None, str, str | None]
    ] = {
        "direct-av": ((), None, None, _PASS_INSTRUCTIONS["direct-av"], "audio-on"),
        "direct-av-alternate": (
            (),
            None,
            None,
            _PASS_INSTRUCTIONS["direct-av-alternate"],
            "audio-on",
        ),
        "global-transcript": (
            global_frames,
            full_transcript,
            None,
            _PASS_INSTRUCTIONS["global-transcript"],
            None,
        ),
        "global-local-timeline": (
            all_frames,
            full_transcript,
            context["timeline_payload"],
            _PASS_INSTRUCTIONS["global-local-timeline"],
            None,
        ),
        "question-specialist": (
            all_frames,
            full_transcript,
            context["timeline_payload"],
            f"{_PASS_INSTRUCTIONS['question-specialist']} Route: {route}.",
            None,
        ),
        "local-only": (
            local_frames,
            overlap_transcript,
            None,
            _PASS_INSTRUCTIONS["local-only"],
            None,
        ),
        "audio-first": (
            global_frames,
            union_transcript,
            None,
            _PASS_INSTRUCTIONS["audio-first"],
            "audio-on",
        ),
        "scene-boundary": (
            boundary_frames,
            full_transcript,
            None,
            _PASS_INSTRUCTIONS["scene-boundary"],
            None,
        ),
        "detail-temporal-route": (
            route_frames,
            full_transcript,
            context["timeline_payload"],
            f"{_PASS_INSTRUCTIONS['detail-temporal-route']} Route: {route}.",
            None,
        ),
        "conservative-synthesis": (
            all_frames,
            union_transcript,
            context["timeline_payload"],
            _PASS_INSTRUCTIONS["conservative-synthesis"],
            None,
        ),
    }
    validated_candidate_texts = tuple(_candidate_text(candidate) for candidate in candidates)
    candidate_texts = validated_candidate_texts[:9]
    requests: dict[str, AnswerRequest] = {}
    for pass_id in PASS_IDS:
        frames, transcript, timeline_payload, instruction, audio_mode = views[pass_id]
        requests[pass_id] = _build_answer_request(
            context,
            pass_id=pass_id,
            frames=frames,
            transcript=transcript,
            timeline_payload=timeline_payload,
            instruction=instruction,
            audio_mode=audio_mode,
            include_media=pass_id in {"direct-av", "direct-av-alternate"},
            candidate_texts=candidate_texts
            if pass_id == "conservative-synthesis"
            else (),
        )
    return requests


@dataclass(frozen=True)
class VerifierCandidate:
    """Candidate shown to the verifier without its private producer identity."""

    public_id: str
    candidate_text: str
    pass_id: str
    support_span_ids: tuple[str, ...]
    confidence: float
    source_request_sha256: str

    def __post_init__(self) -> None:
        _nonempty_text(self.public_id, "public_id")
        if self.pass_id not in PASS_IDS:
            raise ValueError("verifier candidate pass_id is unknown")
        _validate_candidate_text(self.candidate_text)
        support_span_ids = _validate_text_tuple(
            self.support_span_ids, "support_span_ids"
        )
        if len(set(support_span_ids)) != len(support_span_ids):
            raise ValueError("support_span_ids must be unique")
        _finite_number(self.confidence, "confidence")
        _validate_sha256(self.source_request_sha256, "source_request_sha256")
        object.__setattr__(self, "support_span_ids", support_span_ids)

    @property
    def text(self) -> str:
        return self.candidate_text

    @property
    def answer(self) -> str:
        return self.candidate_text

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "public_id": self.public_id,
            "candidate_text": self.candidate_text,
            "support_span_ids": list(self.support_span_ids),
        }


def public_candidate_id(run_id: str, video_id: str, pass_id: str, seed: int) -> str:
    """Return a deterministic opaque public ID for one private pass."""

    if not all(
        isinstance(value, str) and value for value in (run_id, video_id, pass_id)
    ):
        raise ValueError("candidate ID inputs must be non-empty strings")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "run_id": run_id,
                "video_id": video_id,
                "pass_id": pass_id,
                "seed": seed,
            }
        )
    ).hexdigest()
    return f"candidate-{digest[:20]}"


@dataclass(frozen=True)
class VerifierRequest:
    """Reference-blind verifier input with a seeded opaque candidate order."""

    run_id: str
    video_id: str
    question: str
    frame_paths: tuple[Path, ...]
    frame_timestamps: tuple[float, ...]
    transcript: str | None
    timeline_payload: Mapping[str, Any]
    candidates: tuple[VerifierCandidate, ...]
    seed: int
    source_span_ids: tuple[str, ...] = ()
    frame_provenance: tuple[Mapping[str, Any], ...] = ()
    media_bytes_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("run_id", "video_id", "question"):
            _nonempty_text(getattr(self, field_name), field_name)
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        candidates = tuple(self.candidates)
        if len(candidates) != len(PASS_IDS):
            raise ValueError("verifier requires exactly ten candidates")
        if {candidate.pass_id for candidate in candidates} != set(PASS_IDS):
            raise ValueError("verifier candidates must contain the exact ten passes")
        if len({candidate.public_id for candidate in candidates}) != len(candidates):
            raise ValueError("verifier public IDs must be unique")
        paths = tuple(Path(path) for path in self.frame_paths)
        timestamps = tuple(
            _finite_number(value, f"frame_timestamps[{index}]", nonnegative=True)
            for index, value in enumerate(self.frame_timestamps)
        )
        if len(paths) != len(timestamps):
            raise ValueError("Frame paths and timestamps must have equal length")
        if self.transcript is not None:
            _nonempty_text(self.transcript, "transcript")
        source_span_ids = _validate_text_tuple(self.source_span_ids, "source_span_ids")
        if len(set(source_span_ids)) != len(source_span_ids):
            raise ValueError("source_span_ids must be unique")
        audit_label_free_payload(self.timeline_payload)
        if self.media_bytes_sha256 is not None:
            _validate_sha256(self.media_bytes_sha256, "media_bytes_sha256")
        provenance = tuple(_freeze(item) for item in self.frame_provenance)
        if len(provenance) != len(paths):
            raise ValueError("frame_provenance and frame_paths must have equal length")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "frame_paths", paths)
        object.__setattr__(self, "frame_timestamps", timestamps)
        object.__setattr__(self, "source_span_ids", source_span_ids)
        object.__setattr__(self, "frame_provenance", provenance)
        self._model_visible_file_records()

    def to_model_payload(self) -> dict[str, Any]:
        self._model_visible_file_records()
        payload = {
            "run_id": self.run_id,
            "video_id": self.video_id,
            "question": self.question,
            "frame_paths": [str(path.resolve()) for path in self.frame_paths],
            "frame_timestamps": list(self.frame_timestamps),
            "transcript": self.transcript,
            "timeline": _thaw(self.timeline_payload),
            "source_span_ids": list(self.source_span_ids),
            "frame_provenance": _thaw(self.frame_provenance),
            "candidates": [candidate.to_model_dict() for candidate in self.candidates],
            "instruction": "Rank all public candidate IDs by evidence support. Do not create a new semantic response.",
        }
        audit_label_free_payload(payload)
        return payload

    @property
    def request_sha256(self) -> str:
        self._model_visible_file_records()
        return canonical_sha256(self.to_model_payload())

    @property
    def model_visible_bytes_sha256(self) -> str:
        return canonical_sha256(
            {
                "payload": self.to_model_payload(),
                "files": self._model_visible_file_records(),
            }
        )

    def _model_visible_file_records(self) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for index, (path, provenance) in enumerate(
            zip(self.frame_paths, self.frame_provenance, strict=True)
        ):
            declared_path = provenance.get("frame_path")
            if declared_path is not None and Path(declared_path).resolve() != path.resolve():
                raise ValueError("frame provenance path does not match frame path")
            records.append(
                {
                    "role": f"frame-{index}",
                    "path": str(path.resolve()),
                    "sha256": _verify_file_hash(
                        path,
                        provenance.get("image_sha256"),
                        "image_sha256",
                    ),
                }
            )
        return records

    def private_pass_id(self, public_id: str) -> str:
        for candidate in self.candidates:
            if candidate.public_id == public_id:
                return candidate.pass_id
        raise ValueError("unknown verifier public candidate ID")

    def map_public_ranking(
        self,
        ranked_public_ids: Sequence[str],
        support_by_public_id: Mapping[str, str] | None = None,
        *,
        status_by_public_id: Mapping[str, str] | None = None,
        selected_public_id: str | None = None,
    ) -> VerifierDecision:
        ranked_public_ids = tuple(ranked_public_ids)
        expected = {candidate.public_id for candidate in self.candidates}
        if (
            len(ranked_public_ids) != len(expected)
            or len(set(ranked_public_ids)) != len(ranked_public_ids)
            or set(ranked_public_ids) != expected
        ):
            raise ValueError("ranked public IDs must be a complete permutation")
        support_by_public_id = (
            {} if support_by_public_id is None else dict(support_by_public_id)
        )
        if not set(support_by_public_id).issubset(expected):
            raise ValueError("support_by_public_id contains an unknown candidate")
        if status_by_public_id is None or set(status_by_public_id) != expected:
            raise ValueError("status_by_public_id must cover every public ID exactly")
        if any(status not in SUPPORT_STATUSES for status in status_by_public_id.values()):
            raise ValueError("status_by_public_id contains an unsupported status")
        if selected_public_id is None:
            raise ValueError("selected public ID is required")
        if selected_public_id != ranked_public_ids[0]:
            raise ValueError("selected public ID must equal the ranked first candidate")
        return VerifierDecision(
            tuple(self.private_pass_id(public_id) for public_id in ranked_public_ids),
            {
                self.private_pass_id(public_id): support
                for public_id, support in support_by_public_id.items()
            },
            selected_pass_id=self.private_pass_id(selected_public_id),
            status_by_pass={
                self.private_pass_id(public_id): status
                for public_id, status in status_by_public_id.items()
            },
        )


def build_verifier_request(
    row: Any,
    timeline: Any,
    evidence_plan: Any,
    candidates: Sequence[CandidateRecord],
    *,
    seed: int,
) -> VerifierRequest:
    """Build a seeded, producer-identity-blind verifier request."""

    context = _context(row, timeline, evidence_plan)
    candidate_rows = tuple(candidates)
    if len(candidate_rows) != len(PASS_IDS):
        raise ValueError("verifier requires exactly ten candidates")
    if not all(isinstance(candidate, CandidateRecord) for candidate in candidate_rows):
        raise TypeError("verifier candidates must be CandidateRecord values")
    if {candidate.pass_id for candidate in candidate_rows} != set(PASS_IDS):
        raise ValueError("verifier candidates must contain the exact ten passes")
    candidates_by_pass = {candidate.pass_id: candidate for candidate in candidate_rows}
    ordered_candidates = tuple(candidates_by_pass[pass_id] for pass_id in PASS_IDS)
    normalized_answers = set()
    for candidate in ordered_candidates:
        _validate_candidate_text(candidate.answer)
        normalized = _normalized_answer(candidate.answer)
        if normalized in normalized_answers:
            raise ValueError("candidates contain duplicate normalized answers")
        normalized_answers.add(normalized)
    verifier_candidates = [
        VerifierCandidate(
            public_id=public_candidate_id(
                context["run_id"], context["video_id"], candidate.pass_id, seed
            ),
            candidate_text=_validate_candidate_text(candidate.answer),
            pass_id=candidate.pass_id,
            support_span_ids=candidate.support_span_ids,
            confidence=candidate.confidence,
            source_request_sha256=candidate.request_sha256,
        )
        for candidate in ordered_candidates
    ]
    random.Random(seed).shuffle(verifier_candidates)
    frame_paths, frame_timestamps = _frame_values(context["all_frames"])
    return VerifierRequest(
        run_id=context["run_id"],
        video_id=context["video_id"],
        question=context["question"],
        frame_paths=frame_paths,
        frame_timestamps=frame_timestamps,
        transcript=_transcript_for_scope(
            context["full_transcript"],
            context["overlap_transcript"],
            include_overlap=True,
        ),
        timeline_payload=context["timeline_payload"],
        candidates=tuple(verifier_candidates),
        seed=seed,
        source_span_ids=context["source_span_ids"],
        frame_provenance=_frame_provenance(context["all_frames"]),
    )


def build_backfill_request(
    row: Any,
    evidence_plan: Any,
    candidates: Sequence[Any],
    *,
    attempt: int,
) -> AnswerRequest:
    """Build one of exactly two candidate-only alternate-generation requests."""

    if isinstance(attempt, bool) or attempt not in _BACKFILL_ATTEMPTS:
        raise ValueError("backfill attempt must be exactly 1 or 2")
    context = _context(
        row,
        type(
            "TimelineForBackfill",
            (),
            {
                "duration_seconds": evidence_plan.duration_seconds,
                "events": (),
                "support_spans": evidence_plan.support_spans,
                "fallback_reason": None,
            },
        )(),
        evidence_plan,
    )
    excluded = tuple(
        sorted(
            {
                _normalized_answer(_candidate_text(candidate))
                for candidate in candidates
            }
        )
    )
    return _build_answer_request(
        context,
        pass_id=f"backfill-{attempt}",
        frames=tuple(evidence_plan.all_frames),
        transcript=_transcript_for_scope(
            context["full_transcript"],
            context["overlap_transcript"],
            include_overlap=True,
        ),
        timeline_payload=None,
        instruction=(
            "Generate one additional concise response independently supported by the union evidence. "
            "Do not repeat any excluded normalized text."
        ),
        audio_mode=None,
        attempt=attempt,
        excluded_normalized_answers=excluded,
    )


__all__ = [
    "FALLBACK_PASS_PRECEDENCE",
    "MODEL_ID",
    "PASS_IDS",
    "SCHEMA_VERSION",
    "SUPPORT_STATUSES",
    "AnswerRequest",
    "CandidateRecord",
    "RankedPrediction",
    "RunManifest",
    "TimelineSpan",
    "VerifierCandidate",
    "VerifierDecision",
    "VerifierRequest",
    "apply_verifier_ranking",
    "audit_label_free_payload",
    "build_answer_requests",
    "build_backfill_request",
    "build_verifier_request",
    "canonical_sha256",
    "deduplicate_candidates",
    "freeze_predictions",
    "public_candidate_id",
    "route_question",
    "serialize_ag_submission",
    "verify_prediction_freeze",
]
