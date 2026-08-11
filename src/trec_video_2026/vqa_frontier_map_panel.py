"""Label-free boundary helpers for the frontier video-map panel."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import StatisticsError, quantiles
from typing import Any

from trec_video_2026.vqa_benchmark import probe_media, source_fingerprint


SCHEMA_VERSION = 1
RUN_ID = "frontier-video-map-20260811"
SELECTION_POLICY = "question-only-category-purity-duration-topology-v1"
REFERENCE_POLICY = "not-read-not-serialized"
CANONICAL_DATA_ROOT = Path("/home/npatta01/data/competitions/trec_video_2026/data")
CATEGORIES = (
    "temporal",
    "counting",
    "causal",
    "speech",
    "sound",
    "ocr_detail",
    "other_global",
)
EVALUATION_PER_CATEGORY = 5
PREFLIGHT_ROWS = 2
MAX_SELECTED_DURATION_SECONDS = 30.0
SUPPORTED_MEDIA_SUFFIXES = (".mkv", ".mp4", ".webm", ".mov")

_QUESTION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("counting", (r"how many", r"number of", r"count", r"quantity")),
    ("causal", (r"why", r"because", r"reason", r"cause", r"causes")),
    (
        "temporal",
        (
            r"what\s+(?:happens?|happened|occurs?|occurred)\s+(?:before|after)",
            r"what\s+is\s+happening\s+(?:before|after)",
            r"in\s+what\s+order",
            r"order",
            r"first",
            r"then",
            r"later",
            r"earlier",
            r"during",
            r"while",
        ),
    ),
    (
        "speech",
        (
            r"say",
            r"says",
            r"said",
            r"speak",
            r"speaks",
            r"speaking",
            r"tell",
            r"tells",
            r"told",
            r"conversation",
            r"dialogue",
        ),
    ),
    (
        "sound",
        (r"sound", r"sounds", r"hear", r"hears", r"heard", r"noise", r"audio"),
    ),
    (
        "ocr_detail",
        (r"text", r"word", r"written", r"read", r"sign", r"subtitle", r"caption"),
    ),
)

_TOPOLOGY_HINTS = {
    "temporal": ("before-after", "multi-moment", "ordered-sequence"),
    "counting": ("global-count", "multi-instance", "repeated-instance"),
    "causal": ("cause-effect", "state-transition", "action-outcome"),
    "speech": ("speech-span", "speaker-turn", "speech-visual-context"),
    "sound": ("sound-span", "audio-visual-event", "sound-source-context"),
    "ocr_detail": ("text-region", "text-in-scene", "text-before-after"),
    "other_global": ("global-scene", "entity-state", "scene-summary"),
}

_FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "answer",
        "answers",
        "correctanswer",
        "expectedanswer",
        "gold",
        "goldanswer",
        "groundtruth",
        "groundtruthanswer",
        "incorrectanswer",
        "incorrectanswers",
        "label",
        "labels",
        "oracle",
        "reference",
        "references",
        "referenceanswer",
        "referenceanswers",
    }
)


@dataclass(frozen=True, slots=True)
class PanelRow:
    ordinal: int
    phase: str
    video_id: str
    question: str
    category: str
    evidence_topology: str
    selection_rationale: str
    media_relative_path: str
    media_sha256: str
    size_bytes: int
    duration_seconds: float
    has_audio: bool
    has_video: bool


@dataclass(frozen=True, slots=True)
class PanelManifest:
    schema_version: int
    run_id: str
    selection_policy: str
    reference_policy: str
    evaluation_rows: tuple[PanelRow, ...]
    preflight_rows: tuple[PanelRow, ...]

    def to_payload(self) -> dict[str, object]:
        rows = sorted(
            (*self.evaluation_rows, *self.preflight_rows),
            key=lambda row: row.ordinal,
        )
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "selection_policy": self.selection_policy,
            "reference_policy": self.reference_policy,
            "evaluation_count": len(self.evaluation_rows),
            "preflight_count": len(self.preflight_rows),
            "examples": [asdict(row) for row in rows],
        }


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _required_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _required_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _reject_label_fields(value: object, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalized_key(key) in _FORBIDDEN_LABEL_KEYS:
                raise ValueError(
                    f"label-free manifest contains forbidden field at "
                    f"{location}.{key}"
                )
            _reject_label_fields(nested, location=f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_label_fields(nested, location=f"{location}[{index}]")


def classify_question(question: str) -> str:
    """Classify a question with fixed precedence and word boundaries."""
    lowered = _required_text(question, "question").casefold()
    for category, patterns in _QUESTION_RULES:
        for pattern in patterns:
            if re.search(rf"\b(?:{pattern})\b", lowered):
                return category
    return "other_global"


def _annotations_root(data_root: Path) -> Path:
    path = (data_root / "vqa/2025/training/annotations").resolve()
    if not path.is_dir():
        raise ValueError(f"annotation directory does not exist: {path}")
    return path


def _media_root(data_root: Path) -> Path:
    path = (data_root / "vqa/2025/training/videos_original").resolve()
    if not path.is_dir():
        raise ValueError(f"media directory does not exist: {path}")
    return path


def verify_canonical_data_root(
    data_root: Path,
    *,
    expected_root: Path = CANONICAL_DATA_ROOT,
) -> Path:
    resolved = data_root.resolve(strict=True)
    expected = expected_root.resolve(strict=True)
    if resolved != expected:
        raise ValueError(f"data root must resolve to canonical dataset root: {expected}")
    return resolved


def _question_projection(annotation_path: Path) -> tuple[str, str]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"annotation must be a JSON object: {annotation_path}")
    question = _required_text(payload.get("question"), f"{annotation_path.name} question")
    return annotation_path.stem, question


def _locate_media(media_root: Path, video_id: str) -> Path:
    matches = [
        candidate
        for suffix in SUPPORTED_MEDIA_SUFFIXES
        if (candidate := media_root / f"{video_id}{suffix}").is_file()
    ]
    if not matches:
        raise ValueError(f"media file is missing for {video_id}")
    if len(matches) > 1:
        raise ValueError(f"multiple media files found for {video_id}")
    return matches[0]


def _safe_relative_path(data_root: Path, path: Path, *, kind: str) -> str:
    root = data_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{kind} path must stay beneath the dataset root")
    return str(resolved.relative_to(root))


def _candidate_row(data_root: Path, annotation_path: Path) -> dict[str, object]:
    video_id, question = _question_projection(annotation_path)
    category = classify_question(question)
    media_path = _locate_media(_media_root(data_root), video_id)
    media_identity = source_fingerprint(media_path)
    metadata = probe_media(media_path)
    return {
        "video_id": video_id,
        "question": question,
        "category": category,
        "evidence_topology_hints": list(_TOPOLOGY_HINTS[category]),
        "media_relative_path": _safe_relative_path(
            data_root, media_path, kind="Media"
        ),
        "media_sha256": media_identity.sha256,
        "size_bytes": media_identity.size_bytes,
        "duration_seconds": metadata.duration_seconds,
    }


def build_candidate_sheet(data_root: Path) -> dict[str, object]:
    candidates = [
        _candidate_row(data_root, annotation_path)
        for annotation_path in sorted(_annotations_root(data_root).glob("*.json"))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "selection_policy": SELECTION_POLICY,
        "reference_policy": REFERENCE_POLICY,
        "example_count": len(candidates),
        "candidates": candidates,
    }


def _parse_row(raw: object, *, location: str) -> PanelRow:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{location} must be an object")
    category = _required_text(raw.get("category"), f"{location}.category")
    if category not in CATEGORIES:
        raise ValueError(f"{location}.category must be one of {CATEGORIES}")
    phase = _required_text(raw.get("phase"), f"{location}.phase")
    if phase not in {"evaluation", "preflight"}:
        raise ValueError(f"{location}.phase must be evaluation or preflight")
    return PanelRow(
        ordinal=_required_int(raw.get("ordinal"), f"{location}.ordinal"),
        phase=phase,
        video_id=_required_text(raw.get("video_id"), f"{location}.video_id"),
        question=_required_text(raw.get("question"), f"{location}.question"),
        category=category,
        evidence_topology=_required_text(
            raw.get("evidence_topology"),
            f"{location}.evidence_topology",
        ),
        selection_rationale=_required_text(
            raw.get("selection_rationale"),
            f"{location}.selection_rationale",
        ),
        media_relative_path=_required_text(
            raw.get("media_relative_path"),
            f"{location}.media_relative_path",
        ),
        media_sha256=_required_text(raw.get("media_sha256"), f"{location}.media_sha256"),
        size_bytes=_required_int(raw.get("size_bytes"), f"{location}.size_bytes"),
        duration_seconds=_required_float(
            raw.get("duration_seconds"),
            f"{location}.duration_seconds",
        ),
        has_audio=_required_bool(raw.get("has_audio"), f"{location}.has_audio"),
        has_video=_required_bool(raw.get("has_video"), f"{location}.has_video"),
    )


def _rows_by_phase(rows: Sequence[PanelRow], phase: str) -> tuple[PanelRow, ...]:
    return tuple(row for row in rows if row.phase == phase)


def _validate_header(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version does not match the frontier panel schema")
    if payload.get("run_id") != RUN_ID:
        raise ValueError("run_id does not match the frontier panel run")
    if payload.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("selection_policy does not match the frozen contract")
    if payload.get("reference_policy") != REFERENCE_POLICY:
        raise ValueError("reference_policy does not match the frozen contract")


def _validate_counts(
    payload: Mapping[str, object],
    evaluation_rows: Sequence[PanelRow],
    preflight_rows: Sequence[PanelRow],
) -> None:
    if len(evaluation_rows) != len({row.ordinal for row in evaluation_rows}):
        raise ValueError("evaluation ordinals must be distinct")
    if len(preflight_rows) != len({row.ordinal for row in preflight_rows}):
        raise ValueError("preflight ordinals must be distinct")
    if payload.get("evaluation_count") != len(evaluation_rows):
        raise ValueError("evaluation_count does not match the manifest rows")
    if payload.get("preflight_count") != len(preflight_rows):
        raise ValueError("preflight_count does not match the manifest rows")
    if len(evaluation_rows) != EVALUATION_PER_CATEGORY * len(CATEGORIES):
        raise ValueError("panel must include exactly 35 evaluation rows")
    if len(preflight_rows) != PREFLIGHT_ROWS:
        raise ValueError("panel must include exactly 2 preflight rows")
    counts = Counter(row.category for row in evaluation_rows)
    expected = {category: EVALUATION_PER_CATEGORY for category in CATEGORIES}
    if counts != expected:
        raise ValueError("panel must keep exactly five evaluation rows per category")


def _validate_preflight_distinct(
    evaluation_rows: Sequence[PanelRow],
    preflight_rows: Sequence[PanelRow],
) -> None:
    evaluation_video_ids = {row.video_id for row in evaluation_rows}
    preflight_video_ids = [row.video_id for row in preflight_rows]
    if len(preflight_video_ids) != len(set(preflight_video_ids)):
        raise ValueError("preflight rows must use distinct video_ids")
    if evaluation_video_ids & set(preflight_video_ids):
        raise ValueError("preflight rows must not reuse evaluation video_ids")


def _validate_selected_duration_limit(row: PanelRow) -> None:
    if row.duration_seconds > MAX_SELECTED_DURATION_SECONDS:
        raise ValueError(
            f"selected rows must be <= {MAX_SELECTED_DURATION_SECONDS:.1f} seconds: "
            f"{row.video_id}"
        )


def _candidate_index(
    candidate_sheet: Mapping[str, object],
) -> dict[tuple[str, str], dict[str, object]]:
    raw_candidates = candidate_sheet.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidate sheet must contain a candidate list")
    index: dict[tuple[str, str], dict[str, object]] = {}
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("candidate rows must be objects")
        video_id = _required_text(raw.get("video_id"), "candidate.video_id")
        question = _required_text(raw.get("question"), "candidate.question")
        index[(video_id, question)] = raw
    return index


def _validate_row_against_candidate(
    row: PanelRow,
    candidate_index: Mapping[tuple[str, str], dict[str, object]],
) -> None:
    candidate = candidate_index.get((row.video_id, row.question))
    if candidate is None:
        raise ValueError(f"panel row is not present in the candidate sheet: {row.video_id}")
    if candidate.get("category") != row.category:
        raise ValueError(f"panel category does not match the candidate sheet: {row.video_id}")
    if candidate.get("media_relative_path") != row.media_relative_path:
        raise ValueError(f"panel media path does not match the candidate sheet: {row.video_id}")
    if not math.isclose(
        _required_float(candidate.get("duration_seconds"), "candidate.duration_seconds"),
        row.duration_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"panel duration does not match the candidate sheet: {row.video_id}")


def _validate_media_identity(row: PanelRow, data_root: Path) -> None:
    media_path = (data_root / row.media_relative_path).resolve()
    root = data_root.resolve(strict=True)
    if not media_path.is_relative_to(root):
        raise ValueError("media path must stay beneath the dataset root")
    if not media_path.is_file():
        raise ValueError(f"media file does not exist: {media_path}")
    identity = source_fingerprint(media_path)
    if identity.sha256 != row.media_sha256:
        raise ValueError(f"media sha256 does not match current bytes: {row.video_id}")
    if identity.size_bytes != row.size_bytes:
        raise ValueError(f"media size does not match current bytes: {row.video_id}")
    metadata = probe_media(media_path)
    if not math.isclose(
        metadata.duration_seconds,
        row.duration_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"media duration does not match current bytes: {row.video_id}")
    if metadata.has_audio != row.has_audio:
        raise ValueError(f"media audio flag does not match current bytes: {row.video_id}")
    if metadata.has_video != row.has_video:
        raise ValueError(f"media video flag does not match current bytes: {row.video_id}")


def _validate_topology_diversity(evaluation_rows: Sequence[PanelRow]) -> None:
    by_category: dict[str, set[str]] = defaultdict(set)
    for row in evaluation_rows:
        by_category[row.category].add(row.evidence_topology)
    for category in CATEGORIES:
        if len(by_category[category]) < 3:
            raise ValueError(
                f"topology diversity requires at least three topology values in {category}"
            )


def _duration_bands(durations: Sequence[float]) -> tuple[float, float]:
    if len(durations) < 2:
        value = durations[0]
        return value, value
    try:
        lower, _, upper = quantiles(durations, n=4)
    except StatisticsError:
        midpoint = durations[len(durations) // 2]
        return durations[0], midpoint
    return lower, upper


def _validate_duration_coverage(
    evaluation_rows: Sequence[PanelRow],
    candidate_sheet: Mapping[str, object],
) -> None:
    raw_candidates = candidate_sheet.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidate sheet must contain a candidate list")
    candidate_durations: dict[str, list[float]] = defaultdict(list)
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise ValueError("candidate rows must be objects")
        category = _required_text(raw.get("category"), "candidate.category")
        candidate_durations[category].append(
            _required_float(raw.get("duration_seconds"), "candidate.duration_seconds")
        )
    selected_durations: dict[str, list[float]] = defaultdict(list)
    for row in evaluation_rows:
        selected_durations[row.category].append(row.duration_seconds)
    for category in CATEGORIES:
        durations = sorted(candidate_durations[category])
        if not durations:
            raise ValueError(f"candidate sheet is missing rows for {category}")
        lower, upper = _duration_bands(durations)
        selected = selected_durations[category]
        has_short = any(duration <= lower for duration in selected)
        has_typical = any(lower <= duration <= upper for duration in selected)
        has_long = any(duration >= upper for duration in selected)
        if not (has_short and has_typical and has_long):
            raise ValueError(
                f"duration coverage for {category} must include short, typical, "
                f"and long rows"
            )


def validate_panel_manifest(payload: object, data_root: Path) -> PanelManifest:
    if not isinstance(payload, Mapping):
        raise ValueError("panel manifest must be a JSON object")
    _reject_label_fields(payload)
    _validate_header(payload)
    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list):
        raise ValueError("panel manifest must include an examples list")
    rows = tuple(
        _parse_row(raw, location=f"examples[{index}]")
        for index, raw in enumerate(raw_examples)
    )
    if len(rows) != len({row.ordinal for row in rows}):
        raise ValueError("panel ordinals must be distinct")
    for row in rows:
        if classify_question(row.question) != row.category:
            raise ValueError(f"panel category disagrees with classify_question: {row.video_id}")
    evaluation_rows = _rows_by_phase(rows, "evaluation")
    preflight_rows = _rows_by_phase(rows, "preflight")
    _validate_counts(payload, evaluation_rows, preflight_rows)
    _validate_preflight_distinct(evaluation_rows, preflight_rows)
    candidate_sheet = build_candidate_sheet(data_root)
    candidate_index = _candidate_index(candidate_sheet)
    for row in rows:
        _validate_selected_duration_limit(row)
        _validate_row_against_candidate(row, candidate_index)
        _validate_media_identity(row, data_root)
    _validate_topology_diversity(evaluation_rows)
    _validate_duration_coverage(evaluation_rows, candidate_sheet)
    return PanelManifest(
        schema_version=SCHEMA_VERSION,
        run_id=RUN_ID,
        selection_policy=SELECTION_POLICY,
        reference_policy=REFERENCE_POLICY,
        evaluation_rows=tuple(sorted(evaluation_rows, key=lambda row: row.ordinal)),
        preflight_rows=tuple(sorted(preflight_rows, key=lambda row: row.ordinal)),
    )


def write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked output: {path}")
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"Output already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CANONICAL_DATA_ROOT",
    "CATEGORIES",
    "EVALUATION_PER_CATEGORY",
    "MAX_SELECTED_DURATION_SECONDS",
    "PREFLIGHT_ROWS",
    "PanelManifest",
    "PanelRow",
    "build_candidate_sheet",
    "classify_question",
    "validate_panel_manifest",
    "verify_canonical_data_root",
    "write_json_atomically",
]
