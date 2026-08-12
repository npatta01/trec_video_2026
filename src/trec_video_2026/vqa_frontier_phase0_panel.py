"""Build and validate the private, label-free Phase 0 panel."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA_VERSION = 1
PHASE0_RUN_ID = "vqa-frontier-phase0-20260812"
SELECTION_POLICY = "spent-successful-pilot-plus-category-median-duration-v1"
REFERENCE_POLICY = "not-read-not-serialized"
SOURCE_RUN_ID = "frontier-video-map-20260811"
SOURCE_SELECTION_POLICY = "question-only-category-purity-duration-topology-v1"
SOURCE_REFERENCE_POLICY = "not-read-not-serialized"
QWEN_CONDITION_ID = "qwen35-4b-12f-frames-only"
CONDITIONS = ("neutral_map", "direct_answer", "map_guided_answer")
CATEGORIES = (
    "temporal",
    "counting",
    "causal",
    "speech",
    "sound",
    "ocr_detail",
    "other_global",
)

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
        "label",
        "labels",
        "oracle",
        "reference",
        "references",
        "referenceanswer",
        "referenceanswers",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
    condition_ids: tuple[str, ...]
    qwen_condition_id: str | None = None
    qwen_status: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["condition_ids"] = list(self.condition_ids)
        if self.qwen_condition_id is None:
            payload.pop("qwen_condition_id")
        if self.qwen_status is None:
            payload.pop("qwen_status")
        return payload


@dataclass(frozen=True, slots=True)
class Phase0Panel:
    schema_version: int
    run_id: str
    selection_policy: str
    reference_policy: str
    source_panel_sha256: str
    qwen_predictions_sha256: str
    selected_ordered_ids_sha256: str
    source_media_sha256: str
    rows: tuple[PanelRow, ...]
    source_panel_file_sha256: str | None = None
    qwen_prediction_file_sha256: str | None = None

    @property
    def pilot_rows(self) -> tuple[PanelRow, ...]:
        return tuple(row for row in self.rows if row.phase == "pilot")

    @property
    def screen_rows(self) -> tuple[PanelRow, ...]:
        return tuple(row for row in self.rows if row.phase == "screen")

    def to_payload(self) -> dict[str, object]:
        bindings: dict[str, object] = {
            "source_panel_sha256": self.source_panel_sha256,
            "qwen_predictions_sha256": self.qwen_predictions_sha256,
            "selected_ordered_ids_sha256": self.selected_ordered_ids_sha256,
            "source_media_sha256": self.source_media_sha256,
            "selection_policy": self.selection_policy,
        }
        if self.source_panel_file_sha256 is not None:
            bindings["source_panel_file_sha256"] = self.source_panel_file_sha256
        if self.qwen_prediction_file_sha256 is not None:
            bindings["qwen_prediction_file_sha256"] = self.qwen_prediction_file_sha256
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "selection_policy": self.selection_policy,
            "reference_policy": self.reference_policy,
            "conditions": list(CONDITIONS),
            "pilot_count": len(self.pilot_rows),
            "screen_count": len(self.screen_rows),
            "bindings": bindings,
            "examples": [row.to_payload() for row in self.rows],
        }


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _required_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).casefold())


def _reject_label_fields(value: object, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalized_key(key) in _FORBIDDEN_LABEL_KEYS:
                raise ValueError(f"forbidden field at {location}.{key}")
            _reject_label_fields(nested, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_label_fields(nested, location=f"{location}[{index}]")


def _canonical_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_sha256(value: object, field: str) -> str:
    result = _required_text(value, field)
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _parse_source_row(raw: object, *, location: str) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{location} must be an object")
    category = _required_text(raw.get("category"), f"{location}.category")
    if category not in CATEGORIES:
        raise ValueError(f"{location}.category is not a Phase 0 category")
    phase = _required_text(raw.get("phase"), f"{location}.phase")
    if phase not in {"evaluation", "preflight"}:
        raise ValueError(f"{location}.phase must be evaluation or preflight")
    return {
        "ordinal": _required_int(raw.get("ordinal"), f"{location}.ordinal"),
        "phase": phase,
        "video_id": _required_text(raw.get("video_id"), f"{location}.video_id"),
        "question": _required_text(raw.get("question"), f"{location}.question"),
        "category": category,
        "evidence_topology": _required_text(
            raw.get("evidence_topology"), f"{location}.evidence_topology"
        ),
        "selection_rationale": _required_text(
            raw.get("selection_rationale"), f"{location}.selection_rationale"
        ),
        "media_relative_path": _required_text(
            raw.get("media_relative_path"), f"{location}.media_relative_path"
        ),
        "media_sha256": _required_sha256(
            raw.get("media_sha256"), f"{location}.media_sha256"
        ),
        "size_bytes": _required_int(raw.get("size_bytes"), f"{location}.size_bytes"),
        "duration_seconds": _required_float(
            raw.get("duration_seconds"), f"{location}.duration_seconds"
        ),
        "has_audio": _required_bool(raw.get("has_audio"), f"{location}.has_audio"),
        "has_video": _required_bool(raw.get("has_video"), f"{location}.has_video"),
    }


def _source_rows(source_panel: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    _reject_label_fields(source_panel)
    if source_panel.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("source panel schema_version does not match")
    if source_panel.get("run_id") != SOURCE_RUN_ID:
        raise ValueError("source panel run_id does not match the frozen frontier panel")
    if source_panel.get("selection_policy") != SOURCE_SELECTION_POLICY:
        raise ValueError("source panel selection_policy does not match")
    if source_panel.get("reference_policy") != SOURCE_REFERENCE_POLICY:
        raise ValueError("source panel reference_policy does not match")
    raw_examples = source_panel.get("examples")
    if not isinstance(raw_examples, list):
        raise ValueError("source panel must contain an examples list")
    rows = tuple(
        _parse_source_row(raw, location=f"source_panel.examples[{index}]")
        for index, raw in enumerate(raw_examples)
    )
    if len(rows) != 37:
        raise ValueError("source panel must contain exactly 37 rows")
    if len({row["ordinal"] for row in rows}) != len(rows):
        raise ValueError("source panel ordinals must be distinct")
    if len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("source panel video_ids must be distinct")
    evaluation = [row for row in rows if row["phase"] == "evaluation"]
    preflight = [row for row in rows if row["phase"] == "preflight"]
    if len(evaluation) != 35 or len(preflight) != 2:
        raise ValueError("source panel must contain 35 evaluation and 2 preflight rows")
    counts = Counter(row["category"] for row in evaluation)
    if counts != {category: 5 for category in CATEGORIES}:
        raise ValueError(
            "source panel must contain exactly five evaluation rows per category"
        )
    return rows


def _qwen_index(
    qwen_rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    index: dict[str, Mapping[str, object]] = {}
    for index_number, row in enumerate(qwen_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"qwen_rows[{index_number}] must be an object")
        video_id = _required_text(
            row.get("video_id"), f"qwen_rows[{index_number}].video_id"
        )
        if video_id in index:
            raise ValueError(f"Qwen baseline contains duplicate video_id: {video_id}")
        index[video_id] = row
    return index


def _require_qwen_baseline(
    qwen_index: Mapping[str, Mapping[str, object]], video_id: str
) -> Mapping[str, object]:
    row = qwen_index.get(video_id)
    if row is None:
        raise ValueError(f"missing matched Qwen baseline for {video_id}")
    if row.get("condition_id") != QWEN_CONDITION_ID or row.get("status") != "passed":
        raise ValueError(f"matched Qwen baseline is not passed for {video_id}")
    return row


def _phase0_row(
    source: Mapping[str, object],
    *,
    phase: str,
    condition_ids: tuple[str, ...],
    qwen: Mapping[str, object] | None,
) -> PanelRow:
    qwen_condition_id = None if qwen is None else QWEN_CONDITION_ID
    qwen_status = None if qwen is None else "passed"
    rationale = (
        "spent successful preflight pilot"
        if phase == "pilot"
        else "median duration row from the unused evaluation rows in this category"
    )
    return PanelRow(
        ordinal=int(source["ordinal"]),
        phase=phase,
        video_id=str(source["video_id"]),
        question=str(source["question"]),
        category=str(source["category"]),
        evidence_topology=str(source["evidence_topology"]),
        selection_rationale=rationale,
        media_relative_path=str(source["media_relative_path"]),
        media_sha256=str(source["media_sha256"]),
        size_bytes=int(source["size_bytes"]),
        duration_seconds=float(source["duration_seconds"]),
        has_audio=bool(source["has_audio"]),
        has_video=bool(source["has_video"]),
        condition_ids=condition_ids,
        qwen_condition_id=qwen_condition_id,
        qwen_status=qwen_status,
    )


def _screen_choice(rows: Sequence[PanelRow]) -> PanelRow:
    ordered = sorted(rows, key=lambda row: (row.duration_seconds, row.ordinal))
    return ordered[len(ordered) // 2]


def _build_panel_object(
    source_panel: Mapping[str, object],
    successful_pilot_video_id: str,
    qwen_rows: Sequence[Mapping[str, object]],
) -> Phase0Panel:
    source_rows = _source_rows(source_panel)
    pilot_id = _required_text(successful_pilot_video_id, "successful_pilot_video_id")
    preflight = [row for row in source_rows if row["phase"] == "preflight"]
    pilot_source = next((row for row in preflight if row["video_id"] == pilot_id), None)
    if pilot_source is None:
        raise ValueError(
            "successful pilot must be one of the source panel preflight rows"
        )
    qwen_index = _qwen_index(qwen_rows)
    evaluation_by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in source_rows:
        if row["phase"] == "evaluation":
            evaluation_by_category[str(row["category"])].append(row)

    pilot = _phase0_row(
        pilot_source,
        phase="pilot",
        condition_ids=("neutral_map",),
        qwen=None,
    )
    screen: list[PanelRow] = []
    for category in CATEGORIES:
        selected_source = _screen_choice(
            [
                _phase0_row(row, phase="screen", condition_ids=CONDITIONS, qwen=None)
                for row in evaluation_by_category[category]
            ]
        )
        qwen = _require_qwen_baseline(qwen_index, selected_source.video_id)
        screen.append(
            _phase0_row(
                next(
                    row
                    for row in evaluation_by_category[category]
                    if row["video_id"] == selected_source.video_id
                ),
                phase="screen",
                condition_ids=CONDITIONS,
                qwen=qwen,
            )
        )
    rows = (pilot, *screen)
    ordered_ids = [row.video_id for row in rows]
    source_media = [
        {"video_id": row.video_id, "source_sha256": row.media_sha256} for row in rows
    ]
    return Phase0Panel(
        schema_version=SCHEMA_VERSION,
        run_id=PHASE0_RUN_ID,
        selection_policy=SELECTION_POLICY,
        reference_policy=REFERENCE_POLICY,
        source_panel_sha256=_canonical_sha256(source_panel),
        qwen_predictions_sha256=_canonical_sha256(qwen_rows),
        selected_ordered_ids_sha256=_canonical_sha256(ordered_ids),
        source_media_sha256=_canonical_sha256(source_media),
        rows=rows,
    )


def build_phase0_panel(
    source_panel: Mapping[str, object],
    successful_pilot_video_id: str,
    qwen_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select one spent pilot and seven unused category-median rows."""

    return _build_panel_object(
        source_panel, successful_pilot_video_id, qwen_rows
    ).to_payload()


def _parse_phase0_row(raw: object, *, location: str) -> PanelRow:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{location} must be an object")
    phase = _required_text(raw.get("phase"), f"{location}.phase")
    if phase not in {"pilot", "screen"}:
        raise ValueError(f"{location}.phase must be pilot or screen")
    source = dict(raw)
    source["phase"] = "preflight" if phase == "pilot" else "evaluation"
    parsed = _parse_source_row(source, location=location)
    raw_conditions = raw.get("condition_ids")
    if not isinstance(raw_conditions, list) or any(
        not isinstance(condition, str) for condition in raw_conditions
    ):
        raise ValueError(f"{location}.condition_ids must be a list of strings")
    expected_conditions = ("neutral_map",) if phase == "pilot" else CONDITIONS
    if tuple(raw_conditions) != expected_conditions:
        raise ValueError(f"{location}.condition_ids do not match the frozen phase")
    qwen_condition_id = raw.get("qwen_condition_id")
    qwen_status = raw.get("qwen_status")
    if phase == "screen":
        if qwen_condition_id != QWEN_CONDITION_ID or qwen_status != "passed":
            raise ValueError(f"{location} must bind a passed Qwen baseline")
    elif qwen_condition_id is not None or qwen_status is not None:
        raise ValueError(f"{location} pilot must not contain a Qwen baseline binding")
    return PanelRow(
        ordinal=int(parsed["ordinal"]),
        phase=phase,
        video_id=str(parsed["video_id"]),
        question=str(parsed["question"]),
        category=str(parsed["category"]),
        evidence_topology=str(parsed["evidence_topology"]),
        selection_rationale=str(parsed["selection_rationale"]),
        media_relative_path=str(parsed["media_relative_path"]),
        media_sha256=str(parsed["media_sha256"]),
        size_bytes=int(parsed["size_bytes"]),
        duration_seconds=float(parsed["duration_seconds"]),
        has_audio=bool(parsed["has_audio"]),
        has_video=bool(parsed["has_video"]),
        condition_ids=tuple(raw_conditions),
        qwen_condition_id=qwen_condition_id
        if isinstance(qwen_condition_id, str)
        else None,
        qwen_status=qwen_status if isinstance(qwen_status, str) else None,
    )


def _validate_phase0_header(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Phase 0 schema_version does not match")
    if payload.get("run_id") != PHASE0_RUN_ID:
        raise ValueError("Phase 0 run_id does not match")
    if payload.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("Phase 0 selection_policy does not match")
    if payload.get("reference_policy") != REFERENCE_POLICY:
        raise ValueError("Phase 0 reference_policy does not match")
    if payload.get("conditions") != list(CONDITIONS):
        raise ValueError("Phase 0 conditions do not match")


def _validate_bindings(
    payload: Mapping[str, object],
    *,
    source_panel: Mapping[str, object],
    qwen_rows: Sequence[Mapping[str, object]],
    rows: Sequence[PanelRow],
) -> tuple[str, str, str, str, str | None, str | None]:
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Phase 0 manifest must contain bindings")
    expected_source = _canonical_sha256(source_panel)
    expected_qwen = _canonical_sha256(qwen_rows)
    ordered_ids = [row.video_id for row in rows]
    expected_ids = _canonical_sha256(ordered_ids)
    source_media = [
        {"video_id": row.video_id, "source_sha256": row.media_sha256} for row in rows
    ]
    expected_media = _canonical_sha256(source_media)
    values = (
        _required_sha256(bindings.get("source_panel_sha256"), "source_panel_sha256"),
        _required_sha256(
            bindings.get("qwen_predictions_sha256"), "qwen_predictions_sha256"
        ),
        _required_sha256(
            bindings.get("selected_ordered_ids_sha256"), "selected_ordered_ids_sha256"
        ),
        _required_sha256(bindings.get("source_media_sha256"), "source_media_sha256"),
    )
    if values != (expected_source, expected_qwen, expected_ids, expected_media):
        raise ValueError(
            "Phase 0 binding does not match source panel or Qwen predictions"
        )
    if bindings.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("selection policy binding does not match")
    optional = []
    for field in ("source_panel_file_sha256", "qwen_prediction_file_sha256"):
        value = bindings.get(field)
        optional.append(None if value is None else _required_sha256(value, field))
    return (*values, optional[0], optional[1])


def validate_phase0_panel(
    payload: object,
    *,
    source_panel: Mapping[str, object],
    qwen_rows: Sequence[Mapping[str, object]],
) -> Phase0Panel:
    """Validate a Phase 0 panel against its label-free source inputs."""

    if not isinstance(payload, Mapping):
        raise ValueError("Phase 0 manifest must be an object")
    _reject_label_fields(payload)
    _validate_phase0_header(payload)
    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list):
        raise ValueError("Phase 0 manifest must contain an examples list")
    rows = tuple(
        _parse_phase0_row(raw, location=f"examples[{index}]")
        for index, raw in enumerate(raw_examples)
    )
    if len(rows) != 8:
        raise ValueError(
            "Phase 0 manifest must contain one pilot and seven screen rows"
        )
    if len({row.ordinal for row in rows}) != len(rows):
        raise ValueError("Phase 0 row ordinals must be distinct")
    if len({row.video_id for row in rows}) != len(rows):
        raise ValueError("Phase 0 video_ids must be distinct")
    pilot_rows = tuple(row for row in rows if row.phase == "pilot")
    screen_rows = tuple(row for row in rows if row.phase == "screen")
    if len(pilot_rows) != 1 or len(screen_rows) != 7:
        raise ValueError(
            "Phase 0 manifest must contain one pilot and seven screen rows"
        )
    if {row.category for row in screen_rows} != set(CATEGORIES):
        raise ValueError("Phase 0 screen must contain one row per category")
    if payload.get("pilot_count") != 1 or payload.get("screen_count") != 7:
        raise ValueError("Phase 0 counts do not match the manifest rows")

    source_rows = _source_rows(source_panel)
    source_by_id = {str(row["video_id"]): row for row in source_rows}
    qwen_index = _qwen_index(qwen_rows)
    for row in rows:
        source = source_by_id.get(row.video_id)
        if source is None:
            raise ValueError(
                f"Phase 0 row is not present in the source panel: {row.video_id}"
            )
        expected_source_phase = "preflight" if row.phase == "pilot" else "evaluation"
        if source["phase"] != expected_source_phase:
            raise ValueError(
                f"Phase 0 row phase does not match source panel: {row.video_id}"
            )
        for field in (
            "ordinal",
            "question",
            "category",
            "evidence_topology",
            "media_relative_path",
            "media_sha256",
            "size_bytes",
            "duration_seconds",
            "has_audio",
            "has_video",
        ):
            if getattr(row, field) != source[field]:
                raise ValueError(
                    f"Phase 0 row binding changed for {row.video_id}: {field}"
                )
        if row.phase == "screen":
            _require_qwen_baseline(qwen_index, row.video_id)

    selected_by_category = {row.category: row for row in screen_rows}
    for category in CATEGORIES:
        expected = _screen_choice(
            [
                _phase0_row(source, phase="screen", condition_ids=CONDITIONS, qwen=None)
                for source in source_rows
                if source["phase"] == "evaluation" and source["category"] == category
            ]
        )
        if selected_by_category[category].video_id != expected.video_id:
            raise ValueError(
                f"Phase 0 screen row is not the category median: {category}"
            )

    source_sha, qwen_sha, ids_sha, media_sha, source_file_sha, qwen_file_sha = (
        _validate_bindings(
            payload, source_panel=source_panel, qwen_rows=qwen_rows, rows=rows
        )
    )
    return Phase0Panel(
        schema_version=SCHEMA_VERSION,
        run_id=PHASE0_RUN_ID,
        selection_policy=SELECTION_POLICY,
        reference_policy=REFERENCE_POLICY,
        source_panel_sha256=source_sha,
        qwen_predictions_sha256=qwen_sha,
        selected_ordered_ids_sha256=ids_sha,
        source_media_sha256=media_sha,
        rows=rows,
        source_panel_file_sha256=source_file_sha,
        qwen_prediction_file_sha256=qwen_file_sha,
    )


def write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    """Create one immutable JSON file without following an output symlink."""

    if path.is_symlink():
        raise ValueError(f"refusing symlinked output: {path}")
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
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
    "CATEGORIES",
    "CONDITIONS",
    "PHASE0_RUN_ID",
    "QWEN_CONDITION_ID",
    "REFERENCE_POLICY",
    "SELECTION_POLICY",
    "SOURCE_REFERENCE_POLICY",
    "SOURCE_RUN_ID",
    "SOURCE_SELECTION_POLICY",
    "PanelRow",
    "Phase0Panel",
    "build_phase0_panel",
    "validate_phase0_panel",
    "write_json_atomically",
]
