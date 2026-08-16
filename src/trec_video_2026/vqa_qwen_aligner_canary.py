"""Immutable Qwen ASR plus forced-aligner canary and quality-panel harness."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from trec_video_2026.vqa_experiments.temporal_bottleneck import (
    freeze_json_artifact,
    verify_json_artifact,
)


SCHEMA_VERSION = 1
EXPERIMENT_ID = "qwen-aligner-stage05-20260815-v1"
QUALITY_SELECTION_KIND = "qwen_aligner_quality_panel"
FRESH_ENGLISH_SELECTION_KIND = "qwen_aligned_english_fresh_speech_panel"
FRESH_PANEL_SIZE = 12
FRESH_STRATUM_QUOTAS = {
    "reported_speech": 6,
    "conversation": 3,
    "vocalization": 3,
}
GEMINI_CONDITION = "gemini-3.1-pro-native-av-4fps"
CONDITION_ID = "qwen3-asr-1.7b-forced-aligner-0.6b"
ENGLISH_CONDITION_ID = "qwen3-asr-1.7b-forced-aligner-0.6b-english"
ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
ASR_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
RUNTIME_IMAGE = (
    "localhost/trec-vqa-audio-rocm@sha256:"
    "faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234"
)
CANONICAL_DATA_ROOT = Path(
    "/home/npatta01/data/competitions/trec_video_2026/data"
).resolve()
RUNNER_PATH = Path(__file__).with_name("vqa_audio_runner.py").resolve()
DEDICATED_RUNNER_PATH = Path(__file__).with_name(
    "vqa_qwen_empty_output_runner.py"
).resolve()
CONTROLLER_PATH = Path(__file__).resolve()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QWEN_ASR_MARKERS = (
    "Qwen/Qwen3-ASR-1.7B",
    "qwen3_asr",
    "qwen3-asr",
)
_VISIBLE_TEXT = re.compile(
    r"\b(?:sign|caption|subtitle|screen|label|shirt|board|paper|written|writing|text)\b",
    re.IGNORECASE,
)
_SPEECH_PATTERNS = {
    "reported_speech": re.compile(
        r"\b(?:say|says|said|saying|tell|tells|told|telling|ask|asks|asked|asking|"
        r"answer|answers|answered|answering|reply|replies|replied|replying|respond|"
        r"responds|responded|responding|announce|announces|announced|announcing|"
        r"mention|mentions|mentioned|mentioning)\b",
        re.IGNORECASE,
    ),
    "conversation": re.compile(
        r"\b(?:speak|speaks|spoke|spoken|speaking|talk|talks|talked|talking)\b",
        re.IGNORECASE,
    ),
    "vocalization": re.compile(
        r"\b(?:shout|shouts|shouted|shouting|yell|yells|yelled|yelling|scream|"
        r"screams|screamed|screaming|sing|sings|sang|sung|singing|song|songs)\b",
        re.IGNORECASE,
    ),
}
_FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "annotation",
    "annotations",
    "gold",
    "label",
    "labels",
    "map",
    "metric",
    "metrics",
    "question",
    "reference",
    "references",
    "transcript",
    "transcripts",
}


def canonical_sha256(value: Any) -> str:
    """Hash JSON values without depending on the larger VQA submission module."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configure_experiment_id(value: str) -> str:
    """Select a fresh run identity for one CLI process without changing v1."""

    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,80}", value) is None
    ):
        raise ValueError("experiment ID must be a lowercase bounded identifier")
    global EXPERIMENT_ID
    EXPERIMENT_ID = value
    return EXPERIMENT_ID


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_speech_record(value: Any) -> bool:
    return isinstance(value, str) and any(
        line.startswith("SPEECH |") for line in value.splitlines()
    )


def _load_json_or_jsonl(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read exposure source: {path.name}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"exposure source is not valid JSON/JSONL: {path.name}") from exc


def _collect_video_ids(value: Any) -> set[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            video_id = item.get("video_id")
            if isinstance(video_id, str) and video_id:
                found.add(video_id)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return found


def _canonical_private_file(path: Path, *, field: str) -> Path:
    resolved = path.resolve()
    if (
        not resolved.is_relative_to(CANONICAL_DATA_ROOT)
        or not resolved.is_file()
    ):
        raise ValueError(f"{field} must be a file inside the canonical data root")
    return resolved


def build_qwen_asr_exposure_index(
    *,
    source_paths: Sequence[Path],
    oracle_manifest: Path,
    built_at: str,
) -> dict[str, Any]:
    """Freeze identities previously exposed to Qwen3-ASR without output text."""

    if not source_paths:
        raise ValueError("at least one Qwen3-ASR exposure source is required")
    if not isinstance(built_at, str) or not built_at:
        raise ValueError("exposure index timestamp is required")
    resolved_sources = [
        _canonical_private_file(path, field="exposure source")
        for path in source_paths
    ]
    if len(set(resolved_sources)) != len(resolved_sources):
        raise ValueError("exposure source paths must be unique")
    resolved_oracle = _canonical_private_file(
        oracle_manifest, field="oracle manifest"
    )

    excluded: set[str] = set()
    source_records: list[dict[str, Any]] = []
    for path in sorted(resolved_sources, key=str):
        raw = path.read_text(encoding="utf-8")
        matched = [marker for marker in _QWEN_ASR_MARKERS if marker in raw]
        if not matched:
            raise ValueError("exposure source does not contain an exact Qwen3-ASR marker")
        ids = _collect_video_ids(_load_json_or_jsonl(path))
        excluded.update(ids)
        source_records.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "matched_markers": matched,
                "excluded_video_id_count": len(ids),
            }
        )

    oracle_ids = _collect_video_ids(_load_json_or_jsonl(resolved_oracle))
    excluded.update(oracle_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "index_kind": "qwen3_asr_exposure_index",
        "built_at": built_at,
        "markers": list(_QWEN_ASR_MARKERS),
        "sources": source_records,
        "oracle_manifest": {
            "path": str(resolved_oracle),
            "sha256": _sha256_file(resolved_oracle),
            "excluded_video_id_count": len(oracle_ids),
        },
        "excluded_video_ids": sorted(excluded),
    }


def _verify_qwen_asr_exposure_index(index: Mapping[str, Any]) -> set[str]:
    if set(index) != {
        "schema_version",
        "index_kind",
        "built_at",
        "markers",
        "sources",
        "oracle_manifest",
        "excluded_video_ids",
    }:
        raise ValueError("Qwen3-ASR exposure index keys are invalid")
    if (
        index.get("schema_version") != SCHEMA_VERSION
        or index.get("index_kind") != "qwen3_asr_exposure_index"
        or index.get("markers") != list(_QWEN_ASR_MARKERS)
    ):
        raise ValueError("Qwen3-ASR exposure index contract is invalid")
    sources = index.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Qwen3-ASR exposure index sources are invalid")
    seen_paths: set[Path] = set()
    for record in sources:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "matched_markers",
            "excluded_video_id_count",
        }:
            raise ValueError("Qwen3-ASR exposure source record is invalid")
        path = _canonical_private_file(Path(str(record["path"])), field="exposure source")
        if path in seen_paths:
            raise ValueError("Qwen3-ASR exposure source paths must be unique")
        seen_paths.add(path)
        if record.get("sha256") != _sha256_file(path):
            raise ValueError("Qwen3-ASR exposure source hash does not match")
        matched = record.get("matched_markers")
        if (
            not isinstance(matched, list)
            or not matched
            or any(marker not in _QWEN_ASR_MARKERS for marker in matched)
        ):
            raise ValueError("Qwen3-ASR exposure source markers are invalid")

    oracle = index.get("oracle_manifest")
    if not isinstance(oracle, Mapping) or set(oracle) != {
        "path",
        "sha256",
        "excluded_video_id_count",
    }:
        raise ValueError("Qwen3-ASR oracle manifest record is invalid")
    oracle_path = _canonical_private_file(
        Path(str(oracle["path"])), field="oracle manifest"
    )
    if oracle.get("sha256") != _sha256_file(oracle_path):
        raise ValueError("Qwen3-ASR oracle exposure hash does not match")
    excluded = index.get("excluded_video_ids")
    if (
        not isinstance(excluded, list)
        or any(not isinstance(video_id, str) or not video_id for video_id in excluded)
        or excluded != sorted(set(excluded))
    ):
        raise ValueError("Qwen3-ASR excluded identities are invalid")
    return set(excluded)


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
    except (OSError, wave.Error) as exc:
        raise ValueError(f"cannot read WAV metadata: {path.name}") from exc
    if frame_rate <= 0 or frame_count <= 0:
        raise ValueError(f"WAV duration is invalid: {path.name}")
    return frame_count / frame_rate


def _speech_stratum(question: str) -> str | None:
    if _VISIBLE_TEXT.search(question):
        return None
    for stratum in FRESH_STRATUM_QUOTAS:
        if _SPEECH_PATTERNS[stratum].search(question):
            return stratum
    return None


def select_fresh_english_quality_panel(
    *,
    manifest: Mapping[str, Any],
    exposure_index: Mapping[str, Any],
    audio_root: Path,
    selected_at: str,
    source_hashes: Mapping[str, str],
    panel_size: int = FRESH_PANEL_SIZE,
) -> dict[str, Any]:
    """Select the fixed question-only fresh 6/3/3 English speech panel."""

    if panel_size != FRESH_PANEL_SIZE or panel_size != sum(FRESH_STRATUM_QUOTAS.values()):
        raise ValueError("fresh English panel size must match the fixed quota total")
    audio_root = audio_root.resolve()
    if (
        not audio_root.is_relative_to(CANONICAL_DATA_ROOT)
        or not audio_root.is_dir()
    ):
        raise ValueError("audio root must be inside the canonical data root")
    excluded_ids = _verify_qwen_asr_exposure_index(exposure_index)
    examples = manifest.get("examples")
    if not isinstance(examples, list):
        raise ValueError("manifest examples must be an array")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("selection source hashes must be nonempty")
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in source_hashes.values()
    ):
        raise ValueError("selection source hashes must be lowercase SHA-256 values")

    candidates: dict[str, list[dict[str, Any]]] = {
        stratum: [] for stratum in FRESH_STRATUM_QUOTAS
    }
    seen: set[str] = set()
    for source_index, example in enumerate(examples):
        if not isinstance(example, Mapping) or set(example) != {"video_id", "question"}:
            raise ValueError("manifest rows must contain only video_id and question")
        video_id = example.get("video_id")
        question = example.get("question")
        if (
            not isinstance(video_id, str)
            or not video_id
            or video_id in seen
            or not isinstance(question, str)
        ):
            raise ValueError("manifest rows require unique IDs and string questions")
        seen.add(video_id)
        if video_id in excluded_ids:
            continue
        stratum = _speech_stratum(question)
        if stratum is not None:
            candidates[stratum].append(
                {
                    "video_id": video_id,
                    "speech_stratum": stratum,
                    "source_order_index": source_index,
                }
            )

    chosen: list[dict[str, Any]] = []
    for stratum, quota in FRESH_STRATUM_QUOTAS.items():
        rows = candidates[stratum]
        if len(rows) < quota:
            raise ValueError(f"fresh English stratum quota is not satisfied: {stratum}")
        chosen.extend(rows[:quota])
    chosen.sort(key=lambda row: int(row["source_order_index"]))

    rows = []
    for candidate in chosen:
        video_id = str(candidate["video_id"])
        audio_path = (audio_root / f"{video_id}.wav").resolve()
        if not audio_path.is_relative_to(audio_root) or not audio_path.is_file():
            raise ValueError("selected audio is missing or escapes the audio root")
        rows.append(
            {
                "role": "speech_quality",
                "speech_stratum": candidate["speech_stratum"],
                "video_id": video_id,
                "audio_path": str(audio_path),
                "audio_sha256": _sha256_file(audio_path),
                "duration_seconds": _wav_duration(audio_path),
                "source_order_index": candidate["source_order_index"],
                "selection_provenance": (
                    "manifest_question_whole_word+fresh_qwen3_asr+fixed_stratum_quota"
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selection_kind": FRESH_ENGLISH_SELECTION_KIND,
        "panel_size": FRESH_PANEL_SIZE,
        "selected_at": selected_at,
        "selection_policy": (
            "first fresh visible-text-excluded speech rows in fixed 6/3/3 strata, "
            "restored to manifest order"
        ),
        "stratum_counts": dict(FRESH_STRATUM_QUOTAS),
        "rows": rows,
        "source_hashes": dict(source_hashes),
    }


def select_panel(
    *,
    scores: Mapping[str, Any],
    maps: Sequence[Mapping[str, Any]],
    transcripts: Mapping[str, Any],
    prior_qwen_ids: set[str],
    reviewed_failure_ids: set[str],
    audio_root: Path,
    selected_at: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Select one fresh timestamp-overrun target and one strong speech control."""

    audio_root = audio_root.resolve()
    if not audio_root.is_relative_to(CANONICAL_DATA_ROOT):
        raise ValueError("audio root must be inside the canonical data root")
    score_groups = scores.get("rows")
    if not isinstance(score_groups, Mapping):
        raise ValueError("scores rows must be an object")
    score_rows = score_groups.get(GEMINI_CONDITION)
    if not isinstance(score_rows, list):
        raise ValueError("Gemini score rows must be an array")
    map_by_id = {
        row.get("video_id"): row
        for row in maps
        if isinstance(row, Mapping) and isinstance(row.get("video_id"), str)
    }
    transcript_rows = transcripts.get("results")
    if not isinstance(transcript_rows, list):
        raise ValueError("transcript results must be an array")
    transcript_by_id = {
        row.get("video_id"): row
        for row in transcript_rows
        if isinstance(row, Mapping) and isinstance(row.get("video_id"), str)
    }

    targets: list[Mapping[str, Any]] = []
    controls: list[Mapping[str, Any]] = []
    for score in score_rows:
        if not isinstance(score, Mapping) or score.get("question_stratum") != "audio_or_text":
            continue
        video_id = score.get("video_id")
        if not isinstance(video_id, str) or video_id in prior_qwen_ids:
            continue
        map_row = map_by_id.get(video_id)
        transcript = transcript_by_id.get(video_id)
        if not isinstance(map_row, Mapping) or not isinstance(transcript, Mapping):
            continue
        if str(transcript.get("source_language", "")).casefold() != "english":
            continue
        quality = transcript.get("quality")
        quality_status = quality.get("status") if isinstance(quality, Mapping) else None
        speech_record = _has_speech_record(map_row.get("map"))
        if not speech_record and quality_status == "rejected-timestamp-overrun":
            targets.append(score)
            continue
        composite = score.get("semantic_composite")
        if (
            speech_record
            and quality_status == "accepted"
            and transcript.get("transcript_included") is True
            and video_id not in reviewed_failure_ids
            and isinstance(composite, (int, float))
            and not isinstance(composite, bool)
        ):
            controls.append(score)

    if len(targets) != 1:
        raise ValueError("selection requires exactly one fresh English target")
    if not controls:
        raise ValueError("selection requires at least one fresh English control")
    target = targets[0]
    control = sorted(
        controls,
        key=lambda row: (-float(row["semantic_composite"]), str(row["video_id"])),
    )[0]

    def selected_row(score: Mapping[str, Any], role: str) -> dict[str, Any]:
        video_id = str(score["video_id"])
        transcript = transcript_by_id[video_id]
        path = (audio_root / f"{video_id}.wav").resolve()
        if not path.is_relative_to(audio_root) or not path.is_file():
            raise ValueError("selected audio is missing or escapes the audio root")
        duration = transcript.get("source_audio_duration_seconds")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            raise ValueError("selected audio duration must be positive")
        provenance = (
            "audio_or_text+missing_speech_record+english+"
            "whisper_timestamp_overrun+fresh_qwen"
            if role == "target"
            else "audio_or_text+speech_record+english+whisper_accepted+"
            "unreviewed+best_semantic_control+fresh_qwen"
        )
        return {
            "role": role,
            "video_id": video_id,
            "audio_path": str(path),
            "audio_sha256": _sha256_file(path),
            "duration_seconds": float(duration),
            "selection_provenance": provenance,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selected_at": selected_at,
        "selection_policy": (
            "unique fresh English timestamp-overrun target; highest-composite "
            "fresh accepted English speech control outside reviewed failures"
        ),
        "rows": [selected_row(target, "target"), selected_row(control, "control")],
        "source_hashes": dict(source_hashes),
    }


def select_quality_panel(
    *,
    candidates: Sequence[Mapping[str, Any]],
    prior_qwen_ids: set[str],
    audio_root: Path,
    selected_at: str,
    source_hashes: Mapping[str, str],
    panel_size: int = 8,
) -> dict[str, Any]:
    """Select an exact fresh English speech panel without reading labels."""

    if panel_size <= 0:
        raise ValueError("quality panel size must be positive")
    audio_root = audio_root.resolve()
    if not audio_root.is_relative_to(CANONICAL_DATA_ROOT):
        raise ValueError("audio root must be inside the canonical data root")
    eligible: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("quality candidates must be objects")
        video_id = candidate.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise ValueError("quality candidates must have unique video IDs")
        seen.add(video_id)
        language = str(candidate.get("source_language", "")).casefold()
        if video_id in prior_qwen_ids or language not in {"en", "english"}:
            continue
        if candidate.get("whisper_status") != "accepted":
            continue
        eligible.append(candidate)
    if len(eligible) != panel_size:
        raise ValueError(
            f"quality panel requires exactly {panel_size} fresh English rows; "
            f"found {len(eligible)}"
        )

    rows: list[dict[str, Any]] = []
    for candidate in eligible:
        video_id = str(candidate["video_id"])
        path = (audio_root / f"{video_id}.wav").resolve()
        if not path.is_relative_to(audio_root) or not path.is_file():
            raise ValueError("quality panel audio is missing or escapes the audio root")
        duration = candidate.get("duration_seconds")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            raise ValueError("quality panel duration must be positive")
        rows.append(
            {
                "role": "speech_quality",
                "video_id": video_id,
                "audio_path": str(path),
                "audio_sha256": _sha256_file(path),
                "duration_seconds": float(duration),
                "selection_provenance": (
                    "oracle_speech_10+english+whisper_accepted+fresh_qwen"
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selection_kind": QUALITY_SELECTION_KIND,
        "panel_size": panel_size,
        "selected_at": selected_at,
        "selection_policy": (
            "all fresh English exact-speech rows from the declared oracle panel "
            "after prior Qwen exclusion"
        ),
        "rows": rows,
        "source_hashes": dict(source_hashes),
    }


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing artifact: {path.name}") from exc


def _write_text_exclusive(path: Path, value: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(value)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing artifact: {path.name}") from exc


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path.name}")
    return value


def _walk_label_free(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact keys must be strings")
            normalized = re.sub(r"[^a-z]", "", key.casefold())
            if normalized in _FORBIDDEN_KEYS:
                raise ValueError("label-bearing fields are forbidden")
            _walk_label_free(child)
    elif isinstance(value, list):
        for child in value:
            _walk_label_free(child)


def _validate_selection(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    selection_kind = selection.get("selection_kind")
    expected = {
        "schema_version",
        "experiment_id",
        "selection_policy",
        "rows",
        "source_hashes",
    }
    if selection_kind == QUALITY_SELECTION_KIND:
        expected.update({"selection_kind", "panel_size", "selected_at"})
    elif selection_kind == FRESH_ENGLISH_SELECTION_KIND:
        expected.update(
            {
                "selection_kind",
                "panel_size",
                "selected_at",
                "stratum_counts",
            }
        )
    else:
        expected.add("selected_at")
    if set(selection) != expected:
        raise ValueError("selection keys do not match the canary contract")
    if selection["schema_version"] != SCHEMA_VERSION:
        raise ValueError("selection schema version is invalid")
    if selection["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("selection experiment ID is invalid")
    _walk_label_free(selection)
    rows = selection["rows"]
    if not isinstance(rows, list):
        raise ValueError("selection rows must be an array")
    if selection_kind in {QUALITY_SELECTION_KIND, FRESH_ENGLISH_SELECTION_KIND}:
        panel_size = selection.get("panel_size")
        if (
            not isinstance(panel_size, int)
            or isinstance(panel_size, bool)
            or panel_size <= 0
        ):
            raise ValueError("quality panel size is invalid")
        if len(rows) != panel_size:
            raise ValueError(f"selection must contain exactly {panel_size} rows")
        if any(
            not isinstance(row, Mapping) or row.get("role") != "speech_quality"
            for row in rows
        ):
            raise ValueError("quality selection roles are invalid")
        if selection_kind == FRESH_ENGLISH_SELECTION_KIND:
            if panel_size != FRESH_PANEL_SIZE:
                raise ValueError("fresh English panel size is invalid")
            if selection.get("stratum_counts") != FRESH_STRATUM_QUOTAS:
                raise ValueError("fresh English stratum counts are invalid")
    else:
        if len(rows) != 2:
            raise ValueError("selection must contain exactly two rows")
        if [row.get("role") for row in rows if isinstance(row, Mapping)] != [
            "target",
            "control",
        ]:
            raise ValueError("selection order must be target then control")
    expected_row_keys = {
        "role",
        "video_id",
        "audio_path",
        "audio_sha256",
        "duration_seconds",
        "selection_provenance",
    }
    if selection_kind == FRESH_ENGLISH_SELECTION_KIND:
        expected_row_keys.update({"speech_stratum", "source_order_index"})
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_row_keys:
            raise ValueError("selection row keys do not match the canary contract")
        video_id = row["video_id"]
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise ValueError("selection video IDs must be nonempty and unique")
        seen.add(video_id)
        audio_path = Path(str(row["audio_path"]))
        resolved_audio = audio_path.resolve()
        if (
            not audio_path.is_absolute()
            or not resolved_audio.is_relative_to(CANONICAL_DATA_ROOT)
            or not resolved_audio.is_file()
        ):
            raise ValueError(
                "selection audio path must be inside the canonical data root"
            )
        actual_hash = _sha256_file(audio_path)
        if row["audio_sha256"] != actual_hash:
            raise ValueError("selection audio SHA-256 does not match")
        duration = row["duration_seconds"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            raise ValueError("selection duration must be positive")
        if selection_kind == FRESH_ENGLISH_SELECTION_KIND:
            if row.get("speech_stratum") not in FRESH_STRATUM_QUOTAS:
                raise ValueError("fresh English speech stratum is invalid")
            source_index = row.get("source_order_index")
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index < 0
            ):
                raise ValueError("fresh English source order index is invalid")
        normalized.append(dict(row))
    if selection_kind == FRESH_ENGLISH_SELECTION_KIND:
        observed_counts = {
            stratum: sum(row["speech_stratum"] == stratum for row in normalized)
            for stratum in FRESH_STRATUM_QUOTAS
        }
        source_indexes = [row["source_order_index"] for row in normalized]
        if observed_counts != FRESH_STRATUM_QUOTAS:
            raise ValueError("fresh English row strata do not match fixed quotas")
        if source_indexes != sorted(set(source_indexes)):
            raise ValueError("fresh English rows are not in unique source order")
    source_hashes = selection["source_hashes"]
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("selection source hashes must be a nonempty object")
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in source_hashes.values()
    ):
        raise ValueError("selection source hashes must be lowercase SHA-256 values")
    return normalized


def prepare_canary(
    root: Path,
    selection: Mapping[str, Any],
    *,
    runner_path: Path,
) -> dict[str, Any]:
    """Create and freeze one immutable label-free canary input bundle."""

    if root.exists():
        raise ValueError("refusing to overwrite existing canary root")
    rows = _validate_selection(selection)
    selection_kind = selection.get("selection_kind")
    is_fresh_english = selection_kind == FRESH_ENGLISH_SELECTION_KIND
    expected_runner = DEDICATED_RUNNER_PATH if is_fresh_english else RUNNER_PATH
    if runner_path.resolve() != expected_runner or not expected_runner.is_file():
        raise ValueError("runner source must be the exact mounted runner")
    if is_fresh_english and (
        not RUNNER_PATH.is_file() or not CONTROLLER_PATH.is_file()
    ):
        raise ValueError("helper and controller sources must exist")
    root.mkdir(parents=True)
    selection_value = dict(selection)
    condition_id = ENGLISH_CONDITION_ID if is_fresh_english else CONDITION_ID
    requests = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "requests": [
            {
                "run_id": EXPERIMENT_ID,
                "condition_id": condition_id,
                "video_id": row["video_id"],
                "audio_path": row["audio_path"],
                "mode": "transcription",
            }
            for row in rows
        ],
    }
    _walk_label_free(requests)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared",
        "condition_id": condition_id,
        "asr_model_id": ASR_MODEL_ID,
        "asr_revision": ASR_REVISION,
        "forced_aligner_model_id": ALIGNER_MODEL_ID,
        "forced_aligner_revision": ALIGNER_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "selection_kind": selection_value.get(
            "selection_kind", "two_row_operational_canary"
        ),
        "row_count": len(rows),
        "runtime_contract": {
            "accelerator": "rocm",
            "device": "cuda:0",
            "precision": "bfloat16",
            "attention": "sdpa",
            "network": "none",
            "batch_size": 1,
            "return_time_stamps": True,
        },
        "attempt_limit": 1,
        "paid_compute_usd": 0.0,
        "row_order": [row["video_id"] for row in rows],
        "audio_sha256": {row["video_id"]: row["audio_sha256"] for row in rows},
        "duration_seconds": {row["video_id"]: row["duration_seconds"] for row in rows},
        "runner_sha256": _sha256_file(expected_runner),
        "selection_sha256": canonical_sha256(selection_value),
        "requests_sha256": canonical_sha256(requests),
    }
    if is_fresh_english:
        protocol["runtime_contract"]["language"] = "English"
        protocol.update(
            {
                "runner_path": str(DEDICATED_RUNNER_PATH),
                "helper_path": str(RUNNER_PATH),
                "helper_sha256": _sha256_file(RUNNER_PATH),
                "controller_path": str(CONTROLLER_PATH),
                "controller_sha256": _sha256_file(CONTROLLER_PATH),
            }
        )
    _write_json_exclusive(root / "selection.json", selection_value)
    _write_json_exclusive(root / "requests.json", requests)
    _write_json_exclusive(root / "protocol.json", protocol)
    artifacts = {
        name: freeze_json_artifact(
            root / f"{name}.json",
            bindings={"experiment_id": EXPERIMENT_ID},
        )
        for name in ("selection", "requests", "protocol")
    }
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "inputs_frozen",
        "artifacts": artifacts,
        "artifacts_sha256": canonical_sha256(artifacts),
    }
    _write_json_exclusive(root / "inputs-freeze.json", freeze)
    return freeze


def _inference_argv(selection_kind: str) -> list[str]:
    if selection_kind == FRESH_ENGLISH_SELECTION_KIND:
        return [
            "/opt/venv/bin/python",
            "-m",
            "trec_video_2026.vqa_qwen_empty_output_runner",
            "--model-id",
            ASR_MODEL_ID,
            "--revision",
            ASR_REVISION,
            "--forced-aligner-id",
            ALIGNER_MODEL_ID,
            "--forced-aligner-revision",
            ALIGNER_REVISION,
            "--language",
            "English",
            "--timestamps",
            "on",
        ]
    return [
        "/opt/venv/bin/python",
        "-m",
        "trec_video_2026.vqa_audio_runner",
        "--kind",
        "qwen3_asr",
        "--model-id",
        ASR_MODEL_ID,
        "--revision",
        ASR_REVISION,
        "--forced-aligner-id",
        ALIGNER_MODEL_ID,
        "--forced-aligner-revision",
        ALIGNER_REVISION,
    ]


def _podman_prefix(*, hub_cache: Path, data_root: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    container_repo = Path("/workspace/trec_video_2026")
    return [
        "podman",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--read-only",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--group-add",
        "keep-groups",
        "--security-opt",
        "label=disable",
        "--tmpfs",
        "/tmp:rw,size=4g",
        "--volume",
        f"{hub_cache.resolve()}:/root/.cache/huggingface/hub:ro",
        "--volume",
        f"{data_root.resolve()}:{data_root.resolve()}:ro",
        "--volume",
        f"{repo_root}:{container_repo}:ro",
        "--workdir",
        str(container_repo),
        "--env",
        f"PYTHONPATH={container_repo / 'src'}",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "HF_HOME=/root/.cache/huggingface",
        "--env",
        "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub",
        "--env",
        "HOME=/tmp/home",
        "--env",
        f"VQA_RUNTIME_IMAGE={RUNTIME_IMAGE}",
    ]


def build_podman_command(
    root: Path,
    *,
    hub_cache: Path,
    data_root: Path,
) -> list[str]:
    """Build the sole network-disabled local inference command."""

    if not root.is_dir():
        raise ValueError("canary root is missing")
    if not hub_cache.is_dir():
        raise ValueError("Hugging Face cache is missing")
    resolved_data_root = data_root.resolve()
    if resolved_data_root != CANONICAL_DATA_ROOT or not resolved_data_root.is_dir():
        raise ValueError("canonical data root is missing or does not match")
    if not root.resolve().is_relative_to(resolved_data_root):
        raise ValueError("canary root must be inside the canonical data root")
    _verify_runner_binding(root)
    protocol = _read_object(root / "protocol.json")
    selection_kind = str(protocol.get("selection_kind"))
    return [
        *_podman_prefix(hub_cache=hub_cache, data_root=resolved_data_root),
        RUNTIME_IMAGE,
        *_inference_argv(selection_kind),
    ]


def _finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{field} must be finite and positive")
    return number


def validate_results(
    protocol: Mapping[str, Any],
    requests: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate terminal runner rows against the frozen operational contract."""

    request_rows = requests.get("requests")
    if not isinstance(request_rows, list) or not request_rows:
        raise ValueError("requests must contain at least one row")
    required_count = protocol.get("row_count", len(request_rows))
    if (
        isinstance(required_count, bool)
        or not isinstance(required_count, int)
        or required_count <= 0
        or len(request_rows) != required_count
    ):
        raise ValueError(f"requests must contain exactly {required_count} rows")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise ValueError("results must be an array")
    if len(results) != required_count:
        raise ValueError(f"results must contain exactly {required_count} rows")
    expected_ids = protocol.get("row_order")
    if not isinstance(expected_ids, list) or len(expected_ids) != required_count:
        raise ValueError("protocol row order is invalid")
    if [row.get("video_id") for row in request_rows] != expected_ids:
        raise ValueError("request order does not match protocol")
    audio_hashes = protocol.get("audio_sha256")
    durations = protocol.get("duration_seconds")
    if not isinstance(audio_hashes, Mapping) or not isinstance(durations, Mapping):
        raise ValueError("protocol audio bindings are invalid")

    completed = 0
    aligned = 0
    peak_values: list[int] = []
    elapsed_values: list[float] = []
    failure_types: list[str] = []
    for index, (request, result) in enumerate(zip(request_rows, results, strict=True)):
        if not isinstance(request, Mapping) or not isinstance(result, Mapping):
            raise ValueError("request and result rows must be objects")
        video_id = expected_ids[index]
        identities = {
            "run_id": EXPERIMENT_ID,
            "condition_id": protocol.get("condition_id"),
            "video_id": video_id,
            "mode": "transcription",
            "model_id": protocol.get("asr_model_id"),
            "model_revision": protocol.get("asr_revision"),
            "forced_aligner_model_id": protocol.get("forced_aligner_model_id"),
            "forced_aligner_revision": protocol.get("forced_aligner_revision"),
            "runtime_image": protocol.get("runtime_image"),
        }
        for field, expected in identities.items():
            if result.get(field) != expected:
                raise ValueError(f"result {field} does not match frozen protocol")
        status = result.get("status")
        if status != "passed":
            failure_types.append(str(result.get("error_type") or status or "unknown"))
            continue
        completed += 1
        if result.get("source_audio_sha256") != audio_hashes.get(video_id):
            raise ValueError("result source audio hash does not match protocol")
        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("passed result transcript must be nonempty")
        language = result.get("language")
        if not isinstance(language, str) or language.casefold() != "english":
            raise ValueError("passed result language must be English")
        if result.get("device") != "cuda:0 (ROCm)":
            raise ValueError("passed result must record ROCm cuda:0")
        if result.get("precision") != "bfloat16":
            raise ValueError("passed result must record bfloat16")
        runtime = result.get("runtime")
        if not isinstance(runtime, str) or "torch" not in runtime or "transformers" not in runtime:
            raise ValueError("passed result runtime metadata is invalid")
        elapsed_values.append(
            _finite_number(result.get("elapsed_seconds"), "elapsed_seconds", positive=True)
        )
        _finite_number(result.get("model_load_seconds"), "model_load_seconds", positive=True)
        peak = _finite_number(
            result.get("peak_memory_bytes"), "peak_memory_bytes", positive=True
        )
        peak_values.append(int(peak))
        chunks = result.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("passed result must contain aligned timestamp chunks")
        duration = _finite_number(durations.get(video_id), "duration_seconds", positive=True)
        previous_end = 0.0
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise ValueError("timestamp chunks must be objects")
            start = _finite_number(chunk.get("start"), "timestamp start")
            end = _finite_number(chunk.get("end"), "timestamp end")
            chunk_text = chunk.get("text")
            if (
                start < 0
                or end < start
                or end > duration
                or start < previous_end
                or not isinstance(chunk_text, str)
                or not chunk_text.strip()
            ):
                raise ValueError("timestamp chunk is invalid or exceeds source duration")
            previous_end = end
        aligned += 1

    decision = (
        "ready_for_quality_test"
        if completed == aligned == required_count
        else "not_ready"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": decision,
        "completion_count": completed,
        "required_count": required_count,
        "valid_alignment_count": aligned,
        "failure_count": required_count - completed,
        "failure_types": failure_types,
        "paid_compute_usd": 0.0,
        "results_sha256": canonical_sha256(list(results)),
        "max_peak_memory_bytes": max(peak_values, default=0),
        "total_elapsed_seconds": sum(elapsed_values),
    }


def _freeze_bytes(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot freeze artifact: {path.name}") from exc
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _verify_frozen_bytes(path: Path, record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != SCHEMA_VERSION or record.get("status") != "frozen":
        raise ValueError("byte artifact freeze record is invalid")
    if record.get("artifact_sha256") != _sha256_file(path):
        raise ValueError(f"artifact hash does not match freeze: {path.name}")


def _validate_record_command(root: Path, command: Sequence[str]) -> list[str]:
    normalized = list(command)
    if not normalized or any(not isinstance(value, str) for value in normalized):
        raise ValueError("record command must be a nonempty string array")
    if normalized[:2] != ["podman", "run"]:
        raise ValueError("record command must invoke podman run")
    if "-i" not in normalized and "--interactive" not in normalized:
        raise ValueError("podman interactive stdin must be attached")
    image_index = normalized.index(RUNTIME_IMAGE) if RUNTIME_IMAGE in normalized else -1
    interactive_indexes = [
        normalized.index(flag)
        for flag in ("-i", "--interactive")
        if flag in normalized
    ]
    if image_index < 0 or min(interactive_indexes) > image_index:
        raise ValueError("pinned image and interactive stdin order are invalid")
    required = {
        "--read-only",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HOME=/root/.cache/huggingface",
        "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub",
        ASR_REVISION,
        ALIGNER_REVISION,
    }
    if not required.issubset(normalized):
        raise ValueError("record command does not match the offline frozen contract")
    try:
        network_index = normalized.index("--network")
    except ValueError as exc:
        raise ValueError("record command must disable the network") from exc
    if network_index + 1 >= len(normalized) or normalized[network_index + 1] != "none":
        raise ValueError("record command must disable the network")
    rendered = " ".join(normalized).casefold()
    if "hf_token" in rendered or "hugging_face_hub_token" in rendered:
        raise ValueError("record command must not contain credentials")
    protocol = _read_object(root / "protocol.json")
    if protocol.get("selection_kind") == FRESH_ENGLISH_SELECTION_KIND:
        cache_suffix = ":/root/.cache/huggingface/hub:ro"
        cache_mounts = [
            value
            for value in normalized
            if isinstance(value, str) and value.endswith(cache_suffix)
        ]
        if len(cache_mounts) != 1:
            raise ValueError("fresh English command cache mount is invalid")
        cache_source = cache_mounts[0][: -len(cache_suffix)]
        cache_path = Path(cache_source)
        if not cache_path.is_absolute() or not cache_path.is_dir():
            raise ValueError("fresh English command cache source is invalid")
        expected = [
            *_podman_prefix(
                hub_cache=cache_path,
                data_root=CANONICAL_DATA_ROOT,
            ),
            RUNTIME_IMAGE,
            *_inference_argv(FRESH_ENGLISH_SELECTION_KIND),
        ]
        if normalized != expected:
            raise ValueError("record command does not match the exact forced-English contract")
    return normalized


def _verify_runner_binding(root: Path) -> None:
    protocol = _read_object(root / "protocol.json")
    if protocol.get("selection_kind") == FRESH_ENGLISH_SELECTION_KIND:
        bindings = (
            ("runner", DEDICATED_RUNNER_PATH),
            ("helper", RUNNER_PATH),
            ("controller", CONTROLLER_PATH),
        )
        for label, path in bindings:
            if protocol.get(f"{label}_path") != str(path):
                raise ValueError(f"mounted {label} path does not match protocol")
            try:
                actual_hash = _sha256_file(path)
            except OSError as exc:
                raise ValueError(f"mounted {label} source is missing") from exc
            if protocol.get(f"{label}_sha256") != actual_hash:
                raise ValueError(f"mounted {label} source hash does not match protocol")
        return
    expected = protocol.get("runner_sha256")
    if not isinstance(expected, str) or expected != _sha256_file(RUNNER_PATH):
        raise ValueError("mounted runner source hash does not match protocol")


def _sanitize_stderr(value: str, requests: Mapping[str, Any]) -> str:
    sanitized = value[:1_000_000]
    rows = requests.get("requests")
    if isinstance(rows, list):
        for row in rows:
            audio_path = row.get("audio_path") if isinstance(row, Mapping) else None
            if isinstance(audio_path, str) and audio_path:
                sanitized = sanitized.replace(audio_path, "<private-audio-path>")
    token_patterns = (
        r"hf_[A-Za-z0-9]{16,}",
        r"Bearer\s+[A-Za-z0-9._~+/-]{16,}",
        r"AIza[0-9A-Za-z_-]{20,}",
        r"sk-[A-Za-z0-9_-]{20,}",
    )
    for pattern in token_patterns:
        sanitized = re.sub(pattern, "<redacted-secret>", sanitized)
    return sanitized


def record_attempt(
    root: Path,
    command: Sequence[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Execute and preserve the sole terminal inference attempt."""

    if status_canary(root)["status"] != "inputs_frozen":
        raise ValueError("canary inputs are not frozen or an attempt already exists")
    normalized_command = _validate_record_command(root, command)
    output_names = (
        "command.json",
        "raw-results.json",
        "stderr.txt",
        "terminal.json",
        "validated-results.json",
    )
    if any((root / name).exists() for name in output_names):
        raise ValueError("refusing a second attempt or output overwrite")
    requests = _read_object(root / "requests.json")
    protocol = _read_object(root / "protocol.json")
    request_rows = requests.get("requests")
    required_count = protocol.get("row_count")
    if (
        isinstance(required_count, bool)
        or not isinstance(required_count, int)
        or required_count <= 0
    ):
        required_count = (
            len(request_rows) if isinstance(request_rows, list) and request_rows else 2
        )
    _verify_runner_binding(root)
    payload = json.dumps(requests["requests"], ensure_ascii=False, separators=(",", ":"))
    command_hash = canonical_sha256(normalized_command)
    _write_json_exclusive(
        root / "command.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "command_frozen",
            "argv": normalized_command,
            "command_sha256": command_hash,
        },
    )
    started = monotonic()
    launch_error_type = None
    try:
        completed = runner(
            normalized_command,
            input=payload,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - preserve launch failures terminally
        launch_error_type = type(exc).__name__
        completed = subprocess.CompletedProcess(
            normalized_command,
            127,
            stdout="",
            stderr=f"{launch_error_type}: subprocess launch failed\n",
        )
    wall_seconds = max(0.0, monotonic() - started)
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    raw_stderr = completed.stderr if isinstance(completed.stderr, str) else ""
    stderr = _sanitize_stderr(raw_stderr, requests)
    _write_text_exclusive(root / "raw-results.json", stdout)
    _write_text_exclusive(root / "stderr.txt", stderr)
    try:
        parsed = json.loads(stdout)
        if not isinstance(parsed, list):
            raise ValueError("runner output must be an array")
        validation = validate_results(protocol, requests, parsed)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        validation = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "not_ready",
            "completion_count": 0,
            "required_count": required_count,
            "valid_alignment_count": 0,
            "failure_count": required_count,
            "failure_types": [type(exc).__name__],
            "paid_compute_usd": 0.0,
            "results_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "max_peak_memory_bytes": 0,
            "total_elapsed_seconds": 0.0,
        }
    if completed.returncode != 0:
        validation["status"] = "not_ready"
        failure_types = [] if launch_error_type else validation.get("failure_types")
        if not isinstance(failure_types, list):
            failure_types = []
        terminal_failure = launch_error_type or "NonzeroExitCode"
        if terminal_failure not in failure_types:
            failure_types.append(terminal_failure)
        validation["failure_types"] = failure_types
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed" if completed.returncode == 0 else "failed",
        "attempt_index": 1,
        "exit_code": int(completed.returncode),
        "wall_seconds": wall_seconds,
        "command_sha256": command_hash,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "paid_compute_usd": 0.0,
    }
    _write_json_exclusive(root / "validated-results.json", validation)
    _write_json_exclusive(root / "terminal.json", terminal)
    return terminal


def freeze_predictions(root: Path) -> dict[str, Any]:
    """Freeze raw output and its terminal validation without interpreting labels."""

    if status_canary(root)["status"] != "inputs_frozen":
        raise ValueError("inputs must verify before prediction freeze")
    if (root / "predictions-freeze.json").exists():
        raise ValueError("refusing to overwrite prediction freeze")
    validation = _read_object(root / "validated-results.json")
    terminal = _read_object(root / "terminal.json")
    artifacts = {
        "raw-results": _freeze_bytes(root / "raw-results.json"),
        "stderr": _freeze_bytes(root / "stderr.txt"),
        "terminal": freeze_json_artifact(
            root / "terminal.json", bindings={"experiment_id": EXPERIMENT_ID}
        ),
        "validated-results": freeze_json_artifact(
            root / "validated-results.json",
            bindings={"experiment_id": EXPERIMENT_ID},
        ),
    }
    if (root / "command.json").exists():
        artifacts["command"] = freeze_json_artifact(
            root / "command.json", bindings={"experiment_id": EXPERIMENT_ID}
        )
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "predictions_frozen",
        "decision": validation.get("status"),
        "terminal_status": terminal.get("status"),
        "input_freeze_sha256": _sha256_file(root / "inputs-freeze.json"),
        "artifacts": artifacts,
        "artifacts_sha256": canonical_sha256(artifacts),
    }
    _write_json_exclusive(root / "predictions-freeze.json", freeze)
    return freeze


def _verify_prediction_freeze(root: Path) -> dict[str, Any]:
    freeze = _read_object(root / "predictions-freeze.json")
    if freeze.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("prediction freeze experiment ID is invalid")
    if freeze.get("input_freeze_sha256") != _sha256_file(root / "inputs-freeze.json"):
        raise ValueError("prediction freeze input hash is invalid")
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping) or freeze.get("artifacts_sha256") != canonical_sha256(artifacts):
        raise ValueError("prediction freeze artifact hash is invalid")
    expected = {"raw-results", "stderr", "terminal", "validated-results"}
    if "command" in artifacts:
        expected.add("command")
    if set(artifacts) != expected:
        raise ValueError("prediction freeze artifacts are invalid")
    _verify_frozen_bytes(root / "raw-results.json", artifacts["raw-results"])
    _verify_frozen_bytes(root / "stderr.txt", artifacts["stderr"])
    verify_json_artifact(root / "terminal.json", dict(artifacts["terminal"]))
    verify_json_artifact(
        root / "validated-results.json", dict(artifacts["validated-results"])
    )
    if "command" in artifacts:
        verify_json_artifact(root / "command.json", dict(artifacts["command"]))
    return freeze


def require_quality_scoring_ready(root: Path) -> dict[str, Any]:
    """Require a verified, all-row-ready prediction freeze before scoring."""

    freeze = _verify_prediction_freeze(root)
    validation = _read_object(root / "validated-results.json")
    if freeze.get("decision") != validation.get("status"):
        raise ValueError("prediction decision does not match validation")
    if freeze.get("decision") != "ready_for_quality_test":
        raise ValueError("predictions are not ready for quality scoring")
    protocol = _read_object(root / "protocol.json")
    terminal = _read_object(root / "terminal.json")
    required_count = protocol.get("row_count")
    if (
        terminal.get("status") != "completed"
        or validation.get("required_count") != required_count
        or validation.get("completion_count") != required_count
        or validation.get("valid_alignment_count") != required_count
    ):
        raise ValueError("all frozen rows must complete with valid alignment")
    if (
        protocol.get("selection_kind") == FRESH_ENGLISH_SELECTION_KIND
        and required_count != FRESH_PANEL_SIZE
    ):
        raise ValueError("fresh English quality scoring requires exactly 12 rows")
    return freeze


def _normalized_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def score_postfreeze(
    root: Path,
    *,
    references: Mapping[str, str],
    whisper: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score lexical/timestamp diagnostics only after verifying prediction bytes."""

    try:
        prediction_freeze = require_quality_scoring_ready(root)
    except ValueError as exc:
        if str(exc) in {
            "prediction decision does not match validation",
            "predictions are not ready for quality scoring",
        }:
            raise
        raise ValueError("prediction freeze must verify before scoring") from exc
    if (root / "postfreeze-score.json").exists():
        raise ValueError("refusing to overwrite postfreeze score")
    protocol = _read_object(root / "protocol.json")
    fresh_english = protocol.get("selection_kind") == FRESH_ENGLISH_SELECTION_KIND
    results = json.loads((root / "raw-results.json").read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise ValueError("raw results are not scoreable")
    rows = []
    for result in results:
        video_id = result.get("video_id") if isinstance(result, Mapping) else None
        if not isinstance(video_id, str) or video_id not in references:
            raise ValueError("score inputs do not cover frozen result IDs")
        reference = references[video_id]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("official phrase must be nonempty")
        transcript = result.get("text")
        phrase = _normalized_phrase(reference)
        normalized_text = _normalized_phrase(transcript) if isinstance(transcript, str) else ""
        present = bool(phrase) and phrase in normalized_text
        comparison = whisper.get(video_id, {})
        whisper_text = comparison.get("text")
        whisper_quality_status = comparison.get("quality_status")
        normalized_whisper = (
            _normalized_phrase(whisper_text)
            if isinstance(whisper_text, str)
            and (
                not fresh_english
                or not (
                    isinstance(whisper_quality_status, str)
                    and whisper_quality_status.casefold().startswith("rejected")
                )
            )
            else ""
        )
        whisper_present = bool(phrase) and phrase in normalized_whisper
        rows.append(
            {
                "video_id": video_id,
                "official_phrase_present": present,
                "whisper_official_phrase_present": whisper_present,
                "qwen_quality_status": (
                    result.get("quality", {}).get("status")
                    if isinstance(result.get("quality"), Mapping)
                    else None
                ),
                "whisper_quality_status": whisper_quality_status,
                "valid_alignment": bool(result.get("chunks")),
            }
        )
    qwen_count = sum(bool(row["official_phrase_present"]) for row in rows)
    whisper_count = sum(
        bool(row["whisper_official_phrase_present"]) for row in rows
    )
    both_count = sum(
        bool(row["official_phrase_present"])
        and bool(row["whisper_official_phrase_present"])
        for row in rows
    )
    qwen_only_count = sum(
        bool(row["official_phrase_present"])
        and not bool(row["whisper_official_phrase_present"])
        for row in rows
    )
    whisper_only_count = sum(
        not bool(row["official_phrase_present"])
        and bool(row["whisper_official_phrase_present"])
        for row in rows
    )
    neither_count = len(rows) - both_count - qwen_only_count - whisper_only_count
    directional_decision = (
        "directional_qwen_signal"
        if qwen_count > whisper_count
        else "whisper_directional_signal"
        if qwen_count < whisper_count
        else "no_directional_difference"
    )
    score = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "scored_postfreeze",
        "prediction_freeze_sha256": _sha256_file(root / "predictions-freeze.json"),
        "operational_decision": prediction_freeze.get("decision"),
        "official_phrase_present_count": qwen_count,
        "whisper_official_phrase_present_count": whisper_count,
        "both_present_count": both_count,
        "qwen_only_present_count": qwen_only_count,
        "whisper_only_present_count": whisper_only_count,
        "neither_present_count": neither_count,
        "directional_decision": directional_decision,
        "rows": rows,
    }
    _write_json_exclusive(root / "postfreeze-score.json", score)
    score_record = freeze_json_artifact(
        root / "postfreeze-score.json", bindings={"experiment_id": EXPERIMENT_ID}
    )
    score_freeze = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "scored_postfreeze",
        "artifact": score_record,
    }
    _write_json_exclusive(root / "postfreeze-score-freeze.json", score_freeze)
    return score


def status_canary(root: Path) -> dict[str, Any]:
    """Verify immutable inputs and return the current canary phase."""

    freeze = _read_object(root / "inputs-freeze.json")
    if freeze.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("input freeze schema version is invalid")
    if freeze.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("input freeze experiment ID is invalid")
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "selection",
        "requests",
        "protocol",
    }:
        raise ValueError("input freeze artifacts are invalid")
    if freeze.get("artifacts_sha256") != canonical_sha256(artifacts):
        raise ValueError("input freeze artifact hash is invalid")
    for name, record in artifacts.items():
        verify_json_artifact(root / f"{name}.json", dict(record))
    _verify_runner_binding(root)
    status = str(freeze.get("status"))
    decision = None
    if (root / "predictions-freeze.json").exists():
        prediction_freeze = _verify_prediction_freeze(root)
        status = str(prediction_freeze.get("status"))
        decision = prediction_freeze.get("decision")
    if (root / "postfreeze-score-freeze.json").exists():
        score_freeze = _read_object(root / "postfreeze-score-freeze.json")
        if score_freeze.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("postfreeze score experiment ID is invalid")
        record = score_freeze.get("artifact")
        if not isinstance(record, Mapping):
            raise ValueError("postfreeze score freeze is invalid")
        verify_json_artifact(root / "postfreeze-score.json", dict(record))
        status = str(score_freeze.get("status"))
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "decision": decision,
        "paid_compute_usd": 0.0,
    }


__all__ = [
    "DEDICATED_RUNNER_PATH",
    "EXPERIMENT_ID",
    "FRESH_ENGLISH_SELECTION_KIND",
    "FRESH_PANEL_SIZE",
    "FRESH_STRATUM_QUOTAS",
    "GEMINI_CONDITION",
    "QUALITY_SELECTION_KIND",
    "SCHEMA_VERSION",
    "build_podman_command",
    "build_qwen_asr_exposure_index",
    "configure_experiment_id",
    "freeze_predictions",
    "prepare_canary",
    "record_attempt",
    "require_quality_scoring_ready",
    "score_postfreeze",
    "select_fresh_english_quality_panel",
    "select_quality_panel",
    "select_panel",
    "status_canary",
    "validate_results",
]
