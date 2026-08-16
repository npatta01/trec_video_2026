#!/usr/bin/env python3
"""Prepare the treatment-first task-support sufficiency audit.

The preparation boundary is deliberately separate from the provider runner.
It freezes the treatment and all source bytes first, then creates an
all-row, label-free candidate sheet.  Only the later manual-audit freeze may
join references and prior answers, and those fields are never copied into the
provider protocol.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_task_support_verifier.py"
MODEL = "gemini-3.1-pro-preview"
EXPECTED_SETTINGS: dict[str, Any] = {
    "model": MODEL,
    "temperature": 0.0,
    "seed": 42,
    "thinking_level": "HIGH",
    "candidate_count": 1,
    "max_output_tokens": 2048,
    "retry_count": 0,
}
ROLES = (
    "answer_sufficient_failure",
    "insufficient_or_ambiguous",
    "supported_control",
)
NONSELECTABLE_ROLE = "reference_or_question_issue"
AUDIT_ROLES = (*ROLES, NONSELECTABLE_ROLE)
ALLOWED_QUEUE_STRATA = {"temporal", "audio_or_text"}
LOW_SCORE_MAX = 0.45
CONTROL_SCORE_MIN = 0.70
PROMOTION_GATE_DEFAULT: dict[str, Any] = {
    "target_rescues": {"minimum": 3, "denominator": 4},
    "safeguard_verifications": {"required": 4, "denominator": 4},
    "controls_preserved": {"required": 4, "denominator": 4},
    "unsupported_answers": {"maximum": 0},
    "completed_rows": {"required": 12},
    "changed_rows_inspected": "all",
}
LOW_DEFAULT = 24
CONTROL_DEFAULT = 8
MINIMUM_PER_ROLE = 4


def _sha256_path(path: Path) -> str:
    """Hash a file or a deterministic relative-path/file-hash directory."""

    path = path.resolve()
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        files = sorted(
            (item.relative_to(path) for item in path.rglob("*") if item.is_file()),
            key=lambda item: item.as_posix(),
        )
        for relative in files:
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_path(path / relative).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    raise ValueError(f"source path does not exist: {path}")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error


def _write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite immutable artifact: {path}") from error


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _resolve_path(value: Any, *, base: Path, field: str) -> Path:
    if isinstance(value, Mapping):
        value = value.get("path", value.get("file"))
    path = Path(_nonempty(value, field))
    if not path.is_absolute():
        # Configs commonly use repository-relative paths, while test fixtures
        # use paths relative to their temporary config directory.
        candidate = (base / path).resolve()
        path = candidate if candidate.exists() else (ROOT / path).resolve()
    return path


def _config_path(config_path: Path | str | None, config: Any) -> Path:
    if config_path is not None:
        return Path(config_path).resolve()
    if isinstance(config, (str, Path)):
        return Path(config).resolve()
    raise ValueError("config path is required")


def _context(
    config_path: Path | str | None = None,
    output: Path | str | None = None,
    *,
    config: Any = None,
    output_root: Path | str | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], Path, Path]:
    del kwargs
    path = _config_path(config_path, config)
    if isinstance(config, Mapping):
        payload = dict(config)
    else:
        payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("config must be a JSON object")
    root = Path(output_root or output) if (output_root or output) else None
    if root is None:
        raise ValueError("output root is required")
    return dict(payload), path, root.resolve()


def _source_specs(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config.get(
        "source_artifacts",
        config.get("sources", config.get("source_paths")),
    )
    if source is None:
        source = {
            key: config[key]
            for key in (
                "protocol",
                "maps",
                "predictions",
                "scores",
                "annotations",
                "full_panel_protocol",
                "full_panel_maps",
                "full_panel_predictions",
                "full_panel_score",
            )
            if key in config
        }
    if not isinstance(source, Mapping):
        raise ValueError("config source_artifacts must be an object")
    return {str(key): value for key, value in source.items()}


def _exposure_specs(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config.get(
        "exposure_artifacts", config.get("exposures", config.get("exposure_sources", {}))
    )
    if isinstance(source, list):
        return {f"exposure_{index:02d}": value for index, value in enumerate(source)}
    if not isinstance(source, Mapping):
        raise ValueError("config exposure_artifacts must be an object")
    return {str(key): value for key, value in source.items()}


def _expected_hash(spec: Any) -> str | None:
    if isinstance(spec, Mapping):
        value = spec.get("sha256", spec.get("hash"))
        return value if isinstance(value, str) else None
    return None


def _source_paths(config: Mapping[str, Any], config_file: Path) -> dict[str, Path]:
    specs = _source_specs(config)
    return {
        name: _resolve_path(spec, base=config_file.parent, field=f"source_artifacts.{name}")
        for name, spec in specs.items()
    }


def _exposure_paths(config: Mapping[str, Any], config_file: Path) -> dict[str, Path]:
    specs = _exposure_specs(config)
    return {
        name: _resolve_path(spec, base=config_file.parent, field=f"exposure_artifacts.{name}")
        for name, spec in specs.items()
    }


def _verify_declared_hashes(
    config: Mapping[str, Any], config_file: Path, *, include_exposures: bool = True
) -> dict[str, str]:
    paths = _source_paths(config, config_file)
    if include_exposures:
        paths.update({f"exposure:{name}": path for name, path in _exposure_paths(config, config_file).items()})
    specs: dict[str, Any] = {}
    specs.update({name: spec for name, spec in _source_specs(config).items()})
    if include_exposures:
        specs.update({f"exposure:{name}": spec for name, spec in _exposure_specs(config).items()})
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        digest = _sha256_path(path)
        expected = _expected_hash(specs.get(name))
        if expected is not None and digest != expected:
            raise ValueError(f"source hash mismatch for {name}: expected {expected}, got {digest}")
        hashes[name] = digest
    return hashes


def _load_runner_module(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"runner path does not exist: {path}")
    if path.resolve() != RUNNER.resolve():
        return None
    spec = importlib.util.spec_from_file_location("task_support_verifier", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _treatment_hashes(config: Mapping[str, Any], config_file: Path) -> dict[str, str]:
    runner_value = config.get(
        "runner_path", config.get("runner", config.get("runner_script"))
    )
    runner_path = _resolve_path(runner_value or RUNNER, base=config_file.parent, field="runner_path")
    runner_hash = _sha256_path(runner_path)
    expected_runner = config.get("runner_sha256")
    if expected_runner is not None and runner_hash != expected_runner:
        raise ValueError("runner hash does not match config")
    module = _load_runner_module(runner_path)
    prompt_hash = config.get("prompt_template_sha256")
    schema_hash = config.get("response_schema_sha256", config.get("schema_sha256"))
    validator_hash = config.get("validator_sha256")
    if module is not None:
        prompt_hash = prompt_hash or module._prompt_template_sha256()
        schema_hash = schema_hash or module._schema_sha256()
        validator_hash = validator_hash or _sha256_validator(module)
    prompt_hash = _nonempty(prompt_hash or _canonical_hash("task-support-prompt-v1"), "prompt_template_sha256")
    schema_hash = _nonempty(schema_hash or _canonical_hash("task-support-schema-v1"), "response_schema_sha256")
    validator_hash = _nonempty(validator_hash or runner_hash, "validator_sha256")
    return {
        "runner_sha256": runner_hash,
        "prompt_template_sha256": prompt_hash,
        "response_schema_sha256": schema_hash,
        "validator_sha256": validator_hash,
    }


def _sha256_validator(module: Any) -> str:
    source = Path(module.__file__).read_text(encoding="utf-8")
    start = source.index("def validate_answer(")
    end = source.index("\ndef _index_protocol_rows", start)
    return hashlib.sha256(source[start:end].encode("utf-8")).hexdigest()


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = config.get("settings", EXPECTED_SETTINGS)
    if settings != EXPECTED_SETTINGS:
        raise ValueError("config settings do not match frozen settings")
    return dict(EXPECTED_SETTINGS)


def _gate(config: Mapping[str, Any]) -> dict[str, Any]:
    configured = config.get("gate", {})
    if not isinstance(configured, Mapping):
        raise ValueError("gate must be an object")
    return {
        "low_audit_count": int(
            configured.get("low_audit_count", configured.get("low_queue_count", LOW_DEFAULT))
        ),
        "control_audit_count": int(
            configured.get("control_audit_count", configured.get("control_queue_count", CONTROL_DEFAULT))
        ),
        "minimum_per_role": int(configured.get("minimum_per_role", MINIMUM_PER_ROLE)),
        "all_rows_required": True,
        "exposure_is_flag_only": True,
    }


def _promotion_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    configured = config.get("promotion_gate", PROMOTION_GATE_DEFAULT)
    if not isinstance(configured, Mapping):
        raise ValueError("promotion_gate must be an object")
    # Keep a complete, deterministic copy in treatment-freeze.  Requiring
    # exact keys here prevents an omitted safety condition from silently
    # weakening the provider promotion boundary.
    expected = set(PROMOTION_GATE_DEFAULT)
    if set(configured) != expected:
        raise ValueError("promotion_gate fields do not match the frozen gate")
    normalized = json.loads(json.dumps(configured))
    for key in ("target_rescues", "safeguard_verifications", "controls_preserved"):
        if normalized.get(key) != PROMOTION_GATE_DEFAULT[key]:
            raise ValueError(f"promotion_gate.{key} does not match the frozen gate")
    if normalized.get("unsupported_answers") != PROMOTION_GATE_DEFAULT["unsupported_answers"]:
        raise ValueError("promotion_gate.unsupported_answers does not match the frozen gate")
    if normalized.get("completed_rows") != PROMOTION_GATE_DEFAULT["completed_rows"]:
        raise ValueError("promotion_gate.completed_rows does not match the frozen gate")
    if normalized.get("changed_rows_inspected") != "all":
        raise ValueError("promotion_gate.changed_rows_inspected does not match the frozen gate")
    return normalized


def _load_protocol(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    raw_rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_rows, list):
        raise ValueError("full-panel protocol rows must be a list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_rows):
        if not isinstance(item, Mapping):
            raise ValueError(f"protocol row {index} must be an object")
        video_id = _nonempty(item.get("video_id"), f"protocol row {index}.video_id")
        if video_id in seen:
            raise ValueError(f"duplicate video_id in protocol: {video_id}")
        seen.add(video_id)
        question = _nonempty(item.get("question"), f"protocol row {video_id}.question")
        rows.append(
            {
                "video_id": video_id,
                "question": question,
                "question_stratum": item.get("question_stratum", item.get("stratum", "unknown")),
                "protocol": dict(item),
            }
        )
    if not rows:
        raise ValueError("full-panel protocol is empty")
    return rows


def _index_records(payload: Any, *, field: str) -> dict[str, dict[str, Any]]:
    if isinstance(payload, Mapping):
        raw = payload.get("rows", payload.get("examples", payload.get("records")))
        if raw is None and all(isinstance(key, str) for key in payload):
            raw = [dict(value, video_id=key) if isinstance(value, Mapping) else value for key, value in payload.items()]
    else:
        raw = payload
    if not isinstance(raw, list):
        raise ValueError(f"{field} must contain a list of rows")
    indexed: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field} row {index} must be an object")
        video_id = _nonempty(item.get("video_id"), f"{field} row {index}.video_id")
        if video_id in indexed:
            raise ValueError(f"duplicate video_id in {field}: {video_id}")
        indexed[video_id] = dict(item)
    return indexed


def _load_scores(path: Path, *, condition: str) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    records: list[Mapping[str, Any]] = []
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), Mapping):
        values = payload["rows"].get(condition)
        if not isinstance(values, list):
            raise ValueError(f"score artifact has no rows for condition: {condition}")
        records.extend(item for item in values if isinstance(item, Mapping))
    else:
        raw = payload.get("rows", payload) if isinstance(payload, Mapping) else payload
        if isinstance(raw, list):
            records = [item for item in raw if isinstance(item, Mapping)]
    result: dict[str, dict[str, Any]] = {}
    for item in records:
        video_id = item.get("video_id")
        if not isinstance(video_id, str):
            continue
        value = item.get("semantic_composite", item.get("score", item.get("meteor")))
        if isinstance(value, (int, float)):
            result[video_id] = {
                "semantic_composite": float(value),
                "question_stratum": item.get("question_stratum"),
            }
    return result


def _annotation_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_dir():
        raise ValueError(f"annotations path must be a directory: {path}")
    result: dict[str, dict[str, Any]] = {}
    for child in sorted(path.glob("*.json")):
        payload = _load_json(child)
        if not isinstance(payload, Mapping):
            raise ValueError(f"annotation is not an object: {child}")
        video_id = str(payload.get("video_id") or child.stem)
        if video_id in result:
            raise ValueError(f"duplicate annotation video_id: {video_id}")
        result[video_id] = {**dict(payload), "_path": str(child), "_sha256": _sha256_path(child)}
    return result


def _map_lines(record: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    value = record.get("map", record.get("text", record.get("lines")))
    if isinstance(value, str):
        text = value
        lines = [{"line_number": index, "claim": line} for index, line in enumerate(text.splitlines(), 1)]
    elif isinstance(value, list):
        lines = []
        for index, item in enumerate(value, 1):
            if isinstance(item, Mapping):
                line_number = item.get("line_number", item.get("line", index))
                claim = item.get("claim", item.get("text", item.get("content", "")))
            else:
                line_number, claim = index, item
            if not isinstance(line_number, int) or not isinstance(claim, str):
                raise ValueError("map lines must contain integer line numbers and text")
            lines.append({"line_number": line_number, "claim": claim})
        text = "\n".join(item["claim"] for item in lines)
    else:
        raise ValueError("map record must contain map text or lines")
    if not lines or not any(str(item["claim"]).strip() for item in lines):
        raise ValueError("map contains no non-empty lines")
    return text, lines


def _map_records(path: Path) -> dict[str, dict[str, Any]]:
    return _index_records(_load_json(path), field="maps")


def _prediction_records(path: Path) -> dict[str, dict[str, Any]]:
    return _index_records(_load_json(path), field="predictions")


def _extract_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        if isinstance(value.get("video_id"), str):
            result.add(value["video_id"])
        for key, child in value.items():
            if key in {"video_ids", "ids", "selected_ids"} and isinstance(child, list):
                result.update(item for item in child if isinstance(item, str))
                continue
            if (
                isinstance(child, list)
                and child
                and all(isinstance(item, str) for item in child)
                and key in {"panel", "rows", "examples", "selected_rows", "target", "control"}
            ):
                result.update(child)
                continue
            if key not in {"correct_answer", "reference", "answer", "question"}:
                result.update(_extract_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_extract_ids(child))
    return result


def _exposure_ids(config: Mapping[str, Any], config_file: Path) -> dict[str, set[str]]:
    specs = _exposure_specs(config)
    result: dict[str, set[str]] = {}
    for name, spec in specs.items():
        path = _resolve_path(spec, base=config_file.parent, field=f"exposure_artifacts.{name}")
        explicit = spec.get("video_ids") if isinstance(spec, Mapping) else None
        ids = set(item for item in explicit if isinstance(item, str)) if isinstance(explicit, list) else set()
        ids.update(_extract_ids(_load_json(path)))
        result[name] = ids
    return result


def _validate_treatment(
    root: Path, config: Mapping[str, Any], config_file: Path, *, source_hashes: dict[str, str] | None = None
) -> dict[str, Any]:
    path = root / "treatment-freeze.json"
    if not path.is_file():
        raise ValueError("treatment freeze must exist before candidate preparation")
    payload = _load_json(path)
    if not isinstance(payload, Mapping) or payload.get("status") != "frozen":
        raise ValueError("treatment freeze must have exact status=frozen")
    expected_sources = source_hashes or _verify_declared_hashes(config, config_file)
    actual_sources = payload.get("source_artifacts_sha256")
    if actual_sources != expected_sources:
        raise ValueError("source hash drift or treatment freeze tampering detected")
    treatment_hashes = _treatment_hashes(config, config_file)
    for key, value in treatment_hashes.items():
        if payload.get(key) != value:
            label = key.replace("_sha256", " hash")
            raise ValueError(f"treatment freeze {label} mismatch")
    if payload.get("settings") != _settings(config):
        raise ValueError("treatment freeze settings mismatch")
    if payload.get("gate") != _gate(config):
        raise ValueError("treatment freeze gate mismatch")
    if payload.get("promotion_gate") != _promotion_gate(config):
        raise ValueError("treatment freeze promotion gate mismatch")
    if payload.get("config_sha256") != _sha256_path(config_file):
        raise ValueError("treatment freeze config hash mismatch")
    return dict(payload)


def freeze_treatment(
    config_path: Path | str | None = None,
    output: Path | str | None = None,
    *,
    config: Any = None,
    output_root: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Freeze model treatment, runner bytes, and all declared source hashes."""

    cfg, cfg_path, root = _context(config_path, output, config=config, output_root=output_root, **kwargs)
    source_paths = _source_paths(cfg, cfg_path)
    protocol_path = next(
        (source_paths[key] for key in ("protocol", "full_panel_protocol") if key in source_paths),
        None,
    )
    if protocol_path is None:
        raise ValueError("source_artifacts must declare protocol")
    _load_protocol(protocol_path)
    hashes = _verify_declared_hashes(cfg, cfg_path)
    treatment_hashes = _treatment_hashes(cfg, cfg_path)
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "experiment_id": cfg.get("experiment_id", "task-support-sufficiency-audit-issue35"),
        "frozen_at": datetime.now(UTC).isoformat(),
        **treatment_hashes,
        "model": MODEL,
        "settings": _settings(cfg),
        "gate": _gate(cfg),
        "promotion_gate": _promotion_gate(cfg),
        "config_sha256": _sha256_path(cfg_path),
        "source_artifacts_sha256": hashes,
        "label_isolation": "references, predictions, scores, and audit verdicts are excluded from provider inputs",
        "exposure_policy": "all eligible rows retained; exposure recorded only as boolean flags",
    }
    _write_once(root / "treatment-freeze.json", payload)
    return payload


def _build_candidates(
    cfg: Mapping[str, Any], cfg_path: Path, root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    source_paths = _source_paths(cfg, cfg_path)
    protocol_key = next((key for key in ("protocol", "full_panel_protocol") if key in source_paths), None)
    maps_key = next((key for key in ("maps", "full_panel_maps") if key in source_paths), None)
    pred_key = next((key for key in ("predictions", "full_panel_predictions") if key in source_paths), None)
    score_key = next((key for key in ("scores", "score", "full_panel_score") if key in source_paths), None)
    ann_key = next((key for key in ("annotations", "annotation_root") if key in source_paths), None)
    if not all((protocol_key, maps_key, pred_key, score_key, ann_key)):
        raise ValueError("source_artifacts must declare protocol, maps, predictions, scores, and annotations")
    protocol_rows = _load_protocol(source_paths[protocol_key])
    maps = _map_records(source_paths[maps_key])
    predictions = _prediction_records(source_paths[pred_key])
    score_condition = str(cfg.get("score_condition", "gemini-3.1-pro-native-av-4fps"))
    scores = _load_scores(source_paths[score_key], condition=score_condition)
    annotations = _annotation_records(source_paths[ann_key])
    exposures = _exposure_ids(cfg, cfg_path)
    candidates: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for ordinal, row in enumerate(protocol_rows, 1):
        video_id = row["video_id"]
        if video_id not in maps or video_id not in predictions or video_id not in annotations:
            raise ValueError(f"source rows do not cover protocol video_id: {video_id}")
        map_text, lines = _map_lines(maps[video_id])
        annotation = annotations[video_id]
        exposure_flags = {name: video_id in ids for name, ids in exposures.items()}
        candidate = {
            "ordinal": ordinal,
            "video_id": video_id,
            "question": row["question"],
            "question_stratum": row["question_stratum"],
            "exposure_flags": exposure_flags,
            "annotation_file_sha256": annotation["_sha256"],
            "map_sha256": _sha256_path(source_paths[maps_key]),
        }
        # Priority is retained only as a private deterministic queue field;
        # no answer/reference/prediction fields enter candidate-sheet rows.
        enriched.append({
            **candidate,
            "_protocol": row["protocol"],
            "_score": scores.get(video_id, {}).get("semantic_composite", float(ordinal)),
            "_score_stratum": scores.get(video_id, {}).get("question_stratum"),
            "_map_text": map_text,
            "_map_lines": lines,
            "_reference": annotation.get("correct_answer", annotation.get("reference")),
            "_direct_answer": predictions[video_id].get("answer", predictions[video_id].get("text")),
            "_annotation": annotation,
        })
        candidates.append(candidate)
    queue_candidates = [
        item
        for item in enriched
        if item["question_stratum"] in ALLOWED_QUEUE_STRATA
        and item["_score"] is not None
    ]
    low = sorted(
        (item for item in queue_candidates if item["_score"] <= LOW_SCORE_MAX),
        key=lambda item: (item["_score"], item["video_id"]),
    )
    control = sorted(
        (item for item in queue_candidates if item["_score"] >= CONTROL_SCORE_MIN),
        key=lambda item: (-item["_score"], item["video_id"]),
    )
    gate = _gate(cfg)
    low_target = min(gate["low_audit_count"], len(low))
    low = low[:low_target]
    control = [item for item in control if item["video_id"] not in {row["video_id"] for row in low}]
    control_target = min(gate["control_audit_count"], len(control))
    control = control[:control_target]
    return candidates, low, control, _verify_declared_hashes(cfg, cfg_path)


def _audit_row(item: Mapping[str, Any], *, queue: str, index: int) -> dict[str, Any]:
    return {
        "audit_index": index,
        "queue": queue,
        "video_id": item["video_id"],
        "question": item["question"],
        "question_stratum": item["question_stratum"],
        "exposure_flags": item["exposure_flags"],
        "frozen_map": {
            "map_sha256": _canonical_hash(item["_map_text"]),
            "map": item["_map_text"],
            "lines": item["_map_lines"],
        },
        "reference": item["_reference"],
        "direct_answer": item["_direct_answer"],
        "annotation_file_sha256": item["annotation_file_sha256"],
        "role": "",
        "verdict": "",
        "evidence_line_numbers": [],
        "adjudication": {
            "requested_entity": "",
            "requested_fact": "",
            "qualifier": "",
            "answer_unit": "",
            "evidence_line_numbers": [],
            "conflicts": [],
            "role": "",
            "verdict": "",
            "note": "",
        },
    }


def prepare_candidates(
    config_path: Path | str | None = None,
    output: Path | str | None = None,
    *,
    config: Any = None,
    output_root: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create all-row label-free candidates and the private 32-row audit queue."""

    cfg, cfg_path, root = _context(config_path, output, config=config, output_root=output_root, **kwargs)
    source_hashes = _verify_declared_hashes(cfg, cfg_path)
    treatment = _validate_treatment(root, cfg, cfg_path, source_hashes=source_hashes)
    candidates, low, control, _ = _build_candidates(cfg, cfg_path, root)
    candidate_payload = {
        "schema_version": 1,
        "status": "prepared",
        "experiment_id": cfg.get("experiment_id"),
        "row_count": len(candidates),
        "eligible_row_count": len(candidates),
        "all_rows_eligible": True,
        "exposure_policy": "flags_only",
        "rows": candidates,
    }
    _write_once(root / "candidate-sheet.json", candidate_payload)
    candidate_hash = _sha256_path(root / "candidate-sheet.json")
    low_rows = [_audit_row(item, queue="low_score", index=index) for index, item in enumerate(low, 1)]
    control_rows = [_audit_row(item, queue="supported_control", index=len(low) + index) for index, item in enumerate(control, 1)]
    audit_rows = low_rows + control_rows
    template = {
        "schema_version": 1,
        "status": "template",
        "private": True,
        "provider_input": False,
        "row_count": len(audit_rows),
        "low_score_count": len(low_rows),
        "control_count": len(control_rows),
        "treatment_freeze_sha256": _sha256_path(root / "treatment-freeze.json"),
        "source_artifacts_sha256": source_hashes,
        "rows": audit_rows,
    }
    _write_once(root / "audit-template.json", template)
    _write_once(
        root / "manual-audit.json",
        {
            "schema_version": 1,
            "status": "open",
            "private": True,
            "provider_input": False,
            "template_sha256": _sha256_path(root / "audit-template.json"),
            "rows": audit_rows,
        },
    )
    _write_once(
        root / "candidate-sheet-freeze.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "row_count": len(candidates),
            "eligible_row_count": len(candidates),
            "candidate_sheet_sha256": candidate_hash,
            "audit_template_sha256": _sha256_path(root / "audit-template.json"),
            "manual_audit_sha256": _sha256_path(root / "manual-audit.json"),
            "treatment_freeze_sha256": _sha256_path(root / "treatment-freeze.json"),
            "source_artifacts_sha256": source_hashes,
            "exposure_counts": {
                name: sum(1 for row in candidates if row["exposure_flags"].get(name, False))
                for name in _exposure_specs(cfg)
            },
            "queue_counts": {"low_score": len(low_rows), "supported_control": len(control_rows)},
        },
    )
    return treatment


def _provider_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    projection = [
        {
            "video_id": row["video_id"],
            "question": row["question"],
            "evidence": [
                {"evidence_id": item["evidence_id"], "claim": item["claim"]}
                for item in row["evidence"]
            ],
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manual_rows(manual: Mapping[str, Any], template: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = manual.get("rows")
    template_rows = template.get("rows")
    if not isinstance(rows, list) or not isinstance(template_rows, list) or len(rows) != len(template_rows):
        raise ValueError("manual audit row count does not match audit template")
    expected_ids = [row.get("video_id") for row in template_rows]
    actual_ids = [row.get("video_id") for row in rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise ValueError("manual audit IDs or order do not match audit template")
    return [dict(row) for row in rows]


def _canonicalize_audit_rows(
    manual_rows: Sequence[Mapping[str, Any]],
    template: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    protocol_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join only verdict fields to frozen source/candidate identity fields.

    ``manual-audit.json`` is an operator verdict input.  Its copied question,
    map, exposure, reference, and answer fields are deliberately ignored so a
    label or content edit cannot become provider input after candidate freeze.
    """

    candidates = candidate_payload.get("rows")
    templates = template.get("rows")
    if not isinstance(candidates, list) or not isinstance(templates, list):
        raise ValueError("candidate or template rows are invalid")
    candidate_by_id = {row.get("video_id"): row for row in candidates if isinstance(row, Mapping)}
    protocol_by_id = {row.get("video_id"): row for row in protocol_rows if isinstance(row, Mapping)}
    canonical: list[dict[str, Any]] = []
    for manual, template_row in zip(manual_rows, templates, strict=True):
        video_id = template_row.get("video_id")
        candidate = candidate_by_id.get(video_id)
        source = protocol_by_id.get(video_id)
        if candidate is None or source is None:
            raise ValueError(f"frozen source row is missing: {video_id}")
        nested = manual.get("adjudication")
        verdict = dict(nested) if isinstance(nested, Mapping) else {}
        # Support the compact top-level verdict form without accepting any
        # content fields from the editable manual copy.
        for key in (
            "role",
            "verdict",
            "evidence_line_numbers",
            "line_numbers",
            "evidence_lines",
            "requested_entity",
            "requested_fact",
            "qualifier",
            "answer_unit",
            "conflicts",
            "note",
        ):
            if key in manual and key not in verdict:
                verdict[key] = manual[key]
        canonical.append(
            {
                "audit_index": template_row.get("audit_index"),
                "video_id": video_id,
                "question": source.get("question"),
                "question_stratum": source.get("question_stratum", source.get("stratum", "unknown")),
                "exposure_flags": candidate.get("exposure_flags", {}),
                "adjudication": verdict,
            }
        )
    return canonical


def _selected_protocol_rows(
    audited: Sequence[Mapping[str, Any]], source_maps: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_role: dict[str, list[Mapping[str, Any]]] = {role: [] for role in AUDIT_ROLES}
    for row in audited:
        nested_adjudication = row.get("adjudication")
        adjudication = nested_adjudication if isinstance(nested_adjudication, Mapping) else {}
        role = adjudication.get("role") or adjudication.get("verdict") or row.get("role") or row.get("verdict")
        if role not in AUDIT_ROLES:
            raise ValueError(f"invalid or incomplete audit role: {row.get('video_id')}")
        numbers = adjudication.get("evidence_line_numbers")
        if not numbers:
            numbers = adjudication.get("line_numbers", adjudication.get("evidence_lines"))
        if not numbers:
            numbers = row.get("evidence_line_numbers", row.get("line_numbers", row.get("evidence_lines")))
        if not isinstance(numbers, list) or not numbers:
            raise ValueError(f"evidence line numbers are required: {row.get('video_id')}")
        try:
            numbers = [int(number) for number in numbers]
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid evidence line number: {row.get('video_id')}") from error
        _map_text, lines = _map_lines(source_maps[row["video_id"]])
        valid_numbers = {
            line["line_number"] for line in lines if str(line["claim"]).strip()
        }
        if len(set(numbers)) != len(numbers) or not set(numbers).issubset(valid_numbers):
            raise ValueError(f"invalid evidence line number: {row.get('video_id')}")
        by_role[role].append({**dict(row), "_line_numbers": numbers})
    counts = {role: len(rows) for role, rows in by_role.items()}
    if any(counts[role] < MINIMUM_PER_ROLE for role in ROLES):
        raise ValueError(f"fewer than four rows for role: {counts}")
    selected: list[dict[str, Any]] = []
    for role in ROLES:
        selected.extend(by_role[role][:MINIMUM_PER_ROLE])
    return [dict(row) for row in selected], counts


def freeze_selection(
    config_path: Path | str | None = None,
    output: Path | str | None = None,
    *,
    config: Any = None,
    output_root: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Freeze the completed manual audit and emit the Task 1 provider protocol."""

    cfg, cfg_path, root = _context(config_path, output, config=config, output_root=output_root, **kwargs)
    source_hashes = _verify_declared_hashes(cfg, cfg_path)
    _validate_treatment(root, cfg, cfg_path, source_hashes=source_hashes)
    candidate_freeze = _load_json(root / "candidate-sheet-freeze.json")
    candidate_payload = _load_json(root / "candidate-sheet.json")
    template = _load_json(root / "audit-template.json")
    manual = _load_json(root / "manual-audit.json")
    if candidate_freeze.get("candidate_sheet_sha256") != _sha256_path(root / "candidate-sheet.json"):
        raise ValueError("candidate sheet hash drift")
    if candidate_freeze.get("audit_template_sha256") != _sha256_path(root / "audit-template.json"):
        raise ValueError("audit template hash drift")
    if manual.get("template_sha256") != _sha256_path(root / "audit-template.json"):
        raise ValueError("manual audit template hash drift")
    manual_rows = _manual_rows(manual, template)
    source_paths = _source_paths(cfg, cfg_path)
    protocol_path = next(
        (source_paths[key] for key in ("protocol", "full_panel_protocol") if key in source_paths),
        None,
    )
    if protocol_path is None:
        raise ValueError("protocol source is required")
    frozen_protocol_rows = _load_protocol(protocol_path)
    audited = _canonicalize_audit_rows(
        manual_rows,
        template,
        candidate_payload,
        frozen_protocol_rows,
    )
    maps_path = _source_paths(cfg, cfg_path).get("maps", _source_paths(cfg, cfg_path).get("full_panel_maps"))
    if maps_path is None:
        raise ValueError("maps source is required")
    maps = _map_records(maps_path)
    selected, audited_counts = _selected_protocol_rows(audited, maps)
    protocol_rows: list[dict[str, Any]] = []
    selected_metadata: list[dict[str, Any]] = []
    for row in selected:
        evidence: list[dict[str, str]] = []
        lines = {line["line_number"]: line["claim"] for line in _map_lines(maps[row["video_id"]])[1]}
        for number in row["_line_numbers"]:
            evidence.append({"evidence_id": f"L{number:03d}", "claim": lines[number]})
        protocol_rows.append({"video_id": row["video_id"], "question": row["question"], "evidence": evidence})
        adjudication = row.get("adjudication") if isinstance(row.get("adjudication"), Mapping) else row
        selected_metadata.append({"video_id": row["video_id"], "role": adjudication.get("role", adjudication.get("verdict")), "audit_index": row["audit_index"], "evidence_line_numbers": row["_line_numbers"]})
    provider_digest = _provider_digest(protocol_rows)
    selection_payload = {
        "schema_version": 1,
        "status": "frozen",
        "experiment_id": cfg.get("experiment_id"),
        "frozen_at": datetime.now(UTC).isoformat(),
        "treatment_freeze_sha256": _sha256_path(root / "treatment-freeze.json"),
        "candidate_sheet_freeze_sha256": _sha256_path(root / "candidate-sheet-freeze.json"),
        "audit_template_sha256": _sha256_path(root / "audit-template.json"),
        "manual_audit_sha256": _sha256_path(root / "manual-audit.json"),
        "source_artifacts_sha256": source_hashes,
        "promotion_gate": _promotion_gate(cfg),
        "audited_row_count": len(audited),
        "audited_role_counts": audited_counts,
        "selected_row_count": len(protocol_rows),
        "selected_rows": [row["video_id"] for row in selected],
        "selected_row_metadata": selected_metadata,
        "role_counts": {role: MINIMUM_PER_ROLE for role in ROLES},
        "provider_rows_sha256": provider_digest,
        "exposure_stratified_counts": {
            name: {
                "audited": sum(1 for row in audited if row.get("exposure_flags", {}).get(name, False)),
                "selected": sum(1 for row in selected if row.get("exposure_flags", {}).get(name, False)),
            }
            for name in _exposure_specs(cfg)
        },
    }
    _write_once(root / "selection-freeze.json", selection_payload)
    treatment_hashes = _treatment_hashes(cfg, cfg_path)
    protocol_payload = {
        "schema_version": 1,
        "experiment_id": cfg.get("experiment_id"),
        "model": MODEL,
        "settings": _settings(cfg),
        "prompt_template_sha256": treatment_hashes["prompt_template_sha256"],
        "response_schema_sha256": treatment_hashes["response_schema_sha256"],
        "treatment_freeze_sha256": _sha256_path(root / "treatment-freeze.json"),
        "selection_freeze_sha256": _sha256_path(root / "selection-freeze.json"),
        "rows": protocol_rows,
    }
    _write_once(root / "protocol.json", protocol_payload)
    _write_once(
        root / "selection-integrity.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "selection_freeze_sha256": _sha256_path(root / "selection-freeze.json"),
            "protocol_sha256": _sha256_path(root / "protocol.json"),
            "manual_audit_sha256": _sha256_path(root / "manual-audit.json"),
            "provider_rows_sha256": provider_digest,
            "source_artifacts_sha256": source_hashes,
            "selected_row_count": len(protocol_rows),
            "role_counts": selection_payload["role_counts"],
        },
    )
    return selection_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, function in (("freeze-treatment", freeze_treatment), ("prepare-candidates", prepare_candidates), ("freeze-selection", freeze_selection)):
        del function
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", "--config-path", dest="config_path", type=Path, required=True)
        subparser.add_argument("--output", "--output-root", dest="output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    function = {"freeze-treatment": freeze_treatment, "prepare-candidates": prepare_candidates, "freeze-selection": freeze_selection}[args.command]
    function(config_path=args.config_path, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
