"""Frozen transcript-to-answer diagnostic for the Stage 0.5f Qwen panel."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import trec_video_2026.vqa_qwen_aligner_canary as aligner_canary


SCHEMA_VERSION = 1
EXPERIMENT_ID = "qwen-transcript-answer-stage05g-v1"
PANEL_SIZE = 12
MANIFEST_SIZE = 475
QWEN_CONDITION = "qwen_transcript"
WHISPER_CONDITION = "whisper_transcript"
STAGE05F_EXPERIMENT_ID = "qwen-aligner-stage05f-20260816-aligned-english-12-v1"
STAGE05F_CONTROLLER_SHA256 = (
    "b993d8e088502cc6214f77e09c5797279e01d25d536822f28e69101474157582"
)
ANSWER_MODEL_ID = "Qwen/Qwen3.5-4B"
ANSWER_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RUNTIME_IMAGE = (
    "localhost/trec-vqa-gate-rocm@sha256:"
    "510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5"
)
CANONICAL_DATA_ROOT = aligner_canary.CANONICAL_DATA_ROOT
STAGE05F_ROOT = (
    CANONICAL_DATA_ROOT
    / "vqa/benchmarks/model-verifier-canaries-20260814/"
    "stage05f-qwen-aligned-english-quality-v1"
)
STAGE05F_SELECTION_PATH = STAGE05F_ROOT / "selection.json"
STAGE05F_QWEN_RESULTS_PATH = STAGE05F_ROOT / "raw-results.json"
STAGE05F_PREDICTION_FREEZE_PATH = STAGE05F_ROOT / "predictions-freeze.json"
MANIFEST_PATH = aligner_canary.FRESH_MANIFEST_PATH
WHISPER_PATH = aligner_canary.FRESH_WHISPER_PATH
WHISPER_CHECKSUM_PATH = aligner_canary.FRESH_WHISPER_CHECKSUM_PATH
RUNNER_PATH = Path(__file__).with_name("vqa_transformers_runner.py").resolve()
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HUB_CACHE = Path("/home/npatta01/.cache/huggingface/hub").resolve()
_FRESH_SELECTION_KIND = "qwen_aligned_english_fresh_speech_panel"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_FILES = (
    "command.json",
    "raw-results.json",
    "stderr.txt",
    "terminal.json",
    "validated-results.json",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _request(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = _canonical_json(payload)
    return {
        **payload,
        "canonical_request": canonical,
        "request_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON source: {path.name}") from error


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def _source_paths() -> dict[str, Path]:
    return {
        "selection": STAGE05F_SELECTION_PATH.resolve(),
        "qwen_results": STAGE05F_QWEN_RESULTS_PATH.resolve(),
        "prediction_freeze": STAGE05F_PREDICTION_FREEZE_PATH.resolve(),
        "manifest": MANIFEST_PATH.resolve(),
        "whisper": WHISPER_PATH.resolve(),
        "whisper_checksums": WHISPER_CHECKSUM_PATH.resolve(),
        "runner": RUNNER_PATH.resolve(),
    }


def _checksum_binds_whisper() -> bool:
    expected = WHISPER_PATH.resolve().relative_to(
        WHISPER_CHECKSUM_PATH.resolve().parent
    ).as_posix()
    actual_hash = _sha256_file(WHISPER_PATH)
    for line in WHISPER_CHECKSUM_PATH.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            continue
        recorded = Path(fields[1].lstrip("*")).as_posix()
        if fields[0] == actual_hash and recorded == expected:
            return True
    return False


def verify_stage05f_root() -> dict[str, Any]:
    """Verify the complete historical freeze without changing its artifacts."""

    inputs = _load_json(STAGE05F_ROOT / "inputs-freeze.json")
    if not isinstance(inputs, Mapping):
        raise ValueError("Stage 0.5f input freeze is invalid")
    artifacts = inputs.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "selection",
        "requests",
        "protocol",
    }:
        raise ValueError("Stage 0.5f input artifact freeze is invalid")
    if inputs.get("artifacts_sha256") != aligner_canary.canonical_sha256(
        artifacts
    ):
        raise ValueError("Stage 0.5f input artifact hash is invalid")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError("Stage 0.5f input artifact record is invalid")
        aligner_canary.verify_json_artifact(
            STAGE05F_ROOT / f"{name}.json", dict(record)
        )

    protocol = _load_json(STAGE05F_ROOT / "protocol.json")
    if not isinstance(protocol, Mapping):
        raise ValueError("Stage 0.5f protocol is invalid")
    if protocol.get("controller_sha256") != STAGE05F_CONTROLLER_SHA256:
        raise ValueError("Stage 0.5f historical controller hash is invalid")
    if protocol.get("runner_sha256") != _sha256_file(
        Path(str(protocol.get("runner_path")))
    ):
        raise ValueError("Stage 0.5f runner source hash has changed")
    if protocol.get("helper_sha256") != _sha256_file(
        Path(str(protocol.get("helper_path")))
    ):
        raise ValueError("Stage 0.5f helper source hash has changed")

    previous_experiment_id = aligner_canary.EXPERIMENT_ID
    try:
        aligner_canary.configure_experiment_id(STAGE05F_EXPERIMENT_ID)
        prediction_freeze = aligner_canary._verify_prediction_freeze(STAGE05F_ROOT)
    finally:
        aligner_canary.EXPERIMENT_ID = previous_experiment_id
    validation = _load_json(STAGE05F_ROOT / "validated-results.json")
    terminal = _load_json(STAGE05F_ROOT / "terminal.json")
    if (
        prediction_freeze.get("status") != "predictions_frozen"
        or prediction_freeze.get("decision") != "ready_for_quality_test"
        or not isinstance(validation, Mapping)
        or validation.get("status") != "ready_for_quality_test"
        or validation.get("completion_count") != PANEL_SIZE
        or validation.get("valid_alignment_count") != PANEL_SIZE
        or not isinstance(terminal, Mapping)
        or terminal.get("status") != "completed"
        or protocol.get("row_count") != PANEL_SIZE
        or protocol.get("selection_kind") != _FRESH_SELECTION_KIND
    ):
        raise ValueError("Stage 0.5f source freeze is not ready")
    return {
        "status": "predictions_frozen",
        "decision": "ready_for_quality_test",
        "historical_controller_sha256": STAGE05F_CONTROLLER_SHA256,
    }


def _rows(value: Any, *, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    if not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"{field} rows must be objects")
    return value


def _index_rows(
    rows: Sequence[Mapping[str, Any]], *, field: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"{field} video IDs must be nonempty strings")
        if video_id in indexed:
            raise ValueError(f"{field} video IDs must be unique")
        indexed[video_id] = row
    return indexed


def _transcript_text(row: Mapping[str, Any], *, field: str) -> str:
    text = row.get("text")
    if not isinstance(text, str):
        original = row.get("original_language_transcript")
        text = original.get("text") if isinstance(original, Mapping) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{field} transcript must be nonempty")
    return text.strip()


def build_answer_requests(
    *,
    selection: Mapping[str, Any],
    manifest: Mapping[str, Any],
    qwen_results: Sequence[Mapping[str, Any]],
    whisper_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build 24 matched, label-free transcript-only answer requests."""

    if selection.get("selection_kind") != _FRESH_SELECTION_KIND:
        raise ValueError("selection kind is not the frozen fresh English panel")
    if selection.get("panel_size") != PANEL_SIZE:
        raise ValueError("selection panel size must be 12")
    selected_rows = _rows(selection.get("rows"), field="selection")
    if len(selected_rows) != PANEL_SIZE:
        raise ValueError("selection must contain exactly 12 rows")
    selected = _index_rows(selected_rows, field="selection")

    examples = _rows(manifest.get("examples"), field="manifest examples")
    if len(examples) != MANIFEST_SIZE:
        raise ValueError("canonical manifest must contain exactly 475 rows")
    manifest_by_id = _index_rows(examples, field="manifest")

    qwen_rows = _rows(list(qwen_results), field="Qwen results")
    if len(qwen_rows) != PANEL_SIZE:
        raise ValueError("Qwen results must contain exactly 12 rows")
    qwen_by_id = _index_rows(qwen_rows, field="Qwen results")

    whisper_rows = _rows(whisper_payload.get("results"), field="Whisper results")
    whisper_by_id = _index_rows(whisper_rows, field="Whisper results")

    selected_ids = set(selected)
    if set(qwen_by_id) != selected_ids:
        raise ValueError("Qwen results do not exactly cover the selected panel")
    if not selected_ids.issubset(whisper_by_id):
        raise ValueError("Whisper results do not cover the selected panel")
    if not selected_ids.issubset(manifest_by_id):
        raise ValueError("manifest does not cover the selected panel")

    requests: list[dict[str, Any]] = []
    for selected_row in selected_rows:
        video_id = str(selected_row["video_id"])
        audio_sha256 = selected_row.get("audio_sha256")
        if not isinstance(audio_sha256, str) or not _SHA256.fullmatch(audio_sha256):
            raise ValueError("selection audio SHA-256 is invalid")
        question = manifest_by_id[video_id].get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("selected question must be nonempty")

        qwen_row = qwen_by_id[video_id]
        whisper_row = whisper_by_id[video_id]
        for field, row in (("Qwen", qwen_row), ("Whisper", whisper_row)):
            if row.get("status") != "passed":
                raise ValueError(f"{field} transcript row must have passed")
            if row.get("source_audio_sha256") != audio_sha256:
                raise ValueError(f"{field} transcript audio hash does not match")

        for condition_id, transcript in (
            (QWEN_CONDITION, _transcript_text(qwen_row, field="Qwen")),
            (
                WHISPER_CONDITION,
                _transcript_text(whisper_row, field="Whisper"),
            ),
        ):
            requests.append(
                _request(
                    {
                        "run_id": EXPERIMENT_ID,
                        "condition_id": condition_id,
                        "video_id": video_id,
                        "question": question.strip(),
                        "frame_paths": [],
                        "frame_timestamps": [],
                        "transcript": transcript,
                    }
                )
            )
    return requests


def _build_live_payloads(prepared_at: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(prepared_at, str) or not prepared_at.strip():
        raise ValueError("prepared_at must be nonempty")
    selection = _load_json(STAGE05F_SELECTION_PATH)
    manifest = _load_json(MANIFEST_PATH)
    qwen_results = _load_json(STAGE05F_QWEN_RESULTS_PATH)
    whisper = _load_json(WHISPER_PATH)
    if not isinstance(selection, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("selection and manifest must be objects")
    if not isinstance(qwen_results, list) or not isinstance(whisper, Mapping):
        raise ValueError("transcript sources have invalid shape")
    requests = build_answer_requests(
        selection=selection,
        manifest=manifest,
        qwen_results=qwen_results,
        whisper_payload=whisper,
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "inputs_frozen",
        "prepared_at": prepared_at,
        "source_experiment_id": STAGE05F_EXPERIMENT_ID,
        "panel_size": PANEL_SIZE,
        "request_count": PANEL_SIZE * 2,
        "conditions": [QWEN_CONDITION, WHISPER_CONDITION],
        "answer_model_id": ANSWER_MODEL_ID,
        "answer_model_revision": ANSWER_MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "answer_policy": "current",
        "decoding": {"do_sample": False, "max_new_tokens": 32},
        "frame_count": 0,
        "label_state": "closed",
        "paid_compute_usd": 0.0,
    }
    request_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "condition_order": [QWEN_CONDITION, WHISPER_CONDITION],
        "requests": requests,
    }
    return protocol, request_payload


def prepare_answer_test(root: Path, *, prepared_at: str) -> dict[str, Any]:
    """Exclusively freeze the two label-closed answer arms."""

    if root.exists():
        raise FileExistsError(f"answer-test root already exists: {root}")
    verify_stage05f_root()
    source_paths = _source_paths()
    if any(not path.is_file() for path in source_paths.values()):
        raise ValueError("one or more answer-test source files are missing")
    if not _checksum_binds_whisper():
        raise ValueError("frozen checksum manifest does not bind Whisper source")
    protocol, requests = _build_live_payloads(prepared_at)
    source_files = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in source_paths.items()
    }

    root.mkdir(parents=True)
    _write_json_exclusive(root / "protocol.json", protocol)
    _write_json_exclusive(root / "requests.json", requests)
    artifacts = {
        name: {
            "path": f"{name}.json",
            "sha256": _sha256_file(root / f"{name}.json"),
        }
        for name in ("protocol", "requests")
    }
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "inputs_frozen",
        "prepared_at": prepared_at,
        "artifacts": artifacts,
        "artifacts_sha256": hashlib.sha256(
            _canonical_json(artifacts).encode()
        ).hexdigest(),
        "source_files": source_files,
        "source_files_sha256": hashlib.sha256(
            _canonical_json(source_files).encode()
        ).hexdigest(),
    }
    _write_json_exclusive(root / "inputs-freeze.json", freeze)
    return freeze


def verify_answer_inputs(root: Path) -> dict[str, Any]:
    """Re-hash every frozen input and rebuild the label-free requests."""

    freeze = _load_json(root / "inputs-freeze.json")
    if not isinstance(freeze, Mapping):
        raise ValueError("input freeze must be an object")
    if (
        freeze.get("schema_version") != SCHEMA_VERSION
        or freeze.get("experiment_id") != EXPERIMENT_ID
        or freeze.get("status") != "inputs_frozen"
    ):
        raise ValueError("input freeze identity is invalid")
    source_files = freeze.get("source_files")
    expected_paths = _source_paths()
    if not isinstance(source_files, Mapping) or set(source_files) != set(
        expected_paths
    ):
        raise ValueError("input source freeze is incomplete")
    if freeze.get("source_files_sha256") != hashlib.sha256(
        _canonical_json(source_files).encode()
    ).hexdigest():
        raise ValueError("input source freeze hash is invalid")
    for name, expected_path in expected_paths.items():
        record = source_files.get(name)
        if not isinstance(record, Mapping):
            raise ValueError("input source record is invalid")
        if record.get("path") != str(expected_path):
            raise ValueError("input source path has changed")
        if record.get("sha256") != _sha256_file(expected_path):
            raise ValueError("input source hash has changed")
    if not _checksum_binds_whisper():
        raise ValueError("Whisper source checksum binding has changed")
    verify_stage05f_root()

    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "protocol",
        "requests",
    }:
        raise ValueError("input artifact freeze is incomplete")
    if freeze.get("artifacts_sha256") != hashlib.sha256(
        _canonical_json(artifacts).encode()
    ).hexdigest():
        raise ValueError("input artifact freeze hash is invalid")
    for name in ("protocol", "requests"):
        record = artifacts.get(name)
        path = root / f"{name}.json"
        if (
            not isinstance(record, Mapping)
            or record.get("path") != path.name
            or record.get("sha256") != _sha256_file(path)
        ):
            raise ValueError("input artifact hash has changed")

    protocol = _load_json(root / "protocol.json")
    requests = _load_json(root / "requests.json")
    expected_protocol, expected_requests = _build_live_payloads(
        str(freeze.get("prepared_at"))
    )
    if protocol != expected_protocol or requests != expected_requests:
        raise ValueError("frozen answer-test inputs do not match live sources")
    return dict(freeze)


def build_answer_command(root: Path) -> list[str]:
    """Build the sole offline, digest-pinned local inference command."""

    verify_answer_inputs(root)
    for name, path in (
        ("repository", REPOSITORY_ROOT),
        ("canonical data", CANONICAL_DATA_ROOT),
        ("Hugging Face cache", HUB_CACHE),
    ):
        if not path.resolve().is_dir():
            raise ValueError(f"{name} path is not a directory")
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
        f"{HUB_CACHE.resolve()}:/root/.cache/huggingface/hub:ro",
        "--volume",
        f"{CANONICAL_DATA_ROOT.resolve()}:{CANONICAL_DATA_ROOT.resolve()}:ro",
        "--volume",
        f"{REPOSITORY_ROOT.resolve()}:{container_repo}:ro",
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
        "MPLCONFIGDIR=/tmp/matplotlib",
        "--env",
        f"VQA_RUNTIME_IMAGE={RUNTIME_IMAGE}",
        RUNTIME_IMAGE,
        "/opt/venv/bin/python",
        "-m",
        "trec_video_2026.vqa_transformers_runner",
        "--model-id",
        ANSWER_MODEL_ID,
        "--revision",
        ANSWER_MODEL_REVISION,
        "--model-class",
        "image-text",
        "--max-new-tokens",
        "32",
        "--answer-policy",
        "current",
    ]


def _sanitize_stderr(value: str, requests: Sequence[Mapping[str, Any]]) -> str:
    sanitized = value[:1_000_000]
    for request in requests:
        for field in ("question", "transcript"):
            private_text = request.get(field)
            if isinstance(private_text, str) and private_text:
                sanitized = sanitized.replace(private_text, f"<private-{field}>")
    for private_path, replacement in (
        (str(CANONICAL_DATA_ROOT), "<canonical-data-root>"),
        (str(REPOSITORY_ROOT), "<repository-root>"),
        (str(HUB_CACHE), "<model-cache>"),
    ):
        sanitized = sanitized.replace(private_path, replacement)
    for pattern in (
        r"hf_[A-Za-z0-9]{16,}",
        r"Bearer\s+[A-Za-z0-9._~+/-]{16,}",
        r"AIza[0-9A-Za-z_-]{20,}",
        r"sk-[A-Za-z0-9_-]{20,}",
    ):
        sanitized = re.sub(pattern, "<redacted-secret>", sanitized)
    return sanitized


def _validate_answer_results(
    requests: Sequence[Mapping[str, Any]],
    raw_stdout: str,
    *,
    returncode: int,
    launch_error: Exception | None,
) -> dict[str, Any]:
    required = len(requests)
    failures: list[str] = []
    try:
        parsed = json.loads(raw_stdout)
    except (TypeError, json.JSONDecodeError):
        parsed = None
        failures.append("InvalidJSON")
    rows = parsed if isinstance(parsed, list) else []
    if parsed is not None and not isinstance(parsed, list):
        failures.append("InvalidResultShape")
    if len(rows) != required:
        failures.append("WrongResultCount")

    expected = {
        (str(request["condition_id"]), str(request["video_id"]))
        for request in requests
    }
    seen: set[tuple[str, str]] = set()
    completed = 0
    peak_values: list[int] = []
    elapsed_values: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            failures.append("InvalidResultRow")
            continue
        identity = (str(row.get("condition_id")), str(row.get("video_id")))
        valid = True
        if identity not in expected or identity in seen:
            failures.append("InvalidResultIdentity")
            valid = False
        seen.add(identity)
        answer = row.get("answer")
        if row.get("run_id") != EXPERIMENT_ID:
            failures.append("RunIdentityMismatch")
            valid = False
        if row.get("status") != "passed" or not isinstance(answer, str) or not answer.strip():
            failures.append("MissingAnswer")
            valid = False
        if (
            row.get("model_id") != ANSWER_MODEL_ID
            or row.get("model_revision") != ANSWER_MODEL_REVISION
            or row.get("runtime_image") != RUNTIME_IMAGE
            or row.get("answer_policy") != "current"
        ):
            failures.append("ModelContractMismatch")
            valid = False
        if valid:
            completed += 1
        peak = row.get("peak_memory_bytes")
        if isinstance(peak, int) and not isinstance(peak, bool) and peak >= 0:
            peak_values.append(peak)
        elapsed = row.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            elapsed_values.append(float(elapsed))
    if seen != expected:
        failures.append("ResultCoverageMismatch")
    if returncode != 0:
        failures.append("NonzeroExitCode")
    if launch_error is not None:
        failures.append(type(launch_error).__name__)
    failures = sorted(set(failures))
    ready = completed == required and not failures
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "ready_for_scoring" if ready else "not_ready",
        "required_count": required,
        "completion_count": completed,
        "failure_count": required - completed,
        "failure_types": failures,
        "results_sha256": hashlib.sha256(raw_stdout.encode()).hexdigest(),
        "max_peak_memory_bytes": max(peak_values, default=0),
        "total_elapsed_seconds": sum(elapsed_values),
        "paid_compute_usd": 0.0,
    }


def record_answer_attempt(
    root: Path,
    *,
    runner: Callable[..., Any] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Execute and preserve the sole transcript-to-answer attempt."""

    verify_answer_inputs(root)
    if any((root / name).exists() for name in (*_ATTEMPT_FILES, "predictions-freeze.json")):
        raise ValueError("answer attempt already exists")
    requests_payload = _load_json(root / "requests.json")
    requests = (
        requests_payload.get("requests")
        if isinstance(requests_payload, Mapping)
        else None
    )
    if not isinstance(requests, list) or len(requests) != PANEL_SIZE * 2:
        raise ValueError("frozen answer requests are invalid")
    command = build_answer_command(root)
    command_record = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "command_frozen",
        "argv": command,
        "command_sha256": hashlib.sha256(
            _canonical_json(command).encode()
        ).hexdigest(),
    }
    _write_json_exclusive(root / "command.json", command_record)

    started = monotonic()
    launch_error: Exception | None = None
    try:
        completed = runner(
            command,
            input=json.dumps(requests, ensure_ascii=False),
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        returncode = int(completed.returncode)
        raw_stdout = completed.stdout if isinstance(completed.stdout, str) else ""
        raw_stderr = completed.stderr if isinstance(completed.stderr, str) else ""
    except Exception as error:  # noqa: BLE001 - terminal evidence must survive setup
        launch_error = error
        returncode = 127
        raw_stdout = ""
        raw_stderr = f"{type(error).__name__}: {error}"
    wall_seconds = monotonic() - started
    validation = _validate_answer_results(
        requests,
        raw_stdout,
        returncode=returncode,
        launch_error=launch_error,
    )
    _write_text_exclusive(root / "raw-results.json", raw_stdout)
    _write_text_exclusive(root / "stderr.txt", _sanitize_stderr(raw_stderr, requests))
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed" if returncode == 0 and launch_error is None else "failed",
        "attempt_index": 1,
        "exit_code": returncode,
        "wall_seconds": wall_seconds,
        "command_sha256": command_record["command_sha256"],
        "paid_compute_usd": 0.0,
    }
    _write_json_exclusive(root / "terminal.json", terminal)
    _write_json_exclusive(root / "validated-results.json", validation)
    return terminal


def freeze_answer_predictions(root: Path) -> dict[str, Any]:
    """Freeze the sole terminal attempt before any answer labels are opened."""

    verify_answer_inputs(root)
    freeze_path = root / "predictions-freeze.json"
    if freeze_path.exists():
        raise ValueError("prediction freeze already exists")
    if any(not (root / name).exists() for name in _ATTEMPT_FILES):
        raise ValueError("answer attempt evidence is incomplete")
    command = _load_json(root / "command.json")
    validation = _load_json(root / "validated-results.json")
    terminal = _load_json(root / "terminal.json")
    if not all(isinstance(value, Mapping) for value in (command, validation, terminal)):
        raise ValueError("answer attempt evidence is invalid")
    raw_stdout = (root / "raw-results.json").read_text(encoding="utf-8")
    if validation.get("results_sha256") != hashlib.sha256(
        raw_stdout.encode()
    ).hexdigest():
        raise ValueError("raw answer result hash does not match validation")
    if (
        terminal.get("status") == "completed"
        and validation.get("status") == "ready_for_scoring"
    ):
        request_payload = _load_json(root / "requests.json")
        request_rows = (
            request_payload.get("requests")
            if isinstance(request_payload, Mapping)
            else None
        )
        if not isinstance(request_rows, list):
            raise ValueError("frozen answer requests are invalid")
        revalidated = _validate_answer_results(
            request_rows,
            raw_stdout,
            returncode=int(terminal.get("exit_code", -1)),
            launch_error=None,
        )
        if dict(validation) != revalidated:
            raise ValueError("answer validation does not match raw results")
    expected_command = build_answer_command(root)
    if (
        command.get("argv") != expected_command
        or command.get("command_sha256")
        != hashlib.sha256(_canonical_json(expected_command).encode()).hexdigest()
        or terminal.get("command_sha256") != command.get("command_sha256")
    ):
        raise ValueError("answer command freeze is invalid")
    artifact_names = {
        "command.json": "command",
        "raw-results.json": "raw-results",
        "stderr.txt": "stderr",
        "terminal.json": "terminal",
        "validated-results.json": "validated-results",
    }
    artifacts = {
        artifact_names[name]: {
            "path": name,
            "sha256": _sha256_file(root / name),
        }
        for name in _ATTEMPT_FILES
    }
    decision = (
        "ready_for_scoring"
        if terminal.get("status") == "completed"
        and validation.get("status") == "ready_for_scoring"
        else "not_ready"
    )
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "predictions_frozen",
        "decision": decision,
        "input_freeze_sha256": _sha256_file(root / "inputs-freeze.json"),
        "artifacts": artifacts,
        "artifacts_sha256": hashlib.sha256(
            _canonical_json(artifacts).encode()
        ).hexdigest(),
        "paid_compute_usd": 0.0,
    }
    _write_json_exclusive(freeze_path, freeze)
    return freeze


__all__ = [
    "ANSWER_MODEL_ID",
    "ANSWER_MODEL_REVISION",
    "EXPERIMENT_ID",
    "QWEN_CONDITION",
    "RUNTIME_IMAGE",
    "WHISPER_CONDITION",
    "build_answer_command",
    "build_answer_requests",
    "freeze_answer_predictions",
    "prepare_answer_test",
    "record_answer_attempt",
    "verify_answer_inputs",
    "verify_stage05f_root",
]
