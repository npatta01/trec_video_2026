from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
ADJUDICATOR_PATH = ROOT / "scripts/adjudicate_vqa_event_map_canary.py"
RUNNER_PATH = ROOT / "scripts/run-vqa-event-map-dstack.sh"
MODULE_PATH = ROOT / "src/trec_video_2026/vqa_experiments/nemotron3_omni_event_map.py"
INPUT_DIR = ROOT / "data/vqa/benchmarks/vqa-event-map-inputs-20260809-a3/inputs"
CONFIG_PATH = ROOT / "dstack/vqa-event-map-canary.dstack.yml"
ATTEMPT_UUID = "d6d3a33e-84c5-4067-bcea-fe82cdd10e2f"
EXECUTION_RUN_ID = "vqa-event-map-canary-a2"
INPUT_RUN_ID = "vqa-event-map-inputs-20260809-a3"
OUTPUT_IDENTITY = (
    "hf://buckets/Npatta01/trec-video-2026-event-map-20260809-vqa-event-map-canary-a2"
)
DSTACK_RUN_UUID = "11111111-1111-4111-8111-111111111111"
FOREIGN_DSTACK_RUN_UUID = "22222222-2222-4222-8222-222222222222"
SOURCE_BUNDLE_SHA256 = (
    "3dbfa1e31b4550a9ed1c77a2abdbad55c15d1594c7e2a153ad26dcc2101b138b"
)
CONTROLLER_SHA256 = "91815765952ede4e7986b09eec64ce2f831536259a1b3e1f12d0127b244be5a8"
PROTOCOL_SHA256 = "bc140b5ace4f1cfce9feb98434ca5276466602ed7e2487261788fc163c9fc4f2"
REQUEST_SHA256 = "eeb15558494948a77c881e557c14b5728865fd838a38ef4823faf2db5d85709c"
MODULE_SHA256 = "0deaf7eccafc9225ba533982c19ea894fdc8571c9081db4a52a3ddf11d5ca366"
MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
MODEL_REVISION = "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
RUNTIME_IMAGE = (
    "docker.io/vllm/vllm-openai:v0.20.0@"
    "sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
)
RUNTIME_IMAGE_DIGEST = RUNTIME_IMAGE.rsplit("@", 1)[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_adjudicator():
    if not ADJUDICATOR_PATH.is_file():
        pytest.fail("adjudicator is not implemented yet")
    return _load_module("vqa_event_map_canary_adjudicator", ADJUDICATOR_PATH)


def load_event_map_module():
    return _load_module("vqa_event_map_request_contract", MODULE_PATH)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def scheduler_evidence(
    *,
    elapsed_seconds: float = 1000.0,
    price: float = 1.0,
    config_path: Path = CONFIG_PATH,
) -> dict[str, object]:
    started = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=elapsed_seconds)
    return {
        "run_id": DSTACK_RUN_UUID,
        "run_name": EXECUTION_RUN_ID,
        "config_sha256": _sha256_file(config_path),
        "status": "done",
        "cost": elapsed_seconds / 3600 * price,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "backend": "runpod",
        "region": "US-KS-2",
        "gpu": {"name": "L40S", "memory_mib": 49152},
        "price": price,
    }


def _runtime() -> dict[str, object]:
    return {
        "schema_version": 1,
        "runtime_image": RUNTIME_IMAGE,
        "vllm_version": "0.20.0",
        "torch_version": "2.8.0",
        "torch_cuda": "12.8",
        "torch_hip": None,
        "gpu_name": "NVIDIA L40S",
        "gpu_memory_mib": 49152,
        "enable_thinking": False,
        "max_output_tokens": 768,
        "gpu_telemetry_path": "telemetry/nvidia-smi.csv",
    }


def _request_hash(
    request: dict[str, object],
    *,
    retry: bool = False,
    audio_enabled: bool | None = None,
) -> str:
    event_map = load_event_map_module()
    condition = str(request["condition_id"])
    payload = event_map.build_chat_request(
        media_path=Path(f"/dstack/input/normalized/{request['video_id']}.mp4"),
        question=str(request["question"]),
        duration_seconds=float(request["duration_seconds"]),
        audio_enabled=(condition == "av-map")
        if audio_enabled is None
        else audio_enabled,
        response_format_supported=True,
        retry=retry,
    )
    return event_map.canonical_sha256(payload)


def _prediction_row(
    request: dict[str, object], runtime: dict[str, object]
) -> dict[str, object]:
    duration = float(request["duration_seconds"])
    primary_hash = _request_hash(request)
    return {
        "schema_version": 1,
        "video_id": request["video_id"],
        "condition_id": request["condition_id"],
        "request_sha256": primary_hash,
        "primary_request_sha256": primary_hash,
        "retry_request_sha256": None,
        "primary_valid": True,
        "retry_valid": None,
        "retry_used": False,
        "status": "passed",
        "error_type": None,
        "response_sha256": "2" * 64,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "memory_telemetry": {
            "path": "telemetry/nvidia-smi.csv",
            "sampling_interval_seconds": 5,
        },
        "runtime_identity": runtime,
        "answer": "A short answer",
        "abstain": False,
        "confidence": 0.8,
        "events": [
            {
                "event_id": "event-0",
                "start": 0.0,
                "end": duration,
                "visual_summary": "A private visual event",
                "audio_summary": "A private audio event",
            }
        ],
        "selected_intervals": [
            {"event_id": "event-0", "start": 0.0, "end": min(1.0, duration)}
        ],
        "elapsed_seconds": 10.0,
        "run_id": INPUT_RUN_ID,
        "slice_id": "canary",
        "question": request["question"],
        "question_stratum": request["question_stratum"],
        "controller_request_sha256": request["request_sha256"],
        "media_path": request["media_path"],
        "source_media_sha256": request["media_sha256"],
        "normalized_media_sha256": "3" * 64,
        "source_duration_seconds": duration,
        "normalized_duration_seconds": duration,
    }


def _producer_timings(
    *,
    bootstrap_download_seconds: float = 300.0,
    normalization_count: int = 4,
    normalization_seconds: float = 80.0,
    model_startup_seconds: float = 500.0,
    inference_seconds: float = 80.0,
    finalization_seconds: float = 40.0,
) -> dict[str, object]:
    phase_total = (
        bootstrap_download_seconds
        + normalization_seconds
        + model_startup_seconds
        + inference_seconds
        + finalization_seconds
    )
    return {
        "schema_version": 1,
        "clock": "monotonic",
        "bootstrap_download_seconds": bootstrap_download_seconds,
        "normalization": {
            "media_count": normalization_count,
            "seconds": normalization_seconds,
        },
        "model_startup_seconds": model_startup_seconds,
        "inference_seconds": inference_seconds,
        "finalization_seconds": finalization_seconds,
        "phase_total_seconds": phase_total,
    }


def _write_checksums(artifact_dir: Path) -> None:
    payloads = sorted(
        path
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "terminal-status.json"}
    )
    (artifact_dir / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256_file(path)}  ./{path.relative_to(artifact_dir).as_posix()}\n"
            for path in payloads
        ),
        encoding="utf-8",
    )


def _write_terminal(
    artifact_dir: Path,
    *,
    row_count: int = 8,
    timings: dict[str, object] | None = None,
) -> None:
    checksum = _sha256_file(artifact_dir / "SHA256SUMS")
    write_json(
        artifact_dir / "terminal-status.json",
        {
            "schema_version": 1,
            "status": "complete",
            "run_id": EXECUTION_RUN_ID,
            "attempt_uuid": ATTEMPT_UUID,
            "output_identity": OUTPUT_IDENTITY,
            "payload_checksum": checksum,
            "row_count": row_count,
            "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
            "controller_sha256": CONTROLLER_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
            "request_file_sha256": REQUEST_SHA256,
            "inference_module_sha256": MODULE_SHA256,
            "runner_sha256": _sha256_file(RUNNER_PATH),
            "producer_timings": timings or _producer_timings(),
        },
    )


def canary_artifacts(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "canary-output"
    telemetry_dir = artifact_dir / "telemetry"
    telemetry_dir.mkdir(parents=True)
    runtime = _runtime()
    write_json(artifact_dir / "runtime.json", runtime)
    write_json(
        artifact_dir / "gpu-identity.json",
        {"gpu_name": "L40S", "gpu_memory_mib": 49152},
    )
    write_json(
        artifact_dir / "preflight.json",
        {
            "schema_version": 1,
            "run_id": INPUT_RUN_ID,
            "attempt_uuid": ATTEMPT_UUID,
            "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
            "controller_sha256": CONTROLLER_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
            "request_file_sha256": REQUEST_SHA256,
            "inference_module_sha256": MODULE_SHA256,
            "runner_sha256": _sha256_file(RUNNER_PATH),
            "row_count": 8,
            "condition_id": "v-map,av-map",
            "slice_id": "",
        },
    )
    requests = [
        json.loads(line)
        for line in (INPUT_DIR / "event-map-canary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    predictions = [_prediction_row(request, runtime) for request in requests]
    (artifact_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    (telemetry_dir / "nvidia-smi.csv").write_text(
        "timestamp,gpu_name,memory_used_mib,memory_total_mib,utilization_gpu\n"
        "2026-08-09T12:00:00Z,L40S,12000,49152,80\n"
        "2026-08-09T12:00:05Z,L40S,13000,49152,81\n",
        encoding="utf-8",
    )
    _write_checksums(artifact_dir)
    _write_terminal(artifact_dir)
    return artifact_dir


def _refresh_integrity(
    artifact_dir: Path,
    *,
    row_count: int = 8,
    timings: dict[str, object] | None = None,
) -> None:
    if timings is None and (artifact_dir / "terminal-status.json").is_file():
        terminal = json.loads(
            (artifact_dir / "terminal-status.json").read_text(encoding="utf-8")
        )
        timings = terminal.get("producer_timings")
    _write_checksums(artifact_dir)
    _write_terminal(artifact_dir, row_count=row_count, timings=timings)


def _write_predictions(artifact_dir: Path, rows: list[dict[str, object]]) -> None:
    (artifact_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _refresh_integrity(artifact_dir)


def adjudicate(
    adjudicator,
    artifacts: Path,
    scheduler: dict[str, object] | Path,
    *,
    config_path: Path = CONFIG_PATH,
    dstack_run_id: str | None = DSTACK_RUN_UUID,
):
    return adjudicator.adjudicate_canary(
        artifacts,
        scheduler,
        input_dir=INPUT_DIR,
        config_path=config_path,
        dstack_run_id=dstack_run_id,
    )


def test_success_emits_bounded_go_without_private_or_scheduler_details(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "GO"
    assert decision["gates"]["rows"] is True
    assert decision["gates"]["phase_timing"] is True
    assert decision["observed"]["valid_rows"] == 8
    assert decision["observed"]["schema_retries"] == 0
    assert decision["observed"]["normalization_media_count"] == 4
    assert decision["projection"]["calls"] == 950
    assert decision["projection"]["inference_margin"] == pytest.approx(1.1)
    assert decision["projection"]["wall_time_hours"] <= 3.5
    encoded = json.dumps(decision, sort_keys=True)
    assert "A short answer" not in encoded
    assert "private visual" not in encoded
    assert "US-KS-2" not in encoded


def test_checksum_tamper_is_no_go(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    predictions = artifacts / "predictions.jsonl"
    predictions.write_bytes(predictions.read_bytes() + b"tamper")

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "artifact_integrity" in decision["reasons"]


def test_terminal_marker_is_trusted_only_when_its_binding_is_exact(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    terminal = json.loads(
        (artifacts / "terminal-status.json").read_text(encoding="utf-8")
    )
    terminal["payload_checksum"] = "0" * 64
    write_json(artifacts / "terminal-status.json", terminal)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "terminal" in decision["reasons"]


def test_scheduler_sanitizer_requires_exact_bound_record() -> None:
    adjudicator = load_adjudicator()
    evidence = scheduler_evidence()

    sanitized = adjudicator.sanitize_dstack_scheduler_json(
        evidence,
        expected_dstack_run_id=DSTACK_RUN_UUID,
        expected_config_sha256=_sha256_file(CONFIG_PATH),
    )

    assert sanitized["dstack_run_id"] == DSTACK_RUN_UUID
    assert sanitized["run_name"] == EXECUTION_RUN_ID
    assert sanitized["config_sha256"] == _sha256_file(CONFIG_PATH)
    assert set(sanitized) == {
        "schema_version",
        "dstack_run_id",
        "run_name",
        "config_sha256",
        "status",
        "elapsed_seconds",
        "cost",
        "price",
        "backend",
        "gpu_memory_mib",
    }
    with pytest.raises(ValueError, match="schema mismatch"):
        adjudicator.sanitize_dstack_scheduler_json(
            {**evidence, "secret": "must not be accepted"},
            expected_dstack_run_id=DSTACK_RUN_UUID,
            expected_config_sha256=_sha256_file(CONFIG_PATH),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.update({"unexpected": "extra"}),
        lambda config: config.__setitem__(
            "commands", ["echo prefix\n" + config["commands"][0]]
        ),
        lambda config: config.__setitem__(
            "commands", [config["commands"][0] + "echo suffix\n"]
        ),
    ],
    ids=("extra-top-level-key", "command-prefix", "command-suffix"),
)
def test_config_validation_is_exact_and_fail_closed(
    tmp_path: Path,
    mutation,
) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutation(config)
    config_path = tmp_path / "canary.dstack.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    decision = adjudicate(
        adjudicator,
        artifacts,
        scheduler_evidence(config_path=config_path),
        config_path=config_path,
    )

    assert decision["status"] == "NO-GO"
    assert "config_identity" in decision["reasons"]


def test_scheduler_config_hash_must_bind_exact_config_bytes(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    scheduler = scheduler_evidence()
    scheduler["config_sha256"] = "0" * 64

    decision = adjudicate(adjudicator, artifacts, scheduler)

    assert decision["status"] == "NO-GO"
    assert "scheduler_identity" in decision["reasons"]


@pytest.mark.parametrize("dstack_run_id", [None, FOREIGN_DSTACK_RUN_UUID])
def test_api_rejects_omitted_or_foreign_dstack_uuid(
    tmp_path: Path,
    dstack_run_id: str | None,
) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)

    decision = adjudicate(
        adjudicator,
        artifacts,
        scheduler_evidence(),
        dstack_run_id=dstack_run_id,
    )

    assert decision["status"] == "NO-GO"
    assert decision["dstack_run_id"] is None
    assert "scheduler_identity" in decision["reasons"]


def test_cli_requires_dstack_uuid() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ADJUDICATOR_PATH),
            "--artifacts",
            "/does/not/matter",
            "--scheduler-json",
            "/does/not/matter.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--dstack-run-id" in result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [("backend", "aws"), ("cost", 99.0)],
)
def test_scheduler_backend_and_billing_are_reconciled(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    scheduler = scheduler_evidence()
    scheduler[field] = value

    decision = adjudicate(adjudicator, artifacts, scheduler)

    assert decision["status"] == "NO-GO"
    assert "scheduler_identity" in decision["reasons"]


def test_missing_row_is_no_go_even_with_consistent_payload_checksums(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    predictions = artifacts / "predictions.jsonl"
    rows = predictions.read_text(encoding="utf-8").splitlines()[:-1]
    predictions.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _refresh_integrity(artifacts, row_count=8)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "rows" in decision["reasons"]


def test_wrong_audio_pairing_is_no_go(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    predictions = artifacts / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["condition_id"] = "v-map"
    _write_predictions(artifacts, rows)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "row_identity" in decision["reasons"]


def test_swapped_v_av_request_hashes_are_no_go(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    predictions = artifacts / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    first_hash = rows[0]["primary_request_sha256"]
    second_hash = rows[1]["primary_request_sha256"]
    rows[0]["primary_request_sha256"] = second_hash
    rows[0]["request_sha256"] = second_hash
    rows[1]["primary_request_sha256"] = first_hash
    rows[1]["request_sha256"] = first_hash
    _write_predictions(artifacts, rows)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "request_hashes" in decision["reasons"]


def test_wrong_audio_flag_request_hash_is_no_go(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    requests = [
        json.loads(line)
        for line in (INPUT_DIR / "event-map-canary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    predictions = artifacts / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    wrong_hash = _request_hash(requests[0], audio_enabled=True)
    rows[0]["primary_request_sha256"] = wrong_hash
    rows[0]["request_sha256"] = wrong_hash
    _write_predictions(artifacts, rows)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "request_hashes" in decision["reasons"]


def test_label_reference_or_metric_field_is_no_go(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    predictions = artifacts / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["runtime_identity"]["reference_metric"] = "forbidden"
    _write_predictions(artifacts, rows)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "leakage" in decision["reasons"]


def test_retry_transport_failure_row_is_no_go(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    requests = [
        json.loads(line)
        for line in (INPUT_DIR / "event-map-canary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    predictions = artifacts / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    retry_hash = _request_hash(requests[0], retry=True)
    rows[0].update(
        {
            "status": "failed",
            "error_type": "request_failed",
            "primary_valid": False,
            "retry_valid": False,
            "retry_used": True,
            "retry_request_sha256": retry_hash,
            "request_sha256": retry_hash,
            "response_sha256": None,
        }
    )
    _write_predictions(artifacts, rows)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "retry_transport_failure" in decision["reasons"]


def test_schema_retry_gate_is_per_row_and_reports_total_retry_count(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    requests = [
        json.loads(line)
        for line in (INPUT_DIR / "event-map-canary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    predictions = artifacts / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    for request, row in zip(requests[:2], rows[:2], strict=True):
        retry_hash = _request_hash(request, retry=True)
        row.update(
            {
                "primary_valid": False,
                "retry_valid": True,
                "retry_used": True,
                "retry_request_sha256": retry_hash,
                "request_sha256": retry_hash,
            }
        )
    _write_predictions(artifacts, rows)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "GO"
    assert decision["observed"]["schema_retries"] == 2


def test_projection_scales_fixed_phases_over_eight_tasks_and_inference_by_condition() -> (
    None
):
    adjudicator = load_adjudicator()
    row_summary = {
        "elapsed_seconds": 100.0,
        "elapsed_by_condition": {
            "v-map": [30.0, 10.0, 10.0, 10.0],
            "av-map": [10.0] * 4,
        },
    }
    timings = _producer_timings(
        bootstrap_download_seconds=300.0,
        normalization_seconds=80.0,
        model_startup_seconds=500.0,
        inference_seconds=100.0,
        finalization_seconds=40.0,
    )
    scheduler = {"elapsed_seconds": 1020.0, "price": 1.0}

    projection = adjudicator._projection(scheduler, row_summary, timings)

    fixed_seconds = 300.0 + 500.0 + 40.0
    normalization_seconds = (80.0 / 4) * 950
    inference_seconds = 1.1 * 475 * (15.0 + 10.0)
    expected_aggregate = (
        fixed_seconds * 8 + normalization_seconds + inference_seconds
    ) / 3600
    expected_longest = (fixed_seconds + (80.0 / 4) * 119 + 1.1 * 15.0 * 119) / 3600
    assert projection["aggregate_gpu_hours"] == pytest.approx(expected_aggregate)
    assert projection["longest_shard_hours"] == pytest.approx(expected_longest)
    assert projection["wall_time_hours"] == pytest.approx(expected_longest)


def test_scalable_normalization_can_cross_aggregate_and_longest_shard_gates(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    timings = _producer_timings(normalization_seconds=440.0)
    _refresh_integrity(artifacts, timings=timings)

    decision = adjudicate(
        adjudicator,
        artifacts,
        scheduler_evidence(
            elapsed_seconds=float(timings["phase_total_seconds"]), price=0.5
        ),
    )

    assert decision["status"] == "NO-GO"
    assert "aggregate_gpu_hours" in decision["reasons"]
    assert "longest_shard_hours" in decision["reasons"]


def test_phase_totals_must_reconcile_with_scheduler_duration(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)

    decision = adjudicate(
        adjudicator,
        artifacts,
        scheduler_evidence(elapsed_seconds=1400.0),
    )

    assert decision["status"] == "NO-GO"
    assert "phase_timing" in decision["reasons"]


def test_identity_mismatch_is_no_go(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    preflight = json.loads((artifacts / "preflight.json").read_text(encoding="utf-8"))
    preflight["attempt_uuid"] = "511c5f82-68c1-4df8-a19e-ac91e28f9232"
    write_json(artifacts / "preflight.json", preflight)
    _refresh_integrity(artifacts)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "identity" in decision["reasons"]


def test_runner_sha_mismatch_is_no_go(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    preflight = json.loads((artifacts / "preflight.json").read_text(encoding="utf-8"))
    preflight["runner_sha256"] = "4" * 64
    write_json(artifacts / "preflight.json", preflight)
    _refresh_integrity(artifacts)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "identity" in decision["reasons"]


def test_superseded_a1_identity_is_rejected(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    terminal = json.loads(
        (artifacts / "terminal-status.json").read_text(encoding="utf-8")
    )
    terminal.update(
        {
            "run_id": "vqa-event-map-canary",
            "attempt_uuid": "4baaf48c-b690-45f0-b6bf-065b9dc1e527",
            "output_identity": (
                "hf://buckets/Npatta01/trec-video-2026-event-map-20260809-"
                "vqa-event-map-canary"
            ),
        }
    )
    write_json(artifacts / "terminal-status.json", terminal)

    decision = adjudicate(adjudicator, artifacts, scheduler_evidence())

    assert decision["status"] == "NO-GO"
    assert "terminal" in decision["reasons"]


def test_failed_a2_without_payload_stays_no_go(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    missing_a2_output = tmp_path / "a2-produced-no-payload"

    decision = adjudicate(
        adjudicator,
        missing_a2_output,
        scheduler_evidence(elapsed_seconds=300.0),
    )

    assert decision["status"] == "NO-GO"
    assert decision["attempt_uuid"] == ATTEMPT_UUID
    assert "artifact_integrity" in decision["reasons"]


def test_timeout_is_no_go_at_exactly_2700_seconds(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    timings = _producer_timings(bootstrap_download_seconds=2000.0)
    assert timings["phase_total_seconds"] == 2700.0
    _refresh_integrity(artifacts, timings=timings)

    decision = adjudicate(
        adjudicator,
        artifacts,
        scheduler_evidence(elapsed_seconds=2700.0),
    )

    assert decision["status"] == "NO-GO"
    assert "canary_timeout" in decision["reasons"]


def test_projected_cost_gate_is_fail_closed(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)

    decision = adjudicate(
        adjudicator,
        artifacts,
        scheduler_evidence(price=10.0),
    )

    assert decision["status"] == "NO-GO"
    assert "projected_cost" in decision["reasons"]


@pytest.mark.parametrize("malformation", ["missing-artifacts", "bad-json", "bad-utf8"])
def test_python_api_turns_malformed_io_into_bounded_no_go(
    tmp_path: Path,
    malformation: str,
) -> None:
    adjudicator = load_adjudicator()
    artifacts = canary_artifacts(tmp_path)
    scheduler: dict[str, object] | Path = scheduler_evidence()
    if malformation == "missing-artifacts":
        artifacts = tmp_path / "missing"
    else:
        scheduler_path = tmp_path / "scheduler.json"
        if malformation == "bad-json":
            scheduler_path.write_text("{not-json", encoding="utf-8")
        else:
            scheduler_path.write_bytes(b"\xff\xfe")
        scheduler = scheduler_path

    decision = adjudicate(adjudicator, artifacts, scheduler)

    assert decision["status"] == "NO-GO"
    assert set(decision) == {
        "schema_version",
        "status",
        "run_id",
        "attempt_uuid",
        "dstack_run_id",
        "gates",
        "observed",
        "projection",
        "reasons",
    }
    encoded = json.dumps(decision, sort_keys=True)
    assert len(encoded) < 4096
    assert "Traceback" not in encoded
    assert "UnicodeDecodeError" not in encoded
