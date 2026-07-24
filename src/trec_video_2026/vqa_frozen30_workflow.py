"""CLI for the serial, resumable native-audio frozen-30 benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from trec_video_2026.vqa_benchmark import (
    EvidenceParameters,
    build_frame_evidence,
    load_manifest,
    load_references,
    source_fingerprint,
    write_resolved_manifest,
)
from trec_video_2026.vqa_evaluation import (
    evaluate_saved_predictions,
    summarize_paired_conditions,
)
from trec_video_2026.vqa_frozen30 import (
    EXPECTED_FROZEN_COUNT,
    FrozenExecution,
    SerialBenchmarkLock,
    append_attempt_rows,
    append_launch_decision,
    audit_frozen_requests,
    evidence_identity_sha256,
    pending_frozen_requests,
    select_complete_frozen_rows,
    validate_launch_decision,
)
from trec_video_2026.vqa_gate import (
    RunnerRequest,
    build_rocm_runner_command,
    build_runner_request,
    canonical_request_sha256,
    invoke_runner_batch,
    load_conditions,
    saved_request_matches,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUN = DATA / "vqa/benchmarks/runs/native-audio-benchmark-v1"
EVIDENCE = DATA / "vqa/benchmarks/evidence"
MANIFEST_PATH = ROOT / "configs/vqa/native_audio_benchmark_30.json"
MODEL_CONFIG = ROOT / "configs/vqa/model_runs.json"
PROMOTION_CONFIG = ROOT / "configs/vqa/native_audio_frozen30_promotions.json"
HF_CACHE = Path("/home/npatta01/.cache/huggingface")
ATTEMPTS = RUN / "attempts.jsonl"
DECISIONS = RUN / "promotion-decisions.jsonl"
ASR_ATTEMPTS = RUN / "whisper-attempts.jsonl"
ASR_CONDITION = "whisper-large-v3-bf16-frozen30"
ASR_MODEL = "openai/whisper-large-v3"
ASR_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
TRANSCRIPT_CONDITION = "qwen35-4b-12f-whisper-large-v3-frozen30"
CHUNK_SIZE = 10
MAX_CONDITION_SECONDS = 45 * 60
MAX_PEAK_BYTES = 32 * 1024**3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, run, resume, and score the frozen 30 benchmark"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("prepare", help="CPU-only evidence/request preflight")
    subparsers.add_parser("status", help="Read-only progress and validity summary")
    subparsers.add_parser(
        "run-whisper", help="GPU: generate frozen transcript evidence"
    )
    subparsers.add_parser(
        "build-transcript-requests",
        help="CPU-only: bind validated ASR rows to downstream requests",
    )
    for action in ("authorize", "run"):
        command = subparsers.add_parser(action)
        command.add_argument("condition")
    subparsers.add_parser("score", help="Score only after every promoted row validates")
    return parser


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return source_fingerprint(path).sha256


def validate_transcript_sources(
    path: Path,
    *,
    expected_artifact_sha256: str,
    expected_evidence_sha256: str,
) -> None:
    """Verify the external ASR selection artifact and its semantic identity."""

    if _sha256(path) != expected_artifact_sha256:
        raise ValueError("Transcript source artifact checksum does not match preflight")
    selected_sources = _json(path)
    if (
        canonical_request_sha256({"transcript_sources": selected_sources})
        != expected_evidence_sha256
    ):
        raise ValueError("Current transcript source selections do not match preflight")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _promotions() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _json(PROMOTION_CONFIG)
    manifest = load_manifest(MANIFEST_PATH, DATA)
    if (
        payload["manifest_version"] != manifest.manifest_version
        or payload["gate_selection_sha256"] != manifest.gate_selection_sha256
        or payload["recorded_before_any_frozen30_attempt"] is not True
    ):
        raise ValueError("Promotion decisions do not match the frozen manifest")
    promoted = {item["condition_id"]: item for item in payload["promoted_conditions"]}
    return payload, promoted


def _request_path(condition_id: str) -> Path:
    return RUN / "requests" / f"{condition_id}.json"


def _expected_ids() -> tuple[str, ...]:
    return tuple(item.video_id for item in load_manifest(MANIFEST_PATH, DATA).examples)


def _extract_audio(media_path: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(media_path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(temporary),
            ],
            check=True,
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare() -> None:
    if ATTEMPTS.exists() and ATTEMPTS.stat().st_size:
        raise ValueError("Cannot rebuild frozen requests after inference attempts")
    manifest = load_manifest(MANIFEST_PATH, DATA)
    config = load_conditions(MODEL_CONFIG)
    conditions = {item.condition_id: item for item in config.conditions}
    promotion_payload, promoted = _promotions()
    write_resolved_manifest(manifest, RUN / "resolved-manifest.json")
    audits: dict[str, Any] = {}
    evidence_rows: dict[tuple[str, int], dict[str, Any]] = {}

    for specification in promoted.values():
        if specification["condition_id"] == TRANSCRIPT_CONDITION:
            continue
        base = conditions[specification["base_condition_id"]]
        condition = replace(base, condition_id=specification["condition_id"])
        requests = []
        condition_evidence = []
        for example in manifest.examples:
            frame_count = int(specification["frame_count"])
            evidence = build_frame_evidence(
                example.media_path,
                EVIDENCE,
                EvidenceParameters(frame_count=frame_count),
            )
            frames = [
                {"path": str(frame.path), "sha256": frame.sha256}
                for frame in evidence.frames
            ]
            evidence_rows[(example.video_id, frame_count)] = {
                "video_id": example.video_id,
                "frame_count": frame_count,
                "cache_key": evidence.cache_key,
                "source_sha256": evidence.source.sha256,
                "frames": frames,
                "timestamps_seconds": list(evidence.timestamps_seconds),
            }
            requests.append(
                build_runner_request(
                    condition=condition,
                    run_id="native-audio-benchmark-v1",
                    video_id=example.video_id,
                    question=example.question,
                    frame_paths=tuple(frame.path for frame in evidence.frames),
                    frame_timestamps=evidence.timestamps_seconds,
                    transcript=None,
                    media_path=example.media_path,
                ).to_dict()
            )
            condition_evidence.append(evidence_rows[(example.video_id, frame_count)])
        audit = audit_frozen_requests(
            requests,
            expected_video_ids=_expected_ids(),
            condition_id=condition.condition_id,
        )
        audit["evidence_manifest_sha256"] = canonical_request_sha256(
            {"evidence": condition_evidence}
        )
        _atomic_write(
            _request_path(condition.condition_id),
            json.dumps(requests, ensure_ascii=False, indent=2) + "\n",
        )
        audits[condition.condition_id] = audit

    asr_requests = []
    audio_audit = []
    for example in manifest.examples:
        audio_path = RUN / "audio" / f"{example.video_id}.wav"
        _extract_audio(example.media_path, audio_path)
        audio_fingerprint = source_fingerprint(audio_path)
        asr_requests.append(
            {
                "run_id": "native-audio-benchmark-v1",
                "condition_id": ASR_CONDITION,
                "video_id": example.video_id,
                "audio_path": str(audio_path.resolve()),
            }
        )
        audio_audit.append(
            {
                "video_id": example.video_id,
                "path": str(audio_path.resolve()),
                "sha256": audio_fingerprint.sha256,
                "size_bytes": audio_fingerprint.size_bytes,
            }
        )
    _atomic_write(
        RUN / "whisper-requests.json",
        json.dumps(asr_requests, ensure_ascii=False, indent=2) + "\n",
    )
    preflight = {
        "schema_version": 1,
        "manifest_version": manifest.manifest_version,
        "gate_selection_sha256": manifest.gate_selection_sha256,
        "ordered_video_ids": list(_expected_ids()),
        "row_count_per_condition": EXPECTED_FROZEN_COUNT,
        "references_loaded": False,
        "promotion_config_sha256": _sha256(PROMOTION_CONFIG),
        "promotion_recorded_at_utc": promotion_payload["recorded_at_utc"],
        "condition_audits": audits,
        "whisper": {
            "condition_id": ASR_CONDITION,
            "model_id": ASR_MODEL,
            "model_revision": ASR_REVISION,
            "runtime_image": config.runtime_image,
            "requests_sha256": _sha256(RUN / "whisper-requests.json"),
            "audio": audio_audit,
        },
    }
    _atomic_write(
        RUN / "preflight.json",
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({"prepared_conditions": sorted(audits), "asr_rows": 30}))


def _asr_valid(request: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    try:
        audio_sha256 = _sha256(Path(request["audio_path"]))
    except (KeyError, OSError):
        return False
    return (
        row.get("status") == "passed"
        and row.get("run_id") == request.get("run_id")
        and row.get("condition_id") == ASR_CONDITION
        and row.get("video_id") == request.get("video_id")
        and row.get("model_id") == ASR_MODEL
        and row.get("model_revision") == ASR_REVISION
        and row.get("runtime_image")
        == _json(RUN / "preflight.json")["whisper"]["runtime_image"]
        and row.get("device") == "cuda:0 (ROCm)"
        and row.get("precision") == "bfloat16"
        and row.get("source_audio_sha256") == audio_sha256
        and isinstance(row.get("text"), str)
        and bool(row["text"].strip())
        and isinstance(row.get("quality"), Mapping)
        and saved_request_matches(row, request)
        and row.get("error_type") is None
        and row.get("error_message") is None
    )


def _whisper_command() -> list[str]:
    image = _json(RUN / "preflight.json")["whisper"]["runtime_image"]
    root = ROOT.resolve()
    data = DATA.resolve()
    return [
        "podman",
        "run",
        "--rm",
        "-i",
        "--network=none",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--group-add",
        "keep-groups",
        "-v",
        f"{root}:{root}:ro",
        "-v",
        f"{data}:{data}:ro",
        "-v",
        f"{HF_CACHE}:/root/.cache/huggingface:rw",
        "-w",
        str(root),
        "-e",
        f"PYTHONPATH={root / 'src'}",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        f"VQA_RUNTIME_IMAGE={image}",
        image,
        "python",
        "-m",
        "trec_video_2026.vqa_whisper",
        "--model-id",
        ASR_MODEL,
        "--revision",
        ASR_REVISION,
    ]


def run_whisper() -> None:
    requests = _json(RUN / "whisper-requests.json")
    attempts = _jsonl(ASR_ATTEMPTS)
    pending = [
        request
        for request in requests
        if not any(_asr_valid(request, row) for row in attempts)
    ]
    with SerialBenchmarkLock(RUN / "serial.lock", condition_id=ASR_CONDITION):
        for request in pending:
            completed = subprocess.run(
                _whisper_command(),
                input=json.dumps(request),
                text=True,
                capture_output=True,
                timeout=900,
                check=False,
            )
            try:
                result = json.loads(completed.stdout)
            except json.JSONDecodeError:
                result = {
                    "status": "failed",
                    "answer": None,
                    "error_type": "RunnerProtocolError",
                    "error_message": (completed.stderr or completed.stdout)[-4000:],
                }
            row = {
                **result,
                "run_id": request["run_id"],
                "condition_id": ASR_CONDITION,
                "video_id": request["video_id"],
                "request": request,
                "request_sha256": canonical_request_sha256(request),
            }
            append_attempt_rows(ASR_ATTEMPTS, [{**row, "answer": row.get("text")}])
            if not _asr_valid(request, row):
                raise RuntimeError(
                    f"Whisper row failed validation: {request['video_id']}"
                )
    print(json.dumps({"launched": len(pending), "skipped": 30 - len(pending)}))


def build_transcript_requests() -> None:
    if ATTEMPTS.exists() and any(
        row.get("condition_id") == TRANSCRIPT_CONDITION for row in _jsonl(ATTEMPTS)
    ):
        raise ValueError("Cannot rebuild transcript requests after condition attempts")
    manifest = load_manifest(MANIFEST_PATH, DATA)
    _, promoted = _promotions()
    base_requests = _json(_request_path("qwen35-4b-12f-frozen30"))
    asr_requests = _json(RUN / "whisper-requests.json")
    asr_attempts = _jsonl(ASR_ATTEMPTS)
    selected_sources = []
    requests = []
    for example, base, asr_request in zip(
        manifest.examples, base_requests, asr_requests, strict=True
    ):
        matches = [row for row in asr_attempts if _asr_valid(asr_request, row)]
        if not matches:
            raise ValueError(f"Missing valid Whisper evidence: {example.video_id}")
        source = matches[-1]
        accepted = source["quality"]["status"] == "accepted"
        transcript = (
            source.get("timestamped_text") or source["text"] if accepted else None
        )
        requests.append(
            {
                **base,
                "condition_id": TRANSCRIPT_CONDITION,
                "transcript": transcript,
            }
        )
        selected_sources.append(
            {
                "video_id": example.video_id,
                "attempt_sha256": canonical_request_sha256(source),
                "source_audio_sha256": source["source_audio_sha256"],
                "quality": source["quality"],
                "transcript_included": accepted,
            }
        )
    audit = audit_frozen_requests(
        requests,
        expected_video_ids=_expected_ids(),
        condition_id=TRANSCRIPT_CONDITION,
    )
    audit["evidence_manifest_sha256"] = canonical_request_sha256(
        {"transcript_sources": selected_sources}
    )
    _atomic_write(
        _request_path(TRANSCRIPT_CONDITION),
        json.dumps(requests, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        RUN / "transcript-sources.json",
        json.dumps(selected_sources, ensure_ascii=False, indent=2) + "\n",
    )
    preflight = _json(RUN / "preflight.json")
    preflight["condition_audits"][TRANSCRIPT_CONDITION] = audit
    preflight["transcript_sources_sha256"] = _sha256(RUN / "transcript-sources.json")
    preflight["transcript_model_revision"] = promoted[TRANSCRIPT_CONDITION][
        "transcript_source"
    ]["model_revision"]
    _atomic_write(
        RUN / "preflight.json",
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({"condition": TRANSCRIPT_CONDITION, "rows": 30}))


def _condition_inputs(
    condition_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], FrozenExecution]:
    _, promoted = _promotions()
    if condition_id not in promoted:
        raise ValueError(f"Condition is not promoted: {condition_id}")
    specification = promoted[condition_id]
    requests = _json(_request_path(condition_id))
    audit = audit_frozen_requests(
        requests,
        expected_video_ids=_expected_ids(),
        condition_id=condition_id,
    )
    preflight_audit = _json(RUN / "preflight.json")["condition_audits"][condition_id]
    if audit["request_set_sha256"] != preflight_audit["request_set_sha256"]:
        raise ValueError("Current request set does not match preflight")
    if condition_id == TRANSCRIPT_CONDITION:
        validate_transcript_sources(
            RUN / "transcript-sources.json",
            expected_artifact_sha256=_json(RUN / "preflight.json")[
                "transcript_sources_sha256"
            ],
            expected_evidence_sha256=preflight_audit["evidence_manifest_sha256"],
        )
    execution = FrozenExecution(
        condition_id=condition_id,
        model_id=specification["model_id"],
        model_revision=specification["model_revision"],
        runtime_image=specification["runtime_image"],
    )
    return specification, requests, execution


def authorize(condition_id: str) -> None:
    specification, requests, _ = _condition_inputs(condition_id)
    audit = audit_frozen_requests(
        requests,
        expected_video_ids=_expected_ids(),
        condition_id=condition_id,
    )
    existing = [
        row for row in _jsonl(DECISIONS) if row.get("condition_id") == condition_id
    ]
    if existing:
        validate_launch_decision(existing[-1])
        if existing[-1].get("request_set_sha256") != audit["request_set_sha256"]:
            raise ValueError("Existing launch decision binds a different request set")
        print(json.dumps(existing[-1], sort_keys=True))
        return
    preflight = _json(RUN / "preflight.json")["condition_audits"][condition_id]
    decision = append_launch_decision(
        DECISIONS,
        {
            "schema_version": 1,
            "manifest_version": _json(PROMOTION_CONFIG)["manifest_version"],
            "gate_selection_sha256": _json(PROMOTION_CONFIG)["gate_selection_sha256"],
            "condition_id": condition_id,
            "decision": "promote",
            "rationale": specification["rationale"],
            "promotion_path": specification["promotion_path"],
            "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "request_set_sha256": audit["request_set_sha256"],
            "evidence_manifest_sha256": preflight["evidence_manifest_sha256"],
            "model_id": specification["model_id"],
            "model_revision": specification["model_revision"],
            "runtime_image": specification["runtime_image"],
        },
        attempts_path=ATTEMPTS,
    )
    print(json.dumps(decision, sort_keys=True))


def _runner_request(payload: Mapping[str, Any]) -> RunnerRequest:
    return RunnerRequest(
        run_id=payload["run_id"],
        condition_id=payload["condition_id"],
        video_id=payload["video_id"],
        question=payload["question"],
        frame_paths=tuple(Path(value) for value in payload["frame_paths"]),
        frame_timestamps=tuple(payload["frame_timestamps"]),
        frame_descriptions=(
            tuple(payload["frame_descriptions"])
            if payload["frame_descriptions"] is not None
            else None
        ),
        transcript=payload["transcript"],
        media_path=Path(payload["media_path"]) if payload["media_path"] else None,
        audio_path=Path(payload["audio_path"]) if payload["audio_path"] else None,
        audio_mode=payload.get("audio_mode"),
    )


def run_condition(condition_id: str) -> None:
    specification, requests, execution = _condition_inputs(condition_id)
    decisions = [
        row for row in _jsonl(DECISIONS) if row.get("condition_id") == condition_id
    ]
    if len(decisions) != 1:
        raise ValueError("Exactly one pre-launch promotion decision is required")
    decision = decisions[0]
    validate_launch_decision(decision)
    launch_sha256 = decision["launch_decision_sha256"]
    attempts = _jsonl(ATTEMPTS)
    pending = pending_frozen_requests(
        requests,
        attempts,
        execution=execution,
        launch_decision_sha256=launch_sha256,
    )
    config = load_conditions(MODEL_CONFIG)
    base = next(
        item
        for item in config.conditions
        if item.condition_id == specification["base_condition_id"]
    )
    condition = replace(base, condition_id=condition_id)
    command = build_rocm_runner_command(
        image=config.runtime_image,
        repo_root=ROOT,
        dataset_root=DATA,
        hf_cache_root=HF_CACHE,
        condition=condition,
    )
    evidence_manifest_sha256 = decision["evidence_manifest_sha256"]
    with SerialBenchmarkLock(RUN / "serial.lock", condition_id=condition_id):
        for offset in range(0, len(pending), CHUNK_SIZE):
            chunk = pending[offset : offset + CHUNK_SIZE]
            results = invoke_runner_batch(
                command,
                [_runner_request(request) for request in chunk],
                timeout_seconds=1800,
            )
            rows = [
                {
                    **result,
                    "run_id": request["run_id"],
                    "condition_id": request["condition_id"],
                    "video_id": request["video_id"],
                    "request": request,
                    "request_sha256": canonical_request_sha256(request),
                    "launch_decision_sha256": launch_sha256,
                    "evidence_identity_sha256": evidence_identity_sha256(request),
                    "evidence_manifest_sha256": evidence_manifest_sha256,
                }
                for request, result in zip(chunk, results, strict=True)
            ]
            append_attempt_rows(ATTEMPTS, rows)
            all_attempts = _jsonl(ATTEMPTS)
            valid_count = EXPECTED_FROZEN_COUNT - len(
                pending_frozen_requests(
                    requests,
                    all_attempts,
                    execution=execution,
                    launch_decision_sha256=launch_sha256,
                )
            )
            condition_rows = [
                row for row in all_attempts if row.get("condition_id") == condition_id
            ]
            attempted = len({row.get("video_id") for row in condition_rows})
            failures = attempted - valid_count
            peaks = [
                row["peak_memory_bytes"]
                for row in condition_rows
                if isinstance(row.get("peak_memory_bytes"), int)
            ]
            elapsed = [
                row["elapsed_seconds"]
                for row in condition_rows
                if row.get("status") == "passed"
                and isinstance(row.get("elapsed_seconds"), (int, float))
            ]
            if peaks and max(peaks) >= MAX_PEAK_BYTES:
                raise RuntimeError("Stop rule: peak allocation reached 32 GiB")
            if attempted >= 10 and failures / attempted > 0.10:
                raise RuntimeError("Stop rule: more than 10% of rows are unscoreable")
            if (
                elapsed
                and fmean(elapsed) * EXPECTED_FROZEN_COUNT > MAX_CONDITION_SECONDS
            ):
                raise RuntimeError("Stop rule: projected condition time exceeds 45 min")
    print(json.dumps({"launched": len(pending), "skipped": 30 - len(pending)}))


def score() -> None:
    from nltk import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    _, promoted = _promotions()
    attempts = _jsonl(ATTEMPTS)
    selected = []
    for condition_id in promoted:
        _, requests, execution = _condition_inputs(condition_id)
        decision = next(
            row for row in _jsonl(DECISIONS) if row["condition_id"] == condition_id
        )
        validate_launch_decision(decision)
        selected.extend(
            select_complete_frozen_rows(
                requests,
                attempts,
                execution=execution,
                launch_decision_sha256=decision["launch_decision_sha256"],
            )
        )
    selected_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected
    )
    selected_path = RUN / "selected-predictions.jsonl"
    _atomic_write(selected_path, selected_text)

    # Evaluation-only boundary: labels are loaded only after the selected,
    # label-free prediction artifact exists and is hash-frozen.
    manifest = load_manifest(MANIFEST_PATH, DATA)
    references = {
        item.video_id: item.reference_answer for item in load_references(manifest)
    }
    strata = {item.video_id: item.category for item in manifest.examples}
    meteor_rows = []
    for row in selected:
        prediction = " ".join(row["answer"].strip().lower().split())
        reference = " ".join(references[row["video_id"]].strip().lower().split())
        meteor = float(
            meteor_score([word_tokenize(reference)], word_tokenize(prediction))
        )
        meteor_rows.append({**row, "semantic_metrics": {"meteor": meteor}})
    scored_path = RUN / "scored-predictions.jsonl"
    _atomic_write(
        scored_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in meteor_rows
        ),
    )
    evaluation = evaluate_saved_predictions(
        scored_path, references=references, strata=strata
    )
    examples = evaluation["examples"]
    comparisons = {}
    for baseline, treatment in (
        ("qwen35-4b-12f-frozen30", "qwen35-4b-24f-frozen30"),
        ("qwen35-4b-12f-frozen30", TRANSCRIPT_CONDITION),
    ):
        comparisons[f"{treatment}_vs_{baseline}"] = summarize_paired_conditions(
            examples,
            baseline_condition_id=baseline,
            treatment_condition_id=treatment,
            bootstrap_iterations=2000,
            bootstrap_seed=20260724,
        )
    payload = {
        "evaluation_boundary": {
            "references_loaded_only_after_prediction_freeze": True,
            "selected_prediction_sha256": _sha256(selected_path),
            "scored_prediction_sha256": _sha256(scored_path),
            "selected_rows": len(selected),
            "expected_rows": EXPECTED_FROZEN_COUNT * len(promoted),
            "manifest_version": manifest.manifest_version,
            "gate_selection_sha256": manifest.gate_selection_sha256,
        },
        **evaluation,
        "paired_comparisons": comparisons,
    }
    _atomic_write(
        RUN / "evaluation.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(payload["conditions"], sort_keys=True))


def status() -> None:
    attempts = _jsonl(ATTEMPTS)
    decisions = _jsonl(DECISIONS)
    _, promoted = _promotions()
    payload = {
        "run_dir": str(RUN),
        "prepared": (RUN / "preflight.json").exists(),
        "whisper_valid_rows": 0,
        "conditions": {},
    }
    if (RUN / "whisper-requests.json").exists():
        payload["whisper_valid_rows"] = sum(
            any(_asr_valid(request, row) for row in _jsonl(ASR_ATTEMPTS))
            for request in _json(RUN / "whisper-requests.json")
        )
    for condition_id in promoted:
        request_path = _request_path(condition_id)
        decision = next(
            (row for row in decisions if row.get("condition_id") == condition_id),
            None,
        )
        valid = 0
        if request_path.exists() and decision is not None:
            _, requests, execution = _condition_inputs(condition_id)
            valid = EXPECTED_FROZEN_COUNT - len(
                pending_frozen_requests(
                    requests,
                    attempts,
                    execution=execution,
                    launch_decision_sha256=decision["launch_decision_sha256"],
                )
            )
        payload["conditions"][condition_id] = {
            "requests_prepared": request_path.exists(),
            "authorized": decision is not None,
            "valid_rows": valid,
            "expected_rows": EXPECTED_FROZEN_COUNT,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "prepare":
        prepare()
    elif args.action == "status":
        status()
    elif args.action == "run-whisper":
        run_whisper()
    elif args.action == "build-transcript-requests":
        build_transcript_requests()
    elif args.action == "authorize":
        authorize(args.condition)
    elif args.action == "run":
        run_condition(args.condition)
    elif args.action == "score":
        score()
    else:
        raise AssertionError(args.action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
