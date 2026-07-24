"""Orchestration primitives for the local VQA compatibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RUN_STATUSES = frozenset({"passed", "unsupported", "failed", "timed_out"})
AUDIO_MODES = frozenset({"audio-off", "audio-on"})
REQUIRED_RESULT_FIELDS = frozenset(
    {
        "status",
        "answer",
        "elapsed_seconds",
        "model_id",
        "model_revision",
        "runtime",
        "precision",
        "device",
        "peak_memory_bytes",
        "error_type",
        "error_message",
    }
)


def canonical_request_sha256(request: Mapping[str, Any]) -> str:
    """Hash every field in a JSON runner request without lossy normalization."""

    encoded = json.dumps(
        request,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def saved_request_matches(
    attempt: Mapping[str, Any], current_request: Mapping[str, Any]
) -> bool:
    """Fail closed unless a saved attempt contains the exact current request."""

    saved_request = attempt.get("request")
    if not isinstance(saved_request, Mapping):
        return False
    try:
        saved_digest = canonical_request_sha256(saved_request)
        current_digest = canonical_request_sha256(current_request)
    except (TypeError, ValueError):
        return False
    recorded_digest = attempt.get("request_sha256")
    if recorded_digest is not None and recorded_digest != saved_digest:
        return False
    return saved_digest == current_digest


@dataclass(frozen=True)
class Condition:
    condition_id: str
    model_id: str
    model_revision: str
    evidence: str
    download_bytes: int
    runner_kind: str = "image-text"
    runtime_image: str | None = None
    audio_mode: str | None = None
    task_mode: str | None = None

    def __post_init__(self) -> None:
        if self.audio_mode is not None and self.audio_mode not in AUDIO_MODES:
            raise ValueError("Condition audio_mode must be audio-off or audio-on")


@dataclass(frozen=True)
class ConditionsConfig:
    runtime_image: str
    runtime_base_digest: str
    transformers_revision: str
    conditions: tuple[Condition, ...]


def load_conditions(path: Path) -> ConditionsConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported conditions schema version")
    conditions = tuple(Condition(**item) for item in payload.get("conditions", []))
    if not conditions:
        raise ValueError("Conditions cannot be empty")
    seen: set[str] = set()
    for condition in conditions:
        if condition.condition_id in seen:
            raise ValueError(f"Duplicate condition: {condition.condition_id}")
        seen.add(condition.condition_id)
        if len(condition.model_revision) < 7:
            raise ValueError(
                f"Condition {condition.condition_id} needs an immutable revision"
            )
        if condition.download_bytes <= 0:
            raise ValueError("download_bytes must be positive")
    return ConditionsConfig(
        runtime_image=str(payload["runtime_image"]),
        runtime_base_digest=str(payload["runtime_base_digest"]),
        transformers_revision=str(payload["transformers_revision"]),
        conditions=conditions,
    )


def build_rocm_runner_command(
    *,
    image: str,
    repo_root: Path,
    dataset_root: Path,
    hf_cache_root: Path,
    condition: Condition,
    source_root: Path | None = None,
) -> list[str]:
    """Build a rootless, offline Podman invocation for one controlled run."""

    roots = tuple(
        path.resolve(strict=True) for path in (repo_root, dataset_root, hf_cache_root)
    )
    repo, dataset, cache = roots
    runtime_source_root = (
        source_root.resolve(strict=True)
        if source_root is not None
        else (repo / "src").resolve()
    )
    media_root = (dataset / "vqa/2025/training/videos_original").resolve()
    evidence_root = (dataset / "vqa/benchmarks/evidence").resolve()
    runtime_image = condition.runtime_image or image
    command = [
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
        f"{runtime_source_root}:{runtime_source_root}:ro",
        "-v",
        f"{media_root}:{media_root}:ro",
        "-v",
        f"{evidence_root}:{evidence_root}:ro",
        "-v",
        f"{cache}:/root/.cache/huggingface:rw",
        "-w",
        str(runtime_source_root),
        "-e",
        f"PYTHONPATH={runtime_source_root}",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        f"VQA_RUNTIME_IMAGE={runtime_image}",
        runtime_image,
        "python",
        "-m",
    ]
    if condition.runner_kind == "qwen-omni":
        command.append("trec_video_2026.vqa_omni_runner")
    elif condition.runner_kind in {"internvl", "minicpm"}:
        command.extend(
            [
                "trec_video_2026.vqa_specialist_runner",
                "--kind",
                condition.runner_kind,
            ]
        )
    else:
        command.extend(
            [
                "trec_video_2026.vqa_transformers_runner",
                "--model-class",
                condition.runner_kind,
            ]
        )
    command.extend(
        [
            "--model-id",
            condition.model_id,
            "--revision",
            condition.model_revision,
            "--max-new-tokens",
            "32",
        ]
    )
    return command


@dataclass(frozen=True)
class RunnerRequest:
    """Inference-only payload; reference labels deliberately have no field."""

    run_id: str
    condition_id: str
    video_id: str
    question: str
    frame_paths: tuple[Path, ...]
    frame_timestamps: tuple[float, ...]
    transcript: str | None
    media_path: Path | None
    audio_path: Path | None = None
    frame_descriptions: tuple[str, ...] | None = None
    audio_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.condition_id or not self.video_id:
            raise ValueError("Runner request identity fields cannot be empty")
        if not self.question.strip():
            raise ValueError("Runner request question cannot be empty")
        if len(self.frame_paths) != len(self.frame_timestamps):
            raise ValueError("Frame paths and timestamps must have equal length")
        if self.frame_descriptions is not None and len(self.frame_descriptions) != len(
            self.frame_paths
        ):
            raise ValueError("Frame descriptions and paths must have equal length")
        if any(timestamp < 0 for timestamp in self.frame_timestamps):
            raise ValueError("Frame timestamps cannot be negative")
        if self.audio_mode is not None and self.audio_mode not in AUDIO_MODES:
            raise ValueError("Runner request audio_mode must be audio-off or audio-on")

    def to_dict(self) -> dict[str, Any]:
        return {
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
        }


def build_runner_request(
    *,
    condition: Condition,
    run_id: str,
    video_id: str,
    question: str,
    frame_paths: tuple[Path, ...],
    frame_timestamps: tuple[float, ...],
    transcript: str | None,
    media_path: Path | None,
    audio_path: Path | None = None,
    frame_descriptions: tuple[str, ...] | None = None,
) -> RunnerRequest:
    """Construct a label-free request from a configured model condition."""

    return RunnerRequest(
        run_id=run_id,
        condition_id=condition.condition_id,
        video_id=video_id,
        question=question,
        frame_paths=frame_paths,
        frame_timestamps=frame_timestamps,
        transcript=transcript,
        media_path=media_path,
        audio_path=audio_path,
        frame_descriptions=frame_descriptions,
        audio_mode=condition.audio_mode,
    )


def validate_runner_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the one-result JSON runner interface."""

    missing = sorted(REQUIRED_RESULT_FIELDS - payload.keys())
    if missing:
        raise ValueError(f"Runner result is missing {', '.join(missing)}")
    status = payload["status"]
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown runner status: {status!r}")
    answer = payload["answer"]
    if status == "passed" and (not isinstance(answer, str) or not answer.strip()):
        raise ValueError("Passed runner result requires a non-empty answer")
    if status != "passed" and answer is not None:
        raise ValueError("Non-passed runner result must have a null answer")
    elapsed = payload["elapsed_seconds"]
    if not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise ValueError("elapsed_seconds must be non-negative")
    revision = payload["model_revision"]
    if not isinstance(revision, str) or len(revision) < 7:
        raise ValueError("model_revision must be an immutable revision identifier")
    return dict(payload)


def invoke_runner(
    command: Sequence[str], request: RunnerRequest, *, timeout_seconds: float
) -> dict[str, Any]:
    """Invoke one isolated runner and classify process/protocol failures."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    invocation, container_name = _prepare_invocation(command, [request.to_dict()])
    started = time.monotonic()
    try:
        completed = subprocess.run(
            invocation,
            input=json.dumps(request.to_dict()),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result = _orchestration_failure(
            request,
            "timed_out",
            time.monotonic() - started,
            "TimeoutExpired",
            f"Runner exceeded {timeout_seconds:g} seconds",
        )
        return _record_timeout_cleanup(result, container_name)

    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        message = _bounded(completed.stderr or completed.stdout)
        return _orchestration_failure(
            request,
            "failed",
            elapsed,
            "RunnerProcessError",
            message or f"Runner exited {completed.returncode}",
        )
    stdout = completed.stdout.strip()
    try:
        decoded = json.loads(stdout)
        return validate_runner_result(decoded)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return _orchestration_failure(
            request,
            "failed",
            elapsed,
            "RunnerProtocolError",
            _bounded(str(error)),
        )


def invoke_runner_batch(
    command: Sequence[str],
    requests: Sequence[RunnerRequest],
    *,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Invoke one model process for multiple ordered requests."""

    if not requests:
        raise ValueError("Batch requests cannot be empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    request_payloads = [request.to_dict() for request in requests]
    invocation, container_name = _prepare_invocation(command, request_payloads)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            invocation,
            input=json.dumps(request_payloads),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        cleanup = _cleanup_timed_out_container(container_name)
        return [
            _timeout_failure_with_cleanup(
                _orchestration_failure(
                    request,
                    "timed_out",
                    time.monotonic() - started,
                    "TimeoutExpired",
                    f"Runner batch exceeded {timeout_seconds:g} seconds",
                ),
                container_name,
                cleanup,
            )
            for request in requests
        ]
    if completed.returncode != 0:
        message = _bounded(completed.stderr or completed.stdout)
        return [
            _orchestration_failure(
                request,
                "failed",
                time.monotonic() - started,
                "RunnerProcessError",
                message or f"Runner exited {completed.returncode}",
            )
            for request in requests
        ]
    try:
        decoded = json.loads(completed.stdout.strip())
        if not isinstance(decoded, list) or len(decoded) != len(requests):
            raise ValueError("Runner batch result count does not match requests")
        return [validate_runner_result(item) for item in decoded]
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return [
            _orchestration_failure(
                request,
                "failed",
                time.monotonic() - started,
                "RunnerProtocolError",
                _bounded(str(error)),
            )
            for request in requests
        ]


def _orchestration_failure(
    request: RunnerRequest,
    status: str,
    elapsed: float,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "answer": None,
        "elapsed_seconds": elapsed,
        "model_id": request.condition_id,
        "model_revision": "orchestrator",
        "runtime": "subprocess",
        "precision": "unknown",
        "device": "unknown",
        "peak_memory_bytes": None,
        "error_type": error_type,
        "error_message": error_message,
    }


def _prepare_invocation(
    command: Sequence[str], requests: Sequence[Mapping[str, Any]]
) -> tuple[list[str], str | None]:
    invocation = list(command)
    if invocation[:2] != ["podman", "run"]:
        return invocation, None
    if "--name" in invocation:
        raise ValueError("Podman runner command must not supply its own --name")
    identity = {
        "command": invocation,
        "requests": list(requests),
    }
    request_identity = canonical_request_sha256(identity)[:12]
    container_name = f"trec-vqa-{request_identity}-{secrets.token_hex(6)}"
    invocation[2:2] = ["--name", container_name]
    return invocation, container_name


def _cleanup_timed_out_container(container_name: str | None) -> dict[str, Any] | None:
    if container_name is None:
        return None

    def run_cleanup(command: list[str]) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            return {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "error": None,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            return {
                "returncode": None,
                "stdout": "",
                "error": type(error).__name__,
            }

    stop = run_cleanup(["podman", "stop", "--time", "10", container_name])
    remove = run_cleanup(["podman", "rm", "--force", container_name])
    exists = run_cleanup(["podman", "container", "exists", container_name])
    inspect: dict[str, Any] | None = None
    running_state = ""
    if exists["returncode"] == 0:
        inspect = run_cleanup(
            ["podman", "inspect", "--format", "{{.State.Running}}", container_name]
        )
        running_state = str(inspect["stdout"]).strip()
    verified = exists["returncode"] == 1 or (
        exists["returncode"] == 0
        and inspect is not None
        and inspect["returncode"] == 0
        and running_state.lower() == "false"
    )
    return {
        "stop_returncode": stop["returncode"],
        "stop_error": stop["error"],
        "remove_returncode": remove["returncode"],
        "remove_error": remove["error"],
        "exists_returncode": exists["returncode"],
        "exists_error": exists["error"],
        "inspect_returncode": inspect["returncode"] if inspect is not None else None,
        "inspect_error": inspect["error"] if inspect is not None else None,
        "running_state": running_state or None,
        "container_absent": exists["returncode"] == 1,
        "verified": verified,
    }


def _timeout_failure_with_cleanup(
    result: dict[str, Any],
    container_name: str | None,
    cleanup: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if container_name is None:
        return result
    result["container_name"] = container_name
    result["timeout_cleanup"] = dict(cleanup or {})
    if not cleanup or cleanup.get("verified") is not True:
        result["status"] = "failed"
        result["error_type"] = "TimeoutCleanupError"
        result["error_message"] = (
            "Runner timed out and Podman container cleanup could not be verified"
        )
    return result


def _record_timeout_cleanup(
    result: dict[str, Any], container_name: str | None
) -> dict[str, Any]:
    return _timeout_failure_with_cleanup(
        result, container_name, _cleanup_timed_out_container(container_name)
    )


def _bounded(text: str, limit: int = 4_000) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def select_latest_results(
    attempts: Sequence[Mapping[str, Any]],
    *,
    requests: Sequence[Mapping[str, Any]],
    validate_pass: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Select the latest exact-request results with validated pass provenance."""

    return {
        (str(result["condition_id"]), str(result["video_id"])): result
        for result in select_results_for_requests(
            requests, attempts, validate_pass=validate_pass
        )
    }


def select_results_for_requests(
    requests: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    *,
    validate_pass: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select only attempts bound to each exact request and valid provenance."""

    selected: list[dict[str, Any]] = []
    for request in requests:
        valid_pass: dict[str, Any] | None = None
        latest_failure: dict[str, Any] | None = None
        for attempt in attempts:
            if not (
                attempt.get("condition_id") == request.get("condition_id")
                and attempt.get("video_id") == request.get("video_id")
                and saved_request_matches(attempt, request)
            ):
                continue
            if attempt.get("status") != "passed":
                latest_failure = dict(attempt)
                continue
            try:
                validated = validate_pass(request, attempt)
            except (KeyError, TypeError, ValueError):
                continue
            valid_pass = dict(validated)
        chosen = valid_pass or latest_failure
        if chosen is not None:
            selected.append(chosen)
    return selected


def render_gate_report(
    *,
    gate_id: str,
    references: Mapping[str, str],
    attempts: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    validate_pass: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
) -> str:
    """Render a compact report from raw attempts; never imply statistical rank."""

    from trec_video_2026.vqa_evaluation import evaluate_local_answer

    selected = select_latest_results(
        attempts, requests=requests, validate_pass=validate_pass
    )
    lines = [
        f"# {gate_id} results",
        "",
        "This is a descriptive portability gate over three examples, not a statistical ranking.",
        "",
        "| Condition | Video | Status | Prediction | Reference | Normalized EM | Token F1 | Seconds | Peak memory |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for (condition_id, video_id), result in sorted(selected.items()):
        reference = references.get(video_id, "")
        prediction = result.get("answer") or ""
        if result.get("status") == "passed" and video_id in references:
            metrics = evaluate_local_answer(prediction, reference)
            exact = f"{metrics['normalized_exact_match']:.3f}"
            f1 = f"{metrics['token_f1']:.3f}"
        else:
            exact = "—"
            f1 = "—"
        elapsed = result.get("elapsed_seconds")
        elapsed_text = f"{elapsed:.2f}" if isinstance(elapsed, (int, float)) else "—"
        memory = result.get("peak_memory_bytes")
        memory_text = (
            f"{memory / (1024**3):.2f} GiB" if isinstance(memory, int) else "—"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(condition_id),
                    _markdown_cell(video_id),
                    _markdown_cell(str(result.get("status", "unknown"))),
                    _markdown_cell(prediction),
                    _markdown_cell(reference),
                    exact,
                    f1,
                    elapsed_text,
                    memory_text,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Normalized EM and token F1 are local diagnostics. Missing semantic metrics are not treated as zero.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


class GateStore:
    """Append-only attempt storage with resumable successful-result lookup."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.attempts_path = run_dir / "attempts.jsonl"

    def append_attempt(self, attempt: Mapping[str, Any]) -> None:
        required = {"run_id", "condition_id", "video_id", "status", "answer"}
        missing = sorted(required - attempt.keys())
        if missing:
            raise ValueError(f"Attempt is missing {', '.join(missing)}")
        request = attempt.get("request")
        if isinstance(request, Mapping):
            for field in ("run_id", "condition_id", "video_id"):
                if field in request and request[field] != attempt[field]:
                    raise ValueError(f"Attempt/request identity mismatch for {field}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(dict(attempt), sort_keys=True, ensure_ascii=False)
        with self.attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def completed(
        self,
        request: Mapping[str, Any],
        *,
        validate_pass: Callable[
            [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
        ],
    ) -> dict[str, Any] | None:
        if not self.attempts_path.exists():
            return None
        attempts: list[dict[str, Any]] = []
        with self.attempts_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    attempt = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on attempt line {line_number}"
                    ) from error
                attempts.append(attempt)
        selected = select_results_for_requests(
            [request], attempts, validate_pass=validate_pass
        )
        if selected and selected[0].get("status") == "passed":
            return selected[0]
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the three-video VQA gate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--dataset-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        from trec_video_2026.vqa_benchmark import load_manifest, probe_media

        manifest = load_manifest(args.config, args.dataset_root)
        examples = []
        for example in manifest.inference_examples:
            metadata = probe_media(example.media_path)
            examples.append(
                {
                    "video_id": example.video_id,
                    "category": example.category,
                    "media_path": str(example.media_path),
                    "duration_seconds": metadata.duration_seconds,
                    "has_video": metadata.has_video,
                    "has_audio": metadata.has_audio,
                }
            )
        print(
            json.dumps(
                {"gate_id": manifest.gate_id, "examples": examples},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
