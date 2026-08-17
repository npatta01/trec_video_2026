"""Isolated four-mode Qwen3-ASR runner for empty-output diagnostics.

The runner keeps model and audio dependencies inside :func:`main` while
sharing the established request, normalization, quality, and metric helpers
with the production audio runner.  Its language and timestamp switches are
arm-wide command-line controls, so every request in one process uses the same
decode surface.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_audio_runner import (
    _base_result,
    _model_with_parameters,
    audio_evidence_quality,
    classify_error_status,
    finalize_result_metrics,
    load_qwen,
    load_qwen_with_aligner,
    normalize_qwen_asr_result,
    validate_request,
    validate_rocm_bfloat16,
)


def transcribe_qwen_mode(
    model: Any,
    audio_path: str,
    *,
    language: str | None,
    timestamps: bool,
) -> dict[str, Any]:
    """Transcribe one waveform using one of the diagnostic decode surfaces."""

    kwargs: dict[str, Any] = {"audio": audio_path, "language": language}
    if timestamps:
        kwargs["return_time_stamps"] = True
    outputs = model.transcribe(**kwargs)
    if len(outputs) != 1:
        raise ValueError("Qwen3-ASR returned an unexpected result count")
    return normalize_qwen_asr_result(outputs[0])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--forced-aligner-id")
    parser.add_argument("--forced-aligner-revision")
    parser.add_argument("--language", choices=("auto", "English"), default="auto")
    parser.add_argument("--timestamps", choices=("on", "off"), default="off")
    args = parser.parse_args(argv)

    has_aligner_id = args.forced_aligner_id is not None
    has_aligner_revision = args.forced_aligner_revision is not None
    if has_aligner_id != has_aligner_revision:
        parser.error("forced aligner ID and revision must be provided together")
    if args.timestamps == "on" and not has_aligner_id:
        parser.error("timestamps on requires a forced aligner")
    return args


def main(argv: list[str] | None = None) -> int:
    """Read one request or ordered request batch and emit structured JSON."""

    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        import soundfile
        import torch
        import transformers

        from qwen_asr import Qwen3ASRModel
        from trec_video_2026.vqa_benchmark import source_fingerprint

        args = parse_args(argv)
        payload = json.load(sys.stdin)
        is_batch = isinstance(payload, list)
        requests = payload if is_batch else [payload]
        if not requests:
            raise ValueError("Qwen empty-output runner request batch cannot be empty")

        loaded_at = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        runtime = f"transformers {transformers.__version__}; torch {torch.__version__}"
        device = "unverified"
        precision = "unverified"
        try:
            if args.forced_aligner_id is not None:
                model = load_qwen_with_aligner(
                    Qwen3ASRModel,
                    args.model_id,
                    args.revision,
                    args.forced_aligner_id,
                    args.forced_aligner_revision,
                    dtype=torch.bfloat16,
                )
            else:
                model = load_qwen(
                    Qwen3ASRModel,
                    args.model_id,
                    args.revision,
                    dtype=torch.bfloat16,
                )
            device, precision = validate_rocm_bfloat16(
                _model_with_parameters(model), torch
            )
            torch.cuda.synchronize()
            load_seconds = time.monotonic() - loaded_at
        except Exception as error:  # noqa: BLE001 - protocol returns structured failures
            results = [
                {
                    **_base_result(
                        request,
                        args=args,
                        status=classify_error_status(error),
                        started=loaded_at,
                        runtime=runtime,
                        device=device,
                        precision=precision,
                        peak_memory=torch.cuda.max_memory_allocated(),
                        source_audio_sha256=None,
                        error=error,
                    ),
                    "model_load_seconds": None,
                }
                for request in requests
            ]
            print(
                json.dumps(results if is_batch else results[0], ensure_ascii=False),
                file=protocol_stdout,
            )
            return 0

        language = None if args.language == "auto" else args.language
        results = []
        for request in requests:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            audio_hash = None
            try:
                validate_request(request, model_kind="qwen3_asr")
                audio_path = Path(request["audio_path"])
                audio_hash = source_fingerprint(audio_path).sha256
                duration = soundfile.info(str(audio_path)).duration
                result = _base_result(
                    request,
                    args=args,
                    status="passed",
                    started=started,
                    runtime=runtime,
                    device=device,
                    precision=precision,
                    peak_memory=None,
                    source_audio_sha256=audio_hash,
                )
                output = transcribe_qwen_mode(
                    model,
                    str(audio_path),
                    language=language,
                    timestamps=args.timestamps == "on",
                )
                result.update(output)
                result["quality"] = audio_evidence_quality(
                    output["text"], duration_seconds=duration
                )
                finalize_result_metrics(
                    result, started=started, torch_module=torch
                )
            except Exception as error:  # noqa: BLE001 - preserve per-request failures
                result = _base_result(
                    request,
                    args=args,
                    status=classify_error_status(error),
                    started=started,
                    runtime=runtime,
                    device=device,
                    precision=precision,
                    peak_memory=torch.cuda.max_memory_allocated(),
                    source_audio_sha256=audio_hash,
                    error=error,
                )
            result["model_load_seconds"] = load_seconds
            results.append(result)

        print(
            json.dumps(results if is_batch else results[0], ensure_ascii=False),
            file=protocol_stdout,
        )
        return 0
    finally:
        sys.stdout = protocol_stdout


if __name__ == "__main__":
    raise SystemExit(main())
