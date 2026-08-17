#!/usr/bin/env python3
"""Whisper large-v3 transcript evidence for the frozen 2026 test media.

The runner is deliberately media-oriented: each unique video is transcribed
once, and its append-only JSONL result can be resumed after a preemption.  It
does not read answers or annotations.  The final ``accepted-transcripts.json``
is the small adapter consumed by the Qwen3.6 deadline runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

MODEL_ID = "openai/whisper-large-v3"
MODEL_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
TRANSFORMERS_REVISION = "cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
INPUT_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
    "experiments/deadline-qwen36-full-inputs-20260816-a2/input"
)
DEFAULT_OUTPUT_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
    "experiments/deadline-qwen36-whisper-full-20260816-a2/output"
)
EXPECTED_MEDIA_COUNT = 1_946
EXPECTED_NO_AUDIO_COUNT = 5
MEDIA_SUFFIXES = {".mkv", ".webm", ".mp4", ".mov", ".avi"}
TERMINAL_RESULT_STATUSES = {"passed", "no-audio-stream"}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_media(path: Path) -> dict[str, Any]:
    """Return stream presence and duration without decoding the whole video."""

    try:
        import av
    except ImportError:  # local no-model preview fallback; dstack uses PyAV above
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,duration",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        durations = [
            float(stream["duration"])
            for stream in streams
            if isinstance(stream, dict) and stream.get("duration") not in (None, "N/A")
        ]
        raw_duration = payload.get("format", {}).get("duration")
        duration = (
            float(raw_duration)
            if raw_duration not in (None, "N/A")
            else max(durations, default=0.0)
        )
        return {
            "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
            "duration_seconds": duration,
        }

    with av.open(str(path)) as container:
        duration = None
        if container.duration is not None:
            duration = float(container.duration) / 1_000_000.0
        if duration is None or duration <= 0:
            durations = [
                float(stream.duration * stream.time_base)
                for stream in container.streams
                if stream.duration is not None and stream.time_base is not None
            ]
            duration = max(durations, default=0.0)
        return {
            "has_audio": bool(container.streams.audio),
            "duration_seconds": duration,
        }


def _contract_records(root: Path) -> list[dict[str, Any]] | None:
    contract = root / "media-contract.json"
    if not contract.is_file():
        return None
    payload = _load_json(contract)
    media = payload.get("media")
    if not isinstance(media, list):
        raise ValueError("media-contract.json must contain a media list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in media:
        if not isinstance(item, dict):
            raise ValueError("media-contract.json contains a non-object media row")
        video_id = item.get("video_id")
        raw_path = item.get("path")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("media contract video_id must be a non-empty string")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"media contract path is missing for {video_id}")
        if video_id in seen:
            raise ValueError(f"duplicate video_id in media contract: {video_id}")
        seen.add(video_id)
        path = (root / raw_path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"media contract path escapes input root: {raw_path}")
        if not path.is_file():
            raise FileNotFoundError(f"media file from contract is missing: {path}")
        record = {
            "video_id": video_id,
            "path": path,
            "source_sha256": item.get("source_sha256"),
            "size_bytes": item.get("size_bytes"),
            "source_format": item.get("source_format"),
        }
        records.append(record)
    return records


def _directory_records(root: Path) -> list[dict[str, Any]]:
    paths = sorted(
        {
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
        },
        key=lambda path: str(path),
    )
    records = []
    seen: set[str] = set()
    for path in paths:
        video_id = path.stem
        if video_id in seen:
            raise ValueError(
                f"duplicate media stem without a media contract: {video_id}"
            )
        seen.add(video_id)
        records.append({"video_id": video_id, "path": path})
    return records


def scan_media(
    input_root: Path,
    *,
    probe: Callable[[Path], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Discover unique media and attach PyAV (or injected-test) metadata."""

    input_root = input_root.resolve()
    records = _contract_records(input_root) or _directory_records(input_root)
    probe = probe or _probe_media
    for record in records:
        metadata = probe(Path(record["path"]))
        if not isinstance(metadata.get("has_audio"), bool):
            raise ValueError(
                f"audio probe did not return bool for {record['video_id']}"
            )
        record.update(
            {
                "has_audio": metadata["has_audio"],
                "duration_seconds": float(metadata.get("duration_seconds", 0.0)),
            }
        )
    records.sort(key=lambda row: str(row["video_id"]))
    return records


def verify_media_contract(
    input_root: Path,
    records: list[dict[str, Any]],
    *,
    expected_count: int = EXPECTED_MEDIA_COUNT,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Check frozen contract cardinality, paths, sizes, and optional hashes."""

    contract = input_root / "media-contract.json"
    if not contract.is_file():
        return {
            "contract_present": False,
            "media_count": len(records),
            "hashes_verified": False,
        }
    payload = _load_json(contract)
    contract_count = payload.get("unique_media_count", len(payload.get("media", [])))
    if contract_count != len(records):
        raise ValueError(
            f"media contract count {contract_count} does not match discovered {len(records)}"
        )
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"expected {expected_count} media files, discovered {len(records)}"
        )
    missing_size = []
    for record in records:
        expected_size = record.get("size_bytes")
        if (
            isinstance(expected_size, int)
            and Path(record["path"]).stat().st_size != expected_size
        ):
            missing_size.append(record["video_id"])
    if missing_size:
        raise ValueError(f"media size mismatch for {len(missing_size)} contract rows")
    verified = False
    if verify_hashes:
        for record in records:
            expected = record.get("source_sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"missing source_sha256 for {record['video_id']}")
            actual = sha256_file(Path(record["path"]))
            if actual != expected:
                raise ValueError(f"media hash mismatch for {record['video_id']}")
        verified = True
    return {
        "contract_present": True,
        "media_count": len(records),
        "contract_unique_media_count": contract_count,
        "hashes_verified": verified,
        "expected_no_audio_count": EXPECTED_NO_AUDIO_COUNT,
        "observed_no_audio_count": sum(not row["has_audio"] for row in records),
    }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_resumable_results(path: Path) -> dict[str, dict[str, Any]]:
    """Load append-only results and return terminal rows keyed by video ID."""

    if not path.exists():
        return {}
    terminal: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            video_id = row.get("video_id") if isinstance(row, dict) else None
            if not isinstance(video_id, str) or not video_id:
                raise ValueError(f"missing video_id at {path}:{line_number}")
            if video_id in seen:
                raise ValueError(f"duplicate video_id in resumable results: {video_id}")
            seen.add(video_id)
            if row.get("status") in TERMINAL_RESULT_STATUSES:
                terminal[video_id] = row
    return terminal


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_dump(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _accepted_row(row: dict[str, Any]) -> bool:
    selected = row.get("selected_transcript")
    quality = row.get("quality")
    if (
        row.get("status") != "passed"
        or not isinstance(selected, str)
        or not selected.strip()
    ):
        return False
    if isinstance(quality, dict):
        if quality.get("accepted") is True or quality.get("status") == "accepted":
            return True
        return False
    return row.get("transcript_included") is True


def build_accepted_adapter(
    rows: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    if isinstance(rows, dict):
        values = list(rows.values())
    else:
        values = rows
    accepted = [
        {
            "video_id": str(row["video_id"]),
            "transcript": str(row["selected_transcript"]).strip(),
        }
        for row in values
        if _accepted_row(row)
    ]
    accepted.sort(key=lambda row: row["video_id"])
    return {"rows": accepted}


def preview_input(
    input_root: Path,
    *,
    probe: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = scan_media(input_root, probe=probe)
    return {
        "input_root": str(input_root.resolve()),
        "media_count": len(records),
        "audio_count": sum(row["has_audio"] for row in records),
        "no_audio_count": sum(not row["has_audio"] for row in records),
        "total_duration_seconds": sum(row["duration_seconds"] for row in records),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transcription_mode": "translate-only",
        "input_uri": INPUT_URI,
    }


def _extract_audio(media_path: Path, wav_path: Path) -> float:
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - dstack-only
        raise RuntimeError("imageio-ffmpeg is required for audio extraction") from exc
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = wav_path.with_suffix(".tmp.wav")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = build_ffmpeg_command(ffmpeg, media_path, temporary)
    try:
        subprocess.run(command, check=True)
        temporary.replace(wav_path)
    finally:
        temporary.unlink(missing_ok=True)
    try:
        import soundfile
    except ImportError as exc:  # pragma: no cover - dstack-only
        raise RuntimeError("soundfile is required after audio extraction") from exc
    _, sample_rate = soundfile.read(str(wav_path), dtype="float32", always_2d=False)
    if sample_rate != 16_000:
        raise RuntimeError(f"extracted audio is not 16000 Hz: {sample_rate}")
    info = soundfile.info(str(wav_path))
    return float(info.duration)


def build_ffmpeg_command(
    ffmpeg: str | Path, media_path: Path, temporary_wav_path: Path
) -> list[str]:
    """Build an explicit WAV extraction command (the suffix must remain .wav)."""

    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
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
        str(temporary_wav_path),
    ]


def _load_asr() -> tuple[Any, dict[str, Any]]:  # pragma: no cover - dstack-only
    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if not torch.cuda.is_available() or torch.version.cuda is None or torch.version.hip:
        raise RuntimeError("CUDA is unavailable or the runtime is not NVIDIA CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bfloat16 is unavailable")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch.bfloat16,
        device=0,
    )
    return asr, {
        "model_load_seconds": time.monotonic() - started,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": transformers.__version__,
        "transformers_revision": TRANSFORMERS_REVISION,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_bf16_supported": True,
    }


def _transcribe(asr: Any, wav_path: Path, duration_seconds: float) -> dict[str, Any]:
    import soundfile

    from trec_video_2026.vqa_whisper import transcribe_audio, transcript_quality

    audio, sample_rate = soundfile.read(str(wav_path), dtype="float32")
    if sample_rate != 16_000:
        raise RuntimeError(f"expected 16000 Hz audio, got {sample_rate}")
    if getattr(audio, "ndim", 1) != 1:
        raise RuntimeError("expected mono audio")
    result = transcribe_audio(asr, audio, task="translate")
    result["quality"] = transcript_quality(
        result["text"],
        duration_seconds=duration_seconds,
        chunks=result["chunks"],
        mode="translate",
        no_speech_probability=result["no_speech_probability"],
    )
    return result


def sync_output(output_root: Path, *, remote_uri: str | None = None) -> None:
    remote_uri = remote_uri or os.environ.get("HF_OUTPUT_URI")
    if not remote_uri:
        return
    hf = os.environ.get("HF_CLI", "hf")
    subprocess.run([hf, "buckets", "sync", str(output_root), remote_uri], check=True)


def _write_adapter(output_root: Path, rows: dict[str, dict[str, Any]]) -> Path:
    adapter_path = output_root / "requests.accepted-whisper.json"
    adapter_path.write_text(
        json.dumps(build_accepted_adapter(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return adapter_path


def run_transcription(args: argparse.Namespace) -> int:  # pragma: no cover - GPU path
    input_root = args.input_root.resolve()
    output_root = args.output.resolve()
    work_root = args.work_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    records = scan_media(input_root)
    contract_status = verify_media_contract(
        input_root,
        records,
        expected_count=args.expected_media_count,
        verify_hashes=args.verify_input_hashes,
    )
    no_audio_count = sum(not row["has_audio"] for row in records)
    if no_audio_count != EXPECTED_NO_AUDIO_COUNT:
        raise ValueError(
            f"expected {EXPECTED_NO_AUDIO_COUNT} no-audio videos, discovered {no_audio_count}"
        )
    (output_root / "preflight.json").write_text(
        json.dumps({**contract_status, **preview_input(input_root)}, indent=2) + "\n",
        encoding="utf-8",
    )
    results_path = output_root / "whisper-results.jsonl"
    failures_path = output_root / "whisper-failures.jsonl"
    completed = load_resumable_results(results_path)
    targets = records[: args.limit] if args.limit is not None else records
    pending_audio = [
        row for row in targets if row["video_id"] not in completed and row["has_audio"]
    ]
    asr = None
    runtime: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transcription_mode": "translate-only",
    }
    if pending_audio:
        asr, runtime = _load_asr()
    audio_root = work_root / "audio"
    processed_since_sync = 0
    started = time.monotonic()
    for index, record in enumerate(targets, start=1):
        video_id = str(record["video_id"])
        if video_id in completed:
            continue
        row_started = time.monotonic()
        if not record["has_audio"]:
            result = {
                "status": "no-audio-stream",
                "video_id": video_id,
                "media_path": str(record["path"]),
                "source_sha256": record.get("source_sha256"),
                "transcript_included": False,
                "selected_transcript": None,
                "elapsed_seconds": time.monotonic() - row_started,
            }
        else:
            try:
                audio_name = (
                    hashlib.sha256(video_id.encode("utf-8")).hexdigest() + ".wav"
                )
                wav_path = audio_root / audio_name
                duration = (
                    _extract_audio(Path(record["path"]), wav_path)
                    if not wav_path.exists()
                    else float(record["duration_seconds"])
                )
                if asr is None:
                    raise RuntimeError("ASR model was not loaded for an audio row")
                transcription = _transcribe(asr, wav_path, duration)
                accepted = transcription["quality"]["status"] == "accepted"
                selected = transcription["timestamped_text"] or transcription["text"]
                result = {
                    "status": "passed",
                    "video_id": video_id,
                    "media_path": str(record["path"]),
                    "source_sha256": record.get("source_sha256"),
                    "source_audio_duration_seconds": duration,
                    "transcript_included": accepted,
                    "text": transcription["text"],
                    "timestamped_text": transcription["timestamped_text"],
                    "selected_transcript": selected if accepted else None,
                    "quality": transcription["quality"],
                    "source_language": transcription.get("language"),
                    "diagnostics": transcription.get("diagnostics"),
                    "elapsed_seconds": time.monotonic() - row_started,
                }
            except Exception as exc:  # keep the rest of the panel resumable
                append_jsonl(
                    failures_path,
                    {
                        "video_id": video_id,
                        "media_path": str(record["path"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": time.monotonic() - row_started,
                    },
                )
                print(
                    json.dumps(
                        {"video_id": video_id, "status": "failed", "error": str(exc)}
                    ),
                    flush=True,
                )
                continue
        append_jsonl(results_path, result)
        completed[video_id] = result
        processed_since_sync += 1
        if processed_since_sync >= args.sync_every:
            sync_output(output_root, remote_uri=args.output_uri)
            processed_since_sync = 0
        if index % 10 == 0 or index == len(targets):
            print(
                json.dumps({"processed": index, "target_count": len(targets)}),
                flush=True,
            )
    adapter_path = _write_adapter(output_root, completed)
    summary = {
        "schema_version": 1,
        "label_policy": "label-free-transcript-evidence",
        "model": runtime,
        "transcription_mode": "translate-only",
        "media_count": len(records),
        "target_count": len(targets),
        "completed_count": len(completed),
        "accepted_count": sum(_accepted_row(row) for row in completed.values()),
        "no_audio_count": sum(
            row.get("status") == "no-audio-stream" for row in completed.values()
        ),
        "failed_count": sum(1 for _ in failures_path.open(encoding="utf-8"))
        if failures_path.exists()
        else 0,
        "wall_seconds": time.monotonic() - started,
        "accepted_adapter": str(adapter_path),
        "input_uri": INPUT_URI,
        "output_uri": args.output_uri
        or os.environ.get("HF_OUTPUT_URI")
        or DEFAULT_OUTPUT_URI,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    sync_output(output_root, remote_uri=args.output_uri)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/dstack/input"))
    parser.add_argument("--output", type=Path, default=Path("/dstack/output"))
    parser.add_argument("--work-dir", type=Path, default=Path("/dstack/work"))
    parser.add_argument("--output-uri", default=None)
    parser.add_argument("--sync-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--expected-media-count", type=int, default=EXPECTED_MEDIA_COUNT
    )
    parser.add_argument("--verify-input-hashes", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="scan/verify inputs without loading Whisper",
    )
    parser.add_argument(
        "--preview", action="store_true", help="print a no-model input summary"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sync_every < 1:
        raise SystemExit("--sync-every must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    preview = preview_input(args.input_root)
    if args.preview:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    if args.preflight:
        records = scan_media(args.input_root)
        result = verify_media_contract(
            args.input_root,
            records,
            expected_count=args.expected_media_count,
            verify_hashes=args.verify_input_hashes,
        )
        result.update(preview)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return run_transcription(args)


if __name__ == "__main__":
    raise SystemExit(main())
