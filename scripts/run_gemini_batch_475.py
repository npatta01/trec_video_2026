#!/usr/bin/env python3
"""Prepare, submit, and monitor the frozen 475-row Gemini native-AV batches."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

from google import genai
from google.genai import types

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trec_video_2026.vqa_frontier_map_media import (  # noqa: E402
    DeliveryMedia,
    DeliveryMediaError,
    prepare_delivery_media,
)


MODEL = "gemini-3.1-pro-preview"
FPS = 4.0
PANEL_LIMIT = 475
CONDITIONS = {"direct-answer", "transcript-map"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def validate_panel_size(rows: Sequence[dict[str, Any]]) -> None:
    if len(rows) > PANEL_LIMIT:
        raise ValueError("batch panel exceeds frozen 475-row boundary")


def mkv_delivery_strategy(video_codec: str, audio_codec: str | None) -> str:
    if video_codec in {"vp8", "vp9", "av1"} and audio_codec in {None, "opus", "vorbis"}:
        return "webm-copy"
    if audio_codec == "opus" and video_codec not in {"vp8", "vp9", "av1"}:
        return "mp4-audio-aac"
    return "mp4-copy"


def _av_codecs(path: Path) -> tuple[str, str | None]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    video = next(stream["codec_name"] for stream in streams if stream["codec_type"] == "video")
    audio = next((stream["codec_name"] for stream in streams if stream["codec_type"] == "audio"), None)
    return video, audio


def _video_packet_hash(path: Path) -> str:
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-c", "copy", "-f", "hash", "-hash", "sha256", "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().removeprefix("SHA256=")


def _prepare_mp4_fallback(source: Path, run_path: Path, video_id: str) -> DeliveryMedia:
    video_codec, audio_codec = _av_codecs(source)
    strategy = mkv_delivery_strategy(video_codec, audio_codec)
    if strategy == "webm-copy":
        raise DeliveryMediaError("WebM-compatible source unexpectedly failed strict delivery")
    delivery_root = run_path / "delivery-media"
    delivery_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = delivery_root / f"{video_id}.mp4"
    provenance = delivery_root / f"gemini-{video_id}.json"
    if target.is_file() and provenance.is_file():
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        return DeliveryMedia(
            path=target,
            provenance_path=provenance,
            mime_type="video/mp4",
            sha256=payload["delivery_sha256"],
            size_bytes=payload["delivery_size_bytes"],
            source_sha256=payload["source_sha256"],
            source_size_bytes=payload["source_size_bytes"],
            policy_version=payload["policy_version"],
            derived=True,
        )
    with tempfile.TemporaryDirectory(prefix=f".{video_id}-", dir=delivery_root) as temporary_dir:
        temporary = Path(temporary_dir) / target.name
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
            "-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-c:v", "copy",
        ]
        if strategy == "mp4-audio-aac":
            command.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            command.extend(["-c:a", "copy"])
        command.extend(["-movflags", "+faststart", str(temporary)])
        subprocess.run(command, check=True)
        source_video_hash = _video_packet_hash(source)
        delivery_video_hash = _video_packet_hash(temporary)
        if source_video_hash != delivery_video_hash:
            raise DeliveryMediaError("MP4 fallback changed the encoded video stream")
        payload = {
            "schema_version": 1,
            "policy_version": "gemini-mkv-mp4-compat-v1",
            "provider": "gemini",
            "video_id": video_id,
            "derived": True,
            "strategy": strategy,
            "source_name": source.name,
            "source_sha256": _sha256(source),
            "source_size_bytes": source.stat().st_size,
            "source_video_codec": video_codec,
            "source_audio_codec": audio_codec,
            "source_video_packet_sha256": source_video_hash,
            "delivery_name": target.name,
            "delivery_sha256": _sha256(temporary),
            "delivery_size_bytes": temporary.stat().st_size,
            "delivery_mime_type": "video/mp4",
            "delivery_video_packet_sha256": delivery_video_hash,
            "audio_transcoded": strategy == "mp4-audio-aac",
        }
        os.link(temporary, target)
        _write_once(provenance, payload)
    return DeliveryMedia(
        path=target,
        provenance_path=provenance,
        mime_type="video/mp4",
        sha256=payload["delivery_sha256"],
        size_bytes=payload["delivery_size_bytes"],
        source_sha256=payload["source_sha256"],
        source_size_bytes=payload["source_size_bytes"],
        policy_version=payload["policy_version"],
        derived=True,
    )


def _direct_prompt(question: str) -> str:
    return f"""Answer the question using the complete native video and audio. Identify the requested relationship and inspect all relevant moments. For temporal questions, include multiple supported subsequent events when the wording is ambiguous rather than stopping at an incidental micro-action. Give one concise best-supported answer only.

Question: {question}"""


def _map_prompt(duration_seconds: float) -> str:
    return f"""Watch and listen to the complete {duration_seconds:.3f}-second video from beginning to end. Create a question-blind evidence map. Do not answer or anticipate any question.

Represent the overall activity, beginning/middle/end phases, every important scene and transition, repeated patterns and all their instances, state changes, entities and actions, visible text, non-speech sounds, and all intelligible speech. For speech, transcribe the words verbatim, preserve their order, and align each utterance to its time span. Use uncertainty instead of guessing. Cover the complete video, including brief and late events. Use precise evidence-based timestamps rather than schematic intervals.

Return concise plain text using only the applicable record types below, one record per line:
OVERVIEW | concise whole-video synthesis
SCENE | start_seconds-end_seconds | observable description
SPEECH | start_seconds-end_seconds | speaker_or_unknown | verbatim speech | language_or_unknown | uncertainty_or_none
TEXT | start_seconds-end_seconds | verbatim visible text | uncertainty_or_none
SOUND | start_seconds-end_seconds | observable non-speech sound | uncertainty_or_none
PATTERN | count | instance_spans | repeated pattern
STATE | start_seconds-end_seconds | entity | observable state or change
UNCERTAINTY | span_or_global | unresolved detail

Do not omit a record type merely because another record already mentions the evidence."""


def build_inlined_request(
    *,
    condition: str,
    video_id: str,
    question: str,
    duration_seconds: float,
    file_uri: str,
    mime_type: str,
) -> types.InlinedRequest:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    prompt = (
        _direct_prompt(question)
        if condition == "direct-answer"
        else _map_prompt(duration_seconds)
    )
    maximum = 2048 if condition == "direct-answer" else 8192
    video = types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=FPS),
        media_resolution=types.PartMediaResolution(
            level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
        ),
    )
    return types.InlinedRequest(
        contents=types.Content(
            role="user", parts=[video, types.Part(text=prompt)]
        ),
        metadata={"video_id": video_id, "condition": condition},
        config=types.GenerateContentConfig(
            temperature=0.0,
            seed=42,
            max_output_tokens=maximum,
            response_mime_type="text/plain",
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=False,
            ),
        ),
    )


def prepare_protocol(panel: Path, media_dir: Path, root: Path) -> None:
    rows = [json.loads(line) for line in panel.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_panel_size(rows)
    if len(rows) != PANEL_LIMIT or len({row["video_id"] for row in rows}) != PANEL_LIMIT:
        raise ValueError("expected exactly 475 unique frozen rows")
    frozen = []
    for row in rows:
        candidates = sorted(media_dir.glob(f"{row['video_id']}.*"))
        candidates = [path for path in candidates if path.suffix.lower() in {".mkv", ".mp4", ".mov", ".mpeg", ".mpg", ".webm"}]
        if len(candidates) != 1:
            raise ValueError(f"expected one source video for {row['video_id']}")
        source = candidates[0].resolve()
        source_sha256 = _sha256(source)
        if source_sha256 != row["source_media_sha256"]:
            raise ValueError(f"source hash mismatch for {row['video_id']}")
        frozen.append(
            {
                "video_id": row["video_id"],
                "question": row["question"],
                "question_stratum": row["question_stratum"],
                "duration_seconds": row["source_duration_seconds"],
                "source_path": str(source),
                "source_sha256": source_sha256,
            }
        )
    _write_once(
        root / "protocol.json",
        {
            "schema_version": 1,
            "experiment_id": "direct-gemini-batch-475-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "model": MODEL,
            "conditions": sorted(CONDITIONS),
            "fps": FPS,
            "media_resolution": "HIGH",
            "thinking_level": "HIGH",
            "temperature": 0.0,
            "seed": 42,
            "label_isolation": "questions and media only; references unavailable until direct predictions freeze",
            "cost_boundary_usd": 30.0,
            "panel_path": str(panel.resolve()),
            "panel_sha256": _sha256(panel),
            "rows": frozen,
        },
    )


def _selected_rows(root: Path, video_ids: Sequence[str]) -> list[dict[str, Any]]:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    lookup = {row["video_id"]: row for row in protocol["rows"]}
    missing = sorted(set(video_ids) - set(lookup))
    if missing:
        raise ValueError(f"video IDs not in frozen panel: {missing}")
    rows = [lookup[video_id] for video_id in video_ids]
    validate_panel_size(rows)
    return rows


def _upload(client: Any, media: Path, video_id: str) -> Any:
    uploaded = client.files.upload(
        file=str(media),
        config={"mime_type": "video/webm", "display_name": f"vqa-batch-{video_id}"},
    )
    for _ in range(180):
        uploaded = client.files.get(name=uploaded.name)
        state = getattr(uploaded.state, "name", str(uploaded.state))
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            raise RuntimeError(f"Google file processing failed for {video_id}")
        time.sleep(1)
    raise TimeoutError(f"Google file processing timed out for {video_id}")


def upload_rows(root: Path, video_ids: Sequence[str]) -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for row in _selected_rows(root, video_ids):
        record = root / "uploads" / f"{row['video_id']}.json"
        if record.is_file():
            continue
        delivery_root = root / "deliveries" / row["video_id"]
        try:
            delivery = prepare_delivery_media(
                run_path=delivery_root,
                source_path=Path(row["source_path"]),
                video_id=row["video_id"],
                provider="gemini",
            )
        except DeliveryMediaError:
            delivery = _prepare_mp4_fallback(
                Path(row["source_path"]), delivery_root, row["video_id"]
            )
        uploaded = _upload(client, delivery.path, row["video_id"])
        _write_once(
            record,
            {
                "video_id": row["video_id"],
                "source_sha256": row["source_sha256"],
                "delivery_path": str(delivery.path),
                "delivery_sha256": delivery.sha256,
                "delivery_mime_type": delivery.mime_type,
                "file_name": uploaded.name,
                "file_uri": uploaded.uri,
                "file_state": getattr(uploaded.state, "name", str(uploaded.state)),
                "uploaded_at": datetime.now(UTC).isoformat(),
                "expiration_time": str(getattr(uploaded, "expiration_time", None)),
            },
        )
        print(json.dumps({"video_id": row["video_id"], "status": "uploaded"}), flush=True)


def _job_payload(job: Any, *, condition: str, video_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "condition": condition,
        "video_ids": list(video_ids),
        "submitted_at": datetime.now(UTC).isoformat(),
        "job": job.model_dump(mode="json", by_alias=True, exclude_none=True),
    }


def submit(root: Path, condition: str, video_ids: Sequence[str], shard: str) -> None:
    rows = _selected_rows(root, video_ids)
    requests = []
    for row in rows:
        upload_path = root / "uploads" / f"{row['video_id']}.json"
        if not upload_path.is_file():
            raise FileNotFoundError(f"missing upload record for {row['video_id']}")
        upload = json.loads(upload_path.read_text(encoding="utf-8"))
        requests.append(
            build_inlined_request(
                condition=condition,
                video_id=row["video_id"],
                question=row["question"],
                duration_seconds=float(row["duration_seconds"]),
                file_uri=upload["file_uri"],
                mime_type=upload["delivery_mime_type"],
            )
        )
    record = root / "jobs" / f"{shard}-{condition}.json"
    if record.exists():
        raise FileExistsError(f"refusing duplicate paid submission: {record}")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    job = client.batches.create(
        model=MODEL,
        src=requests,
        config={"display_name": f"vqa-475-{shard}-{condition}"},
    )
    _write_once(record, _job_payload(job, condition=condition, video_ids=video_ids))
    print(json.dumps({"condition": condition, "job_name": job.name, "state": str(job.state), "record": str(record)}))


def status(root: Path, record_name: str) -> None:
    record_path = root / "jobs" / record_name
    record = json.loads(record_path.read_text(encoding="utf-8"))
    name = record["job"]["name"]
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    job = client.batches.get(name=name)
    snapshot = {
        "checked_at": datetime.now(UTC).isoformat(),
        "name": job.name,
        "state": getattr(job.state, "name", str(job.state)),
        "job": job.model_dump(mode="json", by_alias=True, exclude_none=True),
    }
    _write_once(root / "status" / record_path.stem / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json", snapshot)
    print(json.dumps({"name": job.name, "state": snapshot["state"]}))


def collect_terminal_outputs(root: Path, condition: str) -> list[dict[str, Any]]:
    """Collect one successful terminal response per frozen row, allowing failed-row retries."""

    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    expected = {row["video_id"]: row for row in protocol["rows"]}
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, list[dict[str, Any]]] = {}
    terminal_jobs: list[tuple[Path, dict[str, Any]]] = []
    for status_dir in sorted((root / "status").glob(f"*-{condition}")):
        terminal = []
        for snapshot_path in sorted(status_dir.glob("*.json")):
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if payload.get("state") == "JOB_STATE_SUCCEEDED":
                terminal.append((snapshot_path, payload))
        if not terminal:
            continue
        snapshot_path, payload = terminal[-1]
        terminal_jobs.append((snapshot_path, payload))

    # Retry directories can sort before their original shards. Collect every
    # terminal error first so successful retry provenance is order-independent.
    for _, payload in terminal_jobs:
        for item in payload["job"].get("dest", {}).get("inlinedResponses", []):
            metadata = item.get("metadata", {})
            video_id = metadata.get("video_id")
            if video_id not in expected or metadata.get("condition") != condition:
                raise ValueError("terminal response metadata is outside the frozen condition")
            if item.get("error") is not None:
                failures.setdefault(video_id, []).append(item["error"])

    for snapshot_path, payload in terminal_jobs:
        job = payload["job"]
        for item in job.get("dest", {}).get("inlinedResponses", []):
            metadata = item.get("metadata", {})
            video_id = metadata.get("video_id")
            if video_id not in expected or metadata.get("condition") != condition:
                raise ValueError("terminal response metadata is outside the frozen condition")
            if item.get("error") is not None:
                continue
            response = item.get("response", {})
            candidates = response.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text = "".join(part.get("text", "") for part in parts if part.get("text"))
            if not text.strip():
                raise ValueError(f"empty successful response for {video_id}")
            if video_id in successes:
                raise ValueError(f"duplicate successful response for {video_id}")
            successes[video_id] = {
                "video_id": video_id,
                "question": expected[video_id]["question"],
                "question_stratum": expected[video_id]["question_stratum"],
                "text": text.strip(),
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "usage_metadata": response.get("usageMetadata", {}),
                "model_version": response.get("modelVersion"),
                "response_id": response.get("responseId"),
                "job_name": job.get("name"),
                "terminal_snapshot": str(snapshot_path.resolve()),
                "terminal_snapshot_sha256": _sha256(snapshot_path),
                "prior_errors": failures.get(video_id, []),
            }
    missing = [row["video_id"] for row in protocol["rows"] if row["video_id"] not in successes]
    if missing:
        raise ValueError(f"missing successful {condition} outputs: {missing}")
    return [successes[row["video_id"]] for row in protocol["rows"]]


def freeze(root: Path) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": "direct-gemini-batch-475-v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": _sha256(root / "protocol.json"),
        "conditions": {
            condition: collect_terminal_outputs(root, condition)
            for condition in sorted(CONDITIONS)
        },
    }
    for condition, rows in payload["conditions"].items():
        if len(rows) != PANEL_LIMIT or len({row["video_id"] for row in rows}) != PANEL_LIMIT:
            raise ValueError(f"invalid frozen row boundary for {condition}")
    _write_once(root / "predictions-freeze.json", payload)
    print(json.dumps({condition: len(rows) for condition, rows in payload["conditions"].items()}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--panel", type=Path, required=True)
    prepare.add_argument("--media-dir", type=Path, required=True)
    prepare.add_argument("--root", type=Path, required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--root", type=Path, required=True)
    upload.add_argument("video_ids", nargs="+")
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--root", type=Path, required=True)
    submit_parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    submit_parser.add_argument("--shard", required=True)
    submit_parser.add_argument("video_ids", nargs="+")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--root", type=Path, required=True)
    status_parser.add_argument("record_name")
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_protocol(args.panel, args.media_dir, args.root)
    elif args.command == "upload":
        upload_rows(args.root, args.video_ids)
    elif args.command == "submit":
        submit(args.root, args.condition, args.video_ids, args.shard)
    elif args.command == "status":
        status(args.root, args.record_name)
    else:
        freeze(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
