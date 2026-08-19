from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_vqa_event_map_evidence.py"
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "vqa_event_map_evidence", BUILDER_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load event-map evidence builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _id_digest(video_ids: list[str]) -> str:
    return _sha256(("\n".join(video_ids) + "\n").encode("utf-8"))


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _fixture(tmp_path: Path) -> dict[str, Any]:
    video_id = "video01abcd"
    media = tmp_path / f"{video_id}.mp4"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"fixture-video-bytes")
    media_sha256 = _sha256(media.read_bytes())
    duration = 12.0
    panel = [
        {
            "video_id": video_id,
            "question": "What happens after the door opens?",
            "duration_seconds": duration,
            "media_path": str(media),
            "media_sha256": media_sha256,
            "whisper_segment_count": 3,
        }
    ]
    event_map = [
        {
            "schema_version": 1,
            "run_id": "event-map-canary",
            "condition_id": "av-map",
            "video_id": video_id,
            "question": panel[0]["question"],
            "status": "passed",
            "error_type": None,
            "source_media_sha256": media_sha256,
            "source_duration_seconds": duration,
            "response_sha256": "a" * 64,
            "answer": "The door opens.",
            "abstain": False,
            "confidence": 0.8,
            "events": [
                {
                    "event_id": "event-a",
                    "start": 0.0,
                    "end": duration,
                    "visual_summary": "Private summary.",
                    "audio_summary": None,
                }
            ],
            "selected_intervals": [{"event_id": "event-a", "start": 4.0, "end": 5.0}],
        }
    ]
    whisper_text = (
        "[00:00:00.000] Before.\n"
        "[00:00:04.000] The door opens.\n"
        "[00:00:08.000] After.\n"
    )
    whisper = [
        {
            "video_id": video_id,
            "status": "passed",
            "text": whisper_text,
            "timestamped_text": whisper_text,
            "selected_transcript": whisper_text,
            "transcript_included": True,
            "quality": {"status": "accepted"},
            "english_translation": {
                "text": "Before. The door opens. After.",
                "timestamped_text": whisper_text,
                "chunks": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Before.",
                        "language": "english",
                    },
                    {
                        "start": 4.0,
                        "end": 6.0,
                        "text": "The door opens.",
                        "language": "english",
                    },
                    {
                        "start": 8.0,
                        "end": 10.0,
                        "text": "After.",
                        "language": "english",
                    },
                ],
            },
            "source_sha256": _sha256(whisper_text.encode()),
        }
    ]
    inventory = [
        {
            "stream_index": 0,
            "frame_index": index,
            "decoded_frame_ordinal": index,
            "actual_pts_seconds": round(index / 10, 6),
            "time_base": "1/1000",
            "raw_pts": index * 100,
            "source_media_sha256": media_sha256,
        }
        for index in range(121)
    ]
    f0_frames = []
    f0_root = tmp_path / "frozen-f0" / video_id
    for ordinal, frame_index in enumerate(range(5, 116, 10), start=1):
        frame_path = f0_root / f"frame-{ordinal:02d}.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(f"frozen-f0-frame-{ordinal:02d}".encode())
        f0_frames.append(
            {
                "source_frame_ordinal": ordinal,
                "source_path": str(frame_path),
                "source_relative_path": f"evidence/frames/{video_id}/{frame_path.name}",
                "requested_timestamp_seconds": float(frame_index) / 10,
                "actual_pts_seconds": float(frame_index) / 10,
                "frame_byte_sha256": _sha256(frame_path.read_bytes()),
            }
        )
    f0 = [
        {
            "video_id": video_id,
            "question": panel[0]["question"],
            "duration_seconds": duration,
            "source_media_sha256": media_sha256,
            "source_artifact_sha256": "b" * 64,
            "source_request_sha256": "c" * 64,
            "source_evidence_sha256": "d" * 64,
            "source_frames_request_sha256": "e" * 64,
            "source_whisper_sha256": "f" * 64,
            "transcript": whisper_text,
            "transcript_sha256": _sha256(whisper_text.encode()),
            "frames": f0_frames,
        }
    ]
    return {
        "panel": panel,
        "event_map": event_map,
        "whisper": whisper,
        "inventory": {video_id: inventory},
        "f0": f0,
        "media_sha256": media_sha256,
        "video_id": video_id,
    }


def _freeze_identity() -> dict[str, Any]:
    return {
        "source_bundle_sha256": "1" * 64,
        "controller_sha256": "2" * 64,
        "protocol_sha256": "3" * 64,
        "inference_module_sha256": "4" * 64,
        "runner_sha256": "5" * 64,
        "model_id": "nvidia/Nemotron-fixture",
        "model_revision": "6" * 40,
        "runtime_image": "registry.invalid/vllm@sha256:" + "7" * 64,
        "runtime_identity": {
            "schema_version": 1,
            "runtime_image": "registry.invalid/vllm@sha256:" + "7" * 64,
            "vllm_version": "fixture",
        },
        "row_run_id": "event-map-input-fixture",
        "producer_run_prefix": "event-map-fixture",
        "output_uri_prefix": "hf://private/event-map-fixture",
    }


def _producer_row(fixture: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(fixture["event_map"][0])
    identity = _freeze_identity()
    row.update(
        {
            "run_id": identity["row_run_id"],
            "slice_id": "00",
            "question_stratum": "temporal",
            "media_path": f"media/{fixture['video_id']}.mp4",
            "normalized_media_sha256": "8" * 64,
            "normalized_duration_seconds": row["source_duration_seconds"],
            "request_sha256": "9" * 64,
            "controller_request_sha256": "a" * 64,
            "primary_request_sha256": "9" * 64,
            "retry_request_sha256": None,
            "primary_valid": True,
            "retry_valid": None,
            "retry_used": False,
            "model_id": identity["model_id"],
            "model_revision": identity["model_revision"],
            "runtime_image": identity["runtime_image"],
            "runtime_identity": identity["runtime_identity"],
            "memory_telemetry": {
                "path": "telemetry/nvidia-smi.csv",
                "sampling_interval_seconds": 5,
            },
            "elapsed_seconds": 1.25,
        }
    )
    return row


def _aggregate_freeze_manifest(
    fixture: dict[str, Any], *, shard_counts: tuple[int, int, int, int] = (1, 0, 0, 0)
) -> dict[str, Any]:
    identity = _freeze_identity()
    rows = [_producer_row(fixture)]
    terminals = []
    terminal_hashes: dict[tuple[str, int], str] = {}
    for condition in ("v-map", "av-map"):
        for shard_index, row_count in enumerate(shard_counts):
            run_id = (
                f"{identity['producer_run_prefix']}-{condition}-chunk-{shard_index:02d}"
            )
            terminal = {
                "schema_version": 1,
                "status": "complete",
                "run_id": run_id,
                "attempt_uuid": f"00000000-0000-4000-8000-{shard_index + (4 if condition == 'av-map' else 0):012d}",
                "output_identity": f"{identity['output_uri_prefix']}-{run_id}",
                "payload_checksum": f"{shard_index + (8 if condition == 'av-map' else 0):064x}",
                "row_count": row_count,
                "source_bundle_sha256": identity["source_bundle_sha256"],
                "controller_sha256": identity["controller_sha256"],
                "protocol_sha256": identity["protocol_sha256"],
                "request_file_sha256": f"{16 + shard_index + (4 if condition == 'av-map' else 0):064x}",
                "inference_module_sha256": identity["inference_module_sha256"],
                "runner_sha256": identity["runner_sha256"],
                "producer_timings": {
                    "schema_version": 1,
                    "clock": "monotonic",
                    "bootstrap_download_seconds": 1.0,
                    "normalization": {"media_count": max(row_count, 1), "seconds": 1.0},
                    "model_startup_seconds": 1.0,
                    "inference_seconds": 1.0,
                    "finalization_seconds": 1.0,
                    "phase_total_seconds": 5.0,
                },
            }
            terminal_sha256 = _canonical_sha256(terminal)
            terminal_hashes[(condition, shard_index)] = terminal_sha256
            terminals.append(
                {
                    "condition_id": condition,
                    "shard_index": shard_index,
                    "terminal_sha256": terminal_sha256,
                    "terminal": terminal,
                }
            )
    shard_bindings = []
    start = 0
    for shard_index, row_count in enumerate(shard_counts):
        stop = start + row_count
        shard_rows = rows[start:stop]
        shard_bindings.append(
            {
                "shard_index": shard_index,
                "row_start": start,
                "row_stop": stop,
                "row_count": row_count,
                "rows_sha256": _canonical_sha256(shard_rows),
                "predictions_sha256": f"{32 + shard_index:064x}",
                "terminal_sha256": terminal_hashes[("av-map", shard_index)],
            }
        )
        start = stop
    return {
        "schema_version": 1,
        "artifact_type": "vqa-event-map-aggregate-freeze",
        "status": "frozen",
        "identities": identity,
        "terminals": terminals,
        "av_projection": {
            "condition_id": "av-map",
            "row_count": len(rows),
            "canonical_id_digest": _id_digest([fixture["video_id"]]),
            "rows_sha256": _canonical_sha256(rows),
            "shards": shard_bindings,
            "rows": rows,
        },
    }


def _expected_av_shards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    terminals = {
        wrapper["shard_index"]: wrapper
        for wrapper in payload["terminals"]
        if wrapper["condition_id"] == "av-map"
    }
    return [
        {
            "shard_index": shard_index,
            "run_id": terminals[shard_index]["terminal"]["run_id"],
            "attempt_uuid": terminals[shard_index]["terminal"]["attempt_uuid"],
            "output_identity": terminals[shard_index]["terminal"]["output_identity"],
            "payload_checksum": terminals[shard_index]["terminal"]["payload_checksum"],
            "row_count": terminals[shard_index]["terminal"]["row_count"],
            "predictions_sha256": payload["av_projection"]["shards"][shard_index][
                "predictions_sha256"
            ],
            "terminal_sha256": terminals[shard_index]["terminal_sha256"],
        }
        for shard_index in range(4)
    ]


def _write_frozen_baseline_bundle(
    tmp_path: Path, fixture: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "frozen-baseline"
    evidence_root = root / "payload" / "evidence"
    frame_root = evidence_root / "frames" / fixture["video_id"]
    frame_root.mkdir(parents=True, exist_ok=True)
    f0_frames = []
    for source in fixture["f0"][0]["frames"]:
        target = frame_root / Path(source["source_path"]).name
        target.write_bytes(Path(source["source_path"]).read_bytes())
        f0_frames.append(
            {
                "path": target.name,
                "requested_timestamp_seconds": source["requested_timestamp_seconds"],
                "selected_frame_timestamp_seconds": source["actual_pts_seconds"],
                "sha256": _sha256(target.read_bytes()),
            }
        )
    manifest = {
        "schema_version": 1,
        "manifest_version": "fixture",
        "label_policy": "label-free-inference-input",
        "example_count": 1,
        "examples": [
            {
                "video_id": fixture["video_id"],
                "question": fixture["panel"][0]["question"],
            }
        ],
    }
    request = {
        "run_id": "frozen-fixture",
        "condition_id": "shared-accepted-whisper",
        "video_id": fixture["video_id"],
        "question": fixture["panel"][0]["question"],
        "frame_paths": [
            f"/dstack/output/evidence/frames/{fixture['video_id']}/{f['path']}"
            for f in f0_frames
        ],
        "frame_timestamps": [f["requested_timestamp_seconds"] for f in f0_frames],
        "transcript": fixture["whisper"][0]["selected_transcript"],
    }
    frames_request = {**request, "condition_id": "frames-only", "transcript": None}
    evidence_row = {
        "audio": {
            "duration_seconds": fixture["panel"][0]["duration_seconds"],
            "path": None,
            "sha256": None,
            "status": "fixture",
        },
        "contact_sheet": f"{fixture['video_id']}.jpg",
        "contact_sheet_sha256": "0" * 64,
        "duration_seconds": fixture["panel"][0]["duration_seconds"],
        "frame_count": 12,
        "frames": f0_frames,
        "media_path": f"{fixture['video_id']}.mp4",
        "media_sha256": fixture["media_sha256"],
        "ordinal": 1,
        "question": fixture["panel"][0]["question"],
        "video_id": fixture["video_id"],
    }
    evidence = {
        "schema_version": 1,
        "label_policy": "label-free-evidence",
        "run_id": "frozen-fixture",
        "frame_policy": "fixture",
        "example_count": 1,
        "examples": [evidence_row],
    }
    whisper = {
        "schema_version": 1,
        "label_policy": "label-free-transcript-evidence",
        "model": {
            "model_id": "openai/whisper-large-v3",
            "model_revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
        },
        "transcription_mode": "translate-only",
        "example_count": 1,
        "accepted_count": 1,
        "no_audio_count": 0,
        "wall_seconds": 1.0,
        "results": fixture["whisper"],
    }
    whisper_summary = {
        "schema_version": 1,
        "label_policy": "label-free-transcript-evidence",
        "example_count": 1,
        "accepted_count": 1,
        "no_audio_count": 0,
        "wall_seconds": 1.0,
        "model": whisper["model"],
    }
    payloads = {
        "manifest.json": manifest,
        "requests.accepted-whisper.json": [request],
        "requests.frames.json": [frames_request],
        "evidence.json": evidence,
        "whisper.json": whisper,
        "whisper-summary.json": whisper_summary,
    }
    for name, payload in payloads.items():
        (evidence_root / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    marker = {
        "schema_version": 1,
        "label_policy": "label-free-evidence",
        "evidence_bundle_id": "frozen-fixture",
        "example_count": 1,
        "manifest_sha256": _sha256((evidence_root / "manifest.json").read_bytes()),
        "requests_sha256": _sha256(
            (evidence_root / "requests.accepted-whisper.json").read_bytes()
        ),
        "whisper_sha256": _sha256((evidence_root / "whisper.json").read_bytes()),
        "canonical_id_digest": _id_digest([fixture["video_id"]]),
        "checksum_file": "SHA256SUMS",
        "frame_policy": "fixture",
        "whisper_policy": "Whisper large-v3 accepted translations only",
    }
    (evidence_root / "evidence-marker.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_lines = []
    for path in sorted(
        item for item in (root / "payload").rglob("*") if item.is_file()
    ):
        checksum_lines.append(
            f"{_sha256(path.read_bytes())}  ./{path.relative_to(root / 'payload').as_posix()}\n"
        )
    checksums = root / "payload" / "SHA256SUMS"
    checksums.write_text("".join(checksum_lines), encoding="utf-8")
    terminal = {
        "payload_sha256": _sha256(checksums.read_bytes()),
        "payload_uri": "hf://private/frozen-fixture/payload",
        "run_name": "frozen-fixture",
        "schema_version": 1,
        "status": "passed",
    }
    terminal_path = root / "terminal-status.json"
    terminal_path.write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        "run_name": "frozen-fixture",
        "payload_uri": terminal["payload_uri"],
        "terminal_sha256": _sha256(terminal_path.read_bytes()),
        "checksums_sha256": _sha256(checksums.read_bytes()),
        "evidence_marker_sha256": _sha256(
            (evidence_root / "evidence-marker.json").read_bytes()
        ),
        "manifest_sha256": marker["manifest_sha256"],
        "requests_sha256": marker["requests_sha256"],
        "frames_requests_sha256": _sha256(
            (evidence_root / "requests.frames.json").read_bytes()
        ),
        "evidence_sha256": _sha256((evidence_root / "evidence.json").read_bytes()),
        "whisper_sha256": marker["whisper_sha256"],
        "whisper_summary_sha256": _sha256(
            (evidence_root / "whisper-summary.json").read_bytes()
        ),
        "whisper_rows_sha256": _canonical_sha256(fixture["whisper"]),
        "model_id": whisper["model"]["model_id"],
        "model_revision": whisper["model"]["model_revision"],
        "example_count": 1,
        "available_count": 1,
        "quality_rejected_count": 0,
        "canonical_id_digest": _id_digest([fixture["video_id"]]),
    }
    return root, contract


def _frame_renderer(
    _row: dict[str, Any],
    frame: dict[str, Any],
    _plan_id: str,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(f"frame-{frame['frame_index']}".encode())
    return {
        "frame_path": str(output_path),
        "stream_index": frame["stream_index"],
        "frame_index": frame["frame_index"],
        "decoded_frame_ordinal": frame["frame_index"],
        "actual_pts_seconds": frame["actual_pts_seconds"],
        "time_base": frame["time_base"],
        "raw_pts": frame.get("raw_pts"),
        "frame_byte_sha256": _sha256(output_path.read_bytes()),
    }


def _contact_sheet(
    *, video_id: str, plan_id: str, frame_paths: list[str], output_path: Path
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(f"{video_id}:{plan_id}:{len(frame_paths)}".encode("utf-8"))
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path.read_bytes()),
    }


def test_build_evidence_materializes_exact_five_plans_and_verbatim_transcripts(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    fixture = _fixture(tmp_path)

    summary = builder.build_evidence(
        panel_rows=fixture["panel"],
        event_map_rows=fixture["event_map"],
        whisper_rows=fixture["whisper"],
        frozen_f0_rows=fixture["f0"],
        output_dir=tmp_path / "evidence",
        frame_inventory_provider=lambda row: fixture["inventory"][row["video_id"]],
        frame_renderer=_frame_renderer,
        contact_sheet_builder=_contact_sheet,
        decoder_runtime_identity={"decoder": "fixture", "version": "1"},
        expected_count=1,
        expected_panel_id_digest=_id_digest([fixture["video_id"]]),
        fixture_mode=True,
        run_id="materializer-test",
    )

    assert summary["row_count"] == 1
    assert summary["plan_ids"] == ["F0", "H/F", "G/S", "H/S", "Shifted H/S"]
    manifest = json.loads((tmp_path / "evidence" / "manifest.json").read_text())
    row = manifest["rows"][0]
    assert set(row["plans"]) == {"F0", "H/F", "G/S", "H/S", "Shifted H/S"}
    assert all(plan["frame_count"] == 12 for plan in row["plans"].values())
    assert row["plans"]["F0"]["transcript_mode"] == "full"
    assert row["plans"]["H/F"]["transcript_mode"] == "full"
    assert row["plans"]["G/S"]["transcript_mode"] == "selected"
    assert row["plans"]["H/S"]["transcript_mode"] == "selected"
    assert row["plans"]["Shifted H/S"]["transcript_mode"] == "selected"
    assert len(list((tmp_path / "evidence" / "contact-sheets").glob("*.jpg"))) == 5
    frame = row["plans"]["H/S"]["frames"][0]
    assert {
        "requested_timestamp_seconds",
        "actual_pts_seconds",
        "stream_index",
        "frame_index",
        "time_base",
        "decoder_runtime_identity",
        "source_media_sha256",
        "frame_byte_sha256",
        "slot_type",
    } <= set(frame)
    requests = json.loads(
        (tmp_path / "evidence" / "requests" / f"{fixture['video_id']}.json").read_text()
    )
    assert requests["F0"]["transcript"] == fixture["whisper"][0]["text"]
    assert requests["G/S"]["transcript"] == "[00:00:04.000] The door opens.\n"
    assert "answer" not in json.dumps(requests).lower()
    assert "visual_summary" not in json.dumps(requests).lower()
    for line in (tmp_path / "evidence" / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        assert digest == _sha256(
            (tmp_path / "evidence" / relative.removeprefix("./")).read_bytes()
        )


def test_f0_reuses_exact_frozen_frame_transcript_and_timestamp_bytes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    _build_with_fixture(tmp_path, fixture=fixture)
    root = tmp_path / "evidence"
    row = json.loads((root / "manifest.json").read_text())["rows"][0]
    request = json.loads((root / "requests" / "video01abcd.json").read_text())["F0"]

    assert request["transcript"].encode() == fixture["f0"][0]["transcript"].encode()
    assert request["frame_timestamps"] == [
        frame["requested_timestamp_seconds"] for frame in fixture["f0"][0]["frames"]
    ]
    for source, materialized in zip(
        fixture["f0"][0]["frames"], row["plans"]["F0"]["frames"], strict=True
    ):
        assert (root / materialized["path"]).read_bytes() == Path(
            source["source_path"]
        ).read_bytes()
        assert materialized["frame_byte_sha256"] == source["frame_byte_sha256"]
        assert (
            materialized["requested_timestamp_seconds"]
            == source["requested_timestamp_seconds"]
        )
        assert materialized["actual_pts_seconds"] == source["actual_pts_seconds"]
    assert row["plans"]["F0"]["frozen_source_sha256"] == "b" * 64


def _build_with_fixture(
    tmp_path: Path,
    *,
    fixture: dict[str, Any] | None = None,
    output_name: str = "evidence",
) -> dict[str, Any]:
    builder = load_builder()
    fixture = fixture or _fixture(tmp_path)
    return builder.build_evidence(
        panel_rows=fixture["panel"],
        event_map_rows=fixture["event_map"],
        whisper_rows=fixture["whisper"],
        frozen_f0_rows=fixture["f0"],
        output_dir=tmp_path / output_name,
        frame_inventory_provider=lambda row: fixture["inventory"][row["video_id"]],
        frame_renderer=_frame_renderer,
        contact_sheet_builder=_contact_sheet,
        decoder_runtime_identity={"decoder": "fixture", "version": "1"},
        expected_count=1,
        expected_panel_id_digest=_id_digest([fixture["video_id"]]),
        fixture_mode=True,
        run_id="materializer-test",
    )


def test_programmatic_defaults_require_canonical_475_panel(tmp_path: Path) -> None:
    builder = load_builder()
    fixture = _fixture(tmp_path)

    with pytest.raises(ValueError, match="exactly 475"):
        builder.build_evidence(
            panel_rows=fixture["panel"],
            event_map_rows=fixture["event_map"],
            whisper_rows=fixture["whisper"],
            frozen_f0_rows=fixture["f0"],
            output_dir=tmp_path / "evidence",
            frame_inventory_provider=lambda row: fixture["inventory"][row["video_id"]],
            frame_renderer=_frame_renderer,
            contact_sheet_builder=_contact_sheet,
            decoder_runtime_identity={"decoder": "fixture", "version": "1"},
        )


def test_materializer_rejects_unsafe_video_id_before_path_construction(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    original = fixture["video_id"]
    unsafe = "../escape"
    fixture["video_id"] = unsafe
    fixture["panel"][0]["video_id"] = unsafe
    fixture["event_map"][0]["video_id"] = unsafe
    fixture["whisper"][0]["video_id"] = unsafe
    fixture["inventory"] = {unsafe: fixture["inventory"][original]}

    with pytest.raises(ValueError, match="YouTube ID"):
        _build_with_fixture(tmp_path, fixture=fixture)
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "evidence").exists()


def test_aggregate_freeze_requires_all_eight_terminals_and_bound_av_projection(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    fixture = _fixture(tmp_path)
    payload = _aggregate_freeze_manifest(fixture)
    path = tmp_path / "aggregate-freeze.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    verified = builder.validate_aggregate_freeze_manifest(
        path,
        expected_count=1,
        expected_panel_id_digest=_id_digest([fixture["video_id"]]),
        expected_shard_counts=(1, 0, 0, 0),
        expected_identities=_freeze_identity(),
        expected_av_shards=_expected_av_shards(payload),
    )

    assert verified["rows"] == [_producer_row(fixture)]
    assert verified["manifest_sha256"] == _sha256(path.read_bytes())
    assert verified["terminal_count"] == 8
    assert verified["av_terminal_count"] == 4

    payload["terminals"].pop()
    missing_path = tmp_path / "missing-terminal-freeze.json"
    missing_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="all 8"):
        builder.validate_aggregate_freeze_manifest(
            missing_path,
            expected_count=1,
            expected_panel_id_digest=_id_digest([fixture["video_id"]]),
            expected_shard_counts=(1, 0, 0, 0),
            expected_identities=_freeze_identity(),
            expected_av_shards=_expected_av_shards(_aggregate_freeze_manifest(fixture)),
        )


def test_aggregate_freeze_rejects_loose_producer_terminal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    loose = _aggregate_freeze_manifest(fixture)["terminals"][0]["terminal"]
    path = tmp_path / "loose-terminal.json"
    path.write_text(json.dumps(loose), encoding="utf-8")

    with pytest.raises(ValueError, match="aggregate freeze"):
        load_builder().validate_aggregate_freeze_manifest(
            path,
            expected_count=1,
            expected_panel_id_digest=_id_digest([fixture["video_id"]]),
            expected_shard_counts=(1, 0, 0, 0),
            expected_identities=_freeze_identity(),
        )


def test_aggregate_freeze_rejects_unexpected_exact_av_attempt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = _aggregate_freeze_manifest(fixture)
    path = tmp_path / "aggregate-freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    expected_av = _expected_av_shards(payload)
    expected_av[0]["attempt_uuid"] = "00000000-0000-4000-8000-000000000099"

    with pytest.raises(ValueError, match="AV shard identity"):
        load_builder().validate_aggregate_freeze_manifest(
            path,
            expected_count=1,
            expected_panel_id_digest=_id_digest([fixture["video_id"]]),
            expected_shard_counts=(1, 0, 0, 0),
            expected_identities=_freeze_identity(),
            expected_av_shards=expected_av,
        )


def test_aggregate_freeze_accepts_explicit_fresh_exact_row_schema(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    fixture = _fixture(tmp_path)
    payload = _aggregate_freeze_manifest(fixture)
    rows = payload["av_projection"]["rows"]
    rows[0]["fresh_protocol_field"] = "fresh-freeze-value"
    payload["av_projection"]["rows_sha256"] = _canonical_sha256(rows)
    payload["av_projection"]["shards"][0]["rows_sha256"] = _canonical_sha256(rows)
    path = tmp_path / "fresh-schema-freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    verified = builder.validate_aggregate_freeze_manifest(
        path,
        expected_count=1,
        expected_panel_id_digest=_id_digest([fixture["video_id"]]),
        expected_shard_counts=(1, 0, 0, 0),
        expected_identities=_freeze_identity(),
        expected_producer_row_fields=sorted(
            {*builder.PRODUCER_ROW_FIELDS, "fresh_protocol_field"}
        ),
        expected_av_shards=_expected_av_shards(payload),
    )

    assert verified["rows"][0]["fresh_protocol_field"] == "fresh-freeze-value"
    assert verified["producer_row_schema_sha256"] == _canonical_sha256(
        sorted({*builder.PRODUCER_ROW_FIELDS, "fresh_protocol_field"})
    )


def test_frozen_baseline_authenticates_whisper_and_exact_f0_sources(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    root, contract = _write_frozen_baseline_bundle(tmp_path, fixture)

    authenticated = load_builder().validate_frozen_baseline_bundle(
        root, contract=contract
    )

    assert authenticated["whisper_rows"] == fixture["whisper"]
    assert authenticated["whisper_binding"]["available_count"] == 1
    assert authenticated["whisper_binding"]["quality_rejected_count"] == 0
    assert (
        authenticated["f0_rows"][0]["source_artifact_sha256"]
        == authenticated["f0_binding"]["artifact_sha256"]
    )
    assert len(authenticated["f0_rows"][0]["frames"]) == 12


def test_frozen_baseline_allows_current_locator_to_replace_terminal_locator(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    root, contract = _write_frozen_baseline_bundle(tmp_path, fixture)
    contract["payload_uri"] = (
        "hf://buckets/Npatta01/trec_video_2026/experiments/frozen-fixture/payload"
    )

    authenticated = load_builder().validate_frozen_baseline_bundle(
        root, contract=contract
    )

    assert authenticated["whisper_rows"] == fixture["whisper"]


def test_frozen_baseline_rejects_tampered_f0_frame(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root, contract = _write_frozen_baseline_bundle(tmp_path, fixture)
    source = next((root / "payload" / "evidence" / "frames").rglob("*.jpg"))
    source.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="frame|checksum|hash"):
        load_builder().validate_frozen_baseline_bundle(root, contract=contract)


def test_superseded_424_51_whisper_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="superseded|427|48"):
        load_builder()._validate_whisper_counts(
            example_count=475,
            available_count=424,
            quality_rejected_count=51,
            expected_count=475,
            expected_available=427,
            expected_quality_rejected=48,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda f: f["event_map"][0].update({"status": "failed"}), "did not pass"),
        (
            lambda f: f["event_map"].append(copy.deepcopy(f["event_map"][0])),
            "duplicate",
        ),
        (
            lambda f: f["event_map"][0].update({"question": "different"}),
            "question mismatch",
        ),
        (
            lambda f: f["event_map"][0].update({"selected_intervals": []}),
            "selected_intervals",
        ),
        (lambda f: f["panel"][0].update({"unexpected": True}), "schema drift"),
        (lambda f: f["panel"][0].update({"media_sha256": "0" * 64}), "media hash"),
        (
            lambda f: f["whisper"][0]["english_translation"].update(
                {"timestamped_text": None}
            ),
            "transcript is missing",
        ),
    ],
)
def test_materializer_rejects_failed_mismatched_duplicate_and_malformed_inputs(
    tmp_path: Path, mutation, message: str
) -> None:
    fixture = _fixture(tmp_path)
    mutation(fixture)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        _build_with_fixture(tmp_path, fixture=fixture)
    assert not (tmp_path / "evidence").exists()
    assert not list(tmp_path.glob(".evidence.staging-*"))


def test_materializer_rejects_missing_media_before_writing_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["panel"][0]["media_path"] = str(tmp_path / "missing.mp4")

    with pytest.raises((ValueError, FileNotFoundError), match="missing|media"):
        _build_with_fixture(tmp_path, fixture=fixture)
    assert not (tmp_path / "evidence").exists()


def test_materializer_rejects_duplicate_or_non_monotonic_actual_pts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["inventory"][fixture["video_id"]][20]["actual_pts_seconds"] = 0.1

    with pytest.raises(ValueError, match="PTS"):
        _build_with_fixture(tmp_path, fixture=fixture)

    fixture = _fixture(tmp_path / "ordered")
    fixture["inventory"][fixture["video_id"]][20]["actual_pts_seconds"] = -1.0
    with pytest.raises(ValueError, match="PTS"):
        _build_with_fixture(tmp_path / "ordered", fixture=fixture)


def test_selected_transcript_preserves_source_lines_ids_and_hashes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["event_map"][0]["selected_intervals"] = [
        {"event_id": "event-a", "start": 1.5, "end": 4.5}
    ]
    _build_with_fixture(tmp_path, fixture=fixture)
    root = tmp_path / "evidence"
    manifest = json.loads((root / "manifest.json").read_text())
    transcript = manifest["rows"][0]["transcript"]
    assert transcript["selected_segment_ids"] == [0, 1]
    assert transcript["selected_segments"][0]["source_line"] == "[00:00:00.000] Before."
    assert transcript["selected_segments"][1]["source_line_bytes_sha256"] == _sha256(
        b"[00:00:04.000] The door opens.\n"
    )
    assert (root / "transcripts" / "video01abcd" / "selected.txt").read_bytes() == (
        b"[00:00:00.000] Before.\n[00:00:04.000] The door opens.\n"
    )
    assert transcript["selected_source_sha256"] == transcript["full_source_sha256"]
    assert transcript["selected_sha256"] == _sha256(
        b"[00:00:00.000] Before.\n[00:00:04.000] The door opens.\n"
    )


def test_whisper_overlap_uses_true_chunk_end_and_preserves_real_gap(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["event_map"][0]["selected_intervals"] = [
        {"event_id": "event-a", "start": 2.5, "end": 3.5}
    ]

    _build_with_fixture(tmp_path, fixture=fixture)
    requests = json.loads(
        (tmp_path / "evidence" / "requests" / "video01abcd.json").read_text()
    )

    assert requests["G/S"]["transcript"] is None
    assert requests["H/S"]["transcript"] is None


def test_accepted_trailing_out_of_duration_chunk_is_explicitly_recorded(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    trailing_line = "[00:00:13.000] to be continued"
    text = fixture["whisper"][0]["timestamped_text"].rstrip("\n") + "\n" + trailing_line
    fixture["whisper"][0].update(
        {
            "text": text,
            "timestamped_text": text,
            "selected_transcript": text,
            "source_sha256": _sha256(text.encode()),
        }
    )
    fixture["f0"][0].update(
        {"transcript": text, "transcript_sha256": _sha256(text.encode())}
    )
    fixture["whisper"][0]["english_translation"]["timestamped_text"] = text
    fixture["whisper"][0]["english_translation"]["text"] += " to be continued"
    fixture["whisper"][0]["english_translation"]["chunks"].append(
        {"start": 13.0, "end": None, "text": "to be continued", "language": "english"}
    )

    _build_with_fixture(tmp_path, fixture=fixture)
    row = json.loads((tmp_path / "evidence" / "manifest.json").read_text())["rows"][0]

    assert row["transcript"]["excluded_segments"] == [
        {
            "segment_id": 3,
            "source_line": trailing_line,
            "source_line_bytes_sha256": _sha256(trailing_line.encode()),
            "start_seconds": 13.0,
            "end_seconds": None,
            "reason": "out_of_duration_unbounded",
        }
    ]


def test_whisper_metadata_preserves_original_chunk_ids_and_source_byte_hashes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    lines = fixture["whisper"][0]["text"].splitlines(keepends=True)
    fixture["event_map"][0]["selected_intervals"] = [
        {"event_id": "event-a", "start": 4.0, "end": 5.0}
    ]
    _build_with_fixture(tmp_path, fixture=fixture)
    transcript = json.loads((tmp_path / "evidence" / "manifest.json").read_text())[
        "rows"
    ][0]["transcript"]
    assert transcript["selected_segment_ids"] == [1]
    assert transcript["selected_segments"][0]["segment_id"] == 1
    assert transcript["selected_segments"][0]["original_segment_sha256"] == _sha256(
        lines[1].encode()
    )


def test_shift_control_preserves_lengths_and_records_impossible_fallback() -> None:
    builder = load_builder()
    selected = [
        {"event_id": "a", "start_seconds": 2.0, "end_seconds": 4.0},
        {"event_id": "b", "start_seconds": 6.0, "end_seconds": 7.0},
    ]
    shifted, reason = builder._shift_intervals(selected, duration=10.0)
    assert reason is None
    assert [item["end_seconds"] - item["start_seconds"] for item in shifted] == [
        2.0,
        1.0,
    ]
    assert all(
        shifted_item["end_seconds"] <= original["start_seconds"]
        or shifted_item["start_seconds"] >= original["end_seconds"]
        for shifted_item in shifted
        for original in selected
    )

    impossible, reason = builder._shift_intervals(
        [{"event_id": "a", "start_seconds": 0.0, "end_seconds": 9.0}],
        duration=10.0,
    )
    assert impossible == []
    assert reason == "no_disjoint_shift_possible"

    reordered, reason = builder._shift_intervals(
        [
            {"event_id": "long", "start_seconds": 1.0, "end_seconds": 3.0},
            {"event_id": "short", "start_seconds": 5.0, "end_seconds": 6.0},
        ],
        duration=6.0,
    )
    assert reason is None
    assert reordered == [
        {"event_id": "short", "start_seconds": 0.0, "end_seconds": 1.0},
        {"event_id": "long", "start_seconds": 3.0, "end_seconds": 5.0},
    ]


def test_impossible_shift_omits_only_paired_control(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["event_map"][0]["selected_intervals"] = [
        {"event_id": "event-a", "start": 0.0, "end": 3.0},
        {"event_id": "event-a", "start": 4.0, "end": 7.0},
    ]

    summary = _build_with_fixture(tmp_path, fixture=fixture)
    root = tmp_path / "evidence"
    row = json.loads((root / "manifest.json").read_text())["rows"][0]
    requests = json.loads((root / "requests" / "video01abcd.json").read_text())

    assert set(row["plans"]) == {"F0", "H/F", "G/S", "H/S"}
    assert set(requests) == {"F0", "H/F", "G/S", "H/S"}
    assert row["shifted_intervals"] == []
    assert row["shifted_control"] == {
        "eligible": False,
        "reason": "no_disjoint_shift_possible",
        "transcript_quantity_measure": "selected_segment_count",
    }
    assert summary["shifted_control_ineligibility_reasons"] == {
        "no_disjoint_shift_possible": 1
    }


def test_shifted_transcript_count_mismatch_omits_paired_control(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    text = (
        "[00:00:00.000] First shifted line.\n"
        "[00:00:00.500] Second shifted line.\n"
        "[00:00:04.000] The door opens.\n"
        "[00:00:08.000] After.\n"
    )
    fixture["panel"][0]["whisper_segment_count"] = 4
    fixture["whisper"][0].update(
        {
            "text": text,
            "timestamped_text": text,
            "selected_transcript": text,
            "source_sha256": _sha256(text.encode()),
        }
    )
    fixture["whisper"][0]["english_translation"] = {
        "text": "First shifted line. Second shifted line. The door opens. After.",
        "timestamped_text": text,
        "chunks": [
            {"start": 0.0, "end": 0.4, "text": "First shifted line."},
            {"start": 0.5, "end": 0.9, "text": "Second shifted line."},
            {"start": 4.0, "end": 6.0, "text": "The door opens."},
            {"start": 8.0, "end": 10.0, "text": "After."},
        ],
    }
    fixture["f0"][0].update(
        {"transcript": text, "transcript_sha256": _sha256(text.encode())}
    )

    _build_with_fixture(tmp_path, fixture=fixture)
    root = tmp_path / "evidence"
    row = json.loads((root / "manifest.json").read_text())["rows"][0]
    requests = json.loads((root / "requests" / "video01abcd.json").read_text())

    assert "Shifted H/S" not in row["plans"]
    assert "Shifted H/S" not in requests
    assert row["shifted_control"]["eligible"] is False
    assert row["shifted_control"]["reason"] == "transcript_segment_count_mismatch"
    assert row["transcript_budget"]["selected_segment_count"] == 1
    assert row["transcript_budget"]["shifted_segment_count"] == 2


def test_shifted_hybrid_has_same_budget_and_disjoint_intervals_when_possible(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["event_map"][0]["selected_intervals"] = [
        {"event_id": "event-a", "start": 4.0, "end": 5.0}
    ]
    _build_with_fixture(tmp_path, fixture=fixture)
    manifest = json.loads((tmp_path / "evidence" / "manifest.json").read_text())
    row = manifest["rows"][0]
    assert row["shift_fallback_reason"] is None
    assert row["plans"]["Shifted H/S"]["frame_count"] == 12
    assert (
        sum(
            frame["slot_type"] == "global"
            for frame in row["plans"]["Shifted H/S"]["frames"]
        )
        == 6
    )
    assert (
        sum(
            frame["slot_type"] == "local"
            for frame in row["plans"]["Shifted H/S"]["frames"]
        )
        == 6
    )
    for selected in row["selected_intervals"]:
        for shifted in row["shifted_intervals"]:
            assert (
                shifted["end_seconds"] <= selected["start_seconds"]
                or shifted["start_seconds"] >= selected["end_seconds"]
            )


def test_shifted_hs_reselects_whisper_from_shifted_intervals(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["event_map"][0]["selected_intervals"] = [
        {"event_id": "event-a", "start": 4.0, "end": 5.0}
    ]
    _build_with_fixture(tmp_path, fixture=fixture)
    requests = json.loads(
        (tmp_path / "evidence" / "requests" / "video01abcd.json").read_text()
    )

    assert requests["H/S"]["transcript"] == "[00:00:04.000] The door opens.\n"
    assert requests["Shifted H/S"]["transcript"] == "[00:00:00.000] Before.\n"
    assert "The door opens." not in requests["Shifted H/S"]["transcript"]


def test_shifted_transcript_provenance_records_distinct_ids_and_budget_match(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["event_map"][0]["selected_intervals"] = [
        {"event_id": "event-a", "start": 4.0, "end": 5.0}
    ]
    _build_with_fixture(tmp_path, fixture=fixture)
    row = json.loads((tmp_path / "evidence" / "manifest.json").read_text())["rows"][0]

    assert row["transcript"]["selected_segment_ids"] == [1]
    assert row["shifted_transcript"]["selected_segment_ids"] == [0]
    assert row["transcript_budget"]["segment_count_equal"] is True
    assert row["transcript_budget"]["byte_count_equal"] is False
    assert row["transcript_budget"]["mismatch_reason"] == "whisper_overlap_bytes_differ"


def test_quality_rejected_whisper_row_keeps_visual_plans_and_null_transcripts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["panel"][0]["whisper_segment_count"] = 0
    fixture["whisper"][0].update(
        {
            "transcript_included": False,
            "text": "rejected ASR must not be copied",
            "timestamped_text": None,
            "selected_transcript": None,
            "quality": {"status": "rejected"},
        }
    )
    fixture["f0"][0].update({"transcript": None, "transcript_sha256": None})
    summary = _build_with_fixture(tmp_path, fixture=fixture)
    root = tmp_path / "evidence"
    requests = json.loads((root / "requests" / "video01abcd.json").read_text())
    row = json.loads((root / "manifest.json").read_text())["rows"][0]

    assert summary["transcript_counts"] == {"available": 0, "unavailable": 1}
    assert all(request["transcript"] is None for request in requests.values())
    assert row["transcript"]["transcript_available"] is False
    assert row["shifted_transcript"]["transcript_available"] is False
    assert row["transcript"]["availability_provenance"]["quality_status"] == "rejected"
    assert len(row["transcript"]["availability_provenance"]["source_row_sha256"]) == 64
    assert "rejected ASR" not in json.dumps(requests)
    assert "rejected ASR" not in json.dumps(row["transcript"])


def test_hashes_are_deterministic_across_output_roots(tmp_path: Path) -> None:
    first_fixture = _fixture(tmp_path / "first-input")
    second_fixture = _fixture(tmp_path / "second-input")
    _build_with_fixture(tmp_path / "first-input", fixture=first_fixture)
    _build_with_fixture(tmp_path / "second-input", fixture=second_fixture)
    first = tmp_path / "first-input" / "evidence"
    second = tmp_path / "second-input" / "evidence"
    for relative in (
        "manifest.json",
        "summary.json",
        "SHA256SUMS",
        "requests/video01abcd.json",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_recursive_reference_leakage_is_rejected_without_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["panel"][0]["reference"] = "must not be read"

    with pytest.raises(ValueError, match="leakage|schema drift"):
        _build_with_fixture(tmp_path, fixture=fixture)
    assert not (tmp_path / "evidence").exists()


def test_rerun_refuses_overwrite_and_preserves_first_atomic_bundle(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _build_with_fixture(tmp_path, fixture=fixture)
    root = tmp_path / "evidence"
    before = (root / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError):
        _build_with_fixture(tmp_path, fixture=fixture)

    assert (root / "manifest.json").read_bytes() == before
    assert not list(tmp_path.glob(".evidence.staging-*"))


def test_completion_marker_binds_all_sources_and_adapter_boundary(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    summary = _build_with_fixture(tmp_path, fixture=fixture)
    root = tmp_path / "evidence"
    marker = json.loads((root / "evidence-marker.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())

    assert marker["status"] == "complete"
    assert (
        marker["aggregate_freeze_manifest_sha256"]
        == summary["aggregate_freeze_manifest_sha256"]
    )
    assert marker["whisper_artifact_sha256"] == summary["whisper_artifact_sha256"]
    assert marker["frozen_f0_artifact_sha256"] == summary["frozen_f0_artifact_sha256"]
    assert marker["manifest_sha256"] == _sha256((root / "manifest.json").read_bytes())
    assert marker["summary_sha256"] == _sha256((root / "summary.json").read_bytes())
    assert marker["adapter_contract"] == manifest["adapter_contract"]
    assert marker["adapter_contract"]["later_adapter_must_flatten"] is True
    checksum_lines = (root / "SHA256SUMS").read_text().splitlines()
    assert any(line.endswith("  ./evidence-marker.json") for line in checksum_lines)


def test_source_media_mutation_window_is_closed_before_atomic_publish(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    mutated = False

    def mutating_renderer(
        row: dict[str, Any],
        frame: dict[str, Any],
        plan_id: str,
        output_path: Path,
    ) -> dict[str, Any]:
        nonlocal mutated
        rendered = _frame_renderer(row, frame, plan_id, output_path)
        if not mutated:
            media = Path(row["media_path"])
            media.write_bytes(media.read_bytes() + b"mutation")
            mutated = True
        return rendered

    with pytest.raises(ValueError, match="mutated during materialization"):
        load_builder().build_evidence(
            panel_rows=fixture["panel"],
            event_map_rows=fixture["event_map"],
            whisper_rows=fixture["whisper"],
            frozen_f0_rows=fixture["f0"],
            output_dir=tmp_path / "output",
            frame_inventory_provider=lambda row: fixture["inventory"][row["video_id"]],
            frame_renderer=mutating_renderer,
            contact_sheet_builder=_contact_sheet,
            decoder_runtime_identity={"decoder": "fixture", "version": "1"},
            expected_count=1,
            expected_panel_id_digest=_id_digest([fixture["video_id"]]),
            fixture_mode=True,
        )
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.staging-*"))


def test_real_current_frozen_baseline_contract_reports_427_48_when_available() -> None:
    root = Path(
        "/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/"
        "vqa-factorial-full-evidence-20260809-f1"
    )
    if not root.is_dir():
        pytest.skip("private frozen baseline is unavailable")

    authenticated = load_builder().validate_frozen_baseline_bundle(root)

    assert len(authenticated["whisper_rows"]) == 475
    assert len(authenticated["f0_rows"]) == 475
    assert authenticated["whisper_binding"]["available_count"] == 427
    assert authenticated["whisper_binding"]["quality_rejected_count"] == 48


def test_real_tiny_video_default_decode_pts_hash_and_contact_sheet(
    tmp_path: Path,
) -> None:
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is unavailable")
    from PIL import Image

    fixture = _fixture(tmp_path)
    media = Path(fixture["panel"][0]["media_path"])
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10:duration=12",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(media),
        ],
        check=True,
    )
    media_sha256 = _sha256(media.read_bytes())
    fixture["panel"][0]["media_sha256"] = media_sha256
    fixture["event_map"][0]["source_media_sha256"] = media_sha256
    fixture["f0"][0]["source_media_sha256"] = media_sha256
    for index, frame in enumerate(fixture["f0"][0]["frames"]):
        source = Path(frame["source_path"])
        Image.new("RGB", (32, 24), (index * 17 % 255, 20, 40)).save(
            source, format="JPEG"
        )
        frame["frame_byte_sha256"] = _sha256(source.read_bytes())

    load_builder().build_evidence(
        panel_rows=fixture["panel"],
        event_map_rows=fixture["event_map"],
        whisper_rows=fixture["whisper"],
        frozen_f0_rows=fixture["f0"],
        output_dir=tmp_path / "real-output",
        expected_count=1,
        expected_panel_id_digest=_id_digest([fixture["video_id"]]),
        fixture_mode=True,
    )
    root = tmp_path / "real-output"
    manifest = json.loads((root / "manifest.json").read_text())
    frames = manifest["rows"][0]["plans"]["H/S"]["frames"]
    assert len({frame["actual_pts_seconds"] for frame in frames}) == 12
    assert all(
        frame["frame_byte_sha256"] == _sha256((root / frame["path"]).read_bytes())
        for frame in frames
    )
    runtime = manifest["decoder_runtime_identity"]
    assert runtime["decoder"] == "PyAV"
    assert runtime["ffmpeg_library_versions"]["libavcodec"]
    assert runtime["libjpeg_version"]
    sheet = root / manifest["rows"][0]["plans"]["H/S"]["contact_sheet"]["path"]
    assert Image.open(sheet).size == (1280, 750)


def test_production_cli_accepts_only_strict_frozen_sources(tmp_path: Path) -> None:
    builder = load_builder()
    args = builder.parse_args(
        [
            "--panel",
            str(tmp_path / "panel.json"),
            "--aggregate-freeze-manifest",
            str(tmp_path / "aggregate-freeze.json"),
            "--expected-freeze-contract",
            str(tmp_path / "freeze-identities.json"),
            "--frozen-baseline-root",
            str(tmp_path / "baseline"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert not hasattr(args, "frame_inventory")
    assert not hasattr(args, "expected_count")
    assert not hasattr(args, "decoder_runtime_version")

    with pytest.raises(SystemExit):
        builder.parse_args(
            [
                "--panel",
                "panel.json",
                "--aggregate-freeze-manifest",
                "freeze.json",
                "--expected-freeze-contract",
                "identities.json",
                "--frozen-baseline-root",
                "baseline",
                "--output-dir",
                "output",
                "--frame-inventory",
                "inventory.json",
            ]
        )


def test_cli_rejects_missing_aggregate_freeze_binding(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        load_builder().main(
            [
                "--panel",
                str(tmp_path / "panel.json"),
                "--frozen-baseline-root",
                str(tmp_path / "baseline"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


def test_aggregate_freeze_rejects_tampered_terminal_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = _aggregate_freeze_manifest(fixture)
    payload["terminals"][4]["terminal"]["payload_checksum"] = "0" * 64
    path = tmp_path / "tampered-freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal hash"):
        load_builder().validate_aggregate_freeze_manifest(
            path,
            expected_count=1,
            expected_panel_id_digest=_id_digest([fixture["video_id"]]),
            expected_shard_counts=(1, 0, 0, 0),
            expected_identities=_freeze_identity(),
        )


def test_default_batch_decoder_decodes_repeated_identity_once(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    calls: list[tuple[tuple[int, int], ...]] = []
    builder = load_builder()

    def batch_decoder(
        _row: dict[str, Any],
        needed: dict[tuple[int, int], dict[str, Any]],
        cache_dir: Path,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        calls.append(tuple(sorted(needed)))
        result = {}
        for identity, frame in needed.items():
            path = cache_dir / f"{identity[0]}-{identity[1]}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"batch-frame-{identity[1]}".encode())
            result[identity] = {
                **frame,
                "frame_path": str(path),
                "decoded_frame_ordinal": frame["frame_index"],
                "frame_byte_sha256": _sha256(path.read_bytes()),
            }
        return result

    _build_with_fixture(tmp_path, fixture=fixture)
    # The injectable renderer above exercises fixture behavior. Re-run through
    # the production-default renderer seam with a counted batch decoder.
    output = tmp_path / "batch-output"
    builder.build_evidence(
        panel_rows=fixture["panel"],
        event_map_rows=fixture["event_map"],
        whisper_rows=fixture["whisper"],
        frozen_f0_rows=fixture["f0"],
        output_dir=output,
        frame_inventory_provider=lambda row: fixture["inventory"][row["video_id"]],
        frame_batch_decoder=batch_decoder,
        contact_sheet_builder=_contact_sheet,
        decoder_runtime_identity={"decoder": "fixture", "version": "1"},
        expected_count=1,
        expected_panel_id_digest=_id_digest([fixture["video_id"]]),
        fixture_mode=True,
        run_id="materializer-test",
    )
    assert len(calls) == 1
    assert len(calls[0]) < 60


def test_contact_sheet_cell_label_contains_requested_and_actual_pts() -> None:
    builder = load_builder()
    label = builder._contact_sheet_cell_label(
        video_id="video01abcd",
        plan_id="H/S",
        index=0,
        frame={
            "requested_timestamp_seconds": 4.25,
            "actual_pts_seconds": 4.2,
        },
    )
    assert "req=4.250000s" in label
    assert "pts=4.200000s" in label


def test_production_runtime_identity_cannot_be_caller_supplied() -> None:
    with pytest.raises(ValueError, match="cannot be caller supplied"):
        load_builder()._runtime_identity(
            {"decoder": "claimed", "decoder_runtime_version": "claimed"},
            fixture_mode=False,
        )
