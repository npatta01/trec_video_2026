from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vqa_deadline_gemini_rescue_mp4 as runner  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(path: Path, body: dict[str, Any]) -> None:
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text(
        json.dumps(
            {**body, "_artifact_sha256": hashlib.sha256(canonical.encode()).hexdigest()},
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def compatibility_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    media_root = tmp_path / "testing"
    videos = media_root / "videos"
    videos.mkdir(parents=True)
    raw_rows: list[dict[str, Any]] = []
    query_ids = [175, 310, 886, 1563, 2279, 2404]
    for query_id in query_ids:
        video_id = f"video-{query_id}.webm"
        source = videos / video_id
        source.write_bytes(f"source-{query_id}".encode())
        raw_rows.append(
            {
                "availability": "available",
                "frame_paths": [],
                "frame_timestamps": [0.0, 2.0],
                "query_id": query_id,
                "question": f"Question {query_id}?",
                "source_sha256": _sha256(source),
                "video_id": video_id,
            }
        )
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps(raw_rows) + "\n", encoding="utf-8")
    loaded = runner.full.load_input_rows(requests, media_root=media_root, expected_count=6)

    base_freeze = tmp_path / "base-predictions.json"
    _write_artifact(base_freeze, {"status": "predictions_frozen", "model": runner.full.MODEL})
    first_manifest = tmp_path / "first-manifest.json"
    _write_artifact(
        first_manifest,
        {
            "status": "inputs_frozen",
            "base_predictions_freeze_sha256": _sha256(base_freeze),
            "input": {"path": str(requests), "sha256": _sha256(requests)},
            "rows": loaded,
        },
    )
    first_freeze = tmp_path / "first-predictions.json"
    _write_artifact(
        first_freeze,
        {
            "status": "predictions_frozen",
            "model": runner.full.MODEL,
            "base_predictions_freeze_sha256": _sha256(base_freeze),
            "manifest_sha256": _sha256(first_manifest),
            "rows": [
                {
                    "query_id": row["query_id"],
                    "video_id": row["video_id"],
                    "status": "failed",
                    "failure_reason": "empty_text",
                    "answer": "",
                }
                for row in loaded
            ],
        },
    )
    monkeypatch.setattr(runner, "ORIGINAL_FREEZE_SHA256", _sha256(base_freeze))
    monkeypatch.setattr(runner, "FIRST_RESCUE_FREEZE_SHA256", _sha256(first_freeze))
    monkeypatch.setattr(runner, "FIRST_RESCUE_MANIFEST_SHA256", _sha256(first_manifest))
    monkeypatch.setattr(runner, "EXPECTED_QUERY_IDS", tuple(query_ids))
    monkeypatch.setattr(runner, "EXPECTED_PANEL_COUNT", 6)
    return {
        "tmp": tmp_path,
        "requests": requests,
        "media_root": media_root,
        "base_freeze": base_freeze,
        "first_freeze": first_freeze,
        "first_manifest": first_manifest,
        "loaded": loaded,
    }


def test_selector_requires_exact_first_rescue_empty_rows_and_source_bindings(
    compatibility_fixture: dict[str, Any],
) -> None:
    rows = runner.select_compatibility_rows(
        original_freeze_path=compatibility_fixture["base_freeze"],
        first_rescue_freeze_path=compatibility_fixture["first_freeze"],
        first_rescue_manifest_path=compatibility_fixture["first_manifest"],
        requests_path=compatibility_fixture["requests"],
        media_root=compatibility_fixture["media_root"],
    )
    assert [row["query_id"] for row in rows] == [175, 310, 886, 1563, 2279, 2404]
    assert all(row["compatibility_source_failure_reason"] == "empty_text" for row in rows)


def test_selector_rejects_first_rescue_manifest_mutation(
    compatibility_fixture: dict[str, Any],
) -> None:
    changed = json.loads(compatibility_fixture["first_manifest"].read_text())
    changed["rows"][0]["question"] = "mutated"
    _write_artifact(compatibility_fixture["first_manifest"], {k: v for k, v in changed.items() if k != "_artifact_sha256"})
    with pytest.raises(ValueError, match="hash|binding|manifest"):
        runner.select_compatibility_rows(
            original_freeze_path=compatibility_fixture["base_freeze"],
            first_rescue_freeze_path=compatibility_fixture["first_freeze"],
            first_rescue_manifest_path=compatibility_fixture["first_manifest"],
            requests_path=compatibility_fixture["requests"],
            media_root=compatibility_fixture["media_root"],
        )


def test_mp4_transcode_is_atomic_and_records_h264_aac_provenance(
    compatibility_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = runner.select_compatibility_rows(
        original_freeze_path=compatibility_fixture["base_freeze"],
        first_rescue_freeze_path=compatibility_fixture["first_freeze"],
        first_rescue_manifest_path=compatibility_fixture["first_manifest"],
        requests_path=compatibility_fixture["requests"],
        media_root=compatibility_fixture["media_root"],
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        Path(command[-1]).write_bytes(b"mp4-delivery")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(runner, "_tool_version", lambda _tool: "tool-version")
    monkeypatch.setattr(
        runner,
        "_probe_media",
        lambda path: {
            "duration_seconds": 2.0,
            "video_codec": "h264",
            "audio_codec": "aac",
        }
        if path.suffix == ".mp4"
        else {
            "duration_seconds": 2.0,
            "video_codec": "vp9",
            "audio_codec": "opus",
        },
    )
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    records = runner.transcode_rows(rows, compatibility_fixture["tmp"] / "deliveries")
    assert len(records) == 6
    assert {record["delivery_mime_type"] for record in records} == {"video/mp4"}
    assert all(record["output_probe"]["video_codec"] == "h264" for record in records)
    assert all(record["output_probe"]["audio_codec"] == "aac" for record in records)
    assert all("libx264" in command and "aac" in command for command in commands)
    assert all(record["ffmpeg_version"] == "tool-version" for record in records)
    assert all(Path(record["delivery_path"]).suffix == ".mp4" for record in records)
    assert all(command[-1].endswith(".mp4") for command in commands)
    runner._freeze_transcode_records(
        compatibility_fixture["tmp"] / "evidence" / "transcodes", records
    )
    assert (compatibility_fixture["tmp"] / "evidence" / "transcodes" / "video-175.webm.json").is_file()
    with pytest.raises(ValueError, match="overwrite"):
        runner.transcode_rows(rows, compatibility_fixture["tmp"] / "deliveries")


def test_request_settings_and_caps_are_fixed(
    compatibility_fixture: dict[str, Any],
) -> None:
    rows = runner.select_compatibility_rows(
        original_freeze_path=compatibility_fixture["base_freeze"],
        first_rescue_freeze_path=compatibility_fixture["first_freeze"],
        first_rescue_manifest_path=compatibility_fixture["first_manifest"],
        requests_path=compatibility_fixture["requests"],
        media_root=compatibility_fixture["media_root"],
    )
    request = runner.build_compatibility_request(
        row=rows[2], upload={"file_uri": "https://provider.invalid/file", "mime_type": "video/mp4"}
    )
    assert request.config.max_output_tokens == 8192
    assert request.contents[0].parts[0].video_metadata.fps == 4.0
    assert request.contents[0].parts[0].media_resolution.level.value == "MEDIA_RESOLUTION_LOW"
    assert request.config.thinking_config.thinking_level.value == "HIGH"
    assert runner.UPLOAD_WORKERS == 1
    assert runner.MAX_ESTIMATED_COST_USD <= 0.50
    assert runner.MAX_OBSERVED_COST_USD <= 0.50


@pytest.mark.parametrize("bad_cap", [float("nan"), float("inf"), -0.01, True, 0.51])
def test_nonfinite_negative_boolean_or_oversized_caps_fail_closed(
    compatibility_fixture: dict[str, Any], bad_cap: float
) -> None:
    with pytest.raises(ValueError, match="cap"):
        runner.run_compatibility(
            original_freeze_path=compatibility_fixture["base_freeze"],
            first_rescue_freeze_path=compatibility_fixture["first_freeze"],
            first_rescue_manifest_path=compatibility_fixture["first_manifest"],
            requests_path=compatibility_fixture["requests"],
            media_root=compatibility_fixture["media_root"],
            max_estimated_cost_usd=bad_cap,
        )
