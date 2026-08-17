from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import overlay_vqa_deadline_gemini_mp4_rescue as overlay  # noqa: E402


def _write(path: Path, body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    value = {**body, "_artifact_sha256": hashlib.sha256(canonical).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest_sha = _write(
        source_manifest,
        {
            "status": "inputs_frozen",
            "row_count": 2,
            "rows": [
                {"query_id": 1, "video_id": "a", "question": "Q1?"},
                {"query_id": 2, "video_id": "b", "question": "Q2?"},
            ],
        },
    )
    base_path = tmp_path / "first-overlay.json"
    base_sha = _write(
        base_path,
        {
            "status": "predictions_frozen",
            "model": "gemini-3.7-flash",
            "row_count": 2,
            "rows": [
                {
                    "query_id": 1,
                    "video_id": "a",
                    "status": "failed",
                    "failure_reason": "empty_text",
                    "answer": "",
                },
                {
                    "query_id": 2,
                    "video_id": "b",
                    "status": "failed",
                    "failure_reason": "empty_text",
                    "answer": "",
                },
            ],
        },
    )
    manifest_path = tmp_path / "mp4" / "manifest.json"
    manifest_sha = _write(
        manifest_path,
        {
            "status": "inputs_frozen",
            "model": "gemini-3.7-flash",
            "original_predictions_freeze_sha256": "o" * 64,
            "first_rescue_predictions_freeze_sha256": "f" * 64,
            "first_rescue_manifest_sha256": "m" * 64,
            "selected_query_ids": [1, 2],
            "model_settings": {
                "video_fps": 4.0,
                "media_resolution": "MEDIA_RESOLUTION_LOW",
                "thinking_level": "HIGH",
                "max_output_tokens": 8192,
            },
            "rows": [
                {"query_id": 1, "video_id": "a", "question": "Q1?"},
                {"query_id": 2, "video_id": "b", "question": "Q2?"},
            ],
        },
    )
    rescue_path = tmp_path / "mp4" / "predictions-freeze.json"
    rescue_sha = _write(
        rescue_path,
        {
            "status": "predictions_frozen",
            "model": "gemini-3.7-flash",
            "manifest_sha256": manifest_sha,
            "original_predictions_freeze_sha256": "o" * 64,
            "first_rescue_predictions_freeze_sha256": "f" * 64,
            "first_rescue_manifest_sha256": "m" * 64,
            "selected_query_ids": [1, 2],
            "max_output_tokens": 8192,
            "row_count": 2,
            "rows": [
                {
                    "query_id": 1,
                    "video_id": "a",
                    "status": "succeeded",
                    "answer": "new",
                    "elapsed_seconds": 1.5,
                },
                {
                    "query_id": 2,
                    "video_id": "b",
                    "status": "failed",
                    "failure_reason": "empty_text",
                    "answer": "",
                },
            ],
        },
    )
    monkeypatch.setattr(overlay, "FIRST_OVERLAY_SHA256", base_sha)
    monkeypatch.setattr(overlay, "MP4_RESCUE_SHA256", rescue_sha)
    monkeypatch.setattr(overlay, "ORIGINAL_FREEZE_SHA256", "o" * 64)
    monkeypatch.setattr(overlay, "FIRST_RESCUE_FREEZE_SHA256", "f" * 64)
    monkeypatch.setattr(overlay, "FIRST_RESCUE_MANIFEST_SHA256", "m" * 64)
    monkeypatch.setattr(overlay, "SOURCE_MANIFEST_SHA256", source_manifest_sha)
    monkeypatch.setattr(overlay, "EXPECTED_QUERY_IDS", (1, 2))
    monkeypatch.setattr(overlay, "EXPECTED_PANEL_COUNT", 2)
    return {"base": base_path, "rescue": rescue_path, "source_manifest": source_manifest}


def test_overlay_replaces_only_successful_mp4_rescue_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = overlay.overlay_mp4_rescue(
        base_freeze=paths["base"],
        rescue_freeze=paths["rescue"],
        source_manifest=paths["source_manifest"],
        output_path=tmp_path / "overlay.json",
    )
    assert result["rescue_success_count"] == 1
    assert result["rescue_preserved_fallback_count"] == 1
    assert result["rows"][0]["answer"] == "new"
    assert result["rows"][1]["failure_reason"] == "empty_text"


def test_overlay_rejects_broken_first_rescue_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    manifest = json.loads(paths["rescue"].with_name("manifest.json").read_text())
    manifest.pop("_artifact_sha256")
    manifest["first_rescue_predictions_freeze_sha256"] = "x" * 64
    _write(paths["rescue"].with_name("manifest.json"), manifest)
    with pytest.raises(overlay.OverlayError, match="manifest"):
        overlay.overlay_mp4_rescue(
            base_freeze=paths["base"],
            rescue_freeze=paths["rescue"],
            source_manifest=paths["source_manifest"],
            output_path=tmp_path / "overlay.json",
        )
