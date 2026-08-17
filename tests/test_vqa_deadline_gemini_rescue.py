from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vqa_deadline_gemini_rescue as runner  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(path: Path, body: dict[str, Any]) -> None:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload = {**body, "_artifact_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def rescue_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    media_root = tmp_path / "testing"
    videos = media_root / "videos"
    videos.mkdir(parents=True)
    raw_requests: list[dict[str, Any]] = []
    for query_id in range(1, 28):
        video_id = f"clip-{query_id - 1}.webm" if query_id > 1 else "clip-0.webm"
        if query_id == 2:
            video_id = "clip-0.webm"
        media = videos / video_id
        if not media.exists():
            media.write_bytes(f"video-{video_id}".encode())
        digest = _sha256(media)
        raw = {
            "availability": "available",
            "frame_paths": [],
            "frame_timestamps": [0.0, 2.0],
            "query_id": query_id,
            "question": f"What happens in clip {query_id}?",
            "source_sha256": digest,
            "video_id": video_id,
        }
        raw_requests.append(raw)
    requests_path = tmp_path / "requests.json"
    requests_path.write_text(json.dumps(raw_requests) + "\n", encoding="utf-8")
    loaded = runner.full.load_input_rows(
        requests_path, media_root=media_root, expected_count=27
    )
    reasons = ["max_tokens"] * 11 + ["empty_text"] * 9 + ["provider_upload_failed"] * 7
    freeze_rows = [
        {
            "query_id": row["query_id"],
            "video_id": row["video_id"],
            "status": "upload_failed" if reason == "provider_upload_failed" else "failed",
            "answer": "",
            "elapsed_seconds": 0.0,
            "failure_reason": reason,
            "raw_response_sha256": None,
        }
        for row, reason in zip(loaded, reasons, strict=True)
    ]
    manifest_body = {
        "schema_version": 1,
        "status": "inputs_frozen",
        "input": {"path": str(requests_path), "sha256": _sha256(requests_path)},
        "rows": loaded,
    }
    manifest_path = tmp_path / "manifest.json"
    _write_artifact(manifest_path, manifest_body)
    freeze_body = {
        "schema_version": 1,
        "status": "predictions_frozen",
        "model": "gemini-3.7-flash",
        "manifest_sha256": _sha256(manifest_path),
        "row_count": 27,
        "rows": freeze_rows,
    }
    freeze_path = tmp_path / "predictions-freeze.json"
    _write_artifact(freeze_path, freeze_body)

    monkeypatch.setattr(runner, "BASE_PREDICTIONS_SHA256", _sha256(freeze_path))
    monkeypatch.setattr(runner, "EXPECTED_QUERY_IDS", tuple(range(1, 28)))
    monkeypatch.setattr(
        runner,
        "EXPECTED_REASON_BY_QUERY",
        {
            query_id: reason
            for query_id, reason in zip(range(1, 28), reasons, strict=True)
        },
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_REASON_COUNTS",
        {"max_tokens": 11, "empty_text": 9, "provider_upload_failed": 7},
    )
    monkeypatch.setattr(runner, "EXPECTED_PANEL_COUNT", 27)
    return {
        "tmp": tmp_path,
        "requests": requests_path,
        "media_root": media_root,
        "freeze": freeze_path,
        "manifest": manifest_path,
        "loaded": loaded,
    }


def test_selector_rederives_exact_reasons_and_26_unique_videos(
    rescue_fixture: dict[str, Any],
) -> None:
    rows = runner.select_rescue_rows(
        freeze_path=rescue_fixture["freeze"],
        manifest_path=rescue_fixture["manifest"],
        requests_path=rescue_fixture["requests"],
        media_root=rescue_fixture["media_root"],
    )
    assert [row["query_id"] for row in rows] == list(range(1, 28))
    assert len({row["video_id"] for row in rows}) == 26
    assert runner.selection_reason_counts(rows) == {
        "max_tokens": 11,
        "empty_text": 9,
        "provider_upload_failed": 7,
    }


def test_request_uses_8192_tokens_and_fixed_native_av_settings(
    rescue_fixture: dict[str, Any],
) -> None:
    row = runner.select_rescue_rows(
        freeze_path=rescue_fixture["freeze"],
        manifest_path=rescue_fixture["manifest"],
        requests_path=rescue_fixture["requests"],
        media_root=rescue_fixture["media_root"],
    )[0]
    request = runner.build_rescue_request(
        row=row,
        upload={"file_uri": "https://provider.invalid/file", "mime_type": "video/webm"},
    )
    assert request.config.max_output_tokens == 8192
    assert request.contents[0].parts[0].video_metadata.fps == 4.0
    assert request.contents[0].parts[0].media_resolution.level.value == "MEDIA_RESOLUTION_LOW"
    assert request.config.thinking_config.thinking_level.value == "HIGH"


def test_dry_run_reports_27_rows_26_videos_and_cost(
    rescue_fixture: dict[str, Any],
) -> None:
    result = runner.run_rescue(
        freeze_path=rescue_fixture["freeze"],
        manifest_path=rescue_fixture["manifest"],
        requests_path=rescue_fixture["requests"],
        media_root=rescue_fixture["media_root"],
        output_root=rescue_fixture["tmp"] / "run",
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert result["row_count"] == 27
    assert result["upload_count"] == 26
    assert result["request_count"] == 27
    assert result["estimated_cost_usd"] < 2.0
    assert result["max_output_tokens"] == 8192


def test_estimated_cap_blocks_before_provider_call(
    rescue_fixture: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="estimated cost"):
        runner.run_rescue(
            freeze_path=rescue_fixture["freeze"],
            manifest_path=rescue_fixture["manifest"],
            requests_path=rescue_fixture["requests"],
            media_root=rescue_fixture["media_root"],
            output_root=rescue_fixture["tmp"] / "run",
            max_estimated_cost_usd=0.0,
        )


def test_source_mutation_is_rejected(rescue_fixture: dict[str, Any]) -> None:
    requests = json.loads(rescue_fixture["requests"].read_text())
    requests[0]["question"] = "mutated question"
    rescue_fixture["requests"].write_text(json.dumps(requests) + "\n")
    with pytest.raises(ValueError, match="input|binding|hash"):
        runner.select_rescue_rows(
            freeze_path=rescue_fixture["freeze"],
            manifest_path=rescue_fixture["manifest"],
            requests_path=rescue_fixture["requests"],
            media_root=rescue_fixture["media_root"],
        )


def test_per_query_reason_swap_is_rejected(
    rescue_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = json.loads(rescue_fixture["freeze"].read_text())
    freeze.pop("_artifact_sha256")
    freeze["rows"][0]["failure_reason"] = "empty_text"
    freeze["rows"][11]["failure_reason"] = "max_tokens"
    _write_artifact(rescue_fixture["freeze"], freeze)
    monkeypatch.setattr(
        runner, "BASE_PREDICTIONS_SHA256", _sha256(rescue_fixture["freeze"])
    )

    with pytest.raises(ValueError, match="per-query reasons"):
        runner.select_rescue_rows(
            freeze_path=rescue_fixture["freeze"],
            manifest_path=rescue_fixture["manifest"],
            requests_path=rescue_fixture["requests"],
            media_root=rescue_fixture["media_root"],
        )


class _FakeFiles:
    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.failed_once = False

    def upload(self, *, file: Path, config: Any) -> Any:
        self.calls.append(file)
        if not self.failed_once:
            self.failed_once = True
            raise genai_errors.ServerError(503, {"message": "retry"})
        return SimpleNamespace(
            name=f"files/{file.stem}",
            uri=f"https://provider.invalid/{file.stem}",
            mime_type=config.mime_type,
            state="ACTIVE",
            size_bytes=file.stat().st_size,
        )


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        index = len(self.calls)
        return {
            "response_id": f"r-{index}",
            "model_version": "gemini-3.7-flash-test",
            "text": "" if index == 1 else f"answer {index}",
            "finish_reason": "MAX_TOKENS" if index == 1 else "STOP",
            "usage_metadata": {
                "prompt_token_count": 100,
                "candidates_token_count": 10,
                "total_token_count": 110,
            },
        }


class _FakeClient:
    def __init__(self) -> None:
        self.files = _FakeFiles()
        self.models = _FakeModels()


def test_live_run_freezes_success_and_failure_terminal_rows(
    rescue_fixture: dict[str, Any],
) -> None:
    client = _FakeClient()
    result = runner.run_rescue(
        freeze_path=rescue_fixture["freeze"],
        manifest_path=rescue_fixture["manifest"],
        requests_path=rescue_fixture["requests"],
        media_root=rescue_fixture["media_root"],
        output_root=rescue_fixture["tmp"] / "run",
        client=client,
        run=True,
        sleep=lambda _seconds: None,
    )
    assert result["status"] == "predictions_frozen"
    assert len(client.files.calls) == 27  # one exact 503 retry
    freeze = json.loads(
        (rescue_fixture["tmp"] / "run" / "predictions-freeze.json").read_text()
    )
    assert freeze["status"] == "predictions_frozen"
    assert freeze["base_predictions_freeze_sha256"] == runner.BASE_PREDICTIONS_SHA256
    assert freeze["selected_query_ids"] == list(range(1, 28))
    assert freeze["max_output_tokens"] == 8192
    rescue_manifest = json.loads(
        (rescue_fixture["tmp"] / "run" / "manifest.json").read_text()
    )
    assert rescue_manifest["status"] == "inputs_frozen"
    assert rescue_manifest["model"] == "gemini-3.7-flash"
    assert rescue_manifest["model_settings"]["max_output_tokens"] == 8192
    assert len(rescue_manifest["rows"]) == 27
    assert freeze["manifest_sha256"] == _sha256(
        rescue_fixture["tmp"] / "run" / "manifest.json"
    )
    assert freeze["rows"][0]["status"] == "failed"
    assert freeze["rows"][0]["failure_reason"] == "max_tokens"
    assert freeze["rows"][1]["status"] == "succeeded"
    summary = json.loads((rescue_fixture["tmp"] / "run" / "summary.json").read_text())
    assert summary["upload_count"] == 26
    source_manifest = json.loads(rescue_fixture["manifest"].read_text())
    rescue_manifest = json.loads(
        (rescue_fixture["tmp"] / "run" / "manifest.json").read_text()
    )
    source_by_id = {row["query_id"]: row for row in source_manifest["rows"]}
    for row in rescue_manifest["rows"]:
        for key, value in source_by_id[row["query_id"]].items():
            assert row[key] == value
