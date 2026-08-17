from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vqa_deadline_gemini_canary as runner  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def panel_fixture(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    source_root = data_root / "vqa" / "2025" / "training" / "videos_original"
    source_root.mkdir(parents=True)
    delivery_root = tmp_path / "historical" / "deliveries"
    rows: list[dict[str, Any]] = []
    strata = ("temporal", "causal", "ocr_detail", "counting", "speech")
    for index, stratum in enumerate(strata):
        video_id = f"A{index:010d}"
        source = source_root / f"{video_id}.mkv"
        source.write_bytes(f"source-{video_id}".encode())
        delivery = delivery_root / video_id / "delivery-media" / f"{video_id}.webm"
        delivery.parent.mkdir(parents=True)
        delivery.write_bytes(f"delivery-{video_id}".encode())
        rows.append(
            {
                "video_id": video_id,
                "stratum": stratum,
                "question": f"What happens in {video_id}?",
                "duration_seconds": 10.0 + index,
                "source_path": f"vqa/2025/training/videos_original/{video_id}.mkv",
                "source_sha256": _sha256(source),
            }
        )
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"schema_version": 1, "rows": rows}) + "\n")
    return {"panel": panel, "data_root": data_root, "historical_root": delivery_root.parent}


class FakeFiles:
    def __init__(self) -> None:
        self.upload_calls: list[tuple[Path, Any]] = []

    def upload(self, *, file: Path, config: Any) -> Any:
        self.upload_calls.append((file, config))
        video_id = file.stem
        return SimpleNamespace(
            name=f"files/{video_id}",
            uri=f"https://provider.invalid/files/{video_id}",
            mime_type="video/webm",
            state="ACTIVE",
            size_bytes=file.stat().st_size,
        )


class FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        index = len(self.calls)
        return {
            "response_id": f"response-{index}",
            "model_version": "gemini-3.7-flash-20260801",
            "text": f"direct answer {index}",
            "finish_reason": "STOP",
            "usage_metadata": {
                "prompt_token_count": 100,
                "candidates_token_count": 12,
                "total_token_count": 112,
            },
        }


class FakeClient:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.models = FakeModels()


def test_build_request_uses_native_av_candidate_model(panel_fixture: dict[str, Path]) -> None:
    row = {
        "video_id": "A0000000000",
        "question": "What happens?",
        "duration_seconds": 10.0,
    }
    upload = {
        "file_uri": "https://provider.invalid/files/A0000000000",
        "mime_type": "video/webm",
    }

    request = runner.build_request(row=row, upload=upload)

    assert request.model == "gemini-3.7-flash"
    assert request.config.max_output_tokens == 2048
    assert len(request.contents[0].parts) == 2
    assert request.contents[0].parts[0].file_data.file_uri == upload["file_uri"]
    assert "What happens?" in request.contents[0].parts[1].text


def test_dry_run_has_five_requests_and_stays_under_cost_cap(
    panel_fixture: dict[str, Path], tmp_path: Path
) -> None:
    result = runner.run_canary(
        panel_path=panel_fixture["panel"],
        data_root=panel_fixture["data_root"],
        historical_root=panel_fixture["historical_root"],
        output_root=tmp_path / "gemini",
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["row_count"] == 5
    assert result["upload_count"] == 5
    assert result["request_count"] == 5
    assert result["estimated_cost_usd"] < 0.15
    assert not (tmp_path / "gemini").exists()


def test_live_run_uploads_and_calls_each_row_once_then_freezes(
    panel_fixture: dict[str, Path], tmp_path: Path
) -> None:
    client = FakeClient()
    output_root = tmp_path / "gemini"

    result = runner.run_canary(
        panel_path=panel_fixture["panel"],
        data_root=panel_fixture["data_root"],
        historical_root=panel_fixture["historical_root"],
        output_root=output_root,
        client=client,
        created_at="2026-08-16T20:00:00Z",
    )

    assert result["status"] == "predictions_frozen"
    assert result["row_count"] == 5
    assert result["completed_count"] == 5
    assert len(client.files.upload_calls) == 5
    assert len(client.models.calls) == 5
    assert (output_root / "manifest.json").is_file()
    assert (output_root / "raw-freeze.json").is_file()
    assert (output_root / "predictions-freeze.json").is_file()
    assert (output_root / "summary.json").is_file()

    freeze = json.loads((output_root / "predictions-freeze.json").read_text())
    assert freeze["status"] == "predictions_frozen"
    assert len(freeze["rows"]) == 5
    assert all(row["answer"] for row in freeze["rows"])


def test_live_run_refuses_duplicate_output_before_provider_calls(
    panel_fixture: dict[str, Path], tmp_path: Path
) -> None:
    output_root = tmp_path / "gemini"
    output_root.mkdir()
    client = FakeClient()

    with pytest.raises(FileExistsError):
        runner.run_canary(
            panel_path=panel_fixture["panel"],
            data_root=panel_fixture["data_root"],
            historical_root=panel_fixture["historical_root"],
            output_root=output_root,
            client=client,
        )

    assert client.files.upload_calls == []
    assert client.models.calls == []
