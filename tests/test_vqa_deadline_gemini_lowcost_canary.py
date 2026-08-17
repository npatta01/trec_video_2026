from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vqa_deadline_gemini_lowcost_canary as runner  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def panel_fixture(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    source_root = data_root / "vqa" / "2025" / "training" / "videos_original"
    source_root.mkdir(parents=True)
    historical_root = tmp_path / "historical"
    rows: list[dict[str, Any]] = []
    for index, stratum in enumerate(("temporal", "causal", "ocr_detail", "counting", "speech")):
        video_id = f"A{index:010d}"
        source = source_root / f"{video_id}.mkv"
        source.write_bytes(f"source-{video_id}".encode())
        delivery = historical_root / "deliveries" / video_id / "delivery-media" / f"{video_id}.webm"
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
    return {"panel": panel, "data_root": data_root, "historical_root": historical_root}


class FakeFiles:
    def __init__(self) -> None:
        self.upload_calls: list[Path] = []

    def upload(self, *, file: Path, config: Any) -> Any:
        self.upload_calls.append(file)
        return SimpleNamespace(
            name=f"files/{file.stem}",
            uri=f"https://provider.invalid/files/{file.stem}",
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
            "model_version": "gemini-3.7-flash-test",
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


def test_build_request_uses_low_cost_settings() -> None:
    request = runner.build_request(
        row={"video_id": "A", "question": "What happens?"},
        upload={"file_uri": "https://provider.invalid/files/A", "mime_type": "video/webm"},
    )
    video = request.contents[0].parts[0]
    assert request.model == "gemini-3.7-flash"
    assert request.config.max_output_tokens == 256
    assert video.video_metadata.fps == 1.0
    assert video.media_resolution.level == runner.MEDIA_RESOLUTION
    assert request.config.thinking_config.thinking_level == runner.THINKING_LEVEL
    assert request.config.thinking_config.include_thoughts is False


def test_dry_run_has_five_requests_and_hard_cap_below_ten_cents(panel_fixture: dict[str, Path]) -> None:
    result = runner.run_canary(
        panel_path=panel_fixture["panel"],
        data_root=panel_fixture["data_root"],
        historical_root=panel_fixture["historical_root"],
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert result["row_count"] == 5
    assert result["upload_count"] == 5
    assert result["request_count"] == 5
    assert result["estimated_cost_usd"] <= 0.10


def test_live_run_makes_exactly_five_calls_and_freezes_usage(
    panel_fixture: dict[str, Path], tmp_path: Path
) -> None:
    client = FakeClient()
    output_root = tmp_path / "lowcost"
    result = runner.run_canary(
        panel_path=panel_fixture["panel"],
        data_root=panel_fixture["data_root"],
        historical_root=panel_fixture["historical_root"],
        output_root=output_root,
        client=client,
        created_at="2026-08-16T20:00:00Z",
    )
    assert result["status"] == "predictions_frozen"
    assert len(client.files.upload_calls) == 5
    assert len(client.models.calls) == 5
    assert result["observed_cost_usd"] == 0.0006
    predictions = json.loads((output_root / "predictions-freeze.json").read_text())
    assert predictions["row_count"] == 5
    assert predictions["prompt_tokens"] == 500
    assert predictions["output_tokens"] == 60
