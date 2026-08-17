from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vqa_deadline_gemini_lowres_canary as runner  # noqa: E402


def test_build_request_uses_4fps_lowres_highthinking() -> None:
    request = runner.build_request(
        row={"video_id": "A", "question": "What happens?"},
        upload={"file_uri": "https://provider.invalid/files/A", "mime_type": "video/webm"},
    )
    video = request.contents[0].parts[0]
    assert request.model == "gemini-3.7-flash"
    assert request.config.max_output_tokens == 2048
    assert video.video_metadata.fps == 4.0
    assert video.media_resolution.level == runner.MEDIA_RESOLUTION
    assert request.config.thinking_config.thinking_level == runner.THINKING_LEVEL
    assert request.config.thinking_config.include_thoughts is False


def test_dry_run_has_five_requests_under_hard_cap(tmp_path: Path) -> None:
    rows = [
        {"duration_seconds": duration}
        for duration in (21.268, 19.188, 13.788, 28.828, 24.588)
    ]
    result = runner._estimated_cost(rows)
    assert result["estimated_cost_usd"] == 0.135669
    assert result["estimated_cost_usd"] <= runner.MAX_ESTIMATED_COST_USD
    assert tmp_path.is_dir()


def test_usage_cost_bills_hidden_thinking_as_output() -> None:
    prompt, billed_output, cost = runner._usage_cost(
        {"prompt_tokens": 100, "output_tokens": 7, "total_tokens": 352}
    )
    assert prompt == 100
    assert billed_output == 252
    assert cost == 0.00102
