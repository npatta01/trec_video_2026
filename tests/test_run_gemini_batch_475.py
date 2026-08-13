from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gemini_batch_475.py"
SPEC = importlib.util.spec_from_file_location("gemini_batch_475", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_direct_request_keeps_native_av_and_question_together() -> None:
    request = MODULE.build_inlined_request(
        condition="direct-answer",
        video_id="video_1",
        question="What happens next?",
        duration_seconds=12.5,
        file_uri="https://example.invalid/files/one",
        mime_type="video/webm",
    )

    assert request.metadata == {
        "condition": "direct-answer",
        "video_id": "video_1",
    }
    parts = request.contents.parts
    assert parts[0].file_data.file_uri == "https://example.invalid/files/one"
    assert parts[0].file_data.mime_type == "video/webm"
    assert parts[0].video_metadata.fps == 4.0
    assert parts[0].media_resolution.level.value == "MEDIA_RESOLUTION_HIGH"
    assert "What happens next?" in parts[1].text
    assert request.config.temperature == 0.0
    assert request.config.seed == 42
    assert request.config.thinking_config.include_thoughts is False
    assert request.config.thinking_config.thinking_level.value == "HIGH"


def test_map_request_is_question_blind_and_requests_transcript() -> None:
    request = MODULE.build_inlined_request(
        condition="transcript-map",
        video_id="video_2",
        question="SECRET QUESTION",
        duration_seconds=17.25,
        file_uri="https://example.invalid/files/two",
        mime_type="video/webm",
    )

    prompt = request.contents.parts[1].text
    assert "SECRET QUESTION" not in prompt
    assert "question-blind" in prompt
    assert "verbatim speech" in prompt
    assert "17.250-second" in prompt
    assert request.config.max_output_tokens == 8192


def test_cost_guard_rejects_more_than_frozen_panel() -> None:
    rows = [{"video_id": str(index)} for index in range(476)]
    try:
        MODULE.validate_panel_size(rows)
    except ValueError as error:
        assert str(error) == "batch panel exceeds frozen 475-row boundary"
    else:
        raise AssertionError("oversized paid panel was accepted")


def test_mkv_delivery_strategy_preserves_video_and_limits_audio_conversion() -> None:
    assert MODULE.mkv_delivery_strategy("vp9", "opus") == "webm-copy"
    assert MODULE.mkv_delivery_strategy("h264", "aac") == "mp4-copy"
    assert MODULE.mkv_delivery_strategy("vp9", "aac") == "mp4-copy"
    assert MODULE.mkv_delivery_strategy("h264", "opus") == "mp4-audio-aac"


def test_collect_terminal_outputs_replaces_failed_row_with_retry(tmp_path: Path) -> None:
    root = tmp_path
    protocol = {
        "rows": [
            {"video_id": "one", "question": "q1", "question_stratum": "temporal"},
            {"video_id": "two", "question": "q2", "question_stratum": "speech"},
        ]
    }
    (root / "protocol.json").write_text(json.dumps(protocol))

    def snapshot(name: str, condition: str, responses: list[dict]) -> None:
        directory = root / "status" / f"{name}-{condition}"
        directory.mkdir(parents=True)
        (directory / "terminal.json").write_text(
            json.dumps(
                {
                    "state": "JOB_STATE_SUCCEEDED",
                    "job": {"name": f"batches/{name}", "dest": {"inlinedResponses": responses}},
                }
            )
        )

    def response(video_id: str, condition: str, *, text: str | None = None, error: str | None = None) -> dict:
        item = {"metadata": {"video_id": video_id, "condition": condition}}
        if error is not None:
            item["error"] = {"code": 13, "message": error}
        else:
            item["response"] = {
                "candidates": [{"content": {"parts": [{"text": text}]}}],
                "usageMetadata": {"totalTokenCount": 10},
            }
        return item

    snapshot(
        "shard",
        "direct-answer",
        [
            response("one", "direct-answer", text="first"),
            response("two", "direct-answer", error="internal"),
        ],
    )
    snapshot(
        "retry",
        "direct-answer",
        [response("two", "direct-answer", text="second")],
    )

    rows = MODULE.collect_terminal_outputs(root, "direct-answer")
    assert [(row["video_id"], row["text"]) for row in rows] == [
        ("one", "first"),
        ("two", "second"),
    ]
    assert rows[1]["prior_errors"] == [{"code": 13, "message": "internal"}]
