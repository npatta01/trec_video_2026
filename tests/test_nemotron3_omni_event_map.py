from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments import nemotron3_omni_event_map as event_map


def _response(duration: float = 10.0) -> dict[str, object]:
    return {
        "answer": "The person opens the door.",
        "abstain": False,
        "confidence": 0.75,
        "events": [
            {
                "event_id": "event-a",
                "start": 0.0,
                "end": duration / 2,
                "visual_summary": "The scene is established.",
                "audio_summary": None,
            },
            {
                "event_id": "event-b",
                "start": duration / 2,
                "end": duration,
                "visual_summary": "The person opens the door.",
                "audio_summary": "A door sound is audible.",
            },
        ],
        "selected_intervals": [
            {"event_id": "event-b", "start": 5.0, "end": 8.0}
        ],
    }


def test_pinned_identity_and_event_budget_are_immutable() -> None:
    assert event_map.MODEL_ID == (
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
    )
    assert event_map.MODEL_REVISION == "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
    assert event_map.RUNTIME_IMAGE == (
        "docker.io/vllm/vllm-openai:v0.20.0@"
        "sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
    )
    assert event_map.MAX_OUTPUT_TOKENS == 768
    assert event_map.MAX_EVENTS == 6
    assert event_map.MAX_SELECTED_SECONDS == 6.0
    assert event_map.SHARD_COUNTS == (119, 119, 119, 118)


def test_question_strata_and_canary_selection_use_questions_only() -> None:
    rows = [
        {"video_id": "speech-video", "question": "What does the woman say?"},
        {"video_id": "sound-video", "question": "What sound is heard at the end?"},
        {"video_id": "ocr-video", "question": "What color is written on the sign?"},
        {"video_id": "temporal-video", "question": "What happens after the man sits?"},
        {
            "video_id": "unused-video",
            "question": "How many people are visible?",
            "reference": "must not affect selection",
        },
    ]

    assert event_map.question_stratum(rows[0]["question"]) == "speech"
    assert event_map.question_stratum(rows[1]["question"]) == "sound"
    assert event_map.question_stratum(rows[2]["question"]) == "ocr_detail"
    assert event_map.question_stratum(rows[3]["question"]) == "temporal"
    selected = event_map.select_canary_rows(rows)
    assert [row["video_id"] for row in selected] == [
        "speech-video",
        "sound-video",
        "ocr-video",
        "temporal-video",
    ]
    assert all("reference" not in row for row in selected)


def test_chat_request_uses_full_local_video_and_only_audio_toggle_changes() -> None:
    video = Path("/tmp/event-map-video.mp4")
    v_map = event_map.build_chat_request(
        media_path=video,
        question="What happens after the bell?",
        duration_seconds=10.0,
        audio_enabled=False,
        response_format_supported=True,
    )
    av_map = event_map.build_chat_request(
        media_path=video,
        question="What happens after the bell?",
        duration_seconds=10.0,
        audio_enabled=True,
        response_format_supported=True,
    )

    assert v_map["messages"][0]["content"][0] == {
        "type": "video_url",
        "video_url": {"url": video.resolve().as_uri()},
    }
    assert "10.000 seconds" in v_map["messages"][0]["content"][1]["text"]
    assert "whisper" not in json.dumps(v_map).lower()
    assert v_map["mm_processor_kwargs"] == {"use_audio_in_video": False}
    assert av_map["mm_processor_kwargs"] == {"use_audio_in_video": True}
    v_without_toggle = dict(v_map)
    av_without_toggle = dict(av_map)
    v_without_toggle.pop("mm_processor_kwargs")
    av_without_toggle.pop("mm_processor_kwargs")
    assert v_without_toggle == av_without_toggle
    assert v_map["response_format"]["type"] == "json_schema"
    assert v_map["max_tokens"] == event_map.MAX_OUTPUT_TOKENS
    assert v_map["temperature"] == 0.0
    assert v_map["top_p"] == 1.0
    assert v_map["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_request_omits_schema_when_server_does_not_support_it() -> None:
    request = event_map.build_chat_request(
        media_path=Path("/tmp/event-map-video.mp4"),
        question="What is visible?",
        duration_seconds=10.0,
        audio_enabled=False,
        response_format_supported=False,
    )
    assert "response_format" not in request


def test_private_response_validator_accepts_bounded_complete_map() -> None:
    value = event_map.validate_private_response(_response(), duration_seconds=10.0)
    assert value["answer"] == "The person opens the door."
    assert value["events"][0]["event_id"] == "event-a"


def test_private_response_validator_rejects_event_gaps_and_overlapping_selections() -> None:
    gap = _response()
    gap["events"][0]["end"] = 4.0
    with pytest.raises(ValueError, match="cover|gap"):
        event_map.validate_private_response(gap, duration_seconds=10.0)

    overlapping = _response()
    overlapping["selected_intervals"] = [
        {"event_id": "event-b", "start": 5.0, "end": 8.0},
        {"event_id": "event-b", "start": 7.0, "end": 9.0},
    ]
    with pytest.raises(ValueError, match="non-overlapping|chronological"):
        event_map.validate_private_response(overlapping, duration_seconds=10.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["events"].append(value["events"][0]), "event_id"),
        (lambda value: value["events"][0].update({"extra": True}), "extra keys"),
        (lambda value: value["events"][0].update({"start": 2.0}), "clip start"),
        (lambda value: value["events"][0].update({"visual_summary": ""}), "visual_summary"),
        (lambda value: value["selected_intervals"].__setitem__(0, {"event_id": "missing", "start": 1.0, "end": 2.0}), "event_id"),
        (lambda value: value["selected_intervals"].__setitem__(0, {"event_id": "event-b", "start": 1.0, "end": 8.1}), "inside event"),
        (lambda value: value.update({"confidence": float("nan")}), "finite"),
        (lambda value: value.update({"answer": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty one"}), "20 words"),
    ],
)
def test_private_response_validator_rejects_contract_violations(
    mutation, message: str
) -> None:
    value = _response()
    mutation(value)
    with pytest.raises(ValueError, match=message):
        event_map.validate_private_response(value, duration_seconds=10.0)


def test_private_response_validator_rejects_recursive_leakage_markdown_and_nan() -> None:
    for field in ("reference", "meteor", "reasoning", "ground_truth"):
        value = _response()
        value["events"][0][field] = "private"
        with pytest.raises(ValueError, match="forbidden"):
            event_map.validate_private_response(value, duration_seconds=10.0)

    value = _response()
    value["events"][0]["visual_summary"] = "```json\n{}\n```"
    with pytest.raises(ValueError, match="markdown"):
        event_map.validate_private_response(value, duration_seconds=10.0)


def test_completion_parser_requires_stop_and_rejects_thinking_or_extra_keys() -> None:
    valid = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(_response())}}]}
    parsed = event_map.parse_completion(valid, duration_seconds=10.0)
    assert parsed["answer"] == "The person opens the door."

    for finish_reason in (None, "length"):
        invalid = {"choices": [{"finish_reason": finish_reason, "message": {"content": json.dumps(_response())}}]}
        with pytest.raises(ValueError, match="finish"):
            event_map.parse_completion(invalid, duration_seconds=10.0)

    thinking = {"choices": [{"finish_reason": "stop", "message": {"content": "<think>secret</think>"}}]}
    with pytest.raises(ValueError, match="thinking"):
        event_map.parse_completion(thinking, duration_seconds=10.0)


def test_one_schema_retry_records_primary_and_retry_validity_without_retaining_raw_text() -> None:
    responses = [
        {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(_response())}}]},
    ]

    result = event_map.run_one_request(
        request={"video_id": "video-a", "question": "What happens?"},
        duration_seconds=10.0,
        post_json=lambda _payload: responses.pop(0),
        media_path=Path("/tmp/video-a.mp4"),
        audio_enabled=False,
        response_format_supported=True,
        elapsed_clock=iter((1.0, 2.5)).__next__,
    )

    assert result["status"] == "passed"
    assert result["schema_version"] == 1
    assert result["memory_telemetry"]["path"] == "telemetry/nvidia-smi.csv"
    assert result["primary_valid"] is False
    assert result["retry_valid"] is True
    assert result["retry_used"] is True
    assert result["answer"] == "The person opens the door."
    assert "raw" not in result
    assert len(result["request_sha256"]) == 64
    assert len(result["primary_request_sha256"]) == 64
    assert len(result["retry_request_sha256"]) == 64
    assert result["primary_request_sha256"] != result["retry_request_sha256"]
    assert result["request_sha256"] == result["retry_request_sha256"]
    assert len(result["response_sha256"]) == 64
    assert result["elapsed_seconds"] == pytest.approx(1.5)


def test_failed_schema_retry_is_bounded_and_technical() -> None:
    invalid = {"choices": [{"finish_reason": "length", "message": {"content": "bad"}}]}
    result = event_map.run_one_request(
        request={"video_id": "video-a", "question": "What happens?"},
        duration_seconds=10.0,
        post_json=lambda _payload: invalid,
        media_path=Path("/tmp/video-a.mp4"),
        audio_enabled=True,
        response_format_supported=False,
        elapsed_clock=iter((1.0, 2.0, 3.0)).__next__,
    )
    assert result["status"] == "failed"
    assert result["schema_version"] == 1
    assert result["model_id"] == event_map.MODEL_ID
    assert result["memory_telemetry"]["path"] == "telemetry/nvidia-smi.csv"
    assert result["primary_valid"] is False
    assert result["retry_valid"] is False
    assert result["retry_used"] is True
    assert len(result["primary_request_sha256"]) == 64
    assert len(result["retry_request_sha256"]) == 64
    assert result["error_type"] == "schema_invalid"
    assert "error_message" not in result


def test_retry_transport_failure_returns_a_durable_failed_row() -> None:
    calls = 0

    def post(_payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "not json"},
                    }
                ]
            }
        raise TimeoutError("retry transport timeout")

    result = event_map.run_one_request(
        request={"video_id": "video-a", "question": "What happens?"},
        duration_seconds=10.0,
        post_json=post,
        media_path=Path("/tmp/video-a.mp4"),
        audio_enabled=True,
        response_format_supported=True,
        elapsed_clock=iter((1.0, 2.0)).__next__,
    )
    assert result["status"] == "failed"
    assert result["error_type"] == "request_failed"
    assert result["retry_used"] is True
    assert result["retry_valid"] is False
    assert len(result["retry_request_sha256"]) == 64


def test_gpu_identity_accepts_real_l40s_reported_memory() -> None:
    assert event_map.validate_gpu_identity("NVIDIA L40S", 46068) == {
        "gpu_name": "L40S",
        "gpu_memory_mib": 46068,
    }
    with pytest.raises(ValueError, match="L40S"):
        event_map.validate_gpu_identity("NVIDIA H100", 81920)
    with pytest.raises(ValueError, match="reported"):
        event_map.validate_gpu_identity("NVIDIA L40S", 44000)


def test_validate_normalized_media_accepts_ffprobe_string_durations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    derived = tmp_path / "derived.mp4"
    source.write_bytes(b"source")
    derived.write_bytes(b"derived")

    identity = event_map.validate_normalized_media(
        source_path=source,
        derived_path=derived,
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_probe={
            "streams": [
                {"codec_type": "video", "codec_name": "vp9"},
                {"codec_type": "audio", "codec_name": "opus"},
            ],
            "format": {"duration": "24.588000"},
        },
        derived_probe={
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "24.574000"},
        },
    )

    assert identity["source_duration_seconds"] == pytest.approx(24.588)
    assert identity["derived_duration_seconds"] == pytest.approx(24.574)


def test_normalize_media_file_disables_metadata_and_chapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(event_map.subprocess, "run", fake_run)
    monkeypatch.setattr(
        event_map,
        "validate_normalized_media",
        lambda **_kwargs: {"status": "validated"},
    )

    result = event_map.normalize_media_file(
        source_path=tmp_path / "source.mkv",
        derived_path=tmp_path / "derived.mp4",
        expected_source_sha256="0" * 64,
    )

    assert result == {"status": "validated"}
    assert len(commands) == 1
    command = commands[0]
    metadata_index = command.index("-map_metadata")
    assert command[metadata_index : metadata_index + 4] == [
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
    ]


def _plaintext_map_payload(*, duration: float = 20.0) -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": (
                        "E1|1.0|Opening shot shows a person entering the room.\n"
                        "E2|10.0|The person places an object on the table.\n"
                        "E3|19.0|The person leaves and the room becomes empty.\n"
                        "ANSWER|The person enters, places an object, then leaves."
                    )
                },
            }
        ]
    }


def _plaintext_selector_payload() -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": (
                        "SELECT|E1,E2\n"
                        "ANSWER|The person places an object on the table."
                    )
                },
            }
        ]
    }


def test_plaintext_requests_disable_thinking_and_guided_decoding() -> None:
    map_request = event_map.build_plaintext_map_request(
        media_path=Path("/tmp/event-map-video.mp4"),
        question="What happens?",
        duration_seconds=20.0,
        audio_enabled=True,
    )
    selector_request = event_map.build_plaintext_selector_request(
        question="What happens?",
        frozen_map="E1|1.0|Opening shot.\nE2|19.0|Ending shot.",
    )

    assert map_request["max_tokens"] == 256
    assert map_request["stop"] == ["\nEND"]
    assert map_request["temperature"] == 0.0
    assert map_request["top_p"] == 1.0
    assert map_request["top_k"] == 1
    assert map_request["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in map_request
    assert map_request["mm_processor_kwargs"] == {"use_audio_in_video": True}
    assert selector_request["max_tokens"] == 64
    assert selector_request["stop"] == ["\nEND"]
    assert selector_request["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in selector_request
    assert "video_url" not in json.dumps(selector_request)
    assert "exactly one strongest event" in selector_request["messages"][0]["content"][0]["text"]
    assert "SELECT|E1" in selector_request["messages"][0]["content"][0]["text"]


def test_plaintext_map_parser_accepts_stop_trimmed_end_marker() -> None:
    parsed = event_map.parse_plaintext_map(
        _plaintext_map_payload(), duration_seconds=20.0
    )

    assert parsed["answer"].startswith("The person enters")
    assert [event["event_id"] for event in parsed["events"]] == ["E1", "E2", "E3"]
    assert [event["timestamp"] for event in parsed["events"]] == [1.0, 10.0, 19.0]

    cells = event_map.derive_anchor_cells(parsed["events"], duration_seconds=20.0)
    assert cells == [
        {"event_id": "E1", "start": 0.0, "end": 5.5},
        {"event_id": "E2", "start": 5.5, "end": 14.5},
        {"event_id": "E3", "start": 14.5, "end": 20.0},
    ]

    windows = event_map.expand_selected_anchors(
        parsed["events"], ["E1", "E2"], duration_seconds=20.0
    )
    assert windows == [
        {"event_id": "E1", "start": 0.0, "end": 3.0},
        {"event_id": "E2", "start": 8.5, "end": 11.5},
    ]


def test_plaintext_map_parser_allows_timeline_without_optional_direct_answer() -> None:
    payload = _plaintext_map_payload()
    payload["choices"][0]["message"]["content"] = (
        "E1|1.0|Opening shot shows a person entering the room.\n"
        "E2|10.0|The person places an object on the table.\n"
        "E3|19.0|The person leaves and the room becomes empty."
    )
    parsed = event_map.parse_plaintext_map(payload, duration_seconds=20.0)
    assert parsed["answer"] is None
    assert "ANSWER|" not in parsed["map_text"]


def test_plaintext_selector_parser_and_edge_shift_are_deterministic() -> None:
    parsed_map = event_map.parse_plaintext_map(
        _plaintext_map_payload(), duration_seconds=20.0
    )
    selected = event_map.parse_plaintext_selector(
        _plaintext_selector_payload(),
        event_ids={event["event_id"] for event in parsed_map["events"]},
    )
    assert selected == {
        "selected_event_ids": ["E1", "E2"],
        "answer": "The person places an object on the table.",
    }

    edge_windows = event_map.expand_selected_anchors(
        [{"event_id": "E1", "timestamp": 0.2, "description": "opening"}],
        ["E1"],
        duration_seconds=2.0,
    )
    assert edge_windows == [{"event_id": "E1", "start": 0.0, "end": 2.0}]


def test_plaintext_map_accepts_last_third_anchor_and_sixteen_word_description() -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": (
                        "E1|0.0|The opening scene establishes the setting.\n"
                        "E2|5.0|The subject performs the central action in the middle of the clip.\n"
                        "E3|10.0|The final scene shows the subject leaving the room.\n"
                        "ANSWER|The subject performs an action and leaves."
                    )
                },
            }
        ]
    }
    parsed = event_map.parse_plaintext_map(payload, duration_seconds=13.765)
    assert parsed["events"][-1]["timestamp"] == 10.0


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("E1|1.0|Only one event.\nANSWER|answer", "2–6"),
        ("E1|10.0|Opening.\nE2|5.0|Backwards.\nANSWER|answer", "increasing"),
        ("E1|1.0|Opening.\nE2|19.0|Ending.\nEXTRA|bad\nANSWER|answer", "line"),
        ("E1|1.0|Opening.", "2–6"),
    ],
)
def test_plaintext_map_parser_rejects_malformed_protocol(
    content: str, message: str
) -> None:
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
    with pytest.raises(ValueError, match=message):
        event_map.parse_plaintext_map(payload, duration_seconds=20.0)


def test_plaintext_selector_rejects_unknown_or_overlapping_selection() -> None:
    unknown = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "SELECT|E9\nANSWER|answer"},
            }
        ]
    }
    with pytest.raises(ValueError, match="unknown"):
        event_map.parse_plaintext_selector(unknown, event_ids={"E1", "E2"})

    events = [
        {"event_id": "E1", "timestamp": 1.0, "description": "opening"},
        {"event_id": "E2", "timestamp": 2.0, "description": "middle"},
    ]
    with pytest.raises(ValueError, match="overlap"):
        event_map.expand_selected_anchors(events, ["E1", "E2"], duration_seconds=4.0)


def test_checksum_manifest_and_terminal_marker_bind_exact_payload(tmp_path: Path) -> None:
    payload = tmp_path / "payload.jsonl"
    payload.write_text('{"video_id":"video-a"}\n', encoding="utf-8")
    checksums = event_map.write_payload_checksums(tmp_path, [payload])
    assert checksums.read_text(encoding="utf-8") == (
        f"{hashlib.sha256(payload.read_bytes()).hexdigest()}  ./payload.jsonl\n"
    )
    marker = event_map.write_terminal_marker(
        tmp_path,
        run_id="vqa-event-map-v-map-chunk-00",
        attempt_uuid="attempt-uuid",
        output_identity="hf://buckets/private/event-map-00",
        payload_checksum=hashlib.sha256(checksums.read_bytes()).hexdigest(),
        row_count=1,
    )
    assert json.loads(marker.read_text(encoding="utf-8"))["payload_checksum"] == hashlib.sha256(checksums.read_bytes()).hexdigest()
