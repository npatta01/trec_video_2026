from __future__ import annotations

import contextlib
import hashlib
import json
import struct
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.qwen3_omni_structured import (
    REPAIR_PROMPT,
    SELECTOR_PROMPT,
    EXPECTED_EOS_TOKEN_ID,
    EXPECTED_PAD_TOKEN_ID,
    _runtime_generator,
    build_selector_messages,
    extract_terminal_selector_json,
    run_selector_with_retry,
    validate_selector_request,
)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def selector_request(tmp_path: Path) -> dict[str, object]:
    media = tmp_path / "media" / "video-01.mkv"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"frozen-video")
    request: dict[str, object] = {
        "schema_version": 1,
        "run_id": "silver-a1",
        "video_id": "video-01",
        "video_path": "media/video-01.mkv",
        "media_sha256": hashlib.sha256(b"frozen-video").hexdigest(),
        "duration_seconds": 20.0,
        "question": "What happens after the bell?",
        "question_stratum": "temporal",
        "reference": "The door opens.",
        "whisper_segments": [
            {
                "segment_id": 0,
                "start_seconds": 0.0,
                "end_seconds": 8.0,
                "source_line": "[00:00:00.000] A bell rings.",
            },
            {
                "segment_id": 1,
                "start_seconds": 8.0,
                "end_seconds": 20.0,
                "source_line": "[00:00:08.000] The door opens.",
            },
        ],
        "whisper_dropped_trailing_line_count": 0,
        "whisper_sha256": "a" * 64,
    }
    request["request_sha256"] = canonical_sha256(request)
    return request


def valid_selector_json() -> str:
    return json.dumps(
        {
            "status": "localized",
            "intervals": [{"start_seconds": 7.0, "end_seconds": 12.0}],
            "whisper_segment_ids": [1],
        }
    )


def test_validate_selector_request_binds_hash_media_and_segments(tmp_path: Path) -> None:
    request = selector_request(tmp_path)

    normalized = validate_selector_request(request, tmp_path)

    assert normalized["video_id"] == "video-01"
    assert normalized["whisper_segment_count"] == 2
    assert normalized["media_path"] == tmp_path / "media" / "video-01.mkv"


def test_validate_selector_request_rejects_reference_or_media_tampering(
    tmp_path: Path,
) -> None:
    request = selector_request(tmp_path)
    request["reference"] = "changed label"
    with pytest.raises(ValueError, match="request SHA-256"):
        validate_selector_request(request, tmp_path)

    request = selector_request(tmp_path)
    (tmp_path / "media" / "video-01.mkv").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="media SHA-256"):
        validate_selector_request(request, tmp_path)


def test_build_selector_messages_uses_one_native_video_and_indexed_transcript(
    tmp_path: Path,
) -> None:
    request = selector_request(tmp_path)

    messages = build_selector_messages(request, tmp_path)

    assert messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(tmp_path / "media" / "video-01.mkv"),
                    "fps": 2.0,
                },
                {
                    "type": "text",
                    "text": (
                        f"{SELECTOR_PROMPT}\n\n"
                        "Video duration: 20.000000 seconds\n"
                        "Question: What happens after the bell?\n"
                        "Reference answer (localization only): The door opens.\n"
                        "Indexed accepted Whisper transcript:\n"
                        "segment_id=0 | 0.000000-8.000000 | "
                        "[00:00:00.000] A bell rings.\n"
                        "segment_id=1 | 8.000000-20.000000 | "
                        "[00:00:08.000] The door opens."
                    ),
                },
            ],
        }
    ]


def test_extract_terminal_selector_json_accepts_thinking_then_terminal_object(
    tmp_path: Path,
) -> None:
    request = selector_request(tmp_path)
    raw = "internal analysis with {noise}\n```json\n" + valid_selector_json() + "\n```"

    parsed = extract_terminal_selector_json(raw, request)

    assert parsed["status"] == "localized"
    assert parsed["intervals"] == [{"start_seconds": 7.0, "end_seconds": 12.0}]
    assert parsed["whisper_segment_ids"] == [1]


def test_extract_terminal_selector_json_rejects_answer_or_trailing_prose(
    tmp_path: Path,
) -> None:
    request = selector_request(tmp_path)
    with pytest.raises(ValueError, match="terminal"):
        extract_terminal_selector_json(valid_selector_json() + " explanation", request)

    payload = json.loads(valid_selector_json())
    payload["answer"] = "The door opens"
    with pytest.raises(ValueError, match="exactly"):
        extract_terminal_selector_json(json.dumps(payload), request)


def test_run_selector_with_retry_uses_fixed_repair_seed_without_saving_raw(
    tmp_path: Path,
) -> None:
    request = selector_request(tmp_path)
    calls: list[tuple[int, str]] = []

    def generate(messages: list[dict[str, object]], seed: int) -> str:
        text = str(messages[0]["content"][1]["text"])
        calls.append((seed, text))
        return "not-json" if seed == 42 else valid_selector_json()

    record = run_selector_with_retry(request, tmp_path, generate)

    assert record["status"] == "passed"
    assert record["selector_attempt_count"] == 2
    assert record["repair_used"] is True
    assert [seed for seed, _ in calls] == [42, 43]
    assert REPAIR_PROMPT not in calls[0][1]
    assert REPAIR_PROMPT in calls[1][1]
    assert record["selector"]["whisper_segment_ids"] == [1]
    serialized = json.dumps(record)
    assert "not-json" not in serialized
    assert "internal analysis" not in serialized
    assert all("raw_output_sha256" in attempt for attempt in record["attempts"])


def test_run_selector_with_retry_does_not_retry_runtime_failure(tmp_path: Path) -> None:
    request = selector_request(tmp_path)
    calls = 0

    def generate(_messages: list[dict[str, object]], _seed: int) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("CUDA failure")

    record = run_selector_with_retry(request, tmp_path, generate)

    assert calls == 1
    assert record["status"] == "failed"
    assert record["failure_stage"] == "generation"
    assert record["selector_attempt_count"] == 1


def test_runtime_generator_keeps_native_video_audio_switch_consistent(
    tmp_path: Path,
) -> None:
    request = selector_request(tmp_path)
    messages = build_selector_messages(request, tmp_path)
    calls: dict[str, object] = {}

    class Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    class Tensor:
        def __init__(self, values: list[int], shape: tuple[int, ...]) -> None:
            self.values = values
            self.shape = shape

        def __getitem__(self, key: object) -> "Tensor | Scalar":
            assert isinstance(key, tuple)
            if key == (0, -1):
                return Scalar(self.values[-1])
            assert isinstance(key[1], slice)
            values = self.values[key[1]]
            return Tensor(values, (1, len(values)))

        def detach(self) -> "Tensor":
            return self

        def cpu(self) -> "Tensor":
            return self

        def contiguous(self) -> "Tensor":
            return self

        def numpy(self) -> "Tensor":
            return self

        def tobytes(self) -> bytes:
            return b"".join(struct.pack("q", value) for value in self.values)

    class Batch(dict[str, object]):
        def to(self, _value: object) -> "Batch":
            return self

    class Processor:
        class Tokenizer:
            eos_token_id = EXPECTED_EOS_TOKEN_ID
            pad_token_id = EXPECTED_PAD_TOKEN_ID

            @staticmethod
            def convert_ids_to_tokens(value: int) -> str:
                return {
                    EXPECTED_EOS_TOKEN_ID: "<|im_end|>",
                    EXPECTED_PAD_TOKEN_ID: "<|endoftext|>",
                }[value]

        tokenizer = Tokenizer()

        def apply_chat_template(self, value: object, **kwargs: object) -> str:
            calls["template"] = (value, kwargs)
            return "templated"

        def __call__(self, **kwargs: object) -> Batch:
            calls["processor"] = kwargs
            return Batch(
                input_ids=Tensor([1, 2, 3], (1, 3)),
                pixel_values=Tensor([1, 2], (2,)),
                video_grid_thw=Tensor([1, 2, 3], (1, 3)),
                input_features=Tensor([4, 5], (1, 2)),
                feature_attention_mask=Tensor([1, 1], (1, 2)),
            )

        def batch_decode(self, _tokens: object, **kwargs: object) -> list[str]:
            calls["decode"] = kwargs
            return [valid_selector_json()]

    class Parameter:
        device = "cuda:0"

    class Model:
        dtype = "bfloat16"

        def __init__(self) -> None:
            self.hooks: list[object] = []

        @staticmethod
        def parameters():
            yield Parameter()

        def register_forward_pre_hook(self, hook: object, *, with_kwargs: bool):
            assert with_kwargs is True
            self.hooks.append(hook)

            class Handle:
                @staticmethod
                def remove() -> None:
                    return None

            return Handle()

        def generate(self, **kwargs: object) -> Tensor:
            calls["generate"] = kwargs
            for hook in self.hooks:
                hook(self, (), kwargs)
            return Tensor([1, 2, 3, 777, EXPECTED_EOS_TOKEN_ID], (1, 5))

    class Cuda:
        @staticmethod
        def manual_seed_all(seed: int) -> None:
            calls["cuda_seed"] = seed

        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def max_memory_allocated() -> int:
            return 123

    class Torch:
        cuda = Cuda()

        @staticmethod
        def manual_seed(seed: int) -> None:
            calls["seed"] = seed

        @staticmethod
        def inference_mode():
            return contextlib.nullcontext()

    def process_mm_info(value: object, **kwargs: object):
        calls["mm"] = (value, kwargs)
        return ["audio"], None, ["video"]

    generate = _runtime_generator(Torch(), Processor(), Model(), process_mm_info)

    raw, metadata = generate(messages, 42)

    assert raw == valid_selector_json()
    assert calls["mm"][1]["use_audio_in_video"] is True
    assert calls["processor"]["use_audio_in_video"] is True
    assert calls["generate"]["use_audio_in_video"] is True
    assert calls["generate"]["eos_token_id"] == EXPECTED_EOS_TOKEN_ID
    assert calls["generate"]["pad_token_id"] == EXPECTED_PAD_TOKEN_ID
    assert calls["seed"] == calls["cuda_seed"] == 42
    assert metadata["video_fps"] == 2.0
    assert metadata["peak_memory_allocated_bytes"] == 123
    assert metadata["ended_with_eos"] is True
    assert metadata["modality_attestation"]["native_video_element_count"] == 1
    assert metadata["modality_attestation"]["native_audio_element_count"] == 1
    assert metadata["modality_attestation"]["same_first_forward"] is True
