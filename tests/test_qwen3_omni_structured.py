from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.qwen3_omni_structured import (
    CANONICAL_PANEL_IDS_SHA256,
    EXPECTED_REQUEST_FILENAMES,
    MODEL_ID,
    MODEL_REVISION,
    MIN_GPU_MEMORY_BYTES,
    PRIMARY_SEED,
    REPAIR_PROMPT,
    REPAIR_SEED,
    RUNTIME_IMAGE,
    SELECTOR_PROMPT,
    EXPECTED_EOS_TOKEN_ID,
    EXPECTED_PAD_TOKEN_ID,
    _runtime_generator,
    authenticate_runtime_preflight_marker,
    build_selector_messages,
    extract_terminal_selector_json,
    run_selector_with_retry,
    validate_generation_memory_headroom,
    validate_h100_gpu_identity,
    validate_protocol,
    validate_selector_request,
)
from trec_video_2026.vqa_experiments.silver_oracle import protocol_sha256


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


def runtime_protocol_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], list[dict[str, str]], Path]:
    unhashed: dict[str, object] = {
        "schema_version": 1,
        "run_id": "silver-test",
        "panel": {
            "count": 475,
            "ids_sha256": CANONICAL_PANEL_IDS_SHA256,
            "canary_count": 11,
            "stratified_canary_count": 10,
            "resource_sentinel_count": 1,
            "resource_sentinel_id_sha256": hashlib.sha256(
                b"resource-sentinel"
            ).hexdigest(),
            "chunk_counts": [119, 119, 119, 118],
        },
        "selector": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_image": RUNTIME_IMAGE,
            "primary_seed": PRIMARY_SEED,
            "repair_seed": REPAIR_SEED,
            "max_retries": 1,
            "prompt": SELECTOR_PROMPT,
            "repair_prompt": REPAIR_PROMPT,
            "max_intervals": 2,
            "max_whisper_segments": 3,
            "interval_duration_budget_formula": (
                "min(video_duration, max(12, min(30, 0.2 * video_duration)))"
            ),
        },
    }
    requests = [
        {"run_id": "silver-test", "video_id": f"video-{index:02d}"}
        for index in range(10)
    ]
    requests.append({"run_id": "silver-test", "video_id": "resource-sentinel"})
    request_path = tmp_path / "selector-canary.jsonl"
    request_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in requests),
        encoding="utf-8",
    )
    ids_sha256 = hashlib.sha256(
        ("\n".join(row["video_id"] for row in requests) + "\n").encode()
    ).hexdigest()
    request_identity = {
        "row_count": len(requests),
        "ids_sha256": ids_sha256,
        "file_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
    }
    unhashed["request_files"] = {
        name: dict(request_identity) for name in EXPECTED_REQUEST_FILENAMES
    }
    unhashed["request_files"][request_path.name] = request_identity
    protocol = {**unhashed, "protocol_sha256": protocol_sha256(unhashed)}
    return protocol, requests, request_path


def executable_runtime_preflight_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    protocol, _, request_path = runtime_protocol_fixture(tmp_path)
    template = selector_request(tmp_path)
    sentinel_id = "resource-sentinel"
    video_ids = [f"video-{index:02d}" for index in range(10)] + [sentinel_id]
    requests: list[dict[str, object]] = []
    for index, video_id in enumerate(video_ids):
        media_bytes = f"frozen-video-{index}".encode()
        media_path = tmp_path / "media" / f"{video_id}.mkv"
        media_path.write_bytes(media_bytes)
        request = copy.deepcopy(template)
        request.update(
            {
                "run_id": "silver-test",
                "video_id": video_id,
                "video_path": f"media/{video_id}.mkv",
                "media_sha256": hashlib.sha256(media_bytes).hexdigest(),
            }
        )
        request.pop("request_sha256")
        request["request_sha256"] = canonical_sha256(request)
        requests.append(request)
    request_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in requests),
        encoding="utf-8",
    )
    request_identity = {
        "row_count": 11,
        "ids_sha256": hashlib.sha256(
            ("\n".join(video_ids) + "\n").encode()
        ).hexdigest(),
        "file_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
    }
    request_files = protocol["request_files"]
    assert isinstance(request_files, dict)
    request_files[request_path.name] = request_identity
    unhashed = {
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    }
    protocol["protocol_sha256"] = protocol_sha256(unhashed)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return request_path, protocol_path, tmp_path / "runtime-preflight-marker.json"


def validate_runtime_protocol(
    protocol: dict[str, object], requests: list[dict[str, str]], path: Path
) -> str:
    return validate_protocol(
        protocol,
        requests,
        requests_filename=path.name,
        request_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("max_intervals", 3),
        ("max_whisper_segments", 4),
        ("interval_duration_budget_formula", "unbounded"),
    ],
)
def test_validate_protocol_binds_all_selector_constraints(
    tmp_path: Path, field: str, invalid: object
) -> None:
    protocol, requests, request_path = runtime_protocol_fixture(tmp_path)
    assert (
        validate_runtime_protocol(protocol, requests, request_path)
        == protocol["protocol_sha256"]
    )

    selector = protocol["selector"]
    assert isinstance(selector, dict)
    selector[field] = invalid
    unhashed = {
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    }
    protocol["protocol_sha256"] = protocol_sha256(unhashed)

    with pytest.raises(ValueError, match=field):
        validate_runtime_protocol(protocol, requests, request_path)


def test_validate_protocol_accepts_stratified_canary_plus_resource_sentinel(
    tmp_path: Path,
) -> None:
    protocol, requests, request_path = runtime_protocol_fixture(tmp_path)

    assert (
        validate_runtime_protocol(protocol, requests, request_path)
        == protocol["protocol_sha256"]
    )


def test_validate_protocol_rejects_stale_ten_row_canary(tmp_path: Path) -> None:
    protocol, requests, request_path = runtime_protocol_fixture(tmp_path)
    stale_requests = requests[:-1]
    stale_path = tmp_path / "selector-canary.jsonl"
    stale_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in stale_requests),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row count"):
        validate_runtime_protocol(protocol, stale_requests, stale_path)


def test_validate_protocol_rejects_wrong_request_filename(tmp_path: Path) -> None:
    protocol, requests, request_path = runtime_protocol_fixture(tmp_path)
    wrong_path = tmp_path / "selector-not-frozen.jsonl"
    wrong_path.write_bytes(request_path.read_bytes())

    with pytest.raises(ValueError, match="filename"):
        validate_runtime_protocol(protocol, requests, wrong_path)


def test_validate_protocol_rejects_wrong_ids_even_when_external_hashes_are_regenerated(
    tmp_path: Path,
) -> None:
    protocol, requests, request_path = runtime_protocol_fixture(tmp_path)
    substituted = [dict(row) for row in requests]
    substituted[0]["video_id"] = "substituted-video"
    request_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in substituted),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ID digest"):
        validate_runtime_protocol(protocol, substituted, request_path)


def test_validate_protocol_rejects_changed_request_bytes_after_remote_checksums_regenerated(
    tmp_path: Path,
) -> None:
    protocol, requests, request_path = runtime_protocol_fixture(tmp_path)
    request_path.write_bytes(request_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="file SHA-256"):
        validate_runtime_protocol(protocol, requests, request_path)


def test_validate_selector_request_binds_hash_media_and_segments(
    tmp_path: Path,
) -> None:
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
                        "Valid interval range: 0.000000 <= start_seconds < "
                        "end_seconds <= 20.000000.\n"
                        "Combined interval duration limit: 12.000000 seconds.\n"
                        "Whisper segment ID range: 0 through 1.\n"
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
    literal_shape = (
        '{"status":"localized","intervals":'
        '[{"start_seconds":0.0,"end_seconds":1.0}],'
        '"whisper_segment_ids":[]}'
    )
    assert literal_shape in SELECTOR_PROMPT
    assert literal_shape in REPAIR_PROMPT


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
    assert "Combined interval duration limit: 12.000000 seconds." in calls[0][1]
    assert "Combined interval duration limit: 12.000000 seconds." in calls[1][1]
    assert "Return no more than two intervals" in calls[1][1]
    assert "Return no more than three Whisper segment IDs" in calls[1][1]
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
                pixel_values_videos=Tensor([1, 2], (2,)),
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
        class Properties:
            total_memory = 80 * 1024**3

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            calls["cuda_seed"] = seed

        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def max_memory_allocated() -> int:
            return 123

        @staticmethod
        def max_memory_reserved() -> int:
            return 456

        @staticmethod
        def get_device_properties(_index: int) -> "Cuda.Properties":
            return Cuda.Properties()

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
    assert metadata["peak_memory_reserved_bytes"] == 456
    assert metadata["gpu_total_memory_bytes"] == 80 * 1024**3
    assert metadata["reserved_memory_margin_bytes"] == 80 * 1024**3 - 456
    assert metadata["ended_with_eos"] is True
    assert metadata["modality_attestation"]["native_video_element_count"] == 1
    assert metadata["modality_attestation"]["native_audio_element_count"] == 1
    assert metadata["modality_attestation"]["same_first_forward"] is True


def test_h100_runtime_memory_floor_is_bound_to_advisor_gate() -> None:
    assert MIN_GPU_MEMORY_BYTES == 78_700_000_000


@pytest.mark.parametrize(
    "gpu_name",
    ["H100", "NVIDIA H100", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"],
)
def test_h100_runtime_identity_accepts_only_h100_names(gpu_name: str) -> None:
    assert validate_h100_gpu_identity(gpu_name) == gpu_name


@pytest.mark.parametrize(
    "gpu_name",
    [
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA H200",
        "NVIDIA H100 + NVIDIA A100",
        "pretend-H100",
        "",
    ],
)
def test_h100_runtime_identity_rejects_non_h100_or_mixed_names(
    gpu_name: str,
) -> None:
    with pytest.raises(RuntimeError, match="H100"):
        validate_h100_gpu_identity(gpu_name)


def test_authenticated_runtime_preflight_marker_binds_media_and_execution_attempt(
    tmp_path: Path,
) -> None:
    requests_path, protocol_path, marker_path = executable_runtime_preflight_fixture(
        tmp_path
    )
    kwargs = {
        "requests_path": requests_path,
        "protocol_path": protocol_path,
        "input_root": tmp_path,
        "marker_path": marker_path,
        "run_id": "vqa-silver-oracle-election-20260809-b4-canary-b4",
        "execution_attempt_id": "2d1cb945-31a7-4ad1-8462-b0acbce30ee1",
        "remote_output_uri": (
            "hf://buckets/private/selector/canary-b4/"
            "2d1cb945-31a7-4ad1-8462-b0acbce30ee1"
        ),
        "source_bundle_sha256": "b" * 64,
        "submitted_code_revision": "c" * 40,
    }

    marker = authenticate_runtime_preflight_marker(**kwargs, create=True)
    assert marker["status"] == "passed"
    assert marker["execution_attempt_id"] == kwargs["execution_attempt_id"]
    assert marker["marker_sha256"] == canonical_sha256(
        {key: value for key, value in marker.items() if key != "marker_sha256"}
    )
    assert authenticate_runtime_preflight_marker(**kwargs, create=False) == marker

    (tmp_path / "media" / "video-00.mkv").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="media SHA-256"):
        authenticate_runtime_preflight_marker(**kwargs, create=False)


def test_runtime_preflight_verification_requires_existing_exact_marker(
    tmp_path: Path,
) -> None:
    requests_path, protocol_path, marker_path = executable_runtime_preflight_fixture(
        tmp_path
    )

    with pytest.raises(FileNotFoundError, match="preflight marker"):
        authenticate_runtime_preflight_marker(
            requests_path=requests_path,
            protocol_path=protocol_path,
            input_root=tmp_path,
            marker_path=marker_path,
            run_id="vqa-silver-oracle-election-20260809-b4-canary-b4",
            execution_attempt_id="2d1cb945-31a7-4ad1-8462-b0acbce30ee1",
            remote_output_uri=(
                "hf://buckets/private/selector/canary-b4/"
                "2d1cb945-31a7-4ad1-8462-b0acbce30ee1"
            ),
            source_bundle_sha256="b" * 64,
            submitted_code_revision="c" * 40,
            create=False,
        )


def test_generation_memory_headroom_enforces_h100_safety_gates() -> None:
    total = 80 * 1024**3
    result = validate_generation_memory_headroom(
        total_memory=total,
        peak_allocated=70_826_560_000,
        peak_reserved=72_000_000_000,
    )
    assert result["reserved_memory_margin_bytes"] == total - 72_000_000_000

    with pytest.raises(RuntimeError, match="90%"):
        validate_generation_memory_headroom(
            total_memory=total,
            peak_allocated=int(total * 0.90) + 1,
            peak_reserved=72_000_000_000,
        )
    with pytest.raises(RuntimeError, match="6 GiB"):
        validate_generation_memory_headroom(
            total_memory=total,
            peak_allocated=70_000_000_000,
            peak_reserved=total - 6 * 1024**3 + 1,
        )
