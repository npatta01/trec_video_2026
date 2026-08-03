from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments import videochat3_pilot as videochat3_module
from trec_video_2026.vqa_experiments.videochat3_pilot import (
    GATE_ROW_IDS,
    PILOT_ROW_IDS,
    build_exact_video_metadata,
    build_videochat3_messages,
    generate_videochat3_answer,
    install_videochat3_generation_compatibility,
    load_frozen_requests,
    prepare_videochat3_inputs,
    select_gate_requests,
    select_label_free_requests,
    validate_cuda_bfloat16,
    verify_input_manifests,
)


TARGET_IDS = (
    "CH8Dp7EgW4I",
    "EtNOkUJd5w8",
    "Gp4LVTn3YbY",
    "KgOvCPOX-Ms",
)
CONTROL_IDS = ("EMI_anpHII4", "LusRurRrmf0")


class _MetadataBase:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _TransformersMetadataBase:
    def __init__(
        self,
        total_num_frames: int,
        fps: float,
        width: int,
        height: int,
        duration: float,
        video_backend: str,
        frames_indices: list[int],
    ) -> None:
        self.total_num_frames = total_num_frames
        self.fps = fps
        self.width = width
        self.height = height
        self.duration = duration
        self.video_backend = video_backend
        self.frames_indices = frames_indices


class _VideoChat3MetadataBase:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)

    @property
    def timestamps(self) -> list[float]:
        return [index / self.fps for index in self.frames_indices]


class _Shape:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)


class _Batch(dict):
    def to(self, value: object) -> "_Batch":
        self.setdefault("to_calls", []).append(value)
        return self


class _VideoKwargs:
    __annotations__ = {
        "do_convert_rgb": bool | None,
        "do_resize": bool | None,
        "size": dict[str, int] | None,
    }


class _Size:
    def __init__(self) -> None:
        self.shortest_edge = 4096
        self.longest_edge = 25165824
        self.height = None


class _VideoProcessor:
    valid_kwargs = _VideoKwargs
    do_convert_rgb = True
    do_resize = True
    size = _Size()


class _Processor:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.tokenizer = self
        self.video_processor = _VideoProcessor()

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        self.messages = messages
        self.template_kwargs = kwargs
        return "rendered-prompt"

    def __call__(self, *args: object, **kwargs: object) -> _Batch:
        self.calls.append((args, kwargs))
        return _Batch(input_ids=_Shape(3), pixel_values_videos="pixels")

    def batch_decode(self, values: object, **kwargs: object) -> list[str]:
        self.decoded = (values, kwargs)
        return ["  final answer: a short answer  "]


class _GeneratedRow:
    def __getitem__(self, item: slice) -> list[int]:
        assert item == slice(3, None)
        return [9, 10]


class _Model:
    device = "cuda:0"
    dtype = "bf16"

    def generate(self, **kwargs: object) -> list[_GeneratedRow]:
        self.generation_kwargs = kwargs
        return [_GeneratedRow()]


class _GenerationBase:
    def prepare_inputs_for_generation(
        self, input_ids: object, **kwargs: object
    ) -> dict[str, object]:
        return {
            "input_ids": input_ids,
            "pixel_values_videos": kwargs.get("pixel_values_videos"),
        }


class _RemoteGenerationModel(_GenerationBase):
    pass


class _Weight:
    def __init__(self, *, shape: tuple[int, ...], dtype: str, device: str) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device


class _EmbeddingModule:
    def __init__(self, weight: _Weight) -> None:
        self.weight = weight


class _UntiedLanguageModel:
    def __init__(self) -> None:
        self.input_embeddings = _EmbeddingModule(
            _Weight(shape=(151936, 2560), dtype="bf16", device="cuda:0")
        )
        self.output_embeddings = _EmbeddingModule(
            _Weight(shape=(151936, 2560), dtype="bf16", device="cuda:0")
        )

    def get_input_embeddings(self) -> _EmbeddingModule:
        return self.input_embeddings

    def get_output_embeddings(self) -> _EmbeddingModule:
        return self.output_embeddings


def _request(video_id: str, *, frame_count: int = 12) -> dict[str, object]:
    return {
        "run_id": "old-run",
        "condition_id": "old-condition",
        "video_id": video_id,
        "question": "What happens?",
        "frame_paths": [
            f"/input/{video_id}/frame-{index:02d}.png"
            for index in range(1, frame_count + 1)
        ],
        "frame_timestamps": [0.25 + index * 0.5 for index in range(frame_count)],
        "frame_descriptions": [f"Frame {index}" for index in range(frame_count)],
        "transcript": None,
    }


def test_exact_video_metadata_preserves_indices_and_original_timestamps() -> None:
    timestamps = (0.672833, 2.0185, 3.364167, 4.709833)

    metadata = build_exact_video_metadata(
        _MetadataBase,
        timestamps=timestamps,
        width=448,
        height=336,
    )

    assert metadata.frames_indices == [0, 1, 2, 3]
    assert metadata.timestamps == list(timestamps)
    assert metadata.total_num_frames == 4
    assert metadata.video_backend == "frozen-frame-tensor"
    assert "_exact_timestamps" not in metadata.__dict__


def test_loader_resolves_both_metadata_types_used_by_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AcceptedVideoMetadata:
        pass

    class ProcessorVideoMetadata:
        pass

    class FakeProcessor:
        video_processor = type(
            "VideoProcessor",
            (),
            {"__module__": "fake_videochat3.processing"},
        )()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeProcessor:
            return FakeProcessor()

    class FakeWeight:
        shape = (151936, 2560)
        dtype = "bf16"
        device = "cuda:0"

    class FakeEmbeddings:
        def __init__(self) -> None:
            self.weight = FakeWeight()

    class FakeModel:
        def __init__(self) -> None:
            self.input_embeddings = FakeEmbeddings()
            self.output_embeddings = FakeEmbeddings()

        def get_input_embeddings(self) -> FakeEmbeddings:
            return self.input_embeddings

        def get_output_embeddings(self) -> FakeEmbeddings:
            return self.output_embeddings

        def eval(self) -> "FakeModel":
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeModel:
            return FakeModel()

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = FakeAutoProcessor
    transformers.AutoModelForCausalLM = FakeAutoModel
    video_utils = types.ModuleType("transformers.video_utils")
    video_utils.VideoMetadata = AcceptedVideoMetadata
    remote_utils = types.ModuleType("fake_videochat3.videochat3_utils")
    remote_utils.VideoChat3VideoMetadata = ProcessorVideoMetadata
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.video_utils", video_utils)
    monkeypatch.setitem(
        sys.modules,
        "fake_videochat3.videochat3_utils",
        remote_utils,
    )

    (
        _,
        model,
        metadata_base,
        processor_metadata_base,
        output_head_tie,
    ) = videochat3_module._load_videochat3(
        Path("/model"), type("Torch", (), {"bfloat16": "bf16"})()
    )

    assert metadata_base is AcceptedVideoMetadata
    assert processor_metadata_base is ProcessorVideoMetadata
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    assert output_head_tie["was_already_tied"] is False
    assert output_head_tie["is_tied"] is True


def test_exact_metadata_uses_transformers_constructor_contract() -> None:
    timestamps = (0.672833, 2.0185, 3.364167, 4.709833)

    metadata = build_exact_video_metadata(
        _TransformersMetadataBase,
        timestamps=timestamps,
        width=448,
        height=336,
    )

    assert metadata.frames_indices == [0, 1, 2, 3]
    assert metadata.timestamps == list(timestamps)


def test_exact_metadata_survives_videochat3_processor_type_conversion() -> None:
    timestamps = (0.672833, 2.0185, 3.364167, 4.709833)

    metadata = build_exact_video_metadata(
        _TransformersMetadataBase,
        processor_metadata_base=_VideoChat3MetadataBase,
        timestamps=timestamps,
        width=448,
        height=336,
    )

    assert isinstance(metadata, _TransformersMetadataBase)
    assert isinstance(metadata, _VideoChat3MetadataBase)
    processor_metadata = (
        metadata
        if isinstance(metadata, _VideoChat3MetadataBase)
        else _VideoChat3MetadataBase(**metadata.__dict__)
    )
    assert processor_metadata is metadata
    assert processor_metadata.timestamps == [
        0.672833,
        2.0185,
        3.364167,
        4.709833,
    ]


def test_input_flow_attestation_checks_post_processor_timestamps() -> None:
    timestamps = (0.125, 0.625, 1.375, 2.25, 3.5, 5.0, 6.75, 8.75)
    metadata = build_exact_video_metadata(
        _TransformersMetadataBase,
        processor_metadata_base=_VideoChat3MetadataBase,
        timestamps=timestamps,
        width=32,
        height=32,
    )

    class Processor:
        video_processor = types.SimpleNamespace(temporal_merge_size=4)

        @staticmethod
        def _calculate_timestamps(value: object, merge_size: int) -> list[float]:
            assert merge_size == 4
            values = value.timestamps
            return [
                (values[index] + values[min(index + 3, len(values) - 1)]) / 2
                for index in range(0, len(values), merge_size)
            ]

    attestation = videochat3_module.attest_videochat3_input_flow(
        processor=Processor(),
        inputs={"video_metadata": [metadata]},
        source_metadata=metadata,
        expected_timestamps=timestamps,
        expected_prompt_timestamps=(1.1875, 6.125),
    )

    assert attestation == {
        "processor_preserved_metadata_identity": True,
        "processor_preserved_exact_timestamps": True,
        "processor_metadata_timestamps": [
            0.125,
            0.625,
            1.375,
            2.25,
            3.5,
            5.0,
            6.75,
            8.75,
        ],
        "processor_prompt_timestamps": [1.1875, 6.125],
    }


def test_input_flow_attestation_accepts_batchfeature_metadata_copy() -> None:
    """Transformers may copy Mapping metadata while tensorizing a batch."""

    timestamps = (0.125, 0.625, 1.375, 2.25, 3.5, 5.0, 6.75, 8.75)
    metadata = build_exact_video_metadata(
        _TransformersMetadataBase,
        processor_metadata_base=_VideoChat3MetadataBase,
        timestamps=timestamps,
        width=32,
        height=32,
    )
    processed_metadata = copy.copy(metadata)
    assert processed_metadata is not metadata

    class Processor:
        video_processor = types.SimpleNamespace(temporal_merge_size=4)

        @staticmethod
        def _calculate_timestamps(value: object, merge_size: int) -> list[float]:
            assert merge_size == 4
            values = value.timestamps
            return [
                (values[index] + values[min(index + 3, len(values) - 1)]) / 2
                for index in range(0, len(values), merge_size)
            ]

    attestation = videochat3_module.attest_videochat3_input_flow(
        processor=Processor(),
        inputs={"video_metadata": [processed_metadata]},
        source_metadata=metadata,
        expected_timestamps=timestamps,
        expected_prompt_timestamps=(1.1875, 6.125),
    )

    assert attestation["processor_preserved_metadata_identity"] is False
    assert attestation["processor_preserved_exact_timestamps"] is True


def test_input_flow_canary_uses_cuda_processor_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = videochat3_module.INPUT_FLOW_TIMESTAMPS
    metadata = build_exact_video_metadata(
        _TransformersMetadataBase,
        processor_metadata_base=_VideoChat3MetadataBase,
        timestamps=timestamps,
        width=32,
        height=32,
    )

    class Processor:
        video_processor = types.SimpleNamespace(temporal_merge_size=4)
        tokenizer = types.SimpleNamespace(
            decode=lambda *args, **kwargs: (
                "<1.2 seconds><6.1 seconds><15.1 seconds>"
            )
        )

        @staticmethod
        def _calculate_timestamps(value: object, merge_size: int) -> list[float]:
            values = value.timestamps
            return [
                (values[index] + values[min(index + 3, len(values) - 1)]) / 2
                for index in range(0, len(values), merge_size)
            ]

    class PixelValues:
        device = "cuda:0"
        dtype = "bf16"

    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip=None, cuda="12.8")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        get_device_name=lambda index: "NVIDIA A40",
    )
    torch.uint8 = "uint8"
    torch.bfloat16 = "bf16"
    torch.zeros = lambda *args, **kwargs: "synthetic-tensor"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        videochat3_module,
        "_load_videochat3_processor",
        lambda model_dir: (
            Processor(),
            _TransformersMetadataBase,
            _VideoChat3MetadataBase,
        ),
    )
    prepare_kwargs: dict[str, object] = {}

    def fake_prepare(**kwargs: object) -> tuple[dict[str, object], object]:
        prepare_kwargs.update(kwargs)
        return (
            {
                "video_metadata": [metadata],
                "pixel_values_videos": PixelValues(),
                "input_ids": [[1, 2, 3]],
            },
            metadata,
        )

    monkeypatch.setattr(videochat3_module, "prepare_videochat3_inputs", fake_prepare)
    args = argparse.Namespace(
        videochat_model_dir=Path("/model"),
        videochat_model_id="MCG-NJU/VideoChat3-4B",
        videochat_revision="revision",
        runtime_image="image@sha256:digest",
    )

    record = videochat3_module.run_input_flow_canary(args)

    assert prepare_kwargs["processor_metadata_base"] is _VideoChat3MetadataBase
    assert prepare_kwargs["model_device"] == "cuda:0"
    assert prepare_kwargs["model_dtype"] == "bf16"
    assert record["mode"] == "synthetic-input-flow-canary"
    assert record["model_loaded"] is False
    assert record["private_media_loaded"] is False
    assert record["processor_prompt_timestamps"] == [1.1875, 6.125, 15.125]
    assert record["prompt_time_labels"] == [
        "<1.2 seconds>",
        "<6.1 seconds>",
        "<15.1 seconds>",
    ]
    assert record["runtime_attestation"]["cuda_device_name"] == "NVIDIA A40"


def test_input_flow_cli_writes_frozen_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "input-flow.json"
    monkeypatch.setattr(
        videochat3_module,
        "run_input_flow_canary",
        lambda args: {
            "mode": "synthetic-input-flow-canary",
            "model_loaded": False,
            "private_media_loaded": False,
        },
    )

    returncode = videochat3_module.main(
        [
            "--mode",
            "input-flow-canary",
            "--output",
            str(output),
            "--videochat-model-dir",
            "/model",
            "--videochat-model-id",
            "MCG-NJU/VideoChat3-4B",
            "--videochat-revision",
            "revision",
            "--runtime-image",
            "image@sha256:digest",
        ]
    )

    assert returncode == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "mode": "synthetic-input-flow-canary",
        "model_loaded": False,
        "private_media_loaded": False,
    }
    assert output.stat().st_mode & 0o777 == 0o444


def test_videochat3_messages_use_one_video_and_the_controlled_short_prompt() -> None:
    messages = build_videochat3_messages("  What happens next?  ")

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {
                    "type": "text",
                    "text": (
                        "Question: What happens next?\n"
                        "Answer with one short phrase or sentence. Return only the answer; "
                        "do not explain your reasoning."
                    ),
                },
            ],
        }
    ]


def test_prepare_videochat3_inputs_passes_one_tensor_and_exact_metadata() -> None:
    processor = _Processor()
    video_tensor = object()
    timestamps = tuple(0.25 + index * 0.5 for index in range(12))

    inputs, metadata = prepare_videochat3_inputs(
        processor=processor,
        metadata_base=_MetadataBase,
        processor_metadata_base=_VideoChat3MetadataBase,
        question="What happens?",
        video_tensor=video_tensor,
        timestamps=timestamps,
        width=448,
        height=336,
        model_device="cuda:0",
        model_dtype="bf16",
    )

    assert processor.template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    _, kwargs = processor.calls[0]
    assert kwargs["text"] == "rendered-prompt"
    assert kwargs["videos"] == [video_tensor]
    assert kwargs["video_metadata"] == [metadata]
    assert kwargs["do_sample_frames"] is False
    assert kwargs["do_convert_rgb"] is True
    assert kwargs["do_resize"] is True
    assert kwargs["size"] == {
        "shortest_edge": 4096,
        "longest_edge": 25165824,
    }
    assert kwargs["return_metadata"] is True
    assert kwargs["return_tensors"] == "pt"
    assert metadata.frames_indices == list(range(12))
    assert metadata.timestamps == list(timestamps)
    assert isinstance(metadata, _VideoChat3MetadataBase)
    assert inputs["to_calls"] == ["cuda:0", "bf16"]


def test_generate_videochat3_answer_is_greedy_and_capped_at_32_tokens() -> None:
    processor = _Processor()
    model = _Model()
    inputs = _Batch(
        input_ids=_Shape(3),
        pixel_values_videos="pixels",
        video_metadata="processor-only-metadata",
    )

    answer = generate_videochat3_answer(processor, model, inputs)

    assert answer == "a short answer"
    assert model.generation_kwargs["do_sample"] is False
    assert model.generation_kwargs["max_new_tokens"] == 32
    assert model.generation_kwargs["pixel_values_videos"] == "pixels"
    assert model.generation_kwargs["inputs_embeds"] is None
    assert "video_metadata" not in model.generation_kwargs


def test_gate_requires_three_nonempty_single_line_short_answers() -> None:
    passing = [
        {"status": "passed", "answer": "one"},
        {"status": "passed", "answer": "two words"},
        {"status": "passed", "answer": "a short sentence."},
    ]

    assert videochat3_module.gate_answers_are_valid(passing) is True
    assert videochat3_module.gate_answers_are_valid(passing[:2]) is False
    assert videochat3_module.gate_answers_are_valid(
        [*passing[:2], {"status": "passed", "answer": "line one\nline two"}]
    ) is False
    assert videochat3_module.gate_answers_are_valid(
        [*passing[:2], {"status": "failed", "answer": "answer"}]
    ) is False


def test_output_head_is_explicitly_tied_to_trained_input_embeddings() -> None:
    model = _UntiedLanguageModel()
    trained_embeddings = model.get_input_embeddings().weight

    attestation = videochat3_module.tie_videochat3_output_embeddings(model)

    assert model.get_output_embeddings().weight is trained_embeddings
    assert attestation == {
        "was_already_tied": False,
        "is_tied": True,
        "shape": [151936, 2560],
        "dtype": "bf16",
        "device": "cuda:0",
    }


def test_output_head_tie_fails_closed_on_incompatible_shapes() -> None:
    model = _UntiedLanguageModel()
    model.output_embeddings.weight = _Weight(
        shape=(151935, 2560), dtype="bf16", device="cuda:0"
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        videochat3_module.tie_videochat3_output_embeddings(model)


def test_generation_compatibility_supplies_missing_inputs_embeds_key() -> None:
    model = _RemoteGenerationModel()
    vision_block = types.SimpleNamespace(attn_impl="flash_attention_2")
    model.model = types.SimpleNamespace(
        vision_tower=types.SimpleNamespace(
            encoder=types.SimpleNamespace(blocks=[vision_block])
        )
    )

    install_videochat3_generation_compatibility(model)
    prepared = model.prepare_inputs_for_generation(
        _Shape(3),
        cache_position=[0],
        pixel_values_videos="pixels",
    )

    assert prepared["inputs_embeds"] is None
    assert prepared["pixel_values_videos"] == "pixels"
    assert vision_block.attn_impl == "sdpa"

    decoding = model.prepare_inputs_for_generation(
        _Shape(3),
        cache_position=[1],
        pixel_values_videos="pixels",
    )

    assert decoding["pixel_values_videos"] is None

    cached_decoding = model.prepare_inputs_for_generation(
        _Shape(3),
        past_key_values=object(),
        cache_position=[0],
        pixel_values_videos="pixels",
    )

    assert cached_decoding["pixel_values_videos"] is None


def test_select_gate_requests_projects_declared_three_without_predictions(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts.jsonl"
    records = []
    for video_id in reversed(GATE_ROW_IDS):
        request = _request(video_id)
        request["transcript"] = None
        records.append({"answer": "old prediction", "request": request})
    attempts.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    selected = select_gate_requests(
        attempts,
        run_id="videochat3-pilot-20260730-a1",
        condition_id="videochat3-compatibility-gate",
    )

    assert tuple(row["video_id"] for row in selected) == GATE_ROW_IDS
    assert all("answer" not in json.dumps(row).lower() for row in selected)
    assert all(
        row["frame_paths"][0]
        == f"/dstack/input/evidence/{row['video_id']}/frame-01.png"
        for row in selected
    )


def test_select_label_free_requests_freezes_exact_six_rows(tmp_path: Path) -> None:
    source = tmp_path / "requests.json"
    source.write_text(
        json.dumps(
            [_request(video_id) for video_id in ("unused", *reversed(PILOT_ROW_IDS))]
        ),
        encoding="utf-8",
    )

    selected = select_label_free_requests(
        source,
        run_id="videochat3-pilot-20260730-a1",
        condition_id="videochat3-4b-12f",
    )

    assert tuple(row["video_id"] for row in selected) == PILOT_ROW_IDS
    assert all(row["run_id"] == "videochat3-pilot-20260730-a1" for row in selected)
    assert all(row["condition_id"] == "videochat3-4b-12f" for row in selected)
    assert all(len(row["frame_paths"]) == 12 for row in selected)
    assert all(len(row["frame_timestamps"]) == 12 for row in selected)
    assert all("answer" not in json.dumps(row).lower() for row in selected)


def test_select_label_free_requests_accepts_exactly_24_frozen_frames(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requests.json"
    source.write_text(
        json.dumps([_request(video_id, frame_count=24) for video_id in PILOT_ROW_IDS]),
        encoding="utf-8",
    )

    selected = select_label_free_requests(
        source,
        run_id="videochat3-24frame-pilot-20260801-a1",
        condition_id="videochat3-4b-24f",
        frame_count=24,
    )

    assert len(selected) == 6
    assert all(len(row["frame_paths"]) == 24 for row in selected)
    assert all(len(row["frame_timestamps"]) == 24 for row in selected)
    assert all(row["condition_id"] == "videochat3-4b-24f" for row in selected)


def test_validate_cached_request_checks_all_24_frame_hashes(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "24-frame-cache"
    evidence_dir.mkdir()
    frames = []
    timestamps = []
    paths = []
    for index in range(1, 25):
        frame_name = f"frame-{index:02d}.png"
        payload = f"frame payload {index}\n".encode()
        path = evidence_dir / frame_name
        path.write_bytes(payload)
        paths.append(str(path))
        timestamps.append(index / 4)
        frames.append(
            {
                "path": frame_name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (evidence_dir / "evidence.json").write_text(
        json.dumps(
            {
                "cache_key": "24-frame-cache",
                "parameters": {"frame_count": 24},
                "timestamps_seconds": timestamps,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    request = {
        **_request(PILOT_ROW_IDS[0], frame_count=24),
        "frame_paths": paths,
        "frame_timestamps": timestamps,
        "evidence_cache_key": "24-frame-cache",
    }

    verified = videochat3_module.validate_cached_request(request, frame_count=24)

    assert verified["cache_key"] == "24-frame-cache"
    assert verified["frame_indices"] == list(range(24))
    assert verified["frame_names"] == [
        f"frame-{index:02d}.png" for index in range(1, 25)
    ]
    assert len(verified["frame_sha256s"]) == 24

    Path(paths[-1]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="frame checksum mismatch"):
        videochat3_module.validate_cached_request(request, frame_count=24)


def test_validate_cached_request_binds_declared_24_frame_cache_key(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "declared-cache-key"
    evidence_dir.mkdir()
    timestamps = [index / 4 for index in range(1, 25)]
    paths = []
    frames = []
    for index in range(1, 25):
        frame_name = f"frame-{index:02d}.png"
        payload = f"frame payload {index}\n".encode()
        path = evidence_dir / frame_name
        path.write_bytes(payload)
        paths.append(str(path))
        frames.append(
            {
                "path": frame_name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    evidence_path = evidence_dir / "evidence.json"
    evidence = {
        "cache_key": "declared-cache-key",
        "parameters": {"frame_count": 24},
        "timestamps_seconds": timestamps,
        "frames": frames,
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    request = {
        **_request(PILOT_ROW_IDS[0], frame_count=24),
        "frame_paths": paths,
        "frame_timestamps": timestamps,
        "evidence_cache_key": "declared-cache-key",
    }

    request["evidence_cache_key"] = "different-request-key"
    with pytest.raises(ValueError, match="evidence cache-key binding mismatch"):
        videochat3_module.validate_cached_request(request, frame_count=24)

    request["evidence_cache_key"] = "declared-cache-key"
    evidence["cache_key"] = "different-evidence-key"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence cache-key binding mismatch"):
        videochat3_module.validate_cached_request(request, frame_count=24)


def test_prepare_24_frame_manifest_binds_exact_cache_without_labels(
    tmp_path: Path,
) -> None:
    from trec_video_2026.vqa_benchmark import (
        EvidenceParameters,
        cache_key,
        source_fingerprint,
    )

    dataset_root = tmp_path / "data"
    media_root = dataset_root / "vqa/2025/training/videos_original"
    evidence_root = tmp_path / "evidence"
    media_root.mkdir(parents=True)
    evidence_root.mkdir()
    source_requests = tmp_path / "source-requests.json"
    source_requests.write_text(
        json.dumps([_request(video_id) for video_id in PILOT_ROW_IDS]),
        encoding="utf-8",
    )
    expected_keys = []
    for row_index, video_id in enumerate(PILOT_ROW_IDS):
        media = media_root / f"{video_id}.mkv"
        media.write_bytes(f"media {row_index}\n".encode())
        fingerprint = source_fingerprint(media)
        parameters = EvidenceParameters(frame_count=24)
        expected_key = cache_key(fingerprint, parameters)
        expected_keys.append(expected_key)
        cache_dir = evidence_root / expected_key
        cache_dir.mkdir()
        frames = []
        for frame_index in range(1, 25):
            frame_name = f"frame-{frame_index:02d}.png"
            payload = f"{video_id} frame {frame_index}\n".encode()
            (cache_dir / frame_name).write_bytes(payload)
            frames.append(
                {
                    "path": frame_name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        (cache_dir / "evidence.json").write_text(
            json.dumps(
                {
                    "cache_key": expected_key,
                    "source": {
                        "sha256": fingerprint.sha256,
                        "size_bytes": fingerprint.size_bytes,
                    },
                    "parameters": {
                        "frame_count": 24,
                        "extractor_version": "ffmpeg-bin-midpoint-v1",
                    },
                    "duration_seconds": 24.0,
                    "timestamps_seconds": [index + 0.5 for index in range(24)],
                    "frames": frames,
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "pilot-24.json"

    result = videochat3_module.prepare_24_frame_manifest(
        source_requests=source_requests,
        dataset_root=dataset_root,
        evidence_root=evidence_root,
        output=output,
        run_id="videochat3-24frame-pilot-20260801-a1",
        remote_evidence_root=Path("/dstack/input/evidence"),
    )

    rows = json.loads(output.read_text(encoding="utf-8"))
    assert result["manifest_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert result["evidence_cache_keys"] == expected_keys
    assert [row["video_id"] for row in rows] == list(PILOT_ROW_IDS)
    assert all(row["condition_id"] == "videochat3-4b-24f" for row in rows)
    assert all(len(row["frame_paths"]) == 24 for row in rows)
    assert all(len(row["frame_timestamps"]) == 24 for row in rows)
    assert all(row["frame_descriptions"] is None for row in rows)
    assert rows[0]["frame_paths"] == [
        f"/dstack/input/evidence/{expected_keys[0]}/frame-{index:02d}.png"
        for index in range(1, 25)
    ]
    serialized = json.dumps(rows).lower()
    assert all(key not in serialized for key in ("answer", "annotation", "reference", "label"))
    assert output.stat().st_mode & 0o777 == 0o444

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        videochat3_module.prepare_24_frame_manifest(
            source_requests=source_requests,
            dataset_root=dataset_root,
            evidence_root=evidence_root,
            output=output,
            run_id="videochat3-24frame-pilot-20260801-a1",
            remote_evidence_root=Path("/dstack/input/evidence"),
        )


def test_load_frozen_requests_returns_verified_full_manifest_hash(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "gate.json"
    manifest.write_bytes(b'[{"video_id":"BDKkVYmWqdM","question":"one"}]\n')

    rows, verified_sha256 = load_frozen_requests(
        manifest,
        expected_sha256=(
            "5c0af485cd6a15c7381e2ac43beda6628bce0e1e58b2c9cb3885b69fa3b1e507"
        ),
        expected_row_ids=("BDKkVYmWqdM",),
        manifest_name="gate",
    )

    assert rows == [{"video_id": "BDKkVYmWqdM", "question": "one"}]
    assert (
        verified_sha256
        == "5c0af485cd6a15c7381e2ac43beda6628bce0e1e58b2c9cb3885b69fa3b1e507"
    )


def test_load_frozen_requests_rejects_changed_content_with_same_row_identity(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "gate.json"
    manifest.write_bytes(b'[{"video_id":"BDKkVYmWqdM","question":"changed"}]\n')

    with pytest.raises(ValueError, match="gate manifest SHA-256 mismatch"):
        load_frozen_requests(
            manifest,
            expected_sha256=(
                "5c0af485cd6a15c7381e2ac43beda6628bce0e1e58b2c9cb3885b69fa3b1e507"
            ),
            expected_row_ids=("BDKkVYmWqdM",),
            manifest_name="gate",
        )


def test_runtime_rejects_manifest_hash_pin_other_than_frozen_declaration(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "gate.json"
    gate.write_bytes(
        b'[{"video_id":"BDKkVYmWqdM","nonce":"frozen"},'
        b'{"video_id":"O60MtMuSpf8","nonce":"frozen"},'
        b'{"video_id":"Fk_j6re_Uiw","nonce":"frozen"}]\n'
    )
    args = argparse.Namespace(
        gate_requests=gate,
        gate_requests_sha256=(
            "714ee55e7e671a18c16ff0a58f6266c38f89055ae6c62917c879d6ca17f82323"
        ),
        pilot_requests=tmp_path / "pilot.json",
        pilot_requests_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="gate manifest SHA-256 pin mismatch"):
        verify_input_manifests(args)


def test_single_gate_manifest_is_verified_without_requiring_pilot_input(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "gate.json"
    gate.write_bytes(
        b'[{"video_id":"BDKkVYmWqdM"},'
        b'{"video_id":"O60MtMuSpf8"},'
        b'{"video_id":"Fk_j6re_Uiw"}]\n'
    )
    expected_sha256 = (
        "9c0309af595c09d55fd5dcf5e9cd989c9f2196a16c8957de32eb5eb152e18f3f"
    )
    args = argparse.Namespace(
        gate_requests=gate,
        gate_requests_sha256=expected_sha256,
    )

    provenance = videochat3_module.verify_frozen_manifest(
        args,
        manifest_name="gate",
        declared_sha256=expected_sha256,
    )

    assert provenance == {
        "schema_version": "videochat3-input-manifest-v1",
        "verified": True,
        "manifest_name": "gate",
        "input_manifest_sha256": expected_sha256,
        "row_ids": list(GATE_ROW_IDS),
    }


def test_single_manifest_preflight_verifies_each_cached_evidence_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = [_request(video_id) for video_id in GATE_ROW_IDS]
    monkeypatch.setattr(
        videochat3_module,
        "verify_frozen_manifest",
        lambda *args, **kwargs: {
            "schema_version": "videochat3-input-manifest-v1",
            "verified": True,
            "manifest_name": "gate",
            "input_manifest_sha256": "a" * 64,
            "row_ids": list(GATE_ROW_IDS),
        },
    )
    monkeypatch.setattr(
        videochat3_module,
        "load_frozen_requests",
        lambda *args, **kwargs: (requests, "a" * 64),
    )
    monkeypatch.setattr(
        videochat3_module,
        "validate_cached_request",
        lambda request, **kwargs: {
            "video_id": request["video_id"],
            "evidence_sha256": request["video_id"].encode().hex().ljust(64, "0")[:64],
        },
    )
    args = argparse.Namespace(
        gate_requests=Path("gate.json"),
        gate_requests_sha256="a" * 64,
    )

    provenance = videochat3_module.verify_frozen_evidence(
        args,
        manifest_name="gate",
        declared_sha256="a" * 64,
    )

    assert provenance["evidence_verified"] is True
    assert [row["video_id"] for row in provenance["evidence"]] == list(GATE_ROW_IDS)


def test_prepare_local_inputs_uses_fresh_gate_and_pilot_run_identities(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(
        "\n".join(
            json.dumps({"request": _request(video_id)})
            for video_id in GATE_ROW_IDS
        )
        + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "pilot-source.json"
    source.write_text(
        json.dumps([_request(video_id) for video_id in PILOT_ROW_IDS]),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        prepare_output_dir=tmp_path / "prepared",
        source_gate_attempts=attempts,
        source_pilot_requests=source,
        gate_run_id="videochat3-gate-20260801-a1",
        pilot_run_id="videochat3-pilot-20260801-a1",
    )

    videochat3_module.prepare_local_inputs(args)

    gate = json.loads((args.prepare_output_dir / "gate-requests.json").read_text())
    pilot = json.loads((args.prepare_output_dir / "pilot-requests.json").read_text())
    assert {row["run_id"] for row in gate} == {"videochat3-gate-20260801-a1"}
    assert {row["run_id"] for row in pilot} == {"videochat3-pilot-20260801-a1"}


def test_gate_run_freezes_only_three_videochat_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch)
    pil = types.ModuleType("PIL")
    pil.Image = object()
    monkeypatch.setitem(sys.modules, "PIL", pil)
    requests = [_request(video_id) for video_id in GATE_ROW_IDS]
    predictions = [
        {"video_id": video_id, "status": "passed", "answer": "short answer"}
        for video_id in GATE_ROW_IDS
    ]
    rows_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        videochat3_module,
        "load_frozen_requests",
        lambda *args, **kwargs: (requests, videochat3_module.GATE_REQUESTS_SHA256),
    )
    monkeypatch.setattr(videochat3_module, "_set_determinism", lambda torch: None)
    monkeypatch.setattr(
        videochat3_module,
        "validate_cached_request",
        lambda request, **kwargs: {"video_id": request["video_id"]},
    )
    monkeypatch.setattr(
        videochat3_module,
        "_load_videochat3",
        lambda *args: (
            "processor",
            "model",
            _MetadataBase,
            _VideoChat3MetadataBase,
            {"is_tied": True},
        ),
    )
    monkeypatch.setattr(
        videochat3_module,
        "validate_cuda_bfloat16",
        lambda model, torch: {"backend": "nvidia-cuda"},
    )
    monkeypatch.setattr(
        videochat3_module,
        "_videochat3_rows",
        lambda *args, **kwargs: rows_kwargs.update(kwargs) or predictions,
    )
    args = argparse.Namespace(
        gate_requests=Path("gate.json"),
        gate_requests_sha256=videochat3_module.GATE_REQUESTS_SHA256,
        videochat_model_dir=Path("model"),
        runtime_image="image",
        videochat_model_id="model-id",
        videochat_revision="revision",
    )

    record = videochat3_module.run_gate(args)

    assert record["mode"] == "compatibility-gate"
    assert record["gate_passed"] is True
    assert record["gate_completion"] == "3/3"
    assert record["gate_predictions"] == predictions
    assert "treatment_predictions" not in record
    assert "control_predictions" not in record
    assert record["references_loaded"] is False
    assert rows_kwargs["processor_metadata_base"] is _VideoChat3MetadataBase


def test_matched_pilot_uses_only_six_pilot_rows_for_both_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    pil = types.ModuleType("PIL")
    pil.Image = object()
    monkeypatch.setitem(sys.modules, "PIL", pil)
    requests = [_request(video_id, frame_count=24) for video_id in PILOT_ROW_IDS]
    treatment = [
        {"video_id": video_id, "status": "passed", "answer": "treatment"}
        for video_id in PILOT_ROW_IDS
    ]
    control = [
        {"video_id": video_id, "status": "passed", "answer": "control"}
        for video_id in PILOT_ROW_IDS
    ]
    monkeypatch.setattr(
        videochat3_module,
        "load_frozen_requests",
        lambda *args, **kwargs: (
            requests,
            videochat3_module.PILOT_24_REQUESTS_SHA256,
        ),
    )
    monkeypatch.setattr(videochat3_module, "_set_determinism", lambda torch: None)
    monkeypatch.setattr(
        videochat3_module,
        "validate_cached_request",
        lambda request, **kwargs: {"video_id": request["video_id"]},
    )
    monkeypatch.setattr(
        videochat3_module,
        "_load_videochat3",
        lambda *args: (
            "processor",
            "model",
            _MetadataBase,
            _VideoChat3MetadataBase,
            {"is_tied": True},
        ),
    )
    monkeypatch.setattr(
        videochat3_module,
        "_load_qwen",
        lambda *args: ("qwen-processor", "qwen-model"),
    )
    monkeypatch.setattr(
        videochat3_module,
        "validate_cuda_bfloat16",
        lambda model, torch: {"backend": "nvidia-cuda"},
    )
    monkeypatch.setattr(
        videochat3_module,
        "_videochat3_rows",
        lambda *args, **kwargs: videochat_kwargs.update(kwargs) or treatment,
    )
    monkeypatch.setattr(
        videochat3_module,
        "_qwen_rows",
        lambda *args, **kwargs: qwen_kwargs.update(kwargs) or control,
    )
    videochat_kwargs: dict[str, object] = {}
    qwen_kwargs: dict[str, object] = {}
    args = argparse.Namespace(
        pilot_requests=Path("pilot.json"),
        pilot_requests_sha256=videochat3_module.PILOT_24_REQUESTS_SHA256,
        videochat_model_dir=Path("videochat"),
        qwen_model_dir=Path("qwen"),
        runtime_image="image",
        videochat_model_id="videochat-id",
        videochat_revision="videochat-revision",
        qwen_model_id="qwen-id",
        qwen_revision="qwen-revision",
        frame_count=24,
    )

    record = videochat3_module.run_matched_pilot(args)

    assert record["mode"] == "matched-six"
    assert record["pilot_completion"] == "6/6+6/6"
    assert record["treatment_predictions"] == treatment
    assert record["control_predictions"] == control
    assert "gate_predictions" not in record
    assert record["references_loaded"] is False
    assert record["grades_loaded"] is False
    assert videochat_kwargs["frame_count"] == 24
    assert qwen_kwargs["frame_count"] == 24


def test_24_frame_pilot_requires_its_complete_frozen_manifest_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_called = False

    def unexpected_load(*args: object, **kwargs: object) -> object:
        nonlocal load_called
        load_called = True
        raise AssertionError("manifest must be rejected before loading")

    monkeypatch.setattr(videochat3_module, "load_frozen_requests", unexpected_load)
    args = argparse.Namespace(
        pilot_requests=Path("pilot-24.json"),
        pilot_requests_sha256=videochat3_module.PILOT_REQUESTS_SHA256,
        frame_count=24,
    )

    with pytest.raises(ValueError, match="pilot manifest SHA-256 pin mismatch"):
        videochat3_module.run_matched_pilot(args)

    assert load_called is False


def test_24_frame_matched_path_rejects_changed_content_with_same_row_ids(
    tmp_path: Path,
) -> None:
    changed_manifest = tmp_path / "pilot-24-changed.json"
    changed_manifest.write_text(
        json.dumps(
            [_request(video_id, frame_count=24) for video_id in PILOT_ROW_IDS],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        pilot_requests=changed_manifest,
        pilot_requests_sha256=videochat3_module.PILOT_24_REQUESTS_SHA256,
        frame_count=24,
    )

    with pytest.raises(ValueError, match="pilot manifest SHA-256 mismatch"):
        videochat3_module.run_matched_pilot(args)


def test_matched_24_frame_gate_runs_exact_predeclared_pair_for_both_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    pil = types.ModuleType("PIL")
    pil.Image = object()
    monkeypatch.setitem(sys.modules, "PIL", pil)
    requests = [_request(video_id, frame_count=24) for video_id in PILOT_ROW_IDS]
    observed: dict[str, object] = {}
    evidence_preflight: list[str] = []

    monkeypatch.setattr(
        videochat3_module,
        "load_frozen_requests",
        lambda *args, **kwargs: (
            requests,
            videochat3_module.PILOT_24_REQUESTS_SHA256,
        ),
    )
    monkeypatch.setattr(videochat3_module, "_set_determinism", lambda torch: None)
    monkeypatch.setattr(
        videochat3_module,
        "validate_cached_request",
        lambda request, **kwargs: evidence_preflight.append(request["video_id"])
        or {"video_id": request["video_id"]},
    )

    def load_videochat3(*args: object) -> tuple[object, ...]:
        assert evidence_preflight == list(PILOT_ROW_IDS)
        return (
            "processor",
            "model",
            _MetadataBase,
            _VideoChat3MetadataBase,
            {"is_tied": True},
        )

    monkeypatch.setattr(videochat3_module, "_load_videochat3", load_videochat3)
    monkeypatch.setattr(
        videochat3_module,
        "_load_qwen",
        lambda *args: ("qwen-processor", "qwen-model"),
    )
    monkeypatch.setattr(
        videochat3_module,
        "validate_cuda_bfloat16",
        lambda model, torch: {"backend": "nvidia-cuda"},
    )

    def treatment_rows(rows: list[dict[str, object]], **kwargs: object) -> list[dict[str, object]]:
        observed["treatment_ids"] = [row["video_id"] for row in rows]
        observed["treatment_frame_count"] = kwargs["frame_count"]
        return [
            {"video_id": row["video_id"], "status": "passed", "answer": "yes"}
            for row in rows
        ]

    def control_rows(rows: list[dict[str, object]], **kwargs: object) -> list[dict[str, object]]:
        observed["control_ids"] = [row["video_id"] for row in rows]
        observed["control_frame_count"] = kwargs["frame_count"]
        return [
            {"video_id": row["video_id"], "status": "passed", "answer": "yes"}
            for row in rows
        ]

    monkeypatch.setattr(videochat3_module, "_videochat3_rows", treatment_rows)
    monkeypatch.setattr(videochat3_module, "_qwen_rows", control_rows)
    args = argparse.Namespace(
        pilot_requests=Path("pilot-24.json"),
        pilot_requests_sha256=videochat3_module.PILOT_24_REQUESTS_SHA256,
        videochat_model_dir=Path("videochat"),
        qwen_model_dir=Path("qwen"),
        runtime_image="image",
        videochat_model_id="videochat-id",
        videochat_revision="videochat-revision",
        qwen_model_id="qwen-id",
        qwen_revision="qwen-revision",
        frame_count=24,
    )

    record = videochat3_module.run_matched_gate(args)

    expected_ids = list(videochat3_module.MATCHED_GATE_ROW_IDS)
    assert observed == {
        "treatment_ids": expected_ids,
        "treatment_frame_count": 24,
        "control_ids": expected_ids,
        "control_frame_count": 24,
    }
    assert record["mode"] == "matched-two-gate"
    assert record["gate_completion"] == "2/2+2/2"
    assert record["gate_passed"] is True
    assert record["references_loaded"] is False
    assert record["grades_loaded"] is False
    assert evidence_preflight == list(PILOT_ROW_IDS)


def test_cuda_memory_tracker_resets_and_records_peak_bytes() -> None:
    calls: list[object] = []
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            reset_peak_memory_stats=lambda: calls.append("reset"),
            max_memory_allocated=lambda: calls.append("read") or 123456,
        )
    )

    videochat3_module._reset_cuda_memory_tracking(torch)
    peak = videochat3_module._cuda_peak_memory_bytes(torch)

    assert calls == ["reset", "read"]
    assert peak == 123456


def test_cli_restricts_frame_count_to_supported_frozen_policies() -> None:
    parsed = videochat3_module.parse_args(
        ["--mode", "canary", "--frame-count", "24"]
    )

    assert parsed.frame_count == 24

    with pytest.raises(SystemExit):
        videochat3_module.parse_args(
            ["--mode", "canary", "--frame-count", "16"]
        )


@pytest.mark.parametrize("frame_count", (16, 23, 25))
def test_declared_pilot_hash_rejects_unsupported_frame_counts(
    frame_count: int,
) -> None:
    with pytest.raises(ValueError, match="unsupported frame count"):
        videochat3_module._declared_pilot_sha256(frame_count)


@pytest.mark.parametrize(
    "forbidden",
    (
        "answer",
        "annotation",
        "reference",
        "label",
        "grade",
        "gold",
        "ground_truth",
        "score",
    ),
)
def test_select_label_free_requests_rejects_recursive_label_fields(
    tmp_path: Path, forbidden: str
) -> None:
    source = tmp_path / "requests.json"
    rows = [_request(video_id) for video_id in PILOT_ROW_IDS]
    rows[0]["nested"] = {forbidden: "leak"}
    source.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden label-bearing field"):
        select_label_free_requests(source, run_id="run", condition_id="condition")


def test_select_label_free_requests_rejects_unexpected_source_fields(
    tmp_path: Path,
) -> None:
    """Catches whole source rows being copied into the frozen inference manifest."""

    source = tmp_path / "requests.json"
    rows = [_request(video_id) for video_id in PILOT_ROW_IDS]
    rows[0]["internal_note"] = "must not enter inference"
    source.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected request fields: internal_note"):
        select_label_free_requests(source, run_id="run", condition_id="condition")


def test_cuda_bfloat16_attestation_rejects_rocm_and_cpu_fallback() -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Version:
        cuda = "12.8"
        hip = None

    class _Torch:
        cuda = _Cuda()
        version = _Version()
        bfloat16 = "bf16"

    with pytest.raises(RuntimeError, match="CUDA accelerator is unavailable"):
        validate_cuda_bfloat16(object(), _Torch())


def test_declared_panel_has_four_targets_then_two_controls() -> None:
    assert PILOT_ROW_IDS == (*TARGET_IDS, *CONTROL_IDS)


def test_cli_guard_runs_only_after_all_runtime_helpers_are_defined() -> None:
    module_path = Path("src/trec_video_2026/vqa_experiments/videochat3_pilot.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    attestation = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_cuda_bfloat16"
    )
    main_guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )

    assert attestation.lineno < main_guard.lineno
