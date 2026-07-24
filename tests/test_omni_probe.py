from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

from trec_video_2026.vqa_experiments import omni_probe
from trec_video_2026.vqa_experiments.omni_probe import (
    MODEL_SPECS,
    ProbeStageError,
    _generate_one,
    _load_backend,
    classify_probe_failure,
    resolve_example_paths,
    validate_run_id,
)


def test_model_specs_pin_hub_revisions_and_joint_input_keys():
    assert MODEL_SPECS["phi4mm"].model_id == ("microsoft/Phi-4-multimodal-instruct")
    assert MODEL_SPECS["phi4mm"].revision == (
        "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
    )
    assert MODEL_SPECS["gemma3n"].model_id == "google/gemma-3n-E4B-it"
    assert MODEL_SPECS["gemma3n"].revision == (
        "c1221e9c62e34a43ab7ffacd1be0ea71f126ef10"
    )
    for spec in MODEL_SPECS.values():
        assert spec.vision_input_keys in (
            frozenset({"input_image_embeds"}),
            frozenset({"pixel_values"}),
        )
        assert spec.audio_input_keys in (
            frozenset({"input_audio_embeds"}),
            frozenset({"input_features"}),
        )


def test_resolve_example_paths_keeps_contract_relative_and_runtime_absolute(
    tmp_path: Path,
):
    example = {
        "frame_paths": ["frames/one.png", "frames/two.png"],
        "audio_path": "audio/source.wav",
        "media_path": "media/source.mkv",
    }

    resolved = resolve_example_paths(example, tmp_path)

    assert example["frame_paths"] == ["frames/one.png", "frames/two.png"]
    assert resolved["frame_paths"] == [
        str(tmp_path / "frames/one.png"),
        str(tmp_path / "frames/two.png"),
    ]
    assert resolved["audio_path"] == str(tmp_path / "audio/source.wav")


def test_run_id_rejects_path_escape_and_empty_values():
    for run_id in ("../escape", "nested/run", "", ".hidden", "white space"):
        with pytest.raises(ValueError, match="run-id"):
            validate_run_id(run_id)

    assert validate_run_id("phi4mm-rocm-bf16-20260724") == ("phi4mm-rocm-bf16-20260724")


def test_probe_failure_classification_distinguishes_contract_and_backend():
    processor = ProbeStageError("processor", ValueError("bad multimodal request"))
    transfer = ProbeStageError("device_transfer", RuntimeError("HIP out of memory"))
    generation = ProbeStageError("generation", RuntimeError("HIP out of memory"))

    assert classify_probe_failure(processor) == (
        processor.original_error,
        "processor",
        True,
    )
    assert classify_probe_failure(generation) == (
        generation.original_error,
        "generation",
        None,
    )
    assert classify_probe_failure(transfer) == (
        transfer.original_error,
        "device_transfer",
        None,
    )
    internal = RuntimeError("unexpected")
    assert classify_probe_failure(internal) == (internal, "internal", None)


def test_generate_one_labels_device_transfer_failure_separately(monkeypatch):
    class FakeInputs(dict):
        def to(self, *_args, **_kwargs):
            raise RuntimeError("HIP out of memory")

    inputs = FakeInputs(input_ids=SimpleNamespace(shape=(1, 3)))
    monkeypatch.setattr(omni_probe, "_processor_inputs_gemma", lambda *_args: inputs)

    with pytest.raises(ProbeStageError) as captured:
        _generate_one(
            model_key="gemma3n",
            spec=MODEL_SPECS["gemma3n"],
            processor=object(),
            model=object(),
            torch=object(),
            example={},
        )

    assert captured.value.stage == "device_transfer"


def test_phi_backend_does_not_require_gemma_class(monkeypatch, tmp_path):
    class FakeModel:
        def eval(self):
            return self

    class FakeAuto:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            cls.last_kwargs = _kwargs
            return FakeModel()

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    torch.bfloat16 = "bfloat16"
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeAuto
    transformers.AutoProcessor = FakeAuto
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    loaded_torch, processor, model = _load_backend("phi4mm", tmp_path)

    assert loaded_torch is torch
    assert isinstance(processor, FakeModel)
    assert isinstance(model, FakeModel)
    assert FakeAuto.last_kwargs["_attn_implementation"] == "sdpa"
