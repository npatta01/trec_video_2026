from pathlib import Path
import json
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

from trec_video_2026.vqa_experiments import omni_probe
from trec_video_2026.vqa_experiments.omni_probe import (
    MAX_NEW_TOKENS,
    MODEL_SPECS,
    ProbeStageError,
    _device_description,
    _generate_one,
    _load_backend,
    _runtime_versions,
    classify_probe_failure,
    resolve_example_paths,
    validate_run_id,
)


def test_gemma_generation_budget_allows_completed_structured_output():
    assert MAX_NEW_TOKENS == {"phi4mm": 160, "gemma3n": 320}


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


def test_runtime_attestation_distinguishes_cuda_from_rocm():
    cuda_torch = SimpleNamespace(
        __version__="2.5.1",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "NVIDIA A100-SXM4-80GB",
        ),
        version=SimpleNamespace(hip=None, cuda="12.4"),
    )
    transformers = SimpleNamespace(__version__="4.48.2")

    assert _device_description(cuda_torch) == (
        "cuda:0 (NVIDIA CUDA 12.4; NVIDIA A100-SXM4-80GB)"
    )
    assert _runtime_versions(cuda_torch, transformers) == {
        "python": omni_probe.platform.python_version(),
        "torch": "2.5.1",
        "transformers": "4.48.2",
        "rocm": None,
        "cuda": "12.4",
    }


def test_run_probe_validates_only_uploaded_inference_evidence(
    monkeypatch, tmp_path: Path
):
    examples = [
        {
            "video_id": f"video-{index}",
            "operational_tag": role,
            "question": "Question?",
        }
        for index, role in enumerate(
            ("lexical_speech", "prosody", "non_speech_sound"), start=1
        )
    ]
    contract = {"schema_version": 1, "examples": examples}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        omni_probe,
        "validate_inference_evidence",
        lambda *args, **kwargs: calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        omni_probe,
        "validate_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source-video validation must not run")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        omni_probe,
        "_load_backend",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("load stopped for test")),
    )

    summary = omni_probe.run_probe(
        model_key="phi4mm",
        model_path=tmp_path / "model",
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=tmp_path / "output",
        runtime_image="example/image",
        runtime_image_digest="sha256:" + "a" * 64,
        run_id="test-run",
        expected_contract_sha256=omni_probe.contract_sha256(contract),
    )

    assert calls
    assert summary["rows"] == 3
