from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.omni_probe import (
    MODEL_SPECS,
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
