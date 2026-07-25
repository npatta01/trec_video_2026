import hashlib
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import trec_video_2026.vqa_experiments.qwen3_omni_quality_dstack as quality_runner
from trec_video_2026.vqa_experiments.qwen3_omni_dstack import build_messages
from trec_video_2026.vqa_experiments.omni_runtime import (
    NEUTRAL_OUTPUT_INSTRUCTION,
)


def _contract() -> dict:
    return {
        "schema_version": 1,
        "panel_id": "qwen3-omni-quality-10",
        "conditions": [
            {"condition_id": "frames-only", "audio_enabled": False},
            {"condition_id": "frames-native-audio", "audio_enabled": True},
        ],
        "source_inputs": [],
        "examples": [
            {
                "video_id": "first-video",
                "question": "What happens?",
                "prompt_repair": False,
                "frame_paths": [f"vqa/frame-{index:02d}.png" for index in range(12)],
                "frame_sha256s": ["0" * 64 for _ in range(12)],
                "frame_timestamps": list(range(12)),
                "audio_path": "vqa/audio/first-video.wav",
                "audio_sha256": "1" * 64,
            },
            {
                "video_id": "second-video",
                "question": "What changes?",
                "prompt_repair": False,
                "frame_paths": [f"vqa/second-{index:02d}.png" for index in range(12)],
                "frame_sha256s": ["2" * 64 for _ in range(12)],
                "frame_timestamps": list(range(12)),
                "audio_path": "vqa/audio/second-video.wav",
                "audio_sha256": "3" * 64,
            },
        ],
    }


def _attestation(audio_enabled: bool) -> dict:
    return {
        "same_request": True,
        "same_forward_pass": True,
        "generation_call_count": 1,
        "generation_input_keys": ["image_grid_thw", "pixel_values"],
        "first_forward_input_keys": ["image_grid_thw", "pixel_values"],
        "vision_input_keys": ["image_grid_thw", "pixel_values"],
        "audio_input_keys": (
            []
            if not audio_enabled
            else ["feature_attention_mask", "input_features"]
        ),
        "vision_element_count": 12,
        "audio_element_count": 1 if audio_enabled else 0,
        "vision_input_sha256": "a" * 64,
        "audio_input_sha256": "b" * 64 if audio_enabled else None,
        "generation": {"ended_with_eos": True},
    }


def _runtime() -> dict:
    return {
        "device": "cuda:0",
        "runtime_versions": {"torch": "test"},
        "gpu_name": "test GPU",
        "gpu_total_memory_bytes": 100,
    }


def test_neutral_profile_keeps_pair_text_identical_and_changes_only_audio_element():
    """Catches a neutral a2 prompt that leaks condition-specific modality language."""
    example = _contract()["examples"][0]
    data_root = Path("/dstack/input")

    frames_only = build_messages(
        example,
        data_root,
        audio_enabled=False,
        output_instruction=NEUTRAL_OUTPUT_INSTRUCTION,
    )[0]["content"]
    audio_on = build_messages(
        example,
        data_root,
        audio_enabled=True,
        output_instruction=NEUTRAL_OUTPUT_INSTRUCTION,
    )[0]["content"]

    assert [item["type"] for item in frames_only] == ["image"] * 12 + ["text"]
    assert [item["type"] for item in audio_on] == ["image"] * 12 + ["audio", "text"]
    assert frames_only == [item for item in audio_on if item["type"] != "audio"]
    neutral_lower = frames_only[-1]["text"].lower()
    assert "audio" not in neutral_lower
    assert "auditory" not in neutral_lower
    assert "modality" not in neutral_lower


def _install_fakes(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, bool]]):
    class Cuda:
        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def max_memory_allocated():
            return 0

        @staticmethod
        def max_memory_reserved():
            return 0

    class Torch:
        cuda = Cuda()

    monkeypatch.setattr(
        quality_runner,
        "validate_uploaded_quality_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        quality_runner,
        "_load_model",
        lambda: (Torch(), object(), object(), _runtime()),
    )

    def generate(**kwargs):
        calls.append((kwargs["example"]["video_id"], kwargs["audio_enabled"]))
        return '{"answer":"test","evidence":"visible cue"}', _attestation(
            kwargs["audio_enabled"]
        )

    monkeypatch.setattr(quality_runner, "_generate_one", generate)


def test_quality_runner_writes_adjacent_condition_records_with_distinct_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catches reordered conditions or request digests that omit the condition."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")

    summary = quality_runner.run_quality_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=tmp_path / "output",
        run_id="test-run",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "output" / "predictions.jsonl").read_text().splitlines()
    ]
    assert calls == [
        ("first-video", False),
        ("first-video", True),
        ("second-video", False),
        ("second-video", True),
    ]
    assert [(row["condition_id"], row["audio_enabled"], row["pair_index"]) for row in rows] == [
        ("frames-only", False, 0),
        ("frames-native-audio", True, 0),
        ("frames-only", False, 1),
        ("frames-native-audio", True, 1),
    ]
    assert rows[0]["request_sha256"] != rows[1]["request_sha256"]
    assert summary["rows"] == 4
    assert summary["canary_passed"] is True
    assert summary["succeeded"] is True
    preflight = json.loads((tmp_path / "output" / "preflight.json").read_text())
    assert preflight == {
        "audio_files": 2,
        "condition_ids": ["frames-only", "frames-native-audio"],
        "frame_files": 24,
        "instruction_profile": "audio-expecting",
        "instruction_sha256": "55c340c622c4b153cd9756b2c2070b88b2b7eb9a8772637130432ea2f1d2f676",
        "quality_contract_sha256": quality_runner.quality_contract_sha256(_contract()),
        "run_id": "test-run",
        "schema_version": 1,
        "video_count": 2,
    }


def test_quality_runner_binds_neutral_profile_and_instruction_hash_to_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catches an a2 result that cannot prove which instruction generated it."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    output_dir = tmp_path / "output"

    summary = quality_runner.run_quality_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=output_dir,
        run_id="test-run",
        instruction_profile="neutral",
    )

    instruction_sha256 = hashlib.sha256(
        NEUTRAL_OUTPUT_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    rows = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    preflight = json.loads((output_dir / "preflight.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert {(row["instruction_profile"], row["instruction_sha256"]) for row in rows} == {
        ("neutral", instruction_sha256)
    }
    assert preflight["instruction_profile"] == "neutral"
    assert preflight["instruction_sha256"] == instruction_sha256
    assert summary["instruction_profile"] == "neutral"
    assert summary["instruction_sha256"] == instruction_sha256
    assert manifest["inputs"]["instruction_profile"] == "neutral"
    assert manifest["inputs"]["instruction_sha256"] == instruction_sha256


def test_quality_runner_stops_after_a_failed_canary_but_persists_both_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catches spending on later pairs after either canary condition breaks."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls)

    def generation_with_bad_audio_canary(**kwargs):
        calls.append((kwargs["example"]["video_id"], kwargs["audio_enabled"]))
        return "not json", _attestation(kwargs["audio_enabled"])

    monkeypatch.setattr(quality_runner, "_generate_one", generation_with_bad_audio_canary)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")

    summary = quality_runner.run_quality_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=tmp_path / "output",
        run_id="test-run",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "output" / "predictions.jsonl").read_text().splitlines()
    ]
    assert calls == [("first-video", False), ("first-video", True)]
    assert len(rows) == 2
    assert rows[-1]["status"] == "failed"
    assert summary["canary_passed"] is False
    assert summary["succeeded"] is False


def test_quality_runner_cli_forwards_wrapper_arguments_and_fails_on_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catches a module entry point that drops wrapper arguments or masks failure."""
    contract_path = tmp_path / "contract.json"
    output_dir = tmp_path / "output"
    observed: dict[str, object] = {}

    def failed_probe(**kwargs: object) -> dict:
        observed.update(kwargs)
        return {"succeeded": False}

    monkeypatch.setattr(quality_runner, "run_quality_probe", failed_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen3_omni_quality_dstack",
            "--contract",
            str(contract_path),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "test-run",
        ],
    )

    assert quality_runner.main() == 1
    assert observed == {
        "contract_path": contract_path,
        "data_root": tmp_path,
        "output_dir": output_dir,
        "run_id": "test-run",
        "instruction_profile": "audio-expecting",
    }


def test_quality_runner_cli_selects_neutral_instruction_for_every_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catches an a2 wrapper whose neutral profile is ignored by the runner."""
    observed: dict[str, object] = {}

    def failed_probe(**kwargs: object) -> dict:
        observed.update(kwargs)
        return {"succeeded": False}

    monkeypatch.setattr(quality_runner, "run_quality_probe", failed_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen3_omni_quality_dstack",
            "--contract",
            str(tmp_path / "contract.json"),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--run-id",
            "test-run",
            "--instruction-profile",
            "neutral",
        ],
    )

    assert quality_runner.main() == 1
    assert observed["instruction_profile"] == "neutral"


def test_uploaded_contract_validation_needs_only_contract_bound_evidence(
    tmp_path: Path,
):
    """Catches remote validation that reaches for unuploaded build-time inputs."""
    from trec_video_2026.vqa_experiments.qwen3_omni_quality_contract import (
        build_contract,
        validate_uploaded_quality_contract,
    )

    contract = build_contract(Path("data"))
    for example in contract["examples"]:
        for relative_path in [*example["frame_paths"], example["audio_path"]]:
            source = Path("data") / relative_path
            destination = tmp_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    validate_uploaded_quality_contract(contract, tmp_path, verify_files=True)


def test_uploaded_contract_validation_rejects_a_video_bound_to_another_audio_file(
    tmp_path: Path,
):
    """Catches a remote contract that swaps a valid WAV between fixed videos."""
    from trec_video_2026.vqa_experiments.qwen3_omni_quality_contract import (
        build_contract,
        validate_uploaded_quality_contract,
    )

    contract = build_contract(Path("data"))
    swapped = deepcopy(contract)
    swapped["examples"][0]["audio_path"] = contract["examples"][1]["audio_path"]
    swapped["examples"][0]["audio_sha256"] = contract["examples"][1]["audio_sha256"]

    with pytest.raises(ValueError, match="audio_path does not match the fixed panel"):
        validate_uploaded_quality_contract(swapped, tmp_path, verify_files=False)
