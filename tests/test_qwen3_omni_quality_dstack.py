import json
from pathlib import Path

import pytest

import trec_video_2026.vqa_experiments.qwen3_omni_quality_dstack as quality_runner


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
        "validate_quality_contract",
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
