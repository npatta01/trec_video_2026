import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

import trec_video_2026.vqa_experiments.qwen3_omni_frozen30_dstack as runner
from trec_video_2026.vqa_experiments.omni_runtime import (
    NEUTRAL_OUTPUT_INSTRUCTION,
    VISUAL_CONCISE_OUTPUT_INSTRUCTION,
)
from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_contract import (
    CONDITION_SPECS,
    RUN_ID,
    VIDEO_IDS,
)
from trec_video_2026.vqa_experiments.qwen3_omni_dstack import (
    AUDIO_INPUT_KEYS,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
)
from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_profiles import (
    A4_PROFILE,
    STAGE1_PROFILE,
    Frozen30ExperimentProfile,
)

EXPECTED_IDS = list(VIDEO_IDS)


def _contract(profile: Frozen30ExperimentProfile = A4_PROFILE) -> dict:
    examples = []
    for pair_index, video_id in enumerate(EXPECTED_IDS):
        examples.append(
            {
                "video_id": video_id,
                "question": f"Question {pair_index}?",
                "frame_paths": [
                    f"vqa/frames/{video_id}-{frame_index:02d}.jpg"
                    for frame_index in range(12)
                ],
                "frame_sha256s": [f"{pair_index:064x}" for _ in range(12)],
                "frame_timestamps": list(range(12)),
                "audio_path": f"vqa/audio/{video_id}.wav",
                "audio_sha256": f"{pair_index + 30:064x}",
            }
        )
    return {
        "schema_version": 1,
        "panel_id": "qwen3-omni-frozen30",
        "run_id": profile.run_id,
        "conditions": [
            {"condition_id": condition_id, "audio_enabled": audio_enabled}
            for condition_id, audio_enabled in profile.condition_specs
        ],
        "canary": {
            "video_id": profile.canary_video_id,
            "condition_id": profile.canary_condition_id,
        },
        "source_inputs": [
            {"path": "vqa/source/manifest.json", "sha256": "a" * 64},
            {"path": "vqa/source/requests.json", "sha256": "b" * 64},
        ],
        "examples": examples,
    }


def _attestation(
    audio_enabled: bool,
    *,
    do_sample: bool = True,
    enable_thinking: bool | None = None,
    seed: int = 20260724,
) -> dict:
    audio_keys = sorted(AUDIO_INPUT_KEYS) if audio_enabled else []
    return {
        "same_request": True,
        "same_forward_pass": True,
        "generation_call_count": 1,
        "generation_input_keys": [
            "image_grid_thw",
            "pixel_values",
            *audio_keys,
        ],
        "first_forward_input_keys": [
            "image_grid_thw",
            "pixel_values",
            *audio_keys,
        ],
        "vision_input_keys": ["image_grid_thw", "pixel_values"],
        "audio_input_keys": audio_keys,
        "vision_element_count": 12,
        "audio_element_count": 1 if audio_enabled else 0,
        "vision_input_sha256": "c" * 64,
        "audio_input_sha256": "d" * 64 if audio_enabled else None,
        "generation": {
            "ended_with_eos": True,
            "eos_token_id": 151645,
            "pad_token_id": 151643,
            "seed": seed,
            "max_new_tokens": 2048,
            "do_sample": do_sample,
            "enable_thinking": enable_thinking,
        },
    }


def _runtime() -> dict:
    return {
        "device": "cuda:0",
        "runtime_versions": {"torch": "test"},
        "gpu_name": "fake H100",
        "gpu_total_memory_bytes": 100 * 1024**3,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, bool]],
    profile: Frozen30ExperimentProfile = A4_PROFILE,
) -> None:
    class Cuda:
        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def max_memory_allocated() -> int:
            return 11

        @staticmethod
        def max_memory_reserved() -> int:
            return 22

    class Torch:
        cuda = Cuda()

    monkeypatch.setattr(
        runner, "validate_uploaded_contract", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner, "_load_model", lambda: (Torch(), object(), object(), _runtime())
    )

    def generate(**kwargs):
        video_id = kwargs["example"]["video_id"]
        audio_enabled = kwargs["audio_enabled"]
        do_sample = kwargs["do_sample"]
        enable_thinking = kwargs["enable_thinking"]
        seed = kwargs["seed"]
        assert do_sample is profile.do_sample
        assert enable_thinking is profile.enable_thinking
        assert seed == profile.generation_seed
        calls.append((video_id, audio_enabled))
        return '{"answer":"test","evidence":"observed cue"}', _attestation(
            audio_enabled,
            do_sample=do_sample,
            enable_thinking=enable_thinking,
            seed=seed,
        )

    monkeypatch.setattr(runner, "_generate_one", generate)


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, bool]],
    profile: Frozen30ExperimentProfile = A4_PROFILE,
) -> tuple[dict, Path]:
    _install_fakes(monkeypatch, calls, profile)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract(profile)), encoding="utf-8")
    output_dir = tmp_path / "output"
    summary = runner.run_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=output_dir,
        run_id=profile.run_id,
        profile=profile,
    )
    return summary, output_dir


def test_stage1_runner_is_frames_only_greedy_and_exactly_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches Stage 1 reusing A4 audio, sampling, instruction, or row topology."""
    calls: list[tuple[str, bool]] = []

    summary, output_dir = _run(tmp_path, monkeypatch, calls, STAGE1_PROFILE)

    canary = json.loads((output_dir / "canary.json").read_text())
    rows = _read_jsonl(output_dir / "predictions.jsonl")
    assert calls == [
        (STAGE1_PROFILE.canary_video_id, False),
        *((video_id, False) for video_id in EXPECTED_IDS),
    ]
    assert len(rows) == 30
    assert [
        (row["video_id"], row["condition_id"], row["pair_index"])
        for row in rows
    ] == [
        (video_id, STAGE1_PROFILE.condition_specs[0][0], pair_index)
        for pair_index, video_id in enumerate(EXPECTED_IDS)
    ]
    for record in [canary, *rows]:
        attestation = record["modality_attestation"]
        assert record["audio_enabled"] is False
        assert record["instruction_profile"] == "visual-concise"
        assert record["instruction_sha256"] == hashlib.sha256(
            VISUAL_CONCISE_OUTPUT_INSTRUCTION.encode("utf-8")
        ).hexdigest()
        assert attestation["audio_element_count"] == 0
        assert attestation["audio_input_keys"] == []
        assert not set(attestation["generation_input_keys"]) & AUDIO_INPUT_KEYS
        assert not set(attestation["first_forward_input_keys"]) & AUDIO_INPUT_KEYS
        assert attestation["generation"]["do_sample"] is False
    assert canary["scored"] is False
    assert summary["scored_rows"] == 30
    assert summary["scored_passed"] == 30
    assert summary["expected_order_passed"] is True
    assert summary["succeeded"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda row: row["modality_attestation"].update(
                audio_input_keys=["input_features"]
            ),
            id="audio-keys",
        ),
        pytest.param(
            lambda row: row["modality_attestation"]["generation"].update(
                do_sample=True
            ),
            id="sampled-generation",
        ),
        pytest.param(
            lambda row: row.update(instruction_sha256="0" * 64),
            id="instruction-digest",
        ),
        pytest.param(
            lambda row: row.update(instruction_profile="neutral"),
            id="instruction-profile",
        ),
    ],
)
def test_stage1_condition_gate_rejects_profile_attestation_mutations(mutation) -> None:
    """Catches eligibility after a Stage 1 modality, decoding, or prompt mutation."""
    instruction_sha256 = hashlib.sha256(
        VISUAL_CONCISE_OUTPUT_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    row = {
        "status": "passed",
        "output_parse_status": "passed",
        "audio_enabled": False,
        "instruction_profile": "visual-concise",
        "instruction_sha256": instruction_sha256,
        "modality_attestation": _attestation(False, do_sample=False),
    }
    mutation(row)

    assert runner._condition_passed(row, STAGE1_PROFILE) is False


def test_runner_rejects_a_mutated_stage1_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches execution under a profile that is not one of the two sealed values."""
    monkeypatch.setattr(
        runner, "validate_uploaded_contract", lambda *_args, **_kwargs: None
    )
    profile = replace(STAGE1_PROFILE, instruction_profile="neutral")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract(STAGE1_PROFILE)), encoding="utf-8")

    with pytest.raises(ValueError, match="sealed profile"):
        runner.preflight_frozen30_probe(
            contract_path=contract_path,
            data_root=tmp_path,
            run_id=profile.run_id,
            profile=profile,
        )


def test_stage1_wrong_row_count_is_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches Stage 1 success using any denominator other than thirty rows."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls, STAGE1_PROFILE)
    contract = _contract(STAGE1_PROFILE)
    contract["examples"].pop()
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    summary = runner.run_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=tmp_path / "output",
        run_id=STAGE1_PROFILE.run_id,
        profile=STAGE1_PROFILE,
    )

    assert summary["scored_rows"] == 29
    assert summary["scored_passed"] == 29
    assert summary["succeeded"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("condition_id", "wrong-stage1-condition"),
        ("pair_index", 1),
    ],
)
def test_stage1_condition_name_or_order_mutation_is_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    """Catches Stage 1 order checks that ignore condition identity or row position."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls, STAGE1_PROFILE)
    execute_condition = runner._execute_condition

    def mutate_first_scored(**kwargs):
        record = execute_condition(**kwargs)
        if kwargs["scored"] and kwargs["pair_index"] == 0:
            record[field] = value
        return record

    monkeypatch.setattr(runner, "_execute_condition", mutate_first_scored)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(_contract(STAGE1_PROFILE)), encoding="utf-8"
    )

    summary = runner.run_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=tmp_path / "output",
        run_id=STAGE1_PROFILE.run_id,
        profile=STAGE1_PROFILE,
    )

    assert summary["scored_rows"] == 30
    assert summary["scored_passed"] == 30
    assert summary["expected_order_passed"] is False
    assert summary["succeeded"] is False


def test_runner_canary_precedes_exact_adjacent_frozen_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches scored work before canary or reordered/unpaired panel calls."""
    calls: list[tuple[str, bool]] = []

    summary, output_dir = _run(tmp_path, monkeypatch, calls)

    scored_rows = _read_jsonl(output_dir / "predictions.jsonl")
    canary = json.loads((output_dir / "canary.json").read_text())
    assert calls[0] == ("NkXPPJoZ-h8", True)
    assert calls[1:] == [
        (video_id, audio_enabled)
        for video_id in EXPECTED_IDS
        for audio_enabled in (False, True)
    ]
    assert len(scored_rows) == 60
    assert [
        (row["video_id"], row["audio_enabled"], row["pair_index"])
        for row in scored_rows
    ] == [
        (video_id, audio_enabled, pair_index)
        for pair_index, video_id in enumerate(EXPECTED_IDS)
        for audio_enabled in (False, True)
    ]
    assert canary["video_id"] == "NkXPPJoZ-h8"
    assert canary["scored"] is False
    assert summary["canary_passed"] is True
    assert summary["scored_passed"] == 60
    assert summary["succeeded"] is True


def test_failed_canary_writes_diagnostics_without_scored_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a failed preflight gate that still spends scored calls."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls)

    def failed_canary(**kwargs):
        calls.append((kwargs["example"]["video_id"], kwargs["audio_enabled"]))
        return "not json", _attestation(True)

    monkeypatch.setattr(runner, "_generate_one", failed_canary)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    output_dir = tmp_path / "output"

    summary = runner.run_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=output_dir,
        run_id=RUN_ID,
    )

    assert calls == [("NkXPPJoZ-h8", True)]
    assert (output_dir / "predictions.jsonl").read_text() == ""
    assert {path.name for path in output_dir.iterdir()} == set(runner.OUTPUT_NAMES)
    assert json.loads((output_dir / "canary.json").read_text())["status"] == "failed"
    assert summary["canary_passed"] is False
    assert summary["scored_rows"] == 0
    assert summary["succeeded"] is False


def test_all_outputs_are_reserved_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches late output reservation that can overwrite evidence after GPU spend."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    output_dir = tmp_path / "output"
    default_load = runner._load_model

    def inspect_reservations():
        assert {path.name for path in output_dir.iterdir()} == {
            "preflight.json",
            "canary.json",
            "predictions.jsonl",
            "summary.json",
            "manifest.json",
        }
        assert (output_dir / "canary.json").read_bytes() == b""
        assert (output_dir / "summary.json").read_bytes() == b""
        assert (output_dir / "manifest.json").read_bytes() == b""
        return default_load()

    monkeypatch.setattr(runner, "_load_model", inspect_reservations)

    summary = runner.run_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=output_dir,
        run_id=RUN_ID,
    )

    assert summary["succeeded"] is True


def test_model_load_failure_finalizes_every_reserved_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches empty reserved artifacts after a model-load exception."""
    monkeypatch.setattr(
        runner, "validate_uploaded_contract", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_load_model",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic load failure")),
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    output_dir = tmp_path / "output"

    summary = runner.run_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=output_dir,
        run_id=RUN_ID,
    )

    assert {path.name for path in output_dir.iterdir()} == set(runner.OUTPUT_NAMES)
    assert (output_dir / "predictions.jsonl").read_bytes() == b""
    canary = json.loads((output_dir / "canary.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert canary["status"] == "failed"
    assert canary["failure_stage"] == "model_load"
    assert summary["scored_rows"] == 0
    assert summary["succeeded"] is False
    for name, digest in manifest["outputs"].items():
        assert digest == hashlib.sha256((output_dir / name).read_bytes()).hexdigest()


def test_scored_failure_is_preserved_and_run_continues_to_complete_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches partial-output truncation or masking a failed scored condition."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls)
    attempt = 0

    def fail_one_scored_call(**kwargs):
        nonlocal attempt
        attempt += 1
        calls.append((kwargs["example"]["video_id"], kwargs["audio_enabled"]))
        if attempt == 10:
            raise RuntimeError("synthetic scored failure")
        return '{"answer":"test","evidence":"observed cue"}', _attestation(
            kwargs["audio_enabled"]
        )

    monkeypatch.setattr(runner, "_generate_one", fail_one_scored_call)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    output_dir = tmp_path / "output"

    summary = runner.run_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=output_dir,
        run_id=RUN_ID,
    )

    rows = _read_jsonl(output_dir / "predictions.jsonl")
    assert len(calls) == 61
    assert len(rows) == 60
    assert rows[8]["status"] == "failed"
    assert rows[8]["failure_stage"] == "internal"
    assert rows[0]["status"] == "passed"
    assert rows[-1]["status"] == "passed"
    assert summary["scored_passed"] == 59
    assert summary["succeeded"] is False


def test_scored_cuda_cleanup_failure_is_contained_and_remaining_rows_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches CUDA cleanup exceptions escaping before a diagnostic row is appended."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls)
    torch, processor, model, runtime = runner._load_model()
    cleanup_calls = 0

    def fail_first_scored_cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 2:
            raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(torch.cuda, "empty_cache", fail_first_scored_cleanup)
    monkeypatch.setattr(
        runner, "_load_model", lambda: (torch, processor, model, runtime)
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    output_dir = tmp_path / "output"

    summary = runner.run_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=output_dir,
        run_id=RUN_ID,
    )

    rows = _read_jsonl(output_dir / "predictions.jsonl")
    assert len(rows) == 60
    assert rows[0]["status"] == "failed"
    assert rows[0]["failure_stage"] == "cuda_cleanup"
    assert rows[-1]["status"] == "passed"
    assert len(calls) == 60
    assert summary["scored_passed"] == 59
    assert summary["succeeded"] is False


def test_final_memory_failure_is_contained_and_remaining_rows_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches final memory measurement suppressing a row and all final artifacts."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls)
    torch, processor, model, runtime = runner._load_model()
    reserved_calls = 0

    def fail_first_scored_reserved_measurement() -> int:
        nonlocal reserved_calls
        reserved_calls += 1
        if reserved_calls == 2:
            raise RuntimeError("synthetic reserved-memory failure")
        return 22

    monkeypatch.setattr(
        torch.cuda, "max_memory_reserved", fail_first_scored_reserved_measurement
    )
    monkeypatch.setattr(
        runner, "_load_model", lambda: (torch, processor, model, runtime)
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    output_dir = tmp_path / "output"

    summary = runner.run_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=output_dir,
        run_id=RUN_ID,
    )

    rows = _read_jsonl(output_dir / "predictions.jsonl")
    assert len(rows) == 60
    assert rows[0]["status"] == "failed"
    assert rows[0]["failure_stage"] == "memory_attestation"
    assert rows[-1]["status"] == "passed"
    assert len(calls) == 61
    assert summary["scored_passed"] == 59
    assert summary["succeeded"] is False


@pytest.mark.parametrize(
    ("mutation", "audio_enabled"),
    [
        (lambda value: value.update(vision_element_count=11), False),
        (lambda value: value.update(audio_element_count=1), False),
        (lambda value: value.update(audio_input_keys=[]), True),
        (lambda value: value.update(generation_call_count=2), False),
        (lambda value: value.update(same_forward_pass=False), False),
        (lambda value: value["generation"].update(ended_with_eos=False), False),
    ],
)
def test_condition_gate_rejects_any_missing_modality_or_execution_attestation(
    mutation, audio_enabled: bool
) -> None:
    """Catches eligibility when any one-call/forward/modality/EOS proof is false."""
    attestation = deepcopy(_attestation(audio_enabled))
    mutation(attestation)
    record = {
        "status": "passed",
        "output_parse_status": "passed",
        "audio_enabled": audio_enabled,
        "modality_attestation": attestation,
    }

    assert runner._condition_passed(record) is False


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda value: value.update(generation_call_count=True), id="call-bool"
        ),
        pytest.param(
            lambda value: value.update(generation_call_count=1.0), id="call-float"
        ),
        pytest.param(
            lambda value: value.update(vision_element_count=True), id="vision-bool"
        ),
        pytest.param(
            lambda value: value.update(vision_element_count=12.0), id="vision-float"
        ),
        pytest.param(
            lambda value: value.update(audio_element_count=False), id="audio-bool"
        ),
        pytest.param(lambda value: value.update(same_request=1), id="request-int"),
        pytest.param(lambda value: value.update(same_forward_pass=1), id="forward-int"),
        pytest.param(
            lambda value: value.update(audio_input_keys=()), id="audio-keys-tuple"
        ),
        pytest.param(
            lambda value: value.update(audio_input_keys=["input_features"]),
            id="audio-keys-missing",
        ),
        pytest.param(
            lambda value: value.update(generation_input_keys=["pixel_values"]),
            id="generation-audio-keys-missing",
        ),
        pytest.param(
            lambda value: value.update(first_forward_input_keys=["pixel_values"]),
            id="forward-audio-keys-missing",
        ),
        pytest.param(lambda value: value.update(generation=[]), id="generation-list"),
        pytest.param(
            lambda value: value["generation"].pop("eos_token_id"), id="eos-missing"
        ),
        pytest.param(
            lambda value: value["generation"].update(eos_token_id=True), id="eos-bool"
        ),
        pytest.param(
            lambda value: value["generation"].update(eos_token_id=151645.0),
            id="eos-float",
        ),
        pytest.param(
            lambda value: value["generation"].update(eos_token_id=151643),
            id="eos-wrong",
        ),
        pytest.param(
            lambda value: value["generation"].update(pad_token_id=151645),
            id="pad-wrong",
        ),
        pytest.param(
            lambda value: value["generation"].update(seed=20260725), id="seed-wrong"
        ),
        pytest.param(
            lambda value: value["generation"].update(max_new_tokens=2047),
            id="tokens-wrong",
        ),
        pytest.param(
            lambda value: value["generation"].update(do_sample=1), id="sample-int"
        ),
        pytest.param(
            lambda value: value["generation"].update(enable_thinking=True),
            id="thinking-wrong",
        ),
        pytest.param(
            lambda value: value["generation"].update(ended_with_eos=1), id="ended-int"
        ),
    ],
)
def test_condition_gate_rejects_wrong_or_missing_pinned_attestation_values(
    mutation,
) -> None:
    """Catches Python equality accepting aliases for pinned generation evidence."""
    attestation = deepcopy(_attestation(True))
    mutation(attestation)
    record = {
        "status": "passed",
        "output_parse_status": "passed",
        "audio_enabled": True,
        "modality_attestation": attestation,
    }

    assert runner._condition_passed(record) is False


def test_wrong_named_condition_makes_exact_scored_order_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches order validation that binds only the condition's boolean alias."""
    calls: list[tuple[str, bool]] = []
    _install_fakes(monkeypatch, calls)
    execute_condition = runner._execute_condition

    def rename_first_scored_condition(**kwargs):
        record = execute_condition(**kwargs)
        if (
            kwargs["scored"]
            and kwargs["pair_index"] == 0
            and not record["audio_enabled"]
        ):
            record["condition_id"] = "wrong-frames-condition"
        return record

    monkeypatch.setattr(runner, "_execute_condition", rename_first_scored_condition)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    output_dir = tmp_path / "output"

    summary = runner.run_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        output_dir=output_dir,
        run_id=RUN_ID,
    )

    assert summary["scored_rows"] == 60
    assert summary["scored_passed"] == 60
    assert summary["expected_order_passed"] is False
    assert summary["succeeded"] is False


@pytest.mark.parametrize("output_name", runner.OUTPUT_NAMES)
def test_existing_output_fails_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_name: str
) -> None:
    """Catches overwriting any reserved evidence artifact after model loading."""
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / output_name).write_text("reserved", encoding="utf-8")
    loaded = False
    monkeypatch.setattr(
        runner, "validate_uploaded_contract", lambda *_args, **_kwargs: None
    )

    def load_model():
        nonlocal loaded
        loaded = True
        raise AssertionError("model must not load")

    monkeypatch.setattr(runner, "_load_model", load_model)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run_frozen30_probe(
            contract_path=contract_path,
            data_root=tmp_path,
            output_dir=output_dir,
            run_id=RUN_ID,
        )

    assert loaded is False


def test_manifest_binds_all_outputs_sources_contract_and_neutral_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an artifact set that cannot prove its code/input/instruction lineage."""
    calls: list[tuple[str, bool]] = []
    _summary, output_dir = _run(tmp_path, monkeypatch, calls)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    instruction_sha256 = hashlib.sha256(
        NEUTRAL_OUTPUT_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    assert manifest["inputs"] == {
        "contract_sha256": runner.contract_sha256(_contract()),
        "instruction_profile": "neutral",
        "instruction_sha256": instruction_sha256,
    }
    assert set(manifest["outputs"]) == {
        "preflight.json",
        "canary.json",
        "predictions.jsonl",
        "summary.json",
    }
    for name, digest in manifest["outputs"].items():
        assert digest == hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
    assert set(manifest["sources"]) == {
        "omni_contract.py",
        "omni_runtime.py",
        "qwen3_omni_dstack.py",
        "qwen3_omni_frozen30_contract.py",
        "qwen3_omni_frozen30_dstack.py",
        "qwen3_omni_frozen30_profiles.py",
    }
    serialized = json.dumps(manifest).lower()
    for forbidden in (
        "frame_paths",
        "audio_path",
        "reference",
        "distractor",
        "transcript",
        "grade",
        "score",
    ):
        assert forbidden not in serialized


def test_preflight_validates_real_inputs_without_model_load_or_output_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract()), encoding="utf-8")
    monkeypatch.setattr(
        runner, "validate_uploaded_contract", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_load_model",
        lambda: (_ for _ in ()).throw(AssertionError("preflight must not load model")),
    )

    record = runner.preflight_frozen30_probe(
        contract_path=contract_path,
        data_root=tmp_path,
        run_id=RUN_ID,
    )

    assert set(tmp_path.iterdir()) == {contract_path}
    assert record == {
        "schema_version": 1,
        "run_id": RUN_ID,
        "contract_sha256": runner.contract_sha256(_contract()),
        "instruction_profile": "neutral",
        "instruction_sha256": hashlib.sha256(
            NEUTRAL_OUTPUT_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "video_count": 30,
        "frame_files": 360,
        "audio_files": 30,
        "condition_ids": [condition_id for condition_id, _ in CONDITION_SPECS],
        "canary_video_id": "NkXPPJoZ-h8",
        "generation": {
            "seed": 20260724,
                "max_new_tokens": 2048,
                "do_sample": True,
                "eos_token_id": 151645,
            "pad_token_id": 151643,
        },
    }


def test_cli_preflight_uses_real_contract_runtime_arguments_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_preflight(**kwargs):
        observed.update(kwargs)
        return {"status": "preflight-passed"}

    monkeypatch.setattr(runner, "preflight_frozen30_probe", fake_preflight)
    monkeypatch.setattr(
        runner,
        "run_frozen30_probe",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("full workload must not run during preflight")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen3_omni_frozen30_dstack",
            "--contract",
            str(tmp_path / "contract.json"),
            "--data-root",
            str(tmp_path),
            "--run-id",
            RUN_ID,
            "--experiment",
            "a4",
            "--preflight",
        ],
    )

    assert runner.main() == 0
    assert observed == {
        "contract_path": tmp_path / "contract.json",
        "data_root": tmp_path,
        "run_id": RUN_ID,
        "profile": A4_PROFILE,
    }
    assert json.loads(capsys.readouterr().out) == {"status": "preflight-passed"}


@pytest.mark.parametrize(("succeeded", "expected"), [(True, 0), (False, 1)])
def test_cli_exit_status_reflects_only_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    succeeded: bool,
    expected: int,
) -> None:
    """Catches a CLI that masks an ineligible experiment as successful."""
    observed: dict[str, object] = {}

    def fake_probe(**kwargs):
        observed.update(kwargs)
        return {"succeeded": succeeded}

    monkeypatch.setattr(runner, "run_frozen30_probe", fake_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen3_omni_frozen30_dstack",
            "--contract",
            str(tmp_path / "contract.json"),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--run-id",
            RUN_ID,
            "--experiment",
            "a4",
        ],
    )

    assert runner.main() == expected
    assert observed == {
        "contract_path": tmp_path / "contract.json",
        "data_root": tmp_path,
        "output_dir": tmp_path / "output",
        "run_id": RUN_ID,
        "profile": A4_PROFILE,
    }


def test_records_bind_exact_model_runtime_run_and_neutral_instruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches drift from the predeclared immutable experiment constants."""
    calls: list[tuple[str, bool]] = []
    _summary, output_dir = _run(tmp_path, monkeypatch, calls)

    row = _read_jsonl(output_dir / "predictions.jsonl")[0]
    assert {
        "run_id": row["run_id"],
        "model_id": row["model_id"],
        "model_revision": row["model_revision"],
        "runtime_image": row["runtime_image"],
        "runtime_image_digest": row["runtime_image_digest"],
        "instruction_profile": row["instruction_profile"],
    } == {
        "run_id": "qwen3-omni-frozen30-audio-20260731-a4",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "instruction_profile": "neutral",
    }
    assert row["condition_request_sha256"]
    assert row["contract_sha256"] == runner.contract_sha256(_contract())
