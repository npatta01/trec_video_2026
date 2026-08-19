import contextlib
import inspect
import json
from pathlib import Path
import re

import pytest

from trec_video_2026.vqa_experiments.dstack_metadata import sanitize_dstack_run
from trec_video_2026.vqa_experiments.qwen3_omni_dstack import (
    CONTRACT_SHA256,
    EXPECTED_EOS_TOKEN_ID,
    EXPECTED_PAD_TOKEN_ID,
    GENERATION_SEED,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    ProbeStageError,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
    _generate_one,
    _preserve_failure_attestation,
    attest_qwen_inputs,
    build_messages,
    generation_special_token_ids,
    normalize_generation_output,
    validate_panel_identity,
)

CONFIG = Path("dstack/qwen3-omni-compatibility.dstack.yml")


def _example() -> dict:
    return {
        "video_id": "NN7G28sBOlY",
        "operational_tag": "lexical_speech",
        "question": "What is said?",
        "frame_paths": [f"frames/frame-{index:02d}.png" for index in range(1, 13)],
        "audio_path": "audio/NN7G28sBOlY.wav",
    }


def test_qwen_runtime_uses_immutable_model_and_image_identifiers():
    text = CONFIG.read_text(encoding="utf-8")
    top_level_images = [line for line in text.splitlines() if line.startswith("image:")]

    assert MODEL_ID == "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    assert MODEL_REVISION == "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
    assert RUNTIME_IMAGE_DIGEST == (
        "sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
    )
    assert RUNTIME_IMAGE.endswith(f"@{RUNTIME_IMAGE_DIGEST}")
    assert MODEL_REVISION in text
    assert top_level_images == [f"image: {RUNTIME_IMAGE}"]
    assert ":latest" not in text


def test_dstack_task_has_fixed_resource_and_cost_envelope():
    text = CONFIG.read_text(encoding="utf-8")

    assert "type: task" in text
    assert "ports:" not in text
    assert "memory: 90GB.." in text
    assert "name: [H100, H200, B200, RTXPRO6000WS]" in text
    assert "backends: [runpod, vastai]" in text
    assert "spot_policy: on-demand" in text
    assert "max_price: 4.50" in text
    assert "max_duration: 45m" in text
    assert "on_events: [no-capacity]" in text
    assert "duration: 30m" in text


def test_dstack_task_uploads_only_frozen_inputs_and_syncs_private_results():
    text = CONFIG.read_text(encoding="utf-8")
    runner = Path("scripts/run-qwen3-omni-dstack.sh").read_text(encoding="utf-8")
    combined = text + runner
    files_section = text.split("files:\n", 1)[1].split("\ncommands:", 1)[0]
    mappings = [
        line.removeprefix("  - ")
        for line in files_section.splitlines()
        if line.startswith("  - ")
    ]

    assert CONTRACT_SHA256 in text
    assert "/dstack/input/contract.json" in text
    assert "/dstack/input/vqa/benchmarks/evidence/" in text
    assert text.count(".png:/dstack/input/vqa/benchmarks/evidence/") == 36
    assert len(mappings) == 42
    assert mappings[:3] == [
        "../scripts/run-qwen3-omni-dstack.sh:"
        "/dstack/run/scripts/run-qwen3-omni-dstack.sh",
        "../src/trec_video_2026:/dstack/run/src/trec_video_2026",
        "../data/vqa/benchmarks/omni-compatibility-3/contract.json:"
        "/dstack/input/contract.json",
    ]
    frame_pattern = re.compile(
        r"\.\./data/vqa/benchmarks/evidence/[0-9a-f]{64}/frame-\d{2}\.png:"
        r"/dstack/input/vqa/benchmarks/evidence/[0-9a-f]{64}/frame-\d{2}\.png"
    )
    assert all(frame_pattern.fullmatch(mapping) for mapping in mappings[3:39])
    assert all(
        mapping.startswith(
            "../data/vqa/benchmarks/runs/native-audio-benchmark-v2/audio/"
        )
        and ":/dstack/input/vqa/benchmarks/runs/native-audio-benchmark-v2/audio/"
        in mapping
        and mapping.endswith(".wav")
        for mapping in mappings[39:]
    )
    assert (
        "/dstack/input/vqa/benchmarks/runs/native-audio-benchmark-v2/audio/"
        "NN7G28sBOlY.wav"
    ) in text
    assert (
        "/dstack/input/vqa/benchmarks/runs/native-audio-benchmark-v2/audio/"
        "NUmb2P3Bzu8.wav"
    ) in text
    assert (
        "/dstack/input/vqa/benchmarks/runs/native-audio-benchmark-v2/audio/"
        "NkXPPJoZ-h8.wav"
    ) in text
    assert 'chmod -R a-w "$DATA_ROOT"' in combined
    assert "HF_TOKEN=${{ secrets.hf_token }}" in text
    assert (
        "hf://buckets/Npatta01/trec_video_2026/"
        "experiments/qwen3-omni-compatibility-3-20260724-a4"
    ) in combined
    assert "name: qwen3-omni-compatibility-3-20260724-a4" in text
    assert 'readonly EXPECTED_RUN_ID="qwen3-omni-compatibility-3-20260724-a4"' in runner
    assert 'uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"' in runner
    assert "--break-system-packages" not in runner
    assert "hf_cli buckets sync" in runner
    assert "--delete" not in combined


def test_dstack_wrapper_syncs_sanitized_artifacts_after_preflight_failure():
    runner = Path("scripts/run-qwen3-omni-dstack.sh").read_text(encoding="utf-8")

    assert "preflight_rc=$?" in runner
    assert "finalize_and_sync" in runner
    assert 'return "$preflight_rc"' in runner
    assert "preflight.stdout.json" in runner
    assert "preflight.stderr" not in runner


def test_dstack_wrapper_fails_closed_on_all_absolute_io_paths():
    runner = Path("scripts/run-qwen3-omni-dstack.sh").read_text(encoding="utf-8")

    assert '[[ "$CONTRACT_PATH" == /dstack/input/contract.json ]]' in runner
    assert '[[ "$DATA_ROOT" == /dstack/input ]]' in runner
    assert '[[ "$OUTPUT_DIR" == /dstack/output ]]' in runner
    assert 'chmod -R a-w "$DATA_ROOT"' in runner
    assert "chmod -R a-w /dstack/input" not in runner


def test_qwen_message_contains_all_frozen_modalities_in_one_turn():
    messages = build_messages(_example(), Path("/dstack/input"))

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [item["type"] for item in content].count("image") == 12
    assert [item["type"] for item in content].count("audio") == 1
    assert [item["type"] for item in content].count("text") == 1
    assert "JSON object" in content[-1]["text"]


def test_frames_only_message_and_attestation_omit_audio():
    """Catches an audio-off request that still sends an audio modality tensor."""
    import array

    class Tensor:
        def __init__(self, values, shape):
            self.values = array.array("f", values)
            self.shape = shape

        def tobytes(self):
            return self.values.tobytes()

    content = build_messages(
        _example(), Path("/dstack/input"), audio_enabled=False
    )[0]["content"]
    frames_only_inputs = {
        "pixel_values": Tensor([1.0, 2.0], (2,)),
        "image_grid_thw": Tensor(range(36), (12, 3)),
        "input_ids": Tensor([1.0, 2.0], (1, 2)),
    }

    attestation = attest_qwen_inputs(
        frames_only_inputs,
        first_forward_input_keys=set(frames_only_inputs),
        audio_enabled=False,
    )

    assert [item["type"] for item in content].count("image") == 12
    assert [item["type"] for item in content].count("audio") == 0
    assert attestation["audio_element_count"] == 0
    assert attestation["audio_input_keys"] == []


def test_thinking_output_normalizes_only_a_terminal_json_object():
    raw = (
        "<think>I should inspect both modalities.</think>\n"
        '{"answer":"speaker raises her voice","evidence":"loud speech and gesture"}'
    )

    normalized = normalize_generation_output(raw)

    assert normalized == (
        '{"answer": "speaker raises her voice", "evidence": "loud speech and gesture"}'
    )
    assert normalize_generation_output("reasoning without a final object") == (
        "reasoning without a final object"
    )
    assert normalize_generation_output('{"answer":"x","evidence":"y"} trailing') == (
        '{"answer":"x","evidence":"y"} trailing'
    )


def test_thinking_output_normalizes_one_terminal_json_fence() -> None:
    raw = (
        "<think>Synthetic reasoning omitted.</think>\n"
        "```json\n"
        '{"answer":"a bell rings","evidence":"the final sound is a bell"}\n'
        "```"
    )

    assert normalize_generation_output(raw) == (
        '{"answer": "a bell rings", "evidence": "the final sound is a bell"}'
    )


@pytest.mark.parametrize(
    "raw",
    (
        '```json\n{"answer":"x",\n```',
        '```text\n{"answer":"x","evidence":"y"}\n```',
        '```json\n{"answer":"x","evidence":"y"}\n```\n'
        '```json\n{"answer":"x","evidence":"y"}\n```',
        '```json\n{"evidence":"y"}\n```',
        '```json\n{"answer":"x"}\n```',
        '```json\n{"answer":"x","evidence":"y","extra":true}\n```',
        '```json\n{"answer":"","evidence":"y"}\n```',
        '```json\n{"answer":"x","evidence":"y"}\n``` trailing',
        '```json\nanswer is x\n```',
    ),
)
def test_terminal_json_fence_rejections_remain_unparseable(raw: str) -> None:
    assert normalize_generation_output(raw) == raw.strip()


def test_generation_policy_has_the_approved_bounded_seeded_values():
    assert MAX_NEW_TOKENS == 2048
    assert GENERATION_SEED == 20260724


def test_generation_policy_wires_tokenizer_termination_ids():
    class Tokenizer:
        eos_token_id = EXPECTED_EOS_TOKEN_ID
        pad_token_id = EXPECTED_PAD_TOKEN_ID

        @staticmethod
        def convert_ids_to_tokens(value):
            return {
                EXPECTED_EOS_TOKEN_ID: "<|im_end|>",
                EXPECTED_PAD_TOKEN_ID: "<|endoftext|>",
            }[value]

    class Processor:
        tokenizer = Tokenizer()

    assert generation_special_token_ids(Processor()) == {
        "eos_token_id": EXPECTED_EOS_TOKEN_ID,
        "pad_token_id": EXPECTED_PAD_TOKEN_ID,
    }


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("eos_token_id", None),
        ("eos_token_id", "151645"),
        ("eos_token_id", 7),
        ("pad_token_id", None),
        ("pad_token_id", "151643"),
        ("pad_token_id", 8),
    ],
)
def test_generation_policy_rejects_missing_or_noninteger_termination_ids(
    attribute, value
):
    class Tokenizer:
        eos_token_id = EXPECTED_EOS_TOKEN_ID
        pad_token_id = EXPECTED_PAD_TOKEN_ID

        @staticmethod
        def convert_ids_to_tokens(value):
            return {
                EXPECTED_EOS_TOKEN_ID: "<|im_end|>",
                EXPECTED_PAD_TOKEN_ID: "<|endoftext|>",
            }.get(value, "<wrong>")

    class Processor:
        tokenizer = Tokenizer()

    setattr(Processor.tokenizer, attribute, value)
    with pytest.raises(ValueError, match=attribute):
        generation_special_token_ids(Processor())


class _Batch(dict):
    def to(self, *_args):
        return self


class _FakeTensor:
    def __init__(self, values, shape):
        self.values = list(values)
        self.shape = shape

    def tobytes(self):
        return json.dumps(self.values).encode()


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeResponseTokens(_FakeTensor):
    def __init__(self, values):
        super().__init__(values, (1, len(values)))

    def __getitem__(self, key):
        row, column = key
        assert row == 0
        return _FakeScalar(self.values[column])


class _FakeGenerated:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, key):
        rows, columns = key
        assert rows == slice(None)
        return _FakeResponseTokens(self.values[columns])


class _FakeProcessor:
    class Tokenizer:
        eos_token_id = EXPECTED_EOS_TOKEN_ID
        pad_token_id = EXPECTED_PAD_TOKEN_ID

        @staticmethod
        def convert_ids_to_tokens(value):
            return {
                EXPECTED_EOS_TOKEN_ID: "<|im_end|>",
                EXPECTED_PAD_TOKEN_ID: "<|endoftext|>",
            }[value]

    tokenizer = Tokenizer()

    def __init__(self):
        self.chat_template_kwargs = None

    def apply_chat_template(self, *_args, **_kwargs):
        self.chat_template_kwargs = _kwargs
        return _Batch(
            {
                "input_ids": _FakeTensor([10, 11], (1, 2)),
                "pixel_values": _FakeTensor([1.0], (1,)),
                "image_grid_thw": _FakeTensor([0] * 36, (12, 3)),
                "input_features": _FakeTensor([0.0, 0.0], (1, 2)),
                "feature_attention_mask": _FakeTensor([1, 1], (1, 2)),
            }
        )

    @staticmethod
    def batch_decode(*_args, **_kwargs):
        return [
            "<think>Inspect both modalities.</think>\n"
            '{"answer":"cat yowl","evidence":"open mouth and loud vocalization"}'
        ]


class _FakeHandle:
    @staticmethod
    def remove():
        return None


class _FakeParameter:
    device = "cuda:0"


class _FakeModel:
    dtype = "bfloat16"

    def __init__(self, final_token=EXPECTED_EOS_TOKEN_ID):
        self.final_token = final_token
        self.generate_kwargs = None
        self.forward_hook = None

    def parameters(self):
        yield _FakeParameter()

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        assert with_kwargs is True
        self.forward_hook = hook
        return _FakeHandle()

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        assert self.forward_hook is not None
        self.forward_hook(self, (), kwargs)
        return _FakeGenerated([10, 11, 20, self.final_token])


class _FakeCuda:
    @staticmethod
    def manual_seed_all(_seed):
        return None


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def manual_seed(_seed):
        return None

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


def test_generation_policy_parameter_is_keyword_only():
    """Catches accidental positional exposure of the Stage 1 sampling policy."""
    parameter = inspect.signature(_generate_one).parameters["do_sample"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True


def test_generate_one_preserves_sampled_a4_and_accepts_stage1_greedy_generation():
    """Catches a generation policy that cannot switch to Stage 1 greediness."""
    stage1_model = _FakeModel()

    output, attestation = _generate_one(
        torch=_FakeTorch(),
        processor=_FakeProcessor(),
        model=stage1_model,
        example=_example(),
        data_root=Path("/dstack/input"),
        do_sample=False,
    )

    assert stage1_model.generate_kwargs["do_sample"] is False
    assert attestation["generation"]["do_sample"] is False
    assert attestation["generation"]["enable_thinking"] is None
    assert stage1_model.generate_kwargs["eos_token_id"] == EXPECTED_EOS_TOKEN_ID
    assert stage1_model.generate_kwargs["pad_token_id"] == EXPECTED_PAD_TOKEN_ID
    assert attestation["generation"]["ended_with_eos"] is True
    assert attestation["generation"]["generated_new_token_count"] == 2
    assert attestation["generation"]["eos_token"] == "<|im_end|>"
    assert attestation["generation"]["pad_token"] == "<|endoftext|>"
    assert output == (
        '{"answer": "cat yowl", '
        '"evidence": "open mouth and loud vocalization"}'
    )

    a4_model = _FakeModel()
    _, a4_attestation = _generate_one(
        torch=_FakeTorch(),
        processor=_FakeProcessor(),
        model=a4_model,
        example=_example(),
        data_root=Path("/dstack/input"),
    )

    assert a4_model.generate_kwargs["do_sample"] is True
    assert a4_model.generate_kwargs["temperature"] == 0.6
    assert a4_model.generate_kwargs["top_p"] == 0.95
    assert a4_model.generate_kwargs["top_k"] == 20
    assert a4_attestation["generation"]["do_sample"] is True
    assert a4_attestation["generation"]["temperature"] == 0.6
    assert a4_attestation["generation"]["top_p"] == 0.95
    assert a4_attestation["generation"]["top_k"] == 20
    assert a4_attestation["generation"]["enable_thinking"] is None


def test_generate_one_rejects_unsupported_thinking_mode():
    """Catches Omni silently ignoring the generic Qwen3 thinking kwarg."""
    with pytest.raises(ValueError, match="enable_thinking is unsupported"):
        _generate_one(
            torch=_FakeTorch(),
            processor=_FakeProcessor(),
            model=_FakeModel(),
            example=_example(),
            data_root=Path("/dstack/input"),
            enable_thinking=False,
        )


@pytest.mark.parametrize("enable_thinking", (0, 1, "false"))
def test_generate_one_rejects_non_boolean_thinking_policy(enable_thinking):
    with pytest.raises(ValueError, match="enable_thinking must be a bool or None"):
        _generate_one(
            torch=_FakeTorch(),
            processor=_FakeProcessor(),
            model=_FakeModel(),
            example=_example(),
            data_root=Path("/dstack/input"),
            enable_thinking=enable_thinking,
        )


@pytest.mark.parametrize("do_sample", (0, 1, "false", None))
def test_generate_one_rejects_non_boolean_generation_policy(do_sample):
    """Catches truthy non-booleans that would silently alter sampling behavior."""
    model = _FakeModel()

    with pytest.raises(ValueError, match="do_sample must be a bool"):
        _generate_one(
            torch=_FakeTorch(),
            processor=_FakeProcessor(),
            model=model,
            example=_example(),
            data_root=Path("/dstack/input"),
            do_sample=do_sample,
        )
    assert model.generate_kwargs is None


def test_generate_one_accepts_and_attests_an_explicit_retry_seed():
    torch = _FakeTorch()
    processor = _FakeProcessor()
    model = _FakeModel()
    _, attestation = _generate_one(
        torch=torch,
        processor=processor,
        model=model,
        example=_example(),
        data_root=Path("/dstack/input"),
        do_sample=True,
        seed=42,
    )
    assert attestation["generation"]["seed"] == 42


def test_generate_one_fails_closed_without_eos_termination():
    with pytest.raises(
        ProbeStageError, match="did not terminate with EOS"
    ) as captured:
        _generate_one(
            torch=_FakeTorch(),
            processor=_FakeProcessor(),
            model=_FakeModel(final_token=42),
            example=_example(),
            data_root=Path("/dstack/input"),
        )

    record = {
        "modality_attestation": {
            "same_request": False,
            "same_forward_pass": False,
            "generation_call_count": 0,
        }
    }
    _preserve_failure_attestation(record, captured.value)
    attestation = record["modality_attestation"]
    assert attestation["same_request"] is True
    assert attestation["same_forward_pass"] is True
    assert attestation["generation_call_count"] == 1
    assert attestation["vision_element_count"] == 12
    assert attestation["audio_element_count"] == 1
    assert attestation["generation"]["ended_with_eos"] is False
    assert attestation["generation"]["generated_new_token_count"] == 2
    assert attestation["generation"]["final_generated_token_id"] == 42
    assert attestation["generation"]["eos_token_id"] == EXPECTED_EOS_TOKEN_ID
    assert attestation["generation"]["eos_token"] == "<|im_end|>"


def test_panel_identity_fails_closed_on_hash_or_video_mismatch():
    contract = {
        "examples": [
            {"video_id": "NN7G28sBOlY"},
            {"video_id": "NUmb2P3Bzu8"},
            {"video_id": "NkXPPJoZ-h8"},
        ]
    }
    validate_panel_identity(contract, CONTRACT_SHA256)

    with pytest.raises(ValueError, match="contract digest"):
        validate_panel_identity(contract, "0" * 64)

    contract["examples"][2]["video_id"] = "unexpected"
    with pytest.raises(ValueError, match="video IDs"):
        validate_panel_identity(contract, CONTRACT_SHA256)


def test_qwen_attestation_uses_cardinality_tensors_and_first_forward():
    import array

    class Tensor:
        def __init__(self, values, shape):
            self.values = array.array("f", values)
            self.shape = shape

        def tobytes(self):
            return self.values.tobytes()

    inputs = {
        "pixel_values": Tensor([1.0, 2.0], (2,)),
        "image_grid_thw": Tensor(range(36), (12, 3)),
        "input_features": Tensor([3.0, 4.0], (1, 2)),
        "feature_attention_mask": Tensor([1.0, 1.0], (1, 2)),
        "input_ids": Tensor([1.0, 2.0], (1, 2)),
    }

    attestation = attest_qwen_inputs(
        inputs,
        first_forward_input_keys=set(inputs),
    )

    assert attestation["same_request"] is True
    assert attestation["same_forward_pass"] is True
    assert attestation["generation_call_count"] == 1
    assert attestation["vision_element_count"] == 12
    assert attestation["audio_element_count"] == 1
    assert set(attestation["vision_input_keys"]) == {
        "image_grid_thw",
        "pixel_values",
    }
    assert set(attestation["audio_input_keys"]) == {
        "feature_attention_mask",
        "input_features",
    }


def test_collector_targets_only_ignored_canonical_directory_and_allowlists_metadata():
    collector = Path("scripts/collect-qwen3-omni-dstack.sh").read_text(encoding="utf-8")

    assert "data/vqa/benchmarks/qwen3-omni-compatibility-3/$RUN_ID" in collector
    assert "dstack run get" in collector
    assert "sha256sum -c SHA256SUMS" in collector
    assert "artifact-hashes.sha256" in collector
    assert "ssh_key" not in collector
    assert "--delete" not in collector


def test_dstack_metadata_uses_exact_provisioned_offer_shape():
    raw = {
        "id": "run-uuid",
        "run_spec": {"run_name": "qwen3-omni-a1"},
        "status": "terminated",
        "status_message": "terminated",
        "termination_reason": "stopped_by_user",
        "submitted_at": "2026-07-24T20:03:52.493209+00:00",
        "cost": 0.1844,
        "latest_job_submission": {
            "status": "terminated",
            "termination_reason": "terminated_by_user",
            "submitted_at": "2026-07-24T20:03:52.495154+00:00",
            "finished_at": "2026-07-24T20:06:23.738912+00:00",
            "exit_status": None,
            "job_provisioning_data": {
                "backend": "runpod",
                "region": "CA-MTL-4",
                "price": 4.39,
                "instance_type": {
                    "name": "NVIDIA H200",
                    "resources": {
                        "cpus": 28,
                        "memory_mib": 257024,
                        "gpus": [
                            {
                                "name": "H200",
                                "memory_mib": 144384,
                                "vendor": "nvidia",
                            }
                        ],
                        "disk": {"size_mib": 204800},
                        "spot": False,
                    },
                },
            },
            "job_runtime_data": {
                "offer": {
                    "backend": "runpod",
                    "region": "CA-MTL-4",
                    "price": 4.39,
                }
            },
        },
    }

    sanitized = sanitize_dstack_run(raw, run_id="qwen3-omni-a1")

    assert sanitized["backend"] == "runpod"
    assert sanitized["region"] == "CA-MTL-4"
    assert sanitized["price"] == 4.39
    assert sanitized["gpu_name"] == "H200"
    assert sanitized["gpu_vendor"] == "nvidia"
    assert sanitized["gpu_count"] == 1
    assert sanitized["cpu_count"] == 28
    assert sanitized["gpu_memory_mib"] == 144384
    assert sanitized["host_memory_mib"] == 257024
    assert sanitized["disk_mib"] == 204800
    assert sanitized["elapsed_seconds"] == pytest.approx(151.245703)
    assert set(sanitized).isdisjoint({"user", "backend_data", "public_ip"})

    incomplete = {**raw, "cost": None}
    with pytest.raises(ValueError, match="run.cost"):
        sanitize_dstack_run(incomplete, run_id="qwen3-omni-a1")

    mismatched = {
        **raw,
        "latest_job_submission": {
            **raw["latest_job_submission"],
            "job_runtime_data": {
                "offer": {
                    "backend": "runpod",
                    "region": "US-TX-3",
                    "price": 4.39,
                }
            },
        },
    }
    with pytest.raises(ValueError, match="region differs"):
        sanitize_dstack_run(mismatched, run_id="qwen3-omni-a1")
