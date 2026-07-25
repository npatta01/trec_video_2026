from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_gate import canonical_request_sha256
from trec_video_2026.vqa_omni_runner import (
    _error_result,
    build_validated_attempt,
    build_conversation,
    generate_text,
    load_omni,
    omni_audio_enabled,
    prepare_inputs,
    select_evaluation_attempts,
    select_pending_requests,
    timeout_cleanup_verified,
    validate_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def request(*, audio_mode: str = "audio-off") -> dict:
    return {
        "run_id": "run-1",
        "condition_id": f"omni-{audio_mode}",
        "video_id": "video-1",
        "question": "What happens?",
        "frame_paths": [],
        "frame_timestamps": [],
        "frame_descriptions": None,
        "transcript": None,
        "media_path": "/data/video.mp4",
        "audio_path": None,
        "audio_mode": audio_mode,
    }


@pytest.mark.parametrize(
    ("mode", "expected"), (("audio-off", False), ("audio-on", True))
)
def test_audio_mode_is_explicit(mode: str, expected: bool) -> None:
    assert omni_audio_enabled(request(audio_mode=mode)) is expected


def test_request_rejects_unknown_fields_so_references_cannot_leak() -> None:
    payload = request()
    payload["reference_answer"] = "a label"

    with pytest.raises(ValueError, match="unknown fields.*reference_answer"):
        validate_request(payload)


def test_audio_off_conversation_preserves_the_video_evidence() -> None:
    conversation = build_conversation(request(audio_mode="audio-off"))

    user_content = conversation[-1]["content"]
    assert user_content[0] == {"type": "video", "video": "/data/video.mp4"}
    assert user_content[1]["type"] == "text"
    assert "What happens?" in user_content[1]["text"]
    assert "transcript" not in user_content[1]["text"].lower()


class FakeBatch(dict):
    def __init__(self):
        super().__init__(input_ids=[[11, 12]])

    def to(self, value):
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.processor_calls = []

    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": False}
        return "templated prompt"

    def __call__(self, **kwargs):
        self.processor_calls.append(kwargs)
        return FakeBatch()

    def batch_decode(self, ids, **kwargs):
        assert ids == [[1, 2, 3]]
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return ["Answer: a bell rings"]


class FakeModel:
    device = "cuda:0"
    dtype = "bfloat16"

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[11, 12, 1, 2, 3]]


@pytest.mark.parametrize("enabled", (False, True))
def test_preprocessing_and_generation_receive_the_same_audio_switch(
    enabled: bool,
) -> None:
    processor = FakeProcessor()
    model = FakeModel()
    mm_calls = []

    def process_mm_info(conversation, *, use_audio_in_video):
        mm_calls.append((conversation, use_audio_in_video))
        return (["audio"] if use_audio_in_video else None, None, ["video"])

    conversation = build_conversation(
        request(audio_mode="audio-on" if enabled else "audio-off")
    )
    inputs = prepare_inputs(
        processor,
        process_mm_info,
        conversation,
        use_audio_in_video=enabled,
        model=model,
    )
    answer = generate_text(
        model,
        processor,
        inputs,
        use_audio_in_video=enabled,
        max_new_tokens=32,
    )

    assert mm_calls == [(conversation, enabled)]
    assert processor.processor_calls[0]["videos"] == ["video"]
    assert processor.processor_calls[0]["use_audio_in_video"] is enabled
    assert model.generate_kwargs["use_audio_in_video"] is enabled
    assert model.generate_kwargs["return_audio"] is False
    assert model.generate_kwargs["do_sample"] is False
    assert answer == "a bell rings"


class FakeLoadedModel:
    def eval(self):
        self.was_evaluated = True
        return self

    def disable_talker(self):
        self.talker_disabled = True


class FakeAutoModel:
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return FakeLoadedModel()


class FakeAutoProcessor:
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return "processor"


class FakeTorch:
    bfloat16 = "bfloat16"


def test_load_omni_uses_exact_local_snapshot_sdpa_bf16_and_disables_talker() -> None:
    FakeAutoModel.calls.clear()
    FakeAutoProcessor.calls.clear()
    resolver_calls = []

    def resolver(model_id, *, revision, local_files_only):
        resolver_calls.append((model_id, revision, local_files_only))
        return "/cache/exact-snapshot"

    model, processor = load_omni(
        FakeAutoModel,
        FakeAutoProcessor,
        FakeTorch,
        "Qwen/Qwen2.5-Omni-3B",
        "f" * 40,
        snapshot_resolver=resolver,
    )

    assert resolver_calls == [("Qwen/Qwen2.5-Omni-3B", "f" * 40, True)]
    assert FakeAutoModel.calls == [
        (
            "/cache/exact-snapshot",
            {
                "torch_dtype": "bfloat16",
                "device_map": "cuda:0",
                "attn_implementation": "sdpa",
                "local_files_only": True,
            },
        )
    ]
    assert FakeAutoProcessor.calls == [
        ("/cache/exact-snapshot", {"local_files_only": True})
    ]
    assert model.was_evaluated is True
    assert model.talker_disabled is True
    assert processor == "processor"


def test_model_config_contains_frozen_matched_3b_pair() -> None:
    payload = json.loads(
        (REPO_ROOT / "configs/vqa/model_runs.json").read_text(encoding="utf-8")
    )
    conditions = {
        item["condition_id"]: item
        for item in payload["conditions"]
        if item["model_id"] == "Qwen/Qwen2.5-Omni-3B"
    }

    assert set(conditions) == {
        "qwen25-omni-3b-native-video-audio-off",
        "qwen25-omni-3b-native-video-audio-on",
    }
    off = conditions["qwen25-omni-3b-native-video-audio-off"]
    on = conditions["qwen25-omni-3b-native-video-audio-on"]
    for key in (
        "model_id",
        "model_revision",
        "download_bytes",
        "runner_kind",
        "runtime_image",
    ):
        assert off[key] == on[key]
    assert off["model_revision"] == "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
    assert off["audio_mode"] == "audio-off"
    assert on["audio_mode"] == "audio-on"


def test_container_provides_pinned_ffmpeg_for_native_video_audio_decode() -> None:
    text = (
        REPO_ROOT / "containers/vqa-gate/Containerfile.qwen-omni"
    ).read_text(encoding="utf-8")

    assert '"imageio-ffmpeg==0.6.0"' in text
    assert "get_ffmpeg_exe" in text
    assert "/usr/local/bin/ffmpeg" in text


def test_empty_exception_messages_keep_actionable_repr_and_truthful_load_state() -> None:
    result = _error_result(
        type("NotInstalledError", (Exception,), {})(),
        0.0,
        args=argparse.Namespace(model_id="Qwen/model", revision="f" * 40),
        runtime="runtime",
        peak_memory=0,
        device="unverified",
        precision="unverified",
        audio_enabled=True,
        talker_disabled=False,
    )

    assert result["error_message"] == "NotInstalledError()"
    assert result["talker_disabled"] is False


def passed_result(*, audio_enabled: bool = False) -> dict:
    return {
        "status": "passed", "answer": "a bell", "elapsed_seconds": 1.0,
        "model_id": "Qwen/Qwen2.5-Omni-3B", "model_revision": "f" * 40,
        "runtime": "runtime", "precision": "bfloat16",
        "device": "cuda:0 (ROCm)", "peak_memory_bytes": 1,
        "error_type": None, "error_message": None, "native_video": True,
        "native_audio": audio_enabled, "talker_disabled": True,
        "return_audio": False, "transcript_included": False,
    }


def test_resume_skips_only_an_exact_previously_passed_request() -> None:
    first = request(audio_mode="audio-off")
    second = request(audio_mode="audio-on")
    second["condition_id"] = "omni-audio-on"
    attempt = build_validated_attempt(first, passed_result(),
                                      expected_model_revision="f" * 40)

    assert select_pending_requests(
        [first, second], [attempt], expected_model_revision="f" * 40
    ) == [second]
    altered = {**first, "question": "A changed question?"}
    assert select_pending_requests(
        [altered], [attempt], expected_model_revision="f" * 40
    ) == [altered]


@pytest.mark.parametrize(
    "mutation",
    (
        {"device": "cpu"},
        {"model_revision": "e" * 40},
        {"native_audio": True},
        {"talker_disabled": False},
        {"return_audio": True},
        {"transcript_included": True},
    ),
)
def test_resume_revalidates_saved_success_provenance(mutation: dict) -> None:
    payload = request(audio_mode="audio-off")
    attempt = build_validated_attempt(
        payload, passed_result(), expected_model_revision="f" * 40
    )
    corrupted = {**attempt, **mutation}

    assert select_pending_requests(
        [payload], [corrupted], expected_model_revision="f" * 40
    ) == [payload]


def test_attempt_validation_fails_closed_on_audio_or_accelerator_mismatch() -> None:
    payload = request(audio_mode="audio-on")
    with pytest.raises(ValueError, match="audio evidence mismatch"):
        build_validated_attempt(payload, passed_result(audio_enabled=False),
                                expected_model_revision="f" * 40)
    wrong_device = {**passed_result(audio_enabled=True), "device": "cpu"}
    with pytest.raises(ValueError, match="ROCm BF16"):
        build_validated_attempt(payload, wrong_device,
                                expected_model_revision="f" * 40)


def test_evaluation_selects_valid_pass_over_retries_and_latest_failure_otherwise() -> None:
    off = request(audio_mode="audio-off")
    on = request(audio_mode="audio-on")
    on["condition_id"] = "omni-audio-on"
    good = build_validated_attempt(
        off, passed_result(), expected_model_revision="f" * 40
    )
    corrupt = {**good, "device": "cpu"}
    old_failure = {
        **good, "status": "failed", "answer": None, "device": "unknown",
        "precision": "unknown",
    }
    latest_on_failure = {
        **old_failure, "condition_id": on["condition_id"],
        "request": on, "request_sha256": canonical_request_sha256(on),
    }

    selected = select_evaluation_attempts(
        [off, on], [old_failure, corrupt, good, latest_on_failure],
        expected_model_revision="f" * 40,
    )

    assert selected == [good, latest_on_failure]


@pytest.mark.parametrize(
    ("exists_returncode", "inspect_returncode", "running_state", "expected"),
    (
        (1, 125, "", True),
        (0, 0, "false", True),
        (0, 0, "true", False),
        (125, 125, "", False),
        (0, 125, "", False),
    ),
)
def test_timeout_cleanup_requires_container_absence_or_stopped_state(
    exists_returncode: int,
    inspect_returncode: int,
    running_state: str,
    expected: bool,
) -> None:
    assert (
        timeout_cleanup_verified(
            exists_returncode, inspect_returncode, running_state
        )
        is expected
    )
