import pytest

from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_profiles import (
    Frozen30ExperimentProfile,
    STAGE1_REPAIR_PROFILE,
    STAGE1_SAMPLED_PROFILE,
    STAGE1_SEED42_PROFILE,
    profile_for_key,
)


def test_stage1_profile_is_the_approved_calibrated_frames_only_run():
    """Catches a Stage 1 run that drifts from the approved calibration boundary."""
    assert profile_for_key("stage1") == Frozen30ExperimentProfile(
        key="stage1",
        run_id="qwen3-omni-f30-calibrated-20260801-s1",
        condition_specs=(("qwen3-omni-frames-only-calibrated-frozen30", False),),
        canary_video_id="NkXPPJoZ-h8",
        canary_condition_id="qwen3-omni-frames-only-calibrated-frozen30",
        instruction_profile="visual-concise",
        do_sample=False,
        expected_scored_rows=30,
        enable_thinking=None,
    )


def test_stage1_repair_profile_is_a_new_immutable_identity_with_thinking_disabled():
    assert STAGE1_REPAIR_PROFILE.key == "stage1_repair"
    assert STAGE1_REPAIR_PROFILE.run_id == "qwen3-omni-f30-calibrated-20260801-s2-nothink"
    assert STAGE1_REPAIR_PROFILE.do_sample is False
    assert STAGE1_REPAIR_PROFILE.enable_thinking is False


def test_stage1_sampled_profile_is_a_fresh_native_thinking_retry():
    assert STAGE1_SAMPLED_PROFILE.key == "stage1_sampled"
    assert STAGE1_SAMPLED_PROFILE.run_id == "qwen3-omni-f30-calibrated-20260801-s3-sampled"
    assert STAGE1_SAMPLED_PROFILE.do_sample is True
    assert STAGE1_SAMPLED_PROFILE.enable_thinking is None


def test_stage1_seed42_profile_is_a_sealed_sampling_retry():
    assert STAGE1_SEED42_PROFILE.key == "stage1_seed42"
    assert STAGE1_SEED42_PROFILE.run_id == "qwen3-omni-f30-calibrated-20260801-s5-seed42"
    assert STAGE1_SEED42_PROFILE.do_sample is True
    assert STAGE1_SEED42_PROFILE.generation_seed == 42


def test_a4_profile_preserves_the_historical_paired_audio_run():
    """Catches a profile registry change that silently changes the A4 baseline."""
    assert profile_for_key("a4") == Frozen30ExperimentProfile(
        key="a4",
        run_id="qwen3-omni-frozen30-audio-20260731-a4",
        condition_specs=(
            ("qwen3-omni-frames-only-frozen30", False),
            ("qwen3-omni-frames-native-audio-frozen30", True),
        ),
        canary_video_id="NkXPPJoZ-h8",
        canary_condition_id="qwen3-omni-frames-native-audio-frozen30",
        instruction_profile="neutral",
        do_sample=True,
        expected_scored_rows=60,
        enable_thinking=None,
    )


def test_profile_for_key_rejects_unknown_profile():
    """Catches an accidental fallback to a different experiment profile."""
    with pytest.raises(ValueError, match="unknown frozen30 experiment profile: unknown"):
        profile_for_key("unknown")
