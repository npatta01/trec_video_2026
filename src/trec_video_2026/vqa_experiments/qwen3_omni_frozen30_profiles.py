"""Sealed profiles for Qwen3-Omni frozen-30 experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Frozen30ExperimentProfile:
    key: str
    run_id: str
    condition_specs: tuple[tuple[str, bool], ...]
    canary_video_id: str
    canary_condition_id: str
    instruction_profile: str
    do_sample: bool
    expected_scored_rows: int
    enable_thinking: bool | None = None
    generation_seed: int = 20260724


A4_PROFILE = Frozen30ExperimentProfile(
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

STAGE1_PROFILE = Frozen30ExperimentProfile(
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

STAGE1_REPAIR_PROFILE = Frozen30ExperimentProfile(
    key="stage1_repair",
    run_id="qwen3-omni-f30-calibrated-20260801-s2-nothink",
    condition_specs=(
        ("qwen3-omni-frames-only-calibrated-frozen30", False),
    ),
    canary_video_id="NkXPPJoZ-h8",
    canary_condition_id="qwen3-omni-frames-only-calibrated-frozen30",
    instruction_profile="visual-concise",
    do_sample=False,
    expected_scored_rows=30,
    enable_thinking=False,
)

STAGE1_SAMPLED_PROFILE = Frozen30ExperimentProfile(
    key="stage1_sampled",
    run_id="qwen3-omni-f30-calibrated-20260801-s3-sampled",
    condition_specs=(
        ("qwen3-omni-frames-only-calibrated-frozen30", False),
    ),
    canary_video_id="NkXPPJoZ-h8",
    canary_condition_id="qwen3-omni-frames-only-calibrated-frozen30",
    instruction_profile="visual-concise",
    do_sample=True,
    expected_scored_rows=30,
    enable_thinking=None,
)

STAGE1_EXPLICIT_SAMPLING_PROFILE = Frozen30ExperimentProfile(
    key="stage1_explicit_sampling",
    run_id="qwen3-omni-f30-calibrated-20260801-s4-explicit-sampling",
    condition_specs=(
        ("qwen3-omni-frames-only-calibrated-frozen30", False),
    ),
    canary_video_id="NkXPPJoZ-h8",
    canary_condition_id="qwen3-omni-frames-only-calibrated-frozen30",
    instruction_profile="visual-concise",
    do_sample=True,
    expected_scored_rows=30,
    enable_thinking=None,
)

STAGE1_SEED42_PROFILE = Frozen30ExperimentProfile(
    key="stage1_seed42",
    run_id="qwen3-omni-f30-calibrated-20260801-s5-seed42",
    condition_specs=(
        ("qwen3-omni-frames-only-calibrated-frozen30", False),
    ),
    canary_video_id="NkXPPJoZ-h8",
    canary_condition_id="qwen3-omni-frames-only-calibrated-frozen30",
    instruction_profile="visual-concise",
    do_sample=True,
    expected_scored_rows=30,
    enable_thinking=None,
    generation_seed=42,
)

PROFILES = {
    "a4": A4_PROFILE,
    "stage1": STAGE1_PROFILE,
    "stage1_repair": STAGE1_REPAIR_PROFILE,
    "stage1_sampled": STAGE1_SAMPLED_PROFILE,
    "stage1_explicit_sampling": STAGE1_EXPLICIT_SAMPLING_PROFILE,
    "stage1_seed42": STAGE1_SEED42_PROFILE,
}


def profile_for_key(key: str) -> Frozen30ExperimentProfile:
    try:
        return PROFILES[key]
    except KeyError as error:
        raise ValueError(f"unknown frozen30 experiment profile: {key}") from error
