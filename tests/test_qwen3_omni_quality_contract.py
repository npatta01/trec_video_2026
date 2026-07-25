import copy
import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.qwen3_omni_quality_contract import (
    VIDEO_IDS,
    build_contract,
    quality_contract_sha256,
    validate_quality_contract,
)


def test_quality_contract_has_ten_examples_and_two_conditions():
    """Catches a missing, reordered, or incorrectly paired panel row."""
    contract = build_contract(Path("data"))

    assert [row["video_id"] for row in contract["examples"]] == list(VIDEO_IDS)
    assert contract["conditions"] == [
        {"condition_id": "frames-only", "audio_enabled": False},
        {"condition_id": "frames-native-audio", "audio_enabled": True},
    ]
    assert all(len(row["frame_paths"]) == 12 for row in contract["examples"])


def test_quality_contract_is_label_free():
    """Catches accidental inclusion of hidden scoring material in inference input."""
    serialized = json.dumps(build_contract(Path("data"))).lower()

    assert "correct_answer" not in serialized
    assert "reference" not in serialized
    assert "distractor" not in serialized
    assert "transcript" not in serialized


def test_quality_contract_hash_is_canonical_and_changes_with_content():
    """Catches hashes that depend on mapping insertion order or omit request data."""
    contract = build_contract(Path("data"))
    reordered = json.loads(json.dumps(contract, sort_keys=True))
    changed = copy.deepcopy(contract)
    changed["examples"][0]["question"] += "?"

    assert quality_contract_sha256(contract) == quality_contract_sha256(reordered)
    assert quality_contract_sha256(contract) != quality_contract_sha256(changed)


def test_quality_contract_rejects_parent_traversal_paths():
    """Catches a validator that would let evidence escape the data root."""
    contract = build_contract(Path("data"))
    contract["examples"][0]["audio_path"] = "vqa/../../outside.wav"

    with pytest.raises(ValueError, match="safe data-root-relative path"):
        validate_quality_contract(contract, Path("data"), verify_files=False)
