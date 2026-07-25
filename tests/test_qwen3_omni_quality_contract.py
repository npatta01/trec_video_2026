import copy
import hashlib
import json
from pathlib import Path

import pytest

import trec_video_2026.vqa_experiments.qwen3_omni_quality_contract as quality_contract
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


@pytest.mark.parametrize("attack", ("duplicate_frame", "changed_timestamps"))
def test_quality_contract_rejects_substituted_frozen_evidence(attack: str):
    """Catches a valid local file recipe substituted for frozen request evidence."""
    contract = build_contract(Path("data"))
    example = contract["examples"][0]
    if attack == "duplicate_frame":
        example["frame_paths"][0] = example["frame_paths"][1]
        example["frame_sha256s"][0] = example["frame_sha256s"][1]
    else:
        example["frame_timestamps"][0] = example["frame_timestamps"][1]

    with pytest.raises(ValueError, match="frozen request evidence"):
        validate_quality_contract(contract, Path("data"), verify_files=True)


def test_quality_contract_rejects_non_horse_question_rewrite():
    """Catches a prompt rewrite outside the single approved horse repair."""
    contract = build_contract(Path("data"))
    contract["examples"][0]["question"] = "A substituted question?"

    with pytest.raises(ValueError, match="frozen request question"):
        validate_quality_contract(contract, Path("data"), verify_files=False)


def test_request_hash_is_checked_before_json_deserialization(monkeypatch: pytest.MonkeyPatch):
    """Catches parsing an unpinned request file before its identity is verified."""
    monkeypatch.setattr(
        quality_contract, "FROZEN12_REQUEST_SHA256", "0" * 64, raising=False
    )

    def unexpected_deserialization(_: str):
        pytest.fail("request JSON was deserialized before its SHA-256 was checked")

    monkeypatch.setattr(quality_contract.json, "loads", unexpected_deserialization)
    with pytest.raises(ValueError, match="frozen request hash"):
        build_contract(Path("data"))


def test_request_reader_rejects_forbidden_row_key_before_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catches an otherwise selected source row carrying prohibited hidden data."""
    request_path = _write_selected_source_row(
        tmp_path, monkeypatch, correct_answer="forbidden test fixture value"
    )

    with pytest.raises(ValueError, match="forbidden source row key"):
        quality_contract._read_requests(tmp_path)


def _write_selected_source_row(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> Path:
    request_path = data_root / quality_contract.FROZEN12_REQUEST_PATH
    request_path.parent.mkdir(parents=True)
    row = {
        "run_id": "test-run",
        "condition_id": "frozen12",
        "video_id": VIDEO_IDS[0],
        "question": "A harmless question?",
        "frame_descriptions": None,
        "frame_paths": [],
        "frame_timestamps": [],
        "transcript": None,
        "media_path": "/unused/media.mkv",
        "audio_path": None,
        "audio_mode": None,
    }
    row.update(overrides)
    request_path.write_text(json.dumps([row]), encoding="utf-8")
    monkeypatch.setattr(
        quality_contract,
        "FROZEN12_REQUEST_SHA256",
        hashlib.sha256(request_path.read_bytes()).hexdigest(),
        raising=False,
    )
    return request_path


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("transcript", "hidden text"),
        ("frame_descriptions", ["hidden description"]),
        ("audio_path", "/hidden/audio.wav"),
        ("audio_mode", "native"),
    ),
)
def test_request_reader_rejects_non_null_placeholder_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
):
    """Catches hidden evidence placed into a null-only frozen source field."""
    _write_selected_source_row(tmp_path, monkeypatch, **{field: value})

    with pytest.raises(ValueError, match="null placeholder"):
        quality_contract._read_requests(tmp_path)


@pytest.mark.parametrize(
    "field",
    ("answer", "prior_answer", "prediction", "reference", "distractor", "score_value"),
)
def test_request_reader_rejects_answer_and_score_like_source_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
):
    """Catches any answer, prediction, reference, or scoring value in a source row."""
    _write_selected_source_row(tmp_path, monkeypatch, **{field: "hidden value"})

    with pytest.raises(ValueError, match="forbidden source row key"):
        quality_contract._read_requests(tmp_path)


def test_safe_lexical_path_rejects_symlink_escape(tmp_path: Path):
    """Catches a safe-looking vqa path whose final symlink target leaves data."""
    data_root = tmp_path / "data"
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"not dataset evidence")
    escape_link = data_root / "vqa" / "evidence" / "escape.wav"
    escape_link.parent.mkdir(parents=True)
    escape_link.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes data_root"):
        quality_contract._resolve_relative(
            data_root, "vqa/evidence/escape.wav", "audio_path"
        )


def test_returned_conditions_cannot_mutate_validator_expectations():
    """Catches shared condition state changed through a previously returned contract."""
    contract = build_contract(Path("data"))
    contract["conditions"][0]["audio_enabled"] = True

    fresh = build_contract(Path("data"))

    assert fresh["conditions"] == [
        {"condition_id": "frames-only", "audio_enabled": False},
        {"condition_id": "frames-native-audio", "audio_enabled": True},
    ]
