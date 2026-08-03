"""Tests for the label-free Qwen3-Omni frozen-30 input contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import trec_video_2026.vqa_experiments.qwen3_omni_frozen30_contract as frozen30
from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_contract import (
    VIDEO_IDS,
    build_contract,
    contract_sha256,
    validate_contract,
    validate_uploaded_contract,
)
from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_profiles import (
    A4_PROFILE,
    STAGE1_PROFILE,
)

EXPECTED_IDS = (
    "LKiaCf1QZFw",
    "Ie-KCtcZ7uc",
    "J1CS47pBX_M",
    "Mi-NpnCBegw",
    "NN7G28sBOlY",
    "NUmb2P3Bzu8",
    "IRTTtnSRTyY",
    "IWSPAfCKtV8",
    "Jc_vZjKwtvg",
    "MdMKHhFFqJQ",
    "NkXPPJoZ-h8",
    "OzNdzgQi1CA",
    "CH8Dp7EgW4I",
    "EtNOkUJd5w8",
    "Gp4LVTn3YbY",
    "KgOvCPOX-Ms",
    "PcDuz7j9LL8",
    "C4j3Pp79KSg",
    "Hw0bRvn1ZIo",
    "J-sG9vePDmI",
    "Jc2BdtyAF48",
    "HyOKrz1C4zU",
    "Msbpqv5aOxk",
    "JOClNGH6yL8",
    "EMI_anpHII4",
    "IGL7j9-JRG4",
    "JSOh_pKFq84",
    "QFljXc4WpFA",
    "PiIrinu42DQ",
    "LusRurRrmf0",
)


def test_contract_has_exact_order_conditions_and_canary() -> None:
    """Catches a missing, reordered, or incorrectly paired frozen panel row."""
    contract = build_contract(Path("data"))

    assert VIDEO_IDS == EXPECTED_IDS
    assert tuple(row["video_id"] for row in contract["examples"]) == EXPECTED_IDS
    assert contract["conditions"] == [
        {"condition_id": "qwen3-omni-frames-only-frozen30", "audio_enabled": False},
        {
            "condition_id": "qwen3-omni-frames-native-audio-frozen30",
            "audio_enabled": True,
        },
    ]
    assert contract["canary"] == {
        "video_id": "NkXPPJoZ-h8",
        "condition_id": "qwen3-omni-frames-native-audio-frozen30",
    }
    assert all(len(row["frame_paths"]) == 12 for row in contract["examples"])
    assert all("prompt_repair" not in row for row in contract["examples"])


def test_contract_is_canonical_label_free_and_hash_stable() -> None:
    """Catches a digest that changes with mapping order or labels leaking to inference."""
    contract = build_contract(Path("data"))
    reordered = json.loads(json.dumps(contract, sort_keys=True))
    changed = copy.deepcopy(contract)
    changed["examples"][0]["question"] += "?"
    serialized = json.dumps(contract).lower()

    assert (
        contract_sha256(contract)
        == "9cfa26eacecde5bd8c916823ce2a59e74ded7f71ae978b973e306a0525b24bb5"
    )
    assert contract_sha256(contract) == contract_sha256(reordered)
    assert contract_sha256(contract) != contract_sha256(changed)
    for forbidden in (
        "correct_answer",
        "reference",
        "distractor",
        "transcript",
        "category",
        "annotation_path",
        "media_path",
    ):
        assert forbidden not in serialized


def test_contract_hash_rejects_non_finite_json_numbers() -> None:
    """Catches canonical serialization emitting non-standard NaN/Infinity tokens."""
    contract = build_contract(Path("data"))
    contract["examples"][0]["frame_timestamps"][0] = float("nan")

    with pytest.raises(ValueError, match="JSON compliant"):
        contract_sha256(contract)


@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), float("-inf")))
def test_uploaded_validator_rejects_non_real_or_finite_timestamps(
    invalid: object,
) -> None:
    """Catches booleans and non-finite floats entering uploaded frame timing."""
    contract = build_contract(Path("data"))
    contract["examples"][0]["frame_timestamps"][0] = invalid

    with pytest.raises(ValueError, match="finite numbers"):
        validate_uploaded_contract(contract, Path("data"), verify_files=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", True, "schema_version"),
        ("conditions.0.audio_enabled", 0, "audio_enabled"),
    ),
)
def test_validator_rejects_type_coercible_identity_primitives(
    field: str, value: object, message: str
) -> None:
    """Catches Python equality treating JSON booleans and integers as identical."""
    contract = build_contract(Path("data"))
    if field == "schema_version":
        contract[field] = value
    else:
        contract["conditions"][0]["audio_enabled"] = value

    with pytest.raises(ValueError, match=message):
        validate_uploaded_contract(contract, Path("data"), verify_files=False)


def test_builder_hashes_sources_before_deserialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches parsing either unpinned source before its raw bytes are verified."""
    monkeypatch.setattr(frozen30, "FROZEN30_MANIFEST_SHA256", "0" * 64)

    def unexpected_deserialization(_: str | bytes) -> object:
        pytest.fail("source JSON was deserialized before its SHA-256 was checked")

    monkeypatch.setattr(frozen30.json, "loads", unexpected_deserialization)
    with pytest.raises(ValueError, match="manifest hash"):
        build_contract(Path("data"))


@pytest.mark.parametrize("attack", ("question", "timestamps", "path"))
def test_local_validator_rejects_changed_frozen_request_evidence(attack: str) -> None:
    """Catches a question, timestamp, or frame substituted after construction."""
    contract = build_contract(Path("data"))
    row = contract["examples"][0]
    if attack == "question":
        row["question"] = "A substituted question?"
    elif attack == "timestamps":
        row["frame_timestamps"][0] = row["frame_timestamps"][1]
    else:
        row["frame_paths"][0] = row["frame_paths"][1]
        row["frame_sha256s"][0] = row["frame_sha256s"][1]

    with pytest.raises(ValueError, match="frozen request"):
        validate_contract(contract, Path("data"), verify_files=False)


def test_uploaded_validator_rejects_swapped_audio() -> None:
    """Catches a valid WAV swapped between two uploaded frozen-panel rows."""
    contract = build_contract(Path("data"))
    first, second = contract["examples"][:2]
    first["audio_path"], second["audio_path"] = (
        second["audio_path"],
        first["audio_path"],
    )
    first["audio_sha256"], second["audio_sha256"] = (
        second["audio_sha256"],
        first["audio_sha256"],
    )

    with pytest.raises(ValueError, match="audio_path does not match"):
        validate_uploaded_contract(contract, Path("data"), verify_files=False)


def test_validator_rejects_parent_traversal() -> None:
    """Catches a contract path that tries to escape the evidence root lexically."""
    contract = build_contract(Path("data"))
    contract["examples"][0]["audio_path"] = "vqa/../../outside.wav"

    with pytest.raises(ValueError, match="safe data-root-relative path"):
        validate_uploaded_contract(contract, Path("data"), verify_files=False)


def test_validator_rejects_symlink_escape(tmp_path: Path) -> None:
    """Catches a safe-looking evidence path whose symlink target leaves the root."""
    data_root = tmp_path / "data"
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"not dataset evidence")
    escaped = data_root / "vqa" / "evidence" / "escape.wav"
    escaped.parent.mkdir(parents=True)
    escaped.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes data_root"):
        frozen30._resolve_relative(data_root, "vqa/evidence/escape.wav", "audio_path")


def _write_request_fixture(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]
) -> None:
    if len(rows) == 1:
        rows = [rows[0]] + [_request_row(video_id) for video_id in EXPECTED_IDS[1:]]
    request_path = data_root / frozen30.FROZEN12_REQUEST_PATH
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(
        frozen30,
        "FROZEN12_REQUEST_SHA256",
        hashlib.sha256(request_path.read_bytes()).hexdigest(),
    )


def _write_manifest_fixture(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = data_root / frozen30.FROZEN30_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "examples": [
                    {"video_id": video_id, "question": "A harmless question?"}
                    for video_id in EXPECTED_IDS
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        frozen30,
        "FROZEN30_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )


def _request_row(video_id: str) -> dict[str, object]:
    return {
        "run_id": "native-audio-benchmark-v2",
        "condition_id": "qwen35-4b-12f-frozen30",
        "video_id": video_id,
        "question": "A harmless question?",
        "frame_paths": [],
        "frame_timestamps": [],
        "frame_descriptions": None,
        "transcript": None,
        "media_path": "/non-null/source.mkv",
        "audio_path": None,
        "audio_mode": None,
    }


@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), float("-inf")))
def test_local_recipe_rejects_non_real_or_finite_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: object
) -> None:
    """Catches invalid frame timing in the pinned local request source."""
    row = _request_row(EXPECTED_IDS[0])
    row["frame_paths"] = ["not-reached"] * 12
    row["frame_timestamps"] = [invalid] + [0.0] * 11
    _write_request_fixture(tmp_path, monkeypatch, [row])
    _write_manifest_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="finite numbers"):
        frozen30._frozen_recipes(tmp_path)


def test_request_reader_rejects_duplicate_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches duplicate source rows before they can collapse into one mapping entry."""
    _write_request_fixture(
        tmp_path,
        monkeypatch,
        [_request_row(EXPECTED_IDS[0]), _request_row(EXPECTED_IDS[0])]
        + [_request_row(video_id) for video_id in EXPECTED_IDS[2:]],
    )

    with pytest.raises(ValueError, match="duplicate frozen request"):
        frozen30._read_requests(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("transcript", "hidden text"),
        ("frame_descriptions", ["hidden description"]),
        ("audio_path", "/hidden/audio.wav"),
        ("audio_mode", "native"),
    ),
)
def test_request_reader_rejects_non_null_hidden_source_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    """Catches non-null content in a frozen request's null-only source fields."""
    row = _request_row(EXPECTED_IDS[0])
    row[field] = value
    _write_request_fixture(tmp_path, monkeypatch, [row])

    with pytest.raises(ValueError, match="null placeholder"):
        frozen30._read_requests(tmp_path)


def test_recursive_forbidden_key_rejection() -> None:
    """Catches labels hidden anywhere inside an otherwise well-formed contract."""
    contract = build_contract(Path("data"))
    contract["examples"][0]["nested"] = {"Reference": "hidden label"}

    with pytest.raises(ValueError, match="label-bearing field"):
        validate_uploaded_contract(contract, Path("data"), verify_files=False)


def test_uploaded_validator_does_not_read_build_time_sources(tmp_path: Path) -> None:
    """Catches remote validation accidentally depending on unuploaded source recipes."""
    contract = build_contract(Path("data"))

    validate_uploaded_contract(contract, tmp_path, verify_files=False)


def test_uploaded_validator_rejects_duplicate_evidence_paths() -> None:
    """Catches 390 sealed members collapsing through a repeated frame path/hash."""
    contract = build_contract(Path("data"), profile=STAGE1_PROFILE)
    contract["examples"][0]["frame_paths"][1] = contract["examples"][0][
        "frame_paths"
    ][0]
    contract["examples"][0]["frame_sha256s"][1] = contract["examples"][0][
        "frame_sha256s"
    ][0]

    with pytest.raises(ValueError, match="390 unique evidence paths"):
        validate_uploaded_contract(
            contract,
            Path("data"),
            verify_files=False,
            profile=STAGE1_PROFILE,
        )


def test_stage1_contract_preserves_the_sealed_frames_and_audio_evidence() -> None:
    """Catches Stage 1 inheriting the historical A4 identity or dropping audio evidence."""
    contract = build_contract(Path("data"), profile=STAGE1_PROFILE)
    serialized = json.dumps(contract).lower()

    assert contract["run_id"] == "qwen3-omni-f30-calibrated-20260801-s1"
    assert contract["conditions"] == [
        {
            "condition_id": "qwen3-omni-frames-only-calibrated-frozen30",
            "audio_enabled": False,
        }
    ]
    assert contract["canary"] == {
        "video_id": "NkXPPJoZ-h8",
        "condition_id": "qwen3-omni-frames-only-calibrated-frozen30",
    }
    assert tuple(row["video_id"] for row in contract["examples"]) == EXPECTED_IDS
    assert sum(len(row["frame_paths"]) for row in contract["examples"]) == 360
    assert len({row["audio_path"] for row in contract["examples"]}) == 30
    for forbidden in frozen30.FORBIDDEN_FIELD_NAMES:
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "mutation",
    (
        "run_id",
        "condition_id",
        "audio_enabled",
        "canary_condition_id",
        "profile",
    ),
)
def test_stage1_uploaded_validation_rejects_each_identity_mutation(
    mutation: str,
) -> None:
    """Catches uploaded Stage 1 evidence being accepted under a different sealed identity."""
    contract = build_contract(Path("data"), profile=STAGE1_PROFILE)
    profile = STAGE1_PROFILE
    if mutation == "run_id":
        contract["run_id"] = A4_PROFILE.run_id
    elif mutation == "condition_id":
        contract["conditions"][0]["condition_id"] = A4_PROFILE.condition_specs[0][0]
    elif mutation == "audio_enabled":
        contract["conditions"][0]["audio_enabled"] = True
    elif mutation == "canary_condition_id":
        contract["canary"]["condition_id"] = A4_PROFILE.canary_condition_id
    else:
        profile = A4_PROFILE

    with pytest.raises(ValueError):
        validate_uploaded_contract(
            contract, Path("data"), verify_files=False, profile=profile
        )


def test_default_a4_contract_canonical_bytes_and_digest_are_unchanged() -> None:
    """Catches profile parameterization altering the historical A4 contract bytes."""
    default_contract = build_contract(Path("data"))
    explicit_a4_contract = build_contract(Path("data"), profile=A4_PROFILE)

    expected_sha256 = "9cfa26eacecde5bd8c916823ce2a59e74ded7f71ae978b973e306a0525b24bb5"
    assert frozen30._canonical_json(default_contract) == frozen30._canonical_json(
        explicit_a4_contract
    )
    assert contract_sha256(default_contract) == expected_sha256
