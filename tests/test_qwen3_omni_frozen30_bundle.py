"""Tests for the immutable private frozen-30 evidence bundle."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

import trec_video_2026.vqa_experiments.qwen3_omni_frozen30_bundle as bundle_module
from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_bundle import (
    derive_reused_bundle_manifest,
    prepare_bundle,
    reconstruct_bundle,
)
from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_contract import (
    CANARY_VIDEO_ID,
    CONDITION_SPECS,
    FROZEN12_REQUEST_PATH,
    FROZEN12_REQUEST_SHA256,
    FROZEN30_MANIFEST_PATH,
    FROZEN30_MANIFEST_SHA256,
    NATIVE_AUDIO_DIRECTORY,
    PANEL_ID,
    RUN_ID,
    VIDEO_IDS,
    contract_sha256,
)
from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_profiles import (
    A4_PROFILE,
    STAGE1_PROFILE,
    Frozen30ExperimentProfile,
)

REUSED_ARCHIVE_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/"
    "qwen3-omni-frozen30-audio-20260731-a4/evidence.tar"
)
REUSED_ARCHIVE_SIZE = 318177280
REUSED_ARCHIVE_SHA256 = (
    "9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any], Path, bytes]:
    source_root = tmp_path / "source"
    frame_relative = Path("vqa/synthetic/frames") / f"{VIDEO_IDS[0]}-0.png"
    frame_payload = f"frame-{VIDEO_IDS[0]}-0".encode()

    examples = []
    for video_id in VIDEO_IDS:
        frame_paths = []
        frame_sha256s = []
        for index in range(12):
            relative = Path("vqa/synthetic/frames") / f"{video_id}-{index}.png"
            payload = f"frame-{video_id}-{index}".encode()
            frame = source_root / relative
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(payload)
            frame_paths.append(relative.as_posix())
            frame_sha256s.append(_sha256(payload))
        audio_relative = Path(NATIVE_AUDIO_DIRECTORY) / f"{video_id}.wav"
        audio_payload = f"audio-{video_id}".encode()
        audio = source_root / audio_relative
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(audio_payload)
        examples.append(
            {
                "video_id": video_id,
                "question": f"Synthetic question for {video_id}?",
                "frame_paths": frame_paths,
                "frame_sha256s": frame_sha256s,
                "frame_timestamps": [float(index) for index in range(12)],
                "audio_path": audio_relative.as_posix(),
                "audio_sha256": _sha256(audio_payload),
            }
        )
    contract = {
        "schema_version": 1,
        "panel_id": PANEL_ID,
        "run_id": RUN_ID,
        "conditions": [
            {"condition_id": condition_id, "audio_enabled": audio_enabled}
            for condition_id, audio_enabled in CONDITION_SPECS
        ],
        "canary": {
            "video_id": CANARY_VIDEO_ID,
            "condition_id": CONDITION_SPECS[1][0],
        },
        "source_inputs": [
            {"path": FROZEN30_MANIFEST_PATH, "sha256": FROZEN30_MANIFEST_SHA256},
            {"path": FROZEN12_REQUEST_PATH, "sha256": FROZEN12_REQUEST_SHA256},
        ],
        "examples": examples,
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(_canonical_json(contract))
    return source_root, contract_path, contract, frame_relative, frame_payload


def test_prepare_is_deterministic_and_reconstructs_exact_bytes(tmp_path: Path) -> None:
    source_root, contract_path, contract, frame_relative, frame_payload = _fixture(
        tmp_path
    )
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = prepare_bundle(contract_path, source_root, first_root)
    second = prepare_bundle(contract_path, source_root, second_root)

    assert first == second
    assert first["contract_sha256"] == contract_sha256(contract)
    assert first["member_count"] == 390
    assert (first_root / "manifest.json").read_bytes() == (
        second_root / "manifest.json"
    ).read_bytes()
    assert (first_root / "evidence.tar").read_bytes() == (
        second_root / "evidence.tar"
    ).read_bytes()
    manifest = json.loads((first_root / "manifest.json").read_bytes())
    assert [item["path"] for item in manifest["members"]] == sorted(
        item["path"] for item in manifest["members"]
    )
    with tarfile.open(first_root / "evidence.tar", mode="r:") as archive:
        assert all(member.isfile() for member in archive.getmembers())
        assert all(
            (member.mtime, member.uid, member.gid, member.uname, member.gname)
            == (0, 0, 0, "", "")
            for member in archive.getmembers()
        )

    remote_root = tmp_path / "remote"
    reconstructed = reconstruct_bundle(
        contract_path,
        remote_root,
        first_root / "manifest.json",
        first_root / "evidence.tar",
    )

    assert reconstructed == first
    assert (remote_root / frame_relative).read_bytes() == frame_payload
    for video_id in VIDEO_IDS:
        audio = remote_root / NATIVE_AUDIO_DIRECTORY / f"{video_id}.wav"
        assert audio.read_bytes() == f"audio-{video_id}".encode()


@pytest.mark.parametrize("change", ("bytes", "symlink"))
def test_prepare_rejects_changed_or_symlink_source(
    tmp_path: Path, change: str
) -> None:
    source_root, contract_path, _, frame_relative, frame_payload = _fixture(tmp_path)
    frame = source_root / frame_relative
    if change == "bytes":
        frame.write_bytes(b"changed-frame-bytes")
    else:
        backing = frame.with_name("backing.png")
        backing.write_bytes(frame_payload)
        frame.unlink()
        frame.symlink_to(backing.name)

    with pytest.raises(ValueError, match="hash|symlink"):
        prepare_bundle(contract_path, source_root, tmp_path / "bundle")


def test_reconstruct_rejects_changed_archive_before_writing(tmp_path: Path) -> None:
    source_root, contract_path, _, frame_relative, _ = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    prepare_bundle(contract_path, source_root, bundle)
    archive = bundle / "evidence.tar"
    archive.write_bytes(archive.read_bytes()[:-1] + b"X")
    remote = tmp_path / "remote"

    with pytest.raises(ValueError, match="archive SHA-256"):
        reconstruct_bundle(contract_path, remote, bundle / "manifest.json", archive)

    assert not (remote / frame_relative).exists()


def test_reconstruct_rejects_traversal_member_even_when_archive_is_rehashed(
    tmp_path: Path,
) -> None:
    source_root, contract_path, _, frame_relative, _ = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    prepare_bundle(contract_path, source_root, bundle)
    archive_path = bundle / "evidence.tar"
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("vqa/../../outside")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["archive"]["size"] = archive_path.stat().st_size
    manifest["archive"]["sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(_canonical_json(manifest))
    remote = tmp_path / "remote"

    with pytest.raises(ValueError, match="safe|member"):
        reconstruct_bundle(contract_path, remote, manifest_path, archive_path)

    assert not (tmp_path / "outside").exists()
    assert not (remote / frame_relative).exists()


def test_reconstruct_rejects_existing_destination_without_overwrite(
    tmp_path: Path,
) -> None:
    source_root, contract_path, _, frame_relative, _ = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    prepare_bundle(contract_path, source_root, bundle)
    remote = tmp_path / "remote"
    destination = remote / frame_relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"must-survive")

    with pytest.raises(ValueError, match="already exists"):
        reconstruct_bundle(
            contract_path,
            remote,
            bundle / "manifest.json",
            bundle / "evidence.tar",
        )

    assert destination.read_bytes() == b"must-survive"


def test_reconstruct_rejects_different_contract(tmp_path: Path) -> None:
    source_root, contract_path, _, frame_relative, _ = _fixture(tmp_path)
    bundle = tmp_path / "bundle"
    prepare_bundle(contract_path, source_root, bundle)
    changed = json.loads(contract_path.read_bytes())
    changed["examples"][0]["question"] += " changed"
    changed_contract = tmp_path / "changed-contract.json"
    changed_contract.write_bytes(_canonical_json(changed))
    remote = tmp_path / "remote"

    with pytest.raises(ValueError, match="contract SHA-256"):
        reconstruct_bundle(
            changed_contract,
            remote,
            bundle / "manifest.json",
            bundle / "evidence.tar",
        )

    assert not (remote / frame_relative).exists()


def _profile_contract(
    source_root: Path, profile: Frozen30ExperimentProfile
) -> dict[str, Any]:
    examples = []
    for video_id in VIDEO_IDS:
        frame_paths = []
        frame_sha256s = []
        for index in range(12):
            relative = Path("vqa/synthetic/frames") / f"{video_id}-{index}.png"
            payload = f"frame-{video_id}-{index}".encode()
            path = source_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            frame_paths.append(relative.as_posix())
            frame_sha256s.append(_sha256(payload))
        audio_relative = Path(NATIVE_AUDIO_DIRECTORY) / f"{video_id}.wav"
        audio_payload = f"audio-{video_id}".encode()
        audio = source_root / audio_relative
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(audio_payload)
        examples.append(
            {
                "video_id": video_id,
                "question": f"Synthetic question for {video_id}?",
                "frame_paths": frame_paths,
                "frame_sha256s": frame_sha256s,
                "frame_timestamps": [float(index) for index in range(12)],
                "audio_path": audio_relative.as_posix(),
                "audio_sha256": _sha256(audio_payload),
            }
        )
    return {
        "schema_version": 1,
        "panel_id": PANEL_ID,
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
            {"path": FROZEN30_MANIFEST_PATH, "sha256": FROZEN30_MANIFEST_SHA256},
            {"path": FROZEN12_REQUEST_PATH, "sha256": FROZEN12_REQUEST_SHA256},
        ],
        "examples": examples,
    }


def _reuse_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    local_archive_identity: bool = False,
) -> tuple[Path, Path, Path, Path]:
    source_root = tmp_path / "source"
    source_contract = _profile_contract(source_root, A4_PROFILE)
    source_contract_path = tmp_path / "contract-a4.json"
    source_contract_path.write_bytes(_canonical_json(source_contract))
    source_bundle = tmp_path / "bundle-a4"
    prepare_bundle(source_contract_path, source_root, source_bundle)

    source_manifest = json.loads((source_bundle / "manifest.json").read_bytes())
    if local_archive_identity:
        monkeypatch.setattr(
            bundle_module,
            "REUSED_ARCHIVE_SIZE",
            source_manifest["archive"]["size"],
        )
        monkeypatch.setattr(
            bundle_module,
            "REUSED_ARCHIVE_SHA256",
            source_manifest["archive"]["sha256"],
        )
    else:
        source_manifest["archive"] = {
            "name": "evidence.tar",
            "size": REUSED_ARCHIVE_SIZE,
            "sha256": REUSED_ARCHIVE_SHA256,
        }
    source_manifest_path = tmp_path / "source-manifest.json"
    source_manifest_path.write_bytes(_canonical_json(source_manifest))
    monkeypatch.setattr(
        bundle_module,
        "REUSED_SOURCE_MANIFEST_SHA256",
        hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
        raising=False,
    )

    stage1_contract = json.loads(json.dumps(source_contract))
    stage1_contract["run_id"] = STAGE1_PROFILE.run_id
    stage1_contract["conditions"] = [
        {"condition_id": condition_id, "audio_enabled": audio_enabled}
        for condition_id, audio_enabled in STAGE1_PROFILE.condition_specs
    ]
    stage1_contract["canary"] = {
        "video_id": STAGE1_PROFILE.canary_video_id,
        "condition_id": STAGE1_PROFILE.canary_condition_id,
    }
    stage1_contract_path = tmp_path / "contract-stage1.json"
    stage1_contract_path.write_bytes(_canonical_json(stage1_contract))
    return (
        stage1_contract_path,
        source_contract_path,
        source_manifest_path,
        source_bundle / "evidence.tar",
    )


def test_reuse_binds_stage1_to_the_exact_existing_a4_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a Stage 1 manifest that rewrites or weakens the sealed A4 archive binding."""
    contract_path, source_contract_path, source_manifest_path, _ = _reuse_fixture(
        tmp_path, monkeypatch
    )
    output_path = tmp_path / "bundle-stage1" / "manifest.json"

    manifest = derive_reused_bundle_manifest(
        contract_path=contract_path,
        source_contract_path=source_contract_path,
        source_manifest_path=source_manifest_path,
        archive_uri=REUSED_ARCHIVE_URI,
        output_path=output_path,
    )

    assert output_path.read_bytes() == _canonical_json(manifest)
    assert manifest["schema_version"] == 2
    assert manifest["archive"] == {
        "name": "evidence.tar",
        "size": 318177280,
        "sha256": "9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943",
        "uri": REUSED_ARCHIVE_URI,
    }
    assert len(manifest["members"]) == 390
    assert not (output_path.parent / "evidence.tar").exists()


@pytest.mark.parametrize(
    "archive_uri",
    (
        "https://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/qwen3-omni-frozen30-audio-20260731-a4/evidence.tar",
        "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/latest/evidence.tar",
    ),
)
def test_reuse_rejects_any_non_pinned_archive_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_uri: str
) -> None:
    """Catches a mutable or differently resolved archive URI entering the Stage 1 binding."""
    contract_path, source_contract_path, source_manifest_path, _ = _reuse_fixture(
        tmp_path, monkeypatch
    )

    with pytest.raises(ValueError, match="archive URI"):
        derive_reused_bundle_manifest(
            contract_path=contract_path,
            source_contract_path=source_contract_path,
            source_manifest_path=source_manifest_path,
            archive_uri=archive_uri,
            output_path=tmp_path / "output.json",
        )


@pytest.mark.parametrize("attack", ("changed_member", "wrong_source", "output", "noncanonical"))
def test_reuse_rejects_changed_evidence_or_untrusted_manifest_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    """Catches reuse bypassing exact A4 provenance, evidence equality, or exclusive output."""
    contract_path, source_contract_path, source_manifest_path, _ = _reuse_fixture(
        tmp_path, monkeypatch
    )
    output_path = tmp_path / "output.json"
    if attack == "changed_member":
        changed = json.loads(contract_path.read_bytes())
        changed["examples"][0]["frame_sha256s"][0] = "0" * 64
        contract_path.write_bytes(_canonical_json(changed))
    elif attack == "wrong_source":
        source_contract_path = contract_path
    elif attack == "output":
        output_path.write_bytes(b"must-survive")
    else:
        parsed = json.loads(source_manifest_path.read_bytes())
        source_manifest_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        monkeypatch.setattr(
            bundle_module,
            "REUSED_SOURCE_MANIFEST_SHA256",
            hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
            raising=False,
        )

    with pytest.raises(ValueError):
        derive_reused_bundle_manifest(
            contract_path=contract_path,
            source_contract_path=source_contract_path,
            source_manifest_path=source_manifest_path,
            archive_uri=REUSED_ARCHIVE_URI,
            output_path=output_path,
        )

    if attack == "output":
        assert output_path.read_bytes() == b"must-survive"


def test_reuse_and_reconstruct_use_the_actual_stage1_schema2_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches reconstruction validating the derived Stage 1 contract as A4."""
    contract_path, source_contract_path, source_manifest_path, archive_path = (
        _reuse_fixture(tmp_path, monkeypatch, local_archive_identity=True)
    )
    reused_manifest_path = tmp_path / "bundle-stage1" / "manifest.json"
    derived = derive_reused_bundle_manifest(
        contract_path=contract_path,
        source_contract_path=source_contract_path,
        source_manifest_path=source_manifest_path,
        archive_uri=REUSED_ARCHIVE_URI,
        output_path=reused_manifest_path,
    )

    reconstructed_root = tmp_path / "reconstructed-stage1"
    reconstructed = reconstruct_bundle(
        contract_path,
        reconstructed_root,
        reused_manifest_path,
        archive_path,
    )

    assert derived["schema_version"] == 2
    assert reconstructed["contract_sha256"] == derived["contract_sha256"]
    assert reconstructed["member_count"] == 390
    assert len([path for path in reconstructed_root.rglob("*") if path.is_file()]) == 390
    assert (
        reconstructed_root / "vqa/synthetic/frames" / f"{VIDEO_IDS[0]}-0.png"
    ).read_bytes() == f"frame-{VIDEO_IDS[0]}-0".encode()


@pytest.mark.parametrize("field", ("uri", "size", "sha256"))
def test_schema2_reader_rejects_each_changed_archive_identity_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Catches schema-2 readers accepting an archive other than the sealed A4 object."""
    contract_path, source_contract_path, source_manifest_path, _ = _reuse_fixture(
        tmp_path, monkeypatch, local_archive_identity=True
    )
    manifest_path = tmp_path / "manifest-stage1.json"
    derive_reused_bundle_manifest(
        contract_path=contract_path,
        source_contract_path=source_contract_path,
        source_manifest_path=source_manifest_path,
        archive_uri=REUSED_ARCHIVE_URI,
        output_path=manifest_path,
    )
    changed = json.loads(manifest_path.read_bytes())
    if field == "uri":
        changed["archive"][field] = "hf://buckets/attacker/changed/evidence.tar"
    elif field == "size":
        changed["archive"][field] += 1
    else:
        changed["archive"][field] = "0" * 64
    manifest_path.write_bytes(_canonical_json(changed))
    contract = json.loads(contract_path.read_bytes())

    with pytest.raises(ValueError, match="approved A4 archive"):
        bundle_module._read_manifest(manifest_path, contract)


def test_reuse_rejects_contracts_whose_evidence_collapses_to_389_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches matching A4/Stage 1 contracts collapsing a duplicate frame member."""
    contract_path, source_contract_path, source_manifest_path, _ = _reuse_fixture(
        tmp_path, monkeypatch
    )
    source_contract = json.loads(source_contract_path.read_bytes())
    stage1_contract = json.loads(contract_path.read_bytes())
    removed_path = source_contract["examples"][0]["frame_paths"][1]
    for contract in (source_contract, stage1_contract):
        contract["examples"][0]["frame_paths"][1] = contract["examples"][0][
            "frame_paths"
        ][0]
        contract["examples"][0]["frame_sha256s"][1] = contract["examples"][0][
            "frame_sha256s"
        ][0]
    source_contract_path.write_bytes(_canonical_json(source_contract))
    contract_path.write_bytes(_canonical_json(stage1_contract))
    source_manifest = json.loads(source_manifest_path.read_bytes())
    source_manifest["contract_sha256"] = contract_sha256(source_contract)
    source_manifest["members"] = [
        member
        for member in source_manifest["members"]
        if member["path"] != removed_path
    ]
    source_manifest_path.write_bytes(_canonical_json(source_manifest))
    monkeypatch.setattr(
        bundle_module,
        "REUSED_SOURCE_MANIFEST_SHA256",
        hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
        raising=False,
    )

    with pytest.raises(ValueError, match="390 unique evidence paths|390 members"):
        derive_reused_bundle_manifest(
            contract_path=contract_path,
            source_contract_path=source_contract_path,
            source_manifest_path=source_manifest_path,
            archive_uri=REUSED_ARCHIVE_URI,
            output_path=tmp_path / "output.json",
        )


def test_reuse_rejects_a_changed_source_manifest_member_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches unsealed member sizes being propagated into the Stage 1 manifest."""
    contract_path, source_contract_path, source_manifest_path, _ = _reuse_fixture(
        tmp_path, monkeypatch
    )
    source_manifest = json.loads(source_manifest_path.read_bytes())
    source_manifest["members"][0]["size"] += 1
    source_manifest_path.write_bytes(_canonical_json(source_manifest))

    with pytest.raises(ValueError, match="source manifest SHA-256"):
        derive_reused_bundle_manifest(
            contract_path=contract_path,
            source_contract_path=source_contract_path,
            source_manifest_path=source_manifest_path,
            archive_uri=REUSED_ARCHIVE_URI,
            output_path=tmp_path / "output.json",
        )


@pytest.mark.parametrize("failure_stage", ("write", "publish"))
def test_reuse_manifest_publication_failure_is_atomic_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """Catches write/publish failures poisoning the final exclusive output path."""
    contract_path, source_contract_path, source_manifest_path, _ = _reuse_fixture(
        tmp_path, monkeypatch
    )
    output_path = tmp_path / "bundle-stage1" / "manifest.json"

    with monkeypatch.context() as failure:
        if failure_stage == "write":

            def fail_write_fsync(_: int) -> None:
                raise OSError("simulated write failure")

            failure.setattr(bundle_module.os, "fsync", fail_write_fsync)
        else:

            def fail_publish(*_: object, **__: object) -> None:
                raise OSError("simulated publish failure")

            failure.setattr(bundle_module.os, "link", fail_publish)

        with pytest.raises(OSError, match=f"simulated {failure_stage} failure"):
            derive_reused_bundle_manifest(
                contract_path=contract_path,
                source_contract_path=source_contract_path,
                source_manifest_path=source_manifest_path,
                archive_uri=REUSED_ARCHIVE_URI,
                output_path=output_path,
            )

    assert not output_path.exists()
    assert list(output_path.parent.glob(f".{output_path.name}.*")) == []

    manifest = derive_reused_bundle_manifest(
        contract_path=contract_path,
        source_contract_path=source_contract_path,
        source_manifest_path=source_manifest_path,
        archive_uri=REUSED_ARCHIVE_URI,
        output_path=output_path,
    )
    assert output_path.read_bytes() == _canonical_json(manifest)
