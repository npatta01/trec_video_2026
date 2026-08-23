"""Deterministic contract-bound private bundle for frozen-30 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_contract import (
    contract_sha256,
    validate_uploaded_contract,
)
from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_profiles import (
    A4_PROFILE,
    STAGE1_PROFILE,
    STAGE1_REPAIR_PROFILE,
    STAGE1_EXPLICIT_SAMPLING_PROFILE,
    STAGE1_SEED42_PROFILE,
    STAGE1_SAMPLED_PROFILE,
)

SCHEMA_VERSION = 1
REUSED_SCHEMA_VERSION = 2
ARCHIVE_NAME = "evidence.tar"
MANIFEST_NAME = "manifest.json"
HEX_DIGITS = frozenset("0123456789abcdef")
REUSED_ARCHIVE_URI = (
    "hf://buckets/Npatta01/trec_video_2026/inputs/"
    "qwen3-omni-frozen30-audio-20260731-a4/evidence.tar"
)
REUSED_ARCHIVE_SIZE = 318177280
REUSED_ARCHIVE_SHA256 = (
    "9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943"
)
REUSED_SOURCE_MANIFEST_SHA256 = (
    "3f88bb004c9d95c433e544ddf0cf6b314deffd0d5a22e5adb2d3e1d09613ca74"
)
EXPECTED_MEMBER_COUNT = 390


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and set(value).issubset(HEX_DIGITS)


def _read_contract(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("contract must be a regular non-symlink file")
    parsed = json.loads(path.read_bytes())
    if not isinstance(parsed, Mapping):
        raise ValueError("contract must be a JSON object")
    return parsed


def _safe_relative(value: Any, field: str = "member path") -> PurePosixPath:
    if type(value) is not str:
        raise ValueError(f"{field} must be a safe data-root-relative path")
    relative = PurePosixPath(value)
    if (
        not value
        or relative.is_absolute()
        or relative.as_posix() != value
        or ".." in relative.parts
        or relative.parts[0] != "vqa"
    ):
        raise ValueError(f"{field} must be a safe data-root-relative path")
    return relative


def _safe_data_path(data_root: Path, value: Any, field: str = "member path") -> Path:
    relative = _safe_relative(value, field)
    root = data_root.resolve()
    candidate = root.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes data_root") from error
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{field} must not use a symlink: {value}")
        if not cursor.exists():
            break
    return candidate


def _evidence(contract: Mapping[str, Any]) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for row in contract["examples"]:
        pairs = list(zip(row["frame_paths"], row["frame_sha256s"], strict=True))
        pairs.append((row["audio_path"], row["audio_sha256"]))
        for relative, digest in pairs:
            _safe_relative(relative, "evidence path")
            if relative in evidence:
                raise ValueError(f"contract has a duplicate evidence path: {relative}")
            evidence[relative] = digest
    if len(evidence) != EXPECTED_MEMBER_COUNT:
        raise ValueError(f"contract must name exactly {EXPECTED_MEMBER_COUNT} members")
    return evidence


def _source_members(
    contract: Mapping[str, Any], data_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    members: list[dict[str, Any]] = []
    sources: dict[str, Path] = {}
    for relative, expected_sha256 in sorted(_evidence(contract).items()):
        source = _safe_data_path(data_root, relative, "evidence path")
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"evidence path is missing or is a symlink: {relative}")
        if _sha256_file(source) != expected_sha256:
            raise ValueError(f"evidence hash does not match: {relative}")
        members.append(
            {
                "path": relative,
                "sha256": expected_sha256,
                "size": source.stat().st_size,
            }
        )
        sources[relative] = source
    return members, sources


def _write_archive(
    archive_path: Path,
    members: list[dict[str, Any]],
    sources: Mapping[str, Path],
) -> None:
    with archive_path.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for member in members:
                info = tarfile.TarInfo(member["path"])
                info.size = member["size"]
                info.mode = 0o444
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with sources[member["path"]].open("rb") as source:
                    archive.addfile(info, source)
        raw.flush()
        os.fsync(raw.fileno())


def _validate_members(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != EXPECTED_MEMBER_COUNT:
        raise ValueError(
            f"bundle members must contain exactly {EXPECTED_MEMBER_COUNT} members"
        )
    members: list[Mapping[str, Any]] = []
    previous: str | None = None
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "size"}:
            raise ValueError("bundle member has an invalid schema")
        relative = _safe_relative(item["path"]).as_posix()
        if previous is not None and relative <= previous:
            raise ValueError("bundle member paths must be unique and sorted")
        previous = relative
        if not _is_sha256(item["sha256"]):
            raise ValueError("bundle member SHA-256 is invalid")
        if type(item["size"]) is not int or item["size"] < 0:
            raise ValueError("bundle member size is invalid")
        members.append(item)
    return members


def _validate_manifest(
    manifest: Any, contract: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_version",
        "contract_sha256",
        "archive",
        "members",
    }:
        raise ValueError("bundle manifest has an invalid schema")
    schema_version = manifest["schema_version"]
    if type(schema_version) is not int or schema_version not in {
        SCHEMA_VERSION,
        REUSED_SCHEMA_VERSION,
    }:
        raise ValueError("bundle manifest schema_version must be the integer 1 or 2")
    if manifest["contract_sha256"] != contract_sha256(contract):
        raise ValueError("bundle manifest contract SHA-256 does not match")
    archive = manifest["archive"]
    archive_keys = {"name", "sha256", "size"}
    if schema_version == REUSED_SCHEMA_VERSION:
        archive_keys.add("uri")
    if (
        not isinstance(archive, Mapping)
        or set(archive) != archive_keys
        or archive["name"] != ARCHIVE_NAME
        or not _is_sha256(archive["sha256"])
        or type(archive["size"]) is not int
        or archive["size"] <= 0
        or (
            schema_version == REUSED_SCHEMA_VERSION
            and (
                type(archive["uri"]) is not str
                or not archive["uri"].startswith("hf://buckets/")
                or "/latest/" in archive["uri"]
                or not archive["uri"].endswith(f"/{ARCHIVE_NAME}")
            )
        )
    ):
        raise ValueError("bundle archive metadata is invalid")
    if schema_version == REUSED_SCHEMA_VERSION and archive != {
        "name": ARCHIVE_NAME,
        "size": REUSED_ARCHIVE_SIZE,
        "sha256": REUSED_ARCHIVE_SHA256,
        "uri": REUSED_ARCHIVE_URI,
    }:
        raise ValueError("schema-2 archive metadata must match the approved A4 archive")
    members = _validate_members(manifest["members"])
    expected = _evidence(contract)
    observed = {item["path"]: item["sha256"] for item in members}
    if observed != expected:
        raise ValueError("bundle members do not match contract evidence")
    return members


def _read_manifest(
    path: Path,
    contract: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("bundle manifest must be a regular non-symlink file")
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("source manifest SHA-256 does not match the sealed A4 manifest")
    parsed = json.loads(raw)
    _validate_manifest(parsed, contract)
    if raw != _canonical_json(parsed):
        raise ValueError("bundle manifest is not canonical JSON")
    return parsed


def _verify_archive(
    archive_path: Path,
    manifest: Mapping[str, Any],
    members: list[Mapping[str, Any]],
) -> None:
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("bundle archive must be a regular non-symlink file")
    archive_spec = manifest["archive"]
    if archive_path.stat().st_size != archive_spec["size"]:
        raise ValueError("bundle archive size does not match")
    if _sha256_file(archive_path) != archive_spec["sha256"]:
        raise ValueError("bundle archive SHA-256 does not match")
    expected_paths = [item["path"] for item in members]
    with tarfile.open(archive_path, mode="r:") as archive:
        actual = archive.getmembers()
        if [item.name for item in actual] != expected_paths:
            raise ValueError("bundle archive member set or order does not match")
        for info, expected in zip(actual, members, strict=True):
            _safe_relative(info.name)
            if not info.isfile() or info.size != expected["size"]:
                raise ValueError("bundle archive member type or size does not match")
            extracted = archive.extractfile(info)
            if extracted is None or _sha256_stream(extracted) != expected["sha256"]:
                raise ValueError("bundle archive member SHA-256 does not match")


def _attestation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    members = manifest["members"]
    return {
        "archive_sha256": manifest["archive"]["sha256"],
        "archive_size": manifest["archive"]["size"],
        "contract_sha256": manifest["contract_sha256"],
        "manifest_sha256": hashlib.sha256(_canonical_json(manifest)).hexdigest(),
        "member_count": len(members),
        "png_count": sum(item["path"].endswith(".png") for item in members),
        "wav_count": sum(item["path"].endswith(".wav") for item in members),
    }


def prepare_bundle(
    contract_path: Path, data_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Create or verify one deterministic bundle of exact contract evidence."""
    contract = _read_contract(contract_path)
    data_root = data_root.resolve()
    validate_uploaded_contract(contract, data_root, verify_files=True)
    members, sources = _source_members(contract, data_root)
    if output_dir.is_symlink():
        raise ValueError("bundle output must not be a symlink")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("bundle output must be a directory")
        manifest = _read_manifest(output_dir / MANIFEST_NAME, contract)
        if manifest["members"] != members:
            raise ValueError("existing bundle members do not match exact source bytes")
        _verify_archive(output_dir / ARCHIVE_NAME, manifest, list(manifest["members"]))
        return _attestation(manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        archive_path = temporary / ARCHIVE_NAME
        _write_archive(archive_path, members, sources)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": contract_sha256(contract),
            "archive": {
                "name": ARCHIVE_NAME,
                "sha256": _sha256_file(archive_path),
                "size": archive_path.stat().st_size,
            },
            "members": members,
        }
        (temporary / MANIFEST_NAME).write_bytes(_canonical_json(manifest))
        _read_manifest(temporary / MANIFEST_NAME, contract)
        _verify_archive(archive_path, manifest, members)
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _attestation(manifest)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_canonical_exclusive(output_path: Path, payload: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise ValueError(
                f"bundle manifest output already exists: {output_path}"
            ) from error
        published = True
        temporary.unlink()
        _fsync_directory(output_path.parent)
    except BaseException:
        if published:
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
            try:
                _fsync_directory(output_path.parent)
            except OSError:
                pass
        raise
    finally:
        if descriptor != -1:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def derive_reused_bundle_manifest(
    *,
    contract_path: Path,
    source_contract_path: Path,
    source_manifest_path: Path,
    archive_uri: str,
    output_path: Path,
) -> dict[str, Any]:
    """Bind Stage 1 evidence to the immutable archive already sealed for A4."""
    if archive_uri != REUSED_ARCHIVE_URI:
        raise ValueError("archive URI must match the approved immutable A4 archive")
    source_contract = _read_contract(source_contract_path)
    validate_uploaded_contract(
        source_contract,
        source_contract_path.parent,
        verify_files=False,
        profile=A4_PROFILE,
    )
    source_manifest = _read_manifest(
        source_manifest_path,
        source_contract,
        expected_sha256=REUSED_SOURCE_MANIFEST_SHA256,
    )
    if source_manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("source bundle manifest must use schema version 1")
    if source_manifest["archive"] != {
        "name": ARCHIVE_NAME,
        "size": REUSED_ARCHIVE_SIZE,
        "sha256": REUSED_ARCHIVE_SHA256,
    }:
        raise ValueError("source bundle archive does not match the approved A4 archive")

    contract = _read_contract(contract_path)
    stage1_profile = {
        STAGE1_PROFILE.run_id: STAGE1_PROFILE,
        STAGE1_REPAIR_PROFILE.run_id: STAGE1_REPAIR_PROFILE,
        STAGE1_SAMPLED_PROFILE.run_id: STAGE1_SAMPLED_PROFILE,
        STAGE1_EXPLICIT_SAMPLING_PROFILE.run_id: STAGE1_EXPLICIT_SAMPLING_PROFILE,
        STAGE1_SEED42_PROFILE.run_id: STAGE1_SEED42_PROFILE,
    }.get(contract.get("run_id"))
    if stage1_profile is None:
        raise ValueError("contract run_id does not match a Stage 1 profile")
    validate_uploaded_contract(
        contract,
        contract_path.parent,
        verify_files=False,
        profile=stage1_profile,
    )
    if _evidence(contract) != _evidence(source_contract):
        raise ValueError("Stage 1 evidence does not match the sealed A4 member set")

    manifest: dict[str, Any] = {
        "schema_version": REUSED_SCHEMA_VERSION,
        "contract_sha256": contract_sha256(contract),
        "archive": {
            "name": ARCHIVE_NAME,
            "size": REUSED_ARCHIVE_SIZE,
            "sha256": REUSED_ARCHIVE_SHA256,
            "uri": archive_uri,
        },
        "members": [dict(member) for member in source_manifest["members"]],
    }
    _validate_manifest(manifest, contract)
    _publish_canonical_exclusive(output_path, _canonical_json(manifest))
    return manifest


def reconstruct_bundle(
    contract_path: Path,
    data_root: Path,
    manifest_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    """Verify a complete bundle, then publish its evidence without overwriting."""
    contract = _read_contract(contract_path)
    data_root = data_root.resolve()
    if contract.get("run_id") == A4_PROFILE.run_id:
        profile = A4_PROFILE
        expected_schema_version = SCHEMA_VERSION
    elif contract.get("run_id") in {
        STAGE1_PROFILE.run_id,
        STAGE1_REPAIR_PROFILE.run_id,
        STAGE1_SAMPLED_PROFILE.run_id,
        STAGE1_EXPLICIT_SAMPLING_PROFILE.run_id,
        STAGE1_SEED42_PROFILE.run_id,
    }:
        profile = (
            STAGE1_PROFILE
            if contract.get("run_id") == STAGE1_PROFILE.run_id
            else STAGE1_REPAIR_PROFILE
            if contract.get("run_id") == STAGE1_REPAIR_PROFILE.run_id
            else STAGE1_SAMPLED_PROFILE
            if contract.get("run_id") == STAGE1_SAMPLED_PROFILE.run_id
            else STAGE1_EXPLICIT_SAMPLING_PROFILE
            if contract.get("run_id") == STAGE1_EXPLICIT_SAMPLING_PROFILE.run_id
            else STAGE1_SEED42_PROFILE
        )
        expected_schema_version = REUSED_SCHEMA_VERSION
    else:
        raise ValueError("contract run_id does not match a reconstructable profile")
    validate_uploaded_contract(
        contract,
        data_root,
        verify_files=False,
        profile=profile,
    )
    manifest = _read_manifest(manifest_path, contract)
    if manifest["schema_version"] != expected_schema_version:
        raise ValueError("bundle manifest schema does not match the contract profile")
    members = list(manifest["members"])
    _verify_archive(archive_path, manifest, members)

    destinations: list[tuple[Mapping[str, Any], Path, Path]] = []
    for member in members:
        target = _safe_data_path(data_root, member["path"])
        temporary = target.with_name(f".{target.name}.reconstructing")
        if target.exists() or target.is_symlink():
            raise ValueError(f"bundle destination already exists: {member['path']}")
        if temporary.exists() or temporary.is_symlink():
            raise ValueError(f"bundle temporary already exists: {temporary}")
        destinations.append((member, target, temporary))

    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            archive_by_name = {item.name: item for item in archive.getmembers()}
            for member, target, temporary in destinations:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(archive_by_name[member["path"]])
                if source is None:
                    raise ValueError("bundle archive member is unavailable")
                with temporary.open("xb") as output:
                    shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                if (
                    temporary.stat().st_size != member["size"]
                    or _sha256_file(temporary) != member["sha256"]
                ):
                    raise ValueError("reconstructed bundle member does not match")
        for member, target, temporary in destinations:
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise ValueError(
                    f"bundle destination already exists: {member['path']}"
                ) from error
            if _sha256_file(target) != member["sha256"]:
                raise ValueError("published bundle member does not match")
    finally:
        for _, _, temporary in destinations:
            if temporary.exists():
                temporary.unlink()
    return _attestation(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--contract", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    reconstruct = commands.add_parser("reconstruct")
    reconstruct.add_argument("--contract", type=Path, required=True)
    reconstruct.add_argument("--data-root", type=Path, required=True)
    reconstruct.add_argument("--manifest", type=Path, required=True)
    reconstruct.add_argument("--archive", type=Path, required=True)
    reuse = commands.add_parser("reuse")
    reuse.add_argument("--contract", type=Path, required=True)
    reuse.add_argument("--source-contract", type=Path, required=True)
    reuse.add_argument("--source-manifest", type=Path, required=True)
    reuse.add_argument("--archive-uri", required=True)
    reuse.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        attestation = prepare_bundle(args.contract, args.data_root, args.output_dir)
    elif args.command == "reconstruct":
        attestation = reconstruct_bundle(
            args.contract, args.data_root, args.manifest, args.archive
        )
    else:
        attestation = derive_reused_bundle_manifest(
            contract_path=args.contract,
            source_contract_path=args.source_contract,
            source_manifest_path=args.source_manifest,
            archive_uri=args.archive_uri,
            output_path=args.output,
        )
    print(json.dumps(attestation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
