"""Byte-exact transport for the one Qwen quality frame over dstack's 2 MiB cap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

SOURCE_RELATIVE_PATH = Path(
    "vqa/benchmarks/evidence/"
    "0b20cdcb075f8a2fcd3640a0a40356146796859051c4c203b9acb8c051ed68a6/"
    "frame-11.png"
)
CHUNK_RELATIVE_DIRECTORY = Path(
    "vqa/benchmarks/qwen3-omni-quality-10/transport/frame-11"
)
CONTRACT_RELATIVE_PATH = Path(
    "vqa/benchmarks/qwen3-omni-quality-10/contract.json"
)
SOURCE_SIZE = 2_441_142
SOURCE_SHA256 = "fc42daf585e038331b0e739cc5aa4761d50c95c90066d006238606718829d99e"
CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class ChunkSpec:
    name: str
    size: int
    sha256: str


CHUNK_SPECS = (
    ChunkSpec(
        "part-0000.bin",
        1_000_000,
        "869372a01a7e53c3eeffe5ddb3a2b92b8a4fb2d60cd8ac056f00c1c7e3bf3090",
    ),
    ChunkSpec(
        "part-0001.bin",
        1_000_000,
        "b73bfd8757e4c2cdee061b0870ad312d6981ba1a9ca5817602638472878ec93e",
    ),
    ChunkSpec(
        "part-0002.bin",
        441_142,
        "1d239ec8b8ded80ef4f1ce07ccf1f0f632bd4487cf4f7a9893e0d330b66f10f1",
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_data_root(data_root: Path, path: Path) -> Path:
    resolved_root = data_root.resolve()
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"path escapes data root: {path}") from error
    return resolved


def _chunk_directory(data_root: Path, chunk_directory: Path | None) -> Path:
    if chunk_directory is None:
        chunk_directory = data_root / CHUNK_RELATIVE_DIRECTORY
    elif not chunk_directory.is_absolute():
        chunk_directory = data_root / chunk_directory
    return _inside_data_root(data_root, chunk_directory)


def _verify_contract_binding(
    data_root: Path, contract_path: Path | None = None
) -> None:
    if contract_path is None:
        contract_path = data_root / CONTRACT_RELATIVE_PATH
    elif not contract_path.is_absolute():
        contract_path = data_root / contract_path
    contract_path = _inside_data_root(data_root, contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    matches: list[str] = []
    for example in contract.get("examples", []):
        frame_paths = example.get("frame_paths", [])
        frame_hashes = example.get("frame_sha256s", [])
        if SOURCE_RELATIVE_PATH.as_posix() in frame_paths:
            index = frame_paths.index(SOURCE_RELATIVE_PATH.as_posix())
            if index >= len(frame_hashes):
                raise ValueError("contract frame hash is missing")
            matches.append(frame_hashes[index])
    if matches != [SOURCE_SHA256]:
        raise ValueError("contract frame hash does not match frozen transport")


def _verify_chunks(chunk_directory: Path) -> None:
    if not chunk_directory.is_dir() or chunk_directory.is_symlink():
        raise ValueError("chunk directory is missing or unsafe")
    actual_names = sorted(path.name for path in chunk_directory.iterdir())
    expected_names = [spec.name for spec in CHUNK_SPECS]
    if actual_names != expected_names:
        raise ValueError("chunk count or names do not match the frozen transport")
    for spec in CHUNK_SPECS:
        path = chunk_directory / spec.name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"chunk is missing or unsafe: {spec.name}")
        if path.stat().st_size != spec.size:
            raise ValueError(f"chunk size does not match: {spec.name}")
        if _sha256_file(path) != spec.sha256:
            raise ValueError(f"chunk hash does not match: {spec.name}")


def _combined_sha256(chunk_directory: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for spec in CHUNK_SPECS:
        with (chunk_directory / spec.name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
    return size, digest.hexdigest()


def _attestation() -> dict[str, object]:
    return {
        "source_relative_path": SOURCE_RELATIVE_PATH.as_posix(),
        "source_size": SOURCE_SIZE,
        "source_sha256": SOURCE_SHA256,
        "chunk_relative_directory": CHUNK_RELATIVE_DIRECTORY.as_posix(),
        "chunks": [asdict(spec) for spec in CHUNK_SPECS],
    }


def prepare_chunks(data_root: Path) -> dict[str, object]:
    """Create and verify deterministic ignored chunks from the pinned source."""
    data_root = data_root.resolve()
    _verify_contract_binding(data_root)
    source = _inside_data_root(data_root, data_root / SOURCE_RELATIVE_PATH)
    if not source.is_file() or source.is_symlink():
        raise ValueError("pinned source frame is missing or unsafe")
    if source.stat().st_size != SOURCE_SIZE or _sha256_file(source) != SOURCE_SHA256:
        raise ValueError("pinned source frame does not match its contract hash")

    chunk_directory = _chunk_directory(data_root, None)
    if chunk_directory.exists():
        _verify_chunks(chunk_directory)
        combined_size, combined_hash = _combined_sha256(chunk_directory)
        if (combined_size, combined_hash) != (SOURCE_SIZE, SOURCE_SHA256):
            raise ValueError("existing chunks do not recombine to the pinned frame")
        return _attestation()

    chunk_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".frame-11.", dir=chunk_directory.parent)
    )
    try:
        with source.open("rb") as handle:
            for spec in CHUNK_SPECS:
                payload = handle.read(CHUNK_SIZE)
                (temporary / spec.name).write_bytes(payload)
            if handle.read(1):
                raise ValueError("pinned source produced an unexpected extra chunk")
        _verify_chunks(temporary)
        combined_size, combined_hash = _combined_sha256(temporary)
        if (combined_size, combined_hash) != (SOURCE_SIZE, SOURCE_SHA256):
            raise ValueError("prepared chunks do not recombine to the pinned frame")
        os.rename(temporary, chunk_directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _attestation()


def reconstruct_frame(
    data_root: Path,
    *,
    chunk_directory: Path | None = None,
    contract_path: Path | None = None,
) -> dict[str, object]:
    """Reconstruct the contract-bound frame and remove verified staging chunks."""
    data_root = data_root.resolve()
    _verify_contract_binding(data_root, contract_path)
    chunks = _chunk_directory(data_root, chunk_directory)
    _verify_chunks(chunks)
    combined_size, combined_hash = _combined_sha256(chunks)
    if (combined_size, combined_hash) != (SOURCE_SIZE, SOURCE_SHA256):
        raise ValueError("chunks do not recombine to the pinned frame")

    target = _inside_data_root(data_root, data_root / SOURCE_RELATIVE_PATH)
    if target.exists() or target.is_symlink():
        raise ValueError("reconstructed target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.reconstructing")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("reconstruction temporary path already exists")
    try:
        with temporary.open("xb") as output:
            for spec in CHUNK_SPECS:
                with (chunks / spec.name).open("rb") as source:
                    shutil.copyfileobj(source, output)
        if temporary.stat().st_size != SOURCE_SIZE:
            raise ValueError("reconstructed frame size does not match")
        if _sha256_file(temporary) != SOURCE_SHA256:
            raise ValueError("reconstructed frame hash does not match")
        temporary.replace(target)
        if _sha256_file(target) != SOURCE_SHA256:
            raise ValueError("published frame hash does not match")
    finally:
        if temporary.exists():
            temporary.unlink()
    shutil.rmtree(chunks)
    return _attestation()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--data-root", type=Path, required=True)
    reconstruct_parser = subparsers.add_parser("reconstruct")
    reconstruct_parser.add_argument("--data-root", type=Path, required=True)
    reconstruct_parser.add_argument("--chunk-directory", type=Path, required=True)
    reconstruct_parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        record = prepare_chunks(args.data_root)
    else:
        record = reconstruct_frame(
            args.data_root,
            chunk_directory=args.chunk_directory,
            contract_path=args.contract,
        )
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
