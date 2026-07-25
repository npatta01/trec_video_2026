import hashlib
import json
import shutil
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.qwen3_omni_quality_transport import (
    CHUNK_RELATIVE_DIRECTORY,
    CHUNK_SPECS,
    SOURCE_SHA256,
    SOURCE_RELATIVE_PATH,
    prepare_chunks,
    reconstruct_frame,
)


def _copy_source(data_root: Path) -> bytes:
    source = Path("data") / SOURCE_RELATIVE_PATH
    payload = source.read_bytes()
    destination = data_root / SOURCE_RELATIVE_PATH
    destination.parent.mkdir(parents=True)
    shutil.copyfile(source, destination)
    contract_source = Path("data/vqa/benchmarks/qwen3-omni-quality-10/contract.json")
    contract_destination = (
        data_root / "vqa/benchmarks/qwen3-omni-quality-10/contract.json"
    )
    contract_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contract_source, contract_destination)
    return payload


def test_chunk_preparation_and_reconstruction_are_byte_exact(tmp_path: Path) -> None:
    """Catches chunk transport that changes the contract-bound frame bytes."""
    payload = _copy_source(tmp_path)

    prepared = prepare_chunks(tmp_path)
    chunks = sorted((tmp_path / CHUNK_RELATIVE_DIRECTORY).iterdir())
    assert [(path.name, path.stat().st_size) for path in chunks] == [
        (spec.name, spec.size) for spec in CHUNK_SPECS
    ]
    assert all(path.stat().st_size < 2 * 1024 * 1024 for path in chunks)
    assert prepared["source_sha256"] == SOURCE_SHA256

    (tmp_path / SOURCE_RELATIVE_PATH).unlink()
    local_contract = (
        tmp_path / "vqa/benchmarks/qwen3-omni-quality-10/contract.json"
    )
    remote_contract = tmp_path / "contract.json"
    local_contract.rename(remote_contract)
    reconstruct_frame(tmp_path, contract_path=remote_contract)

    reconstructed = (tmp_path / SOURCE_RELATIVE_PATH).read_bytes()
    assert reconstructed == payload
    assert hashlib.sha256(reconstructed).hexdigest() == SOURCE_SHA256
    assert not (tmp_path / CHUNK_RELATIVE_DIRECTORY).exists()


@pytest.mark.parametrize("tamper", ("extra_chunk", "changed_bytes"))
def test_reconstruction_fails_closed_on_chunk_count_or_hash(
    tmp_path: Path, tamper: str
) -> None:
    """Catches accepting an incomplete, extra, or modified chunk set."""
    _copy_source(tmp_path)
    prepare_chunks(tmp_path)
    (tmp_path / SOURCE_RELATIVE_PATH).unlink()
    chunk_directory = tmp_path / CHUNK_RELATIVE_DIRECTORY
    if tamper == "extra_chunk":
        (chunk_directory / "unexpected.bin").write_bytes(b"unexpected")
    else:
        first_chunk = chunk_directory / CHUNK_SPECS[0].name
        first_chunk.write_bytes(b"changed" + first_chunk.read_bytes()[7:])

    with pytest.raises(ValueError, match="chunk"):
        reconstruct_frame(tmp_path)

    assert not (tmp_path / SOURCE_RELATIVE_PATH).exists()


def test_chunk_preparation_rejects_a_changed_contract_frame_hash(
    tmp_path: Path,
) -> None:
    """Catches transport constants drifting away from the uploaded contract."""
    _copy_source(tmp_path)
    contract_path = (
        tmp_path / "vqa/benchmarks/qwen3-omni-quality-10/contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for example in contract["examples"]:
        if SOURCE_RELATIVE_PATH.as_posix() in example["frame_paths"]:
            index = example["frame_paths"].index(SOURCE_RELATIVE_PATH.as_posix())
            example["frame_sha256s"][index] = "0" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="contract frame hash"):
        prepare_chunks(tmp_path)
