#!/usr/bin/env python3
"""Fail closed unless a local VideoChat3 CUDA certificate matches its inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


RUNTIME_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
)
MODEL_ID = "MCG-NJU/VideoChat3-4B"
MODEL_REVISION = "37fa901ec5913f84bc31108ebc1e60ad1903634c"
TRANSFORMERS_REVISION = "cbf4d720ec734edb77d25452b2790c7e4be2f8d7"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot_sha256(source_root: Path) -> str:
    """Hash every Python source file with its relative name, in stable order."""

    if not source_root.is_dir():
        raise ValueError("source root not found")
    digest = hashlib.sha256()
    source_files = sorted(source_root.rglob("*.py"))
    if not source_files:
        raise ValueError("source root contains no Python files")
    for path in source_files:
        if not path.is_file():
            continue
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fail(message: str) -> int:
    print(f"not eligible: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.certificate.is_file():
        return _fail("certificate not found")
    if not args.requirements.is_file():
        return _fail("requirements not found")
    try:
        certificate: Any = json.loads(args.certificate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _fail("certificate is not valid JSON")
    try:
        expected = {
            "runtime_image": RUNTIME_IMAGE,
            "requirements_sha256": _sha256_file(args.requirements),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "source_snapshot_sha256": source_snapshot_sha256(args.source_root),
        }
        from trec_video_2026.vqa_experiments.videochat3_runtime_certification import (
            validate_certificate,
        )

        validate_certificate(certificate, expected)
    except (OSError, ValueError, TypeError) as error:
        return _fail(str(error))
    print("eligible: complete NVIDIA CUDA BF16 runtime certificate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
