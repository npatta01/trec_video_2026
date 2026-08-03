#!/usr/bin/env python3
"""Render sealed Qwen3-Omni frozen-30 dstack configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import textwrap
from typing import Any


RUN_ID = "qwen3-omni-frozen30-audio-20260731-a4"
CANARY_RUN_ID = "qwen3-omni-f30-a4-bundle-canary"
DEV_RUN_ID = f"{RUN_ID}-dev"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
MODEL_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
RUNTIME_IMAGE = (
    "docker.io/qwenllm/qwen3-omni@"
    "sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)
OUTPUT_URI = "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/" + RUN_ID
INPUT_URI = "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/" + RUN_ID
CANONICAL_CONTRACT = Path("data/vqa/benchmarks/qwen3-omni-frozen30/contract-a4.json")
CANONICAL_BUNDLE_MANIFEST = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/bundle-a4/manifest.json"
)
STAGE1_RUN_ID = "qwen3-omni-f30-calibrated-20260801-s1"
STAGE1_CANARY_RUN_ID = "qwen3-omni-f30-s1-canary"
STAGE1_DEV_RUN_ID = "qwen3-omni-f30-s1-dev"
STAGE1_OUTPUT_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/"
    f"{STAGE1_RUN_ID}"
)
STAGE1_ARCHIVE_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/"
    "qwen3-omni-frozen30-audio-20260731-a4/evidence.tar"
)
STAGE1_ARCHIVE_SIZE = 318177280
STAGE1_ARCHIVE_SHA256 = (
    "9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943"
)
STAGE1_CONTRACT = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/contract-stage1.json"
)
STAGE1_BUNDLE_MANIFEST = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/bundle-stage1/manifest.json"
)
STAGE1_REPAIR_RUN_ID = "qwen3-omni-f30-calibrated-20260801-s2-nothink"
STAGE1_REPAIR_CANARY_RUN_ID = "qwen3-omni-f30-s2-nothink-canary"
STAGE1_REPAIR_DEV_RUN_ID = "qwen3-omni-f30-s2-nothink-dev"
STAGE1_REPAIR_OUTPUT_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/"
    f"{STAGE1_REPAIR_RUN_ID}"
)
STAGE1_REPAIR_CONTRACT = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/contract-stage1-repair.json"
)
STAGE1_REPAIR_BUNDLE_MANIFEST = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/bundle-stage1-repair/manifest.json"
)
STAGE1_SAMPLED_RUN_ID = "qwen3-omni-f30-calibrated-20260801-s3-sampled"
STAGE1_SAMPLED_CANARY_RUN_ID = "qwen3-omni-f30-s3-sampled-canary"
STAGE1_SAMPLED_DEV_RUN_ID = "qwen3-omni-f30-s3-sampled-dev"
STAGE1_SAMPLED_OUTPUT_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/"
    f"{STAGE1_SAMPLED_RUN_ID}"
)
STAGE1_SAMPLED_CONTRACT = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/contract-stage1-sampled.json"
)
STAGE1_SAMPLED_BUNDLE_MANIFEST = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/bundle-stage1-sampled/manifest.json"
)
STAGE1_EXPLICIT_SAMPLING_RUN_ID = (
    "qwen3-omni-f30-calibrated-20260801-s4-explicit-sampling"
)
STAGE1_EXPLICIT_SAMPLING_CANARY_RUN_ID = (
    "qwen3-omni-f30-s4-exp-canary"
)
STAGE1_EXPLICIT_SAMPLING_DEV_RUN_ID = (
    "qwen3-omni-f30-s4-exp-dev"
)
STAGE1_EXPLICIT_SAMPLING_OUTPUT_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/"
    f"{STAGE1_EXPLICIT_SAMPLING_RUN_ID}"
)
STAGE1_EXPLICIT_SAMPLING_CONTRACT = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/contract-stage1-explicit-sampling.json"
)
STAGE1_EXPLICIT_SAMPLING_BUNDLE_MANIFEST = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/bundle-stage1-explicit-sampling/manifest.json"
)
STAGE1_SEED42_RUN_ID = "qwen3-omni-f30-calibrated-20260801-s5-seed42"
STAGE1_SEED42_CANARY_RUN_ID = "qwen3-omni-f30-s5-seed42-canary"
STAGE1_SEED42_DEV_RUN_ID = "qwen3-omni-f30-s5-seed42-dev"
STAGE1_SEED42_OUTPUT_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/"
    f"{STAGE1_SEED42_RUN_ID}"
)
STAGE1_SEED42_CONTRACT = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/contract-stage1-seed42.json"
)
STAGE1_SEED42_BUNDLE_MANIFEST = Path(
    "data/vqa/benchmarks/qwen3-omni-frozen30/bundle-stage1-seed42/manifest.json"
)
EXPECTED_MEMBER_COUNT = 390
REPO_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "a4": {
        "run_id": RUN_ID,
        "canary_run_id": CANARY_RUN_ID,
        "dev_run_id": DEV_RUN_ID,
        "output_uri": OUTPUT_URI,
        "input_uri": INPUT_URI,
        "contract_path": CANONICAL_CONTRACT,
        "manifest_path": CANONICAL_BUNDLE_MANIFEST,
        "manifest_schema": 1,
        "archive_uri": None,
    },
    "stage1": {
        "run_id": STAGE1_RUN_ID,
        "canary_run_id": STAGE1_CANARY_RUN_ID,
        "dev_run_id": STAGE1_DEV_RUN_ID,
        "output_uri": STAGE1_OUTPUT_URI,
        "input_uri": STAGE1_ARCHIVE_URI.rsplit("/", 1)[0],
        "contract_path": STAGE1_CONTRACT,
        "manifest_path": STAGE1_BUNDLE_MANIFEST,
        "manifest_schema": 2,
        "archive_uri": STAGE1_ARCHIVE_URI,
    },
    "stage1_repair": {
        "run_id": STAGE1_REPAIR_RUN_ID,
        "canary_run_id": STAGE1_REPAIR_CANARY_RUN_ID,
        "dev_run_id": STAGE1_REPAIR_DEV_RUN_ID,
        "output_uri": STAGE1_REPAIR_OUTPUT_URI,
        "input_uri": STAGE1_ARCHIVE_URI.rsplit("/", 1)[0],
        "contract_path": STAGE1_REPAIR_CONTRACT,
        "manifest_path": STAGE1_REPAIR_BUNDLE_MANIFEST,
        "manifest_schema": 2,
        "archive_uri": STAGE1_ARCHIVE_URI,
    },
    "stage1_sampled": {
        "run_id": STAGE1_SAMPLED_RUN_ID,
        "canary_run_id": STAGE1_SAMPLED_CANARY_RUN_ID,
        "dev_run_id": STAGE1_SAMPLED_DEV_RUN_ID,
        "output_uri": STAGE1_SAMPLED_OUTPUT_URI,
        "input_uri": STAGE1_ARCHIVE_URI.rsplit("/", 1)[0],
        "contract_path": STAGE1_SAMPLED_CONTRACT,
        "manifest_path": STAGE1_SAMPLED_BUNDLE_MANIFEST,
        "manifest_schema": 2,
        "archive_uri": STAGE1_ARCHIVE_URI,
    },
    "stage1_explicit_sampling": {
        "run_id": STAGE1_EXPLICIT_SAMPLING_RUN_ID,
        "canary_run_id": STAGE1_EXPLICIT_SAMPLING_CANARY_RUN_ID,
        "dev_run_id": STAGE1_EXPLICIT_SAMPLING_DEV_RUN_ID,
        "output_uri": STAGE1_EXPLICIT_SAMPLING_OUTPUT_URI,
        "input_uri": STAGE1_ARCHIVE_URI.rsplit("/", 1)[0],
        "contract_path": STAGE1_EXPLICIT_SAMPLING_CONTRACT,
        "manifest_path": STAGE1_EXPLICIT_SAMPLING_BUNDLE_MANIFEST,
        "manifest_schema": 2,
        "archive_uri": STAGE1_ARCHIVE_URI,
    },
    "stage1_seed42": {
        "run_id": STAGE1_SEED42_RUN_ID,
        "canary_run_id": STAGE1_SEED42_CANARY_RUN_ID,
        "dev_run_id": STAGE1_SEED42_DEV_RUN_ID,
        "output_uri": STAGE1_SEED42_OUTPUT_URI,
        "input_uri": STAGE1_ARCHIVE_URI.rsplit("/", 1)[0],
        "contract_path": STAGE1_SEED42_CONTRACT,
        "manifest_path": STAGE1_SEED42_BUNDLE_MANIFEST,
        "manifest_schema": 2,
        "archive_uri": STAGE1_ARCHIVE_URI,
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _source_bundle_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative, payload in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _live_source_bundle(repo_root: Path) -> dict[str, bytes]:
    runner_path = repo_root / "scripts/run-qwen3-omni-frozen30-dstack.sh"
    source_root = repo_root / "src/trec_video_2026"
    if runner_path.is_symlink() or not runner_path.is_file():
        raise ValueError("source bundle runner must be a regular non-symlink file")
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source bundle root must be a regular non-symlink directory")
    files = {
        "scripts/run-qwen3-omni-frozen30-dstack.sh": runner_path.read_bytes(),
    }
    for path in source_root.rglob("*.py"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("source bundle Python entries must be regular files")
        relative = f"src/trec_video_2026/{path.relative_to(source_root).as_posix()}"
        files[relative] = path.read_bytes()
    if len(files) == 1:
        raise ValueError("source bundle contains no Python files")
    return files


def _git_source_bundle(revision: str | None, repo_root: Path) -> dict[str, bytes]:
    if (
        revision is None
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("submitted code revision must be a full lowercase commit SHA")
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0 or resolved.stdout.strip() != revision:
        raise ValueError("submitted code revision is not an exact local commit")
    tree = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            revision,
            "--",
            "scripts/run-qwen3-omni-frozen30-dstack.sh",
            "src/trec_video_2026",
        ],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if tree.returncode != 0:
        raise ValueError("submitted code revision tree is unavailable")
    files: dict[str, bytes] = {}
    for record in tree.stdout.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.split(b" ", 2)
        relative = encoded_path.decode("utf-8")
        included = relative == "scripts/run-qwen3-omni-frozen30-dstack.sh" or (
            relative.startswith("src/trec_video_2026/") and relative.endswith(".py")
        )
        if not included:
            continue
        if object_type != b"blob" or mode not in {b"100644", b"100755"}:
            raise ValueError("submitted source bundle entries must be regular files")
        blob = subprocess.run(
            ["git", "cat-file", "blob", object_id.decode("ascii")],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if blob.returncode != 0:
            raise ValueError("submitted source bundle blob is unavailable")
        files[relative] = blob.stdout
    if "scripts/run-qwen3-omni-frozen30-dstack.sh" not in files:
        raise ValueError("submitted source bundle runner is missing")
    if len(files) == 1:
        raise ValueError("submitted source bundle contains no Python files")
    return files


def source_bundle_sha256(
    revision: str | None,
    *,
    repo_root: Path | None = None,
) -> str:
    """Hash runner and Python blobs from an immutable submitted Git tree."""
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    return _source_bundle_digest(_git_source_bundle(revision, root))


def _validated_source_bundle_sha256(revision: str | None) -> str:
    committed = _git_source_bundle(revision, REPO_ROOT)
    if _live_source_bundle(REPO_ROOT) != committed:
        raise ValueError("submitted code revision does not match transported source bundle")
    return _source_bundle_digest(committed)


def _trusted_source_bundle_verifier() -> str:
    script = r'''/usr/bin/python3 -B - \
  /dstack/run/scripts/run-qwen3-omni-frozen30-dstack.sh \
  /dstack/run/src/trec_video_2026 \
  "$EXPECTED_SOURCE_BUNDLE_SHA256" <<'PY'
import hashlib
import sys
from pathlib import Path

runner_path = Path(sys.argv[1])
source_root = Path(sys.argv[2])
expected_digest = sys.argv[3]
if runner_path.is_symlink() or not runner_path.is_file():
    raise SystemExit("source bundle runner must be a regular non-symlink file")
if source_root.is_symlink() or not source_root.is_dir():
    raise SystemExit("source bundle root must be a regular non-symlink directory")
files = {"scripts/run-qwen3-omni-frozen30-dstack.sh": runner_path.read_bytes()}
for path in source_root.rglob("*.py"):
    if path.is_symlink() or not path.is_file():
        raise SystemExit("source bundle Python entries must be regular files")
    relative = f"src/trec_video_2026/{path.relative_to(source_root).as_posix()}"
    files[relative] = path.read_bytes()
if len(files) == 1:
    raise SystemExit("source bundle contains no Python files")
digest = hashlib.sha256()
for relative, payload in sorted(files.items()):
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(len(payload)).encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")
if digest.hexdigest() != expected_digest:
    raise SystemExit("source bundle SHA-256 mismatch")
print("FROZEN30_INLINE_SOURCE_BUNDLE_VERIFIED")
PY'''
    return textwrap.indent(script, "    ")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    parsed = json.loads(path.read_bytes())
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _safe_member_path(value: Any) -> str:
    if type(value) is not str:
        raise ValueError("bundle member path must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or path.parts[0] != "vqa"
    ):
        raise ValueError("bundle member path must be safe and canonical")
    return value


def _contract_evidence(contract: dict[str, Any]) -> dict[str, str]:
    examples = contract.get("examples")
    if not isinstance(examples, list) or len(examples) != 30:
        raise ValueError("contract must contain thirty examples")
    evidence: dict[str, str] = {}
    for row in examples:
        if not isinstance(row, dict):
            raise ValueError("contract examples must be objects")
        pairs = list(zip(row["frame_paths"], row["frame_sha256s"], strict=True))
        pairs.append((row["audio_path"], row["audio_sha256"]))
        for relative, digest in pairs:
            relative = _safe_member_path(relative)
            previous = evidence.setdefault(relative, digest)
            if previous != digest:
                raise ValueError("contract contains conflicting evidence hashes")
    return evidence


def _validated_inputs(
    contract_path: Path, bundle_manifest_path: Path, *, experiment: str
) -> tuple[dict[str, Any], str, str, dict[str, Any], list[str]]:
    try:
        profile = EXPERIMENTS[experiment]
    except KeyError as error:
        raise ValueError(
            "experiment must be a4, stage1, stage1_repair, stage1_sampled, "
            "stage1_explicit_sampling, or stage1_seed42"
        ) from error
    if contract_path != profile["contract_path"]:
        raise ValueError(f"contract path must be {profile['contract_path']}")
    if bundle_manifest_path != profile["manifest_path"]:
        raise ValueError(f"bundle manifest path must be {profile['manifest_path']}")
    contract = _read_json(contract_path, label="contract")
    manifest = _read_json(bundle_manifest_path, label="bundle manifest")
    if contract.get("run_id") != profile["run_id"]:
        raise ValueError(f"contract run ID does not match {experiment}")
    contract_digest = hashlib.sha256(_canonical_json(contract)).hexdigest()
    if set(manifest) != {"schema_version", "contract_sha256", "archive", "members"}:
        raise ValueError("bundle manifest schema does not match")
    if manifest.get("schema_version") != profile["manifest_schema"]:
        raise ValueError("bundle manifest schema version does not match")
    if manifest.get("contract_sha256") != contract_digest:
        raise ValueError(f"bundle manifest is not bound to the {experiment} contract")
    if bundle_manifest_path.read_bytes() != _canonical_json(manifest):
        raise ValueError("bundle manifest must use canonical JSON")
    archive = manifest["archive"]
    if (
        not isinstance(archive, dict)
        or set(archive)
        != {"name", "sha256", "size"}
        | ({"uri"} if profile["manifest_schema"] == 2 else set())
        or archive["name"] != "evidence.tar"
        or type(archive["sha256"]) is not str
        or len(archive["sha256"]) != 64
        or type(archive["size"]) is not int
        or archive["size"] <= 0
    ):
        raise ValueError("bundle archive metadata does not match")
    if profile["manifest_schema"] == 2 and archive != {
        "name": "evidence.tar",
        "sha256": STAGE1_ARCHIVE_SHA256,
        "size": STAGE1_ARCHIVE_SIZE,
        "uri": profile["archive_uri"],
    }:
        raise ValueError("bundle archive does not match the sealed Stage 1 object")
    members = manifest["members"]
    if not isinstance(members, list) or len(members) != EXPECTED_MEMBER_COUNT:
        raise ValueError("bundle manifest must contain 390 members")
    previous: str | None = None
    observed: dict[str, str] = {}
    for item in members:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            raise ValueError("bundle member schema does not match")
        relative = _safe_member_path(item["path"])
        if previous is not None and relative <= previous:
            raise ValueError("bundle member paths must be unique and sorted")
        previous = relative
        if type(item["sha256"]) is not str or type(item["size"]) is not int:
            raise ValueError("bundle member metadata does not match")
        observed[relative] = item["sha256"]
    if observed != _contract_evidence(contract):
        raise ValueError("bundle members do not match contract evidence")
    manifest_digest = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    mappings = [
        "../scripts/run-qwen3-omni-frozen30-dstack.sh:"
        "/dstack/run/scripts/run-qwen3-omni-frozen30-dstack.sh",
        "../src/trec_video_2026:/dstack/run/src/trec_video_2026",
        f"../{contract_path.as_posix()}:/dstack/input/contract.json",
        f"../{bundle_manifest_path.as_posix()}:/dstack/input/bundle-manifest.json",
    ]
    return profile, contract_digest, manifest_digest, archive, mappings


def _environment(
    profile: dict[str, Any],
    contract_digest: str,
    manifest_digest: str,
    archive: dict[str, Any],
    *,
    experiment: str,
    submitted_code_revision: str | None,
) -> str:
    profile_environment = ""
    if experiment.startswith("stage1"):
        if (
            submitted_code_revision is None
            or len(submitted_code_revision) != 40
            or any(character not in "0123456789abcdef" for character in submitted_code_revision)
        ):
            raise ValueError("submitted code revision must be a full lowercase commit SHA")
        profile_environment = (
            f"  - EXPERIMENT_PROFILE={experiment}\n"
            f"  - HF_ARCHIVE_URI={archive['uri']}\n"
            f"  - SUBMITTED_CODE_REVISION={submitted_code_revision}\n"
            "  - EXPECTED_SOURCE_BUNDLE_SHA256="
            f"{_validated_source_bundle_sha256(submitted_code_revision)}\n"
        )
    return f"""  - HF_TOKEN=${{{{ secrets.hf_token }}}}
  - HF_HUB_DISABLE_TELEMETRY=1
{profile_environment}  - EXPERIMENT_RUN_ID={profile['run_id']}
  - HF_INPUT_URI={profile['input_uri']}
  - HF_REMOTE_URI={profile['output_uri']}
  - RUNTIME_IMAGE={RUNTIME_IMAGE}
  - MODEL_ID={MODEL_ID}
  - MODEL_REVISION={MODEL_REVISION}
  - EXPECTED_CONTRACT={contract_digest}
  - EXPECTED_BUNDLE_MANIFEST={manifest_digest}
  - EXPECTED_BUNDLE_ARCHIVE={archive['sha256']}
  - EXPECTED_BUNDLE_SIZE={archive['size']}
  - PYTHON_BIN=/usr/bin/python3
  - PYTHONPATH=/dstack/run/src"""


def render_config(
    contract_path: Path,
    bundle_manifest_path: Path,
    *,
    kind: str = "task",
    experiment: str = "a4",
    submitted_code_revision: str | None = None,
) -> str:
    """Render a task, bundle canary, or dev environment from sealed inputs."""
    profile, contract_digest, manifest_digest, archive, mappings = _validated_inputs(
        Path(contract_path), Path(bundle_manifest_path), experiment=experiment
    )
    files = "\n".join(f"  - {mapping}" for mapping in mappings)
    environment = _environment(
        profile,
        contract_digest,
        manifest_digest,
        archive,
        experiment=experiment,
        submitted_code_revision=submitted_code_revision,
    )
    trusted_verifier = (
        f"{_trusted_source_bundle_verifier()}\n"
        if experiment.startswith("stage1")
        else ""
    )
    if kind == "dev":
        return f"""type: dev-environment
name: {profile['dev_run_id']}

image: {RUNTIME_IMAGE}
shell: bash
working_dir: /dstack/run
ide: vscode

env:
{environment}

files:
{files}

init:
  - |
    set -euo pipefail
{trusted_verifier}    mkdir -p /dstack/output
    echo FROZEN30_DEV_READY

inactivity_duration: 1h

resources:
  gpu:
    vendor: nvidia
    name: H200
    count: 1
    memory: 90GB..
  cpu: 16..
  memory: 192GB..
  disk: 200GB

fleets: [gpu-marketplace]
backends: [runpod, vastai]
spot_policy: on-demand
max_price: 4.50
max_duration: 2h
retry:
  on_events: [no-capacity]
  duration: 30m
"""
    if kind not in {"task", "canary"}:
        raise ValueError("kind must be task, canary, or dev")
    name = profile["run_id"] if kind == "task" else profile["canary_run_id"]
    command = "run" if kind == "task" else "bundle-canary"
    maximum = "2h" if kind == "task" else "15m"
    return f"""type: task
name: {name}
image: {RUNTIME_IMAGE}
entrypoint: /bin/bash -c
shell: bash
working_dir: /dstack/run

env:
{environment}

files:
{files}

commands:
  - |
    set -euo pipefail
{trusted_verifier}    exec bash /dstack/run/scripts/run-qwen3-omni-frozen30-dstack.sh {command}

resources:
  gpu:
    vendor: nvidia
    name: [H100, H200, B200, RTXPRO6000WS]
    count: 1
    memory: 90GB..
  cpu: 16..
  memory: 192GB..
  disk: 200GB

fleets: [gpu-marketplace]
backends: [runpod, vastai]
spot_policy: on-demand
max_price: 4.50
max_duration: {maximum}
retry:
  on_events: [no-capacity]
  duration: 30m
"""


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("task", "canary", "dev"), default="task")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--submitted-code-revision")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_config(
        args.contract,
        args.bundle_manifest,
        kind=args.kind,
        experiment=args.experiment,
        submitted_code_revision=args.submitted_code_revision,
    )
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            parser.error(f"{args.output} does not match the generated configuration")
        return 0
    _atomic_write(args.output, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
