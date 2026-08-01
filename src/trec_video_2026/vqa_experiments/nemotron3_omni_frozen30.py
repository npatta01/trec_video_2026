"""Independent dual-freeze evaluation for the Nemotron frozen-30 run."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any

from trec_video_2026.vqa_evaluation import normalized_exact_match, token_f1
from trec_video_2026.vqa_experiments.nemotron3_omni_grading import (
    BlindGradingBundle,
    GradeResolution,
    reconcile_grades as _reconcile_shared_grades,
)
from trec_video_2026.vqa_experiments.nemotron3_omni_pilot import (
    CONDITIONS,
    FROZEN30_INFERENCE_MANIFEST_SHA256,
    FROZEN30_NORMALIZED_EXAMPLE_FIELDS,
    FROZEN30_ROWS,
    FROZEN30_SOURCE_MEDIA_PATHS,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
    FORBIDDEN_LABEL_KEYS,
    _validate_run_contract,
    build_chat_request,
)
from trec_video_2026.vqa_gate import canonical_request_sha256

QWEN_RUN_ID = "native-audio-benchmark-v2"
QWEN_FULL_FREEZE_SHA256 = (
    "8aa8c76e809402a97e32277e0846b1cdae033c9727535717724b1be0b0235879"
)
QWEN_CONDITION_ORDER = (
    "qwen35-4b-12f-frozen30",
    "qwen35-4b-24f-frozen30",
    "minicpmo45-native-video-audio-off-frozen30",
    "qwen35-4b-12f-whisper-large-v3-frozen30",
)
CANDIDATE_RUN_ID = "nemotron3-omni-frozen30-20260801-a1"
QWEN_MODEL_ID = "Qwen/Qwen3.5-4B"
QWEN_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN_RUNTIME_IMAGE = (
    "localhost/trec-vqa-gate-rocm@"
    "sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5"
)
_CANDIDATE_CONDITIONS = ("audio-off", "audio-on")
_ALL_CONDITIONS = ("qwen12", *_CANDIDATE_CONDITIONS)
_GRADES = frozenset({"C", "P", "W"})
_GRADE_RANK = {"W": 0, "P": 1, "C": 2}
_RUBRIC = {"C": "Correct", "P": "Partially correct", "W": "Wrong"}
_LABEL_KEYS = FORBIDDEN_LABEL_KEYS
_NORMALIZED_MEDIA_ROOT = Path("/dstack/input/normalized")
_DEFAULT_SOURCE_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "configs/vqa/native_audio_benchmark_30_inference.json"
)
_NORMALIZED_EXAMPLE_FIELDS = FROZEN30_NORMALIZED_EXAMPLE_FIELDS
_VERIFIED_OPERATIONAL_TOKEN = object()
_TRUST_ROOT_PATH = "configs/vqa/nemotron3_frozen30_trust_root.json"
_TRUST_ROOT_REPOSITORY = Path(__file__).resolve().parents[3]
_TRUST_ROOT_REVIEWER_ID = "1314528+npatta01@users.noreply.github.com"
_TRUST_ROOT_SIGNING_PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBr9W8B8bSDKcpgO6XXE1Y6cpcgtaPumuEjW/shnQB5O"
)
_TRUST_ROOT_SIGNING_KEY_FINGERPRINT = (
    "SHA256:+w0DSDr+utElA1L6qX7urVSOt8r9kb221ZA8QDm3Gcc"
)
# Pinned after the signed post-collection review attestation commit.
_TRUST_ROOT_COMMIT: str | None = "10c3747e47a2fe1209401698e402d07b664a8818"
_TRUST_ROOT_BLOB: str | None = "bd4a518b3e498a09ff6745329c696eb25e6012c3"
_FROZEN_BACKENDS = frozenset({"runpod", "RunPod", "vastai", "Vast.ai"})
_FROZEN_MAX_HOURLY_PRICE_USD = 2.0
_FROZEN_MIN_CPU_COUNT = 8
_FROZEN_MIN_GPU_MEMORY_MIB = 49_152
_FROZEN_MIN_HOST_MEMORY_MIB = 65_536
_FROZEN_MIN_DISK_MIB = 102_400


@dataclass(frozen=True, init=False)
class OperationalEvidence:
    completed: int
    expected: int
    run_status: str
    duration_minutes: float
    hourly_price_usd: float
    compute_cost_usd: float
    artifacts_verified: bool
    privacy_or_audit_gap: bool
    workload_anchor_sha256: str
    audit_bundle_sha256: str
    trust_root_commit: str
    trust_root_blob: str
    reviewer_id: str
    reviewer_key_fingerprint: str
    reviewed_at: str
    _verification_token: object

    @classmethod
    def _from_verified_artifacts(
        cls,
        *,
        completed: int,
        expected: int,
        run_status: str,
        duration_minutes: float,
        hourly_price_usd: float,
        compute_cost_usd: float,
        privacy_or_audit_gap: bool,
        workload_anchor_sha256: str,
        audit_bundle_sha256: str,
        trust_root_commit: str,
        trust_root_blob: str,
        reviewer_id: str,
        reviewer_key_fingerprint: str,
        reviewed_at: str,
        verification_token: object,
    ) -> OperationalEvidence:
        """Construct only after the retained artifact boundary has validated."""

        if verification_token is not _VERIFIED_OPERATIONAL_TOKEN:
            raise TypeError("operational evidence requires verified retained artifacts")
        value = object.__new__(cls)
        fields = {
            "completed": completed,
            "expected": expected,
            "run_status": run_status,
            "duration_minutes": duration_minutes,
            "hourly_price_usd": hourly_price_usd,
            "compute_cost_usd": compute_cost_usd,
            "artifacts_verified": True,
            "privacy_or_audit_gap": privacy_or_audit_gap,
            "workload_anchor_sha256": workload_anchor_sha256,
            "audit_bundle_sha256": audit_bundle_sha256,
            "trust_root_commit": trust_root_commit,
            "trust_root_blob": trust_root_blob,
            "reviewer_id": reviewer_id,
            "reviewer_key_fingerprint": reviewer_key_fingerprint,
            "reviewed_at": reviewed_at,
            "_verification_token": verification_token,
        }
        for name, field_value in fields.items():
            object.__setattr__(value, name, field_value)
        return value


@dataclass(frozen=True, repr=False)
class Frozen30GradingBundle(BlindGradingBundle):
    """Opaque grader packets plus private verified rows for final scoring."""

    _rows: tuple[dict[str, Any], ...]
    _grader_assignment_artifact: dict[str, Any]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _parse_jsonl(data: bytes, *, description: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must be UTF-8 JSONL") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{description} must contain only object rows")
    return rows


def _contains_label(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            (isinstance(key, str) and key.lower() in _LABEL_KEYS)
            or _contains_label(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_label(item) for item in value)
    return False


def _verified_media_sha256(
    *,
    media_root: Path,
    relative_path: object,
    expected_sha256: object,
    description: str,
) -> str:
    """Hash one independently supplied media file without permitting root escape."""

    if not media_root.is_dir():
        raise ValueError(f"{description} root must be a directory")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{description} path must be a non-empty relative path")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{description} path escapes its media root")
    resolved_root = media_root.resolve()
    try:
        media_path = resolved_root.joinpath(*relative.parts).resolve(strict=True)
        media_path.relative_to(resolved_root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            f"{description} artifact is missing or escapes its root"
        ) from error
    if not media_path.is_file():
        raise ValueError(f"{description} artifact must be a file")
    with media_path.open("rb") as handle:
        actual_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{description} media hash mismatch")
    return actual_sha256


def verify_qwen_freeze(predictions_path: Path) -> tuple[dict[str, Any], ...]:
    """Recompute and validate the pinned original 120-row Qwen v2 freeze."""

    if not predictions_path.is_file():
        raise ValueError("Qwen frozen predictions must be a file")
    frozen_bytes = predictions_path.read_bytes()
    if hashlib.sha256(frozen_bytes).hexdigest() != QWEN_FULL_FREEZE_SHA256:
        raise ValueError("Qwen pinned full-file SHA-256 mismatch")
    rows = _parse_jsonl(frozen_bytes, description="Qwen frozen predictions")
    if len(rows) != 120 or _canonical_jsonl(rows) != frozen_bytes:
        raise ValueError("Qwen freeze must contain 120 canonical rows")

    expected_keys = [
        (condition, video_id)
        for condition in QWEN_CONDITION_ORDER
        for video_id, _category, _role in FROZEN30_ROWS
    ]
    actual_keys = [(row.get("condition_id"), row.get("video_id")) for row in rows]
    if actual_keys != expected_keys:
        raise ValueError("Qwen canonical condition/order mismatch")
    for row, (condition, video_id) in zip(rows, expected_keys, strict=True):
        request = row.get("request")
        if (
            row.get("run_id") != QWEN_RUN_ID
            or row.get("status") != "passed"
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
            or not isinstance(request, Mapping)
            or request.get("run_id") != QWEN_RUN_ID
            or request.get("condition_id") != condition
            or request.get("video_id") != video_id
            or row.get("request_sha256") != canonical_request_sha256(request)
            or not all(
                _is_sha256(row.get(field))
                for field in (
                    "evidence_identity_sha256",
                    "evidence_manifest_sha256",
                    "launch_decision_sha256",
                )
            )
            or _contains_label(request)
        ):
            raise ValueError("Qwen embedded request provenance mismatch")
    for row in rows[:30]:
        if (
            row.get("model_id") != QWEN_MODEL_ID
            or row.get("model_revision") != QWEN_MODEL_REVISION
            or row.get("runtime_image") != QWEN_RUNTIME_IMAGE
        ):
            raise ValueError("Qwen12 model provenance mismatch")
    return tuple(rows[:30])


def _checksum_entries(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError("candidate SHA256SUMS must be a file")
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("candidate SHA256SUMS is malformed") from error
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or not _is_sha256(digest)
            or relative in entries
            or not relative.startswith("./")
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or relative == "./SHA256SUMS"
        ):
            raise ValueError("candidate SHA256SUMS is malformed")
        entries[relative] = digest
    return entries


_AUDIT_EVIDENCE_FILES = frozenset(
    {
        "./INPUTS_COMPLETE.json",
        "./collection-status.json",
        "./dstack-run.json",
        "./preview.txt",
    }
)
_AUDIT_FILES = frozenset(
    {"./SHA256SUMS", "./workload-anchor.json", *_AUDIT_EVIDENCE_FILES}
)
_COLLECTOR_CANDIDATE_FILES = frozenset(
    {"./artifact-hashes.sha256", "./collection-status.json", "./dstack-run.json"}
)


def _file_hashes(root: Path, *, description: str) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError(f"{description} root must be a directory")
    result: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{description} must not contain symlinks")
        if path.is_file():
            relative = f"./{path.relative_to(root).as_posix()}"
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} must be a retained regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _load_trusted_attestation() -> dict[str, Any]:
    """Load the attestation only through an exact immutable Git commit/blob pin."""

    repository = _TRUST_ROOT_REPOSITORY
    commit = _TRUST_ROOT_COMMIT
    blob = _TRUST_ROOT_BLOB
    if not (
        isinstance(commit, str)
        and len(commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in commit)
        and isinstance(blob, str)
        and len(blob) in {40, 64}
        and all(character in "0123456789abcdef" for character in blob)
    ):
        raise ValueError("trust root requires full lowercase Git commit and blob pins")
    resolved_repository = repository.resolve()

    def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(resolved_repository),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise ValueError("trusted attestation Git identity is unavailable")
        return result

    top_level = git("rev-parse", "--show-toplevel").stdout.strip()
    if Path(top_level).resolve() != resolved_repository:
        raise ValueError("trust root repository must be its exact Git top level")
    if git("rev-parse", "--verify", f"{commit}^{{commit}}").stdout.strip() != commit:
        raise ValueError("trusted attestation commit pin mismatch")
    if git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode != 0:
        raise ValueError("trusted attestation commit must be reachable from HEAD")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as allowed_signers:
        allowed_signers.write(
            f'{_TRUST_ROOT_REVIEWER_ID} namespaces="git" '
            f"{_TRUST_ROOT_SIGNING_PUBLIC_KEY}\n"
        )
        allowed_signers.flush()
        signature = git(
            "-c",
            f"gpg.ssh.allowedSignersFile={allowed_signers.name}",
            "verify-commit",
            commit,
            check=False,
        )
    if signature.returncode != 0:
        raise ValueError("trusted attestation commit signature is not authorized")
    changed_paths = git(
        "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit
    ).stdout.splitlines()
    if changed_paths != [_TRUST_ROOT_PATH]:
        raise ValueError("trusted attestation commit must change only its fixed path")
    resolved_blob = git("rev-parse", f"{commit}:{_TRUST_ROOT_PATH}").stdout.strip()
    if resolved_blob != blob:
        raise ValueError("trusted attestation blob pin mismatch")
    tree_entry = git("ls-tree", commit, "--", _TRUST_ROOT_PATH).stdout.strip()
    if tree_entry != f"100644 blob {blob}\t{_TRUST_ROOT_PATH}":
        raise ValueError("trusted attestation must be a regular committed blob")
    data = git("cat-file", "blob", blob).stdout
    try:
        value = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError("trusted attestation must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("trusted attestation must be an object")
    return value


def _audit_bundle_sha256(file_hashes: Mapping[str, str]) -> str:
    """Hash the exact non-manifest audit file map without filesystem metadata."""

    evidence = {
        relative: digest
        for relative, digest in file_hashes.items()
        if relative != "./SHA256SUMS"
    }
    return hashlib.sha256(_canonical_json(evidence)).hexdigest()


def _validate_audit_bundle(
    audit_bundle_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    """Validate the fixed audit set, checksums, and independently retained anchor."""

    actual = _file_hashes(audit_bundle_root, description="audit bundle")
    if set(actual) != _AUDIT_FILES:
        raise ValueError("audit bundle exact file set mismatch")
    checksums = _checksum_entries(audit_bundle_root / "SHA256SUMS")
    if set(checksums) != _AUDIT_FILES - {"./SHA256SUMS"}:
        raise ValueError("audit checksum exact file set mismatch")
    for relative, digest in checksums.items():
        if actual.get(relative) != digest:
            raise ValueError(f"audit checksum mismatch: {relative}")

    anchor_path = audit_bundle_root / "workload-anchor.json"
    anchor = _load_json_object(anchor_path, description="workload anchor")
    if set(anchor) != {
        "schema_version",
        "anchor_type",
        "run_id",
        "candidate_file_count",
        "candidate_files",
        "audit_evidence_file_count",
        "audit_evidence_files",
    }:
        raise ValueError("workload anchor schema mismatch")
    candidate_files = anchor.get("candidate_files")
    audit_evidence_files = anchor.get("audit_evidence_files")
    if (
        anchor.get("schema_version") != 1
        or anchor.get("anchor_type") != "nemotron3-frozen30-collected-workload"
        or anchor.get("run_id") != CANDIDATE_RUN_ID
        or not isinstance(candidate_files, dict)
        or anchor.get("candidate_file_count") != len(candidate_files)
        or not isinstance(audit_evidence_files, dict)
        or anchor.get("audit_evidence_file_count") != len(audit_evidence_files)
        or set(audit_evidence_files) != _AUDIT_EVIDENCE_FILES
        or not all(
            isinstance(relative, str) and _is_sha256(digest)
            for relative, digest in candidate_files.items()
        )
        or not all(
            isinstance(relative, str) and _is_sha256(digest)
            for relative, digest in audit_evidence_files.items()
        )
    ):
        raise ValueError("workload anchor identity mismatch")
    for relative, digest in audit_evidence_files.items():
        if actual.get(relative) != digest:
            raise ValueError(f"workload anchor audit evidence mismatch: {relative}")

    status = _load_json_object(
        audit_bundle_root / "collection-status.json",
        description="collection status",
    )
    dstack = _load_json_object(
        audit_bundle_root / "dstack-run.json", description="dstack run evidence"
    )
    return (
        anchor,
        status,
        dstack,
        hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        _audit_bundle_sha256(actual),
    )


def _validate_candidate_anchor(
    *,
    candidate_root: Path,
    audit_bundle_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    anchor, status, dstack, anchor_sha, audit_sha = _validate_audit_bundle(
        audit_bundle_root
    )
    actual = _file_hashes(candidate_root, description="candidate bundle")
    if anchor["candidate_files"] != actual:
        raise ValueError("candidate bundle differs from retained workload anchor")
    return anchor, status, dstack, anchor_sha, audit_sha


def _validate_trusted_attestation(
    *,
    attestation: Mapping[str, Any],
    audit_bundle_root: Path,
    dstack: Mapping[str, Any],
    workload_anchor_sha256: str,
    audit_bundle_sha256: str,
) -> None:
    marker_path = audit_bundle_root / "INPUTS_COMPLETE.json"
    marker = _load_json_object(marker_path, description="retained input marker")
    reviewer_id = attestation.get("reviewer_id")
    reviewed_at = attestation.get("reviewed_at")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at)
        finished_at = datetime.fromisoformat(dstack.get("finished_at"))
    except (TypeError, ValueError) as error:
        raise ValueError("trusted review timestamps must be ISO-8601") from error
    expected_fields = {
        "schema_version",
        "attestation_type",
        "run_id",
        "workload_anchor_sha256",
        "audit_bundle_schema_version",
        "audit_bundle_sha256",
        "input_marker_sha256",
        "input_contract_sha256",
        "reviewer_id",
        "reviewer_key_fingerprint",
        "reviewed_at",
        "determination",
    }
    if (
        set(attestation) != expected_fields
        or attestation.get("schema_version") != 1
        or attestation.get("attestation_type")
        != "nemotron3-frozen30-post-collection-review"
        or attestation.get("run_id") != CANDIDATE_RUN_ID
        or attestation.get("workload_anchor_sha256") != workload_anchor_sha256
        or attestation.get("audit_bundle_schema_version") != 1
        or attestation.get("audit_bundle_sha256") != audit_bundle_sha256
        or attestation.get("input_marker_sha256")
        != hashlib.sha256(marker_path.read_bytes()).hexdigest()
        or attestation.get("input_contract_sha256") != marker.get("contract_sha256")
        or reviewer_id != _TRUST_ROOT_REVIEWER_ID
        or attestation.get("reviewer_key_fingerprint")
        != _TRUST_ROOT_SIGNING_KEY_FINGERPRINT
        or parsed_reviewed_at.utcoffset() is None
        or finished_at.utcoffset() is None
        or parsed_reviewed_at <= finished_at
        or attestation.get("determination")
        not in {"no-load-bearing-gap", "load-bearing-gap"}
    ):
        raise ValueError("trusted attestation does not bind the collected workload")


def _validate_input_evidence(
    *,
    audit_bundle_root: Path,
    input_bundle_root: Path,
    contract: Mapping[str, Any],
    source_media_root: Path,
) -> None:
    input_contract_path = input_bundle_root / "contract.json"
    input_contract = _load_json_object(
        input_contract_path, description="retained input contract"
    )
    try:
        mode, input_examples, input_conditions = _validate_run_contract(input_contract)
    except (TypeError, ValueError) as error:
        raise ValueError(f"retained input contract is invalid: {error}") from error
    marker = _load_json_object(
        audit_bundle_root / "INPUTS_COMPLETE.json", description="retained input marker"
    )
    if set(marker) != {
        "schema_version",
        "input_count",
        "object_count",
        "contract_sha256",
        "objects",
    }:
        raise ValueError("retained input marker schema mismatch")
    objects = marker.get("objects")
    expected_objects = {
        "contract.json",
        *(example["media_path"] for example in input_examples),
    }
    if (
        mode != "frozen30-paired"
        or input_conditions != [dict(condition) for condition in CONDITIONS]
        or marker.get("schema_version") != 1
        or marker.get("input_count") != 31
        or marker.get("object_count") != 32
        or not isinstance(objects, dict)
        or set(objects) != expected_objects
        or not all(_is_sha256(digest) for digest in objects.values())
        or marker.get("contract_sha256") != objects.get("contract.json")
        or objects.get("contract.json")
        != hashlib.sha256(input_contract_path.read_bytes()).hexdigest()
    ):
        raise ValueError("retained input marker identity mismatch")
    if (
        contract.get("source_contract_sha256")
        != hashlib.sha256(_canonical_json(input_contract)).hexdigest()
    ):
        raise ValueError("candidate does not bind the retained input contract")
    for example in input_examples:
        expected_sha = objects[example["media_path"]]
        if example["media_sha256"] != expected_sha:
            raise ValueError("retained input marker source media mismatch")
        _verified_media_sha256(
            media_root=source_media_root,
            relative_path=example["media_path"],
            expected_sha256=expected_sha,
            description="retained input source",
        )


def _source_contract_from_normalized(
    contract: Mapping[str, Any], *, source_manifest_path: Path
) -> dict[str, Any]:
    """Reconstruct the source contract from its pinned label-free manifest."""

    if not source_manifest_path.is_file():
        raise ValueError("pinned source manifest must be a file")
    source_bytes = source_manifest_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != FROZEN30_INFERENCE_MANIFEST_SHA256:
        raise ValueError("pinned source manifest SHA-256 mismatch")
    try:
        source_manifest = json.loads(source_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("pinned source manifest must be valid JSON") from error
    source_rows = (
        source_manifest.get("examples")
        if isinstance(source_manifest, Mapping)
        else None
    )
    normalized_rows = contract.get("examples")
    if not isinstance(source_rows, list) or len(source_rows) != 30:
        raise ValueError("pinned source manifest must contain exactly 30 rows")
    if not isinstance(normalized_rows, list) or len(normalized_rows) != 30:
        raise ValueError("normalized contract must contain exactly 30 rows")

    source_examples: list[dict[str, Any]] = []
    for index, ((video_id, category, role), source, normalized) in enumerate(
        zip(FROZEN30_ROWS, source_rows, normalized_rows, strict=True)
    ):
        if not isinstance(source, Mapping) or set(source) != {
            "video_id",
            "category",
            "question",
            "media",
        }:
            raise ValueError("pinned source manifest row schema mismatch")
        if not isinstance(normalized, Mapping):
            raise ValueError("normalized contract rows must be objects")
        if set(normalized) != _NORMALIZED_EXAMPLE_FIELDS:
            raise ValueError("normalized contract example schema mismatch")
        question = source.get("question")
        expected_media = FROZEN30_SOURCE_MEDIA_PATHS[video_id]
        if (
            source.get("video_id") != video_id
            or source.get("category") != category
            or source.get("media") != expected_media.rsplit(".", 1)[0]
            or not isinstance(question, str)
            or not question.strip()
            or normalized.get("video_id") != video_id
            or normalized.get("category") != category
            or normalized.get("role") != role
            or normalized.get("question") != question.strip()
            or normalized.get("source_media_path") != expected_media
        ):
            raise ValueError(
                f"normalized source identity or question mismatch at {index}"
            )
        source_examples.append(
            {
                "video_id": video_id,
                "category": category,
                "role": role,
                "question": question.strip(),
                "media_path": expected_media,
                "media_sha256": normalized["source_media_sha256"],
            }
        )
    return {
        "schema_version": 1,
        "mode": "frozen30-paired",
        "panel_id": "nemotron3-omni-frozen30",
        "source_manifest_sha256": FROZEN30_INFERENCE_MANIFEST_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conditions": [dict(condition) for condition in CONDITIONS],
        "examples": source_examples,
    }


def verify_candidate_freeze(
    *,
    predictions_path: Path,
    manifest_path: Path,
    checksums_path: Path,
    source_media_root: Path,
    normalized_media_root: Path,
    audit_bundle_root: Path,
    input_bundle_root: Path,
    source_manifest_path: Path = _DEFAULT_SOURCE_MANIFEST,
) -> tuple[dict[str, Any], ...]:
    """Recompute all candidate hashes, bindings, identities, and pair order."""

    candidate_root = checksums_path.parent.resolve()
    expected_paths = {
        "predictions": candidate_root / "predictions.jsonl",
        "manifest": candidate_root / "manifest.json",
        "checksums": candidate_root / "SHA256SUMS",
        "normalized": candidate_root / "normalized",
    }
    if (
        predictions_path.resolve() != expected_paths["predictions"]
        or manifest_path.resolve() != expected_paths["manifest"]
        or checksums_path.resolve() != expected_paths["checksums"]
        or normalized_media_root.resolve() != expected_paths["normalized"]
    ):
        raise ValueError("candidate paths must resolve inside one exact bundle")
    if not predictions_path.is_file() or not manifest_path.is_file():
        raise ValueError("candidate predictions and manifest must be files")
    prediction_bytes = predictions_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    checksums = _checksum_entries(checksums_path)
    candidate_hashes = _file_hashes(candidate_root, description="candidate bundle")
    expected_checksum_files = (
        set(candidate_hashes) - {"./SHA256SUMS"} - _COLLECTOR_CANDIDATE_FILES
    )
    if set(checksums) != expected_checksum_files:
        raise ValueError("candidate checksum exact file set mismatch")
    for relative, digest in checksums.items():
        if candidate_hashes.get(relative) != digest:
            raise ValueError(f"candidate checksum mismatch: {relative}")
    prediction_sha = hashlib.sha256(prediction_bytes).hexdigest()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if checksums["./predictions.jsonl"] != prediction_sha:
        raise ValueError("candidate prediction SHA-256 mismatch")
    if checksums["./manifest.json"] != manifest_sha:
        raise ValueError("candidate manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("candidate manifest must be valid JSON") from error
    expected_manifest_fields = {
        "schema_version",
        "run_id",
        "mode",
        "model_id",
        "model_revision",
        "runtime_image",
        "runtime_image_digest",
        "contract_sha256",
        "contract_file_sha256",
        "predictions_sha256",
        "frozen",
        "completion",
        "reference_answers_available",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise ValueError("candidate manifest schema mismatch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("run_id") != CANDIDATE_RUN_ID
        or manifest.get("mode") != "frozen30-paired"
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("runtime_image") != RUNTIME_IMAGE
        or manifest.get("runtime_image_digest") != RUNTIME_IMAGE_DIGEST
        or manifest.get("frozen") is not True
        or manifest.get("reference_answers_available") is not False
    ):
        raise ValueError("candidate manifest identity mismatch")
    if manifest.get("completion") != {"completed": 60, "expected": 60}:
        raise ValueError("candidate freeze must be 60/60 complete")
    if manifest.get("predictions_sha256") != prediction_sha:
        raise ValueError("candidate manifest prediction SHA-256 binding mismatch")

    contract_path = manifest_path.with_name("contract.json")
    if not contract_path.is_file():
        raise ValueError("candidate contract must be a file")
    contract_bytes = contract_path.read_bytes()
    if (
        manifest.get("contract_file_sha256")
        != hashlib.sha256(contract_bytes).hexdigest()
    ):
        raise ValueError("candidate contract file SHA-256 binding mismatch")
    try:
        contract = json.loads(contract_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("candidate contract must be valid JSON") from error
    if not isinstance(contract, dict):
        raise ValueError("candidate contract must be an object")
    if _contains_label(contract):
        raise ValueError("label-bearing field leaked into candidate contract")
    if (
        manifest.get("contract_sha256")
        != hashlib.sha256(_canonical_json(contract)).hexdigest()
    ):
        raise ValueError("candidate contract canonical SHA-256 binding mismatch")
    try:
        mode, examples, conditions = _validate_run_contract(contract)
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    if (
        mode != "frozen30-paired"
        or conditions != [dict(x) for x in CONDITIONS]
        or contract.get("media_derivation") is None
    ):
        raise ValueError("candidate contract provenance mismatch")
    source_contract = _source_contract_from_normalized(
        contract, source_manifest_path=source_manifest_path
    )
    if (
        contract.get("source_contract_sha256")
        != hashlib.sha256(_canonical_json(source_contract)).hexdigest()
    ):
        raise ValueError("normalized source contract SHA-256 mismatch")
    _validate_input_evidence(
        audit_bundle_root=audit_bundle_root,
        input_bundle_root=input_bundle_root,
        contract=contract,
        source_media_root=source_media_root,
    )
    for example in examples:
        _verified_media_sha256(
            media_root=source_media_root,
            relative_path=example["source_media_path"],
            expected_sha256=example["source_media_sha256"],
            description="source",
        )
        _verified_media_sha256(
            media_root=normalized_media_root,
            relative_path=example["media_path"],
            expected_sha256=example["derived_media_sha256"],
            description="normalized",
        )

    records = _parse_jsonl(prediction_bytes, description="candidate predictions")
    if len(records) != 60:
        raise ValueError("candidate freeze must be 60/60 complete")
    expected_fields = {
        "schema_version",
        "run_id",
        "mode",
        "pair_index",
        "video_id",
        "category",
        "role",
        "condition_id",
        "audio_enabled",
        "request_sha256",
        "model_id",
        "model_revision",
        "runtime_image_digest",
        "status",
        "answer",
        "answer_sha256",
        "error_type",
        "error_message",
        "elapsed_seconds",
    }
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair_index, ((video_id, category, role), example) in enumerate(
        zip(FROZEN30_ROWS, examples, strict=True)
    ):
        pair = records[pair_index * 2 : pair_index * 2 + 2]
        if video_id in seen:
            raise ValueError("candidate duplicate pair identity")
        seen.add(video_id)
        predictions: dict[str, dict[str, Any]] = {}
        for record, condition in zip(pair, CONDITIONS, strict=True):
            condition_id = condition["condition_id"]
            answer = record.get("answer")
            elapsed = record.get("elapsed_seconds")
            expected_request = build_chat_request(
                media_path=_NORMALIZED_MEDIA_ROOT / example["media_path"],
                question=example["question"],
                audio_enabled=condition["audio_enabled"],
            )
            if set(record) != expected_fields:
                raise ValueError("candidate prediction schema mismatch")
            if (
                record.get("schema_version") != 1
                or record.get("run_id") != CANDIDATE_RUN_ID
                or record.get("mode") != "frozen30-paired"
                or record.get("pair_index") != pair_index
                or record.get("video_id") != video_id
                or record.get("category") != category
                or record.get("role") != role
                or record.get("condition_id") != condition_id
                or record.get("audio_enabled") is not condition["audio_enabled"]
                or record.get("model_id") != MODEL_ID
                or record.get("model_revision") != MODEL_REVISION
                or record.get("runtime_image_digest") != RUNTIME_IMAGE_DIGEST
                or record.get("status") != "passed"
            ):
                raise ValueError(
                    "candidate prediction pair order or provenance mismatch"
                )
            if (
                record.get("request_sha256")
                != canonical_request_sha256(expected_request)
                or not isinstance(answer, str)
                or not answer.strip()
                or record.get("answer_sha256")
                != hashlib.sha256(answer.encode()).hexdigest()
                or record.get("error_type") is not None
                or record.get("error_message") is not None
                or not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError("candidate prediction request provenance mismatch")
            predictions[str(condition_id)] = {"completed": True, "answer": answer}
        result.append(
            {
                "row_key": video_id,
                "category": category,
                "role": role,
                "question": example["question"],
                "predictions": predictions,
            }
        )
    if len(seen) != 30:
        raise ValueError("candidate duplicate pair identity")
    _anchor, _status, dstack, anchor_sha, audit_sha = _validate_candidate_anchor(
        candidate_root=candidate_root,
        audit_bundle_root=audit_bundle_root,
    )
    attestation = _load_trusted_attestation()
    _validate_trusted_attestation(
        attestation=attestation,
        audit_bundle_root=audit_bundle_root,
        dstack=dstack,
        workload_anchor_sha256=anchor_sha,
        audit_bundle_sha256=audit_sha,
    )
    if attestation["determination"] != "no-load-bearing-gap":
        raise ValueError("trusted review blocks reference access")
    return tuple(result)


def _validate_references(
    keys: tuple[str, ...], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    if len(rows) != 30:
        raise ValueError("reference loader must return exactly 30 rows")
    expected_categories = {
        video_id: category for video_id, category, _role in FROZEN30_ROWS
    }
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "row_key",
            "category",
            "question",
            "reference",
        }:
            raise ValueError("reference row schema mismatch")
        key = row.get("row_key")
        if not isinstance(key, str) or key in by_key:
            raise ValueError("reference row identities must be unique")
        if row.get("category") != expected_categories.get(key):
            raise ValueError("reference stratum mismatch")
        if not all(
            isinstance(row.get(field), str) and bool(row[field].strip())
            for field in ("question", "reference")
        ):
            raise ValueError("reference text must be non-empty")
        by_key[key] = row
    if tuple(by_key) != keys:
        raise ValueError("reference rows must match frozen order")
    return by_key


def build_blind_grading_bundle(
    *,
    qwen_predictions_path: Path,
    candidate_predictions_path: Path,
    candidate_manifest_path: Path,
    candidate_checksums_path: Path,
    source_media_root: Path,
    normalized_media_root: Path,
    audit_bundle_root: Path,
    input_bundle_root: Path,
    reference_loader: Callable[[tuple[str, ...]], Sequence[Mapping[str, Any]]],
    grader_ids: tuple[str, str],
    shuffle_seeds: tuple[int, int],
    source_manifest_path: Path = _DEFAULT_SOURCE_MANIFEST,
) -> Frozen30GradingBundle:
    """Verify both immutable freezes before making references accessible."""

    qwen_rows = verify_qwen_freeze(qwen_predictions_path)
    candidate_rows = verify_candidate_freeze(
        predictions_path=candidate_predictions_path,
        manifest_path=candidate_manifest_path,
        checksums_path=candidate_checksums_path,
        source_media_root=source_media_root,
        normalized_media_root=normalized_media_root,
        audit_bundle_root=audit_bundle_root,
        input_bundle_root=input_bundle_root,
        source_manifest_path=source_manifest_path,
    )
    keys = tuple(video_id for video_id, _category, _role in FROZEN30_ROWS)
    references = _validate_references(keys, reference_loader(keys))
    if (
        len(grader_ids) != 2
        or len(set(grader_ids)) != 2
        or not all(isinstance(value, str) and value for value in grader_ids)
    ):
        raise ValueError("exactly two independent grader identities are required")
    if len(shuffle_seeds) != 2 or shuffle_seeds[0] == shuffle_seeds[1]:
        raise ValueError("two distinct shuffle seeds are required")
    # Salt caller seeds with a private frozen-artifact digest. This remains
    # deterministic for score-time reconstruction without giving packet
    # recipients a low-entropy seed oracle through the observable item order.
    shuffle_nonce = hashlib.sha256(candidate_predictions_path.read_bytes()).digest()

    private_rows: list[dict[str, Any]] = []
    canonical_keys: list[tuple[str, str]] = []
    item_content: dict[tuple[str, str], dict[str, str]] = {}
    for index, (qwen_row, candidate_row) in enumerate(
        zip(qwen_rows, candidate_rows, strict=True)
    ):
        video_id, category, role = FROZEN30_ROWS[index]
        reference = references[video_id]
        if candidate_row["question"] != reference["question"]:
            raise ValueError("candidate and reference question mismatch")
        predictions = {
            "qwen12": {"completed": True, "answer": qwen_row["answer"]},
            **candidate_row["predictions"],
        }
        row = {
            "row_key": video_id,
            "category": category,
            "role": role,
            "question": reference["question"],
            "reference": reference["reference"],
            "predictions": predictions,
        }
        private_rows.append(row)
        for condition in _ALL_CONDITIONS:
            key = (video_id, condition)
            canonical_keys.append(key)
            item_content[key] = {
                "question": row["question"],
                "candidate_answer": predictions[condition]["answer"],
                "reference_answer": row["reference"],
            }

    packets: dict[str, dict[str, Any]] = {}
    answer_keys: dict[str, dict[str, tuple[str, str]]] = {}
    assignments: list[dict[str, str]] = []
    prior: list[tuple[str, str]] | None = None
    for packet_index, (private_grader_id, seed) in enumerate(
        zip(grader_ids, shuffle_seeds, strict=True), start=1
    ):
        order = list(canonical_keys)
        protected_seed = int.from_bytes(
            hashlib.sha256(
                shuffle_nonce
                + packet_index.to_bytes(1, "big")
                + str(seed).encode("ascii")
            ).digest()
        )
        random.Random(protected_seed).shuffle(order)
        if order == prior:
            order = order[1:] + order[:1]
        prior = order
        grader_alias = f"grader-{packet_index}"
        answer_key = {}
        items = []
        for item_index, key in enumerate(order, start=1):
            item_id = f"item-{item_index:03d}"
            answer_key[item_id] = key
            items.append({"item_id": item_id, **item_content[key]})
        public_content = {
            "grader_id": grader_alias,
            "rubric": dict(_RUBRIC),
            "items": items,
        }
        packet_id = (
            f"blind-{packet_index}-"
            f"{hashlib.sha256(_canonical_json(public_content)).hexdigest()[:16]}"
        )
        packets[grader_alias] = {
            "packet_id": packet_id,
            **public_content,
        }
        answer_keys[grader_alias] = answer_key
        assignments.append(
            {
                "grader_id": private_grader_id,
                "grader_alias": grader_alias,
                "packet_id": packet_id,
                "packet_sha256": hashlib.sha256(
                    _canonical_json(packets[grader_alias])
                ).hexdigest(),
            }
        )
    return Frozen30GradingBundle(
        packets=packets,
        _answer_keys=answer_keys,
        _canonical_keys=tuple(canonical_keys),
        _item_content=item_content,
        _rows=tuple(private_rows),
        _grader_assignment_artifact={
            "schema_version": 1,
            "assignments": assignments,
        },
    )


def validate_private_grader_assignments(
    bundle: Frozen30GradingBundle, artifact: object
) -> None:
    """Require the independently retained grader assignment for this bundle."""

    if artifact != bundle._grader_assignment_artifact:
        raise ValueError("private grader assignment artifact mismatch")


def derive_operational_evidence(
    *,
    candidate_predictions_path: Path,
    candidate_manifest_path: Path,
    audit_bundle_root: Path,
    privacy_review_path: Path | None = None,
) -> OperationalEvidence:
    """Derive promotion inputs only from retained, anchor-bound artifacts."""

    candidate_root = candidate_manifest_path.parent.resolve()
    if candidate_predictions_path.resolve() != candidate_root / "predictions.jsonl":
        raise ValueError("operational candidate paths must share the exact bundle")
    _anchor, status, dstack, anchor_sha, audit_sha = _validate_candidate_anchor(
        candidate_root=candidate_root,
        audit_bundle_root=audit_bundle_root,
    )
    manifest = _load_json_object(
        candidate_manifest_path, description="candidate manifest"
    )
    records = _parse_jsonl(
        candidate_predictions_path.read_bytes(), description="candidate predictions"
    )
    completed = sum(record.get("status") == "passed" for record in records)
    if (
        manifest.get("run_id") != CANDIDATE_RUN_ID
        or manifest.get("completion") != {"completed": 60, "expected": 60}
        or len(records) != completed
        or completed != 60
    ):
        raise ValueError("operational completion differs from the verified candidate")

    expected_status_fields = {
        "schema_version",
        "run_id",
        "collection_kind",
        "scoreable",
        "reason",
    }
    if (
        set(status) != expected_status_fields
        or status.get("schema_version") != 1
        or status.get("run_id") != CANDIDATE_RUN_ID
        or status.get("collection_kind") != "completed-candidate"
        or status.get("scoreable") is not False
        or status.get("reason") != "pending-independent-freeze-verification"
    ):
        raise ValueError(
            "collection status is not the anchor-bound completed candidate"
        )

    expected_dstack_fields = {
        "schema_version",
        "run_id",
        "id",
        "status",
        "status_message",
        "termination_reason",
        "job_termination_reason",
        "submitted_at",
        "finished_at",
        "elapsed_seconds",
        "cost",
        "backend",
        "region",
        "price",
        "instance_type",
        "cpu_count",
        "gpu_count",
        "gpu_name",
        "gpu_vendor",
        "gpu_memory_mib",
        "host_memory_mib",
        "disk_mib",
        "spot",
        "exit_status",
    }
    elapsed = dstack.get("elapsed_seconds")
    cost = dstack.get("cost")
    price = dstack.get("price")
    resource_fields = {
        "cpu_count": _FROZEN_MIN_CPU_COUNT,
        "gpu_memory_mib": _FROZEN_MIN_GPU_MEMORY_MIB,
        "host_memory_mib": _FROZEN_MIN_HOST_MEMORY_MIB,
        "disk_mib": _FROZEN_MIN_DISK_MIB,
    }
    if (
        set(dstack) != expected_dstack_fields
        or dstack.get("schema_version") != 1
        or dstack.get("run_id") != CANDIDATE_RUN_ID
        or dstack.get("status") != "done"
        or type(dstack.get("exit_status")) is not int
        or dstack.get("exit_status") != 0
        or dstack.get("termination_reason") is not None
        or dstack.get("job_termination_reason") is not None
        or dstack.get("backend") not in _FROZEN_BACKENDS
        or dstack.get("instance_type") != "NVIDIA L40S"
        or type(dstack.get("gpu_count")) is not int
        or dstack.get("gpu_count") != 1
        or dstack.get("gpu_name") != "L40S"
        or dstack.get("gpu_vendor") != "nvidia"
        or dstack.get("spot") is not False
        or any(
            type(dstack.get(field)) is not int or dstack[field] < minimum
            for field, minimum in resource_fields.items()
        )
        or any(
            type(value) not in {int, float} or not math.isfinite(value) or value < 0
            for value in (elapsed, cost, price)
        )
        or price > _FROZEN_MAX_HOURLY_PRICE_USD
    ):
        raise ValueError(
            "dstack run evidence is not a verified successful frozen30 run"
        )

    attestation = _load_trusted_attestation()
    _validate_trusted_attestation(
        attestation=attestation,
        audit_bundle_root=audit_bundle_root,
        dstack=dstack,
        workload_anchor_sha256=anchor_sha,
        audit_bundle_sha256=audit_sha,
    )
    if privacy_review_path is not None:
        review = _load_json_object(
            privacy_review_path, description="privacy and audit review copy"
        )
        expected_review_copy = {
            "schema_version": 1,
            "review_type": "nemotron3-frozen30-privacy-audit",
            "run_id": CANDIDATE_RUN_ID,
            "workload_anchor_sha256": anchor_sha,
            "reviewer_id": attestation["reviewer_id"],
            "reviewed_at": attestation["reviewed_at"],
            "determination": attestation["determination"],
        }
        if review != expected_review_copy:
            raise ValueError("privacy review copy differs from trusted attestation")
    return OperationalEvidence._from_verified_artifacts(
        completed=completed,
        expected=60,
        run_status=dstack["status"],
        duration_minutes=float(elapsed) / 60,
        hourly_price_usd=float(price),
        compute_cost_usd=float(cost),
        privacy_or_audit_gap=attestation["determination"] == "load-bearing-gap",
        workload_anchor_sha256=anchor_sha,
        audit_bundle_sha256=audit_sha,
        trust_root_commit=_TRUST_ROOT_COMMIT,
        trust_root_blob=_TRUST_ROOT_BLOB,
        reviewer_id=attestation["reviewer_id"],
        reviewer_key_fingerprint=attestation["reviewer_key_fingerprint"],
        reviewed_at=attestation["reviewed_at"],
        verification_token=_VERIFIED_OPERATIONAL_TOKEN,
    )


def _bound_adjudication_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    content = {key: value for key, value in packet.items() if key != "packet_id"}
    return {
        "packet_id": "adjudication-"
        + hashlib.sha256(_canonical_json(content)).hexdigest()[:16],
        **content,
    }


def reconcile_frozen30_grades(
    bundle: Frozen30GradingBundle,
    submissions: Sequence[Mapping[str, Any]],
    *,
    adjudication: Mapping[str, Any] | None = None,
) -> GradeResolution:
    """Resolve grades only when every submission binds to its exact packet."""

    if len(submissions) != 2:
        raise ValueError("exactly two independent grader submissions are required")
    stripped_submissions = []
    for submission in submissions:
        if not isinstance(submission, Mapping) or set(submission) != {
            "grader_id",
            "packet_id",
            "grades",
        }:
            raise ValueError("grader submissions require exact packet ID bindings")
        grader_id = submission.get("grader_id")
        assignment = next(
            (
                row
                for row in bundle._grader_assignment_artifact["assignments"]
                if row["grader_id"] == grader_id
            ),
            None,
        )
        if assignment is None:
            raise ValueError(
                "submission does not match an independently assigned grader"
            )
        grader_alias = assignment["grader_alias"]
        packet = bundle.packets.get(grader_alias)
        if packet is None or submission.get("packet_id") != packet["packet_id"]:
            raise ValueError("grader submission packet ID mismatch")
        if (
            assignment["packet_id"] != packet["packet_id"]
            or assignment["packet_sha256"]
            != hashlib.sha256(_canonical_json(packet)).hexdigest()
        ):
            raise ValueError("private grader assignment packet binding mismatch")
        stripped_submissions.append(
            {"grader_id": grader_alias, "grades": submission.get("grades")}
        )

    pending = _reconcile_shared_grades(bundle, stripped_submissions)
    if pending.status == "final":
        if adjudication is not None:
            raise ValueError(
                "adjudication is forbidden when there are no disagreements"
            )
        return pending
    if pending.adjudication_packet is None:
        raise RuntimeError("shared reconciler omitted disagreement packet")
    bound_packet = _bound_adjudication_packet(pending.adjudication_packet)
    if adjudication is None:
        return GradeResolution(
            status="adjudication_required",
            final_grades=None,
            adjudication_packet=bound_packet,
        )
    if not isinstance(adjudication, Mapping) or set(adjudication) != {
        "grader_id",
        "packet_id",
        "grades",
    }:
        raise ValueError("adjudication requires an exact packet ID binding")
    if adjudication.get("packet_id") != bound_packet["packet_id"]:
        raise ValueError("adjudication packet ID mismatch")
    assigned_grader_ids = {
        row["grader_id"] for row in bundle._grader_assignment_artifact["assignments"]
    }
    if adjudication.get("grader_id") in assigned_grader_ids:
        raise ValueError("adjudication requires a distinct third grader")
    return _reconcile_shared_grades(
        bundle,
        stripped_submissions,
        adjudication={
            "grader_id": adjudication.get("grader_id"),
            "grades": adjudication.get("grades"),
        },
    )


def apply_decision_rule(
    *,
    improvements: int,
    regressions: int,
    category_regressions_exceed_improvements: bool,
    visual_control_regressions: int,
    exact_match_delta_rows: float | Decimal,
    token_f1_delta: float | Decimal,
    meteor_delta: float | Decimal,
    operational: OperationalEvidence,
) -> dict[str, Any]:
    """Apply the preregistered semantic and operational boundaries verbatim."""

    if (
        not isinstance(operational, OperationalEvidence)
        or operational._verification_token is not _VERIFIED_OPERATIONAL_TOKEN
    ):
        raise TypeError("operational must be validated OperationalEvidence")
    semantic = (
        improvements >= 6
        and regressions <= 2
        and not category_regressions_exceed_improvements
        and visual_control_regressions == 0
        and Decimal(str(exact_match_delta_rows)) >= Decimal("-1")
        and Decimal(str(token_f1_delta)) >= Decimal("-0.02")
        and Decimal(str(meteor_delta)) >= Decimal("-0.02")
    )
    operational_passed = (
        operational.completed == operational.expected == 60
        and operational.run_status == "done"
        and operational.duration_minutes <= 60
        and operational.hourly_price_usd <= _FROZEN_MAX_HOURLY_PRICE_USD
        and operational.compute_cost_usd <= 1.25
        and operational.artifacts_verified
        and not operational.privacy_or_audit_gap
    )
    if semantic and operational_passed:
        disposition = "Current default"
    elif semantic:
        disposition = "Routed candidate"
    else:
        disposition = "Not promoted under the tested condition"
    return {
        "semantic_gate_passed": semantic,
        "operational_gate_passed": operational_passed,
        "disposition": disposition,
    }


def score_replacement(
    bundle: Frozen30GradingBundle,
    resolution: GradeResolution,
    *,
    meteor_fn: Callable[[str, str], float],
    operational: OperationalEvidence,
) -> dict[str, Any]:
    """Produce private per-row and aggregate metrics plus the disposition."""

    if resolution.status != "final" or resolution.final_grades is None:
        raise ValueError("finalized blind grades are required before scoring")
    if set(resolution.final_grades) != set(bundle._canonical_keys) or any(
        grade not in _GRADES for grade in resolution.final_grades.values()
    ):
        raise ValueError("final grades must cover all 90 frozen answers")
    examples = []
    condition_values: dict[str, dict[str, list[float]]] = {
        condition: {"exact_match": [], "token_f1": [], "meteor": []}
        for condition in _ALL_CONDITIONS
    }
    category_values: dict[str, dict[str, dict[str, list[float]]]] = {
        category: {
            condition: {"exact_match": [], "token_f1": [], "meteor": []}
            for condition in _ALL_CONDITIONS
        }
        for _video_id, category, _role in FROZEN30_ROWS
    }
    grade_counts = {
        condition: {"C": 0, "P": 0, "W": 0} for condition in _ALL_CONDITIONS
    }
    category_comparisons: dict[str, dict[str, int]] = {
        category: {"improvements": 0, "regressions": 0, "ties": 0}
        for _video_id, category, _role in FROZEN30_ROWS
    }
    audio_control_by_category = {
        category: {"improvements": 0, "regressions": 0, "ties": 0}
        for _video_id, category, _role in FROZEN30_ROWS
    }
    improvements = regressions = visual_regressions = 0
    for row in bundle._rows:
        metrics = {}
        for condition in _ALL_CONDITIONS:
            answer = row["predictions"][condition]["answer"]
            values = {
                "exact_match": normalized_exact_match(answer, row["reference"]),
                "token_f1": token_f1(answer, row["reference"]),
                "meteor": float(meteor_fn(answer, row["reference"])),
            }
            metrics[condition] = values
            for name, value in values.items():
                condition_values[condition][name].append(value)
                category_values[row["category"]][condition][name].append(value)
            grade_counts[condition][
                resolution.final_grades[(row["row_key"], condition)]
            ] += 1
        qwen_rank = _GRADE_RANK[resolution.final_grades[(row["row_key"], "qwen12")]]
        on_rank = _GRADE_RANK[resolution.final_grades[(row["row_key"], "audio-on")]]
        off_rank = _GRADE_RANK[resolution.final_grades[(row["row_key"], "audio-off")]]
        comparison = "ties"
        if on_rank > qwen_rank:
            comparison = "improvements"
            improvements += 1
        elif on_rank < qwen_rank:
            comparison = "regressions"
            regressions += 1
            if row["role"] == "control":
                visual_regressions += 1
        category_comparisons[row["category"]][comparison] += 1
        audio_comparison = "ties"
        if on_rank > off_rank:
            audio_comparison = "improvements"
        elif on_rank < off_rank:
            audio_comparison = "regressions"
        audio_control_by_category[row["category"]][audio_comparison] += 1
        examples.append({**row, "automatic_metrics": metrics})
    aggregate = {
        condition: {name: fmean(values) for name, values in metrics.items()}
        for condition, metrics in condition_values.items()
    }
    category_metrics = {
        category: {
            condition: {name: fmean(values) for name, values in metrics.items()}
            for condition, metrics in conditions.items()
        }
        for category, conditions in category_values.items()
    }
    token_f1_delta = (
        sum(Decimal(str(value)) for value in condition_values["audio-on"]["token_f1"])
        - sum(Decimal(str(value)) for value in condition_values["qwen12"]["token_f1"])
    ) / Decimal(30)
    meteor_delta = (
        sum(Decimal(str(value)) for value in condition_values["audio-on"]["meteor"])
        - sum(Decimal(str(value)) for value in condition_values["qwen12"]["meteor"])
    ) / Decimal(30)
    decision = apply_decision_rule(
        improvements=improvements,
        regressions=regressions,
        category_regressions_exceed_improvements=any(
            values["regressions"] > values["improvements"]
            for values in category_comparisons.values()
        ),
        visual_control_regressions=visual_regressions,
        exact_match_delta_rows=(
            sum(condition_values["audio-on"]["exact_match"])
            - sum(condition_values["qwen12"]["exact_match"])
        ),
        token_f1_delta=token_f1_delta,
        meteor_delta=meteor_delta,
        operational=operational,
    )
    return {
        "completion": {"completed": operational.completed, "expected": 60},
        "operational_evidence": {
            "run_status": operational.run_status,
            "duration_minutes": operational.duration_minutes,
            "hourly_price_usd": operational.hourly_price_usd,
            "compute_cost_usd": operational.compute_cost_usd,
            "workload_anchor_sha256": operational.workload_anchor_sha256,
            "audit_bundle_sha256": operational.audit_bundle_sha256,
            "trust_root_commit": operational.trust_root_commit,
            "trust_root_blob": operational.trust_root_blob,
            "reviewer_id": operational.reviewer_id,
            "reviewer_key_fingerprint": operational.reviewer_key_fingerprint,
            "reviewed_at": operational.reviewed_at,
            "privacy_or_audit_gap": operational.privacy_or_audit_gap,
        },
        "conditions": aggregate,
        "grade_counts": grade_counts,
        "category_metrics": category_metrics,
        "qwen12_vs_audio_on": {
            "improvements": improvements,
            "regressions": regressions,
            "ties": 30 - improvements - regressions,
            "by_category": category_comparisons,
        },
        "audio_off_vs_audio_on": {"by_category": audio_control_by_category},
        "examples": examples,
        **decision,
    }
