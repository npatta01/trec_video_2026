#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_DATA_TARGET="/home/npatta01/data/competitions/trec_video_2026/data"
readonly PYTHON_BIN="/usr/bin/python3"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"
readonly EXPANDED12_CANARY_C1_SCHEDULER_ID="nemotron3-expanded12-canary-20260730-c1"
readonly EXPANDED12_CANARY_D1_SCHEDULER_ID="nemotron3-expanded12-canary-20260731-d1"
readonly FROZEN30_CANARY_SCHEDULER_ID="nemotron3-f30-canary-20260801-a1"

RUN_ID="${1:-}"
case "$RUN_ID" in
  nemotron3-omni-runner-canary-20260730-a1|nemotron3-omni-runner-canary-20260730-a2|nemotron3-omni-pilot-20260730-a1|nemotron3-omni-codec-sentinel-20260730-b1|nemotron3-omni-pilot-20260730-b1|nemotron3-omni-expanded12-canary-20260730-c1|nemotron3-omni-expanded12-20260730-c1|nemotron3-omni-expanded12-canary-20260731-d1|nemotron3-omni-expanded12-20260731-d1|nemotron3-omni-frozen30-canary-20260801-a1|nemotron3-omni-frozen30-20260801-a1) ;;
  *) echo "error: unexpected immutable run ID" >&2; exit 2 ;;
esac
DSTACK_RUN_NAME="$RUN_ID"
if [[ "$RUN_ID" == "nemotron3-omni-expanded12-canary-20260730-c1" ]]; then
  DSTACK_RUN_NAME="$EXPANDED12_CANARY_C1_SCHEDULER_ID"
fi
if [[ "$RUN_ID" == "nemotron3-omni-expanded12-canary-20260731-d1" ]]; then
  DSTACK_RUN_NAME="$EXPANDED12_CANARY_D1_SCHEDULER_ID"
fi
if [[ "$RUN_ID" == "nemotron3-omni-frozen30-canary-20260801-a1" ]]; then
  DSTACK_RUN_NAME="$FROZEN30_CANARY_SCHEDULER_ID"
fi
PREVIEW_PATH="${2:-}"
case "$RUN_ID" in
  nemotron3-omni-codec-sentinel-20260730-b1|nemotron3-omni-pilot-20260730-b1)
    [[ -f "$PREVIEW_PATH" ]] \
      || { echo "error: preview output is required for b1" >&2; exit 2; }
    ;;
  nemotron3-omni-expanded12-canary-20260730-c1|nemotron3-omni-expanded12-20260730-c1|nemotron3-omni-expanded12-canary-20260731-d1|nemotron3-omni-expanded12-20260731-d1)
    [[ -f "$PREVIEW_PATH" ]] \
      || { echo "error: preview output is required for c1" >&2; exit 2; }
    ;;
  nemotron3-omni-frozen30-canary-20260801-a1|nemotron3-omni-frozen30-20260801-a1)
    [[ -f "$PREVIEW_PATH" ]] \
      || { echo "error: preview output is required for frozen30 a1" >&2; exit 2; }
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_LINK="$REPO_ROOT/data"
[[ -L "$DATA_LINK" ]] || { echo "error: data must be a symlink" >&2; exit 2; }
[[ "$(readlink -f "$DATA_LINK")" == "$EXPECTED_DATA_TARGET" ]] \
  || { echo "error: data symlink target mismatch" >&2; exit 2; }
[[ -z "${HF_TOKEN:-}" ]] \
  || { echo "error: HF_TOKEN must remain unset outside dstack secrets" >&2; exit 2; }

REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/$RUN_ID"
OUTPUT_DIR="$DATA_LINK/vqa/benchmarks/nemotron3-omni/$RUN_ID"
AUDIT_DIR="$DATA_LINK/vqa/benchmarks/nemotron3-omni/$RUN_ID-audit"
[[ ! -e "$OUTPUT_DIR" ]] \
  || { echo "error: refusing to overwrite $OUTPUT_DIR" >&2; exit 2; }
if [[ "$RUN_ID" == "nemotron3-omni-frozen30-20260801-a1" ]]; then
  [[ ! -e "$AUDIT_DIR" ]] \
    || { echo "error: refusing to overwrite $AUDIT_DIR" >&2; exit 2; }
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
TEMP_OUTPUT_DIR="$(mktemp -d "${OUTPUT_DIR}.tmp.XXXXXX")"
TEMP_AUDIT_DIR="$(mktemp -d /tmp/nemotron3-omni-audit.XXXXXX)"
VERIFY_AUDIT_DIR="$(mktemp -d /tmp/nemotron3-omni-audit-verify.XXXXXX)"
cleanup() {
  rm -rf "$TEMP_OUTPUT_DIR" "$TEMP_AUDIT_DIR" "$VERIFY_AUDIT_DIR"
}
trap cleanup EXIT

uvx_launcher="$(command -v uvx)"
[[ "$uvx_launcher" = /* && -x "$uvx_launcher" ]] \
  || { echo "error: uvx is unavailable" >&2; exit 2; }
hf_cli() {
  "$uvx_launcher" --from "uv==$UV_VERSION" uvx \
    --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

verify_checksum_bundle() {
  local bundle_dir="$1"
  local bundle_name="$2"
  "$PYTHON_BIN" - "$bundle_dir" "$bundle_name" <<'PY'
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1])
bundle_name = sys.argv[2]
checksum_path = root / "SHA256SUMS"
if not checksum_path.is_file():
    raise RuntimeError(f"{bundle_name} checksum file is missing")
declared = set()
for line in checksum_path.read_text(encoding="utf-8").splitlines():
    digest, separator, relative = line.partition("  ")
    path = PurePosixPath(relative)
    if (
        separator != "  "
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not relative.startswith("./")
        or path.is_absolute()
        or ".." in path.parts
        or relative == "./SHA256SUMS"
        or relative in declared
    ):
        raise RuntimeError(f"{bundle_name} checksum manifest is malformed")
    declared.add(relative)
actual = {
    f"./{path.relative_to(root).as_posix()}"
    for path in root.rglob("*")
    if path.is_file()
}
symlinks = sorted(
    f"./{path.relative_to(root).as_posix()}"
    for path in root.rglob("*")
    if path.is_symlink()
)
if symlinks:
    raise RuntimeError(f"{bundle_name} contains forbidden symlink: {symlinks}")
expected = {"./SHA256SUMS", *declared}
if actual != expected:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise RuntimeError(
        f"{bundle_name} checksum file set mismatch: missing={missing}, extra={extra}"
    )
PY
  (
    cd "$bundle_dir"
    sha256sum -c SHA256SUMS
  )
}

create_workload_anchor() {
  local candidate_dir="$1"
  local audit_dir="$2"
  "$PYTHON_BIN" - "$candidate_dir" "$audit_dir" "$RUN_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

candidate = Path(sys.argv[1])
audit = Path(sys.argv[2])
run_id = sys.argv[3]
evidence_names = (
    "INPUTS_COMPLETE.json",
    "collection-status.json",
    "dstack-run.json",
    "input-contract.json",
    "preview.txt",
)
for root, description in ((candidate, "candidate"), (audit, "audit")):
    symlinks = sorted(path.as_posix() for path in root.rglob("*") if path.is_symlink())
    if symlinks:
        raise RuntimeError(f"{description} contains forbidden symlink: {symlinks}")
candidate_files = {
    f"./{path.relative_to(candidate).as_posix()}": hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(candidate.rglob("*"))
    if path.is_file()
}
audit_evidence_files = {
    f"./{name}": hashlib.sha256((audit / name).read_bytes()).hexdigest()
    for name in evidence_names
}
anchor = {
    "schema_version": 1,
    "anchor_type": "nemotron3-frozen30-collected-workload",
    "run_id": run_id,
    "candidate_file_count": len(candidate_files),
    "candidate_files": candidate_files,
    "audit_evidence_file_count": len(audit_evidence_files),
    "audit_evidence_files": audit_evidence_files,
}
(audit / "workload-anchor.json").write_text(
    json.dumps(anchor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

verify_workload_anchor() {
  local candidate_dir="$1"
  local audit_dir="$2"
  "$PYTHON_BIN" - "$candidate_dir" "$audit_dir" "$RUN_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

candidate = Path(sys.argv[1])
audit = Path(sys.argv[2])
run_id = sys.argv[3]
expected_audit = {
    "INPUTS_COMPLETE.json",
    "SHA256SUMS",
    "collection-status.json",
    "dstack-run.json",
    "input-contract.json",
    "preview.txt",
    "workload-anchor.json",
}
actual_audit = {path.name for path in audit.iterdir() if path.is_file()}
if actual_audit != expected_audit or any(path.is_dir() for path in audit.iterdir()):
    raise RuntimeError("audit exact file set mismatch")
anchor = json.loads((audit / "workload-anchor.json").read_text(encoding="utf-8"))
candidate_files = {
    f"./{path.relative_to(candidate).as_posix()}": hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(candidate.rglob("*"))
    if path.is_file()
}
if (
    anchor.get("run_id") != run_id
    or anchor.get("candidate_file_count") != len(candidate_files)
    or anchor.get("candidate_files") != candidate_files
):
    raise RuntimeError("retained workload anchor mismatch")
for relative, digest in anchor.get("audit_evidence_files", {}).items():
    path = audit / relative.removeprefix("./")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("retained workload audit evidence mismatch")
PY
}

classify_frozen30_bundle() {
  local bundle_dir="$1"
  "$PYTHON_BIN" - "$bundle_dir" "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_id = sys.argv[2]
technical_path = root / "technical-error.json"
status = {
    "schema_version": 1,
    "run_id": run_id,
    "collection_kind": "terminal-non-scoreable",
    "scoreable": False,
}
if technical_path.exists():
    technical = json.loads(technical_path.read_text(encoding="utf-8"))
    if (
        set(technical)
        != {
            "schema_version",
            "run_id",
            "failure_stage",
            "exit_status",
            "credentials_recorded",
        }
        or technical.get("schema_version") != 1
        or technical.get("run_id") != run_id
        or not isinstance(technical.get("failure_stage"), str)
        or not technical["failure_stage"]
        or type(technical.get("exit_status")) is not int
        or technical["exit_status"] == 0
        or technical.get("credentials_recorded") is not False
    ):
        raise RuntimeError("frozen30 technical-error artifact is invalid")
    status["reason"] = "technical-error"
else:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("frozen30 terminal bundle omits manifest and technical-error")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("completion") == {"completed": 60, "expected": 60}:
        normalized_root = root / "normalized"
        normalized = sorted(
            path.relative_to(normalized_root).as_posix()
            for path in normalized_root.rglob("*")
            if path.is_file()
        ) if normalized_root.is_dir() else []
        if len(normalized) != 30 or any("/" in path or not path.endswith(".mp4") for path in normalized):
            raise RuntimeError(
                "completed frozen30 bundle must preserve exactly 30 normalized artifacts"
            )
        status.update(
            {
                "collection_kind": "completed-candidate",
                "reason": "pending-independent-freeze-verification",
            }
        )
    else:
        status["reason"] = "incomplete-workload"
(root / "collection-status.json").write_text(
    json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

hf_cli buckets list "${REMOTE_URI}/outputs" --format json >/dev/null
hf_cli buckets sync "${REMOTE_URI}/outputs" "$TEMP_OUTPUT_DIR"
verify_checksum_bundle "$TEMP_OUTPUT_DIR" "workload"
if [[ "$RUN_ID" == "nemotron3-omni-frozen30-20260801-a1" ]]; then
  classify_frozen30_bundle "$TEMP_OUTPUT_DIR"
fi
if [[ -n "$PREVIEW_PATH" ]]; then
  cp -- "$PREVIEW_PATH" "$TEMP_AUDIT_DIR/preview.txt"
  dstack run get "$DSTACK_RUN_NAME" --json \
    | PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" \
        -m trec_video_2026.vqa_experiments.dstack_metadata \
        --run-id "$DSTACK_RUN_NAME" --output "$TEMP_AUDIT_DIR/dstack-run.json"
  if [[ -f "$TEMP_OUTPUT_DIR/collection-status.json" ]]; then
    cp -- "$TEMP_OUTPUT_DIR/collection-status.json" \
      "$TEMP_AUDIT_DIR/collection-status.json"
  fi
  cp -- "$TEMP_AUDIT_DIR/dstack-run.json" "$TEMP_OUTPUT_DIR/dstack-run.json"
  if [[ "$RUN_ID" == "nemotron3-omni-frozen30-20260801-a1" ]]; then
    hf_cli buckets cp "${REMOTE_URI}/inputs/INPUTS_COMPLETE.json" \
      "$TEMP_AUDIT_DIR/INPUTS_COMPLETE.json"
    hf_cli buckets cp "${REMOTE_URI}/inputs/contract.json" \
      "$TEMP_AUDIT_DIR/input-contract.json"
  fi
else
  dstack run get "$DSTACK_RUN_NAME" --json \
  | PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" \
      -m trec_video_2026.vqa_experiments.dstack_metadata \
      --run-id "$DSTACK_RUN_NAME" --output "$TEMP_OUTPUT_DIR/dstack-run.json"
fi
(
  cd "$TEMP_OUTPUT_DIR"
  find . -type f ! -name artifact-hashes.sha256 -print0 \
    | sort -z | xargs -0 sha256sum >artifact-hashes.sha256
  sha256sum -c artifact-hashes.sha256
)
if [[ -n "$PREVIEW_PATH" ]]; then
  if [[ "$RUN_ID" == "nemotron3-omni-frozen30-20260801-a1" ]]; then
    create_workload_anchor "$TEMP_OUTPUT_DIR" "$TEMP_AUDIT_DIR"
  fi
  (
    cd "$TEMP_AUDIT_DIR"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum -c SHA256SUMS
    chmod a-w ./*
  )
  audit_listing="$(hf_cli buckets list "${REMOTE_URI}/audit" --format quiet)"
  [[ -z "$audit_listing" ]] \
    || { echo "error: immutable audit prefix already exists" >&2; exit 2; }
  hf_cli buckets sync "$TEMP_AUDIT_DIR" "${REMOTE_URI}/audit"
  hf_cli buckets list "${REMOTE_URI}/audit" --format json
  hf_cli buckets sync "${REMOTE_URI}/audit" "$VERIFY_AUDIT_DIR"
  verify_checksum_bundle "$VERIFY_AUDIT_DIR" "audit"
  if [[ "$RUN_ID" == "nemotron3-omni-frozen30-20260801-a1" ]]; then
    verify_workload_anchor "$TEMP_OUTPUT_DIR" "$VERIFY_AUDIT_DIR"
  fi
fi
mv -T --no-clobber "$TEMP_OUTPUT_DIR" "$OUTPUT_DIR"
[[ -d "$OUTPUT_DIR" && ! -e "$TEMP_OUTPUT_DIR" ]] \
  || { echo "error: refusing to overwrite or nest output" >&2; exit 2; }
if [[ "$RUN_ID" == "nemotron3-omni-frozen30-20260801-a1" ]]; then
  mv -T --no-clobber "$VERIFY_AUDIT_DIR" "$AUDIT_DIR"
  [[ -d "$AUDIT_DIR" && ! -e "$VERIFY_AUDIT_DIR" ]] \
    || { echo "error: refusing to overwrite or nest audit" >&2; exit 2; }
fi
trap - EXIT
rm -rf "$TEMP_AUDIT_DIR" "$VERIFY_AUDIT_DIR"
echo "$OUTPUT_DIR"
