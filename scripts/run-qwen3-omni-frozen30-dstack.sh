#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUNTIME_IMAGE="docker.io/qwenllm/qwen3-omni@sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
readonly EXPECTED_MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Thinking"
readonly EXPECTED_MODEL_REVISION="2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"
readonly PYTHON_BIN="/usr/bin/python3"

EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-a4}"
EXPECTED_RUN_ID=""
EXPECTED_INPUT_URI=""
EXPECTED_ARCHIVE_URI=""
EXPECTED_REMOTE_URI=""
PINNED_CONTRACT_SHA256=""
PINNED_BUNDLE_MANIFEST_SHA256=""
EXPECTED_BUNDLE_ARCHIVE_SHA256=""
EXPECTED_BUNDLE_ARCHIVE_SIZE=""
DOWNLOAD_REMOTE_MANIFEST=""
case "$EXPERIMENT_PROFILE" in
  a4)
    EXPECTED_RUN_ID="qwen3-omni-frozen30-audio-20260731-a4"
    EXPECTED_INPUT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/qwen3-omni-frozen30-audio-20260731-a4"
    EXPECTED_ARCHIVE_URI="$EXPECTED_INPUT_URI/evidence.tar"
    EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-frozen30-audio-20260731-a4"
    PINNED_CONTRACT_SHA256="9cfa26eacecde5bd8c916823ce2a59e74ded7f71ae978b973e306a0525b24bb5"
    PINNED_BUNDLE_MANIFEST_SHA256="3f88bb004c9d95c433e544ddf0cf6b314deffd0d5a22e5adb2d3e1d09613ca74"
    EXPECTED_BUNDLE_ARCHIVE_SHA256="9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943"
    EXPECTED_BUNDLE_ARCHIVE_SIZE="318177280"
    DOWNLOAD_REMOTE_MANIFEST="true"
    ;;
  stage1)
    EXPECTED_RUN_ID="qwen3-omni-f30-calibrated-20260801-s1"
    EXPECTED_INPUT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/qwen3-omni-frozen30-audio-20260731-a4"
    EXPECTED_ARCHIVE_URI="$EXPECTED_INPUT_URI/evidence.tar"
    EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-f30-calibrated-20260801-s1"
    EXPECTED_BUNDLE_ARCHIVE_SHA256="9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943"
    EXPECTED_BUNDLE_ARCHIVE_SIZE="318177280"
    DOWNLOAD_REMOTE_MANIFEST="false"
    ;;
  stage1_repair)
    EXPECTED_RUN_ID="qwen3-omni-f30-calibrated-20260801-s2-nothink"
    EXPECTED_INPUT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/qwen3-omni-frozen30-audio-20260731-a4"
    EXPECTED_ARCHIVE_URI="$EXPECTED_INPUT_URI/evidence.tar"
    EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-f30-calibrated-20260801-s2-nothink"
    EXPECTED_BUNDLE_ARCHIVE_SHA256="9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943"
    EXPECTED_BUNDLE_ARCHIVE_SIZE="318177280"
    DOWNLOAD_REMOTE_MANIFEST="false"
    ;;
  stage1_sampled)
    EXPECTED_RUN_ID="qwen3-omni-f30-calibrated-20260801-s3-sampled"
    EXPECTED_INPUT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/qwen3-omni-frozen30-audio-20260731-a4"
    EXPECTED_ARCHIVE_URI="$EXPECTED_INPUT_URI/evidence.tar"
    EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-f30-calibrated-20260801-s3-sampled"
    EXPECTED_BUNDLE_ARCHIVE_SHA256="9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943"
    EXPECTED_BUNDLE_ARCHIVE_SIZE="318177280"
    DOWNLOAD_REMOTE_MANIFEST="false"
    ;;
  stage1_explicit_sampling)
    EXPECTED_RUN_ID="qwen3-omni-f30-calibrated-20260801-s4-explicit-sampling"
    EXPECTED_INPUT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/qwen3-omni-frozen30-audio-20260731-a4"
    EXPECTED_ARCHIVE_URI="$EXPECTED_INPUT_URI/evidence.tar"
    EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-f30-calibrated-20260801-s4-explicit-sampling"
    EXPECTED_BUNDLE_ARCHIVE_SHA256="9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943"
    EXPECTED_BUNDLE_ARCHIVE_SIZE="318177280"
    DOWNLOAD_REMOTE_MANIFEST="false"
    ;;
  stage1_seed42)
    EXPECTED_RUN_ID="qwen3-omni-f30-calibrated-20260801-s5-seed42"
    EXPECTED_INPUT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/inputs/qwen3-omni-frozen30-audio-20260731-a4"
    EXPECTED_ARCHIVE_URI="$EXPECTED_INPUT_URI/evidence.tar"
    EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-f30-calibrated-20260801-s5-seed42"
    EXPECTED_BUNDLE_ARCHIVE_SHA256="9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943"
    EXPECTED_BUNDLE_ARCHIVE_SIZE="318177280"
    DOWNLOAD_REMOTE_MANIFEST="false"
    ;;
  *)
    echo "error: EXPERIMENT_PROFILE must be a4, stage1, stage1_repair, stage1_sampled, stage1_explicit_sampling, or stage1_seed42" >&2
    exit 2
    ;;
esac
readonly EXPERIMENT_PROFILE EXPECTED_RUN_ID EXPECTED_INPUT_URI EXPECTED_ARCHIVE_URI
readonly EXPECTED_REMOTE_URI PINNED_CONTRACT_SHA256
readonly PINNED_BUNDLE_MANIFEST_SHA256 EXPECTED_BUNDLE_ARCHIVE_SHA256
readonly EXPECTED_BUNDLE_ARCHIVE_SIZE DOWNLOAD_REMOTE_MANIFEST

UV_BIN=""
UVX_BIN=""
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-}"
HF_INPUT_URI="${HF_INPUT_URI:-}"
HF_ARCHIVE_URI="${HF_ARCHIVE_URI:-$EXPECTED_ARCHIVE_URI}"
HF_REMOTE_URI="${HF_REMOTE_URI:-}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
EXPECTED_CONTRACT="${EXPECTED_CONTRACT:-}"
EXPECTED_BUNDLE_MANIFEST="${EXPECTED_BUNDLE_MANIFEST:-}"
EXPECTED_BUNDLE_ARCHIVE="${EXPECTED_BUNDLE_ARCHIVE:-}"
EXPECTED_BUNDLE_SIZE="${EXPECTED_BUNDLE_SIZE:-}"
CONTRACT_PATH="${CONTRACT_PATH:-/dstack/input/contract.json}"
BUNDLE_MANIFEST_PATH="${BUNDLE_MANIFEST_PATH:-/dstack/input/bundle-manifest.json}"
BUNDLE_STAGING_DIR="${BUNDLE_STAGING_DIR:-/dstack/bundle}"
DATA_ROOT="${DATA_ROOT:-/dstack/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"
SOURCE_BUNDLE_RUNNER="${SOURCE_BUNDLE_RUNNER:-/dstack/run/scripts/run-qwen3-omni-frozen30-dstack.sh}"
SOURCE_BUNDLE_ROOT="${SOURCE_BUNDLE_ROOT:-/dstack/run/src/trec_video_2026}"
EXPECTED_SOURCE_BUNDLE_SHA256="${EXPECTED_SOURCE_BUNDLE_SHA256:-}"
SUBMITTED_CODE_REVISION="${SUBMITTED_CODE_REVISION:-}"

die() {
  echo "error: $*" >&2
  exit 2
}

canonical_json_sha256() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
canonical = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
).encode("utf-8")
print(hashlib.sha256(canonical).hexdigest())
PY
}

is_sha256() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

manifest_archive_uri() {
  "$PYTHON_BIN" - "$BUNDLE_MANIFEST_PATH" "$EXPERIMENT_PROFILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
archive = manifest.get("archive")
profile = sys.argv[2]
expected_schema = 1 if profile == "a4" else 2
expected_archive_keys = {"name", "sha256", "size"}
if expected_schema == 2:
    expected_archive_keys.add("uri")
if (
    set(manifest) != {"schema_version", "contract_sha256", "archive", "members"}
    or manifest.get("schema_version") != expected_schema
    or not isinstance(archive, dict)
    or set(archive) != expected_archive_keys
    or archive.get("name") != "evidence.tar"
):
    raise SystemExit("mapped bundle manifest has an invalid profile schema")
print(archive.get("uri", ""))
PY
}

validate_environment() {
  local mapped_archive_uri
  [[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] \
    || die "PYTHON_BIN must name the exact absolute image interpreter"
  [[ "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" == "$PYTHON_BIN" ]] \
    || die "image interpreter identity mismatch"
  [[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN secret is unavailable"
  [[ "$EXPERIMENT_RUN_ID" == "$EXPECTED_RUN_ID" ]] || die "unexpected run ID"
  [[ "$HF_INPUT_URI" == "$EXPECTED_INPUT_URI" ]] || die "unexpected input URI"
  [[ "$HF_ARCHIVE_URI" == "$EXPECTED_ARCHIVE_URI" ]] || die "unexpected archive URI"
  [[ "$HF_REMOTE_URI" == "$EXPECTED_REMOTE_URI" ]] || die "unexpected output URI"
  [[ "$RUNTIME_IMAGE" == "$EXPECTED_RUNTIME_IMAGE" ]] || die "runtime image mismatch"
  [[ "$MODEL_ID" == "$EXPECTED_MODEL_ID" ]] || die "model ID mismatch"
  [[ "$MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]] || die "model revision mismatch"
  is_sha256 "$EXPECTED_CONTRACT" || die "contract digest pin is invalid"
  is_sha256 "$EXPECTED_BUNDLE_MANIFEST" || die "bundle manifest digest pin is invalid"
  if [[ "$EXPERIMENT_PROFILE" == "a4" ]]; then
    [[ "$EXPECTED_CONTRACT" == "$PINNED_CONTRACT_SHA256" ]] \
      || die "contract digest pin mismatch"
    [[ "$EXPECTED_BUNDLE_MANIFEST" == "$PINNED_BUNDLE_MANIFEST_SHA256" ]] \
      || die "bundle manifest digest pin mismatch"
  fi
  [[ "$EXPECTED_BUNDLE_ARCHIVE" == "$EXPECTED_BUNDLE_ARCHIVE_SHA256" ]] \
    || die "bundle archive digest pin mismatch"
  [[ "$EXPECTED_BUNDLE_SIZE" == "$EXPECTED_BUNDLE_ARCHIVE_SIZE" ]] \
    || die "bundle archive size pin mismatch"
  [[ "$CONTRACT_PATH" == /dstack/input/contract.json ]] || die "contract path mismatch"
  [[ "$BUNDLE_MANIFEST_PATH" == /dstack/input/bundle-manifest.json ]] \
    || die "bundle manifest path mismatch"
  [[ "$BUNDLE_STAGING_DIR" == /dstack/bundle ]] || die "bundle staging path mismatch"
  [[ "$DATA_ROOT" == /dstack/input ]] || die "data root mismatch"
  [[ "$OUTPUT_DIR" == /dstack/output ]] || die "output path mismatch"
  if [[ "$EXPERIMENT_PROFILE" == stage1* ]]; then
    [[ "$SOURCE_BUNDLE_RUNNER" == /dstack/run/scripts/run-qwen3-omni-frozen30-dstack.sh ]] \
      || die "source bundle runner path mismatch"
    [[ "$SOURCE_BUNDLE_ROOT" == /dstack/run/src/trec_video_2026 ]] \
      || die "source bundle root path mismatch"
    is_sha256 "$EXPECTED_SOURCE_BUNDLE_SHA256" \
      || die "source bundle digest pin is invalid"
    [[ "$SUBMITTED_CODE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
      || die "submitted code revision is invalid"
  fi
  [[ -f "$CONTRACT_PATH" && ! -L "$CONTRACT_PATH" ]] || die "contract is missing"
  [[ -f "$BUNDLE_MANIFEST_PATH" && ! -L "$BUNDLE_MANIFEST_PATH" ]] \
    || die "bundle manifest is missing"
  [[ "$(canonical_json_sha256 "$CONTRACT_PATH")" == "$EXPECTED_CONTRACT" ]] \
    || die "uploaded contract digest does not match"
  [[ "$(canonical_json_sha256 "$BUNDLE_MANIFEST_PATH")" == "$EXPECTED_BUNDLE_MANIFEST" ]] \
    || die "uploaded bundle manifest digest does not match"
  mapped_archive_uri="$(manifest_archive_uri)" \
    || die "mapped bundle manifest archive metadata is invalid"
  if [[ "$EXPERIMENT_PROFILE" == stage1* ]]; then
    [[ "$mapped_archive_uri" == "$HF_ARCHIVE_URI" ]] \
      || die "mapped bundle manifest archive URI does not match"
  else
    [[ -z "$mapped_archive_uri" ]] \
      || die "A4 bundle manifest must not carry a remote URI"
  fi
  mkdir -p "$OUTPUT_DIR"
}

verify_source_bundle() {
  [[ "$EXPERIMENT_PROFILE" == stage1* ]] || return 0
  "$PYTHON_BIN" -B - \
    "$SOURCE_BUNDLE_RUNNER" \
    "$SOURCE_BUNDLE_ROOT" \
    "$EXPECTED_SOURCE_BUNDLE_SHA256" \
    "$SUBMITTED_CODE_REVISION" \
    "$OUTPUT_DIR/source-bundle-identity.json" <<'PY' \
    || die "source bundle identity verification failed"
import hashlib
import json
import sys
from pathlib import Path

runner_path = Path(sys.argv[1])
source_root = Path(sys.argv[2])
expected_digest = sys.argv[3]
submitted_revision = sys.argv[4]
output_path = Path(sys.argv[5])
if runner_path.is_symlink() or not runner_path.is_file():
    raise SystemExit("source bundle runner must be a regular non-symlink file")
if source_root.is_symlink() or not source_root.is_dir():
    raise SystemExit("source bundle root must be a regular non-symlink directory")
files = [("scripts/run-qwen3-omni-frozen30-dstack.sh", runner_path)]
for path in source_root.rglob("*.py"):
    if path.is_symlink() or not path.is_file():
        raise SystemExit("source bundle Python entries must be regular files")
    files.append(
        (f"src/trec_video_2026/{path.relative_to(source_root).as_posix()}", path)
    )
if len(files) == 1:
    raise SystemExit("source bundle contains no Python files")
digest = hashlib.sha256()
for relative, path in sorted(files):
    payload = path.read_bytes()
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(len(payload)).encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")
observed_digest = digest.hexdigest()
if observed_digest != expected_digest:
    raise SystemExit("source bundle SHA-256 mismatch")
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("x", encoding="utf-8") as handle:
    json.dump(
        {
            "schema_version": 1,
            "source_bundle_sha256": observed_digest,
            "submitted_code_revision": submitted_revision,
        },
        handle,
        sort_keys=True,
    )
    handle.write("\n")
PY
  echo FROZEN30_SOURCE_BUNDLE_VERIFIED
}

bootstrap_uv() {
  "$PYTHON_BIN" -m pip install --no-deps "uv==$UV_VERSION"
  [[ "$("$PYTHON_BIN" -c \
    'import importlib.metadata; print(importlib.metadata.version("uv"))')" == "$UV_VERSION" ]] \
    || die "uv bootstrap attestation failed"
  local -a resolved
  mapfile -t resolved < <("$PYTHON_BIN" - <<'PY'
import importlib.metadata
import sysconfig
from pathlib import Path

distribution = importlib.metadata.distribution("uv")
scripts = Path(sysconfig.get_path("scripts")).resolve()
owned = {distribution.locate_file(path).resolve() for path in distribution.files or ()}
for name in ("uv", "uvx"):
    executable = (scripts / name).resolve()
    if executable not in owned:
        raise RuntimeError(f"pinned uv distribution does not own {name}")
    print(executable)
PY
  ) || die "could not resolve pinned uv executables"
  [[ "${#resolved[@]}" -eq 2 ]] || die "pinned uv executables are incomplete"
  UV_BIN="${resolved[0]}"
  UVX_BIN="${resolved[1]}"
  [[ "$UV_BIN" = /* && -x "$UV_BIN" && "$UVX_BIN" = /* && -x "$UVX_BIN" ]] \
    || die "pinned uv executables are unavailable"
}

hf_cli() {
  "$UVX_BIN" --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

remote_output_prefix_is_empty() {
  local listing rc
  set +e
  listing="$(hf_cli buckets list "$HF_REMOTE_URI" --format json 2>/dev/null)"
  rc=$?
  set -e
  (( rc == 0 )) || die "could not verify immutable output prefix (rc=$rc)"
  [[ -z "$listing" || "$listing" == "[]" ]]
}

download_and_reconstruct_bundle() {
  [[ ! -e "$BUNDLE_STAGING_DIR" && ! -L "$BUNDLE_STAGING_DIR" ]] \
    || die "bundle staging directory already exists"
  [[ ! -e "$DATA_ROOT/vqa" && ! -L "$DATA_ROOT/vqa" ]] \
    || die "data root already contains evidence"
  mkdir "$BUNDLE_STAGING_DIR"
  if [[ "$DOWNLOAD_REMOTE_MANIFEST" == "true" ]]; then
    hf_cli buckets cp "$HF_INPUT_URI/manifest.json" \
      "$BUNDLE_STAGING_DIR/remote-manifest.json" --quiet \
      || die "private bundle manifest download failed"
    cmp "$BUNDLE_MANIFEST_PATH" "$BUNDLE_STAGING_DIR/remote-manifest.json" \
      || die "remote bundle manifest does not match sealed manifest"
  fi
  hf_cli buckets cp "$HF_ARCHIVE_URI" \
    "$BUNDLE_STAGING_DIR/evidence.tar" --quiet \
    || die "private bundle archive download failed"
  "$PYTHON_BIN" -m \
    trec_video_2026.vqa_experiments.qwen3_omni_frozen30_bundle reconstruct \
    --contract "$CONTRACT_PATH" \
    --data-root "$DATA_ROOT" \
    --manifest "$BUNDLE_MANIFEST_PATH" \
    --archive "$BUNDLE_STAGING_DIR/evidence.tar" \
    >"$OUTPUT_DIR/bundle-reconstruction.json" \
    || die "private bundle reconstruction failed"
  echo FROZEN30_BUNDLE_VERIFIED
}

run_frozen30_child() {
  "$UV_BIN" run \
    --no-project \
    --no-managed-python \
    --no-python-downloads \
    --python "$PYTHON_BIN" \
    python -m trec_video_2026.vqa_experiments.qwen3_omni_frozen30_dstack \
      --contract "$CONTRACT_PATH" \
      --data-root "$DATA_ROOT" \
      --run-id "$EXPERIMENT_RUN_ID" \
      --experiment "$EXPERIMENT_PROFILE" \
      "$@"
}

run_local_preflight() {
  run_frozen30_child --preflight >/dev/null \
    || die "frozen30 local preflight failed"
  echo FROZEN30_PREFLIGHT_PASSED
}

run_workload_preserving_rc() {
  local preflight_rc
  set +e
  run_frozen30_child --preflight >/dev/null
  preflight_rc=$?
  set -e
  if (( preflight_rc != 0 )); then
    workload_rc="$preflight_rc"
    return 0
  fi
  echo FROZEN30_MODEL_WORKLOAD_STARTING
  set +e
  run_frozen30_child --output-dir "$OUTPUT_DIR" >"$OUTPUT_DIR/workload.stdout.json"
  workload_rc=$?
  set -e
}

write_terminal_status() {
  local workload_rc="$1"
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {"artifact_sync_attempted": True, "workload_returncode": int(sys.argv[2])},
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

write_checksums() {
  (
    cd "$OUTPUT_DIR"
    find . -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum >SHA256SUMS
  )
}

finalize_and_sync() {
  local workload_rc="$1"
  local finalization_rc=0 step_rc sync_rc
  set +e
  write_terminal_status "$workload_rc"
  step_rc=$?
  if (( step_rc != 0 && finalization_rc == 0 )); then
    finalization_rc="$step_rc"
  fi
  write_checksums
  step_rc=$?
  if (( step_rc != 0 && finalization_rc == 0 )); then
    finalization_rc="$step_rc"
  fi
  sync_rc=0
  if (( finalization_rc == 0 )); then
    hf_cli buckets sync "$OUTPUT_DIR" "$HF_REMOTE_URI"
    sync_rc=$?
  fi
  set -e
  (( workload_rc == 0 )) || return "$workload_rc"
  (( finalization_rc == 0 )) || return "$finalization_rc"
  return "$sync_rc"
}

prepare_runtime() {
  validate_environment
  verify_source_bundle
  bootstrap_uv
  hf_cli version
  hf_cli auth whoami
  download_and_reconstruct_bundle
  chmod -R a-w "$DATA_ROOT"
}

run_bundle_canary() {
  prepare_runtime
  run_local_preflight
  echo FROZEN30_BUNDLE_CANARY_DONE
}

run_probe() {
  local workload_rc=0
  validate_environment
  verify_source_bundle
  bootstrap_uv
  hf_cli version
  hf_cli auth whoami
  remote_output_prefix_is_empty || die "immutable output prefix already exists"
  download_and_reconstruct_bundle
  chmod -R a-w "$DATA_ROOT"
  run_workload_preserving_rc
  finalize_and_sync "$workload_rc"
}

main() {
  case "${1:-}" in
    bundle-canary)
      run_bundle_canary
      ;;
    run)
      run_probe
      ;;
    *)
      die "usage: $0 {bundle-canary|run}"
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
