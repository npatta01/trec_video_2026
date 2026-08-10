#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUN_ID="vqa-silver-oracle-election-20260809-b4"
readonly EXPECTED_MEDIA_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/artifacts/vqa2025-training-media-475-20260809-a1/media"
readonly EXPECTED_CONTROLLER_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/vqa-silver-oracle-election-20260809-b4/input-controller"
readonly EXPECTED_OUTPUT_BASE="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/vqa-silver-oracle-election-20260809-b4"
readonly EXPECTED_RUNTIME_IMAGE="docker.io/qwenllm/qwen3-omni@sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
readonly EXPECTED_MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Thinking"
readonly EXPECTED_MODEL_REVISION="2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.24.0"
readonly PYTHON_BIN="/usr/bin/python3"
readonly INPUT_ROOT="/dstack/input"
readonly OUTPUT_DIR="/dstack/output"
readonly WORKLOAD_STDOUT="/dstack/workload.stdout.json"
readonly SOURCE_IDENTITY="/dstack/source-bundle-identity.json"
readonly TELEMETRY_CSV="/dstack/nvidia-smi.csv"
readonly PREFLIGHT_MARKER="/dstack/runtime-preflight-marker.json"

EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-}"
MEDIA_URI="${MEDIA_URI:-}"
CONTROLLER_URI="${CONTROLLER_URI:-}"
OUTPUT_BASE="${OUTPUT_BASE:-}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
SELECTOR_SLICE="${SELECTOR_SLICE:-}"
EXPECTED_PROTOCOL_SHA256="${EXPECTED_PROTOCOL_SHA256:-}"
EXPECTED_SOURCE_BUNDLE_SHA256="${EXPECTED_SOURCE_BUNDLE_SHA256:-}"
SUBMITTED_CODE_REVISION="${SUBMITTED_CODE_REVISION:-}"
EXECUTION_ATTEMPT_ID="${EXECUTION_ATTEMPT_ID:-}"
UVX_BIN=""
REQUEST_NAME=""
REMOTE_OUTPUT_URI=""
FINALIZE_READY="false"
TELEMETRY_PID=""

die() {
  echo "error: $*" >&2
  return 2
}

validate_environment() {
  [[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN secret is unavailable"
  [[ "$EXPERIMENT_RUN_ID" == "$EXPECTED_RUN_ID" ]] || die "unexpected run ID"
  [[ "$MEDIA_URI" == "$EXPECTED_MEDIA_URI" ]] || die "unexpected media URI"
  [[ "$CONTROLLER_URI" == "$EXPECTED_CONTROLLER_URI" ]] \
    || die "unexpected controller URI"
  [[ "$OUTPUT_BASE" == "$EXPECTED_OUTPUT_BASE" ]] || die "unexpected output base"
  [[ "$RUNTIME_IMAGE" == "$EXPECTED_RUNTIME_IMAGE" ]] \
    || die "runtime image mismatch"
  [[ "$MODEL_ID" == "$EXPECTED_MODEL_ID" ]] || die "model ID mismatch"
  [[ "$MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]] \
    || die "model revision mismatch"
  [[ "$EXPECTED_PROTOCOL_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    && [[ "$EXPECTED_PROTOCOL_SHA256" != \
      "0000000000000000000000000000000000000000000000000000000000000000" ]] \
    || die "protocol digest pin is invalid"
  [[ "$EXPECTED_SOURCE_BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    && [[ "$EXPECTED_SOURCE_BUNDLE_SHA256" != \
      "0000000000000000000000000000000000000000000000000000000000000000" ]] \
    || die "source bundle digest pin is invalid"
  [[ "$SUBMITTED_CODE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
    && [[ "$SUBMITTED_CODE_REVISION" != "0000000000000000000000000000000000000000" ]] \
    || die "submitted code revision is invalid"
  [[ "$EXECUTION_ATTEMPT_ID" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] \
    && [[ "$EXECUTION_ATTEMPT_ID" != \
      "00000000-0000-0000-0000-000000000000" ]] \
    || die "execution attempt ID pin is invalid"
  [[ "$PYTHON_BIN" == /usr/bin/python3 && -x "$PYTHON_BIN" ]] \
    || die "pinned image Python is unavailable"
  case "$SELECTOR_SLICE" in
    canary-b4) REQUEST_NAME="selector-canary.jsonl" ;;
    chunk-00) REQUEST_NAME="selector-chunk-00.jsonl" ;;
    chunk-01) REQUEST_NAME="selector-chunk-01.jsonl" ;;
    chunk-02) REQUEST_NAME="selector-chunk-02.jsonl" ;;
    chunk-03) REQUEST_NAME="selector-chunk-03.jsonl" ;;
    *) die "SELECTOR_SLICE must be canary-b4 or chunk-00 through chunk-03" ;;
  esac
  REMOTE_OUTPUT_URI="$OUTPUT_BASE/selector/$SELECTOR_SLICE/$EXECUTION_ATTEMPT_ID"
}

verify_source_bundle() {
  "$PYTHON_BIN" -B - \
    /dstack/run/scripts/run-vqa-silver-oracle-selector-dstack.sh \
    /dstack/run/src/trec_video_2026 \
    "$EXPECTED_SOURCE_BUNDLE_SHA256" \
    "$SUBMITTED_CODE_REVISION" \
    "$SOURCE_IDENTITY" \
    "$EXPERIMENT_RUN_ID-$SELECTOR_SLICE" \
    "$EXECUTION_ATTEMPT_ID" \
    "$REMOTE_OUTPUT_URI" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

runner_path = Path(sys.argv[1])
source_root = Path(sys.argv[2])
expected_digest = sys.argv[3]
submitted_revision = sys.argv[4]
output_path = Path(sys.argv[5])
execution_run_id = sys.argv[6]
execution_attempt_id = sys.argv[7]
remote_output_uri = sys.argv[8]
if runner_path.is_symlink() or not runner_path.is_file():
    raise SystemExit("source bundle runner must be a regular file")
if source_root.is_symlink() or not source_root.is_dir():
    raise SystemExit("source bundle root must be a regular directory")
files = {"scripts/run-vqa-silver-oracle-selector-dstack.sh": runner_path.read_bytes()}
for path in source_root.rglob("*.py"):
    if path.is_symlink() or not path.is_file():
        raise SystemExit("source bundle Python entries must be regular files")
    relative = f"src/trec_video_2026/{path.relative_to(source_root).as_posix()}"
    files[relative] = path.read_bytes()
digest = hashlib.sha256()
for relative, payload in sorted(files.items()):
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(len(payload)).encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    digest.update(b"\0")
actual = digest.hexdigest()
if actual != expected_digest:
    raise SystemExit("submitted source bundle SHA-256 mismatch")
output_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "execution_run_id": execution_run_id,
            "execution_attempt_id": execution_attempt_id,
            "remote_output_uri": remote_output_uri,
            "source_bundle_sha256": actual,
            "submitted_code_revision": submitted_revision,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

bootstrap_hf_cli() {
  "$PYTHON_BIN" -m pip install --no-deps "uv==$UV_VERSION"
  UVX_BIN="$($PYTHON_BIN - <<'PY'
import importlib.metadata
import sysconfig
from pathlib import Path

distribution = importlib.metadata.distribution("uv")
candidate = (Path(sysconfig.get_path("scripts")) / "uvx").resolve()
owned = {distribution.locate_file(path).resolve() for path in distribution.files or ()}
if candidate not in owned or not candidate.is_file():
    raise SystemExit("pinned uv distribution does not own uvx")
print(candidate)
PY
)"
  [[ "$UVX_BIN" = /* && -x "$UVX_BIN" ]] || die "pinned uvx is unavailable"
}

hf_cli() {
  "$UVX_BIN" --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

remote_output_prefix_is_empty() {
  local listing
  listing="$(hf_cli buckets list "$REMOTE_OUTPUT_URI" -R --format json)" \
    || die "could not inspect immutable output prefix"
  [[ -z "$listing" || "$listing" == "[]" ]] \
    || die "immutable output prefix already exists"
}

verify_protocol_identity() {
  "$PYTHON_BIN" - "$INPUT_ROOT/controller/protocol.json" \
    "$EXPECTED_PROTOCOL_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
declared = payload.pop("protocol_sha256", None)
canonical = json.dumps(
    payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
actual = hashlib.sha256(canonical).hexdigest()
if actual != expected or declared != expected:
    raise SystemExit("controller protocol identity mismatch")
PY
}

fetch_slice_inputs() {
  mkdir -p "$INPUT_ROOT/controller" "$INPUT_ROOT/media"
  hf_cli buckets sync "$CONTROLLER_URI" "$INPUT_ROOT/controller"
  (
    cd "$INPUT_ROOT/controller"
    sha256sum -c SHA256SUMS
  )
  verify_protocol_identity

  local -a requested_media include_args
  readarray -t requested_media < <(
    "$PYTHON_BIN" - "$INPUT_ROOT/controller/$REQUEST_NAME" <<'PY'
import json
import sys
from pathlib import PurePosixPath

seen = set()
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        path = PurePosixPath(row["video_path"])
        if len(path.parts) != 2 or path.parts[0] != "media" or path.name in seen:
            raise SystemExit("request media path is unsafe or duplicated")
        seen.add(path.name)
        print(path.name)
PY
  )
  ((${#requested_media[@]} > 0)) || die "slice contains no requested media"
  include_args=()
  local basename
  for basename in "${requested_media[@]}"; do
    include_args+=(--include "$basename")
  done
  hf_cli buckets sync "$MEDIA_URI" "$INPUT_ROOT/media" "${include_args[@]}"
  [[ "$(find "$INPUT_ROOT/media" -maxdepth 1 -type f | wc -l)" \
      -eq "${#requested_media[@]}" ]] \
    || die "downloaded media count does not match slice"
  chmod -R a-w "$INPUT_ROOT"
}

run_runtime_preflight() {
  PYTHONPATH=/dstack/run/src "$PYTHON_BIN" -B -m \
    trec_video_2026.vqa_experiments.qwen3_omni_structured \
      --requests "$INPUT_ROOT/controller/$REQUEST_NAME" \
      --protocol "$INPUT_ROOT/controller/protocol.json" \
      --input-root "$INPUT_ROOT" \
      --run-id "$EXPERIMENT_RUN_ID-$SELECTOR_SLICE" \
      --execution-attempt-id "$EXECUTION_ATTEMPT_ID" \
      --remote-output-uri "$REMOTE_OUTPUT_URI" \
      --source-bundle-sha256 "$EXPECTED_SOURCE_BUNDLE_SHA256" \
      --submitted-code-revision "$SUBMITTED_CODE_REVISION" \
      --preflight-marker "$PREFLIGHT_MARKER" \
      --create-preflight-marker \
      >/dev/null
}

runtime_preflight_marker_is_valid() {
  PYTHONPATH=/dstack/run/src "$PYTHON_BIN" -B -m \
    trec_video_2026.vqa_experiments.qwen3_omni_structured \
      --requests "$INPUT_ROOT/controller/$REQUEST_NAME" \
      --protocol "$INPUT_ROOT/controller/protocol.json" \
      --input-root "$INPUT_ROOT" \
      --run-id "$EXPERIMENT_RUN_ID-$SELECTOR_SLICE" \
      --execution-attempt-id "$EXECUTION_ATTEMPT_ID" \
      --remote-output-uri "$REMOTE_OUTPUT_URI" \
      --source-bundle-sha256 "$EXPECTED_SOURCE_BUNDLE_SHA256" \
      --submitted-code-revision "$SUBMITTED_CODE_REVISION" \
      --preflight-marker "$PREFLIGHT_MARKER" \
      --verify-preflight-marker \
      >/dev/null
}

run_selector() {
  PYTHONPATH=/dstack/run/src "$PYTHON_BIN" -B -m \
    trec_video_2026.vqa_experiments.qwen3_omni_structured \
      --requests "$INPUT_ROOT/controller/$REQUEST_NAME" \
      --protocol "$INPUT_ROOT/controller/protocol.json" \
      --input-root "$INPUT_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$EXPERIMENT_RUN_ID-$SELECTOR_SLICE" \
      --execution-attempt-id "$EXECUTION_ATTEMPT_ID" \
      --remote-output-uri "$REMOTE_OUTPUT_URI" \
      --source-bundle-sha256 "$EXPECTED_SOURCE_BUNDLE_SHA256" \
      --submitted-code-revision "$SUBMITTED_CODE_REVISION" \
      --preflight-marker "$PREFLIGHT_MARKER" \
      >"$WORKLOAD_STDOUT"
}

start_telemetry() {
  nvidia-smi \
    --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,nounits \
    -lms 1000 >"$TELEMETRY_CSV" &
  TELEMETRY_PID=$!
}

stop_telemetry() {
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=""
  fi
}

finalize_on_exit() {
  local workload_rc=$?
  trap - EXIT
  stop_telemetry
  if [[ "$FINALIZE_READY" != "true" ]]; then
    exit "$workload_rc"
  fi
  if ! runtime_preflight_marker_is_valid; then
    if (( workload_rc == 0 )); then
      workload_rc=2
    fi
    exit "$workload_rc"
  fi
  set +e
  local finalization_rc=0 step_rc
  if [[ -f "$WORKLOAD_STDOUT" && ! -L "$WORKLOAD_STDOUT" ]]; then
    cp "$WORKLOAD_STDOUT" "$OUTPUT_DIR/workload.stdout.json"
    step_rc=$?
    if (( step_rc != 0 )); then
      finalization_rc=$step_rc
    fi
  fi
  if [[ -f "$SOURCE_IDENTITY" && ! -L "$SOURCE_IDENTITY" ]]; then
    cp "$SOURCE_IDENTITY" "$OUTPUT_DIR/source-bundle-identity.json"
    step_rc=$?
    if (( step_rc != 0 && finalization_rc == 0 )); then
      finalization_rc=$step_rc
    fi
  fi
  if [[ -f "$PREFLIGHT_MARKER" && ! -L "$PREFLIGHT_MARKER" ]]; then
    cp "$PREFLIGHT_MARKER" "$OUTPUT_DIR/runtime-preflight-marker.json"
    step_rc=$?
    if (( step_rc != 0 && finalization_rc == 0 )); then
      finalization_rc=$step_rc
    fi
  fi
  if [[ -f "$TELEMETRY_CSV" && ! -L "$TELEMETRY_CSV" ]]; then
    cp "$TELEMETRY_CSV" "$OUTPUT_DIR/nvidia-smi.csv"
    step_rc=$?
    if (( step_rc != 0 && finalization_rc == 0 )); then
      finalization_rc=$step_rc
    fi
  fi
  "$PYTHON_BIN" - \
    "$OUTPUT_DIR/terminal-status.json" \
    "$workload_rc" \
    "$EXECUTION_ATTEMPT_ID" \
    "$REMOTE_OUTPUT_URI" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "artifact_sync_attempted": True,
            "workload_returncode": int(sys.argv[2]),
            "execution_attempt_id": sys.argv[3],
            "remote_output_uri": sys.argv[4],
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
  step_rc=$?
  if (( step_rc != 0 && finalization_rc == 0 )); then
    finalization_rc=$step_rc
  fi
  if (( finalization_rc == 0 )); then
    (
      cd "$OUTPUT_DIR"
      find . -type f ! -name SHA256SUMS -print0 \
        | sort -z \
        | xargs -0 sha256sum >SHA256SUMS
    )
    finalization_rc=$?
  fi
  if (( finalization_rc == 0 )); then
    hf_cli buckets sync "$OUTPUT_DIR" "$REMOTE_OUTPUT_URI"
    finalization_rc=$?
  fi
  if (( workload_rc != 0 )); then
    exit "$workload_rc"
  fi
  exit "$finalization_rc"
}

main() {
  mkdir -p "$OUTPUT_DIR"
  validate_environment
  bootstrap_hf_cli
  hf_cli version
  hf_cli auth whoami
  remote_output_prefix_is_empty
  trap finalize_on_exit EXIT
  verify_source_bundle
  fetch_slice_inputs
  run_runtime_preflight
  FINALIZE_READY="true"
  start_telemetry
  run_selector
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
