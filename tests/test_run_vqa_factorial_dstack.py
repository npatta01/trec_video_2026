from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-vqa-factorial-dstack.sh"
BUILDER = ROOT / "scripts/build_vqa_factorial_manifest.py"
PROFILE = ROOT / "configs/vqa/vqa_factorial_canary_profile.json"
MANIFEST = ROOT / "configs/vqa/vqa_factorial_canary_3.json"
FULL_PROFILE = ROOT / "configs/vqa/vqa_factorial_full_condition_profiles.json"
FULL_MANIFEST = ROOT / "configs/vqa/qwen35_vqa2025_475_label_free.json"
IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)


def test_runtime_attestation_records_the_pinned_qwen_vl_utils_distribution_version() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'from importlib.metadata import version' in runner
    assert 'version("qwen-vl-utils")' in runner
    assert 'qwen_vl_utils_version != "0.0.14"' in runner
    assert '"qwen-vl-utils": qwen_vl_utils_version' in runner
    assert 'getattr(qwen_vl_utils, "__version__", "installed")' not in runner


def _fake_inference(tmp_path: Path) -> Path:
    path = tmp_path / "fake-inference.py"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys, time\n"
        "rows = json.load(sys.stdin)\n"
        "pid = os.getpid()\n"
        "with open(os.environ['INFERENCE_PID_FILE'], 'w', encoding='utf-8') as h: h.write(str(pid))\n"
        "with open(os.environ['EVENT_LOG'], 'a', encoding='utf-8') as h: h.write(f'inference-start:{pid}\\n')\n"
        "if os.environ.get('FAKE_INFERENCE_SLEEP'): time.sleep(float(os.environ['FAKE_INFERENCE_SLEEP']))\n"
        "if os.environ.get('FAKE_INFERENCE_RC'):\n"
        "    with open(os.environ['EVENT_LOG'], 'a', encoding='utf-8') as h: h.write(f'inference-exit:{pid}\\n')\n"
        "    raise SystemExit(int(os.environ['FAKE_INFERENCE_RC']))\n"
        "model = sys.argv[sys.argv.index('--model-id') + 1]\n"
        "revision = sys.argv[sys.argv.index('--revision') + 1]\n"
        "policy = sys.argv[sys.argv.index('--answer-policy') + 1]\n"
        "minimum = int(sys.argv[sys.argv.index('--min-pixels-per-frame') + 1])\n"
        "maximum = int(sys.argv[sys.argv.index('--max-pixels-per-frame') + 1])\n"
        "mutation = os.environ.get('PREDICTION_MUTATION')\n"
        "out = []\n"
        "for request in rows:\n"
        "    result = {\n"
        "        'status': 'passed', 'answer': 'stable', 'run_id': request['run_id'],\n"
        "        'condition_id': request['condition_id'], 'video_id': request['video_id'],\n"
        "        'model_id': model, 'model_revision': revision, 'runtime': 'transformers 5.15.0.dev0; torch 2.8.0',\n"
        "        'runtime_image': os.environ['VQA_RUNTIME_IMAGE'], 'precision': 'bfloat16',\n"
        "        'device': 'cuda:0 (NVIDIA CUDA)', 'answer_policy': policy, 'question_type': 'general',\n"
        "        'requested_min_pixels_per_frame': minimum, 'requested_max_pixels_per_frame': maximum,\n"
        "        'min_pixels_per_frame': minimum, 'max_pixels_per_frame': maximum,\n"
        "        'elapsed_seconds': 0.01, 'model_load_seconds': 0.25, 'peak_memory_bytes': 11, 'error_type': None, 'error_message': None,\n"
        "        'runtime_attestation': {\n"
        "            'parameter_tensor_count': 2, 'parameter_devices': {'cuda:0': 2},\n"
        "            'parameter_dtypes': {'torch.bfloat16': 2}, 'buffer_tensor_count': 2,\n"
        "            'buffer_devices': {'cuda:0': 2},\n"
        "            'buffer_dtypes': {'torch.bfloat16': 1, 'torch.int64': 1},\n"
        "            'accelerator_backend': 'nvidia-cuda', 'torch_cuda': '12.4', 'torch_hip': None,\n"
        "        },\n"
        "    }\n"
        "    if mutation == 'omit-runtime': result.pop('runtime')\n"
        "    if mutation == 'changed-runtime': result['runtime'] = 'untrusted-runtime 0.0; torch unknown'\n"
        "    if mutation == 'changed-cuda-version': result['runtime_attestation']['torch_cuda'] = '999.0'\n"
        "    if mutation == 'omit-elapsed': result.pop('elapsed_seconds')\n"
        "    if mutation == 'nan-model-load': result['model_load_seconds'] = float('nan')\n"
        "    if mutation == 'negative-model-load': result['model_load_seconds'] = -1\n"
        "    if mutation == 'different-model-load': result['model_load_seconds'] = 0.25 + len(out)\n"
        "    if mutation == 'nan-elapsed': result['elapsed_seconds'] = float('nan')\n"
        "    if mutation == 'negative-elapsed': result['elapsed_seconds'] = -1\n"
        "    if mutation == 'nonnumeric-vram': result['peak_memory_bytes'] = 'unknown'\n"
        "    if mutation == 'negative-vram': result['peak_memory_bytes'] = -1\n"
        "    if mutation == 'extra-field': result['unexpected'] = True\n"
        "    if mutation == 'missing-field': result.pop('runtime')\n"
        "    if mutation == 'partial-parameter-count': result['runtime_attestation']['parameter_tensor_count'] = 100\n"
        "    if mutation == 'partial-parameter-devices': result['runtime_attestation']['parameter_devices'] = {'cuda:0': 1}\n"
        "    if mutation == 'partial-parameter-dtypes': result['runtime_attestation']['parameter_dtypes'] = {'torch.bfloat16': 1}\n"
        "    if mutation == 'partial-buffer-count': result['runtime_attestation']['buffer_tensor_count'] = 100\n"
        "    if mutation == 'partial-buffer-devices': result['runtime_attestation']['buffer_devices'] = {'cuda:0': 1, 'cpu': 1}\n"
        "    if mutation == 'partial-buffer-dtypes': result['runtime_attestation']['buffer_dtypes'] = {'torch.bfloat16': 1}\n"
        "    out.append(result)\n"
        "with open(os.environ['EVENT_LOG'], 'a', encoding='utf-8') as h: h.write(f'inference-exit:{pid}\\n')\n"
        "print(json.dumps(out, allow_nan=True))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run(
    tmp_path: Path,
    *,
    extra: dict[str, str] | None = None,
    persistent: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.Popen[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    events = tmp_path / "events.log"
    remote = tmp_path / "remote"
    fake_hf = tmp_path / "fake-hf"
    fake_hf.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s:%s\\n' \"$1\" \"$2\" >> \"$EVENT_LOG\"\n"
        "case \"$1/$2\" in\n"
        "  buckets/list) if [[ -n \"${FAKE_LIST_WARNING:-}\" ]]; then printf '%s\\n' \"$FAKE_LIST_WARNING\" >&2; fi; printf '%s' \"${FAKE_REMOTE_LISTING:-}\" ;;\n"
        "  buckets/sync)\n"
        "    if [[ \"$3\" == hf://* ]]; then\n"
        "      attempts=0\n"
        "      [[ ! -f \"$INPUT_SYNC_ATTEMPTS_FILE\" ]] || attempts=$(<\"$INPUT_SYNC_ATTEMPTS_FILE\")\n"
        "      attempts=$((attempts + 1))\n"
        "      printf '%s' \"$attempts\" > \"$INPUT_SYNC_ATTEMPTS_FILE\"\n"
        "      mkdir -p \"$4\"\n"
        "      if (( attempts <= ${FAKE_INPUT_SYNC_FAILURES:-0} )); then\n"
        "        if (( attempts == 1 )); then printf 'partial\\n' > \"$4/.partial-download\"; fi\n"
        "        exit 29\n"
        "      fi\n"
        "      if [[ \"${FAKE_REQUIRE_INPUT_REUSE:-0}\" == 1 ]]; then\n"
        "        [[ -f \"$4/.partial-download\" ]] || exit 31\n"
        "        printf 'reused\\n' > \"$INPUT_REUSE_OBSERVED_FILE\"\n"
        "      fi\n"
        "      cp -a \"$FAKE_INPUT_ROOT/.\" \"$4/\"\n"
        "    else\n"
        "      output_attempts=0\n"
        "      [[ ! -f \"$OUTPUT_SYNC_ATTEMPTS_FILE\" ]] || output_attempts=$(<\"$OUTPUT_SYNC_ATTEMPTS_FILE\")\n"
        "      printf '%s' \"$((output_attempts + 1))\" > \"$OUTPUT_SYNC_ATTEMPTS_FILE\"\n"
        "      mkdir -p \"$FAKE_REMOTE_ROOT/payload\"; cp -a \"$3/.\" \"$FAKE_REMOTE_ROOT/payload/\"\n"
        "      if [[ \"${FAKE_TAMPER:-0}\" == 1 ]]; then printf 'tampered\\n' >> \"$OUTPUT_DIR/status.json\"; fi\n"
        "      exit \"${FAKE_SYNC_RC:-0}\"\n"
        "    fi ;;\n"
        "  buckets/cp) if [[ \"${FAKE_TERMINAL_RC:-0}\" == 0 ]]; then cp \"$3\" \"$FAKE_REMOTE_ROOT/terminal-status.json\"; fi; exit \"${FAKE_TERMINAL_RC:-0}\" ;;\n"
        "  *) exit 99 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_hf.chmod(0o755)
    fake_nvidia = tmp_path / "nvidia-smi"
    fake_nvidia.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"nvidia-start:$$\" >> \"$EVENT_LOG\"\n"
        "printf '%s' \"$$\" > \"$NVIDIA_PID_FILE\"\n"
        "printf '0, 1, 2, 3\\n'\n"
        "sleep \"${FAKE_NVIDIA_SLEEP:-0}\"\n"
        "printf '%s\\n' \"nvidia-exit:$$\" >> \"$EVENT_LOG\"\n",
        encoding="utf-8",
    )
    fake_nvidia.chmod(0o755)
    inference = _fake_inference(tmp_path)
    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "BUILDER": str(BUILDER),
        "HF_CLI_BIN": str(fake_hf),
        "HF_TOKEN": "test-token-not-for-output",
        "PROFILE": str(PROFILE),
        "VQA_MODE": "canary",
        "EXPERIMENT_RUN_NAME": "vqa-factorial-canary-20260809-e2",
        "HF_INPUT_URI": (
            "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
            "experiments/qwen35-4b-12f-full-20260803-b1-inputs/inputs"
        ),
        "HF_REMOTE_URI": (
            "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
            "experiments/vqa-factorial-canary-20260809-e2"
        ),
        "VQA_RUNTIME_IMAGE": IMAGE,
        "OUTPUT_DIR": str(tmp_path / "output"),
        "INPUT_DIR": str(tmp_path / "input"),
        "FRAME_DIR": str(tmp_path / "frames"),
        "EVENT_LOG": str(events),
        "FAKE_REMOTE_ROOT": str(remote),
        "MANIFEST_OVERRIDE": str(MANIFEST),
        "VQA_TEST_MODE": "1",
        "VQA_DRY_RUN": "1",
        "INFERENCE_BIN": str(inference),
        "INFERENCE_PID_FILE": str(tmp_path / "inference.pid"),
        "NVIDIA_PID_FILE": str(tmp_path / "nvidia.pid"),
        "CALL_LOG": str(tmp_path / "calls.log"),
        "INPUT_SYNC_ATTEMPTS_FILE": str(tmp_path / "input-sync-attempts"),
        "INPUT_REUSE_OBSERVED_FILE": str(tmp_path / "input-reuse-observed"),
        "OUTPUT_SYNC_ATTEMPTS_FILE": str(tmp_path / "output-sync-attempts"),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }
    env.update(extra or {})
    command = ["bash", str(RUNNER)]
    if persistent:
        return subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(command, env=env, check=False, capture_output=True, text=True)


def _build_fake_full_evidence(tmp_path: Path) -> Path:
    root = tmp_path / "fake-input"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    manifest = json.loads(FULL_MANIFEST.read_text(encoding="utf-8"))
    requests = []
    for example in manifest["examples"]:
        video_id = example["video_id"]
        requests.append(
            {
                "run_id": "shared-evidence",
                "condition_id": "shared-accepted-whisper",
                "video_id": video_id,
                "question": example["question"],
                "frame_paths": [
                    f"/source/evidence/frames/{video_id}/frame-{index:02d}.jpg"
                    for index in range(1, 13)
                ],
                "frame_timestamps": [float(index) for index in range(12)],
                "transcript": f"transcript for {video_id}",
            }
        )
    (evidence / "requests.accepted-whisper.json").write_text(
        json.dumps(requests, indent=2) + "\n", encoding="utf-8"
    )
    (evidence / "evidence-marker.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "label_policy": "label-free-evidence",
                "example_count": 475,
                "checksum_file": "SHA256SUMS",
                "manifest_sha256": "092aec23cfdcc370bbed8e776f7cc7c84752481da6ba0996983883955d3841e8",
                "canonical_id_digest": "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(evidence.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root)
            checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
    (root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return root


def _run_full_condition_with_input_retry(
    tmp_path: Path, *, failures: str = "1"
) -> subprocess.CompletedProcess[str]:
    fake_input = _build_fake_full_evidence(tmp_path)
    result = _run(
        tmp_path,
        extra={
            "PROFILE": str(FULL_PROFILE),
            "MANIFEST_OVERRIDE": str(FULL_MANIFEST),
            "VQA_MODE": "condition",
            "VQA_DRY_RUN": "0",
            "CONDITION_ID": "4b-question-aware",
            "EXPERIMENT_RUN_NAME": "vqa-factorial-full-20260809-f1-4b-question-aware",
            "HF_INPUT_URI": (
                "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
                "experiments/vqa-factorial-full-evidence-20260809-f1"
            ),
            "HF_REMOTE_URI": (
                "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
                "experiments/vqa-factorial-full-20260809-f1/conditions/4b-question-aware"
            ),
            "FAKE_INPUT_ROOT": str(fake_input),
            "FAKE_INPUT_SYNC_FAILURES": failures,
            "FAKE_REQUIRE_INPUT_REUSE": "1" if failures == "1" else "0",
        },
    )
    assert isinstance(result, subprocess.CompletedProcess)
    return result


def _pid_is_gone(path: Path) -> None:
    pid = int(path.read_text(encoding="utf-8"))
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _assert_remote_payload(tmp_path: Path, *, status: str) -> None:
    local = tmp_path / "output"
    remote = tmp_path / "remote" / "payload"
    local_files = sorted(path.relative_to(local) for path in local.rglob("*") if path.is_file())
    remote_files = sorted(path.relative_to(remote) for path in remote.rglob("*") if path.is_file())
    assert remote_files == local_files
    for relative in local_files:
        assert (remote / relative).read_bytes() == (local / relative).read_bytes()
    checked = subprocess.run(
        ["sha256sum", "--check", "SHA256SUMS"], cwd=remote, check=False, capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    terminal = json.loads((tmp_path / "remote" / "terminal-status.json").read_text(encoding="utf-8"))
    assert terminal["status"] == status
    assert terminal["payload_sha256"] == hashlib.sha256((remote / "SHA256SUMS").read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        "omit-runtime", "changed-runtime", "changed-cuda-version", "omit-elapsed", "nan-elapsed", "negative-elapsed",
        "nan-model-load", "negative-model-load", "different-model-load",
        "nonnumeric-vram", "negative-vram", "extra-field", "missing-field",
        "partial-parameter-count", "partial-parameter-devices", "partial-parameter-dtypes",
        "partial-buffer-count", "partial-buffer-devices", "partial-buffer-dtypes",
    ],
)
def test_prediction_schema_and_attestation_are_exact(tmp_path: Path, mutation: str) -> None:
    result = _run(tmp_path, extra={"PREDICTION_MUTATION": mutation})
    assert result.returncode != 0, result.stdout + result.stderr


def test_success_sync_snapshots_frozen_payload_and_terminal_checksum(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    runtime = json.loads((tmp_path / "output/runtime.json").read_text(encoding="utf-8"))
    assert runtime == {
        "runtime": "transformers 5.15.0.dev0; torch 2.8.0",
        "torch_cuda": "12.4",
        "torch_hip": None,
        "runtime_image": IMAGE,
    }
    _assert_remote_payload(tmp_path, status="passed")
    assert (tmp_path / "output.terminal-status.json").is_file()
    assert (tmp_path / "remote" / "terminal-status.json").is_file()
    _pid_is_gone(tmp_path / "inference.pid")
    _pid_is_gone(tmp_path / "nvidia.pid")


def test_empty_remote_prefix_ignores_hf_cli_stderr_diagnostics(tmp_path: Path) -> None:
    result = _run(tmp_path, extra={"FAKE_LIST_WARNING": "fresh client diagnostic"})
    assert result.returncode == 0, result.stderr
    _assert_remote_payload(tmp_path, status="passed")


def test_condition_input_sync_retries_in_place_after_transient_failure(tmp_path: Path) -> None:
    result = _run_full_condition_with_input_retry(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "input-sync-attempts").read_text(encoding="utf-8") == "2"
    assert (tmp_path / "input-reuse-observed").read_text(encoding="utf-8") == "reused\n"
    _assert_remote_payload(tmp_path, status="passed")


def test_condition_input_sync_stops_after_three_failures_without_inference(
    tmp_path: Path,
) -> None:
    result = _run_full_condition_with_input_retry(tmp_path, failures="3")
    assert result.returncode != 0
    assert (tmp_path / "input-sync-attempts").read_text(encoding="utf-8") == "3"
    assert not (tmp_path / "inference.pid").exists()
    assert (tmp_path / "output-sync-attempts").read_text(encoding="utf-8") == "1"
    _assert_remote_payload(tmp_path, status="failed")


def test_post_sync_local_mutation_cannot_be_rewritten_into_remote_payload(tmp_path: Path) -> None:
    result = _run(tmp_path, extra={"FAKE_TAMPER": "1"})
    assert result.returncode == 0, result.stderr
    remote_status = (tmp_path / "remote/payload/status.json").read_bytes()
    local_status = (tmp_path / "output/status.json").read_bytes()
    assert local_status.endswith(b"tampered\n")
    assert remote_status != local_status
    checked = subprocess.run(
        ["sha256sum", "--check", "SHA256SUMS"], cwd=tmp_path / "remote/payload", check=False
    )
    assert checked.returncode == 0
    terminal = json.loads((tmp_path / "remote/terminal-status.json").read_text(encoding="utf-8"))
    assert terminal["payload_sha256"] == hashlib.sha256(
        (tmp_path / "remote/payload/SHA256SUMS").read_bytes()
    ).hexdigest()


def test_payload_or_terminal_failure_leaves_remote_terminal_absent_and_payload_immutable(tmp_path: Path) -> None:
    result = _run(tmp_path, extra={"FAKE_TERMINAL_RC": "19"})
    assert result.returncode != 0
    assert not (tmp_path / "remote/terminal-status.json").exists()
    assert (tmp_path / "remote/payload/SHA256SUMS").is_file()
    assert (tmp_path / "output/status.json").read_bytes() == (
        tmp_path / "remote/payload/status.json"
    ).read_bytes()
    result = _run(tmp_path / "retry", extra={"FAKE_SYNC_RC": "17"})
    assert result.returncode != 0
    assert (tmp_path / "retry/output-sync-attempts").read_text(encoding="utf-8") == "1"
    assert not (tmp_path / "retry/remote/terminal-status.json").exists()


def test_inference_failure_and_sigterm_reap_test_owned_processes(tmp_path: Path) -> None:
    failed = _run(tmp_path / "failed", extra={"FAKE_INFERENCE_RC": "23"})
    assert failed.returncode != 0
    _pid_is_gone(tmp_path / "failed/inference.pid")
    _pid_is_gone(tmp_path / "failed/nvidia.pid")

    live = _run(
        tmp_path / "signal",
        extra={"FAKE_INFERENCE_SLEEP": "60", "FAKE_NVIDIA_SLEEP": "60"},
        persistent=True,
    )
    assert isinstance(live, subprocess.Popen)
    pid_file = tmp_path / "signal/inference.pid"
    for _ in range(100):
        if pid_file.exists():
            break
        time.sleep(0.05)
    assert pid_file.exists()
    live.send_signal(signal.SIGTERM)
    live.communicate(timeout=10)
    assert live.returncode == 143
    _pid_is_gone(pid_file)
    _pid_is_gone(tmp_path / "signal/nvidia.pid")
