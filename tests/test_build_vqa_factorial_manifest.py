from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_vqa_factorial_manifest.py"
RUNNER_PATH = ROOT / "scripts/run-vqa-factorial-dstack.sh"
SOURCE_PATH = ROOT / "configs/vqa/qwen35_vqa2025_475_label_free.json"
CANARY_PROFILE_PATH = ROOT / "configs/vqa/vqa_factorial_canary_profile.json"
EVIDENCE_PROFILE_PATH = ROOT / "configs/vqa/vqa_factorial_full_evidence_profile.json"
CONDITION_PROFILE_PATH = ROOT / "configs/vqa/vqa_factorial_full_condition_profiles.json"
RETRY_CONDITION_PROFILE_PATH = ROOT / "configs/vqa/vqa_factorial_full_condition_profiles_f2.json"
CANARY_MANIFEST_SHA256 = "857d251870d24c4c5c7b43d94030a994bd4d8c39da4db1aa4385c1e01f7a7005"
FULL_MANIFEST_SHA256 = "092aec23cfdcc370bbed8e776f7cc7c84752481da6ba0996983883955d3841e8"
FULL_IDS_SHA256 = "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"


def _load_builder():
    spec = importlib.util.spec_from_file_location("vqa_factorial_manifest", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load manifest builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canary_profile() -> dict[str, object]:
    return json.loads(CANARY_PROFILE_PATH.read_text(encoding="utf-8"))


def _run_shell(
    tmp_path: Path,
    *,
    profile: Path = CANARY_PROFILE_PATH,
    remote_listing: str = "",
    vqa_mode: str = "canary",
    experiment_run_name: str = "vqa-factorial-canary-20260809-e2",
    input_uri: str = (
        "hf://buckets/Npatta01/trec_video_2026/"
        "experiments/qwen35-4b-12f-full-20260803-b1-inputs/inputs"
    ),
    remote_uri: str = (
        "hf://buckets/Npatta01/trec_video_2026/"
        "experiments/vqa-factorial-canary-20260809-e2"
    ),
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str] | subprocess.Popen[str]:
    fake_hf = tmp_path / "fake-hf"
    event_log = tmp_path / "events.log"
    fake_hf.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ": \"${EVENT_LOG:?}\"\n"
        "printf '%s\\n' \"$1:$2\" >> \"$EVENT_LOG\"\n"
        "if [[ \"$1/$2\" == \"buckets/list\" ]]; then\n"
        "  printf '%s' \"${FAKE_REMOTE_LISTING:-}\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1/$2\" == \"buckets/sync\" ]]; then\n"
        "  if [[ \"${FAKE_TAMPER:-0}\" == \"1\" && -n \"${OUTPUT_DIR:-}\" ]]; then printf 'tampered\\n' >> \"$OUTPUT_DIR/status.json\"; fi\n"
        "  exit \"${FAKE_SYNC_RC:-0}\"\n"
        "fi\n"
        "if [[ \"$1/$2\" == \"buckets/cp\" ]]; then\n"
        "  exit \"${FAKE_TERMINAL_RC:-0}\"\n"
        "fi\n"
        "exit 99\n",
        encoding="utf-8",
    )
    fake_hf.chmod(0o755)
    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "BUILDER": str(BUILDER_PATH),
        "HF_CLI_BIN": str(fake_hf),
        "HF_TOKEN": "test-secret-must-not-print",
        "PROFILE": str(profile),
        "VQA_MODE": vqa_mode,
        "EXPERIMENT_RUN_NAME": experiment_run_name,
        "HF_INPUT_URI": input_uri,
        "HF_REMOTE_URI": remote_uri,
        "VQA_RUNTIME_IMAGE": (
            "docker.io/pytorch/pytorch@sha256:"
            "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
        ),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "INPUT_DIR": str(tmp_path / "input"),
        "FRAME_DIR": str(tmp_path / "frames"),
        "EVENT_LOG": str(event_log),
        "FAKE_REMOTE_LISTING": remote_listing,
        "MANIFEST_OVERRIDE": str(ROOT / "configs/vqa/vqa_factorial_canary_3.json"),
        "VQA_TEST_MODE": "1",
    }
    if extra_env:
        env.update(extra_env)
    if extra_env and extra_env.get("PERSIST") == "1":
        return subprocess.Popen(
            ["bash", str(RUNNER_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    return subprocess.run(["bash", str(RUNNER_PATH)], check=False, capture_output=True, text=True, env=env)


def _write_contract_fake_inference(tmp_path: Path) -> Path:
    fake_inference = tmp_path / "fake-inference"
    fake_inference.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "import time\n"
        "rows = json.load(sys.stdin)\n"
        "with open(os.environ['CALL_LOG'], 'a', encoding='utf-8') as handle:\n"
        "    handle.write(sys.argv[sys.argv.index('--model-id') + 1] + ':' + str(len(rows)) + '\\n')\n"
        "if os.environ.get('CHILD_PID_FILE'):\n"
        "    open(os.environ['CHILD_PID_FILE'], 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "if os.environ.get('FAKE_INFERENCE_SLEEP'):\n"
        "    time.sleep(float(os.environ['FAKE_INFERENCE_SLEEP']))\n"
        "if os.environ.get('FAKE_INFERENCE_RC'):\n"
        "    raise SystemExit(int(os.environ['FAKE_INFERENCE_RC']))\n"
        "mutation = os.environ.get('FAKE_PREDICTION_MUTATION')\n"
        "output = []\n"
        "for row in rows:\n"
        "    result = {\n"
        "        'status': 'passed', 'answer': 'stable', 'run_id': row['run_id'],\n"
        "        'condition_id': row['condition_id'], 'video_id': row['video_id'],\n"
        "        'model_id': sys.argv[sys.argv.index('--model-id') + 1],\n"
        "        'model_revision': sys.argv[sys.argv.index('--revision') + 1],\n"
        "        'answer_policy': sys.argv[sys.argv.index('--answer-policy') + 1],\n"
        "        'question_type': 'general',\n"
        "        'requested_min_pixels_per_frame': int(sys.argv[sys.argv.index('--min-pixels-per-frame') + 1]),\n"
        "        'requested_max_pixels_per_frame': int(sys.argv[sys.argv.index('--max-pixels-per-frame') + 1]),\n"
        "        'min_pixels_per_frame': int(sys.argv[sys.argv.index('--min-pixels-per-frame') + 1]),\n"
        "        'max_pixels_per_frame': int(sys.argv[sys.argv.index('--max-pixels-per-frame') + 1]),\n"
        "        'runtime': 'transformers 5.15.0.dev0; torch 2.8.0', 'runtime_image': os.environ['VQA_RUNTIME_IMAGE'],\n"
        "        'precision': 'bfloat16', 'device': 'cuda:0 (NVIDIA CUDA)',\n"
        "        'runtime_attestation': {'parameter_tensor_count': 1, 'parameter_devices': {'cuda:0': 1}, 'parameter_dtypes': {'torch.bfloat16': 1}, 'buffer_tensor_count': 0, 'buffer_devices': {}, 'buffer_dtypes': {}, 'accelerator_backend': 'nvidia-cuda', 'torch_cuda': '12.4', 'torch_hip': None},\n"
        "        'elapsed_seconds': 0.01, 'model_load_seconds': 0.25, 'peak_memory_bytes': 11, 'error_type': None, 'error_message': None,\n"
        "    }\n"
        "    if mutation == 'stale-model': result['model_id'] = 'stale/model'\n"
        "    if mutation == 'stale-policy': result['answer_policy'] = 'current' if result['answer_policy'] == 'question-aware' else 'question-aware'\n"
        "    if mutation == '512-bounds': result['requested_max_pixels_per_frame'] = 512 * 512\n"
        "    if mutation == 'missing-attestation': result.pop('runtime_attestation')\n"
        "    output.append(result)\n"
        "print(json.dumps(output))\n",
        encoding="utf-8",
    )
    fake_inference.chmod(0o755)
    return fake_inference


def test_builder_freezes_the_three_label_free_canary_rows(tmp_path: Path) -> None:
    builder = _load_builder()
    output = tmp_path / "canary.json"

    builder.build_canary_manifest(SOURCE_PATH, output, expected_source_count=475)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["example_count"] == 3
    assert [row["video_id"] for row in payload["examples"]] == [
        "CtinYSH3nVw",
        "LKiaCf1QZFw",
        "IVqpg_1aWXU",
    ]
    assert payload["examples"] == [
        {
            "video_id": "CtinYSH3nVw",
            "question": "What happens before the man gets into a car?",
        },
        {
            "video_id": "LKiaCf1QZFw",
            "question": "What does this woman say at the end of the video?",
        },
        {
            "video_id": "IVqpg_1aWXU",
            "question": "What is written on the toy airplane?",
        },
    ]
    assert all(set(row) == {"video_id", "question"} for row in payload["examples"])


def test_builder_rejects_nested_label_like_keys() -> None:
    builder = _load_builder()
    with pytest.raises(ValueError, match="label-bearing"):
        builder.reject_label_like_keys({"question": {"reference_answer": "x"}})


def test_builder_rejects_wrong_source_count_and_duplicate_ids(tmp_path: Path) -> None:
    builder = _load_builder()
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    source["examples"] = [source["examples"][0], source["examples"][0]]
    source["example_count"] = 2
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 475"):
        builder.build_canary_manifest(path, tmp_path / "out.json", expected_source_count=475)


def test_canary_profile_is_strict_and_has_six_invocations() -> None:
    builder = _load_builder()
    profile = _canary_profile()
    builder.validate_canary_profile(profile)

    assert set(profile) == {
        "run_name",
        "input_uri",
        "remote_uri",
        "manifest",
        "expected_count",
        "frame_count",
        "conditions",
        "repeat_condition",
    }
    assert profile["expected_count"] == 3
    assert len(profile["conditions"]) == 6
    assert profile["repeat_condition"] in {
        condition["condition_id"] for condition in profile["conditions"]
    }
    assert all(
        set(condition)
        == {
            "condition_id",
            "model_id",
            "revision",
            "answer_policy",
            "min_pixels",
            "max_pixels",
            "request_source",
        }
        for condition in profile["conditions"]
    )
    highres = next(
        condition
        for condition in profile["conditions"]
        if condition["request_source"] == "shared-accepted-whisper-ocr-only"
    )
    assert (highres["min_pixels"], highres["max_pixels"]) == (65536, 589824)
    assert all(
        (condition["min_pixels"], condition["max_pixels"]) == (65536, 262144)
        for condition in profile["conditions"]
        if condition is not highres
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda profile: profile.update(extra="reject-me"),
        lambda profile: profile["conditions"].append(profile["conditions"][0]),
        lambda profile: profile["conditions"][0].update(revision="not-a-pinned-revision"),
        lambda profile: profile["conditions"][0].update(answer_policy="labels"),
        lambda profile: profile["conditions"][0].update(min_pixels=0),
        lambda profile: profile["conditions"][0].update(max_pixels=65535),
        lambda profile: profile.update(remote_uri="/unsafe/output"),
    ],
    ids=["unknown-field", "duplicate-condition", "revision", "policy", "zero-bound", "inverted-bound", "unsafe-path"],
)
def test_canary_profile_validation_fails_closed(mutation) -> None:
    builder = _load_builder()
    profile = _canary_profile()
    mutation(profile)
    with pytest.raises(ValueError):
        builder.validate_canary_profile(profile)


def test_full_profiles_define_one_evidence_job_and_four_distinct_conditions() -> None:
    builder = _load_builder()
    evidence = json.loads(EVIDENCE_PROFILE_PATH.read_text(encoding="utf-8"))
    conditions = json.loads(CONDITION_PROFILE_PATH.read_text(encoding="utf-8"))
    builder.validate_full_evidence_profile(evidence)
    builder.validate_full_condition_profiles(conditions)

    assert evidence["expected_count"] == 475
    assert len(conditions["conditions"]) == 4
    assert {item["condition_id"] for item in conditions["conditions"]} == {
        "4b-current",
        "4b-question-aware",
        "8b-current",
        "8b-question-aware",
    }
    assert len(
        {
            conditions["remote_uri"].rstrip("/") + "/" + item["condition_id"]
            for item in conditions["conditions"]
        }
    ) == 4


def test_full_topology_has_one_evidence_yaml_and_four_one_condition_yamls() -> None:
    evidence = yaml.safe_load(
        (ROOT / "dstack/vqa-factorial-full-evidence.dstack.yml").read_text()
    )
    condition_paths = sorted(ROOT.glob("dstack/vqa-factorial-full-*.dstack.yml"))
    condition_paths = [path for path in condition_paths if "evidence" not in path.name]
    assert len(condition_paths) == 4
    assert evidence["resources"]["gpu"].endswith(":1:48GB..")
    for mapping in evidence["files"]:
        assert not mapping.startswith("../src/trec_video_2026:")
    expected_package = {
        "../src/trec_video_2026/__init__.py",
        "../src/trec_video_2026/vqa_models.py",
        "../src/trec_video_2026/vqa_transformers_runner.py",
        "../src/trec_video_2026/vqa_whisper.py",
    }
    assert {
        mapping.split(":", 1)[0]
        for mapping in evidence["files"]
        if mapping.startswith("../src/trec_video_2026/")
    } == expected_package
    assert len({yaml.safe_load(path.read_text())["name"] for path in condition_paths}) == 4
    for path in condition_paths:
        payload = yaml.safe_load(path.read_text())
        assert re.fullmatch(r"[a-z][a-z0-9-]{1,40}", payload["name"])
        assert payload["resources"]["gpu"].endswith(":1:48GB..")
        assert payload["backends"] == ["runpod", "vastai"]
        assert "VQA_MODE=condition" in "\n".join(payload["env"])
        assert "CONDITION_ID=" in "\n".join(payload["env"])
        assert "--mode condition" not in "\n".join(payload["commands"])
        mappings = payload["files"]
        assert not any(mapping.startswith("../src/trec_video_2026:") for mapping in mappings)
        assert {
            mapping.split(":", 1)[0]
            for mapping in mappings
            if mapping.startswith("../src/trec_video_2026/")
        } == {
            "../src/trec_video_2026/__init__.py",
            "../src/trec_video_2026/vqa_models.py",
            "../src/trec_video_2026/vqa_transformers_runner.py",
            "../src/trec_video_2026/vqa_whisper.py",
        }


def test_f2_retry_topology_preserves_f1_and_uses_a_fresh_remote_identity() -> None:
    builder = _load_builder()
    profile = json.loads(RETRY_CONDITION_PROFILE_PATH.read_text(encoding="utf-8"))
    builder.validate_full_condition_profiles(profile)
    assert profile["run_name"] == "vqa-factorial-full-20260809-f2"
    assert profile["remote_uri"] == (
        "hf://buckets/Npatta01/trec_video_2026/"
        "experiments/vqa-factorial-full-20260809-f2/conditions"
    )
    for condition_id in (
        "4b-current",
        "4b-question-aware",
        "8b-current",
        "8b-question-aware",
    ):
        path = ROOT / f"dstack/vqa-factorial-f2-{condition_id}.dstack.yml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = "\n".join(payload["env"])
        assert payload["name"] == f"vqa-fact-f2-{condition_id}"
        assert len(payload["name"]) <= 41
        assert payload["backends"] == ["runpod", "vastai"]
        assert payload["resources"]["gpu"] == "L40S:1:48GB.."
        assert f"EXPERIMENT_RUN_NAME=vqa-factorial-full-20260809-f2-{condition_id}" in env
        assert (
            "HF_REMOTE_URI=hf://buckets/Npatta01/trec_video_2026/"
            f"experiments/vqa-factorial-full-20260809-f2/conditions/{condition_id}"
        ) in env
        assert "PROFILE=/dstack/run/configs/vqa/vqa_factorial_full_condition_profiles_f2.json" in env


def test_shell_refuses_stale_remote_prefix_before_runtime_bootstrap(tmp_path: Path) -> None:
    result = _run_shell(tmp_path, remote_listing="already-there\n")
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert "pip install" not in result.stderr
    assert "test-secret-must-not-print" not in result.stdout + result.stderr
    events = (tmp_path / "events.log").read_text().splitlines()
    assert events.index("lightweight-client-ready") < events.index("buckets:list")
    assert "heavy-runtime-bootstrap" not in events


def test_shell_refuses_profile_revision_before_remote_or_bootstrap(tmp_path: Path) -> None:
    profile = _canary_profile()
    profile["conditions"][0]["revision"] = "bad-revision"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    result = _run_shell(tmp_path, profile=path)
    assert result.returncode != 0
    assert "revision" in result.stderr.lower()
    assert "pip install" not in result.stderr


def test_shell_refuses_label_bearing_manifest_before_bootstrap(tmp_path: Path) -> None:
    manifest = json.loads(
        (ROOT / "configs/vqa/vqa_factorial_canary_3.json").read_text(encoding="utf-8")
    )
    manifest["examples"][0]["reference_answer"] = "must never enter inference"
    manifest_path = tmp_path / "label-bearing.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = _run_shell(
        tmp_path,
        extra_env={"MANIFEST_OVERRIDE": str(manifest_path), "VQA_DRY_RUN": "1"},
    )
    assert result.returncode != 0
    assert "label-bearing" in result.stderr
    assert "pip install" not in result.stderr


def test_shell_refuses_unsafe_output_profile_before_bootstrap(tmp_path: Path) -> None:
    profile = _canary_profile()
    profile["remote_uri"] = "/tmp/unsafe-output"
    path = tmp_path / "unsafe-profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    result = _run_shell(tmp_path, profile=path)
    assert result.returncode != 0
    assert "safe HF bucket URI" in result.stderr
    assert "pip install" not in result.stderr


def test_canary_shell_runs_six_isolated_cells_and_freezes_bundle(
    tmp_path: Path,
) -> None:
    fake_inference = _write_contract_fake_inference(tmp_path)
    call_log = tmp_path / "calls.log"
    fake_nvidia = tmp_path / "nvidia-smi"
    fake_nvidia.write_text("#!/usr/bin/env bash\nprintf '0, 1, 2, 3\\n'\n", encoding="utf-8")
    fake_nvidia.chmod(0o755)
    result = _run_shell(
        tmp_path,
        extra_env={
            "VQA_DRY_RUN": "1",
            "MANIFEST_OVERRIDE": str(ROOT / "configs/vqa/vqa_factorial_canary_3.json"),
            "INFERENCE_BIN": str(fake_inference),
            "CALL_LOG": str(call_log),
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
        },
    )
    assert result.returncode == 0, result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 6
    assert sorted(int(line.rsplit(":", 1)[1]) for line in calls) == [1, 3, 3, 3, 3, 3]
    output = tmp_path / "output"
    assert (output / "conditions/4b-current/predictions.json").is_file()
    assert (output / "conditions/4b-current-repeat/predictions.json").is_file()
    first_predictions = json.loads(
        (output / "conditions/4b-current/predictions.json").read_text()
    )
    repeat_predictions = json.loads(
        (output / "conditions/4b-current-repeat/predictions.json").read_text()
    )
    assert [row["answer"] for row in first_predictions] == [
        row["answer"] for row in repeat_predictions
    ]
    ocr_requests = json.loads(
        (output / "conditions/8b-question-aware-ocr-highres/requests.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(ocr_requests) == 1
    assert ocr_requests[0]["video_id"] == "IVqpg_1aWXU"
    checksum_lines = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert all("SHA256SUMS" not in line for line in checksum_lines)
    assert any("status.json" in line for line in checksum_lines)
    assert any("evidence/evidence.json" in line for line in checksum_lines)
    assert (output / "status.json").is_file()
    assert not list(output.glob("**/*.pid"))


def test_generated_canary_hash_is_stable(tmp_path: Path) -> None:
    builder = _load_builder()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    builder.build_canary_manifest(SOURCE_PATH, first, expected_source_count=475)
    builder.build_canary_manifest(SOURCE_PATH, second, expected_source_count=475)
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(
        second.read_bytes()
    ).hexdigest()


def test_manifest_identities_and_exact_condition_bindings_are_pinned() -> None:
    builder = _load_builder()
    canary = json.loads((ROOT / "configs/vqa/vqa_factorial_canary_3.json").read_text())
    full = json.loads(SOURCE_PATH.read_text())

    assert builder.manifest_sha256(ROOT / "configs/vqa/vqa_factorial_canary_3.json") == CANARY_MANIFEST_SHA256
    assert builder.manifest_sha256(SOURCE_PATH) == FULL_MANIFEST_SHA256
    assert builder.canonical_id_digest(full["examples"]) == FULL_IDS_SHA256
    builder.validate_manifest_identity(
        ROOT / "configs/vqa/vqa_factorial_canary_3.json",
        expected_count=3,
        expected_sha256=CANARY_MANIFEST_SHA256,
        expected_ids=("CtinYSH3nVw", "LKiaCf1QZFw", "IVqpg_1aWXU"),
    )
    builder.validate_manifest_identity(
        SOURCE_PATH,
        expected_count=475,
        expected_sha256=FULL_MANIFEST_SHA256,
        expected_canonical_id_digest=FULL_IDS_SHA256,
    )
    builder.validate_canary_profile(_canary_profile())
    builder.validate_full_condition_profiles(
        json.loads(CONDITION_PROFILE_PATH.read_text(encoding="utf-8"))
    )
    assert len(canary["examples"]) == 3


def test_private_full_manifest_mode_reproduces_the_frozen_identity(tmp_path: Path) -> None:
    builder = _load_builder()
    output = tmp_path / "full.json"

    builder.build_full_manifest(ROOT / "data", output, expected_annotations=481, expected_media=475)

    assert hashlib.sha256(output.read_bytes()).hexdigest() == FULL_MANIFEST_SHA256
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert builder.canonical_id_digest(payload["examples"]) == FULL_IDS_SHA256


def test_builder_rejects_path_escape_video_ids_but_accepts_youtube_punctuation(tmp_path: Path) -> None:
    builder = _load_builder()
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    canary_index = next(
        index for index, row in enumerate(source["examples"]) if row["video_id"] == "CtinYSH3nVw"
    )
    source["examples"][canary_index] = {"video_id": "../../outside", "question": "safe"}
    escaped = tmp_path / "escaped.json"
    escaped.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="video_id"):
        builder.build_canary_manifest(escaped, tmp_path / "out.json", expected_source_count=475)

    source["examples"][canary_index] = {"video_id": "abc-_123XYZ", "question": "safe"}
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="missing frozen canary"):
        builder.build_canary_manifest(valid, tmp_path / "out.json", expected_source_count=475)


@pytest.mark.parametrize(
    "mutation",
    ["stale-model", "stale-policy", "512-bounds", "missing-attestation"],
)
def test_shell_rejects_mutated_prediction_control_contract(tmp_path: Path, mutation: str) -> None:
    fake_inference = _write_contract_fake_inference(tmp_path)
    calls = tmp_path / "calls.log"
    result = _run_shell(
        tmp_path,
        extra_env={
            "VQA_DRY_RUN": "1",
            "MANIFEST_OVERRIDE": str(ROOT / "configs/vqa/vqa_factorial_canary_3.json"),
            "INFERENCE_BIN": str(fake_inference),
            "CALL_LOG": str(calls),
            "FAKE_PREDICTION_MUTATION": mutation,
        },
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "output.terminal-status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("mutation", ["replace", "reorder", "question"])
def test_shell_requires_exact_canary_manifest_identity(tmp_path: Path, mutation: str) -> None:
    manifest = json.loads((ROOT / "configs/vqa/vqa_factorial_canary_3.json").read_text())
    if mutation == "replace":
        manifest["examples"][0]["video_id"] = "../../outside"
    elif mutation == "reorder":
        manifest["examples"] = list(reversed(manifest["examples"]))
    else:
        manifest["examples"][0]["question"] = "Different question"
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _run_shell(
        tmp_path,
        extra_env={"MANIFEST_OVERRIDE": str(path), "VQA_DRY_RUN": "1"},
    )

    assert result.returncode != 0
    assert "identity" in result.stderr.lower() or "manifest" in result.stderr.lower()


def test_shell_rejects_ocr_only_full_condition_binding(tmp_path: Path) -> None:
    profile = json.loads(CONDITION_PROFILE_PATH.read_text(encoding="utf-8"))
    profile["conditions"][0]["request_source"] = "shared-accepted-whisper-ocr-only"
    path = tmp_path / "full-profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    result = _run_shell(
        tmp_path,
        profile=path,
        vqa_mode="condition",
        experiment_run_name="vqa-factorial-full-20260809-f1-4b-current",
        input_uri="hf://buckets/Npatta01/trec_video_2026/experiments/vqa-factorial-full-evidence-20260809-f1",
        remote_uri="hf://buckets/Npatta01/trec_video_2026/experiments/vqa-factorial-full-20260809-f1/conditions/4b-current",
        extra_env={
            "CONDITION_ID": "4b-current",
            "VQA_DRY_RUN": "1",
            "MANIFEST_OVERRIDE": str(SOURCE_PATH),
        },
    )
    assert result.returncode != 0
    assert "request_source" in result.stderr or "condition" in result.stderr


def test_shell_publishes_payload_before_terminal_marker_and_cleans_telemetry(tmp_path: Path) -> None:
    fake_inference = _write_contract_fake_inference(tmp_path)
    calls = tmp_path / "calls.log"
    result = _run_shell(
        tmp_path,
        extra_env={
            "VQA_DRY_RUN": "1",
            "MANIFEST_OVERRIDE": str(ROOT / "configs/vqa/vqa_factorial_canary_3.json"),
            "INFERENCE_BIN": str(fake_inference),
            "CALL_LOG": str(calls),
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
        },
    )
    assert result.returncode == 0, result.stderr
    events = (tmp_path / "events.log").read_text().splitlines()
    assert events.index("buckets:list") < events.index("buckets:sync") < events.index("buckets:cp")
    assert (tmp_path / "output.terminal-status.json").read_text()
    assert not list((tmp_path / "output").glob("**/*.pid"))


def test_shell_records_failed_status_and_no_terminal_marker_when_payload_sync_fails(tmp_path: Path) -> None:
    fake_inference = _write_contract_fake_inference(tmp_path)
    result = _run_shell(
        tmp_path,
        extra_env={
            "VQA_DRY_RUN": "1",
            "MANIFEST_OVERRIDE": str(ROOT / "configs/vqa/vqa_factorial_canary_3.json"),
            "INFERENCE_BIN": str(fake_inference),
            "CALL_LOG": str(tmp_path / "calls.log"),
            "FAKE_SYNC_RC": "17",
        },
    )
    assert result.returncode != 0
    status = json.loads((tmp_path / "output" / "status.json").read_text())
    assert status["status"] == "failed"
    assert not (tmp_path / "output.terminal-status.json").exists()
    assert "buckets:cp" not in (tmp_path / "events.log").read_text()


def test_shell_rejects_payload_checksum_tamper_before_terminal_marker(tmp_path: Path) -> None:
    fake_inference = _write_contract_fake_inference(tmp_path)
    result = _run_shell(
        tmp_path,
        extra_env={
            "VQA_DRY_RUN": "1",
            "MANIFEST_OVERRIDE": str(ROOT / "configs/vqa/vqa_factorial_canary_3.json"),
            "INFERENCE_BIN": str(fake_inference),
            "CALL_LOG": str(tmp_path / "calls.log"),
            "FAKE_TAMPER": "1",
        },
    )
    assert result.returncode == 0
    assert (tmp_path / "output" / "status.json").read_bytes().endswith(b"tampered\n")
    assert "buckets:cp" in (tmp_path / "events.log").read_text()


def test_shell_finalizes_truthful_failed_artifact_after_inference_failure(tmp_path: Path) -> None:
    fake_inference = _write_contract_fake_inference(tmp_path)
    result = _run_shell(
        tmp_path,
        extra_env={
            "VQA_DRY_RUN": "1",
            "MANIFEST_OVERRIDE": str(ROOT / "configs/vqa/vqa_factorial_canary_3.json"),
            "INFERENCE_BIN": str(fake_inference),
            "CALL_LOG": str(tmp_path / "calls.log"),
            "FAKE_INFERENCE_RC": "23",
        },
    )
    assert result.returncode != 0
    status = json.loads((tmp_path / "output" / "status.json").read_text())
    assert status["status"] == "failed"
    terminal = json.loads((tmp_path / "output.terminal-status.json").read_text())
    assert terminal["status"] == "failed"
    assert (tmp_path / "output" / "conditions/4b-current/inference.stderr").is_file()


def test_shell_signal_cleanup_removes_inference_child_and_telemetry(tmp_path: Path) -> None:
    fake_inference = _write_contract_fake_inference(tmp_path)
    child_pid_file = tmp_path / "child.pid"
    process = _run_shell(
        tmp_path,
        extra_env={
            "PERSIST": "1",
            "VQA_DRY_RUN": "1",
            "MANIFEST_OVERRIDE": str(ROOT / "configs/vqa/vqa_factorial_canary_3.json"),
            "INFERENCE_BIN": str(fake_inference),
            "CALL_LOG": str(tmp_path / "calls.log"),
            "CHILD_PID_FILE": str(child_pid_file),
            "FAKE_INFERENCE_SLEEP": "60",
        },
    )
    assert isinstance(process, subprocess.Popen)
    for _ in range(100):
        if child_pid_file.exists():
            break
        time.sleep(0.05)
    assert child_pid_file.exists()
    child_pid = int(child_pid_file.read_text())
    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=10)
    assert process.returncode == 143
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert not list((tmp_path / "output").glob("**/*.pid"))


def test_shell_marks_terminal_publication_failure_failed(tmp_path: Path) -> None:
    fake_inference = _write_contract_fake_inference(tmp_path)
    result = _run_shell(
        tmp_path,
        extra_env={
            "VQA_DRY_RUN": "1",
            "MANIFEST_OVERRIDE": str(ROOT / "configs/vqa/vqa_factorial_canary_3.json"),
            "INFERENCE_BIN": str(fake_inference),
            "CALL_LOG": str(tmp_path / "calls.log"),
            "FAKE_TERMINAL_RC": "19",
        },
    )
    assert result.returncode != 0
    assert json.loads((tmp_path / "output" / "status.json").read_text())["status"] == "passed"
    events = (tmp_path / "events.log").read_text().splitlines()
    assert "buckets:cp" in events
    assert "terminal-marker-published" not in events


def test_shell_refuses_nonempty_local_output_before_remote_inspection(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.txt").write_text("stale")
    result = _run_shell(
        tmp_path,
        extra_env={"OUTPUT_DIR": str(output), "VQA_DRY_RUN": "1"},
    )
    assert result.returncode != 0
    assert "non-empty" in result.stderr
    assert "buckets:list" not in (tmp_path / "events.log").read_text()
