from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
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
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    fake_hf = tmp_path / "fake-hf"
    fake_hf.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1/$2\" == \"buckets/list\" ]]; then\n"
        f"  printf '%s' {remote_listing!r}\n"
        "  exit 0\n"
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
        "VQA_MODE": "canary",
        "EXPERIMENT_RUN_NAME": "vqa-factorial-canary-20260809-e1",
        "HF_INPUT_URI": (
            "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
            "experiments/qwen35-4b-12f-full-20260803-b1-inputs/inputs"
        ),
        "HF_REMOTE_URI": (
            "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
            "experiments/vqa-factorial-canary-20260809-e1"
        ),
        "VQA_RUNTIME_IMAGE": (
            "docker.io/pytorch/pytorch@sha256:"
            "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
        ),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "INPUT_DIR": str(tmp_path / "input"),
        "FRAME_DIR": str(tmp_path / "frames"),
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(RUNNER_PATH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


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
    assert len({yaml.safe_load(path.read_text())["name"] for path in condition_paths}) == 4
    for path in condition_paths:
        payload = yaml.safe_load(path.read_text())
        assert payload["resources"]["gpu"].endswith(":1:48GB..")
        assert "VQA_MODE=condition" in "\n".join(payload["env"])
        assert "CONDITION_ID=" in "\n".join(payload["env"])
        assert "--mode condition" not in "\n".join(payload["commands"])


def test_shell_refuses_stale_remote_prefix_before_runtime_bootstrap(tmp_path: Path) -> None:
    result = _run_shell(tmp_path, remote_listing="already-there\n")
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert "pip install" not in result.stderr
    assert "test-secret-must-not-print" not in result.stdout + result.stderr


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
    fake_inference = tmp_path / "fake-inference"
    call_log = tmp_path / "calls.log"
    fake_inference.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "rows = json.load(sys.stdin)\n"
        "with open(os.environ['CALL_LOG'], 'a', encoding='utf-8') as handle:\n"
        "    handle.write(sys.argv[sys.argv.index('--model-id') + 1] + ':' + str(len(rows)) + '\\n')\n"
        "print(json.dumps([{'status': 'passed', 'answer': 'stable', 'video_id': row['video_id'], 'peak_memory_bytes': 11, 'runtime': 'fake', 'runtime_image': 'fake'} for row in rows]))\n",
        encoding="utf-8",
    )
    fake_inference.chmod(0o755)
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
    assert (output / "conditions/4b-current/predictions.json").read_bytes() == (
        output / "conditions/4b-current-repeat/predictions.json"
    ).read_bytes()
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
