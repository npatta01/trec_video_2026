from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from trec_video_2026.vqa_gate import (
    Condition,
    GateStore,
    RunnerRequest,
    build_rocm_runner_command,
    invoke_runner,
    invoke_runner_batch,
    load_conditions,
    main,
    render_gate_report,
    select_latest_results,
    validate_runner_result,
)


def request(tmp_path: Path) -> RunnerRequest:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    return RunnerRequest(
        run_id="gate",
        condition_id="fake",
        video_id="video000001",
        question="What happens?",
        frame_paths=(frame,),
        frame_timestamps=(0.5,),
        transcript=None,
        media_path=None,
        audio_path=None,
    )


def test_runner_request_never_serializes_reference(tmp_path: Path) -> None:
    payload = request(tmp_path).to_dict()

    assert payload["question"] == "What happens?"
    assert "reference" not in json.dumps(payload).lower()
    assert "answer" not in payload


def test_runner_request_serializes_frame_descriptions(tmp_path: Path) -> None:
    payload = dataclasses.replace(
        request(tmp_path), frame_descriptions=("Full frame:",)
    ).to_dict()
    assert payload["frame_descriptions"] == ["Full frame:"]
    assert "reference" not in json.dumps(payload).lower()


def test_validate_runner_result_requires_provenance() -> None:
    with pytest.raises(ValueError, match="model_revision"):
        validate_runner_result(
            {
                "status": "passed",
                "answer": "A jump",
                "elapsed_seconds": 1.2,
                "model_id": "fake/model",
            }
        )


def test_invoke_runner_classifies_timeout(tmp_path: Path) -> None:
    result = invoke_runner(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        request(tmp_path),
        timeout_seconds=0.01,
    )

    assert result["status"] == "timed_out"
    assert result["answer"] is None
    assert result["error_type"] == "TimeoutExpired"


def test_invoke_runner_accepts_single_json_result(tmp_path: Path) -> None:
    payload = {
        "status": "passed",
        "answer": "A jump",
        "elapsed_seconds": 0.01,
        "model_id": "fake/model",
        "model_revision": "a" * 40,
        "runtime": "fake 1.0",
        "precision": "float32",
        "device": "cpu",
        "peak_memory_bytes": 123,
        "error_type": None,
        "error_message": None,
    }
    script = "import json; print(json.dumps(" + repr(payload) + "))"

    result = invoke_runner(
        [sys.executable, "-c", script], request(tmp_path), timeout_seconds=2
    )

    assert result == payload


def test_invoke_runner_batch_preserves_request_order(tmp_path: Path) -> None:
    payloads = [
        {
            "status": "passed",
            "answer": answer,
            "elapsed_seconds": 0.01,
            "model_id": "fake/model",
            "model_revision": "a" * 40,
            "runtime": "fake 1.0",
            "precision": "float32",
            "device": "cpu",
            "peak_memory_bytes": 123,
            "error_type": None,
            "error_message": None,
        }
        for answer in ("first", "second")
    ]
    script = "import json; print(json.dumps(" + repr(payloads) + "))"
    requests = [request(tmp_path), request(tmp_path)]

    results = invoke_runner_batch(
        [sys.executable, "-c", script], requests, timeout_seconds=2
    )

    assert [result["answer"] for result in results] == ["first", "second"]


def test_store_is_append_only_and_preserves_success_on_resume(tmp_path: Path) -> None:
    store = GateStore(tmp_path / "run")
    failed = {
        "run_id": "gate",
        "condition_id": "fake",
        "video_id": "video000001",
        "status": "failed",
        "answer": None,
    }
    passed = {**failed, "status": "passed", "answer": "A jump"}
    later_failure = {**failed, "error_type": "OutOfMemoryError"}

    store.append_attempt(failed)
    assert store.completed("fake", "video000001") is None
    store.append_attempt(passed)
    store.append_attempt(later_failure)

    assert store.completed("fake", "video000001") == passed
    lines = store.attempts_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["status"] for line in lines] == [
        "failed",
        "passed",
        "failed",
    ]


def test_store_rejects_mismatched_identity(tmp_path: Path) -> None:
    store = GateStore(tmp_path / "run")

    with pytest.raises(ValueError, match="identity"):
        store.append_attempt(
            {
                "run_id": "gate",
                "condition_id": "fake",
                "video_id": "video000001",
                "status": "passed",
                "answer": "A jump",
                "request": {"condition_id": "different"},
            }
        )


def test_load_conditions_rejects_duplicate_ids(tmp_path: Path) -> None:
    config = tmp_path / "models.json"
    condition = {
        "condition_id": "same",
        "model_id": "fake/model",
        "model_revision": "a" * 40,
        "evidence": "controlled-12-frames",
        "download_bytes": 1,
    }
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_image": "local/image:pinned",
                "runtime_base_digest": "sha256:" + "b" * 64,
                "transformers_revision": "c" * 40,
                "conditions": [condition, condition],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate condition"):
        load_conditions(config)


def test_rocm_command_mounts_only_required_roots(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dataset_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    for path in (repo_root, dataset_root, cache_root):
        path.mkdir()
    condition = Condition(
        condition_id="qwen",
        model_id="Qwen/example",
        model_revision="a" * 40,
        evidence="controlled-12-frames",
        download_bytes=123,
    )

    command = build_rocm_runner_command(
        image="local/image:pinned",
        repo_root=repo_root,
        dataset_root=dataset_root,
        hf_cache_root=cache_root,
        condition=condition,
    )

    joined = " ".join(command)
    assert "--network=none" in command
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert "/dev/kfd" in joined and "/dev/dri" in joined
    assert str(repo_root.resolve()) + ":" + str(repo_root.resolve()) + ":ro" in command
    assert str(dataset_root.resolve()) + ":" + str(dataset_root.resolve()) + ":ro" in command
    assert "/home/npatta01:/home/npatta01" not in joined
    assert condition.model_revision in command
    assert command[-1] == "32"
    assert "--model-class" in command


def test_rocm_command_dispatches_specialist_runner(tmp_path: Path) -> None:
    for path in (tmp_path / "repo", tmp_path / "dataset", tmp_path / "cache"):
        path.mkdir()
    condition = Condition(
        condition_id="internvl",
        model_id="OpenGVLab/model",
        model_revision="a" * 40,
        evidence="native-capped-tiles",
        download_bytes=1,
        runner_kind="internvl",
    )

    command = build_rocm_runner_command(
        image="local/image:pinned",
        repo_root=tmp_path / "repo",
        dataset_root=tmp_path / "dataset",
        hf_cache_root=tmp_path / "cache",
        condition=condition,
    )

    assert "trec_video_2026.vqa_specialist_runner" in command
    assert command[command.index("--kind") + 1] == "internvl"


def test_rocm_command_uses_condition_specific_runtime_image(tmp_path: Path) -> None:
    for path in (tmp_path / "repo", tmp_path / "dataset", tmp_path / "cache"):
        path.mkdir()
    condition = Condition(
        condition_id="molmo",
        model_id="allenai/Molmo2-4B",
        model_revision="a" * 40,
        evidence="controlled-12-frames",
        download_bytes=1,
        runtime_image="local/molmo:pinned",
    )

    command = build_rocm_runner_command(
        image="local/default:pinned",
        repo_root=tmp_path / "repo",
        dataset_root=tmp_path / "dataset",
        hf_cache_root=tmp_path / "cache",
        condition=condition,
    )

    assert "local/molmo:pinned" in command
    assert "local/default:pinned" not in command
    assert "VQA_RUNTIME_IMAGE=local/molmo:pinned" in command


def test_latest_results_prefer_success_over_later_failure() -> None:
    attempts = [
        {
            "condition_id": "model-a",
            "video_id": "video-1",
            "status": "passed",
            "answer": "red",
        },
        {
            "condition_id": "model-a",
            "video_id": "video-1",
            "status": "failed",
            "answer": None,
        },
        {
            "condition_id": "model-b",
            "video_id": "video-1",
            "status": "failed",
            "answer": None,
        },
    ]

    selected = select_latest_results(attempts)

    assert selected[("model-a", "video-1")]["answer"] == "red"
    assert selected[("model-b", "video-1")]["status"] == "failed"


def test_report_labels_three_examples_as_descriptive_not_ranked() -> None:
    report = render_gate_report(
        gate_id="three-video-gate",
        references={"video-1": "red"},
        attempts=[
            {
                "condition_id": "model-a",
                "video_id": "video-1",
                "status": "passed",
                "answer": "Red.",
                "elapsed_seconds": 1.5,
                "peak_memory_bytes": 1024,
                "model_revision": "a" * 40,
                "device": "cuda:0 (ROCm)",
            }
        ],
    )

    assert "descriptive portability gate" in report
    assert "not a statistical ranking" in report
    assert "| model-a | video-1 | passed | Red. | red |" in report
    assert "Normalized EM" in report


def test_preflight_cli_reports_all_three_real_media(capsys) -> None:
    root = Path(__file__).resolve().parents[1]

    assert (
        main(
            [
                "preflight",
                "--config",
                str(root / "configs/vqa/three_video_gate.json"),
                "--dataset-root",
                str(root / "data"),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["gate_id"] == "three-video-gate"
    assert len(payload["examples"]) == 3
    assert all(row["has_video"] and row["has_audio"] for row in payload["examples"])
