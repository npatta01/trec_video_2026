from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

import trec_video_2026.vqa_gate as vqa_gate
from trec_video_2026.vqa_gate import (
    Condition,
    GateStore,
    RunnerRequest,
    build_runner_request,
    build_rocm_runner_command,
    canonical_request_sha256,
    invoke_runner,
    invoke_runner_batch,
    load_conditions,
    main,
    render_gate_report,
    saved_request_matches,
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


def validate_revision_a(_request, attempt):
    if attempt.get("model_revision") != "a" * 40:
        raise ValueError("wrong revision")
    return dict(attempt)


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


def test_canonical_request_sha256_is_order_independent_and_mutation_sensitive() -> None:
    first = {
        "run_id": "gate",
        "condition_id": "model",
        "video_id": "video",
        "transcript": None,
        "frame_paths": ["/frames/one.jpg"],
    }
    reordered = dict(reversed(first.items()))

    assert canonical_request_sha256(first) == canonical_request_sha256(reordered)
    assert canonical_request_sha256(first) != canonical_request_sha256(
        {**first, "transcript": "changed"}
    )
    assert canonical_request_sha256(first) != canonical_request_sha256(
        {**first, "frame_paths": ["/frames/two.jpg"]}
    )


def test_saved_request_matches_validates_saved_payload_and_optional_digest() -> None:
    current = {
        "run_id": "gate",
        "condition_id": "model",
        "video_id": "video",
        "question": "What happens?",
        "transcript": None,
        "frame_paths": ["/frames/one.jpg"],
    }
    digest = canonical_request_sha256(current)

    assert saved_request_matches({"request": current}, current)
    assert saved_request_matches(
        {"request": current, "request_sha256": digest}, current
    )
    assert not saved_request_matches({}, current)
    assert not saved_request_matches(
        {"request": current, "request_sha256": "0" * 64}, current
    )
    assert not saved_request_matches(
        {"request": {**current, "question": "Changed?"}}, current
    )


def test_standard_condition_builds_serialized_minicpm_audio_pair_request(
    tmp_path: Path,
) -> None:
    condition = Condition(
        condition_id="minicpm-on",
        model_id="openbmb/MiniCPM-o-4_5",
        model_revision="a" * 40,
        evidence="selected-12-frames-native-audio-on",
        download_bytes=1,
        runner_kind="minicpm",
        audio_mode="audio-on",
    )
    base = request(tmp_path)

    built = build_runner_request(
        condition=condition,
        run_id=base.run_id,
        video_id=base.video_id,
        question=base.question,
        frame_paths=base.frame_paths,
        frame_timestamps=base.frame_timestamps,
        transcript=None,
        media_path=tmp_path / "video.mkv",
    )

    assert built.condition_id == condition.condition_id
    assert built.audio_mode == "audio-on"
    assert built.to_dict()["audio_mode"] == "audio-on"


@pytest.mark.parametrize("value", ["auto", "", 7])
def test_audio_mode_validation_rejects_unknown_values(value) -> None:
    with pytest.raises((TypeError, ValueError), match="audio_mode"):
        dataclasses.replace(request(Path("/tmp")), audio_mode=value)


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


def test_podman_timeout_uses_unique_name_and_records_verified_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    nonces = iter(("1" * 12, "2" * 12))

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        if command[:2] == ["podman", "run"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[:2] == ["podman", "stop"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["podman", "rm"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] == ["podman", "container", "exists"]:
            return subprocess.CompletedProcess(command, 1, "", "")
        raise AssertionError(command)

    monkeypatch.setattr(vqa_gate.subprocess, "run", fake_run)
    monkeypatch.setattr(vqa_gate.secrets, "token_hex", lambda _size: next(nonces))
    runner_request = request(tmp_path)
    command = ["podman", "run", "--rm", "local/test:image", "python"]

    result = invoke_runner(command, runner_request, timeout_seconds=1)

    container_name = calls[0][calls[0].index("--name") + 1]
    assert container_name.startswith("trec-vqa-")
    assert calls[1] == ["podman", "stop", "--time", "10", container_name]
    assert calls[2] == ["podman", "rm", "--force", container_name]
    assert calls[3] == ["podman", "container", "exists", container_name]
    assert result["status"] == "timed_out"
    assert result["container_name"] == container_name
    assert result["timeout_cleanup"]["verified"] is True
    assert result["timeout_cleanup"]["container_absent"] is True

    calls.clear()
    invoke_runner(command, runner_request, timeout_seconds=1)
    assert calls[0][calls[0].index("--name") + 1] != container_name


def test_podman_timeout_fails_closed_when_cleanup_cannot_be_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **kwargs):
        command = list(command)
        if command[:2] == ["podman", "run"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[:2] in (["podman", "stop"], ["podman", "rm"]):
            return subprocess.CompletedProcess(command, 125, "", "cleanup failed")
        if command[:3] == ["podman", "container", "exists"]:
            return subprocess.CompletedProcess(command, 125, "", "unknown")
        if command[:2] == ["podman", "inspect"]:
            return subprocess.CompletedProcess(command, 125, "", "unknown")
        raise AssertionError(command)

    monkeypatch.setattr(vqa_gate.subprocess, "run", fake_run)

    result = invoke_runner(
        ["podman", "run", "--rm", "local/test:image", "python"],
        request(tmp_path),
        timeout_seconds=1,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "TimeoutCleanupError"
    assert result["timeout_cleanup"]["verified"] is False
    assert result["timeout_cleanup"]["stop_returncode"] == 125
    assert result["timeout_cleanup"]["remove_returncode"] == 125


def test_podman_timeout_fails_closed_when_cleanup_command_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **kwargs):
        command = list(command)
        if command[:2] == ["podman", "run"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[:2] == ["podman", "stop"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[:2] == ["podman", "rm"]:
            return subprocess.CompletedProcess(command, 125, "", "")
        if command[:3] == ["podman", "container", "exists"]:
            return subprocess.CompletedProcess(command, 125, "", "")
        raise AssertionError(command)

    monkeypatch.setattr(vqa_gate.subprocess, "run", fake_run)

    result = invoke_runner(
        ["podman", "run", "--rm", "local/test:image", "python"],
        request(tmp_path),
        timeout_seconds=1,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "TimeoutCleanupError"
    assert result["timeout_cleanup"]["stop_error"] == "TimeoutExpired"
    assert result["timeout_cleanup"]["verified"] is False


def test_podman_batch_timeout_cleans_one_named_container_for_every_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        if command[:2] == ["podman", "run"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[:2] in (["podman", "stop"], ["podman", "rm"]):
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] == ["podman", "container", "exists"]:
            return subprocess.CompletedProcess(command, 1, "", "")
        raise AssertionError(command)

    monkeypatch.setattr(vqa_gate.subprocess, "run", fake_run)
    requests = [
        request(tmp_path),
        dataclasses.replace(request(tmp_path), video_id="video000002"),
    ]

    results = invoke_runner_batch(
        ["podman", "run", "--rm", "local/test:image", "python"],
        requests,
        timeout_seconds=1,
    )

    container_name = calls[0][calls[0].index("--name") + 1]
    assert [row["status"] for row in results] == ["timed_out", "timed_out"]
    assert {row["container_name"] for row in results} == {container_name}
    assert all(row["timeout_cleanup"]["verified"] is True for row in results)
    assert [call[:2] for call in calls].count(["podman", "stop"]) == 1


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
    current = request(tmp_path).to_dict()
    failed = {
        "run_id": current["run_id"],
        "condition_id": current["condition_id"],
        "video_id": current["video_id"],
        "status": "failed",
        "answer": None,
        "request": current,
        "request_sha256": canonical_request_sha256(current),
        "model_revision": "a" * 40,
    }
    passed = {**failed, "status": "passed", "answer": "A jump"}
    later_failure = {**failed, "error_type": "OutOfMemoryError"}

    store.append_attempt(failed)
    assert store.completed(current, validate_pass=validate_revision_a) is None
    store.append_attempt(passed)
    store.append_attempt(later_failure)

    assert store.completed(current, validate_pass=validate_revision_a) == passed
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


def test_store_exact_request_resume_rejects_mutated_request_and_provenance(
    tmp_path: Path,
) -> None:
    store = GateStore(tmp_path / "run")
    current = request(tmp_path).to_dict()
    passed = {
        "run_id": current["run_id"],
        "condition_id": current["condition_id"],
        "video_id": current["video_id"],
        "status": "passed",
        "answer": "A jump",
        "request": current,
        "request_sha256": canonical_request_sha256(current),
        "model_revision": "a" * 40,
    }
    store.append_attempt(passed)

    def validate_provenance(_request, attempt):
        if attempt.get("model_revision") != "a" * 40:
            raise ValueError("wrong revision")
        return dict(attempt)

    assert (
        store.completed(
            current,
            validate_pass=validate_provenance,
        )
        == passed
    )
    assert (
        store.completed(
            {**current, "question": "Changed?"},
            validate_pass=validate_provenance,
        )
        is None
    )
    corrupted = {**passed, "model_revision": "b" * 40}
    store.append_attempt(corrupted)
    assert (
        store.completed(
            current,
            validate_pass=validate_provenance,
        )
        == passed
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


def test_minicpm_config_loads_explicit_audio_modes() -> None:
    config = load_conditions(Path("configs/vqa/model_runs.json"))
    modes = {
        condition.condition_id: condition.audio_mode
        for condition in config.conditions
        if condition.runner_kind == "minicpm"
    }

    assert modes == {
        "minicpmo45-native-video-audio-off": "audio-off",
        "minicpmo45-native-video-audio-on": "audio-on",
    }


def test_cached_ten_visual_conditions_are_explicit_and_immutable() -> None:
    config = load_conditions(Path("configs/vqa/model_runs.json"))
    by_id = {condition.condition_id: condition for condition in config.conditions}
    expected = {
        "qwen35-4b-12f": (
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "controlled-12-frames",
            "image-text",
            "localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5",
        ),
        "qwen35-4b-24f": (
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "controlled-24-frames",
            "image-text",
            "localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5",
        ),
        "qwen35-9b-12f": (
            "Qwen/Qwen3.5-9B",
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "controlled-12-frames",
            "image-text",
            "localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5",
        ),
        "molmo2-4b-12f": (
            "allenai/Molmo2-4B",
            "042abfa7a38879a376cec03d949eff0aefaa0600",
            "controlled-12-frames",
            "image-text",
            "localhost/trec-vqa-gate-molmo@sha256:f5b655ab67cd62427819ac0992a626394f42cf786b4816132ae8610c47181fee",
        ),
        "internvl35-4b-12f": (
            "OpenGVLab/InternVL3_5-4B",
            "481f6e32467eab4e922ccd7fd6cf420441a62331",
            "controlled-12-frames-native-tiles",
            "internvl",
            "localhost/trec-vqa-gate-internvl@sha256:64c040ec4e2cb76f055661c227c0fc456f8389091f4d3bad77bbcc643f28de12",
        ),
        "gemma4-12b-12f": (
            "google/gemma-4-12b-it",
            "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            "controlled-12-frames",
            "multimodal",
            "localhost/trec-vqa-gate-gemma@sha256:5bbd6f039622f70ad53d62395f189b07dc8fe28e2336c8aa462081abee9b7aaa",
        ),
    }
    assert {
        condition_id: (
            by_id[condition_id].model_id,
            by_id[condition_id].model_revision,
            by_id[condition_id].evidence,
            by_id[condition_id].runner_kind,
            by_id[condition_id].runtime_image or config.runtime_image,
        )
        for condition_id in expected
    } == expected


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
    assert (
        str(dataset_root.resolve()) + ":" + str(dataset_root.resolve()) + ":ro"
        in command
    )
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


def test_rocm_command_dispatches_qwen_omni_runner(tmp_path: Path) -> None:
    for path in (tmp_path / "repo", tmp_path / "dataset", tmp_path / "cache"):
        path.mkdir()
    condition = Condition(
        condition_id="qwen-omni",
        model_id="Qwen/Qwen2.5-Omni-3B",
        model_revision="f" * 40,
        evidence="native-video-audio-off",
        download_bytes=1,
        runner_kind="qwen-omni",
        audio_mode="audio-off",
    )

    command = build_rocm_runner_command(
        image="local/image:pinned",
        repo_root=tmp_path / "repo",
        dataset_root=tmp_path / "dataset",
        hf_cache_root=tmp_path / "cache",
        condition=condition,
    )

    assert "trec_video_2026.vqa_omni_runner" in command
    assert "--model-class" not in command


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
    current = {
        "run_id": "gate",
        "condition_id": "model-a",
        "video_id": "video-1",
        "question": "What color?",
    }
    attempts = [
        {
            **current,
            "status": "passed",
            "answer": "red",
            "request": current,
            "request_sha256": canonical_request_sha256(current),
            "model_revision": "a" * 40,
        },
        {
            **current,
            "status": "failed",
            "answer": None,
            "request": current,
            "request_sha256": canonical_request_sha256(current),
            "model_revision": "orchestrator",
        },
    ]

    selected = select_latest_results(
        attempts,
        requests=[current],
        validate_pass=validate_revision_a,
    )

    assert selected[("model-a", "video-1")]["answer"] == "red"


def test_storage_selection_and_reporting_require_exact_request_validation(
    tmp_path: Path,
) -> None:
    store = GateStore(tmp_path / "run")

    with pytest.raises(TypeError):
        store.completed(
            {
                "run_id": "gate",
                "condition_id": "model-a",
                "video_id": "video-1",
            }
        )
    with pytest.raises(TypeError):
        select_latest_results([])
    with pytest.raises(TypeError):
        vqa_gate.select_results_for_requests([], [])
    with pytest.raises(TypeError):
        render_gate_report(gate_id="gate", references={}, attempts=[])


def test_exact_request_selection_rejects_mutations_and_invalid_provenance() -> None:
    current = {
        "run_id": "gate",
        "condition_id": "model-a",
        "video_id": "video-1",
        "question": "What color?",
    }
    good = {
        **current,
        "status": "passed",
        "answer": "red",
        "request": current,
        "request_sha256": canonical_request_sha256(current),
        "model_revision": "a" * 40,
    }
    changed_request = {
        **good,
        "answer": "blue",
        "request": {**current, "question": "What shape?"},
    }
    changed_request["request_sha256"] = canonical_request_sha256(
        changed_request["request"]
    )
    corrupt_provenance = {**good, "answer": "green", "model_revision": "b" * 40}

    def validate_provenance(_request, attempt):
        if attempt.get("model_revision") != "a" * 40:
            raise ValueError("wrong revision")
        return dict(attempt)

    selected = vqa_gate.select_results_for_requests(
        [current],
        [changed_request, corrupt_provenance, good],
        validate_pass=validate_provenance,
    )

    assert selected == [good]


def test_latest_result_and_report_can_require_an_exact_request() -> None:
    current = {
        "run_id": "gate",
        "condition_id": "model-a",
        "video_id": "video-1",
        "question": "What color?",
    }
    stale = {
        **current,
        "status": "passed",
        "answer": "blue",
        "request": {**current, "question": "What shape?"},
        "elapsed_seconds": 1.0,
        "peak_memory_bytes": 1,
    }
    stale["request_sha256"] = canonical_request_sha256(stale["request"])

    selected = select_latest_results(
        [stale],
        requests=[current],
        validate_pass=validate_revision_a,
    )
    report = render_gate_report(
        gate_id="exact-gate",
        references={"video-1": "red"},
        attempts=[stale],
        requests=[current],
        validate_pass=validate_revision_a,
    )

    assert selected == {}
    assert "| model-a | video-1 | passed |" not in report


def test_report_labels_three_examples_as_descriptive_not_ranked() -> None:
    current = {
        "run_id": "gate",
        "condition_id": "model-a",
        "video_id": "video-1",
        "question": "What color?",
    }
    report = render_gate_report(
        gate_id="three-video-gate",
        references={"video-1": "red"},
        attempts=[
            {
                **current,
                "status": "passed",
                "answer": "Red.",
                "elapsed_seconds": 1.5,
                "peak_memory_bytes": 1024,
                "model_revision": "a" * 40,
                "device": "cuda:0 (ROCm)",
                "request": current,
                "request_sha256": canonical_request_sha256(current),
            }
        ],
        requests=[current],
        validate_pass=validate_revision_a,
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
