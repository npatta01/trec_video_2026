from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/score-nemotron3-omni-frozen30.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("score_nemotron_f30", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_requires_path_based_freezes_and_private_output_directory() -> None:
    module = _load_script()
    parser = module.build_parser()
    args, unknown = parser.parse_known_args(
        [
            "prepare",
            "--qwen-predictions",
            "/private/qwen.jsonl",
            "--candidate-predictions",
            "/private/predictions.jsonl",
            "--candidate-manifest",
            "/private/manifest.json",
            "--candidate-checksums",
            "/private/SHA256SUMS",
            "--references",
            "/private/references.json",
            "--source-manifest",
            "/repo/configs/vqa/native_audio_benchmark_30_inference.json",
            "--source-media-root",
            "/private/source-media",
            "--normalized-media-root",
            "/private/downloaded-candidate-bundle/normalized",
            "--output-dir",
            "/private/grading",
            "--grader-id",
            "a",
            "--grader-id",
            "b",
            "--shuffle-seed",
            "1709",
            "--shuffle-seed",
            "2903",
        ]
    )
    assert args.command == "prepare"
    assert unknown == []
    assert not hasattr(args, "qwen_verified")
    assert args.output_dir == Path("/private/grading")
    assert args.source_manifest == Path(
        "/repo/configs/vqa/native_audio_benchmark_30_inference.json"
    )
    assert args.source_media_root == Path("/private/source-media")
    assert args.normalized_media_root == Path(
        "/private/downloaded-candidate-bundle/normalized"
    )


def test_score_requires_private_grader_assignment_artifact() -> None:
    module = _load_script()
    parser = module.build_parser()
    common = [
        "--qwen-predictions",
        "/private/qwen.jsonl",
        "--candidate-predictions",
        "/private/predictions.jsonl",
        "--candidate-manifest",
        "/private/manifest.json",
        "--candidate-checksums",
        "/private/SHA256SUMS",
        "--references",
        "/private/references.json",
        "--source-manifest",
        "/repo/configs/vqa/native_audio_benchmark_30_inference.json",
        "--source-media-root",
        "/private/source-media",
        "--normalized-media-root",
        "/private/downloaded-candidate-bundle/normalized",
        "--output-dir",
        "/private/grading",
        "--grader-id",
        "a",
        "--grader-id",
        "b",
        "--shuffle-seed",
        "1709",
        "--shuffle-seed",
        "2903",
    ]
    args = parser.parse_args(
        [
            "score",
            *common,
            "--grader-submission",
            "/private/a.json",
            "--grader-submission",
            "/private/b.json",
            "--grader-assignments",
            "/private/grader-assignments.json",
            "--operational-evidence",
            "/private/operational.json",
        ]
    )

    assert args.grader_assignments == Path("/private/grader-assignments.json")


def test_prepare_retains_private_grader_assignment_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    packet = {"packet_id": "blind-1-abcd", "grader_id": "grader-1", "items": []}
    artifact = {
        "schema_version": 1,
        "assignments": [
            {
                "grader_id": "Alice",
                "grader_alias": "grader-1",
                "packet_id": "blind-1-abcd",
                "packet_sha256": "a" * 64,
            }
        ],
    }
    monkeypatch.setattr(
        module,
        "_bundle",
        lambda args: SimpleNamespace(
            packets={"grader-1": packet},
            _grader_assignment_artifact=artifact,
        ),
    )
    monkeypatch.setattr(module, "require_private_output_directory", lambda path: path)

    assert module._prepare(SimpleNamespace(output_dir=tmp_path / "private")) == 0
    written = json.loads(
        (tmp_path / "private" / "private-grader-assignments.json").read_text()
    )
    assert written == artifact
    assert "Alice" not in json.dumps(packet)


def test_script_refuses_output_paths_inside_repository(tmp_path: Path) -> None:
    module = _load_script()
    module.REPOSITORY_ROOT = tmp_path
    output = tmp_path / "raw-private"
    try:
        module.require_private_output_directory(output)
    except ValueError as error:
        assert "outside the repository" in str(error)
    else:
        raise AssertionError("repository-local raw output was accepted")


def test_script_refuses_output_inside_any_other_git_repository(tmp_path: Path) -> None:
    module = _load_script()
    repository = tmp_path / "other-repository"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)

    with pytest.raises(ValueError, match="Git"):
        module.require_private_output_directory(repository / "private" / "grading")

    assert not (repository / "private").exists()


def test_script_operational_evidence_requires_exact_schema(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "operational.json"
    path.write_text(
        json.dumps(
            {
                "completed": 60,
                "expected": 60,
                "duration_minutes": 59.0,
                "compute_cost_usd": 1.0,
                "artifacts_verified": True,
                "privacy_or_audit_gap": False,
                "caller_verified": True,
            }
        )
    )

    with pytest.raises(ValueError, match="schema"):
        module._operational(path)
