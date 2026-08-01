from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

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
