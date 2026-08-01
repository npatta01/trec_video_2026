from __future__ import annotations

import importlib.util
from pathlib import Path


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
    args = parser.parse_args(
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
    assert not hasattr(args, "qwen_verified")
    assert args.output_dir == Path("/private/grading")


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
