from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_exposes_only_bounded_verifier_transitions() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_vqa_frontier_boundary_verifier.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "{prepare,preflight,run,freeze,status}" in completed.stdout
