from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_BUCKET_PATTERNS = (
    "trec_mlm_2026",
    "hf://buckets/Npatta01/trec-video-2026",
)


def test_tracked_hf_references_use_the_project_bucket() -> None:
    """Catch any tracked configuration that routes project data outside its bucket."""
    for pattern in LEGACY_BUCKET_PATTERNS:
        result = subprocess.run(
            [
                "git",
                "grep",
                "-n",
                "-F",
                pattern,
                "--",
                ".",
                ":(exclude)tests/test_hf_bucket_layout.py",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1, result.stdout
