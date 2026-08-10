#!/usr/bin/env python3
"""Freeze, serialize, and validate the private Qwen safety submission."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from trec_video_2026.vqa_safety_submission import (
    finalize_submission,
    run_nist_validator,
    sha256_path,
)


VALIDATOR_REVISION = "e06cb7854d7defe53e86412c2bba0ca5f636ea40"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validator-checkout", type=Path, required=True)
    return parser.parse_args()


def _verify_validator_checkout(checkout: Path) -> Path:
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != VALIDATOR_REVISION:
        raise ValueError(f"validator revision mismatch: {revision}")
    script = checkout / "validate.ag.run.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    return script


def main() -> int:
    args = parse_args()
    validator = _verify_validator_checkout(args.validator_checkout)
    outputs = finalize_submission(args.manifest, args.predictions, args.output_dir)
    report = args.output_dir / "validator-report.json"
    completed = run_nist_validator(
        validator_script=validator,
        topics_path=args.topics,
        submission_path=outputs["top10"],
        report_path=report,
    )
    (args.output_dir / "validator.stdout").write_text(
        completed.stdout, encoding="utf-8"
    )
    record = {
        "validator_revision": VALIDATOR_REVISION,
        "validator_script_sha256": sha256_path(validator),
        "validator_report_sha256": sha256_path(report),
        "submission_top10_sha256": sha256_path(outputs["top10"]),
        "status": "passed",
    }
    (args.output_dir / "validator-identity.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(outputs["checksums"], args.output_dir / "SHA256SUMS.pre-validator")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
