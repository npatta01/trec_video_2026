#!/usr/bin/env python3
"""Run the V3 MC protocol with Gemini 3.8 Flash as the sole treatment change."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import run_vqa_mc_gemini_direct_full as v3


CONTROL_MODEL = "gemini-3.7-flash"
TREATMENT_MODEL = "gemini-3.8-flash"
V3_PROTOCOL = {
    "MODEL_ARM": "native-av-lowres-4fps-high-thinking-direct-options",
    "VIDEO_FPS": 4.0,
    "MAX_OUTPUT_TOKENS": 2048,
    "BATCH_INPUT_USD_PER_MILLION": 0.375,
    "BATCH_OUTPUT_USD_PER_MILLION": 1.875,
}


def configured_runner(runner: Any = v3) -> Any:
    """Bind the existing runner to 3.8 after fail-closed V3 contract checks."""

    drift = {
        name: getattr(runner, name, None)
        for name, expected in V3_PROTOCOL.items()
        if getattr(runner, name, None) != expected
    }
    if getattr(runner, "MODEL", None) not in {CONTROL_MODEL, TREATMENT_MODEL}:
        drift["MODEL"] = getattr(runner, "MODEL", None)
    if drift:
        raise RuntimeError("V3 protocol drift prevents model-only replication")
    runner.MODEL = TREATMENT_MODEL
    return runner


def main(argv: Sequence[str] | None = None) -> int:
    return configured_runner().main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
