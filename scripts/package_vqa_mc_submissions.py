#!/usr/bin/env python3
"""Package and validate the three TRECVID 2026 VQA-MC candidates."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK2_COMMIT = "34f1d1e"
TASK2_PATH = "datasets/trecvid-2026/vqa/task2/vqa.2026.testingDataset.Task2.json"
OUTPUT_ROOT = ROOT / "submissions/trecvid-2026/vqa-mc"
PRIVATE_ROOT = ROOT / "data/vqa/submissions/trecvid-2026-vqa-mc-final-20260823"
PARENT_ROOT = ROOT / "data/vqa/benchmarks/vqa-mc-gemini-direct-v2-v3-20260822-a1"
REPAIR_ROOT = (
    ROOT
    / "data/vqa/benchmarks/vqa-mc-gemini-direct-v2b-v3b-repair9-20260823-a1"
)
TRANSCODE_ROOT = (
    ROOT
    / "data/vqa/benchmarks/vqa-mc-gemini-direct-v2c-v3c-repair3-transcode-20260823-a1"
)
WEAK_QUEUE = (
    ROOT
    / "data/vqa/benchmarks/vqa-mc-text-audit-20260822-a1/video-review-queue.jsonl"
)
SOURCES = {
    "g37q36_mc_text": (
        ROOT / "data/vqa/benchmarks/vqa-mc-text-rank1-20260822-a1/submission.json"
    ),
    "g37_mc_target": REPAIR_ROOT / "submission-v2b.json",
    "g37_mc_direct": REPAIR_ROOT / "submission-v3b.json",
}
EXPECTED_TAGS = frozenset(SOURCES)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _write_exact(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f"immutable packaged artifact changed: {path}")
        return
    path.write_bytes(data)


def _write_json(path: Path, value: Any) -> None:
    _write_exact(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )


def validate_mc_submission(
    *,
    submission: Any,
    task_rows: Sequence[Mapping[str, Any]],
    expected_count: int = 2_443,
) -> dict[str, Any]:
    """Enforce the official MC JSON shape and exact released option coverage."""

    if not isinstance(submission, list) or len(submission) != expected_count:
        raise ValueError("submission row count differs from Task 2")
    if len(task_rows) != expected_count:
        raise ValueError("Task 2 row count is invalid")
    option_counts: Counter[int] = Counter()
    seen: set[int] = set()
    for index, (task, row) in enumerate(zip(task_rows, submission, strict=True)):
        if not isinstance(row, Mapping) or set(row) != {"Q_ID", "Video_ID", "Answers"}:
            raise ValueError(f"submission row {index} has invalid fields")
        query_id = task.get("Q_ID")
        video_id = task.get("Video_ID")
        if (
            isinstance(query_id, bool)
            or not isinstance(query_id, int)
            or query_id in seen
            or row.get("Q_ID") != query_id
            or row.get("Video_ID") != video_id
        ):
            raise ValueError(f"submission identity/order mismatch at row {index}")
        seen.add(query_id)
        raw_options = task.get("Options")
        if not isinstance(raw_options, list):
            raise ValueError(f"Task 2 options are invalid for Q_ID {query_id}")
        options = [
            option.get("option") if isinstance(option, Mapping) else None
            for option in raw_options
        ]
        if any(not isinstance(option, str) for option in options):
            raise ValueError(f"Task 2 options are invalid for Q_ID {query_id}")
        answers = row.get("Answers")
        if not isinstance(answers, list) or len(answers) != len(options):
            raise ValueError(f"answer count differs from released options for Q_ID {query_id}")
        for rank, answer in enumerate(answers, start=1):
            if (
                not isinstance(answer, Mapping)
                or set(answer) != {"Rank", "Answer"}
                or answer.get("Rank") != rank
                or isinstance(answer.get("Rank"), bool)
                or not isinstance(answer.get("Answer"), str)
            ):
                raise ValueError(f"ranked answer is invalid for Q_ID {query_id}")
        if Counter(answer["Answer"] for answer in answers) != Counter(options):
            raise ValueError(f"ranking does not contain the exact released options for Q_ID {query_id}")
        option_counts[len(options)] += 1
    return {
        "row_count": len(submission),
        "option_count_distribution": {
            str(count): rows for count, rows in sorted(option_counts.items())
        },
    }


def package_candidates(
    *,
    task_rows: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Path],
    output_root: Path,
    expected_count: int = 2_443,
) -> dict[str, Any]:
    """Copy exact validated source bytes to stable organizer-upload paths."""

    if set(sources) != EXPECTED_TAGS:
        raise ValueError("candidate run tags differ from the approved set")
    runs: dict[str, Any] = {}
    for tag in sorted(sources):
        source = Path(sources[tag])
        data = source.read_bytes()
        validation = validate_mc_submission(
            submission=json.loads(data),
            task_rows=task_rows,
            expected_count=expected_count,
        )
        target = Path(output_root) / tag / "submission.json"
        _write_exact(target, data)
        runs[tag] = {
            "path": str(target),
            "sha256": _sha256(data),
            "size_bytes": len(data),
            **validation,
        }
    return {"run_count": len(runs), "runs": runs}


def build_exception_tracker(
    *,
    parent_freeze: Mapping[str, Any],
    composite_repair_freeze: Mapping[str, Any],
    weak_query_ids: set[int],
) -> dict[str, Any]:
    """Record unavailable media and persistent provider failures separately."""

    parent_rows = parent_freeze.get("rows")
    repair_rows = composite_repair_freeze.get("rows")
    if not isinstance(parent_rows, list) or not isinstance(repair_rows, list):
        raise ValueError("exception source rows are invalid")
    exceptions: list[dict[str, Any]] = []
    for row in parent_rows:
        if not isinstance(row, Mapping) or row.get("status") != "unavailable":
            continue
        query_id = int(row["query_id"])
        affected = ["g37_mc_direct"]
        if query_id in weak_query_ids:
            affected.insert(0, "g37_mc_target")
        exceptions.append(
            {
                "Q_ID": query_id,
                "Video_ID": str(row["video_id"]),
                "affected_runs": affected,
                "fallback": "g37q36_mc_text",
                "status": "media_unavailable",
            }
        )
    for row in repair_rows:
        if not isinstance(row, Mapping) or row.get("status") != "invalid":
            continue
        query_id = int(row["query_id"])
        affected = ["g37_mc_direct"]
        if query_id in weak_query_ids:
            affected.insert(0, "g37_mc_target")
        exceptions.append(
            {
                "Q_ID": query_id,
                "Video_ID": str(row["video_id"]),
                "affected_runs": affected,
                "fallback": "g37q36_mc_text",
                "status": "persistent_provider_invalid",
            }
        )
    exceptions.sort(key=lambda row: int(row["Q_ID"]))
    return {
        "schema_version": 1,
        "counts": {
            "media_unavailable": sum(
                row["status"] == "media_unavailable" for row in exceptions
            ),
            "persistent_provider_invalid": sum(
                row["status"] == "persistent_provider_invalid" for row in exceptions
            ),
            "v2_fallback": sum(
                "g37_mc_target" in row["affected_runs"] for row in exceptions
            ),
            "v3_fallback": sum(
                "g37_mc_direct" in row["affected_runs"] for row in exceptions
            ),
        },
        "exceptions": exceptions,
    }


def _task2() -> list[Mapping[str, Any]]:
    try:
        raw = subprocess.run(
            ["git", "show", f"{TASK2_COMMIT}:{TASK2_PATH}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        value = json.loads(raw)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ValueError("frozen Task 2 source is unavailable") from error
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError("frozen Task 2 source is invalid")
    return value


def _weak_query_ids() -> set[int]:
    result: set[int] = set()
    for line in WEAK_QUEUE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.add(int(json.loads(line)["Q_ID"]))
    if len(result) != 818:
        raise ValueError("weak queue coverage is invalid")
    return result


def main() -> int:
    task = _task2()
    packaged = package_candidates(
        task_rows=task,
        sources=SOURCES,
        output_root=OUTPUT_ROOT,
    )
    tracker = build_exception_tracker(
        parent_freeze=_json(PARENT_ROOT / "predictions-freeze.json"),
        composite_repair_freeze=_json(TRANSCODE_ROOT / "composite-repair-freeze.json"),
        weak_query_ids=_weak_query_ids(),
    )
    _write_json(PRIVATE_ROOT / "exceptions.json", tracker)
    catalog = {
        "schema_version": 1,
        "status": "locally_validated_not_uploaded",
        "competition": "TRECVID 2026 Video Question Answering",
        "task": "Multiple Choice",
        "official_format_url": "https://www-nlpir.nist.gov/projects/tv2026/vqa.html#Runs",
        "official_max_runs": 5,
        "selected_run_count": 3,
        "task2_commit": TASK2_COMMIT,
        "task2_path": TASK2_PATH,
        "runs": packaged["runs"],
        "exception_tracker": str(PRIVATE_ROOT / "exceptions.json"),
        "exception_counts": tracker["counts"],
    }
    _write_json(PRIVATE_ROOT / "catalog.json", catalog)
    print(json.dumps(catalog, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
