#!/usr/bin/env python3
"""Run frozen-30 scoring across a label-isolated two-container boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Literal, NamedTuple, Sequence


SCORER_DIGEST = (
    "sha256:effff2801d7677aa4894a641adc5793d6e5072a8cec71817bf925574862c88de"
)
SCORER_IMAGE = f"localhost/trec-vqa-gate-qwen-omni@{SCORER_DIGEST}"
RUN_ID = "native-audio-benchmark-v2"
INSIDE_SCRIPT = Path("/opt/score_frozen30_v2.py")
CANONICAL_ROOT = Path("/home/npatta01/data/competitions/trec_video_2026")
CANONICAL_DATA = CANONICAL_ROOT / "data"
Action = Literal["freeze", "evaluate"]
CONDITIONS = (
    "qwen35-4b-12f-frozen30",
    "qwen35-4b-24f-frozen30",
    "minicpmo45-native-video-audio-off-frozen30",
    "qwen35-4b-12f-whisper-large-v3-frozen30",
)


class Layout(NamedTuple):
    repo_root: Path
    canonical_root: Path
    data_root: Path
    run_dir: Path
    script_path: Path


class InputPaths(NamedTuple):
    media: tuple[Path, ...]
    frame_dirs: tuple[Path, ...]
    annotations: tuple[Path, ...]


class Mount(NamedTuple):
    source: Path
    target: Path
    mode: Literal["ro", "rw"]


class FrozenSnapshot(NamedTuple):
    selected_bytes: bytes
    freeze_bytes: bytes
    rows: tuple[dict[str, Any], ...]
    selected_sha256: str


class _ReferenceBoundary(Exception):
    """Private control-flow sentinel raised at the first label load."""


def container_base_argv() -> list[str]:
    """Return the immutable, offline scorer-container prefix."""

    return [
        "podman",
        "run",
        "--pull=never",
        "--rm",
        "--network=none",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        f"VQA_RUNTIME_IMAGE={SCORER_IMAGE}",
    ]


def build_mounts(
    action: Action,
    layout: Layout,
    inputs: InputPaths,
) -> list[Mount]:
    """Build the exact phase-specific bind-mount allowlist."""

    if action not in {"freeze", "evaluate"}:
        raise ValueError(f"Unknown scoring phase: {action}")
    if len(inputs.media) != 30 or len(inputs.frame_dirs) != 60:
        raise ValueError("Frozen scoring requires exactly 30 media and 60 frame roots")
    staged_source = layout.run_dir / "runtime-src"
    mounts = [
        Mount(layout.run_dir, layout.run_dir, "rw"),
        Mount(staged_source, layout.canonical_root / "src", "ro"),
        Mount(staged_source, staged_source, "ro"),
        Mount(
            layout.repo_root / "configs/vqa/native_audio_benchmark_30_inference.json",
            layout.canonical_root
            / "configs/vqa/native_audio_benchmark_30_inference.json",
            "ro",
        ),
        Mount(
            layout.repo_root / "configs/vqa/native_audio_frozen30_v2_decisions.json",
            layout.canonical_root
            / "configs/vqa/native_audio_frozen30_v2_decisions.json",
            "ro",
        ),
        Mount(layout.script_path, INSIDE_SCRIPT, "ro"),
        *(Mount(path, path, "ro") for path in inputs.media),
        *(Mount(path, path, "ro") for path in inputs.frame_dirs),
    ]
    if action == "evaluate":
        if len(inputs.annotations) != 30:
            raise ValueError("Evaluation requires exactly 30 annotation files")
        mounts.extend(
            [
                Mount(
                    layout.repo_root / "configs/vqa/native_audio_benchmark_30.json",
                    layout.canonical_root
                    / "configs/vqa/native_audio_benchmark_30.json",
                    "ro",
                ),
                *(Mount(path, path, "ro") for path in inputs.annotations),
                Mount(
                    layout.run_dir / "selected-predictions.jsonl",
                    layout.run_dir / "selected-predictions.jsonl",
                    "ro",
                ),
                Mount(
                    layout.run_dir / "selected-predictions-freeze.json",
                    layout.run_dir / "selected-predictions-freeze.json",
                    "ro",
                ),
            ]
        )
    return mounts


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_inputs(action: Action, layout: Layout) -> InputPaths:
    """Resolve and validate the exact saved-request inputs to bind."""

    media: list[Path] = []
    seen_media: set[Path] = set()
    frame_dirs: set[Path] = set()
    frame_files: set[Path] = set()
    expected_video_ids: list[str] | None = None
    media_root = layout.data_root / "vqa/2025/training/videos_original"
    evidence_root = layout.data_root / "vqa/benchmarks/evidence"
    for condition in CONDITIONS:
        request_path = layout.run_dir / "requests" / f"{condition}.json"
        rows = _load_json(request_path)
        if not isinstance(rows, list) or len(rows) != 30:
            raise ValueError(f"{condition} must contain exactly 30 requests")
        video_ids = [str(row.get("video_id")) for row in rows]
        if expected_video_ids is None:
            expected_video_ids = video_ids
        elif video_ids != expected_video_ids:
            raise ValueError("All scoring request files must use one ordered ID set")
        for row in rows:
            media_path = Path(str(row.get("media_path")))
            if not media_path.is_absolute() or not media_path.is_file():
                raise ValueError(f"Missing recorded media file: {media_path}")
            if media_path.resolve() != media_path or media_path.parent != media_root:
                raise ValueError("Recorded media path is outside the exact media root")
            if media_path not in seen_media:
                seen_media.add(media_path)
                media.append(media_path)
            frame_paths = row.get("frame_paths")
            if not isinstance(frame_paths, list):
                raise ValueError("Every saved request must declare frame_paths")
            for raw_path in frame_paths:
                frame_path = Path(str(raw_path))
                if not frame_path.is_absolute() or not frame_path.is_file():
                    raise ValueError(f"Missing recorded frame: {frame_path}")
                if (
                    frame_path.resolve() != frame_path
                    or frame_path.suffix != ".png"
                    or frame_path.parent.parent != evidence_root
                ):
                    raise ValueError(
                        "Recorded frame is outside an exact frame-cache directory"
                    )
                frame_files.add(frame_path)
                frame_dirs.add(frame_path.parent)
    ordered_frame_dirs = tuple(sorted(frame_dirs, key=str))
    if len(media) != 30 or len(ordered_frame_dirs) != 60:
        raise ValueError("Saved requests must resolve to 30 media and 60 frame roots")
    png_count = sum(
        sum(path.is_file() for path in directory.glob("*.png"))
        for directory in ordered_frame_dirs
    )
    if png_count != 1080:
        raise ValueError("The 60 exact frame roots must contain exactly 1,080 PNGs")
    if len(frame_files) != 1080:
        raise ValueError("Saved requests must name exactly 1,080 unique PNGs")

    if action == "freeze":
        return InputPaths(tuple(media), ordered_frame_dirs, ())
    reference_path = layout.repo_root / "configs/vqa/native_audio_benchmark_30.json"
    reference = _load_json(reference_path)
    examples = reference.get("examples") if isinstance(reference, dict) else None
    if not isinstance(examples, list) or len(examples) != 30:
        raise ValueError("Reference manifest must declare exactly 30 examples")
    if [str(item.get("video_id")) for item in examples] != expected_video_ids:
        raise ValueError("Reference manifest order does not match saved requests")
    annotation_root = layout.data_root / "vqa/2025/training/annotations"
    annotations = []
    for item in examples:
        relative = Path(str(item.get("annotation")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Annotation path must be a declared relative path")
        annotation = layout.data_root / relative
        if annotation.parent != annotation_root:
            raise ValueError("Annotation path is outside the exact annotation root")
        annotations.append(annotation)
    annotations = tuple(annotations)
    if len(set(annotations)) != 30 or not all(path.is_file() for path in annotations):
        raise ValueError("Reference manifest must resolve 30 unique annotation files")
    return InputPaths(tuple(media), ordered_frame_dirs, annotations)


def build_container_argv(
    action: Action,
    layout: Layout,
    mounts: Sequence[Mount],
) -> list[str]:
    """Render the reviewed Podman invocation from an exact mount allowlist."""

    argv = container_base_argv()
    argv.extend(["-e", f"PYTHONPATH={layout.canonical_root / 'src'}"])
    for mount in mounts:
        argv.extend(["-v", f"{mount.source}:{mount.target}:{mount.mode}"])
    argv.extend(
        [
            "-w",
            str(layout.canonical_root),
            SCORER_IMAGE,
            "python",
            str(INSIDE_SCRIPT),
            f"--inside={action}",
        ]
    )
    return argv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen-30 label-isolated two-phase scorer"
    )
    parser.add_argument("action", nargs="?", choices=("freeze", "evaluate"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--inside",
        choices=("freeze", "evaluate"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _canonical_selected_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _expected_row_keys(workflow: Any) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for condition in workflow.PROMOTED_ORDER:
        requests = _load_json(workflow.RUN / "requests" / f"{condition}.json")
        if not isinstance(requests, list) or len(requests) != 30:
            raise ValueError("Every frozen condition must have exactly 30 requests")
        keys.extend(
            (str(condition), str(request.get("video_id"))) for request in requests
        )
    if len(keys) != 120:
        raise ValueError("Frozen request order must contain exactly 120 rows")
    return keys


def validate_frozen_artifacts(workflow: Any) -> FrozenSnapshot:
    """Validate exact canonical prediction bytes, order, count, and freeze record."""

    selected_path = workflow.RUN / "selected-predictions.jsonl"
    freeze_path = workflow.RUN / "selected-predictions-freeze.json"
    selected_bytes = selected_path.read_bytes()
    freeze_bytes = freeze_path.read_bytes()
    try:
        rows = tuple(
            json.loads(line)
            for line in selected_bytes.decode("utf-8").splitlines()
            if line.strip()
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Selected predictions are not valid UTF-8 JSONL") from error
    if len(rows) != 120 or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Selected predictions must contain exactly 120 object rows")
    if _canonical_selected_bytes(rows) != selected_bytes:
        raise ValueError("Selected prediction bytes are not canonical")
    actual_keys = [
        (str(row.get("condition_id")), str(row.get("video_id"))) for row in rows
    ]
    if actual_keys != _expected_row_keys(workflow):
        raise ValueError("Selected prediction rows do not match frozen request order")
    selected_sha256 = hashlib.sha256(selected_bytes).hexdigest()
    expected_record = {
        "selected_prediction_sha256": selected_sha256,
        "selected_rows": 120,
    }
    try:
        record = json.loads(freeze_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prediction freeze record is not valid JSON") from error
    if record != expected_record:
        raise ValueError("Prediction freeze record does not match 120 selected rows")
    expected_freeze_bytes = (json.dumps(expected_record, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if freeze_bytes != expected_freeze_bytes:
        raise ValueError("Prediction freeze record bytes are not canonical")
    return FrozenSnapshot(
        selected_bytes,
        freeze_bytes,
        rows,
        selected_sha256,
    )


def inside_freeze(workflow: Any) -> None:
    """Run the staged scorer only through its pre-reference freeze boundary."""

    workflow.configure_run(RUN_ID)
    reference_path = Path(workflow.REFERENCE_MANIFEST_PATH)
    annotations_dir = Path(workflow.DATA) / "vqa/2025/training/annotations"
    if reference_path.exists() or annotations_dir.exists():
        raise ValueError("Freeze phase must not see the reference manifest or labels")
    original_load_manifest = workflow.load_manifest

    def label_boundary(path: Path, *args: Any, **kwargs: Any) -> Any:
        if Path(path) == reference_path:
            raise _ReferenceBoundary
        return original_load_manifest(path, *args, **kwargs)

    workflow.load_manifest = label_boundary
    reached_boundary = False
    try:
        workflow.score()
    except _ReferenceBoundary:
        reached_boundary = True
    finally:
        workflow.load_manifest = original_load_manifest
    if not reached_boundary:
        raise RuntimeError("Staged scorer did not reach the reference-load boundary")
    validate_frozen_artifacts(workflow)


def inside_evaluate(workflow: Any) -> None:
    """Evaluate labels while treating the phase-one freeze as immutable input."""

    workflow.configure_run(RUN_ID)
    before = validate_frozen_artifacts(workflow)
    selected_path = workflow.RUN / "selected-predictions.jsonl"
    freeze_path = workflow.RUN / "selected-predictions-freeze.json"
    original_freeze = workflow.freeze_selected_predictions

    def validate_only_freeze(
        candidate_selected_path: Path,
        candidate_freeze_path: Path,
        rows: Sequence[dict[str, Any]],
    ) -> str:
        if (
            Path(candidate_selected_path) != selected_path
            or Path(candidate_freeze_path) != freeze_path
        ):
            raise ValueError("Scorer requested unexpected prediction freeze paths")
        if len(rows) != 120:
            raise ValueError("Scorer must reselect exactly 120 rows")
        if _canonical_selected_bytes(rows) != before.selected_bytes:
            raise ValueError("Scorer selection differs from frozen prediction bytes")
        current = validate_frozen_artifacts(workflow)
        if current != before:
            raise ValueError("Frozen prediction artifacts changed before evaluation")
        return before.selected_sha256

    workflow.freeze_selected_predictions = validate_only_freeze
    try:
        workflow.score()
    finally:
        workflow.freeze_selected_predictions = original_freeze
    if (
        selected_path.read_bytes() != before.selected_bytes
        or freeze_path.read_bytes() != before.freeze_bytes
    ):
        raise ValueError("Frozen prediction artifacts changed during evaluation")
    after = validate_frozen_artifacts(workflow)
    if (
        after.selected_bytes != before.selected_bytes
        or after.freeze_bytes != before.freeze_bytes
        or after.selected_sha256 != before.selected_sha256
    ):
        raise ValueError("Frozen prediction artifacts changed during evaluation")


def image_inspect_argv() -> list[str]:
    return [
        "podman",
        "image",
        "inspect",
        "--format",
        "{{.Digest}}",
        SCORER_IMAGE,
    ]


def inspect_scorer_image(runner: Any = subprocess.run) -> None:
    """Fail closed unless the local image resolves to the reviewed digest."""

    result = runner(
        image_inspect_argv(),
        check=True,
        capture_output=True,
        text=True,
    )
    observed = result.stdout.strip()
    if observed != SCORER_DIGEST:
        raise ValueError(
            f"Scorer image digest mismatch: expected {SCORER_DIGEST}, got {observed}"
        )


def discover_layout() -> Layout:
    """Resolve the worktree sources and enforce the canonical shared data root."""

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    data_root = (repo_root / "data").resolve(strict=True)
    if data_root != CANONICAL_DATA:
        raise ValueError(
            f"data must resolve to canonical shared dataset: {CANONICAL_DATA}"
        )
    return Layout(
        repo_root=repo_root,
        canonical_root=CANONICAL_ROOT,
        data_root=data_root,
        run_dir=data_root / "vqa/benchmarks/runs" / RUN_ID,
        script_path=script_path,
    )


def validate_mount_sources(mounts: Sequence[Mount]) -> None:
    """Require every reviewed bind source before Podman can create targets."""

    missing = [str(mount.source) for mount in mounts if not mount.source.exists()]
    if missing:
        raise FileNotFoundError(f"Missing bind-mount sources: {missing}")
    writable = [mount for mount in mounts if mount.mode == "rw"]
    if len(writable) != 1 or writable[0].source.name != RUN_ID:
        raise ValueError("The v2 run directory must be the only writable bind mount")


def snapshot_immutable_run(
    run_dir: Path,
    allowed_outputs: set[str],
) -> dict[str, str]:
    """Hash every run file except the phase's explicitly writable outputs."""

    snapshot: dict[str, str] = {}
    for path in sorted(
        run_dir.rglob("*"), key=lambda item: str(item.relative_to(run_dir))
    ):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative in allowed_outputs:
            continue
        snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def require_unchanged_snapshot(
    before: dict[str, str],
    after: dict[str, str],
) -> None:
    if before != after:
        changed = sorted(set(before) | set(after))
        changed = [path for path in changed if before.get(path) != after.get(path)]
        raise ValueError(f"Scoring changed immutable run artifacts: {changed}")


def _snapshot_sha256(snapshot: dict[str, str]) -> str:
    payload = "".join(
        f"{path}\0{digest}\0" for path, digest in sorted(snapshot.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable_plan(
    action: Action,
    mounts: Sequence[Mount],
    container_argv: Sequence[str],
) -> dict[str, Any]:
    return {
        "phase": action,
        "scorer_image": SCORER_IMAGE,
        "inspect_argv": image_inspect_argv(),
        "container_argv": list(container_argv),
        "mounts": [
            {
                "source": str(mount.source),
                "target": str(mount.target),
                "mode": mount.mode,
            }
            for mount in mounts
        ],
    }


def run_host_phase(action: Action, *, dry_run: bool) -> None:
    layout = discover_layout()
    inputs = discover_inputs(action, layout)
    mounts = build_mounts(action, layout, inputs)
    container_argv = build_container_argv(action, layout, mounts)
    plan = _jsonable_plan(action, mounts, container_argv)
    if dry_run:
        print(json.dumps(plan, indent=2))
        return

    validate_mount_sources(mounts)
    inspect_scorer_image()
    allowed_outputs = (
        {"selected-predictions.jsonl", "selected-predictions-freeze.json"}
        if action == "freeze"
        else {"scored-predictions.jsonl", "evaluation.json"}
    )
    before = snapshot_immutable_run(layout.run_dir, allowed_outputs)
    subprocess.run(container_argv, check=True)
    after = snapshot_immutable_run(layout.run_dir, allowed_outputs)
    require_unchanged_snapshot(before, after)
    print(
        json.dumps(
            {
                "phase": action,
                "immutable_run_sha256_before": _snapshot_sha256(before),
                "immutable_run_sha256_after": _snapshot_sha256(after),
            },
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inside is not None:
        if args.action is not None or args.dry_run:
            raise ValueError("--inside cannot be combined with host arguments")
        from trec_video_2026 import vqa_frozen30_workflow as workflow

        if args.inside == "freeze":
            inside_freeze(workflow)
        else:
            inside_evaluate(workflow)
        return 0
    if args.action is None:
        raise ValueError("Host mode requires freeze or evaluate")
    run_host_phase(args.action, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
