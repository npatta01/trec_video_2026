from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/score_frozen30_v2.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("score_frozen30_v2", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_container_base_argv_pins_offline_read_only_scorer() -> None:
    scorer = _load_script()

    argv = scorer.container_base_argv()

    assert argv[:3] == ["podman", "run", "--pull=never"]
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert ["--tmpfs", "/tmp"] == argv[
        argv.index("--tmpfs") : argv.index("--tmpfs") + 2
    ]
    assert f"VQA_RUNTIME_IMAGE={scorer.SCORER_IMAGE}" in argv
    assert scorer.SCORER_IMAGE.endswith(
        "@sha256:effff2801d7677aa4894a641adc5793d6e5072a8cec71817bf925574862c88de"
    )


def _mount_fixture(tmp_path: Path, scorer):
    canonical_root = tmp_path / "canonical-repo"
    data_root = canonical_root / "data"
    run = data_root / "vqa/benchmarks/runs/native-audio-benchmark-v2"
    layout = scorer.Layout(
        repo_root=tmp_path / "worktree",
        canonical_root=canonical_root,
        data_root=data_root,
        run_dir=run,
        script_path=tmp_path / "worktree/scripts/score_frozen30_v2.py",
    )
    inputs = scorer.InputPaths(
        media=tuple(tmp_path / f"media/{index}.mkv" for index in range(30)),
        frame_dirs=tuple(tmp_path / f"frames/{index}" for index in range(60)),
        annotations=tuple(
            data_root / f"vqa/2025/training/annotations/{index}.json"
            for index in range(30)
        ),
    )
    return layout, inputs


def test_freeze_mounts_staged_source_and_exclude_all_labels(tmp_path: Path) -> None:
    scorer = _load_script()
    layout, inputs = _mount_fixture(tmp_path, scorer)

    mounts = scorer.build_mounts("freeze", layout, inputs)
    sources = [mount.source for mount in mounts]
    targets = [mount.target for mount in mounts]

    assert (
        scorer.Mount(
            layout.run_dir / "runtime-src",
            layout.canonical_root / "src",
            "ro",
        )
        in mounts
    )
    assert (
        scorer.Mount(
            layout.run_dir / "runtime-src",
            layout.run_dir / "runtime-src",
            "ro",
        )
        in mounts
    )
    assert layout.repo_root / "src" not in sources
    assert (
        layout.canonical_root / "configs/vqa/native_audio_benchmark_30.json"
        not in targets
    )
    assert not set(inputs.annotations) & set(sources)
    assert len([mount for mount in mounts if mount.source in inputs.media]) == 30
    assert len([mount for mount in mounts if mount.source in inputs.frame_dirs]) == 60


def test_evaluate_mounts_only_declared_labels_and_read_only_freeze_overlays(
    tmp_path: Path,
) -> None:
    scorer = _load_script()
    layout, inputs = _mount_fixture(tmp_path, scorer)

    mounts = scorer.build_mounts("evaluate", layout, inputs)
    annotation_mounts = [
        mount for mount in mounts if mount.source in inputs.annotations
    ]

    assert len(annotation_mounts) == 30
    assert all(mount.mode == "ro" for mount in annotation_mounts)
    for name in ("selected-predictions.jsonl", "selected-predictions-freeze.json"):
        assert (
            scorer.Mount(layout.run_dir / name, layout.run_dir / name, "ro") in mounts
        )


def test_mounts_never_expose_broad_label_or_evidence_roots(tmp_path: Path) -> None:
    scorer = _load_script()
    layout, inputs = _mount_fixture(tmp_path, scorer)
    forbidden = {
        layout.repo_root,
        layout.canonical_root,
        layout.data_root,
        layout.data_root / "vqa/benchmarks/evidence",
        layout.data_root / "vqa/2025/training/annotations",
        layout.repo_root / "src",
    }

    for action in ("freeze", "evaluate"):
        mounts = scorer.build_mounts(action, layout, inputs)
        assert forbidden.isdisjoint(mount.source for mount in mounts)
        assert forbidden.isdisjoint(mount.target for mount in mounts)


def test_container_argv_renders_exact_mounts_and_has_no_image_override(
    tmp_path: Path,
) -> None:
    scorer = _load_script()
    layout, inputs = _mount_fixture(tmp_path, scorer)
    mounts = scorer.build_mounts("freeze", layout, inputs)

    argv = scorer.build_container_argv("freeze", layout, mounts)

    assert argv[-4:] == [
        scorer.SCORER_IMAGE,
        "python",
        str(scorer.INSIDE_SCRIPT),
        "--inside=freeze",
    ]
    volumes = [argv[index + 1] for index, value in enumerate(argv) if value == "-v"]
    assert len(volumes) == len(mounts)
    assert all(volume.endswith(":ro") or volume.endswith(":rw") for volume in volumes)
    assert not any("/annotations/" in value for value in argv)
    with pytest.raises(SystemExit):
        scorer.parse_args(["freeze", "--image", "untrusted:latest"])


def test_discover_inputs_requires_30_media_60_frame_dirs_and_1080_pngs(
    tmp_path: Path,
) -> None:
    scorer = _load_script()
    layout, _ = _mount_fixture(tmp_path, scorer)
    request_dir = layout.run_dir / "requests"
    request_dir.mkdir(parents=True)
    media = [
        layout.data_root / f"vqa/2025/training/videos_original/video-{index:02d}.mkv"
        for index in range(30)
    ]
    for path in media:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    requests_by_condition = {}
    for condition, count, offset in (
        ("qwen35-4b-12f-frozen30", 12, 0),
        ("qwen35-4b-24f-frozen30", 24, 30),
        ("minicpmo45-native-video-audio-off-frozen30", 0, 0),
        ("qwen35-4b-12f-whisper-large-v3-frozen30", 12, 0),
    ):
        rows = []
        for index, media_path in enumerate(media):
            frame_dir = (
                layout.data_root / f"vqa/benchmarks/evidence/{offset + index:02d}"
            )
            frame_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for frame_index in range(count):
                frame = frame_dir / f"frame-{frame_index + 1:02d}.png"
                frame.touch()
                frames.append(str(frame))
            rows.append(
                {
                    "condition_id": condition,
                    "video_id": f"video-{index:02d}",
                    "media_path": str(media_path),
                    "frame_paths": frames,
                }
            )
        requests_by_condition[condition] = rows
        (request_dir / f"{condition}.json").write_text(json.dumps(rows))
    reference_config = layout.repo_root / "configs/vqa/native_audio_benchmark_30.json"
    reference_config.parent.mkdir(parents=True)
    annotations = []
    examples = []
    for index in range(30):
        annotation = (
            layout.data_root / f"vqa/2025/training/annotations/video-{index:02d}.json"
        )
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.touch()
        annotations.append(annotation)
        examples.append(
            {
                "video_id": f"video-{index:02d}",
                "annotation": (f"vqa/2025/training/annotations/video-{index:02d}.json"),
            }
        )
    reference_config.write_text(json.dumps({"examples": examples}))

    discovered = scorer.discover_inputs("evaluate", layout)

    assert discovered.media == tuple(media)
    assert len(discovered.frame_dirs) == 60
    assert sum(len(list(path.glob("*.png"))) for path in discovered.frame_dirs) == 1080
    assert discovered.annotations == tuple(annotations)

    reference_config.unlink()
    label_free = scorer.discover_inputs("freeze", layout)
    assert label_free.annotations == ()


def test_discover_inputs_rejects_mounts_outside_exact_media_and_evidence_roots(
    tmp_path: Path,
) -> None:
    scorer = _load_script()
    layout, _ = _mount_fixture(tmp_path, scorer)
    request_dir = layout.run_dir / "requests"
    request_dir.mkdir(parents=True)
    label = layout.data_root / "vqa/2025/training/annotations/secret.json"
    label.parent.mkdir(parents=True)
    label.touch()
    for condition in scorer.CONDITIONS:
        rows = [
            {
                "condition_id": condition,
                "video_id": f"video-{index:02d}",
                "media_path": str(label),
                "frame_paths": [],
            }
            for index in range(30)
        ]
        (request_dir / f"{condition}.json").write_text(json.dumps(rows))

    with pytest.raises(ValueError, match="media root"):
        scorer.discover_inputs("freeze", layout)


def _frozen_rows(scorer) -> list[dict[str, object]]:
    return [
        {
            "answer": f"answer-{condition_index}-{row_index}",
            "condition_id": condition,
            "video_id": f"video-{row_index:02d}",
        }
        for condition_index, condition in enumerate(scorer.CONDITIONS)
        for row_index in range(30)
    ]


def _write_request_order(run: Path, scorer) -> None:
    request_dir = run / "requests"
    request_dir.mkdir(parents=True)
    for condition in scorer.CONDITIONS:
        rows = [
            {"condition_id": condition, "video_id": f"video-{index:02d}"}
            for index in range(30)
        ]
        (request_dir / f"{condition}.json").write_text(json.dumps(rows))


def _write_frozen_artifacts(run: Path, scorer, rows=None) -> tuple[bytes, bytes]:
    rows = _frozen_rows(scorer) if rows is None else rows
    selected = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()
    digest = hashlib.sha256(selected).hexdigest()
    freeze = (
        json.dumps(
            {
                "selected_prediction_sha256": digest,
                "selected_rows": 120,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    (run / "selected-predictions.jsonl").write_bytes(selected)
    (run / "selected-predictions-freeze.json").write_bytes(freeze)
    return selected, freeze


def _fake_workflow(tmp_path: Path, scorer):
    run = tmp_path / "run"
    run.mkdir()
    _write_request_order(run, scorer)
    reference = tmp_path / "repo/configs/vqa/native_audio_benchmark_30.json"
    data = tmp_path / "data"
    calls = {"score": 0, "configure": []}

    def load_manifest(path, _data):
        return {"path": str(path)}

    workflow = SimpleNamespace(
        RUN=run,
        RUN_ID=scorer.RUN_ID,
        DATA=data,
        REFERENCE_MANIFEST_PATH=reference,
        PROMOTED_ORDER=scorer.CONDITIONS,
        load_manifest=load_manifest,
        freeze_selected_predictions=None,
        configure_run=lambda run_id: calls["configure"].append(run_id),
    )
    return workflow, calls


def test_inside_freeze_catches_label_boundary_only_after_valid_120_row_freeze(
    tmp_path: Path,
) -> None:
    scorer = _load_script()
    workflow, calls = _fake_workflow(tmp_path, scorer)

    def score():
        calls["score"] += 1
        _write_frozen_artifacts(workflow.RUN, scorer)
        workflow.load_manifest(workflow.REFERENCE_MANIFEST_PATH, workflow.DATA)

    workflow.score = score

    scorer.inside_freeze(workflow)

    assert calls == {"score": 1, "configure": [scorer.RUN_ID]}
    assert not workflow.REFERENCE_MANIFEST_PATH.exists()
    assert not (workflow.DATA / "vqa/2025/training/annotations").exists()


def test_inside_freeze_rejects_partial_artifacts_after_label_sentinel(
    tmp_path: Path,
) -> None:
    scorer = _load_script()
    workflow, calls = _fake_workflow(tmp_path, scorer)

    def score():
        _write_frozen_artifacts(workflow.RUN, scorer, _frozen_rows(scorer)[:-1])
        workflow.load_manifest(workflow.REFERENCE_MANIFEST_PATH, workflow.DATA)

    workflow.score = score

    with pytest.raises(ValueError, match="120"):
        scorer.inside_freeze(workflow)


@pytest.mark.parametrize("corruption", ["missing", "partial", "digest", "reordered"])
def test_inside_evaluate_rejects_bad_freeze_before_calling_scorer(
    tmp_path: Path,
    corruption: str,
) -> None:
    scorer = _load_script()
    workflow, calls = _fake_workflow(tmp_path, scorer)
    rows = _frozen_rows(scorer)
    selected, _ = _write_frozen_artifacts(workflow.RUN, scorer, rows)
    if corruption == "missing":
        (workflow.RUN / "selected-predictions-freeze.json").unlink()
    elif corruption == "partial":
        _write_frozen_artifacts(workflow.RUN, scorer, rows[:-1])
    elif corruption == "digest":
        record = json.loads(
            (workflow.RUN / "selected-predictions-freeze.json").read_text()
        )
        record["selected_prediction_sha256"] = "0" * 64
        (workflow.RUN / "selected-predictions-freeze.json").write_text(
            json.dumps(record, sort_keys=True) + "\n"
        )
    else:
        lines = selected.splitlines(keepends=True)
        (workflow.RUN / "selected-predictions.jsonl").write_bytes(
            b"".join([lines[1], lines[0], *lines[2:]])
        )
        reordered = (workflow.RUN / "selected-predictions.jsonl").read_bytes()
        (workflow.RUN / "selected-predictions-freeze.json").write_text(
            json.dumps(
                {
                    "selected_prediction_sha256": hashlib.sha256(reordered).hexdigest(),
                    "selected_rows": 120,
                },
                sort_keys=True,
            )
            + "\n"
        )
    workflow.score = lambda: calls.__setitem__("score", calls["score"] + 1)

    with pytest.raises((FileNotFoundError, ValueError)):
        scorer.inside_evaluate(workflow)

    assert calls["score"] == 0


def test_inside_evaluate_validate_only_hook_writes_nothing(tmp_path: Path) -> None:
    scorer = _load_script()
    workflow, calls = _fake_workflow(tmp_path, scorer)
    selected_before, freeze_before = _write_frozen_artifacts(workflow.RUN, scorer)

    def score():
        calls["score"] += 1
        rows = [
            json.loads(line)
            for line in (workflow.RUN / "selected-predictions.jsonl")
            .read_text()
            .splitlines()
        ]
        digest = workflow.freeze_selected_predictions(
            workflow.RUN / "selected-predictions.jsonl",
            workflow.RUN / "selected-predictions-freeze.json",
            rows,
        )
        assert digest == hashlib.sha256(selected_before).hexdigest()

    workflow.score = score

    scorer.inside_evaluate(workflow)

    assert calls == {"score": 1, "configure": [scorer.RUN_ID]}
    assert (workflow.RUN / "selected-predictions.jsonl").read_bytes() == selected_before
    assert (
        workflow.RUN / "selected-predictions-freeze.json"
    ).read_bytes() == freeze_before


def test_inside_evaluate_rejects_post_score_frozen_artifact_mutation(
    tmp_path: Path,
) -> None:
    scorer = _load_script()
    workflow, _ = _fake_workflow(tmp_path, scorer)
    _write_frozen_artifacts(workflow.RUN, scorer)

    def score():
        (workflow.RUN / "selected-predictions.jsonl").write_text("mutated\n")

    workflow.score = score

    with pytest.raises(ValueError, match="changed"):
        scorer.inside_evaluate(workflow)


def test_image_inspect_requires_exact_hard_coded_digest() -> None:
    scorer = _load_script()
    calls = []

    def good_runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=scorer.SCORER_DIGEST + "\n")

    scorer.inspect_scorer_image(good_runner)

    assert calls[0][0] == [
        "podman",
        "image",
        "inspect",
        "--format",
        "{{.Digest}}",
        scorer.SCORER_IMAGE,
    ]
    assert calls[0][1]["check"] is True
    assert calls[0][1]["text"] is True
    with pytest.raises(ValueError, match="digest"):
        scorer.inspect_scorer_image(
            lambda argv, **kwargs: subprocess.CompletedProcess(
                argv, 0, stdout="sha256:" + "0" * 64 + "\n"
            )
        )


def test_dry_run_prints_inspect_and_container_argv_without_execution(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    scorer = _load_script()
    layout, inputs = _mount_fixture(tmp_path, scorer)
    monkeypatch.setattr(scorer, "discover_layout", lambda: layout)
    monkeypatch.setattr(scorer, "discover_inputs", lambda _action, _layout: inputs)
    monkeypatch.setattr(scorer, "validate_mount_sources", lambda _mounts: None)
    monkeypatch.setattr(
        scorer.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry-run executed Podman"),
    )

    assert scorer.main(["freeze", "--dry-run"]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["phase"] == "freeze"
    assert plan["inspect_argv"][-1] == scorer.SCORER_IMAGE
    assert plan["container_argv"][-1] == "--inside=freeze"
    assert len(plan["mounts"]) == 96


def test_immutable_run_snapshot_allows_only_declared_outputs(tmp_path: Path) -> None:
    scorer = _load_script()
    run = tmp_path / "run"
    run.mkdir()
    (run / "attempts.jsonl").write_text("immutable\n")
    before = scorer.snapshot_immutable_run(
        run, {"selected-predictions.jsonl", "selected-predictions-freeze.json"}
    )
    (run / "selected-predictions.jsonl").write_text("allowed\n")
    after_allowed = scorer.snapshot_immutable_run(
        run, {"selected-predictions.jsonl", "selected-predictions-freeze.json"}
    )
    scorer.require_unchanged_snapshot(before, after_allowed)
    (run / "attempts.jsonl").write_text("mutated\n")
    after_mutation = scorer.snapshot_immutable_run(
        run, {"selected-predictions.jsonl", "selected-predictions-freeze.json"}
    )

    with pytest.raises(ValueError, match="immutable"):
        scorer.require_unchanged_snapshot(before, after_mutation)
