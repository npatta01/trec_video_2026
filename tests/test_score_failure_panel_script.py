from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/score_failure_panel.py"


def _load():
    spec = importlib.util.spec_from_file_location("score_failure_panel", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layout(tmp_path: Path, scorer):
    repo = tmp_path / "repo"
    data = repo / "data"
    output = data / "vqa/benchmarks/followups/failure-panel/scoring"
    return scorer.Layout(
        repo_root=repo,
        data_root=data,
        output_dir=output,
        candidate_path=output.parent / "candidate-predictions.jsonl",
        inference_manifest=repo / "configs/vqa/failure_panel_12_inference.json",
        reference_manifest=repo / "configs/vqa/failure_panel_12.json",
        script_path=repo / "scripts/score_failure_panel.py",
    )


def _inference_rows() -> list[dict[str, str]]:
    return [
        {
            "video_id": video_id,
            "question": f"question {video_id}",
            "category": "dialogue" if index < 2 else "visual",
            "media": f"media/{video_id}",
        }
        for index, video_id in enumerate(("a", "b", "c"))
    ]


def _candidate_rows() -> list[dict[str, str]]:
    return [
        {
            "condition_id": condition,
            "video_id": video_id,
            "answer": f"answer {video_id}",
            "status": "passed",
            "model_id": "model",
            "model_revision": "1" * 40,
            "runtime_image": "image@sha256:" + "2" * 64,
            "source_sha256": "0" * 64,
        }
        for condition in ("baseline", "treatment")
        for video_id in ("a", "b", "c")
    ]


def _pin_manifest(scorer, name: str, path: Path) -> None:
    setattr(scorer, name, scorer.sha256_path(path))


def _runtime() -> dict[str, object]:
    return {
        "nltk_version": "3.9.2",
        "punkt_sha256": "3" * 64,
        "scorer_image": "image@sha256:" + "4" * 64,
        "wordnet_sha256": "5" * 64,
    }


def test_container_is_digest_pinned_offline_and_read_only() -> None:
    scorer = _load()

    argv = scorer.container_base_argv()

    assert argv[:3] == ["podman", "run", "--pull=never"]
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert scorer.SCORER_IMAGE.endswith("@" + scorer.SCORER_DIGEST)


def test_freeze_mounts_no_reference_or_annotations(tmp_path: Path) -> None:
    scorer = _load()
    layout = _layout(tmp_path, scorer)

    mounts = scorer.build_mounts("freeze", layout, ())

    sources = {mount.source for mount in mounts}
    assert layout.candidate_path in sources
    assert layout.inference_manifest in sources
    assert layout.reference_manifest not in sources
    inference_mount = next(
        mount for mount in mounts if mount.source == layout.inference_manifest
    )
    assert inference_mount.target == (
        scorer.CANONICAL_ROOT / "configs/vqa/failure_panel_12_inference.json"
    )
    assert not any("annotations" in str(path) for path in sources)
    assert [mount for mount in mounts if mount.mode == "rw"] == [
        scorer.Mount(layout.output_dir, layout.output_dir, "rw")
    ]


def test_evaluate_mounts_only_declared_labels_and_read_only_freeze(
    tmp_path: Path,
) -> None:
    scorer = _load()
    layout = _layout(tmp_path, scorer)
    annotations = tuple(
        layout.data_root / f"vqa/2025/training/annotations/{name}.json"
        for name in ("a", "b", "c")
    )

    mounts = scorer.build_mounts("evaluate", layout, annotations)

    assert layout.candidate_path not in {mount.source for mount in mounts}
    assert (
        scorer.Mount(
            layout.output_dir / "frozen-scorer.py",
            scorer.INSIDE_SCRIPT,
            "ro",
        )
        in mounts
    )
    assert layout.script_path not in {mount.source for mount in mounts}
    assert (
        scorer.Mount(
            layout.reference_manifest,
            scorer.CANONICAL_ROOT / "configs/vqa/failure_panel_12.json",
            "ro",
        )
        in mounts
    )
    for annotation in annotations:
        assert scorer.Mount(annotation, annotation, "ro") in mounts
    for name in ("selected-predictions.jsonl", "prediction-freeze.json"):
        path = layout.output_dir / name
        assert scorer.Mount(path, path, "ro") in mounts


def test_phase_state_prevents_labels_or_stale_outputs_from_crossing_freeze(
    tmp_path: Path,
) -> None:
    scorer = _load()
    layout = _layout(tmp_path, scorer)
    layout.output_dir.mkdir(parents=True)

    scorer._validate_phase_state("freeze", layout)
    (layout.output_dir / "evaluation.json").write_text('{"reference": "secret"}')
    with pytest.raises(ValueError, match="Unexpected freeze"):
        scorer._validate_phase_state("freeze", layout)


def test_freeze_candidates_rejects_labels_and_writes_canonical_hash(
    tmp_path: Path,
) -> None:
    scorer = _load()
    inference = tmp_path / "inference.json"
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "output"
    inference.write_text(json.dumps({"examples": _inference_rows()}))
    _pin_manifest(scorer, "INFERENCE_MANIFEST_SHA256", inference)
    rows = _candidate_rows()
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows))

    scorer.freeze_candidates(
        candidates,
        inference,
        output,
        runtime_provenance=_runtime(),
        scorer_script_path=SCRIPT,
    )

    selected = output / "selected-predictions.jsonl"
    record = json.loads((output / "prediction-freeze.json").read_text())
    assert record["selected_rows"] == 6
    assert record["condition_counts"] == {"baseline": 3, "treatment": 3}
    assert record["selected_prediction_sha256"] == scorer.sha256_path(selected)
    assert selected.read_bytes() == scorer.canonical_jsonl(rows)
    assert record["scorer_script_sha256"] == scorer.sha256_path(SCRIPT)
    assert (output / "frozen-scorer.py").read_bytes() == SCRIPT.read_bytes()

    rows[0]["reference"] = "secret"
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="label-bearing"):
        scorer.freeze_candidates(
            candidates,
            inference,
            output,
            runtime_provenance=_runtime(),
            scorer_script_path=SCRIPT,
        )


def test_freeze_rejects_unpinned_manifest_and_missing_provenance(
    tmp_path: Path,
) -> None:
    scorer = _load()
    inference = tmp_path / "inference.json"
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "output"
    inference.write_text(json.dumps({"examples": _inference_rows()}))
    candidates.write_text("".join(json.dumps(row) + "\n" for row in _candidate_rows()))

    with pytest.raises(ValueError, match="manifest SHA-256"):
        scorer.freeze_candidates(
            candidates,
            inference,
            output,
            runtime_provenance=_runtime(),
            scorer_script_path=SCRIPT,
        )

    _pin_manifest(scorer, "INFERENCE_MANIFEST_SHA256", inference)
    rows = _candidate_rows()
    rows[0]["model_revision"] = ""
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="provenance"):
        scorer.freeze_candidates(
            candidates,
            inference,
            output,
            runtime_provenance=_runtime(),
            scorer_script_path=SCRIPT,
        )

    partial = _candidate_rows()[:2]
    candidates.write_text("".join(json.dumps(row) + "\n" for row in partial))
    with pytest.raises(ValueError, match="exact ordered panel"):
        scorer.freeze_candidates(
            candidates,
            inference,
            output,
            runtime_provenance=_runtime(),
            scorer_script_path=SCRIPT,
        )


def test_runtime_attestation_rejects_unpinned_host_execution() -> None:
    scorer = _load()

    with pytest.raises(ValueError, match="mounted frozen scorer"):
        scorer._attest_scorer_runtime()


def test_assemble_candidates_projects_only_label_free_panel_rows(
    tmp_path: Path,
) -> None:
    scorer = _load()
    inference = tmp_path / "inference.json"
    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate.jsonl"
    inference.write_text(json.dumps({"examples": _inference_rows()}))
    _pin_manifest(scorer, "INFERENCE_MANIFEST_SHA256", inference)
    source.write_text(
        json.dumps(
            [
                *_candidate_rows()[:3],
                {
                    **_candidate_rows()[0],
                    "video_id": "outside-panel",
                },
            ]
        )
    )

    rows = scorer.assemble_candidates(
        [("baseline", source)],
        inference,
        candidate,
    )

    assert [row["video_id"] for row in rows] == ["a", "b", "c"]
    assert all(set(row) == scorer.ALLOWED_CANDIDATE_KEYS for row in rows)
    assert {row["source_sha256"] for row in rows} == {scorer.sha256_path(source)}

    payload = json.loads(source.read_text())
    payload[0]["reference"] = "secret"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="label-bearing"):
        scorer.assemble_candidates(
            [("baseline", source)],
            inference,
            candidate,
        )


def test_evaluate_frozen_rejects_mutation_and_scores_by_condition(
    tmp_path: Path,
) -> None:
    scorer = _load()
    inference = tmp_path / "inference.json"
    candidates = tmp_path / "candidates.jsonl"
    reference = tmp_path / "reference.json"
    output = tmp_path / "output"
    annotation_root = tmp_path / "annotations"
    annotation_root.mkdir()
    examples = _inference_rows()
    inference.write_text(json.dumps({"examples": examples}))
    _pin_manifest(scorer, "INFERENCE_MANIFEST_SHA256", inference)
    reference_examples = []
    for example in examples:
        annotation = annotation_root / f"{example['video_id']}.json"
        annotation.write_text(
            json.dumps(
                {
                    "question": example["question"],
                    "correct_answer": f"answer {example['video_id']}",
                    "video_url": (
                        "https://www.youtube.com/watch?v=" + example["video_id"]
                    ),
                }
            )
        )
        reference_examples.append(
            {
                **example,
                "annotation": str(annotation),
            }
        )
    reference.write_text(json.dumps({"examples": reference_examples}))
    _pin_manifest(scorer, "REFERENCE_MANIFEST_SHA256", reference)
    rows = _candidate_rows()
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows))
    scorer.freeze_candidates(
        candidates,
        inference,
        output,
        runtime_provenance=_runtime(),
        scorer_script_path=SCRIPT,
    )

    report = scorer.evaluate_frozen(
        output,
        inference,
        reference,
        tuple(annotation_root.glob("*.json")),
        meteor_fn=lambda prediction, reference: float(prediction == reference),
        runtime_provenance=_runtime(),
        scorer_script_path=SCRIPT,
    )

    assert report["conditions"]["baseline"]["normalized_exact_match"] == 1.0
    assert report["conditions"]["treatment"]["meteor"] == 1.0
    assert len(report["examples"]) == 6
    assert report["provenance"]["annotation_sha256"] == [
        {
            "sha256": scorer.sha256_path(annotation_root / f"{video_id}.json"),
            "video_id": video_id,
        }
        for video_id in ("a", "b", "c")
    ]

    selected = output / "selected-predictions.jsonl"
    selected.write_bytes(selected.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="freeze"):
        scorer.evaluate_frozen(
            output,
            inference,
            reference,
            tuple(annotation_root.glob("*.json")),
            meteor_fn=lambda _prediction, _reference: 0.0,
            runtime_provenance=_runtime(),
            scorer_script_path=SCRIPT,
        )
