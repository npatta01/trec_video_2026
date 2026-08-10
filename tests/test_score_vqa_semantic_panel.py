from __future__ import annotations

from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.error import URLError

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/score_vqa_semantic_panel.py"
EXPECTED_BERT_MODEL_SNAPSHOT = (
    "722cf37b1afa9454edce342e7895e588b6ff1d59"
)


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_vqa_semantic_panel", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_inputs(tmp_path: Path) -> tuple[Path, dict[str, Path], Path]:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for video_id, question, answer in (
        ("v1", "What is shown?", "the cat"),
        ("v2", "What is shown?", "a dog"),
    ):
        (annotations / f"{video_id}.json").write_text(
            json.dumps(
                {
                    "question": question,
                    "correct_answer": answer,
                    "incorrect_answers": [],
                }
            ),
            encoding="utf-8",
        )
    conditions = {}
    for condition_id, answers in (
        ("frames", ("cat", "wrong")),
        ("whisper", ("dog", "dog")),
    ):
        path = tmp_path / f"{condition_id}.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "status": "passed",
                        "answer": answer,
                        "video_id": video_id,
                    }
                    for video_id, answer in zip(("v1", "v2"), answers, strict=True)
                ]
            ),
            encoding="utf-8",
        )
        conditions[condition_id] = path
    model = tmp_path / EXPECTED_BERT_MODEL_SNAPSHOT
    model.mkdir()
    (model / "config.json").write_text("frozen model fixture", encoding="utf-8")
    return annotations, conditions, model


def injected_metrics(calls: dict[str, int]):
    def meteor(_prediction: str, _reference: str) -> float:
        return 0.2

    def bertscore(
        predictions: list[str], references: list[str], _model: Path
    ) -> list[float]:
        calls["bert"] += 1
        assert len(predictions) == len(references) == 2
        return [0.8] * len(predictions)

    def sts(_prediction: str, _reference: str) -> float:
        return 0.5

    return meteor, bertscore, sts


def fixture_panel_profile(scorer):
    return scorer.PanelProfile(
        name="fixture-v1-v2",
        expected_annotation_files=2,
        expected_rows=2,
        canonical_id_set_sha256=scorer.sorted_video_id_set_sha256({"v1", "v2"}),
        excluded_annotation_ids=(),
        excluded_reason="test fixture",
    )


def test_score_panel_joins_labels_after_freeze_and_retains_metrics(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    calls = {"bert": 0}
    meteor, bertscore, sts = injected_metrics(calls)

    artifact = scorer.score_panel(
        annotation_root=annotations,
        condition_paths=conditions,
        bert_model=model,
        expected_rows=2,
        expected_annotation_files=2,
        meteor_fn=meteor,
        bertscore_fn=bertscore,
        sts_fn=sts,
    )

    assert artifact["panel"]["completed_per_condition"] == {
        "frames": 2,
        "whisper": 2,
    }
    assert artifact["conditions"]["frames"]["semantic_composite"] == pytest.approx(0.5)
    assert artifact["integrity"]["labels_joined_after_prediction_freeze"] is True
    assert len(artifact["rows"]["frames"]) == 2
    assert artifact["rows"]["frames"][0]["semantic_composite"] == pytest.approx(0.5)
    assert calls["bert"] == 2


def test_artifact_is_sanitized_to_ids_strata_numeric_metrics_counts_and_hashes(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    meteor, bertscore, sts = injected_metrics({"bert": 0})

    artifact = scorer.score_panel(
        annotation_root=annotations,
        condition_paths=conditions,
        bert_model=model,
        expected_rows=2,
        expected_annotation_files=2,
        meteor_fn=meteor,
        bertscore_fn=bertscore,
        sts_fn=sts,
    )

    forbidden_keys = {
        "answer",
        "correct_answer",
        "expected_answer",
        "normalized_answer",
        "normalized_reference",
        "question",
        "reference",
        "transcript",
    }
    forbidden_values = {"What is shown?", "the cat", "a dog", "cat", "dog"}

    def assert_safe(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                assert_safe(child)
        elif isinstance(value, list):
            for child in value:
                assert_safe(child)
        elif isinstance(value, str):
            assert value not in forbidden_values
            assert not value.startswith("/tmp/")

    assert_safe(artifact)
    assert set(artifact["rows"]["frames"][0]) == {
        "condition_id",
        "video_id",
        "question_stratum",
        "answer_token_count",
        "meteor",
        "bertscore_f1",
        "sts",
        "semantic_composite",
        "ndcg",
        "ndcg_degenerate",
    }


def test_score_panel_writes_json_atomically_only_after_complete_scoring(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    meteor, bertscore, sts = injected_metrics({"bert": 0})
    output = tmp_path / "artifact.json"

    scorer.score_panel(
        annotation_root=annotations,
        condition_paths=conditions,
        bert_model=model,
        expected_rows=2,
        expected_annotation_files=2,
        output_path=output,
        meteor_fn=meteor,
        bertscore_fn=bertscore,
        sts_fn=sts,
    )

    assert json.loads(output.read_text(encoding="utf-8"))["panel"][
        "completed_per_condition"
    ] == {"frames": 2, "whisper": 2}
    assert not output.with_name(output.name + ".tmp").exists()


def test_score_panel_rejects_duplicate_prediction_ids(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    conditions["frames"].write_text(
        json.dumps(
            [
                {"status": "passed", "answer": "cat", "video_id": "v1"},
                {"status": "passed", "answer": "wrong", "video_id": "v1"},
            ]
        ),
        encoding="utf-8",
    )
    meteor, bertscore, sts = injected_metrics({"bert": 0})

    with pytest.raises(ValueError, match="Duplicate prediction ID"):
        scorer.score_panel(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            expected_rows=2,
            expected_annotation_files=2,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_score_panel_rejects_missing_references(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    conditions["frames"].write_text(
        json.dumps(
            [
                {"status": "passed", "answer": "cat", "video_id": "v1"},
                {"status": "passed", "answer": "bird", "video_id": "missing"},
            ]
        ),
        encoding="utf-8",
    )
    meteor, bertscore, sts = injected_metrics({"bert": 0})

    with pytest.raises(ValueError, match="Missing reference"):
        scorer.score_panel(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            expected_rows=2,
            expected_annotation_files=2,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_score_panel_rejects_nonfinite_sts(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    meteor, bertscore, _ = injected_metrics({"bert": 0})

    with pytest.raises(ValueError, match="sts.*finite"):
        scorer.score_panel(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            expected_rows=2,
            expected_annotation_files=2,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=lambda _prediction, _reference: float("nan"),
        )


def test_score_panel_refuses_preexisting_output(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    output = tmp_path / "artifact.json"
    output.write_text("prior", encoding="utf-8")
    meteor, bertscore, sts = injected_metrics({"bert": 0})

    with pytest.raises(FileExistsError, match="already exists"):
        scorer.score_panel(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            expected_rows=2,
            expected_annotation_files=2,
            output_path=output,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_umbc_sts_uses_encoded_parameters_timeout_and_bounded_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = load_scorer()
    calls: list[tuple[str, float]] = []

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        if len(calls) == 1:
            raise URLError("temporary")
        return Response(b"0.75")

    monkeypatch.setattr(scorer.urllib.request, "urlopen", fake_urlopen)

    assert scorer.umbc_sts_score("a & b", "hello world", timeout=3, retries=2) == 0.75
    assert len(calls) == 2
    assert "phrase1=hello+world" in calls[0][0]
    assert "phrase2=a+%26+b" in calls[0][0]
    assert all(timeout == 3 for _, timeout in calls)


def test_score_panel_requires_declared_annotation_file_count(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    meteor, bertscore, sts = injected_metrics({"bert": 0})

    with pytest.raises(ValueError, match="annotation files.*expected 3"):
        scorer.score_panel(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            expected_rows=2,
            expected_annotation_files=3,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_score_panel_rejects_prediction_id_outside_declared_annotation_set(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    conditions["frames"].write_text(
        json.dumps(
            [
                {"status": "passed", "answer": "cat", "video_id": "v1"},
                {"status": "passed", "answer": "bird", "video_id": "missing"},
            ]
        ),
        encoding="utf-8",
    )
    meteor, bertscore, sts = injected_metrics({"bert": 0})

    with pytest.raises(ValueError, match="not present in declared annotation set"):
        scorer.score_panel(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            expected_rows=2,
            expected_annotation_files=2,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_score_panel_requires_the_pinned_bert_snapshot_directory(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    wrong_model = tmp_path / "roberta-large"
    wrong_model.mkdir()
    meteor, bertscore, sts = injected_metrics({"bert": 0})

    with pytest.raises(ValueError, match=EXPECTED_BERT_MODEL_SNAPSHOT):
        scorer.score_panel(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=wrong_model,
            expected_rows=2,
            expected_annotation_files=2,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_bert_score_records_module_and_distribution_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = load_scorer()
    monkeypatch.setitem(
        sys.modules,
        "bert_score",
        SimpleNamespace(__version__="0.3.12"),
    )
    original_version = scorer.importlib.metadata.version
    monkeypatch.setattr(
        scorer.importlib.metadata,
        "version",
        lambda name: "0.3.13" if name == "bert-score" else original_version(name),
    )

    assert scorer._bert_score_versions(required=True) == {
        "module": "0.3.12",
        "distribution": "0.3.13",
    }


def test_injected_bertscore_bypasses_installed_module_version_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = load_scorer()
    monkeypatch.setitem(
        sys.modules,
        "bert_score",
        SimpleNamespace(__version__="unexpected-injected-environment"),
    )

    assert scorer._bert_score_versions(required=False) == {
        "module": "injected",
        "distribution": "injected",
    }


def test_canonical_panel_rejects_count_preserving_id_mutation() -> None:
    scorer = load_scorer()
    canonical_ids = {f"v{index:03d}" for index in range(475)}
    excluded_ids = {f"excluded-{index}" for index in range(6)}
    profile = scorer.PanelProfile(
        name="vqa-2025-training-475-test",
        expected_annotation_files=481,
        expected_rows=475,
        canonical_id_set_sha256=scorer.sorted_video_id_set_sha256(canonical_ids),
        excluded_annotation_ids=tuple(sorted(excluded_ids)),
        excluded_reason="test fixture",
    )
    mutated_ids = (canonical_ids - {"v000"}) | {"replacement"}

    with pytest.raises(ValueError, match="canonical panel ID set"):
        scorer._validate_panel_identity(
            profile,
            annotation_ids=canonical_ids | excluded_ids,
            condition_ids={"frames": canonical_ids, "whisper": mutated_ids},
        )


def test_artifact_records_panel_identity_and_excluded_id_metadata(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    meteor, bertscore, sts = injected_metrics({"bert": 0})
    profile = fixture_panel_profile(scorer)

    artifact = scorer.score_panel(
        annotation_root=annotations,
        condition_paths=conditions,
        bert_model=model,
        expected_rows=2,
        expected_annotation_files=2,
        panel_profile=profile,
        meteor_fn=meteor,
        bertscore_fn=bertscore,
        sts_fn=sts,
    )

    assert artifact["panel"]["identity"] == {
        "profile": "fixture-v1-v2",
        "canonical_id_set_sha256": profile.canonical_id_set_sha256,
        "id_set_digest_algorithm": (
            "sha256(UTF-8 sorted video IDs joined by LF plus terminal LF)"
        ),
        "usable_video_id_count": 2,
        "excluded_annotation_ids": [],
        "excluded_annotation_count": 0,
        "excluded_reason": "test fixture",
    }
    assert artifact["scorer"]["bert_score_version_note"] == (
        "module __version__ 0.3.12 is authoritative; distribution metadata "
        "0.3.13 reflects the upstream metadata mismatch"
    )


def test_score_panel_requires_the_declared_bootstrap_seed(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    meteor, bertscore, sts = injected_metrics({"bert": 0})

    with pytest.raises(ValueError, match="20260809"):
        scorer.score_panel(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            expected_rows=2,
            expected_annotation_files=2,
            bootstrap_seed=7,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_atomic_publish_does_not_overwrite_destination_created_after_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    output = tmp_path / "artifact.json"
    real_link = scorer.os.link

    def race(source: Path, destination: Path) -> None:
        Path(destination).write_text("racer", encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(scorer.os, "link", race)

    with pytest.raises(FileExistsError):
        scorer._atomic_write_json(output, {"safe": True})
    assert output.read_text(encoding="utf-8") == "racer"
    assert not output.with_name(output.name + ".tmp").exists()


def test_score_panel_records_single_answer_ndcg_as_degenerate_and_excludes_it(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    meteor, bertscore, sts = injected_metrics({"bert": 0})

    artifact = scorer.score_panel(
        annotation_root=annotations,
        condition_paths=conditions,
        bert_model=model,
        expected_rows=2,
        expected_annotation_files=2,
        meteor_fn=meteor,
        bertscore_fn=bertscore,
        sts_fn=sts,
    )

    assert artifact["scorer"]["single_response_ndcg"] == {
        "value": 1.0,
        "degenerate": True,
        "informative": False,
    }
    assert artifact["rows"]["frames"][0]["ndcg_degenerate"] is True
    assert "ndcg" not in artifact["paired"]["metric_deltas"]


def test_score_panel_fails_closed_for_more_than_two_task1_conditions(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    annotations, conditions, model = make_inputs(tmp_path)
    conditions["third"] = conditions["frames"]
    meteor, bertscore, sts = injected_metrics({"bert": 0})

    with pytest.raises(ValueError, match="exactly two"):
        scorer.score_panel(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            expected_rows=2,
            expected_annotation_files=2,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )
