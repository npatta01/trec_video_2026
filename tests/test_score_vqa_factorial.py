from __future__ import annotations

from collections import OrderedDict
import copy
import importlib.util
import json
from pathlib import Path
from urllib.error import URLError

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/score_vqa_factorial.py"
SEMANTIC_SCRIPT = ROOT / "scripts/score_vqa_semantic_panel.py"
EXPECTED_BERT_MODEL_SNAPSHOT = (
    "722cf37b1afa9454edce342e7895e588b6ff1d59"
)
CONDITION_IDS = (
    "4b-current",
    "4b-question-aware",
    "8b-current",
    "8b-question-aware",
)


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_vqa_factorial", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_profile(scorer):
    return {
        "run_name": "fixture-factorial",
        "input_uri": "hf://buckets/test/input",
        "remote_uri": "hf://buckets/test/output",
        "manifest": "/dstack/run/fixture.json",
        "expected_count": 4,
        "frame_count": 12,
        "conditions": [
            {
                "condition_id": condition_id,
                "model_id": (
                    "Qwen/Qwen3.5-4B"
                    if condition_id.startswith("4b")
                    else "Qwen/Qwen3-VL-8B-Instruct"
                ),
                "revision": (
                    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
                    if condition_id.startswith("4b")
                    else "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
                ),
                "answer_policy": (
                    "question-aware" if "question-aware" in condition_id else "current"
                ),
                "min_pixels": 65536,
                "max_pixels": 262144,
                "request_source": "shared-accepted-whisper",
            }
            for condition_id in CONDITION_IDS
        ],
    }


def make_inputs(tmp_path: Path, scorer):
    ids = ("v1", "v2", "v3", "v4")
    annotations = tmp_path / "annotations"
    annotations.mkdir(parents=True)
    for index, video_id in enumerate(ids):
        (annotations / f"{video_id}.json").write_text(
            json.dumps(
                {
                    "question": (
                        "How many objects are visible?"
                        if index == 0
                        else "What is shown?"
                    ),
                    "correct_answer": f"reference {index}",
                    "incorrect_answers": [],
                }
            ),
            encoding="utf-8",
        )
    conditions = OrderedDict()
    for condition_id in CONDITION_IDS:
        path = tmp_path / f"{condition_id}.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "status": "passed",
                        "answer": f"answer {index}",
                        "video_id": video_id,
                        "condition_id": condition_id,
                    }
                    for index, video_id in enumerate(ids)
                ]
            ),
            encoding="utf-8",
        )
        conditions[condition_id] = path
    model = tmp_path / EXPECTED_BERT_MODEL_SNAPSHOT
    model.mkdir()
    (model / "config.json").write_text("frozen model fixture", encoding="utf-8")
    profile = make_profile(scorer)
    panel = scorer.PanelProfile(
        name="fixture-factorial-panel",
        expected_annotation_files=4,
        expected_rows=4,
        canonical_id_set_sha256=scorer.sorted_video_id_set_sha256(ids),
        excluded_annotation_ids=(),
        excluded_reason="fixture has no excluded IDs",
    )
    return annotations, conditions, model, profile, panel


def injected_metrics(calls: dict[str, int] | None = None):
    calls = calls if calls is not None else {}

    def meteor(prediction: str, reference: str) -> float:
        calls["meteor"] = calls.get("meteor", 0) + 1
        return 0.2

    def bertscore(
        predictions: list[str], references: list[str], _model: Path
    ) -> list[float]:
        calls["bert"] = calls.get("bert", 0) + 1
        calls["bert_pairs"] = len(predictions)
        assert len(predictions) == len(references)
        return [0.8] * len(predictions)

    def sts(prediction: str, reference: str) -> float:
        calls["sts"] = calls.get("sts", 0) + 1
        return 0.5

    return meteor, bertscore, sts


def score_fixture(tmp_path: Path, scorer, **kwargs):
    annotations, conditions, model, profile, panel = make_inputs(tmp_path, scorer)
    meteor, bertscore, sts = injected_metrics(kwargs.pop("calls", None))
    return scorer.score_factorial(
        annotation_root=annotations,
        condition_paths=conditions,
        bert_model=model,
        profile=profile,
        panel_profile=panel,
        expected_rows=4,
        expected_annotation_files=4,
        meteor_fn=kwargs.pop("meteor_fn", meteor),
        bertscore_fn=kwargs.pop("bertscore_fn", bertscore),
        sts_fn=kwargs.pop("sts_fn", sts),
        bootstrap_iterations=2000,
        **kwargs,
    )


def test_requires_exact_four_declared_condition_ids(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model, profile, panel = make_inputs(tmp_path, scorer)
    conditions.pop("8b-question-aware")
    meteor, bertscore, sts = injected_metrics()

    with pytest.raises(ValueError, match="exactly four|condition IDs"):
        scorer.score_factorial(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            profile=profile,
            panel_profile=panel,
            expected_rows=4,
            expected_annotation_files=4,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


@pytest.mark.parametrize("mutation", ["duplicate", "extra", "count-preserving"])
def test_prediction_id_mutations_fail_before_labels(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    annotations, conditions, model, profile, panel = make_inputs(tmp_path, scorer)
    rows = json.loads(conditions["8b-current"].read_text(encoding="utf-8"))
    if mutation == "duplicate":
        rows[-1]["video_id"] = rows[0]["video_id"]
    elif mutation == "extra":
        rows[-1]["video_id"] = "extra"
    else:
        rows[-1]["video_id"] = "replacement"
    conditions["8b-current"].write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(
        scorer,
        "_load_annotations",
        lambda _path: (_ for _ in ()).throw(AssertionError("labels read too early")),
    )
    meteor, bertscore, sts = injected_metrics()

    with pytest.raises(ValueError, match="Duplicate|same.*ID|canonical.*order"):
        scorer.score_factorial(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            profile=profile,
            panel_profile=panel,
            expected_rows=4,
            expected_annotation_files=4,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_rejects_wrong_condition_tuple_and_label_leakage(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model, profile, panel = make_inputs(tmp_path, scorer)
    rows = json.loads(conditions["4b-current"].read_text(encoding="utf-8"))
    rows[0]["answer_policy"] = "question-aware"
    rows[1]["reference"] = "leaked label"
    conditions["4b-current"].write_text(json.dumps(rows), encoding="utf-8")
    meteor, bertscore, sts = injected_metrics()

    with pytest.raises(ValueError, match="label-bearing|binding"):
        scorer.score_factorial(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            profile=profile,
            panel_profile=panel,
            expected_rows=4,
            expected_annotation_files=4,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_incomplete_prediction_fails_closed(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model, profile, panel = make_inputs(tmp_path, scorer)
    rows = json.loads(conditions["8b-current"].read_text(encoding="utf-8"))
    rows[2]["status"] = "failed"
    conditions["8b-current"].write_text(json.dumps(rows), encoding="utf-8")
    meteor, bertscore, sts = injected_metrics()

    with pytest.raises(ValueError, match="incomplete"):
        scorer.score_factorial(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            profile=profile,
            panel_profile=panel,
            expected_rows=4,
            expected_annotation_files=4,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_scores_unique_pairs_once_in_one_deterministic_bert_batch(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    calls: dict[str, int] = {}
    artifact = score_fixture(tmp_path, scorer, calls=calls)

    assert calls == {"meteor": 4, "bert": 1, "bert_pairs": 4, "sts": 4}
    assert artifact["panel"]["completed_per_condition"] == {
        condition_id: 4 for condition_id in CONDITION_IDS
    }
    assert artifact["conditions"]["4b-current"]["semantic_composite"] == pytest.approx(
        0.5
    )
    assert artifact["scorer"]["single_response_ndcg"] == {
        "value": 1.0,
        "degenerate": True,
        "informative": False,
    }


def test_bert_length_and_nonfinite_values_fail_closed(tmp_path: Path) -> None:
    scorer = load_scorer()

    def bad_length(predictions, references, model):
        return [0.8]

    with pytest.raises(ValueError, match="BERTScore returned"):
        score_fixture(tmp_path / "length", scorer, bertscore_fn=bad_length)

    def bad_nonfinite(predictions, references, model):
        return [float("nan")] * len(predictions)

    with pytest.raises(ValueError, match="finite"):
        score_fixture(tmp_path / "nonfinite", scorer, bertscore_fn=bad_nonfinite)


def test_sts_cache_hits_and_identity_are_recorded(tmp_path: Path) -> None:
    scorer = load_scorer()
    cache = scorer.STSCache()
    calls: dict[str, int] = {}
    first = score_fixture(tmp_path / "first", scorer, calls=calls, sts_cache=cache)
    assert first["cache_summary"]["miss_count"] == 4
    assert first["cache_summary"]["live_request_count"] == 4
    calls.clear()
    second = score_fixture(tmp_path / "second", scorer, calls=calls, sts_cache=cache)
    assert second["cache_summary"]["hit_count"] == 4
    assert second["cache_summary"]["live_request_count"] == 0
    assert calls.get("sts", 0) == 0
    assert second["cache_summary"]["identity"] == first["cache_summary"]["identity"]

    wrong = scorer.STSCache(identity="mutated-cache-identity")
    with pytest.raises(ValueError, match="cache identity"):
        score_fixture(tmp_path / "wrong", scorer, sts_cache=wrong)


def test_sts_retry_budget_and_failure_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = load_scorer()
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"0.75"

    def urlopen(_request, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise URLError("temporary")
        return Response()

    monkeypatch.setattr(scorer.urllib.request, "urlopen", urlopen)
    assert scorer.umbc_sts_score("a & b", "hello", timeout=10, retries=3) == 0.75
    assert calls == [10, 10, 10]

    def always_fails(_request, timeout):
        raise URLError("permanent")

    monkeypatch.setattr(scorer.urllib.request, "urlopen", always_fails)
    with pytest.raises(ValueError, match="3 attempts"):
        scorer.umbc_sts_score("a", "b", timeout=10, retries=3)


def test_baseline_reuse_requires_full_binding_and_mutation_rescores(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    annotations, conditions, model, profile, panel = make_inputs(tmp_path, scorer)
    calls: dict[str, int] = {}
    first = score_fixture(tmp_path / "initial", scorer, calls=calls)
    baseline_scorer = copy.deepcopy(first["scorer"])
    baseline_scorer["bert_score_module_version"] = "0.3.12"
    baseline = {
        "schema_version": 1,
        "scorer": baseline_scorer,
        "panel": first["panel"],
        "rows": {"legacy-4b": first["rows"]["4b-current"]},
        "inputs": {
            "annotation_root_sha256": scorer.sha256_path(annotations),
            "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
            "bert_model_snapshot_sha256": scorer.sha256_path(model),
            "scorer_sha256": scorer.sha256_path(SEMANTIC_SCRIPT),
            "profile_sha256": scorer.profile_sha256(profile),
            "normalized_answer_map_sha256": scorer.normalized_answer_map_sha256(
                conditions["4b-current"]
            ),
        },
    }
    baseline["inputs"]["artifact_content_sha256"] = scorer.baseline_content_sha256(
        baseline
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    calls.clear()
    reused = scorer.score_factorial(
        annotation_root=annotations,
        condition_paths=conditions,
        bert_model=model,
        profile=profile,
        panel_profile=panel,
        expected_rows=4,
        expected_annotation_files=4,
        baseline_artifact_path=baseline_path,
        meteor_fn=injected_metrics(calls)[0],
        bertscore_fn=injected_metrics(calls)[1],
        sts_fn=injected_metrics(calls)[2],
    )
    assert reused["baseline_binding"]["reused"] is True
    assert reused["baseline_binding"]["reason"] == "all bindings matched"
    assert "4b-current" in reused["rows"]
    assert calls["bert"] == 1

    mutated = copy.deepcopy(baseline)
    mutated["inputs"]["profile_sha256"] = "0" * 64
    mutated["inputs"]["artifact_content_sha256"] = scorer.baseline_content_sha256(
        mutated
    )
    mutated_path = tmp_path / "mutated-baseline.json"
    mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
    calls.clear()
    rescored = scorer.score_factorial(
        annotation_root=annotations,
        condition_paths=conditions,
        bert_model=model,
        profile=profile,
        panel_profile=panel,
        expected_rows=4,
        expected_annotation_files=4,
        baseline_artifact_path=mutated_path,
        meteor_fn=injected_metrics(calls)[0],
        bertscore_fn=injected_metrics(calls)[1],
        sts_fn=injected_metrics(calls)[2],
    )
    assert rescored["baseline_binding"]["reused"] is False
    assert "config" in rescored["baseline_binding"]["reason"]
    assert calls["bert"] == 1


def test_seed_mutation_is_rejected(tmp_path: Path) -> None:
    scorer = load_scorer()
    with pytest.raises(ValueError, match="20260809"):
        score_fixture(tmp_path, scorer, bootstrap_seed=7)


def test_factorial_effects_compute_known_main_effects_and_interaction() -> None:
    scorer = load_scorer()
    rows = {
        condition_id: [
            {
                "condition_id": condition_id,
                "video_id": "v1",
                "stratum": "other",
                "meteor": values[0],
                "bertscore_f1": values[0],
                "sts": values[0],
                "semantic_composite": values[0],
            }
        ]
        for condition_id, values in {
            "4b-current": (0.10,),
            "4b-question-aware": (0.20,),
            "8b-current": (0.30,),
            "8b-question-aware": (0.60,),
        }.items()
    }
    effects = scorer.compute_factorial_effects(
        rows, bootstrap_iterations=2000, bootstrap_seed=20260809
    )
    assert effects["model"]["overall"]["semantic_composite"] == pytest.approx(0.3)
    assert effects["prompt"]["overall"]["semantic_composite"] == pytest.approx(0.2)
    assert effects["interaction"]["overall"]["semantic_composite"] == pytest.approx(0.2)
    assert effects["model"]["strata"]["other"]["semantic_composite"] == pytest.approx(
        0.3
    )


def test_promotion_boundaries_are_exact() -> None:
    scorer = load_scorer()
    base = {
        "delta_C": 0.015,
        "lower_CI_C": 0.000001,
        "delta_METEOR": 0.000001,
        "delta_BERTScore": 0.000001,
        "delta_STS": -0.005,
        "completed": 475,
        "integrity": True,
    }
    assert scorer.promotion_decision(base)["decision"] == "promote"
    for field, value in (
        ("delta_C", 0.014999),
        ("lower_CI_C", 0.0),
        ("delta_METEOR", 0.0),
        ("delta_BERTScore", 0.0),
        ("delta_STS", -0.005001),
        ("completed", 474),
        ("integrity", False),
    ):
        candidate = dict(base)
        candidate[field] = value
        assert scorer.promotion_decision(candidate)["decision"] == "stop"


def test_recursive_sanitization_and_deterministic_condition_order(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, conditions, model, profile, panel = make_inputs(tmp_path, scorer)
    calls_a: dict[str, int] = {}
    artifact_a = scorer.score_factorial(
        annotation_root=annotations,
        condition_paths=conditions,
        bert_model=model,
        profile=profile,
        panel_profile=panel,
        expected_rows=4,
        expected_annotation_files=4,
        meteor_fn=injected_metrics(calls_a)[0],
        bertscore_fn=injected_metrics(calls_a)[1],
        sts_fn=injected_metrics(calls_a)[2],
    )
    reordered = OrderedDict(reversed(list(conditions.items())))
    artifact_b = scorer.score_factorial(
        annotation_root=annotations,
        condition_paths=reordered,
        bert_model=model,
        profile=profile,
        panel_profile=panel,
        expected_rows=4,
        expected_annotation_files=4,
        meteor_fn=injected_metrics({})[0],
        bertscore_fn=injected_metrics({})[1],
        sts_fn=injected_metrics({})[2],
    )
    assert artifact_a["panel"]["declared_conditions"] == list(CONDITION_IDS)
    assert json.dumps(artifact_a, sort_keys=True) == json.dumps(artifact_b, sort_keys=True)

    forbidden = {"answer", "reference", "question", "transcript", "cache_path"}

    def assert_safe(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                assert_safe(child)
        elif isinstance(value, list):
            for child in value:
                assert_safe(child)
        elif isinstance(value, str):
            assert not value.startswith("/")
            assert "answer 0" not in value

    assert_safe(artifact_a)
    assert set(artifact_a["rows"]["4b-current"][0]) == {
        "condition_id",
        "video_id",
        "stratum",
        "normalized_exact_match",
        "token_f1",
        "answer_token_count",
        "meteor",
        "bertscore_f1",
        "sts",
        "semantic_composite",
        "ndcg",
        "ndcg_degenerate",
    }


def test_atomic_race_and_preexisting_destination_fail_closed(
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
        scorer.atomic_write_json(output, {"safe": True})
    assert output.read_text(encoding="utf-8") == "racer"
    assert not output.with_name(output.name + ".tmp").exists()

    output.write_text("prior", encoding="utf-8")
    with pytest.raises(FileExistsError):
        score_fixture(tmp_path / "preexisting", scorer, output_path=output)


def test_old_two_condition_scorer_still_rejects_more_than_two(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "score_vqa_semantic_panel", SEMANTIC_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    annotations, conditions, model, _profile, _panel = make_inputs(tmp_path, load_scorer())
    meteor, bertscore, sts = injected_metrics()
    with pytest.raises(ValueError, match="exactly two"):
        old.score_panel(
            annotation_root=annotations,
            condition_paths=conditions,
            bert_model=model,
            expected_rows=4,
            expected_annotation_files=4,
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
        )


def test_profile_mutation_changes_binding_and_is_rejected() -> None:
    scorer = load_scorer()
    profile_path = ROOT / "configs/vqa/vqa_factorial_full_condition_profiles.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["conditions"][0]["model_id"] = "Qwen/Qwen3-VL-8B-Instruct"
    with pytest.raises(ValueError, match="binding|model"):
        scorer.validate_factorial_profile(payload, strict=True)
