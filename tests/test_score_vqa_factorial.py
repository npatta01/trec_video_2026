from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
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
    assert reused["baseline_binding"]["reused"] is False
    assert "binding mismatch" in reused["baseline_binding"]["reason"]
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
    assert "binding mismatch" in rescored["baseline_binding"]["reason"]
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


def _write_production_bundle(
    root: Path,
    scorer,
    condition_id: str,
    ids: tuple[str, ...] = ("v1", "v2", "v3", "v4"),
    *,
    evidence_id: str = "evidence-1",
    mutate: dict[str, object] | None = None,
) -> Path:
    """Create the downloaded producer layout without reshaping it for scoring."""

    root.mkdir(parents=True, exist_ok=True)
    payload_root = root / "payload"
    condition_root = payload_root / "conditions" / condition_id
    condition_root.mkdir(parents=True, exist_ok=True)
    profile_condition = make_profile(scorer)["conditions"][
        CONDITION_IDS.index(condition_id)
    ]
    run_id = "fixture-factorial-" + condition_id
    runtime_image = scorer.EXPECTED_RUNTIME_CONTROLS["runtime_image"]
    model_id = profile_condition["model_id"]
    revision = profile_condition["revision"]
    policy = profile_condition["answer_policy"]
    prediction_rows = []
    for index, video_id in enumerate(ids):
        prediction_rows.append(
            {
                "status": "passed",
                "answer": f"answer {index}",
                "elapsed_seconds": 0.25,
                "model_load_seconds": 1.5,
                "run_id": run_id,
                "condition_id": condition_id,
                "video_id": video_id,
                "model_id": model_id,
                "model_revision": revision,
                "runtime": "transformers 5.15.0.dev0; torch 2.8.0",
                "answer_policy": policy,
                "question_type": "general",
                "requested_min_pixels_per_frame": 65536,
                "requested_max_pixels_per_frame": 262144,
                "min_pixels_per_frame": 65536,
                "max_pixels_per_frame": 262144,
                "runtime_image": runtime_image,
                "precision": "bfloat16",
                "device": "cuda:0 (NVIDIA CUDA)",
                "runtime_attestation": {
                    "accelerator_backend": "nvidia-cuda",
                    "parameter_tensor_count": 10,
                    "parameter_devices": {"cuda:0": 10},
                    "parameter_dtypes": {"torch.bfloat16": 10},
                    "buffer_tensor_count": 2,
                    "buffer_devices": {"cuda:0": 2},
                    "buffer_dtypes": {"torch.bfloat16": 2},
                    "torch_cuda": "12.4",
                    "torch_hip": None,
                },
                "peak_memory_bytes": 1024 + index,
                "error_type": None,
                "error_message": None,
            }
        )
    identity = {
        "schema_version": 1,
        "condition_id": condition_id,
        "run_id": run_id,
        "model_id": model_id,
        "model_revision": revision,
        "answer_policy": policy,
        "requested_min_pixels_per_frame": 65536,
        "requested_max_pixels_per_frame": 262144,
        "applied_min_pixels_per_frame": 65536,
        "applied_max_pixels_per_frame": 262144,
        "runtime_image": runtime_image,
        "transformers_revision": scorer.EXPECTED_RUNTIME_CONTROLS[
            "transformers_revision"
        ],
        "frame_count": 12,
        "max_new_tokens": 64,
        "do_sample": False,
        "thinking": False,
        "request_source": "shared-accepted-whisper",
        "model_class": "multimodal",
        "deterministic_decoding": {
            "do_sample": False,
            "enable_thinking": False,
            "max_new_tokens": 64,
        },
    }
    evidence_marker = {
        "schema_version": 1,
        "label_policy": "label-free-evidence",
        "evidence_bundle_id": evidence_id,
        "example_count": len(ids),
        "canonical_id_digest": scorer.sorted_video_id_set_sha256(ids),
        "manifest_sha256": "m" * 64,
        "checksum_file": "SHA256SUMS",
    }
    runtime = {
        "runtime": "transformers 5.15.0.dev0; torch 2.8.0",
        "runtime_image": runtime_image,
        "transformers_version": "5.15.0.dev0",
        "transformers_revision": scorer.EXPECTED_RUNTIME_CONTROLS[
            "transformers_revision"
        ],
        "torch_version": "2.8.0",
        "torch_cuda": "12.4",
        "torch_hip": None,
        "cuda_device_name": "fixture NVIDIA",
        "cuda_bf16_supported": True,
        "dependencies": {
            "Pillow": "11.3.0",
            "soundfile": "0.13.0",
            "accelerate": "1.12.0",
            "av": "16.0.0",
            "imageio-ffmpeg": "0.6.0",
            "qwen-vl-utils": "0.0.14",
        },
    }
    timing = {
        "condition_id": condition_id,
        "row_count": len(ids),
        "wall_seconds": 8.0,
        "prepare_wall_seconds": 2.0,
        # Deliberately below nvidia-smi's independent measurement: the scorer
        # must preserve both valid measurements without inventing an ordering.
        "peak_memory_bytes": 64 * 1024 * 1024,
        "model_load_seconds": 1.5,
        "runtime_identity": [
            [
                "transformers 5.15.0.dev0; torch 2.8.0",
                runtime_image,
                "cuda:0 (NVIDIA CUDA)",
                "bfloat16",
            ]
        ],
    }
    telemetry = (
        "timestamp,index,utilization.gpu [%],memory.used [MiB],memory.total [MiB]\n"
        "2026-08-09T00:00:00Z,0,50,100,1000\n"
        "2026-08-09T00:00:05Z,0,75,200,1000\n"
    )
    (condition_root / "predictions.json").write_text(
        json.dumps(prediction_rows), encoding="utf-8"
    )
    (condition_root / "condition-identity.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    evidence_root = payload_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "evidence-marker.json").write_text(
        json.dumps(evidence_marker), encoding="utf-8"
    )
    marker_hash = hashlib.sha256(
        (evidence_root / "evidence-marker.json").read_bytes()
    ).hexdigest()
    (evidence_root / "evidence-bytes.bin").write_bytes(b"verified shared evidence")
    (evidence_root / "SHA256SUMS").write_text(
        f"{marker_hash}  evidence-marker.json\n"
        f"{hashlib.sha256((evidence_root / 'evidence-bytes.bin').read_bytes()).hexdigest()}  evidence-bytes.bin\n",
        encoding="utf-8",
    )
    (payload_root / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    (condition_root / "timing.json").write_text(json.dumps(timing), encoding="utf-8")
    (condition_root / "telemetry").mkdir()
    (condition_root / "telemetry" / "nvidia-smi.csv").write_text(telemetry, encoding="utf-8")
    (payload_root / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "mode": "condition",
                "run_name": run_id,
                "expected_count": len(ids),
                "frame_count": 12,
                "workload_returncode": 0,
                "remote_payload_sync": "frozen",
                "labels_joined": False,
            }
        ),
        encoding="utf-8",
    )
    if mutate:
        for relative, value in mutate.items():
            path = root / relative
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                payload.update(value)
            else:
                raise AssertionError("bundle mutation must target JSON objects")
            path.write_text(json.dumps(payload), encoding="utf-8")
    checksum_lines = []
    for path in sorted(
        item
        for item in payload_root.rglob("*")
        if item.is_file()
        and (item.name != "SHA256SUMS" or item == evidence_root / "SHA256SUMS")
    ):
        checksum_lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(payload_root).as_posix()}"
        )
    checksum = "\n".join(checksum_lines) + "\n"
    (payload_root / "SHA256SUMS").write_text(checksum, encoding="utf-8")
    terminal = {
        "schema_version": 1,
        "status": "passed",
        "payload_sha256": hashlib.sha256(checksum.encode()).hexdigest(),
        "payload_uri": f"hf://buckets/test/output/{condition_id}/payload",
        "run_name": run_id,
    }
    (root / "terminal-status.json").write_text(json.dumps(terminal), encoding="utf-8")
    return root


def test_production_bundle_contract_and_operational_evidence_are_required(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    bundles = {
        condition_id: _write_production_bundle(
            tmp_path / condition_id, scorer, condition_id
        )
        for condition_id in CONDITION_IDS
    }
    profile = make_profile(scorer)
    model = tmp_path / EXPECTED_BERT_MODEL_SNAPSHOT
    model.mkdir()
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for index, video_id in enumerate(("v1", "v2", "v3", "v4")):
        (annotations / f"{video_id}.json").write_text(
            json.dumps(
                {
                    "question": "What is shown?",
                    "correct_answer": f"reference {index}",
                    "incorrect_answers": [],
                }
            ),
            encoding="utf-8",
        )
    panel = scorer.PanelProfile(
        "fixture-factorial-panel",
        4,
        4,
        scorer.sorted_video_id_set_sha256(("v1", "v2", "v3", "v4")),
        (),
        "fixture",
    )
    artifact = scorer.score_factorial(
        annotation_root=annotations,
        condition_bundles=bundles,
        bert_model=model,
        profile=profile,
        panel_profile=panel,
        expected_rows=4,
        expected_annotation_files=4,
        meteor_fn=injected_metrics()[0],
        bertscore_fn=injected_metrics()[1],
        sts_fn=injected_metrics()[2],
    )
    operational = artifact["conditions"]["4b-current"]["operational_evidence"]
    assert operational["wall_seconds"] == 8.0
    assert operational["utilization_sample_count"] == 2
    assert operational["torch_peak_memory_bytes"] == 64 * 1024 * 1024
    assert operational["nvidia_smi_peak_memory_bytes"] == 200 * 1024 * 1024
    projection = artifact["experiment"]["operational_projection"]
    assert projection["parallel_wall_seconds"] == 8.0
    assert projection["gpu_seconds"] == 32.0
    assert projection["gpu_hours"] == pytest.approx(32.0 / 3600.0)
    assert projection["parallel_rows_per_second"] == pytest.approx(2.0)


def test_downloaded_producer_layout_is_consumed_without_reshaping(tmp_path: Path) -> None:
    scorer = load_scorer()
    bundle = _write_production_bundle(tmp_path / "download", scorer, "4b-current")
    assert (bundle / "terminal-status.json").is_file()
    assert (bundle / "payload" / "SHA256SUMS").is_file()
    assert (bundle / "payload" / "conditions" / "4b-current" / "predictions.json").is_file()
    loaded = scorer._load_condition_bundle(
        bundle,
        condition_id="4b-current",
        profile_condition=make_profile(scorer)["conditions"][0],
        expected_rows=4,
        expected_id_digest=scorer.sorted_video_id_set_sha256(("v1", "v2", "v3", "v4")),
    )
    assert loaded["prediction_path"] == (
        bundle / "payload" / "conditions" / "4b-current" / "predictions.json"
    )


def test_cli_passes_downloaded_roots_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = load_scorer()
    bundles = {
        condition_id: _write_production_bundle(tmp_path / condition_id, scorer, condition_id)
        for condition_id in CONDITION_IDS
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(make_profile(scorer)), encoding="utf-8")
    model = tmp_path / EXPECTED_BERT_MODEL_SNAPSHOT
    model.mkdir()
    seen: dict[str, object] = {}

    def fake_score(**kwargs):
        seen.update(kwargs)
        return {"panel": {"completed_per_condition": {condition_id: 4 for condition_id in CONDITION_IDS}}}

    monkeypatch.setattr(scorer, "score_factorial", fake_score)
    assert scorer.main(
        [
            "--annotations", str(tmp_path / "annotations"),
            *sum((["--condition-bundle", f"{condition_id}={bundles[condition_id]}"] for condition_id in CONDITION_IDS), []),
            "--bert-model", str(model),
            "--profile", str(profile_path),
            "--output", str(tmp_path / "artifact.json"),
        ]
    ) == 0
    assert seen["condition_bundles"] == bundles


def _refresh_download_manifest(bundle: Path) -> None:
    payload_root = bundle / "payload"
    paths = [
        path
        for path in payload_root.rglob("*")
        if path.is_file()
        and (
            path != payload_root / "SHA256SUMS"
            and path != payload_root / "terminal-status.json"
        )
    ]
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(payload_root).as_posix()}"
        for path in sorted(paths)
    ]
    checksum = ("\n".join(lines) + "\n").encode("utf-8")
    (payload_root / "SHA256SUMS").write_bytes(checksum)
    terminal = json.loads((bundle / "terminal-status.json").read_text(encoding="utf-8"))
    terminal["payload_sha256"] = hashlib.sha256(checksum).hexdigest()
    (bundle / "terminal-status.json").write_text(json.dumps(terminal), encoding="utf-8")


def test_marker_only_condition_output_requires_and_binds_external_evidence(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    bundles = {
        condition_id: _write_production_bundle(tmp_path / condition_id, scorer, condition_id)
        for condition_id in CONDITION_IDS
    }
    for bundle in bundles.values():
        (bundle / "payload" / "evidence" / "SHA256SUMS").unlink()
        _refresh_download_manifest(bundle)
    evidence = _write_production_bundle(tmp_path / "evidence", scorer, "4b-current")
    shutil.rmtree(evidence / "payload" / "conditions" / "4b-current")
    status_path = evidence / "payload" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["mode"] = "evidence"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    _refresh_download_manifest(evidence)
    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    annotations, _legacy, model, profile, panel = make_inputs(labels_root, scorer)
    artifact = scorer.score_factorial(
        annotation_root=annotations,
        condition_bundles=bundles,
        evidence_bundle_path=evidence,
        bert_model=model,
        profile=profile,
        panel_profile=panel,
        expected_rows=4,
        expected_annotation_files=4,
        meteor_fn=injected_metrics()[0],
        bertscore_fn=injected_metrics()[1],
        sts_fn=injected_metrics()[2],
    )
    assert artifact["integrity"]["evidence_identity_checked"] is True


def test_nested_evidence_manifest_drift_fails_after_root_manifest_is_rebound(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    bundle = _write_production_bundle(tmp_path / "download", scorer, "4b-current")
    nested = bundle / "payload" / "evidence" / "SHA256SUMS"
    nested.write_text(nested.read_text(encoding="utf-8") + ("0" * 64) + "  missing.bin\n", encoding="utf-8")
    payload_root = bundle / "payload"
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(payload_root).as_posix()}"
        for path in sorted(
            path
            for path in payload_root.rglob("*")
            if path.is_file()
            and (path.name != "SHA256SUMS" or path == payload_root / "evidence" / "SHA256SUMS")
        )
    ]
    checksum = ("\n".join(lines) + "\n").encode()
    (payload_root / "SHA256SUMS").write_bytes(checksum)
    terminal = json.loads((bundle / "terminal-status.json").read_text(encoding="utf-8"))
    terminal["payload_sha256"] = hashlib.sha256(checksum).hexdigest()
    (bundle / "terminal-status.json").write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence|coverage|manifest"):
        scorer._load_condition_bundle(
            bundle,
            condition_id="4b-current",
            profile_condition=make_profile(scorer)["conditions"][0],
            expected_rows=4,
            expected_id_digest=scorer.sorted_video_id_set_sha256(("v1", "v2", "v3", "v4")),
        )


def test_runtime_attestation_requires_exact_parameter_and_buffer_sums(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    bundle = _write_production_bundle(tmp_path / "download", scorer, "4b-current")
    prediction_path = bundle / "payload" / "conditions" / "4b-current" / "predictions.json"
    rows = json.loads(prediction_path.read_text(encoding="utf-8"))
    rows[0]["runtime_attestation"]["buffer_devices"] = {"cuda:0": 1}
    prediction_path.write_text(json.dumps(rows), encoding="utf-8")
    _refresh_download_manifest(bundle)
    with pytest.raises(ValueError, match="attestation|buffer|schema|checksum"):
        scorer._load_condition_bundle(
            bundle,
            condition_id="4b-current",
            profile_condition=make_profile(scorer)["conditions"][0],
            expected_rows=4,
            expected_id_digest=scorer.sorted_video_id_set_sha256(("v1", "v2", "v3", "v4")),
        )


def test_shared_wrong_canonical_ids_fail_before_annotation_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    ids = ("w1", "w2", "w3", "w4")
    bundles = {
        condition_id: _write_production_bundle(
            tmp_path / condition_id, scorer, condition_id, ids
        )
        for condition_id in CONDITION_IDS
    }
    monkeypatch.setattr(
        scorer,
        "_load_annotations",
        lambda _path: (_ for _ in ()).throw(AssertionError("labels read too early")),
    )
    model = tmp_path / EXPECTED_BERT_MODEL_SNAPSHOT
    model.mkdir()
    with pytest.raises(ValueError, match="canonical|digest|ID"):
        scorer.score_factorial(
            annotation_root=tmp_path / "annotations",
            condition_bundles=bundles,
            bert_model=model,
            profile=make_profile(scorer),
            panel_profile=scorer.PanelProfile(
                "fixture-factorial-panel", 4, 4, scorer.sorted_video_id_set_sha256(("v1", "v2", "v3", "v4")), (), "fixture"
            ),
            expected_rows=4,
            expected_annotation_files=4,
        )


def test_artifact_privacy_rejects_nested_variants_and_paths(tmp_path: Path) -> None:
    scorer = load_scorer()
    valid = score_fixture(tmp_path / "valid", scorer)
    for payload in (
        {"nested": {"Questions": "text"}},
        {"nested": {"REFERENCES": ["text"]}},
        {"nested": {"answer_text": "text"}},
        {"nested": {"cacheQuery": "text"}},
        {"nested": {"uri": "https://example.test"}},
        {"nested": {"relative": "./secret.json"}},
    ):
        candidate = copy.deepcopy(valid)
        # Keep the surrounding artifact valid so this specifically exercises
        # recursive traversal rather than a top-level shape check.
        candidate["experiment"]["semantic_composite"] = payload
        with pytest.raises(ValueError):
            scorer._artifact_safe(candidate)


def test_private_cache_write_is_atomic_on_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    cache_path = tmp_path / "private-cache.json"
    cache_path.write_text("prior", encoding="utf-8")
    original_replace = scorer.os.replace

    def interrupted(*_args):
        raise OSError("interrupted")

    monkeypatch.setattr(scorer.os, "replace", interrupted)
    with pytest.raises(OSError):
        scorer._write_private_cache(cache_path, scorer.STSCache({"k": 0.5}, "id"))
    assert cache_path.read_text(encoding="utf-8") == "prior"
    monkeypatch.setattr(scorer.os, "replace", original_replace)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("missing-field", "checksum|schema"),
        ("wrong-tuple", "identity tuple"),
        ("stale-bounds", "runtime control|identity tuple"),
    ],
)
def test_production_bundle_mutations_fail_closed(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    scorer = load_scorer()
    bundle = _write_production_bundle(tmp_path / "bundle", scorer, "4b-current")
    condition_root = bundle / "payload" / "conditions" / "4b-current"
    if mutation == "missing-field":
        payload = json.loads((condition_root / "predictions.json").read_text(encoding="utf-8"))
        payload[0].pop("runtime_attestation")
        (condition_root / "predictions.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "wrong-tuple":
        payload = json.loads((condition_root / "condition-identity.json").read_text(encoding="utf-8"))
        payload["model_revision"] = "0" * 40
        (condition_root / "condition-identity.json").write_text(json.dumps(payload), encoding="utf-8")
        # Rebuild only the root manifest so this mutation tests binding, not stale bytes.
        lines = []
        for path in sorted(
                item
                for item in (bundle / "payload").rglob("*")
                if item.is_file()
                and (item.name != "SHA256SUMS" or item == bundle / "payload" / "evidence" / "SHA256SUMS")
        ):
            lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(bundle / 'payload').as_posix()}"
            )
            (bundle / "payload" / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
            checksum = (bundle / "payload" / "SHA256SUMS").read_bytes()
            (bundle / "terminal-status.json").write_text(
                json.dumps(
                        {
                            "schema_version": 1,
                            "status": "passed",
                            "payload_sha256": hashlib.sha256(checksum).hexdigest(),
                            "payload_uri": "hf://buckets/test/output/4b-current/payload",
                            "run_name": "fixture-factorial-4b-current",
                    }
                ),
                encoding="utf-8",
            )
    else:
        payload = json.loads((condition_root / "condition-identity.json").read_text(encoding="utf-8"))
        payload["applied_max_pixels_per_frame"] = 1024 * 1024
        (condition_root / "condition-identity.json").write_text(json.dumps(payload), encoding="utf-8")
        lines = []
        for path in sorted(
                item
                for item in (bundle / "payload").rglob("*")
                if item.is_file()
                and (item.name != "SHA256SUMS" or item == bundle / "payload" / "evidence" / "SHA256SUMS")
        ):
            lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(bundle / 'payload').as_posix()}"
            )
            (bundle / "payload" / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
            checksum = (bundle / "payload" / "SHA256SUMS").read_bytes()
            (bundle / "terminal-status.json").write_text(
                json.dumps(
                        {
                            "schema_version": 1,
                            "status": "passed",
                            "payload_sha256": hashlib.sha256(checksum).hexdigest(),
                            "payload_uri": "hf://buckets/test/output/4b-current/payload",
                            "run_name": "fixture-factorial-4b-current",
                    }
                ),
                encoding="utf-8",
            )
    with pytest.raises(ValueError, match=expected):
        scorer._load_condition_bundle(
            bundle,
            condition_id="4b-current",
            profile_condition=make_profile(scorer)["conditions"][0],
            expected_rows=4,
            expected_id_digest=scorer.sorted_video_id_set_sha256(("v1", "v2", "v3", "v4")),
        )


def test_single_downloaded_bundle_without_the_other_condition_files_is_rejected(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    bundle = _write_production_bundle(tmp_path / "same", scorer, "4b-current")
    model = tmp_path / EXPECTED_BERT_MODEL_SNAPSHOT
    model.mkdir()
    with pytest.raises(ValueError, match="condition"):
        scorer.score_factorial(
            annotation_root=tmp_path / "annotations",
            condition_bundles={condition_id: bundle for condition_id in CONDITION_IDS},
            bert_model=model,
            profile=make_profile(scorer),
            panel_profile=scorer.PanelProfile(
                "fixture-factorial-panel", 4, 4,
                scorer.sorted_video_id_set_sha256(("v1", "v2", "v3", "v4")), (), "fixture"
            ),
            expected_rows=4,
            expected_annotation_files=4,
        )


def test_wrong_evidence_marker_identity_is_rejected_before_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    bundles = {
        condition_id: _write_production_bundle(tmp_path / condition_id, scorer, condition_id)
        for condition_id in CONDITION_IDS
    }
    marker = (
        bundles["8b-current"]
        / "payload"
        / "evidence"
        / "evidence-marker.json"
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["evidence_bundle_id"] = "wrong-evidence"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    # The stale root manifest is intentionally left unmodified; either verification
    # failure or the common-identity check must happen before labels are opened.
    monkeypatch.setattr(
        scorer,
        "_load_annotations",
        lambda _path: (_ for _ in ()).throw(AssertionError("labels read too early")),
    )
    model = tmp_path / EXPECTED_BERT_MODEL_SNAPSHOT
    model.mkdir()
    with pytest.raises(ValueError, match="checksum|evidence"):
        scorer.score_factorial(
            annotation_root=tmp_path / "annotations",
            condition_bundles=bundles,
            bert_model=model,
            profile=make_profile(scorer),
            panel_profile=scorer.PanelProfile(
                "fixture-factorial-panel", 4, 4,
                scorer.sorted_video_id_set_sha256(("v1", "v2", "v3", "v4")), (), "fixture"
            ),
            expected_rows=4,
            expected_annotation_files=4,
        )


def test_cli_refuses_existing_output_before_profile_or_metric_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    output = tmp_path / "artifact.json"
    output.write_text("prior", encoding="utf-8")
    monkeypatch.setattr(
        scorer,
        "load_profile",
        lambda _path: (_ for _ in ()).throw(AssertionError("profile loaded too early")),
    )
    with pytest.raises(FileExistsError, match="Output already exists"):
        scorer.main(
            [
                "--annotations", str(tmp_path / "annotations"),
                "--condition-bundle", "4b-current=" + str(tmp_path / "4b-current"),
                "--condition-bundle", "4b-question-aware=" + str(tmp_path / "4b-question-aware"),
                "--condition-bundle", "8b-current=" + str(tmp_path / "8b-current"),
                "--condition-bundle", "8b-question-aware=" + str(tmp_path / "8b-question-aware"),
                "--bert-model", str(tmp_path / EXPECTED_BERT_MODEL_SNAPSHOT),
                "--output", str(output),
            ]
        )


def test_pinned_baseline_reuse_requires_explicit_source_and_exact_row_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    annotations, conditions, model, profile, panel = make_inputs(tmp_path, scorer)
    first = score_fixture(tmp_path / "initial", scorer)
    baseline_scorer = copy.deepcopy(first["scorer"])
    baseline_scorer.update(
        {
            "bert_score_module_version": "0.3.12",
            "sts_endpoint": scorer.STS_ENDPOINT,
        }
    )
    baseline = {
        "schema_version": 1,
        "scorer": baseline_scorer,
        "panel": first["panel"],
        "rows": {scorer.EXPECTED_BASELINE_ROW_KEY: first["rows"]["4b-current"]},
        "inputs": {
            "annotation_root_sha256": scorer.sha256_path(annotations),
            "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
            "bert_model_snapshot_sha256": scorer.sha256_path(model),
            "scorer_sha256": "placeholder",
            "profile_sha256": scorer.profile_sha256(profile),
            "normalized_answer_map_sha256": scorer.normalized_answer_map_sha256(
                conditions["4b-current"]
            ),
            "source_prediction_sha256": "placeholder-source",
        },
    }
    baseline_path = tmp_path / "pinned-baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(scorer, "EXPECTED_BASELINE_ARTIFACT_SHA256", scorer.sha256_path(baseline_path))
    monkeypatch.setattr(
        scorer,
        "EXPECTED_BASELINE_SOURCE_PREDICTION_SHA256",
        scorer.sha256_path(conditions["4b-current"]),
    )
    monkeypatch.setattr(scorer, "EXPECTED_BASELINE_SCORER_SHA256", scorer.sha256_path(SEMANTIC_SCRIPT))
    baseline["inputs"]["scorer_sha256"] = scorer.EXPECTED_BASELINE_SCORER_SHA256
    baseline["inputs"]["source_prediction_sha256"] = scorer.EXPECTED_BASELINE_SOURCE_PREDICTION_SHA256
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(scorer, "EXPECTED_BASELINE_ARTIFACT_SHA256", scorer.sha256_path(baseline_path))

    calls: dict[str, int] = {}
    artifact = scorer.score_factorial(
        annotation_root=annotations,
        condition_paths=conditions,
        bert_model=model,
        profile=profile,
        panel_profile=panel,
        expected_rows=4,
        expected_annotation_files=4,
        baseline_artifact_path=baseline_path,
        baseline_source_prediction_path=conditions["4b-current"],
        meteor_fn=injected_metrics(calls)[0],
        bertscore_fn=injected_metrics(calls)[1],
        sts_fn=injected_metrics(calls)[2],
    )
    assert artifact["baseline_binding"]["reused"] is True
    assert artifact["baseline_binding"]["reason"] == "all bindings matched"
    assert calls["bert"] == 1

    mutated = copy.deepcopy(baseline)
    mutated["rows"][scorer.EXPECTED_BASELINE_ROW_KEY][0]["meteor"] = 0.99
    mutated_path = tmp_path / "mutated.json"
    mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
    assert scorer.sha256_path(mutated_path) != scorer.EXPECTED_BASELINE_ARTIFACT_SHA256
