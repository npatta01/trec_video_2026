from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/score_vqa_silver_oracle.py"
FREEZE_BUILDER_TESTS = ROOT / "tests/test_build_vqa_silver_freeze_manifest.py"
BERT_SNAPSHOT = "722cf37b1afa9454edce342e7895e588b6ff1d59"
ARM_IDS = ("G/N", "G/F", "G/S", "H/N", "H/F", "H/S", "L/N", "L/F", "L/S")
FACTORIAL_IDS = (
    "4b-current",
    "4b-question-aware",
    "8b-current",
    "8b-question-aware",
)
POOL_IDS = ("A", "B", "C")
ANSWER_MODEL_ID = "Qwen/Qwen3.5-4B"
ANSWER_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EIGHT_B_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
EIGHT_B_MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TRANSFORMERS_REVISION = "cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
ANSWER_RUNTIME_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
JUDGE_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
JUDGE_MODEL_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
JUDGE_RUNTIME_IMAGE = (
    "docker.io/qwenllm/qwen3-omni@sha256:"
    "71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_vqa_silver_oracle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_freeze_fixture_helpers():
    spec = importlib.util.spec_from_file_location(
        "test_build_vqa_silver_freeze_manifest_helpers", FREEZE_BUILDER_TESTS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_identity(source: str) -> dict[str, object]:
    if source in ARM_IDS:
        model_id = ANSWER_MODEL_ID
        model_revision = ANSWER_MODEL_REVISION
        answer_policy = "current"
        condition_id = source.lower().replace("/", "")
    elif source.startswith("4b-"):
        model_id = ANSWER_MODEL_ID
        model_revision = ANSWER_MODEL_REVISION
        answer_policy = source.removeprefix("4b-")
        condition_id = source
    else:
        model_id = EIGHT_B_MODEL_ID
        model_revision = EIGHT_B_MODEL_REVISION
        answer_policy = source.removeprefix("8b-")
        condition_id = source
    return {
        "source_id": source,
        "condition_id": condition_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "answer_policy": answer_policy,
        "min_pixels_per_frame": 65536,
        "max_pixels_per_frame": 262144,
        "runtime": "transformers fixture; torch fixture",
        "runtime_image": ANSWER_RUNTIME_IMAGE,
        "transformers_revision": TRANSFORMERS_REVISION,
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": 64,
    }


def _make_fixture(
    tmp_path: Path,
    scorer,
    *,
    normalized_duplicate: bool = False,
    legacy_wins: bool = False,
):
    ids = ("v1", "v2", "v3")
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for index, video_id in enumerate(ids):
        _write_json(
            annotations / f"{video_id}.json",
            {
                "question": "What is shown?",
                "correct_answer": f"reference {index}",
                "incorrect_answers": [],
            },
        )

    def answer_for(source: str, index: int) -> str:
        if source == "4b-current":
            score = "0.99" if legacy_wins else "0.30"
        elif source == "4b-question-aware":
            score = "0.35"
        elif source == "8b-current":
            score = "0.20"
        elif source == "8b-question-aware":
            score = "0.10"
        else:
            score = {
                "G/N": "0.41",
                "G/F": "0.40",
                "G/S": "0.42",
                "H/N": "0.43",
                "H/F": "0.44",
                "H/S": "0.70",
                "L/N": "0.45",
                "L/F": "0.46",
                "L/S": "0.47",
            }[source]
        answer = f"score-{score}-{index}"
        if normalized_duplicate and source == "8b-question-aware" and index == 0:
            return "SCORE-0.20-0"
        return answer

    def write_predictions(source: str) -> Path:
        path = tmp_path / f"{source.replace('/', '')}.json"
        identity = _source_identity(source)
        _write_json(
            path,
            {
                "metadata": {
                    "labels_loaded_after_prediction_freeze": True,
                    "answer_model_id": identity["model_id"],
                    "answer_model_revision": identity["model_revision"],
                },
                "predictions": [
                    {
                        "status": "passed",
                        "video_id": video_id,
                        "answer": answer_for(source, index),
                        "condition_id": identity["condition_id"],
                        "model_id": identity["model_id"],
                        "model_revision": identity["model_revision"],
                        "answer_policy": identity["answer_policy"],
                        "requested_min_pixels_per_frame": 65536,
                        "requested_max_pixels_per_frame": 262144,
                        "min_pixels_per_frame": 65536,
                        "max_pixels_per_frame": 262144,
                        "runtime_image": ANSWER_RUNTIME_IMAGE,
                        "runtime": "transformers fixture; torch fixture",
                    }
                    for index, video_id in enumerate(ids)
                ],
            },
        )
        return path

    arms = {arm_id: write_predictions(arm_id) for arm_id in ARM_IDS}
    factorial = {
        condition_id: write_predictions(condition_id) for condition_id in FACTORIAL_IDS
    }

    def candidate(source: str, index: int, candidate_id: str) -> dict[str, str]:
        return {
            "candidate_id": candidate_id,
            "arm_id": source,
            "answer": answer_for(source, index),
        }

    election_payloads = {}
    for pool_id, sources in {
        "A": FACTORIAL_IDS,
        "B": ARM_IDS,
        "C": FACTORIAL_IDS + ARM_IDS,
    }.items():
        rows = []
        for index, video_id in enumerate(ids):
            candidates = [
                candidate(source, index, f"{pool_id.lower()}-{n}")
                for n, source in enumerate(sources)
            ]
            ranking = []
            normalized_answers: set[str] = set()
            for item in candidates:
                normalized = scorer._normalize_candidate_text(item["answer"])
                if normalized not in normalized_answers:
                    normalized_answers.add(normalized)
                    ranking.append(item["candidate_id"])
            if pool_id == "A":
                ranking.insert(0, ranking.pop(1))
            elif pool_id in {"B", "C"}:
                hs_candidate_id = next(
                    item["candidate_id"]
                    for item in candidates
                    if item["arm_id"] == "H/S"
                )
                hs_index = ranking.index(hs_candidate_id)
                ranking.insert(0, ranking.pop(hs_index))
            rows.append(
                {
                    "video_id": video_id,
                    "candidates": candidates,
                    "ranking": ranking,
                }
            )
        election_payloads[pool_id] = _write_json(
            tmp_path / f"pool-{pool_id.lower()}.json",
            {
                "metadata": {
                    "judge_model_id": JUDGE_MODEL_ID,
                    "judge_model_revision": JUDGE_MODEL_REVISION,
                    "labels_loaded_after_prediction_freeze": True,
                },
                "rows": rows,
            },
        )

    model = tmp_path / BERT_SNAPSHOT
    model.mkdir()
    (model / "config.json").write_text("fixture", encoding="utf-8")
    condition_bindings = {}
    for source, path in {**factorial, **arms}.items():
        condition_bindings[source] = {
            **_source_identity(source),
            "prediction_sha256": _sha256_path(path),
            "request_sha256": hashlib.sha256(
                f"frozen-request:{source}".encode()
            ).hexdigest(),
        }
    election_bindings = {
        pool_id: {
            "pool_id": pool_id,
            "ranking_sha256": _sha256_path(path),
            "request_sha256": hashlib.sha256(
                f"frozen-election-request:{pool_id}".encode()
            ).hexdigest(),
            "judge_model_id": JUDGE_MODEL_ID,
            "judge_model_revision": JUDGE_MODEL_REVISION,
            "runtime_image": JUDGE_RUNTIME_IMAGE,
        }
        for pool_id, path in election_payloads.items()
    }
    answer_freeze = {
        source: {
            "prediction_sha256": binding["prediction_sha256"],
            "request_sha256": binding["request_sha256"],
        }
        for source, binding in condition_bindings.items()
    }
    judge_freeze = {
        pool_id: {
            "ranking_sha256": binding["ranking_sha256"],
            "request_sha256": binding["request_sha256"],
        }
        for pool_id, binding in election_bindings.items()
    }
    bindings = {
        "schema_version": 1,
        "protocol_sha256": "1" * 64,
        "evidence_sha256": "2" * 64,
        "selector_sha256": "3" * 64,
        "answer_freeze_sha256": _canonical_sha256(answer_freeze),
        "judge_freeze_sha256": _canonical_sha256(judge_freeze),
        "panel_id_set_sha256": scorer.factorial.sorted_video_id_set_sha256(ids),
        "verified_4b_current_prediction_sha256": condition_bindings["4b-current"][
            "prediction_sha256"
        ],
        "verified_4b_current_source_prediction_sha256": "4" * 64,
        "conditions": condition_bindings,
        "elections": election_bindings,
    }
    return annotations, arms, factorial, election_payloads, model, bindings, ids


def _refresh_bindings(
    bindings: dict[str, object],
    *,
    arms: dict[str, Path],
    factorial: dict[str, Path],
    elections: dict[str, Path],
) -> None:
    conditions = bindings["conditions"]
    assert isinstance(conditions, dict)
    for source, path in {**factorial, **arms}.items():
        conditions[source]["prediction_sha256"] = _sha256_path(path)
    election_bindings = bindings["elections"]
    assert isinstance(election_bindings, dict)
    for pool_id, path in elections.items():
        election_bindings[pool_id]["ranking_sha256"] = _sha256_path(path)
    bindings["verified_4b_current_prediction_sha256"] = conditions["4b-current"][
        "prediction_sha256"
    ]
    bindings["answer_freeze_sha256"] = _canonical_sha256(
        {
            source: {
                "prediction_sha256": binding["prediction_sha256"],
                "request_sha256": binding["request_sha256"],
            }
            for source, binding in conditions.items()
        }
    )
    bindings["judge_freeze_sha256"] = _canonical_sha256(
        {
            pool_id: {
                "ranking_sha256": binding["ranking_sha256"],
                "request_sha256": binding["request_sha256"],
            }
            for pool_id, binding in election_bindings.items()
        }
    )


def _injected_metrics():
    calls = {"meteor": 0, "bert": 0, "sts": 0}

    def value(text: str) -> float:
        return float(text.split("-")[1])

    def meteor(prediction: str, reference: str) -> float:
        del reference
        calls["meteor"] += 1
        return value(prediction)

    def bertscore(
        predictions: list[str], references: list[str], model: Path
    ) -> list[float]:
        del references, model
        calls["bert"] += 1
        return [value(prediction) for prediction in predictions]

    def sts(prediction: str, reference: str) -> float:
        del reference
        calls["sts"] += 1
        return value(prediction)

    return calls, meteor, bertscore, sts


def _score_prepared_fixture(scorer, fixture, **overrides):
    annotations, arms, factorial, elections, model, bindings, ids = fixture
    calls, meteor, bertscore, sts = _injected_metrics()
    kwargs = {
        "annotation_root": annotations,
        "arm_paths": arms,
        "factorial_paths": factorial,
        "election_paths": elections,
        "bert_model": model,
        "expected_rows": len(ids),
        "expected_annotation_files": len(ids),
        "bindings": bindings,
        "evidence_coverage": {video_id: True for video_id in ids},
        "meteor_fn": meteor,
        "bertscore_fn": bertscore,
        "sts_fn": sts,
        "bootstrap_iterations": 10_000,
        "bootstrap_seed": 20260809,
    }
    kwargs.update(overrides)
    return scorer.score_silver_oracle(**kwargs), calls


def _score_fixture(tmp_path: Path, scorer, **overrides):
    return _score_prepared_fixture(scorer, _make_fixture(tmp_path, scorer), **overrides)


def test_scores_nine_arms_and_three_elections_with_pinned_arithmetic(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    artifact, calls = _score_fixture(tmp_path, scorer)

    assert set(artifact["arms"]) == set(ARM_IDS)
    assert set(artifact["elections"]) == set(POOL_IDS)
    assert artifact["arms"]["G/F"]["semantic_composite"] == pytest.approx(0.4)
    assert artifact["arms"]["H/S"]["semantic_composite"] == pytest.approx(0.7)
    assert artifact["elections"]["A"]["top1"]["semantic_composite"] == pytest.approx(
        0.35
    )
    assert artifact["elections"]["B"]["top1"]["semantic_composite"] == pytest.approx(
        0.7
    )
    assert artifact["elections"]["C"]["unique_candidate_count"] == 13
    assert artifact["completion"]["arms"]["H/S"] == {"completed": 3, "expected": 3}
    assert artifact["completion"]["elections"]["A"] == {"complete": 3, "expected": 3}
    assert calls["bert"] == 1
    assert calls["sts"] == calls["meteor"]
    encoded = json.dumps(artifact, sort_keys=True)
    assert "score-" not in encoded
    assert "reference 0" not in encoded


def test_declared_contrasts_bootstrap_and_gates_are_reference_locked(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    artifact, _calls = _score_fixture(tmp_path, scorer)

    assert set(artifact["contrasts"]) == {
        "H/S-G/F",
        "H/F-G/F",
        "L/F-G/F",
        "G/S-G/F",
        "H/S-H/F",
        "L/S-L/F",
    }
    primary = artifact["contrasts"]["H/S-G/F"]
    assert primary["metric_deltas"]["semantic_composite"] == pytest.approx(0.3)
    assert primary["bootstrap"]["semantic_composite"]["iterations"] == 10_000
    assert primary["bootstrap"]["semantic_composite"]["seed"] == 20260809
    for gate_id in ("evidence", "election"):
        gate = artifact["gates"][gate_id]
        assert gate["passed"] is False
        assert gate["gates"]["integrity"] is False
        assert all(
            check is True
            for check_id, check in gate["gates"].items()
            if check_id != "integrity"
        )
    assert artifact["ceilings"]["pool_a"]["capture_fraction"] == pytest.approx(1.0)


def test_reference_aware_maxima_and_ndcg_use_row_local_candidates(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    artifact, _calls = _score_fixture(tmp_path, scorer)

    assert artifact["ceilings"]["nine"]["semantic_composite"] == pytest.approx(0.7)
    assert artifact["ceilings"]["pool_a"]["semantic_composite"] == pytest.approx(0.35)
    assert artifact["elections"]["A"]["ndcg_k"] == 4
    assert artifact["elections"]["A"]["ndcg"] == pytest.approx(1.0)
    assert 0.0 < artifact["elections"]["B"]["ndcg"] < 1.0
    assert artifact["ceilings"]["pool_a"]["capture_fraction"] == pytest.approx(1.0)
    assert "rows" not in artifact


def test_nine_arm_ceiling_excludes_a_legacy_source_that_wins_pool_c(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer, legacy_wins=True)
    artifact, _calls = _score_prepared_fixture(scorer, fixture)

    assert artifact["ceilings"]["nine"]["semantic_composite"] == pytest.approx(0.7)
    assert artifact["ceilings"]["pool_c"]["semantic_composite"] == pytest.approx(0.99)


def test_ndcg_uses_the_nist_ranked_answer_contract_and_is_not_claimed_for_single_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    calls: list[tuple[list[float], int]] = []
    official_ndcg = scorer.ndcg_from_gains

    def recorded_ndcg(gains, k):
        calls.append((list(gains), k))
        return official_ndcg(gains, k)

    monkeypatch.setattr(scorer, "ndcg_from_gains", recorded_ndcg)
    artifact, _metric_calls = _score_fixture(tmp_path, scorer)

    assert len(calls) == 9
    assert all(k == min(10, len(gains)) for gains, k in calls)
    assert artifact["scorer"]["single_answer_ndcg"] == {
        "applicable": False,
        "reason": (
            "competition evaluator marks one free-form answer NDCG not applicable; "
            "a single-item ranking is degenerate at 1.0"
        ),
    }
    assert artifact["scorer"]["election_ndcg"] == {
        "applicable": True,
        "contract": "NIST 2025 order-sensitive ranked-answer NDCG",
        "cutoff": "min(10, unique_candidate_count)",
        "gain": "semantic_composite",
        "role": "local_election_diagnostic",
    }
    assert all("ndcg" not in report for report in artifact["arms"].values())
    assert all("ndcg" in report for report in artifact["elections"].values())
    assert artifact["inputs"]["evaluation_module_sha256"] == _sha256_path(
        ROOT / "src/trec_video_2026/vqa_evaluation.py"
    )


def test_freezes_prediction_bytes_before_loading_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    annotations, arms, factorial, elections, model, bindings, ids = _make_fixture(
        tmp_path, scorer
    )
    original_loader = scorer._load_annotations
    mutated = {"done": False}

    def load_labels_after_freeze(path: Path):
        assert scorer._FROZEN_INPUT_DIGESTS
        arms["G/F"].write_text("[]", encoding="utf-8")
        mutated["done"] = True
        return original_loader(path)

    monkeypatch.setattr(scorer, "_load_annotations", load_labels_after_freeze)
    calls, meteor, bertscore, sts = _injected_metrics()
    artifact = scorer.score_silver_oracle(
        annotation_root=annotations,
        arm_paths=arms,
        factorial_paths=factorial,
        election_paths=elections,
        bert_model=model,
        expected_rows=len(ids),
        expected_annotation_files=len(ids),
        bindings=bindings,
        evidence_coverage={video_id: True for video_id in ids},
        meteor_fn=meteor,
        bertscore_fn=bertscore,
        sts_fn=sts,
        bootstrap_iterations=10_000,
    )
    assert mutated["done"] is True
    assert artifact["completion"]["arms"]["G/F"] == {"completed": 3, "expected": 3}


def test_rejects_incomplete_or_misaligned_inputs_before_metrics(tmp_path: Path) -> None:
    scorer = load_scorer()
    annotations, arms, factorial, elections, model, bindings, ids = _make_fixture(
        tmp_path, scorer
    )
    rows = json.loads(arms["H/S"].read_text(encoding="utf-8"))
    rows["predictions"][-1]["status"] = "failed"
    arms["H/S"].write_text(json.dumps(rows), encoding="utf-8")
    _refresh_bindings(bindings, arms=arms, factorial=factorial, elections=elections)
    calls, meteor, bertscore, sts = _injected_metrics()

    with pytest.raises(ValueError, match="incomplete|completion"):
        scorer.score_silver_oracle(
            annotation_root=annotations,
            arm_paths=arms,
            factorial_paths=factorial,
            election_paths=elections,
            bert_model=model,
            expected_rows=len(ids),
            expected_annotation_files=len(ids),
            bindings=bindings,
            evidence_coverage={video_id: True for video_id in ids},
            meteor_fn=meteor,
            bertscore_fn=bertscore,
            sts_fn=sts,
            bootstrap_iterations=10_000,
        )
    assert calls == {"meteor": 0, "bert": 0, "sts": 0}


def test_reuses_the_pinned_private_sts_cache_without_live_calls(tmp_path: Path) -> None:
    scorer = load_scorer()
    cache = scorer.STSCache()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, first_calls = _score_fixture(first_root, scorer, sts_cache=cache)
    second, second_calls = _score_fixture(second_root, scorer, sts_cache=cache)

    assert (
        first["cache_summary"]["miss_count"]
        == first["cache_summary"]["unique_pair_count"]
    )
    assert (
        second["cache_summary"]["hit_count"]
        == second["cache_summary"]["unique_pair_count"]
    )
    assert first_calls["sts"] > 0
    assert second_calls["sts"] == 0


def test_caller_keys_and_bundle_metadata_cannot_establish_provenance(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    with pytest.raises(ValueError, match="binding|provenance"):
        _score_fixture(tmp_path, scorer, bindings=None)

    unanchored_root = tmp_path / "unanchored"
    unanchored_root.mkdir()
    unanchored, _calls = _score_fixture(unanchored_root, scorer)
    assert unanchored["integrity"]["provenance_checked"] is False
    assert unanchored["gates"]["evidence"]["passed"] is False
    assert unanchored["gates"]["election"]["passed"] is False


def test_canonical_panel_and_verified_4b_current_digests_are_pinned() -> None:
    scorer = load_scorer()

    assert scorer.EXPECTED_CANONICAL_PANEL_SHA256 == (
        "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
    )
    assert scorer.EXPECTED_VERIFIED_4B_CURRENT_PREDICTION_SHA256 == (
        "973989e0e10cffb2a9f211ad754008a373f69171319a67bcf17d6bd9e9b8f623"
    )
    assert scorer.EXPECTED_VERIFIED_4B_CURRENT_SOURCE_PREDICTION_SHA256 == (
        "ad9c9aabace33d3091c076e2f06c10ef8e55d355be9e5e392d790a80dbd78377"
    )


def test_canonical_freeze_manifest_has_a_nonzero_compile_time_pin() -> None:
    scorer = load_scorer()

    assert re.fullmatch(
        r"[0-9a-f]{64}", scorer.EXPECTED_CANONICAL_FREEZE_MANIFEST_SHA256
    )
    assert scorer.EXPECTED_CANONICAL_FREEZE_MANIFEST_SHA256 != "0" * 64
    assert re.fullmatch(
        r"[0-9a-f]{64}", scorer.CANONICAL_FREEZE_MANIFEST_SHA256_PLACEHOLDER
    )
    assert scorer.CANONICAL_FREEZE_MANIFEST_SHA256_PLACEHOLDER != "0" * 64


def test_canonical_manifest_anchor_is_not_caller_overridable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    manifest = _write_json(tmp_path / "freeze-manifest.json", {"schema_version": 1})

    with pytest.raises(ValueError, match="placeholder|pin"):
        scorer._verify_manifest_trust_anchor(manifest, expected_rows=475)

    digest = _sha256_path(manifest)
    monkeypatch.setattr(scorer, "EXPECTED_CANONICAL_FREEZE_MANIFEST_SHA256", digest)
    assert scorer._verify_manifest_trust_anchor(manifest, expected_rows=475) == digest
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="anchor|digest|manifest"):
        scorer._verify_manifest_trust_anchor(manifest, expected_rows=475)


def test_canonical_scorer_checks_manifest_anchor_before_opening_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = load_scorer()
    manifest = _write_json(tmp_path / "freeze-manifest.json", {"schema_version": 1})
    opened = False

    def record_annotation_open(_path: Path):
        nonlocal opened
        opened = True
        return {}

    monkeypatch.setattr(scorer, "_load_annotations", record_annotation_open)
    with pytest.raises(ValueError, match="placeholder|pin"):
        scorer.score_silver_oracle(
            annotation_root=tmp_path / "annotations",
            arm_paths={},
            factorial_paths={},
            election_paths={},
            bert_model=tmp_path / BERT_SNAPSHOT,
            freeze_manifest_path=manifest,
            freeze_inputs={},
        )
    assert opened is False


def _make_anchored_score_fixture(tmp_path: Path, scorer):
    helpers = load_freeze_fixture_helpers()
    freeze_fixture = helpers._build_label_free_fixture(tmp_path)
    builder = helpers.load_builder()
    manifest_path = tmp_path / "freeze-manifest.json"
    builder.write_freeze_manifest(
        manifest_path,
        builder.build_freeze_manifest(freeze_fixture["freeze_inputs"], expected_rows=2),
    )
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for video_id in freeze_fixture["ids"]:
        _write_json(
            annotations / f"{video_id}.json",
            {
                "question": "What is visible?",
                "correct_answer": "fixture reference",
                "incorrect_answers": [],
            },
        )
    model = tmp_path / BERT_SNAPSHOT
    model.mkdir()
    (model / "config.json").write_text("fixture", encoding="utf-8")
    kwargs = {
        "annotation_root": annotations,
        "arm_paths": {
            source_id: freeze_fixture["prediction_paths"][source_id]
            for source_id in ARM_IDS
        },
        "factorial_paths": {
            source_id: freeze_fixture["prediction_paths"][source_id]
            for source_id in FACTORIAL_IDS
        },
        "election_paths": freeze_fixture["ranking_paths"],
        "bert_model": model,
        "expected_rows": 2,
        "expected_annotation_files": 2,
        "freeze_manifest_path": manifest_path,
        "freeze_inputs": freeze_fixture["freeze_inputs"],
        "meteor_fn": lambda _prediction, _reference: 0.5,
        "bertscore_fn": lambda predictions, _references, _model: [
            0.5 for _prediction in predictions
        ],
        "sts_fn": lambda _prediction, _reference: 0.5,
    }
    return freeze_fixture, manifest_path, kwargs


def test_anchored_scorer_uses_the_verified_manifest_and_coverage_bytes(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    freeze_fixture, manifest_path, kwargs = _make_anchored_score_fixture(
        tmp_path, scorer
    )
    artifact = scorer.score_silver_oracle(**kwargs)

    assert artifact["inputs"]["freeze_manifest_sha256"] == _sha256_path(manifest_path)
    assert artifact["provenance"]["freeze_manifest_sha256"] == _sha256_path(
        manifest_path
    )
    assert artifact["inputs"]["localized_coverage_sha256"] == _sha256_path(
        freeze_fixture["coverage_path"]
    )
    assert artifact["provenance"]["localized_construction_count"] == 1
    assert artifact["integrity"]["provenance_checked"] is True


@pytest.mark.parametrize("target", ["coverage", "prediction"])
def test_anchored_scorer_rejects_label_free_mutation_after_annotations_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    scorer = load_scorer()
    freeze_fixture, _manifest_path, kwargs = _make_anchored_score_fixture(
        tmp_path, scorer
    )
    original_loader = scorer._load_annotations

    def mutate_after_open(path: Path):
        annotations = original_loader(path)
        if target == "coverage":
            mutated_path = freeze_fixture["coverage_path"]
        else:
            mutated_path = freeze_fixture["prediction_paths"]["G/N"]
        mutated_path.write_bytes(mutated_path.read_bytes() + b" ")
        return annotations

    monkeypatch.setattr(scorer, "_load_annotations", mutate_after_open)
    with pytest.raises(
        ValueError, match="coverage|manifest|freeze|identity|checksum|prediction"
    ):
        scorer.score_silver_oracle(**kwargs)


@pytest.mark.parametrize(
    ("target", "field", "replacement"),
    [
        ("condition", "source_id", "caller-key-is-not-a-binding"),
        ("condition", "condition_id", "wrong-condition"),
        ("condition", "prediction_sha256", "0" * 64),
        ("condition", "request_sha256", "not-a-digest"),
        ("condition", "model_revision", "0" * 40),
        ("condition", "answer_policy", "question-aware"),
        ("condition", "min_pixels_per_frame", 1),
        ("condition", "runtime_image", "docker.io/unpinned:latest"),
        ("condition", "transformers_revision", "0" * 40),
        ("top", "verified_4b_current_prediction_sha256", "0" * 64),
        ("top", "panel_id_set_sha256", "0" * 64),
        ("election", "pool_id", "B"),
        ("election", "ranking_sha256", "0" * 64),
        ("election", "request_sha256", "not-a-digest"),
    ],
)
def test_requires_exact_frozen_condition_request_runtime_and_election_bindings(
    tmp_path: Path, target: str, field: str, replacement: object
) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer)
    bindings = copy.deepcopy(fixture[5])
    if target == "condition":
        bindings["conditions"]["G/F"][field] = replacement
    elif target == "election":
        bindings["elections"]["A"][field] = replacement
    else:
        bindings[field] = replacement

    with pytest.raises(ValueError, match="binding|digest|identity|panel|runtime"):
        _score_prepared_fixture(scorer, fixture, bindings=bindings)


@pytest.mark.parametrize("coverage", [None, 1.0, {"v1": True, "v2": True}])
def test_localized_coverage_is_an_explicit_panel_complete_input(
    tmp_path: Path, coverage: object
) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer)

    with pytest.raises(ValueError, match="coverage"):
        _score_prepared_fixture(scorer, fixture, evidence_coverage=coverage)


def test_cli_requires_the_pinned_manifest_and_label_free_input_locator() -> None:
    scorer = load_scorer()
    parser = scorer.build_parser()

    assert parser._option_string_actions["--freeze-manifest"].required is True
    assert parser._option_string_actions["--freeze-inputs"].required is True
    assert "--bindings" not in parser._option_string_actions
    assert "--localized-coverage" not in parser._option_string_actions


def test_prediction_rows_must_carry_the_bound_condition_identity(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer)
    _annotations, arms, factorial, elections, _model, bindings, _ids = fixture
    payload = json.loads(arms["H/S"].read_text(encoding="utf-8"))
    payload["predictions"][0]["condition_id"] = "gf"
    _write_json(arms["H/S"], payload)
    _refresh_bindings(bindings, arms=arms, factorial=factorial, elections=elections)

    with pytest.raises(ValueError, match="condition|identity|binding"):
        _score_prepared_fixture(scorer, fixture)


def _rewrite_pool(fixture, pool_id: str, mutation) -> None:
    _annotations, arms, factorial, elections, _model, bindings, _ids = fixture
    payload = json.loads(elections[pool_id].read_text(encoding="utf-8"))
    mutation(payload["rows"][0])
    _write_json(elections[pool_id], payload)
    _refresh_bindings(bindings, arms=arms, factorial=factorial, elections=elections)


def test_every_election_candidate_requires_an_explicit_frozen_source(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer)
    _rewrite_pool(fixture, "A", lambda row: row["candidates"][0].pop("arm_id"))

    with pytest.raises(ValueError, match="arm_id|source"):
        _score_prepared_fixture(scorer, fixture)


def test_election_candidate_answer_must_match_every_frozen_source_byte(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer)

    def rewrite_bytes(row) -> None:
        row["candidates"][0]["answer"] = row["candidates"][0]["answer"].upper()

    _rewrite_pool(fixture, "A", rewrite_bytes)
    with pytest.raises(ValueError, match="rewrites|byte|answer"):
        _score_prepared_fixture(scorer, fixture)


def test_election_candidate_schema_rejects_arbitrary_fields(tmp_path: Path) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer)

    def add_arbitrary_field(row) -> None:
        row["candidates"][0]["note"] = "caller supplied"

    _rewrite_pool(fixture, "A", add_arbitrary_field)
    with pytest.raises(ValueError, match="candidate.*schema|field"):
        _score_prepared_fixture(scorer, fixture)


def test_election_rejects_an_arbitrary_candidate_source(tmp_path: Path) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer)

    def add_rogue_candidate(row) -> None:
        row["candidates"].append(
            {
                "candidate_id": "rogue",
                "arm_id": "not-a-frozen-source",
                "answer": row["candidates"][0]["answer"],
            }
        )
        row["ranking"].append("rogue")

    _rewrite_pool(fixture, "A", add_rogue_candidate)
    with pytest.raises(ValueError, match="arm_id|source|membership"):
        _score_prepared_fixture(scorer, fixture)


def test_pool_membership_is_exact_for_each_row_after_normalized_dedupe(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer)

    def remove_unique_member(row) -> None:
        removed = row["candidates"].pop()
        row["ranking"].remove(removed["candidate_id"])

    _rewrite_pool(fixture, "B", remove_unique_member)
    with pytest.raises(ValueError, match="membership|declared|candidate"):
        _score_prepared_fixture(scorer, fixture)


def test_pool_rankings_are_complete_permutations_of_deduped_candidates(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer)
    _rewrite_pool(fixture, "C", lambda row: row["ranking"].pop())

    with pytest.raises(ValueError, match="complete candidate permutation"):
        _score_prepared_fixture(scorer, fixture)


def test_normalized_dedupe_preserves_first_candidate_and_aggregate_counts(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer, normalized_duplicate=True)
    artifact, _calls = _score_prepared_fixture(scorer, fixture)

    assert artifact["elections"]["A"]["candidate_count_max"] == 4
    assert artifact["elections"]["A"]["unique_candidate_count_min"] == 3
    assert artifact["elections"]["A"]["ndcg_k_min"] == 3
    assert artifact["elections"]["A"]["ndcg_k_max"] == 4
    assert artifact["elections"]["C"]["candidate_count_max"] == 13
    assert artifact["elections"]["C"]["unique_candidate_count_min"] == 12


def test_normalized_dedupe_does_not_hide_a_missing_declared_source(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    fixture = _make_fixture(tmp_path, scorer, normalized_duplicate=True)

    def remove_deduped_source(row) -> None:
        row["candidates"] = [
            candidate
            for candidate in row["candidates"]
            if candidate["arm_id"] != "8b-question-aware"
        ]

    _rewrite_pool(fixture, "A", remove_deduped_source)
    with pytest.raises(ValueError, match="membership|declared"):
        _score_prepared_fixture(scorer, fixture)


def test_safe_result_contains_no_row_ids_strata_or_evaluation_arrays(
    tmp_path: Path,
) -> None:
    scorer = load_scorer()
    artifact, _calls = _score_fixture(tmp_path, scorer)

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            assert not {"video_id", "stratum", "strata", "rows", "captures"} & set(
                value
            )
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            assert not any(isinstance(child, dict) for child in value)
            for child in value:
                inspect(child)

    inspect(artifact)


def test_safe_artifact_allowlist_rejects_labels_paths_and_unknown_fields() -> None:
    scorer = load_scorer()
    safe = {
        "schema_version": 1,
        "experiment": {"name": "fixture"},
        "panel": {"expected_rows": 3},
        "arms": {},
        "factorial": {},
        "elections": {},
        "ceilings": {},
        "contrasts": {},
        "election_contrasts": {},
        "gates": {},
        "completion": {},
        "provenance": {},
        "inputs": {},
        "scorer": {},
        "integrity": {},
        "cache_summary": {},
    }
    scorer.validate_safe_artifact(safe)
    for bad in (
        {**safe, "answer": "private"},
        {**safe, "unknown": True},
        {**safe, "provenance": {"private_path": "/tmp/secret"}},
        {**safe, "rows": []},
        {**safe, "arms": {"G/N": {"video_id": "v1"}}},
        {**safe, "elections": {"A": {"evaluation_rows": [{"meteor": 1.0}]}}},
        {**safe, "elections": {"A": {"top1": [{"meteor": 1.0}]}}},
    ):
        with pytest.raises(ValueError):
            scorer.validate_safe_artifact(bad)


def test_atomic_output_refuses_overwrite(tmp_path: Path) -> None:
    scorer = load_scorer()
    output = tmp_path / "aggregate.json"
    scorer.atomic_write_json(output, {"safe": True})
    with pytest.raises(FileExistsError):
        scorer.atomic_write_json(output, {"safe": True})
