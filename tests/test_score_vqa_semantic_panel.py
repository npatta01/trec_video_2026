from __future__ import annotations

from io import BytesIO
import importlib.util
import json
from pathlib import Path
from urllib.error import URLError

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/score_vqa_semantic_panel.py"


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
    model = tmp_path / "roberta-large"
    model.write_bytes(b"frozen model fixture")
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
