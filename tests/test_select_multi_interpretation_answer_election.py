import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "select_multi_interpretation_answer_election.py"
SPEC = importlib.util.spec_from_file_location("multi_select", MODULE_PATH)
assert SPEC and SPEC.loader
multi_select = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multi_select)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path):
    source = tmp_path / "source"
    protocol_rows = []
    maps = []
    predictions = []
    scores = []
    annotations = tmp_path / "annotations"
    roles = ["premise_variance"] * 4 + ["insufficiency_safeguard"] * 4 + [
        "stability_control"
    ] * 4
    selected = []
    for index, role in enumerate(roles):
        video_id = f"v{index:02d}"
        protocol_rows.append({"video_id": video_id, "question": f"question {index}"})
        maps.append({"video_id": video_id, "status": "passed", "map": "OVERVIEW | one\nSCENE | two"})
        predictions.append({"video_id": video_id, "status": "passed", "answer": f"answer {index}"})
        scores.append({"video_id": video_id, "semantic_composite": index / 20, "question_stratum": "temporal"})
        _write(annotations / f"{video_id}.json", {"correct_answer": f"reference {index}"})
        selected.append({
            "video_id": video_id,
            "role": role,
            "evidence_line_numbers": [1, 2],
            "selection_note": f"adjudicated {role}",
        })
    _write(source / "protocol.json", {"rows": protocol_rows})
    _write(source / "maps.question-blind.json", maps)
    _write(source / "predictions.direct-answer.json", predictions)
    _write(source / "postfreeze-score.json", {"rows": {"gemini-3.1-pro-native-av-4fps": scores}})
    v2 = tmp_path / "v2.json"
    canary = tmp_path / "canary.json"
    _write(v2, {"rows": [{"video_id": "spent-v2"}]})
    _write(canary, {"rows": [{"video_id": "spent-canary"}]})
    config = {
        "schema_version": 1,
        "experiment_id": "fixture",
        "source_sha256": {name: multi_select.sha256_path(source / name) for name in multi_select.SOURCE_FILES},
        "exclusion_sha256": {"issue35_v2_protocol": multi_select.sha256_path(v2), "seven_row_canary_manifest": multi_select.sha256_path(canary)},
        "selected_rows": selected,
        "gate": multi_select.PROMOTION_GATE,
    }
    config_path = tmp_path / "config.json"
    _write(config_path, config)
    return source, annotations, v2, canary, config_path, config


def test_prepare_freezes_fresh_exact_four_by_four_panel(tmp_path: Path):
    source, annotations, v2, canary, config_path, _ = _fixture(tmp_path)
    output = tmp_path / "run"
    multi_select.prepare(source, annotations, v2, canary, config_path, output)
    selection = json.loads((output / "selection-freeze.json").read_text())
    protocol = json.loads((output / "protocol.source.json").read_text())
    gate = json.loads((output / "preinference-gate-freeze.json").read_text())
    assert selection["row_count"] == 12
    assert selection["role_counts"] == {role: 4 for role in multi_select.ROLES}
    assert [row["video_id"] for row in protocol["rows"]] == [f"v{i:02d}" for i in range(12)]
    assert set(protocol["rows"][0]) == {"video_id", "question", "evidence"}
    assert protocol["rows"][0]["evidence"][1] == {
        "evidence_id": "E02",
        "map_line_number": 2,
        "text": "SCENE | two",
    }
    assert "reference" not in json.dumps(protocol).lower()
    assert gate["selection_freeze_sha256"] == multi_select.sha256_path(output / "selection-freeze.json")


@pytest.mark.parametrize("mutation", ["duplicate", "spent", "role_count", "bad_line"])
def test_prepare_rejects_invalid_or_spent_selection(tmp_path: Path, mutation: str):
    source, annotations, v2, canary, config_path, config = _fixture(tmp_path)
    changed = copy.deepcopy(config)
    if mutation == "duplicate":
        changed["selected_rows"][1]["video_id"] = "v00"
    elif mutation == "spent":
        changed["selected_rows"][0]["video_id"] = "spent-v2"
    elif mutation == "role_count":
        changed["selected_rows"][0]["role"] = "stability_control"
    else:
        changed["selected_rows"][0]["evidence_line_numbers"] = [3]
    _write(config_path, changed)
    with pytest.raises(ValueError):
        multi_select.prepare(source, annotations, v2, canary, config_path, tmp_path / "run")


def test_prepare_rejects_source_or_exclusion_hash_drift(tmp_path: Path):
    source, annotations, v2, canary, config_path, _ = _fixture(tmp_path)
    maps = json.loads((source / "maps.question-blind.json").read_text())
    maps[0]["map"] += "\nSTATE | drift"
    _write(source / "maps.question-blind.json", maps)
    with pytest.raises(ValueError, match="source hash"):
        multi_select.prepare(source, annotations, v2, canary, config_path, tmp_path / "run")


def test_verify_selection_detects_tampering(tmp_path: Path):
    source, annotations, v2, canary, config_path, _ = _fixture(tmp_path)
    output = tmp_path / "run"
    multi_select.prepare(source, annotations, v2, canary, config_path, output)
    multi_select.verify_selection(output)
    protocol = json.loads((output / "protocol.source.json").read_text())
    protocol["rows"][0]["evidence"][0]["text"] = "tampered"
    _write(output / "protocol.source.json", protocol)
    with pytest.raises(ValueError, match="integrity"):
        multi_select.verify_selection(output)
