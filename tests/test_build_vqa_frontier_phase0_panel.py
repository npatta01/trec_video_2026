from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from test_vqa_frontier_phase0_panel import qwen_fixture, source_panel_fixture


PILOT_BINDING_KEYS = {
    "delivery_policy_version",
    "panel_sha256",
    "prompt_sha256",
    "provider",
    "provider_schema_sha256",
    "raw_files_sha256",
    "response_format_policy",
    "result_count",
    "results_sha256",
    "schema_sha256",
}


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_vqa_frontier_phase0_panel.py"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_vqa_frontier_phase0_panel", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    source = source_panel_fixture()
    qwen = qwen_fixture(source)
    source_path = _write_json(tmp_path / "frozen-panel.json", source)
    qwen_path = _write_json(tmp_path / "predictions.frames-only.json", qwen)

    pilot = next(row for row in source["examples"] if row["video_id"] == "spent-pilot")
    pilot_result = {
        "schema_version": 1,
        "run_id": "frontier-video-map-20260811",
        "provider": "gemini",
        "phase": "preflight",
        "video_id": "spent-pilot",
        "ordinal": pilot["ordinal"],
        "status": "succeeded",
        "source_sha256": pilot["media_sha256"],
        "panel_sha256": _canonical_sha256(source),
    }
    result_path = tmp_path / "run" / "preflight-gemini-results.json"
    sibling_path = result_path.parent / "results" / "gemini.jsonl"
    sibling_path.parent.mkdir(parents=True, exist_ok=True)
    sibling_path.write_text(
        json.dumps(pilot_result, sort_keys=True) + "\n", encoding="utf-8"
    )
    aggregate = {
        "schema_version": 1,
        "runner_id": "trec-video-frontier-map-runner-v1",
        "run_id": "frontier-video-map-20260811",
        "provider": "gemini",
        "prompt_sha256": "1" * 64,
        "schema_sha256": "2" * 64,
        "delivery_policy_version": "frontier-map-delivery-v2-webm-copy",
        "result_count": 1,
        "succeeded_count": 1,
        "failed_count": 0,
        "panel_sha256": _canonical_sha256(source),
        "results_sha256": _canonical_sha256([pilot_result]),
        "raw_files_sha256": "3" * 64,
        "provider_schema_sha256": "4" * 64,
        "response_format_policy": "prompt-json-local-validation-v3-exact-fence",
    }
    _write_json(result_path, aggregate)
    bindings = {
        "delivery_policy_version": aggregate["delivery_policy_version"],
        "provider": "gemini",
        "panel_sha256": aggregate["panel_sha256"],
        "prompt_sha256": aggregate["prompt_sha256"],
        "provider_schema_sha256": aggregate["provider_schema_sha256"],
        "raw_files_sha256": aggregate["raw_files_sha256"],
        "response_format_policy": aggregate["response_format_policy"],
        "result_count": 1,
        "results_sha256": aggregate["results_sha256"],
        "schema_sha256": aggregate["schema_sha256"],
    }
    freeze_path = result_path.parent / "preflight-gemini-freeze.json"
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "artifact_sha256": _sha256(result_path),
        "bindings": bindings,
        "bindings_sha256": _canonical_sha256(bindings),
    }
    _write_json(freeze_path, freeze)
    return {
        "source": source_path,
        "qwen": qwen_path,
        "results": result_path,
        "freeze": freeze_path,
    }


def test_cli_builds_atomic_private_manifest_and_safe_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "CANONICAL_DATA_ROOT", tmp_path)
    paths = _write_fixture_inputs(tmp_path)
    output = tmp_path / "panel.json"

    assert (
        cli.main(
            [
                "--source-panel",
                str(paths["source"]),
                "--successful-pilot-results",
                str(paths["results"]),
                "--successful-pilot-freeze",
                str(paths["freeze"]),
                "--qwen-predictions",
                str(paths["qwen"]),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    output_text = capsys.readouterr().out
    summary = json.loads(output_text)
    assert summary["run_id"] == "vqa-frontier-phase0-20260812"
    assert summary["pilot_count"] == 1
    assert summary["screen_count"] == 7
    assert summary["categories"] == [
        "causal",
        "counting",
        "ocr_detail",
        "other_global",
        "sound",
        "speech",
        "temporal",
    ]
    assert "private baseline answer" not in output_text
    assert manifest["bindings"]["source_panel_file_sha256"] == _sha256(paths["source"])
    assert manifest["bindings"]["qwen_prediction_file_sha256"] == _sha256(paths["qwen"])
    assert "question" in json.dumps(manifest)
    assert all("answer" not in row for row in manifest["examples"])

    with pytest.raises(FileExistsError):
        cli.main(
            [
                "--source-panel",
                str(paths["source"]),
                "--successful-pilot-results",
                str(paths["results"]),
                "--successful-pilot-freeze",
                str(paths["freeze"]),
                "--qwen-predictions",
                str(paths["qwen"]),
                "--output",
                str(output),
            ]
        )


def test_cli_rejects_tampered_successful_pilot_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "CANONICAL_DATA_ROOT", tmp_path)
    paths = _write_fixture_inputs(tmp_path)
    aggregate = json.loads(paths["results"].read_text(encoding="utf-8"))
    aggregate["succeeded_count"] = 0
    paths["results"].write_text(json.dumps(aggregate), encoding="utf-8")

    with pytest.raises(ValueError, match="freeze|aggregate|successful"):
        cli.main(
            [
                "--source-panel",
                str(paths["source"]),
                "--successful-pilot-results",
                str(paths["results"]),
                "--successful-pilot-freeze",
                str(paths["freeze"]),
                "--qwen-predictions",
                str(paths["qwen"]),
                "--output",
                str(tmp_path / "panel.json"),
            ]
        )


def test_cli_derives_pilot_from_sibling_result_not_an_id_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "CANONICAL_DATA_ROOT", tmp_path)
    paths = _write_fixture_inputs(tmp_path)
    sibling = paths["results"].parent / "results" / "gemini.jsonl"
    row = json.loads(sibling.read_text(encoding="utf-8"))
    row["video_id"] = "unused-pilot"
    sibling.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pilot|source|result"):
        cli.main(
            [
                "--source-panel",
                str(paths["source"]),
                "--successful-pilot-results",
                str(paths["results"]),
                "--successful-pilot-freeze",
                str(paths["freeze"]),
                "--qwen-predictions",
                str(paths["qwen"]),
                "--output",
                str(tmp_path / "panel.json"),
            ]
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("schema_version", "schema_version"),
        ("status", "status"),
        ("binding_key", "exactly"),
        ("binding_value", "binding"),
        ("bindings_sha256", "bindings_sha256"),
        ("artifact_sha256", "artifact_sha256"),
    ],
)
def test_cli_rejects_mutated_pilot_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "CANONICAL_DATA_ROOT", tmp_path)
    paths = _write_fixture_inputs(tmp_path)
    freeze = json.loads(paths["freeze"].read_text(encoding="utf-8"))
    if mutation == "schema_version":
        freeze["schema_version"] = True
    elif mutation == "status":
        freeze["status"] = "unfrozen"
    elif mutation == "binding_key":
        freeze["bindings"]["unexpected"] = "stale"
        freeze["bindings_sha256"] = _canonical_sha256(freeze["bindings"])
    elif mutation == "binding_value":
        freeze["bindings"]["delivery_policy_version"] = "stale-policy"
        freeze["bindings_sha256"] = _canonical_sha256(freeze["bindings"])
    elif mutation == "bindings_sha256":
        freeze["bindings_sha256"] = "f" * 64
    else:
        freeze["artifact_sha256"] = "f" * 64
    paths["freeze"].write_text(json.dumps(freeze), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        cli.main(
            [
                "--source-panel",
                str(paths["source"]),
                "--successful-pilot-results",
                str(paths["results"]),
                "--successful-pilot-freeze",
                str(paths["freeze"]),
                "--qwen-predictions",
                str(paths["qwen"]),
                "--output",
                str(tmp_path / "panel.json"),
            ]
        )


def test_cli_rejects_output_outside_canonical_private_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    root = tmp_path / "private-data"
    root.mkdir()
    monkeypatch.setattr(cli, "CANONICAL_DATA_ROOT", root)
    paths = _write_fixture_inputs(tmp_path)

    with pytest.raises(ValueError, match="canonical private data root"):
        cli.main(
            [
                "--source-panel",
                str(paths["source"]),
                "--successful-pilot-results",
                str(paths["results"]),
                "--successful-pilot-freeze",
                str(paths["freeze"]),
                "--qwen-predictions",
                str(paths["qwen"]),
                "--output",
                str(tmp_path / "outside" / "panel.json"),
            ]
        )


def test_cli_rejects_symlink_escape_from_canonical_private_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    root = tmp_path / "private-data"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(cli, "CANONICAL_DATA_ROOT", root)
    paths = _write_fixture_inputs(tmp_path)

    with pytest.raises(ValueError, match="canonical private data root"):
        cli.main(
            [
                "--source-panel",
                str(paths["source"]),
                "--successful-pilot-results",
                str(paths["results"]),
                "--successful-pilot-freeze",
                str(paths["freeze"]),
                "--qwen-predictions",
                str(paths["qwen"]),
                "--output",
                str(root / "escape" / "panel.json"),
            ]
        )
