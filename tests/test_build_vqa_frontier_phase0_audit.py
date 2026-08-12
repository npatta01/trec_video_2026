from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from trec_video_2026 import vqa_frontier_phase0_audit as audit_module


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_vqa_frontier_phase0_audit.py"


def test_cli_validate_stdout_is_safe_and_reports_only_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert SCRIPT.is_file()
    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    root = tmp_path / "audit"
    root.mkdir()
    (root / "packet-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "audit_id": "frontier-phase0-native-evidence-audit-v1",
                "run_id": "run",
                "panel_sha256": "a" * 64,
                "phase": "pilot",
                "row_count": 0,
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "blind-manifest.json").write_text(
        json.dumps({"schema_version": 1, "rows": []}), encoding="utf-8"
    )
    (root / "aware-manifest.json").write_text(
        json.dumps({"schema_version": 1, "rows": []}), encoding="utf-8"
    )
    (root / "private-manifest.json").write_text(
        json.dumps({"schema_version": 1, "rows": []}), encoding="utf-8"
    )
    (root / "map-verdicts.jsonl").write_text("", encoding="utf-8")
    (root / "answer-verdicts.jsonl").write_text("", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("phase0_audit_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    assert cli.main(["validate", "--audit", str(root)]) == 0
    stdout = capsys.readouterr().out
    assert '"status":"valid"' in stdout
    assert str(root) not in stdout
    assert "sentinel" not in stdout.lower()
