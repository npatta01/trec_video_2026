from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.prepare_vqa_deadline_qwen_full_inputs import prepare_inputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_requests(path: Path, webm: Path, mkv: Path, mp4: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "query_id": 1,
                    "video_id": webm.name,
                    "question": "What happens?",
                    "availability": "available",
                    "source_sha256": _sha256(webm),
                    "frame_paths": ["private-frame.png"],
                },
                {
                    "query_id": 2,
                    "video_id": mkv.stem,
                    "question": "What is said?",
                    "availability": "available",
                    "source_sha256": _sha256(mkv),
                },
                {
                    "query_id": 3,
                    "video_id": mkv.stem,
                    "question": "How many?",
                    "availability": "available",
                    "source_sha256": _sha256(mkv),
                },
                {
                    "query_id": 4,
                    "video_id": mp4.stem,
                    "question": "Who appears?",
                    "availability": "available",
                    "source_sha256": _sha256(mp4),
                },
                {
                    "query_id": 5,
                    "video_id": "missing-id",
                    "question": "Where is it?",
                    "availability": "unavailable",
                    "source_sha256": "0" * 64,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_prepare_inputs_freezes_label_free_queries_and_unique_media(tmp_path: Path) -> None:
    webm_root = tmp_path / "videos"
    mkv_root = tmp_path / "videos_original"
    webm_root.mkdir()
    mkv_root.mkdir()
    webm = webm_root / "clip.webm"
    mkv = mkv_root / "youtube1234.mkv"
    mp4 = mkv_root / "recovered.mp4"
    webm.write_bytes(b"webm")
    mkv.write_bytes(b"mkv")
    mp4.write_bytes(b"mp4")
    requests = tmp_path / "requests.json"
    _write_requests(requests, webm, mkv, mp4)

    result = prepare_inputs(
        requests_path=requests,
        webm_root=webm_root,
        mkv_root=mkv_root,
        output_root=tmp_path / "output",
    )

    assert result == {
        "status": "inputs_frozen",
        "query_count": 5,
        "available_query_count": 4,
        "fallback_query_count": 1,
        "unique_media_count": 3,
    }
    rows = json.loads((tmp_path / "output/requests.json").read_text())
    assert [row["query_id"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["media_file"] == "media/clip.webm"
    assert rows[1]["media_file"] == "media/youtube1234.mkv"
    assert rows[3]["media_file"] == "media/recovered.mp4"
    assert rows[4]["media_file"] is None
    assert all("frame_paths" not in row for row in rows)
    contract = json.loads((tmp_path / "output/media-contract.json").read_text())
    assert [row["path"] for row in contract["media"]] == [
        "media/clip.webm",
        "media/recovered.mp4",
        "media/youtube1234.mkv",
    ]
    assert contract["media"][2]["query_count"] == 2
    assert (tmp_path / "output/inputs-freeze.json").is_file()
    assert (tmp_path / "output/SHA256SUMS").is_file()


def test_prepare_inputs_rejects_a_declared_media_hash_mismatch(tmp_path: Path) -> None:
    webm_root = tmp_path / "videos"
    mkv_root = tmp_path / "videos_original"
    webm_root.mkdir()
    mkv_root.mkdir()
    webm = webm_root / "clip.webm"
    webm.write_bytes(b"webm")
    requests = tmp_path / "requests.json"
    requests.write_text(
        json.dumps(
            [
                {
                    "query_id": 1,
                    "video_id": webm.name,
                    "question": "What happens?",
                    "availability": "available",
                    "source_sha256": "0" * 64,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_inputs(
            requests_path=requests,
            webm_root=webm_root,
            mkv_root=mkv_root,
            output_root=tmp_path / "output",
        )


def test_prepare_inputs_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    webm_root = tmp_path / "videos"
    mkv_root = tmp_path / "videos_original"
    webm_root.mkdir()
    mkv_root.mkdir()
    webm = webm_root / "clip.webm"
    webm.write_bytes(b"webm")
    requests = tmp_path / "requests.json"
    digest = _sha256(webm)
    requests.write_text(
        json.dumps(
            [
                {"query_id": 1, "video_id": webm.name, "question": "One?", "source_sha256": digest},
                {"query_id": 1, "video_id": webm.name, "question": "Two?", "source_sha256": digest},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="query IDs"):
        prepare_inputs(
            requests_path=requests,
            webm_root=webm_root,
            mkv_root=mkv_root,
            output_root=tmp_path / "output",
        )
