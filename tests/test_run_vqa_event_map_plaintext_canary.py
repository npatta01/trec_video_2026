from __future__ import annotations

import json
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_vqa_event_map_plaintext_canary.py"
SPEC = importlib.util.spec_from_file_location("run_plaintext_canary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
run_plaintext_canary = MODULE.run_plaintext_canary


def _map_response() -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": (
                        "E1|1.0|Opening shot shows a person entering the room.\n"
                        "E2|5.0|The person places an object on the table.\n"
                        "E3|9.0|The person leaves and the room becomes empty.\n"
                        "ANSWER|The person enters, places an object, then leaves."
                    )
                },
            }
        ]
    }


def _selector_response() -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": (
                        "SELECT|E1,E2\n"
                        "ANSWER|The person places an object on the table."
                    )
                },
            }
        ]
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    requests = tmp_path / "requests.jsonl"
    rows = []
    for video_id in ("video-a", "video-b", "video-c", "video-d"):
        base = {
            "video_id": video_id,
            "question": "What happens?",
            "question_stratum": "other",
            "duration_seconds": 10.0,
            "media_path": f"media/{video_id}.mkv",
            "media_sha256": "a" * 64,
            "condition_id": "av-map",
        }
        rows.append(base)
        rows.append({**base, "condition_id": "v-map"})
    requests.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    normalized_root = tmp_path / "normalized"
    normalized_root.mkdir()
    controller = tmp_path / "controller.json"
    controller.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "video_id": video_id,
                        "derived_duration_seconds": 10.0,
                        "derived_media_sha256": "b" * 64,
                    }
                    for video_id in ("video-a", "video-b", "video-c", "video-d")
                ]
            }
        ),
        encoding="utf-8",
    )
    for video_id in ("video-a", "video-b", "video-c", "video-d"):
        (normalized_root / f"{video_id}.mp4").write_bytes(video_id.encode())
    return requests, normalized_root, controller


def test_canary_runs_four_av_maps_then_four_text_selectors_without_retry(
    tmp_path: Path,
) -> None:
    requests, normalized_root, controller = _write_inputs(tmp_path)
    calls: list[dict[str, object]] = []

    def post(_url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        if len(calls) <= 4:
            return _map_response()
        return _selector_response()

    output = tmp_path / "output.jsonl"
    rows = run_plaintext_canary(
        requests_path=requests,
        normalized_root=normalized_root,
        normalized_controller_path=controller,
        output_path=output,
        server_url="http://127.0.0.1:8000/v1/chat/completions",
        post=post,
    )

    assert len(calls) == 8
    assert all("response_format" not in call for call in calls)
    assert all(call["chat_template_kwargs"] == {"enable_thinking": False} for call in calls)
    assert all(call["max_tokens"] == (256 if index < 4 else 64) for index, call in enumerate(calls))
    assert all(call["mm_processor_kwargs"] == {"use_audio_in_video": True} for call in calls[:4])
    assert all("mm_processor_kwargs" not in call for call in calls[4:])
    assert all(row["status"] == "passed" for row in rows)
    assert all(row["selector_input_sha256"] == row["map_text_sha256"] for row in rows)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4


def test_canary_fails_closed_and_does_not_repair_bad_map(tmp_path: Path) -> None:
    requests, normalized_root, controller = _write_inputs(tmp_path)
    calls = 0

    def post(_url: str, _payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "ANSWER|answer"},
                }
            ]
        }

    rows = run_plaintext_canary(
        requests_path=requests,
        normalized_root=normalized_root,
        normalized_controller_path=controller,
        output_path=tmp_path / "output.jsonl",
        server_url="http://127.0.0.1:8000/v1/chat/completions",
        post=post,
    )

    assert calls == 4
    assert all(row["status"] == "failed" for row in rows)
    assert all(row["error_type"] == "map_parse_failed" for row in rows)
    assert all(row["failure_detail"] == "map_event_count_invalid" for row in rows)
    assert all("selected_event_ids" not in row for row in rows)
