"""Behavioral contract for the Nemotron 3 Nano Omni paired pilot."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from trec_video_2026.vqa_benchmark import MediaMetadata, MediaStreamSummary
from trec_video_2026.vqa_experiments import nemotron3_omni_pilot
from trec_video_2026.vqa_experiments.nemotron3_omni_pilot import (
    EXPANDED_ROWS,
    MODEL_ID,
    TARGET_ROWS,
    build_canary_contract,
    build_chat_request,
    build_expanded_contract,
    build_label_free_contract,
    parse_short_answer,
    run_probe,
    validate_contract_inputs,
)


def _write_panel_and_media(
    tmp_path: Path, *, rows: tuple[tuple[str, str, str], ...] = TARGET_ROWS
) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    examples = []
    for index, (video_id, category, role) in enumerate(rows):
        media_stem = f"vqa/private/{video_id}"
        media_path = data_root / f"{media_stem}.mkv"
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(f"media-{index}".encode())
        examples.append(
            {
                "video_id": video_id,
                "category": category,
                "question": f"Question {index}?",
                "media": media_stem,
                "annotation": {
                    "reference": f"secret reference {index}",
                    "distractors": ["secret distractor"],
                },
            }
        )
    examples.append(
        {
            "video_id": "excluded",
            "category": "Dialogue and speech",
            "question": "Excluded?",
            "media": "vqa/private/excluded",
            "annotation": {"reference": "secret"},
        }
    )
    panel_path = tmp_path / "failure-panel.json"
    panel_path.write_text(json.dumps({"examples": examples}), encoding="utf-8")
    return panel_path, data_root


def _media_metadata(
    *,
    duration_seconds: float = 2.0,
    video_codec: str = "h264",
    audio_codec: str = "aac",
) -> MediaMetadata:
    return MediaMetadata(
        duration_seconds=duration_seconds,
        has_video=True,
        has_audio=True,
        stream_types=("video", "audio"),
        streams=(
            MediaStreamSummary(codec_type="video", codec_name=video_codec),
            MediaStreamSummary(codec_type="audio", codec_name=audio_codec),
        ),
    )


def _expanded_normalized_inputs(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    panel_path, data_root = _write_panel_and_media(tmp_path, rows=EXPANDED_ROWS)
    source_contract = build_expanded_contract(
        panel_path=panel_path, data_root=data_root
    )
    normalized_root = tmp_path / "normalized"
    normalized_root.mkdir()
    for index, (video_id, _category, _role) in enumerate(EXPANDED_ROWS):
        (normalized_root / f"{video_id}.mp4").write_bytes(
            f"normalized-{index}".encode()
        )
    return source_contract, data_root, normalized_root


def _decode_succeeds(
    *_args: object, **_kwargs: object
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


@pytest.mark.parametrize(
    ("derived_metadata", "message"),
    [
        (None, "media probe failed"),
        (_media_metadata(video_codec="vp9"), "normalized video codec mismatch"),
        (_media_metadata(audio_codec="mp3"), "normalized audio codec mismatch"),
        (_media_metadata(duration_seconds=2.6), "normalized duration drift"),
    ],
)
def test_expanded_normalization_rejects_invalid_media_before_inference(
    tmp_path: Path, derived_metadata: MediaMetadata | None, message: str
) -> None:
    """Catches corrupt, incompatible, or duration-drifted expanded media."""
    source_contract, data_root, normalized_root = _expanded_normalized_inputs(tmp_path)

    def probe(path: Path) -> MediaMetadata:
        if path.suffix == ".mkv":
            return _media_metadata()
        if derived_metadata is None:
            raise subprocess.CalledProcessError(1, ["ffprobe", str(path)])
        return derived_metadata

    with pytest.raises(ValueError, match=message):
        nemotron3_omni_pilot.build_normalized_contract(
            source_contract=source_contract,
            source_media_root=data_root,
            normalized_media_root=normalized_root,
            mode="expanded-pilot",
            media_probe=probe,
            decode_runner=_decode_succeeds,
        )


def test_expanded_probe_rejects_undecodable_media_before_any_request(
    tmp_path: Path,
) -> None:
    """Catches a derived file that probes cleanly but fails stream decode."""
    source_contract, data_root, normalized_root = _expanded_normalized_inputs(tmp_path)
    contract = nemotron3_omni_pilot.build_normalized_contract(
        source_contract=source_contract,
        source_media_root=data_root,
        normalized_media_root=normalized_root,
        mode="expanded-pilot",
        media_probe=lambda _path: _media_metadata(),
        decode_runner=_decode_succeeds,
    )
    requested = False

    def request(_payload: dict[str, object]) -> dict[str, object]:
        nonlocal requested
        requested = True
        return {}

    def decode_fails(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, ["ffmpeg"])

    with pytest.raises(ValueError, match="media decode failed"):
        run_probe(
            contract=contract,
            media_root=normalized_root,
            output_dir=tmp_path / "output",
            run_id="nemotron3-omni-expanded12-20260730-c1",
            request=request,
            media_probe=lambda _path: _media_metadata(),
            decode_runner=decode_fails,
        )
    assert requested is False


def test_expanded_contract_freezes_the_exact_failure_panel_order_without_labels(
    tmp_path: Path,
) -> None:
    """Catches expanded-panel row/category/role drift or label leakage."""
    panel_path, data_root = _write_panel_and_media(tmp_path, rows=EXPANDED_ROWS)

    contract = build_expanded_contract(panel_path=panel_path, data_root=data_root)

    assert EXPANDED_ROWS == (
        ("NN7G28sBOlY", "Dialogue and speech", "target"),
        ("NUmb2P3Bzu8", "Dialogue and speech", "target"),
        ("IRTTtnSRTyY", "Non-speech sound and music", "target"),
        ("IWSPAfCKtV8", "Non-speech sound and music", "target"),
        ("MdMKHhFFqJQ", "Non-speech sound and music", "target"),
        ("OzNdzgQi1CA", "Non-speech sound and music", "target"),
        ("CH8Dp7EgW4I", "Temporal and action", "target"),
        ("EtNOkUJd5w8", "Temporal and action", "target"),
        ("Gp4LVTn3YbY", "Temporal and action", "target"),
        ("KgOvCPOX-Ms", "Temporal and action", "target"),
        ("EMI_anpHII4", "Visual controls", "control"),
        ("LusRurRrmf0", "Visual controls", "control"),
    )
    assert contract["mode"] == "expanded-pilot"
    assert contract["panel_id"] == "nemotron3-omni-expanded-12"
    assert [
        (row["video_id"], row["category"], row["role"]) for row in contract["examples"]
    ] == list(EXPANDED_ROWS)
    assert "annotation" not in json.dumps(contract).lower()


def test_expanded_normalized_contract_requires_twelve_inputs_and_runs_24_adjacent_pairs(
    tmp_path: Path,
) -> None:
    """Catches an expanded run with a wrong input denominator or pair order."""
    panel_path, data_root = _write_panel_and_media(tmp_path, rows=EXPANDED_ROWS)
    source_contract = build_expanded_contract(
        panel_path=panel_path, data_root=data_root
    )
    source_contract_path = tmp_path / "source-contract.json"
    source_contract_path.write_text(json.dumps(source_contract), encoding="utf-8")
    normalized_root = tmp_path / "normalized"
    normalized_root.mkdir()
    for index, (video_id, _category, _role) in enumerate(EXPANDED_ROWS):
        (normalized_root / f"{video_id}.mp4").write_bytes(
            f"normalized-{index}".encode()
        )
    normalized_contract = nemotron3_omni_pilot.build_normalized_contract(
        source_contract=source_contract,
        source_media_root=data_root,
        normalized_media_root=normalized_root,
        mode="expanded-pilot",
        media_probe=lambda _path: _media_metadata(),
        decode_runner=_decode_succeeds,
    )
    marker_path = tmp_path / "INPUTS_COMPLETE.json"
    marker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_count": 13,
                "contract_sha256": __import__("hashlib")
                .sha256(source_contract_path.read_bytes())
                .hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert validate_contract_inputs(
        contract_path=source_contract_path,
        media_root=data_root,
        marker_path=marker_path,
    ) == {"mode": "expanded-pilot", "media_files": 12, "requests": 24}
    calls: list[dict[str, object]] = []

    def request(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "Short answer."}}
            ]
        }

    output_dir = tmp_path / "output"
    summary = run_probe(
        contract=normalized_contract,
        media_root=normalized_root,
        output_dir=output_dir,
        run_id="nemotron3-omni-expanded12-20260730-c1",
        request=request,
        media_probe=lambda _path: _media_metadata(),
        decode_runner=_decode_succeeds,
    )
    assert summary["completion"] == {"completed": 24, "expected": 24}
    assert [
        call["mm_processor_kwargs"]["use_audio_in_video"]  # type: ignore[index]
        for call in calls
    ] == [False, True] * 12
    records = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert [(record["pair_index"], record["condition_id"]) for record in records] == [
        (pair_index, condition_id)
        for pair_index in range(12)
        for condition_id in ("audio-off", "audio-on")
    ]


def test_expanded_contract_cli_writes_the_label_free_twelve_row_source_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a CLI that silently builds the historical six-row contract."""
    panel_path, data_root = _write_panel_and_media(tmp_path, rows=EXPANDED_ROWS)

    exit_code = nemotron3_omni_pilot.main(
        [
            "build-expanded-contract",
            "--panel",
            str(panel_path),
            "--data-root",
            str(data_root),
        ]
    )

    assert exit_code == 0
    contract = json.loads(capsys.readouterr().out)
    assert contract["mode"] == "expanded-pilot"
    assert len(contract["examples"]) == 12


def test_contract_selects_exact_rows_and_removes_every_label(tmp_path: Path) -> None:
    """Catches target drift or annotation/reference leakage into inference."""
    panel_path, data_root = _write_panel_and_media(tmp_path)

    contract = build_label_free_contract(panel_path=panel_path, data_root=data_root)

    assert [row["video_id"] for row in contract["examples"]] == [
        row[0] for row in TARGET_ROWS
    ]
    assert [row["role"] for row in contract["examples"]] == [
        row[2] for row in TARGET_ROWS
    ]
    assert contract["conditions"] == [
        {"condition_id": "audio-off", "audio_enabled": False},
        {"condition_id": "audio-on", "audio_enabled": True},
    ]
    encoded = json.dumps(contract).lower()
    for forbidden in ("annotation", "reference", "distractor", "correct_answer"):
        assert forbidden not in encoded
    assert all(len(row["media_sha256"]) == 64 for row in contract["examples"])
    assert all(row["media_path"].endswith(".mkv") for row in contract["examples"])


def test_normalized_contract_preserves_frozen_rows_and_records_both_media_hashes(
    tmp_path: Path,
) -> None:
    """Catches row drift, question drift, or unverified source/derived media."""
    panel_path, data_root = _write_panel_and_media(tmp_path)
    source_contract = build_label_free_contract(
        panel_path=panel_path, data_root=data_root
    )
    normalized_root = tmp_path / "normalized"
    normalized_root.mkdir()
    for index, (video_id, _category, _role) in enumerate(TARGET_ROWS):
        (normalized_root / f"{video_id}.mp4").write_bytes(
            f"normalized-{index}".encode()
        )

    contract = nemotron3_omni_pilot.build_normalized_contract(
        source_contract=source_contract,
        source_media_root=data_root,
        normalized_media_root=normalized_root,
        mode="pilot",
    )

    assert contract["mode"] == "pilot"
    assert contract["conditions"] == [
        {"condition_id": "audio-off", "audio_enabled": False},
        {"condition_id": "audio-on", "audio_enabled": True},
    ]
    assert [
        (row["video_id"], row["category"], row["role"], row["question"])
        for row in contract["examples"]
    ] == [
        (
            "IRTTtnSRTyY",
            "Non-speech sound and music",
            "target",
            "Question 0?",
        ),
        (
            "IWSPAfCKtV8",
            "Non-speech sound and music",
            "target",
            "Question 1?",
        ),
        (
            "MdMKHhFFqJQ",
            "Non-speech sound and music",
            "target",
            "Question 2?",
        ),
        (
            "OzNdzgQi1CA",
            "Non-speech sound and music",
            "target",
            "Question 3?",
        ),
        ("EMI_anpHII4", "Visual controls", "control", "Question 4?"),
        ("LusRurRrmf0", "Visual controls", "control", "Question 5?"),
    ]
    for source, derived in zip(
        source_contract["examples"], contract["examples"], strict=True
    ):
        assert derived["source_media_path"] == source["media_path"]
        assert derived["source_media_sha256"] == source["media_sha256"]
        assert derived["media_path"] == f"{source['video_id']}.mp4"
        assert len(derived["derived_media_sha256"]) == 64
        assert "media_sha256" not in derived

    (normalized_root / "EMI_anpHII4.mp4").unlink()
    with pytest.raises(ValueError, match="normalized media is missing: EMI_anpHII4"):
        nemotron3_omni_pilot.build_normalized_contract(
            source_contract=source_contract,
            source_media_root=data_root,
            normalized_media_root=normalized_root,
            mode="pilot",
        )


def test_codec_sentinel_runs_exact_three_audio_on_native_video_requests(
    tmp_path: Path,
) -> None:
    """Catches a sentinel that skips an AV1 row or bypasses the native runner."""
    panel_path, data_root = _write_panel_and_media(tmp_path)
    source_contract = build_label_free_contract(
        panel_path=panel_path, data_root=data_root
    )
    normalized_root = tmp_path / "normalized"
    normalized_root.mkdir()
    for index, (video_id, _category, _role) in enumerate(TARGET_ROWS):
        (normalized_root / f"{video_id}.mp4").write_bytes(
            f"normalized-{index}".encode()
        )
    contract = nemotron3_omni_pilot.build_normalized_contract(
        source_contract=source_contract,
        source_media_root=data_root,
        normalized_media_root=normalized_root,
        mode="codec-sentinel",
    )
    calls: list[dict[str, object]] = []

    def request(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "A short answer."},
                }
            ]
        }

    summary = run_probe(
        contract=contract,
        media_root=normalized_root,
        output_dir=tmp_path / "output",
        run_id="nemotron3-omni-codec-sentinel-20260730-b1",
        request=request,
    )

    assert [row["video_id"] for row in contract["examples"]] == [
        "IWSPAfCKtV8",
        "OzNdzgQi1CA",
        "EMI_anpHII4",
    ]
    assert contract["conditions"] == [
        {"condition_id": "audio-on", "audio_enabled": True}
    ]
    assert summary["mode"] == "codec-sentinel"
    assert summary["completion"] == {"completed": 3, "expected": 3}
    assert len(calls) == 3
    assert all(
        call["mm_processor_kwargs"] == {"use_audio_in_video": True} for call in calls
    )
    assert all(
        call["chat_template_kwargs"] == {"enable_thinking": False} for call in calls
    )


def test_normalized_contract_cli_writes_a_runnable_twelve_request_pilot(
    tmp_path: Path,
) -> None:
    """Catches a CLI that emits an incomplete or non-runnable pilot contract."""
    panel_path, data_root = _write_panel_and_media(tmp_path)
    source_contract = build_label_free_contract(
        panel_path=panel_path, data_root=data_root
    )
    source_contract_path = tmp_path / "source-contract.json"
    source_contract_path.write_text(json.dumps(source_contract), encoding="utf-8")
    normalized_root = tmp_path / "normalized"
    normalized_root.mkdir()
    for index, (video_id, _category, _role) in enumerate(TARGET_ROWS):
        (normalized_root / f"{video_id}.mp4").write_bytes(
            f"normalized-{index}".encode()
        )
    output_path = tmp_path / "normalized-contract.json"

    exit_code = nemotron3_omni_pilot.main(
        [
            "build-normalized-contract",
            "--source-contract",
            str(source_contract_path),
            "--source-media-root",
            str(data_root),
            "--normalized-media-root",
            str(normalized_root),
            "--mode",
            "pilot",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    contract = json.loads(output_path.read_text(encoding="utf-8"))
    calls = 0

    def request(_payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "Short answer."}}
            ]
        }

    summary = run_probe(
        contract=contract,
        media_root=normalized_root,
        output_dir=tmp_path / "output",
        run_id="nemotron3-omni-pilot-20260730-b1",
        request=request,
    )
    assert summary["completion"] == {"completed": 12, "expected": 12}
    assert calls == 12


@pytest.mark.parametrize("audio_enabled", [False, True])
def test_chat_request_uses_native_video_audio_toggle_without_thinking(
    audio_enabled: bool,
) -> None:
    """Catches separate audio requests, sampling, or hidden reasoning being enabled."""
    payload = build_chat_request(
        media_path=Path("/dstack/input/media/example.mkv"),
        question="What happens?",
        audio_enabled=audio_enabled,
    )

    assert payload == {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "file:///dstack/input/media/example.mkv"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "What happens?\nAnswer in at most 12 words. "
                            "Return only the answer, with no analysis or explanation."
                        ),
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32,
        "stream": False,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"use_audio_in_video": audio_enabled},
    }


def test_short_answer_parser_discards_protocol_envelope_and_rejects_reasoning() -> None:
    """Catches persistence of chain-of-thought or malformed/long completions."""
    assert (
        parse_short_answer(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "A bell rings."}}
                ]
            }
        )
        == "A bell rings."
    )

    with pytest.raises(ValueError, match="reasoning content is forbidden"):
        parse_short_answer(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "A bell rings.",
                            "reasoning_content": "I inferred this from the waveform.",
                        },
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="thinking markup is forbidden"):
        parse_short_answer(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "<think>hidden</think> answer"},
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="at most 12 words"):
        parse_short_answer(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "one two three four five six seven eight nine ten eleven twelve thirteen"
                        },
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="finish_reason must be stop"):
        parse_short_answer(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "A truncated answer"},
                    }
                ]
            }
        )


def test_canary_contract_and_foreground_probe_freeze_three_native_av_answers(
    tmp_path: Path,
) -> None:
    """Catches a synthetic canary that bypasses the real paired runner seam."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    media_paths = []
    for index, color in enumerate(("red", "green", "blue")):
        media_path = media_dir / f"canary-{index + 1}-{color}.mp4"
        media_path.write_bytes(f"public-av-{color}".encode())
        media_paths.append(media_path)
    contract = build_canary_contract(media_paths)
    calls: list[dict[str, object]] = []

    def request(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "A colored screen."}}
            ]
        }

    output_dir = tmp_path / "output"
    summary = run_probe(
        contract=contract,
        media_root=media_dir,
        output_dir=output_dir,
        run_id="nemotron3-omni-runner-canary-20260730-a1",
        request=request,
    )

    assert summary["mode"] == "canary"
    assert summary["completion"] == {"completed": 3, "expected": 3}
    assert summary["frozen"] is True
    assert summary["reference_answers_available"] is False
    assert len(calls) == 3
    assert all(
        call["mm_processor_kwargs"] == {"use_audio_in_video": True} for call in calls
    )
    records = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert len(records) == 3
    assert all(record["audio_enabled"] is True for record in records)
    assert all(
        set(record).isdisjoint({"response", "reasoning_content"}) for record in records
    )
    assert (output_dir / "contract.json").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["frozen"] is True
    assert manifest["completion"] == {"completed": 3, "expected": 3}


def test_pilot_probe_runs_adjacent_audio_off_on_pairs_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    """Catches condition drift, missing outputs, and post-freeze overwrite."""
    panel_path, data_root = _write_panel_and_media(tmp_path)
    contract = build_label_free_contract(panel_path=panel_path, data_root=data_root)
    calls: list[dict[str, object]] = []

    def request(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "Short answer."}}
            ]
        }

    output_dir = tmp_path / "output"
    summary = run_probe(
        contract=contract,
        media_root=data_root,
        output_dir=output_dir,
        run_id="nemotron3-omni-pilot-20260730-a1",
        request=request,
    )

    assert summary["mode"] == "pilot"
    assert summary["completion"] == {"completed": 12, "expected": 12}
    assert [
        call["mm_processor_kwargs"]["use_audio_in_video"]  # type: ignore[index]
        for call in calls
    ] == [False, True] * 6
    records = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert [(record["pair_index"], record["condition_id"]) for record in records] == [
        (pair_index, condition_id)
        for pair_index in range(6)
        for condition_id in ("audio-off", "audio-on")
    ]
    assert "reference" not in json.dumps(records).lower()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_probe(
            contract=contract,
            media_root=data_root,
            output_dir=output_dir,
            run_id="nemotron3-omni-pilot-20260730-a1",
            request=request,
        )


def test_pilot_rejects_changed_role_before_any_request(tmp_path: Path) -> None:
    """Catches a count-correct contract with changed panel semantics."""
    panel_path, data_root = _write_panel_and_media(tmp_path)
    contract = build_label_free_contract(panel_path=panel_path, data_root=data_root)
    contract["examples"][0]["role"] = "control"
    called = False

    def request(payload: dict[str, object]) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(ValueError, match="pilot row identity mismatch"):
        run_probe(
            contract=contract,
            media_root=data_root,
            output_dir=tmp_path / "output",
            run_id="nemotron3-omni-pilot-20260730-a1",
            request=request,
        )
    assert called is False


def test_probe_flushes_each_frozen_record_before_a_process_interrupt(
    tmp_path: Path,
) -> None:
    """Catches buffering that loses the completion denominator after interruption."""
    panel_path, data_root = _write_panel_and_media(tmp_path)
    contract = build_label_free_contract(panel_path=panel_path, data_root=data_root)
    call_count = 0

    def request(payload: dict[str, object]) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise KeyboardInterrupt
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "First answer."}}
            ]
        }

    output_dir = tmp_path / "output"
    with pytest.raises(KeyboardInterrupt):
        run_probe(
            contract=contract,
            media_root=data_root,
            output_dir=output_dir,
            run_id="nemotron3-omni-pilot-20260730-a1",
            request=request,
        )

    records = (output_dir / "predictions.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["status"] == "passed"


def test_pilot_input_marker_and_every_media_hash_are_verified(tmp_path: Path) -> None:
    """Catches an incomplete upload marker or changed media before model loading."""
    panel_path, data_root = _write_panel_and_media(tmp_path)
    contract = build_label_free_contract(panel_path=panel_path, data_root=data_root)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker_path = tmp_path / "INPUTS_COMPLETE.json"
    marker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_count": 7,
                "contract_sha256": __import__("hashlib")
                .sha256(contract_path.read_bytes())
                .hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert validate_contract_inputs(
        contract_path=contract_path,
        media_root=data_root,
        marker_path=marker_path,
    ) == {"mode": "pilot", "media_files": 6, "requests": 12}

    marker = json.loads(marker_path.read_text())
    marker["contract_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="input marker contract hash mismatch"):
        validate_contract_inputs(
            contract_path=contract_path,
            media_root=data_root,
            marker_path=marker_path,
        )
