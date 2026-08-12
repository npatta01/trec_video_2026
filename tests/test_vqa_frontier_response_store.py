from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from trec_video_2026.vqa_frontier_response_store import (
    RawResponseConflictError,
    parse_prompt_json_response,
    preserve_raw_response,
    verify_raw_response,
)


def test_parse_removes_only_one_exact_json_fence() -> None:
    assert parse_prompt_json_response('```json\n{"ok": true}\n```') == {"ok": True}
    with pytest.raises(json.JSONDecodeError):
        parse_prompt_json_response('```JSON\n{"ok": true}\n```')


def test_preserve_raw_response_never_overwrites_conflicting_billable_body(
    tmp_path: Path,
) -> None:
    preserve_raw_response(
        tmp_path,
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        raw_response='{"a":1}',
    )
    with pytest.raises(RawResponseConflictError):
        preserve_raw_response(
            tmp_path,
            run_id="r",
            condition_id="neutral_map",
            video_id="v",
            raw_response='{"a":2}',
        )
    assert (tmp_path / "raw-conflicts" / "neutral_map" / "v").is_dir()
    primary = json.loads(
        (tmp_path / "raw" / "neutral_map" / "v.json").read_text(encoding="utf-8")
    )
    assert primary["raw_response"] == '{"a":1}'
    conflicts = list((tmp_path / "raw-conflicts" / "neutral_map" / "v").glob("*.json"))
    assert len(conflicts) == 1
    assert json.loads(conflicts[0].read_text(encoding="utf-8"))["raw_response"] == (
        '{"a":2}'
    )


def test_preserve_raw_response_is_idempotent_for_identical_body(tmp_path: Path) -> None:
    first = preserve_raw_response(
        tmp_path,
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        raw_response='{"a":1}',
    )
    target = tmp_path / "raw" / "neutral_map" / "v.json"
    before = target.read_bytes()

    second = preserve_raw_response(
        tmp_path,
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        raw_response='{"a":1}',
    )

    assert first == second
    assert target.read_bytes() == before
    assert first["raw_response_sha256"] == (
        "3dc4eef510cec9eda1ca361042e8152bd62c44288af2d9cc465cc217c381facd"
    )


@pytest.mark.parametrize(
    ("condition_id", "video_id"),
    [
        ("../escape", "v"),
        ("nested/condition", "v"),
        ("neutral_map", "../escape"),
        ("neutral_map", "nested/video"),
        ("", "v"),
    ],
)
def test_preserve_raw_response_rejects_unsafe_path_segments(
    tmp_path: Path, condition_id: str, video_id: str
) -> None:
    with pytest.raises(ValueError, match="safe path segment"):
        preserve_raw_response(
            tmp_path,
            run_id="r",
            condition_id=condition_id,
            video_id=video_id,
            raw_response="body",
        )


def test_preserve_raw_response_rejects_symlinked_condition_directory(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (raw / "neutral_map").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked"):
        preserve_raw_response(
            tmp_path,
            run_id="r",
            condition_id="neutral_map",
            video_id="v",
            raw_response="body",
        )
    assert list(outside.iterdir()) == []


def test_preserve_raw_response_rejects_symlinked_conflict_directory(
    tmp_path: Path,
) -> None:
    preserve_raw_response(
        tmp_path,
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        raw_response="first",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "raw-conflicts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked"):
        preserve_raw_response(
            tmp_path,
            run_id="r",
            condition_id="neutral_map",
            video_id="v",
            raw_response="second",
        )
    assert list(outside.iterdir()) == []


def test_verify_raw_response_accepts_intact_record(tmp_path: Path) -> None:
    raw_response = "billable response"
    preserve_raw_response(
        tmp_path,
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        raw_response=raw_response,
    )

    verify_raw_response(
        tmp_path / "raw" / "neutral_map" / "v.json",
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        response_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
    )


def test_verify_raw_response_rejects_tampered_body(tmp_path: Path) -> None:
    preserve_raw_response(
        tmp_path,
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        raw_response="original",
    )
    target = tmp_path / "raw" / "neutral_map" / "v.json"
    record = json.loads(target.read_text(encoding="utf-8"))
    record["raw_response"] = "tampered"
    target.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="identity changed"):
        verify_raw_response(
            target,
            run_id="r",
            condition_id="neutral_map",
            video_id="v",
            response_sha256=hashlib.sha256(b"original").hexdigest(),
        )


def test_verify_raw_response_rejects_wrong_transport_hash(tmp_path: Path) -> None:
    preserve_raw_response(
        tmp_path,
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        raw_response="original",
    )

    with pytest.raises(ValueError, match="hash changed"):
        verify_raw_response(
            tmp_path / "raw" / "neutral_map" / "v.json",
            run_id="r",
            condition_id="neutral_map",
            video_id="v",
            response_sha256=hashlib.sha256(b"different").hexdigest(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_schema_version",
        "wrong_schema_version",
        "boolean_schema_version",
        "extra_field",
    ],
)
def test_verify_raw_response_requires_exact_typed_record_shape(
    tmp_path: Path, mutation: str
) -> None:
    raw_response = "original"
    preserve_raw_response(
        tmp_path,
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        raw_response=raw_response,
    )
    target = tmp_path / "raw" / "neutral_map" / "v.json"
    record = json.loads(target.read_text(encoding="utf-8"))
    if mutation == "missing_schema_version":
        del record["schema_version"]
    elif mutation == "wrong_schema_version":
        record["schema_version"] = 2
    elif mutation == "boolean_schema_version":
        record["schema_version"] = True
    else:
        record["extra_field"] = "unexpected"
    target.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="record shape"):
        verify_raw_response(
            target,
            run_id="r",
            condition_id="neutral_map",
            video_id="v",
            response_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        )


def test_verify_raw_response_rejects_symlinked_parent_directory(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    raw_response = "original"
    preserve_raw_response(
        actual,
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        raw_response=raw_response,
    )
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        verify_raw_response(
            alias / "raw" / "neutral_map" / "v.json",
            run_id="r",
            condition_id="neutral_map",
            video_id="v",
            response_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        )


def test_invalid_utf8_response_does_not_create_a_record(tmp_path: Path) -> None:
    with pytest.raises(UnicodeDecodeError):
        preserve_raw_response(
            tmp_path,
            run_id="r",
            condition_id="neutral_map",
            video_id="v",
            raw_response=b"invalid\xff",
        )

    assert not (tmp_path / "raw").exists()


def test_invalid_utf8_response_does_not_overwrite_a_record(tmp_path: Path) -> None:
    preserve_raw_response(
        tmp_path,
        run_id="r",
        condition_id="neutral_map",
        video_id="v",
        raw_response="original",
    )
    target = tmp_path / "raw" / "neutral_map" / "v.json"
    before = target.read_bytes()

    with pytest.raises(UnicodeDecodeError):
        preserve_raw_response(
            tmp_path,
            run_id="r",
            condition_id="neutral_map",
            video_id="v",
            raw_response=b"invalid\xff",
        )

    assert target.read_bytes() == before


def test_shared_install_race_is_idempotent_with_descriptor_relative_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = threading.Barrier(2, timeout=5)
    original_link = __import__(
        "trec_video_2026.vqa_frontier_response_store", fromlist=["os"]
    ).os.link

    def synchronized_link(
        source: object, destination: object, *args: object, **kwargs: object
    ) -> None:
        assert kwargs["src_dir_fd"] is not None
        assert kwargs["dst_dir_fd"] is not None
        barrier.wait()
        original_link(source, destination, *args, **kwargs)

    response_store_module = __import__(
        "trec_video_2026.vqa_frontier_response_store", fromlist=["os"]
    )
    monkeypatch.setattr(response_store_module.os, "link", synchronized_link)

    def install() -> object:
        return preserve_raw_response(
            tmp_path,
            run_id="r",
            condition_id="neutral_map",
            video_id="v",
            raw_response="same",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: install(), range(2)))

    assert results[0] == results[1]
    assert len(list((tmp_path / "raw" / "neutral_map").glob("*.json"))) == 1
