"""Tests for the local memory store and configuration (CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernel_memory.store import MemoryConfig, MemoryStore
from tests.memory_helpers import make_attempt, make_task_result


def test_off_mode_is_inert():
    config = MemoryConfig(mode="off")
    config.validate()
    assert config.is_off()
    assert not config.retrieval_enabled()
    assert not config.write_enabled()


def test_collect_mode_writes_but_does_not_retrieve():
    config = MemoryConfig(mode="collect")
    config.validate()
    assert config.write_enabled()
    assert not config.retrieval_enabled()


def test_frozen_mode_never_writes_to_snapshot(tmp_path: Path):
    snapshot = tmp_path / "snap"
    store_dir = tmp_path / "work"
    config = MemoryConfig(
        mode="frozen", store_dir=str(store_dir), snapshot_dir=str(snapshot)
    )
    config.validate()
    assert config.retrieval_dir() == snapshot.resolve()
    assert config.write_dir() == store_dir.resolve()
    assert config.write_dir() != config.retrieval_dir()


def test_frozen_same_dir_still_separates_write(tmp_path: Path):
    shared = tmp_path / "shared"
    config = MemoryConfig(
        mode="frozen", store_dir=str(shared), snapshot_dir=str(shared)
    )
    config.validate()
    assert config.retrieval_dir() == shared.resolve()
    assert config.write_dir() != shared.resolve()


def test_frozen_requires_snapshot_dir():
    config = MemoryConfig(mode="frozen", snapshot_dir=None)
    with pytest.raises(ValueError):
        config.validate()


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        MemoryConfig(mode="bogus").validate()


def test_store_round_trip(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem")
    attempt = make_attempt()
    result = make_task_result()
    store.append_attempts([attempt])
    store.append_task_results([result])

    assert store.read_attempts() == [attempt]
    assert store.read_task_results() == [result]
    counts = store.counts()
    assert counts["attempts"] == 1
    assert counts["task_results"] == 1


def test_memory_version_is_content_addressed(tmp_path: Path):
    directory = tmp_path / "mem"
    store = MemoryStore(directory)
    empty_version = store.memory_version()
    store.append_attempts([make_attempt()])
    after = store.memory_version()
    assert after != empty_version

    # A fresh reader sees the same version for identical content.
    reader = MemoryStore(directory, read_only=True)
    assert reader.memory_version() == after

    store.append_attempts([make_attempt(task_id=2)])
    assert store.memory_version() != after


def test_read_is_robust_to_malformed_lines(tmp_path: Path):
    directory = tmp_path / "mem"
    directory.mkdir(parents=True)
    path = directory / "attempts.jsonl"
    good = json.dumps(make_attempt())
    path.write_text("not-json\n" + good + "\n", encoding="utf-8")
    store = MemoryStore(directory)
    records = store.read_attempts()
    assert len(records) == 1
    assert records[0]["kind"] == "attempt"


def test_freeze_copies_without_mutating_source(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem")
    store.append_attempts([make_attempt()])
    source_version = store.memory_version()

    dest = tmp_path / "frozen"
    info = store.freeze(dest, label="unit")
    assert info["snapshot_label"] == "unit"
    assert info["memory_version"] == source_version

    frozen = MemoryStore(dest, read_only=True)
    assert frozen.read_version_info()["snapshot_label"] == "unit"
    assert len(frozen.read_attempts()) == 1
    # Source is unchanged.
    assert store.memory_version() == source_version


def test_read_only_store_rejects_writes(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem", read_only=True)
    with pytest.raises(PermissionError):
        store.append_attempts([make_attempt()])
