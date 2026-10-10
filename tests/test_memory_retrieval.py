"""Tests for bounded measured-memory retrieval (CPU-only)."""

from __future__ import annotations

from pathlib import Path

from kernel_memory.retrieval import build_experience_context, retrieve_measured
from kernel_memory.schema import task_result_record
from kernel_memory.store import MemoryStore
from kernel_memory.summary import build_summaries
from tests.memory_helpers import compat, make_attempt, task_features


def _stored_task_result(task_id: int, *, speedup: float | None = 5.4, correct: bool = True):
    return task_result_record(
        run_id="r",
        task_id=task_id,
        task_name=f"{task_id}_task.py",
        task_source_sha256="sha",
        task_features=task_features(),
        compatibility=compat(),
        evaluation_status="passed" if correct else "incorrect",
        correctness_status="project_passed" if correct else "project_failed",
        baseline_latency_ms=10.0,
        candidate_latency_ms=(10.0 / speedup) if speedup else None,
        task_speedup=speedup,
        timing_methodology={"method": "do_bench"},
        route="kernelagent",
        failure_category="none" if correct else "correctness_failed",
        memory_version_used="v0",
        memory_mode_used="collect",
    )


def _seeded_store(tmp_path: Path, attempts):
    store = MemoryStore(tmp_path / "mem")
    store.append_attempts(attempts)
    store.append_summaries(build_summaries(attempts))
    return store


def test_retrieval_filters_incompatible_compatibility(tmp_path: Path):
    other_compat = compat(gpu_name="NVIDIA H100")
    attempts = [make_attempt(task_id=7, compatibility=other_compat)]
    store = _seeded_store(tmp_path, attempts)

    items, stats = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="memory",
        current_task_id=99,
    )
    assert items == []
    assert stats["excluded_incompatible"] >= 1


def test_foreign_compat_allowed_as_hypothesis(tmp_path: Path):
    other_compat = compat(gpu_name="NVIDIA H100")
    store = _seeded_store(tmp_path, [make_attempt(task_id=7, compatibility=other_compat)])
    items, stats = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="memory",
        current_task_id=99,
        allow_foreign_compat=True,
    )
    assert items
    assert stats["foreign_used"] is True
    assert "FOREIGN" in items[0]["caveat"]


def test_anti_leakage_excludes_current_task(tmp_path: Path):
    store = _seeded_store(tmp_path, [make_attempt(task_id=42)])
    items, stats = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="memory",
        current_task_id=42,
    )
    assert items == []
    assert stats["excluded_same_task"] >= 1


def test_retrieval_is_bounded_and_ranked(tmp_path: Path):
    attempts = [
        make_attempt(task_id=1, bottleneck="memory", action_summary="a", record_id="r1"),
        make_attempt(task_id=2, bottleneck="compute", action_summary="b", record_id="r2"),
        make_attempt(task_id=3, bottleneck="memory", action_summary="c", record_id="r3"),
    ]
    store = _seeded_store(tmp_path, attempts)
    items, _ = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="memory",
        current_task_id=99,
        max_items=2,
    )
    assert len(items) <= 2
    # Bottleneck matches should outrank non-matches.
    assert items[0]["bottleneck_match"] is True


def test_failures_can_be_excluded(tmp_path: Path):
    store = _seeded_store(
        tmp_path,
        [make_attempt(task_id=1, passed=False, local_speedup=None, failure_category="verification_failed")],
    )
    items, _ = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="memory",
        current_task_id=99,
        include_failures=False,
    )
    assert all(item.get("measurement_available") for item in items)


def test_experience_context_is_labeled_and_bounded(tmp_path: Path):
    store = _seeded_store(tmp_path, [make_attempt(task_id=1)])
    items, _ = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="memory",
        current_task_id=99,
        max_items=3,
    )
    context = build_experience_context(
        items, memory_version="v1", frozen=True, compatibility=compat(), budget_chars=5000
    )
    assert "MEASURED CROSS-TASK EXPERIENCE" in context
    assert "static RAG" in context
    assert "evidence_ids" in context

    tiny = build_experience_context(
        items, memory_version="v1", frozen=False, compatibility=compat(), budget_chars=100
    )
    assert "truncated" in tiny


def test_empty_retrieval_returns_empty_context(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem")
    items, _ = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="memory",
        current_task_id=99,
    )
    assert items == []
    assert build_experience_context(
        items, memory_version=None, frozen=False, compatibility=compat()
    ) == ""


def test_task_result_retrieval_includes_measured_speedup(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem")
    store.append_task_results([_stored_task_result(7, speedup=5.4)])
    items, _ = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="unknown",
        current_task_id=99,
    )
    task_items = [i for i in items if i["kind"] == "task_result"]
    assert task_items
    assert "5.400x" in task_items[0]["observed"]
    assert task_items[0]["measurement_available"] is True


def test_task_result_retrieval_excludes_same_task(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem")
    store.append_task_results([_stored_task_result(7, speedup=5.4)])
    items, _ = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="unknown",
        current_task_id=7,
    )
    assert all(i["kind"] != "task_result" for i in items)


def test_task_result_failures_excluded_when_requested(tmp_path: Path):
    store = MemoryStore(tmp_path / "mem")
    store.append_task_results([_stored_task_result(7, speedup=None, correct=False)])
    items, _ = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="unknown",
        current_task_id=99,
        include_failures=False,
    )
    assert all(i["kind"] != "task_result" for i in items)
