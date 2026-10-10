"""Tests for offline, rule-based strategy summary construction (CPU-only)."""

from __future__ import annotations

from kernel_memory.summary import build_summaries
from tests.memory_helpers import make_attempt


def test_build_summaries_groups_and_counts():
    attempts = [
        make_attempt(local_speedup=1.25, passed=True),
        make_attempt(local_speedup=1.10, passed=True),
        make_attempt(local_speedup=0.90, passed=True),
        make_attempt(passed=False, local_speedup=None, failure_category="verification_failed"),
    ]
    summaries = build_summaries(attempts)
    assert len(summaries) == 1
    summary = summaries[0]
    effect = summary["observed_effect"]
    assert effect["improvements"] == 2
    assert effect["regressions"] == 1
    assert effect["failures"] == 1
    assert summary["support_ids"]  # passed attempts
    assert len(summary["counterexample_ids"]) == 1  # failed attempt
    assert summary["llm_judgment_used"] is False
    assert "measured" in summary["measurement_basis"]


def test_build_summaries_deterministic_ids():
    attempts = [make_attempt()]
    a = build_summaries(attempts, version="v9")
    b = build_summaries(attempts, version="v9")
    assert a[0]["summary_id"] == b[0]["summary_id"]
    assert a[0]["version"] == "v9"


def test_build_summaries_separates_by_bottleneck():
    attempts = [
        make_attempt(bottleneck="memory"),
        make_attempt(bottleneck="compute"),
    ]
    summaries = build_summaries(attempts)
    assert len(summaries) == 2
    assert {s["bottleneck_category"] for s in summaries} == {"memory", "compute"}


def test_build_summaries_min_support():
    attempts = [make_attempt()]
    assert build_summaries(attempts, min_support=2) == []
    assert len(build_summaries(attempts, min_support=1)) == 1


def test_build_summaries_records_evidence_ids_and_confidence():
    attempts = [
        make_attempt(local_speedup=1.4, passed=True, record_id="r1"),
        make_attempt(local_speedup=0.8, passed=True, record_id="r2"),
    ]
    summary = build_summaries(attempts)[0]
    assert summary["support_ids"] == ["r1", "r2"]
    assert 0.0 <= summary["confidence"] <= 1.0
    assert summary["suggested_lesson"] is None
