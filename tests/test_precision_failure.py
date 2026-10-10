"""BF16/FP32 contract violations are correctness failures with no perf credit."""

from __future__ import annotations

from kernel_memory.collector import (
    classify_error,
    classify_task_status,
    task_result_record_from_result,
)
from kernel_memory.retrieval import retrieve_measured
from kernel_memory.store import MemoryStore
from tests.memory_helpers import compat, task_features


def _bf16_result():
    return {
        "task_id": 2,
        "task_name": "2_Standard_matrix_multiplication_.py",
        "agent_success": True,
        "compile_success": False,
        "runtime_success": False,
        "correct": False,
        "baseline_latency_ms": None,
        "candidate_latency_ms": None,
        "speedup": None,
        "iterations": 2,
        "candidate_count": 2,
        "evaluation_status": "compile_failed",
        "route": "kernelagent",
        "error": "Expected bf16",
        "metadata": {},
        "artifacts": {},
    }


def test_precision_mismatch_error_is_correctness_failure():
    assert classify_error("Expected bf16") == "correctness_failed"
    assert classify_error("inputs must be bfloat16") == "correctness_failed"
    assert classify_error("CUDA out of memory") == "oom"


def test_task_status_precision_mismatch_maps_to_correctness():
    assert (
        classify_task_status("compile_failed", "Expected bf16")
        == "correctness_failed"
    )
    # Without a precision signal the raw status mapping is preserved.
    assert classify_task_status("compile_failed") == "compile_error"


def test_bf16_task_result_has_no_performance_credit():
    record = task_result_record_from_result(
        _bf16_result(),
        run_id="r",
        task_id=2,
        task_name="2_Standard_matrix_multiplication_.py",
        task_features=task_features(),
        compatibility=compat(),
        memory_version_used=None,
        memory_mode_used="collect",
    )
    assert record["failure_category"] == "correctness_failed"
    assert record["task_speedup_vs_torch_compile"] is None
    assert record["correctness_status"] == "project_failed"


def test_bf16_task_result_not_used_for_performance_retrieval(tmp_path):
    store = MemoryStore(tmp_path / "mem")
    record = task_result_record_from_result(
        _bf16_result(),
        run_id="r",
        task_id=2,
        task_name="2_Standard_matrix_multiplication_.py",
        task_features=task_features(),
        compatibility=compat(),
        memory_version_used=None,
        memory_mode_used="collect",
    )
    store.append_task_results([record])
    items, _ = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="unknown",
        current_task_id=99,
        include_failures=False,
    )
    # Excluded as a failure; and when included it carries no measured speedup.
    assert all(i["kind"] != "task_result" for i in items)
    all_items, _ = retrieve_measured(
        store,
        compatibility=compat(),
        task_features=task_features(),
        bottleneck_category="unknown",
        current_task_id=99,
        include_failures=True,
    )
    tr = [i for i in all_items if i["kind"] == "task_result"]
    assert tr and tr[0]["measurement_available"] is False
