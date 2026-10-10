"""OOM is a non-evaluable status, not ``incorrect`` (CPU-only)."""

from __future__ import annotations

from kernelbench_runner.runner import _result_status
from kernel_memory.collector import classify_task_status


def test_oom_during_correctness_is_not_incorrect():
    evaluation = {
        "compiled": True,
        "runtime_success": True,
        "correct": False,
        "oom": True,
    }
    status = _result_status(True, evaluation, False, False, None)
    assert status == "oom"


def test_non_oom_failure_still_incorrect():
    evaluation = {
        "compiled": True,
        "runtime_success": True,
        "correct": False,
        "oom": False,
    }
    assert _result_status(True, evaluation, False, False, None) == "incorrect"


def test_oom_failure_category_maps():
    assert classify_task_status("oom") == "oom"
