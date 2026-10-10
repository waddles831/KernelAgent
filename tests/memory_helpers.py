"""Shared builders for cross-task memory tests (CPU-only)."""

from __future__ import annotations

from pathlib import Path

from kernel_memory.features import compatibility_from_environment
from kernel_memory.schema import attempt_record


def compat(gpu_name: str = "NVIDIA A30", torch_version: str = "2.9.1") -> dict:
    return compatibility_from_environment(
        gpu_name=gpu_name,
        torch_version=torch_version,
        triton_version="3.5.1",
        cuda_version="12.2",
        gpu_arch=["Ampere"],
        model="test-model",
    )


def task_features(sha: str | None = "abc", ops: list[str] | None = None) -> dict:
    return {
        "source_sha256": sha,
        "op_hints": ops if ops is not None else ["layernorm", "reduction"],
        "model_class": "Model",
        "unknown_features": ["input_shapes", "dtypes"],
    }


def make_attempt(
    *,
    task_id: int = 1,
    bottleneck: str = "memory",
    action_summary: str = "num_warps 4->8",
    local_speedup: float | None = 1.25,
    passed: bool = True,
    failure_category: str = "none",
    compatibility: dict | None = None,
    phase: str = "optimization",
    round_num: int = 1,
    record_id: str | None = None,
) -> dict:
    measured = passed and local_speedup is not None
    return attempt_record(
        run_id="run-test",
        task_id=task_id,
        task_name=f"{task_id}_task.py",
        phase=phase,
        parent_id="initial",
        candidate_id="initial->cand",
        candidate_sha256="candhash",
        round_num=round_num,
        worker_id=0,
        task_features=task_features(),
        compatibility=compatibility or compat(),
        bottleneck_category=bottleneck,
        action={"summary": action_summary, "config_changes": {"num_warps": "4->8"}},
        transformation_tags=["num_warps"],
        output_effect={"is_improvement": bool(measured and local_speedup and local_speedup > 1)},
        correctness_status="internal_passed" if passed else "internal_failed",
        measurement={
            "available": measured,
            "method": "optimizer_internal_do_bench",
            "parent_relative": True,
            "time_before_ms": 1.0 if measured else None,
            "time_after_ms": (1.0 / local_speedup) if measured and local_speedup else None,
            "local_speedup": local_speedup if measured else None,
        },
        profile={
            "status": "available" if measured else "unavailable",
            "metrics": {"combined_sol_pct": 50.0 if measured else None},
        },
        failure_category=failure_category,
        error_summary=None,
        reflexion={"round_num": round_num, "was_fix_effective": bool(measured)},
        memory_version_used="v0",
        record_id=record_id,
    )


def make_task_result(
    *,
    task_id: int = 1,
    evaluation_status: str = "passed",
    correct: bool = True,
    speedup: float | None = 1.2,
) -> dict:
    """Mimic a ``TaskResult.to_dict()`` for runner-ingestion tests."""
    return {
        "task_id": task_id,
        "task_name": f"{task_id}_task.py",
        "agent_success": True,
        "compile_success": True,
        "runtime_success": True,
        "correct": correct,
        "baseline_latency_ms": 10.0,
        "candidate_latency_ms": 8.0 if correct else None,
        "speedup": speedup,
        "iterations": 2,
        "candidate_count": 4,
        "evaluation_status": evaluation_status,
        "route": "kernelagent",
        "error": None,
        "metadata": {
            "benchmark": {
                "metadata": {
                    "benchmark_method": "triton.testing.do_bench",
                    "benchmark_warmup_ms": 1000.0,
                    "benchmark_rep_ms": 100.0,
                    "benchmark_return_mode": "mean",
                }
            }
        },
        "artifacts": {},
    }


def write_evidence(dir_path: Path, attempts: list[dict]) -> None:
    """Write manager-style raw optimization evidence for ingestion tests."""
    payload = {
        "worker_id": 0,
        "round_num": 1,
        "parent_id": "initial",
        "openai_model": "test-model",
        "kernel_code": "def kernel_function(*a): return a[0]",
        "time_ms": 0.8,
        "success": True,
        "ptx_hash": "ptx0",
        "attempts": attempts,
        "reflexions": [{"round_num": 1, "was_fix_effective": True}],
    }
    import json

    dir_path.mkdir(parents=True, exist_ok=True)
    with (dir_path / "optimization_attempts.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    with (dir_path / "optimization_retrievals.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "kind": "retrieval",
                    "memory_version": "frozenv1",
                    "task_id": 1,
                    "retrieved": [],
                }
            )
            + "\n"
        )
