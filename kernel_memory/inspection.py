"""Inspect existing run artifacts for memory-readiness.

This report is deliberately explicit about *missing* data.  Profiling,
optimization-attempt and reflexion artifacts are frequently absent in
generation-only runs; they are reported as ``missing`` (with a reason) and are
never synthesized.  The report is used to validate the collection/ingestion
plumbing against existing runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _task_artifact_dir(run_dir: Path, result: dict[str, Any]) -> Path:
    task_id = int(result.get("task_id", -1))
    name = str(result.get("task_name", "task"))
    stem = Path(name).stem
    return run_dir / "kernels" / f"{task_id:03d}_{stem}"


def inspect_run_artifacts(run_dir: Path | str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = _read_json(run_dir / "config.json")
    results_path = run_dir / "task_results.jsonl"
    results: list[dict[str, Any]] = []
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                results.append(value)

    optimize_enabled = bool(config.get("optimize", False))
    tasks: list[dict[str, Any]] = []
    for result in results:
        artifact_dir = _task_artifact_dir(run_dir, result)
        optimization_result_path = artifact_dir / "optimization_result.json"
        optimization_result = _read_json(optimization_result_path)
        agent_result = _read_json(artifact_dir / "kernelagent_result.json")
        benchmark_result = _read_json(artifact_dir / "kernelagent_benchmark_result.json")
        benchmark_meta = (benchmark_result.get("metadata") or {}) if benchmark_result else {}

        attempts = optimization_result.get("attempts")
        reflexions = optimization_result.get("reflexions")
        last_attempt = optimization_result.get("last_attempt")
        last_reflexion = optimization_result.get("last_reflexion")

        session_dir = ((agent_result.get("details") or {}).get("session_dir"))
        run_subdir = ((agent_result.get("details") or {}).get("run_dir"))
        generation_history_present = False
        if session_dir:
            session_result = _read_json(Path(session_dir) / "result.json")
            generation_history_present = isinstance(session_result.get("history"), list)
        if not generation_history_present and run_subdir:
            # Fuser route: per-worker attempt traces under <run_dir>/workers.
            run_path = Path(run_subdir)
            generation_history_present = (run_path / "workers").is_dir() or (
                run_path / "compose_out"
            ).is_dir()

        missing: list[str] = []
        if not (artifact_dir / "kernel_function.py").exists():
            missing.append("kernel_function")
        if not (artifact_dir / "ModelNew.py").exists():
            missing.append("modelnew")
        if not benchmark_result:
            missing.append("project_benchmark_result")
        if not optimize_enabled:
            missing.append("optimization_result (reason=not_attempted)")
        elif not optimization_result_path.exists():
            missing.append("optimization_result")
        if not (artifact_dir / "kernel_function_optimized.py").exists():
            missing.append("optimized_kernel")

        has_profiling = any(
            "ncu" in str(key).lower() or "profil" in str(key).lower()
            for key in benchmark_meta
        )
        has_attempts = bool(attempts) or bool(last_attempt)
        has_reflexion = bool(reflexions) or bool(last_reflexion)

        if not has_profiling:
            missing.append("profiling_metadata (ncu unavailable/not recorded)")
        if not has_attempts:
            missing.append("optimization_attempts")
        if not has_reflexion:
            missing.append("reflexions")

        tasks.append(
            {
                "task_id": result.get("task_id"),
                "task_name": result.get("task_name"),
                "evaluation_status": result.get("evaluation_status"),
                "correct": result.get("correct"),
                "speedup": result.get("speedup"),
                "route": result.get("route"),
                "artifacts": {
                    "kernel_function": (artifact_dir / "kernel_function.py").exists(),
                    "modelnew": (artifact_dir / "ModelNew.py").exists(),
                    "project_benchmark_result": bool(benchmark_result),
                    "optimization_result": optimization_result_path.exists(),
                    "optimized_kernel": (artifact_dir / "kernel_function_optimized.py").exists(),
                },
                "has_profiling": has_profiling,
                "has_optimization_attempts": has_attempts,
                "has_reflexion": has_reflexion,
                "generation_history_present": generation_history_present,
                "missing": missing,
            }
        )

    return {
        "run_dir": str(run_dir),
        "experiment_name": config.get("experiment_name"),
        "optimize_enabled": optimize_enabled,
        "task_count": len(tasks),
        "tasks": tasks,
        "coverage": {
            "with_project_benchmark": sum(
                1 for t in tasks if t["artifacts"]["project_benchmark_result"]
            ),
            "with_optimization_result": sum(
                1 for t in tasks if t["artifacts"]["optimization_result"]
            ),
            "with_optimization_attempts": sum(
                1 for t in tasks if t["has_optimization_attempts"]
            ),
            "with_reflexion": sum(1 for t in tasks if t["has_reflexion"]),
            "with_profiling": sum(1 for t in tasks if t["has_profiling"]),
            "with_generation_history": sum(
                1 for t in tasks if t["generation_history_present"]
            ),
        },
        "notes": [
            "Missing profiling/attempt/reflexion data is marked, never inferred.",
            "Generation history is internal-test evidence, not measured performance.",
        ],
    }


def inspect_runs(output_root: Path | str) -> dict[str, Any]:
    root = Path(output_root)
    reports = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "task_results.jsonl").exists():
                reports.append(inspect_run_artifacts(child))
    return {"output_root": str(root), "runs": reports}
