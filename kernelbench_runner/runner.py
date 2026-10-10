"""Sequential KernelAgent → KernelBench-task benchmark runner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

from .adapter import wrap_kernel_as_modelnew
from .config import REPO_ROOT, RunnerConfig
from .schema import TaskResult


def _memory_enabled_helpers():
    """Import memory helpers lazily so off-mode runs have no import cost."""
    from kernel_memory import integration as memory_integration
    from kernel_memory import collector as memory_collector
    from kernel_memory.features import extract_task_features
    from kernel_memory.store import MemoryStore

    return memory_integration, memory_collector, extract_task_features, MemoryStore


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _l1_task_map(kernelbench_root: Path) -> dict[int, Path]:
    level1_dir = kernelbench_root / "KernelBench" / "level1"
    if not level1_dir.is_dir() and kernelbench_root.name == "KernelBench":
        level1_dir = kernelbench_root / "level1"
    if not level1_dir.is_dir():
        raise FileNotFoundError(f"KernelBench L1 directory not found: {level1_dir}")

    task_map: dict[int, Path] = {}
    for path in level1_dir.glob("*.py"):
        match = re.match(r"^(\d+)_", path.name)
        if match:
            task_map[int(match.group(1))] = path.resolve()
    if not task_map:
        raise RuntimeError(f"No numbered L1 task files found in {level1_dir}")
    return task_map


def select_tasks(
    task_map: dict[int, Path], config: RunnerConfig, select_all: bool
) -> list[Path]:
    available_ids = sorted(task_map)
    if select_all:
        selected_ids = available_ids
    elif config.task_ids is not None:
        selected_ids = sorted(config.task_ids)
    elif config.task_range is not None:
        start, end = config.task_range
        selected_ids = [task_id for task_id in available_ids if start <= task_id <= end]
    else:
        raise ValueError("Select tasks with --task-id, --task-range, or --all")

    missing = [task_id for task_id in selected_ids if task_id not in task_map]
    if missing:
        raise ValueError(f"KernelBench L1 task IDs not found: {missing}")
    if not selected_ids:
        raise ValueError("Task selection is empty")
    return [task_map[task_id] for task_id in selected_ids]


def _prepare_environment(config: RunnerConfig) -> dict[str, str]:
    if not os.environ.get(config.api_key_env):
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass
    env = os.environ.copy()
    secret = env.get(config.api_key_env)
    if not secret:
        raise ValueError(
            f"Set the API key in environment variable {config.api_key_env}; "
            "the runner does not read or store API key values from config files"
        )

    # KernelAgent's current OpenAI provider reads OPENAI_API_KEY. Copy the value
    # within the child process environment only; never serialize or print it.
    env["OPENAI_API_KEY"] = secret
    if config.api_base_url:
        env["OPENAI_BASE_URL"] = config.api_base_url
    env["OPENAI_MODEL"] = config.model
    env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    # Reduce allocator fragmentation on memory-heavy L1 tasks (e.g. the
    # 4096x393216 elementwise problems).  Applied to generation, optimization
    # and evaluation children uniformly so both A/B arms see the same setting.
    # PyTorch >= 2.9 renamed the variable; set the new name to avoid the
    # deprecation warning and fall back for older Torch only if unset.
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    python_path = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


def _cleanup_process_group(pid: int) -> None:
    """Terminate any grandchildren left in a finished child's process group.

    Generation and optimization children spawn worker subprocesses that touch
    the GPU.  Letting those linger into the next pipeline stage causes CUDA OOM
    in evaluation, so once the direct child exits we reap its whole group.
    """
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(0.5)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_child(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_s: int,
) -> tuple[int | None, bool]:
    """Run an isolated task process and kill its process group on timeout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            return None, True
        # Reap orphaned workers before returning to the caller.
        _cleanup_process_group(process.pid)
        return code, False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _generation_session_dirs(agent_metadata: dict[str, Any] | None) -> list[Path]:
    """Locate KernelAgent generation sessions for either route.

    Direct route: ``details.session_dir``.
    Fuser route: each subgraph in ``details.kernels_summary`` has its own
    ``session_dir``.  Duplicates are removed while preserving order.
    """
    details = (agent_metadata or {}).get("details") or {}
    candidates: list[str] = []
    direct = details.get("session_dir")
    if direct:
        candidates.append(str(direct))

    summary_path = details.get("kernels_summary")
    if summary_path and Path(summary_path).is_file():
        try:
            items = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            items = None
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("session_dir"):
                    candidates.append(str(item["session_dir"]))

    seen: set[str] = set()
    result: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            result.append(Path(candidate))
    return result


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _agent_iterations(metadata: dict[str, Any]) -> int | None:
    reported = metadata.get("iterations")
    if reported is not None:
        try:
            return int(reported)
        except (TypeError, ValueError):
            pass
    details = metadata.get("details")
    if not isinstance(details, dict):
        return None
    rounds = details.get("rounds")
    if rounds is None:
        composition = details.get("composition")
        if isinstance(composition, dict):
            rounds = composition.get("rounds")
    try:
        return int(rounds) if rounds is not None else None
    except (TypeError, ValueError):
        return None


def _agent_candidate_count(metadata: dict[str, Any]) -> int | None:
    reported = metadata.get("candidate_count")
    if reported is not None:
        try:
            return int(reported)
        except (TypeError, ValueError):
            pass
    details = metadata.get("details")
    if not isinstance(details, dict):
        return None
    session_dir = details.get("session_dir")
    if session_dir:
        seeds = list(Path(session_dir).glob("seed_*.py"))
        return len(seeds) if seeds else None
    return None


def _result_status(
    agent_success: bool,
    evaluation: dict[str, Any] | None,
    generation_timeout: bool,
    evaluation_timeout: bool,
    adapter_error: str | None,
) -> str:
    if generation_timeout:
        return "generation_timeout"
    if not agent_success:
        return "generation_failed"
    if adapter_error:
        return "adapter_failed"
    if evaluation_timeout:
        return "evaluation_timeout"
    if not evaluation:
        return "evaluation_failed"
    if evaluation.get("fatal_error"):
        return "evaluation_failed"
    if not evaluation.get("compiled"):
        return "compile_failed"
    if not evaluation.get("runtime_success"):
        return "runtime_failed"
    if evaluation.get("oom"):
        return "oom"
    if not evaluation.get("correct"):
        return "incorrect"
    if _positive_float(evaluation.get("candidate_latency_ms")) is None:
        return "timing_failed"
    return "passed"


def _compat_from_evaluation(
    evaluation: dict[str, Any] | None,
    config: RunnerConfig,
    *,
    build_compatibility,
) -> dict[str, Any]:
    """Compatibility fingerprint from evaluator metadata, else best-effort."""
    meta = (evaluation or {}).get("metadata") or {}
    return build_compatibility(
        gpu_name=meta.get("gpu"),
        torch_version=meta.get("torch_version"),
        triton_version=meta.get("triton_version"),
        cuda_version=None,
        gpu_arch=config.gpu_arch,
        model=config.model,
    )


def _child_memory_payload(memory_config) -> dict[str, Any]:
    return {
        "mode": memory_config.normalized_mode(),
        "store_dir": str(memory_config.retrieval_dir()),
        "snapshot_dir": (
            str(memory_config.resolved_snapshot_dir())
            if memory_config.is_frozen()
            else None
        ),
        "snapshot_label": memory_config.snapshot_label,
        "max_items": memory_config.max_items,
        "prompt_budget_chars": memory_config.prompt_budget_chars,
        "include_failures": memory_config.include_failures,
        "allow_foreign_compat": memory_config.allow_foreign_compat,
    }


def _ingest_task_memory(
    *,
    memory_config,
    run_id: str,
    task_id: int,
    task_name: str,
    task_path: Path,
    result_dict: dict[str, Any],
    agent_metadata: dict[str, Any] | None,
    optimization_metadata: dict[str, Any] | None,
    compatibility: dict[str, Any],
    write_store,
    generation_retrievals_path: Path | None = None,
) -> dict[str, Any]:
    """Normalize and persist this task's memory evidence.

    Returns a small info dict for the task result metadata.  This is the only
    place the persistent store is appended on the task path.
    """
    _, collector, extract_task_features, _ = _memory_enabled_helpers()
    task_features = extract_task_features(task_path)

    evidence_dir = (
        (optimization_metadata or {}).get("memory", {}).get("evidence_dir")
        if optimization_metadata
        else None
    )
    retrieval_records: list[dict[str, Any]] = []
    if evidence_dir:
        retrieval_records = collector.read_jsonl(
            Path(evidence_dir) / "optimization_retrievals.jsonl"
        )
    generation_retrieval_count = 0
    if generation_retrievals_path is not None and Path(
        generation_retrievals_path
    ).exists():
        generation_retrievals = collector.read_jsonl(generation_retrievals_path)
        generation_retrieval_count = len(generation_retrievals)
        retrieval_records.extend(generation_retrievals)

    memory_versions = sorted(
        {
            record.get("memory_version")
            for record in retrieval_records
            if record.get("memory_version")
        }
    )
    memory_version_used = memory_versions[0] if memory_versions else None

    attempt_records: list[dict[str, Any]] = []
    if evidence_dir:
        raw_evidence = collector.read_jsonl(
            Path(evidence_dir) / "optimization_attempts.jsonl"
        )
        attempt_records.extend(
            collector.attempt_records_from_optimization_evidence(
                raw_evidence,
                run_id=run_id,
                task_id=task_id,
                task_name=task_name,
                task_features=task_features,
                compatibility=compatibility,
                memory_version=memory_version_used,
            )
        )

    generation_count = 0
    if memory_config.record_generation_traces and agent_metadata:
        for session_dir in _generation_session_dirs(agent_metadata):
            generation_records = collector.generation_attempt_records_from_session(
                session_dir,
                run_id=run_id,
                task_id=task_id,
                task_name=task_name,
                task_features=task_features,
                compatibility=compatibility,
                memory_version=memory_version_used,
            )
            generation_count += len(generation_records)
            attempt_records.extend(generation_records)

    task_record = collector.task_result_record_from_result(
        result_dict,
        run_id=run_id,
        task_id=task_id,
        task_name=task_name,
        task_features=task_features,
        compatibility=compatibility,
        memory_version_used=memory_version_used,
        memory_mode_used=memory_config.normalized_mode(),
    )

    write_store.append_task_results([task_record])
    if attempt_records:
        write_store.append_attempts(attempt_records)
    if retrieval_records:
        write_store.append_retrievals(retrieval_records)

    return {
        "mode": memory_config.normalized_mode(),
        "memory_version_used": memory_version_used,
        "memory_versions_used": memory_versions,
        "attempt_records": len(attempt_records),
        "optimization_attempts": len(attempt_records) - generation_count,
        "generation_attempts": generation_count,
        "generation_retrieval_records": generation_retrieval_count,
        "retrieval_records": len(retrieval_records),
    }


def run_experiment(
    config: RunnerConfig,
    *,
    select_all: bool = False,
) -> Path:
    config.validate()
    kernelbench_root = config.resolved_kernelbench_root()
    task_map = _l1_task_map(kernelbench_root)
    task_paths = select_tasks(task_map, config, select_all)
    env = _prepare_environment(config)

    run_id = config.experiment_name
    memory_config = config.memory_config()
    write_store = None
    retrieval_store = None
    memory_build_compatibility = None
    if memory_config is not None:
        memory_integration, _, _, MemoryStore = _memory_enabled_helpers()
        memory_build_compatibility = memory_integration.build_compatibility
        if memory_config.write_enabled():
            write_store = MemoryStore(memory_config.write_dir(), read_only=False)
        if memory_config.retrieval_enabled():
            retrieval_store = MemoryStore(memory_config.retrieval_dir(), read_only=True)

    run_dir = config.resolved_output_root() / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "kernels").mkdir()
    (run_dir / "logs").mkdir()
    config_record = config.public_dict()
    config_record["selected_task_ids"] = [
        int(path.name.split("_", 1)[0]) for path in task_paths
    ]
    config_record["problem_set"] = "KernelBench L1"
    config_record["problem_set_source"] = "local task definitions only"
    config_record["correctness_evaluator"] = "kernelbench_runner.evaluate_one"
    config_record["performance_benchmark"] = {
        "method": "triton.testing.do_bench",
        "baseline": {
            "name": "torch.compile",
            "mode": "default",
            "dynamic_shapes": False,
            "cuda_graphs": False,
        },
        "warmup_ms": config.benchmark_warmup_ms,
        "rep_ms": config.benchmark_rep_ms,
        "return_mode": "mean",
    }
    config_record["gpu_selection"] = {
        "physical_id": config.gpu_id,
        "visible_device_index": 0,
        "gpu_arch": config.gpu_arch,
    }
    if memory_config is not None:
        memory_public = memory_config.public_dict()
        memory_public["retrieval_memory_version_at_start"] = (
            retrieval_store.memory_version() if retrieval_store is not None else None
        )
        memory_public["frozen_snapshot_version"] = (
            retrieval_store.read_version_info().get("memory_version")
            if memory_config.is_frozen() and retrieval_store is not None
            else None
        )
        config_record["memory"] = memory_public
    else:
        config_record["memory"] = {"mode": "off"}
    (run_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    results_path = run_dir / "task_results.jsonl"
    for task_path in task_paths:
        task_id = int(task_path.name.split("_", 1)[0])
        task_name = task_path.name
        task_key = f"{task_id:03d}_{task_path.stem}"
        task_artifacts = run_dir / "kernels" / task_key
        task_logs = run_dir / "logs" / task_key
        task_artifacts.mkdir()
        task_logs.mkdir()
        kernel_path = task_artifacts / "kernel_function.py"
        agent_metadata_path = task_artifacts / "kernelagent_result.json"
        modelnew_path = task_artifacts / "ModelNew.py"
        evaluation_path = task_artifacts / "kernelagent_benchmark_result.json"

        child_memory_path: Path | None = None
        generation_retrievals_path: Path | None = None
        if memory_config is not None:
            child_memory_path = task_artifacts / "memory_config.json"
            child_memory_path.write_text(
                json.dumps(
                    _child_memory_payload(memory_config),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            generation_retrievals_path = (
                task_artifacts / "generation_memory_retrievals.jsonl"
            )

        started = time.perf_counter()
        generation_timed_out = False
        optimization_timed_out = False
        evaluation_timed_out = False
        adapter_error: str | None = None
        agent_metadata: dict[str, Any] = {}
        optimization_metadata: dict[str, Any] = {}
        evaluation: dict[str, Any] | None = None

        generation_command = [
            sys.executable,
            "-m",
            "kernelbench_runner.generate_one",
            "--problem",
            str(task_path),
            "--model",
            config.model,
            "--candidate-count",
            str(config.candidate_count),
            "--cpu-worker-limit",
            str(config.cpu_worker_limit),
            "--search-budget",
            str(config.search_budget),
            "--llm-timeout-s",
            str(config.llm_timeout_s),
            "--test-timeout-s",
            str(config.agent_test_timeout_s),
            "--kernel-output",
            str(kernel_path),
            "--metadata-output",
            str(agent_metadata_path),
        ]
        if (
            memory_config is not None
            and memory_config.retrieval_enabled()
            and child_memory_path is not None
        ):
            generation_command.extend(
                [
                    "--memory-config",
                    str(child_memory_path),
                    "--memory-retrievals-output",
                    str(generation_retrievals_path),
                    "--run-id",
                    run_id,
                    "--task-id",
                    str(task_id),
                    "--task-name",
                    task_name,
                ]
            )
        generation_code, generation_timed_out = _run_child(
            generation_command,
            cwd=REPO_ROOT,
            env=env,
            log_path=task_logs / "kernelagent.log",
            timeout_s=config.generation_timeout_s,
        )
        agent_metadata = _read_json(agent_metadata_path)
        agent_success = bool(
            generation_code == 0
            and agent_metadata.get("success")
            and kernel_path.is_file()
        )

        # Optional hardware-guided optimization stage.  It runs in its own
        # process (the GPU is not yet initialized) and only replaces the
        # generated kernel when it finds a verified-faster best candidate.
        if agent_success and config.optimize:
            optimized_kernel_path = task_artifacts / "kernel_function_optimized.py"
            optimization_result_path = task_artifacts / "optimization_result.json"
            optimization_command = [
                sys.executable,
                "-m",
                "kernelbench_runner.optimize_one",
                "--problem",
                str(task_path),
                "--kernel",
                str(kernel_path),
                "--output",
                str(optimized_kernel_path),
                "--metadata-output",
                str(optimization_result_path),
                "--log-dir",
                str(task_logs / "optimization"),
                "--model",
                config.model,
                "--strategy",
                config.optimization_strategy,
                "--rounds",
                str(config.optimization_rounds),
                "--top-kernels",
                str(config.optimization_top_kernels),
                "--bottlenecks",
                str(config.optimization_bottlenecks),
                "--max-no-improvement",
                str(config.optimization_max_no_improvement),
                "--gpu-id",
                str(config.gpu_id),
            ]
            for architecture in config.gpu_arch:
                optimization_command.extend(["--gpu-arch", architecture])
            if memory_config is not None and child_memory_path is not None:
                optimization_command.extend(
                    [
                        "--memory-config",
                        str(child_memory_path),
                        "--run-id",
                        run_id,
                        "--task-id",
                        str(task_id),
                        "--task-name",
                        task_name,
                    ]
                )
            _, optimization_timed_out = _run_child(
                optimization_command,
                cwd=REPO_ROOT,
                env=env,
                log_path=task_logs / "optimization.log",
                timeout_s=config.optimization_timeout_s,
            )
            optimization_metadata = _read_json(optimization_result_path)
            if optimized_kernel_path.is_file():
                kernel_path = optimized_kernel_path

        if agent_success:
            try:
                kernel_source = kernel_path.read_text(encoding="utf-8")
                reference_source = task_path.read_text(encoding="utf-8")
                modelnew_source = wrap_kernel_as_modelnew(
                    reference_source, kernel_source
                )
                modelnew_path.write_text(modelnew_source, encoding="utf-8")
            except Exception as exc:
                adapter_error = f"{exc.__class__.__name__}: {exc}"

        if agent_success and adapter_error is None:
            evaluation_command = [
                sys.executable,
                "-m",
                "kernelbench_runner.evaluate_one",
                "--problem",
                str(task_path),
                "--modelnew",
                str(modelnew_path),
                "--output",
                str(evaluation_path),
                "--seed",
                str(config.random_seed),
                "--num-correct-trials",
                str(config.num_correct_trials),
                "--warmup-ms",
                str(config.benchmark_warmup_ms),
                "--rep-ms",
                str(config.benchmark_rep_ms),
                "--gpu-lock-id",
                str(config.gpu_id),
            ]
            for architecture in config.gpu_arch:
                evaluation_command.extend(["--gpu-arch", architecture])
            evaluation_code, evaluation_timed_out = _run_child(
                evaluation_command,
                cwd=REPO_ROOT,
                env=env,
                log_path=task_logs / "kernelagent_benchmark.log",
                timeout_s=config.evaluation_timeout_s,
            )
            evaluation = _read_json(evaluation_path)
            if evaluation_code not in (0, None) and not evaluation:
                evaluation = {
                    "error": f"KernelAgent benchmark exited with status {evaluation_code}"
                }

        compile_success = (
            bool(evaluation["compiled"]) if evaluation and "compiled" in evaluation else None
        )
        runtime_success = (
            bool(evaluation["runtime_success"])
            if evaluation and "runtime_success" in evaluation
            else None
        )
        correct = bool(evaluation["correct"]) if evaluation and "correct" in evaluation else None
        baseline_latency = (
            _positive_float(evaluation.get("baseline_latency_ms")) if evaluation else None
        )
        candidate_latency = (
            _positive_float(evaluation.get("candidate_latency_ms"))
            if evaluation
            else None
        )
        speedup = (
            _positive_float(evaluation.get("speedup")) if evaluation else None
        )
        errors = [
            item
            for item in (
                agent_metadata.get("error"),
                adapter_error,
                (evaluation.get("error") or evaluation.get("fatal_error"))
                if evaluation
                else None,
                (
                    f"Generation timed out after {config.generation_timeout_s}s"
                    if generation_timed_out
                    else None
                ),
                (
                    f"KernelAgent benchmark timed out after {config.evaluation_timeout_s}s"
                    if evaluation_timed_out
                    else None
                ),
                (
                    f"optimization: {optimization_metadata.get('error')}"
                    if config.optimize
                    and optimization_metadata
                    and not optimization_metadata.get("optimized")
                    else None
                ),
                (
                    f"Optimization timed out after {config.optimization_timeout_s}s"
                    if optimization_timed_out
                    else None
                ),
            )
            if item
        ]
        status = _result_status(
            agent_success,
            evaluation,
            generation_timed_out,
            evaluation_timed_out,
            adapter_error,
        )
        result = TaskResult(
            task_id=task_id,
            task_name=task_name,
            agent_success=agent_success,
            compile_success=compile_success,
            runtime_success=runtime_success,
            correct=correct,
            baseline_latency_ms=baseline_latency,
            candidate_latency_ms=candidate_latency,
            speedup=speedup,
            iterations=_agent_iterations(agent_metadata),
            candidate_count=_agent_candidate_count(agent_metadata),
            llm_calls=None,
            elapsed_time_s=round(time.perf_counter() - started, 3),
            evaluation_status=status,
            route=agent_metadata.get("route"),
            error="; ".join(str(item) for item in errors) or None,
            metadata={
                "kernelagent": agent_metadata,
                "benchmark": evaluation or {},
                "optimization": optimization_metadata or {},
                "llm_calls_note": (
                    "KernelAgent does not expose aggregate LLM call counts "
                    "in its current result API"
                ),
                "beam_width_note": (
                    "Recorded for reproducibility; OptimizationManager beam "
                    "search runs only when optimize is enabled (beam_width maps "
                    "to optimization_top_kernels)"
                ),
            },
            artifacts={
                "kernel_function": str(kernel_path) if kernel_path.is_file() else "",
                "modelnew": str(modelnew_path) if modelnew_path.is_file() else "",
                "kernelagent_log": str(task_logs / "kernelagent.log"),
                "kernelagent_benchmark_log": str(task_logs / "kernelagent_benchmark.log"),
                "kernelagent_result": str(agent_metadata_path),
                "kernelagent_benchmark_result": str(evaluation_path),
                "optimization_log": str(task_logs / "optimization.log"),
                "optimization_result": str(task_artifacts / "optimization_result.json"),
                "optimized_kernel_function": str(
                    task_artifacts / "kernel_function_optimized.py"
                ),
                "memory_config": (
                    str(child_memory_path) if child_memory_path is not None else ""
                ),
                "generation_memory_retrievals": (
                    str(generation_retrievals_path)
                    if generation_retrievals_path is not None
                    else ""
                ),
            },
        )
        result_path = task_artifacts / "task_result.json"

        if memory_config is not None and write_store is not None:
            try:
                compatibility = _compat_from_evaluation(
                    evaluation, config, build_compatibility=memory_build_compatibility
                )
                memory_info = _ingest_task_memory(
                    memory_config=memory_config,
                    run_id=run_id,
                    task_id=task_id,
                    task_name=task_name,
                    task_path=task_path,
                    result_dict=result.to_dict(),
                    agent_metadata=agent_metadata,
                    optimization_metadata=optimization_metadata,
                    compatibility=compatibility,
                    write_store=write_store,
                    generation_retrievals_path=generation_retrievals_path,
                )
                result.metadata["memory"] = memory_info
            except Exception as exc:  # memory must not break the run
                result.metadata["memory"] = {
                    "error": f"{exc.__class__.__name__}: {exc}"
                }

        result_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with results_path.open("a", encoding="utf-8") as results_file:
            results_file.write(
                json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
            )
        print(
            f"[{task_id}] {task_name}: {status}"
            + (f" ({speedup:.3f}x)" if speedup is not None else "")
        )

    return run_dir
