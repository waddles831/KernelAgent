"""Sequential KernelAgent → KernelBench L1 experiment runner."""

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

from .adapter import wrap_kernel_for_kernelbench
from .config import REPO_ROOT, RunnerConfig
from .schema import TaskResult


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


def _prepare_environment(config: RunnerConfig, kernelbench_root: Path) -> dict[str, str]:
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
    python_path = [str(REPO_ROOT), str(kernelbench_root / "src")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


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
            return process.wait(timeout=timeout_s), False
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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
    if not evaluation.get("correct"):
        return "incorrect"
    if _positive_float(evaluation.get("best_latency_ms")) is None:
        return "timing_failed"
    return "passed"


def run_experiment(
    config: RunnerConfig,
    *,
    select_all: bool = False,
) -> Path:
    config.validate()
    kernelbench_root = config.resolved_kernelbench_root()
    task_map = _l1_task_map(kernelbench_root)
    task_paths = select_tasks(task_map, config, select_all)
    env = _prepare_environment(config, kernelbench_root)

    run_dir = config.resolved_output_root() / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "kernels").mkdir()
    (run_dir / "logs").mkdir()
    config_record = config.public_dict()
    config_record["selected_task_ids"] = [
        int(path.name.split("_", 1)[0]) for path in task_paths
    ]
    config_record["kernelbench_level"] = 1
    config_record["kernelbench_evaluator"] = "kernelbench.eval.eval_kernel_against_ref"
    config_record["gpu_selection"] = {
        "physical_id": config.gpu_id,
        "visible_device_index": 0,
        "gpu_arch": config.gpu_arch,
    }
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
        evaluation_path = task_artifacts / "kernelbench_result.json"

        started = time.perf_counter()
        generation_timed_out = False
        evaluation_timed_out = False
        adapter_error: str | None = None
        agent_metadata: dict[str, Any] = {}
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

        if agent_success:
            try:
                kernel_source = kernel_path.read_text(encoding="utf-8")
                reference_source = task_path.read_text(encoding="utf-8")
                modelnew_source = wrap_kernel_for_kernelbench(
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
                "--num-perf-trials",
                str(config.num_perf_trials),
                "--timing-method",
                config.timing_method,
            ]
            for architecture in config.gpu_arch:
                evaluation_command.extend(["--gpu-arch", architecture])
            evaluation_code, evaluation_timed_out = _run_child(
                evaluation_command,
                cwd=kernelbench_root,
                env=env,
                log_path=task_logs / "kernelbench.log",
                timeout_s=config.evaluation_timeout_s,
            )
            evaluation = _read_json(evaluation_path)
            if evaluation_code not in (0, None) and not evaluation:
                evaluation = {
                    "error": f"KernelBench evaluator exited with status {evaluation_code}"
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
        best_latency = (
            _positive_float(evaluation.get("best_latency_ms")) if evaluation else None
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
                    f"KernelBench evaluation timed out after {config.evaluation_timeout_s}s"
                    if evaluation_timed_out
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
            best_latency_ms=best_latency,
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
                "kernelbench": evaluation or {},
                "llm_calls_note": (
                    "KernelAgent does not expose aggregate LLM call counts "
                    "in its current result API"
                ),
                "beam_width_note": (
                    "Recorded for reproducibility; the KernelAgent generation "
                    "baseline does not invoke OptimizationManager beam search"
                ),
            },
            artifacts={
                "kernel_function": str(kernel_path) if kernel_path.is_file() else "",
                "modelnew": str(modelnew_path) if modelnew_path.is_file() else "",
                "kernelagent_log": str(task_logs / "kernelagent.log"),
                "kernelbench_log": str(task_logs / "kernelbench.log"),
                "kernelagent_result": str(agent_metadata_path),
                "kernelbench_result": str(evaluation_path),
            },
        )
        result_path = task_artifacts / "task_result.json"
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
