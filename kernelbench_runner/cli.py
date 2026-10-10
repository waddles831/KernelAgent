"""Command-line entry point for KernelBench L1 experiments."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import os
import sys

from .config import REPO_ROOT, RunnerConfig
from .runner import _l1_task_map, run_experiment


def _parse_task_range(value: str) -> list[int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("task range must use START:END") from exc
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("task range must be a positive inclusive range")
    return [start, end]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate kernels for selected KernelBench L1 tasks and evaluate "
            "them with KernelAgent's benchmark protocol"
        )
    )
    parser.add_argument("--config", type=Path, help="JSON runner configuration")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--task-id", type=int, action="append", help="L1 task ID; may repeat"
    )
    selection.add_argument(
        "--task-range",
        type=_parse_task_range,
        help="Inclusive ID range, e.g. 1:10",
    )
    selection.add_argument("--all", action="store_true", help="Select every local L1 task")

    overrides = {
        "experiment_name": ("--experiment-name", str),
        "kernelbench_path": ("--kernelbench-path", str),
        "output_root": ("--output-root", str),
        "model": ("--model", str),
        "api_base_url": ("--api-base-url", str),
        "api_key_env": ("--api-key-env", str),
        "gpu_id": ("--gpu-id", int),
        "cpu_worker_limit": ("--cpu-worker-limit", int),
        "candidate_count": ("--candidate-count", int),
        "search_budget": ("--search-budget", int),
        "beam_width": ("--beam-width", int),
        "random_seed": ("--seed", int),
        "num_correct_trials": ("--num-correct-trials", int),
        "benchmark_warmup_ms": ("--benchmark-warmup-ms", float),
        "benchmark_rep_ms": ("--benchmark-rep-ms", float),
        "generation_timeout_s": ("--generation-timeout-s", int),
        "llm_timeout_s": ("--llm-timeout-s", int),
        "agent_test_timeout_s": ("--agent-test-timeout-s", int),
        "evaluation_timeout_s": ("--evaluation-timeout-s", int),
    }
    for field_name, (flag, field_type) in overrides.items():
        parser.add_argument(flag, dest=field_name, type=field_type)
    parser.add_argument(
        "--gpu-arch",
        action="append",
        dest="gpu_arch",
        help="Target GPU architecture for the generated kernel (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass
        config = RunnerConfig.from_json(args.config) if args.config else RunnerConfig()
        args_dict = vars(args)
        for field in fields(RunnerConfig):
            value = args_dict.get(field.name)
            if value is not None:
                setattr(config, field.name, value)
        if args.task_id is not None:
            config.task_ids = args.task_id
            config.task_range = None
        elif args.task_range is not None:
            config.task_range = args.task_range
            config.task_ids = None
        if args.gpu_arch is not None:
            config.gpu_arch = args.gpu_arch

        # Pick up customary endpoint/model environment configuration while
        # persisting only the non-secret endpoint and model name.
        config.api_base_url = config.api_base_url or os.environ.get("OPENAI_BASE_URL")
        if "model" not in {
            key for key, value in args_dict.items() if value is not None
        } and not args.config:
            config.model = os.environ.get("OPENAI_MODEL", config.model)

        kernelbench_root = config.resolved_kernelbench_root()
        task_map = _l1_task_map(kernelbench_root)
        if args.all:
            config.task_ids = sorted(task_map)
            config.task_range = None
        elif config.task_ids is None and config.task_range is None:
            parser.error(
                "choose --task-id, --task-range, --all, or set task_ids/task_range in config"
            )

        run_dir = run_experiment(config, select_all=False)
        print(f"Run artifacts: {run_dir}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"kernelbench-l1: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
