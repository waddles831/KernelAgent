"""Evaluate one ModelNew source using KernelBench's official evaluator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=Path)
    parser.add_argument("--modelnew", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-correct-trials", required=True, type=int)
    parser.add_argument("--num-perf-trials", required=True, type=int)
    parser.add_argument("--timing-method", required=True)
    parser.add_argument("--gpu-arch", action="append", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    payload: dict[str, Any] = {
        "compiled": False,
        "runtime_success": False,
        "correct": False,
        "baseline_latency_ms": None,
        "best_latency_ms": None,
        "speedup": None,
        "metadata": {},
    }
    try:
        import torch
        from kernelbench.eval import eval_kernel_against_ref
        from kernelbench.utils import set_gpu_arch

        set_gpu_arch(args.gpu_arch)
        reference_source = args.problem.read_text(encoding="utf-8")
        modelnew_source = args.modelnew.read_text(encoding="utf-8")
        device = torch.device("cuda:0")

        result = eval_kernel_against_ref(
            original_model_src=reference_source,
            custom_model_src=modelnew_source,
            seed_num=args.seed,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            measure_performance=True,
            timing_method=args.timing_method,
            verbose=False,
            device=device,
            backend="triton",
            precision=torch.float32,
        )
        metadata = _json_safe(result.metadata or {})
        baseline = _positive_float(result.ref_runtime)
        candidate = _positive_float(result.runtime)
        correct = bool(result.correctness)
        payload = {
            "compiled": bool(result.compiled),
            "runtime_success": bool(
                result.compiled and not metadata.get("runtime_error")
            ),
            "correct": correct,
            "baseline_latency_ms": baseline,
            "best_latency_ms": candidate if correct else None,
            "speedup": (
                baseline / candidate
                if correct and baseline is not None and candidate is not None
                else None
            ),
            "metadata": metadata,
        }
        error = (
            metadata.get("compilation_error")
            or metadata.get("runtime_error")
            or metadata.get("error_during_performance")
        )
        if error:
            payload["error"] = str(error)
    except BaseException as exc:
        payload["error"] = str(exc)
        payload["fatal_error"] = str(exc)
        payload["exception"] = exc.__class__.__name__

    payload["elapsed_time_s"] = round(time.perf_counter() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(payload), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
