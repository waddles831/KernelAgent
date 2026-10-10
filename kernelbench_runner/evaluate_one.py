"""Evaluate one candidate with KernelAgent's correctness-gated benchmark."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback
from typing import Any, Iterator


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


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _seed_all(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _args_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def _to_device(value: Any, *, device: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=torch.float32)
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device=device, torch=torch) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device=device, torch=torch) for item in value]
    if isinstance(value, dict):
        return {
            key: _to_device(item, device=device, torch=torch)
            for key, item in value.items()
        }
    return value


def _compare_outputs(
    reference: Any,
    candidate: Any,
    *,
    torch: Any,
    rtol: float,
    atol: float,
    path: str = "output",
) -> tuple[bool, str | None, float | None]:
    if isinstance(reference, torch.Tensor) or isinstance(candidate, torch.Tensor):
        if not isinstance(reference, torch.Tensor) or not isinstance(
            candidate, torch.Tensor
        ):
            return False, f"{path}: output types differ", None
        if reference.shape != candidate.shape:
            return (
                False,
                f"{path}: shape mismatch, expected {tuple(reference.shape)}, "
                f"got {tuple(candidate.shape)}",
                None,
            )
        if reference.is_floating_point() or reference.is_complex():
            try:
                matches = bool(
                    torch.allclose(reference, candidate, rtol=rtol, atol=atol)
                )
                max_abs_diff = float((reference - candidate).abs().max().item())
            except Exception as exc:
                return False, f"{path}: comparison failed: {exc}", None
            if not matches:
                return (
                    False,
                    f"{path}: values differ (max_abs_diff={max_abs_diff:.8g}, "
                    f"rtol={rtol}, atol={atol})",
                    max_abs_diff,
                )
            return True, None, max_abs_diff
        if not torch.equal(reference, candidate):
            return False, f"{path}: integer/bool values differ", None
        return True, None, 0.0

    if isinstance(reference, dict) or isinstance(candidate, dict):
        if not isinstance(reference, dict) or not isinstance(candidate, dict):
            return False, f"{path}: output container types differ", None
        if reference.keys() != candidate.keys():
            return False, f"{path}: output keys differ", None
        max_diff = 0.0
        for key in reference:
            matched, error, diff = _compare_outputs(
                reference[key],
                candidate[key],
                torch=torch,
                rtol=rtol,
                atol=atol,
                path=f"{path}[{key!r}]",
            )
            if not matched:
                return matched, error, diff
            if diff is not None:
                max_diff = max(max_diff, diff)
        return True, None, max_diff

    if isinstance(reference, (tuple, list)) or isinstance(candidate, (tuple, list)):
        if type(reference) is not type(candidate) or len(reference) != len(candidate):
            return False, f"{path}: output sequence structure differs", None
        max_diff = 0.0
        for index, (ref_item, candidate_item) in enumerate(zip(reference, candidate)):
            matched, error, diff = _compare_outputs(
                ref_item,
                candidate_item,
                torch=torch,
                rtol=rtol,
                atol=atol,
                path=f"{path}[{index}]",
            )
            if not matched:
                return matched, error, diff
            if diff is not None:
                max_diff = max(max_diff, diff)
        return True, None, max_diff

    if isinstance(reference, (float, int)) and isinstance(candidate, (float, int)):
        matches = math.isclose(
            float(reference), float(candidate), rel_tol=rtol, abs_tol=atol
        )
        diff = abs(float(reference) - float(candidate))
        return matches, None if matches else f"{path}: scalar values differ", diff
    matches = reference == candidate
    return bool(matches), None if matches else f"{path}: values differ", None


@contextmanager
def _gpu_benchmark_lock(lock_id: str) -> Iterator[str]:
    """Serialize benchmark sections from runner processes on the same host."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - server runtime is Linux
        yield "unavailable"
        return

    lock_path = (
        Path(tempfile.gettempdir()) / f"kernelagent-benchmark-gpu-{lock_id}.lock"
    )
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield str(lock_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _set_architecture(architectures: list[str]) -> None:
    if architectures:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(architectures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=Path)
    parser.add_argument("--modelnew", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-correct-trials", required=True, type=int)
    parser.add_argument("--warmup-ms", required=True, type=float)
    parser.add_argument("--rep-ms", required=True, type=float)
    parser.add_argument("--gpu-lock-id", required=True)
    parser.add_argument("--gpu-arch", action="append", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    phase = "setup"
    payload: dict[str, Any] = {
        "compiled": False,
        "runtime_success": False,
        "correct": False,
        "oom": False,
        "baseline_latency_ms": None,
        "candidate_latency_ms": None,
        "speedup": None,
        "metadata": {
            "evaluator": "kernelbench_runner.evaluate_one",
            "task_dataset": "KernelBench L1 local problem definition",
            "correctness_method": "torch.allclose against task Model",
            "correctness_atol": 1e-3,
            "correctness_rtol": 1e-3,
            "benchmark_method": "triton.testing.do_bench",
            "benchmark_baseline": {
                "name": "torch.compile",
                "mode": "default",
                "dynamic_shapes": False,
                "cuda_graphs": False,
            },
            "benchmark_warmup_ms": args.warmup_ms,
            "benchmark_rep_ms": args.rep_ms,
            "benchmark_return_mode": "mean",
            "gpu_arch": args.gpu_arch,
        },
    }

    try:
        _set_architecture(args.gpu_arch)
        import torch
        import triton
        import triton.testing

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        payload["metadata"].update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "triton_version": triton.__version__,
                "device": str(device),
            }
        )

        phase = "load_problem"
        problem_path = args.problem.resolve()
        candidate_path = args.modelnew.resolve()
        payload["metadata"]["problem_sha256"] = hashlib.sha256(
            problem_path.read_bytes()
        ).hexdigest()
        payload["metadata"]["candidate_sha256"] = hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest()
        problem = _load_module(problem_path, f"ka_problem_{os.getpid()}")
        candidate_module = _load_module(
            candidate_path, f"ka_candidate_{os.getpid()}"
        )
        reference_cls = getattr(problem, "Model")
        candidate_cls = getattr(candidate_module, "ModelNew")
        get_inputs = getattr(problem, "get_inputs")
        get_init_inputs = getattr(problem, "get_init_inputs", lambda: [])

        phase = "initialize_models"
        _seed_all(args.seed, torch)
        init_inputs = [
            _to_device(item, device=device, torch=torch)
            for item in _args_list(get_init_inputs())
        ]
        _seed_all(args.seed, torch)
        reference_model = reference_cls(*init_inputs).to(
            device=device, dtype=torch.float32
        )
        _seed_all(args.seed, torch)
        candidate_model = candidate_cls(*init_inputs).to(
            device=device, dtype=torch.float32
        )
        reference_model.eval()
        candidate_model.eval()

        phase = "correctness"
        trial_results: list[dict[str, Any]] = []
        pass_count = 0
        runtime_success = True
        for trial in range(args.num_correct_trials):
            trial_seed = args.seed + 17_171 * (trial + 1)
            _seed_all(trial_seed, torch)
            ref_inputs = [
                _to_device(item, device=device, torch=torch)
                for item in _args_list(get_inputs())
            ]
            try:
                with torch.inference_mode():
                    expected = reference_model(*ref_inputs)
                    torch.cuda.synchronize(device)
                    del ref_inputs

                    # Recreate the same seeded input instead of keeping two
                    # large device copies alive for correctness comparison.
                    _seed_all(trial_seed, torch)
                    candidate_inputs = [
                        _to_device(item, device=device, torch=torch)
                        for item in _args_list(get_inputs())
                    ]
                    actual = candidate_model(*candidate_inputs)
                    torch.cuda.synchronize(device)
                payload["compiled"] = True
                matched, error, max_diff = _compare_outputs(
                    expected,
                    actual,
                    torch=torch,
                    rtol=1e-3,
                    atol=1e-3,
                )
                trial_results.append(
                    {
                        "trial": trial + 1,
                        "seed": trial_seed,
                        "passed": matched,
                        "max_abs_diff": max_diff,
                        "error": error,
                    }
                )
                pass_count += int(matched)
                del expected, actual, candidate_inputs
            except Exception as exc:
                runtime_success = False
                trial_results.append(
                    {
                        "trial": trial + 1,
                        "seed": trial_seed,
                        "passed": False,
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
                payload["error"] = str(exc)
                payload["exception"] = exc.__class__.__name__
                break

        payload["runtime_success"] = runtime_success
        payload["correct"] = runtime_success and pass_count == args.num_correct_trials
        payload["metadata"]["correctness_trials"] = {
            "passed": pass_count,
            "total": args.num_correct_trials,
            "results": trial_results,
        }
        if any(
            "out of memory" in str(trial.get("error", "")).lower()
            for trial in trial_results
        ):
            payload["oom"] = True
            payload["error"] = payload.get("error") or (
                "CUDA out of memory during correctness trials"
            )

        if payload["correct"]:
            phase = "benchmark"
            perf_seed = args.seed
            _seed_all(perf_seed, torch)
            perf_inputs = [
                _to_device(item, device=device, torch=torch)
                for item in _args_list(get_inputs())
            ]

            compiled_reference = torch.compile(
                reference_model, mode="default", dynamic=False
            )

            def run_reference():
                with torch.inference_mode():
                    return compiled_reference(*perf_inputs)

            def run_candidate():
                with torch.inference_mode():
                    return candidate_model(*perf_inputs)

            with _gpu_benchmark_lock(args.gpu_lock_id) as lock_path:
                payload["metadata"]["benchmark_lock"] = lock_path
                baseline_ms = triton.testing.do_bench(
                    run_reference,
                    warmup=args.warmup_ms,
                    rep=args.rep_ms,
                    return_mode="mean",
                )
                baseline_ms = float(baseline_ms)
                payload["baseline_latency_ms"] = baseline_ms
                candidate_ms = triton.testing.do_bench(
                    run_candidate,
                    warmup=args.warmup_ms,
                    rep=args.rep_ms,
                    return_mode="mean",
                )
            candidate_ms = float(candidate_ms)
            payload["candidate_latency_ms"] = candidate_ms
            payload["speedup"] = baseline_ms / candidate_ms
            payload["metadata"]["performance"] = {
                "baseline_name": "torch.compile default mode",
                "baseline_latency_ms": baseline_ms,
                "candidate_latency_ms": candidate_ms,
                "speedup": payload["speedup"],
            }
    except Exception as exc:
        payload["error"] = str(exc)
        payload["exception"] = exc.__class__.__name__
        if "out of memory" in str(exc).lower():
            payload["oom"] = True
        if phase == "benchmark":
            payload["metadata"]["performance_error"] = str(exc)
        else:
            payload["fatal_error"] = str(exc)
            payload["metadata"]["failure_phase"] = phase
        payload["metadata"]["traceback"] = traceback.format_exc()

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
