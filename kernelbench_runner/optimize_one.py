"""Run KernelAgent's hardware-guided optimizer on one generated kernel.

This is the optional optimization stage of the benchmark runner.  It takes the
``kernel_function`` produced by generation, runs the project's
``OptimizationManager`` (beam search / greedy over LLM-proposed refinements),
and writes the best kernel it found.  Correctness is gated with the same
``ModelNew`` adapter convention used by :mod:`kernelbench_runner.evaluate_one`,
so an optimized kernel is only kept when it still matches the task's ``Model``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import REPO_ROOT


_TEST_TEMPLATE = r'''
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, __REPO_ROOT__)

from kernelbench_runner.adapter import wrap_kernel_as_modelnew

import problem


def _seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(value, device):
    if torch.is_tensor(value):
        value = value.to(device=device)
        if value.is_floating_point():
            value = value.to(dtype=torch.float32)
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_to_device(item, device) for item in value)
    return value


def main():
    device = torch.device("cuda")
    seed = 0
    get_init_inputs = getattr(problem, "get_init_inputs", lambda: [])
    init_inputs = _to_device(get_init_inputs(), device)

    with open("problem.py", encoding="utf-8") as handle:
        reference_source = handle.read()
    with open("kernel.py", encoding="utf-8") as handle:
        kernel_source = handle.read()

    _seed_all(seed)
    reference_model = problem.Model(*init_inputs).to(device=device, dtype=torch.float32)
    reference_model.eval()

    namespace = {}
    wrapped = wrap_kernel_as_modelnew(reference_source, kernel_source)
    modelnew_file = Path("ModelNew_generated.py")
    modelnew_file.write_text(wrapped, encoding="utf-8")
    import importlib.util

    spec = importlib.util.spec_from_file_location("ModelNew_generated", modelnew_file)
    modelnew_module = importlib.util.module_from_spec(spec)
    sys.modules["ModelNew_generated"] = modelnew_module
    spec.loader.exec_module(modelnew_module)
    _seed_all(seed)
    candidate_model = modelnew_module.ModelNew(*init_inputs).to(
        device=device, dtype=torch.float32
    )
    candidate_model.eval()

    inputs = [_to_device(item, device) for item in problem.get_inputs()]
    with torch.inference_mode():
        expected = reference_model(*inputs)
        actual = candidate_model(*inputs)

    if actual is None:
        actual = inputs[0]

    if not torch.is_tensor(expected) or not torch.is_tensor(actual):
        print("FAIL: non-tensor outputs")
        return 1
    if tuple(expected.shape) != tuple(actual.shape):
        print(f"FAIL: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
        return 1
    if expected.is_floating_point() or actual.is_floating_point():
        exp = expected.to(torch.float32)
        act = actual.to(torch.float32)
        if not torch.allclose(exp, act, rtol=1e-3, atol=1e-3):
            print(f"FAIL: max diff {(exp - act).abs().max().item()}")
            return 1
    elif not torch.equal(expected, actual):
        print("FAIL: integer/bool outputs differ")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


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


def build_test_source() -> str:
    """Return a standalone correctness test for the current problem."""
    return _TEST_TEMPLATE.replace("__REPO_ROOT__", repr(str(REPO_ROOT)))


def _detect_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _detect_environment(model: str, gpu_arch: list[str]) -> dict[str, Any]:
    """Toolchain fingerprint for memory compatibility (unknown stays None)."""
    gpu_name = torch_version = triton_version = cuda_version = None
    torch_version = None
    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import triton

        triton_version = getattr(triton, "__version__", None)
    except Exception:
        pass
    from kernel_memory.integration import build_compatibility

    return build_compatibility(
        gpu_name=gpu_name,
        torch_version=torch_version,
        triton_version=triton_version,
        cuda_version=cuda_version,
        gpu_arch=gpu_arch,
        model=model,
    )


def _build_memory_config(args: argparse.Namespace) -> dict[str, Any] | None:
    """Load the user memory config and augment it with task/run context."""
    if not args.memory_config:
        return None
    path = Path(args.memory_config)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or (payload.get("mode") or "off") == "off":
        return None

    from kernel_memory.features import extract_task_features

    payload = dict(payload)
    payload["run_id"] = args.run_id
    payload["task_id"] = args.task_id
    payload["task_name"] = args.task_name
    payload["task_features"] = extract_task_features(args.problem.resolve())
    payload["compatibility"] = _detect_environment(args.model, args.gpu_arch)
    payload["evidence_dir"] = str(args.log_dir)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=Path)
    parser.add_argument("--kernel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--strategy", default="greedy", choices=["greedy", "beam_search"]
    )
    parser.add_argument("--rounds", required=True, type=int)
    parser.add_argument("--top-kernels", required=True, type=int)
    parser.add_argument("--bottlenecks", required=True, type=int)
    parser.add_argument("--max-no-improvement", required=True, type=int)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--gpu-arch", action="append", default=[])
    parser.add_argument(
        "--memory-config",
        type=Path,
        default=None,
        help="Optional JSON memory configuration (mode/store_dir/limits)",
    )
    parser.add_argument("--run-id", default="unknown")
    parser.add_argument("--task-id", type=int, default=-1)
    parser.add_argument("--task-name", default="task")
    args = parser.parse_args()

    metadata: dict[str, Any] = {
        "success": False,
        "strategy": args.strategy,
        "rounds_requested": args.rounds,
        "optimized": False,
    }

    try:
        initial_kernel = args.kernel.read_text(encoding="utf-8")
        args.log_dir.mkdir(parents=True, exist_ok=True)
        test_source = build_test_source()
        (args.log_dir / "test.py").write_text(test_source, encoding="utf-8")

        from triton_kernel_agent.opt_manager import OptimizationManager

        gpu_name = _detect_gpu_name()
        if args.strategy == "beam_search":
            strategy_config: dict[str, Any] = {
                "num_top_kernels": args.top_kernels,
                "num_bottlenecks": args.bottlenecks,
                "num_expanding_parents": args.top_kernels,
                "samples_per_prompt": 1,
            }
            num_workers = args.top_kernels * args.bottlenecks
        else:
            strategy_config = {"max_no_improvement": args.max_no_improvement}
            num_workers = 1

        memory_config = _build_memory_config(args)

        manager = OptimizationManager(
            strategy=args.strategy,
            num_workers=num_workers,
            max_rounds=args.rounds,
            log_dir=args.log_dir,
            database_path=args.log_dir / "program_database.json",
            strategy_config=strategy_config,
            openai_model=args.model,
            high_reasoning_effort=False,
            gpu_ids=[args.gpu_id],
            gpu_name=gpu_name,
            target_platform="cuda",
            memory_config=memory_config,
        )
        result = manager.run_optimization(
            initial_kernel=initial_kernel,
            problem_file=args.problem.resolve(),
            test_code=test_source,
            max_rounds=args.rounds,
        )
        metadata.update(_json_safe(result))
        best_kernel = result.get("kernel_code")
        if result.get("success") and best_kernel:
            args.output.write_text(best_kernel, encoding="utf-8")
            metadata["optimized"] = True
            metadata["output"] = str(args.output.resolve())
    except BaseException as exc:  # keep the pipeline alive on optimizer errors
        metadata = {
            "success": False,
            "strategy": args.strategy,
            "rounds_requested": args.rounds,
            "optimized": False,
            "error": str(exc),
            "exception": exc.__class__.__name__,
        }

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if metadata.get("optimized") else 1


if __name__ == "__main__":
    raise SystemExit(main())
