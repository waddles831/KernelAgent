"""Run one KernelAgent generation task in an isolated child process."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _build_generation_memory(args) -> tuple[Any, str | None]:
    """Build a read-only memory bridge and retrieve seed-generation context.

    Returns ``(bridge_or_None, context_or_None)``.  Retrieval happens before any
    seed exists, so only source-derived task features are known and the
    bottleneck is explicitly ``unknown`` (never a profiled parent).
    """
    if not args.memory_config:
        return None, None
    path = Path(args.memory_config)
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict) or (payload.get("mode") or "off") == "off":
        return None, None

    from kernel_memory.features import extract_task_features
    from kernel_memory.integration import MemoryBridge, detect_environment

    payload = dict(payload)
    payload["run_id"] = args.run_id
    payload["task_id"] = args.task_id
    payload["task_name"] = args.task_name
    payload["task_features"] = extract_task_features(args.problem.resolve())
    payload["compatibility"] = detect_environment(model=args.model)
    payload["phase"] = "generation"

    bridge = MemoryBridge(payload)
    if not bridge.enabled:
        return bridge, None
    context = bridge.build_context(
        bottleneck_category="unknown",
        bottleneck_summary="seed generation (no profile available)",
        round_num=0,
        worker_id=None,
    )
    return bridge, (context or None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate-count", required=True, type=int)
    parser.add_argument("--cpu-worker-limit", required=True, type=int)
    parser.add_argument("--search-budget", required=True, type=int)
    parser.add_argument("--llm-timeout-s", required=True, type=int)
    parser.add_argument("--test-timeout-s", required=True, type=int)
    parser.add_argument("--kernel-output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument(
        "--memory-config",
        type=Path,
        default=None,
        help="Optional JSON memory configuration for seed-generation retrieval",
    )
    parser.add_argument("--memory-retrievals-output", type=Path, default=None)
    parser.add_argument("--run-id", default="unknown")
    parser.add_argument("--task-id", type=int, default=-1)
    parser.add_argument("--task-name", default="task")
    args = parser.parse_args()

    metadata: dict[str, Any] = {"success": False, "route": None, "details": {}}
    bridge = None
    try:
        # Retrieve before constructing the router so the context reaches both
        # the direct KernelAgent and Fuser routes.
        bridge, retrieval_context = _build_generation_memory(args)

        # Import after the parent process has selected CUDA_VISIBLE_DEVICES.
        from Fuser.auto_agent import AutoKernelRouter

        router = AutoKernelRouter(
            ka_model=args.model,
            ka_num_workers=args.candidate_count,
            ka_max_rounds=args.search_budget,
            router_model=args.model,
            extract_model=args.model,
            dispatch_model=args.model,
            compose_model=args.model,
            workers=args.cpu_worker_limit,
            max_iters=args.search_budget,
            compose_max_iters=args.search_budget,
            llm_timeout_s=args.llm_timeout_s,
            run_timeout_s=args.test_timeout_s,
            test_timeout_s=args.test_timeout_s,
            dispatch_jobs=1,
            verify=True,
            allow_fallback=True,
            target_platform="cuda",
            ignore_router_config=True,
            use_router_cache=False,
            retrieval_context=retrieval_context,
        )
        result = router.solve(args.problem.resolve())
        metadata = {
            "success": bool(result.success),
            "route": result.route,
            "details": _json_safe(result.details),
            "worker_id": getattr(result, "worker_id", None),
        }
        details = result.details if isinstance(result.details, dict) else {}
        session_dir = details.get("session_dir")
        if session_dir:
            seed_files = list(Path(session_dir).glob("seed_*.py"))
            metadata["candidate_count"] = len(seed_files) if seed_files else None
        else:
            metadata["candidate_count"] = None
        metadata["iterations"] = details.get("rounds")
        if metadata["iterations"] is None:
            composition = details.get("composition")
            if isinstance(composition, dict):
                metadata["iterations"] = composition.get("rounds")
        if result.kernel_code:
            args.kernel_output.parent.mkdir(parents=True, exist_ok=True)
            args.kernel_output.write_text(result.kernel_code, encoding="utf-8")
            metadata["kernel_output"] = str(args.kernel_output.resolve())
        else:
            metadata["error"] = "KernelAgent returned no kernel source"
    except BaseException as exc:
        metadata = {
            "success": False,
            "route": None,
            "error": str(exc),
            "exception": exc.__class__.__name__,
        }

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    if bridge is not None:
        metadata["memory"] = _json_safe(bridge.summary())
        if args.memory_retrievals_output is not None:
            args.memory_retrievals_output.parent.mkdir(parents=True, exist_ok=True)
            with args.memory_retrievals_output.open("w", encoding="utf-8") as handle:
                for record in bridge.retrieval_records:
                    handle.write(json.dumps(_json_safe(record)) + "\n")
    args.metadata_output.write_text(
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if metadata.get("success") and metadata.get("kernel_output") else 1


if __name__ == "__main__":
    raise SystemExit(main())
