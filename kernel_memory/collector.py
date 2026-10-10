"""Normalize raw run artifacts into memory records.

The optimizer manager is the single writer of the raw optimization trajectory
(one JSON record per worker combining ``attempts``/``reflexions`` with the
parent linkage).  This module converts those raw payloads - and, optionally,
generation worker traces - into schema-versioned :mod:`kernel_memory` records.

Measurements are only attached when both the parent and the candidate were
observed under the same optimizer-internal protocol.  Generation traces have no
performance measurement and are labeled as internal-test-only evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from kernel_memory.schema import (
    attempt_record,
    sha256_text,
    task_result_record,
)

_PRECISION_MISMATCH_PATTERNS = (
    "expected bf16",
    "expected bfloat16",
    "expected fp16",
    "expected float16",
    "expected half",
    "expected float32",
    "expected fp32",
    "must be bf16",
    "must be bfloat16",
    "incompatible dtypes",
    "dtype mismatch",
    "promote_types",
    "not implemented for 'bfloat16'",
    "not implemented for 'half'",
    "not implemented for 'float16'",
)


def _looks_like_precision_mismatch(text: str | None) -> bool:
    """True when an error indicates a dtype/precision-contract violation.

    Such failures are correctness failures: they silently change numerical
    semantics versus the reference and must never earn performance credit.
    """
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in _PRECISION_MISMATCH_PATTERNS)


_ACTION_TAG_PROBES: tuple[tuple[str, str], ...] = (
    ("num_warps", r"\bnum_warps\b"),
    ("num_stages", r"\bnum_stages\b"),
    ("block_size", r"\bBLOCK_[A-Z0-9_]+\b"),
    ("tma", r"\b(?:tma|TensorDescriptor|tensormap)\b"),
    ("persistent", r"persistent"),
    ("vectorization", r"\b(?:float4|vectorize|vectorized)\b"),
    ("coalescing", r"coalesc"),
    ("reduction_decomposition", r"num_warps.*reduction|split.*reduction"),
    ("fusion", r"\bfusion\b"),
    ("swizzle", r"swizzle"),
)


def classify_error(error_text: str | None) -> str:
    """Best-effort failure category from an error string (heuristic)."""
    if not error_text:
        return "unknown"
    text = error_text.lower()
    if _looks_like_precision_mismatch(text):
        return "correctness_failed"
    if "out of memory" in text or "cuda error: out of memory" in text or "oom" in text:
        return "oom"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "notimplemented" in text or "not implemented" in text:
        return "not_implemented"
    if "allclose" in text or "values differ" in text or "max diff" in text or "shape mismatch" in text or "shape" in text and "!=" in text:
        return "correctness_failed"
    if "verification" in text:
        return "verification_failed"
    if "compile" in text or "syntaxerror" in text or "triton" in text and "error" in text:
        return "compile_error"
    if "cuda" in text or "runtimeerror" in text:
        return "runtime_error"
    return "unknown"


def classify_task_status(evaluation_status: str, error_text: str | None = None) -> str:
    if _looks_like_precision_mismatch(error_text):
        return "correctness_failed"
    mapping = {
        "passed": "none",
        "generation_failed": "runtime_error",
        "generation_timeout": "timeout",
        "adapter_failed": "runtime_error",
        "compile_failed": "compile_error",
        "runtime_failed": "runtime_error",
        "incorrect": "correctness_failed",
        "oom": "oom",
        "timing_failed": "timing_failed",
        "evaluation_failed": "runtime_error",
        "evaluation_timeout": "timeout",
    }
    return mapping.get(evaluation_status, "unknown")


def detect_transformation_tags(attempt: dict[str, Any], kernel_code: str | None = None) -> list[str]:
    haystack_parts = []
    if kernel_code:
        haystack_parts.append(kernel_code)
    for key in ("recommended_fix", "root_cause"):
        if attempt.get(key):
            haystack_parts.append(str(attempt[key]))
    for value in (attempt.get("config_changes") or {}).values():
        haystack_parts.append(str(value))
    if attempt.get("bottleneck_category"):
        haystack_parts.append(str(attempt["bottleneck_category"]))
    haystack = "\n".join(haystack_parts)
    return [tag for tag, pattern in _ACTION_TAG_PROBES if re.search(pattern, haystack)]


def _reflexion_for_round(reflexions: list[dict[str, Any]], round_num: int | None) -> dict[str, Any] | None:
    for reflexion in reflexions:
        if reflexion.get("round_num") == round_num:
            return reflexion
    return reflexions[-1] if reflexions else None


def attempt_records_from_optimization_evidence(
    evidence: list[dict[str, Any]],
    *,
    run_id: str,
    task_id: int,
    task_name: str,
    task_features: dict[str, Any],
    compatibility: dict[str, Any],
    memory_version: str | None,
) -> list[dict[str, Any]]:
    """Convert manager-written optimization trajectory records to attempts."""
    records: list[dict[str, Any]] = []
    for payload in evidence:
        parent_id = payload.get("parent_id")
        worker_id = payload.get("worker_id")
        result_kernel = payload.get("kernel_code")
        attempts = payload.get("attempts") or []
        reflexions = payload.get("reflexions") or []
        success = bool(payload.get("success"))

        for index, attempt in enumerate(attempts):
            passed = bool(attempt.get("passed_verification"))
            error_message = attempt.get("error_message") or ""
            reflexion = _reflexion_for_round(reflexions, attempt.get("round_num"))

            measurement_available = False
            local_speedup = None
            if passed:
                before = attempt.get("time_before_ms")
                after = attempt.get("time_after_ms")
                if (
                    isinstance(before, (int, float))
                    and isinstance(after, (int, float))
                    and before > 0
                    and after > 0
                ):
                    local_speedup = round(float(before) / float(after), 6)
                    measurement_available = True

            is_last = index == len(attempts) - 1
            candidate_sha = None
            candidate_id = None
            candidate_code_available = False
            # The worker only returns the best kernel; it equals the candidate
            # exactly when this last attempt passed and strictly improved.
            if is_last and passed and attempt.get("is_improvement") and result_kernel:
                candidate_sha = sha256_text(result_kernel)
                candidate_id = f"{parent_id or 'root'}->{candidate_sha[:10]}"
                candidate_code_available = True

            profile_available = any(
                (attempt.get(key) or 0) > 0
                for key in ("combined_sol_pct", "compute_sol_pct", "memory_sol_pct")
            )
            profile = {
                "status": "available" if profile_available else "unavailable",
                "source": "optimizer_ncu_roofline" if profile_available else None,
                "metrics": {
                    "combined_sol_pct": attempt.get("combined_sol_pct"),
                    "compute_sol_pct": attempt.get("compute_sol_pct"),
                    "memory_sol_pct": attempt.get("memory_sol_pct"),
                },
                "scope": "single_kernel_only",
                "note": (
                    "if the program contains multiple kernels these metrics "
                    "describe at most one kernel and must not be read as "
                    "end-to-end state"
                ),
            }

            if passed:
                correctness_status = "internal_passed"
            else:
                correctness_status = "internal_failed"

            action = {
                "summary": attempt.get("recommended_fix")
                or attempt.get("root_cause")
                or attempt.get("bottleneck_category")
                or "unspecified",
                "root_cause": attempt.get("root_cause"),
                "recommended_fix": attempt.get("recommended_fix"),
                "config_changes": attempt.get("config_changes") or {},
            }

            output_effect = {
                "is_improvement": bool(attempt.get("is_improvement")),
                "improvement_pct": attempt.get("improvement_pct"),
                "expected_outcome": (reflexion or {}).get("expected_outcome"),
                "was_diagnosis_correct": (reflexion or {}).get("was_diagnosis_correct"),
                "was_fix_effective": (reflexion or {}).get("was_fix_effective"),
                "note": (
                    "was_diagnosis_correct is an LLM judgement, not a measured fact"
                ),
            }

            records.append(
                attempt_record(
                    run_id=run_id,
                    task_id=task_id,
                    task_name=task_name,
                    phase="optimization",
                    parent_id=parent_id,
                    candidate_id=candidate_id,
                    candidate_sha256=candidate_sha,
                    round_num=attempt.get("round_num"),
                    worker_id=attempt.get("worker_id", worker_id),
                    task_features=task_features,
                    compatibility=compatibility,
                    bottleneck_category=attempt.get("bottleneck_category") or "unknown",
                    action=action,
                    transformation_tags=detect_transformation_tags(attempt, result_kernel),
                    output_effect=output_effect,
                    correctness_status=correctness_status,
                    measurement={
                        "available": measurement_available,
                        "method": "optimizer_internal_do_bench",
                        "parent_relative": True,
                        "time_before_ms": attempt.get("time_before_ms"),
                        "time_after_ms": attempt.get("time_after_ms"),
                        "local_speedup": local_speedup,
                        "note": (
                            "parent-relative; not comparable to the project "
                            "evaluator's torch.compile speedup"
                        ),
                    },
                    profile=profile,
                    failure_category=("none" if passed else classify_error(error_message)),
                    error_summary=(error_message[:400] if error_message else None),
                    reflexion=reflexion,
                    memory_version_used=memory_version,
                    provenance={
                        "worker_success": success,
                        "candidate_code_available": candidate_code_available,
                        "candidate_code_note": (
                            "worker returns only the best kernel; failed or "
                            "non-improving candidate code is not exported"
                        ),
                        "ptx_hash": payload.get("ptx_hash"),
                    },
                )
            )
    return records


def generation_attempt_records_from_session(
    session_dir: Path | str,
    *,
    run_id: str,
    task_id: int,
    task_name: str,
    task_features: dict[str, Any],
    compatibility: dict[str, Any],
    memory_version: str | None,
) -> list[dict[str, Any]]:
    """Convert generation worker traces into internal-test-only attempts."""
    directory = Path(session_dir)
    result_path = directory / "result.json"
    if not result_path.exists():
        return []
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    history = result.get("history") if isinstance(result, dict) else None
    if not isinstance(history, list):
        return []

    records: list[dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        kernel_code = entry.get("kernel_code") or ""
        passed = bool(entry.get("success"))
        stderr = entry.get("stderr") or ""
        stdout = entry.get("stdout") or ""
        records.append(
            attempt_record(
                run_id=run_id,
                task_id=task_id,
                task_name=task_name,
                phase="generation",
                parent_id=None,
                candidate_id=f"gen_{sha256_text(kernel_code)[:10]}" if kernel_code else None,
                candidate_sha256=sha256_text(kernel_code) if kernel_code else None,
                round_num=entry.get("round"),
                worker_id=None,
                task_features=task_features,
                compatibility=compatibility,
                bottleneck_category="unknown",
                action={"summary": "generation candidate", "root_cause": None, "recommended_fix": None, "config_changes": {}},
                transformation_tags=[],
                output_effect={"internal_test_success": passed, "stdout": stdout[:200]},
                correctness_status=("internal_passed" if passed else "internal_failed"),
                measurement={
                    "available": False,
                    "method": "none",
                    "parent_relative": False,
                    "time_before_ms": None,
                    "time_after_ms": None,
                    "local_speedup": None,
                    "note": "generation traces carry no performance measurement",
                },
                profile={"status": "not_applicable", "source": None, "metrics": {}, "scope": "none"},
                failure_category=("none" if passed else "test_failed"),
                error_summary=(stderr[:400] if stderr else None),
                reflexion=None,
                memory_version_used=memory_version,
                provenance={"source": "generation_session_history", "session_dir": str(directory)},
            )
        )
    return records


def task_result_record_from_result(
    result: dict[str, Any],
    *,
    run_id: str,
    task_id: int,
    task_name: str,
    task_features: dict[str, Any],
    compatibility: dict[str, Any],
    memory_version_used: str | None,
    memory_mode_used: str | None,
) -> dict[str, Any]:
    """Convert a ``TaskResult.to_dict()`` payload into a memory record."""
    metadata = result.get("metadata") or {}
    benchmark = metadata.get("benchmark") or {}
    benchmark_meta = benchmark.get("metadata") or {}
    timing = {
        "method": benchmark_meta.get("benchmark_method", "triton.testing.do_bench"),
        "baseline": benchmark_meta.get("benchmark_baseline"),
        "warmup_ms": benchmark_meta.get("benchmark_warmup_ms"),
        "rep_ms": benchmark_meta.get("benchmark_rep_ms"),
        "return_mode": benchmark_meta.get("benchmark_return_mode"),
    }
    correctness_status = (
        "project_passed" if result.get("correct") is True
        else "project_failed" if result.get("correct") is False
        else "unknown"
    )
    return task_result_record(
        run_id=run_id,
        task_id=task_id,
        task_name=task_name,
        task_source_sha256=task_features.get("source_sha256"),
        task_features=task_features,
        compatibility=compatibility,
        evaluation_status=result.get("evaluation_status", "unknown"),
        correctness_status=correctness_status,
        baseline_latency_ms=result.get("baseline_latency_ms"),
        candidate_latency_ms=result.get("candidate_latency_ms"),
        task_speedup=result.get("speedup"),
        timing_methodology=timing,
        route=result.get("route"),
        failure_category=classify_task_status(
            result.get("evaluation_status", "unknown"), result.get("error")
        ),
        memory_version_used=memory_version_used,
        memory_mode_used=memory_mode_used,
        artifacts=result.get("artifacts") or {},
        metadata={
            "iterations": result.get("iterations"),
            "candidate_count": result.get("candidate_count"),
            "error": result.get("error"),
        },
    )


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records
