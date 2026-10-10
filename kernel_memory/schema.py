"""Versioned record schemas for the cross-task memory store.

All records are plain JSON-serializable dicts.  ``schema_version`` is stored on
every record so that future migrations are explicit.  Absent measurements are
represented as ``None`` and paired with a reason; a failed or unmeasured
candidate must never be written as a valid performance observation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import uuid
from typing import Any

# Bump when the *overall* on-disk contract changes.  Per-artifact versions let
# an individual record type evolve without invalidating the others.
MEMORY_SCHEMA_VERSION = 1
TASK_RESULT_SCHEMA_VERSION = 1
ATTEMPT_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1
RETRIEVAL_SCHEMA_VERSION = 1

# Failure/outcome vocabulary shared by attempts and task results.
FAILURE_CATEGORIES = (
    "none",
    "compile_error",
    "runtime_error",
    "correctness_failed",
    "verification_failed",  # internal optimizer correctness gate
    "test_failed",  # generation-internal test
    "timeout",
    "oom",
    "profiling_failed",
    "timing_failed",
    "not_implemented",
    "unknown",
)

# How a candidate's correctness was (or was not) established.
CORRECTNESS_STATUSES = (
    "project_passed",  # passed kernelbench_runner.evaluate_one
    "project_failed",
    "internal_passed",  # passed the optimizer's own verification
    "internal_failed",
    "unknown",
)

PROFILE_STATUSES = ("available", "unavailable", "unknown", "not_applicable")

MEASUREMENT_BASIS_OPTIMIZER = "optimizer_internal_do_bench_parent_relative"
MEASUREMENT_BASIS_PROJECT = "project_evaluator_do_bench_vs_torch_compile"

SUMMARY_SOURCE = "offline-consolidation-v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_record_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def stable_hash(*parts: Any) -> str:
    """Deterministic short hash over JSON-normalized parts."""
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def task_result_record(
    *,
    run_id: str,
    task_id: int,
    task_name: str,
    task_source_sha256: str | None,
    task_features: dict[str, Any],
    compatibility: dict[str, Any],
    evaluation_status: str,
    correctness_status: str,
    baseline_latency_ms: float | None,
    candidate_latency_ms: float | None,
    task_speedup: float | None,
    timing_methodology: dict[str, Any],
    route: str | None,
    failure_category: str,
    memory_version_used: str | None,
    memory_mode_used: str | None,
    artifacts: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    record_id: str | None = None,
) -> dict[str, Any]:
    """Build a final task-result memory record.

    ``task_speedup`` is the project-evaluator speedup against ``torch.compile``.
    It is intentionally kept separate from any parent-relative attempt speedup.
    """
    return {
        "schema_version": TASK_RESULT_SCHEMA_VERSION,
        "memory_schema_version": MEMORY_SCHEMA_VERSION,
        "kind": "task_result",
        "record_id": record_id or new_record_id("task"),
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "task_id": int(task_id),
        "task_name": task_name,
        "task_source_sha256": task_source_sha256,
        "task_features": task_features,
        "compatibility": compatibility,
        "evaluator": "kernelbench_runner.evaluate_one",
        "evaluation_scope": "project_evaluator",
        "evaluation_status": evaluation_status,
        "correctness_status": correctness_status,
        "baseline_latency_ms": baseline_latency_ms,
        "candidate_latency_ms": candidate_latency_ms,
        "task_speedup_vs_torch_compile": task_speedup,
        "parent_relative_speedup": None,
        "measurement_basis": MEASUREMENT_BASIS_PROJECT,
        "timing_methodology": timing_methodology,
        "route": route,
        "failure_category": failure_category,
        "memory_version_used": memory_version_used,
        "memory_mode_used": memory_mode_used,
        "artifacts": artifacts or {},
        "metadata": metadata or {},
    }


def attempt_record(
    *,
    run_id: str,
    task_id: int,
    task_name: str,
    phase: str,
    parent_id: str | None,
    candidate_id: str | None,
    candidate_sha256: str | None,
    round_num: int | None,
    worker_id: int | None,
    task_features: dict[str, Any],
    compatibility: dict[str, Any],
    bottleneck_category: str,
    action: dict[str, Any],
    transformation_tags: list[str],
    output_effect: dict[str, Any],
    correctness_status: str,
    measurement: dict[str, Any],
    profile: dict[str, Any],
    failure_category: str,
    error_summary: str | None,
    reflexion: dict[str, Any] | None,
    memory_version_used: str | None,
    provenance: dict[str, Any] | None = None,
    record_id: str | None = None,
) -> dict[str, Any]:
    """Build an immutable per-attempt evidence record.

    ``measurement['local_speedup']`` is parent-relative and only set when both
    the parent and candidate were measured under the same optimizer-internal
    protocol.  A failed/unverified candidate gets ``available=False`` and a
    ``None`` speedup.
    """
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "memory_schema_version": MEMORY_SCHEMA_VERSION,
        "kind": "attempt",
        "record_id": record_id or new_record_id("attempt"),
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "task_id": int(task_id),
        "task_name": task_name,
        "phase": phase,
        "parent_id": parent_id,
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "round_num": round_num,
        "worker_id": worker_id,
        "task_features": task_features,
        "compatibility": compatibility,
        "bottleneck_category": bottleneck_category,
        "action": action,
        "transformation_tags": transformation_tags,
        "output_effect": output_effect,
        "correctness_status": correctness_status,
        "measurement": measurement,
        "profile": profile,
        "failure_category": failure_category,
        "error_summary": error_summary,
        "reflexion": reflexion,
        "memory_version_used": memory_version_used,
        "provenance": provenance or {},
    }


def strategy_summary(
    *,
    summary_id: str,
    version: str,
    compatibility_key: str,
    compatibility: dict[str, Any],
    task_feature_predicate: dict[str, Any],
    bottleneck_category: str,
    action: str,
    transformation_tags: list[str],
    applicability: str,
    observed_effect: dict[str, Any],
    support_ids: list[str],
    counterexample_ids: list[str],
    confidence: float,
    caveat: str,
    evidence_hash: str,
    suggested_lesson: str | None = None,
) -> dict[str, Any]:
    """Build a versioned, evidence-linked strategy summary.

    ``suggested_lesson`` is optional natural-language text.  It is explicitly
    labeled as unverified interpretation and never overrides measurements.
    """
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "memory_schema_version": MEMORY_SCHEMA_VERSION,
        "kind": "summary",
        "summary_id": summary_id,
        "version": version,
        "created_at": utc_now_iso(),
        "source": SUMMARY_SOURCE,
        "llm_judgment_used": False,
        "compatibility_key": compatibility_key,
        "compatibility": compatibility,
        "task_feature_predicate": task_feature_predicate,
        "bottleneck_category": bottleneck_category,
        "action": action,
        "transformation_tags": transformation_tags,
        "applicability": applicability,
        "observed_effect": observed_effect,
        "support_ids": support_ids,
        "counterexample_ids": counterexample_ids,
        "confidence": confidence,
        "caveat": caveat,
        "evidence_hash": evidence_hash,
        "measurement_basis": "measured parent-relative optimizer telemetry only",
        "suggested_lesson": suggested_lesson,
    }


def retrieval_record(
    *,
    run_id: str,
    task_id: int,
    round_num: int | None,
    worker_id: int | None,
    query: dict[str, Any],
    compatibility: dict[str, Any],
    memory_mode: str,
    memory_version: str | None,
    frozen: bool,
    snapshot_label: str | None,
    static_rag_enabled: bool,
    retrieved: list[dict[str, Any]],
    prompt_block_chars: int,
    record_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "memory_schema_version": MEMORY_SCHEMA_VERSION,
        "kind": "retrieval",
        "retrieval_id": new_record_id("retr"),
        "record_id": record_id or new_record_id("retrieval"),
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "task_id": int(task_id),
        "round_num": round_num,
        "worker_id": worker_id,
        "query": query,
        "compatibility": compatibility,
        "memory_mode": memory_mode,
        "memory_version": memory_version,
        "frozen": frozen,
        "snapshot_label": snapshot_label,
        "static_rag_enabled": static_rag_enabled,
        "retrieved": retrieved,
        "prompt_block_chars": prompt_block_chars,
    }
