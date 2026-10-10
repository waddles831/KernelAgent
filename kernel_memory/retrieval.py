"""Bounded retrieval of *measured* cross-task experience.

Retrieval ranking is intentionally simple, explainable and hardware-scoped:

1. Compatibility: measured performance is only reused when the compatibility
   key matches (same GPU/toolchain).  Foreign-environment items are admitted
   as qualitative hypotheses only when explicitly allowed, and are always
   labeled for mandatory local revalidation.
2. Bottleneck evidence, then task-feature similarity.  ``unknown`` is used when
   profiling was unavailable; nothing is derived from latency alone.
3. Evidence quality (support count / measurement availability), while always
   keeping room for regressions.
4. Anti-leakage: items produced by the *current* task are excluded by default
   so a task cannot be helped by its own past solution.  This is reported as a
   separate "same-task reuse" case rather than unseen-task transfer.
"""

from __future__ import annotations

from typing import Any

from kernel_memory.features import compatibility_key, task_similarity
from kernel_memory.store import MemoryStore

_CLAIM_LABEL = "MEASURED CROSS-TASK EXPERIENCE"


def _record_compat_key(compat: dict[str, Any] | None) -> str | None:
    """Derive the compatibility key from stored fields.

    Deriving rather than trusting a stored ``compatibility_key`` keeps old
    records matchable when the key definition changes.
    """
    if not compat:
        return None
    return compatibility_key(compat)


def _item_from_summary(
    summary: dict[str, Any],
    *,
    compat_key: str,
    bottleneck: str,
    task_features: dict[str, Any],
    current_task_id: int,
) -> dict[str, Any]:
    evidence_ids = list(summary.get("support_ids") or []) + list(
        summary.get("counterexample_ids") or []
    )
    effect = summary.get("observed_effect") or {}
    observed_bits = []
    if effect.get("mean_local_speedup") is not None:
        observed_bits.append(f"mean local speedup {effect['mean_local_speedup']:.3f}x")
    observed_bits.append(f"support={effect.get('support', 0)}")
    observed_bits.append(f"regressions={effect.get('regressions', 0)}")
    observed_bits.append(f"failures={effect.get('failures', 0)}")
    predicate = summary.get("task_feature_predicate") or {}
    sim = task_similarity(task_features, {
        "source_sha256": None,
        "op_hints": predicate.get("op_hints"),
        "model_class": predicate.get("model_class"),
    })
    source_task_ids = effect.get("source_task_ids") or []
    record_key = _record_compat_key(summary.get("compatibility"))
    return {
        "kind": "summary",
        "source_id": summary.get("summary_id"),
        "version": summary.get("version"),
        "compatibility_match": record_key == compat_key,
        "compatibility_key": record_key,
        "bottleneck_match": (summary.get("bottleneck_category") or "").lower()
        == (bottleneck or "").lower(),
        "bottleneck_category": summary.get("bottleneck_category"),
        "task_similarity": sim,
        "same_task": int(current_task_id) in [int(t) for t in source_task_ids],
        "source_task_ids": source_task_ids,
        "action": summary.get("action"),
        "transformation_tags": summary.get("transformation_tags") or [],
        "applicability": summary.get("applicability"),
        "observed": "; ".join(observed_bits),
        "caveat": summary.get("caveat"),
        "confidence": summary.get("confidence"),
        "support": effect.get("support", 0),
        "regressions": effect.get("regressions", 0),
        "evidence_ids": evidence_ids,
        "correctness_status": "measured (internal verification gate)",
        "measurement_available": effect.get("mean_local_speedup") is not None,
        "is_failure": int(effect.get("support", 0)) == 0,
    }


def _item_from_attempt(
    attempt: dict[str, Any],
    *,
    compat_key: str,
    bottleneck: str,
    task_features: dict[str, Any],
    current_task_id: int,
) -> dict[str, Any]:
    measurement = attempt.get("measurement") or {}
    local_speedup = measurement.get("local_speedup")
    observed_bits = []
    if measurement.get("available") and local_speedup is not None:
        observed_bits.append(f"local speedup {local_speedup:.3f}x")
    else:
        observed_bits.append("no valid measurement")
    observed_bits.append(
        f"correctness={attempt.get('correctness_status', 'unknown')}"
    )
    failure = attempt.get("failure_category", "none")
    if failure and failure != "none":
        observed_bits.append(f"failure={failure}")
    action = attempt.get("action") or {}
    action_text = action.get("summary") or action.get("recommended_fix") or action.get("config_change")
    record_key = _record_compat_key(attempt.get("compatibility"))
    return {
        "kind": "attempt",
        "source_id": attempt.get("record_id"),
        "version": None,
        "compatibility_match": record_key == compat_key,
        "compatibility_key": record_key,
        "bottleneck_match": (attempt.get("bottleneck_category") or "").lower()
        == (bottleneck or "").lower(),
        "bottleneck_category": attempt.get("bottleneck_category"),
        "task_similarity": task_similarity(task_features, attempt.get("task_features") or {}),
        "same_task": int(attempt.get("task_id", -1)) == int(current_task_id),
        "source_task_ids": [attempt.get("task_id")],
        "action": action_text,
        "transformation_tags": attempt.get("transformation_tags") or [],
        "applicability": _attempt_applicability(attempt),
        "observed": "; ".join(observed_bits),
        "caveat": (
            "single attempt; parent-relative optimizer telemetry, not the "
            "project evaluator"
        ),
        "confidence": None,
        "support": 1,
        "regressions": 1 if (measurement.get("available") and local_speedup is not None and local_speedup < 1.0) else 0,
        "evidence_ids": [attempt.get("record_id")],
        "correctness_status": attempt.get("correctness_status", "unknown"),
        "measurement_available": bool(measurement.get("available"))
        and local_speedup is not None,
        "is_failure": (
            attempt.get("correctness_status") in ("internal_failed", "project_failed")
            or not bool(measurement.get("available"))
        ),
    }


def _item_from_task_result(
    record: dict[str, Any],
    *,
    compat_key: str,
    task_features: dict[str, Any],
    current_task_id: int,
) -> dict[str, Any]:
    speedup = record.get("task_speedup_vs_torch_compile")
    source_features = record.get("task_features") or {}
    observed_bits = [f"final status={record.get('evaluation_status')}"]
    if speedup is not None:
        observed_bits.append(f"task speedup vs torch.compile {float(speedup):.3f}x")
    ops = ", ".join(source_features.get("op_hints") or []) or "unknown ops"
    record_key = _record_compat_key(record.get("compatibility"))
    return {
        "kind": "task_result",
        "source_id": record.get("record_id"),
        "version": None,
        "compatibility_match": record_key == compat_key,
        "compatibility_key": record_key,
        "bottleneck_match": False,
        "bottleneck_category": None,
        "task_similarity": task_similarity(task_features, source_features),
        "same_task": int(record.get("task_id", -1)) == int(current_task_id),
        "source_task_ids": [record.get("task_id")],
        "action": "final verified program for a similar task",
        "transformation_tags": [],
        "applicability": f"ops=[{ops}]; route={record.get('route')}",
        "observed": "; ".join(observed_bits),
        "caveat": (
            "task-level project-evaluator result (vs torch.compile); not a "
            "per-kernel action and not parent-relative"
        ),
        "confidence": None,
        "support": 1,
        "regressions": 1 if (speedup is not None and float(speedup) < 1.0) else 0,
        "evidence_ids": [record.get("record_id")],
        "correctness_status": record.get("correctness_status", "unknown"),
        "measurement_available": speedup is not None,
        "is_failure": (
            record.get("correctness_status") in ("project_failed",)
            or speedup is None
        ),
    }


def _attempt_applicability(attempt: dict[str, Any]) -> str:
    features = attempt.get("task_features") or {}
    ops = ", ".join(features.get("op_hints") or []) or "unknown ops"
    profile = attempt.get("profile") or {}
    status = profile.get("status", "unknown")
    return f"ops=[{ops}]; profile={status}; bottleneck={attempt.get('bottleneck_category') or 'unknown'}"


def _score(item: dict[str, Any], *, compat_key: str) -> float:
    score = 0.0
    if item.get("compatibility_match"):
        score += 3.0
    if item.get("bottleneck_match"):
        score += 2.0
    score += 1.5 * float(item.get("task_similarity") or 0.0)
    if item.get("kind") == "summary":
        support = int(item.get("support") or 0)
        confidence = float(item.get("confidence") or 0.0)
        score += min(1.0, support / 5.0) + confidence
    if item.get("measurement_available"):
        score += 0.5
    return round(score, 6)


def retrieve_measured(
    store: MemoryStore,
    *,
    compatibility: dict[str, Any],
    task_features: dict[str, Any],
    bottleneck_category: str,
    current_task_id: int,
    max_items: int = 3,
    include_failures: bool = True,
    allow_foreign_compat: bool = False,
    allow_same_task: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ``(items, stats)`` with bounded, ranked measured experience."""
    compat_key = compatibility.get("compatibility_key")
    summaries = store.read_summaries()
    attempts = store.read_attempts()
    task_results = store.read_task_results()

    candidates: list[dict[str, Any]] = []
    for record in task_results:
        candidates.append(
            _item_from_task_result(
                record,
                compat_key=compat_key,
                task_features=task_features,
                current_task_id=current_task_id,
            )
        )
    for summary in summaries:
        candidates.append(
            _item_from_summary(
                summary,
                compat_key=compat_key,
                bottleneck=bottleneck_category,
                task_features=task_features,
                current_task_id=current_task_id,
            )
        )
    for attempt in attempts:
        candidates.append(
            _item_from_attempt(
                attempt,
                compat_key=compat_key,
                bottleneck=bottleneck_category,
                task_features=task_features,
                current_task_id=current_task_id,
            )
        )

    stats: dict[str, Any] = {
        "total_candidates": len(candidates),
        "excluded_same_task": 0,
        "excluded_incompatible": 0,
        "excluded_failures": 0,
        "foreign_used": False,
    }

    filtered: list[dict[str, Any]] = []
    for item in candidates:
        if item.get("same_task") and not allow_same_task:
            stats["excluded_same_task"] += 1
            continue
        if not item.get("compatibility_match"):
            if not allow_foreign_compat:
                stats["excluded_incompatible"] += 1
                continue
            stats["foreign_used"] = True
            item = dict(item)
            item["compatibility_match"] = False
            item["caveat"] = (
                "FOREIGN environment/toolchain; treat as a hypothesis and "
                "revalidate locally. " + (item.get("caveat") or "")
            )
        if not include_failures and item.get("is_failure"):
            stats["excluded_failures"] += 1
            continue
        filtered.append(item)

    # Deduplicate near-identical items by (kind, bottleneck, action).
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in sorted(filtered, key=lambda i: _score(i, compat_key=compat_key), reverse=True):
        if item.get("kind") == "task_result":
            key = ("task_result", str(item.get("source_id")), "")
        else:
            key = (
                str(item.get("kind")),
                str(item.get("bottleneck_category")),
                str(item.get("action"))[:120],
            )
        if key in seen:
            continue
        seen.add(key)
        item["score"] = _score(item, compat_key=compat_key)
        deduped.append(item)

    return deduped[:max_items], stats


def build_experience_context(
    items: list[dict[str, Any]],
    *,
    memory_version: str | None,
    frozen: bool,
    compatibility: dict[str, Any],
    budget_chars: int = 4000,
) -> str:
    """Render measured experience as a clearly-labeled prompt block."""
    if not items:
        return ""

    gpu = compatibility.get("gpu_name") or "unknown GPU"
    header = (
        f"{_CLAIM_LABEL} (memory_version={memory_version or 'unknown'}, "
        f"frozen={'yes' if frozen else 'no'}, environment={gpu}):\n"
        "These are real measurements from prior KernelAgent optimization runs. "
        "They are conditional and are NOT guaranteed to transfer; revalidate "
        "the idea against this kernel's own measurements before trusting it. "
        "Failed and regression evidence is included deliberately.\n"
    )
    lines = [header]
    for index, item in enumerate(items, start=1):
        evidence = ", ".join(str(e) for e in (item.get("evidence_ids") or [])[:6])
        lines.append(
            f"[M{index}] kind={item.get('kind')} bottleneck={item.get('bottleneck_category')} "
            f"action={item.get('action')}\n"
            f"    observed: {item.get('observed')}\n"
            f"    applicability: {item.get('applicability')}\n"
            f"    caveat: {item.get('caveat')}\n"
            f"    evidence_ids: [{evidence}]"
        )
    lines.append(
        "\nNOTE: static RAG code examples (if any) are provided separately and "
        "are general patterns, not measured experience."
    )
    text = "\n".join(lines)
    if budget_chars and len(text) > budget_chars:
        text = text[:budget_chars] + "\n... (experience block truncated)"
    return text
