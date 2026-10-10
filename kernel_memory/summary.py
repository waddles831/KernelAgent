"""Offline, rule-based construction of strategy summaries.

This pass turns raw attempt evidence into a small number of versioned,
evidence-linked lessons.  It is deterministic and uses **only** measured
telemetry: ``local_speedup`` (parent-relative, optimizer-internal
``do_bench``), the internal verification outcome, and explicit failure
categories.  No LLM is consulted, so nothing here can overwrite a measurement
or promote an unverified candidate.  An optional ``suggested_lesson`` string
may be attached by a caller, but it is stored as interpretation only.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from kernel_memory.features import feature_predicate
from kernel_memory.schema import SUMMARY_SOURCE, stable_hash

BUILDER_VERSION = "v1"


def _action_key(attempt: dict[str, Any]) -> str:
    tags = attempt.get("transformation_tags") or []
    if tags:
        return "+".join(sorted(str(t) for t in tags))
    action = attempt.get("action") or {}
    text = action.get("summary") or action.get("recommended_fix") or action.get("config_change")
    if text:
        return str(text).strip()[:160]
    return "unspecified-action"


def _observed_delta(attempt: dict[str, Any]) -> float | None:
    measurement = attempt.get("measurement") or {}
    if not measurement.get("available"):
        return None
    value = measurement.get("local_speedup")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return value


def build_summaries(
    attempts: list[dict[str, Any]],
    *,
    min_support: int = 1,
    version: str | None = None,
) -> list[dict[str, Any]]:
    """Group attempts and emit one summary per (compat, bottleneck, action)."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        compat_key = (attempt.get("compatibility") or {}).get("compatibility_key") or "unknown"
        bottleneck = attempt.get("bottleneck_category") or "unknown"
        grouped[(compat_key, bottleneck, _action_key(attempt))].append(attempt)

    evidence_hash = stable_hash(
        sorted(
            attempt.get("record_id", "")
            for attempt in attempts
            if attempt.get("record_id")
        )
    )[:16]
    resolved_version = version or f"{BUILDER_VERSION}-{evidence_hash}"

    summaries: list[dict[str, Any]] = []
    for (compat_key, bottleneck, action_key), group in sorted(grouped.items()):
        if len(group) < min_support:
            continue

        support_ids: list[str] = []
        counterexample_ids: list[str] = []
        improvements = 0
        regressions = 0
        failures = 0
        measured: list[float] = []
        task_ids: set[int] = set()
        op_hints: set[str] = set()
        model_classes: set[str] = set()
        compat_example: dict[str, Any] = {}
        profile_statuses: set[str] = set()

        for attempt in group:
            record_id = attempt.get("record_id")
            correctness = attempt.get("correctness_status")
            delta = _observed_delta(attempt)
            if correctness in ("internal_passed", "project_passed"):
                if record_id:
                    support_ids.append(record_id)
            else:
                if record_id:
                    counterexample_ids.append(record_id)
            if delta is not None:
                measured.append(delta)
                if delta > 1.0:
                    improvements += 1
                elif delta < 1.0:
                    regressions += 1
            if correctness in ("internal_failed", "project_failed"):
                failures += 1
            task_ids.add(int(attempt.get("task_id", -1)))
            features = attempt.get("task_features") or {}
            op_hints.update(features.get("op_hints") or [])
            if features.get("model_class"):
                model_classes.add(features["model_class"])
            compat_example = attempt.get("compatibility") or compat_example
            profile_statuses.add((attempt.get("profile") or {}).get("status", "unknown"))

        mean_delta = round(sum(measured) / len(measured), 4) if measured else None
        denom = improvements + regressions + failures
        confidence = round(improvements / denom, 4) if denom else 0.0

        application_ops = sorted(op_hints)
        predicate = feature_predicate(
            {"op_hints": application_ops, "model_class": sorted(model_classes)[0] if len(model_classes) == 1 else None}
        )
        applicability = (
            f"compatible_key={compat_key}; bottleneck={bottleneck}; "
            f"ops={application_ops or ['unknown']}; "
            f"profile={'/'.join(sorted(profile_statuses)) or 'unknown'}"
        )
        caveat_bits = [
            "parent-relative optimizer telemetry only; not the project evaluator",
            "revalidate on the target kernel",
        ]
        if mean_delta is None:
            caveat_bits.append("no valid paired measurements; effect is unquantified")
        if failures:
            caveat_bits.append(f"{failures} failed attempt(s) in support group")
        if profile_statuses and profile_statuses <= {"unavailable", "unknown", "not_applicable"}:
            caveat_bits.append("no NCU profile available")

        summary_id = f"sum_{stable_hash([compat_key, bottleneck, action_key, resolved_version])[:16]}"

        summaries.append(
            {
                "schema_version": 1,
                "memory_schema_version": 1,
                "kind": "summary",
                "summary_id": summary_id,
                "version": resolved_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": SUMMARY_SOURCE,
                "llm_judgment_used": False,
                "compatibility_key": compat_key,
                "compatibility": compat_example,
                "task_feature_predicate": predicate,
                "bottleneck_category": bottleneck,
                "action": action_key,
                "transformation_tags": sorted(
                    {tag for a in group for tag in (a.get("transformation_tags") or [])}
                ),
                "applicability": applicability,
                "observed_effect": {
                    "mean_local_speedup": mean_delta,
                    "support": len(support_ids),
                    "improvements": improvements,
                    "regressions": regressions,
                    "failures": failures,
                    "source_task_ids": sorted(t for t in task_ids if t >= 0),
                },
                "support_ids": support_ids,
                "counterexample_ids": counterexample_ids,
                "confidence": confidence,
                "caveat": "; ".join(caveat_bits),
                "evidence_hash": evidence_hash,
                "measurement_basis": "measured parent-relative optimizer telemetry only",
                "suggested_lesson": None,
            }
        )
    return summaries
