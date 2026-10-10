"""Integration helpers that glue the memory store to the run pipeline.

``MemoryBridge`` runs inside an optimization worker: it reads a frozen or live
store, retrieves bounded measured experience after the diagnosis is available,
and records what it retrieved.  It never writes to the store.

Everything here is inert when memory is disabled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kernel_memory.features import (
    compatibility_from_environment,
    extract_task_features,
)
from kernel_memory.retrieval import build_experience_context, retrieve_measured
from kernel_memory.schema import retrieval_record
from kernel_memory.store import MemoryConfig, MemoryStore


def make_memory_config(payload: dict[str, Any] | None) -> MemoryConfig | None:
    """Build a :class:`MemoryConfig` from a plain dict, or ``None`` for off."""
    if not payload:
        return None
    if str(payload.get("mode", "off")).strip().lower() == "off":
        return None
    config = MemoryConfig(
        mode=payload.get("mode", "off"),
        store_dir=payload.get("store_dir") or str(payload.get("write_dir") or "memory"),
        snapshot_dir=payload.get("snapshot_dir"),
        snapshot_label=payload.get("snapshot_label"),
        max_items=int(payload.get("max_items", 3)),
        prompt_budget_chars=int(payload.get("prompt_budget_chars", 4000)),
        include_failures=bool(payload.get("include_failures", True)),
        allow_foreign_compat=bool(payload.get("allow_foreign_compat", False)),
        record_generation_traces=bool(payload.get("record_generation_traces", False)),
    )
    config.validate()
    return config


def build_compatibility(
    *,
    gpu_name: str | None,
    torch_version: str | None,
    triton_version: str | None,
    cuda_version: str | None = None,
    gpu_arch: list[str] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    return compatibility_from_environment(
        gpu_name=gpu_name,
        torch_version=torch_version,
        triton_version=triton_version,
        cuda_version=cuda_version,
        gpu_arch=gpu_arch,
        model=model,
    )


def detect_environment(model: str | None = None, gpu_arch: list[str] | None = None) -> dict[str, Any]:
    """Best-effort toolchain fingerprint; unknown fields stay ``None``."""
    gpu_name = torch_version = triton_version = cuda_version = None
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
    return build_compatibility(
        gpu_name=gpu_name,
        torch_version=torch_version,
        triton_version=triton_version,
        cuda_version=cuda_version,
        gpu_arch=gpu_arch,
        model=model,
    )


def task_context(problem_path: Path | str, *, task_id: int, task_name: str) -> dict[str, Any]:
    return {
        "task_id": int(task_id),
        "task_name": task_name,
        "task_features": extract_task_features(problem_path),
    }


class MemoryBridge:
    """Worker-side, read-only memory retrieval + retrieval logging."""

    def __init__(self, payload: dict[str, Any], *, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = make_memory_config(payload)
        if self.config is None or not self.config.retrieval_enabled():
            self.enabled = False
            self.store = None
            self.memory_version = None
            self.snapshot_label = None
            self.retrieval_records: list[dict[str, Any]] = []
            return
        self.enabled = True
        self.store = MemoryStore(self.config.retrieval_dir(), read_only=True)
        self.memory_version = self.store.memory_version()
        version_info = self.store.read_version_info()
        self.snapshot_label = version_info.get("snapshot_label") or payload.get("snapshot_label")
        self.compatibility = payload.get("compatibility") or {}
        self.task_features = payload.get("task_features") or {}
        self.task_id = int(payload.get("task_id", -1))
        self.task_name = payload.get("task_name", "")
        self.run_id = payload.get("run_id", "unknown")
        self.static_rag_enabled = bool(payload.get("static_rag_enabled", False))
        self.retrieval_records = []

    def build_context(
        self,
        *,
        bottleneck_category: str,
        bottleneck_summary: str = "",
        round_num: int | None = None,
        worker_id: int | None = None,
    ) -> str:
        if not self.enabled or self.store is None:
            return ""
        items, stats = retrieve_measured(
            self.store,
            compatibility=self.compatibility,
            task_features=self.task_features,
            bottleneck_category=bottleneck_category,
            current_task_id=self.task_id,
            max_items=self.config.max_items,
            include_failures=self.config.include_failures,
            allow_foreign_compat=self.config.allow_foreign_compat,
        )
        context = build_experience_context(
            items,
            memory_version=self.memory_version,
            frozen=self.config.is_frozen(),
            compatibility=self.compatibility,
            budget_chars=self.config.prompt_budget_chars,
        )
        record = retrieval_record(
            run_id=self.run_id,
            task_id=self.task_id,
            round_num=round_num,
            worker_id=worker_id,
            query={
                "bottleneck_category": bottleneck_category,
                "bottleneck_summary": bottleneck_summary[:400],
            },
            compatibility=self.compatibility,
            memory_mode=self.config.normalized_mode(),
            memory_version=self.memory_version,
            frozen=self.config.is_frozen(),
            snapshot_label=self.snapshot_label,
            static_rag_enabled=self.static_rag_enabled,
            retrieved=[
                {
                    "kind": item.get("kind"),
                    "source_id": item.get("source_id"),
                    "score": item.get("score"),
                    "compatibility_match": item.get("compatibility_match"),
                    "bottleneck_match": item.get("bottleneck_match"),
                    "action": item.get("action"),
                    "evidence_ids": item.get("evidence_ids"),
                }
                for item in items
            ],
            prompt_block_chars=len(context),
        )
        record["_retrieval_stats"] = stats
        self.retrieval_records.append(record)
        if items:
            self.logger.info(
                f"[memory] retrieved {len(items)} measured item(s) "
                f"(version={self.memory_version}, frozen={self.config.is_frozen()})"
            )
        return context

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.config.normalized_mode() if self.config else "off",
            "memory_version": self.memory_version,
            "snapshot_label": self.snapshot_label,
            "retrieval_count": len(self.retrieval_records),
        }
