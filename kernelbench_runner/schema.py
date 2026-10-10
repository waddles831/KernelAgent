"""Versioned, JSON-serializable per-task result records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TaskResult:
    task_id: int
    task_name: str
    agent_success: bool
    compile_success: bool | None
    runtime_success: bool | None
    correct: bool | None
    baseline_latency_ms: float | None
    candidate_latency_ms: float | None
    speedup: float | None
    iterations: int | None
    candidate_count: int | None
    llm_calls: int | None
    elapsed_time_s: float
    evaluation_status: str
    route: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
