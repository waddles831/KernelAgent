"""Configuration for reproducible KernelAgent benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNELBENCH_ROOT = REPO_ROOT.parent / "KernelBench"


@dataclass
class RunnerConfig:
    experiment_name: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("l1_%Y%m%dT%H%M%SZ")
    )
    kernelbench_path: str = str(DEFAULT_KERNELBENCH_ROOT)
    output_root: str = str(REPO_ROOT / "runs")

    model: str = "gpt-5"
    api_base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"

    gpu_id: int = 0
    gpu_arch: list[str] = field(default_factory=lambda: ["Ampere"])
    cpu_worker_limit: int = 4
    candidate_count: int = 4
    search_budget: int = 5
    beam_width: int = 2

    random_seed: int = 42
    num_correct_trials: int = 5
    benchmark_warmup_ms: float = 1000.0
    benchmark_rep_ms: float = 100.0
    generation_timeout_s: int = 1800
    llm_timeout_s: int = 1200
    agent_test_timeout_s: int = 300
    evaluation_timeout_s: int = 300

    task_ids: list[int] | None = None
    task_range: list[int] | None = None

    @classmethod
    def from_json(cls, path: Path) -> RunnerConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Runner config must be a JSON object")
        # Accept old config files without misreading trial counts as Triton's
        # millisecond repetition budget. The new benchmark settings otherwise
        # take their current defaults.
        legacy_trials = data.pop("num_perf_trials", None)
        legacy_timing = data.pop("timing_method", None)
        if legacy_trials is not None or legacy_timing is not None:
            warnings.warn(
                "Ignoring legacy KernelBench timing settings; this run uses "
                "triton.testing.do_bench. Set benchmark_warmup_ms and "
                "benchmark_rep_ms to configure the current protocol.",
                stacklevel=2,
            )
        if legacy_timing not in (None, "cuda_event", "do_bench"):
            raise ValueError(
                "Legacy timing_method must be cuda_event or do_bench; "
                "the current evaluator uses triton.testing.do_bench"
            )
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown runner config keys: {', '.join(unknown)}")
        return cls(**data)

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", self.experiment_name):
            raise ValueError(
                "experiment_name must be 1–80 safe filename characters and "
                "start with a letter or digit"
            )
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("api_key_env must be an environment variable name")
        if self.gpu_id not in (0, 1):
            raise ValueError("gpu_id must be 0 or 1 for the two-GPU server allocation")
        if not 1 <= self.cpu_worker_limit <= 80:
            raise ValueError("cpu_worker_limit must be between 1 and 80")
        if not 1 <= self.candidate_count <= self.cpu_worker_limit:
            raise ValueError("candidate_count must be between 1 and cpu_worker_limit")
        if self.search_budget < 1 or self.beam_width < 1:
            raise ValueError("search_budget and beam_width must be positive")
        if self.num_correct_trials < 1:
            raise ValueError("correctness trial count must be positive")
        if (
            not math.isfinite(self.benchmark_warmup_ms)
            or not math.isfinite(self.benchmark_rep_ms)
            or self.benchmark_warmup_ms <= 0
            or self.benchmark_rep_ms <= 0
        ):
            raise ValueError("benchmark warmup and repetition budgets must be positive")
        if any(
            value < 1
            for value in (
                self.generation_timeout_s,
                self.llm_timeout_s,
                self.agent_test_timeout_s,
                self.evaluation_timeout_s,
            )
        ):
            raise ValueError("timeout values must be positive seconds")
        if not self.gpu_arch or any(not item.strip() for item in self.gpu_arch):
            raise ValueError("gpu_arch must contain at least one architecture name")
        if self.task_ids is not None:
            if not self.task_ids or any(task_id < 1 for task_id in self.task_ids):
                raise ValueError("task_ids must contain positive IDs")
            if len(set(self.task_ids)) != len(self.task_ids):
                raise ValueError("task_ids must not contain duplicates")
        if self.task_range is not None:
            if len(self.task_range) != 2:
                raise ValueError("task_range must be [start_id, end_id]")
            start, end = self.task_range
            if start < 1 or end < start:
                raise ValueError("task_range must be an inclusive positive ID range")
        if self.task_ids is not None and self.task_range is not None:
            raise ValueError("Choose task_ids or task_range, not both")

    def resolved_kernelbench_root(self) -> Path:
        root = Path(self.kernelbench_path).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        root = root.resolve()
        if root.name == "KernelBench" and (root / "level1").is_dir():
            # Also accept the dataset directory for convenience.
            return root.parent
        return root

    def resolved_output_root(self) -> Path:
        root = Path(self.output_root).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        return root.resolve()

    def public_dict(self) -> dict[str, Any]:
        """Return the run configuration; API key values are never stored here."""
        result = asdict(self)
        result["kernelbench_path"] = str(self.resolved_kernelbench_root())
        result["output_root"] = str(self.resolved_output_root())
        result["secret_policy"] = "API key value is read from api_key_env and is not serialized"
        return result
