"""Tests for seed-generation cross-task memory retrieval (CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kernel_memory.integration import detect_environment
from kernel_memory.store import MemoryStore
from kernel_memory.summary import build_summaries
from tests.memory_helpers import make_attempt, make_task_result

_PROBLEM = '''\
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)


def get_inputs():
    return [torch.randn(64, 64)]


def get_init_inputs():
    return []
'''


def test_generation_prompt_omits_context_when_absent():
    from triton_kernel_agent.prompt_manager import PromptManager

    pm = PromptManager()
    baseline = pm.render_kernel_generation_prompt(
        problem_description="desc", test_code="test", no_cusolver=False
    )
    assert "MEASURED CROSS-TASK EXPERIENCE" not in baseline
    with_context = pm.render_kernel_generation_prompt(
        problem_description="desc",
        test_code="test",
        no_cusolver=False,
        retrieval_context="MEASURED CROSS-TASK EXPERIENCE (memory_version=v1):\n[M1]",
    )
    assert "MEASURED CROSS-TASK EXPERIENCE" in with_context
    assert with_context.count("MEASURED CROSS-TASK EXPERIENCE") == 1
    # Without a context the prompt is deterministic and unchanged.
    assert (
        pm.render_kernel_generation_prompt(
            problem_description="desc", test_code="test", no_cusolver=False
        )
        == baseline
    )


def test_auto_router_stores_retrieval_context():
    from Fuser.auto_agent import AutoKernelRouter

    router = AutoKernelRouter(retrieval_context="MEASURED BLOCK")
    assert router.retrieval_context == "MEASURED BLOCK"
    assert AutoKernelRouter().retrieval_context is None


def _write_generation_config(directory: Path, *, mode: str, store_dir: Path) -> Path:
    path = directory / "memory_config.json"
    path.write_text(
        json.dumps(
            {
                "mode": mode,
                "store_dir": str(store_dir),
                "snapshot_dir": str(store_dir) if mode == "frozen" else None,
                "max_items": 3,
                "include_failures": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_generation_memory_off(tmp_path: Path):
    from kernelbench_runner.generate_one import _build_generation_memory

    problem = tmp_path / "problem.py"
    problem.write_text(_PROBLEM, encoding="utf-8")
    config = _write_generation_config(tmp_path, mode="off", store_dir=tmp_path / "mem")
    args = SimpleNamespace(
        memory_config=config,
        run_id="r",
        task_id=1,
        task_name="1_task.py",
        problem=problem,
        model="test-model",
    )
    bridge, context = _build_generation_memory(args)
    assert bridge is None
    assert context is None


def test_build_generation_memory_retrieves(tmp_path: Path):
    from kernelbench_runner.generate_one import _build_generation_memory

    problem = tmp_path / "problem.py"
    problem.write_text(_PROBLEM, encoding="utf-8")

    compat = detect_environment(model="test-model")
    store_dir = tmp_path / "snap"
    store = MemoryStore(store_dir)
    store.append_attempts([make_attempt(task_id=40, compatibility=compat)])
    store.append_summaries(build_summaries(store.read_attempts()))

    config = _write_generation_config(tmp_path, mode="frozen", store_dir=store_dir)
    args = SimpleNamespace(
        memory_config=config,
        run_id="r",
        task_id=1,
        task_name="1_task.py",
        problem=problem,
        model="test-model",
    )
    bridge, context = _build_generation_memory(args)
    assert bridge is not None and bridge.enabled
    assert context is not None
    assert "MEASURED CROSS-TASK EXPERIENCE" in context
    assert bridge.retrieval_records
    record = bridge.retrieval_records[0]
    assert record["memory_mode"] == "frozen"
    assert record["frozen"] is True


def test_generation_session_dirs_handles_both_routes(tmp_path: Path):
    from kernelbench_runner.runner import _generation_session_dirs

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {"session_dir": "/a"},
                {"session_dir": "/a"},
                {"session_dir": "/b"},
                {"session_dir": None},
            ]
        ),
        encoding="utf-8",
    )
    fuser_dirs = _generation_session_dirs(
        {"details": {"kernels_summary": str(summary_path)}}
    )
    assert [str(p) for p in fuser_dirs] == ["/a", "/b"]
    direct = _generation_session_dirs({"details": {"session_dir": "/c"}})
    assert [str(p) for p in direct] == ["/c"]
    assert _generation_session_dirs(None) == []


def test_ingest_reads_generation_retrievals(tmp_path: Path):
    from kernelbench_runner.runner import _ingest_task_memory
    from kernel_memory.store import MemoryConfig

    # Generation retrieval log produced by generate_one.
    retrieval_path = tmp_path / "generation_memory_retrievals.jsonl"
    retrieval_path.write_text(
        json.dumps(
            {
                "kind": "retrieval",
                "memory_version": "genv1",
                "task_id": 1,
                "retrieved": [{"kind": "summary"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    write_store = MemoryStore(tmp_path / "mem")
    config = MemoryConfig(mode="collect", store_dir=str(tmp_path / "mem"))
    info = _ingest_task_memory(
        memory_config=config,
        run_id="r",
        task_id=1,
        task_name="1_task.py",
        task_path=tmp_path / "problem.py",
        result_dict=make_task_result(task_id=1),
        agent_metadata=None,
        optimization_metadata=None,
        compatibility=detect_environment(model="m"),
        write_store=write_store,
        generation_retrievals_path=retrieval_path,
    )
    assert info["generation_retrieval_records"] == 1
    assert info["memory_version_used"] == "genv1"
    assert len(write_store.read_retrievals()) == 1
