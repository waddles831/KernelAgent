"""Integration tests: collection, bridge, runner ingestion, CLI (CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path

from kernel_memory.collector import (
    attempt_records_from_optimization_evidence,
    classify_error,
    task_result_record_from_result,
)
from kernel_memory.integration import MemoryBridge, make_memory_config
from kernel_memory.store import MemoryConfig, MemoryStore
from tests.memory_helpers import (
    compat,
    make_attempt,
    make_task_result,
    task_features,
    write_evidence,
)


def test_classify_error_categories():
    assert classify_error("CUDA out of memory") == "oom"
    assert classify_error("timed out after 10s") == "timeout"
    assert classify_error("values differ (max diff ...)") == "correctness_failed"
    assert classify_error("") == "unknown"


def test_collector_marks_measurement_only_when_verified():
    evidence = [
        {
            "worker_id": 0,
            "round_num": 1,
            "parent_id": "initial",
            "kernel_code": "K",
            "time_ms": 1.0,
            "success": True,
            "attempts": [
                {
                    "round_num": 1,
                    "worker_id": 0,
                    "bottleneck_category": "memory",
                    "recommended_fix": "num_warps 4->8",
                    "config_changes": {"num_warps": "4->8"},
                    "time_before_ms": 1.0,
                    "time_after_ms": 0.5,
                    "improvement_pct": 50.0,
                    "is_improvement": True,
                    "passed_verification": True,
                    "error_message": "",
                    "combined_sol_pct": 60.0,
                },
                {
                    "round_num": 1,
                    "worker_id": 0,
                    "bottleneck_category": "compute",
                    "recommended_fix": "reorder",
                    "time_before_ms": 1.0,
                    "time_after_ms": 0.0,
                    "improvement_pct": 0.0,
                    "is_improvement": False,
                    "passed_verification": False,
                    "error_message": "shape mismatch",
                },
            ],
            "reflexions": [],
        }
    ]
    records = attempt_records_from_optimization_evidence(
        evidence,
        run_id="r",
        task_id=1,
        task_name="1_task.py",
        task_features=task_features(),
        compatibility=compat(),
        memory_version="v1",
    )
    assert len(records) == 2
    passed, failed = records
    assert passed["correctness_status"] == "internal_passed"
    assert passed["measurement"]["available"] is True
    assert passed["measurement"]["local_speedup"] == 2.0
    assert passed["measurement"]["parent_relative"] is True
    assert passed["profile"]["status"] == "available"

    assert failed["correctness_status"] == "internal_failed"
    assert failed["measurement"]["available"] is False
    assert failed["measurement"]["local_speedup"] is None
    assert failed["failure_category"] == "correctness_failed"
    assert failed["candidate_sha256"] is None
    # The worker exports only the best kernel; with a later failed attempt the
    # best kernel is the parent, so the candidate hash is honestly absent.
    assert passed["candidate_sha256"] is None


def test_collector_maps_candidate_when_last_improved():
    evidence = [
        {
            "worker_id": 0,
            "round_num": 1,
            "parent_id": "initial",
            "kernel_code": "def k(): pass",
            "time_ms": 0.5,
            "success": True,
            "attempts": [
                {
                    "round_num": 1,
                    "worker_id": 0,
                    "bottleneck_category": "memory",
                    "recommended_fix": "num_warps 4->8",
                    "time_before_ms": 1.0,
                    "time_after_ms": 0.5,
                    "is_improvement": True,
                    "passed_verification": True,
                    "error_message": "",
                }
            ],
            "reflexions": [],
        }
    ]
    records = attempt_records_from_optimization_evidence(
        evidence,
        run_id="r",
        task_id=1,
        task_name="1_task.py",
        task_features=task_features(),
        compatibility=compat(),
        memory_version="v1",
    )
    assert records[0]["parent_id"] == "initial"
    assert records[0]["candidate_sha256"]
    assert records[0]["candidate_id"].startswith("initial->")


def test_task_result_record_exposes_both_speedups():
    record = task_result_record_from_result(
        make_task_result(task_id=3, speedup=1.5),
        run_id="r",
        task_id=3,
        task_name="3_task.py",
        task_features=task_features(),
        compatibility=compat(),
        memory_version_used="v1",
        memory_mode_used="frozen",
    )
    assert record["task_speedup_vs_torch_compile"] == 1.5
    assert record["parent_relative_speedup"] is None
    assert record["evaluation_scope"] == "project_evaluator"
    assert record["memory_mode_used"] == "frozen"


def test_make_memory_config_off_returns_none():
    assert make_memory_config(None) is None
    assert make_memory_config({"mode": "off"}) is None


def test_memory_bridge_retrieves_and_logs(tmp_path: Path):
    read_store = MemoryStore(tmp_path / "snap")
    read_store.append_attempts([make_attempt(task_id=1)])
    from kernel_memory.summary import build_summaries

    read_store.append_summaries(build_summaries(read_store.read_attempts()))

    payload = {
        "mode": "frozen",
        "store_dir": str(tmp_path / "snap"),
        "snapshot_dir": str(tmp_path / "snap"),
        "task_id": 99,
        "task_name": "99_task.py",
        "run_id": "r",
        "compatibility": compat(),
        "task_features": task_features(),
        "max_items": 3,
    }
    bridge = MemoryBridge(payload)
    assert bridge.enabled
    context = bridge.build_context(bottleneck_category="memory", round_num=1)
    assert "MEASURED CROSS-TASK EXPERIENCE" in context
    assert len(bridge.retrieval_records) == 1
    record = bridge.retrieval_records[0]
    assert record["memory_mode"] == "frozen"
    assert record["frozen"] is True
    assert record["memory_version"]


def test_memory_bridge_disabled_for_collect(tmp_path: Path):
    payload = {
        "mode": "collect",
        "store_dir": str(tmp_path / "mem"),
        "task_id": 1,
        "task_name": "1_task.py",
        "run_id": "r",
        "compatibility": compat(),
        "task_features": task_features(),
    }
    bridge = MemoryBridge(payload)
    assert bridge.enabled is False
    assert bridge.build_context(bottleneck_category="memory") == ""


def test_runner_ingest_task_memory(tmp_path: Path):
    from kernelbench_runner.runner import _ingest_task_memory

    evidence_dir = tmp_path / "opt"
    write_evidence(
        evidence_dir,
        [
            {
                "round_num": 1,
                "worker_id": 0,
                "bottleneck_category": "memory",
                "recommended_fix": "num_warps 4->8",
                "time_before_ms": 1.0,
                "time_after_ms": 0.8,
                "is_improvement": True,
                "passed_verification": True,
                "error_message": "",
            }
        ],
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
        optimization_metadata={"memory": {"evidence_dir": str(evidence_dir)}},
        compatibility=compat(),
        write_store=write_store,
    )
    assert info["optimization_attempts"] == 1
    assert info["memory_version_used"] == "frozenv1"
    assert len(write_store.read_attempts()) == 1
    assert len(write_store.read_task_results()) == 1
    assert len(write_store.read_retrievals()) == 1


def test_runner_config_off_by_default():
    from kernelbench_runner.config import RunnerConfig

    config = RunnerConfig()
    assert config.memory_mode == "off"
    assert config.memory_config() is None


def test_cli_parses_memory_flags():
    from kernelbench_runner.cli import _parser

    args = _parser().parse_args(
        [
            "--memory-mode",
            "frozen",
            "--memory-dir",
            "/tmp/mem",
            "--memory-snapshot-dir",
            "/tmp/snap",
            "--memory-max-items",
            "5",
            "--no-memory-include-failures",
        ]
    )
    assert args.memory_mode == "frozen"
    assert args.memory_dir == "/tmp/mem"
    assert args.memory_snapshot_dir == "/tmp/snap"
    assert args.memory_max_items == 5
    assert args.memory_include_failures is False


def test_memory_cli_report_and_snapshot(tmp_path: Path):
    from kernelbench_runner import memory_cli

    store_dir = tmp_path / "mem"
    store = MemoryStore(store_dir)
    store.append_attempts([make_attempt()])

    assert memory_cli.main(["report", "--store", str(store_dir)]) == 0
    assert (
        memory_cli.main(
            [
                "summarize",
                "--store",
                str(store_dir),
                "--version",
                "unit",
            ]
        )
        == 0
    )
    assert len(MemoryStore(store_dir).read_summaries()) == 1

    dest = tmp_path / "frozen"
    assert (
        memory_cli.main(
            [
                "snapshot",
                "--store",
                str(store_dir),
                "--dest",
                str(dest),
                "--label",
                "unit",
            ]
        )
        == 0
    )
    assert MemoryStore(dest, read_only=True).read_version_info()["snapshot_label"] == "unit"
