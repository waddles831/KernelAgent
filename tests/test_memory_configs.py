"""Validate the shipped memory experiment configs (CPU-only, no execution)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernelbench_runner.config import RunnerConfig

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
_CONFIGS = sorted(_CONFIG_DIR.glob("memory_*.json"))


@pytest.mark.parametrize("path", _CONFIGS, ids=[p.name for p in _CONFIGS])
def test_memory_configs_parse_and_validate(path: Path):
    assert path.exists()
    config = RunnerConfig.from_json(path)
    config.validate()
    # 5-task baseline used for existing-task retest; unseen eval must be disjoint.
    if "continual_retest" in path.name:
        assert config.memory_mode == "continual"
        assert set(config.task_ids or []) == {1, 40, 50, 88, 95}
    if "frozen_unseen_eval" in path.name:
        assert config.memory_mode == "frozen"
        assert config.memory_snapshot_dir
        assert not set(config.task_ids or []) & {1, 40, 50, 88, 95}
    if "control_off" in path.name:
        assert config.memory_config() is None


def test_control_and_frozen_share_eval_tasks():
    names = {p.name: RunnerConfig.from_json(p) for p in _CONFIGS}
    control = names["memory_control_off.json"]
    frozen = names["memory_frozen_unseen_eval.json"]
    assert set(control.task_ids or []) == set(frozen.task_ids or [])
