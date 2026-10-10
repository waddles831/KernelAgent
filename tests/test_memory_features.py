"""Tests for static feature extraction and compatibility keys (CPU-only)."""

from __future__ import annotations

from pathlib import Path

from kernel_memory.features import (
    compatibility_from_environment,
    compatibility_key,
    extract_task_features,
    task_similarity,
)

_PROBLEM = '''\
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch.matmul(x, x.transpose(0, 1)) + self.weight


def get_init_inputs():
    return [64]


def get_inputs():
    return [torch.randn(64, 64)]
'''


def test_extract_task_features_marks_unknowns(tmp_path: Path):
    path = tmp_path / "problem.py"
    path.write_text(_PROBLEM, encoding="utf-8")
    features = extract_task_features(path)
    assert features["model_class"] == "Model"
    assert features["source_sha256"]
    assert "matmul" in features["op_hints"]
    assert "unknown_features" in features
    assert "input_shapes" in features["unknown_features"]
    # No runtime shape should be fabricated.
    assert "input_shapes" not in features or features.get("input_shapes") is None


def test_extract_task_features_handles_missing_file(tmp_path: Path):
    features = extract_task_features(tmp_path / "absent.py")
    assert features["source_sha256"] is None
    assert "extraction_error" in features


def test_compatibility_key_ignores_model_name():
    a = compatibility_from_environment(
        gpu_name="NVIDIA A30",
        torch_version="2.9.1",
        triton_version="3.5.1",
        cuda_version="12.2",
        gpu_arch=["Ampere"],
        model="model-a",
    )
    b = compatibility_from_environment(
        gpu_name="NVIDIA A30",
        torch_version="2.9.1",
        triton_version="3.5.1",
        cuda_version="12.2",
        gpu_arch=["Ampere"],
        model="model-b",
    )
    assert a["compatibility_key"] == b["compatibility_key"]


def test_compatibility_key_distinguishes_hardware():
    a = compatibility_from_environment(
        gpu_name="NVIDIA A30", torch_version="2.9.1", triton_version="3.5.1"
    )
    b = compatibility_from_environment(
        gpu_name="NVIDIA H100", torch_version="2.9.1", triton_version="3.5.1"
    )
    assert a["compatibility_key"] != b["compatibility_key"]


def test_compatibility_marks_unknown_fields():
    compat = compatibility_from_environment(
        gpu_name=None, torch_version="2.9.1", triton_version=None
    )
    assert "gpu_name" in compat["unknown_fields"]
    assert "triton_version" in compat["unknown_fields"]


def test_task_similarity_is_explainable():
    a = {"source_sha256": "x", "op_hints": ["matmul", "relu"], "model_class": "A"}
    b = {"source_sha256": "x", "op_hints": ["matmul", "relu"], "model_class": "A"}
    c = {"source_sha256": "y", "op_hints": ["softmax"], "model_class": "B"}
    assert task_similarity(a, b) == 1.0
    assert task_similarity(a, c) < 0.5


def test_compatibility_key_helper_matches():
    compat = compatibility_from_environment(
        gpu_name="NVIDIA A30", torch_version="2.9.1", triton_version="3.5.1"
    )
    assert compatibility_key(compat) == compat["compatibility_key"]
