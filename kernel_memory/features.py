"""Static task-feature extraction and hardware/toolchain compatibility keys.

Only information that can be derived without running the task is extracted
here.  Runtime facts (shapes, dtypes, layouts/strides, scalar parameters) are
recorded as ``None`` with an explicit reason instead of being guessed.  This is
deliberate: the memory notes forbid manufacturing profile- or shape-derived
state from incomplete data.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from kernel_memory.schema import sha256_text, stable_hash

# Ordered (tag, regex) probes over the task source.  These are *hints* about the
# operations a task references, not verified runtime properties.
_OP_PROBES: tuple[tuple[str, str], ...] = (
    ("matmul", r"torch\.(?:matmul|bmm|einsum|mm)\b"),
    ("linear", r"\b(?:nn\.)?Linear\b"),
    ("conv", r"\b(?:nn\.)?Conv[123]d\b|torch\.conv[123]d\b|F\.conv[123]d\b"),
    ("layernorm", r"LayerNorm|layer_norm"),
    ("batchnorm", r"BatchNorm|batch_norm"),
    ("groupnorm", r"GroupNorm|group_norm"),
    ("softmax", r"\bsoftmax\b"),
    ("log_softmax", r"log_softmax"),
    ("gelu", r"\bGELU\b|\bgelu\b"),
    ("silu", r"\bSiLU\b|\bsilu\b"),
    ("relu", r"\bReLU\b|\brelu\b"),
    ("sigmoid", r"\bsigmoid\b"),
    ("tanh", r"\btanh\b"),
    ("cross_entropy", r"CrossEntropyLoss|cross_entropy"),
    ("nll", r"NLLLoss|nll_loss"),
    ("dropout", r"\bDropout\b|dropout"),
    ("embedding", r"\bEmbedding\b"),
    ("reduction", r"torch\.(?:sum|mean|max|min|prod|amax|amin)\b|\.(?:sum|mean|max|min)\("),
    ("normalization", r"normalize"),
    ("transpose", r"\.transpose\(|\.permute\(|torch\.transpose\b"),
    ("reshape", r"\.view\(|\.reshape\(|\.flatten\(|torch\.reshape\b"),
    ("cat", r"torch\.cat\b"),
    ("pool", r"MaxPool|AvgPool|AdaptiveAvgPool|max_pool|avg_pool"),
)

_UNKNOWN_RUNTIME_FEATURES = (
    "input_shapes",
    "dtypes",
    "layouts_strides",
    "scalar_parameters",
    "numerical_tolerance",
)


def extract_task_features(problem_path: Path | str) -> dict[str, Any]:
    """Return source-derived task features with unknown runtime fields marked."""
    path = Path(problem_path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "source_path": str(path),
            "source_sha256": None,
            "extraction_error": f"{exc.__class__.__name__}: {exc}",
            "op_hints": [],
            "model_class": None,
            "unknown_features": list(_UNKNOWN_RUNTIME_FEATURES),
            "unknown_reason": "task source could not be read",
        }

    op_hints = [tag for tag, pattern in _OP_PROBES if re.search(pattern, source)]
    class_match = re.search(r"class\s+(\w+)\s*\(\s*(?:nn\.)?Module\s*\)", source)
    return {
        "source_path": str(path),
        "source_sha256": sha256_text(source),
        "op_hints": op_hints,
        "model_class": class_match.group(1) if class_match else None,
        "has_get_inputs": bool(re.search(r"def\s+get_inputs\s*\(", source)),
        "has_get_init_inputs": bool(re.search(r"def\s+get_init_inputs\s*\(", source)),
        "source_lines": source.count("\n") + 1,
        "unknown_features": list(_UNKNOWN_RUNTIME_FEATURES),
        "unknown_reason": (
            "runtime shapes/dtypes/layouts require executing the task and were "
            "not inferred from source"
        ),
    }


def compatibility_from_environment(
    *,
    gpu_name: str | None,
    torch_version: str | None,
    triton_version: str | None,
    cuda_version: str | None = None,
    gpu_arch: list[str] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Build a compatibility fingerprint for measured-memory reuse."""
    compat: dict[str, Any] = {
        "gpu_name": gpu_name,
        "gpu_arch": list(gpu_arch) if gpu_arch else None,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "triton_version": triton_version,
        "model": model,
    }
    unknown = [key for key, value in compat.items() if value in (None, "", [])]
    compat["unknown_fields"] = unknown
    compat["compatibility_key"] = compatibility_key(compat)
    return compat


def compatibility_key(compat: dict[str, Any]) -> str:
    """Stable key over the fields that constrain measured-memory reuse.

    The model name is intentionally excluded because it does not change the
    measured hardware behavior; it is recorded for provenance only.
    """
    relevant = {
        "gpu_name": compat.get("gpu_name"),
        "gpu_arch": compat.get("gpu_arch"),
        "cuda_version": compat.get("cuda_version"),
        "torch_version": compat.get("torch_version"),
        "triton_version": compat.get("triton_version"),
    }
    return stable_hash(relevant)[:16]


def task_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Explainable similarity in [0, 1] from source-derived features only."""
    if not a or not b:
        return 0.0
    score = 0.0
    weight = 0.0

    # Same task source is the strongest (but still not sufficient) signal.
    if a.get("source_sha256") and a.get("source_sha256") == b.get("source_sha256"):
        score += 1.0
        weight += 1.0
    else:
        weight += 1.0

    # Operation-set Jaccard.
    ops_a = set(a.get("op_hints") or [])
    ops_b = set(b.get("op_hints") or [])
    if ops_a or ops_b:
        jaccard = len(ops_a & ops_b) / max(1, len(ops_a | ops_b))
        score += jaccard
        weight += 1.0

    # Model class name match (weak).
    if a.get("model_class") and a.get("model_class") == b.get("model_class"):
        score += 0.5
        weight += 0.5
    elif a.get("model_class") or b.get("model_class"):
        weight += 0.5

    return round(score / weight, 4) if weight else 0.0


def feature_predicate(task_features: dict[str, Any]) -> dict[str, Any]:
    """A compact applicability predicate stored on summaries."""
    return {
        "op_hints": sorted(task_features.get("op_hints") or []),
        "model_class": task_features.get("model_class"),
    }
