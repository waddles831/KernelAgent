"""Cross-task memory for KernelAgent.

A small, auditable, local experience store.  The memory system is **off by
default**; when disabled it must not alter the existing generation/optimization
behavior.

Three artifact kinds are kept explicitly separate:

* ``task_result`` - the final, project-owned correctness/benchmark result.
* ``attempt``     - immutable raw evidence for every optimization (and,
  optionally, generation) attempt, including regressions and failures.
* ``summary``     - a versioned, compact, rule-based interpretation linked to
  evidence IDs.  LLM text may be attached as a suggestion but never replaces
  measurements or correctness outcomes.

Storage is newline-delimited JSON under a single directory.  No external
service is required.
"""

from kernel_memory.schema import (
    ATTEMPT_SCHEMA_VERSION,
    MEMORY_SCHEMA_VERSION,
    RETRIEVAL_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    TASK_RESULT_SCHEMA_VERSION,
)
from kernel_memory.store import MemoryConfig, MemoryStore

__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "TASK_RESULT_SCHEMA_VERSION",
    "ATTEMPT_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "RETRIEVAL_SCHEMA_VERSION",
    "MemoryConfig",
    "MemoryStore",
]
