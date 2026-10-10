"""Local, file-based persistence for cross-task memory.

Storage is newline-delimited JSON (one directory, three artifact files).  It is
deliberately simple and auditable: no database server, no embeddings, no
network access.  Writes are serialized with an advisory file lock and flushed,
so a single manager/runner process can append safely and a later process can
read a consistent snapshot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:  # pragma: no cover - fcntl is unavailable on non-POSIX
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from kernel_memory.schema import MEMORY_SCHEMA_VERSION, utc_now_iso

_REPO_ROOT = Path(__file__).resolve().parents[1]

_VALID_MODES = ("off", "collect", "frozen", "continual")


@dataclass
class MemoryConfig:
    """User-facing memory configuration.

    Modes
    -----
    off        No retrieval and no writes.  Behavior is identical to a build
               without memory support.
    collect    Write task results and attempt evidence; never retrieve.
    frozen     Retrieve from a frozen snapshot only; write this run's evidence
               to a separate store so the snapshot is never modified.
    continual  Retrieve from, and append to, the same store after each task.
               Only *earlier* tasks are visible to a task's retrieval.
    """

    mode: str = "off"
    store_dir: str = str(_REPO_ROOT / "memory")
    snapshot_dir: str | None = None
    snapshot_label: str | None = None
    max_items: int = 3
    prompt_budget_chars: int = 4000
    include_failures: bool = True
    allow_foreign_compat: bool = False
    record_generation_traces: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def normalized_mode(self) -> str:
        mode = (self.mode or "off").strip().lower()
        return mode

    def is_off(self) -> bool:
        return self.normalized_mode() == "off"

    def retrieval_enabled(self) -> bool:
        return self.normalized_mode() in ("frozen", "continual")

    def write_enabled(self) -> bool:
        return self.normalized_mode() in ("collect", "frozen", "continual")

    def is_frozen(self) -> bool:
        return self.normalized_mode() == "frozen"

    def retrieval_dir(self) -> Path:
        """Directory used for retrieval (never written to in frozen mode)."""
        if self.is_frozen():
            return Path(self.snapshot_dir or self.store_dir).expanduser().resolve()
        return Path(self.store_dir).expanduser().resolve()

    def write_dir(self) -> Path:
        """Directory used for writes.  In frozen mode this differs from the
        retrieval directory by default so the frozen snapshot is preserved."""
        if self.is_frozen():
            base = Path(self.store_dir).expanduser().resolve()
            if self.snapshot_dir:
                snapshot = Path(self.snapshot_dir).expanduser().resolve()
                if base == snapshot:
                    base = base.parent / (base.name + "_current")
            return base
        return Path(self.store_dir).expanduser().resolve()

    def validate(self) -> None:
        if self.normalized_mode() not in _VALID_MODES:
            raise ValueError(
                f"memory mode must be one of {_VALID_MODES}, got {self.mode!r}"
            )
        if self.max_items < 1:
            raise ValueError("memory max_items must be positive")
        if self.prompt_budget_chars < 0:
            raise ValueError("memory prompt_budget_chars must be >= 0")
        if self.is_frozen():
            snapshot = self.resolved_snapshot_dir()
            if snapshot is None:
                raise ValueError("frozen memory requires snapshot_dir")
            if snapshot == self.write_dir() and self.retrieval_dir() == self.write_dir():
                raise ValueError(
                    "frozen memory write_dir must differ from snapshot_dir to "
                    "avoid corrupting the frozen snapshot"
                )

    def resolved_snapshot_dir(self) -> Path | None:
        if self.snapshot_dir:
            return Path(self.snapshot_dir).expanduser().resolve()
        if self.is_frozen():
            return Path(self.store_dir).expanduser().resolve()
        return None

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.normalized_mode()
        data["retrieval_enabled"] = self.retrieval_enabled()
        data["write_enabled"] = self.write_enabled()
        data["retrieval_dir"] = str(self.retrieval_dir()) if not self.is_off() else None
        data["write_dir"] = str(self.write_dir()) if not self.is_off() else None
        return data


class MemoryStore:
    """Read/write access to a memory directory."""

    def __init__(self, directory: Path | str, *, read_only: bool = False):
        self.directory = Path(directory).expanduser().resolve()
        self.read_only = read_only
        self.task_results_path = self.directory / "task_results.jsonl"
        self.attempts_path = self.directory / "attempts.jsonl"
        self.summaries_path = self.directory / "summaries.jsonl"
        self.retrievals_path = self.directory / "retrieval_log.jsonl"
        self.version_path = self.directory / "version.json"

    # -- reading -------------------------------------------------------

    def read_task_results(self) -> list[dict[str, Any]]:
        return self._read(self.task_results_path)

    def read_attempts(self) -> list[dict[str, Any]]:
        return self._read(self.attempts_path)

    def read_summaries(self) -> list[dict[str, Any]]:
        return self._read(self.summaries_path)

    def read_retrievals(self) -> list[dict[str, Any]]:
        return self._read(self.retrievals_path)

    def counts(self) -> dict[str, int]:
        return {
            "task_results": len(self._read(self.task_results_path)),
            "attempts": len(self._read(self.attempts_path)),
            "summaries": len(self._read(self.summaries_path)),
            "retrievals": len(self._read(self.retrievals_path)),
        }

    def _read(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, dict):
                    continue
                records.append(value)
        return records

    # -- writing -------------------------------------------------------

    def append_task_results(self, records: list[dict[str, Any]]) -> None:
        self._append(self.task_results_path, records)

    def append_attempts(self, records: list[dict[str, Any]]) -> None:
        self._append(self.attempts_path, records)

    def append_summaries(self, records: list[dict[str, Any]]) -> None:
        self._append(self.summaries_path, records)

    def append_retrievals(self, records: list[dict[str, Any]]) -> None:
        self._append(self.retrievals_path, records)

    def _append(self, path: Path, records: list[dict[str, Any]]) -> None:
        if self.read_only:
            raise PermissionError(f"memory store is read-only: {self.directory}")
        if not records:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
            for record in records
        )
        with path.open("a", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # -- versioning ----------------------------------------------------

    def memory_version(self) -> str:
        """Deterministic content hash of the three artifact files.

        Two stores with identical records (and ordering) produce the same
        version string.  Retrieval records and the version file are excluded so
        that reading memory does not change its identity.
        """
        digest = hashlib.sha256()
        digest.update(f"memory_schema_version={MEMORY_SCHEMA_VERSION}\n".encode())
        for path in (
            self.task_results_path,
            self.attempts_path,
            self.summaries_path,
        ):
            digest.update(f"--{path.name}--\n".encode())
            if path.exists():
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(65536), b""):
                        digest.update(chunk)
        return digest.hexdigest()[:16]

    def read_version_info(self) -> dict[str, Any]:
        if not self.version_path.exists():
            return {}
        try:
            value = json.loads(self.version_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    # -- freezing ------------------------------------------------------

    def freeze(self, destination: Path | str, *, label: str | None = None) -> dict[str, Any]:
        """Copy the three artifact files into a new immutable-ish snapshot."""
        dest = Path(destination).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        if dest == self.directory:
            raise ValueError("snapshot destination must differ from the source store")
        for path in (self.task_results_path, self.attempts_path, self.summaries_path):
            target = dest / path.name
            if path.exists():
                target.write_bytes(path.read_bytes())
        version_info = {
            "memory_schema_version": MEMORY_SCHEMA_VERSION,
            "memory_version": self.memory_version(),
            "snapshot_label": label or utc_now_iso(),
            "created_at": utc_now_iso(),
            "counts": {
                "task_results": len(self.read_task_results()),
                "attempts": len(self.read_attempts()),
                "summaries": len(self.read_summaries()),
            },
        }
        (dest / "version.json").write_text(
            json.dumps(version_info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return version_info
