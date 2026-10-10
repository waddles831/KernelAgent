"""Command-line utilities for the cross-task memory store.

These commands are offline and CPU-only.  They never start GPU work or call a
model provider.

Examples::

    python -m kernelbench_runner.memory_cli inspect --run runs/<experiment>
    python -m kernelbench_runner.memory_cli inspect-runs --output-root runs
    python -m kernelbench_runner.memory_cli report --store memory
    python -m kernelbench_runner.memory_cli summarize --store memory
    python -m kernelbench_runner.memory_cli snapshot --store memory \
        --dest memory_snapshots/frozen_v1 --label frozen_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from kernel_memory.inspection import inspect_run_artifacts, inspect_runs
from kernel_memory.store import MemoryStore
from kernel_memory.summary import build_summaries


def _cmd_inspect(args: argparse.Namespace) -> int:
    report = inspect_run_artifacts(args.run)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


def _cmd_inspect_runs(args: argparse.Namespace) -> int:
    report = inspect_runs(args.output_root)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    store = MemoryStore(args.store, read_only=True)
    report = {
        "store_dir": str(store.directory),
        "memory_version": store.memory_version(),
        "counts": store.counts(),
        "task_results": [
            {
                "task_id": record.get("task_id"),
                "evaluation_status": record.get("evaluation_status"),
                "correctness_status": record.get("correctness_status"),
                "task_speedup_vs_torch_compile": record.get("task_speedup_vs_torch_compile"),
                "failure_category": record.get("failure_category"),
            }
            for record in store.read_task_results()
        ],
        "attempt_breakdown": _attempt_breakdown(store.read_attempts()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


def _attempt_breakdown(attempts: list[dict]) -> dict:
    by_phase: dict[str, int] = {}
    by_failure: dict[str, int] = {}
    measured = 0
    improved = 0
    regressed = 0
    for attempt in attempts:
        phase = attempt.get("phase", "unknown")
        by_phase[phase] = by_phase.get(phase, 0) + 1
        failure = attempt.get("failure_category", "unknown")
        by_failure[failure] = by_failure.get(failure, 0) + 1
        measurement = attempt.get("measurement") or {}
        if measurement.get("available") and measurement.get("local_speedup") is not None:
            measured += 1
            if measurement["local_speedup"] > 1.0:
                improved += 1
            elif measurement["local_speedup"] < 1.0:
                regressed += 1
    return {
        "total": len(attempts),
        "by_phase": by_phase,
        "by_failure_category": by_failure,
        "measured_parent_relative": measured,
        "improved": improved,
        "regressed": regressed,
    }


def _cmd_summarize(args: argparse.Namespace) -> int:
    store = MemoryStore(args.store, read_only=args.dry_run)
    attempts = store.read_attempts()
    summaries = build_summaries(
        attempts, min_support=args.min_support, version=args.version
    )
    existing = {record.get("summary_id") for record in store.read_summaries()}
    new_summaries = [s for s in summaries if s.get("summary_id") not in existing]
    print(
        json.dumps(
            {
                "attempts_read": len(attempts),
                "summaries_built": len(summaries),
                "summaries_new": len(new_summaries),
                "version": summaries[0]["version"] if summaries else None,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    for summary in new_summaries:
        print(
            f"  {summary['summary_id']}: bottleneck={summary['bottleneck_category']} "
            f"action={summary['action']} support={summary['observed_effect']['support']} "
            f"counterexamples={len(summary['counterexample_ids'])}"
        )
    if not args.dry_run and new_summaries:
        store.append_summaries(new_summaries)
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    store = MemoryStore(args.store, read_only=True)
    info = store.freeze(args.dest, label=args.label)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Report artifact coverage for one run")
    inspect.add_argument("--run", required=True, type=Path)
    inspect.add_argument("--output", type=Path, default=None)
    inspect.set_defaults(func=_cmd_inspect)

    inspect_runs_p = sub.add_parser("inspect-runs", help="Report coverage for all runs")
    inspect_runs_p.add_argument("--output-root", required=True, type=Path)
    inspect_runs_p.add_argument("--output", type=Path, default=None)
    inspect_runs_p.set_defaults(func=_cmd_inspect_runs)

    report = sub.add_parser("report", help="Summarize a memory store")
    report.add_argument("--store", required=True, type=Path)
    report.add_argument("--output", type=Path, default=None)
    report.set_defaults(func=_cmd_report)

    summarize = sub.add_parser(
        "summarize", help="Offline, rule-based strategy-summary construction"
    )
    summarize.add_argument("--store", required=True, type=Path)
    summarize.add_argument("--min-support", type=int, default=1)
    summarize.add_argument("--version", default=None)
    summarize.add_argument("--dry-run", action="store_true")
    summarize.set_defaults(func=_cmd_summarize)

    snapshot = sub.add_parser("snapshot", help="Freeze a memory store")
    snapshot.add_argument("--store", required=True, type=Path)
    snapshot.add_argument("--dest", required=True, type=Path)
    snapshot.add_argument("--label", default=None)
    snapshot.set_defaults(func=_cmd_snapshot)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"memory-cli: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
