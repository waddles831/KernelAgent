# Cross-Task Memory Integration Notes

Cross-task memory is intentionally not implemented in this WSL2 phase. The
following integration points are based on the current KernelAgent state and
task flow.

## Memory write

Write one task-level memory record only after the final candidate has passed
the official KernelBench correctness check and benchmark. The project runner
has the most complete verified view at that point: task ID/name, reference and
best latency, speedup, generated kernel source, KernelAgent route/rounds, and
KernelBench metadata. Keep failed generations and failed candidates as
separate records with explicit error categories; never label an unverified
candidate as a successful memory.

Within the existing optimization loop, `OptimizationManager.run_optimization()`
and `OptimizationOrchestrator._finalize_results()` are the task-final hooks.
`OptimizationManager._run_workers()` receives worker attempt/reflexion
results. These are useful if memory needs within-run optimization events, but
the outer KernelBench runner is the better initial write hook because it can
attach the official task result.

## Memory retrieve

For initial generation, retrieve after the task source and ID are known but
before `TritonKernelAgent._generate_kernel_seeds()` builds candidate prompts.
In the current integrated path, that is most naturally exposed by the runner
before it calls `AutoKernelRouter.solve()` and supplied to a future explicit
prompt/context hook.

For iterative optimization, retrieve when `OptimizationManager` has selected
the parent kernel and task context, before it dispatches workers in
`_run_workers()`. This allows retrieved evidence to inform a specific
parent/diagnosis without modifying the beam-selection algorithm.

## Existing state to reuse

- `ProgramEntry` stores a successful candidate's code, measured `time_ms`,
  `problem_id`, parent ID, generation, optional normalized-PTX hash, and
  timestamp.
- `AttemptRecord` stores candidate code, time, outcome, timestamp, and parent.
- `OptimizationAttempt` and `Reflexion` hold round-level diagnosis, kernel
  changes, outcomes, and lessons used inside optimization prompts.
- `JSONProgramDatabase` provides JSON persistence and same-problem filtering;
  `sample_inspirations()` currently supports an optional `problem_id` filter.
- `OptimizationManager` passes bounded attempt/reflexion history among workers
  in one optimization run.

These are starting points, not a cross-task memory backend. Existing program
history is mainly successful-performance data and does not include enough
task features to support reliable retrieval across unrelated operators.

## Data still needed

Future records should add versioned task characteristics (operator/shape/dtype,
layout/strides, scalar parameters), KernelBench code hash/version, GPU identity
and software versions, successful transformation tags and diagnosis, explicit
failure category/error summary, correctness status, timing methodology, and
provenance for each observation. Keep full kernel source/artifacts linked from
the JSON record rather than duplicating unlimited code in one database.

Retrieval should filter for hardware/toolchain compatibility and task
similarity, expose provenance and correctness status to the prompt, and cap
context size. Similarity rules, weighting of failed attempts, persistence
format, and whether experiences transfer between L1 operators should be
decided after an A30 baseline and qualitative review of actual trajectories.

## Decisions deferred until A30 baseline

- Whether cross-task experiences improve generation or optimization quality.
- Which task features predict useful transfer and whether similar shapes are
  sufficient.
- Which failure categories are stable and useful across runs.
- How A30, Triton, PyTorch, CUDA, and KernelBench versions constrain reuse.
- Storage growth/retention limits and how many retrieved examples fit prompts.
- Whether memory should influence seed generation, optimization prompts, or
  both in the first experiment.

The first A30 baseline should establish complete per-task measurements and
artifacts before these decisions or any memory backend are implemented.
