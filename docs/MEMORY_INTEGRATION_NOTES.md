# Cross-Task Memory Integration Notes

Cross-task memory is intentionally not implemented in this WSL2 phase. The
following integration points are based on the current KernelAgent state and
task flow. The KernelBlaster-informed proposal below adds a concrete design
and experiment sequence; it does not mean that cross-task memory is already
implemented or that its benefit on A30 has been established.

## Memory write

Write one successful task-level memory record only after the final candidate
has passed the project-owned correctness check and benchmark. The project runner has the
most complete verified view at that point: task ID/name, `torch.compile`
reference and candidate latency, speedup, generated kernel source, KernelAgent
route/rounds, and task metadata. Keep failed generations and failed candidates
as separate records with explicit error categories; never label an unverified
candidate as a successful memory.

Within the existing optimization loop, `OptimizationManager.run_optimization()`
and `OptimizationOrchestrator._finalize_results()` are the task-final hooks.
`OptimizationManager._run_workers()` receives worker attempt/reflexion
results. These are useful if memory needs within-run optimization events, but
the outer task runner is the better initial write hook because it can attach
the project-owned benchmark result.

## Memory retrieve

For initial generation, retrieve after the task source and ID are known but
before `TritonKernelAgent._generate_kernel_seeds()` builds candidate prompts.
In the current integrated path, that is most naturally exposed by the runner
before it calls `AutoKernelRouter.solve()` and supplied to a future explicit
prompt/context hook.

For iterative optimization, retrieve when `OptimizationManager` has selected
the parent kernel and task context, before it dispatches workers in
`_run_workers()` (after `_populate_baseline_cache()` has supplied available
parent profiling). This allows retrieved evidence to inform a specific
parent/diagnosis without modifying the beam-selection algorithm.
The worker also has a more precise, diagnosis-conditioned insertion point:
`OptimizationOrchestrator.optimize_kernel()` already constructs optional
`rag_context` after selecting `primary` and before rendering the optimization
prompt. Use this point for the first optimization-memory experiment.

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
- The optional `RAGPrescriber` retrieves static optimization patterns from
  `kernel_perf_agent/kernel_opt/database/code_samples` using embeddings and
  builds bounded prompt context. `OptimizationWorker.use_rag` defaults to
  false. This is a reusable prompt interface, not a measured-experience store
  or a cross-task policy-update mechanism.

These are starting points, not a cross-task memory backend. Existing program
history is mainly successful-performance data and does not include enough
task features to support reliable retrieval across unrelated operators.

## Data still needed

Future records should add versioned task characteristics
(operator/shape/dtype, layout/strides, scalar parameters), KernelBench
task-file hash/version, GPU identity and software versions, successful
transformation tags and diagnosis, explicit failure category/error summary,
correctness status, timing methodology, and provenance for each observation.
Keep full kernel source/artifacts linked from the JSON record rather than
duplicating unlimited code in one database.

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
- Whether optimization-prompt memory earns enough benefit to retain, and
  whether to extend it to seed generation in a separate experiment.

The first A30 baseline should establish complete per-task measurements and
artifacts before these decisions or any memory backend are implemented.

## KernelBlaster reference and evidence

Reviewed on 2026-10-10: Dong et al., **KernelBlaster: Continual Cross-Task CUDA
Optimization via Memory-Augmented In-Context Reinforcement Learning**,
[arXiv:2602.14293v1](https://arxiv.org/abs/2602.14293v1).
The [full paper](https://arxiv.org/html/2602.14293v1) is the source for the
following research observations; the [official repository](https://github.com/NVlabs/KernelBlaster)
is an implementation reference.

- **Method (§3, Algorithm 2):** profiling-derived states index optimization
  actions and expected scores. Weighted exploration and replay feed textual
  updates based on expected-versus-observed performance; model weights stay
  fixed. Both successful and unsuccessful experience inform the knowledge base.
- **Memory/profiling ablations (§6.1, §6.3):** the full system outperforms
  profiling without memory; Level-2 speedup is 1.57x with detailed NCU feedback
  versus 1.22x with cycle-only feedback in that comparison.
- **Search/cost observations (§6.2, §6.4):** returns diminish beyond roughly
  eight trajectories or four optimization steps, although some tasks benefit
  from more depth. The minimal agent uses 2.4x the tokens in the reported setup.
- **Transfer limits (§4.1–4.2, §4.6–4.7):** optimization starts from verified
  CUDA; profile-free initial generation is future work. Evaluation uses the
  faster of eager/compiled PyTorch and summed NCU kernel cycles. These differ
  from this project's Triton generation and `do_bench`/default-compile protocol.

These observations motivate the project-specific proposals below. They are
not measured KernelAgent results, a reproduction of MAIC-RL, or recommended
A30 hyperparameters copied from the paper.

## Proposed KernelAgent integration

### Preserve the existing search and add an experience layer

Implement an opt-in sidecar store after the baseline, keeping the existing
program database responsible for candidate code, lineage, and beam ranking.
Removing the `problem_id` filter from `sample_inspirations()` is insufficient:
that function ranks by absolute latency, which cannot compare different tasks.
Use explicit compatibility and state matching for cross-task retrieval.

Keep three artifacts separate:

1. **Task result:** the final project-owned correctness/benchmark result.
   Write through `kernelbench_runner/runner.py` alongside `task_result.json`.
2. **Attempt evidence:** immutable observations for every completed or failed
   optimization attempt, including regressions and timeouts. Collect through
   the manager; workers return evidence rather than concurrently rewriting
   a shared database. Audit early-exit/error paths before claiming coverage.
3. **Strategy summary:** a versioned, compact interpretation linked to evidence
   IDs. An LLM may suggest a lesson, but it cannot overwrite measurements or
   promote a candidate to verified status.

`OptimizationOrchestrator._finalize_results()` currently returns the last
attempt/reflexion for shared history. Complete trajectory export will require
an explicit extension; the existing last-result API is not a replay buffer.
Also, `kernelbench_runner.optimize_one` uses an internal test with a single
seed and tensor-output restriction, while the outer evaluator runs the
project's configured trials. Record which checker passed; an internal pass
does not imply a final project-evaluator pass.

### Candidate record schema

The following is a proposed schema, not an existing API. Use a versioned JSON
representation initially; choose indexing/retention after observing volume.

| Group | Required information and intended use |
|---|---|
| Identity | `schema_version`, `record_id`, `run_id`, task ID/source hash, parent/candidate hashes, round/worker, timestamp, language/backend, model/prompt version |
| Compatibility | GPU model and capability, CUDA/driver/Triton/PyTorch versions; operator, shapes, dtypes, layouts/strides, scalar parameters, numerical tolerance |
| State | Primary/secondary bottleneck, roofline/SOL metrics, available stall/occupancy/traffic evidence, profile artifact and metric scope, `profile_status` and extraction version |
| Action | Normalized transformation tag, concrete config/code change, preconditions, expected effect captured before execution; mark combined transformations explicitly |
| Outcome | Compile/runtime/verification status, checker and input seeds, failure category, parent/candidate latency, measurement method/budget, uncertainty or repeated samples when available |
| Interpretation | Observed effect, expected-versus-actual discrepancy, conditional lesson, confidence, supporting and contradicting record IDs |
| Provenance | Source/profile/log artifact paths, retrieval IDs, memory snapshot/version, action-selection reason, token/GPU-time cost |

Represent absent measurements as null with a reason. A compile error has no
valid speedup; a correctly measured slowdown is useful negative performance
evidence. Extend the existing `Outcome` classification (`IMPROVED`, `REGRESSED`,
`FAILED`) with explicit compiler/runtime/correctness failure, OOM, and timeout
categories; record profiling availability separately from candidate validity.

For an attempt, store `local_speedup = parent_ms / candidate_ms` only when both
measurements are valid under the same protocol. Keep this separate from
`task_speedup = compiled_reference_ms / final_candidate_ms`. Neither a faster
parent-relative attempt nor a positive LLM judgment proves that the final
candidate beats the project reference. Preserve raw observations when timing
noise makes the sign of an improvement uncertain.

### Retrieval and prompt contract

For the first experiment, retrieve after the worker has a diagnosis and before
`render_kernel_optimization_prompt()`. Extend the existing context interface
with a separately labeled experience block; keep static RAG examples separately
identified and hold their enablement constant across comparison arms.

Use the following project-specific retrieval order:

1. Filter measured examples for compatible backend, hardware/toolchain,
   correctness constraints, and task preconditions. Initially restrict measured
   performance claims to the same A30 environment. Foreign-GPU/CUDA examples
   may later supply qualitative hypotheses, with mandatory local revalidation.
2. Rank by bottleneck evidence and task features, then by evidence quality.
   Shape similarity alone is insufficient. Use `unknown` when NCU data is
   unavailable; do not manufacture a profile-derived state from latency alone.
3. Retrieve a small, configurable set of conditional lessons, including relevant
   regressions. Bound total context and deduplicate near-identical attempts.
   Each item should state applicability, proposed action, measured outcome,
   caveat, and source IDs; load full code only when necessary.
4. Log retrieved items and whether the worker applied their suggestions.
   Preserve a configurable exploration allocation so popular historical
   actions do not consume every candidate. Keep beam survival based on the
   existing correctness gates and measured candidate latency.

Seed-generation retrieval remains a later extension through
`kernelbench_runner/generate_one.py`, `AutoKernelRouter`, and the generation
prompt API. Before a seed exists, only source-derived task features are known;
it must not be represented as a profiled parent. Check both direct and Fuser
routes when propagating context, and evaluate this extension separately.

### Reflection updates and Triton-specific lessons

Reuse `Reflexion.expected_outcome`, `actual_outcome`, `lessons`, `avoid_patterns`,
and `try_patterns` as inputs to an offline consolidation pass. Require each
proposed summary to identify the evidence and applicability conditions.
Store summaries as new versions with support counts and conflicting examples;
do not treat `was_diagnosis_correct` as a measured fact. Freeze a memory snapshot
during a comparison run so concurrent updates cannot change its treatment.

For A30/Triton, start with a small action vocabulary such as block-size changes,
`num_warps`, `num_stages`, layout/coalescing changes, reduction decomposition,
and fusion. These are candidate labels for local experiments, not established
winning strategies. Translate CUDA lessons into Triton hypotheses and inspect
generated behavior before assigning a low-level cause. In particular, record
precision and alignment requirements instead of importing a faster example
that silently changes numerical semantics.

An illustrative record could say: "For this contiguous FP32 reduction and
parent profile, increasing block size was expected to reduce overhead but
increased measured latency; avoid repeating that setting on matching cases."
Only attach a register-pressure explanation when profiling supports it.
If several transformations changed together, attribute the outcome to the
bundle until isolated experiments separate their effects.

For Fuser/multi-kernel candidates, retain per-kernel identities and metric
scope. The current `_get_triton_kernel_metrics()` chooses the first non-PyTorch
kernel's metrics; those must not be labeled as an end-to-end bottleneck.
Initially restrict state-based experiments to interpretable single-kernel
cases, or add explicit multi-kernel attribution before broadening them.

## Execution sequence and acceptance criteria

1. **A30 baseline:** finish the existing correctness/timing protocol and save
   artifacts with memory disabled. Confirm NCU availability separately from
   benchmark availability. Keep current latency ranking and task definitions.
2. **Evidence-only instrumentation:** export complete attempt records and final
   task results without changing prompts. Manually audit a small set of
   improvements, regressions, and failures for correct parent linkage,
   measurement provenance, and useful state/action labels.
3. **Frozen-memory pilot:** build summaries from a source-task subset, freeze
   them, and evaluate disjoint held-out tasks. Compare the unchanged optimizer,
   a bounded code-example retrieval baseline, and the proposed state-conditioned
   lessons. Hold starting kernels, model, static RAG setting, profiling,
   correctness checks, and candidate budget constant. Record extra retrieval
   and reflection costs, rather than treating them as free.
4. **Continual-update experiment:** only after the frozen pilot, update memory
   between tasks. Record task order and snapshot versions, repeat task orders,
   and allow only past-task evidence. Report this separately from held-out
   frozen-memory performance; reusing a task's own past solution measures reuse,
   not unseen-task transfer.
5. **Optional extensions:** independently ablate negative evidence, profiling
   features, exploration allocation, and initial-generation context. Tune
   breadth/depth against the actual A30 cost curve rather than adopting paper
   trajectory counts as beam-search parameters.

Report correctness rate, fraction of tasks beating chosen speedup thresholds,
geometric-mean speedup on a declared comparable task set, attempts to first
verified improvement, tokens, GPU profiling/benchmark time, and total wall time.
Include memory-construction cost separately from amortized reuse cost. Declare
OOM/non-evaluable exclusions and failure denominators; do not hide unsuccessful
tasks by reporting only fast successful kernels. See the unchanged
[local evaluation protocol](KERNELBENCH_L1.md).

Retain memory only if repeated comparisons show better verified performance
or lower cost to reach the same performance, without reduced correctness.
An unsuccessful pilot is still informative: use retrieval logs to distinguish
irrelevant states, missing evidence, ignored lessons, and measurement noise
before adding a more complex retrieval backend. All stages above remain
planned work; this document update itself introduces no runtime behavior.
