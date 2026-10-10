# KernelAgent Architecture and Execution Flow

This map is based on the code in the checked-out KernelAgent repository. It
describes two related paths: general kernel generation/Fuser composition and
hardware-guided optimization of an existing Triton kernel.

## General kernel generation

```text
PyTorch problem file or problem description
        ↓
Fuser AutoKernelRouter (optional)
        ↓
TritonKernelAgent.generate_kernel()
        ↓
LLM seeds → VerificationWorker refinement
        ↓
isolated candidate test subprocesses
        ↓
first successful candidate returned
```

- **Entry points:** `Fuser/auto_agent.py` exposes `AutoKernelRouter.solve()` and
  `python -m Fuser.auto_agent --problem ...`. The direct API is
  `triton_kernel_agent/agent.py:TritonKernelAgent.generate_kernel()`.
- **Routing and orchestration:** `Fuser/auto_agent.py` analyzes the problem,
  may ask an LLM to route it to the direct KernelAgent path or the Fuser
  pipeline, and can fall back to the other path. `Fuser/pipeline.py` runs
  extract → dispatch → compose. It writes artifacts under `.fuse/<run_id>/`.
- **Task representation:** generation accepts a problem description string
  and optional test source. `PromptManager` builds the prompts; the generated
  kernel source and tests are saved in a `session_<timestamp>` directory.
- **LLM abstraction:** `utils/providers/` contains provider interfaces and
  OpenAI, Anthropic, and relay implementations. Model-name/provider mapping is
  in `utils/providers/available_models.py` and `utils/providers/models.py`.
  Unknown model names currently fall back to `RelayProvider`.
- **Seed generation:** `TritonKernelAgent._generate_kernel_seeds()` requests
  candidate code from the selected provider. The test code is also supplied as
  prompt context.
- **Candidate representation:** generation candidates are source strings and
  worker result dictionaries. `VerificationWorker` writes the candidate and
  test harness to a worker directory; there is no separate durable generation
  candidate class.
- **Correctness and failure handling:** `VerificationWorker` validates
  candidate structure, rejects PyTorch compute fallbacks, and executes the
  test in a subprocess through `Fuser/runner.py`. A candidate passes only when
  the test's success sentinel is present and the subprocess exits successfully.
  Failed candidates may be refined with compiler/runtime feedback, bounded by
  `max_rounds` and `test_timeout_s`. `WorkerManager` returns the first worker
  that succeeds, rather than ranking all successful seeds by measured latency.
- **Best kernel:** `TritonKernelAgent.generate_kernel()` returns the successful
  code, worker ID, rounds, and session directory. Performance ranking is not
  part of this generation result.

## KernelBench-style Fuser path

`Fuser/compose_end_to_end.py` asks for a complete Python program containing
Triton kernels, a top-level `kernel_function(...)`, and a self-test. Its
composition prompt receives the original PyTorch model and subgraph
descriptions. With `--verify`, the result is executed by `Fuser/runner.py` and
can be refined when verification fails. The final program and composition
summary are written under the Fuser run directory.

KernelAgent accepts KernelBench problem files as task definitions. The
implemented `kernelbench_runner/` calls `AutoKernelRouter`, adapts
`kernel_function(...)` to a `ModelNew` interface, and then performs its own
correctness and performance evaluation in a separate process. It uses the
task's `Model` and `get_inputs()` as reference/workload data, checks FP32
correctness over deterministic trials, and times the candidate against
default `torch.compile` with `triton.testing.do_bench`. It does not call
`kernelbench.eval.eval_kernel_against_ref()` or use KernelBench timing code.
A configured `OPENAI_BASE_URL` selects the OpenAI-compatible provider for
custom model IDs; the runner copies the API key from the configured
environment-variable name into `OPENAI_API_KEY` only in the child process
environment.

## Hardware-guided optimization path

```text
initial Triton kernel
        ↓
OptimizationManager: initial verification and reference/kernel timings
        ↓
BeamSearchStrategy or GreedyStrategy selects parent candidates
        ↓
OptimizationWorker / OptimizationOrchestrator
        ↓
NCU profile → roofline classification → LLM bottleneck analysis
        ↓
LLM generates a changed kernel
        ↓
VerificationWorker correctness → benchmark subprocess
        ↓
candidate ranking, PTX deduplication, and best-so-far result
```

- **Profile:** `KernelProfiler` invokes the Nsight Compute profiler interface
  and stores metric data plus metadata.
- **Diagnose:** `RooflineAnalyzer` classifies utilization/bottlenecks from NCU
  metrics. `BottleneckAnalyzer` sends kernel code, metrics, and roofline
  findings to the LLM and parses bottleneck categories, causes, and fixes.
- **Prescribe:** `OptimizationOrchestrator` turns the diagnosis and round
  context into an optimization prompt; the provider returns changed Triton
  source. The profiler/analyzer produce recommendations, while the
  orchestrator coordinates their use.
- **Orchestrate:** `OptimizationManager.run_optimization()` checks the initial
  candidate, obtains PyTorch/compiled-reference and initial-kernel timings,
  launches worker rounds, updates the strategy, and finalizes the best result.
  `OptimizationOrchestrator.optimize_kernel()` handles one worker's round
  lifecycle and artifacts.
- **Explore:** `BeamSearchStrategy` selects up to the configured beam parents
  and bottleneck directions; `GreedyStrategy` is also available. Successful
  candidates are deduplicated by normalized PTX fingerprint when available,
  sorted by measured runtime, and truncated to beam width. Candidate workers
  can run in parallel, while benchmark/profile access is guarded by GPU locks.
- **Measure:** `VerificationWorker` checks correctness. The optimization
  `Benchmark` uses isolated subprocesses and records latency in milliseconds;
  lower latency is better, and speedup is calculated against the chosen
  reference timing. NCU profiling supplies metrics, not the ranking latency.
- **History:** `OptimizationAttempt` and `Reflexion` hold per-round attempts
  and lessons. `ProgramEntry` stores successful kernel code, `time_ms`,
  `problem_id`, parent/generation lineage, optional PTX hash, and timestamp.
  `JSONProgramDatabase` persists those entries. `OptimizationManager` also
  shares bounded attempt/reflexion lists across workers in a run.

## Phase-name mapping

| Project concept | Code location / behavior |
|---|---|
| Profile | `triton_kernel_agent/opt_worker_component/profiling/kernel_profiler.py`; Nsight Compute metric collection |
| Diagnose | `kernel_perf_agent/kernel_opt/roofline/ncu_roofline.py` and `triton_kernel_agent/opt_worker_component/prescribing/bottleneck_analyzer.py` |
| Prescribe | Bottleneck response plus optimization prompt construction in `OptimizationOrchestrator` |
| Orchestrate | `OptimizationManager` and `OptimizationOrchestrator` |
| Explore | `searching/strategy/beam_search.py`, `greedy.py`, and `OptimizationManager` worker dispatch |
| Measure | `searching` candidate metrics populated by `Benchmark`; correctness by `VerificationWorker` |

## Server verification

- **TODO(server verification):** verify the current KernelBench L1 model
  signatures against the composed `kernel_function(...)` adapter and its
  model-parameter handling.
- **TODO(server verification):** confirm the A30 CUDA/Triton toolchain,
  `sm_80` compilation, correctness, benchmark, and NCU paths.
- **TODO(server verification):** compare any custom LLM endpoint/model with
  the provider API and response format expected by the current KernelAgent
  provider layer.
