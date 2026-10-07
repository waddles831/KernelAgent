# KernelBench Level 1

This guide reflects the checked-out sibling repository `../KernelBench` and
does not modify its evaluator.

## Task definitions and identity

Level 1 problems are Python files under `KernelBench/level1/`. In this
checkout there are 100 files. Each defines a PyTorch `Model(nn.Module)`,
`get_init_inputs()`, and `get_inputs()`. A task's integer `problem_id` is the
numeric prefix of its filename; `Problem.name` is the full filename. For
example, `19_ReLU.py` maps to ID `19` and name `19_ReLU.py`.

`kernelbench.dataset.LocalKernelBenchDataset(level=1)` loads those local task
files. It supports selecting problem IDs or an inclusive ID range. The dataset
can also be read from Hugging Face, but the current integration uses the local
repository at `../KernelBench`.

## Official evaluation API

The primary evaluator is `kernelbench.eval.eval_kernel_against_ref()`:

1. It requires CUDA to be available and selects the requested GPU.
2. It executes the reference task source and instantiates `Model` with
   `get_init_inputs()`.
3. It loads custom source with entry point `ModelNew`; the `triton` backend
   imports the custom file from a temporary `.py` file so Triton JIT decorators
   work correctly.
4. It instantiates `ModelNew` with the same initialization inputs.
5. Correctness is checked on `num_correct_trials` deterministically seeded
   inputs. Every trial must return the same output shape and pass
   `torch.allclose` using the precision-specific tolerance. Any runtime
   exception or failed trial makes `correctness` false.
6. If `measure_performance=True`, timing runs only after correctness passes.
   `timing_method` selects the official timing implementation; the default is
   `cuda_event`. The default timing trial count in the evaluator is 10; batch
   scripts commonly configure 100.
7. If performance is measured, KernelBench also times the PyTorch reference
   under the same timing method and reports `ref_runtime`. The speedup is
   `ref_runtime / runtime`.

`KernelExecResult` contains `compiled`, `correctness`, `runtime`,
`runtime_stats`, `ref_runtime`, `ref_runtime_stats`, and metadata. The timing
implementation and `get_timing_stats()` report milliseconds; the model's
comments calling these values microseconds are stale. Use explicit `_ms`
field names in experiment output. A correctness failure is not assigned a
meaningful speedup.

The evaluator distinguishes compilation/load failures (`compiled=False`),
execution failures after loading (`compiled=True`, `correctness=False`, with
runtime-error metadata), and output mismatches (`compiled=True`,
`correctness=False`, with correctness metadata). Timing exceptions are recorded
as `error_during_performance`; the result may have no usable runtime. The core
`eval_kernel_against_ref()` call has no overall wall-clock timeout. The batch
script `scripts/eval_from_generations.py` enforces per-task timeouts in a
separate process (default 180 seconds) and records a timeout as a failed
evaluation.

The optional pattern-based `kernel_static_checker` is invoked by
`scripts/run_and_check.py` by default. It is a separate heuristic check; it
does not define the evaluator's shape/tolerance correctness semantics.

## Baseline and speedup

KernelBench can use checked-in baseline timing JSON for analysis, but those
numbers are hardware-specific and are not the A30 reference result. For an A30
experiment, measure the PyTorch reference on the same GPU and run using
`ref_runtime` from `eval_kernel_against_ref()` (or the official reference-time
helper). Store the candidate and reference latency in milliseconds. Do not
compare to another GPU's stored latency as if it were an A30 baseline.

The `scripts/run_and_check.py` single-sample CLI also measures eager and
`torch.compile` reference timings separately. `scripts/eval_from_generations.py`
evaluates generated `ModelNew` files saved in a run directory and writes
`eval_results.json`; it is a batch evaluator, not an agent-generation command.
`scripts/verify_bench.py` checks reference models against themselves across
levels and is not an L1 candidate benchmark.

## Commands and integration path

KernelBench documents a single problem check in this form:

```bash
uv run python scripts/run_and_check.py \
  ref_origin=kernelbench dataset_src=local level=1 problem_id=19 \
  kernel_src_path=/path/to/ModelNew.py eval_mode=local backend=triton
```

The checked-out KernelAgent can already consume a KernelBench problem file
through `python -m Fuser.auto_agent --problem ... --verify`. Its generated
`kernel_function(...)` source is not itself KernelBench's required `ModelNew`
class, and its built-in test pass is not the official KernelBench result. The
integration layer will reuse that existing generation entry point, add a thin
`ModelNew` wrapper, and call KernelBench's official `eval_kernel_against_ref()`
without changing that evaluator's code or semantics.

The project runner will support one ID, an explicit list/range, or all local
L1 IDs. It writes each task's source, wrapper, logs, structured result, and
summary under a run directory. Generation and official evaluation remain
separate phases so failures can be attributed to KernelAgent generation,
compilation, execution, correctness, or timing.

The implementation is `kernelbench_runner/` and the CLI is
`python -m kernelbench_runner.cli` (also installed as `kernelbench-l1`). The
example configuration is `configs/kernelbench_l1.example.json`. It contains a
model name, OpenAI-compatible API base URL, API-key environment variable name,
KernelBench path, output path, search budget, beam width, task selection, GPU
ID, CPU-worker limit, trial counts, and random seed. The runner maps the
configured API-key environment variable to KernelAgent's `OPENAI_API_KEY` only
in its child environment; the key value is not accepted in or written to JSON.

Example task-selection commands (run on the A30 server only):

```bash
python -m kernelbench_runner.cli --config configs/kernelbench_l1.example.json --task-id 19
python -m kernelbench_runner.cli --config configs/kernelbench_l1.example.json --task-range 1:10
python -m kernelbench_runner.cli --config configs/kernelbench_l1.example.json --all
```

Each experiment directory has `config.json`, `task_results.jsonl`,
`kernels/<task>/` (generated function, `ModelNew` wrapper, task result, and
KernelBench result), and `logs/<task>/` (generation/evaluation output). Result
latencies use milliseconds and KernelBench result metadata is retained. The
current KernelAgent API does not report aggregate LLM calls, so `llm_calls` is
`null`; candidate counts are also `null` when the selected upstream route does
not expose them. `beam_width` is recorded for the baseline configuration but
does not activate OptimizationManager beam search in this generation runner.

The wrapper currently instantiates the original task model with the same init
inputs and forwards its parameters/buffers when `kernel_function` declares
additional required positional arguments. This is a thin, provisional
adapter; verify it on A30 for tasks with module parameters, scalar arguments,
and unusual signatures before starting a full L1 run.

## Server verification

- **TODO(server verification):** validate that generated Fuser signatures and
  initialized module parameters map to the `ModelNew` adapter for representative
  elementwise, reduction, and parameterized L1 problems.
- **TODO(server verification):** run one L1 task through KernelBench's official
  local evaluator on A30 before using the batch runner.
- **TODO(server verification):** use a small selected subset before all 100
  tasks, and check timeout, compilation, runtime, and timing metadata.
