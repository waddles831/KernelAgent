# KernelAgent evaluation on KernelBench L1 tasks

KernelBench supplies the local L1 problem files and task IDs. The runner loads
each task's `Model`, `get_init_inputs()`, and `get_inputs()` as the workload and
reference implementation. It does not call `kernelbench.eval` or import
KernelBench timing utilities.

## Evaluation protocol

The project-owned evaluator is `kernelbench_runner.evaluate_one`:

1. Load the task's PyTorch `Model` and the generated `ModelNew` candidate.
   Initialize both with the same seed so parameterized tasks start with matching
   weights.
2. Compare outputs on five deterministically seeded FP32 input sets by default.
   Every output must have the same structure and shape and pass
   `torch.allclose(rtol=1e-3, atol=1e-3)`. Performance is measured only after
   all correctness trials pass.
3. Compile the reference with default `torch.compile` mode, static shapes, and
   CUDA graphs disabled. Measure it and the candidate with
   `triton.testing.do_bench`, reporting mean latency. Defaults are a 1000 ms
   warmup budget and a 100 ms repetition budget. A host-local file lock
   serializes benchmark sections from runner processes using the same GPU.
4. Report `speedup = torch.compile latency / candidate latency`.

This follows the public KernelAgent performance protocol's use of its own
correctness-gated Benchmarking Agent and `triton.testing.do_bench`; the public
report describes mean latency, a warmup longer than one second, and comparison
against default `torch.compile` on H100. Our selected workload is still the
KernelBench L1 set, while the local runs execute on the configured hardware
(currently A30), so their absolute latencies and aggregate speedups are not the
published H100 results. See the [KernelAgent performance report](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/).

The correctness tolerance and number of input trials are explicit project
settings. KernelBench remains the source of the reference model and input
generator because those define each task; its evaluation API, timing code, and
reported speedup are not used.

## Configure and run

The example configuration is `configs/kernelbench_l1.example.json`. It
contains the local dataset path, task selection, model/provider settings, GPU,
correctness trial count, and `benchmark_warmup_ms` / `benchmark_rep_ms`.
API key values are read from the named environment variable and are never
written to run artifacts.

```bash
python -m kernelbench_runner.cli --config configs/kernelbench_l1.example.json --task-id 19
python -m kernelbench_runner.cli --config configs/kernelbench_l1.example.json --task-range 1:10
python -m kernelbench_runner.cli --config configs/kernelbench_l1.example.json --all
```

Each new run stores `config.json`, `task_results.jsonl`, generated source,
`kernelagent_benchmark_result.json`, and generation/benchmark logs under
`runs/<experiment_name>/`. New result metadata records the evaluator,
benchmark method, reference baseline, GPU, Torch/Triton versions, and timing
budgets. Existing runs created with `kernelbench.eval.eval_kernel_against_ref`
are historical results under the former protocol and remain unchanged; do not
aggregate them with new KernelAgent-benchmark runs.

## Hardware and task-size notes

The runner processes tasks sequentially on the configured visible GPU. A local
lock prevents concurrent benchmark sections from this runner on the same host.
It cannot coordinate workloads running in a different container or host, so
record GPU contention when interpreting results.

Some L1 tasks exceed the available A30 memory during correctness or
`torch.compile` benchmarking. Keep those as explicit non-evaluable/OOM cases
for this hardware and exclude them from speedup aggregates; do not modify the
upstream KernelBench evaluator or task definitions to work around capacity.
