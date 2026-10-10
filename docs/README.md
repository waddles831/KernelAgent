# Working in this repo on the A30 server

Operational guide for running the KernelBench L1 runner here. Read this before
running anything. It records the interpreter, server configuration, exact
commands, and the mistakes that waste the most time.

Related docs:
- `docs/KERNELBENCH_L1.md` — evaluation protocol and task semantics.
- `docs/KERNELAGENT_ARCHITECTURE.md` — generation pipeline overview.
- `docs/MEMORY_INTEGRATION_NOTES.md` — cross-task memory schema, anti-leakage
  rules, status (implemented vs planned), and hook points.
- `docs/A30_SERVER_RUNBOOK.md` — the Docker-based runbook (do **not** use it on
  this host; see "Hard rules").

## Quick facts

| Item | Value |
| --- | --- |
| Git root / repo root | `/data/root/work/KernelAgent` |
| KernelBench dataset | `/data/root/work/KernelBench` (tasks in `KernelBench/level1/`) |
| Python interpreter | `/data/root/venvs/kernelagent/bin/python3.10` |
| Python / Torch / Triton | 3.10.12 / `2.9.1+cu126` / `3.5.1` |
| GPUs | 4 × NVIDIA A30 (compute cap 8.0), driver 535.86.10, CUDA 12.2 |
| GPUs allowed by the runner | `--gpu-id 0` or `1` only (mapped to `CUDA_VISIBLE_DEVICES`) |

There is **no activated virtualenv by default** in a fresh shell. `python3.10`
resolves to `/usr/bin/python3.10` (no torch/triton). Always call the venv
interpreter by absolute path, or activate it first:

```bash
source /data/root/venvs/kernelagent/bin/activate
# or, preferred for scripts:
PY=/data/root/venvs/kernelagent/bin/python3.10
```

## Server configuration (model / endpoint / key)

The server runs the DeepSeek-compatible endpoint. Effective values:

| Setting | Value |
| --- | --- |
| `model` | `deepseek-flash` |
| `api_base_url` | `https://api.deepseek.com` |
| `api_key_env` | `DEEPSEEK_API_KEY` |

The key value lives in the untracked `.env` at the repo root and is loaded by
`kernelbench_runner` at startup; the runner copies it into `OPENAI_API_KEY` for
the KernelAgent provider. **Never print, echo, or write the key value** (it will
otherwise end up in logs/artifacts). The tracked example
`configs/kernelbench_l1.example.json` uses a different model and must not be
edited; use a temporary config or CLI overrides instead.

Note: `deepseek-flash` is a reasoning model whose reasoning tokens count against
the completion budget. `utils/providers/openai_provider.py` raises its limit to
32768; without that, reasoning-heavy tasks return empty content and the
generation stage fails.

## Preflight checks

```bash
cd /data/root/work/KernelAgent
PY=/data/root/venvs/kernelagent/bin/python3.10

# interpreter + CUDA
$PY -c "import torch,triton;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), triton.__version__)"

# KernelBench task files exist
ls /data/root/work/KernelBench/KernelBench/level1 | head

# GPU state; only run when the target GPU is idle (memory ~21MiB, util 0%)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

## Run a smoke test (the supported path)

`kernelbench_runner` is the only supported entry point. It generates kernels via
KernelAgent and evaluates them with `kernelbench_runner.evaluate_one`; it does
**not** use KernelBench's `eval.py` or any eval-only harness.

1. Create a temporary config (server settings; keep it out of git). Example at
   `/tmp/opencode/kernelbench_l1.deepseek.json`:

   ```json
   {
     "kernelbench_path": "/data/root/work/KernelBench",
     "output_root": "/data/root/work/KernelAgent/runs",
     "model": "deepseek-flash",
     "api_base_url": "https://api.deepseek.com",
     "api_key_env": "DEEPSEEK_API_KEY",
     "gpu_id": 0,
     "gpu_arch": ["Ampere"],
     "cpu_worker_limit": 4,
     "candidate_count": 4,
     "search_budget": 5,
     "beam_width": 2,
     "random_seed": 42,
     "num_correct_trials": 5,
     "benchmark_warmup_ms": 1000,
     "benchmark_rep_ms": 100,
     "generation_timeout_s": 1800,
     "llm_timeout_s": 1200,
     "agent_test_timeout_s": 300,
     "evaluation_timeout_s": 300
   }
   ```

2. Clean only the repo-root `runs/` (verify the path first):

   ```bash
   GROOT=$(git rev-parse --show-toplevel)      # /data/root/work/KernelAgent
   rm -rf "$GROOT/runs" && mkdir -p "$GROOT/runs"
   ```

3. Launch from the repo root with a **unique** experiment name. Pass multiple
   `--task-id` (never `--all`). Use `setsid` so the run survives shell/tool
   timeouts:

   ```bash
   EXP="l1_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
   LOG="/tmp/opencode/${EXP}.log"
   cd /data/root/work/KernelAgent
   setsid /data/root/venvs/kernelagent/bin/python3.10 -m kernelbench_runner.cli \
     --config /tmp/opencode/kernelbench_l1.deepseek.json \
     --experiment-name "$EXP" \
     --kernelbench-path /data/root/work/KernelBench \
     --gpu-id 0 \
     --seed 42 --num-correct-trials 5 \
     --benchmark-warmup-ms 1000 --benchmark-rep-ms 100 \
     --task-id 1 --task-id 40 --task-id 88 --task-id 50 --task-id 95 \
     >"$LOG" 2>&1 </dev/null &
   ```

   If the shell that launched it is killed, the run keeps going because it is in
   its own session. Do not run the launcher through a foreground wait that can
   be timed out.

4. Monitor progress (results are written only after each task completes):

   ```bash
   EXP=$(ls -1t /data/root/work/KernelAgent/runs | head -1)
   ls  /data/root/work/KernelAgent/runs/$EXP/kernels
   wc -l /data/root/work/KernelAgent/runs/$EXP/task_results.jsonl
   pgrep -af kernelbench_runner.cli
   ```

## Outputs

Everything lands under `runs/<experiment_name>/`:

- `config.json` — resolved configuration plus the evaluation/benchmark policy;
  includes a `memory` block (mode, directories, and the retrieval memory version
  at run start) when memory is enabled.
- `task_results.jsonl` — one schema-v2 record per task. When memory is enabled
  each record's `metadata.memory` reports the memory version actually used, the
  number of attempt/retrieval records ingested, and the mode.
- `kernels/<NNN_task>/` — `kernel_function.py`, `ModelNew.py`,
  `task_result.json`, `kernelagent_result.json`,
  `kernelagent_benchmark_result.json`; with memory on also
  `memory_config.json`, and (with `--optimize`) `optimization_result.json`,
  `kernel_function_optimized.py`.
- `logs/<NNN_task>/` — `kernelagent.log`, `kernelagent_benchmark.log`; with
  memory + `--optimize`, `optimization/optimization_attempts.jsonl` and
  `optimization/optimization_retrievals.jsonl`.
- `runner_console.log` — copy of the launcher stdout if you saved it.
- Memory store (when enabled, default `memory/`) — `task_results.jsonl`,
  `attempts.jsonl`, `summaries.jsonl`, `retrieval_log.jsonl`.

Per-task `evaluation_status` is one of `passed`, `generation_failed`,
`generation_timeout`, `adapter_failed`, `compile_failed`, `runtime_failed`,
`oom`, `incorrect`, `timing_failed`, `evaluation_failed`, `evaluation_timeout`.
`speedup = torch.compile(reference) / candidate`. `oom` is a non-evaluable
status (correctness/benchmark ran out of GPU memory); exclude it from speedup
aggregates instead of counting it as incorrect. Memory-heavy L1 tasks
(e.g. 19–32, 38–39 with ~6–8 GiB input tensors) may OOM on a 24 GiB A30.

For the smoke set the expected shape is: simple elementwise/reduction tasks pass
(e.g. tasks 1, 88, 95); harder tasks (40 LayerNorm, 50 Conv2d 11×11 s4 p2) only
pass once the `deepseek-flash` token budget fix is in place. `speedup < 1` is
normal here because the baseline is `torch.compile` and the generated kernels are
correctness-oriented, not tuned.

**Baseline scope:** the current 5-task run (`runs/l1_smoke_*`, `runs/l1_verify_*`)
is for **engineering flow validation only**. A full L1 baseline and any
memory-benefit evaluation are deferred and are not prerequisites for the memory
implementation.

## Optional optimization stage (`--optimize`) — currently OFF and blocked

`kernelbench_runner` can run KernelAgent's hardware-guided `OptimizationManager`
on a generated kernel before evaluation. It is **off by default**. Enable with
`--optimize` plus `--optimization-strategy greedy|beam_search`,
`--optimization-rounds`, etc.

It cannot run in this container: `OptimizationManager` requires NVIDIA Nsight
Compute (`ncu`), which needs GPU performance-counter access. NCU 2023.2.2 is
installed, but profiling fails with `ERR_NVGPUCTRPERM` because the container
lacks `CAP_SYS_ADMIN` / `nvidia-caps` device access and the host driver has
`RmProfilingAdminOnly=1`. This must be fixed on the host
(`--cap-add SYS_ADMIN` + `/dev/nvidia-caps`, or
`NVreg_RestrictProfilingToAdminUsers=0`). Until then `--optimize` degrades to a
no-op and the original kernel is used.

## Cross-task memory (`--memory-mode`) — implemented, default OFF

`kernel_memory/` stores versioned, local, newline-delimited JSON experience.
It is **off by default**; `--memory-mode off` (the default) preserves the
previous behavior exactly.

| Mode | Retrieval | Writes |
| --- | --- | --- |
| `off` (default) | none | none |
| `collect` | none | task results + attempt evidence |
| `frozen` | frozen snapshot only | this run's evidence (separate dir) |
| `continual` | store at task start | append after each task |

Flags: `--memory-mode`, `--memory-dir`, `--memory-snapshot-dir`,
`--memory-max-items`, `--memory-prompt-budget-chars`,
`--memory-include-failures`, `--memory-allow-foreign-compat`,
`--memory-record-generation-traces`.

Offline, CPU-only tooling (never starts GPU work or model calls):

```bash
PY=/data/root/venvs/kernelagent/bin/python3.10
# Explicit coverage report for existing runs (missing data is marked, not inferred)
$PY -m kernelbench_runner.memory_cli inspect-runs --output-root runs
# Audit a memory store
$PY -m kernelbench_runner.memory_cli report --store memory
# Offline, rule-based strategy-summary construction
$PY -m kernelbench_runner.memory_cli summarize --store memory
# Freeze a snapshot for a frozen-memory experiment
$PY -m kernelbench_runner.memory_cli snapshot --store memory \
  --dest memory_snapshots/source_v1 --label source_v1
```

Retrieval is wired into **both** prompt-construction sites: seed generation
(before any seed exists, using only source-derived features with
`bottleneck=unknown`) and the optimization stage (after the bottleneck
diagnosis). It is therefore active even with `--optimize` off, which is what
makes the memory factor independent of `optimize`. Static RAG examples and
measured memory are rendered as separate prompt blocks. When NCU is
unavailable, generation-arm memory carries internal-test evidence only (no
parent-relative performance or profile); see `docs/MEMORY_INTEGRATION_NOTES.md`.

Control configs (no secrets; set `model`/`api_*` for your server). The
`O+` rows need NCU; the `O-` row runs without it:

| Arm | Config | optimize | memory |
| --- | --- | --- | --- |
| O- M- | `configs/memory_gen_off.json` | off | off |
| O- M+ | `configs/memory_gen_frozen.json` | off | frozen (generation retrieval) |
| O+ M- | `configs/memory_control_off.json` | on | off |
| O+ M+ | `configs/memory_frozen_unseen_eval.json` | on | frozen |
| source | `configs/memory_source_collect.json` | on | collect |
| reuse | `configs/memory_continual_retest.json` | on | continual |

Frozen arms read `memory_snapshots/source_v1`, built from the source config via
`memory_cli snapshot`. These configs are templates; running them is a separate,
explicitly-started GPU experiment. See `docs/MEMORY_INTEGRATION_NOTES.md` for the
schema, anti-leakage rules, and what is implemented versus planned.

## Numerical precision contract

Benchmarking only runs after all correctness trials pass. A
dtype/precision-contract violation (e.g. a bfloat16-only kernel for a float32
task) is recorded as a **correctness failure** and never earns performance
credit (no `candidate_latency_ms`, no `speedup`).

Two mechanisms enforce the contract:

- **Generation** verifies candidates against a self-contained project-contract
  test (`kernelbench_runner.project_test`) instead of an LLM-generated test:
  seeded float32 inputs, structural comparison, `torch.allclose(rtol=atol=1e-3)`
  and an explicit output-dtype match. Disable with
  `--no-generation-project-contract`.
- **Evaluation** pins the reference/benchmark precision policy; by default
  `torch.backends.cudnn.allow_tf32 = False` and
  `torch.backends.cuda.matmul.allow_tf32 = False` so correctness is not
  confounded by a hardware TF32 path. The policy is recorded in
  `kernelagent_benchmark_result.json` (`metadata.precision_policy`). Override
  with `--correctness-allow-tf32`.

## Hard rules

- Do **not** use Docker here, and do not run `scripts/build_a30_image.sh` or
  `scripts/server_check.sh`.
- Do **not** install Python dependencies into the system interpreter; use the
  `kernelagent` venv.
- Do **not** modify `configs/kernelbench_l1.example.json`, KernelBench task
  files, or KernelBench's upstream evaluator. Use a temp config/CLI overrides.
- Do **not** use KernelBench's `eval.py`. Evaluation goes through
  `kernelbench_runner.evaluate_one` only.
- Never commit `.env`, `runs/`, `.fuse/`, `triton_kernel_logs/`, `memory/`,
  `memory_snapshots/`, or key values.

## Common failure modes

- **`No module named 'torch'` / `'dotenv'`**: you used system `python3.10`.
  Use `/data/root/venvs/kernelagent/bin/python3.10`.
- **Run stops after the first task**: the launcher was killed with its process
  group (e.g. a foreground wait timed out). Relaunch with `setsid`.
- **`generation_failed` with empty content**: reasoning overflow on a model
  whose completion budget is too small (see the `deepseek-flash` note above).
- **Another run's results vanished**: `runs/` was cleaned. Confirm the Git root
  and `runs/` absolute path first; only clean `$(git rev-parse --show-toplevel)/runs`.
- **Memory seems to do nothing**: retrieval is only wired into the optimization
  stage, so `--memory-mode frozen|continual` needs `--optimize` too. `collect`
  writes evidence without retrieving.
- **Frozen mode errors on startup**: `frozen` requires `--memory-snapshot-dir`
  (or `memory_snapshot_dir`); create it with
  `python -m kernelbench_runner.memory_cli snapshot`.
