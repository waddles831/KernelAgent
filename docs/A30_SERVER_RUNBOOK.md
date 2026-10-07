# A30 Server Runbook

This runbook is for the A30 server. Do not run these Docker/GPU commands in the
WSL2 development environment.

## Prepare model configuration

Edit a copy of `configs/kernelbench_l1.example.json` for the selected model and
its OpenAI-compatible endpoint (for a DeepSeek-compatible service, use its
model ID, endpoint, and key variable). Set `api_key_env` to the environment
variable name that holds the key, then export that variable on the server or
put it in the untracked `.env` file. If exporting a non-default variable for
the Docker scripts, also set `KERNELAGENT_API_KEY_ENV` to that variable name.
Never place the key value in the JSON config or Git.
The example uses the Ampere target, physical GPU 0, one selected smoke task,
five correctness trials, and 100 performance trials.

## First server sequence

From the KernelAgent repository on the server:

```bash
./scripts/build_a30_image.sh
./scripts/server_check.sh
./scripts/run_single_l1.sh 19 configs/kernelbench_l1.example.json
```

The image follows the CUDA 13.0 / Ubuntu 22.04 / Python 3.10 base used by the
current KernelBench Modal runner and the `torch==2.9.*`, `triton==3.5.*` family
in KernelBench's requirements. `../KernelBench` is mounted read-only beside
KernelAgent at runtime. Check that the host and container each see exactly two
A30 GPUs with compute capability 8.0 before evaluation.

Inspect the single-task output under `runs/<experiment_name>/`: confirm
`kernelagent_result.json` shows a successful generation, then inspect
`kernelbench_result.json` and `task_result.json` for compilation, runtime,
correctness, same-device reference latency, candidate latency, and speedup. A
KernelAgent self-test pass alone does not establish KernelBench correctness.

After the first task and a second representative task (for example, a
parameterized convolution or reduction) pass, run the full local L1 baseline:

```bash
./scripts/run_l1_baseline.sh configs/kernelbench_l1.example.json
```

That command evaluates all task IDs in the mounted `KernelBench/level1`
directory sequentially on the configured GPU. To use the other allocated GPU,
set `gpu_id` to `1` in a separate config and use a distinct experiment name.
There is no automatic multi-GPU task scheduler in this phase.

## First-run decisions to record

- Confirm the driver and container CUDA compatibility and that the pinned Torch
  and Triton wheels compile for `sm_80`.
- Verify the generated `ModelNew` adapter on an elementwise task and on tasks
  with initialized module parameters and non-default scalar arguments.
- Confirm KernelBench correctness, timing, and timeout records before running
  the full 100-task batch.
- Save the exact image tag, driver output, model ID, endpoint, seed, and run
  artifacts for the baseline.
