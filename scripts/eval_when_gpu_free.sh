#!/usr/bin/env bash
set -euo pipefail

# Wait for a GPU with enough free memory, then run a command pinned to that GPU
# exactly once. Useful when the box is shared and you only race for a free card.
#
# Usage:
#   eval_when_gpu_free.sh [options] -- COMMAND [ARGS...]
#
# Options / env vars:
#   --min-free-mib N   Required free memory per GPU, MiB (env MIN_FREE_MIB, default 18000)
#   --poll-seconds N   Poll interval, seconds          (env POLL_SECONDS, default 15)
#   --confirm N        Consecutive qualifying polls    (env CONFIRM, default 2)
#
# Example:
#   PYTHONPATH="/data/root/work/KernelBench/src:/data/root/work/KernelAgent" \
#   scripts/eval_when_gpu_free.sh --min-free-mib 18000 -- \
#     python -m kernelbench_runner.evaluate_one \
#       --problem  /data/root/work/KernelBench/KernelBench/level1/19_ReLU.py \
#       --modelnew runs/l1_deepseek_task19_smoke/kernels/019_19_ReLU/ModelNew.py \
#       --output   runs/l1_deepseek_task19_smoke/kernels/019_19_ReLU/kernelbench_result_gpu1.json \
#       --seed 42 --num-correct-trials 5 --num-perf-trials 100 \
#       --timing-method cuda_event --gpu-arch Ampere
#
# NOTE: do NOT set CUDA_VISIBLE_DEVICES yourself; this script picks a free GPU.

MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
POLL_SECONDS="${POLL_SECONDS:-15}"
CONFIRM="${CONFIRM:-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --confirm)      CONFIRM="$2"; shift 2 ;;
    --)             shift; break ;;
    -h|--help)      sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "No command given. Usage: $0 [--min-free-mib N] [--poll-seconds N] -- CMD [ARGS...]" >&2
  exit 2
fi

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',[[:space:]]*' -v min="$MIN_FREE_MIB" '$2+0 >= min {print $2, $1}' \
    | sort -rn | awk 'NR==1 {print $2}'
}

echo "[wait-gpu] waiting for a GPU with >= ${MIN_FREE_MIB} MiB free (poll ${POLL_SECONDS}s, confirm ${CONFIRM}x)"
last=""
streak=0
while true; do
  gpu="$(pick_gpu)"
  if [[ -n "$gpu" ]]; then
    if [[ "$gpu" == "$last" ]]; then
      streak=$((streak + 1))
    else
      streak=1
      last="$gpu"
    fi
    if (( streak >= CONFIRM )); then
      echo "[wait-gpu] GPU ${gpu} has enough free memory (${streak} consecutive polls); launching"
      break
    fi
  else
    streak=0
    last=""
  fi
  sleep "$POLL_SECONDS"
done

export CUDA_VISIBLE_DEVICES="$gpu"
# kernelbench.utils imports litellm, whose remote cost-map fetch spawns a
# background thread that races the lazy rust_bridge import on offline boxes.
export LITELLM_LOCAL_MODEL_COST_MAP="${LITELLM_LOCAL_MODEL_COST_MAP:-True}"
echo "[wait-gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[wait-gpu] exec: $*"
exec "$@"
