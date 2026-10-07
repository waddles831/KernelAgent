#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [CONFIG_JSON]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNELBENCH_ROOT="${KERNELBENCH_PATH:-$REPO_ROOT/../KernelBench}"
IMAGE_NAME="${KERNELAGENT_IMAGE:-kernelagent-a30:latest}"
API_KEY_ENV="${KERNELAGENT_API_KEY_ENV:-OPENAI_API_KEY}"
RUN_NAME="${KERNELAGENT_EXPERIMENT_NAME:-l1_baseline_$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG_INPUT="${1:-$REPO_ROOT/configs/kernelbench_l1.example.json}"

[[ -f "$CONFIG_INPUT" ]] || { echo "Config not found: $CONFIG_INPUT" >&2; exit 2; }
[[ -d "$KERNELBENCH_ROOT/KernelBench/level1" ]] || {
  echo "KernelBench checkout not found under $KERNELBENCH_ROOT" >&2
  exit 2
}

DOCKER_ARGS=(run --rm --gpus all
  --volume "$REPO_ROOT:/workspace/KernelAgent"
  --volume "$KERNELBENCH_ROOT:/workspace/KernelBench:ro"
  --volume "$CONFIG_INPUT:/tmp/kernelbench_l1.json:ro"
  --workdir /workspace/KernelAgent
)
if printenv "$API_KEY_ENV" >/dev/null 2>&1; then
  DOCKER_ARGS+=(--env "$API_KEY_ENV")
elif [[ -f "$REPO_ROOT/.env" ]]; then
  DOCKER_ARGS+=(--volume "$REPO_ROOT/.env:/workspace/KernelAgent/.env:ro")
else
  echo "Export $API_KEY_ENV or add it to the ignored .env file" >&2
  exit 2
fi

docker "${DOCKER_ARGS[@]}" "$IMAGE_NAME" \
  python3.10 -m kernelbench_runner.cli \
    --config /tmp/kernelbench_l1.json \
    --experiment-name "$RUN_NAME" \
    --kernelbench-path /workspace/KernelBench \
    --all
