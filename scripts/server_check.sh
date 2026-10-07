#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${KERNELAGENT_IMAGE:-kernelagent-a30:latest}"

command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is not available on the A30 host" >&2
  exit 1
}
command -v docker >/dev/null || {
  echo "docker is not available on the A30 host" >&2
  exit 1
}
docker image inspect "$IMAGE_NAME" >/dev/null || {
  echo "Image $IMAGE_NAME not found; build it with scripts/build_a30_image.sh" >&2
  exit 1
}

echo "Host GPU inventory:"
HOST_GPU_INFO="$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader)"
printf '%s\n' "$HOST_GPU_INFO"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$GPU_COUNT" -ne 2 ]]; then
  echo "Expected the allocated 2 A30 GPUs, found $GPU_COUNT" >&2
  exit 1
fi
if [[ "$(printf '%s\n' "$HOST_GPU_INFO" | grep -ci 'A30')" -ne 2 ]]; then
  echo "Expected two A30 GPUs in the server allocation" >&2
  exit 1
fi

echo "Container GPU passthrough:"
docker run --rm --gpus all "$IMAGE_NAME" nvidia-smi \
  --query-gpu=index,name,compute_cap,memory.total \
  --format=csv,noheader

echo "Nsight Compute CLI availability:"
docker run --rm "$IMAGE_NAME" bash -lc 'command -v ncu && ncu --version | head -n 2'

echo "PyTorch visibility and architecture (no kernel workload):"
docker run --rm --gpus all "$IMAGE_NAME" python3.10 -c '
import torch
count = torch.cuda.device_count()
print("torch", torch.__version__, "CUDA", torch.version.cuda, "visible GPUs", count)
for index in range(count):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    print(index, name, capability)
    if "A30" not in name:
        raise SystemExit(f"Expected A30, found {name}")
if count != 2:
    raise SystemExit("Expected exactly two visible GPUs")
if any(torch.cuda.get_device_capability(index) != (8, 0) for index in range(count)):
    raise SystemExit("Expected A30 compute capability 8.0")
'
