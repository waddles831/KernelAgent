# KernelBench's current Modal runner uses this CUDA base and Python 3.10.
# Do not build or run this image from the WSL2 development workflow.
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/KernelBench/src:/workspace/KernelAgent

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cuda-nsight-compute-13-0 \
        git \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/KernelAgent
COPY . /workspace/KernelAgent

# KernelBench requirements.txt pins the tested Torch/Triton family. The
# KernelBench source is mounted beside this repository when the container runs.
# Its core evaluator imports litellm through kernelbench.utils, so install that
# runtime dependency without the unrelated dataset/Modal/DSL extras.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3.10 -m pip install --no-cache-dir -e . \
    && python3.10 -m pip install --no-cache-dir \
        "torch==2.9.*" \
        "triton==3.5.*" \
        "litellm[proxy]>=1.80.10" \
        "tqdm>=4.67.1" \
        "ninja>=1.13.0" \
        "einops>=0.8.1"

CMD ["/bin/bash"]
