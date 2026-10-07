# WSL2 Development Environment

## Local virtual environment

The repository's `pyproject.toml` is the authoritative Python constraint for
KernelAgent: Python `>=3.10,<3.13`. The README's older `3.8–3.12` range does not
match that metadata. This WSL2 machine provides Python 3.12, so the local
development environment uses it:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install 'pytest>=7.0.0' pytest-cov
```

The current development virtual environment contains only the tools listed in
KernelAgent's `[project.optional-dependencies].dev` group. It does not install
KernelAgent's runtime/UI dependencies, PyTorch, Triton, CUDA, or KernelBench's
experiment dependencies. `.venv/` is ignored by Git.

## KernelBench compatibility

The sibling `../KernelBench/pyproject.toml` currently pins Python to
`==3.10.*`. The future A30 environment should therefore use Python 3.10 so both
repositories can share one environment. This local Python 3.12 environment is
for framework and pure-Python development; it is not the KernelBench execution
environment.

## GPU work

Do not install or run GPU dependencies in this WSL2 environment. KernelAgent's
README expects the user to install PyTorch and Triton separately for the target
GPU/toolchain, and KernelBench's `gpu` extra includes CUDA-oriented packages.
Install those only in the later A30 server environment, pinned to the
upstream-compatible versions established during server setup.
