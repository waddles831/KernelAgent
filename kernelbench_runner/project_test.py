"""Project-contract correctness test for KernelAgent generation.

KernelAgent normally verifies candidate kernels against a *self-generated*
test, which can differ from :mod:`kernelbench_runner.evaluate_one` in input
dtype, input scale and tolerance.  That is why a kernel can pass internal
verification and still fail the project evaluator (e.g. a bfloat16-only kernel
for a float32 task, or a solver that over-fits the wrong scale).

This module builds a self-contained test script that mirrors the project
evaluator's contract exactly:

* inputs are seeded and cast to float32 (as ``evaluate_one`` does);
* the reference ``Model`` is executed on those inputs;
* the candidate ``kernel_function`` is called using the same parameter/buffer
  forwarding rule as :func:`kernelbench_runner.adapter.wrap_kernel_as_modelnew`;
* outputs are compared structurally with ``torch.allclose(rtol=atol=1e-3)`` and
  an explicit output-dtype match, so silent reduced-precision casts fail.

The test prints ``PASS``/``FAIL: ...`` and exits 0/1, matching the KernelAgent
generation worker's test harness (``from kernel import kernel_function``).
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import tempfile

_DEFAULT_RTOL = 1e-3
_DEFAULT_ATOL = 1e-3

_TEMPLATE = '''\
import inspect
import random
import sys
import types

import torch

_PROBLEM_SRC = {problem_src!r}
_SEED = {seed!r}
_RTOL = {rtol!r}
_ATOL = {atol!r}


def _seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(value, device):
    if torch.is_tensor(value):
        value = value.to(device=device)
        if value.is_floating_point():
            value = value.to(dtype=torch.float32)
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {{key: _to_device(item, device) for key, item in value.items()}}
    return value


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, (tuple, list)):
        return type(value)(_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {{key: _to_cpu(item) for key, item in value.items()}}
    return value


def _compare(reference, candidate, path="output"):
    if torch.is_tensor(reference) or torch.is_tensor(candidate):
        if not (torch.is_tensor(reference) and torch.is_tensor(candidate)):
            return False, path + ": output types differ"
        if tuple(reference.shape) != tuple(candidate.shape):
            return False, "{{}}: shape {{}} != {{}}".format(
                path, tuple(candidate.shape), tuple(reference.shape)
            )
        if reference.is_floating_point() or reference.is_complex():
            if candidate.dtype != reference.dtype:
                return False, "{{}}: dtype {{}} != {{}}".format(
                    path, candidate.dtype, reference.dtype
                )
            if not torch.allclose(reference, candidate, rtol=_RTOL, atol=_ATOL):
                diff = float((reference - candidate).abs().max().item())
                return False, "{{}}: values differ (max_abs_diff={{:.3e}})".format(
                    path, diff
                )
            return True, None
        if not torch.equal(reference, candidate):
            return False, path + ": integer/bool values differ"
        return True, None
    if isinstance(reference, dict) or isinstance(candidate, dict):
        if not (isinstance(reference, dict) and isinstance(candidate, dict)):
            return False, path + ": container types differ"
        if reference.keys() != candidate.keys():
            return False, path + ": output keys differ"
        for key in reference:
            ok, message = _compare(reference[key], candidate[key], "{{}}[{{!r}}]".format(path, key))
            if not ok:
                return False, message
        return True, None
    if isinstance(reference, (tuple, list)) or isinstance(candidate, (tuple, list)):
        if type(reference) is not type(candidate) or len(reference) != len(candidate):
            return False, path + ": output sequence structure differs"
        for index, (ref_item, cand_item) in enumerate(zip(reference, candidate)):
            ok, message = _compare(ref_item, cand_item, "{{}}[{{}}]".format(path, index))
            if not ok:
                return False, message
        return True, None
    if isinstance(reference, (float, int)) and isinstance(candidate, (float, int)):
        if abs(float(reference) - float(candidate)) > _ATOL + _RTOL * abs(float(reference)):
            return False, path + ": scalar values differ"
        return True, None
    if reference != candidate:
        return False, path + ": values differ"
    return True, None


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA unavailable")
        return 1

    from kernel import kernel_function

    namespace = {{}}
    exec(compile(_PROBLEM_SRC, "problem.py", "exec"), namespace)
    model_cls = namespace["Model"]
    get_inputs = namespace["get_inputs"]
    get_init_inputs = namespace.get("get_init_inputs", lambda: [])

    device = torch.device("cuda")
    _seed_all(_SEED)
    init_inputs = [_to_device(item, device) for item in (get_init_inputs() or [])]
    _seed_all(_SEED)
    reference_model = model_cls(*init_inputs).to(device=device, dtype=torch.float32)
    reference_model.eval()

    inputs = [_to_device(item, device) for item in get_inputs()]

    signature = inspect.signature(kernel_function)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    required_count = sum(
        parameter.default is inspect.Parameter.empty for parameter in positional
    )
    additional_count = max(0, required_count - len(inputs))
    state = list(reference_model.parameters())
    state.extend(reference_model.buffers())

    with torch.inference_mode():
        expected = reference_model(*inputs)
        torch.cuda.synchronize(device)
        expected_cpu = _to_cpu(expected)
        del expected
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        actual = kernel_function(*inputs, *state[:additional_count])
        torch.cuda.synchronize(device)
        if actual is None:
            actual = inputs[0]
        actual_cpu = _to_cpu(actual)

    ok, message = _compare(expected_cpu, actual_cpu)
    if not ok:
        print("FAIL: " + str(message))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def build_kernelagent_test_source(
    problem_source: str,
    *,
    seed: int = 42,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
) -> str:
    """Return a standalone project-contract test for the given problem source."""
    return _TEMPLATE.format(
        problem_src=problem_source, seed=seed, rtol=rtol, atol=atol
    )


def build_kernelagent_test_from_path(
    problem_path: Path | str,
    *,
    seed: int = 42,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
) -> str:
    source = Path(problem_path).read_text(encoding="utf-8")
    return build_kernelagent_test_source(source, seed=seed, rtol=rtol, atol=atol)


def run_contract_test(
    kernel_source: str,
    test_source: str,
    *,
    timeout_s: int = 300,
    workdir: Path | str | None = None,
) -> tuple[bool, str]:
    """Run an already-built contract test against a kernel source.

    Writes ``kernel.py`` and ``contract_test.py`` into a temporary directory and
    executes the test (which imports ``from kernel import kernel_function``).
    Returns ``(passed, output)``.
    """
    directory = Path(workdir) if workdir is not None else Path(
        tempfile.mkdtemp(prefix="kb_contract_")
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(kernel_source, encoding="utf-8")
    test_path = directory / "contract_test.py"
    test_path.write_text(test_source, encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return False, f"contract test timed out after {timeout_s}s"
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()
