"""Project-contract generation test and evaluator precision pinning."""

from __future__ import annotations

from kernelbench_runner.project_test import (
    build_kernelagent_test_from_path,
    build_kernelagent_test_source,
)

_PROBLEM = '''\
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)


def get_inputs():
    return [torch.randn(4, 8)]


def get_init_inputs():
    return []
'''


def test_contract_source_compiles_and_checks_dtype(tmp_path):
    source = build_kernelagent_test_source(_PROBLEM, seed=7, rtol=1e-3, atol=1e-3)
    compile(source, "contract_test.py", "exec")
    assert "from kernel import kernel_function" in source
    assert "torch.allclose" in source
    assert "dtype" in source
    assert "_SEED = 7" in source


def test_contract_from_path_reads_file(tmp_path):
    problem = tmp_path / "problem.py"
    problem.write_text(_PROBLEM, encoding="utf-8")
    source = build_kernelagent_test_from_path(problem, seed=1)
    compile(source, "contract_test.py", "exec")


def test_router_uses_provided_test_only(monkeypatch):
    import Fuser.auto_agent as auto_agent

    captured: dict = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def generate_kernel(
            self, problem_description, test_code=None, generate_default_test=True
        ):
            captured["test_code"] = test_code
            captured["generate_default_test"] = generate_default_test
            return {"success": True, "kernel_code": "def kernel_function(): pass"}

        def cleanup(self):
            pass

    monkeypatch.setattr(auto_agent, "TritonKernelAgent", FakeAgent)
    router = auto_agent.AutoKernelRouter(ka_model="test-model")

    result = router._solve_with_kernelagent("PROBLEM", "CONTRACT")
    assert captured["test_code"] == "CONTRACT"
    assert captured["generate_default_test"] is False
    assert result.success is True

    router._solve_with_kernelagent("PROBLEM", None)
    assert captured["test_code"] is None
    assert captured["generate_default_test"] is True


def test_runner_config_defaults_for_contract_and_precision():
    from kernelbench_runner.config import RunnerConfig

    config = RunnerConfig()
    assert config.generation_project_contract is True
    assert config.correctness_allow_tf32 is False


def test_cli_parses_contract_and_precision_flags():
    from kernelbench_runner.cli import _parser

    args = _parser().parse_args(
        [
            "--no-generation-project-contract",
            "--correctness-allow-tf32",
        ]
    )
    assert args.generation_project_contract is False
    assert args.correctness_allow_tf32 is True


def test_run_contract_test_plumbing(tmp_path):
    from kernelbench_runner.project_test import run_contract_test

    ok, output = run_contract_test(
        "def kernel_function(x): return x", "print('PASS')", workdir=tmp_path / "a"
    )
    assert ok and "PASS" in output

    ok, output = run_contract_test(
        "def kernel_function(x): return x",
        "import sys\nprint('FAIL: bad')\nsys.exit(1)\n",
        workdir=tmp_path / "b",
    )
    assert not ok and "FAIL" in output


def test_router_gate_rejects_failing_fuser_result():
    from Fuser.auto_agent import AutoKernelRouter, RouteResult

    failing = AutoKernelRouter(contract_verifier=lambda code: (False, "diff too big"))
    result = failing._gate_fuser_with_contract(
        RouteResult(route="fuser", success=True, details={}, kernel_code="K"), "TEST"
    )
    assert result.success is False
    assert "contract_test_failed" in result.details

    passing = AutoKernelRouter(contract_verifier=lambda code: (True, "PASS"))
    result = passing._gate_fuser_with_contract(
        RouteResult(route="fuser", success=True, details={}, kernel_code="K"), "TEST"
    )
    assert result.success is True
    assert result.details["contract_test_passed"] is True


def test_router_gate_is_noop_without_verifier_or_test():
    from Fuser.auto_agent import AutoKernelRouter, RouteResult

    router = AutoKernelRouter()
    res = RouteResult(route="fuser", success=True, details={}, kernel_code="K")
    assert router._gate_fuser_with_contract(res, "TEST").success is True
    assert router._gate_fuser_with_contract(res, None).success is True

