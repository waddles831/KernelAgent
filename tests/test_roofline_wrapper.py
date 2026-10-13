"""The platform roofline wrappers must expose ``config`` for the orchestrator."""

from __future__ import annotations


def test_nvidia_roofline_analyzer_exposes_config():
    from triton_kernel_agent.platform.nvidia import NvidiaRooflineAnalyzer

    analyzer = NvidiaRooflineAnalyzer()
    assert hasattr(analyzer.config, "early_stop")


def test_noop_roofline_analyzer_exposes_config():
    from triton_kernel_agent.platform.noop import NoOpRooflineAnalyzer

    assert hasattr(NoOpRooflineAnalyzer().config, "early_stop")
