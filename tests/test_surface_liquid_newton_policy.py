"""CPU nonlinear reuse keeps coupled physics and strict endpoint checks."""
import pytest
import torch

from check_liquid_transient import coupled_model
from surface import torch_liquid
from surface.torch_linear import TorchNewton as LegacyNewton


def test_cpu_full_trial_reduces_work_with_same_coupled_solution(monkeypatch):
    evaluations = []
    original = torch_liquid.TorchLiquidEquations.evaluate
    def counted(self, *args, **kwargs):
        evaluations.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch_liquid.TorchLiquidEquations, "evaluate", counted)
    def run():
        m = coupled_model(.001)
        state = m.initialize({"h":550000., "t":500000.}, {"line":.002}, {"a":100., "b":110.})
        evaluations.clear()
        out = m.advance(state, .02, motor_torques_nm={"a":5., "b":7.})
        return out, len(evaluations)
    modern, new_count = run()
    monkeypatch.setattr(torch_liquid, "_CpuNewton", LegacyNewton)
    legacy, old_count = run()
    assert new_count < old_count
    assert len(modern["steps"]) == len(legacy["steps"])
    for key, atol in (("pressures_pa", 1e-6), ("pipe_volume_rates_m3_s",1e-12), ("speeds_rad_s",1e-9)):
        assert modern["state"][key] == pytest.approx(legacy["state"][key], abs=atol, rel=0)
    assert max(abs(s["energy_residual_j"]) for s in modern["steps"]) < 1e-8


def test_postprojection_rechecks_actual_requested_residual(monkeypatch):
    model = coupled_model(.001)
    e = torch_liquid.TorchLiquidEquations(model)
    fixed = [{"in":400000., "out":480000.}.get(n, 500000.) for n in model._nodes]
    args = tuple(e.tensor(v) for v in ([550000.,500000.], [.002], [100.,110.],
                                      .001, [5.,7.], fixed, [1.]))
    x, _, _, ok, _ = e._solve_impl(*args)
    assert bool(ok)
    disturbed = x.clone()
    disturbed[0] += 1e-10
    def false_success(*a, **kw):
        return disturbed, torch.zeros_like(disturbed), disturbed.new_tensor(True, dtype=torch.bool), 1
    monkeypatch.setattr(e.newton, "solve", false_success)
    _, residual, _, accepted, _ = e._solve_impl(*args)
    assert 2e-11 < float(residual.abs().max()) < 1e-8
    assert not bool(accepted)
