"""PID input must match OPM's squared norm, with restart-safe history."""
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator.simulation import Simulator


@pytest.mark.parametrize('pressure_scale', [0., 1., 1e5])
def test_pid_uses_squared_change_of_pressure_and_all_saturations(pressure_scale):
    p0 = torch.tensor([80., 90.], dtype=torch.float64) * pressure_scale
    p1 = torch.tensor([100., 100.], dtype=torch.float64) * pressure_scale
    s0 = torch.tensor([[.2, .4], [.1, .2]], dtype=torch.float64)
    s1 = torch.tensor([[.3, .4], [.2, .2]], dtype=torch.float64)
    model = SimpleNamespace(fluid=SimpleNamespace(pressure=p1, s_w=s1[0], s_g=s1[1]),
                            _pid_err=[.001] * 3)
    old = torch.cat([p0, s0.flatten(), 1-s0.sum(0)])
    new = torch.cat([p1, s1.flatten(), 1-s1.sum(0)])
    expected = float((new-old).square().sum() / new.square().sum())
    dt = Simulator._opm_dt(model, 100., 8, p0, s0, 8, .001)
    assert model._pid_err[-1] == pytest.approx(expected, rel=1e-14)
    assert dt == pytest.approx(100. * .001 / expected)


@pytest.mark.parametrize('version', [None, 2])
def test_pid_resets_only_incompatible_checkpoint_history(version):
    sim = Simulator.__new__(Simulator)
    sim.sim_params = {'dt_control': 'opm'}
    sim.current_time = 0.
    sim.fluid = SimpleNamespace(pressure=torch.tensor([1e7]),
                               s_w=torch.tensor([.3]), s_g=torch.tensor([.1]))
    sim.fim_solver = SimpleNamespace(debug=SimpleNamespace(enabled=False),
                                    step=lambda *a, **kw: True, last_newton_iters=8)
    sim._pid_err = [.01, .02, .03]
    if version is not None:
        sim._pid_error_version = version
    assert sim._fully_implicit_step(1.)
    assert sim._pid_error_version == 2
    assert sim._pid_err == ([.1, .1, 0.] if version is None else [.02, .03, 0.])
