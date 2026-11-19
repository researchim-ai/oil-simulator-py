import copy
import pytest
import torch

from simulator.blackoil_fim import BlackOilFIMSolver
from simulator.reservoir_time_control import ReservoirTimeControl
from test_blackoil_fim import BASE, _objects


def build(device='cpu'):
    cfg = copy.deepcopy(BASE)
    r, f, w = _objects(cfg, torch.device(device))
    return BlackOilFIMSolver(r, f, w, cfg['simulation'])


def control(solver, **options):
    cfg = dict(pressure_tolerance_pa=1e8, saturation_tolerance=1.,
               component_relative_tolerance=.9, withdrawal_relative_tolerance=.9,
               withdrawal_absolute_tolerance_kg=1.)
    cfg.update(options)
    return ReservoirTimeControl(solver, cfg)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_accepted_state_and_cumulative_flows_are_only_two_half_steps(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    s, reference = build(device), build(device)
    driver = control(s)
    saved = (s.tol_cnv, s.tol_mb, s.min_newton, s.lin_tol_max, s.lin_tol_min, s.lin_accept)
    reference.tol_cnv, reference.tol_mb = min(reference.tol_cnv, 1e-5), min(reference.tol_mb, 1e-8)
    reference.min_newton = max(reference.min_newton, 1)
    reference.lin_tol_max, reference.lin_tol_min = min(reference.lin_tol_max, 1e-8), min(reference.lin_tol_min, 1e-9)
    reference.lin_accept = min(reference.lin_accept, 1e-7)
    reference.relaxed_max_pv_fraction = 0.
    assert driver.step(3600., current_time=0.)
    assert saved == (s.tol_cnv, s.tol_mb, s.min_newton, s.lin_tol_max, s.lin_tol_min, s.lin_accept)
    assert reference.step(1800., current_time=0.)
    assert reference.step(1800., current_time=1800.)
    for name in ['pressure', 's_w', 's_o', 's_g', 'rs', 'rv']:
        torch.testing.assert_close(getattr(s.fluid, name), getattr(reference.fluid, name), rtol=0, atol=0)
    assert driver.report['reservoir_step_calls'] == 3
    assert driver.report['indicator'] <= 1
    for actual, expected in zip(s.wells.wells, reference.wells.wells):
        assert actual.cum_surface == expected.cum_surface


@pytest.mark.parametrize('failure', ['error', 'second_half', 'exception'])
def test_rejection_restores_fields_and_well_totals(monkeypatch, failure):
    s = build()
    assert s.step(60., current_time=0.)
    driver = control(s, pressure_tolerance_pa=1e-12) if failure == 'error' else control(s)
    settings = (s.tol_cnv, s.tol_mb, s.min_newton, s.lin_tol_min, s.relaxed_max_pv_fraction)
    before = {name: getattr(s.fluid, name).clone() for name in ['pressure','s_w','s_g','rs']}
    totals = [dict(w.cum_surface) for w in s.wells.wells]
    original = s.step
    calls = []
    def step(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3 and failure != 'error':
            if failure == 'exception':
                raise RuntimeError('stage failed')
            return False
        return original(*args, **kwargs)
    monkeypatch.setattr(s, 'step', step)
    if failure == 'exception':
        with pytest.raises(RuntimeError, match='stage failed'):
            driver.step(3600., current_time=60.)
    else:
        assert not driver.step(3600., current_time=60.)
    for name, value in before.items():
        torch.testing.assert_close(getattr(s.fluid, name), value, rtol=0, atol=0)
    assert [w.cum_surface for w in s.wells.wells] == totals
    assert driver.report is None
    assert settings == (s.tol_cnv, s.tol_mb, s.min_newton, s.lin_tol_min, s.relaxed_max_pv_fraction)


def test_missing_or_invalid_tolerances_are_rejected():
    s = build()
    with pytest.raises(ValueError, match='Explicit'):
        ReservoirTimeControl(s, {})
    with pytest.raises(ValueError, match='positive'):
        control(s, component_relative_tolerance=float('nan'))


def test_simulator_reaches_report_boundary_and_checkpoint_repeats():
    import pickle
    from simulator.simulation import Simulator
    cfg = copy.deepcopy(BASE)
    cfg['simulation']['reservoir_time_control'] = dict(
        pressure_tolerance_pa=1e8, saturation_tolerance=1.,
        component_relative_tolerance=.9, withdrawal_relative_tolerance=.9,
        withdrawal_absolute_tolerance_kg=1.)
    r, f, w = _objects(cfg, torch.device('cpu'))
    sim = Simulator(r, f, w, cfg['simulation'], device=torch.device('cpu'))
    assert sim.run_step(3600.)
    assert sim.current_time == 3600.
    assert sim.reservoir_time_control.report['indicator'] <= 1
    resumed = pickle.loads(pickle.dumps(sim))
    assert sim.run_step(1200.) and resumed.run_step(1200.)
    assert sim.current_time == resumed.current_time == 4800.
    torch.testing.assert_close(sim.fluid.pressure, resumed.fluid.pressure, rtol=0, atol=0)
