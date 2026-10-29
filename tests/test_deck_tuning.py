"""Deck timestep limits must be enforced, including schedule updates and overrides."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator.eclipse_deck import EclipseCase
from simulator.eclipse_run import iter_deck


def _deck(tmp_path):
    source = Path(__file__).parent / 'data/spe1/SPE1CASE1.DATA'
    text = source.read_text().split('TSTEP')[0] + '''
TUNING
 .4 .5 .01 1* 2 1* .25 2* .15 /
 .05 /
 /
TSTEP
 1.1 /
TUNING
 .2 .25 /
 /
 /
TSTEP
 .6 /
END
'''
    path = tmp_path / 'tuning.DATA'
    path.write_text(text)
    return path


def test_zero_tmaxwc_disables_event_restart(tmp_path):
    path = _deck(tmp_path)
    path.write_text(path.read_text().replace('.25 2* .15', '.25 2* 0'))
    assert EclipseCase(path).tuning_steps[0]['time_step_after_well_event_days'] == 0


@pytest.mark.parametrize('override', [False, True])
def test_scheduled_tuning_controls_actual_steps(tmp_path, override):
    path = _deck(tmp_path)
    case = EclipseCase(path)
    assert [v['max_time_step_days'] for v in case.tuning_steps] == [.5, .25]
    assert case.tuning_restart_steps == [.4, .2]
    overrides = {'max_time_step_days': .1, 'initial_time_step_days': .1} if override else None
    context = {}
    gen = iter_deck(path, device=torch.device('cpu'), sim_overrides=overrides,
                    context=context, yield_start=True)
    next(gen)
    sim = context['sim']
    original = sim.fim_solver.step
    calls = []

    def step(dt, current_time=None):
        calls.append((current_time / 86400., dt / 86400.))
        return original(dt, current_time=current_time)

    sim.fim_solver.step = step
    reports = list(gen)
    assert reports[-1][1] == pytest.approx(1.7)
    for start, dt in calls:
        limit = .1 if override else (.5 if start < 1.1 - 1e-9 else .25)
        assert dt <= limit + 1e-12
    assert calls[0][1] == pytest.approx(.1 if override else .15)
    assert next(dt for t, dt in calls if abs(t - 1.1) < 1e-9) == pytest.approx(.1 if override else .2)
    assert sim.fim_solver.tol_cnv == 1e-3 and sim.fim_solver.tol_mb == 1e-7
    assert sim.sim_params['dt_restart_factor'] == .25
    assert sim.sim_params['dt_pid_tol'] == .05
