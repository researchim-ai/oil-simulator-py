"""Report physical advances, not discarded nonlinear trials or log lines."""
from pathlib import Path

import pytest
import torch

from simulator.eclipse_run import iter_deck
from simulator.eclipse_run import run_deck


@pytest.mark.parametrize('quiet', [False, True])
@pytest.mark.parametrize('temporal', [False, True])
def test_step_count_is_independent_of_logging_and_temporal_probes(tmp_path, quiet, temporal):
    source = Path(__file__).parent / 'data/spe1/SPE1CASE1.DATA'
    deck = tmp_path / 'short.DATA'
    deck.write_text(source.read_text().split('TSTEP')[0] + '\nTSTEP\n .01 .01 /\nEND\n')
    overrides = dict(initial_time_step_days=.01, max_time_step_days=.01,
                     min_time_step_days=1e-8)
    if temporal:
        overrides['reservoir_time_control'] = dict(
            pressure_tolerance_pa=1e8, saturation_tolerance=1.,
            component_relative_tolerance=.9, withdrawal_relative_tolerance=.9,
            withdrawal_absolute_tolerance_kg=1.)
    context = {}
    iterator = iter_deck(deck, device=torch.device('cpu'), quiet=quiet,
                         sim_overrides=overrides, context=context, yield_start=True)
    next(iterator)
    solver, sim = context['solver'], context['sim']
    original = solver.step
    successful_solves = []
    iterations = []

    def step(dt, current_time=None):
        result = original(dt, current_time=current_time)
        if result:
            successful_solves.append(dt)
            iterations.append(solver.last_newton_iters)
        return result

    solver.step = step
    accepted = 0
    while True:
        try:
            next(iterator)
            accepted += sim.last_fim_interval_stats['accepted_steps']
        except StopIteration as stop:
            summary, info = stop.value
            break
    assert summary['TIME'][-1] == pytest.approx(.02)
    assert accepted > 0
    assert info['substeps'] == accepted
    assert len(successful_solves) == accepted * (3 if temporal else 1)
    expected = sum(n for i, n in enumerate(iterations) if not temporal or i % 3 != 0)
    assert info['accepted_newton_iters'] == expected
    assert expected > 0


def test_accepted_iteration_count_survives_checkpoint(tmp_path):
    source = Path(__file__).parent / 'data/spe1/SPE1CASE1.DATA'
    deck = tmp_path / 'restart.DATA'
    deck.write_text(source.read_text().split('TSTEP')[0] + '\nTSTEP\n .01 .01 /\nEND\n')
    checkpoint = tmp_path / 'state.pt'
    options = dict(device=torch.device('cpu'), sim_overrides=dict(
        initial_time_step_days=.01, max_time_step_days=.01, min_time_step_days=1e-8))
    _, uninterrupted = run_deck(deck, **options)
    _, prefix = run_deck(deck, max_days=.01, checkpoint=str(checkpoint), **options)
    _, resumed = run_deck(deck, resume=str(checkpoint), **options)
    assert 0 < prefix['accepted_newton_iters'] < resumed['accepted_newton_iters']
    assert resumed['accepted_newton_iters'] == uninterrupted['accepted_newton_iters']
    assert resumed['substeps'] == uninterrupted['substeps']
