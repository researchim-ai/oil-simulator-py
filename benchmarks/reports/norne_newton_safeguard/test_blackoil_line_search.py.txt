import copy
import pytest
import torch

from simulator.blackoil_fim import BlackOilFIMSolver
from test_blackoil_fim import BASE, _objects


def build(device):
    config = copy.deepcopy(BASE)
    config['simulation'].update(newton_line_search=True, newton_relaxed_max_pv_fraction=0.)
    reservoir, fluid, wells = _objects(config, torch.device(device))
    return BlackOilFIMSolver(reservoir, fluid, wells, config['simulation'])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_rejected_trial_is_backtracked_and_only_one_commit(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    solver = build(device)
    original = solver._trial_merit
    calls = []
    def merit(*args):
        calls.append(1)
        value = original(*args)
        # First call is the base merit; force rejection of the full trial.
        return float('inf') if len(calls) == 2 else value
    monkeypatch.setattr(solver, '_trial_merit', merit)
    commits = []
    store = solver._store_well_results
    def commit(*args):
        commits.append(1)
        return store(*args)
    monkeypatch.setattr(solver, '_store_well_results', commit)
    assert solver.step(3600., current_time=0.)
    assert len(calls) > 2
    assert len(commits) == 1
    assert torch.isfinite(solver.fluid.pressure).all()
    torch.testing.assert_close(solver.fluid.s_w + solver.fluid.s_o + solver.fluid.s_g,
                               torch.ones_like(solver.fluid.s_w), rtol=0, atol=1e-12)


@pytest.mark.parametrize('raises', [False, True])
def test_exhausted_or_exceptional_search_restores_state(monkeypatch, raises):
    solver = build('cpu')
    before = {name: getattr(solver.fluid, name).clone()
              for name in ['pressure', 's_w', 's_g', 's_o', 'rs', 'rv']}
    assert solver.wells.p_bh is None

    calls = []
    def merit(*args):
        calls.append(1)
        if len(calls) > 1:
            # Assembly may modify well caches. Rejected trials cannot retain it.
            solver.wells.p_bh.add_(12345.)
            if raises:
                raise RuntimeError('trial assembly failure')
        return 1. if len(calls) == 1 else 2.
    monkeypatch.setattr(solver, '_trial_merit', merit)
    if raises:
        with pytest.raises(RuntimeError, match='trial assembly failure'):
            solver.step(3600., current_time=0.)
    else:
        assert not solver.step(3600., current_time=0.)
        assert len(calls) == 10  # base plus nine bounded trial evaluations
    for name, value in before.items():
        torch.testing.assert_close(getattr(solver.fluid, name), value, rtol=0, atol=0)
    assert solver.wells.p_bh is None


def test_rescue_restarts_from_rolled_back_state(monkeypatch):
    reference = build('cpu')
    assert reference.step(3600., current_time=0.)
    solver = build('cpu')
    solver.newton_line_search = False
    solver.newton_line_search_fallback = True
    original = solver.solve_linear
    calls = []
    def fail_first(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        return None if len(calls) == 1 else result
    monkeypatch.setattr(solver, 'solve_linear', fail_first)
    assert solver.step(3600., current_time=0.)
    assert solver.newton_line_search_retries == 1
    assert solver.newton_line_search is False
    for name in ['pressure', 's_w', 's_g', 'rs', 'rv']:
        torch.testing.assert_close(getattr(solver.fluid, name), getattr(reference.fluid, name), rtol=0, atol=0)


def test_rescue_is_bounded_and_does_not_retry_success(monkeypatch):
    solver = build('cpu')
    solver.newton_line_search = False
    solver.newton_line_search_fallback = True
    assert solver.step(60., current_time=0.)
    assert getattr(solver, 'newton_line_search_retries', 0) == 0
    initial = solver.fluid.pressure.clone()
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        return None
    monkeypatch.setattr(solver, 'solve_linear', fail)
    assert not solver.step(3600., current_time=60.)
    assert len(calls) == 2
    assert solver.newton_line_search_retries == 1
    torch.testing.assert_close(solver.fluid.pressure, initial, rtol=0, atol=0)


def test_search_rebuilds_failed_reused_amg_without_double_schur(monkeypatch):
    reference, solver = build('cpu'), build('cpu')
    assert reference.step(3600., current_time=0.)
    original = solver.solve_linear
    injected = []
    rebuilds = []
    def solve(*args, **kwargs):
        result = original(*args, **kwargs)
        rebuilds.append(kwargs['rebuild_amg'])
        if not kwargs['rebuild_amg'] and not injected:
            injected.append(True)
            return None  # J/R already changed by the real linear solve
        return result
    monkeypatch.setattr(solver, 'solve_linear', solve)
    assert solver.step(3600., current_time=0.)
    assert injected
    failed_index = rebuilds.index(False)
    assert rebuilds[failed_index + 1]
    for name in ['pressure', 's_w', 's_g']:
        torch.testing.assert_close(getattr(solver.fluid, name), getattr(reference.fluid, name),
                                   rtol=1e-7, atol=1e-7)
