"""Physical audits must survive scheduling changes, not EOS changes."""
from dataclasses import replace

import pytest

from surface.liquid import ConservationError
from test_surface_pr_equilibrium_transport import pipe


@pytest.mark.parametrize('replacement', [False, True])
def test_transport_rechecks_stocks_after_heat_capacity_change(replacement):
    model, state = pipe(320.)
    model.inspect(state)
    if replacement:
        model.fluid.cp = model.fluid.cp * 1.1
    else:
        model.fluid.cp.mul_(1.1)
    with pytest.raises(ConservationError, match='checkpoint'):
        model.inspect(state)


def test_clock_only_change_reuses_physics_but_checks_step_bounds(monkeypatch):
    model, state = pipe(320.)
    model.inspect(state)
    def unexpected(*args, **kwargs):
        raise AssertionError('unchanged physical state was decoded again')
    monkeypatch.setattr(model, '_state_data', unexpected)
    model.inspect(replace(state, time_s=state.time_s + 1., next_step_s=model.minimum_step_s))
    with pytest.raises(ValueError, match='timestep'):
        model.inspect(replace(state, next_step_s=model.maximum_step_s * 2))


def test_physical_stock_change_is_not_a_scheduling_cache_hit():
    model, state = pipe(320.)
    nodes = list(state.nodes)
    name, body = nodes[0]
    nodes[0] = (name, replace(body, total_energy_j=body.total_energy_j + 100.))
    with pytest.raises(ConservationError, match='checkpoint'):
        model.inspect(replace(state, nodes=tuple(nodes)))


def test_parameter_sensitivity_bypasses_detached_validation(monkeypatch):
    model, state = pipe(320.)
    model.fluid.cp.requires_grad_(True)
    calls = []
    original = model._state_data
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(model, '_state_data', counted)
    model.inspect(state)
    model.inspect(state)
    assert len(calls) == 2 * len(state.nodes)
    assert not model._verified


@pytest.mark.parametrize('replacement', [False, True])
def test_gravity_column_rechecks_after_eos_parameter_change(replacement):
    from test_surface_stratified_pr import binary
    from surface.separator_geometry import SeparatorGeometry
    from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
    model = StratifiedPRColumn(binary('cpu'), SeparatorGeometry(.1, 1., 'vertical'))
    state = StratifiedPRState.from_dict(model.initialize(1e6, 320., [.5, .5])['state'])
    model.inspect(state)
    if replacement:
        model.fluid.cp = model.fluid.cp * 1.1
    else:
        model.fluid.cp.mul_(1.1)
    with pytest.raises(ConservationError, match='equilibrium/inventory'):
        model.inspect(state)
