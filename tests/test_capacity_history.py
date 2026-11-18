"""Historical derivatives propose points but never certify their feasibility."""
import math
from dataclasses import replace

import pytest

from simulator.facility_coupling import CoupledFacilityControl
from surface.capacity_bracket import CapacityBracket, capacity_history_slopes


def row(x, value, **extra):
    return {'fraction': x, 'constraint_margins': {'gas': value}, **extra}


def test_local_secant_ignores_certificates_tiny_brackets_and_remote_points():
    rows = [row(.004, 100), row(.003, 0), row(.003+1e-10, .5),
            row(.0031, .02), row(.00305, 100, certified_from_trace_index=0)]
    assert capacity_history_slopes(rows, .003, 1e-9) == pytest.approx({'gas': 200.})


@pytest.mark.parametrize('value', [None, math.nan, math.inf, -.1, 0])
def test_missing_flat_negative_or_nonfinite_derivatives_fall_back(value):
    assert capacity_history_slopes([row(.003, 0), row(.0031, value)], .003, 1e-9) == {}


def test_changed_constraint_does_not_use_an_unrelated_historical_slope():
    b = CapacityBracket(.001, .004, 1e-9, -.5, 1.,
                        low_margins={'other': -.5}, high_margins={'other': 1.})
    assert b.tangent_proposal({'gas': 200.}, {'gas': 200.}) is None


def test_bad_historical_prediction_retains_solved_bracket_and_fallback_converges():
    root = .003
    b = CapacityBracket(.001, .004, 1e-9, -.002, .001,
                        low_margins={'gas': -.002}, high_margins={'gas': .001})
    proposed = b.tangent_proposal({'gas': 100.}, {'gas': 100.})
    assert b.low < proposed < b.high
    b.update(proposed, proposed <= root, proposed-root, margins={'gas': proposed-root})
    count = 0
    while not b.complete:
        x=b.proposal()
        b.update(x,x<=root,x-root,margins={'gas':x-root})
        assert b.low <= root < b.high
        count += 1
        assert count < 40


def test_history_requires_boolean():
    with pytest.raises(ValueError, match='reuse_capacity_history'):
        replace(CoupledFacilityControl(), reuse_capacity_history=1)


def test_real_history_resets_at_events_and_changed_caps_and_survives_rollback(monkeypatch):
    import copy

    from test_fim_facility_coupling import build

    import simulator.facility_coupling as module

    c=build()
    c.control=replace(c.control,search_method='facet_safeguarded',
                      reuse_capacity_history=True,fraction_tolerance=1e-7)
    calls=[]
    original=module.capacity_history_slopes
    def tracked(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module,'capacity_history_slopes',tracked)
    assert c.step(3600.)
    assert not calls and c.report['capacity_history_anchor'] is None
    assert c.step(3600.,current_time=3600.)
    assert len(calls)==1 and c.report['capacity_history_anchor'] is not None
    c.controls={'gas_valve_max_opening':.8}
    assert c.step(3600.,current_time=7200.)
    assert len(calls)==1 and c.report['capacity_history_anchor'] is None
    for w in c.solver.wells.wells:w.control_value *= .9
    assert c.step(3600.,current_time=10800.)
    assert len(calls)==1 and c.report['capacity_history_anchor'] is None
    saved=copy.deepcopy(c.report)
    def fail(*args):
        raise ValueError('ledger failure')
    monkeypatch.setattr(c,'_material_account',fail)
    with pytest.raises(ValueError,match='ledger failure'):
        c.step(3600.,current_time=14400.)
    assert c.report==saved
