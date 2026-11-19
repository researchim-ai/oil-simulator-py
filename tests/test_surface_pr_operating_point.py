from dataclasses import replace

import pytest
from test_surface_stratified_pr_flow import setup

from surface.component_energy import ComponentEnergyStream
from surface.compositional_separator import EquilibriumOutlet
from surface.liquid import PropertyModelError
from surface.pr_operating_point import initialize_operating_point
from surface.stratified_pr_flow import StratifiedFeed, StratifiedPRFlow
from surface.vessel_wall import VesselWall


def case():
    c, column = setup()
    feed = ComponentEnergyStream.from_pt(
        c.fluid, [3.5, 7.4], 1040000., 330., elevation_m=c.elevation_m+1.)
    model = StratifiedPRFlow(c, VesselWall(1000., 2., 1., 310.),
        feeds=[StratifiedFeed(1., feed)],
        outlets=[EquilibriumOutlet('export', 990000., 3e-6, height_m=1., diameter_m=.05)])
    return model, model.initialize(column, column.temperature_k)


def test_operating_point_has_no_transient_inventory_or_energy_drift():
    model, initial = case()
    snapshot = initial.to_dict()
    state, report = initialize_operating_point(model, initial)
    assert initial.to_dict() == snapshot
    assert state.time_s == 0.
    assert state.column.component_moles != initial.column.component_moles
    assert max(map(abs, report['component_balance_mol_s'])) < 1e-8
    assert report['reservoir_feedback_solved'] is False
    assert report['verification_drift']['pressure_pa'] < .01
    # A longer independent interval must retain the stationary solution.
    end, ledger = model._step(state, 1.)
    assert end.column.component_moles == pytest.approx(state.column.component_moles, abs=1e-8)
    assert end.column.temperature_k == pytest.approx(state.column.temperature_k, abs=1e-6)
    assert abs(ledger['energy_residual_j']) < 1e-5


def test_operating_point_rejects_live_state_and_closed_system():
    model, initial = case()
    with pytest.raises(ValueError, match='time zero'):
        initialize_operating_point(model, replace(initial, time_s=1.))
    closed = StratifiedPRFlow(model.column, model.wall)
    with pytest.raises(PropertyModelError, match='throughflow'):
        initialize_operating_point(closed, replace(initial, model_id=closed.model_id))
