"""Explicit actuator commands must override autonomous outlet controllers."""
from copy import deepcopy
from dataclasses import replace

import pytest
from test_fim_dynamic_receiver import build

from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM
from simulator.pr_surface_boundary import PRSurfaceBoundary
from surface.phase_geometry import EquilibriumLevelControl
from surface.stratified_pr_flow import StratifiedPRFlow
from surface.stratified_pr_network import StratifiedPRNetwork


def controlled_receiver(kind):
    base = build()
    port = base.receiver.outlets[0]
    if kind == 'pressure':
        port = replace(port, pressure_setpoint_pa=1e6,
                       opening_gain_per_pa=1e-5, opening_bias=.9)
    else:
        port = replace(port, level_control=EquilibriumLevelControl(
            base.boundary.fluid.names[0], 0., 1., opening_bias=.9,
            minimum_mole_fraction=0.))
    flow = StratifiedPRFlow(base.receiver.column, base.receiver.wall, outlets=[port])
    state = replace(base.receiver_state, model_id=flow.model_id)
    return DynamicReceiverCoupledFIM(
        base.solver, flow, state, base.assay, base.thermal_conditions, base.inlet_height_m)


@pytest.mark.parametrize('kind', ['pressure', 'level'])
@pytest.mark.parametrize('network', [False, True])
def test_manual_override_is_local_and_auto_control_returns(kind, network):
    model = controlled_receiver(kind)
    original = model.receiver.outlets[0]
    receiver = StratifiedPRNetwork({'tank': model.receiver}, []) if network else model.receiver
    boundary = PRSurfaceBoundary(receiver, 'tank' if network else None)
    name = 'tank/export' if network else 'export'
    for opening in (0., .25, 1.):
        trial = boundary.build([], 0., {name: opening}, {})
        port = (trial.vessels['tank'] if network else trial).outlets[0]
        assert port.opening == opening
        assert port.pressure_setpoint_pa is None
        assert port.opening_gain_per_pa == port.opening_bias == 0.
        assert port.level_control is None
    auto = boundary.build([], 0., {}, {})
    assert (auto.vessels['tank'] if network else auto).outlets[0] == original
    assert model.receiver.outlets[0] == original


@pytest.mark.parametrize('kind', ['pressure', 'level'])
def test_actual_closed_controlled_receiver_exports_nothing_and_resumes(kind):
    from check_dynamic_receiver import audit_step

    model = controlled_receiver(kind)
    original = model.receiver.outlets[0]
    before = model.receiver_state
    model.outlet_openings = {'export': 0.}
    assert model.step(.02), model.last_attempt
    assert model.report['cumulative']['export_component_moles'] == [0., 0.]
    assert model.report['cumulative']['export_energy_j'] == 0.
    assert audit_step({**model.to_dict(), 'initial_state': before.to_dict()}, model.report)['accepted']
    wrong_command = deepcopy(model.report)
    wrong_command['outlet_openings'] = {'export': 1.}
    with pytest.raises(ValueError):
        audit_step({**model.to_dict(), 'initial_state': before.to_dict()}, wrong_command)
    model.outlet_openings = {}
    assert model.step(.02, current_time=.02), model.last_attempt
    assert sum(model.report['stages'][-1]['flow']['outlet_component_moles']) > 0.
    assert model.receiver.outlets[0] == original
    assert audit_step({**model.to_dict(), 'initial_state': before.to_dict()}, model.report)['accepted']


def test_actual_graph_manual_closure_passes_independent_audit():
    from check_dynamic_receiver import audit_step

    base = controlled_receiver('pressure')
    network = StratifiedPRNetwork({'tank': base.receiver}, [])
    state = network.initialize({'tank': base.receiver_state})
    model = DynamicReceiverCoupledFIM(
        base.solver, network, state, base.assay, base.thermal_conditions,
        base.inlet_height_m, inlet_vessel='tank')
    model.outlet_openings = {'tank/export': 0.}
    assert model.step(.02), model.last_attempt
    assert model.report['cumulative']['export_component_moles'] == [0., 0.]
    assert audit_step({**model.to_dict(), 'initial_state': state.to_dict()}, model.report)['accepted']
