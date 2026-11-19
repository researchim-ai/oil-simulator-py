from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_fim_dynamic_receiver import build

from simulator.dynamic_receiver_coupling import DynamicReceiverCoupledFIM
from simulator.receiver_operating_point import fixed_reservoir_wells
from surface.stratified_pr_flow import StratifiedPRFlow


def case():
    base = build()
    flow = StratifiedPRFlow(base.receiver.column, base.receiver.wall,
        outlets=[replace(base.receiver.outlets[0], volume_conductance_m3_s_pa=3e-6)])
    return DynamicReceiverCoupledFIM(base.solver,flow,
        replace(base.receiver_state,model_id=flow.model_id),base.assay,
        base.thermal_conditions,base.inlet_height_m)


def test_fixed_reservoir_wells_are_transactional_and_pressure_sensitive():
    m=case(); s=m.solver
    fields={k:getattr(s.fluid,k).clone() for k in ('pressure','s_w','s_g','rs')}
    old=deepcopy([w.__dict__ for w in s.wells.wells]); network=s.wells.surface_network
    a,work=fixed_reservoir_wells(s,1e6)
    b,_=fixed_reservoir_wells(s,1.02e6)
    assert work['well_error_ratio'] <= 1.
    assert sum(sum(w['component_mass_rates_kg_s']) for w in b.values()) < sum(sum(w['component_mass_rates_kg_s']) for w in a.values())
    for name,value in fields.items(): assert torch.equal(getattr(s.fluid,name),value)
    assert [w.__dict__ for w in s.wells.wells] == old
    assert s.wells.surface_network is network and m.report is None


def test_stationary_initialization_satisfies_wells_and_preserves_reservoir():
    m=case(); s=m.solver; initial=m.receiver_state; pressure=s.fluid.pressure.clone()
    state,report=m.prepare_operating_point()
    assert m.receiver_state == initial and m.report is None
    assert torch.equal(s.fluid.pressure,pressure)
    assert state.time_s == 0.
    last=report['pressure_trials'][-1]
    assert last['physical'] and abs(last['defect_pa']) < .012
    assert report['reservoir_advanced'] is False
    assert report['vessel_verification']['verification_drift']['pressure_pa'] < .01
    # Explicit adoption, then the existing full coupled accuracy check.
    m.receiver_state=state
    assert m.step(.1),m.last_attempt
    assert m.report['temporal_error_ratio'] <= 1.
    with pytest.raises(ValueError,match='time-zero'):
        m.prepare_operating_point()


def test_failed_well_initialization_restores_original_state(monkeypatch):
    m=case(); s=m.solver; network=s.wells.surface_network
    pressure=s.fluid.pressure.clone(); bhp=s.wells.p_bh
    report=s.wells.surface_report
    def fail(*args,**kwargs):
        raise RuntimeError('injected factor solve failure')
    monkeypatch.setattr(s.wells,'_solve_D',fail)
    with pytest.raises(RuntimeError,match='injected'):
        fixed_reservoir_wells(s,1e6)
    assert torch.equal(s.fluid.pressure,pressure)
    assert s.wells.p_bh is bhp
    assert s.wells.surface_report is report
    assert s.wells.surface_network is network
