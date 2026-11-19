from copy import deepcopy

import pytest
import torch
from test_fim_receiver_operating_point import case

from simulator.dynamic_receiver_coupling import DynamicReceiverControl, DynamicReceiverCoupledFIM


def build():
    original=case()
    # Consistent initial stocks avoid confusing artificial startup with subcycling.
    state,_=original.prepare_operating_point()
    return DynamicReceiverCoupledFIM(original.solver,original.receiver,state,
        original.assay,original.thermal_conditions,original.inlet_height_m,
        {'surface_subcycling':True})


def test_local_surface_steps_are_conservative_and_independently_audited(monkeypatch):
    from check_dynamic_receiver import audit_step

    m=build(); old=m.receiver_state
    calls=[]; solve=m.solver.step
    def counted(dt,*args,**kwargs):
        calls.append(dt)
        return solve(dt,*args,**kwargs)
    monkeypatch.setattr(m.solver,'step',counted)
    assert m.step(.1),m.last_attempt
    assert set(calls) <= {.1,.05}
    assert all(len(s['flow']['surface_steps'])>=2 for s in m.report['stages'])
    assert audit_step({**m.to_dict(),'initial_state':old.to_dict()},m.report)['accepted']
    tampered=deepcopy(m.report)
    tampered['stages'][0]['flow']['surface_steps'][0]['outlet_component_moles'][0] += .1
    with pytest.raises(ValueError):
        audit_step({**m.to_dict(),'initial_state':old.to_dict()},tampered)
    state,report=m.receiver_state,m.report; pressure=m.solver.fluid.pressure.clone()
    monkeypatch.setattr(m,'_temporal_error',lambda *args:2.)
    assert not m.step(.1,current_time=.1)
    assert m.receiver_state==state and m.report is report
    assert torch.equal(m.solver.fluid.pressure,pressure)


def test_subcycling_is_opt_in_and_serializable():
    m=case()
    assert 'surface_subcycling' not in m.to_dict()['control']
    with pytest.raises(ValueError,match='boolean'):
        DynamicReceiverControl(surface_subcycling=1)
    enabled=build()
    raw={**enabled.to_dict(),'initial_state':enabled.receiver_state.to_dict()}
    restored=DynamicReceiverCoupledFIM.from_dict(enabled.solver,raw)
    assert restored.control.surface_subcycling
    assert restored.model_id==enabled.model_id


def test_partial_trajectory_pressure_error_exposes_required_supply():
    from surface.stratified_pr_flow import ReceiverFeedPressureError

    error=ReceiverFeedPressureError(1e6,1e6+1.)
    supply=error.minimum_supply_pressure_pa
    assert supply < error.receiver_pressure_pa
    assert supply*(1.+error.pressure_relative_tolerance)==pytest.approx(error.receiver_pressure_pa)
