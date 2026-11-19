"""Coupled isothermal separation must retain one stock-component PVT model."""
import copy

import pytest
from test_fim_facility_coupling import facility_spec, solver

from simulator.facility_coupling import FacilityCoupledFIM
from surface import PropertyModelError, SurfaceFacility


@pytest.mark.parametrize("column", ["Bo", "Bw", "Bg", "Rs", "mu_o"])
def test_downstream_stage_pvt_mismatch_rejected(column):
    facility = SurfaceFacility.from_dict(facility_spec())
    stage = facility.stages[1]
    stage.pvt = copy.deepcopy(stage.pvt)
    values = getattr(stage.pvt, column).copy()
    values[-1] *= 1.01
    setattr(stage.pvt, column, values)
    with pytest.raises(PropertyModelError, match=f"stock_tank: {column}"):
        FacilityCoupledFIM(solver(), facility)


def test_independent_identical_stage_pvt_is_accepted():
    facility = SurfaceFacility.from_dict(facility_spec())
    facility.stages[1].pvt = copy.deepcopy(facility.stages[1].pvt)
    coupled = FacilityCoupledFIM(solver(), facility)
    assert coupled.facility is facility
