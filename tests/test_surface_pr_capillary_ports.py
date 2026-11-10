"""Per-end material and surface work must match the actual accepted fluxes."""

import math

import numpy as np
import pytest
from check_pr_capillary_axial import audit
from test_surface_pr_capillary_contact import setup

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_pressure import PRCapillaryPressure


@pytest.mark.parametrize("velocity", [-0.3, 0.3])
def test_two_moving_open_ports_have_independently_reconstructed_material_and_work(velocity):
    base, seed = setup(velocity)
    raw = base.pipe.to_dict()
    raw["boundary_mode"] = "walls"
    pipe = PRActiveMultifluidPipe.from_dict(raw)
    old = pipe.initialize([dict(c) for c in seed.cells])
    params = {k: v for k, v in base.source.to_dict().items() if k not in ("model", "pipe_model_id")}
    params["surface_tension_n_m"] = 0.02
    model = PRCapillaryAxialPipe(
        PRCapillaryPressure(pipe, **params),
        boundaries=[dict(old.cells[1]), dict(old.cells[2])],
        boundary_geometry="reservoir_cell",
    )
    final, report = model.step(old, 1e-7)
    expected = audit(model, old, final, report)["port_transfers"]
    actual = model.port_transfers(old, final)
    assert all(abs(p["capillary_work_j"]) > 1e-20 for p in actual)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a["transfer_n_m_e"], b["transfer_n_m_e"], rtol=1e-9, atol=1e-24)
        assert a["capillary_work_j"] == pytest.approx(b["capillary_work_j"], rel=1e-10, abs=1e-25)
    assert math.fsum(p["capillary_work_j"] for p in actual) == pytest.approx(
        report["external_capillary_work_j"], rel=1e-10, abs=1e-25
    )
    np.testing.assert_allclose(
        np.sum([p["transfer_n_m_e"] for p in actual], axis=0),
        report["external_transfer_n_m_e"],
        rtol=1e-9,
        atol=1e-23,
    )
