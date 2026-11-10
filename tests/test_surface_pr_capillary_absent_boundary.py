"""A missing exterior phase owns neither EOS state nor a ghost inventory."""

import json

import pytest
from check_pr_capillary_axial import audit
from test_surface_pr_capillary_contact import setup

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_pressure import PRCapillaryPressure


@pytest.mark.parametrize("phase", ["liquid", "gas"])
@pytest.mark.parametrize("side", [0, 1])
def test_pure_boundary_transfers_only_from_the_real_phase_and_pairs_capillary_work(phase, side):
    base, old = setup()
    raw = base.pipe.to_dict()
    raw["boundary_mode"] = "walls"
    pipe = PRActiveMultifluidPipe.from_dict(raw)
    old = pipe.initialize([dict(c) for c in old.cells])
    params = {k: v for k, v in base.source.to_dict().items() if k not in ("model", "pipe_model_id")}
    params["surface_tension_n_m"] = 0.02
    source = PRCapillaryPressure(pipe, **params)
    rows = [None, None]
    rows[side] = {phase: dict(old.cells[0 if side == 0 else -1])[phase]}
    model = PRCapillaryAxialPipe(source, boundaries=rows, boundary_geometry="reservoir_cell")
    final, report = model.step(old, 1e-7)
    assert set(model.boundaries[side]) == {phase}
    assert audit(model, old, final, report)["accepted"]
    restored = PRCapillaryAxialPipe.from_dict(json.loads(json.dumps(model.to_dict())))
    same, _ = restored.step(old, 1e-7)
    assert same == final
    with pytest.raises(ValueError, match="acoustic volume"):
        PRCapillaryAxialPipe(
            source, boundaries=rows, boundary_geometry="reservoir_cell", volume_diffusion="none"
        )
