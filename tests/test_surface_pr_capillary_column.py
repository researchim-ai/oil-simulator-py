"""A finite gravitational separator supplies actual capillary pipe stocks."""

import json

import numpy as np
import pytest
import torch
from check_pr_capillary_axial import audit as axial_audit
from test_surface_compositional_separator import binary
from test_surface_pr_active_multifluid_pipe import phase

from surface.equilibrium_properties import EquilibriumFluid
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from surface.pr_capillary_column import PRCapillaryColumnNetwork, PRCapillaryColumnState
from surface.separator_geometry import SeparatorGeometry
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState


def setup(*, reverse=False, device="cpu", small=False):
    f = EquilibriumFluid.from_dict(binary().to_dict(), device=device)
    column = StratifiedPRColumn(
        f,
        SeparatorGeometry(0.001 if small else 0.1, 0.01 if small else 0.1, "vertical"),
        quadrature_order=2,
        maximum_quadrature_order=8,
    )
    cs = StratifiedPRState.from_dict(column.initialize(1e6, 196.2, [0.5, 0.5])["state"])
    height = cs.interface_heights_m[0]
    pipe = PRActiveMultifluidPipe(
        f,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [0.001 if small else 0.01] * 3,
        0.0004 if small else 0.004,
        boundary_mode="walls",
        face_elevations_m=[height] * 4,
        gravity_m_s2=0.0,
    )
    parameters = {
        "liquid_slot": "liquid",
        "gas_slot": "gas",
        "width_m": 0.0004 if small else 0.004,
        "surface_tension_n_m": 0.02,
        "volume_mobility_m3_pa_s": 1e-12,
    }
    model = PRCapillaryColumnNetwork(pipe, column, parameters, port_side=0, port_height_m=height)
    boundary, _ = model._port(cs)
    cells = []
    for _ in pipe.lengths:
        row = {}
        for name, q in boundary.items():
            volume = q.volume_m3
            if small:
                volume = pipe.volumes[0] * (0.4 if name == "liquid" else 0.6)
            row[name] = phase(
                f,
                q.pressure_pa * (1.01 if reverse else 0.99),
                q.temperature_k,
                "liquid" if name == "liquid" else "vapor",
                volume,
                x=tuple(n / sum(q.component_moles) for n in q.component_moles),
            )
        cells.append(row)
    return model, model.initialize(pipe.initialize(cells), cs, next_step_s=1e-7)


@pytest.mark.parametrize("key", ["boundary_heights_m", "boundary_distances_m"])
def test_column_owns_the_aperture_geometry_instead_of_ignoring_prescribed_values(key):
    model, _ = setup()
    with pytest.raises(ValueError, match="owns the coupled face geometry"):
        PRCapillaryColumnNetwork(
            model.pipe,
            model.column,
            {**model.parameters, key: [0.001, None]},
            port_side=model.port_side,
            port_height_m=model.height,
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_finite_column_and_capillary_pipe_share_components_augmented_energy_and_profile(reverse):
    from copy import deepcopy

    from check_pr_capillary_column import audit

    model, old = setup(reverse=reverse)
    snapshot = old.to_dict()
    final, report = model.step(old, 1e-7)
    assert old.to_dict() == snapshot
    transfer = report["pipe"]["axial"]["external_transfer_n_m_e"]
    assert (sum(transfer[:2]) < 0) == reverse
    np.testing.assert_allclose(
        np.array(old.column.component_moles) - final.column.component_moles,
        transfer[:2],
        rtol=1e-5,
        atol=1e-14,
    )
    assert abs(report["conservation_residuals"][-1]) <= report["energy_roundoff_allowance_j"]
    assert report["entropy_production_j_k"] >= -1e-10
    axial = report["pipe"]["axial"]
    base, _ = model._axial(old.column)
    assert axial_audit(base, old.pipe, PRActiveMultifluidState.from_dict(axial["state"]), axial)[
        "accepted"
    ]
    assert all(0 < p["area_fraction"] < 1 for p in report["end_aperture"])
    restored = PRCapillaryColumnNetwork.from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PRCapillaryColumnState.from_dict(json.loads(json.dumps(old.to_dict())))
    same, _ = restored.step(checkpoint, 1e-7)
    assert same == final
    assert audit(model.to_dict(), old, final, report)["accepted"]
    if not reverse:
        damaged = deepcopy(report)
        damaged["boundary_shape_work_from_column_j"] += 1e-8
        with pytest.raises(ValueError, match="shape work"):
            audit(model.to_dict(), old, final, damaged)


def test_coupled_interval_covers_time_and_preserves_checkpoint_continuation():
    model, old = setup()
    final, r = model.advance(old, 2e-7)
    assert final.time_s == 2e-7
    assert r["steps"][-1]["state"] == final.to_dict()
    again, _ = model.advance(final, 1e-7)
    restored = PRCapillaryColumnNetwork.from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PRCapillaryColumnState.from_dict(json.loads(json.dumps(final.to_dict())))
    expected, _ = restored.advance(checkpoint, 1e-7)
    assert again == expected


def test_moving_aperture_geometry_work_is_funded_by_the_finite_column():
    from check_pr_capillary_column import audit

    model, old = setup(small=True)
    final, report = model.step(old, 1e-7)
    assert report["column_energy_iterations"] >= 2
    assert (
        abs(report["boundary_shape_work_from_column_j"])
        > 10 * report["energy_roundoff_allowance_j"]
    )
    assert audit(model.to_dict(), old, final, report)["accepted"]


def test_coupled_time_refinement_approaches_the_declared_first_order():
    model, old = setup()
    solutions = []
    for count in (1, 2, 4):
        state = old
        for _ in range(count):
            state, _ = model.step(state, 2e-7 / count)
        solutions.append(np.array([p.momentum_kg_m_s for c in state.pipe.cells for _, p in c]))
    errors = [np.linalg.norm(a - b) for a, b in zip(solutions, solutions[1:])]
    assert errors[1] > 1e-20
    assert 1.8 < errors[0] / errors[1] < 2.2


def test_gas_pickup_above_a_liquid_layer_has_no_phantom_liquid_boundary():
    from check_pr_capillary_column import audit

    base, old = setup()
    raw = base.pipe.to_dict()
    raw["face_elevations_m"] = [0.08] * 4
    pipe = PRActiveMultifluidPipe.from_dict(raw)
    model = PRCapillaryColumnNetwork(
        pipe, base.column, base.parameters, port_side=0, port_height_m=0.08
    )
    initial = model.initialize(pipe.initialize([dict(c) for c in old.pipe.cells]), old.column)
    boundary = model.inspect(initial)
    assert set(boundary.boundaries[0]) == {"gas"}
    final, report = model.step(initial, 1e-7)
    assert len(final.column.phase_roots) == 2
    assert report["end_aperture"][0]["area_fraction"] == 0.0
    assert report["end_aperture"][0]["sample"] is None
    assert audit(model.to_dict(), initial, final, report)["accepted"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_coupled_heat_cycle_removes_and_recreates_the_actual_column_interface(device):
    from check_pr_capillary_column import audit

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, old = setup()
    if device == "cuda":
        model = PRCapillaryColumnNetwork.from_dict(
            json.loads(json.dumps(model.to_dict())), device=device
        )
        old = PRCapillaryColumnState.from_dict(json.loads(json.dumps(old.to_dict())))
    hot, heating = model.step(old, 1e-7, column_heat_w=1e11, heat_temperature_k=450.0)
    assert len(hot.column.phase_roots) == len(hot.column.phase_seeds) == 1
    assert hot.column.interface_heights_m == ()
    assert hot.layer_slots == ("gas",)
    assert set(heating["end_pipe_model"]["boundaries"][0]) == {"gas"}
    assert audit(model.to_dict(), old, hot, heating, column_heat_w=1e11, heat_temperature_k=450.0)[
        "accepted"
    ]
    cold, cooling = model.step(hot, 1e-7, column_heat_w=-1e11, heat_temperature_k=160.0)
    assert len(cold.column.phase_roots) == len(cold.column.interface_heights_m) + 1 == 2
    assert cold.layer_slots == ("liquid", "gas")
    assert audit(
        model.to_dict(), hot, cold, cooling, column_heat_w=-1e11, heat_temperature_k=160.0
    )["accepted"]
    assert (
        abs(model.ledger(old, cold)[-1])
        <= heating["energy_roundoff_allowance_j"] + cooling["energy_roundoff_allowance_j"]
    )
    restored = PRCapillaryColumnNetwork.from_dict(
        json.loads(json.dumps(model.to_dict())), device=device
    )
    checkpoint = PRCapillaryColumnState.from_dict(json.loads(json.dumps(hot.to_dict())))
    repeated, _ = restored.step(checkpoint, 1e-7, column_heat_w=-1e11, heat_temperature_k=160.0)
    assert repeated == cold


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_physical_checkpoint_couples_on_cpu_and_cuda_with_independent_audits():
    from check_pr_capillary_column import audit

    cpu, old = setup()
    cuda = PRCapillaryColumnNetwork.from_dict(json.loads(json.dumps(cpu.to_dict())), device="cuda")
    gpu_old = PRCapillaryColumnState.from_dict(json.loads(json.dumps(old.to_dict())))
    expected, _ = cpu.step(old, 1e-7)
    actual, report = cuda.step(gpu_old, 1e-7)
    assert audit(cuda.to_dict(), gpu_old, actual, report)["accepted"]
    np.testing.assert_allclose(
        actual.column.component_moles, expected.column.component_moles, rtol=1e-12, atol=1e-15
    )
    assert actual.column.temperature_k == pytest.approx(expected.column.temperature_k, abs=1e-8)
    for a, b in zip(actual.pipe.cells, expected.pipe.cells):
        for (_, x), (_, y) in zip(a, b):
            np.testing.assert_allclose(x.component_moles, y.component_moles, rtol=1e-10, atol=1e-20)
            assert x.temperature_k == pytest.approx(y.temperature_k, abs=1e-8)
