"""Short release checks for actual profile-cell transport (no pipe wrapper)."""

import json
import math
from pathlib import Path

import pytest
import torch

from surface.pr_layered_transport import LayeredTransportState, PRLayeredTransportNetwork
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState

ROOT = Path(__file__).resolve().parents[1]


def model(kind="surface_stratified_pr", angle=0.1, elevation=0.0, cells=2, face_order=2):
    cfg = json.loads((ROOT / "configs" / f"{kind}.json").read_text())
    columns = []
    for i in range(cells):
        c = {
            **cfg["system"],
            "elevation_m": elevation + i * 0.1 * math.sin(angle),
            "geometry": {
                "orientation": "inclined",
                "diameter_m": 0.1,
                "length_m": 0.1,
                "inclination_rad": angle,
            },
        }
        columns.append(StratifiedPRColumn.from_dict(c))
    return PRLayeredTransportNetwork(
        columns, face_order=face_order, wave_speed_bound_m_s=10000
    ), cfg["initial"]


@pytest.fixture(scope="module")
def mixed():
    n, initial = model()
    return n, n.initialize_equilibrium(**initial)


def test_mixed_rest_inside_and_shared_face(mixed, monkeypatch):
    n, state = mixed
    assert [len(c.profile.phase_roots) for c in state.cells] == [2, 2]
    original = state.to_dict()

    # Only initialization may equilibrate/sample; residual/acceptance cannot.
    def forbidden(*args, **kwargs):
        raise AssertionError("Nested equilibrium/profile sampling")

    monkeypatch.setattr(StratifiedPRColumn, "equilibrate", forbidden)
    monkeypatch.setattr(StratifiedPRColumn, "sample", forbidden)
    out, report = n.step(state, 1e-5)
    assert max(abs(c.momentum_kg_m_s / n._mass(c.profile)) for c in out.cells) < 1e-8
    assert report["source_frozen"] is False
    assert report["source_n"] == pytest.approx(report["gravity_mass_n"], abs=1e-7)
    assert max(map(abs, report["component_residuals_mol"])) < 1e-9
    assert abs(report["energy_residual_j"]) < 1e-6
    assert abs(report["momentum_residual_kg_m_s"]) < 1e-8
    assert report["face_patch_count"] == 4
    assert state.to_dict() == original
    y, _, _, evaluate = n._problem(out, 1e-5)
    _, data = evaluate(y)
    assert len(data["shared"]) == 4
    area = sum(float(q[2].sum()) for q in data["shared"])
    assert area == pytest.approx(math.pi * 0.1**2 / 4, abs=1e-13)
    assert max(abs(v) for patch in report["face_contact_velocity_m_s"] for v in patch) < 1e-8
    assert max(abs(v) for patch in report["face_pressure_jump_pa"] for v in patch) < 3e-5


def test_mixed_pressure_driven_both_directions(mixed):
    n, _s = mixed
    cols = n.columns
    # Separate initial thermodynamic profiles have a real pressure mismatch.
    for pressures, sign in [((1.01e6, 1e6), 1), ((1e6, 1.01e6), -1)]:
        profiles = [
            StratifiedPRState.from_dict(c.initialize(p, 196.2, [0.19, 0.81])["state"])
            for c, p in zip(cols, pressures)
        ]
        initial = n.initialize(profiles)
        out, report = n.step(initial, 1e-5)
        assert all(sign * c.momentum_kg_m_s > 0 for c in out.cells)
        assert max(map(abs, report["component_residuals_mol"])) < 1e-8
        assert abs(report["energy_residual_j"]) < 1e-6
        assert any(
            abs(a.total_energy_j - b.total_energy_j) > 1e-5
            for a, b in zip(initial.cells, out.cells)
        )
        for cell in out.cells:
            assert cell.total_energy_j - cell.profile.total_energy_j == pytest.approx(
                0.5 * cell.momentum_kg_m_s**2 / n._mass(cell.profile), abs=1e-10
            )


def test_joint_jacobian_fd_and_live_source(mixed):
    n, s = mixed
    y, _, _, evaluate = n._problem(s, 1e-5)
    # Offset coincident interface kinks; compare at a smooth driven iterate.
    y = y + torch.linspace(-2e-5, 3e-5, len(y), dtype=y.dtype)
    direction = torch.linspace(-0.003, 0.004, len(y), dtype=y.dtype)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    eps = 1e-6
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert torch.isfinite(jac).all()
    assert float((jac @ direction - fd).abs().max()) < 2e-6
    sourcejac = torch.func.jacfwd(lambda z: torch.stack(evaluate(z)[1]["sources"]))(y)
    assert float(sourcejac.abs().max()) > 1


def test_datum_reverse_slope_and_restart():
    results = []
    for elevation in (0.0, 10000.0):
        n, initial = model("surface_stratified_pr_gas", angle=-0.1, elevation=elevation)
        s = n.initialize_equilibrium(**initial)
        out, report = n.step(s, 1e-5)
        assert report["source_n"][0] > 0
        restored = PRLayeredTransportNetwork.from_dict(json.loads(json.dumps(n.to_dict())))
        checkpoint = LayeredTransportState.from_dict(json.loads(json.dumps(out.to_dict())))
        replay, _ = restored.step(checkpoint, 1e-5)
        direct, _ = n.step(out, 1e-5)
        assert replay.to_dict() == direct.to_dict()
        results.append(out)
    for a, b in zip(*[r.cells for r in results]):
        assert a.profile.component_moles == pytest.approx(b.profile.component_moles, rel=1e-9)
        assert a.momentum_kg_m_s == pytest.approx(b.momentum_kg_m_s, abs=1e-9)


def test_reject_unsupported_closures(mixed):
    n, _s = mixed
    with pytest.raises(ValueError, match="slip"):
        PRLayeredTransportNetwork(n.columns, slip=True)
    with pytest.raises(ValueError, match="lateral"):
        PRLayeredTransportNetwork(n.columns, lateral_ports=("side",))


def test_sparse_matches_dense_and_pattern_covers_ad(mixed):
    dense, state = mixed
    sparse = PRLayeredTransportNetwork(
        dense.columns, face_order=dense.face_order, wave_speed_bound_m_s=dense.alpha, dense_limit=1
    )
    s = sparse.initialize([c.profile for c in state.cells], [0.03, -0.01])
    d = dense.initialize([c.profile for c in state.cells], [0.03, -0.01])
    y, _, _, evaluate = sparse._problem(s, 1e-5)
    y = y + torch.linspace(-1e-5, 2e-5, len(y), dtype=y.dtype)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    mask[evaluate.sparsity[0], evaluate.sparsity[1]] = True
    assert float(jac[~mask].abs().max()) < 1e-14
    assert sorted(v for block in evaluate.linear_blocks for v in block) == list(range(len(y)))
    assert max(map(len, evaluate.linear_blocks)) <= 256
    a, _ra = dense.step(d, 1e-5)
    b, rb = sparse.step(s, 1e-5)
    for x, z in zip(a.cells, b.cells):
        assert x.profile.component_moles == pytest.approx(z.profile.component_moles, abs=1e-9)
        assert x.momentum_kg_m_s == pytest.approx(z.momentum_kg_m_s, abs=1e-9)
        assert x.total_energy_j == pytest.approx(z.total_energy_j, abs=1e-6)
    assert rb["max_scaled_residual"] < 1e-9


def test_heat_birth_death_and_adaptive_restart():
    n, _ = model("surface_stratified_pr_gas", angle=0.0, cells=1)
    c = n.columns[0]
    f = n.fluid
    ethane = f.components[1]
    temperature = 200.0
    pressure = ethane.saturation_at_temperature(temperature)["pressure_pa"]
    liquid = ethane.state(pressure, temperature, "liquid", derivatives=False)
    vapor = ethane.state(pressure, temperature, "vapor", derivatives=False)
    mw = ethane.molar_mass_kg_mol
    volume = 0.4 * liquid["specific_volume_m3_kg"] * mw + 0.6 * vapor["specific_volume_m3_kg"] * mw
    from surface.peng_robinson import R

    attraction, _ = ethane._a(220.0)
    p0 = R * 220 / (volume - ethane.b) - attraction / (
        volume**2 + 2 * ethane.b * volume - ethane.b**2
    )
    profile = StratifiedPRState.from_dict(c.initialize(p0, 220.0, [0.0, 1.0])["state"])
    state = n.initialize([profile])
    assert len(profile.phase_roots) == 1
    amount = profile.component_moles[1]
    target = amount * (
        0.4 * (liquid["enthalpy_j_kg"] * mw - pressure * liquid["specific_volume_m3_kg"] * mw)
        + 0.6 * (vapor["enthalpy_j_kg"] * mw - pressure * vapor["specific_volume_m3_kg"] * mw)
    )
    # Include the unchanged volume-integrated gravitational potential.
    target += n._mass(profile) * n.gravity * c.geometry.height_m / 2
    dt = 1e-3
    heat = (target - state.cells[0].total_energy_j) / dt
    born, r = n.step(state, dt, heat_w=heat)
    assert len(born.cells[0].profile.phase_roots) == 2
    assert r["topology_retries"] > 0
    assert born.cells[0].total_energy_j == pytest.approx(target, abs=1e-6)
    dead, r = n.step(born, dt, heat_w=-heat)
    assert len(dead.cells[0].profile.phase_roots) == 1
    assert dead.cells[0].total_energy_j == pytest.approx(state.cells[0].total_energy_j, abs=1e-6)
    out, report = n.advance(dead, 2e-5, maximum_step_s=1e-5)
    checkpoint = LayeredTransportState.from_dict(json.loads(json.dumps(out.to_dict())))
    a, _ = n.advance(checkpoint, 2e-5, maximum_step_s=1e-5)
    b, _ = n.advance(out, 2e-5, maximum_step_s=1e-5)
    assert a.to_dict() == b.to_dict()
    assert report and report[-1]["source_frozen"] is False


def test_three_layers_in_cells_and_face():
    n, initial = model("surface_stratified_pr_3phase", angle=0.001)
    s = n.initialize_equilibrium(**initial)
    assert [len(c.profile.phase_roots) for c in s.cells] == [3, 3]
    out, r = n.step(s, 1e-5)
    assert r["face_patch_count"] == 9
    assert max(abs(c.momentum_kg_m_s) for c in out.cells) < 1e-8
    assert max(abs(v) for patch in r["face_contact_velocity_m_s"] for v in patch) < 1e-8


def test_common_residual_finite_vessel_bridge(mixed):
    from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork
    from surface.torch_linear import TorchNewton

    n, state = mixed
    center = n._disk_bottom(0, -1) + n.diameter * n.cosine / 2
    ordinary = PREquilibriumTransportNetwork(
        n.fluid,
        [PREquilibriumNode(name, 0.001, elevation_m=center) for name in ("junction", "unrelated")],
        gravity_m_s2=n.gravity,
        wave_speed_bound_m_s=n.alpha,
    )
    old = ordinary.initialize(
        {
            name: {
                "pressure_pa": 1.005e6,
                "temperature_k": 196.2,
                "mole_fractions": [0.19, 0.81],
            }
            for name in ordinary.names
        }
    )
    seeds = [ordinary._seed(i, s) for i, (_, s) in enumerate(old.nodes)]
    op = ordinary._problem(
        old, seeds, 1e-5, {}, [0.0] * len(ordinary.nodes), external_ports=((0, None),)
    )
    op[3].sparsity = ordinary._sparsity(op[4], {})
    y, lo, hi, evaluate = n.joint_vessel_problem(
        ordinary, op, state, 1e-5, boundary_nodes=("junction", None)
    )
    direction = torch.linspace(-0.001, 0.002, len(y), dtype=y.dtype)
    smooth = y + torch.linspace(-1e-5, 2e-5, len(y), dtype=y.dtype)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(smooth)
    eps = 1e-6
    fd = (evaluate(smooth + eps * direction)[0] - evaluate(smooth - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 2e-6
    assert float(jac[: len(op[0]), len(op[0]) :].abs().max()) > 0
    assert float(jac[len(op[0]) :, : len(op[0])].abs().max()) > 0
    mask = torch.zeros_like(jac, dtype=torch.bool)
    mask[evaluate.sparsity[0], evaluate.sparsity[1]] = True
    assert float(jac[~mask].abs().max()) < 1e-14
    # An unrelated finite vessel must retain zero cross-block support.
    unrelated = range(op[4][1], op[4][2])
    assert not mask[list(unrelated), len(op[0]) :].any()
    assert not mask[len(op[0]) :, list(unrelated)].any()
    solver = TorchNewton(n.fluid.device, maximum_iterations=40, dense_limit=768)
    solver.host_convergence = True
    with torch.no_grad():
        y, _, ok, _ = solver.solve(lambda z: evaluate(z)[0], y, lo, hi, tolerance=3e-12)
        residual, d = evaluate(y)
    assert ok and float(residual.abs().max()) < 1e-9
    assert d["ordinary"]["valid"]
    assert abs(float(d["vessel_support_force_n"][0])) > 1
    orate = d["ordinary"]["net"][0]
    prates = torch.stack(d["profile"]["rates"]).sum(0)
    assert float((orate[:-2] + prates[:-2]).abs().max()) < 1e-10
    # Convert ordinary datum to native datum before checking opposite energy.
    massrate = (orate[:-2] * n.fluid.molar_masses).sum()
    energy = orate[-1] + massrate * n.gravity * (ordinary.reference_elevation_m - n.datum)
    assert float((energy + prates[-1]).abs()) < 1e-8
    assert float(d["profile"]["velocities"][0]) > 0
    accepted, report = n.accept_solved(
        state, 1e-5, y[len(op[0]) :], d["profile"], external_ends=(True, False)
    )
    assert accepted.time_s == 1e-5
    assert max(map(abs, report["component_residuals_mol"])) < 1e-9
    assert abs(report["energy_residual_j"]) < 1e-6
    assert abs(report["momentum_residual_kg_m_s"]) < 1e-8
    assert report["entropy_production_j_k"] > -1e-6
    assert float(
        d["ordinary"]["exchange"]["incoming_entropy"] + d["profile"]["exchange"]["incoming_entropy"]
    ) == pytest.approx(0.0, abs=1e-12)


def test_native_runner_and_independent_auditor(mixed):
    import sys
    from copy import deepcopy

    sys.path.insert(0, str(ROOT / "scripts"))
    from check_pr_layered_transport import audit
    from run_pr_layered_transport import run

    n, state = mixed
    config = {
        "model": n.to_dict(),
        "initial": {"state": state.to_dict()},
        "schedule": [{"duration_s": 1e-5, "maximum_step_s": 1e-5}],
    }
    config["model"].pop("dynamic_viscosity_pa_s")
    config["model"].pop("roughness_m")
    report = run(config)
    result = audit(config, report)
    assert result["passed"] and result["checked_steps"] == 2
    broken = deepcopy(report)
    broken["intervals"][0]["steps"][0]["state"]["cells"][0]["total_energy_j"] += 100.0
    with pytest.raises(ValueError):
        audit(config, broken)


@pytest.mark.parametrize("kind", ["surface_stratified_pr_gas", "surface_stratified_pr"])
@pytest.mark.parametrize("cells", [1, 2])
def test_reflected_closed_walls_damp_uniform_motion(kind, cells):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from reference_pr_layered_transport import reconstruct

    n, initial = model(kind, angle=0.0, cells=cells)
    rest = n.initialize_equilibrium(**initial)
    profiles = [cell.profile for cell in rest.cells]

    def internal(state):
        return sum(
            float(
                col._problem(
                    n.fluid.tensor(cell.profile.component_moles),
                    cell.profile.total_energy_j,
                    n._seed(col, cell.profile),
                    cell.profile.quadrature_order,
                )[3](n.fluid.tensor(cell.profile.solution_variables))[1]["internal_energy_j"]
            )
            for col, cell in zip(n.columns, state.cells)
        )

    def kinetic(state):
        return sum(0.5 * cell.momentum_kg_m_s**2 / n._mass(cell.profile) for cell in state.cells)

    for velocity in (-0.03, 0.03):
        old = n.initialize(profiles, [velocity] * cells)
        out, report = n.step(old, 1e-5)
        before = sum(c.momentum_kg_m_s for c in old.cells)
        after = sum(c.momentum_kg_m_s for c in out.cells)
        assert 0 < after / before < 0.9
        assert before * report["wall_force_n"] < 0
        assert after - before == pytest.approx(1e-5 * report["wall_force_n"], abs=1e-9)
        assert report["source_n"] == pytest.approx([0.0] * cells, abs=1e-8)
        assert sum(c.total_energy_j for c in out.cells) == pytest.approx(
            sum(c.total_energy_j for c in old.cells), abs=1e-7
        )
        assert kinetic(out) < kinetic(old)
        assert internal(out) > internal(old)
        assert sum(c.profile.total_energy_j for c in out.cells) - sum(
            c.profile.total_energy_j for c in old.cells
        ) == pytest.approx(kinetic(old) - kinetic(out), abs=1e-7)
        assert report["entropy_production_j_k"] > 0
        assert report["incoming_entropy_j_k"] == 0.0
        for flux in report["wall_fluxes"]:
            assert flux[:-2] == [0.0] * len(n.fluid.names)
            assert flux[-1] == 0.0
        assert report["wall_pressure"][0][0] != pytest.approx(report["pressure"][0][0], rel=1e-5)
        if cells == 1:
            # Horizontal extrusion: rho-face integral=M/L. Reflected fans
            # give F_left-F_right=-2*alpha*M*u/L exactly.
            assert after / before == pytest.approx(1 / (1 + 2e-5 * n.alpha / 0.1), rel=1e-7)
        ref = reconstruct(n.to_dict(), out.to_dict())
        assert ref["wall_pressure"].ravel().tolist() == pytest.approx(
            [p for row in report["wall_pressure"] for p in row], rel=1e-6, abs=1e-6
        )
        # The independently reconstructed source still uses thermodynamic P.
        assert ref["source"].tolist() == pytest.approx(report["source_n"], abs=1e-7)


@pytest.mark.parametrize("kind", ["surface_stratified_pr_gas", "surface_stratified_pr"])
def test_declared_effective_mixture_friction(kind):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from reference_pr_layered_transport import reconstruct

    inviscid, initial = model(kind, angle=0.0, cells=1)
    rest = inviscid.initialize_equilibrium(**initial)
    # Synthetic declared viscosity for a strong, analytically laminar check;
    # this value is not a viscosity estimate for either PR test mixture.
    mu = 1e4
    viscous = PRLayeredTransportNetwork(
        inviscid.columns,
        face_order=inviscid.face_order,
        wave_speed_bound_m_s=inviscid.alpha,
        dynamic_viscosity_pa_s=mu,
        roughness_m=0.001,
    )
    restored = PRLayeredTransportNetwork.from_dict(viscous.to_dict())
    assert restored.to_dict() == viscous.to_dict()
    rest_v = viscous.initialize([rest.cells[0].profile])
    still, rs = viscous.step(rest_v, 1e-5)
    assert rs["friction_force_n"] == [0.0]
    assert rs["source_n"] == pytest.approx([0.0], abs=1e-8)
    assert still.cells[0].momentum_kg_m_s == pytest.approx(0.0, abs=1e-12)
    for velocity in (-0.03, 0.03):
        old = inviscid.initialize([rest.cells[0].profile], [velocity])
        old_v = viscous.initialize([rest.cells[0].profile], [velocity])
        base, _ = inviscid.step(old, 1e-5)
        out, r = viscous.step(old_v, 1e-5)
        before = old_v.cells[0].momentum_kg_m_s
        after = out.cells[0].momentum_kg_m_s
        assert abs(after) < abs(base.cells[0].momentum_kg_m_s)
        assert velocity * r["friction_force_n"][0] < 0
        assert after - before == pytest.approx(
            1e-5 * (r["wall_force_n"] + r["friction_force_n"][0]), abs=1e-9
        )
        assert out.cells[0].total_energy_j == pytest.approx(old_v.cells[0].total_energy_j, abs=1e-7)
        assert r["entropy_production_j_k"] > 0
        assert r["source_n"] == pytest.approx([0.0], abs=1e-8)
        density = viscous._mass(out.cells[0].profile) / viscous.columns[0].geometry.volume_m3
        assert after / before == pytest.approx(
            1 / (1 + 1e-5 * (2 * viscous.alpha / 0.1 + 32 * mu / (density * 0.1**2))), rel=1e-7
        )
        ref = reconstruct(viscous.to_dict(), out.to_dict())
        assert ref["friction_force_n"].tolist() == pytest.approx(r["friction_force_n"], rel=1e-7)
    y, _, _, evaluate = viscous._problem(old_v, 1e-5)
    direction = torch.linspace(-0.001, 0.002, len(y), dtype=y.dtype)
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    eps = 1e-6
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 2e-6
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError):
            PRLayeredTransportNetwork(inviscid.columns, dynamic_viscosity_pa_s=bad)
    with pytest.raises(ValueError, match="Roughness"):
        PRLayeredTransportNetwork(inviscid.columns, roughness_m=0.05)
