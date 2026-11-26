"""Multi-depth injection: hydrostatic datum, valve law and full Schur tangents."""

import copy

import pytest
import torch
from test_fim_waterflood import configuration, simulation

from simulator.fim_wells import G
from surface.liquid import HydraulicConvergenceError


def configuration_3d():
    c = configuration()
    c["reservoir"]["dimensions"] = [3, 1, 3]
    c["reservoir"]["grid"] = {"dz": [20, 30, 50]}
    c["wells"][1]["ref_depth"] = 40
    network = c["simulation"]["physical_surface_network"]["network"]
    for node in network["nodes"]:
        if node["name"].startswith("bottom_"):
            node["elevation_m"] = -10
    c["wells"][1]["perforations"] = [[1, 0, k] for k in range(3)]
    return c


@pytest.mark.parametrize("density_model", ["cell", "flash"])
def test_water_column_head_and_perforation_valves_at_different_depths(density_model):
    s = simulation(configuration_3d()).fim_solver
    s.wells.wellbore_density = density_model
    p = s.fluid.pressure.flatten().clone()
    sw = s.fluid.s_w.flatten()
    child = s.wells.injection
    pb = p.new_tensor([3.2e6])
    expected = 1000 * G * p.new_tensor([-30, -5, 35])
    # Exactly equilibrated column; pressure is the oil-pressure potential
    # used by the existing standard-well law (no capillary pressure here).
    p[child.conn_cell] = pb[0] + expected
    v, d = s.cell_state(p, sw, s.x, s.sat, s.ug)
    s.wells.begin_step(v, d, 0)
    torch.testing.assert_close(child.conn_head, expected, rtol=0, atol=1e-9)
    q, _, _ = child._conn_flows(v, d, pb)
    torch.testing.assert_close(q, torch.zeros_like(q), rtol=0, atol=1e-12)
    # Opposite signs around the datum: only the upper valve blocks.
    p[child.conn_cell] += p.new_tensor([1e4, -1e4, -2e4])
    v, d = s.cell_state(p, sw, s.x, s.sat, s.ug)
    q, dq, cv = child._conn_flows(v, d, pb)
    assert child._injection_valve_open.tolist() == [False, True, True]
    assert torch.count_nonzero(q[0]) == torch.count_nonzero(dq[0]) == torch.count_nonzero(cv[0]) == 0
    assert (q[1:, 1] < 0).all()
    assert torch.count_nonzero(q[:, [0, 2]]) == 0


def test_multidepth_full_well_and_reservoir_jacobians():
    s = simulation(configuration_3d()).fim_solver
    child = s.wells.injection
    child.speed_ratios = {"injection_booster": 0.6}
    child.wells[0].control_value = 10000
    p = s.fluid.pressure.flatten().clone()
    p.fill_(3e6)  # Deliberate local Jacobian state, independent of initialization.
    sw = s.fluid.s_w.flatten().clone()
    x = s.x.clone()
    old = s.masses(p, sw, x, s.sat, s.ug)
    s.wells.begin_step(*s.cell_state(p, sw, x, s.sat, s.ug), 0)
    pb = p.new_tensor([2.4e6, 3.2e6, 2.4e6])

    def assemble(pp=p, ww=sw, xx=x, bhp=pb):
        return s.assemble_bo(pp, ww, xx, s.sat, old, 864, bhp, ug=s.ug)

    _, _, system = assemble()
    assert child.step_mode == ["surface"]
    assert child._injection_valve_open.tolist() == [False, True, True]
    B, D, cv = (system[k].clone() for k in ["B_full", "D_full", "Cv"])
    a, b = pb.clone(), pb.clone()
    a[1] += 2
    b[1] -= 2
    ra, _, sa = assemble(bhp=a)
    rb, _, sb = assemble(bhp=b)
    torch.testing.assert_close((sa["r_w"] - sb["r_w"]) / 4, D[:, 1], rtol=4e-5, atol=1e-5)
    mask = s.wells.conn_well == 1
    expected = cv.new_zeros((s.n_cells, 3)).index_add_(0, s.wells.conn_cell[mask], cv[mask])
    torch.testing.assert_close((ra - rb).view(-1, 3) / 4, expected, rtol=4e-5, atol=1e-9)
    for cell in child.conn_cell.tolist():
        for k, h in enumerate((2.0, 1e-6, 1e-6)):
            a = [p.clone(), sw.clone(), x.clone()]
            b = [v.clone() for v in a]
            a[k][cell] += h
            b[k][cell] -= h
            numerical = (assemble(*a)[2]["r_w"] - assemble(*b)[2]["r_w"]) / (2 * h)
            torch.testing.assert_close(
                numerical, B[:, s.wells.conn_cell == cell, k].sum(1), rtol=4e-5, atol=0.003
            )


def test_bhp_reference_change_preserves_physical_perforation_pressure_and_flow():
    first = simulation(configuration_3d()).fim_solver
    config = configuration_3d()
    config["wells"][1]["ref_depth"] = 65
    network = config["simulation"]["physical_surface_network"]["injection_network"]["network"]
    next(n for n in network["nodes"] if n["name"] == "injection_1")["elevation_m"] = -65
    second = simulation(config).fim_solver
    results = []
    for s, bhp in [(first, 3.2e6), (second, 3.2e6 + 1000 * G * 25)]:
        v, d = s.cell_state(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug)
        s.wells.begin_step(v, d, 0)
        w = s.wells.injection
        results.append((w._conn_flows(v, d, v.new_tensor([bhp])), bhp + w.conn_head))
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=1e-9)
    for a, b in zip(results[0][0], results[1][0]):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_multidepth_coupled_step_restart_and_report_audit():
    from check_fim_waterflood import audit
    from run_fim_waterflood import run

    config = configuration_3d()
    config["simulation"].update(newton_tol_cnv=1e-9, newton_tol_mb=1e-13)
    events = [
        {"time_days": 0, "controls": {}},
        {"time_days": 0.01, "controls": {"injection_enabled": False}},
        {"time_days": 0.02, "controls": {"injection_enabled": True}},
    ]
    result = run(config, 0.03, 0.01, events, save_fields=True)
    assert audit(config, result)["accepted"]
    assert result["accepted_substeps"][-1]["coupling"]["water_rates_kg_s"]["injected"] > 0
    damaged = copy.deepcopy(result)
    damaged["accepted_substeps"][-1]["coupling"]["injection_network"]["perforations"][0][
        "hydrostatic_head_pa"
    ] += 100
    with pytest.raises(ValueError, match="audit mismatch"):
        audit(config, damaged)


def test_connected_column_below_vapor_pressure_cannot_commit():
    s = simulation(configuration_3d()).fim_solver
    p = s.fluid.pressure.flatten()
    sw = s.fluid.s_w.flatten()
    old = s.masses(p, sw, s.x, s.sat, s.ug)
    s.wells.begin_step(*s.cell_state(p, sw, s.x, s.sat, s.ug), 0)
    pb = p.new_tensor([2.4e6, 1e5, 2.4e6])
    s.assemble_bo(p, sw, s.x, s.sat, old, 864, pb, ug=s.ug)
    child = s.wells.injection
    assert not child.injection_report["feasible"]
    assert "water_column_vapor_pressure:I1" in child.injection_report["violations"]
    with pytest.raises(HydraulicConvergenceError, match="envelope rejected"):
        child.commit(pb[1:2])
