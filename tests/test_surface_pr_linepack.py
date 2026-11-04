"""Finite PR pipe storage, conservative coupling, isolation and restart."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]
from surface.pr_linepack import HorizontalPRPipe, PRLinepackNetwork
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
from surface.stratified_pr_flow import StratifiedPRFlow
from surface.stratified_pr_network import StratifiedNetworkState
from surface.vessel_wall import VesselWall

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def setup(device="cpu", cells=2, wall=None, reverse=False, gravity=0.0):
    cfg = json.loads((ROOT / "configs/surface_stratified_pr_gas.json").read_text())["system"]
    cfg["geometry"]["length_m"] = 2.0
    cfg["gravity_m_s2"] = gravity
    models = {}
    states = {}
    for name, p, x in [
        ("a", 999990.0 if reverse else 1e6, [0.7, 0.3]),
        ("b", 1e6 if reverse else 999990.0, [0.2, 0.8]),
    ]:
        col = StratifiedPRColumn.from_dict(cfg, device=device)
        s = StratifiedPRState.from_dict(col.initialize(p, 320.0, x)["state"])
        v = StratifiedPRFlow(col, VesselWall(1000.0, 0.0))
        models[name] = v
        states[name] = v.initialize(s, 320.0)
    pipe = HorizontalPRPipe(
        "export",
        "a",
        "b",
        1.0,
        1.0,
        1.0,
        0.05,
        cells,
        0.01,
        wall or VesselWall(1000.0, 0.0),
        maximum_velocity_m_s=10.0,
    )
    model = PRLinepackNetwork(models, [pipe], initial_step_s=0.005)
    initial = model.initialize(
        states,
        {"export": {"pressure_pa": 999995.0, "temperature_k": 320.0, "mole_fractions": [0.3, 0.7]}},
    )
    return model, initial


@pytest.mark.parametrize("device", DEVICES)
def test_pipe_volume_is_real_inventory_with_half_cell_end_resistances(device):
    m, s = setup(device)
    cells = m.cell_names["export"]
    assert len(cells) == 2
    inventory = m._inventory(s, m.pipes[0])
    assert inventory["volume_m3"] == pytest.approx(math.pi * 0.05**2 / 4)
    assert inventory["mass_kg"] > 0
    assert sum(m.network.vessels[n].column.geometry.volume_m3 for n in cells) == pytest.approx(
        inventory["volume_m3"]
    )
    assert [c.length_m for c in m.network.connections] == pytest.approx([0.25, 0.5, 0.25])
    assert sum(c.length_m for c in m.network.connections) == 1.0
    assert sum(m.network.vessels[n].wall.heat_capacity_j_k for n in cells) == 1000.0
    reconstructed = PRLinepackNetwork.from_dict(json.loads(json.dumps(m.to_dict())), device=device)
    assert reconstructed.model_id == m.model_id
    reconstructed.inspect(StratifiedNetworkState.from_dict(s.to_dict()))


def test_linepack_component_energy_ledgers_and_composition_delay():
    m, s = setup()
    before = s.to_dict()
    result = m.advance(s, 0.005)
    p = result["pipes"]["export"]
    assert max(map(abs, p["component_residuals_mol"])) < 1e-9
    assert abs(p["energy_residual_j"]) < 1e-6
    assert abs(result["energy_residual_j"]) < 1e-6
    old = dict(s.vessels)
    changes = []
    for cell in p["cells"]:
        oldx = old[cell["name"]].column.component_moles
        previous = oldx[0] / sum(oldx)
        current = cell["component_moles"][0] / sum(cell["component_moles"])
        changes.append(current - previous)
    assert changes[0] > 0
    assert 0 < changes[1] < changes[0] * 0.1
    assert s.to_dict() == before
    assert result["state"]["time_s"] == 0.005


def test_two_end_isolation_keeps_linepack_inside_pipe_and_heats_it():
    m, s = setup(wall=VesselWall(1000.0, 10.0))
    children = dict(s.vessels)
    for name in m.cell_names["export"]:
        children[name] = replace(children[name], wall_temperature_k=350.0)
    s = m.network.initialize(children)
    faces = m.face_names["export"]
    result = m.advance(s, 0.005, openings={faces[0]: 0.0, faces[-1]: 0.0})
    p = result["pipes"]["export"]
    assert p["source_component_transfer_mol"] == [0.0, 0.0]
    assert p["target_component_transfer_mol"] == [0.0, 0.0]
    assert p["component_moles"] == pytest.approx(
        m._inventory(s, m.pipes[0])["component_moles"], abs=1e-10
    )
    assert all(c["temperature_k"] > 320.0 for c in p["cells"])
    final = dict(StratifiedNetworkState.from_dict(result["state"]).vessels)
    assert final["a"].column.component_moles == children["a"].column.component_moles
    assert final["b"].column.component_moles == children["b"].column.component_moles
    assert abs(p["energy_residual_j"]) < 1e-6


def test_reverse_flow_uses_reverse_donor_composition():
    m, s = setup(reverse=True)
    result = m.advance(s, 0.005)
    p = result["pipes"]["export"]
    assert sum(p["source_component_transfer_mol"]) < 0
    assert sum(p["target_component_transfer_mol"]) < 0
    assert abs(p["energy_residual_j"]) < 1e-6
    old = dict(s.vessels)[m.cell_names["export"][-1]].column.component_moles
    new = p["cells"][-1]["component_moles"]
    assert new[0] / sum(new) < old[0] / sum(old)


@pytest.mark.parametrize("device", DEVICES)
def test_monolithic_pipe_step_has_conservative_face_fluxes(device):
    m, s = setup(device)
    final, report = m.network._step(s, 0.001, m.network._commands(None))
    old, new = m._inventory(s, m.pipes[0]), m._inventory(final, m.pipes[0])
    transfers = {p["name"]: p for p in report["transfers"]}
    first, last = transfers[m.face_names["export"][0]], transfers[m.face_names["export"][-1]]
    for i, (a, b) in enumerate(zip(old["component_moles"], new["component_moles"])):
        assert b - a == pytest.approx(
            0.001 * (first["component_rates_mol_s"][i] - last["component_rates_mol_s"][i]),
            abs=1e-10,
        )
    assert new["total_energy_j"] - old["total_energy_j"] == pytest.approx(
        0.001 * (first["total_energy_rate_w"] - last["total_energy_rate_w"]), abs=1e-6
    )
    assert all(p["hydraulic_model"] == "darcy_weisbach" for p in transfers.values())
    assert report["entropy_production_j_k"] >= -1e-7


def test_pipe_validation_and_axial_initial_arrays():
    m, s = setup()
    pipe = m.pipes[0]
    with pytest.raises(ValueError, match="elevations differ"):
        PRLinepackNetwork(dict(m.terminals), [replace(pipe, target_height_m=0.8)])
    with pytest.raises(ValueError, match="1..64"):
        replace(pipe, cells=True)
    children = {n: child for n, child in s.vessels if n in m.terminals}
    resumed = m.initialize(
        children,
        {
            "export": {
                "pressure_pa": [999997.0, 999993.0],
                "temperature_k": [320.0, 321.0],
                "mole_fractions": [[0.3, 0.7], [0.4, 0.6]],
            }
        },
    )
    assert dict(resumed.vessels)[m.cell_names["export"][1]].column.temperature_k == pytest.approx(
        321.0, abs=1e-6
    )
    with pytest.raises(TypeError):
        m.cell_names["export"] = ("wrong",)


def test_independent_auditor_rejects_geometry_and_face_energy_tampering():
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_pr_linepack import audit

    m, s = setup()
    report = m.advance(s, 0.001)
    cfg = {
        "model": m.to_dict(),
        "initial": {"terminals": {}, "pipes": {}},
        "schedule": [{"duration_s": 0.001}],
    }
    result = {
        "config": cfg,
        "model": m.to_dict(),
        "compiled_network": m.network.to_dict(),
        "initial": s.to_dict(),
        "intervals": [report],
    }
    assert audit(cfg, result)["accepted"]
    broken = json.loads(json.dumps(result))
    broken["compiled_network"]["connections"][0]["length_m"] *= 2
    with pytest.raises(ValueError, match="geometry/resistance"):
        audit(cfg, broken)
    broken = json.loads(json.dumps(result))
    broken["intervals"][0]["steps"][0]["transfers"][0]["total_energy_rate_w"] += 1.0
    with pytest.raises(ValueError, match="audit mismatch"):
        audit(cfg, broken)
    broken = json.loads(json.dumps(result))
    broken["intervals"][0]["pipes"]["export"]["mass_kg"] *= 2
    with pytest.raises(ValueError, match="audit mismatch"):
        audit(cfg, broken)


def test_source_check_valve_keeps_internal_redistribution_and_backfill():
    old, state = setup(reverse=True)
    pipe = replace(old.pipes[0], allow_reverse=False)
    model = PRLinepackNetwork(dict(old.terminals), [pipe], initial_step_s=0.001)
    initial = model.network.initialize(dict(state.vessels))
    assert [c.allow_reverse for c in model.network.connections] == [False, True, True]
    result = model.advance(initial, 0.001)
    p = result["pipes"]["export"]
    assert p["source_component_transfer_mol"] == [0.0, 0.0]
    assert sum(p["target_component_transfer_mol"]) < 0
    assert sum(p["component_moles"]) > sum(model._inventory(initial, pipe)["component_moles"])
    assert abs(p["energy_residual_j"]) < 1e-6


def test_horizontal_pipe_gravity_energy_and_port_elevations():
    m, s = setup(gravity=9.80665)
    center = m.terminals["a"].column.elevation_m + 1.0
    for name in m.cell_names["export"]:
        col = m.network.vessels[name].column
        assert col.elevation_m + col.geometry.height_m / 2 == pytest.approx(center)
    final, report = m.network._step(s, 0.001, m.network._commands(None))
    old, new = m._inventory(s, m.pipes[0]), m._inventory(final, m.pipes[0])
    fluxes = report["transfers"]
    expected = 0.001 * (fluxes[0]["total_energy_rate_w"] - fluxes[-1]["total_energy_rate_w"])
    assert new["total_energy_j"] - old["total_energy_j"] == pytest.approx(expected, abs=1e-6)
    assert abs(report["energy_residual_j"]) < 1e-6
