"""Conservative positive guesses and actual transport of a previously absent species."""

import json
import sys
from pathlib import Path

import pytest
import torch
from test_surface_pr_open_gravity_transport import build

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_pr_open_gravity_transport import audit

from surface.component_energy import ComponentEnergyStream
from surface.pr_material_predictor import carrier_inventory_guess
from surface.pr_open_gravity_transport import PROpenGravityTransportNetwork
from surface.pr_open_transport import PRTransportFeed, _BoundaryProperties, _OpenCoupling
from surface.stratified_pr import StratifiedPRState

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_passive_cycle_predictor_is_positive_conservative_and_keeps_old_stocks(device):
    stocks = torch.tensor([[10.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64, device=device)
    before = stocks.clone()
    incoming = torch.zeros_like(stocks)
    incoming[0, 1] = 1.0
    outgoing = stocks.new_zeros(3)
    edges = [
        (0, 1, stocks.new_tensor(2.0)),
        (1, 2, stocks.new_tensor(1.0)),
        (0, 2, stocks.new_tensor(-0.5)),
    ]
    result = carrier_inventory_guess(stocks, edges, incoming, outgoing, 0.5)
    assert result is not None and result.device == stocks.device
    assert bool((result >= 0).all()) and bool((result[:, 1] > 0).all())
    torch.testing.assert_close(
        result.sum(0), stocks.sum(0) + 0.5 * incoming.sum(0), atol=1e-13, rtol=0.0
    )
    assert torch.equal(stocks, before)
    outgoing[2] = 0.25
    result = carrier_inventory_guess(stocks, edges, incoming, outgoing, 0.5)
    out = (outgoing[:, None] * result / stocks.sum(1)[:, None]).sum(0)
    torch.testing.assert_close(
        result.sum(0) - stocks.sum(0), 0.5 * (incoming.sum(0) - out), atol=1e-13, rtol=0.0
    )


def test_predictor_budget_and_failed_factorization_leave_original_path_available(monkeypatch):
    stocks = torch.ones((257, 2), dtype=torch.float64)
    assert carrier_inventory_guess(stocks, [], stocks * 0, stocks[:, 0] * 0, 1.0) is None

    def fail(a, b, **kw):
        return torch.full_like(b, float("nan")), torch.tensor(1)

    monkeypatch.setattr(torch.linalg, "solve_ex", fail)
    stocks = stocks[:3].clone()
    assert carrier_inventory_guess(stocks, [], stocks * 0, stocks[:, 0] * 0, 1.0) is None


@pytest.mark.parametrize("device", DEVICES)
def test_fresh_boundary_species_reaches_column_as_actual_port_flux_not_seed_stock(device, tmp_path):
    model, _ = build(device)
    g, t, f = model.graph, model.transport, model.fluid
    column = StratifiedPRState.from_dict(
        g.columns["separator"].initialize(1e6, 320.0, [1.0, 0.0])["state"]
    )
    spec = {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [1.0, 0.0]}
    native = t.initialize(
        {
            name: {**spec, "velocity_m_s": 1.0 if t.nodes[i].kind == "pipe_cell" else 0.0}
            for i, name in enumerate(t.names)
        }
    )
    state = g.initialize(native, {"separator": column})
    before = state.to_dict()
    feed = ComponentEnergyStream.from_pt(f, [0.0, 1.0], 1.01e6, 320.0, gravity_m_s2=t.gravity_m_s2)
    m = PROpenGravityTransportNetwork(
        g, [PRTransportFeed("new", "inlet", feed)], model.pressure_ports
    )
    commands = {**t.default_openings, **{p.name: True for p in g.ports}}
    dt = 0.01
    material = _OpenCoupling(
        m, _BoundaryProperties(m, {"boundary": 1.0}), state.transport, dt, commands
    )
    end, raw = g._step(state, dt, commands, {}, {}, {}, _material=material)
    n = dict(end.columns)["separator"].component_moles[1]
    transferred = -dt * sum(p["flux"][1] for p in raw["columns"]["ports"].values())
    assert n > 1e-18 and n == pytest.approx(transferred, abs=1e-24, rel=1e-6)
    assert column.component_moles[1] == 0.0 and state.to_dict() == before
    result = {
        "state": end.to_dict(),
        "steps": [raw],
        "openings": commands,
        "loss_coefficients": {},
        "total_heat_j": raw["total_heat_j"],
        "total_work_j": raw["total_work_j"],
        "conservation_residuals": raw["conservation_residuals"],
        "material_component_moles": raw["material_component_moles"],
        "material_energy_j": raw["material_energy_j"],
        "open_model_id": m.model_id,
        "port_openings": {"boundary": 1.0},
        "boundary_component_moles": raw["material_component_moles"],
        "boundary_energy_j": raw["material_energy_j"],
    }
    checked = audit(m.to_dict(), before, result)
    assert checked["accepted"]
    (tmp_path / f"species-{device}.json").write_text(
        json.dumps(
            {
                "model": m.to_dict(),
                "initial": before,
                "result": result,
                "audit": checked,
                "qualification": "Single nonadaptive implicit physical step. New component accumulation matches actual finite-aperture flux, not the guessed inventory. No temporal-accuracy certificate for a 0.01 s transient.",
            },
            indent=2,
        )
        + "\n"
    )
