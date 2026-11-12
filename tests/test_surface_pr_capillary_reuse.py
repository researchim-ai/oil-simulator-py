"""Existing capillary physics is the accounting authority in shared graphs."""

import json
import math

import pytest
import torch
from check_pr_capillary_facility import audit as capillary_audit
from check_pr_open_gravity_transport import audit as open_audit
from test_surface_pr_capillary_facility import setup

from surface.component_energy import ComponentEnergyStream
from surface.pr_capillary_facility import PRCapillaryFacility, PRCapillaryFacilityState
from surface.pr_capillary_surface_plant import PRCapillarySurfacePlant, PRCapillarySurfaceState
from surface.pr_equilibrium_transport import PREquilibriumNode
from surface.pr_open_gravity_transport import PROpenGravityTransportNetwork
from surface.pr_open_transport import PRTransportFeed, PRTransportPressurePort
from surface.pr_surface_builder import PRColumnTerminal, build_pr_surface_network
from surface.pr_transport_builder import PREquilibriumPipe, initialize_grouped


def shared_case(device="cpu", **transport_options):
    capillary, old = setup(device=device, self_loop=True)
    column = capillary.columns["A"]
    fluid = column.fluid
    cs = dict(old.columns)["A"]
    height = 0.05
    z = column.elevation_m + height
    graph = build_pr_surface_network(
        fluid,
        [PREquilibriumNode(n, 0.1, elevation_m=z) for n in ("inlet", "export")],
        [
            PREquilibriumPipe("supply", "inlet", "pickup", 0.1, 0.0005, 2),
            PREquilibriumPipe("delivery", "pickup2", "export", 0.1, 0.0005, 2),
        ],
        columns=capillary.columns,
        column_terminals=[PRColumnTerminal(n, "A", height) for n in ("pickup", "pickup2")],
        initial_step_s=1e-7,
        maximum_step_s=1e-5,
        dense_limit=768,
        **transport_options,
    )
    local = column.sample(cs, height, side="below")
    spec = {
        "pressure_pa": local["pressure_pa"],
        "temperature_k": cs.temperature_k,
        "mole_fractions": local["mole_fractions"],
    }
    initial = graph.initialize(
        initialize_grouped(
            graph.transport, {n: spec for n in ("inlet", "export", "supply", "delivery")}
        ),
        {"A": cs},
    )
    stream = ComponentEnergyStream.from_pt(
        fluid,
        [0.001, 0.002],
        1.01e6,
        197.0,
        elevation_m=z,
        gravity_m_s2=graph.transport.gravity_m_s2,
    )
    opened = PROpenGravityTransportNetwork(
        graph,
        [PRTransportFeed("feed", "inlet", stream)],
        [PRTransportPressurePort("boundary", "export", 0.99e6, 196.2, (0.5, 0.5), 1e-10)],
    )
    plant = PRCapillarySurfacePlant(opened, capillary)
    return plant, plant.initialize(initial, old)


def test_shared_stocks_exclude_columns_and_transform_the_whole_energy_datum():
    model, state = setup(self_loop=True)
    full = model.conserved_terms(state)
    pipes = model.conserved_terms(state, include_columns=False)
    cs = dict(state.columns)["A"]
    for j, amount in enumerate(cs.component_moles):
        assert math.fsum(full[j]) == math.fsum((*pipes[j], amount))
    assert math.fsum(full[-1]) == math.fsum((*pipes[-1], cs.total_energy_j))
    shifted = model.conserved_terms(state, reference_elevation_m=11.0)
    fluid = model.columns["A"].fluid
    mass = math.fsum(math.fsum(ns) * c.molar_mass_kg_mol for ns, c in zip(full, fluid.components))
    expected = -mass * model.columns["A"].gravity_m_s2 * 11.0
    assert math.fsum(shifted[-1]) - math.fsum(full[-1]) == pytest.approx(expected, abs=2e-11)
    with pytest.raises(ValueError, match="inclusion switch"):
        model.conserved_terms(state, include_columns=1)
    with pytest.raises(ValueError):
        model.conserved_terms(state, reference_elevation_m=math.nan)


def test_shared_column_rebinding_uses_existing_phase_appearance_and_restart():
    model, old = setup(self_loop=True)
    snapshot = old.to_dict()
    hot, _ = model.step(old, 1e-7, heaters={"A": {"power_w": 1e11, "temperature_k": 450}})
    rebound = model.rebind_columns(old, dict(hot.columns))
    assert dict(rebound.layer_slots)["A"] == ("gas",)
    assert rebound.pipes == old.pipes and rebound.time_s == old.time_s
    assert all(set(m.boundaries[0]) == {"gas"} for m in model.inspect(rebound).values())
    cold = model.rebind_columns(rebound, dict(old.columns))
    assert cold == old
    restored = PRCapillaryFacility.from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PRCapillaryFacilityState.from_dict(json.loads(json.dumps(rebound.to_dict())))
    assert restored.rebind_columns(checkpoint, dict(old.columns)) == old
    with pytest.raises(ValueError, match="Exactly"):
        model.rebind_columns(old, {})
    assert old.to_dict() == snapshot


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_open_graph_reuses_checked_suboperators_without_duplicate_column_stock(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    plant, old = shared_case(device)
    snapshot = old.to_dict()
    end, report = plant.step(old, 1e-7)
    assert old.to_dict() == snapshot
    assert end.time_s == 1e-7 and len(end.columns) == 1
    cap_old = plant.capillary_state(old)
    cap_end = PRCapillaryFacilityState.from_dict(report["capillary_end"])
    assert capillary_audit(plant.capillary.to_dict(), cap_old, cap_end, report["capillary"])[
        "accepted"
    ]
    raw = report["surface"]
    surface_result = {
        **raw,
        "steps": [raw],
        "rejected": [],
        "openings": report["openings"],
        "open_model_id": plant.surface.model_id,
        "port_openings": report["port_openings"],
        "boundary_component_moles": raw["material_component_moles"],
        "boundary_energy_j": raw["material_energy_j"],
    }
    assert open_audit(
        plant.surface.to_dict(),
        report["midpoint"],
        surface_result,
        openings=report["openings"],
        column_work_j={n: -v for n, v in report["geometry_work_j"].items()},
    )["accepted"]
    # Authoritative stock API supplies only capillary pipes to the host.
    terms = plant.capillary.conserved_terms(plant.capillary_state(end), include_columns=False)
    assert plant.capillary_stocks(end) == ([math.fsum(v) for v in terms[:-1]], math.fsum(terms[-1]))
    assert all(
        abs(r) <= t for r, t in zip(report["conservation_residuals"], report["roundoff_allowances"])
    )
    assert all(v > 0 for v in raw["material_component_moles"])
    restored = PRCapillarySurfacePlant.from_dict(
        json.loads(json.dumps(plant.to_dict())), device=device
    )
    checkpoint = PRCapillarySurfaceState.from_dict(json.loads(json.dumps(old.to_dict())))
    repeated, _ = restored.step(checkpoint, 1e-7)
    assert repeated == end
