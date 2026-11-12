"""Actual open distributed pipes, passive boundaries and independent ledgers."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from test_surface_compositional_separator import binary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_pr_open_transport import audit

from surface.component_energy import ComponentEnergyStream
from surface.pr_equilibrium_transport import (
    PREquilibriumNode,
    PREquilibriumTransportState,
)
from surface.pr_open_transport import (
    PROpenTransportNetwork,
    PRTransportFeed,
    PRTransportPressurePort,
    _BoundaryProperties,
    _OpenCoupling,
)
from surface.pr_transport_builder import (
    PREquilibriumPipe,
    build_equilibrium_transport,
    initialize_grouped,
)
from surface.stratified_pr_flow import ReceiverFeedPressureError

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def build(device="cpu", *, temperature=320.0, reverse=False, parallel=False, sparse=False):
    f = binary(device)
    nodes = [PREquilibriumNode("inlet", 0.02), PREquilibriumNode("export", 0.02)]
    pipes = [PREquilibriumPipe("line", "inlet", "export", 1.0, 0.02, 2)]
    if parallel:
        pipes.append(PREquilibriumPipe("parallel", "inlet", "export", 1.0, 0.015, 2))
    t = build_equilibrium_transport(
        f, nodes, pipes, initial_step_s=1e-5, maximum_step_s=1e-4, dense_limit=1 if sparse else 768
    )
    spec = {
        "pressure_pa": 1e6,
        "temperature_k": temperature,
        "mole_fractions": [0.5, 0.5],
        "velocity_m_s": 0.0,
    }
    state = initialize_grouped(t, {n: spec for n in ["inlet", "export", *[p.name for p in pipes]]})
    stream = ComponentEnergyStream.from_pt(
        f, [0.01, 0.02], 1.01e6, temperature + 1, gravity_m_s2=t.gravity_m_s2
    )
    model = PROpenTransportNetwork(
        t,
        [PRTransportFeed("feed", "inlet", stream)],
        [
            PRTransportPressurePort(
                "boundary",
                "export",
                1.01e6 if reverse else 0.99e6,
                temperature,
                (0.6, 0.4),
                1e-9,
                allow_reverse=reverse,
            )
        ],
    )
    return model, state


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reverse", [False, True])
def test_open_pipe_faces_and_independent_scalar_balances(device, reverse, tmp_path):
    m, s = build(device, reverse=reverse, parallel=True)
    before = s.to_dict()
    r = m.advance(s, 1e-5)
    check = audit(m.to_dict(), before, r)
    assert check["accepted"]
    assert all(
        step["coupling"]["ports"]["boundary"]["volume_rate_m3_s"] < 0
        if reverse
        else step["coupling"]["ports"]["boundary"]["volume_rate_m3_s"] > 0
        for step in r["steps"]
    )
    assert s.to_dict() == before
    assert any(
        abs(c["momentum_kg_m_s"]) > 0
        for n, c in r["state"]["nodes"].items()
        if n.startswith("line__")
    )
    bad = copy.deepcopy(r)
    bad["steps"][0]["coupling"]["ports"]["boundary"]["total_energy_rate_w"] *= 1.1
    with pytest.raises(ValueError):
        audit(m.to_dict(), before, bad)
    (tmp_path / f"open-{device}-{reverse}.json").write_text(
        json.dumps({"model": m.to_dict(), "initial": before, "result": r, "audit": check}, indent=2)
        + "\n"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_two_phase_restart_and_exact_external_isolation(device):
    m, s = build(device, temperature=196.2)
    assert any(len(v.phase_roots) == 2 for _, v in s.nodes)
    r = m.advance(s, 1e-5, port_openings={"boundary": 0.0})
    assert audit(m.to_dict(), s.to_dict(), r, port_openings={"boundary": 0.0})["accepted"]
    assert all(
        step["coupling"]["ports"]["boundary"]["volume_rate_m3_s"] == 0 for step in r["steps"]
    )
    saved = PREquilibriumTransportState.from_dict(json.loads(json.dumps(r["state"])))
    copied = PROpenTransportNetwork.from_dict(json.loads(json.dumps(m.to_dict())), device=device)
    a = m.advance(saved, 1e-5)
    b = copied.advance(
        PREquilibriumTransportState.from_dict(json.loads(json.dumps(saved.to_dict()))), 1e-5
    )
    assert a == b
    assert audit(m.to_dict(), saved.to_dict(), a)["accepted"]


@pytest.mark.parametrize("reverse", [False, True])
def test_boundary_jacobian_contains_pressure_temperature_and_composition_derivatives(reverse):
    m, s = build(reverse=reverse)
    t = m.transport
    prepared = _BoundaryProperties(m, {"boundary": 1.0})
    c = _OpenCoupling(m, prepared, s, 1e-5, t.default_openings)
    seeds = [t._seed(i, v) for i, (_, v) in enumerate(s.nodes)]
    y, lo, hi, base, offsets, _, _ = t._problem(
        s, seeds, 1e-5, t.default_openings, [0.0] * len(t.names)
    )
    y, _, _, ev, _ = c.prepare(y, lo, hi, base, offsets, 1e-5, None)
    jac = torch.func.jacfwd(lambda v: ev(v)[0])(y)
    direction = torch.linspace(-0.01, 0.01, y.numel(), dtype=y.dtype)
    eps = 1e-5
    fd = (ev(y + eps * direction)[0] - ev(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-6


def test_sparse_dense_open_graph_physical_state_matches():
    m, s = build()
    n, u = build(sparse=True)
    a = m.advance(s, 1e-5)
    b = n.advance(u, 1e-5)
    for name, v in a["state"]["nodes"].items():
        w = b["state"]["nodes"][name]
        assert v["pressure_pa"] == pytest.approx(w["pressure_pa"], abs=1e-5)
        assert v["component_moles"] == pytest.approx(w["component_moles"], abs=1e-9)
    assert audit(n.to_dict(), u.to_dict(), b)["accepted"]


def test_command_identity_pressure_and_geometry_errors_leave_input_unchanged():
    m, s = build()
    before = s.to_dict()
    with pytest.raises(ValueError, match="Unknown external"):
        m.advance(s, 1e-5, port_openings={"typo": 0.0})
    packet = ComponentEnergyStream.from_pt(
        m.fluid, [0.01, 0.02], 0.99e6, 321, gravity_m_s2=m.transport.gravity_m_s2
    )
    bad = PROpenTransportNetwork(m.transport, [PRTransportFeed("feed", "inlet", packet)])
    with pytest.raises(ReceiverFeedPressureError, match="pressure"):
        bad.advance(s, 1e-5)
    assert s.to_dict() == before
    with pytest.raises(ValueError, match="supported stirred"):
        PROpenTransportNetwork(
            m.transport,
            pressure_ports=[
                PRTransportPressurePort("bad", "line__0000", 1e6, 320, (0.5, 0.5), 1e-9)
            ],
        )


@pytest.mark.parametrize("kind", ["pump", "compressor"])
def test_open_mapped_machine_rotor_and_material_boundary_share_energy_ledger(kind):
    from test_surface_pr_machine_drive import setup

    t, s, _ = setup(kind)
    inlet = "feed" if kind == "pump" else "suction"
    outlet = "return" if kind == "pump" else "delivery"
    si = dict(s.nodes)[inlet]
    so = dict(s.nodes)[outlet]
    fractions = [v / sum(si.component_moles) for v in si.component_moles]
    stream = ComponentEnergyStream.from_pt(
        t.fluid,
        [1e-4 * x for x in fractions],
        si.pressure_pa * 1.01,
        si.temperature_k,
        gravity_m_s2=t.gravity_m_s2,
    )
    m = PROpenTransportNetwork(
        t,
        [PRTransportFeed("supply", inlet, stream)],
        [
            PRTransportPressurePort(
                "export", outlet, so.pressure_pa * 0.99, so.temperature_k, tuple(fractions), 1e-10
            )
        ],
    )
    r = m.advance(s, 1e-6)
    assert r["total_work_j"] > 0
    assert audit(m.to_dict(), s.to_dict(), r)["accepted"]
    machine = t.drives[0].machine
    assert all(step["drives"][machine]["shaft_work_j"] > 0 for step in r["steps"])
    stopped = m.advance(
        s,
        1e-6,
        openings={name: False for name in t.default_openings},
        port_openings={"export": 0.0},
    )
    assert all(step["drives"][machine]["shaft_work_j"] == 0 for step in stopped["steps"])
    assert audit(
        m.to_dict(),
        s.to_dict(),
        stopped,
        openings={name: False for name in t.default_openings},
        port_openings={"export": 0.0},
    )["accepted"]


@pytest.mark.parametrize("mode", ["feed", "reverse"])
def test_absent_component_enters_only_through_real_accepted_boundary(mode):
    from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork

    f = binary()
    t = PREquilibriumTransportNetwork(
        f, [PREquilibriumNode("v", 0.02)], initial_step_s=1e-4, maximum_step_s=1e-4
    )
    state = t.initialize(
        {"v": {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [1.0, 0.0]}}
    )
    if mode == "feed":
        stream = ComponentEnergyStream.from_pt(
            f, [0.0, 0.01], 1.01e6, 320.0, gravity_m_s2=t.gravity_m_s2
        )
        m = PROpenTransportNetwork(t, [PRTransportFeed("new", "v", stream)])
    else:
        m = PROpenTransportNetwork(
            t,
            pressure_ports=[
                PRTransportPressurePort(
                    "new", "v", 1.01e6, 320.0, (0.0, 1.0), 1e-9, allow_reverse=True
                )
            ],
        )
    r = m.advance(state, 1e-4)
    assert r["state"]["nodes"]["v"]["component_moles"][1] > 0
    assert state.nodes[0][1].component_moles[1] == 0
    assert audit(m.to_dict(), state.to_dict(), r)["accepted"]
