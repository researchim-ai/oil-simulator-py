"""Reservoir total-head traces participate in the common physical Jacobian."""

import json
import math

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)


def setup(reverse=False, dense_limit=128, reservoir_side="source"):
    f = binary()
    d = 0.01
    nodes = [
        PREquilibriumNode("reservoir", 0.01),
        PREquilibriumNode(
            "pipe", math.pi * d * d / 4 * 0.1, "pipe_cell", length_m=0.1, diameter_m=d
        ),
    ]
    link = (
        PREquilibriumLink("port", "reservoir", "pipe", d, target_port="left")
        if reservoir_side == "source"
        else PREquilibriumLink("port", "pipe", "reservoir", d, source_port="right")
    )
    m = PREquilibriumTransportNetwork(
        f,
        nodes,
        [link],
        wave_fan="local_single_phase",
        reservoir_ports="stagnation_single_phase",
        dense_limit=dense_limit,
    )
    s = m.initialize(
        {
            "reservoir": {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]},
            "pipe": {
                "pressure_pa": 1010000.0 if reverse else 990000.0,
                "temperature_k": 320.0,
                "mole_fractions": [0.5, 0.5],
            },
        }
    )
    return m, s


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dense_limit", [1, 128])
@pytest.mark.parametrize("reservoir_side", ["source", "target"])
def test_total_head_and_receiving_mixing_are_direction_consistent(
    reverse, dense_limit, reservoir_side
):
    m, s = setup(reverse, dense_limit, reservoir_side)
    out, r = m._step(s, 1e-6, {"port": True}, [0.0] * 2, [320.0] * 2)
    trace = r["reservoir_traces"]["port"][reservoir_side]
    if reverse:
        assert trace["mode"] == "mixed_inflow"
        assert trace["outward_velocity_m_s"] == 0.0
    else:
        assert trace["mode"] == "reversible_outflow"
        assert trace["trace_entropy_j_mol_k"] == pytest.approx(
            trace["reservoir_entropy_j_mol_k"], abs=1e-7
        )
        assert trace["trace_total_enthalpy_j_mol"] == pytest.approx(
            trace["reservoir_total_enthalpy_j_mol"], abs=1e-5
        )
        assert trace["outward_velocity_m_s"] > 0.0
        # No numerical inlet loss: physical source enthalpy is transported.
        flux = r["links"]["port"]["flux"]
        molar = sum(flux[:-2])
        assert flux[-1] / molar == pytest.approx(
            trace["reservoir_total_enthalpy_j_mol"], rel=1e-7, abs=1e-5
        )
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    assert max(map(abs, trace["constraint_residuals"])) < 1e-8
    restored = PREquilibriumTransportNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(PREquilibriumTransportState.from_dict(json.loads(json.dumps(out.to_dict()))))


def test_stagnation_trace_cross_blocks_and_jacobian_are_not_frozen(monkeypatch):
    m, s = setup()
    seeds = [m._seed(i, v) for i, (_, v) in enumerate(s.nodes)]
    y, _, _, evaluate, offsets, _, _ = m._problem(s, seeds, 1e-6, {"port": True}, [0.0] * 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("Flash nested in prepared stagnation residual")

    monkeypatch.setattr(type(m.fluid), "flash_pt", forbidden)
    monkeypatch.setattr(type(m.fluid), "flash_uv", forbidden)
    # Differentiate on a fixed outflow branch, away from the reversal switch.
    y = y.clone()
    y[-1] = 0.001
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-0.001, 0.001, len(y), dtype=y.dtype)
    eps = 1e-5
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert float((jac @ direction - fd).abs().max()) < 1e-6
    assert len(y) == offsets[-1] + 3
    assert bool((jac[: offsets[-1], offsets[-1] :].abs() > 1e-10).any())
    pattern = m._sparsity(offsets, {"port": True}, evaluate.reservoir_traces.entries)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    mask[pattern[0], pattern[1]] = True
    if bool((~mask).any()):
        assert float(jac[~mask].abs().max()) == 0.0


@pytest.mark.parametrize("dense_limit", [1, 128])
def test_gravity_column_pipe_and_reservoir_share_trace_unknowns(dense_limit, monkeypatch):
    from test_surface_pr_gravity_transport import setup as gravity_setup

    from surface.pr_gravity_transport import PRGravityTransportNetwork

    base, state, commands = gravity_setup()
    pipe = base.transport.nodes[0]
    sink = PREquilibriumNode("sink", 0.01, elevation_m=pipe.elevation_m)
    link = PREquilibriumLink(
        "drain", pipe.name, sink.name, pipe.diameter_m, source_port="right"
    )
    transport = PREquilibriumTransportNetwork(
        base.transport.fluid,
        [pipe, sink],
        [link],
        gravity_m_s2=base.transport.gravity_m_s2,
        reservoir_ports="stagnation_single_phase",
        wave_fan="local_single_phase",
        dense_limit=dense_limit,
    )
    ts = transport.initialize(
        {
            name: {
                "pressure_pa": 990000.0 if name == "pipe" else 989900.0,
                "temperature_k": 320.0,
                "mole_fractions": [0.5, 0.5],
            }
            for name in transport.names
        }
    )
    model = PRGravityTransportNetwork(transport, base.columns, base.ports)
    old = model.initialize(ts, dict(state.columns))
    commands["drain"] = True
    new, report = model._step(old, 1e-7, commands, {}, {}, {})
    assert new.transport.time_s == 1e-7
    trace = report["transport"]["reservoir_traces"]["drain"]["target"]
    assert trace["mode"] == "mixed_inflow"
    assert max(map(abs, trace["constraint_residuals"])) < 1e-8
    assert max(map(abs, report["conservation_residuals"])) < 1e-6
    if dense_limit == 128:
        import copy
        from pathlib import Path

        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        from check_pr_gravity_transport import audit
        from run_pr_gravity_transport import run

        config = json.loads(json.dumps({
            "model": model.to_dict(),
            "initial": {
                "transport": {
                    name: {
                        "pressure_pa": 990000.0 if name == "pipe" else 989900.0,
                        "temperature_k": 320.0,
                        "mole_fractions": [0.5, 0.5],
                    }
                    for name in transport.names
                },
                "columns": {
                    "separator": {
                        "pressure_pa": 1e6,
                        "temperature_k": 320.0,
                        "mole_fractions": [0.5, 0.5],
                    }
                },
                "wall_temperatures": {},
            },
            "schedule": [{"duration_s": 1e-8}],
        }))
        result = json.loads(json.dumps(run(config)))
        assert audit(config, result)["accepted"]
        broken = copy.deepcopy(result)
        broken["intervals"][0]["steps"][0]["transport"]["reservoir_traces"]["drain"][
            "target"
        ]["trace_total_enthalpy_j_mol"] += 1.0
        with pytest.raises(ValueError):
            audit(config, broken)


def test_independent_audit_includes_total_head_and_contact_constraint(monkeypatch):
    import copy
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    runner = load("run_pr_equilibrium_transport")
    checker = load("check_pr_equilibrium_transport")
    m, _ = setup()
    cfg = {
        "model": m.to_dict(),
        "initial": {
            "nodes": {
                "reservoir": {
                    "pressure_pa": 1e6,
                    "temperature_k": 320.0,
                    "mole_fractions": [0.5, 0.5],
                },
                "pipe": {
                    "pressure_pa": 990000.0,
                    "temperature_k": 320.0,
                    "mole_fractions": [0.5, 0.5],
                },
            }
        },
        "schedule": [{"duration_s": 1e-6}],
    }
    cfg = json.loads(json.dumps(cfg))
    report = json.loads(json.dumps(runner.run(cfg)))
    assert checker.audit(cfg, report)["accepted"]
    broken = copy.deepcopy(report)
    broken["intervals"][0]["steps"][0]["reservoir_traces"]["port"]["source"][
        "trace_total_enthalpy_j_mol"
    ] += 1.0
    with pytest.raises(ValueError):
        checker.audit(cfg, broken)
