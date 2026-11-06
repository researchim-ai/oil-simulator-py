"""Shared junctions must exchange with every native branch in one solve."""

import json
from dataclasses import replace

import pytest
from test_surface_pr_layered_transport import model as profile_model

from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork
from surface.pr_profile_graph import ProfileGraphState, PRProfileGraphNetwork


def setup(*, sparse=False, count=2):
    pipe, spec = profile_model(cells=1)
    native = pipe.initialize_equilibrium(**spec)
    left = pipe._disk_bottom(0, -1) + pipe.diameter * pipe.cosine / 2
    right = pipe._disk_bottom(0, 1) + pipe.diameter * pipe.cosine / 2
    t = PREquilibriumTransportNetwork(
        pipe.fluid,
        [
            PREquilibriumNode("a", 0.001, elevation_m=left),
            PREquilibriumNode("b", 0.001, elevation_m=right),
        ],
        gravity_m_s2=pipe.gravity,
        wave_speed_bound_m_s=pipe.alpha,
        dense_limit=1 if sparse else 768,
        initial_step_s=1e-6,
        maximum_step_s=1e-4,
    )
    ordinary = t.initialize(
        {
            n: {
                "pressure_pa": 1.005e6 if n == "a" else 0.995e6,
                "temperature_k": 196.2,
                "mole_fractions": [0.19, 0.81],
            }
            for n in t.names
        }
    )
    names = [f"line_{i}" for i in range(count)]
    model = PRProfileGraphNetwork(t, {n: (pipe, ("a", "b")) for n in names})
    return model, model.initialize(ordinary, dict.fromkeys(names, native))


@pytest.fixture(scope="module")
def parallel():
    return setup()


def test_parallel_shared_junctions_and_restart(parallel):
    m, s = parallel
    before = s.to_dict()
    end, report = m.step(s, 1e-6)
    assert s.to_dict() == before
    assert max(map(abs, report["conservation_residuals"])) < 1e-5
    profiles = dict(end.profiles)
    assert profiles["line_0"].cells[0].momentum_kg_m_s > 0
    a, b = profiles["line_0"].cells[0], profiles["line_1"].cells[0]
    assert a.momentum_kg_m_s == pytest.approx(b.momentum_kg_m_s, abs=1e-15)
    assert a.total_energy_j == pytest.approx(b.total_energy_j, abs=1e-8)
    assert a.profile.component_moles == pytest.approx(b.profile.component_moles, abs=1e-12)
    restored = PRProfileGraphNetwork.from_dict(json.loads(json.dumps(m.to_dict())))
    restored.inspect(ProfileGraphState.from_dict(json.loads(json.dumps(end.to_dict()))))
    _, follow = restored.step(end, 1e-7)
    assert max(map(abs, follow["conservation_residuals"])) < 1e-5


def test_isolation_and_reopening_keep_finite_stocks(parallel):
    m, s = parallel
    isolated, r = m.step(s, 1e-6, profile_openings={"line_1": [False, False]})
    initial, final = dict(s.profiles)["line_1"], dict(isolated.profiles)["line_1"]
    assert final.cells[0].profile.component_moles == pytest.approx(
        initial.cells[0].profile.component_moles, abs=1e-10
    )
    assert final.cells[0].total_energy_j == pytest.approx(initial.cells[0].total_energy_j, abs=1e-7)
    assert r["boundary_nodes"]["line_1"] == (None, None)
    reopened, _ = m.step(isolated, 1e-6, profile_openings={"line_1": [True, True]})
    assert dict(reopened.profiles)["line_1"].cells[0].momentum_kg_m_s > 0


def test_sparse_parallel_solution_and_last_allowed_attempt(parallel):
    m, s = parallel
    raw = m.to_dict()
    raw["transport"]["dense_limit"] = 1
    sparse = PRProfileGraphNetwork.from_dict(json.loads(json.dumps(raw)))
    ss = sparse.initialize(
        # The material stocks do not depend on linear-solver policy.
        replace(s.transport, model_id=sparse.transport.model_id),
        dict(s.profiles),
    )
    a, _ = m.step(s, 1e-7)
    b, _ = sparse.step(ss, 1e-7)
    for (_, x), (_, y) in zip(a.profiles, b.profiles):
        assert x.cells[0].momentum_kg_m_s == pytest.approx(
            y.cells[0].momentum_kg_m_s, abs=1e-10, rel=1e-5
        )
    report = sparse.advance(ss, 1e-7, maximum_attempts=1)
    end = ProfileGraphState.from_dict(report["state"])
    assert end.time_s == 1e-7
    assert all(p.time_s == end.time_s for _, p in end.profiles)


def test_registry_and_endpoint_commands_reject_atomically(parallel):
    m, s = parallel
    before = s.to_dict()
    with pytest.raises(ValueError):
        m.step(s, 1e-6, profile_openings={"missing": [True, True]})
    with pytest.raises(ValueError):
        m.step(s, 1e-6, profile_openings={"line_0": [1, True]})
    with pytest.raises(ValueError):
        m.initialize(s.transport, {"line_0": s.profiles[0][1]})
    assert s.to_dict() == before


def test_one_ordinary_assembly_and_exact_sparse_cross_support(parallel, monkeypatch):
    import torch

    from surface.pr_profile_graph import _ProfileGraphCoupling

    original, state = parallel
    raw = json.loads(json.dumps(original.to_dict()))
    raw["transport"]["dense_limit"] = 1
    m = PRProfileGraphNetwork.from_dict(raw)
    s = m.initialize(replace(state.transport, model_id=m.transport.model_id), dict(state.profiles))
    observed = {}
    calls = []
    original_problem = m.transport._problem
    original_prepare = _ProfileGraphCoupling.prepare

    def problem(*args, **kwargs):
        parts = list(original_problem(*args, **kwargs))
        base = parts[3]

        def counted(*args, **kwargs):
            calls.append(1)
            return base(*args, **kwargs)

        counted.__dict__.update(base.__dict__)
        parts[3] = counted
        return tuple(parts)

    def prepare(self, *args, **kwargs):
        result = original_prepare(self, *args, **kwargs)
        observed.update(y=result[0], evaluate=result[3], slices=dict(self.slices), cut=self.cut)
        # A dense solve still publishes the structural sparse pattern through
        # the native blocks. Request the sparse policy for this regression.
        observed["pattern"] = result[4]
        return result

    monkeypatch.setattr(m.transport, "_problem", problem)
    monkeypatch.setattr(_ProfileGraphCoupling, "prepare", prepare)
    m.step(s, 1e-7)
    evaluate, y = observed["evaluate"], observed["y"]
    count = len(calls)
    evaluate(y)
    assert len(calls) == count + 1
    jac = torch.func.jacrev(lambda z: evaluate(z)[0])(y)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    rows, cols = observed["pattern"]
    mask[rows, cols] = True
    assert float(jac[~mask].abs().max()) < 1e-12
    # The same junction rows must depend on each branch, not just the last one.
    for sl in observed["slices"].values():
        block = jac[: observed["cut"], sl]
        index = int(block.abs().argmax())
        row, local = divmod(index, block.shape[1])
        col = sl.start + local
        assert abs(float(jac[row, col])) > 1e-12
        delta = torch.zeros_like(y)
        h = 1e-5 * max(1.0, abs(float(y[col])))
        delta[col] = h
        finite_difference = (evaluate(y + delta)[0][row] - evaluate(y - delta)[0][row]) / (2 * h)
        assert float(jac[row, col]) == pytest.approx(float(finite_difference), rel=1e-4, abs=1e-10)


@pytest.fixture(scope="module")
def graph_report():
    from pathlib import Path

    from run_pr_profile_graph import run

    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_pr_profile_graph.json").read_text()
    )
    return cfg, json.loads(json.dumps(run(cfg)))


def test_independent_graph_audit_and_tampering(graph_report, monkeypatch):
    from check_pr_profile_graph import audit

    from surface.pr_layered_transport import PRLayeredTransportNetwork

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent audit called a runtime initializer")

    monkeypatch.setattr(PRProfileGraphNetwork, "initialize", forbidden)
    monkeypatch.setattr(PRLayeredTransportNetwork, "initialize_equilibrium", forbidden)
    cfg, report = graph_report
    assert audit(cfg, report)["accepted"]
    damaged = json.loads(json.dumps(report))
    damaged["intervals"][0]["steps"][0]["profiles"]["line_0"]["vessel_support_force_n"][0] += 1.0
    with pytest.raises(ValueError):
        audit(cfg, damaged)
    changed = json.loads(json.dumps(cfg))
    changed["initial"]["profiles"]["line_1"]["pressure_pa"] *= 1.1
    damaged = json.loads(json.dumps(report))
    damaged["config"] = changed
    with pytest.raises(ValueError):
        audit(changed, damaged)
    damaged = json.loads(json.dumps(report))
    damaged["intervals"][0]["steps"][0]["boundary_nodes"]["line_1"][0] = "a"
    with pytest.raises(ValueError):
        audit(cfg, damaged)


def test_complete_graph_runner_restart(graph_report):
    from check_pr_profile_graph import audit
    from run_pr_profile_graph import run

    cfg, report = graph_report
    cfg = {**cfg, "initial": {"state": report["state"]}, "schedule": [{"duration_s": 1e-8}]}
    continued = json.loads(json.dumps(run(cfg)))
    assert continued["initial"] == report["state"]
    assert audit(cfg, continued)["accepted"]


def test_profile_ring_has_both_flow_directions_and_conserves():
    pipe, initial = profile_model(cells=1, angle=0.0)
    native = pipe.initialize_equilibrium(**initial)
    center = pipe.diameter / 2
    t = PREquilibriumTransportNetwork(
        pipe.fluid,
        [PREquilibriumNode(n, 0.001, elevation_m=center) for n in ("a", "b", "c")],
        gravity_m_s2=pipe.gravity,
        wave_speed_bound_m_s=pipe.alpha,
        dense_limit=1,
        initial_step_s=1e-6,
    )
    ordinary = t.initialize(
        {
            n: {"pressure_pa": p, "temperature_k": 196.2, "mole_fractions": [0.19, 0.81]}
            for n, p in zip(t.names, (1.005e6, 0.995e6, 1e6))
        }
    )
    m = PRProfileGraphNetwork(
        t,
        {
            "ab": (pipe, ("a", "b")),
            "bc": (pipe, ("b", "c")),
            "ca": (pipe, ("c", "a")),
        },
    )
    state = m.initialize(ordinary, dict.fromkeys(m.names, native))
    end, report = m.step(state, 1e-7)
    assert max(map(abs, report["conservation_residuals"])) < 1e-5
    momentum = {n: s.cells[0].momentum_kg_m_s for n, s in end.profiles}
    assert momentum["ab"] > 0 and momentum["bc"] < 0 and momentum["ca"] < 0
    assert report["entropy_production_j_k"] >= -1e-6
