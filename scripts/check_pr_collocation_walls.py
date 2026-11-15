"""Independent isochoric PR/fluid/wall/ambient dynamics; CPU verification.

The reference integrates explicit scalar PR Cv, wall heat capacities and UA
laws. No production flash/AD/transport RHS is called by its ODE.
"""

import argparse
import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from audit_pr_collocation import audit_gauss_step
from check_pr_collocation import ROOT, IsochoricReference, thermal_fixture
from scipy.integrate import solve_ivp

from surface import PRCollocation, PREquilibriumTransportNetwork, PREquilibriumTransportState
from surface.torch_backend import runtime_fingerprint
from surface.vessel_wall import VesselWall


def wall_fixture(*, dense_limit=128, insulated=False, partial=False):
    model, _ = thermal_fixture()
    walls = [
        VesselWall(40.0, 10.0, 0.0 if insulated else 3.0, 285.0, max_temperature_change_k=20.0),
        VesselWall(25.0, 8.0, 0.0 if insulated else 1.0, 380.0, max_temperature_change_k=20.0),
    ]
    nodes = [
        replace(node, wall=None if partial and i == 1 else walls[i])
        for i, node in enumerate(model.nodes)
    ]
    model = PREquilibriumTransportNetwork(
        model.fluid,
        nodes,
        heat_links=model.heat_links,
        initial_step_s=0.02,
        maximum_step_s=0.25,
        minimum_step_s=1e-8,
        relative_tolerance=1e-7,
        temperature_tolerance_k=1e-5,
        dense_limit=dense_limit,
    )
    initial = {
        node.name: {
            "pressure_pa": 1e6,
            "temperature_k": [320.0, 360.0][i],
            "mole_fractions": [1.0, 0.0],
            **({"wall_temperature_k": [350.0, 300.0][i]} if node.wall else {}),
        }
        for i, node in enumerate(model.nodes)
    }
    state = model.initialize(initial)
    return model, state


class WallReference(IsochoricReference):
    def __init__(self, model, state):
        super().__init__(model, state)
        self.walls = [n.wall for n in model.nodes]
        self.tref = model.fluid.components[0].reference_temperature_k

    def rhs(self, _time, y):
        fluid = y[:2]
        wall = y[2:4]
        contact = self.ua * (fluid[0] - fluid[1])
        qh = np.array([w.fluid_ua_w_k * (wall[i] - fluid[i]) for i, w in enumerate(self.walls)])
        qa = np.array(
            [
                w.ambient_ua_w_k * (w.ambient_temperature_k - wall[i])
                for i, w in enumerate(self.walls)
            ]
        )
        return np.r_[
            (np.array([-contact, contact]) + qh) / (self.n * self.cv(fluid)),
            (qa - qh) / [w.heat_capacity_j_k for w in self.walls],
            qa.sum(),
        ]

    def fluid_energy(self, t):
        tr = np.sqrt(t / self.tc)
        alpha = 1 + self.k * (1 - tr)
        attraction = self.a0 * alpha**2
        derivative = -self.a0 * self.k * alpha / (self.tc * tr)
        return self.n * (
            self.cp * (t - self.tref)
            - 8.31446261815324 * t
            + (t * derivative - attraction) * self.log / (2 * np.sqrt(2) * self.b)
        )


def check():
    model, state = wall_fixture()
    ref = WallReference(model, state)
    initial = np.array(
        [s.temperature_k for _, s in state.nodes]
        + [s.wall_temperature_k for _, s in state.nodes]
        + [0.0]
    )
    solution = solve_ivp(
        ref.rhs, (0.0, 0.5), initial, method="DOP853", rtol=1e-13, atol=1e-12, dense_output=True
    )
    tighter = solve_ivp(
        ref.rhs, (0.0, 0.5), initial, method="DOP853", rtol=3e-14, atol=1e-13, dense_output=True
    )
    assert solution.success and tighter.success
    refinement = float(
        abs(solution.sol(np.linspace(0, 0.5, 51)) - tighter.sol(np.linspace(0, 0.5, 51))).max()
    )
    assert refinement < 1e-8
    fixed = []
    for dt in (0.25, 0.125, 0.0625):
        solver = PRCollocation(model, phase_fallback=False)
        current = state
        for _ in range(round(0.5 / dt)):
            before = current
            current, row = solver._step(
                current, dt, model.default_openings, [0.0, 0.0], [320.0, 360.0], {}
            )
            audit_gauss_step(
                model.to_dict(),
                before.to_dict(),
                row,
                model.fluid,
                {"openings": model.default_openings},
            )
        actual = np.array(
            [s.temperature_k for _, s in current.nodes]
            + [s.wall_temperature_k for _, s in current.nodes]
        )
        fixed.append(
            {
                "dt_s": dt,
                "maximum_temperature_error_k": float(abs(actual - solution.y[:4, -1]).max()),
            }
        )
    ratios = [
        fixed[i]["maximum_temperature_error_k"] / fixed[i + 1]["maximum_temperature_error_k"]
        for i in (0, 1)
    ]
    assert min(ratios) > 14, ratios
    result = PRCollocation(model, phase_fallback=False).advance(state, 0.5)
    before = state
    max_t = max_p = max_budget = 0.0
    for row in result["steps"]:
        audit_gauss_step(
            model.to_dict(),
            before.to_dict(),
            row,
            model.fluid,
            {"openings": model.default_openings},
        )
        after = PREquilibriumTransportState.from_dict(row["state"])
        expected = solution.sol(after.time_s)
        temperatures = np.array(
            [s.temperature_k for _, s in after.nodes]
            + [s.wall_temperature_k for _, s in after.nodes]
        )
        max_t = max(max_t, float(abs(temperatures - expected[:4]).max()))
        max_p = max(
            max_p,
            float(
                abs(
                    np.array([s.pressure_pa for _, s in after.nodes]) - ref.pressure(expected[:2])
                ).max()
            ),
        )
        previous_t = np.array([s.temperature_k for _, s in before.nodes])
        fluid_delta = (ref.fluid_energy(temperatures[:2]) - ref.fluid_energy(previous_t)).sum()
        wall_delta = sum(
            node.wall.heat_capacity_j_k * (b.wall_temperature_k - a.wall_temperature_k)
            for node, (_, a), (_, b) in zip(model.nodes, before.nodes, after.nodes)
        )
        max_budget = max(max_budget, abs(fluid_delta + wall_delta - row["ambient_heat_j"]))
        before = after
    assert max_t < 1e-5 and max_p < 1.0 and max_budget < 1e-6
    assert result["fallback_intervals"] == 0
    summary = {
        "scope": "Closed pure-methane PR vessels, finite wall heat storage, thermal contact and passive ambient exchange. Synthetic CPU numerical validation only",
        "runtime": runtime_fingerprint(),
        "simulated_duration_s": 0.5,
        "fixed_step_runs": fixed,
        "observed_refinement_ratios": ratios,
        "reference_refinement_max_state_difference": refinement,
        "all_endpoint_temperature_error_k": max_t,
        "all_endpoint_pressure_error_pa": max_p,
        "independent_max_step_energy_defect_j": max_budget,
        "accepted_substeps": len(result["steps"]),
        "attempts": result["attempts"],
        "fallback_intervals": result["fallback_intervals"],
        "conservation_residuals": result["conservation_residuals"],
        "total_heat_j": result["total_heat_j"],
        "passed": True,
    }
    return summary, {
        "model": model.to_dict(),
        "initial": state.to_dict(),
        "result": result,
        "reference_initial": initial.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        ROOT / "tests/data/surface/pr_mixture_coolprop.json",
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    summary, raw = check()
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]
        for p in paths
    ):
        raise RuntimeError("Sources changed during wall validation")
    summary["source_sha256"] = hashes
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output / "trajectory.json.gz", "wt") as stream:
        json.dump(raw, stream, allow_nan=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
