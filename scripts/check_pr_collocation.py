"""Independent pure-PR isochoric heat-transfer ODE for Gauss4 validation.

The scalar reference directly implements PR alpha, Cv and pressure at fixed
molar volume. No production flash, AD, residual or transport RHS is used.
SciPy is used only by this validation reference, never by PRCollocation.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surface import PRCollocation, PREquilibriumNode, PREquilibriumTransportNetwork
from surface.equilibrium_properties import EquilibriumFluid
from surface.peng_robinson import PengRobinsonFluid
from surface.pr_heat_link import PRHeatLink

R = 8.31446261815324


def thermal_fixture(*, dense_limit=128):
    data = json.loads((ROOT / "tests/data/surface/pr_mixture_coolprop.json").read_text())
    fluid = EquilibriumFluid(
        [PengRobinsonFluid(**c) for c in data["components"]], [[0.0, 0.0], [0.0, 0.0]]
    )
    model = PREquilibriumTransportNetwork(
        fluid,
        [PREquilibriumNode("a", 0.001), PREquilibriumNode("b", 0.001)],
        heat_links=[PRHeatLink("contact", "a", "b", 5.0)],
        initial_step_s=0.05,
        maximum_step_s=0.25,
        minimum_step_s=1e-8,
        relative_tolerance=1e-7,
        temperature_tolerance_k=1e-5,
        dense_limit=dense_limit,
    )
    state = model.initialize(
        {
            name: {"pressure_pa": 1e6, "temperature_k": t, "mole_fractions": [1.0, 0.0]}
            for name, t in [("a", 320.0), ("b", 360.0)]
        }
    )
    return model, state


class IsochoricReference:
    def __init__(self, model, state):
        c = model.fluid.components[0]
        self.tc, self.pc, self.cp, self.omega = (
            c.critical_temperature_k,
            c.critical_pressure_pa,
            c.ideal_cp_j_mol_k,
            c.acentric_factor,
        )
        self.b = 0.07779607390388846 * R * self.tc / self.pc
        self.a0 = 0.4572355289213822 * R**2 * self.tc**2 / self.pc
        self.k = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        self.n = np.array([sum(s.component_moles) for _, s in state.nodes])
        self.v = np.array([node.volume_m3 / n for node, n in zip(model.nodes, self.n)])
        self.log = np.log(
            (self.v + (1 + math.sqrt(2)) * self.b) / (self.v + (1 - math.sqrt(2)) * self.b)
        )
        self.ua = 5.0

    def cv(self, t):
        a_tt = self.a0 * self.k * (1 + self.k) / (2 * self.tc**2 * (t / self.tc) ** 1.5)
        return self.cp - R + t * a_tt / (2 * math.sqrt(2) * self.b) * self.log

    def pressure(self, t):
        a = self.a0 * (1 + self.k * (1 - np.sqrt(t / self.tc))) ** 2
        return R * t / (self.v - self.b) - a / (self.v**2 + 2 * self.b * self.v - self.b**2)

    def rhs(self, _time, t):
        q = self.ua * (t[0] - t[1])
        return np.array([-q, q]) / (self.n * self.cv(t))


def check():
    model, state = thermal_fixture()
    ref = IsochoricReference(model, state)
    t0 = np.array([s.temperature_k for _, s in state.nodes])
    np.testing.assert_allclose(
        ref.pressure(t0), [s.pressure_pa for _, s in state.nodes], rtol=1e-12
    )
    reference = solve_ivp(
        ref.rhs, (0.0, 0.5), t0, method="DOP853", rtol=1e-13, atol=1e-12, dense_output=True
    )
    assert reference.success
    fixed = []
    for dt in (0.25, 0.125, 0.0625):
        solver = PRCollocation(model, phase_fallback=False)
        current = state
        max_energy = 0.0
        for _ in range(round(0.5 / dt)):
            current, record = solver._step(
                current, dt, model.default_openings, [0.0, 0.0], [320.0, 360.0], {}
            )
            max_energy = max(max_energy, abs(record["conservation_residuals"][-1]))
        actual = np.array([s.temperature_k for _, s in current.nodes])
        fixed.append(
            {
                "dt_s": dt,
                "temperature_error_k": float(abs(actual - reference.y[:, -1]).max()),
                "energy_residual_j": max_energy,
            }
        )
    ratios = [fixed[j]["temperature_error_k"] / fixed[j + 1]["temperature_error_k"] for j in (0, 1)]
    assert min(ratios) > 14, ratios
    result = PRCollocation(model, phase_fallback=False).advance(state, 0.5)
    # Report state uses the existing transport JSON schema.
    from surface import PREquilibriumTransportState

    endpoint = PREquilibriumTransportState.from_dict(result["state"])
    errors = abs(np.array([s.temperature_k for _, s in endpoint.nodes]) - reference.y[:, -1])
    assert max(errors) < 1e-5
    assert result["fallback_intervals"] == 0
    return {
        "scope": "Closed, pure methane PR thermal contact, isochoric independent scalar ODE; no experimental calibration or full facility qualification",
        "fixed_step_runs": fixed,
        "observed_refinement_ratios": ratios,
        "adaptive_temperature_error_k": float(max(errors)),
        "adaptive_accepted_substeps": len(result["steps"]),
        "adaptive_attempts": result["attempts"],
        "conservation_residuals": result["conservation_residuals"],
        "fallback_intervals": result["fallback_intervals"],
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        Path(__file__).resolve(),
        ROOT / "scripts/audit_pr_collocation.py",
        ROOT / "scripts/run_pr_equilibrium_transport.py",
        ROOT / "tests/data/surface/pr_mixture_coolprop.json",
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = check()
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]
        for p in paths
    ):
        raise RuntimeError("PR collocation sources changed during validation")
    result["source_sha256"] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
