"""Independent NumPy audit of BDF/BE cell equations and history-weighted ports.

Reuses the independent caloric PR, Rusanov and Darcy reference operators.
Never calls runtime step, decode, inspect or the BDF error estimator.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from check_pr_momentum_pipe import decode, faces, stock, summary
from check_stratified_pr_flow import close
from check_stratified_pr_network import reference_darcy

from surface.operation import _digest
from surface.pr_momentum_pipe import PRMomentumPipe


def audit(report):
    specification = report["model"]
    if specification["model"] != "single_phase_pr_momentum_bdf2_v1":
        raise ValueError("Unknown BDF operator")
    # Metadata only; reference dynamics do not use the runtime solver.
    parsed = PRMomentumPipe.from_dict(specification["pipe"])
    model, fluid = parsed.to_dict(), parsed.fluid
    nc = len(fluid.names)
    history = list(report["initial_state"]["history"])
    boundary = report["left_boundary"], report["right_boundary"]
    boundary_id = _digest({"left": boundary[0], "right": boundary[1]})
    reset = report["initial_state"]["boundary_id"] != boundary_id
    if reset != report["history_reset"]:
        raise ValueError("BDF boundary reset metadata differs")
    if reset:
        history = [
            {
                **history[-1],
                "left_exchange": [0.0] * (nc + 2),
                "right_exchange": [0.0] * (nc + 2),
                "friction_impulse": 0.0,
                "entropy_exchange": 0.0,
                "order": 0,
            }
        ]
    start = history[-1]["state"]
    current = start
    data_old = decode(current, model, fluid)
    left_total, right_total = np.zeros(nc + 2), np.zeros(nc + 2)
    impulse_total = entropy_total = 0.0
    area = math.pi * model["diameter_m"] ** 2 / 4
    maximum = np.zeros(nc + 2)
    for step in report["steps"]:
        point = step["point"]
        state = point["state"]
        h = step["dt_s"]
        if not math.isfinite(h) or h <= 0 or state["model_id"] != parsed.model_id:
            raise ValueError("BDF clock/registry differs")
        close(state["time_s"], current["time_s"] + h, atol=1e-12, rtol=1e-12)
        data = decode(state, model, fluid)
        flux, entropy_flux = faces(data, model, boundary, fluid)
        drag = np.zeros(model["cells"])
        if model["dynamic_viscosity_pa_s"]:
            drag = np.array(
                [
                    reference_darcy(
                        v,
                        rho,
                        1.0,
                        model["diameter_m"],
                        model["dynamic_viscosity_pa_s"],
                        model["roughness_m"],
                    )[0]
                    for v, rho in zip(data["v"], data["rho"])
                ]
            )
        source = np.zeros_like(data["u"])
        source[:, -2] = -data["volume"] * drag
        rate = area * (flux[:-1] - flux[1:]) + source
        if point["order"] == 1:
            beta, gamma = 0.0, 1.0
            expected = h * rate
        elif point["order"] == 2 and len(history) >= 3:
            previous = history[-2]["state"]
            r = h / (current["time_s"] - previous["time_s"])
            if r > 1.5 * (1 + 1e-10):
                raise ValueError("BDF history step ratio rejected")
            # Independently form the differentiated BDF2 equation.
            a0, a1, a2 = (1 + 2 * r) / (1 + r), -(1 + r), r * r / (1 + r)
            differentiated = (
                a0 * stock(state, nc) + a1 * stock(current, nc) + a2 * stock(previous, nc)
            )
            close(differentiated, h * rate, atol=np.r_[[1e-9] * nc, 1e-9, 2e-6], rtol=2e-6)
            beta, gamma = r * r / (1 + 2 * r), (1 + r) / (1 + 2 * r)
            expected = beta * (stock(current, nc) - stock(previous, nc)) + gamma * h * rate
        else:
            raise ValueError("BDF order/history unsupported")
        difference = stock(state, nc) - stock(current, nc) - expected
        close(
            difference,
            np.zeros_like(difference),
            atol=np.r_[[1e-9] * nc, 1e-9, 2e-6] + 2e-6 * np.abs(expected),
            rtol=0,
        )
        left = beta * np.array(history[-1]["left_exchange"]) + gamma * h * area * flux[0]
        right = beta * np.array(history[-1]["right_exchange"]) + gamma * h * area * flux[-1]
        impulse = beta * history[-1]["friction_impulse"] - gamma * h * data["volume"] * drag.sum()
        entropy = beta * history[-1]["entropy_exchange"] + gamma * h * area * (
            entropy_flux[0] - entropy_flux[-1]
        )
        close(point["left_exchange"], left, atol=2e-6, rtol=2e-8)
        close(point["right_exchange"], right, atol=2e-6, rtol=2e-8)
        close(point["friction_impulse"], impulse, atol=1e-9, rtol=1e-7)
        close(point["entropy_exchange"], entropy, atol=2e-7, rtol=1e-7)
        if not math.isfinite(step["temporal_error"]) or not 0 <= step["temporal_error"] <= 1:
            raise ValueError("Unaccepted temporal estimate")
        left_total += left
        right_total += right
        impulse_total += impulse
        entropy_total += entropy
        maximum = np.maximum(maximum, np.abs(difference).max(axis=0))
        current, data_old = state, data
        history.append(point)
        history = history[-3:]
    if not report["steps"]:
        raise ValueError("BDF report has no accepted trajectory")
    close(
        current["time_s"], start["time_s"] + report["requested_duration_s"], atol=1e-12, rtol=1e-12
    )
    final = report["state"]
    if final["boundary_id"] != boundary_id or final["integrator_id"] != _digest(specification):
        raise ValueError("Final checkpoint registry differs")
    if len(final["history"]) != len(history):
        raise ValueError("Final BDF history differs")
    for saved, expected in zip(final["history"], history):
        # next_step is the new controller proposal, not part of the PDE state.
        a, b = dict(saved["state"]), dict(expected["state"])
        a.pop("next_step_s")
        b.pop("next_step_s")
        if a != b or any(
            saved[k] != expected[k]
            for k in [
                "left_exchange",
                "right_exchange",
                "friction_impulse",
                "entropy_exchange",
                "order",
            ]
        ):
            raise ValueError("Final BDF checkpoint differs from accepted trajectory")
    delta = (stock(current, nc) - stock(start, nc)).sum(0) - left_total + right_total
    delta[-2] -= impulse_total
    close(report["conservation_residuals"], delta, atol=2e-6, rtol=1e-7)
    close(delta, np.zeros(nc + 2), atol=2e-6)
    first = decode(start, model, fluid)
    entropy_balance = (data_old["s"].sum() - first["s"].sum()) * data_old["volume"] - entropy_total
    close(report["entropy_balance_diagnostic_j_k"], entropy_balance, atol=2e-7, rtol=1e-7)
    summary(report["final_summary"], data_old)
    return {
        "accepted": True,
        "checked_steps": len(report["steps"]),
        "max_cell_equation_residual": maximum.tolist(),
        "entropy_policy": "BDF entropy balance diagnostic; no BE per-step inequality claimed",
    }


def audit_schedule(report):
    if report.get("integrator") != "bdf2":
        raise ValueError("A BDF2 schedule report is required")
    parsed = PRMomentumPipe.from_dict(report["config"]["model"])
    if parsed.to_dict() != report["model"]:
        raise ValueError("Schedule physical model differs")
    previous = report["initial"]
    results = []
    for command, interval in zip(report["config"]["schedule"], report["intervals"], strict=True):
        if interval["initial_state"] != previous or interval["model"]["pipe"] != report["model"]:
            raise ValueError("BDF schedule/checkpoint chain differs")
        if interval["requested_duration_s"] != command["duration_s"]:
            raise ValueError("Requested BDF duration differs")
        for name in ["left", "right"]:
            expected = command.get(name)
            if isinstance(expected, dict):
                expected = {**expected, "fluid_id": parsed.fluid.model_id}
            if interval[name + "_boundary"] != expected:
                raise ValueError("BDF boundary command differs")
        results.append(audit(interval))
        previous = interval["state"]
    if not results:
        raise ValueError("Empty BDF schedule")
    return {
        "accepted": True,
        "checked_intervals": len(results),
        "checked_steps": sum(r["checked_steps"] for r in results),
        "intervals": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    print(json.dumps(audit_schedule(report) if "intervals" in report else audit(report), indent=2))


if __name__ == "__main__":
    main()
