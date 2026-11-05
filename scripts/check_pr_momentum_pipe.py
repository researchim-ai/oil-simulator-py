"""Independent NumPy PR/Euler/kinetic and signed boundary audit.

Recomputes finite-volume Rusanov fluxes, wall reactions, viscous impulses,
U+K and numerical entropy; no runtime decode, solve or inspect is called.
Single-phase stability is the runtime's checked domain, not a global certificate.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from check_stratified_pr import Reference, softmax
from check_stratified_pr_flow import close
from check_stratified_pr_network import reference_darcy

from surface.pr_momentum_pipe import PRMomentumPipe


def stock(state, nc):
    n = np.array(state["component_moles"])
    momentum = np.array(state["momentum_kg_m_s"])
    energy = np.array(state["total_energy_j"])
    if n.shape != (len(momentum), nc) or energy.shape != momentum.shape:
        raise ValueError("Conserved pipe shape differs")
    return np.column_stack((n, momentum, energy))


def decode(state, model, fluid):
    n = np.array(state["component_moles"])
    active = np.flatnonzero(n[0] > 0)
    nc = len(fluid.names)
    c = len(active)
    count = model["cells"]
    if n.shape != (count, nc) or not np.all((n > 0) == (n[0] > 0)) or np.any(n < 0):
        raise ValueError("Invalid active support")
    y = np.array(state["solution_variables"]).reshape(count, c + 2)
    ref = Reference(fluid, active)
    pressure = np.exp(y[:, 0])
    temperature = y[:, c] * fluid.tref
    velocity = y[:, c + 1] * model["maximum_velocity_m_s"]
    rows = []
    flux = []
    entropy = []
    rho = []
    internal = []
    kinetic = []
    fractions = []
    for p, t, v, row in zip(pressure, temperature, velocity, y):
        if (
            not fluid.min_pressure_pa * (1 - 1e-12) <= p <= fluid.max_pressure_pa * (1 + 1e-12)
            or not fluid.min_temperature_k <= t <= fluid.max_temperature_k
            or abs(v) > model["maximum_velocity_m_s"] * (1 + 1e-12)
        ):
            raise ValueError("Primitive outside acoustic/EOS domain")
        x = np.zeros(nc)
        x[active] = softmax(np.r_[row[1:c], 0.0])
        q = ref.state(p, t, x, "stable")
        concentration = x / q["vm"]
        m = q["rho"] * v
        i = q["u"] / q["vm"]
        k = 0.5 * q["rho"] * v * v
        rows.append(np.r_[concentration, m, i + k])
        flux.append(np.r_[concentration * v, m * v + p, (i + k + p) * v])
        entropy.append(q["s"] / q["vm"])
        rho.append(q["rho"])
        internal.append(i)
        kinetic.append(k)
        fractions.append(x)
        dp = max(1e-4, p * 1e-5)
        dt = max(1e-5, t * 1e-5)
        a, b = ref.state(p + dp, t, x, "stable"), ref.state(p - dp, t, x, "stable")
        cc, d = ref.state(p, t + dt, x, "stable"), ref.state(p, t - dt, x, "stable")
        st = (cc["s"] - d["s"]) / (2 * dt)
        den = (a["rho"] - b["rho"]) / (2 * dp) - (cc["rho"] - d["rho"]) / (2 * dt) * (
            a["s"] - b["s"]
        ) / (2 * dp) / st
        if (
            st <= 0
            or den <= 0
            or 1 / math.sqrt(den) + abs(v) > model["wave_speed_bound_m_s"] * (1 + 2e-6)
        ):
            raise ValueError("Independent acoustic speed bound rejected")
    volume = math.pi * model["diameter_m"] ** 2 / 4 * model["length_m"] / count
    u = np.array(rows)
    target = stock(state, nc)
    close(u[:, :nc] * volume, target[:, :nc], atol=1e-10, rtol=2e-8)
    close(u[:, -2] * volume, target[:, -2], atol=1e-10, rtol=2e-8)
    close(u[:, -1] * volume, target[:, -1], atol=1e-6, rtol=2e-8)
    return {
        "p": pressure,
        "t": temperature,
        "v": velocity,
        "x": np.array(fractions),
        "u": u,
        "flux": np.array(flux),
        "s": np.array(entropy),
        "rho": np.array(rho),
        "internal": np.array(internal),
        "kinetic": np.array(kinetic),
        "ref": ref,
        "volume": volume,
    }


def faces(data, model, boundaries, fluid):
    if model["boundary_mode"] == "periodic":
        ul = np.vstack((data["u"][-1], data["u"]))
        ur = np.vstack((data["u"], data["u"][0]))
        fl = np.vstack((data["flux"][-1], data["flux"]))
        fr = np.vstack((data["flux"], data["flux"][0]))
        sl = np.r_[data["s"][-1], data["s"]]
        sr = np.r_[data["s"], data["s"][0]]
        gl = np.r_[(data["s"] * data["v"])[-1], data["s"] * data["v"]]
        gr = np.r_[data["s"] * data["v"], (data["s"] * data["v"])[0]]
    else:
        ghosts = []
        for index, boundary in zip((0, -1), boundaries):
            if boundary is None or boundary == "wall":
                u = data["u"][index].copy()
                u[-2] *= -1
                v = -data["v"][index]
                p = data["p"][index]
                s = data["s"][index]
            else:
                if boundary["fluid_id"] != fluid.model_id:
                    raise ValueError("Boundary reference differs")
                p, t = boundary["pressure_pa"], boundary["temperature_k"]
                x = np.array(boundary["mole_fractions"])
                v = data["v"][index]
                if (
                    np.any((x > 0) & (np.array(data["x"][index]) == 0))
                    or np.any(x < 0)
                    or abs(sum(x) - 1) > 1e-10
                ):
                    raise ValueError("Boundary activates an absent component")
                q = data["ref"].state(p, t, x, "stable")
                n = x / q["vm"]
                rho = q["rho"]
                energy = q["u"] / q["vm"] + 0.5 * rho * v * v
                u = np.r_[n, rho * v, energy]
                s = q["s"] / q["vm"]
            f = np.r_[u[:-2] * v, u[-2] * v + p, (u[-1] + p) * v]
            ghosts.append((u, f, s, s * v))
        a, b = ghosts
        ul = np.vstack((a[0], data["u"]))
        ur = np.vstack((data["u"], b[0]))
        fl = np.vstack((a[1], data["flux"]))
        fr = np.vstack((data["flux"], b[1]))
        sl = np.r_[a[2], data["s"]]
        sr = np.r_[data["s"], b[2]]
        gl = np.r_[a[3], data["s"] * data["v"]]
        gr = np.r_[data["s"] * data["v"], b[3]]
    alpha = model["wave_speed_bound_m_s"]
    return (fl + fr) / 2 - alpha * (ur - ul) / 2, (gl + gr) / 2 - alpha * (sr - sl) / 2


def summary(saved, data):
    for key, value in {
        "pressure_pa": data["p"],
        "temperature_k": data["t"],
        "velocity_m_s": data["v"],
        "mole_fractions": data["x"],
        "density_kg_m3": data["rho"],
        "internal_energy_j": data["internal"] * data["volume"],
        "kinetic_energy_j": data["kinetic"] * data["volume"],
    }.items():
        close(saved[key], value, atol=1e-7, rtol=2e-8)
    if "potential" in data:
        close(saved["potential_energy_j"], data["potential"] * data["volume"], atol=1e-7, rtol=2e-8)
    close(saved["entropy_j_k"], data["s"].sum() * data["volume"], atol=1e-7)


def audit(config, report):
    if set(config) != {"model", "initial", "schedule"} or config != report["config"]:
        raise ValueError("Momentum request differs")
    # Constructor only normalizes/validates metadata; dynamics remain independent.
    parsed = PRMomentumPipe.from_dict(config["model"])
    model = parsed.to_dict()
    fluid = parsed.fluid
    decode_fn, faces_fn = decode, faces
    if model["model"] == "inclined_single_phase_pr_momentum_v1":
        from check_pr_inclined_pipe import decode as decode_fn
        from check_pr_inclined_pipe import faces as faces_fn
    nc = len(fluid.names)
    if model != report["model"]:
        raise ValueError("Momentum normalized model differs")
    old = report["initial"]
    time = old["time_s"]
    maximum_error = np.zeros(nc + 2)
    checked = 0
    data_old = decode_fn(old, model, fluid)
    if old["model_id"] != parsed.model_id:
        raise ValueError("Initial model identity differs")
    area = math.pi * model["diameter_m"] ** 2 / 4
    volume = data_old["volume"]

    def boundary(spec):
        if spec is None or spec == "wall":
            return spec
        return {**spec, "fluid_id": fluid.model_id}

    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        duration = command["duration_s"]
        end = time + duration
        if not math.isfinite(duration) or duration <= 0 or not interval["steps"]:
            raise ValueError("Invalid momentum interval")
        bounds = boundary(command.get("left")), boundary(command.get("right"))
        if (interval["left_boundary"], interval["right_boundary"]) != bounds:
            raise ValueError("Boundary command differs")
        if model["boundary_mode"] == "periodic" and any(v is not None for v in bounds):
            raise ValueError("Periodic boundary mismatch")
        summary(interval["initial_summary"], data_old)
        interval_old = stock(old, nc)
        boundary_total = np.zeros(nc + 2)
        friction_total = entropy_total = 0.0
        for step in interval["steps"]:
            state = step["state"]
            dt = step["dt_s"]
            if not math.isfinite(dt) or dt <= 0 or state["model_id"] != parsed.model_id:
                raise ValueError("Substep/model identity differs")
            data = decode_fn(state, model, fluid)
            flux, sflux = faces_fn(data, model, bounds, fluid)
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
            gravity = data.get("gravity_force_n", np.zeros(model["cells"]))
            source[:, -2] = -drag + gravity / volume
            expected = dt * area * (flux[:-1] - flux[1:]) + dt * volume * source
            residual = stock(state, nc) - stock(old, nc) - expected
            allowed = np.r_[[1e-9] * nc, 1e-9, 1e-6] + 2e-6 * np.abs(expected)
            if np.any(~np.isfinite(residual)) or np.any(abs(residual) > allowed):
                raise ValueError("Independent FV momentum/energy residual rejected")
            close(step["left_flux"], area * flux[0], atol=1e-7, rtol=2e-8)
            close(step["right_flux"], area * flux[-1], atol=1e-7, rtol=2e-8)
            impulse = -dt * volume * drag.sum()
            close(step["friction_impulse_kg_m_s"], impulse, atol=1e-10)
            total = residual.sum(0)
            close(step["conservation_residuals"], total, atol=2e-6)
            production = (data["s"].sum() - data_old["s"].sum()) * volume - dt * area * (
                sflux[0] - sflux[-1]
            )
            if production < -1e-8 - 1e-8 * max(1, abs(data_old["s"].sum() * volume)):
                raise ValueError("Independent numerical entropy rejected")
            close(step["entropy_production_j_k"], production, atol=2e-7)
            maximum_error = np.maximum(maximum_error, abs(total))
            checked += 1
            time += dt
            close(state["time_s"], time, atol=1e-12, rtol=1e-12)
            boundary_total += dt * area * (flux[0] - flux[-1])
            gravitational_impulse = dt * gravity.sum()
            close(step.get("body_force_impulse_kg_m_s", 0.0), gravitational_impulse, atol=1e-10)
            friction_total += impulse + gravitational_impulse
            entropy_total += production
            old = state
            data_old = data
        final = interval["state"]
        for key in ("component_moles", "momentum_kg_m_s", "total_energy_j", "solution_variables"):
            close(final[key], old[key], atol=1e-10, rtol=1e-12)
        close(time, end, atol=1e-12, rtol=1e-12)
        close(final["time_s"], end, atol=1e-12, rtol=1e-12)
        if final["model_id"] != parsed.model_id:
            raise ValueError("Final pipe identity differs")
        total = (
            (stock(final, nc) - interval_old).sum(0)
            - boundary_total
            - np.r_[np.zeros(nc), friction_total, 0.0]
        )
        close(interval["conservation_residuals"], total, atol=2e-6)
        close(interval["entropy_production_j_k"], entropy_total, atol=2e-7)
        summary(interval["final_summary"], data_old)
        old = final
    return {
        "accepted": True,
        "checked_substeps": checked,
        "maximum_component_residual_mol": float(maximum_error[:nc].max()),
        "maximum_momentum_residual_kg_m_s": float(maximum_error[-2]),
        "maximum_energy_residual_j": float(maximum_error[-1]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(json.loads(args.config.read_text()), json.loads(args.result.read_text())),
            indent=2,
        )
    )
