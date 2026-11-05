"""Independent NumPy PR, HLLC and wall balances for finite vessel pipe runs.

Does not call runtime decode/step/inspect. Initial inventories are checkpoints,
not evidence of correct initial condition generation or universal PR stability.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_pr_momentum_pipe import decode, faces, stock
from check_stratified_pr import Reference, softmax
from check_stratified_pr_flow import close
from check_stratified_pr_network import reference_darcy

from surface.pr_momentum_vessels import PRMomentumVesselNetwork


def reservoir_decode(state, volume, fluid):
    n = np.asarray(state["component_moles"])
    if not np.isfinite(n).all() or np.any(n < 0) or n.sum() <= 0:
        raise ValueError("Invalid finite vessel inventory")
    active = np.flatnonzero(n > 0)
    y = np.asarray(state["solution_variables"])
    if y.shape != (len(active) + 1,) or len(n) != len(fluid.names):
        raise ValueError("Vessel dimensions differ")
    pressure = np.exp(y[0])
    temperature = y[-1] * fluid.tref
    if (
        not fluid.min_pressure_pa <= pressure <= fluid.max_pressure_pa
        or not fluid.min_temperature_k <= temperature <= fluid.max_temperature_k
    ):
        raise ValueError("Vessel primitive outside domain")
    x = np.zeros(len(n))
    x[active] = softmax(np.r_[y[1:-1], 0.0])
    q = Reference(fluid, active).state(pressure, temperature, x, "stable")
    u = np.r_[x / q["vm"], 0.0, q["u"] / q["vm"]]
    close(u[:-2] * volume, n, atol=1e-10, rtol=2e-8)
    close(u[-1] * volume, state["internal_energy_j"], atol=1e-6, rtol=2e-8)
    return (u, np.r_[np.zeros(len(n)), pressure, 0.0], pressure, 0.0, q["rho"], q["s"] / q["vm"])


def port_flux(left, right, alpha):
    u0, f0, p0, v0, rho0, s0 = left
    u1, f1, p1, v1, rho1, s1 = right
    a = rho0 * (-alpha - v0)
    b = rho1 * (alpha - v1)
    contact = (p1 - p0 + a * v0 - b * v1) / (a - b)
    ps = p0 + a * (contact - v0)
    if abs(contact) >= alpha or ps <= 0:
        raise ValueError("Invalid acoustic/contact fan")
    if contact >= 0:
        u, f, p, v, rho, s, speed = u0, f0, p0, v0, rho0, s0, -alpha
    else:
        u, f, p, v, rho, s, speed = u1, f1, p1, v1, rho1, s1, alpha
    ratio = (speed - v) / (speed - contact)
    if ratio <= 0:
        raise ValueError("Invalid star density")
    us = np.r_[
        u[:-2] * ratio,
        rho * ratio * contact,
        ((speed - v) * u[-1] - p * v + ps * contact) / (speed - contact),
    ]
    return f + speed * (us - u), s * ratio * contact, contact, ps


def wall(old, new, fluid_temperature, wall, dt, parts=1):
    if not wall.get("min_temperature_k", 1.0) <= new <= wall.get("max_temperature_k", 2000.0):
        raise ValueError("Wall outside domain")
    capacity = wall["heat_capacity_j_k"] / parts
    heat = dt * wall["fluid_ua_w_k"] / parts * (new - fluid_temperature)
    ambient = (
        dt
        * wall.get("ambient_ua_w_k", 0.0)
        / parts
        * (wall.get("ambient_temperature_k", 298.15) - new)
    )
    close(capacity * (new - old), ambient - heat, atol=1e-7, rtol=2e-7)
    return (
        heat,
        ambient,
        capacity * math.log(new / old) - ambient / wall.get("ambient_temperature_k", 298.15),
    )


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Request differs")
    model = PRMomentumVesselNetwork.from_dict(config["model"])
    if json.loads(json.dumps(model.to_dict())) != report["model"]:
        raise ValueError("Model differs")
    pm = model.pipe.pipe.to_dict()
    fluid = model.pipe.pipe.fluid
    area = math.pi * pm["diameter_m"] ** 2 / 4
    volume = area * pm["length_m"] / pm["cells"]
    nc = len(fluid.names)
    old = report["initial"]

    def identity(state):
        if (
            state["model_id"] != model.model_id
            or state["pipe"]["model_id"] != model.pipe.model_id
            or state["pipe"]["pipe"]["model_id"] != model.pipe.pipe.model_id
            or not pm["minimum_step_s"]
            <= state["pipe"]["pipe"]["next_step_s"]
            <= pm["maximum_step_s"]
        ):
            raise ValueError("Checkpoint model identity or carried step differs")

    identity(old)
    decode(old["pipe"]["pipe"], pm, fluid)
    checked = 0
    maximum = np.zeros(nc + 2)
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        if (
            not math.isfinite(command["duration_s"])
            or command["duration_s"] <= 0
            or not interval["steps"]
        ):
            raise ValueError("Empty or invalid report interval")
        interval_residual = np.zeros(nc + 1)
        interval_ambient = interval_entropy = 0.0
        end = old["pipe"]["pipe"]["time_s"] + command["duration_s"]
        opened = command.get("openings", [True, True])
        if interval["openings"] != opened:
            raise ValueError("Isolation command differs")
        for step in interval["steps"]:
            dt = step["dt_s"]
            new = step["state"]
            identity(new)
            if dt <= 0:
                raise ValueError("Nonpositive step")
            close(
                new["pipe"]["pipe"]["time_s"] - old["pipe"]["pipe"]["time_s"],
                dt,
                atol=1e-14,
                rtol=1e-10,
            )
            pd = decode(new["pipe"]["pipe"], pm, fluid)
            od = decode(old["pipe"]["pipe"], pm, fluid)
            ff, ss = faces(pd, pm, (None, None), fluid)
            ports = []
            reservoir_entropy = 0.0
            ambient_total = 0.0
            wall_delta = 0.0
            for i, index in enumerate((0, -1)):
                rd = reservoir_decode(new["vessels"][i], model.volumes[i], fluid)
                ro = reservoir_decode(old["vessels"][i], model.volumes[i], fluid)
                pipe = (
                    pd["u"][index],
                    pd["flux"][index],
                    pd["p"][index],
                    pd["v"][index],
                    pd["rho"][index],
                    pd["s"][index],
                )
                if opened[i]:
                    port = (
                        port_flux(rd, pipe, pm["wave_speed_bound_m_s"])
                        if i == 0
                        else port_flux(pipe, rd, pm["wave_speed_bound_m_s"])
                    )
                    ff[index], ss[index] = port[:2]
                else:
                    port = (ff[index], ss[index], 0.0, pd["p"][index])
                ports.append(port)
                close(step["port_contact_velocity_m_s"][i], port[2], atol=1e-8, rtol=2e-7)
                close(step["port_pressure_pa"][i], port[3], atol=1e-4, rtol=2e-8)
                oldv, newv = old["vessels"][i], new["vessels"][i]
                wt = model.to_dict()["vessel_walls"][i]
                q, qa, ds = wall(
                    oldv["wall_temperature_k"],
                    newv["wall_temperature_k"],
                    newv["solution_variables"][-1] * fluid.tref,
                    wt,
                    dt,
                )
                transferred = (
                    dt * area * ff[index] * (-1 if i == 0 else 1) if opened[i] else np.zeros(nc + 2)
                )
                close(
                    np.array(newv["component_moles"]) - oldv["component_moles"],
                    transferred[:-2],
                    atol=1e-10,
                    rtol=3e-6,
                )
                close(
                    newv["internal_energy_j"] - oldv["internal_energy_j"],
                    transferred[-1] + q,
                    atol=1e-6,
                    rtol=3e-6,
                )
                close(step["vessel_ambient_heat_j"][i], qa, atol=1e-9, rtol=2e-7)
                impulse = dt * area * ff[index, -2] * (1 if i == 0 else -1) if opened[i] else 0.0
                close(step["vessel_support_impulse_kg_m_s"][i], impulse, atol=1e-10, rtol=2e-7)
                reservoir_entropy += (rd[-1] - ro[-1]) * model.volumes[i] + ds
                ambient_total += qa
                wall_delta += wt["heat_capacity_j_k"] * (
                    newv["wall_temperature_k"] - oldv["wall_temperature_k"]
                )
            close(step["pipe"]["left_flux"], area * ff[0], atol=1e-6, rtol=2e-7)
            close(step["pipe"]["right_flux"], area * ff[-1], atol=1e-6, rtol=2e-7)
            heats = []
            ambients = []
            wall_entropy = 0.0
            wt = model.pipe.to_dict()["wall"]
            for i in range(pm["cells"]):
                q, qa, ds = wall(
                    old["pipe"]["wall_temperature_k"][i],
                    new["pipe"]["wall_temperature_k"][i],
                    pd["t"][i],
                    wt,
                    dt,
                    pm["cells"],
                )
                heats.append(q)
                ambients.append(qa)
                wall_entropy += ds
                wall_delta += (
                    wt["heat_capacity_j_k"]
                    / pm["cells"]
                    * (new["pipe"]["wall_temperature_k"][i] - old["pipe"]["wall_temperature_k"][i])
                )
            close(step["pipe"]["fluid_heat_j"], heats, atol=1e-9, rtol=2e-7)
            close(step["pipe"]["ambient_heat_j"], ambients, atol=1e-9, rtol=2e-7)
            drag = np.array(
                [
                    reference_darcy(
                        v,
                        rho,
                        1.0,
                        pm["diameter_m"],
                        pm["dynamic_viscosity_pa_s"],
                        pm["roughness_m"],
                    )[0]
                    if pm["dynamic_viscosity_pa_s"]
                    else 0.0
                    for v, rho in zip(pd["v"], pd["rho"])
                ]
            )
            target = dt * area * (ff[:-1] - ff[1:])
            target[:, -2] -= dt * volume * drag
            target[:, -1] += heats
            delta = stock(new["pipe"]["pipe"], nc) - stock(old["pipe"]["pipe"], nc)
            close(delta[:, :nc], target[:, :nc], atol=1e-10, rtol=3e-6)
            close(delta[:, -2], target[:, -2], atol=1e-10, rtol=3e-6)
            close(delta[:, -1], target[:, -1], atol=1e-6, rtol=3e-6)
            close(
                step["pipe"]["friction_impulse_kg_m_s"],
                -dt * volume * drag.sum(),
                atol=1e-10,
                rtol=2e-7,
            )
            maximum = np.maximum(maximum, np.max(np.abs(delta - target), axis=0))
            entropy = (pd["s"] - od["s"]).sum() * volume + wall_entropy + reservoir_entropy
            if entropy < -1e-7:
                raise ValueError("Independent total entropy decreased")
            close(step["entropy_production_j_k"], entropy, atol=1e-7, rtol=2e-6)
            dn = (stock(new["pipe"]["pipe"], nc) - stock(old["pipe"]["pipe"], nc)).sum(0)
            for a, b in zip(old["vessels"], new["vessels"]):
                dn[:-2] += np.array(b["component_moles"]) - a["component_moles"]
                dn[-1] += b["internal_energy_j"] - a["internal_energy_j"]
            combined = dn[-1] + wall_delta - ambient_total - sum(ambients)
            close(dn[:-2], np.zeros(nc), atol=1e-9, rtol=0)
            close(combined, 0.0, atol=1e-5, rtol=0)
            close(step["conservation_residuals"], np.r_[dn[:-2], combined], atol=1e-6, rtol=2e-6)
            interval_residual += np.r_[dn[:-2], combined]
            interval_ambient += ambient_total + sum(ambients)
            interval_entropy += entropy
            old = new
            checked += 1
        close(interval["state"]["pipe"]["pipe"]["time_s"], end, atol=1e-13, rtol=0)
        identity(interval["state"])
        close(interval["conservation_residuals"], interval_residual, atol=1e-6, rtol=2e-6)
        close(interval["ambient_heat_j"], interval_ambient, atol=1e-8, rtol=2e-6)
        close(interval["entropy_production_j_k"], interval_entropy, atol=1e-7, rtol=2e-6)
        # Only carried time/next-step metadata may differ from the last halfstep.
        final = json.loads(json.dumps(interval["state"]))
        last = json.loads(json.dumps(old))
        for key in ("time_s", "next_step_s"):
            last["pipe"]["pipe"][key] = final["pipe"]["pipe"][key]
        if last != final:
            raise ValueError("Published interval differs from accepted final state")
        old = interval["state"]
    return {
        "accepted": True,
        "checked_substeps": checked,
        "maximum_pipe_fv_residuals": maximum.tolist(),
    }


def main():
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


if __name__ == "__main__":
    main()
