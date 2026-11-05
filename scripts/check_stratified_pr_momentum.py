"""Independent NumPy audit: gravity profile, local chemical potential and HLLC.

Reference integrates each column on 32 nodes and solves port chemical states
with SciPy. Runtime column/pipe decode, inspect and dynamics are not called.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_pr_inclined_pipe import decode as inclined_decode
from check_pr_inclined_pipe import faces as inclined_faces
from check_pr_momentum_pipe import decode as horizontal_decode
from check_pr_momentum_pipe import faces as horizontal_faces
from check_pr_momentum_pipe import stock
from check_pr_momentum_vessels import port_flux, wall
from check_stratified_pr import grid
from check_stratified_pr_flow import close, profile
from check_stratified_pr_network import reference_darcy

from surface.stratified_pr_momentum import StratifiedPRMomentumNetwork


def column(state, system, height, datum=None):
    relative_height = height if datum is None else datum - system["elevation_m"]
    if len(state["phase_roots"]) != 1:
        raise ValueError("Auditor requires one phase per gravitational column")
    fluid, geometry, sample, _, _, _ = profile(state, system)
    hh, ww = grid(geometry, [], 32)
    n = np.zeros(len(fluid.names))
    energy_relative = entropy = 0.0
    for h, weight in zip(hh[0], ww[0]):
        _, x, q = sample(h, 0)
        n += weight * x / q["vm"]
        energy_relative += weight * (
            q["u"] / q["vm"] + q["rho"] * system["gravity_m_s2"] * (h - relative_height)
        )
        entropy += weight * q["s"] / q["vm"]
    mass = sum(v * c.molar_mass_kg_mol for v, c in zip(state["component_moles"], fluid.components))
    target = state["total_energy_j"] - mass * system["gravity_m_s2"] * (
        system["elevation_m"] + relative_height
    )
    close(n, state["component_moles"], atol=1e-8, rtol=5 * system.get("relative_tolerance", 1e-6))
    close(energy_relative, target, atol=1e-5, rtol=5 * system.get("relative_tolerance", 1e-6))
    pp, x, q = sample(height, 0)
    u = np.r_[x / q["vm"], 0.0, q["u"] / q["vm"]]
    port = (u, np.r_[np.zeros(len(x)), pp, 0.0], pp, 0.0, q["rho"], q["s"] / q["vm"])
    return port, entropy, target


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Request differs")
    parsed = StratifiedPRMomentumNetwork.from_dict(config["model"])
    model = json.loads(json.dumps(parsed.to_dict()))
    if model != report["model"]:
        raise ValueError("Normalized model differs")
    pm = model["pipe"]["pipe"]
    inclined = pm["model"] == "inclined_single_phase_pr_momentum_v1"
    decode = inclined_decode if inclined else horizontal_decode
    faces = inclined_faces if inclined else horizontal_faces
    fluid = parsed.pipe.pipe.fluid
    nc = len(fluid.names)
    area = math.pi * pm["diameter_m"] ** 2 / 4
    volume = area * pm["length_m"] / pm["cells"]
    old = report["initial"]
    checked = 0
    maximum = np.zeros(nc + 2)

    def identity(state):
        p = state["pipe"]["pipe"]
        if (
            state["model_id"] != parsed.model_id
            or state["pipe"]["model_id"] != parsed.pipe.model_id
            or p["model_id"] != parsed.pipe.pipe.model_id
            or not pm["minimum_step_s"] <= p["next_step_s"] <= pm["maximum_step_s"]
        ):
            raise ValueError("Checkpoint identity or step differs")
        for s, c in zip(state["vessels"], parsed.columns, strict=True):
            if s["model_id"] != parsed.model_id or s["column"]["model_id"] != c.model_id:
                raise ValueError("Column identity differs")
            close(s["time_s"], p["time_s"], atol=1e-13, rtol=0.0)
            close(s["next_step_s"], p["next_step_s"], atol=1e-15, rtol=0.0)

    identity(old)
    pdold = decode(old["pipe"]["pipe"], pm, fluid)
    cold = [
        column(v["column"], c, h, parsed.elevation)
        for v, c, h in zip(old["vessels"], model["columns"], parsed.heights)
    ]
    # New anchor-based initialization has directly checkable physical inputs.
    initial_spec = config["initial"]
    if "equilibrium" in initial_spec:
        anchor = initial_spec["equilibrium"]
        endpoint = pdold["reconstruction"] if inclined else pdold
        close(endpoint["p"][0], anchor["pressure_pa"], atol=1e-4, rtol=2e-8)
        close(endpoint["t"][0], anchor["temperature_k"], atol=1e-7, rtol=0.0)
        close(endpoint["x"][0], anchor["mole_fractions"], atol=1e-8, rtol=2e-8)
    else:
        for spec, vessel, system in zip(initial_spec["columns"], old["vessels"], model["columns"]):
            if "anchor_height_m" in spec:
                reference = profile(vessel["column"], system)
                pp, xx, _ = reference[2](spec["anchor_height_m"], 0)
                close(pp, spec["pressure_pa"], atol=1e-4, rtol=2e-8)
                close(xx, spec["mole_fractions"], atol=1e-8, rtol=2e-8)
                close(vessel["column"]["temperature_k"], spec["temperature_k"], atol=1e-7, rtol=0.0)
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        if (
            not math.isfinite(command["duration_s"])
            or command["duration_s"] <= 0
            or not interval["steps"]
        ):
            raise ValueError("Empty or invalid interval")
        end = old["pipe"]["pipe"]["time_s"] + command["duration_s"]
        opened = command.get("openings", [True, True])
        if interval["openings"] != opened:
            raise ValueError("Isolation command differs")
        interval_residual = np.zeros(nc + 1)
        interval_ambient = interval_entropy = 0.0
        for step in interval["steps"]:
            new = step["state"]
            dt = step["dt_s"]
            identity(new)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Invalid substep")
            close(
                new["pipe"]["pipe"]["time_s"] - old["pipe"]["pipe"]["time_s"],
                dt,
                atol=1e-13,
                rtol=1e-10,
            )
            pd = decode(new["pipe"]["pipe"], pm, fluid)
            ff, ss = faces(pd, pm, (None, None), fluid)
            cols = [
                column(v["column"], c, h, parsed.elevation)
                for v, c, h in zip(new["vessels"], model["columns"], parsed.heights)
            ]
            vessel_entropy = ambient_total = wall_delta = 0.0
            for i, index in enumerate((0, -1)):
                rd, entropy, erel = cols[i]
                endpoint = pd["reconstruction"] if inclined else pd
                if inclined:
                    rd = (rd[0].copy(), *rd[1:])
                    rd[0][-1] += (
                        rd[4] * parsed.gravity * (parsed.end_elevations[i] - parsed.elevation)
                    )
                pipe = (
                    endpoint["u"][index],
                    endpoint["flux"][index],
                    endpoint["p"][index],
                    endpoint["v"][index],
                    endpoint["rho"][index],
                    endpoint["s"][index],
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
                close(step["port_contact_velocity_m_s"][i], port[2], atol=1e-8, rtol=2e-7)
                # Pressure reports refer to the column even for a closed pipe end.
                close(step["port_pressure_pa"][i], rd[2], atol=1e-4, rtol=2e-8)
                x = rd[0][:-2] / rd[0][:-2].sum()
                close(step["port_mole_fractions"][i], x, atol=1e-8, rtol=2e-7)
                head = rd[4] * parsed.gravity * pm["diameter_m"] / rd[2]
                if head > parsed.maximum_transverse_head_fraction:
                    raise ValueError("Unresolved transverse head exceeds model domain")
                a, b = old["vessels"][i], new["vessels"][i]
                wt = model["vessel_walls"][i]
                heat, ambient, ds = wall(
                    a["wall_temperature_k"],
                    b["wall_temperature_k"],
                    b["column"]["temperature_k"],
                    wt,
                    dt,
                )
                transfer = (
                    dt * area * ff[index] * (-1 if i == 0 else 1) if opened[i] else np.zeros(nc + 2)
                )
                close(
                    np.array(b["column"]["component_moles"]) - a["column"]["component_moles"],
                    transfer[:-2],
                    atol=1e-10,
                    rtol=3e-6,
                )
                # Compare in the common port datum; includes changing N*g*z.
                close(erel - cold[i][2], transfer[-1] + heat, atol=2e-6, rtol=3e-6)
                rate = (
                    area
                    * (
                        ff[index, -1]
                        + sum(
                            ff[index, j] * c.molar_mass_kg_mol
                            for j, c in enumerate(fluid.components)
                        )
                        * parsed.gravity
                        * parsed.elevation
                    )
                    if opened[i]
                    else 0.0
                )
                close(step["port_energy_rates_with_pe_w"][i], rate, atol=1e-5, rtol=2e-7)
                close(step["vessel_ambient_heat_j"][i], ambient, atol=1e-9, rtol=2e-7)
                impulse = dt * area * ff[index, -2] * (1 if i == 0 else -1) if opened[i] else 0.0
                close(step["vessel_support_impulse_kg_m_s"][i], impulse, atol=1e-10, rtol=2e-7)
                vessel_entropy += entropy - cold[i][1] + ds
                ambient_total += ambient
                wall_delta += wt["heat_capacity_j_k"] * (
                    b["wall_temperature_k"] - a["wall_temperature_k"]
                )
            heats = []
            ambients = []
            wall_entropy = 0.0
            wt = model["pipe"]["wall"]
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
            close(step["pipe"]["left_flux"], area * ff[0], atol=1e-6, rtol=2e-7)
            close(step["pipe"]["right_flux"], area * ff[-1], atol=1e-6, rtol=2e-7)
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
            gravity = pd.get("gravity_force_n", np.zeros(pm["cells"]))
            target[:, -2] += dt * (gravity - volume * drag)
            close(
                step["pipe"].get("body_force_impulse_kg_m_s", 0.0),
                dt * gravity.sum(),
                atol=1e-10,
                rtol=2e-7,
            )
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
            total_entropy = (pd["s"] - pdold["s"]).sum() * volume + wall_entropy + vessel_entropy
            if total_entropy < -2e-6:
                raise ValueError("Independent entropy production rejected")
            close(step["entropy_production_j_k"], total_entropy, atol=2e-6, rtol=2e-6)
            dn = delta.sum(0)
            for i, (a, b) in enumerate(zip(old["vessels"], new["vessels"])):
                dn[:-2] += np.array(b["column"]["component_moles"]) - a["column"]["component_moles"]
                dn[-1] += cols[i][2] - cold[i][2]
            energy = dn[-1] + wall_delta - ambient_total - sum(ambients)
            close(dn[:-2], np.zeros(nc), atol=1e-9, rtol=0.0)
            close(energy, 0.0, atol=1e-5, rtol=0.0)
            close(step["conservation_residuals"], np.r_[dn[:-2], energy], atol=1e-6, rtol=2e-6)
            interval_residual += np.r_[dn[:-2], energy]
            interval_ambient += ambient_total + sum(ambients)
            interval_entropy += total_entropy
            old = new
            pdold = pd
            cold = cols
            checked += 1
        final = interval["state"]
        identity(final)
        close(final["pipe"]["pipe"]["time_s"], end, atol=1e-13, rtol=0.0)
        close(interval["conservation_residuals"], interval_residual, atol=1e-6, rtol=2e-6)
        close(interval["ambient_heat_j"], interval_ambient, atol=1e-8, rtol=2e-6)
        close(interval["entropy_production_j_k"], interval_entropy, atol=2e-6, rtol=2e-6)
        # Synchronize only carried clocks; physical stocks must match the last halfstep.
        last = json.loads(json.dumps(old))
        last["pipe"]["pipe"]["time_s"] = final["pipe"]["pipe"]["time_s"]
        last["pipe"]["pipe"]["next_step_s"] = final["pipe"]["pipe"]["next_step_s"]
        for i in range(2):
            last["vessels"][i]["time_s"] = final["vessels"][i]["time_s"]
            last["vessels"][i]["next_step_s"] = final["vessels"][i]["next_step_s"]
        if last != final:
            raise ValueError("Published interval physical state differs")
        old = final
    return {
        "accepted": True,
        "checked_substeps": checked,
        "reference_column_quadrature_order": 32,
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
