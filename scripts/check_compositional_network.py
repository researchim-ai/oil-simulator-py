"""Independently reconstruct coupled PR journals without solving network steps."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compositional_geometry_audit import check_layout, feedback, weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.component_energy import ComponentEnergyStream
from surface.compositional_network import CompositionalNetwork, CompositionalNetworkState
from surface.operation import _digest


def phase_mask(properties, selection, component, minimum, names):
    mask = np.ones(len(properties["n"]))
    if selection in ("densest", "lightest"):
        i = np.argmax(properties["rho"]) if selection == "densest" else np.argmin(properties["rho"])
        mask[:] = 0
        mask[i] = 1
    elif selection == "component_rich":
        column = properties["x"][:, names.index(component)]
        i = np.argmax(column)
        mask[:] = 0
        mask[i] = float(column[i] >= minimum)
    return mask


def transfer(
    body,
    vessel,
    conductance,
    dp,
    opening,
    selection,
    component,
    minimum,
    names,
    masses,
    height=None,
    diameter=None,
):
    mask = phase_mask(body, selection, component, minimum, names)
    volume = conductance * opening * max(dp, 0) * weights(body, vessel, height, diameter) * mask
    molar = volume / body["vm"]
    components = molar @ body["x"]
    energy = float(
        molar @ body["h"] + components @ masses * vessel.gravity_m_s2 * vessel.elevation_m
    )
    entropy = float(molar @ body["s"])
    return np.r_[components, energy, entropy, volume.sum()]


def check(config, path):
    model = CompositionalNetwork.from_dict(config["system"])
    data = json.loads(path.read_text())
    payload = dict(data)
    checksum = payload.pop("checksum")
    if (
        checksum != _digest(payload)
        or data["next_period"] != len(config["periods"])
        or len(data["records"]) != len(config["periods"])
    ):
        raise ValueError("Incomplete or checksum-invalid network report")
    configured = model.initialize(config["initial"])
    expected = None
    nc = len(model.fluid.names)
    masses = model.fluid.molar_masses.cpu().numpy()
    maxima = {
        k: 0.0
        for k in (
            "component_mol",
            "fluid_energy_j",
            "wall_energy_j",
            "volume_m3",
            "log_fugacity",
            "link_component_mol_s",
            "link_energy_w",
            "global_energy_j",
        )
    }
    signs = {c.name: set() for c in model.connections}
    phase_counts = set()
    steps = failed = 0
    for record, period in zip(data["records"], config["periods"]):
        if expected is None:
            # CPU reconstruction and a CUDA initialization can differ by a few
            # ulps in N/U. Compare physical initial conditions, then require
            # exact serialized continuity between all actual report periods.
            supplied = CompositionalNetworkState.from_dict(record["initial_state"])
            model.inspect(supplied)
            if supplied.next_dt_s != configured.next_dt_s:
                raise ValueError("Initial network timestep differs")
            for (_, a), (_, b) in zip(configured.vessels, supplied.vessels):
                ia, ib = a.inventory, b.inventory
                if (
                    a.model_id != b.model_id
                    or a.next_dt_s != b.next_dt_s
                    or a.wall_temperature_k != b.wall_temperature_k
                    or any(
                        getattr(ia, k) != getattr(ib, k)
                        for k in (
                            "model_id",
                            "pressure_pa",
                            "temperature_k",
                            "volume_m3",
                            "elevation_m",
                            "gravity_m_s2",
                            "time_s",
                        )
                    )
                    or not np.allclose(
                        ia.component_moles, ib.component_moles, rtol=1e-12, atol=1e-13
                    )
                    or abs(ia.internal_energy_j - ib.internal_energy_j)
                    > 1e-6 + 1e-10 * abs(ia.internal_energy_j)
                ):
                    raise ValueError("Initial physical inventory differs from configuration")
            expected = record["initial_state"]
        if record["initial_state"] != expected:
            raise ValueError("Network history/initial conditions differ")
        initial = CompositionalNetworkState.from_dict(expected)
        model.inspect(initial)
        previous = {}
        for name, v, (_, state) in zip(model.names, model.vessels, initial.vessels):
            q = v.inspect(state)
            previous[name] = {
                "N": np.array(state.inventory.component_moles),
                "E": state.inventory.total_energy_j(model.fluid),
                "S": sum(state.inventory.component_moles) * float(q["molar_entropy_j_mol_k"]),
                "Tw": state.wall_temperature_k,
            }
        first = {
            n: {k: (x.copy() if isinstance(x, np.ndarray) else x) for k, x in row.items()}
            for n, row in previous.items()
        }
        feeds = {n: np.zeros(nc + 2) for n in model.names}
        for name, specs in period.get("feeds", {}).items():
            vessel = model.vessels[model.names.index(name)]
            for spec in specs:
                if set(spec) == {"from_pt"}:
                    args = dict(spec["from_pt"])
                    args.setdefault("gravity_m_s2", vessel.gravity_m_s2)
                    args.setdefault("elevation_m", vessel.elevation_m)
                    stream = ComponentEnergyStream.from_pt(model.fluid, **args)
                else:
                    stream = ComponentEnergyStream.from_dict(spec["stream"])
                feeds[name][:nc] += stream.component_molar_rates_mol_s
                feeds[name][nc] += stream.total_energy_rate_w
                if stream.molar_rate_mol_s:
                    q = stream.equilibrate(
                        model.fluid,
                        temperature_guess_k=expected["vessels"][name]["inventory"]["temperature_k"],
                    )
                    feeds[name][nc + 1] += stream.molar_rate_mol_s * float(
                        q["molar_entropy_j_mol_k"]
                    )
        ext_n = np.zeros(nc)
        ext_e = 0.0
        elapsed = 0.0
        for step in record["steps"]:
            dt = step["dt_s"]
            if (
                dt <= 0
                or abs(step["time_s"] - initial.vessels[0][1].inventory.time_s - elapsed) > 1e-11
            ):
                raise ValueError("Noncontinuous network step time")
            bodies = {}
            for name, vessel in zip(model.names, model.vessels):
                row = step["vessels"][name]
                p, t = row["pressure_pa"], row["temperature_k"]
                n = np.array(row["phase_moles"])
                x = np.array(row["phase_mole_fractions"])
                target_vm = np.array(row["phase_volume_fractions"]) * vessel.volume_m3 / n
                phases = []
                for xx, vm in zip(x, target_vm):
                    roots = [model.fluid.homogeneous(p, t, xx, tag) for tag in ("liquid", "vapor")]
                    phases.append(
                        min(roots, key=lambda q: abs(float(q["molar_volume_m3_mol"]) - vm))
                    )
                props = {
                    key: np.array([float(q[key]) for q in phases])
                    for key in (
                        "molar_volume_m3_mol",
                        "molar_enthalpy_j_mol",
                        "molar_internal_energy_j_mol",
                        "molar_entropy_j_mol_k",
                        "density_kg_m3",
                    )
                }
                mu = np.log(np.maximum(x, 1e-300)) + np.array(
                    [q["log_fugacity_coefficients"].cpu().numpy() for q in phases]
                )
                active = x.sum(0) > 0
                chemical = float(np.max(abs(mu[:, active] - mu[-1, active])))
                volume = float(n @ props["molar_volume_m3_mol"])
                maxima["volume_m3"] = max(maxima["volume_m3"], abs(volume - vessel.volume_m3))
                maxima["log_fugacity"] = max(maxima["log_fugacity"], chemical)
                if (
                    abs(volume - vessel.volume_m3) > 1e-10
                    or chemical > 1e-8
                    or np.any(n <= 0)
                    or np.max(abs(x.sum(1) - 1)) > 1e-10
                ):
                    raise ValueError("Independent phase volume/fugacity failed")
                bodies[name] = {
                    "p": p,
                    "t": t,
                    "n": n,
                    "x": x,
                    "vm": props["molar_volume_m3_mol"],
                    "h": props["molar_enthalpy_j_mol"],
                    "s": props["molar_entropy_j_mol_k"],
                    "rho": props["density_kg_m3"],
                    "N": n @ x,
                    "E": float(
                        n @ props["molar_internal_energy_j_mol"]
                        + (n @ x) @ masses * vessel.gravity_m_s2 * vessel.elevation_m
                    ),
                    "S": float(n @ props["molar_entropy_j_mol_k"]),
                    "Tw": row["wall_temperature_k"],
                }
                phase_counts.add(len(n))
                check_layout(row.get("geometry"), bodies[name], vessel)
            net = {n: np.zeros(nc + 3) for n in model.names}
            for c in model.connections:
                a, b = bodies[c.source], bodies[c.target]
                opening = period.get("connection_openings", {}).get(c.name)
                dp = a["p"] - b["p"]
                va = model.vessels[model.names.index(c.source)]
                vb = model.vessels[model.names.index(c.target)]
                if opening is None:
                    opening = c.opening
                    if c.level_control is not None:
                        opening = feedback(
                            c.level_control,
                            a if c.control_side == "source" else b,
                            va if c.control_side == "source" else vb,
                            model.fluid.names,
                        )[0]
                forward = transfer(
                    a,
                    va,
                    c.volume_conductance_m3_s_pa,
                    dp,
                    opening,
                    c.source_selection,
                    c.source_component,
                    c.minimum_mole_fraction,
                    model.fluid.names,
                    masses,
                    c.source_height_m,
                    c.source_diameter_m,
                )
                reverse = transfer(
                    b,
                    vb,
                    c.volume_conductance_m3_s_pa,
                    -dp,
                    opening * int(c.allow_reverse),
                    c.target_selection,
                    c.target_component,
                    c.minimum_mole_fraction,
                    model.fluid.names,
                    masses,
                    c.target_height_m,
                    c.target_diameter_m,
                )
                flow = forward - reverse
                for body, part in ((a, forward), (b, reverse)):
                    if part[-1] and abs(dp) / body["p"] > c.maximum_pressure_drop_fraction + 1e-12:
                        raise ValueError("Connection outside declared pressure-drop domain")
                net[c.source] -= flow
                net[c.target] += flow
                logged = step["connections"][c.name]
                if abs(logged["opening"] - opening) > 1e-9:
                    raise ValueError("Connection opening differs from independent feedback")
                for side, body, vessel in (("source", a, va), ("target", b, vb)):
                    height = getattr(c, side + "_height_m")
                    if (
                        height is not None
                        and np.max(
                            abs(
                                weights(body, vessel, height, getattr(c, side + "_diameter_m"))
                                - logged[side + "_pickup_area_fractions"]
                            )
                        )
                        > 1e-9
                    ):
                        raise ValueError("Connection aperture exposure differs")
                if c.level_control is not None:
                    _, level, present = feedback(
                        c.level_control,
                        a if c.control_side == "source" else b,
                        va if c.control_side == "source" else vb,
                        model.fluid.names,
                    )
                    if (
                        abs(logged["controlled_level_m"] - level) > 1e-9
                        or logged["controlled_phase_present"] != present
                    ):
                        raise ValueError("Connection level sensor differs")
                en = float(np.max(abs(flow[:nc] - logged["component_molar_rates_mol_s"])))
                ee = abs(flow[nc] - logged["energy_rate_w"])
                maxima["link_component_mol_s"] = max(maxima["link_component_mol_s"], en)
                maxima["link_energy_w"] = max(maxima["link_energy_w"], ee)
                if (
                    en > 1e-10
                    or ee > 1e-6
                    or abs(flow[nc + 1] - logged["entropy_rate_w_k"]) > 1e-8
                    or abs(flow[nc + 2] - logged["donor_volume_rate_m3_s"]) > 1e-12
                ):
                    raise ValueError("Logged link differs from independent upwind law")
                if abs(flow[:nc].sum()) > 1e-14:
                    signs[c.name].add(1 if flow[:nc].sum() > 0 else -1)
            for name, vessel in zip(model.names, model.vessels):
                b = bodies[name]
                old = previous[name]
                outlet = np.zeros(nc + 3)
                for port in vessel.outlets:
                    opening = period.get("outlet_openings", {}).get(name, {}).get(port.name)
                    if opening is None:
                        opening = (
                            port.opening
                            if port.pressure_setpoint_pa is None
                            else np.clip(
                                port.opening_bias
                                + port.opening_gain_per_pa * (b["p"] - port.pressure_setpoint_pa),
                                0,
                                1,
                            )
                        )
                        if port.level_control is not None:
                            opening = feedback(port.level_control, b, vessel, model.fluid.names)[0]
                    dp = b["p"] - port.back_pressure_pa
                    flow = transfer(
                        b,
                        vessel,
                        port.volume_conductance_m3_s_pa,
                        dp,
                        opening,
                        port.selection,
                        port.component,
                        port.minimum_mole_fraction,
                        model.fluid.names,
                        masses,
                        port.height_m,
                        port.diameter_m,
                    )
                    logged = step["vessels"][name]["ports"][port.name]
                    if abs(logged["opening"] - opening) > 1e-9:
                        raise ValueError("Outlet opening differs from independent feedback")
                    if (
                        port.height_m is not None
                        and np.max(
                            abs(
                                weights(b, vessel, port.height_m, port.diameter_m)
                                - logged["pickup_area_fractions"]
                            )
                        )
                        > 1e-9
                    ):
                        raise ValueError("Outlet aperture exposure differs")
                    if flow[-1] and dp / b["p"] > port.maximum_pressure_drop_fraction + 1e-12:
                        raise ValueError("External outlet outside declared domain")
                    outlet += flow
                heat = period.get("heat_w", {}).get(name, 0.0)
                utility = period.get("utility_temperature_k", {}).get(name, b["t"])
                exchange = ambient = wall_error = wall_entropy = 0.0
                if vessel.wall:
                    w = vessel.wall
                    exchange = w.fluid_ua_w_k * (b["Tw"] - b["t"])
                    ambient = w.ambient_ua_w_k * (w.ambient_temperature_k - b["Tw"])
                    wall_error = w.heat_capacity_j_k * (b["Tw"] - old["Tw"]) - dt * (
                        ambient - exchange
                    )
                    wall_entropy = w.heat_capacity_j_k * math.log(b["Tw"] / old["Tw"])
                mass_error = float(
                    np.max(
                        abs(
                            b["N"]
                            - old["N"]
                            - dt * (feeds[name][:nc] + net[name][:nc] - outlet[:nc])
                        )
                    )
                )
                energy_error = (
                    b["E"]
                    - old["E"]
                    - dt * (feeds[name][nc] + net[name][nc] - outlet[nc] + heat + exchange)
                )
                production = (
                    b["S"]
                    - old["S"]
                    + wall_entropy
                    - dt
                    * (
                        feeds[name][nc + 1]
                        + net[name][nc + 1]
                        - outlet[nc + 1]
                        + heat / utility
                        + (ambient / vessel.wall.ambient_temperature_k if vessel.wall else 0)
                    )
                )
                maxima["component_mol"] = max(maxima["component_mol"], mass_error)
                maxima["fluid_energy_j"] = max(maxima["fluid_energy_j"], abs(energy_error))
                maxima["wall_energy_j"] = max(maxima["wall_energy_j"], abs(wall_error))
                if (
                    mass_error > 1e-8
                    or abs(energy_error) > 1e-4
                    or abs(wall_error) > 1e-4
                    or production < -1e-6
                    or heat * (utility - b["t"]) < -1e-8
                ):
                    raise ValueError("Independent local mass/energy/entropy balance failed")
                for spec in period.get("feeds", {}).get(name, []):
                    pressure = spec.get("from_pt", spec.get("stream", {}))["pressure_pa"]
                    if b["p"] > pressure + 1e-5:
                        raise ValueError("Passive feed pressure below vessel")
                ext_n += dt * (feeds[name][:nc] - outlet[:nc])
                ext_e += dt * (feeds[name][nc] - outlet[nc] + heat + ambient)
                previous[name] = b
            elapsed = math.fsum((elapsed, dt))
            steps += 1
        if abs(elapsed - period["duration_s"]) > 1e-11:
            raise ValueError("Network report interval not covered")
        final = CompositionalNetworkState.from_dict(record["state"])
        model.inspect(final)
        total_n = np.zeros(nc)
        total_e = 0.0
        for name, vessel, (_, state) in zip(model.names, model.vessels, final.vessels):
            b = previous[name]
            a = first[name]
            if (
                not np.allclose(b["N"], state.inventory.component_moles, rtol=1e-10, atol=1e-10)
                or abs(b["E"] - state.inventory.total_energy_j(model.fluid)) > 1e-4
                or abs(b["p"] - state.inventory.pressure_pa) > 1e-5
                or abs(b["t"] - state.inventory.temperature_k) > 1e-8
            ):
                raise ValueError("Final state differs from journal")
            total_n += b["N"] - a["N"]
            total_e += b["E"] - a["E"]
            if vessel.wall:
                total_e += vessel.wall.energy(b["Tw"]) - vessel.wall.energy(a["Tw"])
        maxima["global_energy_j"] = max(maxima["global_energy_j"], abs(total_e - ext_e))
        if np.max(abs(total_n - ext_n)) > 1e-8 or abs(total_e - ext_e) > 1e-4:
            raise ValueError("Independent global network ledger failed")
        expected = record["state"]
        failed += record["failed_steps"]
    if expected != data["state"]:
        raise ValueError("Checkpoint differs from final journal state")
    return data, {
        "accepted_steps": steps,
        "failed_steps": failed,
        "phase_counts": sorted(phase_counts),
        "connection_directions": {k: sorted(v) for k, v in signs.items()},
        "max_residuals": maxima,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("cuda", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    a, ma = check(config, args.cpu)
    b, mb = check(config, args.cuda)
    pressure = temperature = component = wall = 0.0
    for name, sa in a["state"]["vessels"].items():
        sb = b["state"]["vessels"][name]
        ia = sa["inventory"]
        ib = sb["inventory"]
        pressure = max(pressure, abs(ia["pressure_pa"] - ib["pressure_pa"]))
        temperature = max(temperature, abs(ia["temperature_k"] - ib["temperature_k"]))
        component = max(
            component, float(np.max(abs(np.array(ia["component_moles"]) - ib["component_moles"])))
        )
        if sa["wall_temperature_k"] is not None:
            wall = max(wall, abs(sa["wall_temperature_k"] - sb["wall_temperature_k"]))
    if pressure > 1e-3 or temperature > 1e-6 or component > 1e-9 or wall > 1e-6:
        raise ValueError("CPU/CUDA network states disagree")
    result = {
        "cpu": ma,
        "cuda": mb,
        "max_pressure_difference_pa": pressure,
        "max_temperature_difference_k": temperature,
        "max_component_difference_mol": component,
        "max_wall_temperature_difference_k": wall,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
