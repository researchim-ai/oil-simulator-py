"""Rebuild separator balances/port laws independently from saved CPU/CUDA states."""

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
from surface.compositional_separator import CompositionalSeparator, CompositionalSeparatorState
from surface.operation import _digest


def check(config, path):
    model = CompositionalSeparator.from_dict(config["system"])
    data = json.loads(path.read_text())
    payload = dict(data)
    checksum = payload.pop("checksum")
    if checksum != _digest(payload) or data["next_period"] != len(config["periods"]):
        raise ValueError("Incomplete or checksum-invalid separator report")
    if len(data["records"]) != len(config["periods"]):
        raise ValueError("Period count differs from config")
    max_mass = max_energy = max_wall = max_volume = max_mu = max_port_n = max_port_e = 0.0
    phases = set()
    accepted = 0
    failed = 0
    prior = None
    for record, period in zip(data["records"], config["periods"]):
        initial = CompositionalSeparatorState.from_dict(record["initial_state"])
        model.inspect(initial)
        if prior is not None and record["initial_state"] != prior:
            raise ValueError("Noncontinuous separator periods")
        n = np.array(initial.inventory.component_moles)
        energy = initial.inventory.total_energy_j(model.fluid)
        q = model.inspect(initial)
        entropy = n.sum() * float(q["molar_entropy_j_mol_k"])
        tw = initial.wall_temperature_k
        feed_n = np.zeros(len(n))
        feed_e = feed_s = 0.0
        feed_pressure = math.inf
        for spec in period.get("feeds", []):
            if "from_pt" in spec:
                args = dict(spec["from_pt"])
                args.setdefault("gravity_m_s2", model.gravity_m_s2)
                args.setdefault("elevation_m", model.elevation_m)
                stream = ComponentEnergyStream.from_pt(model.fluid, **args)
            else:
                stream = ComponentEnergyStream.from_dict(spec["stream"])
            feed_n += stream.component_molar_rates_mol_s
            feed_e += stream.total_energy_rate_w
            if stream.molar_rate_mol_s:
                eq = stream.equilibrate(
                    model.fluid, temperature_guess_k=initial.inventory.temperature_k
                )
                feed_s += stream.molar_rate_mol_s * float(eq["molar_entropy_j_mol_k"])
                feed_pressure = min(feed_pressure, stream.pressure_pa)
        times = []
        for step in record["steps"]:
            dt = step["dt_s"]
            p = step["pressure_pa"]
            t = step["temperature_k"]
            nt = np.array(step["phase_moles"])
            x = np.array(step["phase_mole_fractions"])
            values = [model.fluid.homogeneous(p, t, row) for row in x]
            vm = np.array([float(row["molar_volume_m3_mol"]) for row in values])
            h = np.array([float(row["molar_enthalpy_j_mol"]) for row in values])
            u = np.array([float(row["molar_internal_energy_j_mol"]) for row in values])
            s = np.array([float(row["molar_entropy_j_mol_k"]) for row in values])
            rho = np.array([float(row["density_kg_m3"]) for row in values])
            total_n = nt @ x
            total_mass = float(total_n @ model.fluid.molar_masses.numpy())
            stored = float(nt @ u) + total_mass * model.gravity_m_s2 * model.elevation_m
            volume = float(nt @ vm)
            holdup = nt * vm / model.volume_m3
            body = {"n": nt, "vm": vm, "rho": rho, "x": x}
            check_layout(step.get("geometry"), body, model)
            active = total_n > 0
            mu = (
                np.log(x[:, active])
                + np.array([row["log_fugacity_coefficients"].numpy() for row in values])[:, active]
            )
            mu_error = float(np.max(abs(mu - mu[-1])))
            port_n = np.zeros(len(n))
            port_e = port_s = 0.0
            for port in model.outlets:
                reported = step["ports"][port.name]
                opening = period.get("openings", {}).get(port.name)
                if opening is None:
                    opening = (
                        port.opening
                        if port.pressure_setpoint_pa is None
                        else np.clip(
                            port.opening_bias
                            + port.opening_gain_per_pa * (p - port.pressure_setpoint_pa),
                            0,
                            1,
                        )
                    )
                    if port.level_control is not None:
                        opening = feedback(port.level_control, body, model, model.fluid.names)[0]
                mask = np.ones(len(nt))
                if port.selection in ("densest", "lightest"):
                    index = int(rho.argmax() if port.selection == "densest" else rho.argmin())
                    mask[:] = 0
                    mask[index] = 1
                elif port.selection == "component_rich":
                    column = x[:, model.fluid.names.index(port.component)]
                    index = int(column.argmax())
                    mask[:] = 0
                    if column[index] >= port.minimum_mole_fraction:
                        mask[index] = 1
                vdot = (
                    port.volume_conductance_m3_s_pa
                    * opening
                    * max(0.0, p - port.back_pressure_pa)
                    * (
                        weights(body, model, port.height_m, port.diameter_m)
                        if port.height_m is not None
                        else holdup
                    )
                    * mask
                )
                rate = vdot / vm
                component = rate @ x
                mass = float(component @ model.fluid.molar_masses.numpy())
                power = float(rate @ h) + mass * model.gravity_m_s2 * model.elevation_m
                entr = float(rate @ s)
                if (
                    port.height_m is not None
                    and np.max(
                        abs(
                            weights(body, model, port.height_m, port.diameter_m)
                            - reported["pickup_area_fractions"]
                        )
                    )
                    > 1e-9
                ):
                    raise ValueError("Saved outlet aperture exposure differs")
                if port.level_control is not None:
                    _, level, present = feedback(port.level_control, body, model, model.fluid.names)
                    if (
                        abs(reported["controlled_level_m"] - level) > 1e-9
                        or reported["controlled_phase_present"] != present
                    ):
                        raise ValueError("Saved outlet level sensor differs")
                discrepancy = float(
                    np.max(abs(component - reported["component_molar_rates_mol_s"]))
                )
                max_port_n = max(max_port_n, discrepancy)
                max_port_e = max(max_port_e, abs(power - reported["energy_rate_w"]))
                if (
                    discrepancy > 1e-9
                    or abs(power - reported["energy_rate_w"]) > 1e-6
                    or abs(opening - reported["opening"]) > 1e-10
                ):
                    raise ValueError("Saved outlet differs from independently reconstructed law")
                port_n += component
                port_e += power
                port_s += entr
            new_tw = step["wall_temperature_k"]
            exchange = model.wall.fluid_ua_w_k * (new_tw - t) if model.wall else 0.0
            ambient = (
                model.wall.ambient_ua_w_k * (model.wall.ambient_temperature_k - new_tw)
                if model.wall
                else 0.0
            )
            mass_error = float(np.max(abs(total_n - n - dt * (feed_n - port_n))))
            energy_error = (
                stored - energy - dt * (feed_e - port_e + period.get("heat_w", 0.0) + exchange)
            )
            wall_error = (
                model.wall.heat_capacity_j_k * (new_tw - tw) - dt * (ambient - exchange)
                if model.wall
                else 0.0
            )
            wall_entropy = (
                model.wall.heat_capacity_j_k * math.log(new_tw / tw) if model.wall else 0.0
            )
            production = (
                float(nt @ s)
                - entropy
                + wall_entropy
                - dt
                * (
                    feed_s
                    - port_s
                    + period.get("heat_w", 0.0) / period.get("utility_temperature_k", 1.0)
                    + (ambient / model.wall.ambient_temperature_k if model.wall else 0.0)
                )
            )
            if (
                mass_error > 1e-8
                or abs(energy_error) > 1e-5
                or abs(wall_error) > 1e-5
                or abs(volume - model.volume_m3) > 1e-10
                or mu_error > 1e-8
                or production < -1e-6
                or p > feed_pressure + 1e-5
            ):
                raise ValueError("Independently reconstructed balance/equilibrium/domain failed")
            max_mass = max(max_mass, mass_error)
            max_energy = max(max_energy, abs(energy_error))
            max_wall = max(max_wall, abs(wall_error))
            max_volume = max(max_volume, abs(volume - model.volume_m3))
            max_mu = max(max_mu, mu_error)
            n, energy, entropy, tw = total_n, stored, float(nt @ s), new_tw
            phases.add(len(nt))
            times.append(dt)
            accepted += 1
        if abs(math.fsum(times) - period["duration_s"]) > 1e-12:
            raise ValueError("Report interval not fully covered")
        final = CompositionalSeparatorState.from_dict(record["state"])
        model.inspect(final)
        if (
            not np.allclose(n, final.inventory.component_moles, rtol=1e-10, atol=1e-10)
            or abs(energy - final.inventory.total_energy_j(model.fluid)) > 1e-5
        ):
            raise ValueError("Final state differs from accepted step history")
        prior = record["state"]
        failed += record["failed_steps"]
    if prior != data["state"]:
        raise ValueError("Final checkpoint state differs from history")
    return data, {
        "accepted_steps": accepted,
        "failed_steps": failed,
        "phase_counts": sorted(phases),
        "max_component_residual_mol": max_mass,
        "max_fluid_energy_residual_j": max_energy,
        "max_wall_energy_residual_j": max_wall,
        "max_volume_residual_m3": max_volume,
        "max_log_fugacity_error": max_mu,
        "max_port_component_residual_mol_s": max_port_n,
        "max_port_energy_residual_w": max_port_e,
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
    ia = a["state"]["inventory"]
    ib = b["state"]["inventory"]
    if (
        not np.allclose(ia["component_moles"], ib["component_moles"], rtol=1e-9, atol=1e-10)
        or abs(ia["pressure_pa"] - ib["pressure_pa"]) > 1e-3
        or abs(ia["temperature_k"] - ib["temperature_k"]) > 1e-6
    ):
        raise ValueError("CPU/CUDA final states disagree")
    result = {
        "cpu": ma,
        "cuda": mb,
        "pressure_difference_pa": abs(ia["pressure_pa"] - ib["pressure_pa"]),
        "temperature_difference_k": abs(ia["temperature_k"] - ib["temperature_k"]),
        "max_component_difference_mol": float(
            np.max(abs(np.array(ia["component_moles"]) - ib["component_moles"]))
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
