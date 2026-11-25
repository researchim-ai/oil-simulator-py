"""Scalar/NumPy machine rates for the conservative PR Radau audit.

Uses the existing independent card, PS/PH and throat references. No runtime
fluid decoder, machine evaluator, flux or Newton is used here.
"""

import numpy as np
from reference_pr_compressors import audit_compressors
from reference_pr_drives import machine_model, rotor_energy
from reference_pr_pump_connections import audit_pump_connections
from reference_pr_pumps import audit_pumps
from reference_pr_restrictions import audit_restrictions


def hosts(model):
    units = {
        unit.get("name", (unit.get("card") or {}).get("name")): unit
        for family in ("pumps", "pump_connections", "compressors")
        for unit in model.get(family, [])
    }
    return {
        d["machine"]: units[d["machine"]].get(
            "node",
            units[d["machine"]].get("target", units[d["machine"]].get("card", {}).get("end")),
        )
        for d in model.get("drives", [])
    }


def stored_energy(model, rotors):
    rotor_energy({"rotors": list(rotors.items())}, model)
    result = {}
    host = hosts(model)
    for d in model.get("drives", []):
        name = d["machine"]
        node = host[name]
        result[node] = result.get(node, 0.0) + 0.5 * d["drive"]["inertia_kg_m2"] * rotors[name] ** 2
    return result


def stage_rates(model, data, rotors, saved, fluid, commands):
    state = {"rotors": list(rotors.items())}
    adjusted = machine_model(model, state, state)  # instantaneous speed, not midpoint
    openings = commands["openings"]
    forces, pump_powers = audit_pumps(adjusted, data, openings, saved["pumps"])
    pump_net, _ = audit_pump_connections(adjusted, data, openings, saved["pump_connections"], fluid)
    compressor_net, _ = audit_compressors(adjusted, data, openings, saved["compressors"], fluid)
    valve_net = audit_restrictions(
        adjusted,
        data,
        openings,
        commands.get("restriction_openings", {}),
        saved["restrictions"],
        fluid,
    )
    net = {name: np.zeros(len(fluid.names) + 2) for name in data}
    for packets in (pump_net, compressor_net, valve_net):
        for name, value in packets.items():
            net[name] += value
    for name, force in forces.items():
        net[name][-2] += force
        net[name][-1] += pump_powers[name]
    machines = {
        name: row
        for family in ("pumps", "pump_connections", "compressors")
        for name, row in saved[family].items()
    }
    driven = {d["machine"] for d in model.get("drives", [])}
    work = sum(row["shaft_power_w"] for name, row in machines.items() if name not in driven)
    environment = entropy_environment = 0.0
    angular_rates, drive_rates = {}, {}
    host = hosts(model)
    for d in model.get("drives", []):
        name, physics = d["machine"], d["drive"]
        speed = rotors[name]
        row = machines[name]
        running = openings[name] and not row.get("valve_closed", False)
        power = row["shaft_power_w"] if running else 0.0
        load = row.get("load_torque_nm", power / max(speed, 1e-30)) if running else 0.0
        torque = commands.get("drive_torque_nm", {}).get(name, d["motor_torque_nm"])
        if not -physics["max_braking_torque_nm"] <= torque <= physics["max_motor_torque_nm"]:
            raise ValueError("Independent motor torque outside limits")
        signed = any(
            u.get("signed_map") is not None
            and u.get("name", (u.get("card") or {}).get("name")) == name
            for family in ("pumps", "pump_connections", "compressors")
            for u in model.get(family, [])
        )
        net_torque = torque - load - physics["viscous_drag_nm_s"] * speed
        if not signed and speed <= 0:
            net_torque = max(0.0, net_torque)
        motor = torque * speed
        electrical = (
            motor / physics["motor_efficiency"]
            if motor >= 0
            else motor * physics["regeneration_efficiency"]
        )
        for key, value in (
            ("max_electrical_draw_w", max(0.0, electrical)),
            ("max_electrical_export_w", max(0.0, -electrical)),
        ):
            if physics[key] is not None and value > physics[key]:
                raise ValueError("Independent electrical rating exceeded")
        losses = electrical - motor + physics["viscous_drag_nm_s"] * speed**2
        sink = d["heat_sink_node"]
        if sink is None:
            environment -= losses
            entropy_environment += losses / d["ambient_temperature_k"]
        else:
            net[sink][-1] += losses
        kinetic_rate = speed * net_torque
        net[host[name]][-1] += kinetic_rate
        angular_rates[name] = net_torque / physics["inertia_kg_m2"]
        work += electrical
        drive_rates[name] = {
            "motor_work_j": motor,
            "shaft_work_j": power,
            "drag_heat_j": physics["viscous_drag_nm_s"] * speed**2,
            "electrical_energy_j": electrical,
            "motor_heat_j": electrical - motor,
            "heat_to_environment_j": losses if sink is None else 0.0,
            "heat_to_node_j": losses if sink is not None else 0.0,
            "kinetic_rate": kinetic_rate,
        }
    return net, angular_rates, work, environment, entropy_environment, drive_rates
