"""Independent shaft audit: NumPy cards and ODE-integrated isolated motion."""

import copy
import math

from check_stratified_pr_flow import close
from scipy.integrate import solve_ivp


def rotor_energy(state, model):
    configured = {d["machine"]: d for d in model.get("drives", [])}
    raw = state.get("rotors", [])
    speeds = dict(raw)
    if len(raw) != len(speeds) or set(speeds) != set(configured):
        raise ValueError("Independent rotor inventory registry differs")
    for name, speed in speeds.items():
        if (
            not math.isfinite(speed)
            or not 0 <= speed <= configured[name]["drive"]["max_speed_rad_s"]
        ):
            raise ValueError("Independent rotor speed outside bounds")
    return sum(
        0.5 * configured[name]["drive"]["inertia_kg_m2"] * speed**2
        for name, speed in speeds.items()
    )


def machine_model(model, before, after):
    rotor_energy(before, model)
    rotor_energy(after, model)
    adjusted = copy.deepcopy(model)
    old, new = dict(before.get("rotors", [])), dict(after.get("rotors", []))
    for d in model.get("drives", []):
        name = d["machine"]
        speed = (old[name] + new[name]) / 2
        for unit in [*adjusted.get("pumps", []), *adjusted.get("pump_connections", [])]:
            if unit["card"]["name"] == name:
                unit["card"]["speed_ratio"] = speed / d["nominal_speed_rad_s"]
        for unit in adjusted.get("compressors", []):
            if unit["name"] == name:
                unit["speed_rad_s"] = speed
    return adjusted


def initial_rotors(model, state):
    rotor_energy(state, model)
    actual = dict(state.get("rotors", []))
    for d in model.get("drives", []):
        speed = d["initial_speed_rad_s"]
        if speed is None:
            pumps = [
                u
                for u in [*model.get("pumps", []), *model.get("pump_connections", [])]
                if u["card"]["name"] == d["machine"]
            ]
            speed = (
                d["nominal_speed_rad_s"] * pumps[0]["card"]["speed_ratio"]
                if pumps
                else next(
                    u["speed_rad_s"] for u in model["compressors"] if u["name"] == d["machine"]
                )
            )
        close(actual[d["machine"]], speed, atol=1e-12, rtol=1e-12)


def audit_drives(
    model, before, after, openings, commands, dt, saved, pump_powers, compressor_records
):
    configured = {d["machine"]: d for d in model.get("drives", [])}
    if set(saved) != set(configured):
        raise ValueError("Independent drive report registry differs")
    old, new = dict(before.get("rotors", [])), dict(after.get("rotors", []))
    if set(commands) - set(configured):
        raise ValueError("Unknown motor torque command")
    pump_names = {u["card"]["name"]: u["node"] for u in model.get("pumps", [])}
    connection_names = {u["card"]["name"] for u in model.get("pump_connections", [])}
    work_correction = heat_loss = entropy = 0.0
    heat_sources = {}
    for name, d in configured.items():
        p = d["drive"]
        omega0, omega = old[name], new[name]
        tau = commands.get(name, d["motor_torque_nm"])
        if not -p["max_braking_torque_nm"] <= tau <= p["max_motor_torque_nm"]:
            raise ValueError("Independent motor torque limit exceeded")
        mid = (omega0 + omega) / 2
        power = (
            pump_powers[pump_names[name]]
            if name in pump_names
            else compressor_records[name]["shaft_power_w"]
        )
        inertia, drag = p["inertia_kg_m2"], p["viscous_drag_nm_s"]
        closed_connection = name in connection_names and compressor_records[name]["valve_closed"]
        if openings[name] and not closed_connection:
            if name not in pump_names and name not in connection_names:
                unit = next(u for u in model["compressors"] if u["name"] == name)
                axis = unit["card"]["corrected_speed_rad_s"]
                for state, w in ((before, omega0), (after, omega)):
                    source = state["nodes"][unit["source"]]
                    corrected = w / math.sqrt(
                        source["temperature_k"] / unit["card"]["reference_temperature_k"]
                    )
                    if not axis[0] <= corrected <= axis[-1]:
                        raise ValueError("Compressor endpoint speed outside map")
                source = before["nodes"][unit["source"]]
                node = next(n for n in model["nodes"] if n["name"] == unit["source"])
                z = (
                    source["pressure_pa"]
                    * node["volume_m3"]
                    / (sum(source["component_moles"]) * 8.31446261815324 * source["temperature_k"])
                )
                if len(source["phase_roots"]) != 1 or z < unit["minimum_compressibility_factor"]:
                    raise ValueError("Compressor inlet was not dry gas before opening")
            if mid <= 0:
                raise ValueError("Running machine has no positive shaft speed")
            close(
                (inertia * (omega - omega0) - dt * (tau - power / mid - drag * mid))
                / (inertia * d["nominal_speed_rad_s"]),
                0.0,
                atol=1e-9,
                rtol=0.0,
            )
            motor = dt * tau * mid
            friction = dt * drag * mid**2
        else:
            if name in pump_names:
                node = pump_names[name]
                if any(
                    openings[link["name"]]
                    for link in model["links"]
                    if node in (link["source"], link["target"])
                ):
                    raise ValueError("Coasting pump is not hydraulically isolated")

            def rhs(t, y, tau=tau, drag=drag, inertia=inertia):
                return [(tau - drag * y[0]) / inertia, y[0], y[0] ** 2]

            def stop(t, y):
                return y[0]

            stop.terminal = True
            stop.direction = -1
            if omega0 == 0 and tau < 0:
                endpoint, angle, square = 0.0, 0.0, 0.0
            else:
                ode = solve_ivp(
                    rhs,
                    (0.0, dt),
                    [omega0, 0.0, 0.0],
                    method="DOP853",
                    rtol=1e-12,
                    atol=1e-12,
                    events=stop if tau < 0 else None,
                )
                if not ode.success:
                    raise ValueError("Independent isolated rotor ODE failed")
                endpoint, angle, square = ode.y[:, -1]
                if tau < 0 and len(ode.t_events[0]):
                    endpoint = 0.0
            close(omega, endpoint, atol=1e-8, rtol=2e-9)
            motor, friction = tau * angle, drag * square
            close(power, 0.0, atol=1e-12, rtol=0.0)
        shaft = dt * power
        kinetic = 0.5 * inertia * (omega - omega0) * (omega + omega0)
        electrical = (
            motor / p["motor_efficiency"] if motor >= 0 else motor * p["regeneration_efficiency"]
        )
        motor_heat = electrical - motor
        losses = friction + motor_heat
        production = losses / d["ambient_temperature_k"] if d["heat_sink_node"] is None else 0.0
        draw = max(0.0, tau) * max(omega0, omega) / p["motor_efficiency"]
        export = max(0.0, -tau) * max(omega0, omega) * p["regeneration_efficiency"]
        for key, value in [("max_electrical_draw_w", draw), ("max_electrical_export_w", export)]:
            if p[key] is not None and value > p[key] * (1 + 1e-12):
                raise ValueError("Independent electrical rating exceeded")
        residual = motor - shaft - friction - kinetic
        close(
            residual,
            0.0,
            atol=1e-8 + 1e-9 * (abs(motor) + abs(shaft) + abs(friction) + abs(kinetic)),
            rtol=0.0,
        )
        expected = {
            "speed_rad_s": omega,
            "load_evaluation_speed_rad_s": mid,
            "motor_torque_nm": tau,
            "shaft_work_j": shaft,
            "motor_work_j": motor,
            "drag_heat_j": friction,
            "electrical_energy_j": electrical,
            "motor_heat_j": motor_heat,
            "kinetic_energy_change_j": kinetic,
            "energy_residual_j": residual,
            "environment_entropy_j_k": production,
            "heat_to_environment_j": losses if d["heat_sink_node"] is None else 0.0,
            "heat_to_node_j": losses if d["heat_sink_node"] is not None else 0.0,
            "peak_electrical_draw_w": draw,
            "peak_electrical_export_w": export,
        }
        if set(expected) != set(saved[name]):
            raise ValueError("Drive report fields differ")
        for key, value in expected.items():
            close(saved[name][key], value, atol=1e-8, rtol=2e-7)
        work_correction += electrical - shaft
        heat_loss += losses if d["heat_sink_node"] is None else 0.0
        entropy += production
        if d["heat_sink_node"] is not None:
            node = d["heat_sink_node"]
            heat_sources[node] = heat_sources.get(node, 0.0) + losses / dt
    return work_correction, heat_loss, entropy, heat_sources
