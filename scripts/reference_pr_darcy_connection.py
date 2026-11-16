"""Independent scalar Darcy equation and conservative packet audit."""

import math

import numpy as np
from check_stratified_pr_flow import close
from check_stratified_pr_network import reference_darcy


def reconstruct(model, nodes, data, commands, saved, fluid):
    from check_pr_equilibrium_transport import reference_sound
    from check_stratified_pr import Reference

    connections = model.get("darcy_connections", [])
    expected = {c["name"] for c in connections if commands[c["name"]]}
    if set(saved) != expected:
        raise ValueError("Darcy connection registry differs")
    nc = len(next(iter(data.values()))["n"])
    net = {name: np.zeros(nc + 2) for name in nodes}
    for c in connections:
        if c["name"] not in expected:
            continue
        a, b = c["source"], c["target"]
        row = saved[c["name"]]
        velocity = row["velocity_m_s"]
        donor = a if velocity >= 0 else b
        d = data[donor]
        if len(data[a]["pn"]) != 1 or len(data[b]["pn"]) != 1:
            raise ValueError("Darcy line outside single-phase envelope")
        dp = data[a]["p"] - data[b]["p"]
        if not c["allow_reverse"]:
            dp = max(0, dp)
        loss, reynolds = reference_darcy(
            velocity,
            d["rho"],
            c["length_m"],
            c["diameter_m"],
            c["dynamic_viscosity_pa_s"],
            c["roughness_m"],
        )
        loss += 0.5 * c["local_loss_coefficient"] * d["rho"] * velocity * abs(velocity)
        close(loss, dp, atol=1e-7, rtol=2e-7)
        if (
            abs(velocity) > c["maximum_velocity_m_s"]
            or abs(velocity)
            > c["maximum_mach_number"] * reference_sound(fluid, d["p"], d["t"], d["x"][0])
            or abs(dp) > c["maximum_pressure_drop_fraction"] * min(data[a]["p"], data[b]["p"])
        ):
            raise ValueError("Darcy line outside declared operating envelope")
        volume = math.pi * c["diameter_m"] ** 2 / 4 * velocity
        n = volume * d["n"] / nodes[donor]["volume_m3"]
        energy = (
            volume
            * (d["internal"] + d["p"] * nodes[donor]["volume_m3"] + d["potential"])
            / nodes[donor]["volume_m3"]
        )
        close(row["volume_rate_m3_s"], volume, atol=1e-14, rtol=2e-7)
        close(row["component_rates_mol_s"], n, atol=1e-10, rtol=2e-7)
        close(row["total_energy_rate_w"], energy, atol=1e-7, rtol=2e-7)
        close(row["reynolds_number"], reynolds, atol=1e-7, rtol=2e-7)
        x = d["x"][0]
        receiver = b if donor == a else a
        if volume == 0:
            if row["delivery_temperature_k"] is not None:
                raise ValueError("Zero flow has no delivery trace")
        else:
            delivered = Reference(fluid, np.flatnonzero(x > 0)).state(
                data[receiver]["p"], row["delivery_temperature_k"], x, "stable"
            )
            close(delivered["h"], float(d["pn"] @ d["h"] / d["pn"].sum()), atol=1e-5, rtol=2e-7)
        packet = np.r_[n, 0.0, energy]
        net[a] -= packet
        net[b] += packet
    return net
