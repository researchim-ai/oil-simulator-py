"""Independent scalar external PR boundary flux reconstruction, shared by audits."""

import numpy as np
from check_dynamic_receiver import stream_state
from check_stratified_pr import Reference
from check_stratified_pr_flow import close

from surface.component_energy import ComponentEnergyStream


def reconstruct(model, physical, f, data, saved, openings):
    nodes = {v["name"]: v for v in physical["nodes"]}
    nc = len(f.names)
    mw = np.array([c.molar_mass_kg_mol for c in f.components])
    net = {n: np.zeros(nc + 2) for n in nodes}
    entropy = 0.0
    records = {}
    for feed in model["feeds"]:
        packet = feed["stream"]
        ComponentEnergyStream(**packet).validate_fluid(f)
        n = np.asarray(packet["component_molar_rates_mol_s"])
        mass = n @ mw
        node = nodes[feed["node"]]
        if (
            packet["gravity_m_s2"] != physical["gravity_m_s2"]
            or packet["elevation_m"] != node["elevation_m"]
            or packet["velocity_m_s"] != 0
        ):
            raise ValueError("Feed physical coordinates differ")
        if n.sum() and packet["pressure_pa"] < data[feed["node"]]["p"]:
            raise ValueError("Feed pressure below receiver")
        e = (
            packet["total_energy_rate_w"]
            - mass * physical["gravity_m_s2"] * physical["reference_elevation_m"]
        )
        s = stream_state(packet, f) if n.sum() else 0.0
        net[feed["node"]] += np.r_[n, 0.0, e]
        entropy += s
        records[feed["name"]] = ("feed", n, e, s, 0.0)
    for port in model["pressure_ports"]:
        d, node = data[port["node"]], nodes[port["node"]]
        dp = d["p"] - port["pressure_pa"]
        q = openings[port["name"]] * port["volume_conductance_m3_s_pa"] * dp
        if not port["allow_reverse"]:
            q = max(0.0, q)
        if q and abs(dp) > port["maximum_pressure_drop_fraction"] * max(
            d["p"], port["pressure_pa"]
        ):
            raise ValueError("Boundary pressure law outside its domain")
        if q >= 0:
            n = q * d["n"] / node["volume_m3"]
            e = q * (d["pn"] @ d["h"] + d["potential"]) / node["volume_m3"]
            s = q * d["entropy"] / node["volume_m3"]
        else:
            x = np.asarray(port["mole_fractions"])
            ref = Reference(f, np.flatnonzero(x > 0))
            if ref.pure is not None:
                state = ref.state(port["pressure_pa"], port["temperature_k"], x, "stable")
                vm, h, se = state["vm"], state["h"], state["s"]
            else:
                state = ref.mixture.flash_pt(port["pressure_pa"], port["temperature_k"], x[x > 0])
                vm = float(x @ mw) / state["density_kg_m3"]
                h, se = state["molar_enthalpy_j_mol"], state["molar_entropy_j_mol_k"]
            n = q * x / vm
            e = q * h / vm + (n @ mw) * physical["gravity_m_s2"] * (
                node["elevation_m"] - physical["reference_elevation_m"]
            )
            s = q * se / vm
        net[port["node"]] -= np.r_[n, 0.0, e]
        entropy -= s
        records[port["name"]] = ("outward", n, e, s, q)
    if set(records) != set(saved):
        raise ValueError("Boundary port registry differs")
    for name, (orientation, n, e, s, q) in records.items():
        if saved[name]["orientation"] != orientation:
            raise ValueError("Boundary orientation differs")
        for key, value, atol in [
            ("component_rates_mol_s", n, 1e-10),
            ("total_energy_rate_w", e, 1e-6),
            ("entropy_rate_w_k", s, 1e-7),
            ("volume_rate_m3_s", q, 1e-14),
        ]:
            close(saved[name][key], value, atol=atol, rtol=2e-7)
    return {"net": net, "incoming_entropy": entropy}
