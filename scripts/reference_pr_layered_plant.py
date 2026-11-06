"""Independent physical face exchange between profile cells and finite vessels."""

import math

import numpy as np
from check_pr_momentum_vessels import port_flux
from check_stratified_pr_flow import profile
from reference_pr_layered_transport import patch, reconstruct


def exchanges(model, state, ordinary, order=24):
    native = model["layered"]
    columns = native["columns"]
    cells = state["layered"]["cells"]
    decoded = reconstruct(native, state["layered"], order=order)
    dims = columns[0]["geometry"]
    diameter = dims["diameter_m"]
    angle = dims["inclination_rad"]
    cosine = math.cos(angle)
    gravity = decoded["gravity_acceleration"]
    datum = decoded["datum"]
    mw = decoded["molar_masses"]
    nn = len(mw)
    rates = {n: np.zeros(nn + 2) for n in ordinary}
    incoming = 0.0
    reactions = {n: 0.0 for n in ordinary}
    boundary = np.zeros_like(decoded["rates"])
    nodes = {v["name"]: v for v in model["transport"]["nodes"]}
    for side, name in zip((-1, 1), model["boundary_nodes"]):
        if name is None:
            continue
        i = 0 if side == -1 else len(columns) - 1
        decoded["wall_pressure"][i, 0 if side == -1 else 1] = 0.0
        cfg, cell = columns[i], cells[i]
        _, geometry, sample, _, _, _ = profile(cell["profile"], cfg)
        center = (
            cfg["elevation_m"]
            + geometry.height_m / 2
            + side * cfg["geometry"]["length_m"] / 2 * math.sin(angle)
        )
        bottom = center - diameter * cosine / 2
        bounds = np.r_[
            cfg["elevation_m"],
            np.asarray(cell["profile"]["interface_heights_m"]) + cfg["elevation_m"],
            cfg["elevation_m"] + geometry.height_m,
        ]
        d = ordinary[name]
        vessel = nodes[name]
        u = cell["momentum_kg_m_s"] / decoded["mass"][i]
        for phase, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            heights, weights = patch(
                bottom,
                max(lo, bottom),
                min(hi, bottom + diameter * cosine),
                diameter,
                cosine,
                order,
            )
            for h, w in zip(heights, weights):
                p, x, q = sample(h - cfg["elevation_m"], phase)
                energy = q["u"] / q["vm"] + q["rho"] * (0.5 * u * u + gravity * (h - datum))
                conserved = np.r_[x / q["vm"], q["rho"] * u, energy]
                flux = np.r_[x / q["vm"] * u, q["rho"] * u * u + p, (energy + p) * u]
                pp = (conserved, flux, p, u, q["rho"], q["s"] / q["vm"])
                ve = d["internal"] / vessel["volume_m3"] + d["rho"] * gravity * (h - datum)
                vp = (
                    np.r_[d["n"] / vessel["volume_m3"], 0.0, ve],
                    np.r_[np.zeros(nn), d["p"], 0.0],
                    d["p"],
                    0.0,
                    d["rho"],
                    d["entropy"] / vessel["volume_m3"],
                )
                fan = (
                    port_flux(vp, pp, native["wave_speed_bound_m_s"])
                    if side == -1
                    else port_flux(pp, vp, native["wave_speed_bound_m_s"])
                )
                flow = -side * w * fan[0]
                boundary[i] += flow
                other = -flow
                reactions[name] += other[-2]
                other[-1] += (
                    (other[:-2] @ mw)
                    * gravity
                    * (datum - model["transport"]["reference_elevation_m"])
                )
                rates[name] += np.r_[other[:-2], 0.0, other[-1]]
                incoming += side * w * fan[1]
    decoded["rates"] += boundary
    decoded.update(
        boundary=boundary, ordinary_net=rates, incoming_entropy=incoming, support=reactions
    )
    return decoded
