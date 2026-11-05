"""Independent NumPy reconstruction and FV geometry for inclined PR pipes.

Shared by the coupled-system auditor; no runtime hydrostatic solver is called.
"""

import numpy as np
from check_pr_momentum_pipe import decode as horizontal_decode
from check_stratified_pr import softmax
from check_stratified_pr_flow import close

from surface.peng_robinson import R


def decode(state, model, fluid):
    n = np.asarray(state["component_moles"])
    active = np.flatnonzero(n[0] > 0)
    c = len(active)
    count = model["cells"]
    y = np.asarray(state["solution_variables"])
    if y.shape != (count * (3 * c + 2),):
        raise ValueError("Inclined reconstructed state dimensions differ")
    zc = (np.arange(count) + 0.5) * model["elevation_change_m"] / count
    zf = np.arange(count + 1) * model["elevation_change_m"] / count
    mm = np.array([q.molar_mass_kg_mol for q in fluid.components])
    g = model["gravity_m_s2"]
    body = dict(state)
    body["solution_variables"] = y[: count * (c + 2)].tolist()
    body["total_energy_j"] = (np.asarray(state["total_energy_j"]) - n @ mm * g * zc).tolist()
    data = horizontal_decode(body, model, fluid)
    potential = data["rho"] * g * zc
    data["u"][:, -1] += potential
    data["flux"][:, -1] += potential * data["v"]
    data["potential"] = potential
    local = y[count * (c + 2) :].reshape(2 * count, c)
    zz = np.column_stack((zf[:-1], zf[1:])).flatten()
    temps = data["t"].repeat(2)
    velocity = data["v"].repeat(2)
    values = []
    flux = []
    pressure = []
    entropy = []
    rho = []
    compositions = []
    reference = data["ref"]
    for j, (row, t, v, z) in enumerate(zip(local, temps, velocity, zz)):
        p = float(np.exp(row[0]))
        x = np.zeros(len(fluid.names))
        x[active] = softmax(np.r_[row[1:], 0.0])
        if not fluid.min_pressure_pa <= p <= fluid.max_pressure_pa:
            raise ValueError("Hydrostatic face pressure outside domain")
        q = reference.state(p, t, x, "stable")
        i = j // 2
        qc = reference.state(data["p"][i], data["t"][i], data["x"][i], "stable")
        muf = np.log(x[active]) + q["phi"] + np.log(p / fluid.pref)
        muc = np.log(data["x"][i, active]) + qc["phi"] + np.log(data["p"][i] / fluid.pref)
        close(muf - muc + mm[active] * g * (z - zc[i]) / (R * t), np.zeros(c), atol=2e-8, rtol=0.0)
        u = np.r_[
            x / q["vm"], q["rho"] * v, q["u"] / q["vm"] + 0.5 * q["rho"] * v * v + q["rho"] * g * z
        ]
        values.append(u)
        flux.append(np.r_[u[:-2] * v, q["rho"] * v * v + p, (u[-1] + p) * v])
        pressure.append(p)
        entropy.append(q["s"] / q["vm"])
        rho.append(q["rho"])
        compositions.append(x)
        dp = p * 1e-5
        dt = t * 1e-5
        a, b = reference.state(p + dp, t, x, "stable"), reference.state(p - dp, t, x, "stable")
        aa, bb = reference.state(p, t + dt, x, "stable"), reference.state(p, t - dt, x, "stable")
        st = (aa["s"] - bb["s"]) / (2 * dt)
        den = (a["rho"] - b["rho"]) / (2 * dp) - (aa["rho"] - bb["rho"]) / (2 * dt) * (
            a["s"] - b["s"]
        ) / (2 * dp) / st
        if st <= 0 or den <= 0 or den**-0.5 + abs(v) > model["wave_speed_bound_m_s"] * (1 + 2e-6):
            raise ValueError("Reconstructed acoustic bound rejected")
    data["reconstruction"] = {
        "u": np.array(values),
        "flux": np.array(flux),
        "p": np.array(pressure),
        "s": np.array(entropy),
        "rho": np.array(rho),
        "x": np.array(compositions),
        "v": velocity,
        "t": temps,
    }
    data["gravity_force_n"] = (
        np.pi
        * model["diameter_m"] ** 2
        / 4
        * np.diff(np.array(pressure).reshape(count, 2), axis=1).flatten()
    )
    return data


def faces(data, model, boundaries, fluid):
    if any(b is not None and b != "wall" for b in boundaries):
        raise ValueError(
            "Independent inclined auditor supports wall ports or coupled HLLC overrides"
        )
    d = data["reconstruction"]
    alpha = model["wave_speed_bound_m_s"]
    ff = (d["flux"][1:-1:2] + d["flux"][2::2]) / 2 - alpha * (d["u"][2::2] - d["u"][1:-1:2]) / 2
    ss = (d["s"][1:-1:2] * d["v"][1:-1:2] + d["s"][2::2] * d["v"][2::2]) / 2 - alpha * (
        d["s"][2::2] - d["s"][1:-1:2]
    ) / 2
    if model["boundary_mode"] == "periodic":
        end = (d["flux"][-1] + d["flux"][0]) / 2 - alpha * (d["u"][0] - d["u"][-1]) / 2
        se = (d["s"][-1] * d["v"][-1] + d["s"][0] * d["v"][0]) / 2 - alpha * (
            d["s"][0] - d["s"][-1]
        ) / 2
        return np.vstack((end, ff, end)), np.r_[se, ss, se]
    ends = []
    for i in (0, -1):
        end = np.zeros(len(fluid.names) + 2)
        end[-2] = (
            d["rho"][i] * d["v"][i] ** 2 + d["p"][i] + (-alpha if i == 0 else alpha) * d["u"][i, -2]
        )
        ends.append(end)
    return np.vstack((ends[0], ff, ends[1])), np.r_[0.0, ss, 0.0]
