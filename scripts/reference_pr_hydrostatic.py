"""Independent NumPy EOS audit of chemical-potential face reconstruction."""

import math

import numpy as np
from check_stratified_pr import Reference
from check_stratified_pr_flow import close

from surface.peng_robinson import R


def attach_faces(model, nodes, data, saved, fluid, sound_function):
    expected = {
        name
        for name, d in data.items()
        if nodes[name].get("hydrostatic_reconstruction", False)
        and model["gravity_m_s2"]
        and nodes[name]["elevation_change_m"]
        and len(d["pn"]) == 1
    }
    if set(saved) != expected:
        raise ValueError("Hydrostatic reconstruction registry differs")
    for name in expected:
        d, node = data[name], nodes[name]
        if set(saved[name]) != {"left", "right"}:
            raise ValueError("Both hydrostatic faces required")
        active = np.flatnonzero(d["n"] > 0)
        ref = Reference(fluid, active)
        origin = ref.state(d["p"], d["t"], d["x"][0], "stable")
        target = np.log(d["x"][0, active]) + origin["phi"] + math.log(d["p"] / fluid.pref)
        faces = {}
        for port, normal in (("left", -1.0), ("right", 1.0)):
            item = saved[name][port]
            p = item["pressure_pa"]
            x = np.asarray(item["mole_fractions"])
            if x.shape != d["n"].shape or np.any(x < 0) or np.any(x[active] <= 0):
                raise ValueError("Hydrostatic face composition differs")
            close(x.sum(), 1.0, atol=1e-12, rtol=0.0)
            if np.any(x[np.setdiff1d(np.arange(len(x)), active)] != 0):
                raise ValueError("Spurious hydrostatic face component")
            options = [(tag, ref.state(p, d["t"], x, tag)) for tag in ("liquid", "vapor")]
            tag, q = min(
                options, key=lambda pair: abs(pair[1]["z"] - item["compressibility_factor"])
            )
            close(q["z"], item["compressibility_factor"], atol=1e-9, rtol=1e-8)
            if x[active] @ q["phi"] > min(x[active] @ v["phi"] for _, v in options) + 1e-7:
                raise ValueError("Metastable hydrostatic EOS root")
            residual = np.log(x[active]) + q["phi"] + math.log(p / fluid.pref) - target
            residual += (
                ref.mw
                * model["gravity_m_s2"]
                * node["elevation_change_m"]
                * normal
                / (2 * R * d["t"])
            )
            close(residual, np.zeros(len(active)), atol=2e-8, rtol=0.0)
            close(item["chemical_residuals"], residual, atol=2e-8, rtol=0.0)
            sound = sound_function(fluid, p, d["t"], x, tag)
            if sound + abs(d["u"]) > model["wave_speed_bound_m_s"] * (1 + 1e-7):
                raise ValueError("Hydrostatic face exceeds acoustic bound")
            faces[port] = {
                **d,
                "p": p,
                "x": x[None, :],
                "vm": np.array([q["vm"]]),
                "rho_phase": np.array([q["rho"]]),
                "phase_z": np.array([q["z"]]),
                "rho": q["rho"],
                "h": np.array([q["h"]]),
                "s_phase": np.array([q["s"]]),
                "alpha": np.ones(1),
                "sound": sound,
            }
        d["hydrostatic_faces"] = faces


def gravity_force(node, data, model, face_area):
    left, right = face_area(node, "left"), face_area(node, "right")
    if "hydrostatic_faces" in data:
        pl = data["hydrostatic_faces"]["left"]["p"]
        pr = data["hydrostatic_faces"]["right"]["p"]
        return (left + right) / 2 * (pr - pl) + (pl + pr) / 2 * (right - left)
    return -data["mass"] * model["gravity_m_s2"] * node["elevation_change_m"] / node[
        "length_m"
    ] + data["p"] * (right - left)
