"""Independent scalar PR and 64-node path quadrature for active multifluids."""

import math

import numpy as np
from check_pr_equilibrium_transport import reference_sound
from check_stratified_pr import Reference
from reference_pr_interphase_relaxation import decode

from surface.peng_robinson import R


def packets(model, state):
    nc = len(model.fluid.names)
    k = len(model.names)
    count = len(model.volumes)
    q = np.zeros((count, k, nc + 2))
    alpha = np.zeros((count, k))
    p = alpha.copy()
    t = alpha.copy()
    u = alpha.copy()
    entropy = 0.0
    precision = alpha.copy()
    if (
        set(state) != {"cells", "time_s", "model_id"}
        or state["model_id"] != model.model_id
        or len(state["cells"]) != count
        or not math.isfinite(state["time_s"])
        or state["time_s"] < 0
    ):
        raise ValueError("Active checkpoint registry or clock differs")
    for i, cell in enumerate(state["cells"]):
        if not cell or tuple(cell) != tuple(n for n in model.names if n in cell):
            raise ValueError("Canonical nonempty active phase registry required")
        if set(cell) - set(model.names):
            raise ValueError("Unknown active phase")
        for name, phase in cell.items():
            j = model.names.index(name)
            slot = model.slots[j]
            (data,) = decode(model.fluid, {"phases": [phase]}, [slot["root"]])
            entropy += data["entropy"]
            amount = data["n"].sum()
            x = data["n"] / amount
            marker = slot.get("component_marker")
            if marker is not None and x[marker] < slot.get("minimum_marker_fraction", 0.5):
                raise ValueError("Active component marker violated")
            q[i, j] = (
                np.r_[data["n"], phase["momentum_kg_m_s"], phase["total_energy_j"]]
                / model.volumes[i]
            )
            alpha[i, j] = phase["volume_m3"] / model.volumes[i]
            p[i, j] = data["p"]
            t[i, j] = data["t"]
            u[i, j] = data["velocity"]
            acoustic = reference_sound(model.fluid, data["p"], data["t"], x, slot["root"])
            sound = abs(data["velocity"]) + acoustic
            ref = Reference(model.fluid, np.flatnonzero(x > 0))

            def params(temp, ref=ref, x=x):
                if ref.pure:
                    return (*ref.pure._a(temp), ref.pure.b)
                a, at, b, *_ = ref.mixture._parameters(temp, x[ref.active])
                return a, at, b

            h = data["t"] * 1e-5
            _, _, bm = params(data["t"])
            att = (params(data["t"] + h)[1] - params(data["t"] - h)[1]) / (2 * h)
            ideal_cp = sum(v * c.ideal_cp_j_mol_k for v, c in zip(x, model.fluid.components))
            vm = phase["volume_m3"] / amount
            ratio = math.log((vm + (1 + math.sqrt(2)) * bm) / (vm + (1 - math.sqrt(2)) * bm))
            cv = ideal_cp - R + data["t"] * att * ratio / (2 * math.sqrt(2) * bm)
            bulk = data["mass"] / phase["volume_m3"] * acoustic**2
            escale = abs(phase["total_energy_j"]) / amount + (ideal_cp + R) * (
                data["t"] + model.fluid.tref
            )
            precision[i, j] = (
                64
                * np.finfo(float).eps
                * (abs(data["p"]) + bulk + cv * data["t"] / vm)
                * (1 + escale / (cv * data["t"]))
            )
            if sound > model.wave_speed * (1 + 1e-6):
                raise ValueError("Active acoustic envelope violated")
        if abs(math.fsum(alpha[i]) - 1.0) > 1e-9:
            raise ValueError("Active phases do not fill a cell")
    flux = np.concatenate(
        (
            q[..., :nc] * u[..., None],
            (q[..., -2] * u + alpha * p)[..., None],
            ((q[..., -1] + alpha * p) * u)[..., None],
        ),
        -1,
    )
    return {
        "conserved": q,
        "alpha": alpha,
        "pressure": p,
        "temperature": t,
        "velocity": u,
        "flux": flux,
        "entropy": entropy,
        "pressure_precision": precision,
    }


def face(left, right, wave_speed):
    al, ar = left["alpha"], right["alpha"]
    present_l = al > 0
    present_r = ar > 0
    present = present_l | present_r

    def ghosts(key, default):
        return np.where(present, np.where(present_l, left[key], right[key]), default), np.where(
            present, np.where(present_r, right[key], left[key]), default
        )

    pl, pr = ghosts("pressure", 0.0)
    tl, tr = ghosts("temperature", 1.0)
    ul, ur = ghosts("velocity", 0.0)
    da = ar - al
    count = len(al)
    nc = left["conserved"].shape[-1] - 2
    pvalues = np.r_[pl[present], pr[present]]
    uvalues = np.r_[ul[present], ur[present]]
    eps = 32 * np.finfo(float).eps
    contact = np.ptp(pvalues) <= max(
        eps * max(1.0, max(abs(pvalues))),
        2 * max(np.r_[left["pressure_precision"], right["pressure_precision"]]),
    ) and np.ptp(uvalues) <= eps * max(1.0, max(abs(uvalues)))
    dq = right["conserved"] - left["conserved"]
    if contact:
        velocity = 0.5 * (min(uvalues) + max(uvalues))
        pressure = 0.5 * (min(pvalues) + max(pvalues))
        dm = min(velocity, 0.0) * dq
        dp = max(velocity, 0.0) * dq
        am = min(velocity, 0.0) * da
        ap = max(velocity, 0.0) * da
        path = np.c_[np.zeros((count, nc)), -pressure * da, -pressure * velocity * da]
        alpha_path = velocity * da
        upwind = left["conserved"] if velocity >= 0 else right["conserved"]
        flux = velocity * upwind.copy()
        flux[:, -2] += 0.5 * (al + ar) * pressure
        flux[:, -1] += 0.5 * (al + ar) * pressure * velocity
    else:
        nodes, weights = np.polynomial.legendre.leggauss(64)
        nodes = (nodes + 1) / 2
        weights /= 2
        path = np.zeros_like(dq)
        alpha_path = np.zeros(count)
        for i in range(count):
            for j in range(i + 1, count):
                gamma = al[j] * da[i] - al[i] * da[j]
                if gamma == 0.0:
                    continue
                a = al[None, :] + nodes[:, None] * da[None, :]
                pressure = pl[None, :] + nodes[:, None] * (pr - pl)[None, :]
                temperature = tl[None, :] + nodes[:, None] * (tr - tl)[None, :]
                velocity = ul[None, :] + nodes[:, None] * (ur - ul)[None, :]
                pair_u = (a[:, i] * velocity[:, i] + a[:, j] * velocity[:, j]) / (a[:, i] + a[:, j])
                pair_p = (
                    a[:, j] * pressure[:, i] / temperature[:, i]
                    + a[:, i] * pressure[:, j] / temperature[:, j]
                ) / (a[:, j] / temperature[:, i] + a[:, i] / temperature[:, j])
                vp = gamma * (weights @ pair_u)
                mp = gamma * (weights @ pair_p)
                ep = gamma * (weights @ (pair_p * pair_u))
                alpha_path[i] += vp
                alpha_path[j] -= vp
                path[i, -2] -= mp
                path[j, -2] += mp
                path[i, -1] -= ep
                path[j, -1] += ep
        jump = right["flux"] - left["flux"] + path
        dm = 0.5 * (jump - wave_speed * dq)
        dp = 0.5 * (jump + wave_speed * dq)
        am = 0.5 * (alpha_path - wave_speed * da)
        ap = 0.5 * (alpha_path + wave_speed * da)
        flux = 0.5 * (left["flux"] + right["flux"]) - 0.5 * wave_speed * dq
    return (
        dm,
        dp,
        am,
        ap,
        {"flux": flux, "phase_path": path, "alpha_path": alpha_path, "contact": bool(contact)},
    )


def reconstruct(model, state):
    data = packets(model, state)
    count = len(model.volumes)
    fields = (
        "conserved",
        "alpha",
        "pressure",
        "temperature",
        "velocity",
        "pressure_precision",
        "flux",
    )

    def cell(index):
        return {key: data[key][index].copy() for key in fields}

    faces = []
    if model.boundary_mode == "periodic":
        for i in range(count):
            faces.append(face(cell(i), cell((i + 1) % count), model.wave_speed))
        delta = np.stack([faces[i][0] + faces[(i - 1) % count][1] for i in range(count)])
        adelta = np.stack([faces[i][2] + faces[(i - 1) % count][3] for i in range(count)])
    else:
        cells = [cell(i) for i in range(count)]
        import copy

        gl, gr = copy.deepcopy(cells[0]), copy.deepcopy(cells[-1])
        for ghost in (gl, gr):
            ghost["velocity"] *= -1
            ghost["conserved"][:, -2] *= -1
            ghost["flux"][:, :-2] *= -1
            ghost["flux"][:, -1] *= -1
        cells = [gl, *cells, gr]
        for i in range(count + 1):
            faces.append(face(cells[i], cells[i + 1], model.wave_speed))
        delta = np.stack([faces[i + 1][0] + faces[i][1] for i in range(count)])
        adelta = np.stack([faces[i + 1][2] + faces[i][3] for i in range(count)])
    return data, delta, adelta, [f[-1] for f in faces]
