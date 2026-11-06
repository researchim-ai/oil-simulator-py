"""Independent NumPy PR / SciPy PS / finite-difference critical-throat audit.

No runtime throat evaluator, implicit tangent, flash or transport solve is used.
Phase stability outside the serialized branch is not certified by this audit.
"""

import math

import numpy as np
from check_stratified_pr import Reference
from check_stratified_pr_flow import close
from scipy.optimize import root
from scipy.special import softmax


def audit_restrictions(model, data, openings, areas, saved, fluid):
    units = model.get("restrictions", [])
    if set(saved) != {u["name"] for u in units} or set(areas) - {u["name"] for u in units}:
        raise ValueError("Restriction registry or area command differs")
    nodes = {n["name"]: n for n in model["nodes"]}
    masses = np.array([c.molar_mass_kg_mol for c in fluid.components])
    net = {}
    for unit in units:
        name, a, b = unit["name"], unit["source"], unit["target"]
        row = saved[name]
        opening = areas.get(name, unit["opening"])
        if not 0 <= opening <= 1:
            raise ValueError("Restriction area outside bounds")
        enabled = openings[name] and opening > 0
        if row["enabled"] is not enabled:
            raise ValueError("Restriction mode differs")
        forward = data[a]["p"] >= data[b]["p"]
        if row["forward"] is not forward:
            raise ValueError("Restriction donor differs")
        if not enabled:
            if row["choked"] is not False or any(
                row[k] for k in ("phase_fractions", "phase_mole_fractions", "phase_roots")
            ):
                raise ValueError("Disabled restriction has a throat")
            for k, v in row.items():
                if isinstance(v, (float, int)) and not isinstance(v, bool):
                    close(v, 0.0, atol=1e-12, rtol=0.0)
            continue
        if not forward and not unit["allow_reverse"]:
            raise ValueError("Reverse restriction flow outside domain")
        source, target = (data[a], data[b]) if forward else (data[b], data[a])
        p0, pb = source["p"], target["p"]
        h0 = source["pn"] @ source["h"] / source["pn"].sum()
        s0 = source["entropy"] / source["pn"].sum()
        z = source["n"] / source["n"].sum()
        active = np.flatnonzero(z > 0)
        za = z[active]
        p, t = row["throat_pressure_pa"], row["throat_temperature_k"]
        beta, x = np.array(row["phase_fractions"]), np.array(row["phase_mole_fractions"])
        tags = row["phase_roots"]
        if (
            x.shape != (len(beta), len(z))
            or not len(beta)
            or len(tags) != len(beta)
            or np.any(beta < 0)
            or (not row.get("critical_phase_boundary", False) and np.any(beta == 0))
            or np.any(x < 0)
        ):
            raise ValueError("Invalid throat phase registry")
        close(beta.sum(), 1.0, atol=1e-10, rtol=0.0)
        close(x.sum(-1), np.ones(len(beta)), atol=1e-10, rtol=0.0)
        close(beta @ x, z, atol=1e-10, rtol=1e-8)
        count = len(beta)
        boundary = row.get("critical_phase_boundary", False)
        ref = Reference(fluid, active)
        allocation = beta[:, None] * x[:, active]
        if boundary:
            w0 = np.r_[
                math.log(t),
                beta[:-1],
                (np.log(x[:, active[:-1]]) - np.log(x[:, active[-1:]])).ravel(),
            ]
        else:
            w0 = np.r_[math.log(t), (np.log(allocation[:-1]) - np.log(allocation[-1])).ravel()]

        size = len(z)

        def decode(
            w,
            pressure,
            tags=tags,
            ref=ref,
            za=za,
            active=active,
            count=count,
            size=size,
            boundary=boundary,
        ):
            if boundary:
                bb = np.r_[w[1:count], 1 - w[1:count].sum()]
                xx = softmax(
                    np.c_[w[count:].reshape(count, len(active) - 1), np.zeros(count)], axis=-1
                )
            elif count > 1:
                allocations = (
                    softmax(
                        np.r_[w[1:].reshape(count - 1, len(active)), np.zeros((1, len(active)))],
                        axis=0,
                    )
                    * za
                )
                bb = allocations.sum(-1)
                xx = allocations / bb[:, None]
            else:
                bb, xx = np.ones(1), za[None, :]
            full = np.zeros((count, size))
            full[:, active] = xx
            qs = [ref.state(pressure, math.exp(w[0]), full[j], tag) for j, tag in enumerate(tags)]
            mu = np.array([np.log(xx[j]) + q["phi"] for j, q in enumerate(qs)])
            vals = [sum(bb[j] * q[k] for j, q in enumerate(qs)) for k in ("h", "s", "vm")]
            constraints = (mu[:-1] - mu[-1]).ravel()
            if boundary:
                constraints = np.r_[(bb @ xx - za)[:-1], constraints]
            return np.array(vals), constraints

        values, mu = decode(w0, p)
        if mu.size:
            close(mu, np.zeros_like(mu), atol=1e-7, rtol=0.0)
        h, s, v = values
        close(s, s0, atol=1e-6, rtol=1e-9)
        mw = z @ masses
        velocity = row["throat_velocity_m_s"]
        if (
            not 0 <= velocity <= unit["maximum_throat_velocity_m_s"]
            or not unit["minimum_pressure_pa"] <= p <= p0 * (1 + 1e-8)
            or p < pb * (1 - 1e-8)
        ):
            raise ValueError("Throat pressure/velocity outside domain")
        energy = h + 0.5 * mw * velocity**2 - h0
        close(energy, 0.0, atol=1e-5, rtol=0.0)

        def along(pressure, w0=w0, s0=s0, decode=decode, return_state=False):
            def residual(w):
                q, mu = decode(w, pressure)
                return np.r_[mu, (q[1] - s0) / 8.31446261815324]

            solved = root(residual, w0, tol=1e-11)
            if np.max(abs(residual(solved.x))) > 1e-8:
                raise ValueError("Independent throat PS tangent failed")
            result = decode(solved.x, pressure)[0]
            return (result, solved.x) if return_state else result

        dp = p * 1e-4
        first = (along(p + dp)[2] - along(p - dp)[2]) / (2 * dp)
        second = (along(p + dp / 2)[2] - along(p - dp / 2)[2]) / dp
        vp = (4 * second - first) / 3
        criterion = 1 + 2 * (h0 - h) * vp / v**2
        close(row["isentropic_volume_derivative"], vp, atol=1e-15, rtol=2e-6)
        close(row["sonic_criterion"], criterion, atol=2e-6, rtol=2e-6)
        if boundary:
            if count < 2 or min(beta) > 1e-9:
                raise ValueError("Invalid critical phase-onset state")
            present = [j for j in range(count) if beta[j] > 1e-9]
            parent_tags = [tags[j] for j in present]
            allocation = beta[present, None] * x[present][:, active]
            logs = np.log(allocation)
            parent_w = np.r_[math.log(t), (logs[:-1] - logs[-1]).ravel()]

            def parent(pressure, parent_w=parent_w, parent_tags=parent_tags, s0=s0, decode=decode):
                def residual(w):
                    vals, mu = decode(
                        w, pressure, tags=parent_tags, count=len(parent_tags), boundary=False
                    )
                    return np.r_[mu, (vals[1] - s0) / 8.31446261815324]

                solution = root(residual, parent_w, tol=1e-11)
                if np.max(abs(residual(solution.x))) > 1e-8:
                    raise ValueError("Independent parent-phase PS tangent failed")
                return decode(
                    solution.x, pressure, tags=parent_tags, count=len(parent_tags), boundary=False
                )[0][2]

            parent_vp = (parent(p + dp) - parent(p - dp)) / (2 * dp)
            parent_criterion = 1 + 2 * (h0 - h) * parent_vp / v**2
            close(row["parent_sonic_criterion"], parent_criterion, atol=2e-6, rtol=2e-6)
            _, positive = along(p + dp, return_state=True)
            _, negative = along(p - dp, return_state=True)
            vanished = int(beta.argmin())
            slope = (
                (positive[1 + vanished] - negative[1 + vanished]) / (2 * dp)
                if vanished < count - 1
                else -(positive[1:count].sum() - negative[1:count].sum()) / (2 * dp)
            )
            birth = slope < 0
            if bool(row["phase_boundary_birth"]) != bool(birth) or abs(slope) * p <= 1e-10:
                raise ValueError("Phase-boundary direction differs from independent tangent")
            before, after = (
                (parent_criterion, criterion) if birth else (criterion, parent_criterion)
            )
            if before < -1e-6 or after > 1e-6 or p < pb * (1 - 1e-8):
                raise ValueError("Phase onset is not a mass-flux maximum")
        else:
            close(row["parent_sonic_criterion"], criterion, atol=2e-6, rtol=2e-6)
            if row.get("phase_boundary_birth", False):
                raise ValueError("Smooth throat cannot declare phase-boundary direction")
            if vp >= 0 or min((p - pb) / p0, criterion) < -2e-6:
                raise ValueError("Critical-throat complementarity inequality fails")
            close(min((p - pb) / p0, criterion), 0.0, atol=2e-6, rtol=0.0)
        choked = p > pb * (1 + 1e-8)
        if row["choked"] is not choked:
            raise ValueError("Critical-throat mode differs")
        molar = unit["discharge_coefficient"] * unit["throat_area_m2"] * opening * velocity / v
        if molar > unit["maximum_molar_rate_mol_s"]:
            raise ValueError("Restriction flow limit exceeded")
        signed = molar if forward else -molar
        expected = {
            "molar_rate_mol_s": signed,
            "mass_rate_kg_s": signed * mw,
            "throat_molar_enthalpy_j_mol": h,
            "throat_molar_entropy_j_mol_k": s,
            "throat_molar_volume_m3_mol": v,
            "stagnation_enthalpy_j_mol": h0,
            "opening": opening,
            "energy_residual_j_mol": energy,
        }
        for k, value in expected.items():
            close(row[k], value, atol=1e-8, rtol=2e-7)
        phi = model["gravity_m_s2"] * (nodes[a]["elevation_m"] - model["reference_elevation_m"])
        flux = np.r_[signed * z, 0.0, signed * (h0 + mw * phi)]
        net[a] = net.get(a, 0) - flux
        net[b] = net.get(b, 0) + flux
    return net
