"""Scalar PR/64-point BN and virtual-work audit of the joint axial step."""

import math

import numpy as np
from check_pr_capillary_pressure import close
from check_stratified_pr import Reference
from reference_pr_active_multifluid import face

from surface.operation import _digest


def audit(model, before, after, report):
    s, p = model.source, model.pipe
    active = bool(report.get("active_phase_support", False))
    f = p.fluid
    names = (s.liquid, s.gas)
    n, nc = len(p.lengths), len(f.names)
    if (
        report["model_id"] != model.model_id
        or report["source_model_id"] != s.model_id
        or _digest(report["source_model"]) != _digest(s.to_dict())
        or _digest(report["state"]) != _digest(after.to_dict())
    ):
        raise ValueError("Joint axial model/state differs")
    dt = after.time_s - before.time_s
    if dt <= 0 or report["dt_s"] != dt:
        raise ValueError("Joint axial clock differs")
    mw = np.array([c.molar_mass_kg_mol for c in f.components])
    q, vv, velocities, pressures, temperatures, entropies = [], [], [], [], [], []
    entropy_density, entropy_flux = [], []
    phase_entropy_density = []
    for state in (before, after):
        if state.model_id != p.model_id or len(state.cells) != n:
            raise ValueError("Joint axial checkpoint differs")
        qq, volumes, uu, pp, tt, entropy = [], [], [], [], [], []
        densities, entropyflows = [], []
        phase_densities = []
        for i, cell in enumerate(state.cells):
            local_entropy = []
            rowq, rowv, rowu, rowp, rowt = [], [], [], [], []
            for name, root in zip(names, ("liquid", "vapor")):
                phase = dict(cell).get(name)
                if phase is None and active:
                    rowq.append(np.zeros(nc + 2))
                    rowv.append(0.0)
                    rowu.append(0.0)
                    rowp.append(0.0)
                    rowt.append(0.0)
                    local_entropy.append(0.0)
                    continue
                if phase is None:
                    raise ValueError("Missing positive-support phase")
                numbers = np.array(phase.component_moles)
                amount, mass = numbers.sum(), numbers @ mw
                ref = Reference(f, np.flatnonzero(numbers > 0))
                eos = ref.state(phase.pressure_pa, phase.temperature_k, numbers / amount, root)
                close(
                    phase.volume_m3,
                    amount * eos["vm"],
                    "scalar PR axial volume",
                    atol=1e-25,
                    rtol=1e-10,
                )
                energy = amount * eos["u"] + 0.5 * phase.momentum_kg_m_s**2 / mass
                close(
                    phase.total_energy_j,
                    energy,
                    "scalar PR axial energy",
                    atol=256 * math.ulp(energy) + 1e-22,
                    rtol=0,
                )
                entropy.append(amount * eos["s"])
                local_entropy.append(amount * eos["s"])
                rowq.append(
                    np.r_[numbers, phase.momentum_kg_m_s, phase.total_energy_j] / p.volumes[i]
                )
                rowv.append(phase.volume_m3)
                rowu.append(phase.momentum_kg_m_s / mass)
                rowp.append(phase.pressure_pa)
                rowt.append(phase.temperature_k)
            phase_densities.append(np.array(local_entropy) / p.volumes[i])
            densities.append(math.fsum(local_entropy) / p.volumes[i])
            entropyflows.append(float(np.dot(local_entropy, rowu)) / p.volumes[i])
            qq.append(rowq)
            volumes.append(rowv)
            uu.append(rowu)
            pp.append(rowp)
            tt.append(rowt)
        q.append(np.array(qq))
        vv.append(np.array(volumes))
        velocities.append(np.array(uu))
        pressures.append(np.array(pp))
        temperatures.append(np.array(tt))
        entropies.append(math.fsum(entropy))
        phase_entropy_density.append(np.array(phase_densities))
        entropy_density.append(np.array(densities))
        entropy_flux.append(np.array(entropyflows))
    volume = np.array(p.volumes)
    amid = (vv[0] + vv[1]) / (2 * volume[:, None])
    u, pressure = velocities[1], pressures[1]
    trace_entropy = None
    if active:
        trace = report["primitive_traces"]
        rho = np.asarray(trace["component_densities_mol_m3"], dtype=float)
        u = np.asarray(trace["velocity_m_s"], dtype=float)
        pressure = np.asarray(trace["pressure_pa"], dtype=float)
        temperatures[1] = np.asarray(trace["temperature_k"], dtype=float)
        if rho.shape != (n, 2, nc) or np.any(rho < 0):
            raise ValueError("Invalid active intrinsic trace")
        intrinsic = np.zeros((n, 2, nc + 2))
        trace_entropy = np.zeros((n, 2))
        for i in range(n):
            for j, root in enumerate(("liquid", "vapor")):
                if amid[i, j] == 0 and not rho[i, j].any():
                    continue
                amount = rho[i, j].sum()
                if amount <= 0:
                    raise ValueError("Missing finite active trace")
                eos = Reference(f, np.flatnonzero(rho[i, j] > 0)).state(
                    pressure[i, j], temperatures[1][i, j], rho[i, j] / amount, root
                )
                close(amount * eos["vm"], 1.0, "trace PR density", atol=1e-9)
                mass = rho[i, j] @ mw
                intrinsic[i, j] = np.r_[
                    rho[i, j], mass * u[i, j], amount * eos["u"] + 0.5 * mass * u[i, j] ** 2
                ]
                trace_entropy[i, j] = amount * eos["s"]
                close(
                    q[1][i, j] * volume[i],
                    intrinsic[i, j] * vv[1][i, j],
                    "owned stock versus trace",
                    atol=1e-20,
                    rtol=1e-9,
                )
        transported = intrinsic * amid[..., None]
        prepared_entropy = trace_entropy * amid
    else:
        transported = q[1] / (vv[1] / volume[:, None])[..., None] * amid[..., None]
        prepared_entropy = phase_entropy_density[1] / (vv[1] / volume[:, None]) * amid
    fluxphysical = np.concatenate(
        (
            transported[..., :nc] * u[..., None],
            (transported[..., -2] * u + amid * pressure)[..., None],
            ((transported[..., -1] + amid * pressure) * u)[..., None],
        ),
        -1,
    )
    periodic = p.boundary_mode == "periodic"
    cells = [
        {
            "alpha": amid[i],
            "pressure": pressure[i],
            "temperature": temperatures[1][i],
            "velocity": u[i],
            "conserved": transported[i],
            "flux": fluxphysical[i],
            "pressure_precision": np.zeros(2),
            "phase_entropy_density": prepared_entropy[i],
            "entropy_density": np.array(np.sum(prepared_entropy[i])),
            "entropy_flux": np.array(np.sum(prepared_entropy[i] * u[i])),
            "boundary_absent": np.zeros(2, dtype=bool),
        }
        for i in range(n)
    ]

    def reflected(cell):
        ghost = {k: v.copy() for k, v in cell.items()}
        ghost["velocity"] *= -1
        ghost["conserved"][:, -2] *= -1
        ghost["flux"][:, :-2] *= -1
        ghost["flux"][:, -1] *= -1
        return ghost

    def maintained(side, adjacent):
        stocks = model.boundaries[side]
        if stocks is None:
            return reflected(adjacent)
        total = math.fsum(v.volume_m3 for v in stocks.values())
        face_alpha = np.array(
            [stocks[name].volume_m3 / total if name in stocks else 0 for name in names]
        )
        alpha = (
            face_alpha
            if model.boundary_geometry == "reservoir_cell"
            else 2 * face_alpha - adjacent["alpha"]
        )
        if np.any(alpha < 0) or (
            model.boundary_geometry == "extrapolated_face" and np.any(alpha == 0)
        ):
            raise ValueError("Maintained face extrapolation lost positive support")
        pressure = []
        temp = []
        vel = []
        density = []
        ent = []
        for name, root in zip(names, ("liquid", "vapor")):
            if name not in stocks:
                pressure.append(0.0)
                temp.append(1.0)
                vel.append(adjacent["velocity"][names.index(name)])
                density.append(np.zeros(nc + 2))
                ent.append(0.0)
                continue
            phase = stocks[name]
            nn = np.array(phase.component_moles)
            mass = nn @ mw
            eos = Reference(f, np.flatnonzero(nn > 0)).state(
                phase.pressure_pa, phase.temperature_k, nn / nn.sum(), root
            )
            close(
                phase.volume_m3, nn.sum() * eos["vm"], "boundary PR volume", atol=1e-25, rtol=1e-10
            )
            ee = nn.sum() * eos["u"] + 0.5 * phase.momentum_kg_m_s**2 / mass
            close(
                phase.total_energy_j,
                ee,
                "boundary PR energy",
                atol=256 * math.ulp(ee) + 1e-22,
                rtol=0,
            )
            density.append(np.r_[nn, phase.momentum_kg_m_s, phase.total_energy_j] / phase.volume_m3)
            pressure.append(phase.pressure_pa)
            temp.append(phase.temperature_k)
            vel.append(phase.momentum_kg_m_s / mass)
            ent.append(nn.sum() * eos["s"] / phase.volume_m3)
        conserved = np.array(density) * alpha[:, None]
        vel = np.array(vel)
        pressure = np.array(pressure)
        flux = np.c_[
            conserved[:, :nc] * vel[:, None],
            conserved[:, -2] * vel + alpha * pressure,
            (conserved[:, -1] + alpha * pressure) * vel,
        ]
        return {
            "alpha": alpha,
            "pressure": pressure,
            "temperature": np.array(temp),
            "velocity": vel,
            "conserved": conserved,
            "flux": flux,
            "pressure_precision": np.zeros(2),
            "boundary_absent": np.array([name not in stocks for name in names]),
            "phase_entropy_density": alpha * np.array(ent),
            "entropy_density": np.dot(alpha, ent),
            "entropy_flux": np.dot(alpha * np.array(ent), vel),
        }

    pairs = (
        [(cells[i], cells[(i + 1) % n]) for i in range(n)]
        if periodic
        else [
            (maintained(0, cells[0]), cells[0]),
            *zip(cells[:-1], cells[1:]),
            (cells[-1], maintained(1, cells[-1])),
        ]
    )
    path = np.array([face(a, b, p.wave_speed)[-1]["phase_path"] for a, b in pairs])
    numerical_fluxes = []
    for leftp, rightp in pairs:
        jump = rightp["conserved"] - leftp["conserved"]
        if model.volume_diffusion == "none":
            il = leftp["conserved"] / np.where(leftp["alpha"] > 0, leftp["alpha"], 1)[:, None]
            ir = rightp["conserved"] / np.where(rightp["alpha"] > 0, rightp["alpha"], 1)[:, None]
            il, ir = (
                np.where(leftp["alpha"][:, None] > 0, il, ir),
                np.where(rightp["alpha"][:, None] > 0, ir, il),
            )
            jump = (leftp["alpha"] + rightp["alpha"])[:, None] / 2 * (ir - il)
        numerical_fluxes.append((leftp["flux"] + rightp["flux"]) / 2 - p.wave_speed / 2 * jump)
    flux = np.array(numerical_fluxes)
    if periodic:
        delta = flux - np.roll(flux, 1, axis=0) + (path + np.roll(path, 1, axis=0)) / 2
        wall_impulse = 0.0
    else:
        delta = flux[1:] - flux[:-1] + (path[1:] + path[:-1]) / 2
        wall_impulse = (
            dt
            * p.area
            * (
                (flux[0, :, -2].sum() if model.boundaries[0] is None else 0)
                - (flux[-1, :, -2].sum() if model.boundaries[1] is None else 0)
            )
        )
    close(report["wall_impulse_kg_m_s"], wall_impulse, "wall impulse", atol=1e-24, rtol=1e-9)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes = (nodes + 1) / 2
    weights = weights / 2

    def volume_rate(velocity, include_boundary=True):
        rates = []
        for face_index, (aa_packet, bb_packet) in enumerate(pairs):
            if periodic:
                vl, vr = velocity[face_index], velocity[(face_index + 1) % n]
            elif face_index == 0:
                vl = (
                    aa_packet["velocity"]
                    if include_boundary and model.boundaries[0] is not None
                    else np.zeros(2)
                )
                vr = velocity[0]
                vl = np.where(aa_packet["boundary_absent"], vr, vl)
            elif face_index == n:
                vl = velocity[-1]
                vr = (
                    bb_packet["velocity"]
                    if include_boundary and model.boundaries[1] is not None
                    else np.zeros(2)
                )
                vr = np.where(bb_packet["boundary_absent"], vl, vr)
            else:
                vl, vr = velocity[face_index - 1], velocity[face_index]
            al, ar = aa_packet["alpha"], bb_packet["alpha"]
            if active and (periodic or 0 < face_index < n):
                vl, vr = np.where(al > 0, vl, vr), np.where(ar > 0, vr, vl)
            aa = al + nodes[:, None] * (ar - al)
            vv = vl + nodes[:, None] * (vr - vl)
            rates.append((ar[0] - al[0]) * (weights @ (aa * vv).sum(-1)))
        rates = np.array(rates)
        return (
            -p.area / 2 * (rates + np.roll(rates, 1))
            if periodic
            else -p.area / 2 * (rates[1:] + rates[:-1])
        )

    h = [(v[:, 0] / (s.width * np.array(p.lengths))) for v in vv]
    mid = (h[0] + h[1]) / 2
    coefficient = s.sigma * s.width / p.lengths[0]
    curvature = 2 * mid - np.roll(mid, 1) - np.roll(mid, -1)
    if not periodic:
        curvature[0] = mid[0] - mid[1]
        curvature[-1] = mid[-1] - mid[-2]
    for side, height in enumerate(s.boundary_heights):
        if height is not None:
            ci = 0 if side == 0 else n - 1
            curvature[ci] += p.lengths[ci] / s.boundary_distances[side] * (mid[ci] - height)
    mu = coefficient * curvature / (s.width * np.array(p.lengths))
    if s.wetting is not None:
        from reference_wetting_potential import pressure as wall_pressure

        mu += wall_pressure(s.wetting.to_dict(), s.sigma, *h)
    force = np.zeros((n, 2))
    for i in range(n):
        for j in range(2):
            basis = np.zeros((n, 2))
            basis[i, j] = 1
            force[i, j] = -mu @ volume_rate(basis, False)
    wall_force = np.zeros_like(force)
    if s.wetting is not None:
        from reference_wetting_potential import pressure as wall_pressure

        material = wall_pressure(s.wetting.to_dict(), s.sigma, *h)
        for i in range(n):
            for phase in range(2):
                basis = np.zeros((n, 2))
                basis[i, phase] = 1
                wall_force[i, phase] = -material @ volume_rate(basis, False)
    material_impulse = dt * float(wall_force.sum())
    close(
        report["material_wall_impulse_kg_m_s"],
        material_impulse,
        "material wall impulse",
        atol=1e-25,
        rtol=1e-8,
    )
    fraction = amid[:, 0]
    laplace = np.roll(fraction, -1) - 2 * fraction + np.roll(fraction, 1)
    if not periodic:
        laplace[0] = fraction[1] - fraction[0]
        laplace[-1] = fraction[-2] - fraction[-1]
        for side, boundary in enumerate(model.boundaries):
            if boundary is not None:
                ci = 0 if side == 0 else n - 1
                ghost = pairs[0][0] if side == 0 else pairs[-1][1]
                laplace[ci] += ghost["alpha"][0] - fraction[ci]
    numerical = p.area * p.wave_speed / 2 * laplace
    if model.volume_diffusion == "none":
        numerical = np.zeros_like(numerical)
    heat = -mu * numerical
    open_faces = any(b is not None for b in model.boundaries)
    cap_work = dt * float(mu @ volume_rate(np.zeros((n, 2)))) if open_faces else 0.0
    cap_impulse = dt * float(force.sum()) if open_faces else material_impulse
    external = np.zeros(nc + 2)
    incoming_entropy = 0.0
    port_transfers = [None, None]
    if not periodic:
        for side, boundary in enumerate(model.boundaries):
            if boundary is None:
                continue
            index = 0 if side == 0 else -1
            sign = 1 if side == 0 else -1
            external += sign * dt * p.area * flux[index].sum(0)
            leftp, rightp = pairs[index]
            al, ar = leftp["alpha"], rightp["alpha"]
            outside = leftp if side == 0 else rightp
            external_velocity = np.where(outside["boundary_absent"], 0.0, outside["velocity"])
            vv_left, vv_right = (
                (external_velocity, np.zeros(2)) if side == 0 else (np.zeros(2), external_velocity)
            )
            integral = weights @ (
                (al + nodes[:, None] * (ar - al))
                * (vv_left + nodes[:, None] * (vv_right - vv_left))
            ).sum(-1)
            port_transfers[side] = {
                "transfer_n_m_e": (sign * dt * p.area * flux[index].sum(0)).tolist(),
                "capillary_work_j": float(
                    -dt * p.area / 2 * mu[index] * (ar[0] - al[0]) * integral
                ),
            }
            jump = rightp["entropy_density"] - leftp["entropy_density"]
            if model.volume_diffusion == "none":
                il = leftp["phase_entropy_density"] / np.where(
                    leftp["alpha"] > 0, leftp["alpha"], 1
                )
                ir = rightp["phase_entropy_density"] / np.where(
                    rightp["alpha"] > 0, rightp["alpha"], 1
                )
                il, ir = np.where(leftp["alpha"] > 0, il, ir), np.where(rightp["alpha"] > 0, ir, il)
                jump = np.sum((leftp["alpha"] + rightp["alpha"]) / 2 * (ir - il))
            incoming_entropy += (
                sign
                * dt
                * p.area
                * ((leftp["entropy_flux"] + rightp["entropy_flux"]) / 2 - p.wave_speed / 2 * jump)
            )
    # The two actual endpoint impulses can be large and almost opposite.
    # Compare their sum with a forward arithmetic bound on the operands;
    # the small cancelled resultant is not its own roundoff scale.
    external_roundoff = (
        64
        * np.finfo(float).eps
        * np.sum(
            [np.abs(row["transfer_n_m_e"]) for row in port_transfers if row is not None], axis=0
        )
        if any(row is not None for row in port_transfers)
        else np.zeros(nc + 2)
    )
    close(
        report["external_transfer_n_m_e"],
        external,
        "external transport",
        atol=1e-24 + external_roundoff,
        rtol=1e-8,
    )
    close(
        report["external_capillary_work_j"],
        cap_work,
        "external capillary work",
        atol=1e-25,
        rtol=1e-8,
    )
    close(
        report["external_capillary_impulse_kg_m_s"],
        cap_impulse,
        "external capillary impulse",
        atol=1e-24,
        rtol=1e-8,
    )
    close(report["capillary_force_n"], force, "independent adjoint force", atol=1e-22, rtol=1e-8)
    close(
        report["capillary_diffusion_heat_j"],
        dt * heat,
        "derived capillary heat",
        atol=1e-25,
        rtol=1e-8,
    )
    boundary_entropy = incoming_entropy
    side_n = np.zeros((n, 2, nc))
    side_m, side_e = np.zeros((n, 2)), np.zeros((n, 2))
    side_v = np.zeros(n)
    side_total = np.zeros(nc + 2)
    actuator = 0.0
    if active:
        rows = report["controls"]
        commands = report["phase_controls"]
        if len(rows) != len(commands) or len({r["cell"] for r in rows}) != len(rows):
            raise ValueError("Side-port registry differs")
        for row in rows:
            i = row["cell"]
            command = commands[str(i)]
            j = names.index(command["inlet_slot"])
            out = 1 - j
            rate = row["volume_rate_m3_s"]
            if rate < 0 or not math.isfinite(rate):
                raise ValueError("Negative side-port rate")
            close(
                vv[1][i, 0] / volume[i],
                command["target_liquid_fraction"],
                "requested holdup",
                atol=1e-14,
            )
            inlet = command["inlet_phase"]
            ni = np.asarray(inlet["component_moles"])
            vi = inlet["volume_m3"]
            pin, tin = inlet["pressure_pa"], inlet["temperature_k"]
            eos = Reference(f, np.flatnonzero(ni > 0)).state(
                pin, tin, ni / ni.sum(), ("liquid", "vapor")[j]
            )
            close(vi, ni.sum() * eos["vm"], "inlet PR volume", atol=1e-25)
            mi = inlet["momentum_kg_m_s"]
            ei = ni.sum() * eos["u"] + 0.5 * mi**2 / (ni @ mw)
            close(ei, inlet["total_energy_j"], "inlet PR energy", atol=1e-20)
            side_n[i, j] = rate * ni / vi
            side_n[i, out] = -rate * rho[i, out]
            side_m[i, j] = rate * mi / vi
            side_m[i, out] = -rate * intrinsic[i, out, -2]
            side_e[i, j] = rate * (ei / vi + pin)
            side_e[i, out] = -rate * (intrinsic[i, out, -1] + pressure[i, out])
            side_total += dt * np.r_[side_n[i].sum(0), side_m[i].sum(), side_e[i].sum()]
            qvl = rate * (1 if j == 0 else -1)
            qv = np.array([qvl, -qvl])
            close(row["phase_volume_transfer_m3"], dt * qv, "side-port volume", atol=1e-24)
            w = dt * qvl * (mu[i] - pressure[i, 0] + pressure[i, 1])
            close(row["actuator_work_j"], w, "actuator work", atol=1e-20)
            actuator += w
            side_v[i] = qvl
            side_e[i] -= pressure[i] * qv
            incoming_entropy += dt * rate * (ni.sum() / vi * eos["s"] - trace_entropy[i, out])
        close(report["sideport_transfer_n_m_e"], side_total, "side-port N/M/H", atol=1e-20)
        close(report["sideport_actuator_work_j"], actuator, "total actuator work", atol=1e-20)
        close(
            report["sideport_incoming_entropy_j_k"],
            incoming_entropy - boundary_entropy,
            "side-port entropy",
            atol=1e-20,
            rtol=1e-8,
        )
    close(
        report["incoming_entropy_j_k"],
        incoming_entropy,
        "full external entropy",
        atol=1e-20,
        rtol=1e-8,
    )
    mass_res = (q[1][..., :nc] - q[0][..., :nc]) * volume[:, None, None] + dt * p.area * delta[
        ..., :nc
    ]
    mass_res -= dt * side_n
    momentum_res = (q[1][..., -2] - q[0][..., -2]) * volume[:, None] + dt * (
        p.area * delta[..., -2] - force
    )
    momentum_res -= dt * side_m
    energy_res = (q[1][..., -1] - q[0][..., -1]) * volume[:, None] + dt * (
        p.area * delta[..., -1]
        - u * force
        - heat[:, None]
        * (
            np.where(
                (amid > 0).all(-1)[:, None],
                np.array([s.share, 1 - s.share]),
                (amid > 0).astype(float),
            )
            if active
            else np.array([s.share, 1 - s.share])
        )
    )
    energy_res -= dt * side_e
    vol_res = vv[1][:, 0] - vv[0][:, 0] - dt * (volume_rate(u) + numerical + side_v)
    close(
        mass_res,
        np.zeros_like(mass_res),
        "component equations",
        atol=5e-11
        * np.maximum(
            np.maximum(abs(q[0][..., :nc]), abs(q[1][..., :nc])) * volume[:, None, None], 1e-30
        )
        + 1e-22,
        rtol=0,
    )
    mscale = (q[0][..., :nc] * volume[:, None, None] * mw).sum(-1) * np.maximum(
        abs(velocities[0]), 1.0
    )
    if active:
        mscale = np.maximum(mscale, (rho * mw).sum(-1) * volume[:, None] * np.maximum(abs(u), 1.0))
    close(
        momentum_res, np.zeros_like(momentum_res), "momentum equations", atol=5e-11 * mscale, rtol=0
    )
    close(
        energy_res,
        np.zeros_like(energy_res),
        "energy equations",
        atol=5e-11 * np.maximum(abs(q[0][..., -1]), abs(q[1][..., -1])) * volume[:, None] + 1e-22,
        rtol=0,
    )
    close(vol_res, np.zeros_like(vol_res), "volume equations", atol=5e-11 * volume, rtol=0)
    augmented = []
    for packet, height in zip(q, h, strict=True):
        stock = (packet * volume[:, None, None]).sum(axis=(0, 1))
        surface = (
            0.5
            * coefficient
            * np.sum((np.roll(height, -1) - height) ** 2 if periodic else np.diff(height) ** 2)
        )
        for side, height_face in enumerate(s.boundary_heights):
            if height_face is not None:
                ci = 0 if side == 0 else n - 1
                surface += (
                    0.5
                    * s.sigma
                    * s.width
                    / s.boundary_distances[side]
                    * (height[ci] - height_face) ** 2
                )
        if s.wetting is not None:
            from reference_wetting_potential import potential as wall_energy

            surface += float(
                np.dot(
                    s.width * np.array(p.lengths), wall_energy(s.wetting.to_dict(), s.sigma, height)
                )
            )
        stock[-1] += surface
        augmented.append(stock)
    # Conserved packet order is N, M, E; the runtime pipe ledger is N, E, M.
    global_defect = augmented[1] - augmented[0]
    global_defect -= external + side_total
    global_defect[nc] -= wall_impulse + cap_impulse
    global_defect[-1] -= cap_work + actuator
    allowance = 5e-11 * np.sum(abs(q[0] * volume[:, None, None]), axis=(0, 1)) + 1e-22
    # An externally supplied component can have exactly zero old stock.
    # Subtracting its actual incoming transfer still incurs floating-point
    # cancellation. Bound that arithmetic by the real operands, not a floor
    # on the absent material and not a relaxed nonlinear tolerance.
    allowance += 64 * np.finfo(float).eps * (abs(side_total) + abs(external))
    # A pipe starting from rest has zero old momentum, but its wall and open
    # face pressure impulses can be large and nearly cancel. Include only a
    # machine-roundoff bound on those actual operands, not a new relative
    # convergence tolerance on the pressure forces.
    allowance[nc] += (
        64
        * np.finfo(float).eps
        * (
            abs(wall_impulse)
            + abs(cap_impulse)
            + abs(external[nc])
            + np.sum(abs(q[1][..., -2] * volume[:, None]))
            + dt
            * (
                p.area * (2 * np.abs(flux[..., -2]).sum() + np.abs(path[..., -2]).sum())
                + np.abs(force).sum()
            )
        )
    )
    close(global_defect, np.zeros_like(global_defect), "global N/M/(E+F)", atol=allowance, rtol=0)
    if not open_faces:
        close(
            (force - wall_force).sum(),
            0.0,
            "total capillary impulse",
            atol=5e-12 * abs(force).sum() + 1e-24,
            rtol=0,
        )
    close(
        report["conservation_residuals"],
        global_defect[[*range(nc), nc + 1, nc]],
        "reported augmented ledger",
        atol=allowance[[*range(nc), nc + 1, nc]],
        rtol=0,
    )
    entropy = entropies[1] - entropies[0] - incoming_entropy
    if entropy < -128 * (math.ulp(entropies[0]) + math.ulp(entropies[1])) - 1e-22:
        raise ValueError("Independent axial entropy decreased")
    return {
        "accepted": True,
        "cells": n,
        "entropy_production_j_k": entropy,
        "maximum_component_equation_residual_mol": float(abs(mass_res).max()),
        "maximum_energy_equation_residual_j": float(abs(energy_res).max()),
        "port_transfers": port_transfers,
    }
