"""Scalar PR/64-point BN and virtual-work audit of the joint axial step."""

import math

import numpy as np
from check_pr_capillary_pressure import close
from check_stratified_pr import Reference
from reference_pr_active_multifluid import face

from surface.operation import _digest


def audit(model, before, after, report):
    s, p = model.source, model.pipe
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
                phase = dict(cell)[name]
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
    fluxphysical = np.concatenate(
        (
            q[1][..., :nc] * u[..., None],
            (q[1][..., -2] * u + amid * pressure)[..., None],
            ((q[1][..., -1] + amid * pressure) * u)[..., None],
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
            "conserved": q[1][i],
            "flux": fluxphysical[i],
            "pressure_precision": np.zeros(2),
            "phase_entropy_density": phase_entropy_density[1][i],
            "entropy_density": np.array(entropy_density[1][i]),
            "entropy_flux": np.array(entropy_flux[1][i]),
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
        face_alpha = np.array([stocks[name].volume_m3 / total for name in names])
        alpha = 2 * face_alpha - adjacent["alpha"]
        if np.any(alpha <= 0):
            raise ValueError("Maintained face extrapolation lost positive support")
        pressure = []
        temp = []
        vel = []
        density = []
        ent = []
        for name, root in zip(names, ("liquid", "vapor")):
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
            jump = (
                (leftp["alpha"] + rightp["alpha"])[:, None]
                / 2
                * (
                    rightp["conserved"] / rightp["alpha"][:, None]
                    - leftp["conserved"] / leftp["alpha"][:, None]
                )
            )
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
            elif face_index == n:
                vl = velocity[-1]
                vr = (
                    bb_packet["velocity"]
                    if include_boundary and model.boundaries[1] is not None
                    else np.zeros(2)
                )
            else:
                vl, vr = velocity[face_index - 1], velocity[face_index]
            al, ar = aa_packet["alpha"], bb_packet["alpha"]
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
            curvature[ci] += 2 * (mid[ci] - height)
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
    if not periodic:
        for side, boundary in enumerate(model.boundaries):
            if boundary is None:
                continue
            index = 0 if side == 0 else -1
            sign = 1 if side == 0 else -1
            external += sign * dt * p.area * flux[index].sum(0)
            leftp, rightp = pairs[index]
            jump = rightp["entropy_density"] - leftp["entropy_density"]
            if model.volume_diffusion == "none":
                jump = np.sum(
                    (leftp["alpha"] + rightp["alpha"])
                    / 2
                    * (
                        rightp["phase_entropy_density"] / rightp["alpha"]
                        - leftp["phase_entropy_density"] / leftp["alpha"]
                    )
                )
            incoming_entropy += (
                sign
                * dt
                * p.area
                * ((leftp["entropy_flux"] + rightp["entropy_flux"]) / 2 - p.wave_speed / 2 * jump)
            )
    close(report["external_transfer_n_m_e"], external, "external transport", atol=1e-24, rtol=1e-8)
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
    close(
        report["incoming_entropy_j_k"], incoming_entropy, "external entropy", atol=1e-22, rtol=1e-8
    )
    close(report["capillary_force_n"], force, "independent adjoint force", atol=1e-22, rtol=1e-8)
    close(
        report["capillary_diffusion_heat_j"],
        dt * heat,
        "derived capillary heat",
        atol=1e-25,
        rtol=1e-8,
    )
    mass_res = (q[1][..., :nc] - q[0][..., :nc]) * volume[:, None, None] + dt * p.area * delta[
        ..., :nc
    ]
    momentum_res = (q[1][..., -2] - q[0][..., -2]) * volume[:, None] + dt * (
        p.area * delta[..., -2] - force
    )
    energy_res = (q[1][..., -1] - q[0][..., -1]) * volume[:, None] + dt * (
        p.area * delta[..., -1] - u * force - heat[:, None] * np.array([s.share, 1 - s.share])
    )
    vol_res = vv[1][:, 0] - vv[0][:, 0] - dt * (volume_rate(u) + numerical)
    close(
        mass_res,
        np.zeros_like(mass_res),
        "component equations",
        atol=5e-11 * np.maximum(abs(q[0][..., :nc] * volume[:, None, None]), 1e-30),
        rtol=0,
    )
    mscale = (q[0][..., :nc] * volume[:, None, None] * mw).sum(-1) * np.maximum(
        abs(velocities[0]), 1.0
    )
    close(
        momentum_res, np.zeros_like(momentum_res), "momentum equations", atol=5e-11 * mscale, rtol=0
    )
    close(
        energy_res,
        np.zeros_like(energy_res),
        "energy equations",
        atol=5e-11 * abs(q[0][..., -1]) * volume[:, None] + 1e-22,
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
                surface += coefficient * (height[ci] - height_face) ** 2
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
    global_defect -= external
    global_defect[nc] -= wall_impulse + cap_impulse
    global_defect[-1] -= cap_work
    allowance = 5e-11 * np.sum(abs(q[0] * volume[:, None, None]), axis=(0, 1)) + 1e-22
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
    }
