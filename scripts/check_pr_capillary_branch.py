"""Independent scalar PR, 64-node paths and finite-chamber momentum audit."""

import math

import numpy as np
from check_pr_capillary_pressure import close
from check_stratified_pr import Reference
from reference_pr_active_multifluid import face

from surface.operation import _digest


def audit(network, before, after, report):
    model = network.solver
    s, p = network.source, network.source.pipe
    f = p.fluid
    n, nc = len(p.lengths), len(f.names)
    names = (s.liquid, s.gas)
    if (
        report["model_id"] != model.model_id
        or report["source_model_id"] != s.model_id
        or _digest(report["source_model"]) != _digest(s.to_dict())
        or _digest(report["state"]) != _digest(after.to_dict())
    ):
        raise ValueError("Finite branch model/checkpoint differs")
    if _digest(network.to_dict()) != network.model_id:
        raise ValueError("Changed physical branch configuration")
    expected_faces = []
    cursor = 0
    offsets = {}
    lengths = []
    for name, entry in network.pipes.items():
        pipe = entry["pipe"]
        count = len(pipe.lengths)
        offsets[name] = (cursor, cursor + count)
        lengths.extend(pipe.lengths)
        cursor += count
    junction_indices = {name: cursor + i for i, name in enumerate(network.junction_volumes)}
    if junction_indices != network.junction_indices:
        raise ValueError("Finite junction ordering differs")
    lengths.extend(value / p.area for value in network.junction_volumes.values())
    close(p.lengths, lengths, "real control-volume lengths", atol=0, rtol=0)
    for name, entry in network.pipes.items():
        start, end = offsets[name]
        axis = tuple(entry["axis"])
        expected_faces.extend((i, i + 1, -1, axis) for i in range(start, end - 1))
        for side, node in enumerate(entry["boundary_nodes"]):
            cell = start if side == 0 else end - 1
            if node is None:
                expected_faces.append((cell, cell, side, axis))
            elif network.openings[f"{name}:{side}"]:
                expected_faces.append(
                    (junction_indices[node], cell, -1, axis)
                    if side == 0
                    else (cell, junction_indices[node], -1, axis)
                )
            else:
                expected_faces.append((cell, cell, side, axis))
                expected_faces.append(
                    (junction_indices[node], junction_indices[node], 1 - side, axis)
                )
    if (
        tuple(expected_faces) != model.faces
        or tuple((a, b) for a, b, wall, _ in expected_faces if wall == -1) != s.surface_edges
    ):
        raise ValueError("Physical chamber/pipe face binding differs")
    dt = after.time_s - before.time_s
    if dt <= 0 or dt != report["dt_s"]:
        raise ValueError("Finite branch clock differs")
    mw = np.array([c.molar_mass_kg_mol for c in f.components])
    volume = np.array(p.volumes)
    phase_entropy = []
    q = []
    vv = []
    pp = []
    tt = []
    uu = []
    ss = []
    for state in (before, after):
        if state.model_id != p.model_id or len(state.cells) != n:
            raise ValueError("Physical body view differs")
        qq = []
        volumes = []
        pressure = []
        temperature = []
        velocity = []
        entropy = []
        entropy_rows = []
        for i, cell in enumerate(state.cells):
            entropy_row = []
            row = []
            vr = []
            pr = []
            tr = []
            ur = []
            for name, root in zip(names, ("liquid", "vapor")):
                phase = dict(cell)[name]
                nn = np.array(phase.component_moles)
                amount = nn.sum()
                mass = nn @ mw
                eos = Reference(f, np.flatnonzero(nn > 0)).state(
                    phase.pressure_pa, phase.temperature_k, nn / amount, root
                )
                close(phase.volume_m3, amount * eos["vm"], "body PR volume", atol=1e-25, rtol=1e-10)
                energy = amount * eos["u"] + 0.5 * phase.momentum_kg_m_s**2 / mass
                close(
                    phase.total_energy_j,
                    energy,
                    "body PR energy",
                    atol=256 * math.ulp(energy) + 1e-22,
                    rtol=0,
                )
                row.append(np.r_[nn, phase.momentum_kg_m_s, phase.total_energy_j] / volume[i])
                vr.append(phase.volume_m3)
                pr.append(phase.pressure_pa)
                tr.append(phase.temperature_k)
                ur.append(phase.momentum_kg_m_s / mass)
                entropy.append(amount * eos["s"])
                entropy_row.append(eos["s"])
            entropy_rows.append(entropy_row)
            qq.append(row)
            volumes.append(vr)
            pressure.append(pr)
            temperature.append(tr)
            velocity.append(ur)
        q.append(np.array(qq))
        vv.append(np.array(volumes))
        pp.append(np.array(pressure))
        tt.append(np.array(temperature))
        uu.append(np.array(velocity))
        ss.append(math.fsum(entropy))
        phase_entropy.append(np.array(entropy_rows))
    alpha = (vv[0] + vv[1]) / (2 * volume[:, None])
    u = uu[1]
    fluxphysical = np.concatenate(
        (
            q[1][..., :nc] * u[..., None],
            (q[1][..., -2] * u + alpha * pp[1])[..., None],
            ((q[1][..., -1] + alpha * pp[1]) * u)[..., None],
        ),
        -1,
    )
    packets = [
        {
            "alpha": alpha[i],
            "pressure": pp[1][i],
            "temperature": tt[1][i],
            "velocity": u[i],
            "conserved": q[1][i],
            "flux": fluxphysical[i],
            "pressure_precision": np.zeros(2),
        }
        for i in range(n)
    ]

    def reflect(packet):
        value = {k: v.copy() for k, v in packet.items()}
        value["velocity"] *= -1
        value["conserved"][:, -2] *= -1
        value["flux"][:, :-2] *= -1
        value["flux"][:, -1] *= -1
        return value

    distances = [(p.lengths[a] + p.lengths[b]) / 2 for a, b, _, _ in model.faces]
    speeds = [p.wave_speed * max(distances) / d for d in distances]
    pairs = []
    paths = []
    fluxes = []
    for (a, b, wall, _), speed in zip(model.faces, speeds):
        left = reflect(packets[a]) if wall == 0 else packets[a]
        right = reflect(packets[b]) if wall == 1 else packets[b]
        pairs.append((left, right))
        paths.append(face(left, right, speed)[-1])
        jump = right["conserved"] - left["conserved"]
        if model.volume_diffusion == "none":
            jump = (
                (left["alpha"] + right["alpha"])[:, None]
                / 2
                * (
                    right["conserved"] / right["alpha"][:, None]
                    - left["conserved"] / left["alpha"][:, None]
                )
            )
        fluxes.append((left["flux"] + right["flux"]) / 2 - speed / 2 * jump)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes = (nodes + 1) / 2
    weights = weights / 2

    def volume_rate(velocity):
        rate = np.zeros(n)
        for (a, b, wall, _), (left, right) in zip(model.faces, pairs):
            if wall != -1:
                continue
            al, ar = left["alpha"], right["alpha"]
            aa = al + nodes[:, None] * (ar - al)
            v = velocity[a] + nodes[:, None] * (velocity[b] - velocity[a])
            value = -p.area / 2 * (ar[0] - al[0]) * (weights @ (aa * v).sum(-1))
            rate[a] += value
            rate[b] += value
        return rate

    heights = [v[:, 0] / (s.width * np.array(p.lengths)) for v in vv]
    mid = (heights[0] + heights[1]) / 2
    potential = np.zeros(n)
    surface = [0.0, 0.0]
    for a, b in s.surface_edges:
        coefficient = s.sigma * s.width / ((p.lengths[a] + p.lengths[b]) / 2)
        potential[a] += coefficient * (mid[a] - mid[b]) / (s.width * p.lengths[a])
        potential[b] += coefficient * (mid[b] - mid[a]) / (s.width * p.lengths[b])
        for i, h in enumerate(heights):
            surface[i] += 0.5 * coefficient * (h[b] - h[a]) ** 2
    if s.wetting is not None:
        from reference_wetting_potential import potential as wall_energy
        from reference_wetting_potential import pressure as wall_pressure

        potential += wall_pressure(s.wetting.to_dict(), s.sigma, *heights)
        for i, h in enumerate(heights):
            surface[i] += float(
                np.dot(s.width * np.array(p.lengths), wall_energy(s.wetting.to_dict(), s.sigma, h))
            )
    force = np.zeros((n, 2))
    for i in range(n):
        for phase in range(2):
            basis = np.zeros((n, 2))
            basis[i, phase] = 1
            force[i, phase] = -potential @ volume_rate(basis)
    delta = np.zeros_like(q[1])
    diffusion = np.zeros(n)
    reaction = {name: np.zeros(3) for name in network.junction_indices}
    pinned = {i: name for name, i in network.junction_indices.items()}
    wall_impulse = np.zeros(3)
    surface_impulse = np.zeros(3)
    for (a, b, wall, direction), pair, path, flux, speed in zip(
        model.faces, pairs, paths, fluxes, speeds
    ):
        left, right = pair
        direction = np.array(direction)
        if wall != 0:
            delta[a] += flux + path["phase_path"] / 2
        if wall != 1:
            delta[b] += -flux + path["phase_path"] / 2
        if wall == -1:
            d = p.area * speed / 2 * (right["alpha"][0] - left["alpha"][0])
            diffusion[a] += d
            diffusion[b] -= d
            al, ar = left["alpha"], right["alpha"]
            aa = al + nodes[:, None] * (ar - al)
            coeff_l = (weights[:, None] * aa * (1 - nodes[:, None])).sum(0)
            coeff_r = (weights[:, None] * aa * nodes[:, None]).sum(0)
            factor = p.area / 2 * (potential[a] + potential[b]) * (ar[0] - al[0])
            fl, fr = factor * coeff_l, factor * coeff_r
            surface_impulse += dt * (fl + fr).sum() * direction
            if a in pinned:
                reaction[pinned[a]] += (
                    dt
                    * (p.area * (flux[:, -2] + path["phase_path"][:, -2] / 2) - fl).sum()
                    * direction
                )
            if b in pinned:
                reaction[pinned[b]] += (
                    dt
                    * (p.area * (-flux[:, -2] + path["phase_path"][:, -2] / 2) - fr).sum()
                    * direction
                )
        else:
            wall_impulse += dt * p.area * flux[:, -2].sum() * (1 if wall == 0 else -1) * direction
    # A sealed chamber port has its own physical pressure wall. Its support
    # reaction cancels that wall impulse because its mean momentum is pinned.
    for (a, b, wall, direction), path, flux in zip(model.faces, paths, fluxes):
        if wall == 0 and b in pinned:
            reaction[pinned[b]] += (
                dt
                * p.area
                * (-flux[:, -2] + path["phase_path"][:, -2] / 2).sum()
                * np.array(direction)
            )
        if wall == 1 and a in pinned:
            reaction[pinned[a]] += (
                dt
                * p.area
                * (flux[:, -2] + path["phase_path"][:, -2] / 2).sum()
                * np.array(direction)
            )
    source_n = np.zeros((n, 2, nc))
    source_e = np.zeros((n, 2))
    entropy_rate = 0.0
    for name, config in network.sources.items():
        index = network.junction_indices[name]
        for item in config["inlets"]:
            phase = names.index(item["phase"])
            data = item["state"]
            numbers = np.array(data["component_moles"])
            amount = numbers.sum()
            eos = Reference(f, np.flatnonzero(numbers > 0)).state(
                data["pressure_pa"], data["temperature_k"], numbers / amount, p.slots[phase]["root"]
            )
            close(data["volume_m3"], amount * eos["vm"], "source PR volume", atol=1e-25, rtol=1e-10)
            close(
                data["total_energy_j"],
                amount * eos["u"],
                "source PR energy",
                atol=256 * math.ulp(data["total_energy_j"]) + 1e-22,
                rtol=0,
            )
            rate = item["molar_rate_mol_s"]
            source_n[index, phase] += rate * numbers / amount
            source_e[index, phase] += rate * (eos["u"] + data["pressure_pa"] * eos["vm"])
            entropy_rate += rate * eos["s"]
        for item in config["withdrawals"]:
            phase = names.index(item["phase"])
            rate = item["molar_rate_mol_s"]
            amount = q[1][index, phase, :nc].sum() * volume[index]
            source_n[index, phase] -= rate * q[1][index, phase, :nc] * volume[index] / amount
            source_e[index, phase] -= (
                rate
                * (
                    q[1][index, phase, -1] * volume[index]
                    + pp[1][index, phase] * vv[1][index, phase]
                )
                / amount
            )
            entropy_rate -= rate * phase_entropy[1][index, phase]
        for item in config["heaters"]:
            phase = names.index(item["phase"])
            source_e[index, phase] += item["power_w"]
            entropy_rate += item["power_w"] / item["temperature_k"]
    delta = (
        delta
        - np.concatenate((source_n, np.zeros_like(source_e)[..., None], source_e[..., None]), -1)
        / p.area
    )
    external = dt * np.r_[source_n.sum(axis=(0, 1)), 0.0, source_e.sum()]
    close(
        report["external_transfer_n_m_e"],
        external,
        "junction external transfers",
        atol=1e-24,
        rtol=1e-8,
    )
    close(
        report["incoming_entropy_j_k"],
        dt * entropy_rate,
        "junction incoming entropy",
        atol=1e-22,
        rtol=1e-8,
    )
    if model.volume_diffusion == "none":
        diffusion = np.zeros_like(diffusion)
    heat = -potential * diffusion
    close(report["capillary_force_n"], force, "branch virtual-work force", atol=1e-22, rtol=1e-8)
    close(
        report["capillary_diffusion_heat_j"],
        dt * heat,
        "branch derived heat",
        atol=1e-25,
        rtol=1e-8,
    )
    rn = (q[1][..., :nc] - q[0][..., :nc]) * volume[:, None, None] + dt * p.area * delta[..., :nc]
    re = (q[1][..., -1] - q[0][..., -1]) * volume[:, None] + dt * (
        p.area * delta[..., -1] - u * force - heat[:, None] * np.array([s.share, 1 - s.share])
    )
    rm = (q[1][..., -2] - q[0][..., -2]) * volume[:, None] + dt * (p.area * delta[..., -2] - force)
    for i in pinned:
        rm[i] = q[1][i, :, -2] * volume[i]
    rv = vv[1][:, 0] - vv[0][:, 0] - dt * (volume_rate(u) + diffusion)
    close(
        rn,
        np.zeros_like(rn),
        "branch component equations",
        atol=5e-11 * abs(q[0][..., :nc]) * volume[:, None, None],
        rtol=0,
    )
    close(
        re,
        np.zeros_like(re),
        "branch energy equations",
        atol=5e-11 * abs(q[0][..., -1]) * volume[:, None] + 1e-22,
        rtol=0,
    )
    masses = (q[0][..., :nc] * volume[:, None, None] * mw).sum(-1)
    close(
        rm,
        np.zeros_like(rm),
        "branch momentum equations",
        atol=5e-11 * masses * np.maximum(abs(uu[0]), 1.0),
        rtol=0,
    )
    close(rv, np.zeros_like(rv), "branch volume equations", atol=5e-11 * volume, rtol=0)
    stocks = [(v * volume[:, None, None]).sum(axis=(0, 1)) for v in q]
    for value, energy in zip(stocks, surface):
        value[-1] += energy
    defect = stocks[1] - stocks[0] - external
    close(
        defect[[*range(nc), nc + 1]],
        np.zeros(nc + 1),
        "branch global N/(E+F)",
        atol=1e-10 * abs(stocks[0][[*range(nc), nc + 1]]) + 1e-22,
        rtol=0,
    )
    mb = report["momentum_balance"]
    for name, value in reaction.items():
        close(
            mb["junction_support_impulses_kg_m_s"][name],
            value,
            "junction support",
            atol=1e-23,
            rtol=1e-8,
        )
    close(
        mb["wall_impulse_vector_kg_m_s"], wall_impulse, "branch wall impulse", atol=1e-23, rtol=1e-8
    )
    close(
        mb["surface_force_resultant_impulse_kg_m_s"],
        surface_impulse,
        "surface resultant",
        atol=1e-23,
        rtol=1e-8,
    )
    axes = np.array(network.axes)
    momenta = [((v[..., -2] * volume[:, None]).sum(-1)[:, None] * axes).sum(0) for v in q]
    vector_defect = (
        momenta[1] - momenta[0] - wall_impulse - surface_impulse - sum(reaction.values())
    )
    close(
        vector_defect,
        np.zeros(3),
        "branch vector momentum",
        atol=1e-10 * (abs(momenta[0]) + abs(momenta[1])) + 1e-22,
        rtol=0,
    )
    entropy = ss[1] - ss[0] - dt * entropy_rate
    if entropy < -128 * (math.ulp(ss[0]) + math.ulp(ss[1])) - 1e-22:
        raise ValueError("Branch entropy decreased")
    return {
        "accepted": True,
        "physical_cells": n,
        "junctions": len(network.junction_volumes),
        "entropy_production_j_k": entropy,
        "maximum_energy_equation_residual_j": float(abs(re).max()),
    }


def audit_switch(before_model, before, after_model, after, report):
    """Independent energy/work of an ideal binary boundary-condition event."""
    old_config = before_model.to_dict()
    new_config = after_model.to_dict()
    old_config.pop("openings", None)
    new_config.pop("openings", None)
    if _digest(old_config) != _digest(new_config):
        raise ValueError("Port event changed physical equipment/fluid/source parameters")
    if (
        before.pipes != after.pipes
        or before.junctions != after.junctions
        or before.time_s != after.time_s
    ):
        raise ValueError("Port event changed actual phase inventory or clock")
    if (
        report["clock_advanced"]
        or report["old_model_id"] != before_model.model_id
        or report["new_model_id"] != after_model.model_id
    ):
        raise ValueError("Port event identity differs")
    if _digest(report["state"]) != _digest(after.to_dict()) or _digest(report["model"]) != _digest(
        after_model.to_dict()
    ):
        raise ValueError("Port event snapshot differs")
    expected = {**before_model.openings, **report["commands"]}
    if expected != after_model.openings:
        raise ValueError("Port command/physical endpoint binding differs")

    def energy(model, state):
        view = model._view(state)
        width = model.source.width
        sigma = model.source.sigma
        dx = model.source.pipe.lengths
        height = [
            dict(c)[model.source.liquid].volume_m3 / (width * x) for c, x in zip(view.cells, dx)
        ]
        result = math.fsum(
            0.5 * sigma * width / ((dx[i] + dx[j]) / 2) * (height[j] - height[i]) ** 2
            for i, j in model.source.surface_edges
        )

        if model.source.wetting is not None:
            from reference_wetting_potential import potential as wall_energy

            result += float(
                np.dot(
                    width * np.array(dx), wall_energy(model.source.wetting.to_dict(), sigma, height)
                )
            )
        return result

    old_energy, new_energy = energy(before_model, before), energy(after_model, after)
    close(
        report["surface_energy_before_j"], old_energy, "old switch surface", atol=1e-25, rtol=1e-10
    )
    close(
        report["surface_energy_after_j"], new_energy, "new switch surface", atol=1e-25, rtol=1e-10
    )
    close(
        report["external_actuator_work_j"],
        new_energy - old_energy,
        "isolation actuator work",
        atol=1e-25,
        rtol=1e-10,
    )
    close(report["augmented_energy_residual_j"], 0.0, "isolation total energy", atol=1e-25, rtol=0)
    return {
        "accepted": True,
        "external_actuator_work_j": new_energy - old_energy,
        "qualification": "Reversible declared-geometry event; no real valve/contact-angle calibration",
    }
