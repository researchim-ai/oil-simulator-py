"""Independent scalar PR/64-point path and 32-point column audit.

No runtime pipe step, homogeneous recovery or column equilibrium solve is used.
SciPy is confined to this CPU reference verifier.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from check_pr_thermochemical_relaxation import audit as source_audit
from check_stratified_pr import Reference, grid
from reference_pr_active_multifluid import face, packets, reconstruct
from scipy.optimize import brentq, root

from surface.peng_robinson import R
from surface.pr_active_surface import PRActiveSurfaceNetwork, PRActiveSurfaceState
from surface.pr_phase_equilibration import PRPhaseEquilibriumState

FIELDS = ("conserved", "alpha", "pressure", "temperature", "velocity", "pressure_precision", "flux")


def close(a, b, *, atol=1e-7, rtol=1e-9):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise ValueError(f"Independent active surface audit differs: {a!r} vs {b!r}")


def sample_packet(pipe, cell):
    volume = math.fsum(p["volume_m3"] for p in cell.values())
    proxy = SimpleNamespace(
        fluid=pipe.fluid,
        names=pipe.names,
        slots=pipe.slots,
        volumes=[volume],
        wave_speed=pipe.wave_speed,
        model_id=pipe.model_id,
    )
    data = packets(proxy, {"cells": [cell], "time_s": 0.0, "model_id": pipe.model_id})
    return {k: data[k][0] for k in FIELDS}


def column_stock(column, state):
    f = column.fluid
    n = np.array(state.component_moles)
    active = np.flatnonzero(n > 0)
    ref = Reference(f, active)
    mw = np.array([c.molar_mass_kg_mol for c in f.components])
    lam = np.array(state.chemical_constants)[active]
    t = state.temperature_k
    heights, volumes = grid(column.geometry, state.interface_heights_m, 32)
    total_n = np.zeros(len(n))
    energy = entropy = 0.0
    for i, (hh, vv) in enumerate(zip(heights, volumes)):
        x = np.array(state.phase_seeds[i])
        p = math.exp(state.solution_variables[i * state.quadrature_order * len(active)])
        initial = np.r_[math.log(p), np.log(x[active][:-1] / x[active][-1])]
        for h, volume in zip(hh, vv):

            def residual(y, h=h, i=i):
                p, x, q = ref.local(y, t, state.phase_roots[i])
                return (
                    np.log(x[active])
                    + q["phi"]
                    + math.log(p / f.pref)
                    + mw[active] * column.gravity_m_s2 * h / (R * t)
                    - lam
                )

            result = root(residual, initial, tol=1e-10)
            if np.max(np.abs(residual(result.x))) > 1e-8:
                raise ValueError("Independent column chemical equilibrium failed")
            p, x, q = ref.local(result.x, t, state.phase_roots[i])
            initial = result.x
            amount = volume / q["vm"]
            total_n += amount * x
            energy += amount * q["u"] + volume * q["rho"] * column.gravity_m_s2 * (
                column.elevation_m + h
            )
            entropy += amount * q["s"]
    close(total_n, n, atol=1e-7, rtol=1e-8)
    close(energy, state.total_energy_j, atol=1e-5, rtol=1e-8)
    return total_n, energy, entropy


def friction_audit(pipe, before, after, report):
    if pipe.friction.to_dict()["model"] == "laminar_annular_wallis_interface_v1":
        from reference_pr_annular_source import audit

        return audit(pipe, before, after, report, close)
    a = packets(pipe, before.to_dict())
    b = packets(pipe, after.to_dict())
    mw = np.array([c.molar_mass_kg_mol for c in pipe.fluid.components])
    losses = []
    impulses = []
    factors = []
    re_values = []
    for i, cell in enumerate(before.cells):
        if tuple(n for n, _ in cell) != tuple(n for n, _ in after.cells[i]):
            raise ValueError("Wall source changed active registry")
        for name, stock in cell:
            target = dict(after.cells[i])[name]
            mass = np.array(stock.component_moles) @ mw
            rho = mass / stock.volume_m3
            old_u = stock.momentum_kg_m_s / mass
            mu = pipe.friction.viscosity[name]
            re = rho * abs(old_u) * pipe.diameter / mu
            if re == 0:
                factor = 0.0
            elif re <= 2000:
                factor = 64 / re
            else:
                z = brentq(
                    lambda z, re=re: (
                        z
                        + 2
                        * math.log10(pipe.friction.roughness / pipe.diameter / 3.7 + 2.51 * z / re)
                    ),
                    1,
                    100,
                    xtol=1e-13,
                )
                fraction = min(1.0, (re - 2000) / 2000)
                blend = fraction * fraction * (3 - 2 * fraction)
                factor = (1 - blend) * 64 / re + blend / z**2
            dt = report["dt_s"]
            if re <= 2000:
                expected = old_u / (1 + dt * 32 * mu / (rho * pipe.diameter**2))
            else:
                expected = (
                    2
                    * old_u
                    / (1 + math.sqrt(1 + 4 * dt * factor / (2 * pipe.diameter) * abs(old_u)))
                )
            close(target.momentum_kg_m_s, mass * expected, atol=1e-12, rtol=1e-10)
            close(target.component_moles, stock.component_moles, atol=0.0, rtol=0.0)
            close(target.volume_m3, stock.volume_m3, atol=0.0, rtol=0.0)
            close(target.total_energy_j, stock.total_energy_j, atol=0.0, rtol=0.0)
            loss = 0.5 * mass * (old_u - expected) * (old_u + expected)
            losses.append(loss)
            impulses.append(mass * (expected - old_u))
            factors.append(factor)
            re_values.append(re)
    close(losses, report["dissipation_j"], atol=1e-9, rtol=1e-9)
    close(factors, report["darcy_factors"], atol=1e-12, rtol=1e-10)
    close(re_values, report["reynolds_numbers"], atol=1e-8, rtol=1e-10)
    close(math.fsum(impulses), report["wall_impulse_kg_m_s"], atol=1e-12, rtol=1e-9)
    if b["entropy"] - a["entropy"] < -1e-7:
        raise ValueError("Independent wall entropy decreased")
    close(b["entropy"] - a["entropy"], report["entropy_production_j_k"], atol=1e-7, rtol=1e-9)


def pipe_audit(pipe, old, new, report):
    data = packets(pipe, old.to_dict())
    packets(pipe, new.to_dict())
    count = len(old.cells)

    reference_delta = reference_alpha = reference_faces = None
    if pipe.reference_specs is not None:
        ref = {"cells": pipe.reference_specs, "time_s": 0.0, "model_id": pipe.model_id}
        reference_data, reference_delta, reference_alpha, reference_faces = reconstruct(pipe, ref)
        mw = np.array([c.molar_mass_kg_mol for c in pipe.fluid.components])
        for i in range(count):
            reference_delta[i, ..., -2] += (
                pipe.gravity
                * (reference_data["conserved"][i, ..., :-2] @ mw)
                * (pipe.elevations[i + 1] - pipe.elevations[i])
            )
            fl, fr = (
                reference_faces[i]["flux"][..., :-2] @ mw,
                reference_faces[i + 1]["flux"][..., :-2] @ mw,
            )
            reference_delta[i, ..., -1] += pipe.gravity * (
                (pipe.elevations[i + 1] - pipe.centers[i]) * fr
                - (pipe.elevations[i] - pipe.centers[i]) * fl
            )

    def cell(i):
        return {k: data[k][i].copy() for k in FIELDS}

    def reflected(i):
        d = cell(i)
        d["velocity"] *= -1
        d["conserved"][..., -2] *= -1
        d["flux"][..., :-2] *= -1
        d["flux"][..., -1] *= -1
        return d

    faces = []
    for index in range(count + 1):
        if pipe.boundary_mode == "periodic":
            l, r = cell((index - 1) % count), cell(index % count)
        else:
            l = cell(index - 1) if index > 0 else reflected(0)
            r = cell(index) if index < count else reflected(-1)
        side = 0 if index == 0 else 1 if index == count else None
        rows = report["boundary_sample_fluxes"][side] if side is not None else []
        if rows:
            aggregate = None
            weight_sum = 0.0
            for row in rows:
                ghost = sample_packet(pipe, row["cell"])
                values = list(
                    face(ghost, r, pipe.wave_speed)
                    if side == 0
                    else face(l, ghost, pipe.wave_speed)
                )
                # Copy only arrays, not contact metadata, before gravity correction.
                dm, dp, am, ap, ff = values
                fan = ff["flux"].copy()
                zface = pipe.elevations[0 if side == 0 else -1]
                massflux = fan[..., :-2] @ np.array(
                    [c.molar_mass_kg_mol for c in pipe.fluid.components]
                )
                correction = pipe.gravity * (row["elevation_m"] - zface) * massflux
                fan[..., -1] += correction
                if side == 0:
                    dp[..., -1] -= correction
                else:
                    dm[..., -1] += correction
                close(fan, row["flux"], atol=1e-5, rtol=1e-10)
                current = [dm, dp, am, ap, fan]
                weight = row["weight"]
                if not 0 < weight <= 1:
                    raise ValueError("Invalid aperture area weight")
                aggregate = (
                    [weight * v for v in current]
                    if aggregate is None
                    else [a + weight * b for a, b in zip(aggregate, current)]
                )
                weight_sum += weight
            close(weight_sum, 1.0, atol=1e-12, rtol=0.0)
            values = aggregate
        else:
            boundary = report["boundary_cells"][side] if side is not None else None
            if boundary is not None:
                if side == 0:
                    l = sample_packet(pipe, boundary)
                else:
                    r = sample_packet(pipe, boundary)
            dm, dp, am, ap, ff = face(l, r, pipe.wave_speed)
            values = [dm, dp, am, ap, ff["flux"]]
        close(
            values[-1] if reference_faces is None else values[-1] - reference_faces[index]["flux"],
            report["face_fluxes"][
                (index - 1) % count if pipe.boundary_mode == "periodic" else index
            ],
            atol=1e-5,
            rtol=1e-10,
        )
        faces.append(values)
    dt = report["dt_s"]
    mw = np.array([c.molar_mass_kg_mol for c in pipe.fluid.components])
    transport = PRActiveMultifluidState.from_dict(report["transport_state"])
    after = packets(pipe, transport.to_dict())
    for i, length in enumerate(pipe.lengths):
        d = faces[i + 1][0] + faces[i][1]
        ad = faces[i + 1][2] + faces[i][3]
        if pipe.gravity:
            mass = data["conserved"][i, ..., :-2] @ mw
            d[..., -2] += pipe.gravity * mass * (pipe.elevations[i + 1] - pipe.elevations[i])
            fl, fr = faces[i][-1][..., :-2] @ mw, faces[i + 1][-1][..., :-2] @ mw
            d[..., -1] += pipe.gravity * (
                (pipe.elevations[i + 1] - pipe.centers[i]) * fr
                - (pipe.elevations[i] - pipe.centers[i]) * fl
            )
        if reference_delta is not None:
            d -= reference_delta[i]
            ad -= reference_alpha[i]
        close(
            after["conserved"][i] * pipe.volumes[i],
            (data["conserved"][i] - dt / length * d) * pipe.volumes[i],
            atol=1e-6,
            rtol=1e-9,
        )
        close(after["alpha"][i], data["alpha"][i] - dt / length * ad, atol=1e-12, rtol=1e-9)
    heated = transport
    if report.get("thermal") is not None:
        from reference_pr_active_heat import audit as heat_audit

        heated = PRActiveMultifluidState.from_dict(report["heated_state"])
        heat_audit(pipe, transport, heated, report["thermal"], close)
        close(
            report.get("external_heat_j", 0.0),
            report["thermal"]["external_heat_j"],
            atol=1e-11,
            rtol=1e-12,
        )
    elif report.get("external_heat_j", 0.0) != 0:
        raise ValueError("Unaccounted active pipe heat")
    chemical = PRActiveMultifluidState.from_dict(report["chemical_state"])
    if pipe.chemical is not None:
        for cell, source in zip(heated.cells, report["sources"]):
            start = PRPhaseEquilibriumState(
                tuple(p for _, p in cell),
                tuple(pipe.slots[pipe.names.index(n)]["root"] for n, _ in cell),
                pipe.chemical.model_id,
                old.time_s,
            )
            target = PRPhaseEquilibriumState.from_dict(source["state"])
            source_audit(
                *json.loads(
                    json.dumps([pipe.chemical.to_dict(), start.to_dict(), target.to_dict(), source])
                )
            )
        if len(report["sources"]) != len(chemical.cells):
            raise ValueError("Chemical source registry differs")
        for cell, source in zip(chemical.to_dict()["cells"], report["sources"]):
            actual = sorted(json.dumps(p, sort_keys=True) for p in cell.values())
            expected = sorted(json.dumps(p, sort_keys=True) for p in source["state"]["phases"])
            if actual != expected:
                raise ValueError("Chemical phase stocks differ from accepted pipe cells")
    elif report["sources"]:
        raise ValueError("Unexpected chemical source in transport-only model")
    elif chemical.cells != heated.cells:
        raise ValueError("Source-free transport state changed before wall source")
    if pipe.friction is not None:
        friction_audit(pipe, chemical, new, report["friction"])
    elif report["friction"] is not None or chemical.cells != new.cells:
        raise ValueError("Unconfigured wall source changed stocks")
    return True


# Import after reference functions so this module remains usable as a verifier library.
from surface.pr_active_multifluid_pipe import PRActiveMultifluidState


def aperture_audit(model, before, report):
    mw = np.array([c.molar_mass_kg_mol for c in model.fluid.components])
    rates = {n: np.zeros(len(mw) + 1) for n in model.columns}
    for name, entry in model.pipes.items():
        pipe = entry["pipe"]
        r = report["pipes"][name]
        for side, node in enumerate(entry["boundary_nodes"]):
            if node not in model.ports:
                continue
            port = model.ports[node]
            column = model.columns[port.column]
            state = dict(before.columns)[port.column]
            rows = r["boundary_sample_fluxes"][side]
            opened = report["port_positions"][node] > 0
            if not opened:
                if rows:
                    raise ValueError("Isolated aperture carries quadrature rays")
                continue
            if not rows:
                raise ValueError("Open aperture has no finite-area rays")
            active = np.flatnonzero(np.array(state.component_moles) > 0)
            ref = Reference(model.fluid, active)
            occupied = np.zeros(len(state.phase_roots))
            wall_weight = 0.0
            for row in rows:
                if row["kind"] == "wall":
                    wall_weight += row["weight"]
                    current = dict(before.pipes)[name].to_dict()["cells"][0 if side == 0 else -1]
                    if row["cell"].keys() != current.keys():
                        raise ValueError("Closed aperture wall ghost registry differs")
                    for phase_name, stock in row["cell"].items():
                        expected = dict(current[phase_name])
                        expected["momentum_kg_m_s"] *= -1
                        for key, value in expected.items():
                            close(stock[key], value, atol=0.0, rtol=0.0)
                    continue
                h = row["elevation_m"] - column.elevation_m
                if (
                    len(row["cell"]) != 1
                    or not port.height_m - pipe.diameter / 2
                    <= h
                    <= port.height_m + pipe.diameter / 2
                ):
                    raise ValueError("Aperture ray lies outside the actual finite opening")
                phase = next(iter(row["cell"].values()))
                x = np.array(phase["component_moles"]) / sum(phase["component_moles"])
                index = int(np.searchsorted(state.interface_heights_m, h, side="left"))
                q = ref.state(
                    phase["pressure_pa"], phase["temperature_k"], x, state.phase_roots[index]
                )
                chemical = (
                    np.log(x[active])
                    + q["phi"]
                    + math.log(phase["pressure_pa"] / model.fluid.pref)
                    + mw[active] * model.gravity * h / (R * state.temperature_k)
                )
                close(chemical, np.array(state.chemical_constants)[active], atol=1e-8, rtol=0.0)
                close(phase["temperature_k"], state.temperature_k, atol=1e-9, rtol=0.0)
                occupied[index] += row["weight"]
            radius = pipe.diameter / 2

            def area_primitive(y, radius=radius, port=port):
                y = max(-radius, min(radius, y - port.height_m))
                return y * math.sqrt(
                    max(0.0, radius * radius - y * y)
                ) + radius * radius * math.asin(y / radius)

            bounds = [0.0, *state.interface_heights_m, column.geometry.height_m]
            expected = [
                (area_primitive(b) - area_primitive(a)) / (math.pi * radius * radius)
                for a, b in zip(bounds, bounds[1:])
            ]
            position = report["port_positions"][node]
            close(occupied, np.array(expected) * position, atol=1e-12, rtol=1e-9)
            close(wall_weight, 1 - position, atol=1e-12, rtol=0.0)
            index = 0 if side == 0 else -1
            fan = np.array(r["face_fluxes"][index]).sum(0)
            energy = fan[-1] + model.gravity * pipe.elevations[index] * (fan[:-2] @ mw)
            rates[port.column] += (-1.0 if side == 0 else 1.0) * pipe.area * np.r_[fan[:-2], energy]
    for name, values in rates.items():
        close(values, report["columns"][name]["incoming_rates"], atol=1e-7, rtol=1e-9)


def audit(model, before, after, report):
    if after.time_s - before.time_s != report["dt_s"]:
        raise ValueError("Surface physical clock differs")
    aperture_audit(model, before, report)
    for name, old in before.pipes:
        pipe_audit(model.pipes[name]["pipe"], old, dict(after.pipes)[name], report["pipes"][name])
    for name, old in before.columns:
        c = model.columns[name]
        new = dict(after.columns)[name]
        a = column_stock(c, old)
        b = column_stock(c, new)
        rates = report["columns"][name]["incoming_rates"]
        close(b[0] - a[0], np.array(rates[:-1]) * report["dt_s"], atol=2e-7, rtol=1e-7)
        thermal = report["columns"][name]["thermal"]
        heat = 0.0 if thermal is None else thermal["fluid_heat_j"]
        close(b[1] - a[1], rates[-1] * report["dt_s"] + heat, atol=2e-5, rtol=1e-7)
        if name in model.walls:
            if thermal is None:
                raise ValueError("Physical column wall missing in report")
            wall = model.walls[name]
            ta = dict(before.wall_temperatures)[name]
            tb = dict(after.wall_temperatures)[name]
            dt = report["dt_s"]
            q = dt * wall.fluid_ua_w_k * (tb - new.temperature_k)
            qa = dt * wall.ambient_ua_w_k * (wall.ambient_temperature_k - tb)
            close(heat, q, atol=1e-7, rtol=1e-8)
            close(thermal["ambient_heat_j"], qa, atol=1e-7, rtol=1e-8)
            close(wall.heat_capacity_j_k * (tb - ta), qa - q, atol=1e-5, rtol=1e-8)
            production = (
                b[2]
                - a[2]
                + wall.heat_capacity_j_k * math.log(tb / ta)
                - dt * report["columns"][name]["incoming_entropy_w_k"]
                - qa / wall.ambient_temperature_k
            )
            close(
                production, report["columns"][name]["entropy_production_j_k"], atol=1e-6, rtol=1e-7
            )
        elif thermal is not None:
            raise ValueError("Unexpected thermal wall in report")

    # Sum stored components and absolute-datum energies independently.
    def totals(s):
        n = np.zeros(len(model.fluid.names))
        energy = 0.0
        mw = np.array([c.molar_mass_kg_mol for c in model.fluid.components])
        for name, ps in s.pipes:
            p = model.pipes[name]["pipe"]
            for row, z in zip(ps.cells, p.centers):
                for _, phase in row:
                    n += phase.component_moles
                    energy += phase.total_energy_j + model.gravity * z * (
                        np.array(phase.component_moles) @ mw
                    )
        for _, cs in s.columns:
            n += cs.component_moles
            energy += cs.total_energy_j
        for name, t in s.wall_temperatures:
            energy += model.walls[name].heat_capacity_j_k * (
                t - model.walls[name].reference_temperature_k
            )
        if s.equipment is not None:
            for node, (_, stock) in zip(model.equipment.nodes, s.equipment.nodes):
                n += stock.component_moles
                energy += (
                    stock.total_energy_j
                    + model.gravity
                    * model.equipment.reference_elevation_m
                    * (np.array(stock.component_moles) @ mw)
                )
                if node.wall is not None:
                    energy += node.wall.heat_capacity_j_k * stock.wall_temperature_k
            for name, speed in s.equipment.rotors:
                energy += 0.5 * model.equipment._drives[name].drive.inertia_kg_m2 * speed**2
        return n, energy

    a, b = totals(before), totals(after)
    close(a[0], b[0], atol=1e-7, rtol=1e-10)
    close(b[1] - a[1], report["external_energy_j"], atol=1e-5, rtol=1e-9)
    return {"accepted": True, "column_quadrature_order": 32, "path_quadrature_order": 64}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path, nargs="?")
    args = parser.parse_args()
    path = args.report or args.config
    data = json.loads(path.read_text())
    if args.report is not None and json.loads(args.config.read_text()) != data["request"]:
        raise ValueError("Surface report request differs from configuration")
    from run_pr_active_surface import fingerprints

    from surface.operation import _digest

    payload = dict(data)
    checksum = payload.pop("checksum")
    if (
        data["format"] != "pr_active_surface_run_v1"
        or _digest(payload) != checksum
        or data["source_hashes"] != fingerprints()
    ):
        raise ValueError("Stale or modified active surface report")
    model = PRActiveSurfaceNetwork.from_dict(data["system"])
    old = PRActiveSurfaceState.from_dict(data["initial"])
    schedule = data["request"].get("schedule", [{"duration_s": data["request"].get("duration_s")}])
    if len(schedule) != len(data["intervals"]):
        raise ValueError("Operating schedule interval count differs")
    commands = {k: dict(v) for k, v in data["request"].get("commands", {}).items()}
    flattened = []
    for period, interval in zip(schedule, data["intervals"], strict=True):
        for name, values in period.get("commands", {}).items():
            commands[name] = {**commands.get(name, {}), **values}
        if interval["commands"] != commands:
            raise ValueError("Persisted operating commands differ from schedule")
        start_time = old.time_s
        for step in interval["steps"]:
            expected = {n: True for n in model.ports}
            expected.update(commands.get("port_openings", {}))
            if step["port_openings"] != expected:
                raise ValueError("Aperture commands differ from requested isolation")
            positions = {n: 1.0 for n in model.ports}
            positions.update(commands.get("port_positions", {}))
            for n, v in expected.items():
                if not v:
                    positions[n] = 0.0
            if step["port_positions"] != positions:
                raise ValueError("Aperture area fractions differ from request")
            new = PRActiveSurfaceState.from_dict(step["state"])
            from reference_pr_active_heat import check_commands

            for name, record in step["pipes"].items():
                check_commands(
                    record,
                    commands.get("pipe_heat_w", {}).get(name),
                    commands.get("pipe_heat_temperatures_k", {}).get(name),
                )
            audit(model, old, new, step)
            old = new
        close(
            old.time_s, start_time + period["duration_s"], atol=8 * math.ulp(old.time_s), rtol=0.0
        )
        if _digest(interval["state"]) != _digest(old.to_dict()):
            # advance changes only the next-step recommendation on publication.
            published = PRActiveSurfaceState.from_dict(interval["state"])
            if _digest({**old.to_dict(), "next_step_s": published.next_step_s}) != _digest(
                published.to_dict()
            ):
                raise ValueError("Published interval differs from accepted physical stocks")
            old = published
        flattened.extend(interval["steps"])
    if flattened != data["result"]["steps"] or _digest(old.to_dict()) != _digest(
        data["result"]["state"]
    ):
        raise ValueError("Final surface result differs from scheduled intervals")
    print(json.dumps({"accepted": True, "steps": len(data["result"]["steps"])}))


if __name__ == "__main__":
    main()
