"""Independent finite-volume/PR audit of spatial HEM phase appearance."""

import math

import numpy as np
from check_pr_equilibrium_transport import reference_sound
from check_stratified_pr import Reference
from check_stratified_pr_flow import close
from reference_pr_interphase_relaxation import decode

from surface.pr_equilibrium_euler_pipe import PREquilibriumEulerPipe


def audit(config, report, *, external_exchange=None):
    if config != report["config"]:
        raise ValueError("Euler request differs")
    parsed = PREquilibriumEulerPipe.from_dict(config["model"])  # Metadata only.
    import json

    model = json.loads(json.dumps(parsed.to_dict()))
    if model != report["model"]:
        raise ValueError("Euler normalized model differs")
    f = parsed.fluid
    nc = len(f.names)
    count = len(parsed.lengths)
    volumes = np.array(parsed.volumes)
    mw = np.array([c.molar_mass_kg_mol for c in f.components])

    def packets(state):
        if (
            set(state) != {"cells", "bulk_stocks", "time_s", "next_step_s", "model_id"}
            or state["model_id"] != parsed.model_id
            or len(state["cells"]) != count
            or len(state["bulk_stocks"]) != count
        ):
            raise ValueError("Euler checkpoint registry differs")
        if (
            not math.isfinite(state["time_s"])
            or state["time_s"] < 0
            or not math.isfinite(state["next_step_s"])
            or state["next_step_s"] <= 0
        ):
            raise ValueError("Invalid Euler checkpoint clock")
        bulk = np.array(state["bulk_stocks"], dtype=float)
        if (
            bulk.shape != (count, nc + 2)
            or not np.isfinite(bulk).all()
            or np.any(bulk[:, :nc] < 0)
            or np.any(bulk[:, :nc].sum(1) <= 0)
        ):
            raise ValueError("Invalid conservative stocks")
        conserved = bulk / volumes[:, None]
        velocity = bulk[:, -2] / (bulk[:, :nc] @ mw)
        pressure = []
        sound = []
        entropy = 0.0
        for i, cell in enumerate(state["cells"]):
            if (
                set(cell) != {"phases", "phase_roots", "model_id", "time_s"}
                or cell["time_s"] != state["time_s"]
                or cell["model_id"] != parsed.equilibration.model_id
                or not 1 <= len(cell["phases"]) <= f.max_phases
            ):
                raise ValueError("Invalid active Euler cell metadata")
            data = decode(f, cell, cell["phase_roots"])
            close(sum(q["volume"] for q in data), volumes[i], atol=1e-12, rtol=1e-9)
            actual = np.r_[
                np.sum([q["n"] for q in data], axis=0),
                sum(p["momentum_kg_m_s"] for p in cell["phases"]),
                sum(p["total_energy_j"] for p in cell["phases"]),
            ]
            close(actual, bulk[i], atol=np.array([1e-8] * nc + [1e-8, 1e-6]), rtol=1e-9)
            chemical = []
            phase_sound = []
            for d, root in zip(data, cell["phase_roots"]):
                close(
                    [d["p"], d["t"], d["velocity"]],
                    [data[0]["p"], data[0]["t"], velocity[i]],
                    atol=np.array([1e-6, 1e-8, 1e-9]),
                    rtol=1e-10,
                )
                x = d["n"] / d["n"].sum()
                active = np.flatnonzero(x > 0)
                q = Reference(f, active).state(d["p"], d["t"], x, root)
                chemical.append((active, np.log(x[active]) + q["phi"]))
                phase_sound.append(reference_sound(f, d["p"], d["t"], x, root))
                entropy += d["entropy"]
            for a, b in zip(chemical, chemical[1:]):
                if not np.array_equal(a[0], b[0]):
                    raise ValueError("Active equilibrium component supports differ")
                close(a[1], b[1], atol=1e-7, rtol=0.0)
            bound = abs(velocity[i]) + max(phase_sound)
            if bound > parsed.wave_speed * (1 + 1e-6):
                raise ValueError("Euler acoustic envelope violated")
            pressure.append(data[0]["p"])
            sound.append(bound)
        p = np.array(pressure)
        flux = np.c_[
            conserved[:, :nc] * velocity[:, None],
            conserved[:, -2] * velocity + p,
            (conserved[:, -1] + p) * velocity,
        ]
        return {
            "bulk": bulk,
            "conserved": conserved,
            "flux": flux,
            "sound": np.array(sound),
            "entropy": entropy,
        }

    def ledger(old, new, heat, wall, incoming=None):
        a = np.array(old["bulk_stocks"])
        b = np.array(new["bulk_stocks"])
        r = np.sum(b - a, axis=0)
        if incoming is not None:
            r -= incoming
        r[-1] -= heat
        r[-2] -= wall
        return r

    def verify(old, new, heat, wall, saved, incoming=None):
        r = ledger(old, new, heat, wall, incoming)
        close(r, np.zeros(nc + 2), atol=np.array([1e-8] * nc + [1e-8, 1e-6]), rtol=0.0)
        close(saved, r, atol=np.array([1e-11] * nc + [1e-11, 1e-8]), rtol=0.0)

    old = report["initial"]
    packets(old)
    first = old
    if set(config["initial"]) == {"state"}:
        if old != config["initial"]["state"]:
            raise ValueError("Euler restart differs from request")
    elif set(config["initial"]) == {"cells"}:
        if old["time_s"] != 0.0 or len(config["initial"]["cells"]) != count:
            raise ValueError("Euler initialization grid/clock differs")
        for spec, bulk in zip(config["initial"]["cells"], old["bulk_stocks"]):
            close(
                bulk,
                [*spec["component_moles"], spec["momentum_kg_m_s"], spec["total_energy_j"]],
                atol=1e-12,
                rtol=0.0,
            )
    else:
        raise ValueError("Unknown Euler initialization")
    if len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Euler schedule count differs")
    incoming_total = np.zeros(nc + 2)
    heat_total = 0.0
    wall_total = 0.0
    checked = 0
    max_energy = 0.0
    for command, interval in zip(config["schedule"], report["intervals"]):
        if (
            set(command)
            - {
                "duration_s",
                "heat_rates_w",
                "heat_temperatures_k",
                "maximum_step_s",
                "minimum_step_s",
                "maximum_attempts",
            }
            or not interval["steps"]
        ):
            raise ValueError("Invalid Euler schedule/accepted steps")
        start = old
        incoming_interval = np.zeros(nc + 2)
        heat_interval = 0.0
        wall_interval = 0.0
        heat = np.array(command.get("heat_rates_w", [0.0] * count), dtype=float)
        baths = command.get("heat_temperatures_k", [None] * count)
        if heat.shape != (count,) or not np.isfinite(heat).all() or len(baths) != count:
            raise ValueError("Heat boundary registry differs")
        for power, bath in zip(heat, baths):
            if (
                power
                and bath is None
                or (bath is not None and (not math.isfinite(bath) or bath <= 0))
            ):
                raise ValueError("Missing valid heat reservoir")
        for step in interval["steps"]:
            dt = step["dt_s"]
            new = step["state"]
            if not math.isfinite(dt) or dt <= 0 or dt > parsed.maximum_step_s * (1 + 1e-12):
                raise ValueError("Euler CFL/time limit violated")
            close(
                new["time_s"] - old["time_s"],
                dt,
                atol=8 * max(math.ulp(new["time_s"]), math.ulp(old["time_s"])),
                rtol=0.0,
            )
            a = packets(old)
            b = packets(new)
            q, ff = a["conserved"], a["flux"]
            incoming = np.zeros(nc + 2)
            boundary_entropy = 0.0
            if parsed.boundary_mode == "periodic":
                flux = 0.5 * (ff + np.roll(ff, -1, axis=0)) - 0.5 * parsed.wave_speed * (
                    np.roll(q, -1, axis=0) - q
                )
                div = flux - np.roll(flux, 1, axis=0)
                wall = 0.0
            else:
                interior = 0.5 * (ff[:-1] + ff[1:]) - 0.5 * parsed.wave_speed * (q[1:] - q[:-1])
                left = np.zeros(nc + 2)
                right = left.copy()
                left[-2] = ff[0, -2] - a["sound"][0] * q[0, -2]
                right[-2] = ff[-1, -2] + a["sound"][-1] * q[-1, -2]
                if min(left[-2], right[-2]) <= 0:
                    raise ValueError("Unsupported rigid-wall traction")
                if external_exchange is not None:
                    exchange = external_exchange(old, new, a, dt)
                    faces = exchange["boundary_fluxes"]
                    entropy_faces = exchange["boundary_entropy_fluxes"]
                    if len(faces) != 2 or len(entropy_faces) != 2:
                        raise ValueError("External face registry differs")
                    if faces[0] is not None:
                        left = np.asarray(faces[0], dtype=float)
                    if faces[1] is not None:
                        right = np.asarray(faces[1], dtype=float)
                    if (
                        left.shape != (nc + 2,)
                        or right.shape != (nc + 2,)
                        or not np.isfinite([left, right]).all()
                    ):
                        raise ValueError("Invalid external face")
                    incoming = dt * parsed.area * (left - right)
                    incoming[-2] = 0.0  # Momentum is included in wall/support impulse.
                    boundary_entropy = dt * parsed.area * (entropy_faces[0] - entropy_faces[1])
                flux = np.vstack((left, interior, right))
                div = np.diff(flux, axis=0)
                wall = dt * parsed.area * (left[-2] - right[-2])
            close(step["face_fluxes"], flux, atol=np.array([1e-7] * nc + [1e-5, 1e-3]), rtol=1e-7)
            target = -dt * parsed.area * div
            target[:, -1] += dt * heat
            close(
                b["bulk"] - a["bulk"],
                target,
                atol=np.array([1e-10] * nc + [1e-10, 1e-7]),
                rtol=1e-7,
            )
            supplied = dt * sum(
                power / bath for power, bath in zip(heat, baths) if bath is not None
            )
            close(step["incoming_component_moles"], incoming[:nc], atol=1e-10, rtol=1e-7)
            close(
                [step["boundary_energy_j"], step["boundary_entropy_j_k"]],
                [incoming[-1], boundary_entropy],
                atol=1e-8,
                rtol=1e-7,
            )
            production = b["entropy"] - a["entropy"] - supplied - boundary_entropy
            if production < -1e-7:
                raise ValueError("Euler entropy balance violated")
            external = dt * sum(heat)
            close(
                [
                    step["external_heat_j"],
                    step["entropy_supply_j_k"],
                    step["entropy_production_j_k"],
                    step["wall_impulse_kg_m_s"],
                ],
                [external, supplied, production, wall],
                atol=np.array([1e-8, 1e-9, 1e-7, 1e-9]),
                rtol=1e-7,
            )
            if step["phase_counts"] != [len(c["phases"]) for c in new["cells"]]:
                raise ValueError("Euler phase count diagnostics differ")
            verify(old, new, external, wall, step["conservation_residuals"], incoming)
            max_energy = max(max_energy, abs(ledger(old, new, external, wall, incoming)[-1]))
            old = new
            incoming_interval += incoming
            heat_interval += external
            wall_interval += wall
            checked += 1
        published = interval["state"]
        packets(published)
        if {k: v for k, v in published.items() if k != "next_step_s"} != {
            k: v for k, v in old.items() if k != "next_step_s"
        }:
            raise ValueError("Published Euler interval state differs")
        close(
            published["time_s"],
            start["time_s"] + command["duration_s"],
            atol=8 * math.ulp(published["time_s"]),
            rtol=0.0,
        )
        verify(
            start,
            published,
            heat_interval,
            wall_interval,
            interval["conservation_residuals"],
            incoming_interval,
        )
        incoming_total += incoming_interval
        heat_total += heat_interval
        wall_total += wall_interval
        old = published
    if old != report["state"]:
        raise ValueError("Final Euler state differs")
    verify(first, old, heat_total, wall_total, report["conservation_residuals"], incoming_total)
    return {
        "accepted": True,
        "cells": count,
        "checked_substeps": checked,
        "maximum_energy_residual_j": max_energy,
        "closure": "homogeneous_equilibrium_Euler",
        "phase_counts": [len(c["phases"]) for c in old["cells"]],
    }
