"""Independent NumPy audit of a local gas/film source and its wall reaction.

No runtime profile/force/recovery/inspect methods are called. Scalar reference
PR checks reuse the existing independent EOS implementation. This verifies
constitutive equations, not experimental adequacy of the supplied conductance.
"""

import math

import numpy as np
from check_stratified_pr import Reference


def close(actual, expected, name, *, atol=1e-24, rtol=2e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or np.any(np.abs(a - b) > atol + rtol * np.maximum(np.abs(a), np.abs(b)))
    ):
        raise ValueError("Gas/film source audit differs: " + name)


def audit(source, old, gas, end, new_gas, dt, conductance, report):
    film = source.film
    s, f = film.mechanics.surface, film.fluid
    if report["interface_model"] != source.interface_model or report["model_id"] != source.model_id:
        raise ValueError("Gas/film source audit configuration differs")
    close(report["dt_s"], dt, "source interval", atol=0, rtol=0)
    if report["clock_advanced"] is not False or end.time_s != old.time_s:
        raise ValueError("Local gas/film source advanced report clock")
    if old.mechanics.thickness_m != end.mechanics.thickness_m:
        raise ValueError("Gas/film source changed film geometry")
    if old.mechanics.external_work_j != end.mechanics.external_work_j:
        raise ValueError("Internal gas/film work counted as external drive")
    h = np.asarray(old.mechanics.thickness_m)
    hf = (h + np.roll(h, -1)) / 2
    area = s.spacing[0] * s.width
    ml = s.rho * area * hf
    u0, u1 = (
        np.asarray(old.mechanics.face_velocity_m_s),
        np.asarray(end.mechanics.face_velocity_m_s),
    )
    mw = np.array([c.molar_mass_kg_mol for c in f.components])
    mg = np.array([np.asarray(p.component_moles) @ mw for p in gas])
    pg0 = np.array([p.momentum_kg_m_s for p in gas])
    pg1 = np.array([p.momentum_kg_m_s for p in new_gas])
    k = np.asarray(conductance)
    if not (k.shape == h.shape and np.isfinite(k).all() and (k >= 0).all()):
        raise ValueError("Invalid audit conductance")
    if source.interface_model == "stokes_profile":
        # Integrated no-slip/Navier profile, algebra independent of helper.
        a = hf**3 / (3 * s.mu) + s.slip * hf**2 / s.mu
        b = hf**2 / (2 * s.mu) + s.slip * hf / s.mu
        eta = hf * b / a
        beta = (hf**2 / 12 + s.slip * hf / 3) / (s.mu * (hf / 3 + s.slip))
    else:
        eta, beta = np.ones_like(hf), np.zeros_like(hf)
    j = np.asarray(report["gas_to_liquid_impulse_kg_m_s"])
    tau = (j / dt) / area
    ui = eta * u1 + beta * tau
    wall = (eta - 1) * j
    close(ml * (u1 - u0), eta * j, "liquid impulse")
    close(pg1 - pg0, -j, "gas impulse")
    # An equivalent compliance equation avoids huge k times the cancellation
    # of nearly equal final velocities. At k=0, verify exact zero exchange.
    if np.any(k == 0):
        close(j[k == 0], np.zeros_like(j[k == 0]), "closed interface", atol=0, rtol=0)
    active = k > 0
    response = j / mg + eta**2 * j / ml + beta * tau
    rhs = pg0 / mg - eta * u0
    response[active] += (j[active] / dt) / k[active]
    close(response[active], rhs[active], "interface constitutive response", atol=1e-13)
    qg = np.zeros_like(j)
    qg[active] = (j[active] / dt) * (j[active] / k[active])
    ql = j * beta * tau
    ng, nl = j * j / (2 * mg), (eta * j) ** 2 / (2 * ml)
    work = j * ui
    for key, value in {
        "liquid_net_impulse_kg_m_s": eta * j,
        "stationary_wall_impulse_kg_m_s": wall,
        "interface_shear_pa": tau,
        "interface_velocity_m_s": ui,
        "gas_physical_heat_j": qg,
        "gas_numerical_heat_j": ng,
        "liquid_numerical_heat_j": nl,
        "liquid_profile_heat_j": ql,
        "gas_to_liquid_work_j": work,
    }.items():
        close(report[key], value, key, atol=1e-24)
    close(
        end.mechanics.physical_heat_j - old.mechanics.physical_heat_j,
        float(ql.sum()),
        "film heat ledger",
    )
    close(
        end.mechanics.numerical_heat_j - old.mechanics.numerical_heat_j,
        float(nl.sum()),
        "film temporal ledger",
    )
    close(0.5 * ml * (u1 - u0) * (u1 + u0) + nl + ql, work, "liquid kinetic/surface work")
    close(-work - 0.5 * (pg1 - pg0) * (pg1 + pg0) / mg, qg + ng, "gas kinetic/surface work")
    close(
        report["local_liquid_work_residual_j"],
        0.5 * ml * (u1 - u0) * (u1 + u0) + nl + ql - work,
        "reported liquid work residual",
    )
    close(
        report["local_gas_heat_residual_j"],
        -work - 0.5 * (pg1 - pg0) * (pg1 + pg0) / mg - qg - ng,
        "reported gas heat residual",
    )
    cache = {}
    for before, after in zip(gas, new_gas, strict=True):
        if before.component_moles != after.component_moles or before.volume_m3 != after.volume_m3:
            raise ValueError("Gas/film source changed gas component/volume inventories")
        for p in (before, after):
            n = np.asarray(p.component_moles)
            nt = n.sum()
            x = n / nt
            key = (p.pressure_pa, p.temperature_k, tuple(x))
            if key not in cache:
                cache[key] = Reference(f, np.flatnonzero(x > 0)).state(
                    p.pressure_pa, p.temperature_k, x, "vapor"
                )
            q = cache[key]
            mass = n @ mw
            close(p.volume_m3, nt * q["vm"], "independent gas volume", atol=1e-24)
            energy = nt * q["u"] + 0.5 * p.momentum_kg_m_s**2 / mass
            close(p.total_energy_j, energy, "independent gas calorics", atol=1e-20)
    for before, after in zip(old.phases, end.phases, strict=True):
        if (
            before.component_moles != after.component_moles
            or before.volume_m3 != after.volume_m3
            or before.pressure_pa != after.pressure_pa
            or before.temperature_k != after.temperature_k
        ):
            raise ValueError("Source changed isothermal film material/geometry")
    eg0 = np.array([p.total_energy_j for p in gas])
    eg1 = np.array([p.total_energy_j for p in new_gas])
    close(
        eg1 - eg0,
        -work,
        "gas energy transfer",
        atol=max(math.ulp(p.total_energy_j) for p in gas) * 8 + 1e-24,
    )
    # Check actual stored primal material momentum/energy, not only face fields.
    uc0, uc1 = (u0 + np.roll(u0, 1)) / 2, (u1 + np.roll(u1, 1)) / 2
    for p, v in zip(end.phases, uc1, strict=True):
        close(p.momentum_kg_m_s, s.rho * p.volume_m3 * v, "film material momentum")
    film_bulk_change = []
    for a, b, va, vb in zip(old.phases, end.phases, uc0, uc1, strict=True):
        close(
            b.total_energy_j - a.total_energy_j,
            0.5 * s.rho * a.volume_m3 * (vb - va) * (vb + va),
            "film caloric/kinetic change",
            atol=8 * max(math.ulp(a.total_energy_j), math.ulp(b.total_energy_j)) + 1e-24,
        )
        film_bulk_change.append(b.total_energy_j - a.total_energy_j)
    variance = (
        0.125 * s.rho * area * np.sum(h * ((u1 - np.roll(u1, 1)) ** 2 - (u0 - np.roll(u0, 1)) ** 2))
    )
    energy = math.fsum(
        [*film_bulk_change, float(variance), float(ql.sum() + nl.sum()), *(eg1 - eg0)]
    )
    momentum = math.fsum(
        [float(np.sum(s.rho * area * h * (uc1 - uc0))), *(pg1 - pg0), -float(wall.sum())]
    )
    allowance = (
        128
        * math.fsum(
            math.ulp(p.total_energy_j)
            for phases in (gas, new_gas, old.phases, end.phases)
            for p in phases
        )
        + 1e-22
    )
    close(energy, 0.0, "coupled energy", atol=allowance, rtol=0)
    close(
        report["energy_roundoff_allowance_j"],
        allowance,
        "reported roundoff allowance",
        atol=0,
        rtol=1e-15,
    )
    close(report["energy_residual_j"], energy, "reported energy residual", atol=allowance, rtol=0)
    momentum_allowance = 1e-24 + 2e-11 * float(
        (np.abs(pg0) + np.abs(pg1) + np.abs(ml * u0) + np.abs(ml * u1) + np.abs(wall)).sum()
    )
    close(momentum, 0.0, "coupled impulse with wall reaction", atol=momentum_allowance, rtol=0)
    close(
        report["momentum_residual_kg_m_s"],
        momentum,
        "reported impulse residual",
        atol=momentum_allowance,
    )
    return {
        "accepted": True,
        "cells": len(h),
        "interface_model": source.interface_model,
        "energy_residual_j": energy,
        "energy_allowance_j": allowance,
        "momentum_residual_kg_m_s": momentum,
        "stationary_wall_impulse_kg_m_s": float(wall.sum()),
        "external_drive_work_unchanged": True,
        "qualification": "Independent constitutive/stock audit, not experimental calibration",
    }
