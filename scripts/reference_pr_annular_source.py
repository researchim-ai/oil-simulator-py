"""Scalar annular-source audit, independent radial integration and linear solve.

Does not import runtime mobility, friction, UV recovery or time step methods.
SciPy is restricted to this reference checker, never the runtime.
"""

import math

import numpy as np
from reference_pr_active_multifluid import packets
from scipy.integrate import quad
from scipy.optimize import brentq


def resistance(alpha, diameter, viscosity):
    radius = diameter / 2
    inner = radius * math.sqrt(1 - alpha)

    def w(r):
        return (radius - r) * (radius + r) / 4 + inner**2 / 2 * math.log1p((r - radius) / radius)

    def h(r):
        return -inner * math.log1p((r - radius) / radius)

    volume = radius**2 - inner**2
    wm = quad(lambda r: 2 * r * w(r) / volume, inner, radius, epsabs=1e-18)[0]
    hm = quad(lambda r: 2 * r * h(r) / volume, inner, radius, epsabs=1e-18)[0]
    eta = w(inner) / wm
    beta = (h(inner) - eta * hm) / viscosity
    wall = math.pi * volume * viscosity / wm
    return eta, beta, wall, 2 * math.pi * inner


def audit(pipe, before, after, report, close):
    if report["model"] != "laminar_annular_wallis_interface_v1":
        raise ValueError("Annular source model differs")
    old_q, new_q = packets(pipe, before.to_dict()), packets(pipe, after.to_dict())
    mw = np.array([c.molar_mass_kg_mol for c in pipe.fluid.components])
    dt = report["dt_s"]
    heat, physical, temporal, work, impulses, locations, diagnostics = [], [], [], [], [], [], []
    wetting = []
    for i, cell in enumerate(before.cells):
        names = tuple(n for n, _ in cell)
        if names != tuple(n for n, _ in after.cells[i]):
            raise ValueError("Annular source changed active registry")
        stocks = dict(cell)
        masses = {n: np.array(p.component_moles) @ mw for n, p in cell}
        speeds = {n: p.momentum_kg_m_s / masses[n] for n, p in cell}
        final_u, powers, phys = {}, {}, {}
        if len(cell) == 2:
            gas, liquid = pipe.friction.gas_slot, pipe.friction.liquid_slot
            mg, ml = masses[gas], masses[liquid]
            ag = stocks[gas].volume_m3
            alpha = stocks[liquid].volume_m3 / pipe.volumes[i]
            eta, beta, c, perimeter = resistance(
                alpha, pipe.diameter, pipe.friction.viscosity[liquid]
            )
            delta = pipe.diameter / 2 * (1 - math.sqrt(1 - alpha))
            k0 = 0.5 * 0.005 * (1 + 300 * delta / pipe.diameter) * mg / ag
            drive = speeds[gas] - eta * speeds[liquid]
            re_gas = (
                mg
                / ag
                * abs(drive)
                * pipe.diameter
                * math.sqrt(1 - alpha)
                / pipe.friction.viscosity[gas]
            )
            b = max(0.0, min(1.0, (re_gas - 2000) / 2000))
            b = b * b * (3 - 2 * b)
            linear = (
                (1 - b)
                * 4
                * pipe.friction.viscosity[gas]
                / (pipe.diameter / 2 * math.sqrt(1 - alpha))
            )
            quadratic = b * k0
            aa = 1 + beta * linear
            slip = 2 * drive / (aa + math.sqrt(aa**2 + 4 * beta * quadratic * abs(drive)))
            effective = (linear + quadratic * abs(slip)) / (
                1 + beta * (linear + quadratic * abs(slip))
            )
            k = pipe.lengths[i] * perimeter * effective
            wall = pipe.lengths[i] * c
            matrix = np.array([[mg / dt + k, -eta * k], [-eta * k, ml / dt + eta**2 * k + wall]])
            ug, ul = np.linalg.solve(matrix, [mg / dt * speeds[gas], ml / dt * speeds[liquid]])
            tau = effective * (ug - eta * ul)
            ui = eta * ul + beta * tau
            if pipe.friction.wetting is not None:
                spec = pipe.friction.wetting.to_dict()
                radius = pipe.diameter / 2
                inner = radius * math.sqrt(1 - alpha)
                mu = pipe.friction.viscosity[liquid]
                driving = (perimeter * eta * tau - c * ul) / (math.pi * radius**2 * alpha)
                constant = (-inner * tau - driving * inner**2 / 2) / mu

                def profile(r, driving=driving, constant=constant, mu=mu, radius=radius):
                    return driving * (r - radius) * (r + radius) / (4 * mu) + constant * math.log1p(
                        (r - radius) / radius
                    )

                rho = ml / stocks[liquid].volume_m3
                force = (
                    0.5
                    * rho
                    * quad(
                        lambda r, radius=radius: profile(r) ** 2 * r / radius,
                        inner,
                        radius,
                        epsabs=1e-15,
                    )[0]
                )
                lo = spec["surface_tension_n_m"] * (
                    1 - math.cos(math.radians(spec["receding_contact_angle_deg"]))
                )
                hi = spec["surface_tension_n_m"] * (
                    1 - math.cos(math.radians(spec["advancing_contact_angle_deg"]))
                )
                ratio = 1 - inner / radius
                wall_slope = (driving * radius * alpha / 2 - inner / radius * tau) / mu
                valid = (
                    ratio <= spec["maximum_curvature_ratio"]
                    and ul * ui >= 0
                    and ul * wall_slope <= 0
                    and (ul != 0 or tau == 0)
                )
                supported = valid and force >= hi
                dry = valid and force < lo
                wetting.append(
                    {
                        "cell": i,
                        "momentum_force_n_m": force,
                        "receding_force_n_m": lo,
                        "advancing_force_n_m": hi,
                        "profile_valid": valid,
                        "full_wetting_supported": supported,
                        "dry_patch_supported": dry,
                        "history_dependent": valid and not supported and not dry,
                        "curvature_ratio": ratio,
                    }
                )
                if spec["require_full_wetting"] and not supported:
                    raise ValueError("Unsupported wetting accepted by annular source")
            transfer = dt * pipe.lengths[i] * perimeter * tau * ui
            powers = {gas: -transfer, liquid: transfer}
            final_u = {gas: ug, liquid: ul}
            phys = {
                gas: dt * pipe.lengths[i] * perimeter * tau * (ug - ui),
                liquid: dt * (wall * ul**2 + pipe.lengths[i] * perimeter * beta * tau**2),
            }
            diagnostics.append((i, alpha, eta, beta, c, effective, tau, ui))
        else:
            name, stock = cell[0]
            rho, u = masses[name] / stock.volume_m3, speeds[name]
            mu = pipe.friction.viscosity[name]
            re = rho * abs(u) * pipe.diameter / mu
            if re <= 2000:
                un = u / (1 + dt * 32 * mu / (rho * pipe.diameter**2))
            else:
                z = brentq(
                    lambda z, re=re: (
                        z
                        + 2
                        * math.log10(pipe.friction.roughness / pipe.diameter / 3.7 + 2.51 * z / re)
                    ),
                    1.0,
                    100.0,
                )
                t = min(1.0, (re - 2000) / 2000)
                b = t * t * (3 - 2 * t)
                factor = (1 - b) * 64 / re + b / z**2
                un = 2 * u / (1 + math.sqrt(1 + 2 * dt * factor / pipe.diameter * abs(u)))
            final_u[name], powers[name] = un, 0.0
            phys[name] = masses[name] * un * (u - un)
        for name, stock in cell:
            target = dict(after.cells[i])[name]
            close(target.component_moles, stock.component_moles, atol=0.0, rtol=0.0)
            close(target.volume_m3, stock.volume_m3, atol=0.0, rtol=0.0)
            close(target.momentum_kg_m_s, masses[name] * final_u[name], atol=1e-11, rtol=1e-9)
            close(target.total_energy_j, stock.total_energy_j + powers[name], atol=1e-9, rtol=1e-12)
            gain = powers[name] + 0.5 * masses[name] * (speeds[name] - final_u[name]) * (
                speeds[name] + final_u[name]
            )
            temporal.append(0.5 * masses[name] * (speeds[name] - final_u[name]) ** 2)
            physical.append(phys[name])
            work.append(powers[name])
            heat.append(gain)
            impulses.append(masses[name] * (final_u[name] - speeds[name]))
            locations.append([i, name])
    if [list(x) for x in report["active_locations"]] != locations:
        raise ValueError("Annular source locations differ")
    for values, key in [
        (heat, "dissipation_j"),
        (physical, "physical_dissipation_j"),
        (temporal, "temporal_dissipation_j"),
        (work, "interface_work_j"),
    ]:
        close(values, report[key], atol=1e-9, rtol=1e-8)
    if len(diagnostics) != len(report["annular_cells"]):
        raise ValueError("Annular diagnostics registry differs")
    keys = (
        "cell",
        "alpha_l",
        "eta",
        "beta",
        "wall_resistance_per_m",
        "effective_shear",
        "tau_pa",
        "interface_velocity_m_s",
    )
    for expected, found in zip(diagnostics, report["annular_cells"]):
        close(expected, [found[k] for k in keys], atol=1e-9, rtol=1e-8)
        cell = dict(before.cells[int(found["cell"])])
        p = cell[pipe.friction.liquid_slot]
        m = np.array(p.component_moles) @ mw
        re = (
            m
            / pipe.lengths[int(found["cell"])]
            * abs(p.momentum_kg_m_s / m)
            / (math.pi * pipe.diameter * pipe.friction.viscosity[pipe.friction.liquid_slot])
        )
        close(re, found["film_reynolds"], atol=1e-10, rtol=1e-9)
        gas = cell[pipe.friction.gas_slot]
        mg = np.array(gas.component_moles) @ mw
        drive = gas.momentum_kg_m_s / mg - found["eta"] * p.momentum_kg_m_s / m
        gas_re = (
            mg
            / gas.volume_m3
            * abs(drive)
            * pipe.diameter
            * math.sqrt(1 - found["alpha_l"])
            / pipe.friction.viscosity[pipe.friction.gas_slot]
        )
        close(gas_re, found["gas_driving_reynolds"], atol=1e-8, rtol=1e-9)
    close(math.fsum(impulses), report["wall_impulse_kg_m_s"], atol=1e-11, rtol=1e-9)
    if pipe.friction.wetting is not None:
        if len(report.get("wetting", [])) != len(wetting):
            raise ValueError("Wetting profile registry differs")
        for expected, actual in zip(wetting, report["wetting"]):
            if actual.keys() != expected.keys():
                raise ValueError("Wetting diagnostics differ")
            for key, value in expected.items():
                if isinstance(value, bool):
                    if actual[key] is not value:
                        raise ValueError("Wetting classification differs")
                else:
                    close(actual[key], value, atol=1e-11, rtol=1e-8)
    elif report.get("wetting"):
        raise ValueError("Unrequested wetting diagnostic")
    production = new_q["entropy"] - old_q["entropy"]
    if production < -1e-7:
        raise ValueError("Annular reference entropy decreased")
    close(production, report["entropy_production_j_k"], atol=1e-7, rtol=1e-9)
