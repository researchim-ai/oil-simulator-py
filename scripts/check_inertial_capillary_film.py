"""Independent NumPy momentum, volume, energy and scalar PR trajectory audit."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from check_stratified_pr import Reference

from surface.inertial_capillary_film import InertialCapillaryFilm
from surface.pr_isothermal_capillary_film import PRIsothermalCapillaryFilm

ROOT = Path(__file__).resolve().parents[1]


def close(a, b, label, *, atol=1e-20, rtol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    if (
        a.shape != b.shape
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
        or np.any(np.abs(a - b) > atol + rtol * np.maximum(np.abs(a), np.abs(b)))
    ):
        raise ValueError("Joint capillary audit differs: " + label)


def audit(config, report):
    if config != report["request"]:
        raise ValueError("Joint capillary request differs")
    for p, sha in report["source_sha256"].items():
        path = (ROOT / p).resolve()
        path.relative_to(ROOT)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Changed joint capillary source")
    is_pr = report["family"] == "pr_isothermal"
    driving = config.get("driving", {})
    gravity = driving.get("tangential_gravity_m_s2", 0.0)
    shear = driving.get("interface_shear_pa", 0.0)
    # Runtime constructors are used only to normalize metadata and registry.
    # Numeric trajectory checks below do not call step/inspect/force/energy.
    metadata = (PRIsothermalCapillaryFilm if is_pr else InertialCapillaryFilm).from_dict(
        config["model"]
    )
    if metadata.to_dict() != report["normalized_model"]:
        from surface.operation import _digest

        if _digest(metadata.to_dict()) != _digest(report["normalized_model"]):
            raise ValueError("Normalized mechanical metadata differs")
    mc = report["normalized_model"]["mechanics"] if is_pr else report["normalized_model"]
    c = mc["surface"]
    n = c["shape"][0]
    dx = c["lengths_m"][0] / n
    area = dx * c["transverse_width_m"]
    rho, sigma, mu, slip = (
        c[k] for k in ("density_kg_m3", "surface_tension_n_m", "viscosity_pa_s", "slip_length_m")
    )
    normal = c["normal_gravity_m_s2"]

    def face(h):
        return (h + np.roll(h, -1)) / 2

    def center(u):
        return (u + np.roll(u, 1)) / 2

    def grad(x):
        return (np.roll(x, -1) - x) / dx

    def div(x):
        return (x - np.roll(x, 1)) / dx

    def stored(st):
        h = np.asarray(st["thickness_m"])
        u = np.asarray(st["face_velocity_m_s"])
        if (
            h.shape != (n,)
            or u.shape != (n,)
            or not np.all(np.isfinite(h))
            or not np.all(np.isfinite(u))
            or np.any(h <= 0)
            or np.max(np.abs(grad(h))) > c["maximum_slope"]
            or np.max(np.abs(u)) > mc["maximum_velocity_m_s"]
        ):
            raise ValueError("Invalid mechanical field/domain")
        if st["physical_heat_j"] < 0 or st["numerical_heat_j"] < 0:
            raise ValueError("Negative heat ledger")
        if not math.isfinite(st.get("external_work_j", 0.0)):
            raise ValueError("Nonfinite signed external work")
        return h, u

    def energy(h, u):
        return area * np.sum(
            0.5 * rho * face(h) * u * u + sigma / 2 * grad(h) ** 2 + rho * normal * h * h / 2
        )

    def momentum(h, u):
        return rho * area * np.sum(face(h) * u)

    old = report["initial"]
    om = old["mechanics"] if is_pr else old
    if old["model_id"] != metadata.model_id:
        raise ValueError("Initial mechanical identity differs")
    if "initial_state" in config:
        if json.loads(json.dumps(old)) != json.loads(json.dumps(config["initial_state"])):
            raise ValueError("Initial mechanical restart differs")
    else:
        close(om["thickness_m"], config["initial_thickness_m"], "initial height", atol=0, rtol=0)
        close(
            om["face_velocity_m_s"],
            config.get("initial_face_velocity_m_s", [0.0] * n),
            "initial velocity",
            atol=0,
            rtol=0,
        )
    if is_pr:
        pc = report["normalized_model"]
        fluid = metadata.fluid
        x = np.array(pc["mole_fractions"])
        pref, T = pc["reference_pressure_pa"], pc["temperature_k"]
        ref = Reference(fluid, np.flatnonzero(x > 0))
        qref = ref.state(pref, T, x, "liquid")
        close(rho, qref["rho"], "reference PR density", atol=1e-8, rtol=1e-9)

        def material(st):
            h, u = stored(st["mechanics"])
            stocks = st["phases"]
            if len(stocks) != n:
                raise ValueError("Material mesh differs")
            uc = center(u)
            volume = h * area
            for i, p in enumerate(stocks):
                nn = np.array(p["component_moles"])
                amount = volume[i] / qref["vm"]
                close(p["pressure_pa"], pref, "PR reference pressure", atol=0, rtol=0)
                close(p["temperature_k"], T, "PR thermostat temperature", atol=0, rtol=0)
                close(nn, amount * x, "PR components", atol=1e-22, rtol=1e-9)
                close(p["volume_m3"], volume[i], "PR volume", atol=1e-24, rtol=1e-11)
                close(
                    p["momentum_kg_m_s"],
                    rho * volume[i] * uc[i],
                    "PR momentum",
                    atol=1e-22,
                    rtol=1e-9,
                )
                close(
                    p["total_energy_j"],
                    amount * qref["u"] + 0.5 * rho * volume[i] * uc[i] ** 2,
                    "scalar PR calorics",
                    atol=256
                    * np.finfo(float).eps
                    * (abs(amount * qref["u"]) + 0.5 * rho * volume[i] * uc[i] ** 2)
                    + 1e-24,
                    rtol=0.0,
                )
            return np.array([p["component_moles"] for p in stocks])

        oldn = material(old)

        def material_energy(st):
            hh, uu = stored(st["mechanics"])
            gg = grad(hh)
            variance = 0.125 * rho * area * hh * (uu - np.roll(uu, 1)) ** 2
            potential = area * (
                sigma / 4 * (gg**2 + np.roll(gg, 1) ** 2) + rho * normal * hh**2 / 2
            )
            return np.array([p["total_energy_j"] for p in st["phases"]]) + variance + potential

    h0, u0 = stored(om)
    clock = om["time_s"]
    target = clock + config["duration_s"]
    checked = 0
    maximumE = 0.0
    maximumM = 0.0
    for step in report["result"]["steps"]:
        new = step["state"]
        nm = new["mechanics"] if is_pr else new
        r = step["mechanics"] if is_pr else step
        dt = r["dt_s"]
        h, u = stored(nm)
        if not 0 < dt <= config["initial_step_s"] * (1 + 1e-12) or not math.isfinite(dt):
            raise ValueError("Invalid joint step")
        if nm["model_id"] != om["model_id"] or new["model_id"] != old["model_id"]:
            raise ValueError("Mechanical checkpoint identity changed")
        hm = (h + h0) / 2
        hf = face(hm)
        hf0 = face(h0)
        stiff = mc["viscous_drag"] and (
            np.max(dt * mu / (rho * hf0 * (hf0 / 3 + slip))) > 2
            or np.max(dt * mu / (rho * hf * (hf / 3 + slip))) > 2
        )
        scheme = (
            ("backward" if stiff else "roe") if mc["time_scheme"] == "auto" else mc["time_scheme"]
        )
        if r["time_scheme"] != scheme:
            raise ValueError("Mechanical time scheme differs")
        ut = (
            u
            if scheme == "backward"
            else (np.sqrt(face(h)) * u + np.sqrt(hf0) * u0) / (np.sqrt(face(h)) + np.sqrt(hf0))
        )
        mass_flux = hf * ut
        adv_flux = center(mass_flux) * center(ut)
        chemical = -sigma * (np.roll(hm, -1) - 2 * hm + np.roll(hm, 1)) / dx**2 + rho * normal * hm
        force = -area * hf * grad(chemical)
        resistance = mu / (hf / 3 + slip) if mc["viscous_drag"] else hf * 0
        if mc["viscous_drag"]:
            # Independent integrated planar profile, without runtime helper.
            mobility = hf**3 / (3 * mu) + slip * hf**2 / mu
            coupling = hf**2 / (2 * mu) + slip * hf / mu
            eta = hf * coupling / mobility
            beta = (hf**2 / 12 + slip * hf / 3) / (mu * (hf / 3 + slip))
        else:
            eta, beta = hf * 0 + 1, hf * 0
        body = rho * hf * gravity + eta * shear
        eps = np.finfo(float).eps
        hround = (
            128
            * eps
            * (np.max(np.abs(h0)) + np.max(np.abs(h)) + dt / dx * np.max(np.abs(mass_flux)))
        )
        mround = (
            128
            * eps
            * (
                np.max(np.abs(face(h) * u))
                + np.max(np.abs(hf0 * u0))
                + dt / dx * np.max(np.abs(adv_flux))
                + dt
                / rho
                * np.max(hf)
                * (sigma * np.max(hm) / dx**3 + rho * abs(normal) * np.max(hm) / dx)
            )
        )
        close(h - h0, -dt * div(mass_flux), "local volume", atol=hround)
        close(
            face(h) * u - hf0 * u0,
            -dt * grad(adv_flux) + dt * force / (rho * area) - dt / rho * (resistance * ut - body),
            "local impulse",
            atol=mround,
        )
        close(r["face_volume_flux_m2_s"], mass_flux, "mass flux", atol=1e-21)
        close(r["momentum_flux_m3_s2"], adv_flux, "momentum flux", atol=1e-21)
        close(r["face_time_velocity_m_s"], ut, "time velocity", atol=1e-14)
        force_round = (
            128
            * eps
            * area
            * np.max(hf)
            * (sigma * np.max(hm) / dx**3 + rho * abs(normal) * np.max(hm) / dx)
        )
        close(r["capillary_force_n"], force, "capillary force", atol=force_round)
        physical = dt * area * np.sum(resistance * ut * ut + beta * shear**2)
        numerical = 0.5 * rho * area * np.sum(hf0 * (u - u0) ** 2) if scheme == "backward" else 0.0
        wall = dt * area * (-resistance * ut + (eta - 1) * shear)
        gravity_impulse = dt * area * rho * hf * gravity
        interface_impulse = np.full(n, dt * area * shear)
        impulse = np.sum(wall)
        interface_velocity = eta * ut + beta * shear
        gravity_work = dt * area * rho * hf * gravity * ut
        interface_work = dt * area * shear * interface_velocity
        work = gravity_work + interface_work
        close(
            r.get("interface_velocity_m_s", interface_velocity),
            interface_velocity,
            "interface velocity",
            atol=1e-14,
        )
        close(r.get("external_work_j", 0), work.sum(), "external work", atol=1e-22)
        close(r.get("interface_work_j", 0), interface_work.sum(), "interface work", atol=1e-22)
        close(r.get("gravity_work_j", 0), gravity_work.sum(), "gravity work", atol=1e-22)
        close(
            r.get("gravity_impulse_kg_m_s", 0), gravity_impulse.sum(), "gravity impulse", atol=1e-22
        )
        close(
            r.get("interface_impulse_kg_m_s", 0),
            interface_impulse.sum(),
            "interface impulse",
            atol=1e-22,
        )
        close(
            nm.get("external_work_j", 0) - om.get("external_work_j", 0),
            work.sum(),
            "external work ledger",
            atol=1e-22,
        )
        # Local energy and impulse fluxes, independently reconstructed from
        # full stored states rather than runtime Newton increments.
        dh = h - h0
        kinetic_center = 0.5 * rho * np.roll(ut, 1) * ut * center(mass_flux)
        shift = 0.125 * rho * dx**2 * (grad(h) * u**2 - grad(h0) * u0**2)
        j = (
            face(kinetic_center)
            + face(chemical) * mass_flux
            - sigma * grad(hm) * face(dh) / dt
            + shift / dt
        )
        jround = (
            256
            * eps
            * c["transverse_width_m"]
            * (
                sigma * np.max(hm) ** 2 / (dx * dt)
                + rho * dx * np.max(hm) * (np.max(np.abs(u)) ** 2 + np.max(np.abs(u0)) ** 2) / dt
            )
            + 1e-23
        )
        close(
            r["face_mechanical_energy_flux_w"],
            j * c["transverse_width_m"],
            "mechanical energy flux",
            atol=jround,
            rtol=1e-9,
        )
        cell_physical = center(dt * area * (resistance * ut**2 + beta * shear**2))
        cell_numerical = (
            center(0.5 * rho * area * hf0 * (u - u0) ** 2)
            if scheme == "backward"
            else cell_physical * 0
        )
        close(r["cell_physical_heat_j"], cell_physical, "cell physical heat", atol=1e-22)
        close(r["cell_numerical_heat_j"], cell_numerical, "cell numerical heat", atol=1e-22)
        close(
            r.get("cell_external_work_j", np.zeros(n)),
            center(work),
            "cell external work",
            atol=1e-22,
        )
        close(
            r.get("cell_external_impulse_kg_m_s", np.zeros(n)),
            center(gravity_impulse + interface_impulse),
            "cell external impulse",
            atol=1e-22,
        )

        def density(hh, uu):
            gg = grad(hh)
            return (
                0.25 * rho * hh * (uu**2 + np.roll(uu, 1) ** 2)
                + sigma / 4 * (gg**2 + np.roll(gg, 1) ** 2)
                + rho * normal * hh**2 / 2
            )

        local_energy = (
            area * (density(h, u) - density(h0, u0) + dt * div(j))
            + cell_physical
            + cell_numerical
            - center(work)
        )
        local_budget = (
            area * (np.abs(density(h, u)) + np.abs(density(h0, u0)))
            + dt * c["transverse_width_m"] * (np.abs(j) + np.abs(np.roll(j, 1)))
            + cell_physical
            + cell_numerical
            + np.abs(center(work))
        )
        if np.any(np.abs(local_energy) > 1e-23 + 2e-9 * local_budget):
            raise ValueError("Local mechanical energy flux failed")
        pm_shift = 0.25 * rho * dx**2 * (grad(h) * u - grad(h0) * u0)
        jm = rho * face(adv_flux) + pm_shift / dt
        jmround = (
            256
            * eps
            * c["transverse_width_m"]
            * rho
            * dx
            * np.max(hm)
            * (np.max(np.abs(u)) + np.max(np.abs(u0)))
            / dt
            + 1e-23
        )
        close(
            r["face_material_momentum_flux_n"],
            jm * c["transverse_width_m"],
            "material momentum flux",
            atol=jmround,
        )
        close(
            r["cell_potential_force_n"], center(force), "material potential force", atol=force_round
        )
        close(
            r["cell_wall_impulse_kg_m_s"],
            center(wall),
            "cell wall impulse",
            atol=1e-22,
        )
        local_momentum = (
            rho * area * (h * center(u) - h0 * center(u0))
            + dt * area * div(jm)
            - dt * center(force)
            - center(wall + gravity_impulse + interface_impulse)
        )
        mbudget = (
            rho * area * (np.abs(h * center(u)) + np.abs(h0 * center(u0)))
            + dt * c["transverse_width_m"] * (np.abs(jm) + np.abs(np.roll(jm, 1)))
            + dt * np.abs(center(force))
            + np.abs(center(wall))
            + np.abs(center(gravity_impulse + interface_impulse))
        )
        if np.any(np.abs(local_momentum) > 1e-23 + 2e-9 * mbudget):
            raise ValueError("Local material momentum balance failed")
        close(r["physical_heat_j"], physical, "physical heat", atol=1e-22)
        close(r["numerical_heat_j"], numerical, "numerical heat", atol=1e-22)
        close(r["wall_impulse_kg_m_s"], impulse, "wall impulse", atol=1e-22)
        defect = energy(h, u) - energy(h0, u0) + physical + numerical - work.sum()
        momdef = (
            momentum(h, u)
            - momentum(h0, u0)
            - impulse
            - gravity_impulse.sum()
            - interface_impulse.sum()
        )
        ebudget = (
            abs(energy(h, u)) + abs(energy(h0, u0)) + physical + numerical + np.abs(work).sum()
        )
        close(defect, 0.0, "mechanical energy", atol=1e-22 + 2e-9 * ebudget, rtol=0)
        close(
            r["energy_residual_j"],
            defect,
            "reported energy residual",
            atol=1e-22 + 2e-9 * ebudget,
            rtol=0,
        )
        close(
            momdef,
            0.0,
            "total momentum",
            atol=1e-22
            + 1e-10
            * (
                abs(momentum(h, u))
                + abs(momentum(h0, u0))
                + abs(impulse)
                + abs(gravity_impulse.sum())
                + abs(interface_impulse.sum())
            ),
            rtol=0,
        )
        close(
            nm["physical_heat_j"] - om["physical_heat_j"],
            physical,
            "physical heat ledger",
            atol=1e-22,
        )
        close(
            nm["numerical_heat_j"] - om["numerical_heat_j"],
            numerical,
            "numerical heat ledger",
            atol=1e-22,
        )
        close(nm["time_s"], clock + dt, "clock", atol=1e-15)
        if is_pr:
            if r["state"] != nm:
                raise ValueError("Duplicated mechanical checkpoint differs")
            if (
                step["pressure_work_convention"]
                != "uniform_exterior_pressure_enthalpy_and_surface_work_cancelled"
            ):
                raise ValueError("Unsupported mechanical pressure-work convention")
            newn = material(new)
            flux = mass_flux[:, None] * c["transverse_width_m"] * x / qref["vm"]
            close(step["component_flux_mol_s"], flux, "component flux", atol=1e-20)
            close(
                newn - oldn,
                -dt * (flux - np.roll(flux, 1, axis=0)),
                "local PR components",
                atol=1e-20,
                rtol=1e-8,
            )
            total_flux = (
                j * c["transverse_width_m"]
                + qref["u"] / qref["vm"] * mass_flux * c["transverse_width_m"]
            )
            close(
                step["entropy_flux_w_k"],
                qref["s"] / qref["vm"] * mass_flux * c["transverse_width_m"],
                "isothermal entropy flux",
                atol=1e-21,
            )
            close(
                step["face_total_energy_flux_w"],
                total_flux,
                "total PR/film energy flux",
                atol=1e-22 + 256 * eps * np.max(np.abs(total_flux)),
                rtol=1e-11,
            )
            e0, e1 = material_energy(old), material_energy(new)
            pr_local = (
                e1
                - e0
                + dt * (total_flux - np.roll(total_flux, 1))
                + cell_physical
                + cell_numerical
                - center(work)
            )
            pr_budget = (
                np.abs(e0)
                + np.abs(e1)
                + dt * (np.abs(total_flux) + np.abs(np.roll(total_flux, 1)))
                + cell_physical
                + cell_numerical
                + np.abs(center(work))
            )
            if np.any(np.abs(pr_local) > 1e-23 + 512 * eps * pr_budget):
                raise ValueError("Local PR/film energy failed")
            bath = (nm["physical_heat_j"] - om["physical_heat_j"]) + (
                nm["numerical_heat_j"] - om["numerical_heat_j"]
            )
            close(
                step["thermostat_physical_heat_j"], physical, "PR physical thermostat", atol=1e-22
            )
            close(
                step["thermostat_numerical_heat_j"],
                numerical,
                "PR numerical thermostat",
                atol=1e-22,
            )
            roundoff = (
                64 * sum(math.ulp(p["total_energy_j"]) for st in (old, new) for p in st["phases"])
                + 1e-21
            )
            close(
                float(np.sum(e1) - np.sum(e0) + bath - work.sum()),
                0.0,
                "joint PR energy",
                atol=roundoff,
                rtol=0.0,
            )
            oldn = newn
        maximumE = max(maximumE, abs(defect))
        maximumM = max(maximumM, abs(momdef))
        clock += dt
        old = new
        om = nm
        h0, u0 = h, u
        checked += 1
    if old != report["final"] or not checked:
        raise ValueError("Final joint state differs")
    close(clock, target, "full report interval", atol=1e-14)
    return {
        "accepted": True,
        "checked_steps": checked,
        "maximum_energy_residual_j": maximumE,
        "maximum_momentum_residual_kg_m_s": maximumM,
        "qualification": "Independent periodic mechanical and optional scalar PR audit; no external experimental qualification",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(json.loads(a.config.read_text()), json.loads(a.report.read_text()))))


if __name__ == "__main__":
    main()
