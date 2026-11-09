"""Independent NumPy PR pressure quadrature and local momentum/work audit."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from check_inertial_capillary_film import close
from check_stratified_pr import Reference

from surface.operation import _digest
from surface.peng_robinson import OMEGA_A, OMEGA_B, R
from surface.pr_gas_loaded_capillary import PRGasLoadedCapillaryFilm

ROOT = Path(__file__).resolve().parents[1]


def audit(config, report):
    if report["request"] != config or report["format"] != "pr_gas_loaded_capillary_run_v1":
        raise ValueError("Confined gas request differs")
    for p, sha in report["source_sha256"].items():
        path = (ROOT / p).resolve()
        path.relative_to(ROOT)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Changed confined gas source")
    metadata = PRGasLoadedCapillaryFilm.from_dict(config["model"])
    if _digest(metadata.to_dict()) != _digest(report["normalized_model"]):
        raise ValueError("Confined gas metadata differs")
    f = metadata.film.fluid
    c = report["normalized_model"]["film"]
    s = c["mechanics"]["surface"]
    count = s["shape"][0]
    dx = s["lengths_m"][0] / count
    area = dx * s["transverse_width_m"]
    rho = s["density_kg_m3"]
    sigma = s["surface_tension_n_m"]
    normal = s["normal_gravity_m_s2"]
    ceilings = np.array(metadata.heights)
    face = lambda x: (x + np.roll(x, -1)) / 2
    center = lambda x: (x + np.roll(x, 1)) / 2
    grad = lambda x: (np.roll(x, -1) - x) / dx
    div = lambda x: (x - np.roll(x, 1)) / dx
    lx = np.array(c["mole_fractions"])
    lref = Reference(f, np.flatnonzero(lx > 0)).state(
        c["reference_pressure_pa"], c["temperature_k"], lx, "liquid"
    )

    def stocks(st):
        liquid = st["film"]
        mech = liquid["mechanics"]
        if (
            st["model_id"] != metadata.model_id
            or liquid["model_id"] != metadata.film.model_id
            or mech["model_id"] != metadata.film.mechanics.model_id
        ):
            raise ValueError("Confined gas checkpoint identity differs")
        h = np.asarray(mech["thickness_m"])
        u = np.asarray(mech["face_velocity_m_s"])
        if (
            h.shape != (count,)
            or u.shape != (count,)
            or np.any(h <= 0)
            or np.any(h >= ceilings)
            or np.max(abs(grad(h))) > s["maximum_slope"]
        ):
            raise ValueError("Invalid confined gas geometry")
        if not np.isfinite(np.r_[h, u, st["gas_thermostat_energy_j"]]).all():
            raise ValueError("Nonfinite confined gas state")
        if len(liquid["phases"]) != count or len(st["gas"]) != count:
            raise ValueError("Confined gas registry differs")
        for i, p in enumerate(liquid["phases"]):
            volume = area * h[i]
            amount = volume / lref["vm"]
            close(p["component_moles"], amount * lx, "liquid N", atol=1e-23)
            close(p["volume_m3"], volume, "liquid V", atol=1e-24)
            close(p["momentum_kg_m_s"], rho * volume * center(u)[i], "liquid M", atol=1e-23)
            target = amount * lref["u"] + 0.5 * rho * volume * center(u)[i] ** 2
            close(
                p["total_energy_j"],
                target,
                "liquid E",
                atol=256 * np.finfo(float).eps * abs(target) + 1e-24,
                rtol=0.0,
            )
        for i, p in enumerate(st["gas"]):
            n = np.array(p["component_moles"])
            x = n / n.sum()
            q = Reference(f, np.flatnonzero(x > 0)).state(
                p["pressure_pa"], p["temperature_k"], x, "vapor"
            )
            close(p["temperature_k"], metadata.temperature, "gas T", atol=0, rtol=0)
            close(p["volume_m3"], area * (ceilings[i] - h[i]), "gas geometry", atol=1e-24)
            close(p["volume_m3"], n.sum() * q["vm"], "gas EOS", atol=1e-24)
            close(
                p["total_energy_j"],
                n.sum() * q["u"],
                "gas calorics",
                atol=128 * np.finfo(float).eps * abs(p["total_energy_j"]) + 1e-24,
                rtol=0,
            )
            close(p["momentum_kg_m_s"], 0.0, "chamber axial momentum", atol=0, rtol=0)
        return h, u

    def energy(st, h, u):
        mechanical = area * np.sum(
            0.5 * rho * face(h) * u * u + 0.5 * sigma * grad(h) ** 2 + 0.5 * rho * normal * h * h
        )
        internal = sum(area * h / lref["vm"] * lref["u"])
        bath = (
            st["film"]["mechanics"]["physical_heat_j"]
            + st["film"]["mechanics"]["numerical_heat_j"]
            + st["gas_thermostat_energy_j"]
        )
        return mechanical + internal + bath + math.fsum(p["total_energy_j"] for p in st["gas"])

    old = report["initial"]
    h0, u0 = stocks(old)
    clock = old["film"]["mechanics"]["time_s"]
    if "initial_state" in config:
        if old != config["initial_state"]:
            raise ValueError("Confined restart differs")
    else:
        close(h0, config["initial_thickness_m"], "initial h", atol=0, rtol=0)
        close(
            u0, config.get("initial_face_velocity_m_s", [0.0] * count), "initial u", atol=0, rtol=0
        )
        for i, p in enumerate(old["gas"]):
            close(
                p["pressure_pa"],
                config.get("gas_pressures_pa", [c["reference_pressure_pa"]] * count)[i],
                "initial gas p",
                atol=0,
                rtol=0,
            )
            nn = np.array(p["component_moles"])
            close(nn / nn.sum(), config["gas_mole_fractions"], "initial gas x", atol=1e-15)
    target = clock + config["duration_s"]
    checked = 0
    nodes, weights = np.polynomial.legendre.leggauss(24)
    for entry in report["result"]["steps"]:
        new = entry["state"]
        h, u = stocks(new)
        r = entry["mechanics"]
        dt = r["dt_s"]
        if not math.isfinite(dt) or not 0 < dt <= config["initial_step_s"] * (1 + 1e-12):
            raise ValueError("Invalid confined gas dt")
        pressure = []
        for i, (a, b) in enumerate(zip(old["gas"], new["gas"], strict=True)):
            close(
                a["component_moles"], b["component_moles"], "closed gas components", atol=0, rtol=0
            )
            n = np.array(a["component_moles"])
            amount = n.sum()
            x = n / amount
            tc = np.array([v.critical_temperature_k for v in f.components])
            pc = np.array([v.critical_pressure_pa for v in f.components])
            w = np.array([v.acentric_factor for v in f.components])
            kap = 0.37464 + 1.54226 * w - 0.26992 * w * w
            ai = (
                OMEGA_A
                * R
                * R
                * tc
                * tc
                / pc
                * (1 + kap * (1 - np.sqrt(metadata.temperature / tc))) ** 2
            )
            bm = np.sum(x * OMEGA_B * R * tc / pc)
            am = np.sum(
                x[:, None]
                * x[None, :]
                * np.sqrt(ai[:, None] * ai[None, :])
                * (1 - np.array(f.binary_interactions))
            )
            vv = (a["volume_m3"] + (b["volume_m3"] - a["volume_m3"]) * (nodes + 1) / 2) / amount
            pp = R * metadata.temperature / (vv - bm) - am / (vv * vv + 2 * bm * vv - bm * bm)
            pressure.append(float(np.dot(weights, pp) / 2))
        pressure = np.array(pressure)
        dh = h - h0
        hm = (h + h0) / 2
        hf = face(hm)
        ut = (
            u
            if r["time_scheme"] == "backward"
            else (np.sqrt(face(h)) * u + np.sqrt(face(h0)) * u0)
            / (np.sqrt(face(h)) + np.sqrt(face(h0)))
        )
        flux = hf * ut
        chemical = -sigma * div(grad(hm)) + rho * normal * hm + pressure
        resistance = (
            s["viscosity_pa_s"] / (hf / 3 + s["slip_length_m"])
            if c["mechanics"]["viscous_drag"]
            else hf * 0
        )
        close(dh, -dt * div(flux), "confined local volume", atol=1e-20)
        expected = -dt * grad(center(flux) * center(ut)) - dt / rho * (
            hf * grad(chemical) + resistance * ut
        )
        close(
            face(h) * u - face(h0) * u0, expected, "confined local momentum", atol=1e-18, rtol=1e-7
        )
        close(
            r["cell_normal_pressure_pa"],
            pressure,
            "integrated PR normal pressure",
            atol=1e-6,
            rtol=1e-11,
        )
        work = area * dh * pressure
        close(r["cell_normal_load_potential_change_j"], work, "gas work", atol=1e-20, rtol=1e-8)
        du = u - u0
        physical = dt * area * np.sum(resistance * ut * ut)
        numerical = (
            0.5 * rho * area * np.sum(face(h0) * du * du) if r["time_scheme"] == "backward" else 0.0
        )
        close(
            new["film"]["mechanics"]["physical_heat_j"]
            - old["film"]["mechanics"]["physical_heat_j"],
            physical,
            "liquid physical heat",
            atol=1e-22,
        )
        close(
            new["film"]["mechanics"]["numerical_heat_j"]
            - old["film"]["mechanics"]["numerical_heat_j"],
            numerical,
            "liquid numerical heat",
            atol=1e-22,
        )
        du_gas = math.fsum(
            b["total_energy_j"] - a["total_energy_j"]
            for a, b in zip(old["gas"], new["gas"], strict=True)
        )
        close(
            new["gas_thermostat_energy_j"] - old["gas_thermostat_energy_j"],
            sum(work) - du_gas,
            "gas thermostat",
            atol=1e-19,
            rtol=1e-8,
        )
        allowance = (
            256
            * sum(
                math.ulp(p["total_energy_j"])
                for state in (old, new)
                for p in (*state["gas"], *state["film"]["phases"])
            )
            + 1e-20
        )
        close(
            energy(new, h, u) - energy(old, h0, u0),
            0.0,
            "complete confined energy",
            atol=allowance,
            rtol=0,
        )
        clock += dt
        close(new["film"]["mechanics"]["time_s"], clock, "clock", atol=1e-18)
        old, h0, u0 = new, h, u
        checked += 1
    if not checked or old != report["final"]:
        raise ValueError("Confined final state differs")
    close(clock, target, "complete interval", atol=1e-18)
    return {
        "accepted": True,
        "checked_steps": checked,
        "qualification": "Independent PR quadrature, calorics and local volume/momentum; closed isothermal chambers only",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    print(
        json.dumps(audit(json.loads(args.config.read_text()), json.loads(args.report.read_text())))
    )


if __name__ == "__main__":
    main()
