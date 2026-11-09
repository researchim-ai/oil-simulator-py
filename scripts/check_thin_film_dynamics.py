"""Independent NumPy finite-volume and free-energy audit of wall-film runs.

Does not import the runtime film model. Checks every accepted substep,
periodic face flux, material stocks, discrete work and dissipation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def audit(config, report):
    if config != report["request"]:
        raise ValueError("Film request differs")
    root = Path(__file__).resolve().parents[1]
    for path, digest in report["source_sha256"].items():
        p = (root / path).resolve()
        p.relative_to(root)
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError("Changed film source")
    c = config["model"]
    shape = tuple(c["shape"])
    dx = np.asarray(c["lengths_m"]) / shape
    area = float(np.prod(dx)) * (c.get("transverse_width_m", 1.0) if len(shape) == 1 else 1.0)
    sigma, mu, rho = c["surface_tension_n_m"], c["viscosity_pa_s"], c["density_kg_m3"]
    he, slip = c["adsorption_thickness_m"], c["slip_length_m"]
    scale = 10 / 3 * sigma * (1 - math.cos(math.radians(c["equilibrium_contact_angle_deg"])))
    normal = c.get("normal_gravity_m_s2", 0.0)
    gravity = config.get("tangential_gravity_m_s2", [0.0] * len(shape))
    shear = config.get("interface_shear_pa", [0.0] * len(shape))

    def grad(h, k):
        return (np.roll(h, -1, k) - h) / dx[k]

    def energy(h):
        r = he / h
        return float(
            area
            * np.sum(
                scale * (-(r**2) / 2 + r**5 / 5)
                + rho * normal * h**2 / 2
                + sigma / 2 * sum(grad(h, k) ** 2 for k in range(len(shape)))
            )
        )

    def close(a, b, label, *, atol=1e-21, rtol=1e-8):
        a, b = np.asarray(a), np.asarray(b)
        if (
            a.shape != b.shape
            or not np.all(np.isfinite(a))
            or not np.all(np.isfinite(b))
            or np.any(np.abs(a - b) > atol + rtol * np.maximum(np.abs(a), np.abs(b)))
        ):
            raise ValueError("Film audit differs: " + label)

    old = np.asarray(config["initial_thickness_m"], dtype=float).reshape(shape)
    if not np.all(np.isfinite(old)) or np.any(old <= 0):
        raise ValueError("Invalid initial film")
    close(report["initial"]["thickness_m"], old.ravel(), "initial", atol=0, rtol=0)
    start_time = report["initial"]["time_s"]
    clock = start_time
    heat = report["initial"]["dissipated_energy_j"]
    work = report["initial"]["external_work_j"]
    checked = 0
    for step in report["result"]["steps"]:
        new = np.asarray(step["state"]["thickness_m"]).reshape(shape)
        if not np.all(np.isfinite(new)) or np.any(new <= 0):
            raise ValueError("Nonpositive film")
        dt = step["dt_s"]
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid film dt")
        if (
            tuple(step["state"]["shape"]) != shape
            or step["state"]["model_id"] != report["initial"]["model_id"]
        ):
            raise ValueError("Film checkpoint registry differs")
        if any(
            np.any(np.abs(grad(new, k)) > c.get("maximum_slope", 0.2)) for k in range(len(shape))
        ):
            raise ValueError("Film outside long-wave slope domain")
        courant = sum(
            dt
            / dx[k]
            * np.max(
                np.abs(
                    ((old**2 + 2 * slip * old) * rho * gravity[k] + (old + slip) * shear[k]) / mu
                )
            )
            for k in range(len(shape))
        )
        if courant > 0.25:
            raise ValueError("Explicit film driving outside Courant domain")
        mid = (new + old) / 2
        lap = sum(
            (np.roll(mid, -1, k) - 2 * mid + np.roll(mid, 1, k)) / dx[k] ** 2
            for k in range(len(shape))
        )

        # Independent polynomial quotient, expressed in physical heights.
        def quot(p, a=new, b=old):
            return -sum(a**j * b ** (p - 1 - j) for j in range(p)) / (a**p * b**p)

        chemical = (
            -sigma * lap
            + scale * (-(he**2) / 2 * quot(2) + he**5 / 5 * quot(5))
            + rho * normal * mid
        )
        mobility = old**3 / (3 * mu) + slip * old**2 / mu
        shear_mobility = old**2 / (2 * mu) + slip * old / mu
        diss = 0.0
        external = 0.0
        div = 0.0
        for k in range(len(shape)):
            mf = (mobility + np.roll(mobility, -1, k)) / 2
            driving = (
                mf * rho * gravity[k]
                + (shear_mobility + np.roll(shear_mobility, -1, k)) / 2 * shear[k]
            )
            g = grad(chemical, k)
            flux = -mf * g + driving
            close(step["face_volume_flux_m2_s"][k], flux, "face flux", atol=1e-19)
            div = div + (flux - np.roll(flux, 1, k)) / dx[k]
            force = rho * gravity[k] - g
            # Independently integrate u'(z)^2 and Navier wall slip in each
            # of the two physical strips defining this averaged face.
            diss_face = 0.0
            ui_face = 0.0
            points, weights = np.polynomial.legendre.leggauss(4)
            for height in (old, np.roll(old, -1, k)):
                z = height[..., None] * (points + 1) / 2
                derivative = (force[..., None] * (height[..., None] - z) + shear[k]) / mu
                wall_speed = slip * (force * height + shear[k]) / mu
                viscous = mu * height / 2 * np.sum(weights * derivative**2, axis=-1)
                wall_loss = slip * (force * height + shear[k]) ** 2 / mu
                diss_face = diss_face + (viscous + wall_loss) / 2
                ui_face = (
                    ui_face
                    + (force * height**2 / (2 * mu) + shear[k] * height / mu + wall_speed) / 2
                )
            close(step["interface_velocity_m_s"][k], ui_face, "interface velocity", atol=1e-15)
            close(step["face_dissipation_j"][k], dt * area * diss_face, "face dissipation")
            diss += dt * area * np.sum(diss_face)
            external += dt * area * np.sum(rho * gravity[k] * flux + shear[k] * ui_face)
        close(new - old, -dt * div, "local volume", atol=1e-17, rtol=1e-7)
        close(new.sum() * area, old.sum() * area, "global volume", atol=1e-24, rtol=1e-11)
        change = energy(new) - energy(old)
        close(change + diss, external, "free energy", atol=1e-19)
        close(step["dissipation_j"], diss, "dissipation")
        close(step["external_work_j"], external, "work")
        close(step["free_energy_change_j"], change, "energy change", atol=1e-19)
        if diss < 0:
            raise ValueError("Negative dissipation")
        clock += dt
        heat += diss
        work += external
        close(step["state"]["time_s"], clock, "clock", atol=1e-14)
        close(step["state"]["dissipated_energy_j"], heat, "heat ledger")
        close(step["state"]["external_work_j"], work, "work ledger")
        old = new
        checked += 1
    close(clock, start_time + config["duration_s"], "report interval", atol=1e-14)
    if report["final"] != report["result"]["steps"][-1]["state"]:
        raise ValueError("Final film checkpoint differs")
    return {
        "accepted": True,
        "checked_steps": checked,
        "liquid_volume_m3": float(old.sum() * area),
        "dissipated_energy_j": float(heat),
        "external_work_j": float(work),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(json.loads(a.config.read_text()), json.loads(a.report.read_text()))))


if __name__ == "__main__":
    main()
