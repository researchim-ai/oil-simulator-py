"""CPU reference audit of PR gravity columns; no Torch column solve/inspect.

Independent NumPy PR roots and SciPy local chemical-potential solves are used
only by this verifier. A finer independent quadrature checks conserved N/E.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.component_energy import ComponentEnergyStream
from surface.equilibrium_properties import EquilibriumFluid
from surface.operation import _digest
from surface.peng_robinson import R
from surface.peng_robinson_mixture import PengRobinsonMixture
from surface.separator_geometry import SeparatorGeometry


def softmax(y):
    x = np.exp(y - np.max(y))
    return x / x.sum()


class Reference:
    def __init__(self, fluid, active):
        self.fluid = fluid
        self.active = active
        components = [fluid.components[i] for i in active]
        self.mw = np.array([c.molar_mass_kg_mol for c in components])
        self.pure = components[0] if len(active) == 1 else None
        kij = np.array(fluid.binary_interactions)[np.ix_(active, active)]
        self.mixture = None if self.pure else PengRobinsonMixture(components, kij)

    def state(self, p, t, x, tag):
        xa = np.array(x)[self.active]
        if self.pure:
            q = self.pure.state(p, t, tag, derivatives=False)
            mw = self.pure.molar_mass_kg_mol
            v = q["specific_volume_m3_kg"] * mw
            h = q["enthalpy_j_kg"] * mw
            s = q["entropy_j_kg_k"] * mw
            phi = np.array([q["log_fugacity_coefficient"]])
            z = q["compressibility_factor"]
        else:
            q = self.mixture.state(p, t, xa, tag)
            v = q["molar_volume_m3_mol"]
            h = q["molar_enthalpy_j_mol"]
            s = q["molar_entropy_j_mol_k"]
            phi = np.array(q["log_fugacity_coefficients"])
            z = q["compressibility_factor"]
        return {
            "vm": v,
            "h": h,
            "u": h - p * v,
            "s": s,
            "rho": float(xa @ self.mw) / v,
            "phi": phi,
            "z": z,
        }

    def local(self, y, t, tag):
        x = np.zeros(len(self.fluid.names))
        x[self.active] = softmax(np.r_[y[1:], 0.0])
        p = float(np.exp(y[0]))
        return p, x, self.state(p, t, x, tag)


def grid(geometry, interfaces, order):
    bounds = np.r_[0, interfaces, geometry.height_m]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    heights = []
    volumes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        volume = geometry.liquid_volume_m3(float(b)) - geometry.liquid_volume_m3(float(a))
        if geometry.orientation == "vertical":
            h = (a + b) / 2 + (b - a) / 2 * nodes
            w = weights * volume / 2
        else:
            low = np.arcsin(np.clip(2 * a / geometry.height_m - 1, -1, 1))
            high = np.arcsin(np.clip(2 * b / geometry.height_m - 1, -1, 1))
            angle = (low + high) / 2 + (high - low) / 2 * nodes
            h = geometry.height_m / 2 * (1 + np.sin(angle))
            w = weights * np.cos(angle) ** 2
            w = w / w.sum() * volume
        heights.append(h)
        volumes.append(w)
    return heights, volumes


def moment(geometry, h):
    if geometry.orientation == "vertical":
        return geometry.volume_m3 * h * h / (2 * geometry.height_m)
    return (
        geometry.diameter_m / 2 * geometry.liquid_volume_m3(h)
        - 2 / 3 * geometry.length_m * max(0, h * (geometry.height_m - h)) ** 1.5
    )


def check(config, path):
    data = json.loads(path.read_text())
    payload = dict(data)
    checksum = payload.pop("checksum")
    if _digest(payload) != checksum or data["format"] != "stratified_pr_column_v1":
        raise ValueError("Column checksum/format mismatch")
    system = data["system"]
    f = EquilibriumFluid.from_dict(system["fluid"])
    geo = SeparatorGeometry(**system["geometry"])
    # Defaults are made explicit in the saved model; verify supplied fields.
    for key, value in config["system"].items():
        if system[key] != value:
            raise ValueError("Column system differs from config")
    if data["initial"] != config["initial"]:
        raise ValueError("Column initial request differs")
    rootdir = Path(__file__).resolve().parents[1] / "src/surface"
    for name, digest in data["provenance"]["source_hashes"].items():
        if hashlib.sha256((rootdir / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Stale column physical source")
    if (
        hashlib.sha256((Path(__file__).parent / "run_stratified_pr.py").read_bytes()).hexdigest()
        != data["provenance"]["runner_sha256"]
    ):
        raise ValueError("Stale column runner")
    report = data["result"]
    state = report["state"]
    n = np.array(state["component_moles"])
    active = np.flatnonzero(n > 0)
    ref = Reference(f, active)
    c = len(active)
    m = len(state["phase_roots"])
    order = state["quadrature_order"]
    g = system["gravity_m_s2"]
    base = system["elevation_m"]
    t = state["temperature_k"]
    count = m * order + 2 * (m - 1)
    y = np.array(state["solution_variables"])
    local = y[: count * c].reshape(count, c)
    lam = y[count * c : count * c + c]
    if (
        abs(y[count * c + c] * f.tref - t) > 1e-9
        or np.max(abs(lam - np.array(state["chemical_constants"])[active])) > 1e-10
    ):
        raise ValueError("Column metadata/collocation differs")
    fractions = softmax(np.r_[y[count * c + c + 1 :], 0.0])
    interfaces = np.cumsum(fractions)[:-1] * geo.height_m
    if (
        len(interfaces)
        and np.max(abs(interfaces - state["interface_heights_m"])) > 1e-9 * geo.height_m
    ):
        raise ValueError("Column interface metadata differs")
    hh, vw = grid(geo, interfaces, order)
    total = np.zeros_like(n)
    internal = potential = entropy = 0.0
    max_mu = 0.0
    interface_pressures = []
    phase_amounts = []
    for i in range(m):
        layer = report["layers"][i]
        if np.max(abs(np.array(layer["height_m"]) - hh[i])) > 1e-9 * max(1, geo.height_m) or np.max(
            abs(np.array(layer["volume_weights_m3"]) - vw[i])
        ) > 1e-11 * max(1, geo.volume_m3):
            raise ValueError("Column quadrature geometry differs")
        amount = np.zeros_like(n)
        for j, (height, weight) in enumerate(zip(hh[i], vw[i])):
            p, x, q = ref.local(local[i * order + j], t, state["phase_roots"][i])
            mu = (
                np.log(x[active])
                + q["phi"]
                + np.log(p / f.pref)
                + ref.mw * g * height / (R * t)
                - lam
            )
            max_mu = max(max_mu, float(np.max(abs(mu))))
            if (
                abs(p - layer["pressure_pa"][j]) > 1e-7 * max(1, p)
                or np.max(abs(x - layer["mole_fractions"][j])) > 1e-10
            ):
                raise ValueError("Saved profile differs from collocation")
            if (
                abs(q["rho"] - layer["density_kg_m3"][j]) > 1e-7 * max(1, q["rho"])
                or abs(q["u"] - layer["molar_internal_energy_j_mol"][j]) > 1e-5
            ):
                raise ValueError("Column EOS fields differ from NumPy reference")
            dn = weight * x / q["vm"]
            amount += dn
            internal += weight * q["u"] / q["vm"]
            potential += weight * q["rho"] * g * (base + height)
            entropy += weight * q["s"] / q["vm"]
        if np.max(abs(amount - report["phase_component_moles"][i])) > 1e-8 + 1e-9 * n.sum():
            raise ValueError("Layer inventories differ")
        phase_amounts.append(amount)
        total += amount
    max_gap = 0.0
    for i, h in enumerate(interfaces):
        values = []
        for j in (i, i + 1):
            p, x, q = ref.local(local[m * order + 2 * i + (j - i)], t, state["phase_roots"][j])
            mu = np.log(x[active]) + q["phi"] + np.log(p / f.pref) + ref.mw * g * h / (R * t) - lam
            max_mu = max(max_mu, float(np.max(abs(mu))))
            values.append((p, q["rho"]))
        max_gap = max(max_gap, abs(values[0][0] - values[1][0]))
        interface_pressures.append(values[0][0])
        if values[0][1] <= values[1][1] or abs(np.log(values[0][0] / values[1][0])) > 1e-8:
            raise ValueError("Interface mechanical/density equilibrium failed")
    mass_error = float(np.max(abs(total - n)))
    energy_error = internal + potential - state["total_energy_j"]
    if (
        max_mu > 1e-7
        or mass_error > 1e-8 + 1e-9 * n.sum()
        or abs(energy_error) > 1e-5 + 1e-9 * max(1, abs(state["total_energy_j"]))
    ):
        raise ValueError("Independent column chemical/N/E audit failed")
    if "component_moles" in config["initial"]:
        if (
            not np.allclose(n, config["initial"]["component_moles"], rtol=0, atol=1e-12)
            or state["total_energy_j"] != config["initial"]["total_energy_j"]
        ):
            raise ValueError("Conserved input changed")
    else:
        uniform = report["initial_uniform_state"]
        p0 = config["initial"]["pressure_pa"]
        t0 = config["initial"]["temperature_k"]
        initial_n = np.zeros_like(n)
        initial_u = initial_pe = initial_v = initial_s = 0.0
        prior_moment = 0.0
        mu0 = []
        if uniform["pressure_pa"] != p0 or uniform["temperature_k"] != t0:
            raise ValueError("Uniform input PT differs")
        for amount, x, z in zip(
            uniform["phase_moles"],
            uniform["phase_mole_fractions"],
            uniform["phase_compressibility_factors"],
        ):
            phases = [ref.state(p0, t0, x, tag) for tag in ("liquid", "vapor")]
            q = min(phases, key=lambda v: abs(v["z"] - z))
            dv = amount * q["vm"]
            initial_v += dv
            h = geo.level_m(min(geo.volume_m3, initial_v))
            next_moment = moment(geo, h)
            initial_pe += q["rho"] * g * (base * dv + next_moment - prior_moment)
            prior_moment = next_moment
            initial_u += amount * q["u"]
            initial_s += amount * q["s"]
            initial_n += amount * np.array(x)
            mu0.append(np.log(np.array(x)[active]) + q["phi"])
        if (
            abs(initial_v - geo.volume_m3) > 1e-9 * geo.volume_m3
            or np.max(abs(initial_n - n)) > 1e-8 + 1e-9 * n.sum()
            or np.max(abs(initial_n / initial_n.sum() - config["initial"]["mole_fractions"])) > 1e-9
            or np.max(abs(np.array(mu0) - mu0[0])) > 1e-7
            or abs(initial_u + initial_pe - state["total_energy_j"])
            > 1e-5 + 1e-9 * max(1, abs(state["total_energy_j"]))
        ):
            raise ValueError("Uniform initial N/V/E/fugacity differs")
        if entropy < initial_s - 1e-6 - 1e-10 * max(1, abs(initial_s)):
            raise ValueError("Entropy decreased at conserved N/E/V")
    # Solve the same chemical-potential field on a separate, finer reference
    # grid. Balances at the original nodes alone cannot certify quadrature.
    fine_h, fine_v = grid(geo, interfaces, min(64, 2 * order))
    fine_n = np.zeros_like(n)
    fine_energy = 0.0
    for i in range(m):
        old = local[i * order : (i + 1) * order]
        for h, w in zip(fine_h[i], fine_v[i]):
            guess = np.array([np.interp(h, hh[i], old[:, j]) for j in range(c)])

            def chemical(v, phase=i, height=h):
                p, x, q = ref.local(v, t, state["phase_roots"][phase])
                return (
                    np.log(x[active])
                    + q["phi"]
                    + np.log(p / f.pref)
                    + ref.mw * g * height / (R * t)
                    - lam
                )

            solved = root(chemical, guess, tol=1e-9)
            if not solved.success or np.max(abs(chemical(solved.x))) > 1e-8:
                raise ValueError("Independent fine-grid local solve failed")
            p, x, q = ref.local(solved.x, t, state["phase_roots"][i])
            fine_n += w * x / q["vm"]
            fine_energy += w * (q["u"] / q["vm"] + q["rho"] * g * (base + h))
    fine_n_error = float(np.max(abs(fine_n - n)))
    fine_e_error = fine_energy - state["total_energy_j"]
    if fine_n_error > 1e-8 + 5 * system["relative_tolerance"] * n.sum() or abs(
        fine_e_error
    ) > 1e-4 + 5 * system["relative_tolerance"] * max(1, abs(internal), abs(potential)):
        raise ValueError("Independent refined quadrature did not conserve N/E")
    if len(data["samples"]) != len(config["samples"]):
        raise ValueError("Missing local port samples")
    for row, request in zip(data["samples"], config["samples"]):
        if row["request"] != request:
            raise ValueError("Sample request differs")
        point = row["point"]
        h = (
            request["height_m"]
            if "height_m" in request
            else float(interfaces[request["interface_index"]])
        )
        index = sum(h > v for v in interfaces)
        if "interface_index" in request:
            index = request["interface_index"] + (request.get("side") == "above")
        if (
            point["phase_index"] != index
            or abs(point["height_m"] - h) > 1e-9
            or point["temperature_k"] != t
        ):
            raise ValueError("Local phase/height/T differs")
        q = ref.state(point["pressure_pa"], t, point["mole_fractions"], state["phase_roots"][index])
        mu = (
            np.log(np.array(point["mole_fractions"])[active])
            + q["phi"]
            + np.log(point["pressure_pa"] / f.pref)
            + ref.mw * g * h / (R * t)
            - lam
        )
        if np.max(abs(mu)) > 1e-7:
            raise ValueError("Local sample violates generalized chemical equilibrium")
        for key, value in [
            ("density_kg_m3", q["rho"]),
            ("molar_enthalpy_j_mol", q["h"]),
            ("molar_entropy_j_mol_k", q["s"]),
            ("molar_volume_m3_mol", q["vm"]),
        ]:
            if abs(point[key] - value) > 1e-6 + 1e-8 * abs(value):
                raise ValueError("Local port property differs")
        if "molar_rate_mol_s" in request:
            stream = ComponentEnergyStream.from_dict(row["stream"])
            stream.validate_fluid(f)
            rate = request["molar_rate_mol_s"]
            components = rate * np.array(point["mole_fractions"])
            mass = components[active] @ ref.mw
            power = rate * q["h"] + mass * g * (base + h)
            if (
                np.max(abs(components - stream.component_molar_rates_mol_s)) > 1e-10
                or abs(power - stream.total_energy_rate_w) > 1e-6
                or stream.pressure_pa != point["pressure_pa"]
                or stream.gravity_m_s2 != g
                or abs(stream.elevation_m - base - h) > 1e-9
            ):
                raise ValueError("Port component/energy reference differs")
    weight_error = None
    if geo.orientation == "vertical":
        bottom = [v["point"] for v in data["samples"] if abs(v["point"]["height_m"]) < 1e-10]
        top = [
            v["point"]
            for v in data["samples"]
            if abs(v["point"]["height_m"] - geo.height_m) < 1e-10
        ]
        if bottom and top:
            head = float(n[active] @ ref.mw) * g / (geo.volume_m3 / geo.height_m)
            weight_error = bottom[0]["pressure_pa"] - top[0]["pressure_pa"] - head
            if abs(weight_error) > 1e-3 + 1e-7 * abs(head):
                raise ValueError("End pressures do not support column weight")
    return data, {
        "phases": m,
        "quadrature_order": order,
        "max_generalized_chemical_residual": max_mu,
        "max_interface_pressure_gap_pa": max_gap,
        "component_residual_mol": mass_error,
        "energy_residual_j": energy_error,
        "independent_fine_component_residual_mol": fine_n_error,
        "independent_fine_energy_residual_j": fine_e_error,
        "internal_energy_j": internal,
        "potential_energy_j": potential,
        "interface_pressures_pa": interface_pressures,
        "weight_balance_residual_pa": weight_error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("cuda", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    a, ma = check(cfg, args.cpu)
    b, mb = check(cfg, args.cuda)
    sa = a["result"]["state"]
    sb = b["result"]["state"]
    dt = abs(sa["temperature_k"] - sb["temperature_k"])
    dh = max(
        [abs(x - y) for x, y in zip(sa["interface_heights_m"], sb["interface_heights_m"])] or [0.0]
    )
    dp = max(
        [
            abs(x["point"]["pressure_pa"] - y["point"]["pressure_pa"])
            for x, y in zip(a["samples"], b["samples"])
        ]
        or [0.0]
    )
    if (
        dt > 1e-6
        or dh > 1e-7 * max(1, SeparatorGeometry(**cfg["system"]["geometry"]).height_m)
        or dp > 1e-2
    ):
        raise ValueError("CPU/CUDA column solutions disagree")
    result = {
        "cpu": ma,
        "cuda": mb,
        "max_temperature_difference_k": dt,
        "max_interface_difference_m": dh,
        "max_sample_pressure_difference_pa": dp,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
