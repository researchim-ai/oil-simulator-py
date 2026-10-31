"""Deterministic numerical envelope: pressure-coordinate solver vs spatial ODE.

The reference independently evaluates tabulated PVT in NumPy and solves
Colebrook with scipy.brentq; it does not call the production flash or quadrature.
This checks equations/implementation, NOT agreement with field experiments.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.blackoil_pipe import BlackOilPipe
from surface.blackoil_properties import BlackOilMixture
from surface.process import StockDensities


def spatial_reference(pipe, pin, masses, spec):
    pvt = spec["pvt"]
    rho = np.array([spec["stock_densities"][k] for k in ("oil_kg_m3", "water_kg_m3", "gas_kg_m3")])
    qo, qw, qg = np.array(masses) / rho
    mass = sum(masses)
    flux = mass / (math.pi * pipe.diameter_m**2 / 4)
    axis = np.array(pvt["pressure_MPa"]) * 1e6
    sample_pressure = pin
    undersaturated = False

    def column(key, p):
        values = np.array(pvt[key])
        i = np.searchsorted(axis, sample_pressure, side="right") - 1
        i = np.clip(i, 0, len(axis) - 2)
        slope = (values[i + 1] - values[i]) / (axis[i + 1] - axis[i])
        return values[i] + slope * (p - axis[i]), slope

    def oil(key, p, rs):
        ext = pvt["extended"]["oil"]
        ratios = np.array(ext["rs_values"])
        i = np.clip(np.searchsorted(ratios, rs, side="right") - 1, 0, len(ratios) - 2)
        weight = (rs - ratios[i]) / (ratios[i + 1] - ratios[i])
        answer = []
        for ratio in ratios[i : i + 2]:
            table = ext["tables"][str(float(ratio))]
            xp = np.array(table["pressure_MPa"]) * 1e6
            values = np.array(table[key])
            j = np.clip(np.searchsorted(xp, sample_pressure, side="right") - 1, 0, len(xp) - 2)
            slope = (values[j + 1] - values[j]) / (xp[j + 1] - xp[j])
            answer.append((values[j] + slope * (p - xp[j]), slope))
        return tuple((1 - weight) * a + weight * b for a, b in zip(*answer))

    def rhs(_s, y):
        p = float(y[0])
        bo, bop = column("Bo", p)
        bw, bwp = column("Bw", p)
        bg, bgp = column("Bg", p)
        rs, rsp = column("Rs_m3m3", p)
        muo, _ = column("mu_o_cP", p)
        if undersaturated:
            rs = qg / qo
            rsp = 0.0
            bo, bop = oil("Bo", p, rs)
            muo, _ = oil("mu_o_cP", p, rs)
        if qo == 0:
            rs = rsp = 0.0
        gas = max(0.0, qg - qo * rs)
        volumes = np.array([qo * bo, qw * bw, gas * bg])
        volume = sum(volumes)
        vp = (qo * bop + qw * bwp + gas * bgp - qo * rsp * bg) / mass
        mu = (
            np.dot(volumes, [muo, column("mu_w_cP", p)[0], column("mu_g_cP", p)[0]])
            * 0.001
            / volume
        )
        re = flux * pipe.diameter_m / mu
        if re <= 2000:
            f = 64 / re
        else:
            u = brentq(
                lambda u: (
                    u + 2 * math.log10(pipe.roughness_m / (3.7 * pipe.diameter_m) + 2.51 * u / re)
                ),
                1,
                100,
            )
            f = 1 / u**2
            if re < 4000:
                t = (re - 2000) / 2000
                t = t * t * (3 - 2 * t)
                f = (1 - t) * 64 / re + t * f
        v = volume / mass
        return [
            -(
                f * flux**2 * v / (2 * pipe.diameter_m)
                + 9.80665 * pipe.elevation_change_m / pipe.length_m / v
            )
            / (1 + flux**2 * vp)
        ]

    knots = [pin, *axis[axis < pin].tolist()]
    for table in pvt.get("extended", {}).get("oil", {}).get("tables", {}).values():
        knots.extend(p * 1e6 for p in table["pressure_MPa"] if axis[0] <= p * 1e6 < pin)
    rs_axis = np.array(pvt["Rs_m3m3"])
    if qo > 0 and rs_axis[0] < qg / qo < rs_axis[-1]:
        i = int(np.searchsorted(rs_axis, qg / qo, side="right")) - 1
        bubble = axis[i] + (qg / qo - rs_axis[i]) * (axis[i + 1] - axis[i]) / (
            rs_axis[i + 1] - rs_axis[i]
        )
        if axis[0] < bubble < pin:
            knots.append(float(bubble))
    knots = sorted(set(knots), reverse=True)
    position = 0.0
    pressure = pin
    for upper, lower in zip(knots, knots[1:]):
        sample_pressure = (upper + lower) / 2
        undersaturated = qo > 0 and qg / qo < np.interp(sample_pressure, axis, rs_axis)

        def event(_s, y, target=lower):
            return y[0] - target

        event.terminal = True
        event.direction = -1
        # Keep this segment's interpolation/phase branch fixed while the ODE
        # localizes the event. Restart on the other side at the exact pressure.
        sol = solve_ivp(
            rhs,
            (position, pipe.length_m),
            [pressure],
            rtol=1e-11,
            atol=1e-5,
            max_step=pipe.length_m / 40,
            events=event,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        if sol.status == 0:
            return float(sol.y[0, -1])
        position = float(sol.t[-1])
        pressure = lower
    raise RuntimeError("Spatial reference reached PVT boundary before the pipe outlet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/surface_gathering_facility.json",
    )
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--min-inlet-mpa", type=float, default=2.6)
    parser.add_argument("--max-inlet-mpa", type=float, default=4.3)
    parser.add_argument("--jacobian-every", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not math.isfinite(args.min_inlet_mpa)
        or not math.isfinite(args.max_inlet_mpa)
        or not 0 < args.min_inlet_mpa < args.max_inlet_mpa
    ):
        parser.error("Expected a positive finite inlet pressure range")
    if args.cases < 1 or args.jacobian_every < 1:
        parser.error("Positive case counts required")
    cfg = json.loads(args.config.read_text())["gathering"]
    model = BlackOilMixture(cfg["pvt"], StockDensities(**cfg["stock_densities"]))
    rng = np.random.default_rng(20261031)
    rows = []
    start = time.perf_counter()
    for index in range(args.cases):
        pin = float(rng.uniform(args.min_inlet_mpa * 1e6, args.max_inlet_mpa * 1e6))
        length = float(rng.uniform(100, 800))
        rise = float(rng.uniform(0, 60))
        diameter = float(rng.uniform(0.07, 0.14))
        rough = float(rng.uniform(0, 2e-5))
        qo = float(rng.uniform(0.0002, 0.0012))
        qw = float(rng.uniform(0.0004, 0.002))
        gor = float(rng.uniform(8, 115))
        scale = 10 ** float(rng.uniform(-3, 0))
        d = cfg["stock_densities"]
        masses = scale * np.array(
            [qo * d["oil_kg_m3"], qw * d["water_kg_m3"], qo * gor * d["gas_kg_m3"]]
        )
        pipe = BlackOilPipe(f"envelope_{index}", length, diameter, rise, rough)
        r = pipe.run(pin, masses, model)
        reference = spatial_reference(pipe, pin, masses, cfg)
        error = r["outlet_pressure_pa"] - reference
        row = {
            "index": index,
            "inlet_pressure_pa": pin,
            "pipe": pipe.__dict__,
            "component_mass_rates_kg_s": masses.tolist(),
            "outlet_pressure_pa": r["outlet_pressure_pa"],
            "reference_outlet_pa": reference,
            "reference_error_pa": error,
            "mechanical_balance_error_w": r["mechanical_balance_error_w"],
            "length_residual_m": r["length_residual_m"],
            "gas_appears_inside": r["inlet_state"]["phase_mass_rates_kg_s"][2] == 0
            and r["outlet_state"]["phase_mass_rates_kg_s"][2] > 0,
        }
        if abs(error) > 0.1:
            raise RuntimeError(f"Case {index}: spatial reference mismatch {error} Pa")
        if index % args.jacobian_every == 0:
            inverse = pipe.required_inlet(r["outlet_pressure_pa"], masses, model)
            row["inverse_error_pa"] = inverse["inlet_pressure_pa"] - pin
            if abs(row["inverse_error_pa"]) > 0.001:
                raise RuntimeError("Inverse pressure inconsistency")
            discrepancies = {}
            for j, (name, h) in enumerate(
                zip(
                    ("oil_kg_s", "water_kg_s", "gas_kg_s"),
                    tuple(max(abs(v) * 2e-5, 1e-11) for v in masses),
                )
            ):
                a = masses.copy()
                b = masses.copy()
                a[j] += h
                b[j] -= h
                fd = (
                    pipe.run(pin, a, model)["outlet_pressure_pa"]
                    - pipe.run(pin, b, model)["outlet_pressure_pa"]
                ) / (2 * h)
                actual = r["pressure_derivatives"][name]
                relative = abs(actual - fd) / max(1.0, abs(fd))
                discrepancies[name] = relative
                if relative > 2e-5:
                    raise RuntimeError(f"Case {index} Jacobian {name}: relative error {relative}")
            row["jacobian_scaled_errors"] = discrepancies
        rows.append(row)
        if (index + 1) % 32 == 0:
            print(f"{index + 1}/{args.cases} checked", flush=True)
    result = {
        "seed": 20261031,
        "inlet_range_mpa": [args.min_inlet_mpa, args.max_inlet_mpa],
        "cases": rows,
        "elapsed_s": time.perf_counter() - start,
        "max_reference_error_pa": max(abs(r["reference_error_pa"]) for r in rows),
        "max_mechanical_balance_error_w": max(abs(r["mechanical_balance_error_w"]) for r in rows),
        "scope": "numerical cross-check of homogeneous Rs-only closure; no experimental validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
