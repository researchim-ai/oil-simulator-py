"""Independently audit serialized CPU/CUDA equilibrium-process reports."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.operation import _digest


def audit(path):
    data = json.loads(path.read_text())
    saved = dict(data)
    checksum = saved.pop("checksum")
    if checksum != _digest(saved):
        raise ValueError("Report checksum mismatch")
    root = Path(__file__).resolve().parents[1]
    for name, digest in data["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Report source differs from current implementation: {name}")
    process = data["process"]
    incoming = [x["stream"] for x in data["boundary"]["boundaries"].values()]
    outgoing = process["products"]
    components = data["configuration"]["fluid"]["components"]
    n = len(components)
    dm = [
        math.fsum(s["component_molar_rates_mol_s"][i] for s in outgoing)
        - math.fsum(s["component_molar_rates_mol_s"][i] for s in incoming)
        for i in range(n)
    ]
    energy = (
        math.fsum(s["total_energy_rate_w"] for s in outgoing)
        - math.fsum(s["total_energy_rate_w"] for s in incoming)
        - process["balances"]["heat_w"]
        - process["balances"]["shaft_power_w"]
    )
    energy_scale = max(
        1.0, math.fsum(abs(s["total_energy_rate_w"]) for s in (*incoming, *outgoing))
    )
    if max(map(abs, dm)) > 1e-8 or abs(energy) > 1e-6 + 1e-9 * energy_scale:
        raise ValueError("Serialized component/energy balance failed")
    eq = process["equilibrium"]
    active = np.array(eq["mole_fractions"]) > 0
    beta = np.array([p["molar_fraction"] for p in eq["phases"]])
    x = np.array([p["mole_fractions"] for p in eq["phases"]])
    mu = (
        np.log(x[:, active])
        + np.array([p["log_fugacity_coefficients"] for p in eq["phases"]])[:, active]
    )
    fugacity = float(np.max(np.abs(mu - mu[-1])))
    if fugacity > 1e-8 or np.max(np.abs(beta @ x - eq["mole_fractions"])) > 1e-10:
        raise ValueError("Serialized phase equilibrium failed")
    mixed = process["mixed_stream"]
    total = math.fsum(mixed["component_molar_rates_mol_s"])
    if len(outgoing) != len(eq["phases"]):
        raise ValueError("Serialized products do not match equilibrium phases")
    for product, phase in zip(outgoing, eq["phases"]):
        expected_rates = total * phase["molar_fraction"] * np.array(phase["mole_fractions"])
        if not np.allclose(
            product["component_molar_rates_mol_s"], expected_rates, rtol=1e-10, atol=1e-10
        ):
            raise ValueError("Serialized product differs from its equilibrium phase")
        for key in ("pressure_pa", "elevation_m", "velocity_m_s", "gravity_m_s2"):
            if product[key] != mixed[key]:
                raise ValueError("Serialized product pressure/mechanical reference differs")
        mass = math.fsum(
            rate * c["molar_mass_kg_mol"] for rate, c in zip(expected_rates, components)
        )
        expected_energy = total * phase["molar_fraction"] * phase["molar_enthalpy_j_mol"] + mass * (
            mixed["gravity_m_s2"] * mixed["elevation_m"] + 0.5 * mixed["velocity_m_s"] ** 2
        )
        if abs(product["total_energy_rate_w"] - expected_energy) > 1e-6 + 1e-9 * max(
            1.0, abs(expected_energy)
        ):
            raise ValueError("Serialized product energy differs from its equilibrium phase")
    for p in eq["phases"]:
        if (
            abs(
                p["molar_internal_energy_j_mol"]
                - (p["molar_enthalpy_j_mol"] - eq["pressure_pa"] * p["molar_volume_m3_mol"])
            )
            > 1e-7
        ):
            raise ValueError("Serialized U/H/PV identity failed")
    incoming_mass = math.fsum(
        s["component_molar_rates_mol_s"][i] * components[i]["molar_mass_kg_mol"]
        for s in incoming
        for i in range(n)
    )
    stock_mass = math.fsum(
        math.fsum(w["component_mass_rates_kg_s"]) for w in data["configuration"]["wells"].values()
    )
    if abs(incoming_mass - stock_mass) > 1e-9:
        raise ValueError("Serialized well characterization changed mass")
    return data, {
        "phase_count": eq["phase_count"],
        "temperature_k": eq["temperature_k"],
        "mass_rate_kg_s": incoming_mass,
        "component_residuals_mol_s": dm,
        "energy_residual_w": energy,
        "max_log_fugacity_error": fugacity,
        "entropy_production_w_k": process["balances"]["entropy_production_w_k"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("cuda", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu, a = audit(args.cpu)
    cuda, b = audit(args.cuda)
    if (
        cpu["configuration"] != cuda["configuration"]
        or cpu["process"]["model_id"] != cuda["process"]["model_id"]
    ):
        raise ValueError("CPU/CUDA configurations or physical model identities differ")
    if a["phase_count"] != b["phase_count"]:
        raise ValueError("CPU/CUDA phase counts differ")
    ca = np.array([s["component_molar_rates_mol_s"] for s in cpu["process"]["products"]])
    cb = np.array([s["component_molar_rates_mol_s"] for s in cuda["process"]["products"]])
    if (
        not np.allclose(ca, cb, rtol=2e-8, atol=2e-7)
        or abs(a["temperature_k"] - b["temperature_k"]) > 1e-6
    ):
        raise ValueError("CPU/CUDA physical results differ beyond declared comparison tolerance")
    result = {
        "cpu": a,
        "cuda": b,
        "max_product_component_difference_mol_s": float(np.max(np.abs(ca - cb))),
        "temperature_difference_k": abs(a["temperature_k"] - b["temperature_k"]),
        "source_sha256": cpu["source_sha256"],
        "configuration_sha256": _digest(cpu["configuration"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
