"""Unfitted binary PR bubble predictions versus May et al. experimental data.

Temperature and liquid composition are the ONLY experimental model inputs.
Measured pressure/vapor composition never seed the solve. Archived 95%
expanded bounds are reported marginally, not a joint qualification test.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from surface.equilibrium_cache import cached_flashes
from surface.equilibrium_properties import EquilibriumFluid
from surface.liquid import HydraulicConvergenceError, PropertyModelError
from surface.torch_linear import TorchNewton

ROOT = Path(__file__).resolve().parents[1]
SCALES = (0.2, 0.5, 1.0, 2.0, 4.0)


def bubble_prediction(fluid, temperature, liquid_methane):
    """Solve two fugacity equations with fixed, observation-independent starts."""
    f = fluid
    t, x = f.tensor(temperature), f.tensor([liquid_methane, 1 - liquid_methane])
    if len(f.components) != 2 or not 0 < liquid_methane < 1:
        raise ValueError("Binary interior liquid composition required")
    f._validate(f.tensor(1e5), t, x)
    wilson = f.pc * torch.exp(
        5.373 * (1 + f.tensor([c.acentric_factor for c in f.components])) * (1 - f.tc / t)
    )
    p0 = float((x * wilson).sum().cpu())
    y0 = float(((x * wilson)[0] / (x * wilson).sum()).cpu())
    lower = f.tensor(
        [
            math.log(f.min_pressure_pa),
            math.log((liquid_methane + 1e-8) / (1 - liquid_methane - 1e-8)),
        ]
    )
    upper = f.tensor([math.log(f.max_pressure_pa), math.log((1 - 1e-10) / 1e-10)])

    def states(point):
        p = point[0].exp()
        y1 = torch.sigmoid(point[1])
        y = torch.stack((y1, 1 - y1))
        return p, y, f.tensor_homogeneous(p, t, x, "liquid"), f.tensor_homogeneous(p, t, y, "vapor")

    def residual(point):
        _p, y, l, g = states(point)
        return (
            torch.log(x)
            + l["log_fugacity_coefficients"]
            - torch.log(y)
            - g["log_fugacity_coefficients"]
        )

    candidates, attempts = [], []
    for scale in SCALES:
        solver = TorchNewton(device=f.device, maximum_iterations=32)
        solver.host_convergence = True
        try:
            point, r, ok, evaluations = solver.solve(
                residual,
                f.tensor([math.log(p0 * scale), math.log(y0 / (1 - y0))]),
                lower,
                upper,
                tolerance=2e-10,
            )
            p, y, l, g = states(point)
            pressure = float(p.detach().cpu())
            yvalue = float(y[0].detach().cpu())
            norm = float(r.abs().max().detach().cpu())
            valid = (
                bool(ok.detach().cpu())
                and norm <= 2e-10
                and bool((l["valid"] & g["valid"]).detach().cpu())
                and yvalue - liquid_methane > 1e-7
                and float((g["molar_volume_m3_mol"] / l["molar_volume_m3_mol"]).detach().cpu())
                > 1 + 1e-6
            )
            attempts.append(
                {
                    "wilson_pressure_scale": scale,
                    "evaluations": evaluations,
                    "pressure_pa": pressure,
                    "vapor_methane": yvalue,
                    "maximum_log_fugacity_residual": norm,
                    "nontrivial_root": valid,
                }
            )
            if valid and not any(
                abs(pressure - q["pressure_pa"]) <= 1e-7 * pressure for q in candidates
            ):
                # A coexistence root alone can be metastable. Reuse the actual
                # production global-search flash on a constructed 50/50 feed.
                z = (x + y.detach()) / 2
                check = f.flash_pt(pressure, temperature, z)
                stable = (
                    check["phase_count"] == 2
                    and float((check["phases"][0]["mole_fractions"] - x).abs().max().cpu()) < 1e-6
                    and float((check["phases"][1]["mole_fractions"] - y.detach()).abs().max().cpu())
                    < 1e-6
                )
                if stable:
                    candidates.append(
                        {
                            "pressure_pa": pressure,
                            "vapor_ethane_mole_fraction": 1 - yvalue,
                            "maximum_log_fugacity_residual": norm,
                            "production_flash_phase_count": check["phase_count"],
                            "production_flash_component_balance_error": float(
                                check["component_balance_error"].cpu()
                            ),
                            "production_flash_vapor_fraction_for_constructed_feed": float(
                                check["phases"][1]["molar_fraction"].cpu()
                            ),
                        }
                    )
        except (HydraulicConvergenceError, PropertyModelError, ValueError, RuntimeError) as exc:
            attempts.append({"wilson_pressure_scale": scale, "failure": str(exc)})
    if len(candidates) != 1:
        return {
            "status": "no_unique_stable_prediction",
            "candidates": candidates,
            "attempts": attempts,
        }
    return {"status": "predicted", **candidates[0], "attempts": attempts}


def run(reference, model_config, *, device="cpu"):
    reference, model_config = Path(reference).resolve(), Path(model_config).resolve()
    data = json.loads(reference.read_bytes())
    protocol = json.loads(model_config.read_bytes())
    if data["format"] != "may2015_methane_ethane_measurements_v1" or len(data["points"]) != 17:
        raise ValueError("Complete fixed 17-point experimental reference required")
    if protocol["format"] != "unfitted_may2015_pr_protocol_v1":
        raise ValueError("Explicit validation protocol required")
    origin = ROOT / protocol["model_origin_config"]
    if json.loads(origin.read_bytes())["system"]["fluid"] != protocol["fluid"]:
        raise ValueError("Protocol differs from the declared production fluid")
    manifest_path = ROOT / "tests/data/surface/may2015_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if hashlib.sha256(reference.read_bytes()).hexdigest() != manifest["reference_sha256"]:
        raise ValueError("Experimental reference differs from frozen provenance")
    f = EquilibriumFluid.from_dict(protocol["fluid"], device=device)
    if f.names != ("Methane", "Ethane") or any(
        v != 0 for row in protocol["fluid"]["binary_interactions"] for v in row
    ):
        raise ValueError("This unfitted protocol requires ordered methane/ethane and kij=0")
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        ROOT / "scripts/validate_may2015_vle.py",
        ROOT / "scripts/import_may2015_vle.py",
        reference,
        model_config,
        origin,
        manifest_path,
    ]
    hashes = {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in paths
    }
    rows = []
    with cached_flashes(f):
        for index, point in enumerate(data["points"]):
            prediction = bubble_prediction(
                f, point["temperature_k"], point["liquid_methane_mole_fraction"]
            )
            row = {"point_index": index, "measurement": point, "prediction": prediction}
            if prediction["status"] == "predicted":
                dp = prediction["pressure_pa"] - point["pressure_pa"]
                dy = prediction["vapor_ethane_mole_fraction"] - point["vapor_ethane_mole_fraction"]
                row.update(
                    pressure_error_pa=dp,
                    pressure_relative_error=dp / point["pressure_pa"],
                    vapor_ethane_error=dy,
                    pressure_error_over_archived_expanded_bound=dp
                    / point["pressure_expanded_uncertainty_pa"],
                    vapor_error_over_archived_expanded_bound=dy
                    / point["vapor_expanded_uncertainty"],
                    pressure_within_archived_bound=abs(dp)
                    <= point["pressure_expanded_uncertainty_pa"],
                    vapor_within_archived_bound=abs(dy) <= point["vapor_expanded_uncertainty"],
                )
            rows.append(row)
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha for p, sha in hashes.items()):
        raise RuntimeError("Validation inputs or source changed during comparison")
    scored = [r for r in rows if r["prediction"]["status"] == "predicted"]
    return {
        "format": "may2015_pr_experimental_validation_v1",
        "device": str(f.device),
        "reference_doi": data["doi"],
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "protocol": protocol,
        "source_sha256": hashes,
        "fit_performed": False,
        "experimental_points": len(rows),
        "predicted_points": len(scored),
        "failed_point_indices": [
            r["point_index"] for r in rows if r["prediction"]["status"] != "predicted"
        ],
        "pressure_within_archived_bounds": sum(r["pressure_within_archived_bound"] for r in scored),
        "vapor_within_archived_bounds": sum(r["vapor_within_archived_bound"] for r in scored),
        "pressure_mean_absolute_relative_error": sum(
            abs(r["pressure_relative_error"]) for r in scored
        )
        / len(scored)
        if scored
        else None,
        "maximum_pressure_absolute_relative_error": max(
            (abs(r["pressure_relative_error"]) for r in scored), default=None
        ),
        "vapor_ethane_mean_absolute_error": sum(abs(r["vapor_ethane_error"]) for r in scored)
        / len(scored)
        if scored
        else None,
        "cases": rows,
        "uncertainty_scope": "Marginal comparison with archived combined expanded 95% property bounds; no covariance or joint statistical acceptance claim. Wide bounds retained, no errors used for selection.",
        "qualification": "Only conditional binary bubble pressure and vapor composition. Constructed half-liquid/half-vapor feed is a numerical stability cross-check, not a measured vapor fraction. No caloric, water, pipe-hydraulics or complete-plant qualification.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", type=Path, default=ROOT / "tests/data/surface/may2015_methane_ethane.json"
    )
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/validation_may2015_pr.json")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a completed comparison")
    r = run(args.reference, args.protocol, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in r.items() if k not in ("cases", "protocol", "source_sha256")},
            indent=2,
        )
    )
    if r["predicted_points"] != r["experimental_points"]:
        raise SystemExit(1)
