"""Independent scalar PR profile/aperture and stock/energy-transfer audit.

Uses NumPy/scalar PR, not the coupled pressure iteration or Torch vessel
solver. Does not re-solve the reservoir or certify a field stock assay.
The stream reference currently supports at most two equilibrium phases.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from check_stratified_pr import Reference
from check_stratified_pr_flow import audit as flow_audit
from check_stratified_pr_flow import close, profile

from surface.operation import _digest


def stream_state(stream, fluid):
    n = np.asarray(stream["component_molar_rates_mol_s"])
    active = np.flatnonzero(n > 0)
    if not len(active):
        close(stream["total_energy_rate_w"], 0.0, atol=0.0, rtol=0.0)
        return None
    ref = Reference(fluid, active)
    total = sum(n)
    mass = n[active] @ ref.mw
    enthalpy = (
        stream["total_energy_rate_w"]
        - mass
        * (stream["gravity_m_s2"] * stream["elevation_m"] + 0.5 * stream["velocity_m_s"] ** 2)
    ) / total
    if ref.pure:
        q = ref.pure.flash_ph(stream["pressure_pa"], enthalpy / ref.mw[0])
        return q["entropy_j_kg_k"] * ref.mw[0] * total
    q = ref.mixture.flash_ph(stream["pressure_pa"], enthalpy, n[active] / total)
    return q["molar_entropy_j_mol_k"] * total


def audit_step(spec, report):
    model = {k: v for k, v in spec.items() if k != "initial_state"}
    if report["model_id"] != _digest(model) or report["model"] != model["model"]:
        raise ValueError("Dynamic receiver model differs from requested model")
    if (
        report["reservoir_energy_equation_solved"] is not False
        or report["pressure_feedback_from_caloric_process"] is not True
        or report["pvt_eos_characterization_calibrated"] is not False
    ):
        raise ValueError("Incorrect physical scope markers")
    receiver = spec["receiver"]
    old = report["initial_state"]
    if (
        not math.isfinite(report["temporal_error_ratio"])
        or not 0 <= report["temporal_error_ratio"] <= 1
    ):
        raise ValueError("Rejected whole-system temporal estimate published")
    system, wall = receiver["column"], receiver["wall"]
    height = spec["inlet_height_m"]
    assay = np.array(spec["assay"]["stock_mass_fractions"])
    pmax = 0.0
    checks = []
    incoming = np.zeros(assay.shape[1])
    outgoing = incoming.copy()
    stock_total = np.zeros(3)
    energy_in = energy_out = ambient = duration = 0.0
    for stage in report["stages"]:
        dt, step = stage["dt_s"], stage["flow"]
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid coupled substep")
        final = {
            "column": step["column_state"],
            "wall_temperature_k": step["wall_temperature_k"],
            "time_s": dt,
        }
        cfg = {
            "column": system,
            "wall": wall,
            "wall_temperature_k": old["wall_temperature_k"],
            "durations_s": [dt],
            "feeds": [{"height_m": height, "stream": s} for s in stage["inlet_streams"]],
            "outlets": [
                {**p, "opening": report["outlet_openings"].get(p["name"], p["opening"])}
                for p in receiver["outlets"]
            ],
        }
        independent = flow_audit(
            cfg,
            {
                "config": cfg,
                "initial": {"state": old["column"]},
                "intervals": [
                    {
                        "steps": [step],
                        "state": final,
                        **{
                            k: step[k]
                            for k in ("ambient_heat_j", "feed_energy_j", "outlet_energy_j")
                        },
                    }
                ],
            },
        )
        f, _, sample, _, _, _ = profile(final["column"], system)
        index = sum(height > h for h in final["column"]["interface_heights_m"])
        p = sample(height, index)[0]
        trial = stage["pressure_trials"][-1]
        if not trial["physical_trial"]:
            raise ValueError("Rejected pressure trial published")
        defect = p - trial["pressure_pa"]
        tol = spec["control"]["pressure_tolerance_pa"] + spec["control"][
            "pressure_relative_tolerance"
        ] * max(p, trial["pressure_pa"])
        if abs(defect) > tol:
            raise ValueError("Independent END-pressure coupling residual rejected")
        close(trial["receiver_pressure_pa"], p, atol=1e-4, rtol=1e-10)
        pmax = max(pmax, abs(defect))
        mw = np.array([c.molar_mass_kg_mol for c in f.components])
        stock = np.zeros(3)
        conditions = spec["thermal_conditions"]
        if set(conditions) != set(stage["accepted_wells"]) or set(conditions) != set(
            stage["well_boundary"]
        ):
            raise ValueError("Accepted well/thermal boundary registry differs")
        for name, well in stage["accepted_wells"].items():
            mass = np.array(well["component_mass_rates_kg_s"])
            n = mass @ assay / mw
            stock += dt * mass
            saved = stage["well_boundary"][name]["stream"]
            close(saved["component_molar_rates_mol_s"], n, atol=1e-12, rtol=1e-10)
            close(saved["pressure_pa"], well["bhp_pa"], atol=1e-8, rtol=0.0)
            condition = conditions[name]
            close(saved["elevation_m"], condition["elevation_m"], atol=1e-10, rtol=0)
            close(saved["velocity_m_s"], condition.get("velocity_m_s", 0), atol=1e-10, rtol=0)
            if "total_energy_rate_w" in condition:
                close(
                    saved["total_energy_rate_w"],
                    condition["total_energy_rate_w"],
                    atol=1e-9,
                    rtol=1e-10,
                )
            elif sum(n):
                ref = Reference(f, np.flatnonzero(n > 0))
                z = n / sum(n)
                if ref.pure:
                    q = ref.pure.state(
                        well["bhp_pa"], condition["temperature_k"], "stable", derivatives=False
                    )
                    h = q["enthalpy_j_kg"] * ref.mw[0]
                else:
                    q = ref.mixture.flash_pt(
                        well["bhp_pa"], condition["temperature_k"], z[ref.active]
                    )
                    h = q["molar_enthalpy_j_mol"]
                expected = sum(n) * h + sum(mass) * (
                    saved["gravity_m_s2"] * saved["elevation_m"] + 0.5 * saved["velocity_m_s"] ** 2
                )
                close(saved["total_energy_rate_w"], expected, atol=1e-6, rtol=1e-8)
        close(stock, stage["actual_stock_component_mass_kg"], atol=1e-12, rtol=1e-10)
        # Every accepted per-well packet is transported once, preserving N/E;
        # the independent PH states also check passive transfer entropy.
        streams = stage["inlet_streams"]
        if len(streams) != len(stage["well_boundary"]):
            raise ValueError("Missing accepted inlet")
        names = stage["inlet_well_names"]
        if len(names) != len(set(names)) or set(names) != set(stage["well_boundary"]):
            raise ValueError("Inlet well identities differ from accepted wells")
        for name, inlet in zip(names, streams, strict=True):
            source = stage["well_boundary"][name]["stream"]
            for key in ("component_molar_rates_mol_s", "total_energy_rate_w", "model_id"):
                if key == "model_id":
                    if source[key] != inlet[key] or source[key] != f.model_id:
                        raise ValueError("Stream EOS/reference identity changed")
                else:
                    close(inlet[key], source[key], atol=1e-9, rtol=1e-10)
            close(inlet["elevation_m"], system["elevation_m"] + height, atol=1e-10, rtol=0)
            close(inlet["pressure_pa"], trial["pressure_pa"], atol=1e-8, rtol=0)
            source_s = stream_state(source, f)
            inlet_s = stream_state(inlet, f)
            if source_s is not None and inlet_s < source_s - 1e-6:
                raise ValueError("Passive well-to-vessel transport lowers entropy")
        checks.append(independent)
        incoming += np.array(step["feed_component_moles"])
        outgoing += np.array(step["outlet_component_moles"])
        energy_in += step["feed_energy_j"]
        energy_out += step["outlet_energy_j"]
        ambient += step["ambient_heat_j"]
        stock_total += stock
        duration += dt
        old = {**old, **final, "time_s": old["time_s"] + dt}
    close(duration, report["dt_s"], atol=1e-12, rtol=1e-12)
    close(old["time_s"], report["state"]["time_s"], atol=1e-12, rtol=1e-12)
    close(old["wall_temperature_k"], report["state"]["wall_temperature_k"], atol=1e-10, rtol=0)
    if _digest(old["column"]) != _digest(report["state"]["column"]):
        raise ValueError("Final receiver inventory differs from accepted stages")
    return {
        "accepted": True,
        "checked_stages": len(checks),
        "stage_audits": checks,
        "maximum_pressure_defect_pa": pmax,
        "feed_component_moles": incoming.tolist(),
        "export_component_moles": outgoing.tolist(),
        "produced_stock_component_kg": stock_total.tolist(),
        "thermal_boundary_energy_j": energy_in,
        "export_energy_j": energy_out,
        "ambient_heat_j": ambient,
        "qualification": "Scalar PR profile/32-node finite-aperture audit, explicit stock assay and passive stream transfer. Does not re-solve reservoir, certify temporal error estimate, global stability or experimental equipment calibration.",
    }


def audit(config, result):
    if result["config"] != config or result["format"] != "dynamic_pr_receiver_run_v1":
        raise ValueError("Dynamic receiver request/result differ")
    root = Path(__file__).resolve().parents[1]
    for path, sha in result["source_sha256"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Changed source: {path}")
    spec = config["simulator"]["simulation"]["dynamic_receiver"]
    old = spec["initial_state"]
    audits = []
    schedule = config["intervals"]
    period, period_end = 0, schedule[0]["duration_s"]
    totals = {}
    for report in result["records"]:
        while old["time_s"] >= period_end and period + 1 < len(schedule):
            period += 1
            period_end += schedule[period]["duration_s"]
        if report["outlet_openings"] != schedule[period]["outlet_openings"]:
            raise ValueError("Accepted commands differ from requested schedule")
        if report["state"]["time_s"] > period_end + 1e-12:
            raise ValueError("Coupled substep crosses a scheduled command")
        if _digest(report["initial_state"]) != _digest(old):
            raise ValueError("Discontinuous coupled checkpoint")
        checked = audit_step(spec, report)
        audits.append(checked)
        for key in (
            "feed_component_moles",
            "export_component_moles",
            "produced_stock_component_kg",
            "thermal_boundary_energy_j",
            "export_energy_j",
            "ambient_heat_j",
        ):
            totals[key] = np.asarray(totals.get(key, np.zeros_like(checked[key]))) + np.asarray(
                checked[key]
            )
            close(report["cumulative"][key], totals[key], atol=1e-8, rtol=1e-9)
        close(report["cumulative"]["elapsed_s"], report["state"]["time_s"], atol=1e-12, rtol=0)
        old = report["state"]
    if not audits:
        raise ValueError("No accepted coupled intervals")
    close(old["time_s"], sum(i["duration_s"] for i in config["intervals"]), atol=1e-10, rtol=1e-12)
    return {"accepted": True, "intervals": audits}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(json.loads(args.config.read_text()), json.loads(args.report.read_text())),
            indent=2,
        )
    )
