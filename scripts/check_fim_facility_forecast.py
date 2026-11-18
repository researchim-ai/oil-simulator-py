"""Audit accepted forecast material and controls without running either solver.

This is an accounting/constraint audit, not independent validation of the PVT
or homogeneous gathering closure. The production FIM CNV/MB remain required.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulator.capacity_amg_reuse import audit_capacity_amg
from simulator.capacity_sensitivity_audit import audit_capacity_sensitivity
from simulator.fim_initial_cell_cache import audit_initial_cell_cache
from simulator.rate_cap_invariance import audit_inactive_cap_trace

COMPONENTS = ("oil", "water", "gas")


def audit(
    config, report, *, expected_schedule=None, expected_days=None, mass_atol_kg=0.01, mass_rtol=1e-8
):
    def close(a, b, *, atol=1e-9, rtol=1e-10):
        if (
            not math.isfinite(a)
            or not math.isfinite(b)
            or not math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
        ):
            raise ValueError(f"Forecast audit mismatch: {a} != {b}")

    density = config["simulation"]["coupled_facility"]["facility"]["stock_densities"]
    rho = [density[c + "_kg_m3"] for c in COMPONENTS]
    steps = report["accepted_substeps"]
    maximum_days = config["simulation"].get("max_time_step_days")
    if maximum_days is not None and (isinstance(maximum_days, bool) or not math.isfinite(maximum_days)
                                     or maximum_days <= 0):
        raise ValueError("Invalid configured maximum time step")
    maximum_s = math.inf if maximum_days is None else maximum_days * 86400
    if not steps:
        raise ValueError("No accepted coupled steps")
    clock = 0.0
    withdrawn = [0.0] * 3
    max_surface_residual = 0.0
    evaluations = 0
    reservoir_evaluations = 0
    initial_cell_evaluations = initial_cell_hits = 0
    accepted_sensitivity_linear = accepted_sensitivity_plant = 0
    estimator_reservoir_evaluations = 0
    estimator_evaluations = 0
    events = report.get("facility_control_events", [])
    if expected_days is not None:
        close(report["elapsed_days"], expected_days, atol=1e-12, rtol=0)
    if (
        expected_schedule is not None
        and [{"time_days": e["time_days"], "controls": e["controls"]} for e in events]
        != expected_schedule
    ):
        raise ValueError("Requested operating schedule was not completed")
    previous_event = -1.0
    previous_rates, previous_controls = None, None
    for event in events:
        if not previous_event < event["time_days"] < report["elapsed_days"]:
            raise ValueError("Invalid forecast event order")
        close(event["applied_time_days"], event["time_days"], atol=1e-12, rtol=0)
        previous_event = event["time_days"]
    for step_index, step in enumerate(steps):
        dt = step["dt_s"]
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid accepted duration")
        if dt > maximum_s + 1e-7:
            raise ValueError("Accepted stage exceeds configured maximum time step")
        end = step["time_days"] * 86400
        close(end, clock + dt, atol=1e-7, rtol=1e-12)
        controls = {}
        for event in events:
            when = event["time_days"] * 86400
            if clock + 8 * math.ulp(clock) < when < end - 8 * math.ulp(end):
                raise ValueError("Accepted step crossed an operating event")
            if when <= clock + 8 * math.ulp(clock):
                controls = event["controls"]
        if step["facility_controls"] != controls:
            raise ValueError("Wrong controls on accepted reservoir interval")
        coupling = step["facility_coupling"]
        audit_capacity_amg(coupling, config['simulation']['coupled_facility'].get('control') or {})
        if "operator_controls" in coupling and coupling["operator_controls"] != controls:
            raise ValueError("Coupling reports the wrong operator controls")
        plant = coupling["facility"]
        if (
            not plant["feasible"]
            or plant["violations"]
            or not all(x["passed"] for x in plant["limits"])
        ):
            raise ValueError("Infeasible plant was accepted")
        reuse = (config['simulation']['coupled_facility'].get('control') or {}).get('reuse_inactive_caps', False)
        if type(reuse) is not bool:
            raise ValueError('Invalid inactive-cap reuse policy')
        reservoir_evaluations += audit_inactive_cap_trace(coupling, config['wells'], reuse)
        cell_evaluations, cell_hits = audit_initial_cell_cache(
            coupling, config['simulation']['coupled_facility'].get('control') or {})
        initial_cell_evaluations += cell_evaluations
        initial_cell_hits += cell_hits
        if ((config['simulation']['coupled_facility'].get('control') or {}).get('search_method')
                == 'sensitivity_safeguarded' and coupling.get('search_method') != 'sensitivity_safeguarded'):
            raise ValueError('Sensitivity execution policy differs from config')
        sensitivity_linear, sensitivity_plant = audit_capacity_sensitivity(coupling)
        accepted_sensitivity_linear += sensitivity_linear
        accepted_sensitivity_plant += sensitivity_plant
        close(coupling["dt_s"], dt, atol=1e-7)
        if config["simulation"]["coupled_facility"].get("temporal_control") is not None and step_index % 2 == 1:
            # Every accepted macro interval must publish both actual half
            # stages. The discarded coarse solve contributes work, not mass.
            temporal = coupling["temporal_control"]
            errors = temporal["errors"]
            bhp_tol = config["simulation"]["coupled_facility"]["temporal_control"].get("well_bhp_tolerance_pa")
            if bhp_tol is not None:
                if isinstance(bhp_tol, bool) or not math.isfinite(bhp_tol) or bhp_tol <= 0:
                    raise ValueError("Invalid temporal well BHP tolerance")
                coarse = temporal.get("coarse_well_bhp_pa", {})
                fine_bhp = {k: w["bhp_pa"] for k, w in step["surface"]["wells"].items()}
                if not isinstance(coarse, dict) or set(coarse) != set(fine_bhp) or "well_bhp" not in errors:
                    raise ValueError("Missing temporal well BHP evidence")
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in [*coarse.values(), *fine_bhp.values()]):
                    raise ValueError("Nonfinite temporal well BHP evidence")
                actual = max((abs(coarse[k] - fine_bhp[k]) for k in coarse), default=0.) / bhp_tol
                close(errors["well_bhp"], actual, atol=1e-12)
            if not errors or any(not math.isfinite(v) or v < 0 or v > 1 for v in errors.values()):
                raise ValueError("Unresolved accepted temporal defect")
            close(temporal["indicator"], max(errors.values()), atol=0, rtol=0)
            first = steps[step_index - 1]
            policy = config["simulation"]["coupled_facility"]["temporal_control"]
            scaling = policy.get("withdrawal_error_scale", "increment")
            if scaling not in ("increment", "cumulative"):
                raise ValueError("Invalid withdrawal error scaling")
            if scaling == "cumulative":
                coarse_inc = temporal.get("coarse_withdrawal_kg")
                coarse_total = temporal.get("coarse_withdrawn_total_kg")
                if temporal.get("withdrawal_error_scale") != scaling or any(
                    not isinstance(v, list) or len(v) != 3 for v in (coarse_inc, coarse_total)
                ):
                    raise ValueError("Missing cumulative withdrawal error evidence")
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in [*coarse_inc, *coarse_total]):
                    raise ValueError("Nonfinite cumulative withdrawal evidence")
                atol = policy.get("withdrawal_absolute_tolerance_kg", 1e-5)
                rtol = policy.get("withdrawal_relative_tolerance", .002)
                if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in (atol, rtol)):
                    raise ValueError("Invalid withdrawal error budget")
                ratios = []
                for i in range(3):
                    base = withdrawn[i] - first["withdrawn_component_mass_kg"][i]
                    fine_inc = first["withdrawn_component_mass_kg"][i] + step["withdrawn_component_mass_kg"][i]
                    close(coarse_total[i], base + coarse_inc[i])
                    scale = atol + rtol * max(abs(coarse_total[i]), abs(base + fine_inc))
                    ratios.append(abs(coarse_inc[i] - fine_inc) / scale)
                close(errors["withdrawal"], max(ratios), atol=1e-9)
            elif temporal.get("withdrawal_error_scale", "increment") != "increment":
                raise ValueError("Unexpected cumulative withdrawal error evidence")
            close(temporal["macro_dt_s"], first["dt_s"] + dt, atol=1e-7)
            if temporal["macro_dt_s"] > maximum_s + 1e-7:
                raise ValueError("Temporal macro interval exceeds configured maximum time step")
            if first["facility_controls"] != controls:
                raise ValueError("Temporal macro step crossed an operating event")
            total = temporal["total_capacity_evaluations"]
            fine = first["facility_coupling"]["evaluations"] + coupling["evaluations"]
            if type(total) is not int or total < fine + 1:
                raise ValueError("Missing discarded coarse-solve work")
            estimator_evaluations += total - fine
            actual_total = temporal.get('total_reservoir_evaluations', total)
            actual_fine = sum(s['facility_coupling'].get('reservoir_evaluations',
                              s['facility_coupling']['evaluations']) for s in (first, step))
            if (type(actual_total) is not int or not actual_fine + 1 <= actual_total <= total
                    or reuse and 'total_reservoir_evaluations' not in temporal):
                raise ValueError('Invalid temporal reservoir evaluation accounting')
            estimator_reservoir_evaluations += actual_total - actual_fine
            amg_control = config['simulation']['coupled_facility'].get('control') or {}
            if amg_control.get('reuse_capacity_amg', False):
                total_amg = temporal.get('total_capacity_amg_work')
                audit_capacity_amg({'evaluations': actual_total, 'capacity_amg_work': total_amg}, amg_control)
                fine_amg = [audit_capacity_amg(s['facility_coupling'], amg_control) for s in (first, step)]
                if any(total_amg[key] < sum(w[key] for w in fine_amg) for key in total_amg):
                    raise ValueError('Missing discarded coarse AMG work')
            if coupling.get('search_method') == 'sensitivity_safeguarded':
                fine_linear = fine_plant = 0
                for stage in (first, step):
                    nl, np = audit_capacity_sensitivity(stage['facility_coupling'])
                    fine_linear += nl
                    fine_plant += np
                total_linear = temporal.get('total_sensitivity_linear_solves')
                total_plant = temporal.get('total_sensitivity_plant_evaluations')
                if (type(total_linear) is not int or not fine_linear <= total_linear <= actual_total
                        or type(total_plant) is not int or not fine_plant <= total_plant <= 4*total_linear):
                    raise ValueError('Invalid discarded-stage sensitivity work')
        if not 0 < coupling["fraction"] <= 1:
            raise ValueError("Invalid rate-cap fraction")
        if "capacity_bracket" in coupling:
            bracket = coupling["capacity_bracket"]
            if coupling["fraction"] == 1:
                if bracket is not None:
                    raise ValueError("Requested rate has an unexpected capacity bracket")
            else:
                if bracket is None:
                    raise ValueError("Limited rate requires a solved capacity bracket")
                low, high = bracket["feasible_fraction"], bracket["infeasible_fraction"]
                tolerance = (config["simulation"]["coupled_facility"].get("control") or {}).get(
                    "fraction_tolerance", 1e-4
                )
                close(low, coupling["fraction"], atol=0, rtol=0)
                close(bracket["width"], high - low, atol=1e-15)
                close(bracket["fraction_tolerance"], tolerance, atol=0, rtol=0)
                if not 0 < low < high <= 1 or high - low > tolerance:
                    raise ValueError("Unresolved rate-cap bracket")
                if not any(t["fraction"] == high and not t["feasible"] for t in coupling["trace"]):
                    raise ValueError("Upper capacity bracket was not solved")
                method = (config["simulation"]["coupled_facility"].get("control") or {}).get("commit_method", "replay")
                index = -1
                if method == "state_restore":
                    commit = coupling["capacity_commit"]
                    index = commit["solved_trace_index"]
                    if (commit["method"] != method or type(index) is not int
                            or not 0 <= index < len(coupling["trace"])):
                        raise ValueError("Invalid selected capacity state reference")
                elif method != "replay":
                    raise ValueError("Unknown capacity commit method")
                if not coupling["trace"][index]["feasible"] or coupling["trace"][index]["fraction"] != low:
                    raise ValueError("Selected capacity has no matching solved feasible trial")
        for applied, well in zip(
            coupling["applied_rate_caps_m3_day"], config["wells"], strict=True
        ):
            close(applied, well["control_value"] * coupling["fraction"])
        for limit in config["simulation"]["coupled_facility"]["facility"].get("limits", []):
            measurement = plant["measurements"][limit["metric"]]
            value = measurement["value"]
            if (
                measurement["unit"] != limit["unit"]
                or not math.isfinite(value)
                or value < limit.get("minimum", -math.inf)
                or value > limit.get("maximum", math.inf)
            ):
                raise ValueError("Configured equipment constraint violated")
        evaluations += coupling["evaluations"]
        wells = list(step["surface"]["wells"].values())
        current_rates = [
            math.fsum(w["component_mass_rates_kg_s"][i] for w in wells) for i in range(3)
        ]
        policy = config["simulation"]["coupled_facility"].get("time_control")
        if policy is not None:
            saved_indicator = coupling["rate_time_control"]["indicator"]
            if previous_rates is None or previous_controls != controls:
                if saved_indicator is not None or dt > policy.get(
                    "startup_step_days", 1.0
                ) * 86400 * (1 + 1e-12):
                    raise ValueError("Invalid startup/event rate-control interval")
            else:
                indicator = max(
                    0.5
                    * abs(x - y)
                    / (
                        policy.get("rate_absolute_tolerance_kg_s", 1e-8)
                        + policy.get("rate_relative_tolerance", 0.01) * max(abs(x), abs(y))
                    )
                    for x, y in zip(previous_rates, current_rates, strict=True)
                )
                close(indicator, saved_indicator, atol=1e-10)
                if indicator > 1:
                    raise ValueError("Unresolved accepted rate-change indicator")
        previous_rates, previous_controls = current_rates, controls
        for i, component in enumerate(COMPONENTS):
            well_rate = math.fsum(w["component_mass_rates_kg_s"][i] for w in wells)
            feed_rate = plant["feed"][component + "_m3_s"] * rho[i]
            export_rate = math.fsum(
                s[component + "_m3_s"] * rho[i] for s in plant["products"].values()
            )
            close(feed_rate, well_rate, atol=1e-12)
            close(export_rate, well_rate, atol=1e-12)
            max_surface_residual = max(max_surface_residual, abs(export_rate - well_rate))
            amount = well_rate * dt
            close(step["withdrawn_component_mass_kg"][i], amount)
            withdrawn[i] = math.fsum((withdrawn[i], amount))
            account = coupling["material_account"]
            close(account["produced_kg"][component], withdrawn[i])
            close(account["elapsed_s"], end, atol=1e-7)
        clock = end
    close(clock, report["elapsed_days"] * 86400, atol=1e-7)
    facility_control = config['simulation']['coupled_facility'].get('control') or {}
    if config['simulation']['coupled_facility'].get('temporal_control') is not None and facility_control.get('reuse_capacity_amg', False):
        work = report.get('temporal_work') or {}
        actual = work.get('reservoir_evaluations')
        if type(actual) is not int or actual < reservoir_evaluations:
            raise ValueError('Invalid total AMG reservoir work')
        total_amg = audit_capacity_amg({'evaluations': actual, 'capacity_amg_work': work.get('capacity_amg_work')}, facility_control)
        fine_amg = [audit_capacity_amg(s['facility_coupling'], facility_control) for s in steps]
        if any(total_amg[key] < sum(w[key] for w in fine_amg) for key in total_amg):
            raise ValueError('Missing accepted AMG work')
    if config["simulation"]["coupled_facility"].get("temporal_control") is not None and len(steps) % 2:
        raise ValueError("Incomplete temporal stage pair")
    residuals = []
    for i in range(3):
        close(report["withdrawn_component_mass_kg"][i], withdrawn[i])
        residual = (
            report["initial_component_mass_kg"][i]
            - report["final_component_mass_kg"][i]
            - withdrawn[i]
        )
        close(residual, report["mass_balance_error_kg"][i], atol=1e-8)
        budget = mass_atol_kg + mass_rtol * max(abs(withdrawn[i]), 1.0)
        if abs(residual) > budget:
            raise ValueError("Reservoir-to-export mass budget exceeded")
        residuals.append(residual)
    return {
        "accepted": True,
        "checked_substeps": len(steps),
        "elapsed_days": clock / 86400,
        "minimum_step_s": min(s["dt_s"] for s in steps),
        "maximum_step_s": max(s["dt_s"] for s in steps),
        "capacity_evaluations": evaluations,
        "reservoir_evaluations": reservoir_evaluations,
        "accepted_initial_cell_evaluations": initial_cell_evaluations,
        "accepted_initial_cell_reuses": initial_cell_hits,
        "accepted_sensitivity_linear_solves": accepted_sensitivity_linear,
        "accepted_sensitivity_plant_evaluations": accepted_sensitivity_plant,
        "accepted_macro_estimator_reservoir_evaluations": estimator_reservoir_evaluations,
        "accepted_macro_estimator_capacity_evaluations": estimator_evaluations,
        "operating_events": len(events),
        "mass_balance_error_kg": residuals,
        "maximum_surface_component_residual_kg_s": max_surface_residual,
        "mass_atol_kg": mass_atol_kg,
        "mass_rtol": mass_rtol,
        "scope": __doc__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.bundle.suffix == ".gz":
        import gzip

        with gzip.open(args.bundle, "rt") as handle:
            bundle = json.load(handle)
    else:
        bundle = json.loads(args.bundle.read_text())
    print(
        json.dumps(
            audit(
                bundle["config"],
                bundle["result"],
                expected_schedule=bundle["schedule"],
                expected_days=bundle["requested_days"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
