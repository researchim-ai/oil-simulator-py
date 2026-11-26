"""Independent accepted-clock, stock-mass, water-resource and control audit.

Hydraulic closure is checked by derivative tests and existing equipment-law
regressions. This accounting audit is not experimental physical validation.
"""

import math

from surface.water_injection import WaterInjectionTree


def _audit_temporal(config, stages, close):
    options = config["simulation"]["coupled_waterflood"].get("temporal_control")
    if options is None:
        if any("temporal_control" in s["coupling"] for s in stages):
            raise ValueError("Unexpected temporal control without a configured policy")
        return
    if len(stages) % 2:
        raise ValueError("Missing accepted temporal half stage")
    atol = options.get("withdrawal_absolute_tolerance_kg", 1e-5)
    rtol = options.get("withdrawal_relative_tolerance", .002)
    def defect(a, b, absolute=atol, relative=rtol):
        if len(a) != len(b):
            raise ValueError("Different temporal evidence dimensions")
        if any(isinstance(x, bool) or not math.isfinite(x) for x in [*a, *b]):
            raise ValueError("Nonfinite temporal evidence")
        return max((abs(x - y) / (absolute + relative * max(abs(x), abs(y)))
                    for x, y in zip(a, b)), default=0.)
    def transfers(stage):
        r = stage["coupling"]
        return (r["material_account"]["external_in_kg"]
                + r["material_account"]["external_out_kg"]
                + [r["water_rates_kg_s"][n] * r["dt_s"]
                   for n in ["produced", "injected", "recycled", "makeup", "disposed"]])
    previous = None
    for index in range(0, len(stages), 2):
        first, second = stages[index:index + 2]
        a, b = first["coupling"], second["coupling"]
        if "temporal_control" in a or "temporal_control" not in b:
            raise ValueError("Invalid accepted temporal stage pair")
        row, probe = b["temporal_control"], b["temporal_probe"]
        errors = row["errors"]
        required = {"pressure", "s_w", "s_g", "withdrawal", "component_inventory",
                    "resource_transfers", "drive_energy"}
        if options.get("well_bhp_tolerance_pa") is not None:
            required.add("well_bhp")
        if set(errors) != required or any(not math.isfinite(x) or not 0 <= x <= 1 for x in errors.values()):
            raise ValueError("Unresolved accepted waterflood temporal defect")
        close(row["indicator"], max(errors.values()), atol=1e-12)
        close(row["macro_dt_s"], a["dt_s"] + b["dt_s"], atol=1e-7)
        close(a["dt_s"], b["dt_s"], atol=1e-7)
        maximum = config["simulation"].get("max_time_step_days", math.inf) * 86400
        if row["macro_dt_s"] > maximum + 1e-7:
            raise ValueError("Temporal macro step exceeds configured maximum")
        if a["operator_controls"] != b["operator_controls"]:
            raise ValueError("Temporal macro step crossed a control change")
        if (config["simulation"].get("coupled_time_control", True)
                and previous != a["operator_controls"]
                and row["macro_dt_s"] > options.get("startup_step_days", .25) * 86400 + 1e-7):
            raise ValueError("Temporal startup or event step exceeds policy")
        previous = b["operator_controls"]
        fine_work = len(a["trace"]) + len(b["trace"])
        if row["total_capacity_evaluations"] <= fine_work:
            raise ValueError("Missing coarse temporal solve evidence")
        order = probe["well_order"]
        if order != list(first["surface"]["wells"]) or order != list(second["surface"]["wells"]):
            raise ValueError("Different temporal well ordering")
        coarse = probe["coarse_well_withdrawal_kg"]
        if len(coarse) != len(order) or any(len(x) != 3 for x in coarse):
            raise ValueError("Missing per-well temporal mass evidence")
        fine = [[sum(stage["surface"]["wells"][name]["component_mass_rates_kg_s"][j]
                     * stage["coupling"]["dt_s"] for stage in [first, second])
                 for j in range(3)] for name in order]
        close(errors["withdrawal"], defect(
            [x for row in coarse for x in row], [x for row in fine for x in row]), atol=1e-9)
        total = [x + y for x, y in zip(transfers(first), transfers(second))]
        close(errors["resource_transfers"], defect(probe["coarse_transfers_kg"], total), atol=1e-9)
        energy = sum(x["coupling"]["total_drive_power_w"] * x["coupling"]["dt_s"] for x in [first, second])
        close(errors["drive_energy"], defect(
            [probe["coarse_drive_energy_j"]], [energy],
            options.get("drive_energy_absolute_tolerance_j", 1.),
            options.get("drive_energy_relative_tolerance", .002)), atol=1e-9)
        if "well_bhp" in errors:
            bhp = {name: w["bhp_pa"] for name, w in second["surface"]["wells"].items()
                   if w.get("bhp_status") != "isolated_gauge"}
            coarse = probe["coarse_well_bhp_pa"]
            if set(coarse) != set(bhp):
                raise ValueError("Different temporal physical pressure status")
            if any(not math.isfinite(x) for x in [*coarse.values(), *bhp.values()]):
                raise ValueError("Nonfinite temporal BHP evidence")
            close(errors["well_bhp"], max((abs(coarse[n] - bhp[n]) for n in bhp), default=0.)
                  / options["well_bhp_tolerance_pa"], atol=1e-9)


def audit(config, result):
    def close(a, b, atol=1e-9, rtol=1e-10):
        if (
            not math.isfinite(a)
            or not math.isfinite(b)
            or not math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
        ):
            raise ValueError(f"Waterflood audit mismatch: {a} != {b}")

    spec = config["simulation"]["coupled_waterflood"]
    densities = spec["facility"]["stock_densities"]
    rho = [densities[k + "_kg_m3"] for k in ["oil", "water", "gas"]]
    wells = {w["name"]: w for w in config["wells"]}
    injection_spec = config["simulation"]["physical_surface_network"]["injection_network"]
    tree = WaterInjectionTree(injection_spec["network"])
    inventory = result["initial_component_mass_kg"]
    initial = list(inventory)
    inputs = [0.0] * 3
    outputs = [0.0] * 3
    clock = 0.0
    stages = result["accepted_substeps"]
    if not stages:
        raise ValueError("No accepted waterflood stages")
    _audit_temporal(config, stages, close)
    controls = {
        "injection_enabled": True,
        "pump_speed_ratios": {},
        "makeup_water_max_kg_s": spec.get("makeup_water_max_kg_s", 0.0),
        "facility_controls": {},
    }
    events = result["schedule"]
    index = 0
    maximum_residual = 0.0
    for step in stages:
        dt = step["coupling"]["dt_s"]
        end = step["time_days"] * 86400
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid accepted duration")
        close(end, clock + dt, atol=1e-7)
        maximum = config["simulation"].get("max_time_step_days", math.inf) * 86400
        if dt > maximum + 1e-7:
            raise ValueError("Accepted step exceeds maximum")
        while index < len(events) and events[index]["time_days"] * 86400 <= clock + 1e-7:
            controls.update(events[index]["controls"])
            index += 1
        if index < len(events) and events[index]["time_days"] * 86400 < end - 1e-7:
            raise ValueError("Accepted waterflood stage crossed an event")
        report = step["coupling"]
        if report.get("model") == "coupled_waterflood_v2":
            commit = report["dispatch_commit"]
            trace = report["trace"]
            selected = commit["solved_trace_index"]
            if type(selected) is not int or not 0 <= selected < len(trace):
                raise ValueError("Invalid solved waterflood state reference")
            chosen = trace[selected]
            if (chosen["feasible"] is not True
                    or chosen["production_fraction"] != report["production_fraction"]
                    or chosen["injection_fraction"] != report["injection_fraction"]):
                raise ValueError("Waterflood commit does not reference the chosen solved controls")
            options = spec.get("dispatch_solver", {})
            if commit["method"] != options.get("commit", "state_restore"):
                raise ValueError("Wrong waterflood commit policy")
            points = [(r["production_fraction"], r["injection_fraction"]) for r in trace]
            if commit["method"] == "state_restore" and len(set(points)) != len(points):
                raise ValueError("Repeated FIM capacity point in state-restore dispatch")
            if report["reservoir_evaluations"] != len(trace):
                raise ValueError("Inconsistent waterflood FIM evaluation count")
            for name, value in report["water_rates_kg_s"].items():
                close(value, chosen["water_rates_kg_s"][name])
            from simulator.fim_initial_cell_cache import audit_initial_cell_cache
            audit_initial_cell_cache(
                {**report, "evaluations": len(trace)},
                {"reuse_initial_cell_state": options.get("reuse_initial_cell_state", True),
                 "initial_cell_state_cache_bytes": options.get("initial_cell_state_cache_bytes", 64 * 1024**2)},
            )
        resolution = report.get("dispatch_resolution")
        if resolution is not None:
            for name in ["production", "injection"]:
                bracket = resolution[name]
                gap = max(
                    (
                        abs(bracket["low_bhp_pa"][well] - bracket["high_bhp_pa"][well])
                        for well in bracket["low_bhp_pa"].keys() & bracket["high_bhp_pa"].keys()
                    ),
                    default=0.0,
                )
                close(gap, bracket["maximum_common_bhp_gap_pa"], atol=1e-8)
                if (
                    gap > resolution["bhp_tolerance_pa"]
                    or bracket["high_fraction"] - bracket["low_fraction"]
                    > resolution["fraction_tolerance"]
                ):
                    raise ValueError("Unresolved waterflood dispatch bracket")
        if report["operator_controls"] != controls:
            raise ValueError("Wrong controls on accepted stage")
        if (
            not report["facility"]["feasible"]
            or report["facility"]["violations"]
            or not report["injection_network"]["feasible"]
            or report["injection_network"]["violations"]
        ):
            raise ValueError("Infeasible equipment state accepted")
        if any(not limit["passed"] for limit in report["facility"]["limits"]):
            raise ValueError("Facility limit violated")
        produced = [0.0] * 3
        injected = [0.0] * 3
        if set(step["surface"]["wells"]) != set(wells):
            raise ValueError("Different well registry")
        for name, row in step["surface"]["wells"].items():
            rates = row["component_mass_rates_kg_s"]
            if not math.isfinite(row["bhp_pa"]) or row["bhp_pa"] <= 0:
                raise ValueError("Invalid well pressure")
            for j, q in enumerate(rates):
                if not math.isfinite(q):
                    raise ValueError("Non-finite well rate")
                if wells[name]["type"] == "producer":
                    if q < -1e-10:
                        raise ValueError("Producer backflow in supported case")
                    produced[j] += q
                else:
                    if (j != 1 and abs(q) > 1e-10) or q > 1e-10:
                        raise ValueError("Non-water or reverse injection")
                    injected[j] -= q
            if wells[name]["type"] == "injector":
                maximum_bhp = wells[name].get("bhp_max")
                if (
                    maximum_bhp is not None
                    and -rates[1] > 1e-9
                    and row["bhp_pa"] > maximum_bhp * 1e6 + 10
                ):
                    raise ValueError("Injecting BHP exceeds declared maximum")
                close(row["standard_control_rate_m3_s"], -rates[1] / rho[1])
        water = report["water_rates_kg_s"]
        # Optional for historical trajectories. New reports expose the exact
        # incompressible column and each nonreturn completion's actual flux.
        injection = report["injection_network"]
        if "perforations" in injection:
            totals = {name: [0.0] * 3 for name in injection_spec["well_nodes"]}
            counts = dict.fromkeys(totals, 0)
            flags = []
            for perf in injection["perforations"]:
                name = perf["well"]
                row = injection["wells"][name]
                counts[name] += 1
                head = rho[1] * 9.80665 * (perf["depth_m"] - row["reference_depth_m"])
                close(head, perf["hydrostatic_head_pa"], atol=1e-6)
                close(row["bhp_pa"] + head, perf["well_pressure_pa"], atol=1e-6)
                if (
                    row["mode"] != "shut"
                    and perf["well_pressure_pa"] <= tree.fluid.vapor_pressure_pa
                ):
                    raise ValueError("Connected water column crosses vapor pressure")
                flags.append(perf["valve_open"])
                if perf["valve_open"] and (
                    row["mode"] == "shut"
                    or perf["well_pressure_pa"] < perf["reservoir_pressure_pa"] - 1e-6
                ):
                    raise ValueError("Open injection valve has reverse pressure or isolated well")
                for j, mass in enumerate(perf["component_mass_rates_kg_s"]):
                    if (j != 1 and abs(mass) > 1e-10) or mass > 1e-10:
                        raise ValueError("Invalid nonreturn perforation component flux")
                    if not perf["valve_open"]:
                        close(mass, 0.0)
                    totals[name][j] += mass
            if flags != injection["perforation_valves_open"]:
                raise ValueError("Inconsistent perforation valve registry")
            for name, total in totals.items():
                expected = len(wells[name].get("perforations", [None]))
                if counts[name] != expected:
                    raise ValueError("Incomplete injection perforation registry")
                for j in range(3):
                    close(total[j], injection["wells"][name]["component_mass_rates_kg_s"][j])
        for key, value in [
            ("produced", produced[1]),
            ("injected", injected[1]),
            ("recycled", min(produced[1], injected[1])),
            ("makeup", max(0, injected[1] - produced[1])),
            ("disposed", max(0, produced[1] - injected[1])),
        ]:
            close(water[key], value)
        if water["makeup"] > controls["makeup_water_max_kg_s"] + 1e-10:
            raise ValueError("Makeup availability exceeded")
        if not controls["injection_enabled"] and abs(injected[1]) > 1e-12:
            raise ValueError("Disabled injection still flows")
        close(report["injection_network"]["supply_mass_rate_kg_s"], injected[1])
        if controls["injection_enabled"]:
            by_leaf = {
                node: -step["surface"]["wells"][name]["component_mass_rates_kg_s"][1]
                for name, node in injection_spec["well_nodes"].items()
            }
            hydraulic = tree.evaluate(by_leaf, speed_ratios=controls["pump_speed_ratios"])
            if not hydraulic["feasible"]:
                raise ValueError("Recomputed injection equipment envelope failed")
            close(hydraulic["shaft_power_w"], report["injection_network"]["shaft_power_w"])
            for node, pressure in hydraulic["node_pressures_pa"].items():
                close(pressure, report["injection_network"]["node_pressures_pa"][node], atol=1e-5)
            for name, node in injection_spec["well_nodes"].items():
                row = step["surface"]["wells"][name]
                if row["mode"] in ("surface", "no_head", "bhp_limit"):
                    target = min(
                        hydraulic["node_pressures_pa"][node], wells[name].get("bhp_max", 1000) * 1e6
                    )
                    close(row["bhp_pa"], target, atol=10, rtol=0)
                elif (
                    row["mode"] == "rate"
                    and row["bhp_pa"] > hydraulic["node_pressures_pa"][node] + 10
                ):
                    raise ValueError("Rate-controlled injector requires unavailable pressure")
        plant = report["facility"]
        feed = [
            plant["feed"][key] * density
            for key, density in zip(["oil_m3_s", "water_m3_s", "gas_m3_s"], rho)
        ]
        for a, b in zip(feed, produced):
            close(a, b)
        products = [
            math.fsum(stream[key] * density for stream in plant["products"].values())
            for key, density in zip(["oil_m3_s", "water_m3_s", "gas_m3_s"], rho)
        ]
        for a, b in zip(products, produced):
            close(a, b)
        power = (
            plant["measurements"]["facility_drive_power_total_w"]["value"]
            + report["injection_network"]["shaft_power_w"]
        )
        close(power, report["total_drive_power_w"])
        if power > spec.get("max_total_drive_power_w", math.inf) + 1e-8:
            raise ValueError("Shared power limit exceeded")
        ledger = report["material_account"]
        incoming = [0, water["makeup"] * dt, 0]
        outgoing = [produced[0] * dt, water["disposed"] * dt, produced[2] * dt]
        for j in range(3):
            close(ledger["inventory_before_kg"][j], inventory[j], atol=1e-6)
            close(ledger["external_in_kg"][j], incoming[j], atol=1e-7)
            close(ledger["external_out_kg"][j], outgoing[j], atol=1e-7)
            residual = inventory[j] + incoming[j] - ledger["inventory_after_kg"][j] - outgoing[j]
            close(ledger["residual_kg"][j], residual, atol=1e-7)
            if abs(residual) > 0.01 + 1e-8 * max(incoming[j], outgoing[j]):
                raise ValueError("Per-stage component balance failed")
            maximum_residual = max(maximum_residual, abs(residual))
            inputs[j] = math.fsum((inputs[j], incoming[j]))
            outputs[j] = math.fsum((outputs[j], outgoing[j]))
        inventory = ledger["inventory_after_kg"]
        clock = end
    close(clock / 86400, result["requested_days"], atol=1e-12, rtol=0)
    if index != len(events):
        raise ValueError("Incomplete operating schedule")
    residual = []
    for j in range(3):
        close(inventory[j], result["final_component_mass_kg"][j], atol=1e-6)
        value = initial[j] + inputs[j] - inventory[j] - outputs[j]
        residual.append(value)
        if abs(value) > 0.01 + 1e-8 * max(inputs[j], outputs[j]):
            raise ValueError("Whole-trajectory component balance failed")
    return {
        "accepted": True,
        "days": clock / 86400,
        "accepted_stages": len(stages),
        "operating_events": len(events),
        "external_in_kg": inputs,
        "external_out_kg": outputs,
        "residual_kg": residual,
        "maximum_stage_component_residual_kg": maximum_residual,
        "scope": __doc__,
    }
