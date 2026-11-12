"""Accepted FIM packets plus independent capillary/open-gravity composition."""

import json
from copy import deepcopy

import numpy as np
from check_dynamic_receiver import audit_inlet_packets
from check_pr_capillary_surface import audit as plant_audit
from check_pr_capillary_surface import capillary_stocks, ordinary_stocks
from check_stratified_pr_flow import close

from surface.equilibrium_properties import EquilibriumFluid
from surface.operation import _digest
from surface.pr_capillary_facility import PRCapillaryFacilityState


def audit_capillary_step(spec, report):
    spec, report = json.loads(json.dumps([spec, report], allow_nan=False))
    plant = spec["receiver"]
    physical = plant["surface"]["graph"]["transport"]
    fluid = EquilibriumFluid.from_dict(physical["fluid"])
    inlet = next(n for n in physical["nodes"] if n["name"] == spec["inlet_vessel"])
    if spec["inlet_height_m"] != 0 or physical["reference_elevation_m"] != 0:
        raise ValueError("Capillary FIM requires an explicit absolute energy datum")
    old = report["initial_state"]
    feed = np.zeros(len(fluid.names))
    exported, stock = feed.copy(), np.zeros(3)
    ein = eout = heat = work = pmax = 0.0
    checks = []
    for stage in report["stages"]:
        dt, flow = stage["dt_s"], stage["flow"]
        expected = deepcopy(plant)
        opened = expected["surface"]
        opened["feeds"] = [
            {"name": f"fim_feed_{i}", "node": spec["inlet_vessel"], "stream": stream}
            for i, stream in enumerate(stage["inlet_streams"])
        ]
        for port in opened["pressure_ports"]:
            port["opening"] = report["outlet_openings"].get(port["name"], port["opening"])
            if port["allow_reverse"]:
                raise ValueError("Independent surface supply is not a FIM producer packet")
        if expected != flow["plant_model"]:
            raise ValueError("Accepted packet or physical plant differs")
        trial_old = {**old, "model_id": _digest(expected)}
        raw = flow["plant_report"]
        if raw["dt_s"] != dt:
            raise ValueError("Accepted plant clock differs")
        check = plant_audit(expected, trial_old, raw)
        # No unrequested heat, bypass sources or altered physical isolation.
        from surface.pr_gravity_transport import PRGravityTransportNetwork

        graph = PRGravityTransportNetwork.from_dict(opened["graph"])
        commands = {
            **graph.transport.default_openings,
            **{p.name: True for p in (*graph.ports, *graph.heat_links)},
            **report["connection_openings"],
        }
        if raw["openings"] != commands:
            raise ValueError("Plant isolation differs from accepted FIM command")
        if (
            raw["total_heat_j"] != 0
            or raw["capillary"]["phase_controls"]
            or raw["capillary"]["normal_relaxation"] is not True
        ):
            raise ValueError("Unrequested capillary FIM source")
        records = raw["surface"]["columns"].get(
            "material", raw["surface"]["transport"].get("coupling")
        )["ports"]

        def flux(rows, key, dt=dt, records=records):
            return dt * sum(
                (np.asarray(records[p["name"]][key]) for p in rows),
                np.zeros(len(fluid.names)) if key == "component_rates_mol_s" else 0.0,
            )

        dn = flux(opened["feeds"], "component_rates_mol_s")
        on = flux(opened["pressure_ports"], "component_rates_mol_s")
        de = flux(opened["feeds"], "total_energy_rate_w")
        oe = flux(opened["pressure_ports"], "total_energy_rate_w")
        for key, value in (
            ("feed_component_moles", dn),
            ("outlet_component_moles", on),
            ("feed_energy_j", de),
            ("outlet_energy_j", oe),
            ("ambient_heat_j", raw["total_heat_j"]),
            ("external_energy_j", raw["total_work_j"]),
            ("entropy_production_j_k", raw["entropy_production_j_k"]),
        ):
            close(flow[key], value, atol=1e-10, rtol=1e-9)
        stock += audit_inlet_packets(
            spec,
            stage,
            fluid,
            {"elevation_m": inlet["elevation_m"], "gravity_m_s2": physical["gravity_m_s2"]},
            0.0,
        )
        end = raw["state"]
        pressure = end["surface"]["transport"]["nodes"][spec["inlet_vessel"]]["pressure_pa"]
        trial = stage["pressure_trials"][-1]
        defect = pressure - trial["pressure_pa"]
        close(trial["receiver_pressure_pa"], pressure, atol=1e-8, rtol=1e-10)
        close(trial["pressure_defect_pa"], defect, atol=1e-8, rtol=1e-10)
        if not trial["physical_trial"] or abs(defect) > spec["control"][
            "pressure_tolerance_pa"
        ] + spec["control"]["pressure_relative_tolerance"] * max(pressure, trial["pressure_pa"]):
            raise ValueError("Capillary FIM end-pressure defect rejected")
        old = {**end, "model_id": _digest(plant)}
        feed += dn
        exported += on
        ein += de
        eout += oe
        heat += raw["total_heat_j"]
        work += raw["total_work_j"]
        pmax = max(pmax, abs(defect))
        checks.append(check)
    if _digest(old) != _digest(report["state"]):
        raise ValueError("Final capillary FIM checkpoint differs")

    def totals(state):
        cs = PRCapillaryFacilityState.from_dict(
            {
                "pipes": state["capillary_pipes"],
                "columns": {
                    n: state["surface"]["columns"][n] for n in plant["capillary"]["columns"]
                },
                "layer_slots": state["layer_slots"],
                "next_step_s": state["next_step_s"],
                "model_id": _digest(plant["capillary"]),
            }
        )
        n, e, _ = capillary_stocks(plant["capillary"], cs)
        gn, ge = ordinary_stocks(plant["surface"]["graph"], state["surface"])
        return n + gn, e + ge

    n0, e0 = totals(report["initial_state"])
    n1, e1 = totals(old)
    close(report["component_residuals_mol"], n1 - n0 - feed + exported, atol=1e-8, rtol=1e-9)
    close(report["energy_residual_j"], e1 - e0 - ein + eout - heat - work, atol=1e-7, rtol=1e-9)
    close(
        report["reservoir_residual_kg"],
        np.array(report["stages"][0]["reservoir_inventory_before_kg"])
        - np.array(report["stages"][-1]["reservoir_inventory_after_kg"])
        - stock,
        atol=1e-8,
        rtol=1e-9,
    )
    return {
        "accepted": True,
        "checked_stages": len(checks),
        "stage_audits": checks,
        "maximum_pressure_defect_pa": pmax,
        "feed_component_moles": feed.tolist(),
        "export_component_moles": exported.tolist(),
        "produced_stock_component_kg": stock.tolist(),
        "thermal_boundary_energy_j": ein,
        "export_energy_j": eout,
        "ambient_heat_j": heat,
        "surface_external_energy_j": work,
        "qualification": "Independent scalar capillary/gravity suboperators, shared-stock/work closure and accepted FIM packets; no independent reservoir solution or field calibration.",
    }
