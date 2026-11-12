"""Independent accepted well packets and open resolved gravity/pipe graph audit."""

import json
from copy import deepcopy

import numpy as np
from check_dynamic_receiver import audit_inlet_packets
from check_pr_open_gravity_transport import audit as gravity_audit
from check_stratified_pr_flow import close

from surface.equilibrium_properties import EquilibriumFluid
from surface.operation import _digest
from surface.pr_gravity_transport import PRGravityTransportNetwork


def audit_gravity_step(spec, report):
    spec, report = json.loads(json.dumps([spec, report], allow_nan=False))
    receiver = spec["receiver"]
    graph = receiver["graph"]
    physical = graph["transport"]
    fluid = EquilibriumFluid.from_dict(physical["fluid"])
    inlet = next(v for v in physical["nodes"] if v["name"] == spec["inlet_vessel"])
    if spec["inlet_height_m"] != 0 or physical["reference_elevation_m"] != 0:
        raise ValueError("Resolved FIM graph requires the explicit absolute energy datum")
    parsed = PRGravityTransportNetwork.from_dict(graph)
    old = report["initial_state"]
    feed = np.zeros(len(fluid.names))
    export = feed.copy()
    stock = np.zeros(3)
    ein = eout = ambient = external = pmax = 0.0
    checks = []
    for stage in report["stages"]:
        dt, flow = stage["dt_s"], stage["flow"]
        expected = deepcopy(receiver)
        expected["feeds"] = [
            {"name": f"fim_feed_{i}", "node": spec["inlet_vessel"], "stream": s}
            for i, s in enumerate(stage["inlet_streams"])
        ]
        for p in expected["pressure_ports"]:
            p["opening"] = report["outlet_openings"].get(p["name"], p["opening"])
            if p["allow_reverse"]:
                raise ValueError("FIM export cannot supply independent reservoir material")
        if expected != flow["gravity_model"]:
            raise ValueError("Actual well packets or resolved graph differ")
        raw = flow["gravity_report"]
        end = raw["state"]
        records = raw["columns"].get("material", raw["transport"].get("coupling"))["ports"]
        dn = dt * sum(
            (np.array(records[p["name"]]["component_rates_mol_s"]) for p in expected["feeds"]),
            np.zeros(len(fluid.names)),
        )
        on = dt * sum(
            (
                np.array(records[p["name"]]["component_rates_mol_s"])
                for p in expected["pressure_ports"]
            ),
            np.zeros(len(fluid.names)),
        )
        de = dt * sum(records[p["name"]]["total_energy_rate_w"] for p in expected["feeds"])
        oe = dt * sum(records[p["name"]]["total_energy_rate_w"] for p in expected["pressure_ports"])
        commands = {
            **parsed.transport.default_openings,
            **{p.name: True for p in (*parsed.ports, *parsed.heat_links)},
            **report["connection_openings"],
        }
        interval = {
            "state": end,
            "steps": [raw],
            "openings": commands,
            "loss_coefficients": {},
            "total_heat_j": raw["total_heat_j"],
            "total_work_j": raw["total_work_j"],
            "conservation_residuals": raw["conservation_residuals"],
            "material_component_moles": raw["material_component_moles"],
            "material_energy_j": raw["material_energy_j"],
            "open_model_id": _digest(expected),
            "port_openings": {p["name"]: p["opening"] for p in expected["pressure_ports"]},
            "boundary_component_moles": (dn - on).tolist(),
            "boundary_energy_j": de - oe,
        }
        check = gravity_audit(expected, old, interval, openings=report["connection_openings"])
        for key, value in [
            ("feed_component_moles", dn),
            ("outlet_component_moles", on),
            ("feed_energy_j", de),
            ("outlet_energy_j", oe),
            ("ambient_heat_j", raw["total_heat_j"]),
            ("external_energy_j", raw["total_work_j"]),
        ]:
            close(flow[key], value, atol=1e-8, rtol=1e-9)
        st = audit_inlet_packets(
            spec,
            stage,
            fluid,
            {"elevation_m": inlet["elevation_m"], "gravity_m_s2": physical["gravity_m_s2"]},
            0.0,
        )
        pressure = end["transport"]["nodes"][spec["inlet_vessel"]]["pressure_pa"]
        trial = stage["pressure_trials"][-1]
        if not trial["physical_trial"]:
            raise ValueError("Rejected pressure trial published")
        defect = pressure - trial["pressure_pa"]
        close(trial["receiver_pressure_pa"], pressure, atol=1e-8, rtol=1e-10)
        close(trial["pressure_defect_pa"], defect, atol=1e-8, rtol=1e-10)
        if abs(defect) > spec["control"]["pressure_tolerance_pa"] + spec["control"][
            "pressure_relative_tolerance"
        ] * max(pressure, trial["pressure_pa"]):
            raise ValueError("Resolved graph end-pressure defect rejected")
        feed += dn
        export += on
        stock += st
        ein += de
        eout += oe
        ambient += raw["total_heat_j"]
        external += raw["total_work_j"]
        pmax = max(pmax, abs(defect))
        old = end
        checks.append(check)
    if _digest(old) != _digest(report["state"]):
        raise ValueError("Resolved final checkpoint differs")

    def totals(state):
        n = np.sum(
            [
                v["component_moles"]
                for v in [*state["transport"]["nodes"].values(), *state["columns"].values()]
            ],
            axis=0,
        )
        e = sum(v["total_energy_j"] for v in state["transport"]["nodes"].values()) + sum(
            v["total_energy_j"] for v in state["columns"].values()
        )
        for node in physical["nodes"]:
            if node["wall"]:
                e += (
                    node["wall"]["heat_capacity_j_k"]
                    * state["transport"]["nodes"][node["name"]]["wall_temperature_k"]
                )
        e += sum(
            w["heat_capacity_j_k"] * state["walls"][name] for name, w in graph["walls"].items()
        )
        from reference_pr_drives import rotor_energy

        return n, e + rotor_energy(state["transport"], physical)

    na, ea = totals(report["initial_state"])
    nb, eb = totals(old)
    close(report["component_residuals_mol"], nb - na - feed + export, atol=1e-8, rtol=1e-9)
    close(
        report["energy_residual_j"], eb - ea - ein + eout - ambient - external, atol=1e-7, rtol=1e-9
    )
    a = np.array(report["stages"][0]["reservoir_inventory_before_kg"])
    b = np.array(report["stages"][-1]["reservoir_inventory_after_kg"])
    close(report["reservoir_residual_kg"], a - b - stock, atol=1e-8, rtol=1e-9)
    return {
        "accepted": True,
        "checked_stages": len(checks),
        "stage_audits": checks,
        "maximum_pressure_defect_pa": pmax,
        "feed_component_moles": feed.tolist(),
        "export_component_moles": export.tolist(),
        "produced_stock_component_kg": stock.tolist(),
        "thermal_boundary_energy_j": ein,
        "export_energy_j": eout,
        "ambient_heat_j": ambient,
        "surface_external_energy_j": external,
        "qualification": "Scalar profiles, finite apertures, internal/external material and equipment balances plus accepted FIM packets. No independent reservoir solution, field calibration or capillary-interface validation.",
    }
