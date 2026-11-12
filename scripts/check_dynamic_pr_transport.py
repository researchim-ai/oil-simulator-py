"""Accepted well packets and independent open distributed surface audit."""

import json
from copy import deepcopy

import numpy as np
from check_dynamic_receiver import audit_inlet_packets
from check_pr_equilibrium_transport import ledger
from check_pr_open_transport import audit as open_audit
from check_stratified_pr_flow import close

from surface.equilibrium_properties import EquilibriumFluid
from surface.operation import _digest


def audit_transport_step(spec, report):
    spec, report = json.loads(json.dumps([spec, report], allow_nan=False))
    receiver = spec["receiver"]
    old = report["initial_state"]
    physical = receiver["transport"]
    fluid = EquilibriumFluid.from_dict(physical["fluid"])
    inlet = next(v for v in physical["nodes"] if v["name"] == spec["inlet_vessel"])
    if spec["inlet_height_m"] != 0 or physical["reference_elevation_m"] != 0:
        raise ValueError("Distributed well coupling uses an absolute stirred boundary")
    checks, feed, export, stock = (
        [],
        np.zeros(len(fluid.names)),
        np.zeros(len(fluid.names)),
        np.zeros(3),
    )
    ein = eout = ambient = external = pmax = 0.0
    for stage in report["stages"]:
        dt, flow = stage["dt_s"], stage["flow"]
        expected = deepcopy(receiver)
        expected["feeds"] = [
            {"name": f"fim_feed_{i}", "node": spec["inlet_vessel"], "stream": s}
            for i, s in enumerate(stage["inlet_streams"])
        ]
        for port in expected["pressure_ports"]:
            port["opening"] = report["outlet_openings"].get(port["name"], port["opening"])
            if port["allow_reverse"]:
                raise ValueError("Coupled distributed exports must be one-way")
        if expected != flow["transport_model"]:
            raise ValueError("Actual FIM packet or distributed boundary model differs")
        raw = flow["transport_report"]
        end = raw["state"]
        # Native defaults also include pump/compressor/restriction states.
        from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork

        t = PREquilibriumTransportNetwork.from_dict(physical)
        opens = {**t.default_openings, **report["connection_openings"]}
        losses = {
            v["name"]: v.get("local_loss_coefficient", 0.0)
            for v in physical["nodes"]
            if v["kind"] == "pipe_cell"
        }
        records = raw["coupling"]["ports"]
        dn = dt * sum(
            (
                np.asarray(records[f"fim_feed_{i}"]["component_rates_mol_s"])
                for i in range(len(stage["inlet_streams"]))
            ),
            np.zeros(len(fluid.names)),
        )
        de = dt * sum(
            records[f"fim_feed_{i}"]["total_energy_rate_w"]
            for i in range(len(stage["inlet_streams"]))
        )
        on = dt * sum(
            (
                np.asarray(records[p["name"]]["component_rates_mol_s"])
                for p in expected["pressure_ports"]
            ),
            np.zeros(len(fluid.names)),
        )
        oe = dt * sum(records[p["name"]]["total_energy_rate_w"] for p in expected["pressure_ports"])
        interval = {
            "state": end,
            "steps": [raw],
            "openings": opens,
            "loss_coefficients": losses,
            "open_model_id": _digest(expected),
            "port_openings": {p["name"]: p["opening"] for p in expected["pressure_ports"]},
            "total_heat_j": raw["ambient_heat_j"] + raw["external_heat_j"],
            "total_work_j": raw["external_work_j"],
            "conservation_residuals": raw["conservation_residuals"],
            "entropy_production_j_k": raw["entropy_production_j_k"],
            "boundary_component_moles": (dn - on).tolist(),
            "boundary_energy_j": de - oe,
        }
        check = open_audit(expected, old, interval, openings=report["connection_openings"])
        for key, value in [
            ("feed_component_moles", dn),
            ("outlet_component_moles", on),
            ("feed_energy_j", de),
            ("outlet_energy_j", oe),
            ("ambient_heat_j", raw["ambient_heat_j"]),
            ("external_energy_j", raw["external_heat_j"] + raw["external_work_j"]),
        ]:
            close(flow[key], value, atol=1e-8, rtol=1e-9)
        st = audit_inlet_packets(
            spec,
            stage,
            fluid,
            {"elevation_m": inlet["elevation_m"], "gravity_m_s2": physical["gravity_m_s2"]},
            0.0,
        )
        pressure = end["nodes"][spec["inlet_vessel"]]["pressure_pa"]
        trial = stage["pressure_trials"][-1]
        if not trial["physical_trial"]:
            raise ValueError("Unphysical pressure trial published")
        close(trial["receiver_pressure_pa"], pressure, atol=1e-8, rtol=1e-10)
        defect = pressure - trial["pressure_pa"]
        close(trial["pressure_defect_pa"], defect, atol=1e-8, rtol=1e-10)
        if abs(defect) > spec["control"]["pressure_tolerance_pa"] + spec["control"][
            "pressure_relative_tolerance"
        ] * max(pressure, trial["pressure_pa"]):
            raise ValueError("Distributed pressure feedback not converged")
        pmax = max(pmax, abs(defect))
        feed += dn
        export += on
        stock += st
        ein += de
        eout += oe
        ambient += raw["ambient_heat_j"]
        external += flow["external_energy_j"]
        old = end
        checks.append(check)
    if _digest(old) != _digest(report["state"]):
        raise ValueError("Final distributed checkpoint differs")
    before = report["initial_state"]
    nodes = {n["name"]: n for n in physical["nodes"]}
    residual = ledger(
        before, old, nodes, ein - eout + ambient + external, physical, incoming_n=feed - export
    )
    close(report["component_residuals_mol"], residual[:-1], atol=1e-8, rtol=1e-9)
    close(report["energy_residual_j"], residual[-1], atol=1e-7, rtol=1e-9)
    a = np.asarray(report["stages"][0]["reservoir_inventory_before_kg"])
    b = np.asarray(report["stages"][-1]["reservoir_inventory_after_kg"])
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
        "qualification": "Independent scalar open PR pipe/equipment faces, local N/M/E/S and accepted well packets; not an independent reservoir solve or field characterization.",
    }
