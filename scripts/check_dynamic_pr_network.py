"""Independent open-network and accepted-FIM boundary audit, scalar PR."""

import math
from copy import deepcopy

import numpy as np
from check_stratified_pr_flow import close, profile
from check_stratified_pr_network import audit as network_audit

from surface.operation import _digest


def audit_network_step(spec, report):
    from check_dynamic_receiver import audit_inlet_packets

    base = spec["receiver"]
    old = report["initial_state"]
    ports = {n + "/" + p["name"] for n, v in base["vessels"].items() for p in v["outlets"]}
    links = {p["name"] for p in base["connections"]}
    if set(report["outlet_openings"]) - ports or set(report["connection_openings"]) - links:
        raise ValueError("Unknown graph control in report")
    inlet = spec["inlet_vessel"]
    system = base["vessels"][inlet]["column"]
    height = spec["inlet_height_m"]
    checks = []
    incoming = np.zeros(len(spec["assay"]["component_names"]))
    outgoing = incoming.copy()
    stock = np.zeros(3)
    ein = eout = ambient = duration = pmax = 0.0
    for stage in report["stages"]:
        dt, flow = stage["dt_s"], stage["flow"]
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid network coupled substep")
        expected = deepcopy(base)
        for name, v in expected["vessels"].items():
            v["feeds"] = (
                [{"height_m": height, "stream": s} for s in stage["inlet_streams"]]
                if name == inlet
                else []
            )
            for port in v["outlets"]:
                port["opening"] = report["outlet_openings"].get(
                    name + "/" + port["name"], port["opening"]
                )
        if _digest(flow["network_model"]) != _digest(expected):
            raise ValueError("Network trial changed physical model or accepted feeds")
        raw = flow["network_report"]
        cfg = {
            "vessels": {n: {"model": v} for n, v in expected["vessels"].items()},
            "connections": expected["connections"],
            "schedule": [{"duration_s": dt, "openings": report["connection_openings"]}],
        }
        checked = network_audit(
            cfg,
            {
                "config": cfg,
                "initial": old,
                "intervals": [
                    {
                        "state": raw["state"],
                        "steps": [raw],
                        **{
                            k: raw[k]
                            for k in (
                                "component_residuals_mol",
                                "energy_residual_j",
                                "entropy_production_j_k",
                                "ambient_heat_j",
                                "feed_energy_j",
                                "outlet_energy_j",
                            )
                        },
                    }
                ],
            },
        )
        if raw["state"]["model_id"] != _digest(expected):
            raise ValueError("Trial graph state identity differs")
        final = deepcopy(raw["state"])
        final["model_id"] = _digest(base)
        for name, s in final["vessels"].items():
            if s["column"]["model_id"] != _digest(
                {"model": "stratified_pr_uv_v1", **base["vessels"][name]["column"]}
            ):
                raise ValueError("Trial column identity differs")
            if s["model_id"] != _digest(expected["vessels"][name]):
                raise ValueError("Trial vessel identity differs")
            s["model_id"] = _digest(base["vessels"][name])
        f, _, sample, *_ = profile(final["vessels"][inlet]["column"], system)
        phase = np.searchsorted(
            final["vessels"][inlet]["column"]["interface_heights_m"], height, side="left"
        )
        pressure = sample(height, phase)[0]
        trial = stage["pressure_trials"][-1]
        defect = pressure - trial["pressure_pa"]
        tol = spec["control"]["pressure_tolerance_pa"] + spec["control"][
            "pressure_relative_tolerance"
        ] * max(pressure, trial["pressure_pa"])
        if not trial["physical_trial"] or abs(defect) > tol:
            raise ValueError("Independent network END-pressure closure rejected")
        close(trial["receiver_pressure_pa"], pressure, atol=1e-4, rtol=1e-10)
        pmax = max(pmax, abs(defect))
        stock += audit_inlet_packets(spec, stage, f, system, height)
        for key in (
            "feed_component_moles",
            "outlet_component_moles",
            "feed_energy_j",
            "outlet_energy_j",
            "ambient_heat_j",
        ):
            close(flow[key], raw[key], atol=1e-8, rtol=1e-9)
        incoming += flow["feed_component_moles"]
        outgoing += flow["outlet_component_moles"]
        ein += flow["feed_energy_j"]
        eout += flow["outlet_energy_j"]
        ambient += flow["ambient_heat_j"]
        duration += dt
        checks.append(checked)
        old = final
    if _digest(old) != _digest(report["state"]):
        raise ValueError("Accepted graph checkpoint differs from actual network solve")
    close(duration, report["dt_s"], atol=1e-12, rtol=1e-12)
    return {
        "accepted": True,
        "checked_stages": len(checks),
        "stage_audits": checks,
        "maximum_pressure_defect_pa": pmax,
        "feed_component_moles": incoming.tolist(),
        "export_component_moles": outgoing.tolist(),
        "produced_stock_component_kg": stock.tolist(),
        "thermal_boundary_energy_j": ein,
        "export_energy_j": eout,
        "ambient_heat_j": ambient,
        "qualification": "Scalar PR profiles, 32-node internal/external finite apertures, every vessel balance and passive accepted-FIM streams. No independent reservoir solve, global phase-stability or experimental qualification.",
    }
