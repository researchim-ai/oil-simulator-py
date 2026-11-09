"""Independent active-phase finite-volume audit with 64-node path quadrature."""

import json
import math

import numpy as np
from check_pr_thermochemical_relaxation import audit as audit_source
from check_pr_thermochemical_relaxation import reference_limits
from check_stratified_pr_flow import close
from reference_pr_active_multifluid import packets, reconstruct

from surface.operation import _digest
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe


def totals(state, nc):
    phases = [p for cell in state["cells"] for p in cell.values()]
    return np.r_[
        [math.fsum(p["component_moles"][j] for p in phases) for j in range(nc)],
        math.fsum(p["total_energy_j"] for p in phases),
        math.fsum(p["momentum_kg_m_s"] for p in phases),
    ]


def audit_extended(model, config, report):
    from check_pr_active_surface import pipe_audit

    from surface.pr_active_multifluid_pipe import PRActiveMultifluidState

    mw = np.array([c.molar_mass_kg_mol for c in model.fluid.components])

    def stored(s):
        value = totals(s, len(mw))
        value[-2] += math.fsum(
            model.gravity * z * (np.array(p["component_moles"]) @ mw)
            for z, cell in zip(model.centers, s["cells"])
            for p in cell.values()
        )
        return value

    initial = old = report["initial"]
    impulse = 0.0
    total_heat = 0.0
    checked = 0
    maximum_energy = 0.0
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        if (
            set(command)
            - {
                "duration_s",
                "maximum_step_s",
                "minimum_step_s",
                "maximum_attempts",
                "heat_rates_w",
                "heat_temperatures_k",
            }
            or not interval["steps"]
        ):
            raise ValueError("Invalid extended pipe schedule")
        start = old
        interval_impulse = 0.0
        interval_heat = 0.0
        for r in interval["steps"]:
            from reference_pr_active_heat import check_commands

            check_commands(r, command.get("heat_rates_w"), command.get("heat_temperatures_k"))
            new = r["state"]
            if (
                not 0
                < r["dt_s"]
                <= min(model.maximum_step_s, command.get("maximum_step_s", model.maximum_step_s))
                * (1 + 1e-12)
            ):
                raise ValueError("Extended pipe CFL violated")
            close(
                new["time_s"] - old["time_s"], r["dt_s"], atol=8 * math.ulp(new["time_s"]), rtol=0.0
            )
            pipe_audit(
                model,
                PRActiveMultifluidState.from_dict(old),
                PRActiveMultifluidState.from_dict(new),
                r,
            )
            transfer = r["wall_impulse_kg_m_s"] + r["gravity_impulse_kg_m_s"]
            residual = stored(new) - stored(old)
            residual[-1] -= transfer
            residual[-2] -= r.get("external_heat_j", 0.0)
            close(residual, np.zeros(len(mw) + 2), atol=1e-6, rtol=0.0)
            close(r["conservation_residuals"], residual, atol=1e-7, rtol=0.0)
            maximum_energy = max(maximum_energy, abs(residual[-2]))
            interval_impulse += transfer
            interval_heat += r.get("external_heat_j", 0.0)
            old = new
            checked += 1
        if interval["state"] != old:
            raise ValueError("Extended pipe published interval differs")
        close(
            old["time_s"],
            start["time_s"] + command["duration_s"],
            atol=8 * math.ulp(old["time_s"]),
            rtol=0.0,
        )
        residual = stored(old) - stored(start)
        residual[-1] -= interval_impulse
        residual[-2] -= interval_heat
        close(interval["conservation_residuals"], residual, atol=1e-6, rtol=0.0)
        impulse += interval_impulse
        total_heat += interval_heat
    if old != report["state"]:
        raise ValueError("Extended pipe final state differs")
    residual = stored(old) - stored(initial)
    residual[-1] -= impulse
    residual[-2] -= total_heat
    close(report["conservation_residuals"], residual, atol=1e-6, rtol=0.0)
    return {
        "accepted": True,
        "checked_substeps": checked,
        "maximum_energy_residual_j": maximum_energy,
        "active_phase_counts": [len(c) for c in old["cells"]],
    }


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Active pipe request differs")
    model = PRActiveMultifluidPipe.from_dict(config["model"])
    normalized = json.loads(json.dumps(model.to_dict()))
    if normalized != report["model"]:
        raise ValueError("Active pipe normalized model differs")
    nc = len(model.fluid.names)
    initial = old = report["initial"]
    packets(model, old)
    if config["initial"] != {"state": old} and (
        set(config["initial"]) != {"cells"}
        or config["initial"]["cells"] != old["cells"]
        or old["time_s"] != 0.0
    ):
        raise ValueError("Active initial stocks differ")
    if len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Active schedule count differs")
    if (
        model.elevations is not None
        or model.friction is not None
        or any("heat_rates_w" in c or "heat_temperatures_k" in c for c in config["schedule"])
    ):
        return audit_extended(model, config, report)
    checked = 0
    impulse = 0.0
    maximum_energy = 0.0

    def limits(a, b):
        x = {"phases": [p for cell in a["cells"] for p in cell.values()]}
        y = {"phases": [p for cell in b["cells"] for p in cell.values()]}
        full = reference_limits(x, y)
        return np.r_[full[:nc], full[-2:]]

    atol = limits(initial, initial)
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        if (
            set(command) - {"duration_s", "maximum_step_s", "minimum_step_s", "maximum_attempts"}
            or not interval["steps"]
        ):
            raise ValueError("Invalid active pipe command")
        start = old
        wall_interval = 0.0
        maximum = min(model.maximum_step_s, command.get("maximum_step_s", model.maximum_step_s))
        for step in interval["steps"]:
            dt = step["dt_s"]
            new = step["state"]
            transport = step["transport_state"]
            if not math.isfinite(dt) or not 0 < dt <= maximum * (1 + 1e-12):
                raise ValueError("Active CFL violated")
            close(
                new["time_s"] - old["time_s"],
                dt,
                atol=8 * max(math.ulp(new["time_s"]), math.ulp(old["time_s"])),
                rtol=0.0,
            )
            if transport["time_s"] != new["time_s"]:
                raise ValueError("Split source clock differs")
            a, delta, adelta, faces = reconstruct(model, old)
            b = packets(model, transport)
            c = packets(model, new)
            if step["path_quadrature_order"] != 8:
                raise ValueError("Unknown active path quadrature")
            for key in ("flux", "phase_path", "alpha_path"):
                saved = "face_fluxes" if key == "flux" else key
                close(
                    step[saved],
                    np.array([v[key] for v in faces]),
                    atol=1e-10 if key == "alpha_path" else 1e-6,
                    rtol=1e-11,
                )
            if step["contact_faces"] != [v["contact"] for v in faces]:
                raise ValueError("Contact recognition differs")
            predicted = (
                a["conserved"] - np.array([dt / l for l in model.lengths])[:, None, None] * delta
            )
            predicted_alpha = (
                a["alpha"] - np.array([dt / l for l in model.lengths])[:, None] * adelta
            )
            for i, volume in enumerate(model.volumes):
                expected_names = tuple(
                    n
                    for j, n in enumerate(model.names)
                    if predicted[i, j, :nc].sum() > 0 and predicted_alpha[i, j] > 0
                )
                if tuple(transport["cells"][i]) != expected_names:
                    raise ValueError("Phase floor, phantom phase or dropped transported inventory")
                close(
                    b["conserved"][i] * volume,
                    predicted[i] * volume,
                    atol=np.r_[[1e-10] * nc, 1e-10, 1e-8],
                    rtol=1e-8,
                )
                close(b["alpha"][i], predicted_alpha[i], atol=1e-12, rtol=1e-9)
            wall = (
                0.0
                if model.boundary_mode == "periodic"
                else dt
                * model.area
                * (faces[0]["flux"][:, -2].sum() - faces[-1]["flux"][:, -2].sum())
            )
            close(step["wall_impulse_kg_m_s"], wall, atol=1e-11, rtol=1e-8)
            if b["entropy"] - a["entropy"] < -max(
                1e-7, 64 * np.finfo(float).eps * (abs(b["entropy"]) + abs(a["entropy"]))
            ):
                raise ValueError("Transport entropy loss was masked by a source")
            if model.chemical is None:
                if step["sources"] or transport != new:
                    raise ValueError("Undeclared active thermochemical source")
            else:
                if len(step["sources"]) != len(model.volumes):
                    raise ValueError("Missing cell sources")
                source_model = model.chemical.to_dict()
                for i, source_report in enumerate(step["sources"]):
                    cell = transport["cells"][i]
                    before = {
                        "phases": list(cell.values()),
                        "phase_roots": [model.slots[model.names.index(n)]["root"] for n in cell],
                        "time_s": old["time_s"],
                        "model_id": _digest(source_model),
                    }
                    after = source_report["state"]
                    audit_source(source_model, before, after, source_report)
                    source_phases = sorted(json.dumps(p, sort_keys=True) for p in after["phases"])
                    final_phases = sorted(
                        json.dumps(p, sort_keys=True) for p in new["cells"][i].values()
                    )
                    if source_phases != final_phases:
                        raise ValueError(
                            "Source-to-slot mapping changed or dropped physical inventories"
                        )
            close(
                [step["transport_entropy_production_j_k"], step["entropy_production_j_k"]],
                [b["entropy"] - a["entropy"], c["entropy"] - a["entropy"]],
                atol=1e-7,
                rtol=1e-8,
            )
            if c["entropy"] - a["entropy"] < -max(
                1e-7, 64 * np.finfo(float).eps * (abs(c["entropy"]) + abs(a["entropy"]))
            ):
                raise ValueError("Active total entropy balance failed")
            if step["active_phase_counts"] != [len(cell) for cell in new["cells"]]:
                raise ValueError("Active phase diagnostics differ")
            residual = totals(new, nc) - totals(old, nc)
            residual[-1] -= wall
            close(residual, np.zeros(nc + 2), atol=limits(old, new), rtol=0.0)
            close(
                step["conservation_residuals"],
                residual,
                atol=np.r_[[1e-11] * nc, 1e-8, 1e-11],
                rtol=0.0,
            )
            maximum_energy = max(maximum_energy, abs(residual[-2]))
            old = new
            checked += 1
            wall_interval += wall
        if interval["state"] != old:
            raise ValueError("Active published interval differs")
        close(
            old["time_s"],
            start["time_s"] + command["duration_s"],
            atol=8 * math.ulp(old["time_s"]),
            rtol=0.0,
        )
        residual = totals(old, nc) - totals(start, nc)
        residual[-1] -= wall_interval
        close(interval["conservation_residuals"], residual, atol=atol, rtol=0.0)
        impulse += wall_interval
    if old != report["state"]:
        raise ValueError("Active final checkpoint differs")
    residual = totals(old, nc) - totals(initial, nc)
    residual[-1] -= impulse
    close(residual, np.zeros(nc + 2), atol=limits(initial, old), rtol=0.0)
    close(report["conservation_residuals"], residual, atol=atol, rtol=0.0)
    return {
        "accepted": True,
        "checked_substeps": checked,
        "maximum_energy_residual_j": maximum_energy,
        "active_phase_counts": [len(cell) for cell in old["cells"]],
    }


if __name__ == "__main__":
    from check_pr_thermochemical_relaxation import main

    main()
