"""Independent local phase inertia, pressure/heat work and entropy audit."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from check_stratified_pr_flow import close
from reference_pr_interphase_relaxation import decode, momentum

from surface.pr_interphase_relaxation import PRInterphaseRelaxation


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Relaxation request differs")
    parsed = PRInterphaseRelaxation.from_dict(config["model"])
    model = json.loads(json.dumps(parsed.to_dict()))
    if model != report["model"] or len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Relaxation model/schedule differs")
    roots, fluid = model["phase_roots"], parsed.fluid
    nph = len(roots)
    pressure = np.array(model["pressure_conductance_m3_pa_s"])
    thermal = np.array(model["heat_conductance_w_k"])
    pressure_edges = [
        (i, j, pressure[i, j]) for i in range(nph) for j in range(i + 1, nph) if pressure[i, j] > 0
    ]
    heat_edges = [
        (i, j, thermal[i, j]) for i in range(nph) for j in range(i + 1, nph) if thermal[i, j] > 0
    ]

    def metadata(state):
        if (
            set(state) != {"phases", "time_s", "model_id", "next_step_s"}
            or state["model_id"] != parsed.model_id
        ):
            raise ValueError("Relaxation checkpoint model/schema differs")
        if (
            not math.isfinite(state["time_s"])
            or state["time_s"] < 0
            or not math.isfinite(state["next_step_s"])
            or state["next_step_s"] <= 0
        ):
            raise ValueError("Invalid relaxation clock/step")
        return decode(fluid, state, roots)

    def ledger(a, b, impulse):
        nn = len(fluid.names)
        return [
            math.fsum(s["component_moles"][i] for s in b["phases"])
            - math.fsum(s["component_moles"][i] for s in a["phases"])
            for i in range(nn)
        ] + [
            math.fsum(s["volume_m3"] for s in b["phases"])
            - math.fsum(s["volume_m3"] for s in a["phases"]),
            math.fsum(s["total_energy_j"] for s in b["phases"])
            - math.fsum(s["total_energy_j"] for s in a["phases"]),
            math.fsum(s["momentum_kg_m_s"] for s in b["phases"])
            - math.fsum(s["momentum_kg_m_s"] for s in a["phases"])
            - impulse,
        ]

    ledger_atol = np.array([1e-12] * len(fluid.names) + [1e-14, 1e-9, 1e-12])

    old = report["initial"]
    data = metadata(old)
    if set(config["initial"]) == {"state"}:
        if old != config["initial"]["state"]:
            raise ValueError("Requested phase checkpoint differs")
    elif set(config["initial"]) == {"phases"}:
        if old["time_s"] != 0 or len(config["initial"]["phases"]) != nph:
            raise ValueError("Cold phase registry/clock differs")
        for spec, stock, d in zip(config["initial"]["phases"], old["phases"], data, strict=True):
            if set(spec) != {
                "pressure_pa",
                "temperature_k",
                "volume_m3",
                "mole_fractions",
                "velocity_m_s",
            }:
                raise ValueError("Unknown/missing phase initialization fields")
            close(d["p"], spec["pressure_pa"], atol=1e-6, rtol=1e-12)
            close(d["t"], spec["temperature_k"], atol=1e-8, rtol=0)
            close(d["velocity"], spec["velocity_m_s"], atol=1e-12, rtol=1e-12)
            close(d["volume"], spec["volume_m3"], atol=1e-14, rtol=1e-10)
            close(d["n"] / d["n"].sum(), spec["mole_fractions"], atol=1e-12, rtol=1e-12)
    else:
        raise ValueError("Unknown phase initial request")
    first = old
    total_impulse = 0.0
    checked, max_e, max_p = 0, 0.0, 0.0
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        if set(command) - {
            "duration_s",
            "relative_tolerance",
            "velocity_tolerance_m_s",
            "maximum_step_s",
            "minimum_step_s",
            "maximum_attempts",
        }:
            raise ValueError("Unknown relaxation schedule command")
        start_state = old
        interval_impulse = 0.0
        start = old["time_s"]
        if not interval["steps"]:
            raise ValueError("Missing accepted relaxation steps")
        for step in interval["steps"]:
            new, dt = step["state"], step["dt_s"]
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Invalid relaxation timestep")
            close(
                new["time_s"] - old["time_s"],
                dt,
                atol=8 * max(math.ulp(new["time_s"]), math.ulp(old["time_s"])),
                rtol=0,
            )
            final = metadata(new)
            mechanical = momentum(
                [d["mass"] for d in data],
                [d["velocity"] for d in data],
                dt,
                model["pair_drag_kg_s"],
                model["wall_drag_kg_s"],
            )
            close([d["velocity"] for d in final], mechanical["velocity"], atol=2e-11, rtol=1e-10)
            close(
                step["velocity_coordinates_m_s"], mechanical["coordinates"], atol=2e-11, rtol=1e-10
            )
            close(step["pair_impulses_kg_m_s"], mechanical["pair_impulses"], atol=1e-11, rtol=1e-9)
            close(step["physical_dissipation_j"], mechanical["physical"], atol=1e-10, rtol=1e-8)
            close(step["temporal_dissipation_j"], mechanical["numerical"], atol=1e-10, rtol=1e-8)
            dv, work, heat = [np.zeros(nph) for _ in range(3)]
            vp, hp = step["volume_transfers_m3"], step["heat_transfers_j"]
            if len(vp) != len(pressure_edges) or len(hp) != len(heat_edges):
                raise ValueError("Exchange impulse registry differs")
            sp = sh = 0.0
            for amount, (i, j, c) in zip(vp, pressure_edges, strict=True):
                delta = final[i]["p"] - final[j]["p"]
                close(
                    delta - amount / (dt * c),
                    0.0,
                    atol=1e-5 + 1e-9 * max(final[i]["p"], final[j]["p"]),
                    rtol=0,
                )
                dv[i] += amount
                dv[j] -= amount
                ww = 0.5 * (final[i]["p"] + final[j]["p"]) * amount
                work[i] += ww
                work[j] -= ww
                sp += 0.5 * delta * amount * (1 / final[i]["t"] + 1 / final[j]["t"])
            for amount, (i, j, c) in zip(hp, heat_edges, strict=True):
                close((final[j]["t"] - final[i]["t"]) - amount / (dt * c), 0.0, atol=2e-8, rtol=0)
                heat[i] += amount
                heat[j] -= amount
                sh += amount * (1 / final[i]["t"] - 1 / final[j]["t"])
            close(step["pressure_work_j"], work, atol=1e-9, rtol=1e-9)
            close(step["net_heat_j"], heat, atol=1e-9, rtol=1e-9)
            for i, (a, b) in enumerate(zip(data, final, strict=True)):
                if old["phases"][i]["component_moles"] != new["phases"][i]["component_moles"]:
                    raise ValueError("Source step changed phase component stocks")
                close(b["volume"] - a["volume"], dv[i], atol=2e-13, rtol=2e-7)
                close(
                    b["internal"] - a["internal"] + work[i] - heat[i],
                    mechanical["physical"][i] + mechanical["numerical"][i],
                    atol=5e-8,
                    rtol=2e-7,
                )
            v_error = sum(s["volume_m3"] for s in new["phases"]) - sum(
                s["volume_m3"] for s in old["phases"]
            )
            e_error = math.fsum(s["total_energy_j"] for s in new["phases"]) - math.fsum(
                s["total_energy_j"] for s in old["phases"]
            )
            p_error = (
                math.fsum(s["momentum_kg_m_s"] for s in new["phases"])
                - math.fsum(s["momentum_kg_m_s"] for s in old["phases"])
                - mechanical["wall_impulse"]
            )
            close(v_error, 0.0, atol=1e-12, rtol=0)
            close(e_error, 0.0, atol=1e-7, rtol=0)
            close(p_error, 0.0, atol=1e-10, rtol=0)
            close(step["volume_residual_m3"], v_error, atol=1e-14, rtol=0)
            close(step["energy_residual_j"], e_error, atol=1e-9, rtol=0)
            close(step["momentum_residual_kg_m_s"], p_error, atol=1e-12, rtol=0)
            close(step["wall_impulse_kg_m_s"], mechanical["wall_impulse"], atol=1e-12, rtol=1e-10)
            temperatures = np.array([d["t"] for d in final])
            sd = np.sum(mechanical["physical"] / temperatures)
            sn = np.sum(mechanical["numerical"] / temperatures)
            bound = sp + sh + sd + sn
            entropy = sum(d["entropy"] for d in final) - sum(d["entropy"] for d in data)
            if min(sp, sh, sd, sn) < -1e-8 or entropy < max(0.0, bound) - 1e-8:
                raise ValueError("Independent phase entropy inequality failed")
            for key, expected in (
                ("pressure_relaxation_entropy_j_k", sp),
                ("heat_relaxation_entropy_j_k", sh),
                ("drag_physical_entropy_j_k", sd),
                ("drag_temporal_entropy_j_k", sn),
                ("entropy_source_lower_bound_j_k", bound),
                ("entropy_change_j_k", entropy),
                ("thermodynamic_temporal_entropy_j_k", entropy - bound),
            ):
                close(step[key], expected, atol=1e-9, rtol=1e-7)
            interval_impulse += mechanical["wall_impulse"]
            max_e, max_p = max(max_e, abs(e_error)), max(max_p, abs(p_error))
            old, data = new, final
            checked += 1
        close(
            old["time_s"], start + command["duration_s"], atol=8 * math.ulp(old["time_s"]), rtol=0
        )
        published = interval["state"]
        metadata(published)
        if {k: v for k, v in old.items() if k != "next_step_s"} != {
            k: v for k, v in published.items() if k != "next_step_s"
        }:
            raise ValueError("Published relaxation state differs from accepted half steps")
        close(
            interval["conservation_residuals"],
            ledger(start_state, published, interval_impulse),
            atol=ledger_atol,
            rtol=0,
        )
        total_impulse += interval_impulse
        old = published
    if report["state"] != old:
        raise ValueError("Final phase checkpoint differs")
    close(
        report["conservation_residuals"],
        ledger(first, old, total_impulse),
        atol=ledger_atol,
        rtol=0,
    )
    return {
        "accepted": True,
        "checked_substeps": checked,
        "maximum_energy_residual_j": max_e,
        "maximum_momentum_residual_kg_m_s": max_p,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = json.loads(args.report.read_text())
    root = Path(__file__).resolve().parents[1]
    if not result.get("sources_unchanged_during_run") or not result.get("source_sha256"):
        raise ValueError("Missing phase source provenance")
    for path, h in result["source_sha256"].items():
        source = (root / path).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != h:
            raise ValueError("Changed phase relaxation source")
    print(json.dumps(audit(json.loads(args.config.read_text()), result), indent=2))


if __name__ == "__main__":
    main()
