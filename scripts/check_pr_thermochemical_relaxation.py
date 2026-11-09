"""Independent scalar PR, extensive BGK and equilibrium-target audit."""

import math

import numpy as np
from check_stratified_pr import Reference
from check_stratified_pr_flow import close
from reference_pr_interphase_relaxation import decode

from surface.equilibrium_properties import EquilibriumFluid
from surface.operation import _digest
from surface.peng_robinson import R


def families(fluid, state, data):
    # Solve the critical-volume polynomial independently of the runtime's
    # closed-form expression; constitutive parameters use scalar NumPy PR.
    critical = max(q.real for q in np.roots([1.0, -3.0, -3.0, -3.0]) if abs(q.imag) < 1e-10)
    threshold = (critical**2 + 2 * critical - 1) ** 2 / (2 * (critical + 1) * (critical - 1) ** 2)
    result = []
    for row in data:
        x = row["n"] / row["n"].sum()
        active = np.flatnonzero(x > 0)
        ref = Reference(fluid, active)
        if ref.pure:
            a = ref.pure._a(row["t"])[0]
            b = ref.pure.b
        else:
            a, _, b, *_ = ref.mixture._parameters(row["t"], x[active])
        ratio = a / (R * row["t"] * b)
        result.append(
            "stable"
            if ratio <= threshold * (1 + 1e-12)
            else "liquid"
            if row["volume"] / row["n"].sum() / b < critical
            else "vapor"
        )
    return result


def totals(state):
    nc = len(state["phases"][0]["component_moles"])
    return np.array(
        [
            *(math.fsum(p["component_moles"][j] for p in state["phases"]) for j in range(nc)),
            math.fsum(p["volume_m3"] for p in state["phases"]),
            math.fsum(p["total_energy_j"] for p in state["phases"]),
            math.fsum(p["momentum_kg_m_s"] for p in state["phases"]),
        ]
    )


def reference_limits(before, after):
    phases = list(before["phases"]) + list(after["phases"])
    nc = len(phases[0]["component_moles"])
    scales = [math.fsum(abs(p["component_moles"][j]) for p in phases) for j in range(nc)]
    scales.extend(
        (
            math.fsum(p["volume_m3"] for p in phases),
            math.fsum(abs(p["total_energy_j"]) + p["pressure_pa"] * p["volume_m3"] for p in phases),
            math.fsum(abs(p["momentum_kg_m_s"]) for p in phases),
        )
    )
    return np.maximum(
        np.r_[[1e-8] * nc, 1e-12, 1e-6, 1e-9], 64 * np.finfo(float).eps * np.array(scales)
    )


def audit(model, before, after, report):
    if (
        set(model) != {"model", "fluid", "relaxation_time_s", "maximum_active_phases"}
        or model["model"] != "pr_thermochemical_relaxation_v1"
    ):
        raise ValueError("Unknown thermochemical model")
    tau = model["relaxation_time_s"]
    budget = model["maximum_active_phases"]
    if not math.isfinite(tau) or tau < 0 or type(budget) is not int or not 1 <= budget <= 6:
        raise ValueError("Invalid source time constant or phase budget")
    fluid = EquilibriumFluid.from_dict(model["fluid"])
    nc = len(fluid.names)
    for state in (before, after):
        if (
            set(state) != {"phases", "phase_roots", "model_id", "time_s"}
            or state["model_id"] != _digest(model)
            or not 1 <= len(state["phases"]) <= budget
        ):
            raise ValueError("Thermochemical checkpoint registry differs")
        if not math.isfinite(state["time_s"]) or state["time_s"] < 0:
            raise ValueError("Invalid source clock")
    if after != report["state"]:
        raise ValueError("Published source state differs")
    dt = report["dt_s"]
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Positive physical source duration required")
    close(
        after["time_s"] - before["time_s"],
        dt,
        atol=8 * max(math.ulp(before["time_s"]), math.ulp(after["time_s"])),
        rtol=0.0,
    )
    old = decode(fluid, before, before["phase_roots"])
    new = decode(fluid, after, after["phase_roots"])
    eq_model = {"model": "pr_phase_equilibration_v1", "fluid": model["fluid"]}
    target = report["equilibrium_target"]
    raw = report["flash_target"]
    for point in (target, raw):
        if (
            set(point) != {"phases", "phase_roots", "model_id", "time_s"}
            or point["model_id"] != _digest(eq_model)
            or point["time_s"] != before["time_s"]
        ):
            raise ValueError("Equilibrium target metadata differs")
    target_data = decode(fluid, target, target["phase_roots"])
    raw_data = decode(fluid, raw, raw["phase_roots"])
    if target["phase_roots"] != raw["phase_roots"]:
        raise ValueError("Canonical refinement changed the selected phase topology")
    production_target = math.fsum(q["entropy"] for q in target_data) - math.fsum(
        q["entropy"] for q in old
    )
    tolerance = reference_limits(before, target)
    close(totals(target) - totals(before), np.zeros(nc + 3), atol=tolerance, rtol=0.0)
    raw_tolerance = tolerance.copy()
    raw_tolerance[nc] = max(raw_tolerance[nc], 1e-12 + 1e-9 * totals(before)[nc])
    raw_tolerance[nc + 1] = max(raw_tolerance[nc + 1], 1e-6 + 1e-9 * abs(totals(before)[nc + 1]))
    close(totals(raw) - totals(before), np.zeros(nc + 3), atol=raw_tolerance, rtol=0.0)
    for point, data in ((target, target_data), (raw, raw_data)):
        for row in data:
            close(
                [row["p"], row["t"], row["velocity"]],
                [data[0]["p"], data[0]["t"], data[0]["velocity"]],
                atol=np.array([1e-6, 1e-8, 1e-9]),
                rtol=1e-10,
            )
        chemical_rows = []
        for row, root in zip(data, point["phase_roots"]):
            x = row["n"] / row["n"].sum()
            support = np.flatnonzero(x > 0)
            q = Reference(fluid, support).state(row["p"], row["t"], x, root)
            chemical_rows.append((support, np.log(x[support]) + q["phi"]))
        for support, mu in chemical_rows:
            if not np.array_equal(support, chemical_rows[0][0]):
                raise ValueError("Target phase supports differ")
            close(mu, chemical_rows[0][1], atol=1e-7, rtol=0.0)
    # Independent numerical TPD search of the declared equilibrium target.
    n = totals(target)[:nc]
    active = np.flatnonzero(n > 0)
    z = n[active] / n.sum()
    ref = Reference(fluid, active)
    first = target_data[0]
    x = first["n"] / first["n"].sum()
    q = ref.state(first["p"], first["t"], x, target["phase_roots"][0])
    chemical = np.log(x[active]) + q["phi"]
    if ref.pure:
        minimum_tpd = min(
            ref.state(first["p"], first["t"], x, r)["phi"][0] - chemical[0]
            for r in ("liquid", "vapor")
        )
    else:
        minimum_tpd = ref.mixture._tpd(first["p"], first["t"], z, chemical)[0]["tpd"]
    if minimum_tpd < -1e-8:
        raise ValueError("Equilibrium target failed independent numerical TPD search")
    old_families = families(fluid, before, old)
    eq_families = families(fluid, target, target_data)
    if (
        list(report["old_families"]) != old_families
        or list(report["target_families"]) != eq_families
    ):
        raise ValueError("Mechanical cubic branch classification differs")
    a, b = (0.0, 1.0) if tau == 0 else (math.exp(-dt / tau), -math.expm1(-dt / tau))
    close([report["old_weight"], report["equilibrium_weight"]], [a, b], atol=1e-15, rtol=1e-14)
    if tau > 0 and (a <= 0 or b <= 0):
        raise ValueError("Unresolved finite-rate source weights")
    pairs = report["phase_pairs"]
    if len(pairs) != len(new):
        raise ValueError("Source phase-pair registry differs")
    owners = []
    groups = {}
    for i, j, share in pairs:
        if i is not None:
            if type(i) is not int or not 0 <= i < len(old) or i in owners:
                raise ValueError("Invalid old phase ownership")
            owners.append(i)
        if j is not None:
            if (
                type(j) is not int
                or not 0 <= j < len(target_data)
                or not math.isfinite(share)
                or share <= 0
            ):
                raise ValueError("Invalid target phase share")
            groups.setdefault(j, []).append((i, share))
            if (
                i is not None
                and old_families[i] != eq_families[j]
                and "stable" not in (old_families[i], eq_families[j])
            ):
                raise ValueError("Different mechanical branches merged")
        elif share != 0 or i is None:
            raise ValueError("Empty phase pair")
    if (
        set(groups) != set(range(len(target_data)))
        or (a and set(owners) != set(range(len(old))))
        or (not a and owners)
    ):
        raise ValueError("A source or target phase was dropped or duplicated")
    for j, rows in groups.items():
        close(sum(v for _, v in rows), 1.0, atol=1e-14, rtol=0.0)
        total = sum(sum(before["phases"][i]["component_moles"]) for i, _ in rows if i is not None)
        for i, share in rows:
            close(
                share,
                1.0 if i is None else sum(before["phases"][i]["component_moles"]) / total,
                atol=1e-14,
                rtol=1e-12,
            )
    zero = np.zeros(nc + 3)

    def phase_stocks(state, index):
        if index is None:
            return zero
        p = state["phases"][index]
        return np.r_[
            p["component_moles"], p["volume_m3"], p["total_energy_j"], p["momentum_kg_m_s"]
        ]

    for k, (i, j, share) in enumerate(pairs):
        expected = a * phase_stocks(before, i) + b * share * phase_stocks(target, j)
        close(
            phase_stocks(after, k),
            expected,
            atol=np.r_[[1e-11] * nc, 1e-14, 1e-8, 1e-12],
            rtol=1e-10,
        )
    residual = totals(after) - totals(before)
    close(residual, zero, atol=reference_limits(before, after), rtol=0.0)
    close(
        report["conservation_residuals"],
        residual,
        atol=np.r_[[1e-12] * nc, 1e-14, 1e-9, 1e-12],
        rtol=0.0,
    )
    production = math.fsum(q["entropy"] for q in new) - math.fsum(q["entropy"] for q in old)
    bound = b * production_target
    close(
        [report["entropy_production_j_k"], report["entropy_lower_bound_j_k"]],
        [production, bound],
        atol=1e-7,
        rtol=1e-8,
    )
    entropy_limit = max(
        1e-7,
        64
        * np.finfo(float).eps
        * (
            abs(math.fsum(q["entropy"] for q in old))
            + abs(math.fsum(q["entropy"] for q in new))
            + abs(math.fsum(q["entropy"] for q in target_data))
        ),
    )
    close(
        report["conservation_tolerances"], reference_limits(before, after), atol=1e-15, rtol=1e-12
    )
    close(report["entropy_tolerance_j_k"], entropy_limit, atol=1e-15, rtol=1e-12)
    if production < max(0.0, bound) - entropy_limit:
        raise ValueError("Extensive BGK entropy bound failed")
    return {
        "accepted": True,
        "phase_count_before": len(old),
        "phase_count_after": len(new),
        "energy_residual_j": residual[nc + 1],
        "minimum_numerical_tpd": minimum_tpd,
    }


def audit_run(config, report):
    import json

    if config != report["config"]:
        raise ValueError("Thermochemical request differs")
    from surface.pr_thermochemical_relaxation import PRThermochemicalRelaxation

    model = PRThermochemicalRelaxation.from_dict(config["model"])
    if json.loads(json.dumps(model.to_dict())) != report["model"]:
        raise ValueError("Thermochemical normalized model differs")
    old = first = report["initial"]
    if config["initial"] != {"state": old}:
        requested = config["initial"]
        if (
            set(requested) != {"phases", "phase_roots"}
            or requested["phases"] != old["phases"]
            or requested["phase_roots"] != old["phase_roots"]
            or old["time_s"] != 0.0
        ):
            raise ValueError("Thermochemical restart or initial phases differ")
    if len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Thermochemical schedule differs")
    checked = 0
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        start = old
        entropy = 0.0
        if (
            set(command) - {"duration_s", "maximum_step_s", "minimum_step_s", "maximum_attempts"}
            or not interval["steps"]
        ):
            raise ValueError("Invalid thermochemical command")
        for step in interval["steps"]:
            if step["dt_s"] > command.get("maximum_step_s", command["duration_s"]) * (1 + 1e-12):
                raise ValueError("Source maximum step exceeded")
            audit(config["model"], old, step["state"], step)
            old = step["state"]
            entropy += step["entropy_production_j_k"]
            checked += 1
        if old != interval["state"]:
            raise ValueError("Published source interval differs")
        close(
            old["time_s"],
            start["time_s"] + command["duration_s"],
            atol=8 * math.ulp(old["time_s"]),
            rtol=0.0,
        )
        close(interval["conservation_residuals"], totals(old) - totals(start), atol=1e-8, rtol=1e-8)
        close(interval["entropy_production_j_k"], entropy, atol=1e-7, rtol=1e-8)
    if old != report["state"]:
        raise ValueError("Final source state differs")
    close(report["conservation_residuals"], totals(old) - totals(first), atol=1e-8, rtol=1e-8)
    return {"accepted": True, "checked_substeps": checked, "phase_count_after": len(old["phases"])}


def main():
    import argparse
    import hashlib
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = json.loads(args.report.read_text())
    root = Path(__file__).resolve().parents[1]
    for path, digest in report.get("source_sha256", {}).items():
        source = (root / path).resolve()
        source.relative_to(root)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Changed source after calculation: " + path)
    if config["model"].get("model") in (
        "pr_active_multifluid_pipe_v1",
        "pr_active_multifluid_pipe_v2",
        "pr_active_multifluid_pipe_v3",
    ):
        from check_pr_active_multifluid_pipe import audit as audit_active

        result = audit_active(config, report)
    else:
        result = audit_run(config, report)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
