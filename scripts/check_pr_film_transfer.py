"""Independent scalar-PR/NumPy check of every material/thermal split step."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from check_thin_film_dynamics import audit as film_audit
from reference_pr_active_multifluid import packets

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe


def audit(config, report):
    if (
        config != report["request"]
        or config["initial"] != report["initial"]
        or config["initial_film"] != report["initial_film"]
    ):
        raise ValueError("Film transfer request/initial state differs")
    fc = {
        "model": config["film"],
        "initial_thickness_m": config["initial_film"]["thickness_m"],
        "duration_s": config["duration_s"],
    }
    steps = [{**r["film"], "state": r["film_state"]} for r in report["steps"]]
    film_result = film_audit(
        fc,
        {
            "request": fc,
            "source_sha256": report["source_sha256"],
            "initial": report["initial_film"],
            "final": report["final_film"],
            "result": {"steps": steps},
        },
    )
    pipe = PRActiveMultifluidPipe.from_dict(config["pipe"])
    nc = len(pipe.fluid.names)
    liquid = config["liquid_slot"]
    gas = config["gas_slot"]
    width = config["film"].get("transverse_width_m", 1.0)

    def close(a, b, label, atol=1e-18, rtol=1e-10):
        a, b = np.array(a), np.array(b)
        if (
            a.shape != b.shape
            or not np.all(np.isfinite(a))
            or not np.all(np.isfinite(b))
            or np.any(np.abs(a - b) > atol + rtol * np.maximum(np.abs(a), np.abs(b)))
        ):
            raise ValueError("Film transfer audit differs: " + label)

    old = report["initial"]
    packets(pipe, old)
    checked = 0
    for r in report["steps"]:
        new = r["state"]
        packets(pipe, new)
        dt = r["film"]["dt_s"]
        q = np.array(r["film"]["face_volume_flux_m2_s"][0]) * width
        d = np.array(r["film"]["face_dissipation_j"][0])
        heat = (d + np.roll(d, 1)) / 2
        close(r["phase_heat_j"], heat, "liquid heat")
        for name in (liquid, gas):
            stocks = np.array(
                [
                    [
                        *c[name]["component_moles"],
                        c[name]["volume_m3"],
                        c[name]["momentum_kg_m_s"],
                        c[name]["total_energy_j"],
                    ]
                    for c in old["cells"]
                ]
            )
            flow = q if name == liquid else -q
            transported = []
            for i, v in enumerate(flow):
                donor = stocks[i if v >= 0 else (i + 1) % len(flow)]
                transported.append(v * donor / donor[nc])
            transported = np.array(transported)
            expected = stocks - dt * (transported - np.roll(transported, 1, 0))
            if name == liquid:
                expected[:, -1] += heat
            actual = np.array(
                [
                    [
                        *c[name]["component_moles"],
                        c[name]["volume_m3"],
                        c[name]["momentum_kg_m_s"],
                        c[name]["total_energy_j"],
                    ]
                    for c in new["cells"]
                ]
            )
            close(actual, expected, "local " + name, atol=1e-20, rtol=1e-11)
        energy = lambda s: math.fsum(p["total_energy_j"] for c in s["cells"] for p in c.values())
        defect = energy(new) - energy(old) + r["film"]["free_energy_change_j"]
        roundoff = 64 * math.fsum(
            math.ulp(p["total_energy_j"])
            for s in (old, new)
            for c in s["cells"]
            for p in c.values()
        )
        close(defect, 0.0, "joint energy", atol=roundoff + 1e-18, rtol=0.0)
        close(new["time_s"], r["film_state"]["time_s"], "clock", atol=0, rtol=0)
        old = new
        checked += 1
    if report["final"] != old:
        raise ValueError("Final PR transfer checkpoint differs")
    return {"accepted": True, "checked_steps": checked, "film": film_result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(json.loads(a.config.read_text()), json.loads(a.report.read_text()))))


if __name__ == "__main__":
    main()
