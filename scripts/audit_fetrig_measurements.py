"""Audit raw FETRig CCI observations without fitting or certifying a flow model.

Reuse the authors' GO-only mean/std convention; retain ALARM counts explicitly.
No outlier replacement, interpolation, zero adjustment or inferred standard error.
The raw LabView books remain available for a separate timing/conditions audit.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from pathlib import Path


def cci_statistics(raw):
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    header = next(i for i, row in enumerate(rows) if row and row[0] == "Trigger count")
    if rows[header] != ["Trigger count", "OUT01(mm)", ""]:
        raise ValueError("Unexpected CCI output units/header")
    metadata = {row[0]: row[1] for row in rows[:header] if len(row) == 2 and row[0]}
    cycle = metadata["Sampling cycle"]
    if not cycle.endswith("us"):
        raise ValueError("Unconfirmed CCI sampling cycle unit")
    dt = float(cycle[:-2]) * 1e-6
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid CCI sampling period")
    records = [row for row in rows[header + 1 :] if row]
    if len(records) != int(metadata["Data count"]):
        raise ValueError("CCI record count differs from header")
    triggers, accepted = [], []
    counts = {"GO": 0, "ALARM": 0}
    for row in records:
        if len(row) != 3 or row[2] not in counts:
            raise ValueError("Unknown CCI sensor acceptance status")
        trigger = int(row[0])
        if triggers and trigger <= triggers[-1]:
            raise ValueError("CCI trigger order is not strictly increasing")
        triggers.append(trigger)
        counts[row[2]] += 1
        if row[2] == "GO":
            value = float(row[1]) * 1e-3  # OUT01(mm), not the display resolution.
            if not math.isfinite(value):
                raise ValueError("Accepted CCI thickness is nonfinite")
            accepted.append(value)
    if not triggers or triggers[0] != int(metadata["First index"]) or len(accepted) < 2:
        raise ValueError("Unusable CCI trigger/accepted sample record")
    return {
        "records": len(records),
        "sensor_status_counts": counts,
        "sensor_acceptance_fraction": counts["GO"] / len(records),
        "sampling_period_s": dt,
        "first_trigger": triggers[0],
        "last_trigger": triggers[-1],
        "elapsed_first_to_last_s": (triggers[-1] - triggers[0]) * dt,
        "missing_triggers": triggers[-1] - triggers[0] + 1 - len(triggers),
        "go_mean_thickness_m": statistics.fmean(accepted),
        "go_sample_std_thickness_m": statistics.stdev(accepted),
        "go_minimum_thickness_m": min(accepted),
        "go_maximum_thickness_m": max(accepted),
        "standard_error_or_confidence_interval_estimated": False,
        "outliers_replaced": False,
        "zero_offset_adjusted": False,
    }


def run(data):
    data = Path(data)
    manifest_raw = (data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    cases = []
    for member in manifest["members"]:
        raw = gzip.decompress((data / member["file"]).read_bytes())
        if (
            hashlib.sha256(raw).hexdigest() != member["sha256"]
            or len(raw) != member["original_size"]
        ):
            raise ValueError("FETRig extracted source changed")
        if member["original_member"].endswith(".csv"):
            cases.append(
                {
                    "original_member": member["original_member"],
                    "raw_sha256": member["sha256"],
                    **cci_statistics(raw),
                }
            )
    return {
        "format": "fetrig_cci_input_audit_v1",
        "source": manifest["source"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "geometry": manifest["geometry"],
        "cases": cases,
        "total_cci_records": sum(row["records"] for row in cases),
        "model_fitted": False,
        "model_scored": False,
        "experimentally_qualified": False,
        "interpretation": "GO-only descriptive thickness statistics following source Keyence_Data_Manager.m; ALARM is missing sensor information, not a zero-thickness observation. Correlated samples and possibly nonrandom dropouts do not define an independent-sample confidence interval.",
        "remaining_before_model_scoring": [
            "LabView channel units, actual mass/gas flow and fluid properties",
            "clock alignment and steady-input selection independent of model error",
            "sensor zero reference and acceptance/dropout effects",
            "compatible rectangular-channel physical model and wave-domain assumptions",
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/data/surface/fetrig",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "cases": len(result["cases"]),
                "cci_records": result["total_cci_records"],
                "experimentally_qualified": result["experimentally_qualified"],
            }
        )
    )
