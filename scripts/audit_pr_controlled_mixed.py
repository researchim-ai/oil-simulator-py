"""Audit common SDIRK2 and declared BE fallback, including report continuity.

Dispatches to existing independent physical/controller auditors per accepted
record. Only the outer execution-adapter id is normalized to the identical
reference model; physical states and ledgers are never recomputed by runtime.
"""

import copy
import json
import math

from audit_pr_surface_collocation import audit as audit_common
from check_pr_controlled_surface import audit as audit_legacy
from check_pr_subcycled_surface import _clock, _ledger, _reported_ledger, _same_state
from check_stratified_pr_flow import close

from surface.pr_surface_control import PRControlledSurface


def audit(config, report):
    if config != report["config"] or len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Mixed surface request differs")
    source = PRControlledSurface.from_dict(config["model"])
    parsed = getattr(source, "reference_surface", source)
    if json.loads(json.dumps(source.to_dict())) != report["model"]:
        raise ValueError("Mixed surface model differs")
    current = copy.deepcopy(report["initial"])
    if set(config["initial"]) != {"state"}:
        raise ValueError("Mixed audit requires an explicit initial checkpoint")
    if current != config["initial"]["state"] or current["model_id"] != source.model_id:
        raise ValueError("Mixed surface initial checkpoint differs")
    first = copy.deepcopy(current)
    checked = {"sdirk2-common": 0, "controlled_backward_euler_fallback": 0}
    total_heat = total_work = 0.0

    def normalized(state):
        state = copy.deepcopy(state)
        if state["model_id"] not in (source.model_id, parsed.model_id):
            raise ValueError("Mixed checkpoint adapter differs")
        state["model_id"] = parsed.model_id
        return state

    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        before = copy.deepcopy(current)
        end = _clock(current) + command["duration_s"]
        heat = work = 0.0
        for record in interval["steps"]:
            method = record.get("method")
            if method not in checked:
                raise ValueError("Unknown mixed integration record")
            dt = record["dt_s"]
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Invalid mixed integration step")
            close(_clock(record["state"]) - _clock(current), dt, atol=1e-14, rtol=1e-12)
            one = copy.deepcopy(record)
            one["state"] = normalized(one["state"])
            initial = normalized(current)
            request = {
                "model": parsed.to_dict(),
                "initial": {"state": initial},
                "schedule": [{**command, "duration_s": dt}],
            }
            request = json.loads(json.dumps(request))
            ledger = _ledger(
                parsed, initial, one["state"], one["total_heat_j"] + one.get("total_work_j", 0.0)
            )
            single = {
                "state": one["state"],
                "steps": [one],
                "total_heat_j": one["total_heat_j"],
                "total_work_j": one.get("total_work_j", 0.0),
                "conservation_residuals": ledger,
            }
            evidence = {
                "config": request,
                "model": request["model"],
                "initial": initial,
                "intervals": [single],
                "total_heat_j": single["total_heat_j"],
                "total_work_j": single["total_work_j"],
                "conservation_residuals": ledger,
            }
            checker = audit_common if method == "sdirk2-common" else audit_legacy
            if checker(request, evidence)["accepted"] is not True:
                raise ValueError("Mixed physical audit rejected")
            checked[method] += 1
            heat += single["total_heat_j"]
            work += single["total_work_j"]
            current = copy.deepcopy(record["state"])
        close(_clock(current), end, atol=8 * math.ulp(end), rtol=0)
        _same_state(normalized(interval["state"]), normalized(current), schedule_hint=True)
        current = copy.deepcopy(interval["state"])
        close(interval["total_heat_j"], heat, atol=1e-6, rtol=2e-7)
        close(interval.get("total_work_j", 0.0), work, atol=1e-8, rtol=2e-7)
        actual = _ledger(parsed, before, current, heat + work)
        _reported_ledger(interval["conservation_residuals"], actual, before, current)
        total_heat += heat
        total_work += work
    close(report["total_heat_j"], total_heat, atol=1e-6, rtol=2e-7)
    close(report.get("total_work_j", 0.0), total_work, atol=1e-8, rtol=2e-7)
    actual = _ledger(parsed, first, current, total_heat + total_work)
    _reported_ledger(report["conservation_residuals"], actual, first, current)
    return {
        "accepted": True,
        "checked_records": checked,
        "conservation_residuals": actual,
        "scope": "independent stage/BE physics, controller records, clocks, endpoints and global N/E; explicit checkpoint, no walls in common-stage auditor or experimental calibration",
    }
