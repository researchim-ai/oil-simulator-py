"""Actual shared columns, closed graph cycles and separate accepted ports."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from check_pr_capillary_axial import audit
from check_pr_capillary_facility import audit as facility_audit
from test_surface_pr_capillary_column import setup as column_case

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_facility import PRCapillaryFacility, PRCapillaryFacilityState
from surface.stratified_pr import StratifiedPRColumn


def setup(device="cpu", self_loop=False):
    base, old = column_case(device=device)
    names = ["A"] if self_loop else ["A", "B"]
    columns = {n: StratifiedPRColumn.from_dict(base.column.to_dict(), device=device) for n in names}
    row = {
        "pipe": base.pipe,
        "parameters": base.parameters,
        "maximum_unknowns": 4096,
        "linear_solver": "auto",
        "volume_diffusion": "acoustic",
        "ports": [{"column": n, "height_m": base.height} for n in ("A", names[-1])],
    }
    lines = {"out": row, "back": {**row, "ports": list(reversed(row["ports"]))}}
    model = PRCapillaryFacility(lines, columns)
    state = model.initialize({n: old.pipe for n in lines}, {n: old.column for n in names})
    return model, state


@pytest.mark.parametrize("self_loop", [False, True])
def test_shared_stocks_are_updated_once_from_every_real_endpoint(self_loop):
    model, old = setup(self_loop=self_loop)
    snapshot = old.to_dict()
    final, report = model.step(old, 1e-7)
    assert old.to_dict() == snapshot
    assert facility_audit(model.to_dict(), old, final, report)["accepted"]
    assert final.time_s == 1e-7
    assert len(final.columns) == (1 if self_loop else 2)
    for name, cs in old.columns:
        new = dict(final.columns)[name]
        owned = model.column_ports[name]
        for j, amount in enumerate(cs.component_moles):
            actual = math.fsum(
                report["port_transfers"][line][side]["transfer_n_m_e"][j] for line, side in owned
            )
            assert new.component_moles[j] == amount - actual
            assert actual > 0
    for name, pipe_record in report["pipes"].items():
        axial = pipe_record["axial"]
        line = PRCapillaryAxialPipe.from_dict(report["pipe_models"][name])
        checked = audit(
            line, dict(old.pipes)[name], PRActiveMultifluidState.from_dict(axial["state"]), axial
        )
        for actual, expected in zip(report["port_transfers"][name], checked["port_transfers"]):
            np.testing.assert_allclose(
                actual["transfer_n_m_e"], expected["transfer_n_m_e"], rtol=1e-9, atol=1e-23
            )
            assert actual["capillary_work_j"] == pytest.approx(
                expected["capillary_work_j"], abs=1e-23
            )
        summed = np.sum([p["transfer_n_m_e"] for p in report["port_transfers"][name]], axis=0)
        np.testing.assert_allclose(summed, axial["external_transfer_n_m_e"], rtol=1e-10, atol=1e-22)
    assert all(
        abs(v) <= t for v, t in zip(report["conservation_residuals"], report["roundoff_allowances"])
    )
    restored = PRCapillaryFacility.from_dict(json.loads(json.dumps(model.to_dict())))
    repeated, _ = restored.step(
        PRCapillaryFacilityState.from_dict(json.loads(json.dumps(old.to_dict()))), 1e-7
    )
    assert repeated == final


def test_full_interval_and_continuation_restart_are_exact_and_failed_interval_is_atomic():
    model, old = setup()
    snapshot = old.to_dict()
    final, report = model.advance(old, 2e-7)
    assert final.time_s == 2e-7 and len(report["steps"]) == 2
    restored = PRCapillaryFacility.from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PRCapillaryFacilityState.from_dict(json.loads(json.dumps(final.to_dict())))
    a, _ = model.advance(final, 1e-7)
    b, _ = restored.advance(checkpoint, 1e-7)
    assert a == b
    with pytest.raises(ValueError, match="Unknown facility heater"):
        model.step(old, 1e-7, heaters={"missing": {"power_w": 1, "temperature_k": 400}})
    assert old.to_dict() == snapshot


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_one_shared_column_loses_and_recreates_its_interface_without_cloning_ports(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, old = setup(device=device)
    hot_command = {"A": {"power_w": 1e11, "temperature_k": 450}}
    cold_command = {"A": {"power_w": -1e11, "temperature_k": 160}}
    hot, heating = model.step(old, 1e-7, heaters=hot_command)
    assert len(dict(hot.columns)["A"].phase_roots) == 1
    assert len(dict(hot.columns)["B"].phase_roots) == 2
    assert dict(hot.layer_slots)["A"] == ("gas",)
    models = model.inspect(hot)
    assert set(models["out"].boundaries[0]) == {"gas"}
    assert set(models["back"].boundaries[1]) == {"gas"}
    assert facility_audit(model.to_dict(), old, hot, heating, heaters=hot_command)["accepted"]
    cold, cooling = model.step(hot, 1e-7, heaters=cold_command)
    assert facility_audit(model.to_dict(), hot, cold, cooling, heaters=cold_command)["accepted"]
    assert len(dict(cold.columns)["A"].phase_roots) == 2
    assert all(r["entropy_production_j_k"] > 0 for r in (heating, cooling))
    defect, allowed = model.ledger(old, cold)
    assert all(abs(v) <= t for v, t in zip(defect, allowed))


def test_numerical_column_failure_rolls_back_the_whole_graph_and_cuts_the_step(monkeypatch):
    from surface.liquid import HydraulicConvergenceError

    model, old = setup()
    snapshot = old.to_dict()
    original = model.columns["B"].equilibrate
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HydraulicConvergenceError("forced column failure after both pipe solves")
        return original(*args, **kwargs)

    monkeypatch.setattr(model.columns["B"], "equilibrate", fail_once)
    final, report = model.advance(old, 1e-7)
    assert final.time_s == 1e-7
    assert len(report["rejected"]) == 1 and len(report["steps"]) == 2
    assert old.to_dict() == snapshot
    previous = old
    for step in report["steps"]:
        current = PRCapillaryFacilityState.from_dict(step["state"])
        assert facility_audit(model.to_dict(), previous, current, step)["accepted"]
        previous = current


def test_one_column_two_elevations_use_each_actual_gravitational_port_energy():
    base, seed = setup(self_loop=True)
    raw = base.lines["back"]["pipe"].to_dict()
    raw["face_elevations_m"] = [0.08] * (len(raw["lengths_m"]) + 1)
    high = PRActiveMultifluidPipe.from_dict(raw)
    rows = dict(base.lines)
    rows["back"] = {
        **rows["back"],
        "pipe": high,
        "ports": [{"column": "A", "height_m": 0.08}, None],
    }
    model = PRCapillaryFacility(rows, base.columns)
    initial = model.initialize(
        {
            "out": dict(seed.pipes)["out"],
            "back": high.initialize([dict(c) for c in dict(seed.pipes)["back"].cells]),
        },
        dict(seed.columns),
    )
    final, report = model.step(initial, 1e-7)
    assert facility_audit(model.to_dict(), initial, final, report)["accepted"]
    assert model.adapters["out", 0].elevation != model.adapters["back", 0].elevation
    assert set(model.inspect(initial)["back"].boundaries[0]) == {"gas"}


def test_opposed_pressure_impulses_use_operand_roundoff_without_changing_newton_accuracy():
    config = json.loads(
        (Path(__file__).parents[1] / "configs/surface_pr_capillary_facility_cycle.json").read_text()
    )
    model = PRCapillaryFacility.from_dict(config["system"])
    state = PRCapillaryFacilityState.from_dict(config["initial"])
    exercised = False
    for period in config["schedule"]:
        old = state
        state, report = model.step(state, period["duration_s"], heaters=period["heaters"])
        assert facility_audit(model.to_dict(), old, state, report, heaters=period["heaters"])[
            "accepted"
        ]
        for name, body in report["pipes"].items():
            axial = body["axial"]
            pipe = model.lines[name]["pipe"]
            transported = PRActiveMultifluidState.from_dict(axial["state"])
            old_m, new_m = pipe._totals(dict(old.pipes)[name])[-1], pipe._totals(transported)[-1]
            cancelled_total_limit = 2e-10 * (abs(old_m) + abs(new_m)) + 1e-22
            actual = abs(axial["conservation_residuals"][-1])
            if actual > cancelled_total_limit:
                exercised = True
                assert actual <= cancelled_total_limit + axial["momentum_roundoff_allowance_kg_m_s"]
            assert axial["maximum_scaled_residual"] <= 2e-12
    assert exercised
