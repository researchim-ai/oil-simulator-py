"""Constraint identities accelerate proposals while physical brackets certify acceptance."""

from dataclasses import replace

import pytest
import torch
from test_fim_facility_coupling import build

from surface.capacity_bracket import (
    CapacityBracket,
    facility_constraint_margin,
    facility_constraint_margins,
)


def test_plateau_of_unrelated_constraint_does_not_hide_active_root():
    root = 0.003

    def margins(x):
        return {"pressure": -0.05, "gas_export.valve": x / root - 1}

    b = CapacityBracket(
        0.0001,
        0.5,
        1e-9,
        max(margins(0.0001).values()),
        max(margins(0.5).values()),
        low_margins=margins(0.0001),
        high_margins=margins(0.5),
    )
    calls = 0
    while not b.complete:
        x = b.proposal()
        m = margins(x)
        ok = max(m.values()) <= 0
        b.update(x, ok, max(m.values()), margins=m)
        assert b.low <= root < b.high
        calls += 1
        assert calls < 50
    assert calls <= 4


def test_changed_active_constraint_and_missing_maps_keep_solved_bracket():
    b = CapacityBracket(
        0.01,
        1.0,
        1e-7,
        -0.3,
        2.0,
        low_margins={"a": -0.3, "b": -0.4},
        high_margins={"a": 1.0, "b": 2.0},
    )
    x = b.proposal()
    assert 0.01 < x < 1.0
    b.update(x, False, 0.5, margins={"a": 0.5})
    y = b.proposal()
    assert b.low < y < b.high
    b.update(y, True, -0.1, margins=None)
    assert b.low < b.proposal() < b.high


def test_constraint_identifiers_preserve_scalar_margin_and_gas_error_identity():
    report = {
        "limits": [{"minimum": 1.0, "maximum": 3.0, "value": 2.0}],
        "gas_export": {"valve": {"status": "flowing", "utilization": 1.2}},
    }
    values = facility_constraint_margins(report)
    assert values == pytest.approx(
        {"limit:0:minimum": -1 / 3, "limit:0:maximum": -1 / 3, "gas_export.valve": 0.2}
    )
    assert facility_constraint_margin(report) == max(values.values())


def test_real_coupled_constraint_root_matches_legacy_with_same_tolerance():
    a, b = build(), build()
    a.control = replace(
        a.control, commit_method="state_restore", fraction_tolerance=1e-9, max_evaluations=60
    )
    b.control = replace(a.control, search_method="facet_safeguarded")
    assert a.step(3600.0) and b.step(3600.0)
    assert abs(a.report["fraction"] - b.report["fraction"]) < 1e-9
    torch.testing.assert_close(a.solver.fluid.pressure, b.solver.fluid.pressure, rtol=0, atol=0.1)
    torch.testing.assert_close(a.solver.fluid.s_g, b.solver.fluid.s_g, rtol=0, atol=1e-8)
    assert b.report["capacity_bracket"]["width"] <= b.control.fraction_tolerance
    assert b.report["evaluations"] < a.report["evaluations"]


def test_zero_capacity_and_nonfinite_named_margins_fall_back_safely():
    b = CapacityBracket(
        0.01,
        1.0,
        1e-7,
        -0.3,
        None,
        low_margins={"valve": -0.3, "other": None},
        high_margins={"valve": None, "other": float("nan")},
    )
    assert b.proposal() == pytest.approx(0.505)


def test_actual_cuda_faceted_root_retains_physical_solution():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    a, b = build(device="cuda"), build(device="cuda")
    for c in [a, b]:
        c.control = replace(
            c.control, commit_method="state_restore", fraction_tolerance=1e-9, max_evaluations=60
        )
    b.control = replace(b.control, search_method="facet_safeguarded")
    assert a.step(3600.0) and b.step(3600.0)
    torch.testing.assert_close(a.solver.fluid.pressure, b.solver.fluid.pressure, rtol=0, atol=0.1)
    assert abs(a.report["fraction"] - b.report["fraction"]) <= 1e-9
    assert b.report["evaluations"] < a.report["evaluations"]
