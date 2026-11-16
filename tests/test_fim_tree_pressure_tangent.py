"""Actual FIM assembly and accepted state using compact network sensitivities."""

import pytest
import torch
from test_fim_blackoil_surface import config, context, solver

from surface import PropertyModelError


def operator_solver():
    c = config()
    c["simulation"]["physical_surface_network"]["sensitivity_backend"] = "tree_operator"
    return solver(c)


def test_operator_network_assembly_matches_dense_well_blocks():
    a, b = solver(), operator_solver()
    *_, fa = context(a)
    b.wells.maximum_surface_product_entries = b.wells.n_w * 3 * 3
    *_, fb = context(b)
    ra, ja, wa = fa()
    rb, jb, wb = fb()
    torch.testing.assert_close(ra, rb, rtol=1e-12, atol=1e-9)
    torch.testing.assert_close(ja, jb, rtol=1e-12, atol=1e-9)
    for key in ("B_full", "D_full", "r_w"):
        torch.testing.assert_close(wa[key], wb[key], rtol=1e-12, atol=1e-9)
    assert "pressure_linearization" in b.wells.surface_report["network"]
    assert "pressure_derivatives_per_kg_s" not in b.wells.surface_report["network"]


def test_actual_reservoir_step_and_explicit_regulator_boundary():
    a, b = solver(), operator_solver()
    assert a.step(86400) and b.step(86400)
    torch.testing.assert_close(a.fluid.pressure, b.fluid.pressure, rtol=1e-11, atol=1e-4)
    torch.testing.assert_close(a.fluid.s_w, b.fluid.s_w, rtol=1e-11, atol=1e-10)
    with pytest.raises(PropertyModelError, match="pressure regulation"):
        b.wells.set_surface_pressure_control(
            {"valve": "tube1", "node": "bottom_1", "target_pressure_pa": 2.4e6}
        )


def test_legacy_checkpoint_without_sensitivity_setting_uses_dense_default():
    import pickle

    a = solver()
    del a.wells.sensitivity_backend
    b = pickle.loads(pickle.dumps(a))
    assert b.step(86400)
    assert b.wells.surface_report["execution"]["sensitivity_backend"] == "dense"
