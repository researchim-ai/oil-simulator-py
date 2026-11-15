"""Independent thermal oracle and strict adaptive benchmark comparison gates."""

import copy

import numpy as np
import pytest
from benchmark_pr_adaptive_preconditioner import (
    ThermalReference,
    audit_results,
    integrate,
    pair_errors,
)
from benchmark_pr_block_preconditioner import fixture


def test_independent_cv_pressure_and_thermal_energy_derivative():
    model, state = fixture(4)
    ref = ThermalReference(model, state)
    np.testing.assert_allclose(
        ref.pressure(ref.initial[:4]), [s.pressure_pa for _, s in state.nodes], rtol=0, atol=1e-6
    )
    rate = ref.rhs(0.0, ref.initial)
    energy_rate = (ref.n * ref.cv(ref.initial[:4]) * rate[:4]).sum() + (ref.c * rate[4:]).sum()
    assert abs(energy_rate) < 1e-8
    assert rate[:4].max() > 0 and rate[4:].min() < 0


def test_actual_adaptive_trajectories_pass_independent_reference_and_balance_gates():
    model, state = fixture(4)
    duration = 2e-4
    ref = ThermalReference(model, state)
    ref.solve(duration)
    compared = []
    for enabled in (False, True):
        model._collocation_block_preconditioning = enabled
        result = integrate(model, state, [{"duration_s": duration}])
        assert audit_results("thermal32", model, result, [{"duration_s": duration}]) == len(
            result[0][1]["steps"]
        )
        errors, _ = ref.errors(result)
        assert errors["temperature_k"] < 0.005
        compared.append(result)
    pair_errors(*compared)
    wrong = copy.deepcopy(compared[1])
    wrong[0][1]["state"]["nodes"][model.names[0]]["pressure_pa"] += 1.0
    with pytest.raises(AssertionError):
        pair_errors(compared[0], wrong)
    wrong = copy.deepcopy(compared[1])
    wrong[0][1]["state"]["time_s"] += 1e-3
    with pytest.raises(AssertionError):
        pair_errors(compared[0], wrong)
