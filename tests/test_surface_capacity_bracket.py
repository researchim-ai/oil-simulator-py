"""Certified sampled brackets: smooth, opaque and badly curved constraints."""

import math

import pytest

from surface.capacity_bracket import CapacityBracket, facility_constraint_margin


@pytest.mark.parametrize("root", [0.01001, 0.025, 0.4, 0.999])
@pytest.mark.parametrize("law", ["linear", "cubic", "opaque", "discontinuous"])
def test_search_keeps_solved_bracket_and_terminates(root, law):
    def margin(x):
        if law == "opaque":
            return None
        if law == "linear":
            return x - root
        if law == "cubic":
            return (x - root) ** 3
        return -1 if x <= root else 2

    bracket = CapacityBracket(0.01, 1, 1e-5, margin(0.01), margin(1))
    calls = 0
    while not bracket.complete:
        x = bracket.proposal()
        assert bracket.low < x < bracket.high
        bracket.update(x, x <= root, margin(x))
        assert bracket.low <= root < bracket.high
        calls += 1
        assert calls < 100
    assert bracket.high - bracket.low <= 1e-5
    if law == "linear":
        assert calls <= 3


def test_invalid_signed_information_falls_back_and_interior_is_required():
    b = CapacityBracket(0.01, 1, 1e-5, 1, float("inf"))
    assert b.low_margin is None and b.high_margin is None
    assert b.proposal() == 0.505
    with pytest.raises(ValueError):
        b.update(1, False, 1)
    with pytest.raises(ValueError):
        CapacityBracket(1, 0.01, 1e-5)
    with pytest.raises(ValueError):
        CapacityBracket(0.01, 1, float("nan"))


def test_measured_constraint_margin_has_correct_sign_and_units():
    r = {
        "limits": [{"minimum": None, "maximum": 0.001, "value": 0.0009}],
        "gas_export": {"valve": {"status": "flowing", "utilization": 0.8}},
    }
    assert facility_constraint_margin(r) == pytest.approx(-0.1)
    r["limits"][0]["value"] = 0.0012
    assert facility_constraint_margin(r) == pytest.approx(0.2)
    assert math.isfinite(facility_constraint_margin({"limits": []}))


def test_structured_capacity_failure_preserves_margin_through_pickle():
    import pickle

    from surface.gas_valve import GasValveCapacityError

    error = GasValveCapacityError("export", 0.1, 0.12)
    restored = pickle.loads(pickle.dumps(error))
    assert str(restored) == str(error)
    assert restored.normalized_margin == pytest.approx(0.2)
    assert GasValveCapacityError("closed", 0, 0.12).normalized_margin is None
