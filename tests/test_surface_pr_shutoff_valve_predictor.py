"""Check-valve opening uses zero-flow head, not the running discharge guess."""

from test_surface_pr_pump_connection import setup

from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork
from surface.pr_pump_connection import PumpConnectionBundle


def fixture(backpressure=1.028e6, *, device="cpu"):
    original, _, initial = setup(backpressure=backpressure)
    data = original.to_dict()
    data["pump_connections"][0]["outlet_loss_pa_s2_m6"] = 0.0
    model = PREquilibriumTransportNetwork.from_dict(data, device=device)
    return model, model.initialize(initial)


def predicted_valves(model, state):
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    evaluate = model._problem(state, seeds, 1e-5, model.default_openings, [0.0, 0.0])[3]
    return evaluate.pump_connections.valve_states


def step(model, state, **kwargs):
    return model._step(state, 1e-5, model.default_openings, [0.0, 0.0], [350.0, 350.0], **kwargs)


def test_zero_resistance_operating_pressure_does_not_mean_closed_valve(monkeypatch):
    model, old = fixture()
    before = old.to_dict()
    monkeypatch.setattr(PumpConnectionBundle, "_shutoff_valve_predictor", False)
    assert predicted_valves(model, old) == {"booster": True}
    reference, old_report = step(model, old)
    assert old_report["pump_check_valve_retries"] == 1
    monkeypatch.setattr(PumpConnectionBundle, "_shutoff_valve_predictor", True)
    assert predicted_valves(model, old) == {"booster": False}
    actual, new_report = step(model, old)
    assert new_report["pump_check_valve_retries"] == 0
    assert new_report["pump_connections"]["booster"]["flow_m3_s"] > 0
    assert not new_report["pump_connections"]["booster"]["valve_closed"]
    assert model._error(reference, actual) < 1e-5
    assert max(map(abs, new_report["conservation_residuals"])) < 1e-7
    assert old.to_dict() == before


def test_head_insufficient_keeps_closed_zero_flow_branch():
    model, state = fixture(backpressure=1.1e6)
    assert predicted_valves(model, state) == {"booster": True}
    _, report = step(model, state)
    assert report["pump_check_valve_retries"] == 0
    row = report["pump_connections"]["booster"]
    assert row["valve_closed"]
    assert row["flow_m3_s"] == row["shaft_power_w"] == 0


def test_complementarity_still_repairs_an_explicitly_wrong_branch():
    model, state = fixture()
    _, report = step(model, state, _pump_valve_states={"booster": True})
    assert report["pump_check_valve_retries"] == 1
    assert not report["pump_connections"]["booster"]["valve_closed"]
