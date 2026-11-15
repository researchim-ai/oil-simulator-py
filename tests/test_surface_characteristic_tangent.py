"""Joint AD preserves the complete conservative/equilibrium wave tangent."""

import pytest
import torch
from test_surface_pr_equilibrium_transport import pipe


@pytest.mark.parametrize("temperature,velocity", [(320., 0.), (320., -2.), (196.2, 0.), (196.2, 2.)])
def test_joint_tangent_matches_independently_differentiated_blocks(monkeypatch, temperature, velocity):
    model, state = pipe(temperature, velocity, True)
    data = model._state_data(0, state.nodes[0][1])
    active = torch.nonzero(data["n"] > 0).flatten()
    actual = model._characteristics(0, data, active)
    original = torch.func.jacfwd
    calls = []

    def separate(function, *args, **kwargs):
        if function.__name__ != "quantities":
            return original(function, *args, **kwargs)
        calls.append(True)
        def derivative(y):
            return (original(lambda z: function(z)[0], *args, **kwargs)(y),
                    original(lambda z: function(z)[1], *args, **kwargs)(y))
        return derivative

    monkeypatch.setattr(torch.func, "jacfwd", separate)
    reference = model._characteristics(0, data, active)
    assert len(calls) == 1
    assert torch.equal(actual, reference)
