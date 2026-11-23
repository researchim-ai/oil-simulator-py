"""Local step merges may reuse only unchanged, independently validated nodes."""

from dataclasses import replace

import pytest
import torch
from test_surface_pr_equilibrium_transport import pipe

from surface.liquid import ConservationError, PropertyModelError
from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def enable_node_reuse(monkeypatch):
    monkeypatch.setattr(PREquilibriumTransportNetwork, "_reuse_node_validation", True)


def clear(model):
    model._verified.clear()
    model._verified_nodes.clear()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("temperature", [320.0, 196.2])
def test_merge_revalidates_changed_node_at_original_index(device, temperature, monkeypatch):
    model, first = pipe(temperature)
    if device != "cpu":
        model = PREquilibriumTransportNetwork.from_dict(model.to_dict(), device=device)
    specs = {
        name: {
            "pressure_pa": body.pressure_pa,
            "temperature_k": body.temperature_k,
            "mole_fractions": [0.5, 0.5],
        }
        for name, body in first.nodes
    }
    specs["1"]["temperature_k"] += 0.1
    second = model.initialize(specs)
    # Keep node 0 bit-for-bit unchanged; independently initialized phase seeds
    # are not required to serialize with identical rounding on both devices.
    second = replace(second, nodes=(first.nodes[0], second.nodes[1]))
    clear(model)
    model.inspect(first)
    decoded, checked = [], []
    original_data, original_stable = model._state_data, model._stable

    def data(i, state):
        decoded.append(i)
        return original_data(i, state)

    def stable(i, *a, **kw):
        checked.append(i)
        return original_stable(i, *a, **kw)

    monkeypatch.setattr(model, "_state_data", data)
    monkeypatch.setattr(model, "_stable", stable)
    model.inspect(second)
    assert decoded == checked == [1]
    assert model._verified_node_hits >= 1


def test_changed_stock_cannot_hit_node_certificate():
    model, state = pipe(320.0)
    nodes = list(state.nodes)
    name, body = nodes[1]
    nodes[1] = name, replace(body, total_energy_j=body.total_energy_j + 100)
    with pytest.raises(ConservationError, match="checkpoint"):
        model.inspect(replace(state, nodes=nodes))


@pytest.mark.parametrize("replace_parameter", [False, True])
def test_eos_change_invalidates_node_certificates(replace_parameter):
    model, state = pipe(320.0)
    assert model._verified_nodes
    if replace_parameter:
        model.fluid.cp = model.fluid.cp * 1.1
    else:
        model.fluid.cp.mul_(1.1)
    with pytest.raises(ConservationError, match="checkpoint"):
        model.inspect(state)
    assert not model._verified_nodes


def test_failed_phase_audit_does_not_publish_pending_certificates(monkeypatch):
    model, state = pipe(320.0)
    clear(model)
    monkeypatch.setattr(model, "_stable_all", lambda data: [object() for _ in data])
    with pytest.raises(PropertyModelError, match="unstable"):
        model.inspect(state)
    assert not model._verified_nodes and not model._verified


def test_certificate_does_not_transfer_to_different_cell_geometry():
    base, _ = pipe(320.0)
    config = base.to_dict()
    config["nodes"][1]["length_m"] *= 2
    config["nodes"][1]["volume_m3"] *= 2
    model = PREquilibriumTransportNetwork.from_dict(config)
    state = model.initialize(
        {
            name: {"pressure_pa": 1e6, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]}
            for name in model.names
        }
    )
    wrong = replace(state, nodes=(state.nodes[0], ("1", state.nodes[0][1])))
    with pytest.raises(ConservationError, match="checkpoint"):
        model.inspect(wrong)


def test_bounded_node_registry():
    model, state = pipe(320.0)
    clear(model)
    model._node_validation_entries = 1
    model.inspect(state)
    assert len(model._verified_nodes) == 1
