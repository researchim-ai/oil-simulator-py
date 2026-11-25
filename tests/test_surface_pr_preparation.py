"""Exact prepared geometry/EOS operators preserve phase and thermal derivatives."""

import pytest
import torch
from test_surface_stratified_pr_network import setup


def prepare(model, enabled):
    model._prepare_residual = enabled
    model._batch_network_eos = enabled
    for vessel in model.vessels.values():
        vessel._prune_unused = enabled
        vessel.column._prepare_geometry = enabled


@pytest.mark.parametrize("two", [False, True])
@pytest.mark.parametrize("temperature_offset", [0.0, 0.003])
def test_prepared_network_preserves_off_equilibrium_thermal_jacobian(two, temperature_offset):
    model, state = setup(two=two)
    states, seeds = dict(state.vessels), model._seeds(dict(state.vessels))
    prepare(model, False)
    old = model._problem(states, seeds, 0.02, 4, model._commands(None))
    y = old[0].clone()
    t_b = len(old[4]["a"][0]) + len(old[4]["b"][0]) - len(seeds["b"][1])
    y += torch.linspace(-1e-5, 1e-5, len(y), dtype=y.dtype)
    t_a = len(old[4]["a"][0]) - len(seeds["a"][1])
    y[t_a] = old[0][t_a]
    y[t_b] = old[0][t_b] + temperature_offset
    residual_old, data_old = old[3](y)
    jac_old = torch.func.jacfwd(lambda z: old[3](z)[0])(y)
    prepare(model, True)
    new = model._problem(states, seeds, 0.02, 4, model._commands(None))
    residual_new, data_new = new[3](y)
    jac_new = torch.func.jacfwd(lambda z: new[3](z)[0])(y)
    torch.testing.assert_close(residual_new, residual_old, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(jac_new, jac_old, rtol=2e-9, atol=2e-9)
    direction = torch.linspace(-0.01, 0.01, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (new[3](y + eps * direction)[0] - new[3](y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac_new @ direction, fd, rtol=2e-5, atol=1e-6)
    for a, b in zip(data_old["transfers"], data_new["transfers"]):
        for key in ("component_rates", "total_energy_rate", "entropy_rate", "area"):
            torch.testing.assert_close(a[key], b[key], rtol=2e-12, atol=2e-12)
    for vessel in model.vessels.values():
        for heights, weights in vessel.column._uniform_grids.values():
            assert not any(
                torch._C._functorch.is_functorch_wrapped_tensor(v) for v in (*heights, *weights)
            )


@pytest.mark.parametrize("order", [2, 4, 8])
def test_uniform_grid_is_exact_but_moving_interface_keeps_derivatives(order):
    model, _state = setup()
    column = model.vessels["a"].column
    empty = column.fluid.tensor([])
    cached = column._grid(empty, order)
    direct = column._dynamic_grid(empty, order, _prepare=False)
    for a, b in zip((*cached[0], *cached[1]), (*direct[0], *direct[1])):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    interface = column.fluid.tensor([0.4 * column.geometry.height_m])
    jac = torch.func.jacfwd(lambda h: torch.cat(column._grid(h, order)[1]))(interface)
    old_jac = torch.func.jacfwd(
        lambda h: torch.cat(column._dynamic_grid(h, order, _prepare=False)[1])
    )(interface)
    torch.testing.assert_close(jac, old_jac, rtol=0, atol=0)
    assert jac.abs().max() > 0


def test_no_unused_phase_density_geometry_in_closed_vessel_residual(monkeypatch):
    model, state = setup()
    vessel = model.vessels["a"]
    seeds = model._seeds(dict(state.vessels))
    initial, _lo, _hi, evaluate, *_rest = vessel._problem(
        dict(state.vessels)["a"], 0.01, seeds["a"], 4
    )

    def unnecessary(*args):
        pytest.fail(
            "Closed one-phase vessel should reuse its fixed quadrature, not calculate unused outlet density"
        )

    monkeypatch.setattr(vessel.column, "_volume", unnecessary)
    assert torch.isfinite(evaluate(initial)[0]).all()


def test_uniform_grid_never_caches_an_ad_graph_on_first_use():
    model, _state = setup()
    column = model.vessels["a"].column
    column._uniform_grids.clear()
    empty = torch.empty(0, dtype=torch.float64, requires_grad=True)
    column._grid(empty, 4)
    assert not column._uniform_grids
    torch.func.jacfwd(lambda h: torch.cat(column._grid(h[:0], 4)[0]))(
        torch.zeros(1, dtype=torch.float64)
    )
    assert not column._uniform_grids
    heights, weights = column._grid(empty.detach(), 4)
    assert column._uniform_grids
    assert all(not v.requires_grad and v.grad_fn is None for v in (*heights, *weights))


def test_uniform_grid_does_not_retain_fake_trace_tensors():
    from torch.fx.experimental.proxy_tensor import make_fx

    model, _state = setup()
    column = model.vessels["a"].column
    column._uniform_grids.clear()
    graph = make_fx(
        lambda h: torch.cat(column._grid(h[:0], 4)[0]),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(torch.zeros(1, dtype=torch.float64))
    assert not column._uniform_grids
    expected = torch.cat(column._dynamic_grid(column.fluid.tensor([]), 4)[0])
    torch.testing.assert_close(graph(torch.zeros(1, dtype=torch.float64)), expected, rtol=0, atol=0)


@pytest.mark.parametrize("temperature", [320.0, 196.2])
def test_prepared_inertial_column_coupling_preserves_jacobian_and_shares_endpoints(
    temperature, monkeypatch
):
    from test_surface_pr_gravity_transport import setup as gravity_setup

    from surface.pr_gravity_collocation import PRGravityCollocation

    model, state, commands = gravity_setup(temperature=temperature)
    solver = PRGravityCollocation(model, phase_fallback=False)
    calls = []
    endpoint = model.transport._endpoint

    def counted(*args, **kwargs):
        calls.append(1)
        return endpoint(*args, **kwargs)

    monkeypatch.setattr(model.transport, "_endpoint", counted)

    def problem(enabled):
        model._prepare_residual = enabled
        model.transport._prepare_endpoint = enabled
        for column in model.columns.values():
            column._prepare_geometry = enabled
        prepared, _coupling = solver._problem(state, 2, commands, {}, {}, {})
        initial, _lo, _hi, evaluate, rates, *_rest = prepared

        def row(z):
            storage, point = evaluate(z)
            return storage - 1e-6 * rates(point)

        return initial, row

    y, old = problem(False)
    y = y + torch.linspace(-1e-6, 1e-6, len(y), dtype=y.dtype)
    calls.clear()
    old_r = old(y)
    old_calls = len(calls)
    old_jac = torch.func.jacfwd(old)(y)
    _initial, new = problem(True)
    calls.clear()
    new_r = new(y)
    new_calls = len(calls)
    new_jac = torch.func.jacfwd(new)(y)
    torch.testing.assert_close(new_r, old_r, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(new_jac, old_jac, rtol=2e-9, atol=2e-9)
    if temperature == 196.2:
        assert new_calls < old_calls
