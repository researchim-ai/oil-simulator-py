"""Per-stream knot sets preserve selected-table kinks and phase derivatives."""

import numpy as np
import pytest
import torch
from test_surface_blackoil_properties import DENSITIES, pvt_data

from surface.blackoil_pipe import BlackOilPipe
from surface.blackoil_pipe_batch import PipeBatch
from surface.blackoil_pipe_grid import get_pressure_grid
from surface.blackoil_properties import BlackOilMixture


def ragged_pvt(count=5):
    data = pvt_data()
    ratios = np.linspace(0, 80, count)
    assert 20.0 in ratios
    tables = {}
    for i, r in enumerate(ratios):
        pb = float(np.interp(r, data["Rs_m3m3"], data["pressure_MPa"]))
        # Each Rs row owns a distinct interpolation knot, even where the
        # physical Bo/mu functions happen to be linear.
        extra = 0.15 + 4.7 * (i + 1) / (count + 1)
        axis = sorted({0.1, 1.0, 5.0, extra})
        tables[str(float(r))] = {
            "pressure_MPa": axis,
            "Bo": [1 + 0.002 * r - 0.001 * (p - pb) for p in axis],
            "mu_o_cP": [5 - 0.04 * r + 0.01 * (p - pb) for p in axis],
        }
    data["extended"]["oil"] = {"rs_values": ratios.tolist(), "tables": tables}
    return data


class GlobalGrid:
    def __init__(self, model):
        self.model = model

    def fixed(self, masses):
        return self.model._fixed_pressures[None, :].expand(len(masses), -1)


def test_only_relevant_knots_and_common_axis_are_selected():
    model = BlackOilMixture(ragged_pvt(65), DENSITIES)
    grid = get_pressure_grid(model)
    assert grid is get_pressure_grid(model)
    assert grid.maximum_fixed_knots == 5
    assert len(model._fixed_values) > 60
    masses = torch.tensor([[0.85, 2, 0.036], [0.85, 2, 0.06]], dtype=torch.float64)
    fixed = grid.fixed(masses)
    assert fixed.shape == (2, 5)
    for i, ratio in enumerate([30.0, 50.0]):
        index = (
            int(
                torch.searchsorted(
                    model.oil_rs, torch.tensor(ratio, dtype=torch.float64), right=True
                )
            )
            - 1
        )
        tables = list(model.oil_tables.values())
        expected = sorted(
            set(
                model.pressures.tolist() + tables[index][0].tolist() + tables[index + 1][0].tolist()
            )
        )
        actual = sorted(v for v in fixed[i].tolist() if np.isfinite(v))
        assert actual == expected


@pytest.mark.parametrize("boundary,rates", [(2.72e6, [0.85, 2, 0.06]), (4e6, [0.85, 2, 0.036])])
def test_ragged_pipe_pressure_and_implicit_derivatives_match_global_reference(boundary, rates):
    model = BlackOilMixture(ragged_pvt(), DENSITIES)
    pipes = [BlackOilPipe("a", 300, 0.08, 30, 1e-5), BlackOilPipe("b", 230, 0.1, 10, 2e-5)]
    selected = PipeBatch(pipes, model)
    global_kernel = PipeBatch(pipes, model)
    global_kernel.pressure_grid = GlobalGrid(model)
    a = selected.solve([boundary] * 2, [rates] * 2)
    b = global_kernel.solve([boundary] * 2, [rates] * 2)
    scalar = pipes[0].required_inlet(boundary, rates, model)
    assert a[0]["inlet_pressure_pa"] == pytest.approx(scalar["inlet_pressure_pa"], abs=0.001)
    for x, y in zip(a, b):
        assert x["inlet_pressure_pa"] == pytest.approx(y["inlet_pressure_pa"], abs=0.001)
        assert x["pressure_derivatives"] == pytest.approx(
            y["pressure_derivatives"], rel=1e-7, abs=0.002
        )
        assert abs(x["mechanical_balance_error_w"]) < 1e-5


def test_selected_table_pressure_kink_is_not_discarded():
    data = ragged_pvt()
    row = data["extended"]["oil"]["tables"]["40.0"]
    axis = sorted(set(row["pressure_MPa"] + [4.1]))
    pb = 1.0 + (40.0 - 20.0) / 60.0 * 4.0
    row["pressure_MPa"] = axis
    row["Bo"] = [1 + 0.002 * 40 - 0.001 * (p - pb) - 0.003 * max(p - 4.1, 0) for p in axis]
    row["mu_o_cP"] = [5 - 0.04 * 40 + 0.01 * (p - pb) for p in axis]
    model = BlackOilMixture(data, DENSITIES)
    rates = torch.tensor([[0.85, 2, 0.036]], dtype=torch.float64, requires_grad=True)
    kernel = PipeBatch([BlackOilPipe("kink", 300, 0.08, 30)], model)
    assert 4.1 * 1e6 in kernel.pressure_grid.fixed(rates)[0].tolist()
    lower, upper = (
        torch.tensor([4e6], dtype=torch.float64),
        torch.tensor([4.5e6], dtype=torch.float64),
    )
    K = torch.zeros(1, dtype=torch.float64)
    a, _, _ = kernel.checked(lower, upper, rates, K)
    ga = torch.autograd.grad(a[:, 0].sum(), rates, retain_graph=True)[0]
    kernel.pressure_grid = GlobalGrid(model)
    b, _, _ = kernel.checked(lower, upper, rates, K)
    gb = torch.autograd.grad(b[:, 0].sum(), rates)[0]
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-8)
    torch.testing.assert_close(ga, gb, rtol=1e-8, atol=1e-7)


def test_grid_cache_roundtrip_and_new_device_snapshot_are_independent():
    import pickle

    model = BlackOilMixture(ragged_pvt(), DENSITIES)
    grid = get_pressure_grid(model)
    restored = pickle.loads(pickle.dumps(model))
    assert get_pressure_grid(restored) is restored._pipe_pressure_grid
    moved = model.to("cpu")
    assert get_pressure_grid(moved) is not grid
    masses = torch.tensor([[0.85, 2, 0.06]], dtype=torch.float64)
    torch.testing.assert_close(get_pressure_grid(moved).fixed(masses), grid.fixed(masses))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_stream_knot_selection_uses_device_local_compositions():
    model = BlackOilMixture(ragged_pvt(), DENSITIES).to("cuda")
    masses = torch.tensor([[0.85, 2, 0.06], [0.85, 2, 0.036]], dtype=torch.float64, device="cuda")
    result = get_pressure_grid(model).fixed(masses)
    assert result.is_cuda
    assert result.shape[0] == 2


def test_rich_pvto_trunk_uses_stream_grid_and_preserves_network_pressures():
    from test_surface_blackoil_network import feeds, network

    from surface.blackoil_network import BlackOilProductionNetwork

    spec = network().to_dict()
    spec["pvt"] = ragged_pvt(65)
    scalar = BlackOilProductionNetwork.from_dict(spec)
    spec["hydraulic_backend"] = "torch_batch"
    selected = BlackOilProductionNetwork.from_dict(spec)
    a, b = scalar.solve(feeds()), selected.solve(feeds())
    assert b["hydraulic_execution"]["torch_batch_pipes"] == 3
    assert b["hydraulic_execution"]["scalar_retry_pipes"] == 0
    assert a["node_pressures_pa"] == pytest.approx(b["node_pressures_pa"], abs=0.001)
