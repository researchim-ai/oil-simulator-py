"""Numerical backend contracts: no SciPy solve, real device work, conservative transfer."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_liquid_transient import coupled_model
from check_thermal_liquid import pipe_probe

from surface import LiquidHeatNetwork, LiquidTransientNetwork, ThermalLiquidStation
from surface.liquid import HydraulicConvergenceError
from surface.torch_heat import TorchHeatEquations, TorchHeatState
from surface.torch_heat_interval import TorchHeatInterval
from surface.torch_linear import ColoredJacobian, TorchIncidence, TorchNewton, checked_device
from surface.torch_liquid import TorchLiquidEquations

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def model_for(device="cpu", backend="torch"):
    data = coupled_model().to_dict()
    data.update(device=device, backend=backend)
    return LiquidTransientNetwork.from_dict(data)


def native_args(e):
    return tuple(
        e.tensor(v)
        for v in (
            [550000, 500000],
            [0.002],
            [100, 110],
            0.001,
            [5, 7],
            [400000, 550000, 500000, 480000],
            [1],
        )
    )


@pytest.mark.parametrize("device", DEVICES)
def test_transient_root_matches_explicit_scipy_reference(device):
    reference = model_for(backend="scipy_reference")
    m = model_for(device)
    p, q, w = map(np.asarray, ([550000, 500000], [0.002], [100, 110]))
    fixed, valves = reference._controls(None, None)
    ref, _, _, _ = reference._solve(p, q, w, 0.001, np.asarray([5, 7]), fixed, valves)
    e = TorchLiquidEquations(m, device)
    x, residual, data, ok, _ = e.solve(*native_args(e))
    assert bool(ok)
    assert float(residual.abs().max()) < 1e-9
    for actual, expected in zip(data[:9], ref[:9]):
        torch.testing.assert_close(
            actual.cpu(), torch.as_tensor(expected, dtype=torch.float64), rtol=1e-8, atol=1e-7
        )
    metrics = e.mechanical_ledger(*native_args(e)[:4], data, residual)
    assert bool(metrics["conservation_valid"])
    assert abs(float(metrics["energy_residual_j"])) < 1e-5
    # A second replay must use the NEW inputs, not captured old values.
    a = list(native_args(e))
    a[4] = e.tensor([6, 4])
    second = e.solve(*a)
    assert bool(second[3])
    assert not torch.equal(x, second[0])


@pytest.mark.parametrize("device", DEVICES)
def test_heat_reference_and_resident_interval(device):
    m, initial, record = pipe_probe(cells=80, dt=0.0005, duration=0.004)
    d = m.to_dict()
    d["backend"] = "scipy_reference"
    d["device"] = "cpu"
    reference = LiquidHeatNetwork.from_dict(d)
    result = reference.advance_record(reference.initialize(reference.physics.initialize()), record)
    e = TorchHeatEquations(m, device)
    state = TorchHeatState(
        e.tensor([dict(initial.temperatures_k)[n] for n in m.names]),
        e.tensor([dict(initial.masses_kg)[n] for n in m.names]),
    )
    frozen = state.temperature.clone()
    flows = e.tensor([record["midpoint_branch_volume_rates_m3_s"][n] for n in e.branch_names])
    out = TorchHeatInterval(e).advance(state, state.mass, flows, e.tensor([0.0, 0.0]), 0.004, 8)
    expected = e.tensor([result["state"]["temperatures_k"][n] for n in m.names])
    torch.testing.assert_close(out.state.temperature, expected, rtol=0, atol=2e-11)
    assert out.status_reads == 1
    assert out.graph_replays == (8 if device.startswith("cuda") else 0)
    assert abs(float(out.audit[5])) < 1e-6
    assert float(out.audit[6]) < 1e-6
    assert torch.equal(state.temperature, frozen)
    # Depletion is transactional; it cannot partially overwrite the input.
    with pytest.raises(HydraulicConvergenceError, match="rejected"):
        TorchHeatInterval(e).advance(state, -state.mass, flows, e.tensor([0.0, 0.0]), 0.004, 8)
    assert torch.equal(state.temperature, frozen)
    assert e.identity is None  # larger than the explicitly bounded dense limit


@pytest.mark.parametrize("device", DEVICES)
def test_incidence_derivatives_and_colored_sparse_newton(device):
    incidence = TorchIncidence(3, [0, 1], [1, 2], device)
    rates = torch.tensor([2.0, 3.0], dtype=torch.float64, device=device)
    torch.testing.assert_close(incidence.mv(rates), rates.new_tensor([2.0, 1.0, -3.0]))
    J = torch.func.jacfwd(incidence.mv)(rates)
    torch.testing.assert_close(J, rates.new_tensor([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]]))
    size = 140
    rows = list(range(size))
    cols = list(range(size))

    def f(x):
        return x + 0.05 * x.square() - 1

    x = torch.zeros(size, dtype=torch.float64, device=device)
    coloring = ColoredJacobian(size, rows, cols, device)
    assert coloring.seeds.shape == (1, size)
    torch.testing.assert_close(coloring.values(f, x), torch.ones_like(x))
    solver = TorchNewton(device, maximum_iterations=6)
    _out, residual, success, _ = solver.solve(f, x, x - 10, x + 10, (rows, cols))
    assert bool(success)
    assert float(residual.abs().max()) < 1e-10
    assert solver.matrix.values.numel() == size
    assert solver.coloring.seeds.shape == (1, size)


def test_device_and_precision_contracts(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        checked_device("cuda:0")
    e = TorchLiquidEquations(model_for())
    args = list(native_args(e))
    args[0] = args[0].float()
    with pytest.raises(ValueError, match="float64"):
        e.solve(*args)
    with pytest.raises(ValueError, match="CPU-only"):
        model_for("cuda:0", "scipy_reference")


@pytest.mark.parametrize("device", DEVICES)
def test_full_automatic_station_does_not_call_scipy_solvers(monkeypatch, device):
    import scipy.optimize
    import scipy.sparse.linalg

    import surface.liquid_heat as heat
    import surface.liquid_transient as transient
    from surface import liquid

    config = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    from surface.torch_backend import liquid_backend_config

    station = ThermalLiquidStation.from_dict(
        liquid_backend_config(config["system"], "torch", device)
    )
    state = station.initialize(**config["initial"])

    def forbidden(*args, **kwargs):
        raise AssertionError("Hidden SciPy numerical solver in Torch path")

    for module in (scipy.optimize, scipy.sparse.linalg, liquid, transient, heat):
        for name in ("least_squares", "root", "brentq", "splu", "spsolve"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    result = station.advance(state, 0.001, mode="auto")
    assert result["steps"]
    assert result["thermal_steps"]
    assert station.recycling.physics.backend == "torch"


@pytest.mark.parametrize("device", DEVICES)
def test_short_coupled_thermal_station_matches_reference(device):
    from surface.torch_backend import liquid_backend_config

    config = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    ref = ThermalLiquidStation.from_dict(
        liquid_backend_config(config["system"], "scipy_reference", "cpu")
    )
    native = ThermalLiquidStation.from_dict(
        liquid_backend_config(config["system"], "torch", device)
    )
    commands = {k: v for k, v in config["periods"][0].items() if k != "duration_s"}
    expected = ref.advance(ref.initialize(**config["initial"]), 0.01, **commands)
    actual = native.advance(native.initialize(**config["initial"]), 0.01, **commands)
    assert actual["state"]["heat"]["temperatures_k"] == pytest.approx(
        expected["state"]["heat"]["temperatures_k"], abs=2e-8
    )
    assert actual["state"]["heat"]["wall_temperatures_k"] == pytest.approx(
        expected["state"]["heat"]["wall_temperatures_k"], abs=2e-8
    )
    assert all(abs(item["combined_energy_residual_j"]) < 1e-5 for item in actual["thermal_audits"])


@pytest.mark.parametrize("device", DEVICES)
def test_thermal_iteration_failure_is_rejected_without_dense_fallback(device):
    model, state, record = pipe_probe(cells=80, dt=0.001, duration=0.001)
    e = TorchHeatEquations(model, device, iterations=1, dense_limit=0)
    t, m = dict(state.temperatures_k), dict(state.masses_kg)
    s = TorchHeatState(e.tensor([t[n] for n in model.names]), e.tensor([m[n] for n in model.names]))
    flow = e.tensor([record["midpoint_branch_volume_rates_m3_s"][n] for n in e.branch_names])
    with pytest.raises(HydraulicConvergenceError, match="rejected"):
        TorchHeatInterval(e).advance(s, s.mass, flow, e.tensor([0.0, 0.0]), 0.001, 1)
    assert e.identity is None


def test_backend_config_copy_and_source_fingerprints():
    from surface.torch_backend import TORCH_SOURCES, liquid_backend_config, runtime_fingerprint

    config = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())["system"]
    original = json.dumps(config, sort_keys=True)
    override = liquid_backend_config(config, "torch", "cpu")
    assert json.dumps(config, sort_keys=True) == original
    station = ThermalLiquidStation.from_dict(override)
    assert station.heat.device == station.recycling.physics.device == "cpu"
    assert "torch_heat_interval.py" in TORCH_SOURCES
    assert all((ROOT / "src/surface" / name).is_file() for name in TORCH_SOURCES)
    assert runtime_fingerprint()["torch"] == str(torch.__version__)


def test_sparse_physical_acoustic_wave_matches_independent_mode():
    from check_liquid_transient import wave_model, wave_reference

    model = wave_model(80)
    initial = model.initialize(dict(zip(model.free, wave_reference(80, 0))))
    result = model.advance(initial, 0.0001)
    actual = np.asarray([result["state"]["pressures_pa"][n] for n in model.free])
    assert max(abs(actual - wave_reference(80, 0.0001))) < 1e-4
    assert result["failed_steps"] == 0
    assert max(abs(s["energy_residual_j"]) for s in result["steps"]) < 1e-8
    solver = model._torch_bridge.equations.newton
    assert solver.matrix.size == 161
    assert solver.matrix.values.numel() < 4 * solver.matrix.size
    assert solver.coloring.seeds.shape[0] <= 4


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_operator_empty_duplicate_and_batched_rows(device):
    from surface.torch_linear import TorchCOO

    rows, cols = [3, 0, 3, 2, 0, 3], [1, 0, 1, 4, 2, 5]
    values = torch.tensor([3.0, -2.0, 4.0, 1.0, -3.0, 2.0], dtype=torch.float64, device=device)
    operator = TorchCOO(rows, cols, values, 6, device)
    dense = torch.zeros((6, 6), dtype=torch.float64, device=device)
    for r, c, value in zip(rows, cols, values):
        dense[r, c] += value
    batch = torch.arange(18, dtype=torch.float64, device=device).reshape(3, 6)
    torch.testing.assert_close(operator.mv(batch), batch @ dense.T)
    torch.testing.assert_close(operator.mv(batch[0]), dense @ batch[0])
    torch.testing.assert_close(operator.diagonal(), dense.diagonal())
    torch.testing.assert_close(
        operator.row_scale(), values.new_tensor([5.0, 1e-12, 1.0, 9.0, 1e-12, 1e-12])
    )


@pytest.mark.parametrize("device", DEVICES)
def test_current_backend_checkpoint_replays_identically(device):
    from surface import run_thermal_liquid_schedule
    from surface.torch_backend import liquid_backend_config

    config = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    model = ThermalLiquidStation.from_dict(liquid_backend_config(config["system"], "torch", device))
    initial = model.initialize(**config["initial"])
    periods = [{**config["periods"][0], "duration_s": h} for h in (0.01, 0.015)]
    full = run_thermal_liquid_schedule(model, initial, periods)
    prefix = run_thermal_liquid_schedule(model, initial, periods, stop_after=1)
    resumed = run_thermal_liquid_schedule(model, initial, periods, checkpoint=prefix)
    assert resumed == full
