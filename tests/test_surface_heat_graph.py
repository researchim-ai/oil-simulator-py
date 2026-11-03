"""Reusable CUDA linear-step graphs retain exact eager audits and ownership."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_thermal_liquid import pipe_probe

from surface import LiquidHeatNetwork, ThermalLiquidStation
from surface.torch_backend import liquid_backend_config

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification required")


def probe(cells, temporal):
    model, _, rec = pipe_probe(cells=cells, dt=0.0005, duration=0.002)
    data = model.to_dict()
    data["device"] = data["physics"]["device"] = "cuda:0"
    data["temporal_error_control"] = temporal
    model = LiquidHeatNetwork.from_dict(data)
    initial = model.initialize(model.physics.initialize())
    records = []
    time = 0.0
    for i, duration in enumerate((0.002, 0.001, 0.003, 0.0015)):
        record = copy.deepcopy(rec)
        record.update(time_s=time, dt_s=duration)
        q = 0.001 if i % 2 else -0.001
        record["midpoint_branch_volume_rates_m3_s"] = {"line": q, "v": q}
        record["boundary_mass_transfers_in_kg"] = {
            "in": 1000 * q * duration,
            "out": -1000 * q * duration,
        }
        # Manufactured heat input tests deposition; not a momentum oracle.
        record["friction_heat_j"] = 0.01
        record["boundary_work_in_j"] = 0.01
        record["dissipation_heat_j"]["friction"]["line"] = 0.01
        records.append(record)
        time += duration
    return model, initial, records


def physical(result):
    return {k: v for k, v in result.items() if k != "transfer_stats"}


@CUDA
@pytest.mark.parametrize("cells", [4, 80])
@pytest.mark.parametrize("temporal", [False, True])
def test_graph_matches_every_eager_record_with_changed_inputs(cells, temporal):
    model, initial, records = probe(cells, temporal)
    eager = model.advance_records(initial, records, execution="eager")
    graph = model.advance_records(initial, records, execution="graph")
    assert physical(graph) == physical(eager)
    assert graph["transfer_stats"]["linear_graph_replays"] > 0
    assert graph["transfer_stats"]["linear_graph_captures"] == 1
    again = model.advance_records(initial, records, execution="graph")
    assert physical(again) == physical(eager)
    assert again["transfer_stats"]["linear_graph_captures"] == 0
    # The first returned histories must not alias later graph replays.
    assert physical(graph) == physical(again)
    assert len(model._torch_bridge.linear_graphs.entries) == 1
    json.dumps(physical(graph), allow_nan=False)


@CUDA
def test_graph_failed_window_is_retryable_and_never_overwrites_snapshot():
    model, initial, records = probe(80, True)
    session = model.open_session(initial, execution="graph")
    frozen = session.snapshot().to_dict()
    bad = copy.deepcopy(records)
    bad[-1]["friction_heat_j"] += 1
    from surface import ConservationError

    with pytest.raises(ConservationError, match="ledger"):
        session.advance_records(bad)
    assert session.snapshot().to_dict() == frozen
    graph = session.advance_records(records)
    eager = model.advance_records(initial, records, execution="eager")
    assert graph["steps"] == eager["steps"]
    assert graph["audits"] == eager["audits"]
    assert session.close().to_dict() == eager["state"]


@CUDA
def test_graph_complete_controller_window_preserves_trips_and_events():
    config = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    config["system"]["temperature_trips"][0]["threshold_k"] = 300.0001
    config["system"]["temperature_trips"][0]["delay_s"] = 0
    model = ThermalLiquidStation.from_dict(
        liquid_backend_config(config["system"], "torch", "cuda:0")
    )
    initial = model.initialize(**config["initial"])
    commands = {k: v for k, v in config["periods"][0].items() if k != "duration_s"}
    eager = model.advance(initial, 0.12, thermal_execution="eager", **commands)
    graph = model.advance(initial, 0.12, thermal_execution="graph", **commands)
    assert graph == eager
    assert graph["temperature_events"]


def test_graph_mode_does_not_silently_fall_back_to_cpu():
    model, initial, record = pipe_probe()
    with pytest.raises(ValueError, match="requires CUDA"):
        model.advance_record(initial, record, execution="graph")
    with pytest.raises(ValueError, match="thermal_execution"):
        model.advance_records(initial, [record], execution="invalid")
    with pytest.raises(ValueError, match="requires CUDA"):
        model.open_session(initial, execution="graph")


@CUDA
@pytest.mark.parametrize("cells", [4, 80])
def test_graph_variable_mass_matches_eager_unfactored_solve(cells):
    model, initial, records = probe(cells, True)
    for i, record in enumerate(records):
        record["final_pressures_pa"] = {
            n: p + 10 * (i + 1) for n, p in record["final_pressures_pa"].items()
        }
    eager = model.advance_records(initial, records, execution="eager")
    graph = model.advance_records(initial, records, execution="graph")
    assert physical(graph) == physical(eager)
    assert graph["transfer_stats"]["linear_graph_replays"] > 0
    assert graph["linear_factorizations"] == eager["linear_factorizations"]


@CUDA
def test_graph_cache_is_bounded_and_does_not_reuse_changed_iteration_budget():
    model, initial, records = probe(80, False)
    model.advance_record(initial, records[0], execution="eager")
    kernel = model._torch_bridge.kernel
    for count in range(30, 36):
        kernel.iterations = count
        eager = model.advance_record(initial, records[0], execution="eager")
        graph = model.advance_record(initial, records[0], execution="graph")
        assert graph == eager
    cache = model._torch_bridge.linear_graphs
    assert cache.captures == 6
    assert len(cache.entries) == cache.capacity == 4


@CUDA
def test_capture_failure_is_explicit_and_preserves_input_and_operator(monkeypatch):
    model, initial, records = probe(4, True)
    expected = model.advance_record(initial, records[0], execution="eager")
    bridge = model._torch_bridge
    before = initial.to_dict()
    operator_values = bridge.kernel.operator.values

    def fail_capture():
        raise RuntimeError("capture failure sentinel")

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "CUDAGraph", fail_capture)
        with pytest.raises(RuntimeError, match="capture failure sentinel"):
            model.advance_record(initial, records[0], execution="graph")
    assert initial.to_dict() == before
    assert not bridge.linear_graphs.entries
    # compile_operator refreshed the live coefficients. The graph's private
    # packed input must not replace these coefficients after failure.
    assert bridge.kernel.operator.values.shape == operator_values.shape
    assert model.advance_record(initial, records[0], execution="graph") == expected


@CUDA
def test_explicit_graph_schedule_restart_and_execution_fingerprint():
    from surface import run_thermal_liquid_schedule

    config = json.loads((ROOT / "configs/surface_thermal_liquid.json").read_text())
    model = ThermalLiquidStation.from_dict(
        liquid_backend_config(config["system"], "torch", "cuda:0")
    )
    initial = model.initialize(**config["initial"])
    periods = [
        {
            **config["periods"][0],
            "duration_s": 0.01,
            "thermal_execution": "graph",
            "hydraulic_heat_transfer": "device",
        }
        for _ in range(3)
    ]
    full = run_thermal_liquid_schedule(model, initial, periods)
    prefix = run_thermal_liquid_schedule(model, initial, periods, stop_after=1)
    resumed = run_thermal_liquid_schedule(model, initial, periods, checkpoint=prefix)
    assert resumed == full
    with pytest.raises(ValueError, match="fingerprint"):
        run_thermal_liquid_schedule(
            model,
            initial,
            [{**p, "thermal_execution": "eager"} for p in periods],
            checkpoint=prefix,
        )


@CUDA
def test_tiny_energy_journal_keeps_eager_reduction_alignment():
    # Odd-sized sparse fields expose one-ulp reduction changes when a packed
    # delta starts at an unaligned offset. Temperature equality alone misses it.
    from check_liquid_transient import wave_model, wave_reference

    from surface import LiquidTransientNetwork
    from surface.torch_heat_session import TorchHeatWindow

    results = []
    for execution in ("eager", "graph"):
        physics = LiquidTransientNetwork.from_dict(
            {**wave_model(80).to_dict(), "device": "cuda:0", "temporal_error_control": True}
        )
        heat = LiquidHeatNetwork(
            physics,
            4200,
            {"left": 310, "right": 300},
            reference_temperature_k=300,
            maximum_dt_s=physics.maximum_dt_s,
            temporal_error_control=False,
            store_step_temperatures=False,
        )
        initial = physics.initialize(dict(zip(physics.free, wave_reference(80, 0))))
        session = heat.open_session(heat.initialize(initial), execution=execution)
        window = TorchHeatWindow(session, None, None, {}, 100000)
        hydraulic = physics.advance(initial, 0.0002, _on_accepted_step=window.push)
        thermal = window.finish()
        results.append(
            {
                "hydraulic": hydraulic,
                "state": session.close().to_dict(),
                "steps": thermal["steps"],
                "audits": thermal["audits"],
            }
        )
    assert 0 < abs(results[0]["steps"][0]["fluid_energy_change_j"]) < 1e-10
    assert results[1] == results[0]
