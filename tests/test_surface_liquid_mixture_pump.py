"""Dense liquid mixture PS/PH work, exact component transport and device AD."""

import copy
import json
from pathlib import Path

import pytest
import torch
from test_surface_pr_pump_connection import setup

from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork


def mixture_setup(*, dense_limit=128, backpressure=1e6):
    model, _, initial = setup(dense_limit=dense_limit, backpressure=backpressure, driven=True)
    config = model.to_dict()
    config["pump_connections"][0]["equilibrium_liquid_mixture"] = True
    model = PREquilibriumTransportNetwork.from_dict(config)
    for n in initial:
        initial[n]["mole_fractions"] = [0.0, 0.5, 0.5]
    return model, model.initialize(initial), initial


@pytest.mark.parametrize("dense_limit", [1, 128])
@pytest.mark.parametrize("backpressure", [1e6, 1.1e6])
def test_liquid_mixture_work_and_independent_audit(dense_limit, backpressure):
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    model, old, initial = mixture_setup(dense_limit=dense_limit, backpressure=backpressure)
    assert all(len(s.phase_roots) == 2 for _, s in old.nodes)
    cfg = {
        "model": model.to_dict(),
        "initial": {"nodes": initial},
        "schedule": [{"duration_s": 1e-5}],
    }
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    row = result["intervals"][0]["steps"][0]["pump_connections"]["booster"]
    if backpressure > 1e6:
        assert row["valve_closed"]
        assert row["shaft_power_w"] == row["flow_m3_s"] == 0
    else:
        assert row["shaft_power_w"] > row["hydraulic_power_w"] > 0
        assert row["entropy_production_w_k"] > 0
        # Both liquids transported: water may not be silently selected as phase zero.
        after = result["intervals"][0]["state"]["nodes"]["feed"]
        before = dict(old.nodes)["feed"]
        removed = [before.component_moles[i] - after["component_moles"][i] for i in (1, 2)]
        assert min(removed) > 0
        assert removed[0] == pytest.approx(removed[1], rel=1e-5)
    corrupted = copy.deepcopy(result)
    trace = corrupted["intervals"][0]["steps"][0]["pump_connections"]["booster"][
        "equilibrium_traces"
    ][0]
    trace["phase_fractions"][0] += 0.01
    with pytest.raises(ValueError):
        audit(cfg, corrupted)


def test_mixture_global_jacobian_and_sparsity():
    model, state, _ = mixture_setup(dense_limit=1)
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, offsets, *_ = model._problem(
        state, seeds, 1e-5, model.default_openings, [0.0, 0.0]
    )
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    mask = torch.zeros_like(jac, dtype=torch.bool)
    support = (
        set(zip(*model._sparsity(offsets, model.default_openings))) | evaluate.additional_sparsity
    )
    for r, c in support:
        mask[r, c] = True
    assert not bool((jac[~mask].abs() > 1e-12).any())
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=3e-4, atol=1e-7)


def test_legacy_model_hash_unchanged_without_opt_in():
    model, _, _ = setup()
    assert "equilibrium_liquid_mixture" not in model.to_dict()["pump_connections"][0]
    restored = type(model).from_dict(model.to_dict())
    assert restored.model_id == model.model_id


def test_gas_is_not_accepted_as_a_liquid_mixture():
    from surface.liquid import EquipmentOperatingError
    from surface.pr_pump_connection import PumpConnectionBundle

    model, _, initial = mixture_setup()
    initial["feed"]["mole_fractions"] = [1.0, 0.0, 0.0]
    old = model.initialize(initial)
    with pytest.raises(EquipmentOperatingError, match="every inlet phase to be dense"):
        PumpConnectionBundle(model, old, model.default_openings, 0)


def test_single_liquid_opt_in_matches_existing_machine():
    model, old, _ = setup(driven=True)
    d = model.to_dict()
    d["pump_connections"][0]["equilibrium_liquid_mixture"] = True
    mixture = type(model).from_dict(d)
    from dataclasses import replace

    mixture_old = replace(old, model_id=mixture.model_id)
    _, reference = model._step(old, 1e-5, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    _, actual = mixture._step(
        mixture_old, 1e-5, mixture.default_openings, [0.0, 0.0], [350.0, 350.0]
    )
    for key, value in reference["pump_connections"]["booster"].items():
        if not isinstance(value, bool):
            assert actual["pump_connections"]["booster"][key] == pytest.approx(
                value, rel=2e-8, abs=1e-8
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_liquid_mixture_cpu_cuda_state_and_trace_parity():
    cpu, state, _ = mixture_setup(dense_limit=1)
    gpu = type(cpu).from_dict(cpu.to_dict(), device="cuda")
    results = []
    for model in (cpu, gpu):
        new, report = model._step(state, 1e-5, model.default_openings, [0.0, 0.0], [350.0, 350.0])
        values = []
        for _, stock in new.nodes:
            values.extend(
                (
                    *stock.component_moles,
                    stock.pressure_pa,
                    stock.temperature_k,
                    stock.total_energy_j,
                )
            )
        row = report["pump_connections"]["booster"]
        values.extend(row[k] for k in ("flow_m3_s", "shaft_power_w", "entropy_production_w_k"))
        for trace in row["equilibrium_traces"]:
            values.extend(trace["phase_fractions"])
            for composition in trace["phase_mole_fractions"]:
                values.extend(composition)
        results.append(torch.tensor(values, dtype=torch.float64))
    torch.testing.assert_close(*results, rtol=1e-8, atol=1e-9)


def test_unstable_single_trace_is_replaced_in_common_solve():
    model, state, _ = mixture_setup()
    f = model.fluid
    source = model._state_data(model.names.index("feed"), dict(state.nodes)["feed"])
    z = source["n"] / source["n"].sum()
    homogeneous = f.tensor_homogeneous(source["p"], source["t"], z, "liquid")
    seed = {
        "pressure_pa": source["p"],
        "temperature_k": source["t"],
        "phase_count": 1,
        "phases": [{**homogeneous, "molar_fraction": f.tensor(1.0), "mole_fractions": z}],
    }
    _, report = model._step(
        state,
        1e-5,
        model.default_openings,
        [0.0, 0.0],
        [350.0, 350.0],
        _pump_connection_seeds={"booster": [seed, seed, seed]},
    )
    row = report["pump_connections"]["booster"]
    assert all(len(t["phase_fractions"]) == 2 for t in row["equilibrium_traces"])
    assert row["shaft_power_w"] > 0


def test_existing_positional_pump_arguments_keep_their_meaning():
    from surface.pr_pump_connection import PRPumpConnection

    model, _, _ = setup()
    unit = PRPumpConnection(
        model.pump_connections[0].card, 0.0, 400.0, 0.01, None, None, 0.5, None, None
    )
    assert unit.signed_map is None and unit.mapped_speed_rad_s is None
    assert unit.equilibrium_liquid_mixture is False


@pytest.mark.parametrize("backpressure", [1e6, 1.1e6])
def test_roundoff_is_restricted_to_active_enthalpy_rows(backpressure):
    model, state, _ = mixture_setup(backpressure=backpressure)
    seeds = [model._seed(i, stock) for i, (_, stock) in enumerate(state.nodes)]
    y, _, _, evaluate, *_ = model._problem(state, seeds, 1e-5, model.default_openings, [0.0, 0.0])
    _, physical = evaluate(y)
    bounds = physical["roundoff_bounds"]
    offset = evaluate.pump_connections.entries[0][3]
    allowed = {offset} if backpressure > 1e6 else {offset, offset + 2, offset + 3}
    nonzero = set(torch.nonzero(bounds).flatten().cpu().tolist())
    assert nonzero == allowed
    assert bool(torch.isfinite(bounds).all()) and bool((bounds >= 0).all())
    assert float(bounds.max()) <= 1e-10


def test_trace_branch_predictor_preserves_stock_and_obeys_explicit_rebranch():
    from surface.pr_liquid_mixture_pump import LiquidMixtureTrace
    from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState

    cfg = json.loads(
        (
            Path(__file__).parents[1] / "configs/surface_pr_integrated_facility_liquid_mixture.json"
        ).read_text()
    )
    full = PRControlledSurface.from_dict(cfg["model"])
    state = PRControlledSurfaceState.from_dict(cfg["initial"]["state"])
    model = full.plant.transport
    before = state.to_dict()
    unit = next(p for p in model.pump_connections if p.name == "water_pump")
    i = model.names.index(unit.source)
    source = model._state_data(i, dict(state.plant.transport.nodes)[unit.source])
    helper = LiquidMixtureTrace(model, unit, source)
    assert helper.base_seed["phase_count"] == 2
    assert all(t.count == 1 for t in helper.traces)
    z = source["n"] / source["n"].sum()
    for trace in helper.traces:
        _, _, _, _, beta, x, _ = trace.values(trace.initial, source["p"], z)
        torch.testing.assert_close((beta[:, None] * x).sum(0), z, rtol=0, atol=0)
    explicit = LiquidMixtureTrace(model, unit, source, [helper.base_seed] * 3)
    assert all(t.count == 2 for t in explicit.traces)
    assert state.to_dict() == before


def test_thermodynamic_guess_reduces_machine_residual_without_changing_stocks(monkeypatch):
    from surface.pr_pump_connection import PumpConnectionBundle

    model, state, _ = setup(driven=True)
    before = state.to_dict()
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    norms = []
    for enabled in (False, True):
        monkeypatch.setattr(PumpConnectionBundle, "_thermodynamic_predictor", enabled)
        y, _, _, evaluate, *_ = model._problem(
            state, seeds, 1e-5, model.default_openings, [0.0, 0.0]
        )
        row, _ = evaluate(y)
        offset = evaluate.pump_connections.entries[0][3]
        norms.append(float(row[offset : offset + 5].abs().max()))
    assert norms[1] < norms[0] / 100
    assert state.to_dict() == before
