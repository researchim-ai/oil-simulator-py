"""Exactly empty owned phases, finite outflow traces and independent balances."""

import copy
import json

import pytest
import torch
from check_pr_capillary_axial import audit
from check_pr_capillary_pressure import audit as normal_audit
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_pr_active_multifluid_pipe import phase

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_pressure import PRCapillaryPressure


def setup(device="cpu", *, wetting=True):
    f = fluid(device=device)
    pipe = PRActiveMultifluidPipe(
        f,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [0.01] * 2,
        0.004,
        boundary_mode="walls",
    )
    v = pipe.volumes[0]
    liquid = phase(f, 1e6, 196.2, "liquid", v * 0.4, x=(0.5, 0.5))
    gas = phase(f, 1e6, 196.2, "vapor", v * 0.6, x=(0.5, 0.5))
    material = (
        {
            "adsorption_thickness_m": 1e-4,
            "equilibrium_contact_angle_deg": 5.0,
            "dry_extension": "c2_half_adsorption",
        }
        if wetting
        else None
    )
    source = PRCapillaryPressure(
        pipe, "liquid", "gas", 0.004, 0.02, 1e-12, phase_support="active", wetting=material
    )
    model = PRCapillaryAxialPipe(source, volume_diffusion="none")
    return model, pipe.initialize([{"liquid": liquid, "gas": gas}] * 2), liquid, gas


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nonisothermal_depletion_and_refill_have_exact_phase_registry_and_independent_work(
    device, tmp_path
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, old, liquid, gas = setup(device)
    initial = old.to_dict()
    inlet = phase(model.pipe.fluid, 1e6, 210.0, "vapor", gas.volume_m3, x=(0.5, 0.5))
    records = []
    for name, target, reference in [("gas", 0.0, inlet), ("liquid", 0.4, liquid)]:
        end, report = model.step(
            old,
            0.1,
            phase_controls={
                i: {"inlet_slot": name, "inlet_phase": reference, "target_liquid_fraction": target}
                for i in range(2)
            },
        )
        assert audit(model, old, end, report)["accepted"]
        assert report["maximum_scaled_residual"] <= 2e-12
        if name == "gas":
            assert report["newton_iterations"] > 0
            assert all(tuple(dict(cell)) == ("gas",) for cell in end.cells)
        else:
            assert all(set(dict(cell)) == {"liquid", "gas"} for cell in end.cells)
        normal, nr = model.source.apply(end, 1e-7)
        assert normal_audit(model.source, end, normal, nr)["accepted"]
        records.append({"before": old.to_dict(), "axial": report, "normal": nr})
        old = normal
    assert initial != old.to_dict()
    (tmp_path / ("active-cycle-" + device + ".json")).write_text(
        json.dumps(
            {"model": model.to_dict(), "initial": initial, "records": records},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def test_single_present_phase_has_no_liquid_eos_or_stock_and_tampered_ledger_is_rejected():
    model, _old, _, _gas = setup(wetting=False)
    v = model.pipe.volumes[0]
    only = model.pipe.initialize(
        [{"gas": phase(model.pipe.fluid, 1e6, 210.0, "vapor", v, x=(0.5, 0.5))}] * 2
    )
    end, report = model.step(only, 1e-6)
    assert all(tuple(dict(cell)) == ("gas",) for cell in end.cells)
    assert audit(model, only, end, report)["accepted"]
    altered = copy.deepcopy(report)
    altered["primitive_traces"]["component_densities_mol_m3"][0][1][0] *= 1.01
    with pytest.raises(ValueError):
        audit(model, only, end, altered)


def test_normal_source_skips_only_the_dry_cell_and_retains_global_surface_work():
    model, _old, _, _ = setup(wetting=False)
    f = model.pipe.fluid
    # Small holdup keeps the resolved long-wave slope admissible.
    v = model.pipe.volumes[0]
    mixed = model.pipe.initialize(
        [
            {"gas": phase(f, 1e6, 196.2, "vapor", v, x=(0.5, 0.5))},
            {
                "gas": phase(f, 1e6, 196.2, "vapor", v * 0.99, x=(0.5, 0.5)),
                "liquid": phase(f, 1.001e6, 196.2, "liquid", v * 0.01, x=(0.5, 0.5)),
            },
        ]
    )
    end, report = model.source.apply(mixed, 1e-7)
    assert report["active_cells"] == [1]
    assert mixed.cells[0] == end.cells[0]
    assert normal_audit(model.source, mixed, end, report)["accepted"]


def test_active_equations_agree_with_positive_support_for_contact_transport():
    from test_surface_pr_capillary_contact import setup as contact

    base, old = contact()
    raw = base.source.to_dict()
    raw["phase_support"] = "active"
    active = PRCapillaryAxialPipe(PRCapillaryPressure.from_dict(base.pipe, raw))
    a, _ra = base.step(old, 1e-6)
    b, rb = active.step(old, 1e-6)
    assert audit(active, old, b, rb)["accepted"]
    for ca, cb in zip(a.cells, b.cells):
        for (_, pa), (_, pb) in zip(ca, cb):
            assert pb.volume_m3 == pytest.approx(pa.volume_m3, rel=1e-9)
            assert pb.component_moles == pytest.approx(pa.component_moles, rel=1e-9)
            assert pb.total_energy_j == pytest.approx(pa.total_energy_j, rel=1e-9)


def test_real_spatial_transport_populates_exactly_empty_cells():
    model, _old, _, _ = setup(wetting=False)
    f = model.pipe.fluid
    p = PRActiveMultifluidPipe(
        f,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [0.01] * 3,
        0.004,
        boundary_mode="periodic",
    )
    model = PRCapillaryAxialPipe(
        PRCapillaryPressure(p, "liquid", "gas", 0.004, 1e-24, 1e-12, phase_support="active"),
        volume_diffusion="acoustic",
    )
    v = p.volumes[0]
    old = p.initialize(
        [
            {
                "gas": phase(f, 1e6, 196.2, "vapor", v * (1 - a), 0.3, x=(0.5, 0.5)),
                **(
                    {"liquid": phase(f, 1e6, 196.2, "liquid", v * a, 0.3, x=(0.5, 0.5))}
                    if a
                    else {}
                ),
            }
            for a in [0.1, 0.0, 0.0]
        ]
    )
    snapshot = old.to_dict()
    end, report = model.step(old, 1e-6)
    assert old.to_dict() == snapshot
    assert all(set(dict(cell)) == {"liquid", "gas"} for cell in end.cells)
    assert report["newton_iterations"] > 0
    assert not report["phase_controls"]
    assert audit(model, old, end, report)["accepted"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_active_pipe_cycle_inside_global_newton_graph_is_independently_audited(device, tmp_path):
    from pathlib import Path

    from run_pr_capillary_facility import run

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = json.loads(
        (
            Path(__file__).parents[1] / "configs/surface_pr_active_capillary_facility_cycle.json"
        ).read_text()
    )
    snapshot = copy.deepcopy(config)
    result = run(config, device=device)
    assert config == snapshot
    assert len(result["independent_audits"]) == 2
    for interval, count in zip(result["intervals"], (1, 2)):
        assert not interval["result"]["rejected"]
        for report in interval["result"]["steps"]:
            assert all(len(c) == count for c in report["state"]["pipes"]["out"]["cells"])
            assert report["schur_newton"]["jacobian_builds"] >= 1
            assert report["schur_newton"]["linear_solves"] >= 1
    (tmp_path / ("active-facility-" + device + ".json")).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def test_inlet_introduces_a_component_absent_from_the_entire_initial_pipe():
    model, _old, _, _ = setup(wetting=False)
    f = model.pipe.fluid
    pressure = f.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    v = model.pipe.volumes[0]
    old = model.pipe.initialize(
        [
            {
                "liquid": phase(f, pressure, 200.0, "liquid", v * 0.4, x=(0.0, 1.0)),
                "gas": phase(f, pressure, 200.0, "vapor", v * 0.6, x=(0.0, 1.0)),
            }
        ]
        * 2
    )
    inlet = phase(f, pressure, 240.0, "vapor", v, x=(0.5, 0.5))
    end, report = model.step(
        old,
        0.1,
        phase_controls={
            i: {"inlet_slot": "gas", "inlet_phase": inlet, "target_liquid_fraction": 0.0}
            for i in range(2)
        },
    )
    assert all(dict(cell)["gas"].component_moles[0] > 0 for cell in end.cells)
    assert audit(model, old, end, report)["accepted"]
    tampered = copy.deepcopy(report)
    tampered["sideport_transfer_n_m_e"][0] *= 1.001
    with pytest.raises(ValueError):
        audit(model, old, end, tampered)


def test_controlled_advance_reaches_clock_and_failed_control_does_not_mutate_checkpoint():
    model, old, _liquid, gas = setup()
    snapshot = old.to_dict()
    controls = {
        i: {"inlet_slot": "gas", "inlet_phase": gas, "target_liquid_fraction": 0.0}
        for i in range(2)
    }
    end, report = model.advance(
        old,
        0.1,
        initial_step_s=0.1,
        maximum_attempts=1,
        phase_controls=controls,
        normal_relaxation=True,
    )
    assert end.time_s == 0.1 and len(report["steps"]) == 1
    assert all(tuple(dict(cell)) == ("gas",) for cell in end.cells)
    controls[0]["target_liquid_fraction"] = 1.0
    with pytest.raises(ValueError, match="direction"):
        model.step(old, 0.1, phase_controls=controls)
    assert old.to_dict() == snapshot


def test_exactly_dry_checkpoint_restarts_without_reconstructing_phantom_stock():
    from surface.pr_active_multifluid_pipe import PRActiveMultifluidState

    model, old, liquid, gas = setup()
    dry, _ = model.step(
        old,
        0.1,
        phase_controls={
            i: {"inlet_slot": "gas", "inlet_phase": gas, "target_liquid_fraction": 0.0}
            for i in range(2)
        },
    )
    raw = json.loads(json.dumps(dry.to_dict()))
    restored = PRCapillaryAxialPipe.from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PRActiveMultifluidState.from_dict(raw)
    assert checkpoint == dry
    assert all(tuple(dict(cell)) == ("gas",) for cell in checkpoint.cells)
    controls = {
        i: {"inlet_slot": "liquid", "inlet_phase": liquid, "target_liquid_fraction": 0.4}
        for i in range(2)
    }
    expected, _ = model.step(dry, 0.1, phase_controls=controls)
    actual, report = restored.step(checkpoint, 0.1, phase_controls=controls)
    assert actual == expected
    assert json.loads(json.dumps(checkpoint.to_dict())) == raw
    assert audit(restored, checkpoint, actual, report)["accepted"]
