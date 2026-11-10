"""Transport must not compress phases merely because holdup changes."""

import math

import pytest
from check_pr_capillary_axial import audit
from test_surface_equilibrium_thermodynamics import fluid
from test_surface_pr_active_multifluid_pipe import phase

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_capillary_axial import PRCapillaryAxialPipe
from surface.pr_capillary_pressure import PRCapillaryPressure


def setup(velocity=0.3, fractions=(0.15, 0.35, 0.65, 0.45)):
    f = fluid()
    pipe = PRActiveMultifluidPipe(
        f,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [0.01] * 4,
        0.001,
        boundary_mode="periodic",
    )
    old = pipe.initialize(
        [
            {
                "liquid": phase(f, 1e6, 200.0, "liquid", v * a, velocity, x=(0.05, 0.95)),
                "gas": phase(f, 1e6, 300.0, "vapor", v * (1 - a), velocity, x=(0.95, 0.05)),
            }
            for v, a in zip(pipe.volumes, fractions)
        ]
    )
    # Negligible capillarity isolates contact transport while exercising the
    # actual joint integrator and its volume/momentum/energy equations.
    source = PRCapillaryPressure(pipe, "liquid", "gas", 0.001, 1e-24, 1e-12)
    model = PRCapillaryAxialPipe(source)
    return model, old


@pytest.mark.parametrize("velocity", [-0.3, 0.3])
def test_contact_changes_holdup_without_creating_intrinsic_density_or_pressure_errors(velocity):
    model, old = setup(velocity)
    end, report = model.step(old, 1e-6)
    assert audit(model, old, end, report)["accepted"]
    assert any(
        abs(dict(a)["liquid"].volume_m3 - dict(b)["liquid"].volume_m3) > 1e-12
        for a, b in zip(old.cells, end.cells)
    )
    reference = dict(old.cells[0])
    for cell in end.cells:
        for name, actual in cell:
            initial = reference[name]
            density = math.fsum(actual.component_moles) / actual.volume_m3
            expected = math.fsum(initial.component_moles) / initial.volume_m3
            assert density == pytest.approx(expected, rel=1e-10)
            assert actual.pressure_pa == pytest.approx(1e6, abs=1e-4)
            assert actual.temperature_k == pytest.approx(initial.temperature_k, abs=1e-8)


def test_actual_high_contrast_positivity_failure_cuts_step_without_losing_report_time():
    from surface.liquid import HydraulicConvergenceError

    model, old = setup(fractions=(0.01, 0.01, 0.01, 0.99))
    checkpoint = old.to_dict()
    with pytest.raises(HydraulicConvergenceError):
        model.step(old, 1e-4)
    final, report = model.advance(old, 1e-4, initial_step_s=1e-4, maximum_attempts=64)
    assert final.time_s == 1e-4
    assert report["rejected"]
    previous = old
    for step in report["steps"]:
        current = type(old).from_dict(step["state"])
        assert audit(model, previous, current, step)["accepted"]
        previous = current
    assert old.to_dict() == checkpoint
