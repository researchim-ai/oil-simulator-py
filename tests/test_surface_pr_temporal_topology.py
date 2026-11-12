"""A real disappearing column layer cannot vanish from the time-error norm."""

from dataclasses import replace

from test_surface_pr_capillary_reuse import shared_case

from surface.stratified_pr import StratifiedPRState


def test_disappearing_interface_is_detected_even_with_loose_continuous_tolerances():
    model, initial = shared_case(temperature_tolerance_k=1e6, relative_tolerance=1e6)
    column = model.graph.columns["A"]
    old = dict(initial.columns)["A"]
    hot = StratifiedPRState.from_dict(
        column.equilibrate(
            old.component_moles,
            old.total_energy_j + 1e4,
            pressure_guess_pa=1e6,
            temperature_guess_k=old.temperature_k,
        )["state"]
    )
    assert len(old.phase_roots) == 2 and len(hot.phase_roots) == 1
    assert hot.component_moles == old.component_moles
    end = replace(initial.surface, columns=(("A", hot),))
    model.graph.inspect(end)
    # All remaining continuous terms are deliberately below one. Before the
    # guard the empty interface zip made this physical event acceptable.
    continuous = abs(hot.temperature_k - old.temperature_k) / 1e6
    assert continuous < 1
    assert model.graph._error(initial.surface, end) >= 2
