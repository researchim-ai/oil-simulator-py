"""Explicit journal projection; numerical acceptance is independent of export policy."""

FULL_ONLY_FIELDS = frozenset(
    {
        "midpoint_pressures_pa",
        "final_pressures_pa",
        "final_pipe_volume_rates_m3_s",
        "final_speeds_rad_s",
        "boundary_mass_transfers_in_kg",
        "dissipation_heat_j",
    }
)
METRIC_ARRAYS = frozenset(
    {
        "unit_shaft_energy_residuals_j",
        "boundary_mass_transfers_in_kg",
        "friction_unit_heat_j",
        "pump_unit_heat_j",
        "drag_unit_heat_j",
        "motor_unit_heat_j",
    }
)
METRIC_GUARDS = frozenset(
    {
        "max_speed_change_rad_s",
        "minimum_speed_rad_s",
        "max_pressure_change_pa",
        "conservation_valid",
    }
)


def journal_branches(physics, mode, branches):
    if mode not in ("full", "compact"):
        raise ValueError("hydraulic_journal must be full or compact")
    if branches is None:
        branches = ()
    if not isinstance(branches, (list, tuple)) or any(not isinstance(n, str) for n in branches):
        raise ValueError("Journal branches require a list or tuple of branch names")
    if not branches:
        return ()
    requested = set(branches)
    order = physics._journal_branch_order
    if len(branches) != len(requested) or requested - order.keys():
        raise ValueError("Unknown or duplicate journal branch")
    return tuple(sorted(requested, key=order.__getitem__))


def compact_record(record, branches=()):
    """Reference projection used for paired qualification, not export optimization."""
    return {
        k: (
            {n: v for n, v in value.items() if n in branches}
            if k == "midpoint_branch_volume_rates_m3_s"
            else value
        )
        for k, value in record.items()
        if k not in FULL_ONLY_FIELDS
    }


def actuator_report(valves, dt):
    return {
        v.name: {
            **v.actuator.advance(v.position, v.target, dt, v.mode),
            "midpoint_position": v.at(dt / 2).opening,
        }
        for v in valves
        if hasattr(v, "actuator")
    }
