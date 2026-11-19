"""Conservative vessel-only integration inside an existing FIM pressure trial."""
import math

try:
    from ..surface.liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
    from ..surface.stratified_pr_thermal import StratifiedThermalState
except ImportError:
    from surface.liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
    from surface.stratified_pr_thermal import StratifiedThermalState


def advance_receiver_locally(
    flow, initial, dt, *, pressure_tolerance_pa=1.0, pressure_height_m=None,
):
    # Ordinary calls keep immediate pressure rejection. The coupled driver
    # explicitly requests a complete endpoint probe. Even violations in coarse
    # or subsequently discarded steps reject it: no domain check is relaxed.
    failures = []

    def observe(failure):
        # Retain only the worst supply bound, independent of trajectory length.
        if not failures or failure.minimum_supply_pressure_pa > failures[0].minimum_supply_pressure_pa:
            failures[:] = [failure]

    def step(state, step_dt):
        try:
            return flow._step(state, step_dt, _feed_pressure_observer=observe)
        except (ConservationError, HydraulicConvergenceError, PropertyModelError):
            if failures:
                # This already-infeasible probe cannot supply a trustworthy
                # endpoint. Do not spend adaptive retries on its other failure.
                raise failures[0] from None
            raise

    kwargs = {} if pressure_height_m is None else {"_stepper": step}
    try:
        raw = flow.advance(initial, dt, pressure_tolerance_pa=pressure_tolerance_pa, **kwargs)
    except (ConservationError, HydraulicConvergenceError, PropertyModelError):
        if failures:
            # No endpoint exists: retain the original pressure-bound fallback.
            raise failures[0] from None
        raise
    end = StratifiedThermalState.from_dict(raw['state'])
    if failures:
        failure = failures[0]
        failure.endpoint_pressure_pa = flow.column.sample(
            end.column, pressure_height_m, side="below"
        )["pressure_pa"]
        raise failure
    steps = raw['steps']
    result = {key:raw[key] for key in (
        'ambient_heat_j', 'feed_energy_j', 'outlet_energy_j',
        'energy_residual_j', 'entropy_production_j_k')}
    result.update(dt_s=dt, column_state=end.column.to_dict(),
                  wall_temperature_k=end.wall_temperature_k,
                  surface_steps=steps, surface_attempts=raw['attempts'])
    for key in ('feed_component_moles', 'outlet_component_moles'):
        result[key] = [math.fsum(s[key][j] for s in steps)
                       for j in range(len(end.column.component_moles))]
    return end, result
