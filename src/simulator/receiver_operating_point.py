"""Stationary surface preparation against a fixed reservoir state.

Only the existing well/gathering equations are solved. No artificial reservoir
step, production accumulation, or field reset is used to obtain initial rates.
"""
import math
from copy import deepcopy

import torch

from .fim_transaction import _StepState
from .thermal_surface_boundary import characterize_well_thermal_boundaries

try:
    from ..surface.liquid import HydraulicConvergenceError, PropertyModelError
    from ..surface.stratified_pr_flow import ReceiverFeedPressureError, StratifiedPRFlow
except ImportError:
    from surface.liquid import HydraulicConvergenceError, PropertyModelError
    from surface.stratified_pr_flow import ReceiverFeedPressureError, StratifiedPRFlow


def fixed_reservoir_wells(solver, receiver_pressure_pa, *, maximum_iterations=30):
    """Return converged well reports without advancing or modifying the reservoir."""
    if isinstance(receiver_pressure_pa, bool) or not math.isfinite(receiver_pressure_pa) or receiver_pressure_pa <= 0:
        raise ValueError('Positive finite receiver pressure required')
    if type(maximum_iterations) is not int or maximum_iterations < 1:
        raise ValueError('Positive well iteration budget required')
    snapshot = _StepState(solver)
    w = solver.wells
    network = w.surface_network
    try:
        w.set_surface_boundary_pressures({network.root: receiver_pressure_pa})
        with torch.no_grad():
            v, d = solver.cell_state(solver.fluid.pressure.flatten(),
                solver.fluid.s_w.flatten(), solver.x, solver.sat, solver.ug)
            w.begin_step(v, d, 0.)
            bhp = w.p_bh.clone()
            residual = torch.zeros((solver.n_cells, 3), dtype=torch.float64, device=solver.device)
            jacobian = torch.zeros((solver.builder.n_nonzero, 3, 3), dtype=torch.float64, device=solver.device)

            def assemble(p):
                residual.zero_(); jacobian.zero_()
                system = w.assemble(residual, jacobian, solver.builder.diag_map, v, d, p)
                scale = torch.where(system['is_rate'],
                    system['target'].abs().clamp_min(.01/86400.) * solver.tol_well_rate,
                    torch.full_like(system['target'], solver.tol_well_bhp))
                norm = float((system['r_w']/scale).abs().max().cpu())
                return system, norm

            system, norm = assemble(bhp)
            for iteration in range(maximum_iterations):
                if math.isfinite(norm) and norm <= 1.:
                    return deepcopy(w.surface_report['wells']), {
                        'iterations': iteration, 'well_error_ratio': norm,
                        'reservoir_advanced': False,
                    }
                if not math.isfinite(norm) or bool(system['_surface_factor_bad'].cpu()):
                    raise HydraulicConvergenceError('Fixed-state well factorization/residual failed')
                direction = w._solve_D(system, -system['r_w'])
                if not bool(torch.isfinite(direction).all().cpu()):
                    raise HydraulicConvergenceError('Nonfinite fixed-state well update')
                for cut in range(16):
                    candidate = bhp + direction * (0.5**cut)
                    if bool((candidate <= 0).any().cpu()):
                        continue
                    try:
                        trial, error = assemble(candidate)
                    except HydraulicConvergenceError:
                        continue
                    if math.isfinite(error) and (error < norm or error <= 1.):
                        bhp, system, norm = candidate, trial, error
                        break
                else:
                    raise HydraulicConvergenceError('Fixed-state well line search failed')
            raise HydraulicConvergenceError('Fixed-state well iteration budget exhausted')
    finally:
        snapshot.restore()
        w.surface_network = network


def prepare_receiver_operating_point(coupled, *, verification_duration_s=.1):
    """Return proposed t=0 receiver stocks; caller explicitly adopts the result.

    Reservoir is held at its initial fields, not claimed to be stationary.
    The single-phase receiver is stationary under the resulting well rates.
    """
    coupled._validate_binding()
    old = coupled.receiver_state
    if old.time_s != 0 or coupled.report is not None:
        raise ValueError('Receiver operating point requires a fresh time-zero calculation')
    if type(coupled.receiver) is not StratifiedPRFlow:
        raise PropertyModelError('Operating-point preparation currently supports one PR vessel')
    boundary = coupled.boundary
    point = coupled._pressure(old)
    trace = []
    for _ in range(coupled.control.maximum_pressure_iterations):
        wells, work = fixed_reservoir_wells(coupled.solver, point)
        characterized = characterize_well_thermal_boundaries(
            wells, coupled.thermal_conditions, boundary.fluid, coupled.assay)
        streams = [s.transport(boundary.fluid, point,
            elevation_m=boundary.inlet_elevation(coupled.inlet_height_m),
            velocity_m_s=0., temperature_guess_k=boundary.temperature(old))
            for s in characterized['streams'].values()]
        flow = coupled._flow(streams)
        initial = boundary.rebind(old, flow)
        try:
            candidate, verification = flow.initialize_operating_point(
                initial, verification_duration_s=verification_duration_s)
            pressure = coupled._pressure(candidate)
            physical = True
        except ReceiverFeedPressureError as exc:
            pressure = exc.receiver_pressure_pa
            physical = False
        defect = pressure-point
        tolerance = (coupled.control.pressure_tolerance_pa
            + coupled.control.pressure_relative_tolerance*max(point,pressure))
        trace.append({'pressure_pa':point,'receiver_pressure_pa':pressure,
                      'defect_pa':defect,'physical':physical,'well_solve':work})
        if physical and abs(defect) <= tolerance:
            result = boundary.rebind(candidate, coupled.receiver)
            coupled.receiver.inspect(result)
            return result, {
                'pressure_trials':trace, 'accepted_wells':wells,
                'well_boundaries':characterized['boundaries'],
                'inlet_streams':[s.to_dict() for s in streams],
                'vessel_verification':verification,
                'reservoir_advanced':False, 'reservoir_stationary':False,
                'receiver_stationary':True, 'initial_state':old.to_dict(),
                'state':result.to_dict(),
            }
        point = pressure + .25*tolerance
    raise HydraulicConvergenceError('Stationary well/receiver pressure iteration budget exhausted')
