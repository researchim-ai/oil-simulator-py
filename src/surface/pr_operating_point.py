"""Explicit stationary initialization of a single-phase gravitational PR vessel.

Uses the transient model's own aperture and thermodynamic equations. Fixed
feeds are inputs, not a reservoir response. This changes initial inventories;
it is never applied to an evolving state or silently used to skip a startup.
"""
import math

import torch

from .liquid import HydraulicConvergenceError, PropertyModelError, _number
from .stratified_pr_flow import StratifiedPRFlow
from .stratified_pr_thermal import StratifiedThermalState
from .torch_nonlinear import TorchNewton


def initialize_operating_point(model, guess, *, verification_duration_s=0.1):
    if type(model) is not StratifiedPRFlow:
        raise TypeError('Operating-point initialization requires a single PR flow vessel')
    duration = _number(verification_duration_s, 'verification duration', positive=True)
    model.inspect(guess)
    if guess.time_s != 0:
        raise ValueError('Operating-point initialization is only allowed at time zero')
    old = guess.column
    if len(old.phase_roots) != 1:
        raise PropertyModelError('Operating-point initializer currently requires one phase')
    c, f = model.column, model.column.fluid
    feed = sum((f.tensor(p.stream.component_molar_rates_mol_s) for p in model.feeds),
               torch.zeros_like(f.tensor(old.component_moles)))
    if not model.outlets or not bool((feed.sum() > 0).cpu()):
        raise PropertyModelError('A positive throughflow and an outlet are required')
    if not bool(((feed > 0) == (f.tensor(old.component_moles) > 0)).all().cpu()):
        raise PropertyModelError('Feed and initial phase must have the same active species')
    seed = ([f.tensor(x) for x in old.phase_seeds], list(old.phase_roots),
            f.tensor(old.interface_heights_m), math.exp(old.solution_variables[0]),
            old.temperature_k, f.tensor(old.chemical_constants))
    order = max(old.quadrature_order, c.quadrature_order)
    initial, lower, upper, evaluate, active, roots, feed_n, _feed_e = model._problem(
        guess, 1.0, seed, order, stationary=True)
    solver = TorchNewton(f.device, maximum_iterations=60, dense_limit=768)
    solver.host_convergence = True
    with torch.no_grad():
        y, _, ok, _ = solver.solve(lambda z: evaluate(z)[0], initial, lower, upper,
                                   tolerance=3e-12)
        residual, result = evaluate(y)
    if not bool((ok & torch.isfinite(residual).all() & (residual.abs().max() < 1e-9)).cpu()):
        raise HydraulicConvergenceError('Stationary vessel Newton rejected')
    if any(not bool(p['valid'].cpu()) for p in result['ports']):
        raise PropertyModelError('Stationary outlet outside its physical domain')
    component_residual = feed_n - result['out_n']
    energy_residual = float((result['heat'] - result['out_e']).cpu()) + (
        _feed_e - float((feed_n * f.molar_masses).sum().cpu())
        * c.gravity_m_s2 * c.elevation_m
    )
    if (float(component_residual.abs().max().cpu()) > 1e-8 + 1e-9*float(feed_n.sum().cpu())
            or abs(energy_residual) > 1e-5 + 1e-9*max(1., abs(_feed_e))):
        raise HydraulicConvergenceError('Stationary component/energy rate balance rejected')
    data = result['data']
    n = data['component_moles']
    energy = float((data['internal_energy_j'] + data['relative_potential_energy_j']
                    + (n * f.molar_masses).sum() * c.gravity_m_s2 * c.elevation_m).cpu())
    constants = torch.zeros_like(n).index_copy(0, active, data['constants'])
    seed = ([x[0] for x in data['compositions'][:len(roots)]], roots,
            data['interfaces'], float(data['pressures'][0][0].cpu()),
            float(data['temperature'].cpu()), constants)
    column, _ = c._accept(n, energy, seed, order, result['column_variables'], solver_success=ok)
    state = StratifiedThermalState(column, float(result['wall'].cpu()), 0.,
                                   guess.next_step_s, model.model_id)
    model.inspect(state)
    # An independent transient solve enforces the existing flux, phase
    # stability, aperture refinement and energy/entropy acceptance checks.
    end, audit = model._step(state, duration)
    p0 = c.sample(column, 0., side='below')['pressure_pa']
    p1 = c.sample(end.column, 0., side='below')['pressure_pa']
    errors = {
        'pressure_pa': abs(p1-p0),
        'temperature_k': abs(end.column.temperature_k-column.temperature_k),
        'wall_temperature_k': abs(end.wall_temperature_k-state.wall_temperature_k),
        'component_moles': max(abs(a-b) for a,b in zip(end.column.component_moles, column.component_moles)),
    }
    if (errors['pressure_pa'] > .01 or errors['temperature_k'] > 1e-6
            or errors['wall_temperature_k'] > 1e-6
            or errors['component_moles'] > 1e-8 + 1e-9*sum(column.component_moles)):
        raise HydraulicConvergenceError(f'Stationary state drifts in transient verification: {errors}')
    return state, {
        'model_id': model.model_id, 'fixed_feeds': True,
        'reservoir_feedback_solved': False,
        'component_balance_mol_s': component_residual.cpu().tolist(),
        'energy_balance_w': energy_residual,
        'verification_duration_s': duration, 'verification_drift': errors,
        'verification_flow': audit,
        'initial_inventory_change_mol': [a-b for a,b in zip(column.component_moles, old.component_moles)],
        'initial_energy_change_j': column.total_energy_j-old.total_energy_j,
    }
