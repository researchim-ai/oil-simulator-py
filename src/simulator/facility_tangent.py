"""Proposal-only coupled rate-cap response; no accepted or extrapolated state.

Solve the existing sparse reservoir/well Jacobian with the cap derivative in
the well RHS. The physical step, its residuals and its well controls are never
updated. One transient sparse-value copy is budgeted, never a dense reservoir
matrix. A failed auxiliary solve must fall back to the ordinary bracket search.
"""

import math
from contextlib import contextmanager


class FacilityTangentCapture:
    def __init__(self, solver, base_caps_m3_day, budget_bytes=256 * 1024**2, *, fraction=None):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ValueError('Sensitivity sparse budget must be a positive integer')
        self.solver = solver
        self.fraction = fraction
        self.base_caps = tuple(base_caps_m3_day)
        if len(self.base_caps) != solver.wells.n_w or any(
                not math.isfinite(q) or q <= 0 for q in self.base_caps):
            raise ValueError('One positive finite base cap per producing well required')
        self.budget = budget_bytes
        self.data = None
        self.reason = 'not_converged'
        self.linear_solves = 0

    @contextmanager
    def installed(self):
        name = '_facility_tangent_capture'
        if hasattr(self.solver, name):
            raise RuntimeError('Nested facility tangent capture')
        setattr(self.solver, name, self)
        try:
            yield
        finally:
            delattr(self.solver, name)

    def record(self, jacobian, system):
        size = 2 * jacobian.numel() * jacobian.element_size()
        if size > self.budget:
            self.reason = 'sparse_value_budget'
            return
        flows = getattr(self.solver.wells, '_surface_flow_linearization', None)
        if system is None or flows is None:
            self.reason = 'unsupported_wells'
            return
        self.data = jacobian, system, flows
        self.reason = None

    def response(self):
        if self.data is None:
            return None
        jacobian, system, (q, dq, cv) = self.data
        self.data = None
        s = self.solver
        if self.fraction is not None and any(
                control != 'rate' or value != cap*self.fraction/86400
                for (control, value), cap in zip(s.wells.step_targets, self.base_caps, strict=True)):
            self.reason = 'nonlinear_target_policy'
            return None
        host_wells = s.wells.surface_report['wells']
        if not any(row['mode'] == 'rate' for row in host_wells.values()):
            self.reason = 'no_active_rate_rows'
            return None
        rhs = system['is_rate'].to(jacobian.dtype) * jacobian.new_tensor(self.base_caps) / 86400
        tangent_system = dict(system, r_w=-rhs)
        previous_stats = getattr(s, '_last_linear_stats', None)
        self.linear_solves += 1
        try:
            answer = s.solve_linear(jacobian.clone(), jacobian.new_zeros(s.n_cells * jacobian.shape[1]),
                                    tangent_system, newton_rel=0., rebuild_amg=False)
            stats = dict(getattr(s, '_last_linear_stats', {}))
        finally:
            s._last_linear_stats = previous_stats
        if answer is None:
            self.reason = 'auxiliary_linear_failure'
            return None
        residual = stats.get('true_rel_res', float('nan'))
        if not math.isfinite(residual) or residual > 1e-5:
            self.reason = 'inaccurate_auxiliary_linear_solve'
            return None
        dx, dp = answer
        w = s.wells
        direction = (dq * dx[w.conn_cell, None, :]).sum(2) + cv * dp[w.conn_well, None]
        direction = q.new_zeros((w.n_w, 3)).index_add_(0, w.conn_well, direction)
        packet = direction.detach().cpu().tolist()
        if any(not math.isfinite(value) for row in packet for value in row):
            self.reason = 'nonfinite_direction'
            return None
        return {'rates': [host_wells[well.name]['component_mass_rates_kg_s'] for well in w.wells],
                'direction': packet,
                'linear_stats': stats}


def capture_facility_tangent(solver, jacobian, system):
    capture = getattr(solver, '_facility_tangent_capture', None)
    if capture is not None:
        capture.record(jacobian, system)


def plant_margin_tangent(plant, response, fraction, controls, previous_running):
    """Central h/h2 directional checks of the small plant, never a FIM step.

The coupled reservoir direction comes from its sparse Jacobian. Here only
equipment margins are perturbed; no produced mass or state is committed.
Discontinuous, invalid-domain or numerically unstable facets are not used.
"""
    from copy import copy

    try:
        from ..surface import (
            ComponentStream,
            EquipmentOperatingError,
            InvalidPressureError,
            PropertyModelError,
        )
        from ..surface.capacity_bracket import facility_constraint_margins
        from ..surface.gas_valve import GasValveCapacityError
    except ImportError:
        from surface import (
            ComponentStream,
            EquipmentOperatingError,
            InvalidPressureError,
            PropertyModelError,
        )
        from surface.capacity_bracket import facility_constraint_margins
        from surface.gas_valve import GasValveCapacityError

    rates, direction = response['rates'], response['direction']
    h = 1e-4 * max(abs(fraction), 1e-4)
    for row, delta in zip(rates, direction, strict=True):
        for q, dq in zip(row, delta, strict=True):
            if dq:
                h = min(h, .25 * q / abs(dq))
    if not math.isfinite(h) or h <= 32 * math.ulp(fraction):
        return {}, 0, 'composition_boundary'
    evaluations = 0
    def margins(offset):
        nonlocal evaluations
        evaluations += 1
        probe = copy(plant)
        probe.feeds = tuple(ComponentStream.from_mass_rates(
            *(q + offset * dq for q, dq in zip(row, delta, strict=True)), plant.densities)
            for row, delta in zip(rates, direction, strict=True))
        try:
            return facility_constraint_margins(probe.run(previous_running=previous_running, **controls))
        except GasValveCapacityError as error:
            return {'gas_export.valve': error.normalized_margin}
    try:
        plus, minus, half_plus, half_minus = [margins(x) for x in (h, -h, h/2, -h/2)]
    except (EquipmentOperatingError, InvalidPressureError, PropertyModelError):
        return {}, evaluations, 'plant_perturbation_domain'
    slopes = {}
    for key in plus.keys() & minus.keys() & half_plus.keys() & half_minus.keys():
        values = [p[key] for p in (plus, minus, half_plus, half_minus)]
        if any(x is None or not math.isfinite(x) for x in values):
            continue
        coarse = (values[0] - values[1]) / (2*h)
        fine = (values[2] - values[3]) / h
        if (math.isfinite(coarse) and math.isfinite(fine) and fine > 0
                and abs(coarse-fine) <= 1e-3 * max(abs(coarse), abs(fine), 1e-12)):
            slopes[key] = fine
    return slopes, evaluations, None if slopes else 'no_stable_increasing_facet'
