"""Opt-in local BE defect control for reservoir/gathering/steady-facility steps.

The accepted solution is two independently converged half steps. The full
step is discarded. This is a local estimator, not a global error guarantee.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import torch

from .fim_transaction import _StepState

try:
    from ..surface import HydraulicConvergenceError
    from ..surface.time_interval import split_step_s
except ImportError:
    from surface import HydraulicConvergenceError
    from surface.time_interval import split_step_s


@dataclass(frozen=True)
class FacilityTemporalControl:
    pressure_tolerance_pa: float = 100.
    saturation_tolerance: float = 2e-5
    solution_ratio_tolerance: float = .01
    withdrawal_relative_tolerance: float = .002
    withdrawal_absolute_tolerance_kg: float = 1e-5
    startup_step_days: float = .25
    safety_factor: float = .9
    maximum_growth: float = 2.

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Positive finite {name} required')
        if self.safety_factor >= 1 or self.maximum_growth < 1 or self.withdrawal_relative_tolerance >= 1:
            raise ValueError('Invalid temporal control policy')
        if not math.isfinite(self.startup_step_days * 86400):
            raise ValueError('Startup step must be representable in seconds')

    def limit(self, proposed_s, dt_s=None, indicator=None, *, clipped=False):
        if not math.isfinite(proposed_s) or proposed_s <= 0:
            raise ValueError('Positive finite proposed step required')
        if indicator is None:
            return min(proposed_s, self.startup_step_days * 86400)
        if not math.isfinite(indicator) or indicator < 0 or dt_s is None or not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError('Finite nonnegative temporal indicator and duration required')
        allowed = proposed_s if indicator == 0 else dt_s * self.safety_factor / math.sqrt(indicator)
        if not clipped:
            allowed = min(allowed, dt_s * self.maximum_growth)
        return min(proposed_s, allowed)

    def defect(self, coarse_fields, fine_fields, coarse_mass, fine_mass):
        errors = {}
        for name, tolerance in (('pressure', self.pressure_tolerance_pa),
                                ('s_w', self.saturation_tolerance),
                                ('s_g', self.saturation_tolerance),
                                ('rs', self.solution_ratio_tolerance),
                                ('rv', self.solution_ratio_tolerance)):
            if name in coarse_fields:
                errors[name] = float(((fine_fields[name] - coarse_fields[name]).abs() / tolerance).max().cpu())
        scale = self.withdrawal_absolute_tolerance_kg + self.withdrawal_relative_tolerance * torch.maximum(
            coarse_mass.abs(), fine_mass.abs())
        errors['withdrawal'] = float(((coarse_mass - fine_mass).abs() / scale).max().cpu())
        if any(not math.isfinite(e) for e in errors.values()):
            raise HydraulicConvergenceError('Nonfinite temporal defect')
        return errors

    def rejected_step(self, proposed_s, failed_dt_s, indicator):
        # O(dt²) local BE defect. Bound a single reduction so an active-set
        # change cannot turn an extrapolated estimate into an enormous jump.
        allowed = self.limit(failed_dt_s, failed_dt_s, indicator)
        return min(proposed_s, max(.1 * failed_dt_s, allowed))


def _fields(solver):
    return {name: getattr(solver.fluid, name).detach().clone()
            for name in ('pressure', 's_w', 's_g', 'rs', 'rv') if hasattr(solver.fluid, name)}


def _stage(coupled, duration, end):
    s = coupled.solver
    surface = deepcopy(s.wells.surface_report)
    rates = torch.tensor([w['component_mass_rates_kg_s'] for w in surface['wells'].values()],
                         dtype=torch.float64).sum(0)
    return {'time_days': end / 86400, 'dt_s': duration,
            'withdrawn_component_mass_kg': (duration * rates).tolist(),
            'newton_iterations': s.last_newton_iters, 'surface': surface,
            'facility_coupling': deepcopy(coupled.report),
            'facility_controls': deepcopy(coupled.controls)}


def temporal_step(coupled, dt, current_time=None):
    """Three whole-system solves, transactional acceptance of fine stages only."""
    policy, s = coupled.temporal_control, coupled.solver
    if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive and finite')
    start = 0. if current_time is None else current_time
    a, b = split_step_s(start, dt)
    initial, previous = _StepState(s), coupled.report
    old_fields = _fields(s)
    old_metrics = {k: getattr(s, k) for k in ('last_newton_iters', 'last_max_dp', 'last_max_dsw') if hasattr(s, k)}
    accepted, total_evaluations, solves = False, 0, 0
    coupled.accepted_stages = ()
    def solve(duration, time):
        nonlocal total_evaluations, solves
        solves += 1
        try:
            ok = coupled._step_once(duration, current_time=time)
        finally:
            total_evaluations += (coupled.last_attempt or {}).get('evaluations',
                                len((coupled.last_attempt or {}).get('trace', [])))
        if not ok:
            raise HydraulicConvergenceError((coupled.last_attempt or {}).get('reason', 'Coupled step failed'))
    try:
        solve(dt, start)
        coarse_fields = _fields(s)
        coarse = _stage(coupled, dt, start + dt)
        initial.restore()
        initial = _StepState(s)  # restored well arrays become live mutable bindings
        coupled.report = previous
        solve(a, start)
        first = _stage(coupled, a, start + a)
        solve(b, start + a)
        second = _stage(coupled, b, start + dt)
        fine_mass = torch.tensor(first['withdrawn_component_mass_kg'], dtype=torch.float64) + torch.tensor(
            second['withdrawn_component_mass_kg'], dtype=torch.float64)
        errors = policy.defect(coarse_fields, _fields(s),
                               torch.tensor(coarse['withdrawn_component_mass_kg'], dtype=torch.float64), fine_mass)
        ratio = max(errors.values())
        if ratio > 1:
            coupled.last_attempt = {'success': False, 'reason': f'Whole-system temporal defect exceeds tolerance: {errors}',
                                    'temporal_errors': errors, 'evaluations': total_evaluations}
            return False
        metadata = {'indicator': ratio, 'errors': errors, 'macro_dt_s': dt,
                    'total_capacity_evaluations': total_evaluations,
                    'scope': 'Local full-step/two-half-step BE defect; accepted two half steps; requires independent global refinement'}
        coupled.report['temporal_control'] = deepcopy(metadata)
        second['facility_coupling']['temporal_control'] = deepcopy(metadata)
        coupled.accepted_stages = (first, second)
        s.last_newton_iters = max(first['newton_iterations'], second['newton_iterations'])
        s.last_max_dp = float((s.fluid.pressure - old_fields['pressure']).abs().max().cpu())
        sw, sg = s.fluid.s_w - old_fields['s_w'], s.fluid.s_g - old_fields['s_g']
        s.last_max_dsw = float(torch.stack([sw.abs().max(), sg.abs().max(), (sw + sg).abs().max()]).max().cpu())
        coupled.last_attempt = {'success': True, 'evaluations': total_evaluations, 'temporal_errors': errors}
        accepted = True
        return True
    except HydraulicConvergenceError as exc:
        coupled.last_attempt = {'success': False, 'reason': str(exc), 'evaluations': total_evaluations}
        return False
    finally:
        work = getattr(coupled, 'temporal_work', None)
        if work is None:
            work = coupled.temporal_work = dict(macro_attempts=0, rejected_macro_steps=0,
                                                coupled_solves=0, capacity_evaluations=0)
        work['macro_attempts'] += 1
        work['rejected_macro_steps'] += int(not accepted)
        work['coupled_solves'] += solves
        work['capacity_evaluations'] += total_evaluations
        if not accepted:
            initial.restore()
            coupled.report = previous
            coupled.accepted_stages = ()
            for k, value in old_metrics.items():
                setattr(s, k, value)
