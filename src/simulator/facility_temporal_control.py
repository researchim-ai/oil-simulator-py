"""Opt-in local BE defect control for reservoir/gathering/steady-facility steps.

The accepted solution is two independently converged half steps. The full
step is discarded. This is a local estimator, not a global error guarantee.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

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
    well_bhp_tolerance_pa: float | None = None
    withdrawal_error_scale: str = 'increment'

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == 'withdrawal_error_scale':
                if value not in ('increment', 'cumulative'):
                    raise ValueError('withdrawal_error_scale must be increment or cumulative')
                continue
            if name == 'well_bhp_tolerance_pa' and value is None:
                continue
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

    def defect(self, coarse_fields, fine_fields, coarse_mass, fine_mass,
               *, coarse_total_mass=None, fine_total_mass=None):
        terms = {}
        for name, tolerance in (('pressure', self.pressure_tolerance_pa),
                                ('s_w', self.saturation_tolerance),
                                ('s_g', self.saturation_tolerance),
                                ('rs', self.solution_ratio_tolerance),
                                ('rv', self.solution_ratio_tolerance)):
            if name in coarse_fields:
                terms[name] = ((fine_fields[name] - coarse_fields[name]).abs() / tolerance).max()
        # One bounded packet of field reductions instead of a synchronizing
        # scalar device-to-host read for every reservoir quantity.
        values = torch.stack(tuple(terms.values())).detach().cpu().tolist()
        errors = dict(zip(terms, values, strict=True))
        scale_mass = (coarse_mass, fine_mass)
        if self.withdrawal_error_scale == 'cumulative':
            scale_mass = (coarse_total_mass, fine_total_mass)
            if any(x is None or x.shape != coarse_mass.shape or not bool(torch.isfinite(x).all())
                   for x in scale_mass):
                raise HydraulicConvergenceError('Finite cumulative withdrawal totals required')
        scale = self.withdrawal_absolute_tolerance_kg + self.withdrawal_relative_tolerance * torch.maximum(
            scale_mass[0].abs(), scale_mass[1].abs())
        errors['withdrawal'] = float(((coarse_mass - fine_mass).abs() / scale).max().cpu())
        if any(not math.isfinite(e) for e in errors.values()):
            raise HydraulicConvergenceError('Nonfinite temporal defect')
        return errors

    def rejected_step(self, proposed_s, failed_dt_s, indicator):
        # O(dt²) local BE defect. Bound a single reduction so an active-set
        # change cannot turn an extrapolated estimate into an enormous jump.
        allowed = self.limit(failed_dt_s, failed_dt_s, indicator)
        return min(proposed_s, max(.1 * failed_dt_s, allowed))

    def well_defect(self, coarse, fine):
        """BHP values already reside in the host reports; no device polling."""
        if self.well_bhp_tolerance_pa is None:
            return {}
        if set(coarse) != set(fine):
            raise HydraulicConvergenceError('Different well registries in temporal BHP comparison')
        values = [*coarse.values(), *fine.values()]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in values):
            raise HydraulicConvergenceError('Nonfinite temporal well BHP')
        return {'well_bhp': max((abs(coarse[k] - fine[k]) for k in coarse), default=0.)
                / self.well_bhp_tolerance_pa}


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


def temporal_step(coupled, dt, current_time=None, *, stage_factory=None,
                  field_factory=None, defect_evaluator=None):
    """Three whole-system solves, transactional acceptance of fine stages only."""
    policy, s = coupled.temporal_control, coupled.solver
    stage_factory = _stage if stage_factory is None else stage_factory
    field_factory = _fields if field_factory is None else field_factory
    if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive and finite')
    start = 0. if current_time is None else current_time
    a, b = split_step_s(start, dt)
    initial, previous = _StepState(s), coupled.report
    old_fields = field_factory(s)
    old_metrics = {k: getattr(s, k) for k in ('last_newton_iters', 'last_max_dp', 'last_max_dsw') if hasattr(s, k)}
    accepted, total_evaluations, solves = False, 0, 0
    reservoir_evaluations = 0
    amg_work = {key: 0 for key in ('fresh_requests', 'reuse_requests', 'retries', 'builds', 'updates', 'linear_attempts')}
    sensitivity_linear_solves = sensitivity_plant_evaluations = 0
    coupled.accepted_stages = ()
    def solve(duration, time):
        nonlocal total_evaluations, reservoir_evaluations, solves
        nonlocal sensitivity_linear_solves, sensitivity_plant_evaluations
        solves += 1
        coupled.last_attempt = None
        try:
            ok = coupled._step_once(duration, current_time=time)
        finally:
            attempt = coupled.last_attempt or {}
            evaluations = attempt.get('evaluations', len(attempt.get('trace', [])))
            total_evaluations += evaluations
            reservoir_evaluations += attempt.get('reservoir_evaluations', evaluations)
            sensitivity_work = attempt.get('sensitivity_work') or {}
            sensitivity_linear_solves += sensitivity_work.get('linear_solves', 0)
            sensitivity_plant_evaluations += sensitivity_work.get('plant_evaluations', 0)
            for key, value in (attempt.get('capacity_amg_work') or {}).items():
                amg_work[key] += value
        if not ok:
            raise HydraulicConvergenceError((coupled.last_attempt or {}).get('reason', 'Coupled step failed'))
    try:
        solve(dt, start)
        coarse_fields = field_factory(s)
        coarse = stage_factory(coupled, dt, start + dt)
        initial.restore()
        initial = _StepState(s)  # restored well arrays become live mutable bindings
        coupled.report = previous
        solve(a, start)
        first = stage_factory(coupled, a, start + a)
        solve(b, start + a)
        second = stage_factory(coupled, b, start + dt)
        fine_mass = torch.tensor(first['withdrawn_component_mass_kg'], dtype=torch.float64) + torch.tensor(
            second['withdrawn_component_mass_kg'], dtype=torch.float64)
        coarse_total = fine_total = None
        if defect_evaluator is None and policy.withdrawal_error_scale == 'cumulative':
            coarse_total, fine_total = [torch.tensor([
                stage['facility_coupling']['material_account']['produced_kg'][k]
                for k in ('oil', 'water', 'gas')], dtype=torch.float64)
                for stage in (coarse, second)]
        errors = (defect_evaluator(coarse_fields, field_factory(s), coarse, first, second)
                  if defect_evaluator is not None else policy.defect(coarse_fields, field_factory(s),
                               torch.tensor(coarse['withdrawn_component_mass_kg'], dtype=torch.float64), fine_mass,
                               coarse_total_mass=coarse_total, fine_total_mass=fine_total))
        coarse_bhp = None
        if defect_evaluator is None and policy.well_bhp_tolerance_pa is not None:
            coarse_bhp = {k: w['bhp_pa'] for k, w in coarse['surface']['wells'].items()}
            fine_bhp = {k: w['bhp_pa'] for k, w in second['surface']['wells'].items()}
            errors.update(policy.well_defect(coarse_bhp, fine_bhp))
        ratio = max(errors.values())
        if ratio > 1:
            coupled.last_attempt = {'success': False, 'reason': f'Whole-system temporal defect exceeds tolerance: {errors}',
                                    'temporal_errors': errors, 'evaluations': total_evaluations}
            return False
        metadata = {'indicator': ratio, 'errors': errors, 'macro_dt_s': dt,
                    'total_capacity_evaluations': total_evaluations,
                    'total_reservoir_evaluations': reservoir_evaluations,
                    'total_capacity_amg_work': dict(amg_work),
                    'total_sensitivity_linear_solves': sensitivity_linear_solves,
                    'total_sensitivity_plant_evaluations': sensitivity_plant_evaluations,
                    'scope': 'Local full-step/two-half-step BE defect; accepted two half steps; requires independent global refinement'}
        if coarse_bhp is not None:
            metadata['coarse_well_bhp_pa'] = coarse_bhp
        if coarse_total is not None:
            metadata['withdrawal_error_scale'] = 'cumulative'
            metadata['coarse_withdrawal_kg'] = coarse['withdrawn_component_mass_kg']
            metadata['coarse_withdrawn_total_kg'] = coarse_total.tolist()
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
    except Exception as exc:
        coupled.last_attempt = {'success': False, 'reason': str(exc), 'evaluations': total_evaluations}
        raise
    finally:
        if coupled.last_attempt is not None:
            coupled.last_attempt['reservoir_evaluations'] = reservoir_evaluations
        work = getattr(coupled, 'temporal_work', None)
        if work is None:
            work = coupled.temporal_work = {'macro_attempts': 0, 'rejected_macro_steps': 0,
                                            'coupled_solves': 0, 'capacity_evaluations': 0}
        work['macro_attempts'] += 1
        work['rejected_macro_steps'] += int(not accepted)
        work['coupled_solves'] += solves
        work['capacity_evaluations'] += total_evaluations
        work['reservoir_evaluations'] = work.get('reservoir_evaluations', 0) + reservoir_evaluations
        total_amg = work.setdefault('capacity_amg_work', {key: 0 for key in amg_work})
        for key, value in amg_work.items():
            total_amg[key] += value
        work['sensitivity_linear_solves'] = work.get('sensitivity_linear_solves', 0) + sensitivity_linear_solves
        work['sensitivity_plant_evaluations'] = work.get('sensitivity_plant_evaluations', 0) + sensitivity_plant_evaluations
        if not accepted:
            initial.restore()
            coupled.report = previous
            coupled.accepted_stages = ()
            for k, value in old_metrics.items():
                setattr(s, k, value)
