"""Rollback physical FIM state when a substep fails or raises during commit.

Only dynamic well data and mutable fluid fields are copied. Grid/PVT data,
Jacobian blocks, AMG and CUDA graphs remain outside the transaction.
"""
from functools import wraps

import torch

_MISSING = object()
_WELL_FIELDS = (
    'p_bh', 'mode', 'rho_wb', 'conn_head', 'step_targets', 'step_mode',
    'switches', 'well_switches', 'conn_a', 'conn_eff', 'conn_open',
    'p_avg', 'rs_avg', 'rv_avg', 'group_controlled', 'reinit_bhp',
    'alq_opt', '_lift_t', 'pbh_override', 'node_p', '_net_iter',
    '_resv_idx', '_active_resv_idx', '_group_capacity_limited',
    '_group_base_targets', '_group_base_a',
)
# These caches are replaced, rather than modified, by assembly. Preserve their
# bindings without copying dense cross-perforation Jacobian blocks.
_WELL_BINDINGS = ('_Q', '_lin', 'p_cell_max', '_mix', '_mix_sel', '_producer_mix')
_SOLVER_BINDINGS = (
    'x', 'sat', 'ug', 'so_max', '_rs_cap', '_th_cache', '_p_bh_cur',
    '_last_well_sys', '_prev_norm', '_newton_iter',
    'last_newton_iters', 'last_max_dsw', 'last_max_dp',
)
_HYST_BINDINGS = ('sw_min', 'sg_max', 'shift_w', 'shift_g', 'on_w', 'on_g', 'flags')


def _copy_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _copy_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(v) for v in value)
    if isinstance(value, set):
        return set(value)
    return value


def _capture(obj, names, copy=False):
    return {name: _copy_value(getattr(obj, name)) if copy and hasattr(obj, name)
            else getattr(obj, name, _MISSING) for name in names}


def _restore(obj, values):
    for name, value in values.items():
        if value is _MISSING:
            if hasattr(obj, name):
                delattr(obj, name)
        else:
            setattr(obj, name, value)


class _StepState:
    def __init__(self, solver):
        self.solver = solver
        self.bindings = _capture(solver, _SOLVER_BINDINGS)
        fluid = solver.fluid
        # Existing pressure/Sw snapshots are also the old-time values used by
        # Newton: do not allocate a second reservoir-sized copy inside step().
        self.old = {name: getattr(fluid, name).detach().double().clone()
                    for name in ('pressure', 's_w')}
        self.fluid = {name: (getattr(fluid, name, _MISSING),
                            self.old[name] if name in self.old else
                            getattr(fluid, name).detach().clone())
                      for name in ('pressure', 's_w', 's_o') if hasattr(fluid, name)}
        self.fluid_bindings = _capture(fluid, ('s_g', 'rs', 'rv'))
        wells = solver.wells
        self.well_state = _capture(wells, _WELL_FIELDS, copy=True)
        self.well_bindings = _capture(wells, _WELL_BINDINGS)
        self.segments = [(m, _capture_dict(m, ('head', 'p_seg', 'rho')))
                         for m in getattr(wells, 'msw', {}).values()]
        self.wells = []
        for well in wells.wells:
            values = well.__dict__.copy()
            dictionaries = []
            for name, value in values.items():
                if name.startswith(('cum_', 'last_')) and isinstance(value, dict):
                    dictionaries.append((value, _copy_value(value)))
            self.wells.append((well, values, dictionaries))
        self.hyst = getattr(getattr(solver, 'kr', None), 'hyst', None)
        self.hyst_state = _capture(self.hyst, _HYST_BINDINGS) if self.hyst is not None else None

    def restore(self):
        solver = self.solver
        with torch.no_grad():
            for name, (original, value) in self.fluid.items():
                original.copy_(value)
                setattr(solver.fluid, name, original)
        _restore(solver.fluid, self.fluid_bindings)
        _restore(solver, self.bindings)
        _restore(solver.wells, self.well_state)
        _restore(solver.wells, self.well_bindings)
        for segment, values in self.segments:
            for name, value in values.items():
                if value is _MISSING:
                    segment.pop(name, None)
                else:
                    segment[name] = value
        for well, values, dictionaries in self.wells:
            for original, saved in dictionaries:
                original.clear()
                original.update(saved)
            well.__dict__.clear()
            well.__dict__.update(values)
        if self.hyst is not None:
            _restore(self.hyst, self.hyst_state)


def _capture_dict(mapping, names):
    return {name: _copy_value(mapping[name]) if name in mapping else _MISSING for name in names}


def atomic_fim_step(step):
    @wraps(step)
    def wrapped(self, *args, **kwargs):
        state = _StepState(self)
        self._step_initial = state.old
        previous_record = getattr(self.debug, 'step_rec', None)
        accepted = False
        try:
            result = step(self, *args, **kwargs)
            accepted = bool(result)
            return result
        except BaseException:
            if (self.debug.enabled and self.debug.step_rec is not previous_record
                    and self.debug.step_rec is not None):
                self.debug.end_step(False, 'exception')
            raise
        finally:
            if not accepted:
                state.restore()
            del self._step_initial
    return wrapped
