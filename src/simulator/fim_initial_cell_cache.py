"""Reuse only old-state cell properties inside one transactional capacity search.

The caller must restore the same physical old state before every trial. The
cache is accessed only at the explicit initial-state site in FIM.step; nonlinear
iterates always evaluate their own properties and derivatives. No tensor
comparison, device scalar read, persistent cache, or approximate state key.
"""

from contextlib import contextmanager


class InitialCellStateCache:
    def __init__(self, solver, dt, current_time, budget_bytes=64 * 1024**2):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ValueError('Initial cell cache budget must be a positive integer')
        self.solver, self.dt, self.current_time = solver, dt, current_time
        self.budget_bytes = budget_bytes
        self.values = None
        self.hits = self.evaluations = self.stored_bytes = 0
        self.bypassed = False

    @contextmanager
    def installed(self):
        name = '_capacity_initial_cell_cache'
        if hasattr(self.solver, name):
            raise RuntimeError('Nested initial cell cache scope is not supported')
        setattr(self.solver, name, self)
        try:
            yield
        finally:
            delattr(self.solver, name)

    def evaluate(self, solver, dt, current_time, compute):
        if solver is not self.solver or dt != self.dt or current_time != self.current_time:
            raise RuntimeError('Initial cell cache used outside its old-state/dt scope')
        if self.values is not None:
            self.hits += 1
            # Neither CUDA graph replay nor a caller can mutate the cache.
            return tuple(value.clone() for value in self.values)
        self.evaluations += 1
        result = compute()
        if not self.bypassed:
            size = sum(value.numel() * value.element_size() for value in result)
            if size <= self.budget_bytes:
                self.values = tuple(value.detach().clone() for value in result)
                self.stored_bytes = size
            else:
                self.bypassed = True
        return result

    def report(self):
        return {'evaluations': self.evaluations, 'hits': self.hits,
                'stored_bytes': self.stored_bytes, 'budget_bytes': self.budget_bytes,
                'bypassed': self.bypassed}


def initial_cell_state(solver, dt, current_time, p, sw, x, sat, ug):
    cache = getattr(solver, '_capacity_initial_cell_cache', None)
    if cache is None:
        return solver.cell_state(p, sw, x, sat, ug)
    return cache.evaluate(solver, dt, current_time, lambda: solver.cell_state(p, sw, x, sat, ug))


def audit_initial_cell_cache(coupling, control):
    """Check per-search property work and the independently configured memory bound."""
    enabled = control.get('reuse_initial_cell_state', False)
    budget = control.get('initial_cell_state_cache_bytes', 64 * 1024**2)
    if type(enabled) is not bool or type(budget) is not int or budget <= 0:
        raise ValueError('Invalid initial cell cache policy')
    actual = coupling.get('reservoir_evaluations', coupling['evaluations'])
    stats = coupling.get('initial_cell_state_cache')
    if not enabled:
        if stats is not None:
            raise ValueError('Disabled initial cell cache cannot report reuse')
        return actual, 0
    if stats is None:
        raise ValueError('Initial cell cache work report is required')
    if not isinstance(stats, dict):
        raise TypeError('Initial cell cache work report must be a dictionary')
    for key in ('evaluations', 'hits', 'stored_bytes', 'budget_bytes'):
        if type(stats.get(key)) is not int or stats[key] < 0:
            raise ValueError('Invalid initial cell cache work counter')
    if type(stats.get('bypassed')) is not bool or stats['budget_bytes'] != budget:
        raise ValueError('Invalid initial cell cache memory policy')
    if stats['evaluations'] + stats['hits'] != actual:
        raise ValueError('Initial cell cache work differs from actual FIM solves')
    if stats['bypassed']:
        if stats['stored_bytes'] or stats['hits'] or stats['evaluations'] != actual:
            raise ValueError('Bypassed initial cell cache cannot retain or reuse tensors')
    elif not (stats['evaluations'] == 1 and 0 < stats['stored_bytes'] <= budget):
        raise ValueError('Initial cell cache exceeds its single-state/memory bound')
    return stats['evaluations'], stats['hits']
