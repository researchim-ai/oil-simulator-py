"""Bounded hierarchy reuse across consecutive black-oil timesteps.

Numerical coarse operators are still updated by CPR on every linear solve.
This policy chooses only when to rebuild the interpolation hierarchy.
"""
import math


class AdaptiveAMG:
    def __init__(self, max_steps=4):
        self.max_steps = max_steps
        self.age = 0
        self.end = self.dt = self.key = self.hierarchy = None
        self.fresh_iters = None
        self.refresh = True
        self.builds = self.updates = self.retries = 0

    @staticmethod
    def structure(solver):
        def token(t):
            return (t.data_ptr(), tuple(t.shape), None if t.is_inference() else t._version)
        return (id(solver.builder), token(solver.builder.indptr),
                token(solver.builder.indices), token(solver.wells.conn_cell),
                token(solver.wells.conn_well))

    def reuse(self, solver, dt, current_time, k):
        if k:
            return not self.refresh
        key = self.structure(solver)
        permitted = (not self.refresh and solver._amg is self.hierarchy
                     and solver._amg is not None and key == self.key
                     and current_time is not None and current_time == self.end
                     and self.dt is not None and .5 <= dt / self.dt <= 2.
                     and self.age < self.max_steps - 1)
        self.age = self.age + 1 if permitted else 0
        self.refresh = not permitted
        self.key, self.dt = key, dt
        self.end = None if current_time is None else current_time + dt
        return permitted

    def record(self, solver, stats, accepted):
        rebuilt = bool(stats.get('amg_rebuilt'))
        self.builds += int(rebuilt)
        self.updates += int(bool(stats.get('amg_levels')) and not rebuilt)
        iters = stats.get('iters', 0)
        if rebuilt:
            self.age = 0
            self.fresh_iters = max(iters, 1)
        limit = max(20, 2 * (self.fresh_iters or 1))
        residual = stats.get('true_rel_res', float('inf'))
        self.refresh = (not accepted or solver._amg is None
                        or not stats.get('converged', False)
                        or not math.isfinite(residual) or residual > solver.lin_accept
                        or iters > limit)
        self.hierarchy = solver._amg

    def report(self):
        return dict(builds=self.builds, updates=self.updates, retries=self.retries,
                    age=self.age, refresh=self.refresh)
