"""One pressure hierarchy per restored-state capacity search, never a new equation."""

from contextlib import contextmanager


class CapacityAMGReuse:
    binding_name = "_capacity_amg_scope"

    def __init__(self, solver, dt, current_time, *, enabled=True):
        self.solver, self.dt, self.current_time = solver, dt, current_time
        self.enabled = enabled
        self.ready = False
        self.reuse_requests = self.fresh_requests = self.retries = 0
        self.builds = self.updates = self.linear_attempts = 0

    def record(self, stats):
        self.linear_attempts += 1
        self.builds += int(bool(stats.get("amg_rebuilt", False)))
        self.updates += int(bool(stats.get("amg_levels")) and not stats.get("amg_rebuilt", False))

    @contextmanager
    def installed(self):
        if hasattr(self.solver, self.binding_name):
            raise RuntimeError("Nested capacity AMG reuse scope")
        setattr(self.solver, self.binding_name, self)
        try:
            yield
        finally:
            delattr(self.solver, self.binding_name)

    def reuse(self, solver, dt, current_time, k):
        if solver is not self.solver or dt != self.dt or current_time != self.current_time:
            raise RuntimeError("AMG reuse outside restored-state search")
        permitted = (
            self.enabled
            and k == 0
            and self.ready
            and solver._amg is not None
            and solver.amg_rebuild != "iteration"
        )
        if k == 0:
            if permitted:
                self.reuse_requests += 1
            else:
                self.fresh_requests += 1
        return permitted

    def report(self):
        return {
            "reuse_requests": self.reuse_requests,
            "fresh_requests": self.fresh_requests,
            "retries": self.retries,
            "builds": self.builds,
            "updates": self.updates,
            "linear_attempts": self.linear_attempts,
        }


def audit_capacity_amg(coupling, control):
    enabled = control.get("reuse_capacity_amg", False)
    if type(enabled) is not bool:
        raise ValueError("Invalid capacity AMG policy")
    work = coupling.get("capacity_amg_work")
    if not enabled and work is None:
        return {}
    keys = {"fresh_requests", "reuse_requests", "retries", "builds", "updates", "linear_attempts"}
    if (
        not isinstance(work, dict)
        or set(work) != keys
        or any(type(v) is not int or v < 0 for v in work.values())
    ):
        raise ValueError("Invalid capacity AMG work counters")
    actual = coupling.get("reservoir_evaluations", coupling["evaluations"])
    if (
        work["fresh_requests"] + work["reuse_requests"] > actual
        or work["builds"] + work["updates"] > work["linear_attempts"]
        or work["fresh_requests"] + work["reuse_requests"] + work["retries"]
        > work["linear_attempts"]
        or (work["reuse_requests"] and not work["fresh_requests"])
    ):
        raise ValueError("Inconsistent capacity AMG work")
    if not enabled and (work["reuse_requests"] or work["retries"]):
        raise ValueError("Disabled capacity AMG reuse changed execution")
    return work
