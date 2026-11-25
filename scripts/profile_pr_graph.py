"""CPU common-plant phase timings; diagnostic instrumentation, not paired speedup."""

import argparse
import functools
import json
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from audit_pr_controlled_mixed import audit
from run_pr_controlled_surface import run

from surface.pr_collocation import PRCollocation
from surface.torch_nonlinear import TorchNewton


@contextmanager
def profile_graph():
    counters, solves = {}, []
    active, differentiating = 0, 0

    def add(key, elapsed):
        item = counters.setdefault(key, {"calls": 0, "wall_s": 0.0})
        item["calls"] += 1
        item["wall_s"] += elapsed

    @contextmanager
    def replace(owner, name, replacement):
        original = getattr(owner, name)
        setattr(owner, name, replacement(original))
        try:
            yield
        finally:
            setattr(owner, name, original)

    def transform(name):
        def decorate(original):
            @functools.wraps(original)
            def factory(*args, **kwargs):
                function = original(*args, **kwargs)

                @functools.wraps(function)
                def execute(*a, **kw):
                    nonlocal differentiating
                    if not active or differentiating:
                        return function(*a, **kw)
                    differentiating += 1
                    start = time.perf_counter()
                    try:
                        return function(*a, **kw)
                    finally:
                        add(name, time.perf_counter() - start)
                        differentiating -= 1

                return execute

            return factory

        return decorate

    def linear(name):
        def decorate(original):
            @functools.wraps(original)
            def execute(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    if active and not differentiating:
                        add(name, time.perf_counter() - start)

            return execute

        return decorate

    def nonlinear(original):
        @functools.wraps(original)
        def execute(self, function, x, *args, **kwargs):
            nonlocal active

            @functools.wraps(function)
            def residual(y):
                if differentiating:
                    return function(y)
                start = time.perf_counter()
                try:
                    return function(y)
                finally:
                    add("ordinary_newton_residual", time.perf_counter() - start)

            active += 1
            start = time.perf_counter()
            try:
                return original(self, residual, x, *args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                active -= 1
                add("newton_total", elapsed)
                solves.append(
                    {
                        "unknowns": x.numel(),
                        "dense": x.numel() <= self.dense_limit,
                        "wall_s": elapsed,
                        "jacobian_mode": self._dense_jacobian_mode,
                        "jacobian_builds": self.jacobian_builds,
                        "jacobian_reuses": self.jacobian_reuses,
                    }
                )

        return execute

    with ExitStack() as stack:
        stack.enter_context(replace(TorchNewton, "solve", nonlinear))
        for name in ("jacfwd", "jacrev"):
            stack.enter_context(replace(torch.func, name, transform(name)))
        for name in ("lu_factor_ex", "lu_solve"):
            stack.enter_context(replace(torch.linalg, name, linear(name)))
        yield counters, solves


def profile(config, *, reference=False, stage_reference=False):
    previous = (
        PRCollocation._dense_cpu_jacobian_mode,
        PRCollocation._reuse_sdirk_points,
        PRCollocation._reuse_sdirk_tangent,
        PRCollocation._predict_sdirk_stage,
    )
    try:
        if reference:
            PRCollocation._dense_cpu_jacobian_mode = "forward"
            PRCollocation._reuse_sdirk_points = False
        if reference or stage_reference:
            PRCollocation._reuse_sdirk_tangent = False
            PRCollocation._predict_sdirk_stage = False
        with profile_graph() as (counters, solves):
            start = time.perf_counter()
            report = run(config, integrator="sdirk2-common")
            elapsed = time.perf_counter() - start
        certificate = audit(config, json.loads(json.dumps(report)))
    finally:
        (
            PRCollocation._dense_cpu_jacobian_mode,
            PRCollocation._reuse_sdirk_points,
            PRCollocation._reuse_sdirk_tangent,
            PRCollocation._predict_sdirk_stage,
        ) = previous
    newton = counters.get("newton_total", {}).get("wall_s", 0.0)
    accounted = sum(
        counters.get(k, {}).get("wall_s", 0.0)
        for k in ("jacfwd", "jacrev", "ordinary_newton_residual", "lu_factor_ex", "lu_solve")
    )
    return {
        "accepted": certificate["accepted"],
        "reference": reference,
        "stage_reference": stage_reference,
        "instrumented_total_s": elapsed,
        "phases": counters,
        "solves": solves,
        "outside_newton_s": elapsed - newton,
        "other_newton_s": newton - accounted,
        "audit": certificate,
        "scope": "instrumented CPU eager; construction and acceptance included; audit excluded; phases partition Newton only, not independent benchmark timings",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--stage-reference", action="store_true")
    args = parser.parse_args()
    result = profile(
        json.loads(args.config.read_text()),
        reference=args.reference,
        stage_reference=args.stage_reference,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
