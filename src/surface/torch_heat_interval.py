"""Transactional device-resident caloric interval under prescribed hydraulic fluxes.

This is a transport integrator, not a replacement for station control/event
orchestration. Fluxes and heat rates are held for one hydraulic record. All
substeps and audits stay on device; one status read commits or rejects the
whole interval. The caller retains the untouched input state on rejection.
"""

import math
from dataclasses import dataclass

import torch

from .liquid import HydraulicConvergenceError
from .torch_heat import TorchHeatState


@dataclass(frozen=True)
class TorchHeatIntervalResult:
    state: TorchHeatState
    # [stored change, boundary enthalpy, wall exchange, generated, duty,
    #  total energy residual, max local residual, max temporal error]
    audit: torch.Tensor
    substeps: int
    status_reads: int
    graph_replays: int


class TorchHeatInterval:
    def __init__(self, equations):
        self.e = equations

    def advance(
        self,
        state,
        end_mass,
        flows,
        heat_j,
        duration_s,
        substeps,
        boundary=None,
        ambient=None,
        duties=None,
    ):
        e = self.e
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        if isinstance(duration_s, bool) or not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        h = duration_s / substeps
        if h > e.model.maximum_dt_s * (1 + 1e-12):
            raise ValueError("Resident heat step exceeds configured maximum_dt_s")
        boundary = e.default_boundary if boundary is None else boundary
        ambient = e.default_ambient if ambient is None else ambient
        duties = torch.zeros(e.size, dtype=e.dtype, device=e.device) if duties is None else duties
        arrays = (state.temperature, state.mass, end_mass, flows, heat_j, boundary, ambient, duties)
        shapes = (
            (e.size,),
            (e.nf,),
            (e.nf,),
            (len(e.branch_names),),
            (len(e.branch_names),),
            (len(e.boundary_names),),
            (e.nw,),
            (e.size,),
        )
        if any(
            v.dtype != e.dtype or v.device != e.device or tuple(v.shape) != shape
            for v, shape in zip(arrays, shapes)
        ):
            raise ValueError("Resident heat inputs require configured float64 device and shapes")
        # Detach/clone: neither success nor a rejected GPU interval may modify
        # the caller's restart state or hold an unbounded autograd trajectory.
        arrays = tuple(v.detach().clone() for v in arrays)
        initial_t, initial_m, final_m, flows, heat_j, boundary, ambient, duties = arrays
        dt = e.tensor(h)
        forcing, absorbed, diagonal, dense = e.compile_operator(
            flows, heat_j, e.tensor(duration_s), boundary, ambient, duties
        )
        t = initial_t.clone()
        mass = initial_m.clone()
        counter = torch.zeros((), dtype=e.dtype, device=e.device)
        audit = torch.zeros(8, dtype=e.dtype, device=e.device)
        valid = (
            torch.stack([torch.isfinite(v).all() for v in arrays]).all()
            & (initial_m > 0).all()
            & (final_m > 0).all()
        )
        change_limit = min(
            [e.model.maximum_temperature_change_k]
            + [w.max_temperature_change_k for w in e.model.walls.values()]
        )

        def step():
            target = initial_m + (final_m - initial_m) * ((counter + 1) / substeps)
            middle = (mass + target) / 2
            rough = e.linear_step(t, mass, target, dt, forcing, diagonal, dense, boundary, ambient)
            temporal = e.model.temporal_error_control
            if temporal:
                first = e.linear_step(
                    t, mass, middle, dt / 2, forcing, diagonal, dense, boundary, ambient
                )
                second = e.linear_step(
                    first[0], middle, target, dt / 2, forcing, diagonal, dense, boundary, ambient
                )
                final = second[0]
                delta = first[1].sum() + second[1].sum()
                adv = first[2] + second[2]
                env = first[3] + second[3]
                local = torch.maximum(first[4], second[4])
                domain = first[-1] & second[-1]
                scale = (
                    e.model.temperature_tolerance_k
                    + e.model.relative_tolerance * torch.maximum(t.abs(), final.abs())
                )
                error = ((final - rough[0]).abs() / (3 * scale)).max()
            else:
                final = rough[0]
                delta, adv, env, local, domain = (
                    rough[1].sum(),
                    rough[2],
                    rough[3],
                    rough[4],
                    rough[-1],
                )
                error = torch.zeros_like(counter)
            generated = dt * absorbed.sum()
            external = dt * duties.sum()
            residual = delta - adv - env - generated - external
            scale = (
                delta.abs() + adv.abs() + env.abs() + generated.abs() + external.abs()
            ).clamp_min(1)
            capacity = torch.cat((torch.minimum(mass, target) * e.cp, e.wall_capacity))
            positive = (
                dt * (-diagonal) <= e.model.positivity_courant * (1 + 1e-12) * capacity
            ).all()
            accepted = (
                domain
                & positive
                & (target > 0).all()
                & torch.isfinite(error)
                & (error <= 1)
                & ((final - t).abs().max() <= change_limit)
                & (local <= 1e-7 + 1e-10 * scale)
                & (residual.abs() <= 1e-7 + 1e-10 * scale)
            )
            valid.logical_and_(accepted)
            increments = torch.stack((delta, adv, env, generated, external, residual))
            audit[:6].add_(increments)
            audit[6].copy_(torch.maximum(audit[6], local))
            audit[7].copy_(torch.maximum(audit[7], error))
            t.copy_(final)
            mass.copy_(target)
            counter.add_(1)

        with torch.no_grad():
            if e.device.type == "cuda":
                with torch.cuda.device(e.device):
                    stream = torch.cuda.Stream(device=e.device)
                    stream.wait_stream(torch.cuda.current_stream(e.device))
                    snapshot_valid = valid.clone()
                    with torch.cuda.stream(stream):
                        step()
                        step()
                    torch.cuda.current_stream(e.device).wait_stream(stream)
                    t.copy_(initial_t)
                    mass.copy_(initial_m)
                    counter.zero_()
                    audit.zero_()
                    valid.copy_(snapshot_valid)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        step()
                    # Capture executes the step once. Discard it before replay.
                    torch.cuda.current_stream(e.device).wait_stream(stream)
                    t.copy_(initial_t)
                    mass.copy_(initial_m)
                    counter.zero_()
                    audit.zero_()
                    valid.copy_(snapshot_valid)
                    for _ in range(substeps):
                        graph.replay()
            else:
                for _ in range(substeps):
                    step()
            committed = bool(valid.detach().cpu())
        if not committed:
            raise HydraulicConvergenceError(
                "Resident thermal interval rejected: domain, positivity, energy or temporal audit; "
                "input state retained, retry with smaller steps"
            )
        return TorchHeatIntervalResult(
            TorchHeatState(t.clone(), mass.clone()),
            audit.clone(),
            substeps,
            1,
            substeps if e.device.type == "cuda" else 0,
        )
