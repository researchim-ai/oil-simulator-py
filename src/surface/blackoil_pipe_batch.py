"""Bounded independent pressure-coordinate pipe batches, with implicit AD.

Same PVT knots, moving bubble point, Colebrook closure and error budgets as
BlackOilPipe. A failed batch retries the existing scalar solver explicitly;
physical failures are never converted to an accepted approximate result.
"""

import math

import torch

from .blackoil_batch_admission import validate_inputs
from .blackoil_pipe import BlackOilPipe, _darcy, _gauss
from .blackoil_pipe_grid import get_pressure_grid
from .blackoil_pressure_lookup import PressureLookup
from .blackoil_pvto_batch import PackedPVTO
from .blackoil_quadrature_plan import make_quadrature_plan
from .liquid import HydraulicConvergenceError


def tensor_state(model, pressure, masses, *, pvto=None):
    """Independent rows; trailing pressure axes are quadrature/port points."""
    shape = (len(masses),) + (1,) * (pressure.ndim - 1)
    q = masses / model.rho
    oil, water, gas = (q[:, i].reshape(shape) for i in range(3))
    ratio = torch.where(oil > 0, gas / torch.where(oil > 0, oil, torch.ones_like(oil)), oil * 0)
    lookup = PressureLookup(pressure, model.pressures)
    interp = lambda key: lookup.interpolate(model.columns[key])
    sat_rs = interp("Rs")
    saturated = torch.where(oil > 0, sat_rs <= ratio, (gas > 0) | (len(model.oil_rs) < 2))
    rs = torch.where(oil > 0, torch.minimum(sat_rs, ratio), pressure * 0)
    bo, muo = interp("Bo"), interp("mu_o")
    if len(model.oil_rs) >= 2:
        index = (
            torch.searchsorted(model.oil_rs, ratio.detach().contiguous(), right=True) - 1
        ).clamp(0, len(model.oil_rs) - 2)
        weight = (ratio - model.oil_rs[index]) / (model.oil_rs[index + 1] - model.oil_rs[index])
        packed = PackedPVTO(model) if pvto is None else pvto
        selected = index.reshape(len(masses))
        shared = hasattr(packed, "prepare_lookup")
        lo_lookup = packed.prepare_lookup(pressure, selected) if shared else None
        hi_lookup = packed.prepare_lookup(pressure, selected + 1) if shared else None
        values = []
        for column in (1, 2):
            if shared:
                lo = packed.interpolate(pressure, selected, column, lookup=lo_lookup)
                hi = packed.interpolate(pressure, selected + 1, column, lookup=hi_lookup)
            else:
                lo = packed.interpolate(pressure, selected, column)
                hi = packed.interpolate(pressure, selected + 1, column)
            values.append(lo * (1 - weight) + hi * weight)
        bo, muo = torch.where(saturated, bo, values[0]), torch.where(saturated, muo, values[1])
    dissolved = oil * rs
    volumes = torch.stack([oil * bo, water * interp("Bw"), (gas - dissolved) * interp("Bg")], -1)
    phase_mass = torch.stack(
        [
            masses[:, 0].reshape(shape) + dissolved * model.rho[2],
            masses[:, 1].reshape(shape) + pressure * 0,
            masses[:, 2].reshape(shape) - dissolved * model.rho[2],
        ],
        -1,
    )
    volume = volumes.sum(-1)
    viscosity = torch.stack([muo, interp("mu_w"), interp("mu_g")], -1) * 0.001
    return {
        "specific_volume": volume / masses.sum(-1).reshape(shape),
        "viscosity_pa_s": (volumes * viscosity).sum(-1) / volume,
        "phase_volume_rates_m3_s": volumes,
        "phase_mass_rates_kg_s": phase_mass,
        "rs_m3_m3": rs,
    }


class PipeBatch:
    def __init__(self, pipes, model):
        self.pipes, self.model = tuple(pipes), model
        if not self.pipes or not all(isinstance(p, BlackOilPipe) for p in self.pipes):
            raise TypeError("Nonempty BlackOilPipe batch required")
        self.pvto = PackedPVTO(model) if len(model.oil_rs) >= 2 else None
        self.pressure_grid = get_pressure_grid(model)
        self.geometry = model.rho.new_tensor(
            [[p.length_m, p.diameter_m, p.elevation_change_m, p.roughness_m] for p in self.pipes]
        )

    def _pressure_points(self, lower, upper, masses):
        model = self.model
        q = masses / model.rho
        ratio = torch.where(
            q[:, 0] > 0, q[:, 2] / torch.where(q[:, 0] > 0, q[:, 0], torch.ones_like(q[:, 0])), 0.0
        )
        rs = model.columns["Rs"]
        i = (torch.searchsorted(rs, ratio.detach(), right=True) - 1).clamp(0, len(rs) - 2)
        # Flat Rs intervals are not a moving bubble point. Clip outside-range
        # ratios to an endpoint, so they introduce no internal breakpoint.
        span = rs[i + 1] - rs[i]
        bubble = model.pressures[i] + (ratio - rs[i]) * (
            model.pressures[i + 1] - model.pressures[i]
        ) / torch.where(span > 0, span, torch.ones_like(span))
        bubble = torch.where(
            (q[:, 0] > 0) & (ratio > rs[0]) & (ratio < rs[-1]) & (span > 0), bubble, lower
        )
        fixed = self.pressure_grid.fixed(masses)
        inner = torch.cat([fixed, bubble[:, None]], 1).clamp(min=lower[:, None], max=upper[:, None])
        points = torch.cat([lower[:, None], inner, upper[:, None]], 1).sort(1).values
        return points

    def _integrand(self, p, point_widths, masses, resistance):
        model = self.model
        length, diameter, dz, rough = self.geometry.unbind(1)
        differentiate = torch.is_grad_enabled()
        with torch.enable_grad():
            if not p.requires_grad:
                p.requires_grad_(True)
            state = tensor_state(model, p, masses, pvto=self.pvto)
            v, mu = state["specific_volume"], state["viscosity_pa_s"]
            vp = torch.autograd.grad(v.sum(), p, create_graph=differentiate)[0]
        flux = masses.sum(1) / (math.pi * diameter.square() / 4)
        f = _darcy(
            flux[:, None, None] * diameter[:, None, None] / mu, (rough / diameter)[:, None, None]
        )
        margin = 1 + flux[:, None, None].square() * vp
        bad = ((margin <= 1e-6) | ~torch.isfinite(margin)) & (point_widths > 0)
        if bool(bad.any()):
            raise HydraulicConvergenceError("Batch crosses a non-subcritical momentum branch")
        drag = (
            (f / diameter[:, None, None] + (resistance / length)[:, None, None])
            * flux[:, None, None].square()
            * v
            / 2
        )
        slope = (drag + (9.80665 * dz / length)[:, None, None] / v) / margin
        inverse = slope.reciprocal()
        return inverse, drag * v * inverse, v

    def integral(self, lower, upper, masses, resistance, order, *, plan=None):
        if plan is None:
            plan = make_quadrature_plan(self._pressure_points(lower, upper, masses))
        nodes, weights = _gauss(order, str(masses.device))
        return plan.evaluate(self, masses, resistance, nodes, weights)

    def checked(self, lower, upper, masses, resistance):
        previous = None
        plan = make_quadrature_plan(self._pressure_points(lower, upper, masses))
        for order in (16, 32, 64, 128, 256):
            if previous is None:
                with torch.no_grad():
                    result = self.integral(lower, upper, masses, resistance, order, plan=plan)
            else:
                result = self.integral(lower, upper, masses, resistance, order, plan=plan)
            detached = result.detach()
            if not bool(torch.isfinite(detached).all()):
                raise HydraulicConvergenceError("Nonfinite batch quadrature")
            if previous is not None:
                tol = torch.maximum(detached.abs() * 1e-9, detached.new_tensor([1e-7, 1e-8, 1e-8]))
                if bool(((detached - previous).abs() <= tol).all()):
                    return result, order, (detached[:, 0] - previous[:, 0]).abs()
            previous = detached
        raise HydraulicConvergenceError("Batch quadrature requires scalar subdivision")

    @torch.enable_grad()
    def solve(self, boundaries, rates):
        model = self.model
        n = len(self.pipes)
        boundary, masses = validate_inputs(model, boundaries, rates, n)
        resistance = model.rho.new_tensor([p.loss_coefficient for p in self.pipes])
        length, diameter, dz, rough = self.geometry.unbind(1)
        port = boundary.detach().requires_grad_(True)
        state = tensor_state(model, port, masses, pvto=self.pvto)
        v, mu = state["specific_volume"], state["viscosity_pa_s"]
        vp = torch.autograd.grad(v.sum(), port)[0]
        flux = masses.sum(1) / (math.pi * diameter.square() / 4)
        margin = 1 + flux.square() * vp
        if bool(((margin <= 1e-6) | ~torch.isfinite(margin)).any()):
            raise HydraulicConvergenceError("Batch known port is non-subcritical")
        f = _darcy(flux * diameter / mu.detach(), rough / diameter)
        slope = (
            (f / diameter + resistance / length) * flux.square() * v.detach() / 2
            + 9.80665 * dz / length / v.detach()
        ) / margin
        increment = (1.1 * length * slope).clamp_min(1e-3).detach()
        high = (boundary + increment).clamp_max(model.pressure_bounds_pa[1])
        calls = 0
        for _ in range(64):
            with torch.no_grad():
                value, _, _ = self.checked(boundary, high, masses, resistance)
            calls += 1
            found = value[:, 0].detach() >= length
            if bool(found.all()):
                break
            if bool((~found & (high >= model.pressure_bounds_pa[1])).any()):
                raise HydraulicConvergenceError("Batch pressure bracket exceeds PVT coverage")
            increment = torch.where(found, increment, increment * 2)
            high = torch.where(
                found, high, (boundary + increment).clamp_max(model.pressure_bounds_pa[1])
            )
        else:
            raise HydraulicConvergenceError("Batch bracket budget exhausted")
        low = boundary.clone()
        # Reuse the known-port momentum slope already computed for bracketing.
        # This is a proposal only: the full integral and original tolerances
        # still decide convergence. Invalid/out-of-bracket seeds use bisection.
        estimate = boundary + length * slope.detach()
        trial = torch.where(
            torch.isfinite(estimate) & (estimate > low) & (estimate < high),
            estimate,
            (low + high) / 2,
        )
        # Safeguarded Newton uses the exact endpoint integrand dL/dp, keeping
        # the full pressure-coordinate integral as the acceptance equation.
        done = torch.zeros(n, dtype=torch.bool, device=boundary.device)
        for _ in range(64):
            trial = trial.detach().requires_grad_(True)
            value, _, _ = self.checked(boundary, trial, masses, resistance)
            residual = value[:, 0] - length
            calls += 1
            okay = residual.detach().abs() <= torch.maximum(
                length * 1e-10, length.new_full((n,), 1e-8)
            )
            done |= okay
            if bool(done.all()):
                root = trial.detach()
                # Release the unused trial tape before the final sensitivity
                # graph; skipping backward must not increase peak memory.
                del value, residual
                break
            high = torch.where(~done & (residual.detach() > 0), trial.detach(), high)
            low = torch.where(~done & (residual.detach() <= 0), trial.detach(), low)
            # A converged root needs no next Newton proposal. Final implicit
            # sensitivities are computed independently below for all inputs.
            deriv = torch.autograd.grad(value[:, 0].sum(), trial)[0]
            newton = trial.detach() - residual.detach() / deriv.detach()
            safe = torch.isfinite(newton) & (newton > low) & (newton < high)
            trial = torch.where(done, trial.detach(), torch.where(safe, newton, (low + high) / 2))
        else:
            raise HydraulicConvergenceError("Batch pressure inversion budget exhausted")
        x = (
            torch.cat([root[:, None], boundary[:, None], masses, resistance[:, None]], 1)
            .detach()
            .requires_grad_(True)
        )
        integral, order, error = self.checked(x[:, 1], x[:, 0], x[:, 2:5], x[:, 5])
        gradient = torch.autograd.grad(integral[:, 0].sum(), x)[0]
        derivative = -gradient / gradient[:, :1]
        ports = x[:, :2].detach().requires_grad_(True)
        states = tensor_state(model, ports, masses, pvto=self.pvto)
        pv = states["specific_volume"]
        dv = torch.autograd.grad(pv.sum(), ports)[0]
        margins = 1 + flux[:, None].square() * dv
        mechanical = (
            -integral[:, 2]
            + 0.5 * flux.square() * (pv[:, 1].square() - pv[:, 0].square())
            + 9.80665 * dz
            + integral[:, 1]
        )
        scale = torch.stack(
            [torch.ones_like(length), integral[:, 2].abs(), integral[:, 1].abs(), 9.80665 * dz], 1
        ).amax(1)
        valid = (
            torch.isfinite(derivative).all()
            & torch.isfinite(margins).all()
            & (margins > 1e-6).all()
            & (mechanical.abs() <= torch.maximum(scale * 1e-8, scale.new_full((n,), 1e-5))).all()
            & (
                (integral[:, 0] - length).abs()
                <= torch.maximum(length * 1e-8, length.new_full((n,), 1e-5))
            ).all()
        )
        if not bool(valid):
            raise HydraulicConvergenceError("Batch final physical/residual check failed")
        # Reporting is a host boundary; no live AD graph escapes the batch.
        states = {k: v.detach().tolist() for k, v in states.items()}
        values = (
            torch.cat(
                [
                    root[:, None],
                    boundary[:, None],
                    masses,
                    margins,
                    integral.detach(),
                    derivative.detach(),
                    error[:, None],
                    (masses.sum(1) * mechanical.detach())[:, None],
                ],
                1,
            )
            .detach()
            .tolist()
        )
        rows = []
        for i, p in enumerate(self.pipes):
            a = values[i]
            rows.append(
                {
                    "name": p.name,
                    "compute_device": str(masses.device),
                    "model": "homogeneous_equilibrium_blackoil_rs_only",
                    "inlet_pressure_pa": a[0],
                    "outlet_pressure_pa": a[1],
                    "loss_coefficient": p.loss_coefficient,
                    "resistance_model": "uniform_distributed_K",
                    "component_mass_rates_kg_s": dict(zip(("oil", "water", "gas"), a[2:5])),
                    "inlet_acceleration_margin": a[5],
                    "outlet_acceleration_margin": a[6],
                    "inlet_state": {k: v[i][0] for k, v in states.items()},
                    "outlet_state": {k: v[i][1] for k, v in states.items()},
                    "friction_dissipation_w": sum(a[2:5]) * a[8],
                    "mechanical_balance_error_w": a[17],
                    "length_residual_m": a[7] - p.length_m,
                    "quadrature_order": order,
                    "quadrature_length_error_m": a[16],
                    "pressure_evaluations": calls,
                    "derivative_direction": "required_inlet",
                    "pressure_derivatives": dict(
                        zip(
                            (
                                "boundary_pressure_pa",
                                "oil_kg_s",
                                "water_kg_s",
                                "gas_kg_s",
                                "loss_coefficient",
                            ),
                            a[11:16],
                        )
                    ),
                    "hydraulic_backend": "torch_batch",
                }
            )
        return rows


def required_inlets(pipes, boundaries, rates, model):
    """Retry scalar subdivision/choke-bracketing if the bounded batch cannot pass."""
    from .blackoil_pipe import ChokedFlowError

    try:
        return PipeBatch(pipes, model).solve(boundaries, rates)
    except (HydraulicConvergenceError, ChokedFlowError):
        rows = [
            p.required_inlet(b, m, model) for p, b, m in zip(pipes, boundaries, rates, strict=True)
        ]
        for row in rows:
            row["hydraulic_backend"] = "scalar_retry"
        return rows
