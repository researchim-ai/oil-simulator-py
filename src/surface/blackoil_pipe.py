"""Monotone upward/horizontal no-slip black-oil pipe with exact AD sensitivities.

Solve the separated steady momentum integral in pressure coordinates. Splitting
at PVT knots and the moving bubble pressure preserves composition derivatives.
No slip/temperature prediction; constant area and one PVT region per network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.optimize import brentq

from .blackoil_properties import BlackOilMixture
from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    PropertyModelError,
    _number,
)
from .multiphase import HomogeneousPipe


class ChokedFlowError(EquipmentOperatingError):
    """Known physical flow-capacity boundary, distinct from numerical failure."""


@lru_cache(maxsize=32)
def _gauss(order, device="cpu"):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return torch.tensor(nodes, dtype=torch.float64, device=device), torch.tensor(
        weights, dtype=torch.float64, device=device
    )


def _darcy(reynolds, relative_roughness):
    """Differentiable Colebrook root and declared laminar/transition closure."""
    r = reynolds.clamp_min(1e-15)
    # The turbulent root is inactive below Re=2000. Evaluate it only in
    # its valid domain: zero blend does not suppress NaN in autograd.
    turbulent_r = r.clamp_min(2000.0)
    u = torch.full_like(r, 7.0)
    for _ in range(10):
        arg = relative_roughness / 3.7 + 2.51 * u / turbulent_r
        value = u + 2 * torch.log10(arg)
        derivative = 1 + 2 / math.log(10) * (2.51 / turbulent_r) / arg
        u = u - value / derivative
    turbulent = u.reciprocal().square()
    blend = ((r - 2000) / 2000).clamp(0, 1)
    blend = blend.square() * (3 - 2 * blend)
    return (1 - blend) * 64 / r + blend * turbulent


@dataclass(frozen=True)
class BlackOilPipe:
    name: str
    length_m: float
    diameter_m: float
    elevation_change_m: float = 0.0
    roughness_m: float = 0.0
    loss_coefficient: float = 0.0

    def __post_init__(self):
        HomogeneousPipe(
            self.name, self.length_m, self.diameter_m, self.elevation_change_m, self.roughness_m
        )
        _number(self.loss_coefficient, "loss_coefficient", nonnegative=True)
        if self.elevation_change_m < 0:
            raise PropertyModelError(
                "Pressure-coordinate pipe currently requires horizontal or upward flow"
            )

    def _integral(self, lower, upper, masses, model, order, resistance=None):
        resistance = self.loss_coefficient if resistance is None else resistance
        nodes, weights = _gauss(order, str(masses.device))
        pieces = model.split_pressures(lower, upper, masses)
        length = upper * 0
        dissipated = upper * 0
        work = upper * 0
        for a, b in zip(pieces, pieces[1:]):
            half = (b - a) / 2
            p = (a + b) / 2 + half * nodes
            if not p.requires_grad:
                p.requires_grad_(True)
            state = model.tensor_state(p, masses)
            v, mu = state["specific_volume"], state["viscosity_pa_s"]
            vp = torch.autograd.grad(v.sum(), p, create_graph=True)[0]
            flux = masses.sum() / (math.pi * self.diameter_m**2 / 4)
            f = _darcy(flux * self.diameter_m / mu, self.roughness_m / self.diameter_m)
            margin = 1 + flux.square() * vp
            if not bool(torch.isfinite(margin).all()) or bool((margin <= 1e-6).any()):
                raise ChokedFlowError("Black-oil pipe reaches non-subcritical momentum branch")
            drag = (f / self.diameter_m + resistance / self.length_m) * flux.square() * v / 2
            slope = (drag + 9.80665 * self.elevation_change_m / self.length_m / v) / margin
            inverse = slope.reciprocal()
            length = length + half * (weights * inverse).sum()
            dissipated = dissipated + half * (weights * drag * v * inverse).sum()
            work = work + half * (weights * v).sum()
        return length, dissipated, work

    def _checked_integral(self, lower, upper, masses, model, depth=0, resistance=None):
        previous = None
        for order in (16, 32, 64, 128, 256):
            result = self._integral(lower, upper, masses, model, order, resistance)
            values = [float(r.detach()) for r in result]
            value = values[0]
            if not all(math.isfinite(v) for v in values):
                raise HydraulicConvergenceError("Nonfinite pipe quadrature")
            if previous is not None and all(
                abs(v - old) <= max(tol * 0.5**depth, abs(v) * 1e-9)
                for v, old, tol in zip(values, previous, (1e-7, 1e-8, 1e-8))
            ):
                return result, order, abs(value - previous[0])
            previous = values
        if depth >= 8:
            raise HydraulicConvergenceError(
                "Black-oil pipe quadrature did not meet its accuracy target"
            )
        # Re-transition closures are C1 rather than analytic. Subdivide an
        # unresolved interval instead of weakening its error tolerance.
        mid = (lower + upper) / 2
        left, lo, le = self._checked_integral(lower, mid, masses, model, depth + 1, resistance)
        right, ro, re = self._checked_integral(mid, upper, masses, model, depth + 1, resistance)
        return tuple(a + b for a, b in zip(left, right)), max(lo, ro), le + re

    @torch.enable_grad()
    def _solve(self, boundary_pressure_pa, component_mass_rates, model, from_outlet):
        if not isinstance(model, BlackOilMixture):
            raise TypeError("BlackOilPipe requires BlackOilMixture")
        masses = model.validate(boundary_pressure_pa, component_mass_rates).detach().clone()
        boundary = float(boundary_pressure_pa)
        low, high = model.pressure_bounds_pa
        scalar = lambda p: torch.tensor(p, dtype=torch.float64, device=masses.device)
        calls = 0

        def distance(other):
            nonlocal calls
            calls += 1
            a, b = (boundary, other) if from_outlet else (other, boundary)
            if a == b:
                return 0.0
            integral, _, _ = self._checked_integral(scalar(a), scalar(b), masses, model)
            return float(integral[0].detach())

        # Grow the pressure interval, rather than evaluating an inadmissible
        # low-pressure endpoint before finding a valid subcritical solution.
        limit = high if from_outlet else low
        known_state = model.state(boundary, masses.tolist(), derivatives=True)
        v0, mu0 = known_state["specific_volume"], known_state["viscosity_pa_s"]
        flux0 = float(masses.sum()) / (math.pi * self.diameter_m**2 / 4)
        margin0 = 1 + flux0**2 * known_state["derivatives"]["specific_volume"][0]
        if margin0 <= 1e-6:
            raise ChokedFlowError("Known pipe port is outside the subcritical domain")
        f0 = float(
            _darcy(
                torch.tensor(
                    flux0 * self.diameter_m / mu0, dtype=torch.float64, device=masses.device
                ),
                self.roughness_m / self.diameter_m,
            )
        )
        slope0 = (
            (f0 / self.diameter_m + self.loss_coefficient / self.length_m) * flux0**2 * v0 / 2
            + 9.80665 * self.elevation_change_m / self.length_m / v0
        ) / margin0
        increment = max(1.1 * self.length_m * slope0, 1e-3)
        previous = boundary
        found = None
        for _ in range(64):
            trial = boundary + math.copysign(
                min(increment, abs(limit - boundary)), limit - boundary
            )
            try:
                value = distance(trial)
            except ChokedFlowError:
                valid, invalid = previous, trial
                for _ in range(40):
                    mid = (valid + invalid) / 2
                    try:
                        value = distance(mid)
                    except ChokedFlowError:
                        invalid = mid
                        continue
                    if value >= self.length_m:
                        found = mid
                        break
                    valid = mid
                if found is None:
                    raise ChokedFlowError(
                        "Pipe length exceeds its admissible subcritical domain"
                    ) from None
                break
            if value >= self.length_m:
                found = trial
                break
            previous = trial
            if trial == limit:
                break
            increment *= 2
        if found is None:
            raise PropertyModelError("Required pipe pressure is outside PVT coverage")
        root = brentq(
            lambda p: distance(p) - self.length_m,
            min(boundary, found),
            max(boundary, found),
            xtol=1e-5,
            rtol=1e-13,
        )
        pin, pout = (root, boundary) if from_outlet else (boundary, root)
        # Differentiate the quadrature residual with respect to both ports and
        # all component masses, then apply the implicit function theorem.
        x = torch.tensor(
            [pin, pout, *masses.tolist(), self.loss_coefficient],
            dtype=torch.float64,
            device=masses.device,
            requires_grad=True,
        )
        (length, dissipation, work), order, error = self._checked_integral(
            x[1], x[0], x[2:5], model, resistance=x[5]
        )
        grad = torch.autograd.grad(length, x, retain_graph=True)[0]
        unknown = 0 if from_outlet else 1
        known = 1 - unknown
        derivative = -grad / grad[unknown]
        port = torch.tensor(
            [pin, pout], dtype=torch.float64, device=masses.device, requires_grad=True
        )
        pv = model.tensor_state(port, masses)["specific_volume"]
        derivative_v = torch.autograd.grad(pv.sum(), port)[0]
        flux_port = float(masses.sum()) / (math.pi * self.diameter_m**2 / 4)
        margins = 1 + flux_port**2 * derivative_v
        if not bool(torch.isfinite(margins).all()) or bool((margins <= 1e-6).any()):
            raise ChokedFlowError("Black-oil pipe port is outside the subcritical domain")
        inlet = model.state(pin, masses.tolist())
        outlet = model.state(pout, masses.tolist())
        vin, vout = inlet["specific_volume"], outlet["specific_volume"]
        flux = float(masses.sum()) / (math.pi * self.diameter_m**2 / 4)
        mechanical = (
            -float(work.detach())
            + 0.5 * flux**2 * (vout * vout - vin * vin)
            + 9.80665 * self.elevation_change_m
            + float(dissipation.detach())
        )
        scale = max(
            1.0,
            abs(float(work.detach())),
            abs(float(dissipation.detach())),
            9.80665 * self.elevation_change_m,
        )
        if abs(mechanical) > max(1e-5, 1e-8 * scale):
            raise ConservationError(
                "Black-oil pipe mechanical balance failed; check PVT continuity"
            )
        if abs(float(length.detach()) - self.length_m) > max(1e-5, 1e-8 * self.length_m):
            raise HydraulicConvergenceError(
                "Black-oil pipe pressure inversion did not meet length tolerance"
            )
        if not bool(torch.isfinite(derivative).all()):
            raise HydraulicConvergenceError("Nonfinite black-oil pipe sensitivity")
        return {
            "name": self.name,
            "compute_device": str(masses.device),
            "model": "homogeneous_equilibrium_blackoil_rs_only",
            "inlet_pressure_pa": pin,
            "outlet_pressure_pa": pout,
            "loss_coefficient": self.loss_coefficient,
            "resistance_model": "uniform_distributed_K",
            "component_mass_rates_kg_s": dict(zip(("oil", "water", "gas"), masses.tolist())),
            "inlet_acceleration_margin": float(margins[0]),
            "outlet_acceleration_margin": float(margins[1]),
            "inlet_state": inlet,
            "outlet_state": outlet,
            "friction_dissipation_w": float(masses.sum()) * float(dissipation.detach()),
            "mechanical_balance_error_w": float(masses.sum()) * mechanical,
            "length_residual_m": float(length.detach()) - self.length_m,
            "quadrature_order": order,
            "quadrature_length_error_m": error,
            "pressure_evaluations": calls,
            "derivative_direction": "required_inlet" if from_outlet else "outlet",
            "pressure_derivatives": {
                "boundary_pressure_pa": float(derivative[known].detach()),
                "loss_coefficient": float(derivative[5].detach()),
                **dict(
                    zip(("oil_kg_s", "water_kg_s", "gas_kg_s"), derivative[2:5].detach().tolist())
                ),
            },
        }

    def run(self, inlet_pressure_pa, component_mass_rates, model):
        return self._solve(inlet_pressure_pa, component_mass_rates, model, False)

    def required_inlet(self, outlet_pressure_pa, component_mass_rates, model):
        return self._solve(outlet_pressure_pa, component_mass_rates, model, True)
