"""Localized liquid-only K valve in an Rs-only black-oil network.

Upstream PVT density, small density variation, equal port areas/elevations.
Free gas, flashing/cavitation and static zero flow are explicitly unsupported.
K and recovery factor are equipment inputs, not fitted reservoir parameters.
"""

import math
from dataclasses import dataclass

import torch
from scipy.optimize import brentq

from .blackoil_properties import BlackOilMixture
from .liquid import (
    EquipmentOperatingError,
    HydraulicConvergenceError,
    PropertyModelError,
    _name,
    _number,
)


@dataclass(frozen=True)
class BlackOilLiquidValve:
    name: str
    diameter_m: float
    loss_coefficient: float
    vapor_pressure_pa: float
    pressure_recovery_factor: float

    def __post_init__(self):
        _name(self.name)
        for key in (
            "diameter_m",
            "loss_coefficient",
            "vapor_pressure_pa",
            "pressure_recovery_factor",
        ):
            _number(getattr(self, key), key, positive=True)
        if self.pressure_recovery_factor > 1:
            raise ValueError("pressure_recovery_factor must be in (0,1]")

    @torch.enable_grad()
    def _solve(self, boundary_pressure_pa, component_mass_rates, model, from_outlet):
        if not isinstance(model, BlackOilMixture):
            raise TypeError("Expected BlackOilMixture")
        masses = model.validate(boundary_pressure_pa, component_mass_rates).detach().clone()
        boundary = float(boundary_pressure_pa)
        area = math.pi * self.diameter_m**2 / 4
        flux = masses.sum() / area
        coefficient = self.loss_coefficient

        def residual(pin, pout):
            p = masses.new_tensor(pin)
            v = model.tensor_state(p, masses)["specific_volume"]
            return float((pin - pout - coefficient * flux.square() * v / 2).detach())

        if from_outlet:
            _low, high = model.pressure_bounds_pa
            if residual(high, boundary) < 0:
                raise PropertyModelError("Required local-valve inlet is outside PVT coverage")
            pin = brentq(lambda p: residual(p, boundary), boundary, high, xtol=1e-6, rtol=1e-13)
            pout = boundary
        else:
            pin = boundary
            pout = pin + residual(pin, pin)
            model.validate(pout, masses)
        model.validate(pin, masses)
        inlet = model.state(pin, masses.tolist())
        outlet = model.state(pout, masses.tolist())
        dp = pin - pout
        vena = pin - dp / self.pressure_recovery_factor**2
        if vena <= self.vapor_pressure_pa:
            raise EquipmentOperatingError(
                "Local liquid valve violates vena-contracta vapor-pressure margin"
            )
        model.validate(vena, masses)
        throat = model.state(vena, masses.tolist())
        if any(
            s["phase_mass_rates_kg_s"][2] > 1e-12 * float(masses.sum())
            for s in (inlet, outlet, throat)
        ):
            raise PropertyModelError(
                f"Local liquid valve {self.name} does not support free gas or flashing "
                f"(pin={pin:.6g}, pout={pout:.6g}, vena={vena:.6g} Pa)"
            )
        vin = inlet["specific_volume"]
        volume_change = max(abs(s["specific_volume"] / vin - 1) for s in (outlet, throat))
        density_change = max(abs(vin / s["specific_volume"] - 1) for s in (outlet, throat))
        if density_change > 0.01:
            raise PropertyModelError("Local liquid valve density change exceeds 1 percent")
        x = masses.new_tensor([pin, pout, *masses.tolist(), coefficient], requires_grad=True)
        v = model.tensor_state(x[0], x[2:5])["specific_volume"]
        r = x[0] - x[1] - x[5] * (x[2:5].sum() / area).square() * v / 2
        gradient = torch.autograd.grad(r, x)[0]
        unknown, known = (0, 1) if from_outlet else (1, 0)
        if not bool(torch.isfinite(gradient).all()) or abs(float(gradient[unknown])) < 1e-12:
            raise HydraulicConvergenceError("Singular/nonfinite local valve tangent")
        derivative = -gradient / gradient[unknown]
        if abs(float(r.detach())) > max(1e-5, 1e-9 * dp):
            raise HydraulicConvergenceError("Local valve pressure inversion failed tolerance")
        # Nominal incompressible hydraulic work, not a thermal enthalpy balance.
        dissipation = float(masses.sum()) * dp * vin
        return {
            "name": self.name,
            "model": "liquid_only_local_K_upstream_density",
            "compute_device": str(masses.device),
            "inlet_pressure_pa": pin,
            "outlet_pressure_pa": pout,
            "component_mass_rates_kg_s": dict(zip(("oil", "water", "gas"), masses.tolist())),
            "inlet_state": inlet,
            "outlet_state": outlet,
            "loss_coefficient": coefficient,
            "vena_contracta_pressure_pa": vena,
            "vapor_pressure_margin_pa": vena - self.vapor_pressure_pa,
            "relative_specific_volume_change": volume_change,
            "relative_density_change": density_change,
            "vena_contracta_state": throat,
            "friction_dissipation_w": dissipation,
            "mechanical_balance_error_w": float(masses.sum()) * vin * float(r.detach()),
            "pressure_residual_pa": float(r.detach()),
            "pressure_derivatives": {
                "boundary_pressure_pa": float(derivative[known].detach()),
                "loss_coefficient": float(derivative[5].detach()),
                **dict(
                    zip(("oil_kg_s", "water_kg_s", "gas_kg_s"), derivative[2:5].detach().tolist())
                ),
            },
        }

    def required_inlet(self, outlet_pressure_pa, component_mass_rates, model):
        return self._solve(outlet_pressure_pa, component_mass_rates, model, True)

    def run(self, inlet_pressure_pa, component_mass_rates, model):
        return self._solve(inlet_pressure_pa, component_mass_rates, model, False)
