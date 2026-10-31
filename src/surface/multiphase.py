"""Frozen-composition homogeneous isothermal pipe, not a black-oil flash model.

Oil/water are incompressible, free gas ideal; all phases have equal velocity.
No dissolution, slip, emulsion rheology, flow-pattern correlations or heat model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

from scipy.integrate import solve_ivp

from .gas import IdealGas, IsothermalGasPipe
from .liquid import HydraulicConvergenceError, PropertyModelError, _name, _number


@dataclass(frozen=True)
class HomogeneousFluid:
    oil_density_kg_m3: float = 850.0
    water_density_kg_m3: float = 1000.0
    oil_viscosity_pa_s: float = 0.005
    water_viscosity_pa_s: float = 0.001
    gas: IdealGas = field(default_factory=lambda: IdealGas(500.0, 1.3))
    minimum_pressure_pa: float = 10000.0

    def __post_init__(self):
        for key in (
            "oil_density_kg_m3",
            "water_density_kg_m3",
            "oil_viscosity_pa_s",
            "water_viscosity_pa_s",
            "minimum_pressure_pa",
        ):
            _number(getattr(self, key), key, positive=True)
        if not isinstance(self.gas, IdealGas):
            raise TypeError("Homogeneous fluid requires IdealGas")


@dataclass(frozen=True)
class HomogeneousPipe:
    name: str
    length_m: float
    diameter_m: float
    elevation_change_m: float = 0.0
    roughness_m: float = 0.0

    def __post_init__(self):
        _name(self.name)
        _number(self.elevation_change_m, "elevation_change_m")
        IsothermalGasPipe(self.name, self.length_m, self.diameter_m, self.roughness_m)
        if abs(self.elevation_change_m) > self.length_m:
            raise ValueError("Elevation change cannot exceed pipe length")

    def run(self, inlet_pressure_pa, temperature_k, oil_kg_s, water_kg_s, gas_kg_s, fluid):
        """Integrate steady momentum in the positive flow direction.

        Mixture viscosity is volume-fraction weighted (a declared closure).
        Returned component rates are conserved phase masses, NOT stock gas
        equivalents: dissolved gas must not be passed as free gas here.
        """
        pin = _number(inlet_pressure_pa, "inlet_pressure_pa", positive=True)
        temp = _number(temperature_k, "temperature_k", positive=True)
        rates = [
            _number(x, "phase mass rate", nonnegative=True)
            for x in (oil_kg_s, water_kg_s, gas_kg_s)
        ]
        mass = math.fsum(rates)
        if mass == 0:
            raise PropertyModelError(
                "Zero-flow multiphase holdup is undefined; specify a static fluid model"
            )
        fractions = [q / mass for q in rates]
        a = fractions[0] / fluid.oil_density_kg_m3 + fractions[1] / fluid.water_density_kg_m3
        b = fractions[2] * fluid.gas.specific_gas_constant_j_kg_k * temp
        area = math.pi * self.diameter_m**2 / 4
        flux = mass / area
        critical = flux * math.sqrt(b)
        stop = max(fluid.minimum_pressure_pa, critical * (1 + 1e-6))
        if pin <= stop:
            raise PropertyModelError("Inlet is outside the positive subcritical pressure domain")
        friction = IsothermalGasPipe(self.name, self.length_m, self.diameter_m, self.roughness_m)
        gravity_gradient = 9.80665 * self.elevation_change_m / self.length_m

        def properties(p):
            volumes = [
                fractions[0] / fluid.oil_density_kg_m3,
                fractions[1] / fluid.water_density_kg_m3,
                b / p,
            ]
            v = math.fsum(volumes)
            mu = (
                math.fsum(
                    q * m
                    for q, m in zip(
                        volumes,
                        [
                            fluid.oil_viscosity_pa_s,
                            fluid.water_viscosity_pa_s,
                            fluid.gas.viscosity_pa_s,
                        ],
                    )
                )
                / v
            )
            return v, mu

        def rhs(_s, y):
            # Event localization can evaluate beyond the admissible domain.
            # This extension is never accepted as a physical outlet state.
            p = max(float(y[0]), stop)
            v, mu = properties(p)
            f = friction.friction_factor(mass, SimpleNamespace(viscosity_pa_s=mu))
            drag = f * flux**2 * v / (2 * self.diameter_m)
            dp = -(drag + gravity_gradient / v) / (1 - flux**2 * b / p**2)
            return [dp, drag * v]

        def domain_event(_s, y):
            return y[0] - stop

        domain_event.terminal = True
        domain_event.direction = -1
        sol = solve_ivp(
            rhs,
            (0, self.length_m),
            [pin, 0.0],
            rtol=1e-9,
            atol=[1e-4, 1e-7],
            events=domain_event,
            max_step=self.length_m / 20,
        )
        if not sol.success or sol.status == 1 or sol.t[-1] != self.length_m:
            raise HydraulicConvergenceError("Pipe reaches choking or minimum-pressure boundary")
        pout, dissipation = map(float, sol.y[:, -1])
        vin, _ = properties(pin)
        vout, _ = properties(pout)
        pressure_work = a * (pout - pin) + b * math.log(pout / pin)
        kinetic = 0.5 * flux**2 * (vout**2 - vin**2)
        gravity = 9.80665 * self.elevation_change_m
        return {
            "name": self.name,
            "model": "homogeneous_frozen_isothermal",
            "inlet_pressure_pa": pin,
            "outlet_pressure_pa": pout,
            "temperature_k": temp,
            "phase_mass_rates_kg_s": dict(zip(("oil", "water", "gas"), rates)),
            "component_mass_residuals_kg_s": dict.fromkeys(("oil", "water", "gas"), 0.0),
            "inlet_gas_volume_fraction": (b / pin) / vin,
            "outlet_gas_volume_fraction": (b / pout) / vout,
            "inlet_velocity_m_s": flux * vin,
            "outlet_velocity_m_s": flux * vout,
            "outlet_acceleration_margin": 1 - flux**2 * b / pout**2,
            "friction_dissipation_w": mass * dissipation,
            "mechanical_balance_error_w": mass * (pressure_work + kinetic + gravity + dissipation),
            "rhs_evaluations": sol.nfev,
        }
