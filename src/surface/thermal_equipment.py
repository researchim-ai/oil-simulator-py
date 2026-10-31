"""Pure-fluid PR thermal letdown and homogeneous-equilibrium nozzle reference.

Separate from black-oil equipment: EOS/caloric inputs are explicit. No slip,
metastable nucleation delay, oil/water mixing or phase-relaxation kinetics.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .peng_robinson import PengRobinsonFluid


@dataclass(frozen=True)
class ThermalLetdown:
    name: str
    outlet_pressure_pa: float

    def __post_init__(self):
        _name(self.name)
        _number(self.outlet_pressure_pa, "outlet_pressure_pa", positive=True)

    def run(self, inlet_pressure_pa, inlet_temperature_k, mass_rate_kg_s, fluid):
        if not isinstance(fluid, PengRobinsonFluid):
            raise TypeError("Expected PengRobinsonFluid")
        mass = _number(mass_rate_kg_s, "mass_rate_kg_s", nonnegative=True)
        inlet = fluid.state(inlet_pressure_pa, inlet_temperature_k)
        if self.outlet_pressure_pa > inlet_pressure_pa:
            raise PropertyModelError("Letdown cannot boost pressure")
        if mass == 0:
            return {
                "name": self.name,
                "status": "no_flow",
                "mass_rate_kg_s": 0.0,
                "inlet": inlet,
                "outlet": None,
            }
        outlet = (
            dict(inlet)
            if self.outlet_pressure_pa == inlet_pressure_pa
            else fluid.flash_ph(self.outlet_pressure_pa, inlet["enthalpy_j_kg"])
        )
        entropy = outlet["entropy_j_kg_k"] - inlet["entropy_j_kg_k"]
        if entropy < -1e-6:
            raise ConservationError("Adiabatic letdown decreases entropy")
        return {
            "name": self.name,
            "model": "pure_PR_isenthalpic_letdown",
            "status": "flowing",
            "mass_rate_kg_s": mass,
            "inlet": inlet,
            "outlet": outlet,
            "energy_residual_w": mass * (outlet["enthalpy_j_kg"] - inlet["enthalpy_j_kg"]),
            "entropy_generation_w_k": mass * entropy,
        }


@dataclass(frozen=True)
class EquilibriumNozzle:
    name: str
    throat_area_m2: float
    minimum_pressure_pa: float
    discharge_coefficient: float = 1.0
    scan_points: int = 33
    pressure_tolerance_pa: float = 0.01

    def __post_init__(self):
        _name(self.name)
        for key in (
            "throat_area_m2",
            "minimum_pressure_pa",
            "discharge_coefficient",
            "pressure_tolerance_pa",
        ):
            _number(getattr(self, key), key, positive=True)
        if self.discharge_coefficient > 1:
            raise ValueError("Cd must not exceed one")
        if (
            isinstance(self.scan_points, bool)
            or not isinstance(self.scan_points, int)
            or self.scan_points < 9
        ):
            raise ValueError("scan_points must be integer >= 9")

    def run(self, inlet_pressure_pa, inlet_temperature_k, back_pressure_pa, fluid, opening=1.0):
        if not isinstance(fluid, PengRobinsonFluid):
            raise TypeError("Expected PengRobinsonFluid")
        a = _number(opening, "opening", nonnegative=True)
        _number(back_pressure_pa, "back_pressure_pa", positive=True)
        if a > 1:
            raise ValueError("opening must be <= 1")
        if not self.minimum_pressure_pa <= back_pressure_pa <= inlet_pressure_pa:
            raise PropertyModelError("Nozzle pressure outside declared forward-flow domain")
        inlet = fluid.state(inlet_pressure_pa, inlet_temperature_k)
        if a == 0 or back_pressure_pa == inlet_pressure_pa:
            return {
                "name": self.name,
                "regime": "closed" if a == 0 else "no_pressure_drop",
                "mass_rate_kg_s": 0.0,
                "inlet": inlet,
                "throat": None,
                "outlet": None,
            }
        h0, s0 = inlet["enthalpy_j_kg"], inlet["entropy_j_kg_k"]
        cache = {}

        def state(p):
            if p not in cache:
                q = inlet if p == inlet_pressure_pa else fluid.flash_ps(p, s0)
                drop = h0 - q["enthalpy_j_kg"]
                if drop < -1e-4:
                    raise ConservationError("Nozzle expansion increases isentropic enthalpy")
                u = math.sqrt(2 * max(0.0, drop))
                cache[p] = (u / q["specific_volume_m3_kg"], u, q)
            return cache[p]

        grid = np.geomspace(self.minimum_pressure_pa, inlet_pressure_pa, self.scan_points).tolist()
        values = [state(p)[0] for p in grid]
        candidates = [(values[0], grid[0])]
        peaks = []
        for i in range(1, len(grid) - 1):
            if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                peaks.append(i)
                optimum = minimize_scalar(
                    lambda p: -state(float(p))[0],
                    bounds=(grid[i - 1], grid[i + 1]),
                    method="bounded",
                    options={"xatol": self.pressure_tolerance_pa, "maxiter": 100},
                )
                if not optimum.success:
                    raise HydraulicConvergenceError("Thermal nozzle critical search failed")
                candidates.append((-float(optimum.fun), float(optimum.x)))
        if len(peaks) > 1:
            raise PropertyModelError(
                "Multiple sampled equilibrium mass-flux maxima require branch analysis"
            )
        # A phase boundary may be a sharp maximum, missed by a smooth optimizer.
        high_t = min(fluid.max_temperature_k, 0.999 * fluid.critical_temperature_k)
        if fluid.min_temperature_k < high_t:
            satlo = fluid.saturation_at_temperature(fluid.min_temperature_k)["pressure_pa"]
            sathi = fluid.saturation_at_temperature(high_t)["pressure_pa"]
            for phase in ("liquid", "vapor"):

                def boundary(p, phase=phase):
                    return fluid.saturation_at_pressure(p)[phase]["entropy_j_kg_k"] - s0

                valid = [p for p in grid if satlo <= p <= sathi]
                for left, right in zip(valid, valid[1:]):
                    if boundary(left) * boundary(right) < 0:
                        p = brentq(
                            boundary, left, right, xtol=self.pressure_tolerance_pa, rtol=1e-13
                        )
                        candidates.append((state(p)[0], p))
        capacity, critical_p = max(candidates)
        critical_resolved = critical_p > self.minimum_pressure_pa + 2 * self.pressure_tolerance_pa
        choked = critical_resolved and back_pressure_pa <= critical_p
        throat_p = critical_p if choked else back_pressure_pa
        flux, velocity, throat = state(throat_p)
        outlet = fluid.flash_ph(back_pressure_pa, h0)
        entropy = outlet["entropy_j_kg_k"] - s0
        if entropy < -1e-6:
            raise ConservationError("Nozzle/mixing decreases total entropy")
        mass = self.discharge_coefficient * self.throat_area_m2 * a * flux
        energy = mass * (throat["enthalpy_j_kg"] + velocity**2 / 2 - h0)
        return {
            "name": self.name,
            "model": "pure_PR_homogeneous_equilibrium_nozzle",
            "regime": "choked" if choked else "subcritical",
            "critical_pressure_resolved": critical_resolved,
            "critical_pressure_pa": critical_p if critical_resolved else None,
            "critical_mass_flux_kg_m2_s": capacity if critical_resolved else None,
            "mass_rate_kg_s": mass,
            "opening": a,
            "inlet": inlet,
            "throat": throat,
            "outlet": outlet,
            "throat_velocity_m_s": velocity,
            "isentropic_energy_residual_w": energy,
            "outlet_energy_residual_w": mass * (outlet["enthalpy_j_kg"] - h0),
            "entropy_generation_w_k": mass * entropy,
            "pressure_evaluations": len(cache),
        }
