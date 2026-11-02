"""CPU homogeneous-equilibrium nozzle for explicit-composition PR mixtures.

Equilibrium phase velocities coincide. No water/association, slip, relaxation,
IEC sizing or composition loss. Inlet is a stagnation state; bulk outlet is a
fully mixed, adiabatic PH state, distinct from the isentropic throat.
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .peng_robinson_mixture import PengRobinsonMixture
from .thermal_equipment import EquilibriumNozzle


class MixtureEquilibriumNozzle(EquilibriumNozzle):
    def run(
        self,
        inlet_pressure_pa,
        inlet_temperature_k,
        back_pressure_pa,
        mole_fractions,
        fluid,
        opening=1.0,
        with_outlet=True,
        _cache=None,
    ):
        if not isinstance(fluid, PengRobinsonMixture):
            raise TypeError("Expected PengRobinsonMixture")
        opening = _number(opening, "opening", nonnegative=True)
        _number(back_pressure_pa, "back_pressure_pa", positive=True)
        if opening > 1:
            raise ValueError("opening must be <= 1")
        if not self.minimum_pressure_pa <= back_pressure_pa <= inlet_pressure_pa:
            raise PropertyModelError("Nozzle pressure outside declared forward-flow domain")
        z = fluid._composition(mole_fractions)
        storage = {} if _cache is None else _cache
        key = (
            "HEM_capacity",
            fluid,
            inlet_pressure_pa,
            inlet_temperature_k,
            tuple(z),
            self.minimum_pressure_pa,
            self.scan_points,
            self.pressure_tolerance_pa,
        )
        if key not in storage:
            storage[key] = {
                "inlet": fluid.flash_pt(inlet_pressure_pa, inlet_temperature_k, z),
                "path": {},
            }
        work = storage[key]
        inlet = work["inlet"]
        if opening == 0 or back_pressure_pa == inlet_pressure_pa:
            return {
                "name": self.name,
                "regime": "closed" if opening == 0 else "no_pressure_drop",
                "mass_rate_kg_s": 0.0,
                "inlet": inlet,
                "throat": None,
                "outlet": None,
                "component_mass_rates_kg_s": [0.0] * len(z),
            }
        mw = sum(x * c.molar_mass_kg_mol for x, c in zip(z, fluid.components))
        hm, sm = inlet["molar_enthalpy_j_mol"], inlet["molar_entropy_j_mol_k"]
        cache = work["path"]

        def state(p):
            p = float(p)
            if p not in cache:
                q = inlet if p == inlet_pressure_pa else fluid.flash_ps(p, sm, z)
                drop = (hm - q["molar_enthalpy_j_mol"]) / mw
                if drop < -1e-4:
                    raise ConservationError("Isentropic expansion increases enthalpy")
                velocity = math.sqrt(2 * max(0.0, drop))
                cache[p] = (q["density_kg_m3"] * velocity, velocity, q)
            return cache[p]

        if "capacity" not in work:
            grid = np.geomspace(self.minimum_pressure_pa, inlet_pressure_pa, self.scan_points)
            values = [state(p)[0] for p in grid]
            candidates = [(values[0], float(grid[0]))]
            peaks = []
            for i in range(1, len(grid) - 1):
                if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
                    peaks.append(i)
                    optimum = minimize_scalar(
                        lambda p: -state(p)[0],
                        bounds=(grid[i - 1], grid[i + 1]),
                        method="bounded",
                        options={"xatol": self.pressure_tolerance_pa, "maxiter": 100},
                    )
                    if not optimum.success:
                        raise HydraulicConvergenceError("Mixture nozzle critical search failed")
                    candidates.append((-float(optimum.fun), float(optimum.x)))
            if len(peaks) > 1:
                raise PropertyModelError(
                    "Multiple sampled mass-flux maxima require branch analysis"
                )
            # Phase onset can be a non-smooth maximum. Locate every sampled change
            # of phase count, retaining both sides rather than differentiating flash.
            for left, right in zip(grid, grid[1:]):
                count = state(left)[2]["phase_count"]
                if count == state(right)[2]["phase_count"]:
                    continue
                for _ in range(80):
                    if right - left <= self.pressure_tolerance_pa:
                        break
                    mid = (left + right) / 2
                    if state(mid)[2]["phase_count"] == count:
                        left = mid
                    else:
                        right = mid
                else:
                    raise HydraulicConvergenceError("Mixture nozzle phase boundary unresolved")
                candidates.extend([(state(left)[0], float(left)), (state(right)[0], float(right))])
            capacity, critical_p = max(candidates)
            resolved = critical_p > self.minimum_pressure_pa + 2 * self.pressure_tolerance_pa
            work["capacity"] = (capacity, critical_p, resolved)
        capacity, critical_p, resolved = work["capacity"]
        choked = resolved and back_pressure_pa <= critical_p
        throat_p = critical_p if choked else back_pressure_pa
        flux, velocity, throat = state(throat_p)
        outlet = fluid.flash_ph(back_pressure_pa, hm, z) if with_outlet else None
        entropy = (outlet["molar_entropy_j_mol_k"] - sm) / mw if outlet is not None else None
        if entropy is not None and entropy < -1e-6:
            raise ConservationError("Nozzle and mixing decrease total entropy")
        mass = self.discharge_coefficient * self.throat_area_m2 * opening * flux
        component_mass = [mass * x * c.molar_mass_kg_mol / mw for x, c in zip(z, fluid.components)]
        return {
            "name": self.name,
            "model": "mixture_PR_homogeneous_equilibrium_nozzle",
            "regime": "choked" if choked else "subcritical",
            "critical_pressure_resolved": resolved,
            "critical_pressure_pa": critical_p if resolved else None,
            "critical_mass_flux_kg_m2_s": capacity if resolved else None,
            "critical_search": "sampled pressure scan with bounded refinement; not a global certificate",
            "mass_rate_kg_s": mass,
            "component_mass_rates_kg_s": component_mass,
            "opening": opening,
            "inlet": inlet,
            "throat": throat,
            "outlet": outlet,
            "throat_velocity_m_s": velocity,
            "isentropic_energy_residual_w": mass
            * ((throat["molar_enthalpy_j_mol"] - hm) / mw + velocity**2 / 2),
            "outlet_energy_residual_w": mass * (outlet["molar_enthalpy_j_mol"] - hm) / mw
            if outlet is not None
            else None,
            "entropy_generation_w_k": mass * entropy if entropy is not None else None,
            "pressure_evaluations": len(cache),
        }
