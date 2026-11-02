"""Pure-fluid PR thermodynamics with explicit caloric data and stable flash.

No mixture rules or black-oil pseudo-component inference. SI, mass-specific h/s.
Constant ideal-gas Cp is valid only over the caller's declared temperature range.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.optimize import brentq

from .liquid import HydraulicConvergenceError, PropertyModelError, _name, _number

R = 8.31446261815324
OMEGA_A = 0.4572355289213822
OMEGA_B = 0.07779607390388846


@dataclass(frozen=True)
class PengRobinsonFluid:
    name: str
    critical_temperature_k: float
    critical_pressure_pa: float
    acentric_factor: float
    molar_mass_kg_mol: float
    ideal_cp_j_mol_k: float
    min_temperature_k: float
    max_temperature_k: float
    reference_temperature_k: float = 298.15
    reference_pressure_pa: float = 100000.0

    def __post_init__(self):
        _name(self.name)
        for key in (
            "critical_temperature_k",
            "critical_pressure_pa",
            "molar_mass_kg_mol",
            "ideal_cp_j_mol_k",
            "min_temperature_k",
            "max_temperature_k",
            "reference_temperature_k",
            "reference_pressure_pa",
        ):
            _number(getattr(self, key), key, positive=True)
        _number(self.acentric_factor, "acentric_factor")
        if self.ideal_cp_j_mol_k <= R:
            raise PropertyModelError("Ideal Cp must exceed the molar gas constant")
        if not self.critical_temperature_k * 0.5 <= self.min_temperature_k < self.max_temperature_k:
            raise PropertyModelError("Declared temperature range must start at or above 0.5 Tc")
        if not self.min_temperature_k <= self.reference_temperature_k <= self.max_temperature_k:
            raise PropertyModelError("Reference temperature must lie inside the declared range")

    @property
    def b(self):
        return OMEGA_B * R * self.critical_temperature_k / self.critical_pressure_pa

    @property
    def attraction_coefficient(self):
        return OMEGA_A * R**2 * self.critical_temperature_k**2 / self.critical_pressure_pa

    def _a(self, t):
        k = 0.37464 + 1.54226 * self.acentric_factor - 0.26992 * self.acentric_factor**2
        tr = (t / self.critical_temperature_k) ** 0.5
        m = 1 + k * (1 - tr)
        if float(m.detach() if torch.is_tensor(m) else m) <= 0:
            raise PropertyModelError("PR alpha outside supported branch")
        constant = self.attraction_coefficient
        return constant * m * m, -constant * k * m / (self.critical_temperature_k * tr)

    def _validate(self, p, t):
        _number(p, "pressure_pa", positive=True)
        _number(t, "temperature_k", positive=True)
        if not self.min_temperature_k <= t <= self.max_temperature_k:
            raise PropertyModelError("Temperature outside declared caloric range")

    def _roots(self, p, t):
        a, _ = self._a(t)
        A = a * p / (R * t) ** 2
        B = self.b * p / (R * t)
        roots = np.roots([1, B - 1, A - 3 * B * B - 2 * B, -A * B + B * B + B**3])
        real = sorted(
            float(z.real) for z in roots if abs(z.imag) < 1e-9 * max(1, abs(z.real)) and z.real > B
        )
        stable = []
        for z in real:
            v = z * R * t / p
            dpdv = (
                -R * t / (v - self.b) ** 2
                + a * (2 * v + 2 * self.b) / (v * v + 2 * self.b * v - self.b**2) ** 2
            )
            if dpdv < 0 and (not stable or abs(z - stable[-1]) > 1e-9 * max(1, abs(z))):
                stable.append(z)
        if not stable:
            raise HydraulicConvergenceError("No mechanically stable PR root")
        return stable

    def _values(self, p, t, z):
        a, at = self._a(t)
        B = self.b * p / (R * t)
        ratio = (
            torch.log((z + (1 + math.sqrt(2)) * B) / (z + (1 - math.sqrt(2)) * B))
            if torch.is_tensor(z)
            else math.log1p(2 * math.sqrt(2) * B / (z + (1 - math.sqrt(2)) * B))
        )
        log = torch.log if torch.is_tensor(z) else math.log
        lnphi = z - 1 - log(z - B) - a / (2 * math.sqrt(2) * self.b * R * t) * ratio
        h = (
            self.ideal_cp_j_mol_k * (t - self.reference_temperature_k)
            + R * t * (z - 1)
            + (t * at - a) / (2 * math.sqrt(2) * self.b) * ratio
        )
        s = (
            self.ideal_cp_j_mol_k * log(t / self.reference_temperature_k)
            - R * log(p / self.reference_pressure_pa)
            + R * log(z - B)
            + at / (2 * math.sqrt(2) * self.b) * ratio
        )
        v = z * R * t / p
        return (
            lnphi,
            h / self.molar_mass_kg_mol,
            s / self.molar_mass_kg_mol,
            v / self.molar_mass_kg_mol,
        )

    @torch.enable_grad()
    def state(self, pressure_pa, temperature_k, phase="stable", derivatives=True):
        self._validate(pressure_pa, temperature_k)
        if phase not in ("stable", "liquid", "vapor"):
            raise ValueError("phase must be stable, liquid or vapor")
        roots = self._roots(pressure_pa, temperature_k)
        fugacities = [self._values(pressure_pa, temperature_k, z)[0] for z in roots]
        if phase == "stable" and len(roots) > 1 and abs(fugacities[0] - fugacities[-1]) < 1e-8:
            raise PropertyModelError("PT at coexistence needs vapor quality or an explicit phase")
        index = (
            int(np.argmin(fugacities))
            if phase == "stable"
            else (0 if phase == "liquid" else len(roots) - 1)
        )
        z = roots[index]
        lnphi, h, s, v = self._values(pressure_pa, temperature_k, z)
        result = {
            "pressure_pa": float(pressure_pa),
            "temperature_k": float(temperature_k),
            "phase": "single" if len(roots) == 1 else ("liquid" if index == 0 else "vapor"),
            "stable": fugacities[index] <= min(fugacities) + 1e-8,
            "compressibility_factor": z,
            "density_kg_m3": 1 / v,
            "specific_volume_m3_kg": v,
            "enthalpy_j_kg": h,
            "entropy_j_kg_k": s,
            "log_fugacity_coefficient": lnphi,
        }
        if not derivatives:
            return result
        x = torch.tensor([pressure_pa, temperature_k], dtype=torch.float64, requires_grad=True)
        p, t = x.unbind()
        a, _ = self._a(t)
        A = a * p / (R * t) ** 2
        B = self.b * p / (R * t)
        z0 = x.new_tensor(z)
        f = z0**3 + (B - 1) * z0**2 + (A - 3 * B**2 - 2 * B) * z0 - A * B + B**2 + B**3
        fz = 3 * z0 * z0 + 2 * (B - 1) * z0 + A - 3 * B * B - 2 * B
        if abs(float(fz.detach())) < 1e-9:
            raise HydraulicConvergenceError("PR tangent near critical/spinodal point is singular")
        implicit = z0 - f / fz
        _, hh, ss, vv = self._values(p, t, implicit)
        grads = {
            key: torch.autograd.grad(value, x, retain_graph=True)[0].detach().numpy()
            for key, value in (
                ("enthalpy_j_kg", hh),
                ("entropy_j_kg_k", ss),
                ("specific_volume_m3_kg", vv),
            )
        }
        cp = float(grads["enthalpy_j_kg"][1])
        cv = (
            cp
            + temperature_k
            * grads["specific_volume_m3_kg"][1] ** 2
            / grads["specific_volume_m3_kg"][0]
        )
        if (
            not cp > 0
            or not cv > 0
            or not all(np.isfinite(value).all() for value in grads.values())
        ):
            raise PropertyModelError(
                "PR caloric model has nonpositive Cp/Cv or nonfinite derivatives"
            )
        result.update(
            cp_j_kg_k=cp,
            cv_j_kg_k=float(cv),
            sound_speed_m_s=math.sqrt(-v * v * cp / (cv * grads["specific_volume_m3_kg"][0])),
            derivatives={
                key: {"pressure_pa": float(g[0]), "temperature_k": float(g[1])}
                for key, g in grads.items()
            },
        )
        return result

    @lru_cache(maxsize=128)  # noqa: B019 - bounded immutable, scalar-only fluid value keys
    def _saturation_pressure(self, t):
        self._validate(1.0, t)
        if not 0.5 * self.critical_temperature_k <= t <= 0.999 * self.critical_temperature_k:
            raise PropertyModelError("Saturation is supported only from 0.5 Tc through 0.999 Tc")
        a, _ = self._a(t)
        tau = R * t * self.b / a
        spin = np.roots([tau, 4 * tau - 2, 2 * tau + 2, -4 * tau + 2, tau - 2])
        vols = sorted(self.b * float(v.real) for v in spin if abs(v.imag) < 1e-8 and v.real > 1)
        if len(vols) != 2:
            raise HydraulicConvergenceError("PR spinodal bracket unresolved")
        pressures = [R * t / (v - self.b) - a / (v * v + 2 * self.b * v - self.b**2) for v in vols]
        pmin, pmax = min(pressures), max(pressures)
        low = max(pmin, pmax * 1e-10) + pmax * 1e-9
        high = pmax * (1 - 1e-9)

        def delta(p):
            roots = self._roots(p, t)
            if len(roots) != 2:
                raise HydraulicConvergenceError("PR coexistence requires two stable roots")
            return self._values(p, t, roots[0])[0] - self._values(p, t, roots[-1])[0]

        if delta(low) * delta(high) > 0:
            raise HydraulicConvergenceError("PR saturation not bracketed")
        return brentq(delta, low, high, xtol=1e-6, rtol=1e-13)

    def saturation_at_temperature(self, temperature_k):
        p = self._saturation_pressure(temperature_k)
        liquid = self.state(p, temperature_k, "liquid")
        vapor = self.state(p, temperature_k, "vapor")
        return {
            "pressure_pa": p,
            "temperature_k": temperature_k,
            "liquid": liquid,
            "vapor": vapor,
            "fugacity_residual": liquid["log_fugacity_coefficient"]
            - vapor["log_fugacity_coefficient"],
        }

    @lru_cache(maxsize=128)  # noqa: B019 - bounded immutable, scalar-only fluid value keys
    def _saturation_temperature(self, p):
        _number(p, "pressure_pa", positive=True)
        lo = self.min_temperature_k
        hi = min(self.max_temperature_k, 0.999 * self.critical_temperature_k)
        if lo >= hi:
            raise PropertyModelError("No saturation interval in declared temperature range")
        if not self._saturation_pressure(lo) <= p <= self._saturation_pressure(hi):
            raise PropertyModelError("Saturation pressure outside supported interval")
        return brentq(
            lambda t: math.log(self._saturation_pressure(t) / p), lo, hi, xtol=1e-10, rtol=1e-14
        )

    def saturation_at_pressure(self, pressure_pa):
        return self.saturation_at_temperature(self._saturation_temperature(pressure_pa))

    def _flash(self, p, target, key):
        _number(p, "pressure_pa", positive=True)
        _number(target, key)
        lo, hi = self.min_temperature_k, self.max_temperature_k
        phase = "stable"
        sat = None
        if lo < 0.999 * self.critical_temperature_k and p < self.critical_pressure_pa:
            top = min(hi, 0.999 * self.critical_temperature_k)
            plow, phigh = self._saturation_pressure(lo), self._saturation_pressure(top)
            if plow <= p <= phigh:
                sat = self.saturation_at_pressure(p)
            elif p > phigh and hi > top:
                raise PropertyModelError(
                    "Flash crosses unsupported near-critical coexistence region"
                )
        if sat is not None:
            l, v = sat["liquid"], sat["vapor"]
            a, b = l[key], v[key]
            if not a < b:
                raise PropertyModelError("Nonpositive latent change in caloric model")
            boundary_tolerance = max(1e-9, abs(b - a) * 1e-11)
            if a - boundary_tolerance <= target <= b + boundary_tolerance:
                quality = max(0.0, min(1.0, (target - a) / (b - a)))
                result = {
                    "pressure_pa": p,
                    "temperature_k": sat["temperature_k"],
                    "phase": "two_phase",
                    "vapor_quality": quality,
                    "stable": True,
                }
                for prop in ("enthalpy_j_kg", "entropy_j_kg_k", "specific_volume_m3_kg"):
                    result[prop] = (1 - quality) * l[prop] + quality * v[prop]
                result["density_kg_m3"] = 1 / result["specific_volume_m3_kg"]
                result["flash_residual"] = result[key] - target
                return result
            if target < a:
                hi = sat["temperature_k"]
                phase = "liquid"
            else:
                lo = sat["temperature_k"]
                phase = "vapor"

        def evaluate(t):
            return self.state(p, t, phase, derivatives=False)

        if (evaluate(lo)[key] - target) * (evaluate(hi)[key] - target) > 0:
            raise PropertyModelError("Flash target outside declared caloric range")
        t = brentq(lambda t: evaluate(t)[key] - target, lo, hi, xtol=1e-10, rtol=1e-14)
        result = self.state(p, t, phase)
        result["flash_residual"] = result[key] - target
        return result

    def flash_ph(self, pressure_pa, enthalpy_j_kg):
        return self._flash(pressure_pa, enthalpy_j_kg, "enthalpy_j_kg")

    def flash_ps(self, pressure_pa, entropy_j_kg_k):
        return self._flash(pressure_pa, entropy_j_kg_k, "entropy_j_kg_k")
