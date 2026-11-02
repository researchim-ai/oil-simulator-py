"""PR single-phase sonic nozzle, with analytic Cp/Cv and acoustic properties.

Homogeneous PS roots are checked against the full mixture PT stability/flash.
No perfect-gas substitution, phase-change throat, or IEC equipment sizing.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .peng_robinson import R


def single_phase_properties(fluid, pressure_pa, temperature_k, mole_fractions):
    z = fluid._composition(mole_fractions)
    eq = fluid.flash_pt(pressure_pa, temperature_k, z)
    if eq["phase_count"] != 1:
        raise PropertyModelError("Single-phase nozzle path enters a multiphase state")
    q = fluid.state(pressure_pa, temperature_k, z)
    a, at, b, aij, _ = fluid._parameters(temperature_k, z)
    ai = []
    ati = []
    atti = []
    for c in fluid.components:
        av, atv = c._a(temperature_k)
        ai.append(av)
        ati.append(atv)
        k = 0.37464 + 1.54226 * c.acentric_factor - 0.26992 * c.acentric_factor**2
        a0 = c.attraction_coefficient
        atti.append(
            a0 * k * (1 + k) / (2 * math.sqrt(c.critical_temperature_k) * temperature_k**1.5)
        )
    ai = np.array(ai)
    ati = np.array(ati)
    atti = np.array(atti)
    logs = ati / ai
    log_derivative = 0.5 * (logs[:, None] + logs[None, :])
    log_second = atti / ai - logs * logs
    aijtt = aij * (log_derivative**2 + 0.5 * (log_second[:, None] + log_second[None, :]))
    att = z @ aijtt @ z
    vm = q["molar_volume_m3_mol"]
    denom = vm * vm + 2 * b * vm - b * b
    pv = -R * temperature_k / (vm - b) ** 2 + a * (2 * vm + 2 * b) / denom**2
    pt = R / (vm - b) - at / denom
    ratio = math.log1p(2 * math.sqrt(2) * b / (vm + (1 - math.sqrt(2)) * b))
    cv = (
        sum(x * c.ideal_cp_j_mol_k for x, c in zip(z, fluid.components))
        - R
        + temperature_k * att * ratio / (2 * math.sqrt(2) * b)
    )
    cp = cv - temperature_k * pt * pt / pv
    mw = q["molar_mass_kg_mol"]
    sound2 = -vm * vm * cp * pv / (mw * cv)
    if (
        not pv < 0
        or not cv > 0
        or not cp > 0
        or not sound2 > 0
        or not all(math.isfinite(x) for x in (pv, cv, cp, sound2))
    ):
        raise PropertyModelError("Single-phase PR acoustic/caloric tangent is invalid")
    return {
        **q,
        "cp_j_mol_k": float(cp),
        "cv_j_mol_k": float(cv),
        "sound_speed_m_s": math.sqrt(sound2),
        "molar_volume_pressure_derivative": 1 / pv,
        "molar_volume_temperature_derivative": -pt / pv,
    }


@dataclass(frozen=True)
class PRSinglePhaseNozzle:
    name: str
    throat_area_m2: float
    minimum_pressure_ratio: float = 0.4
    discharge_coefficient: float = 1.0
    pressure_tolerance_pa: float = 0.001

    def __post_init__(self):
        _name(self.name)
        for key in (
            "throat_area_m2",
            "minimum_pressure_ratio",
            "discharge_coefficient",
            "pressure_tolerance_pa",
        ):
            _number(getattr(self, key), key, positive=True)
        if self.minimum_pressure_ratio >= 1 or self.discharge_coefficient > 1:
            raise ValueError("Require minimum pressure ratio < 1 and Cd <= 1")

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
        opening = _number(opening, "opening", nonnegative=True)
        _number(back_pressure_pa, "back_pressure_pa", positive=True)
        if opening > 1 or back_pressure_pa > inlet_pressure_pa:
            raise PropertyModelError("Single-phase nozzle requires forward flow and opening <= 1")
        z = tuple(fluid._composition(mole_fractions))
        cache = {} if _cache is None else _cache
        key = (
            fluid,
            inlet_pressure_pa,
            inlet_temperature_k,
            z,
            self.minimum_pressure_ratio,
            self.pressure_tolerance_pa,
        )
        if key not in cache:
            inlet = single_phase_properties(fluid, inlet_pressure_pa, inlet_temperature_k, z)
            cache[key] = {"inlet": inlet, "path": {inlet_pressure_pa: inlet}}
        work = cache[key]
        inlet = work["inlet"]
        hm = inlet["molar_enthalpy_j_mol"]
        sm = inlet["molar_entropy_j_mol_k"]
        mw = inlet["molar_mass_kg_mol"]
        if opening == 0 or inlet_pressure_pa == back_pressure_pa:
            return {
                "regime": "closed" if opening == 0 else "no_pressure_drop",
                "mass_rate_kg_s": 0.0,
                "inlet": inlet,
                "throat": None,
                "outlet": None,
            }

        def state(p):
            p = float(p)
            if p not in work["path"]:

                def entropy(t):
                    return fluid.state(p, t, z)["molar_entropy_j_mol_k"] - sm

                lo = max(c.min_temperature_k for c in fluid.components)
                hi = min(c.max_temperature_k for c in fluid.components)
                if entropy(lo) * entropy(hi) > 0:
                    raise PropertyModelError(
                        "Single-phase isentrope outside declared temperature range"
                    )
                t = brentq(entropy, lo, hi, xtol=1e-9, rtol=1e-13)
                q = single_phase_properties(fluid, p, t, z)
                if abs(q["molar_entropy_j_mol_k"] - sm) > 1e-7:
                    raise HydraulicConvergenceError("Single-phase PS root has unresolved entropy")
                work["path"][p] = q
            return work["path"][p]

        def sonic(p):
            q = state(p)
            return 2 * (hm - q["molar_enthalpy_j_mol"]) / mw - q["sound_speed_m_s"] ** 2

        lower = self.minimum_pressure_ratio * inlet_pressure_pa
        if "critical_pressure_pa" not in work:
            # Follow the inlet-connected subsonic branch, stopping at its first
            # sonic bracket. States below a resolved throat are not required.
            previous = inlet_pressure_pa
            bracket = None
            for proposed in np.geomspace(inlet_pressure_pa, lower, 5)[1:]:
                trial = float(proposed)
                for _ in range(40):
                    try:
                        value = sonic(trial)
                    except PropertyModelError as error:
                        if previous - trial <= self.pressure_tolerance_pa:
                            raise PropertyModelError(
                                "Single-phase domain ends before a sonic throat"
                            ) from error
                        trial = (previous + trial) / 2
                        continue
                    break
                else:
                    raise HydraulicConvergenceError("Single-phase sonic bracket unresolved")
                if value >= 0:
                    bracket = (trial, previous)
                    break
                previous = trial
            work["critical_pressure_pa"] = (
                brentq(sonic, *bracket, xtol=self.pressure_tolerance_pa, rtol=1e-12)
                if bracket
                else None
            )
        critical = work["critical_pressure_pa"]
        if critical is None and back_pressure_pa < lower:
            raise PropertyModelError("Critical pressure unresolved in single-phase domain")
        choked = critical is not None and back_pressure_pa <= critical
        throat = state(critical if choked else back_pressure_pa)
        drop = (hm - throat["molar_enthalpy_j_mol"]) / mw
        if drop < -1e-5:
            raise ConservationError("Single-phase nozzle expansion increases enthalpy")
        velocity = math.sqrt(2 * max(0.0, drop))
        mass = (
            self.discharge_coefficient
            * self.throat_area_m2
            * opening
            * throat["density_kg_m3"]
            * velocity
        )
        outlet = fluid.flash_ph(back_pressure_pa, hm, z) if with_outlet else None
        if outlet is not None and outlet["molar_entropy_j_mol_k"] < sm - 1e-7:
            raise ConservationError("Single-phase nozzle bulk mixing decreases entropy")
        return {
            "model": "single_phase_PR_sonic_nozzle",
            "regime": "choked" if choked else "subcritical",
            "mass_rate_kg_s": mass,
            "critical_pressure_pa": critical,
            "throat": throat,
            "inlet": inlet,
            "outlet": outlet,
            "throat_velocity_m_s": velocity,
            "throat_mach": velocity / throat["sound_speed_m_s"],
            "isentropic_energy_residual_w": mass
            * ((throat["molar_enthalpy_j_mol"] - hm) / mw + velocity**2 / 2),
            "outlet_energy_residual_w": mass * (outlet["molar_enthalpy_j_mol"] - hm) / mw
            if outlet
            else None,
            "path_states": len(work["path"]),
        }
