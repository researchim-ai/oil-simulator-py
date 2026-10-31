"""Explicit-composition PR reference: constant kij, PT VLE, molar balances.

CPU only. Multistart tangent-plane tests are numerical searches, not proofs of
absolute global stability. Water/association, LLE and three phases unsupported.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import softmax

from .liquid import HydraulicConvergenceError, PropertyModelError
from .peng_robinson import PengRobinsonFluid, R


@dataclass(frozen=True)
class PengRobinsonMixture:
    components: tuple
    binary_interactions: tuple

    def __post_init__(self):
        components = tuple(self.components)
        if not 2 <= len(components) <= 12 or not all(
            isinstance(c, PengRobinsonFluid) for c in components
        ):
            raise PropertyModelError("Mixture requires 2..12 explicit PengRobinsonFluid components")
        if len({c.name for c in components}) != len(components):
            raise PropertyModelError("Component names must be unique")
        if len({(c.reference_temperature_k, c.reference_pressure_pa) for c in components}) != 1:
            raise PropertyModelError("Components must share the caloric reference state")
        k = np.asarray(self.binary_interactions, dtype=float)
        n = len(components)
        if (
            k.shape != (n, n)
            or not np.isfinite(k).all()
            or not np.allclose(k, k.T, atol=1e-14, rtol=0)
            or np.any(np.diag(k) != 0)
            or np.any(k >= 1)
        ):
            raise PropertyModelError(
                "Explicit kij must be finite, symmetric, diagonal zero and < 1"
            )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "binary_interactions", tuple(map(tuple, (k + k.T) / 2)))

    def _composition(self, fractions):
        x = np.asarray(fractions, dtype=float)
        if (
            x.shape != (len(self.components),)
            or not np.isfinite(x).all()
            or np.any(x <= 0)
            or abs(x.sum() - 1) > 1e-10
        ):
            raise PropertyModelError(
                "Mole fractions must be positive and sum to one; remove absent species"
            )
        return x / x.sum()

    def _parameters(self, t, x):
        a, at = np.array([c._a(t) for c in self.components]).T
        b = np.array([c.b for c in self.components])
        aij = np.sqrt(a[:, None] * a[None, :]) * (1 - np.array(self.binary_interactions))
        aijt = aij * 0.5 * (at[:, None] / a[:, None] + at[None, :] / a[None, :])
        return x @ aij @ x, x @ aijt @ x, x @ b, aij, b

    def state(self, pressure_pa, temperature_k, mole_fractions, phase="stable"):
        """Homogeneous root; 'stable' selects Gibbs minimum AT FIXED composition.

        Compositional phase stability requires stability()/flash_pt(), separately.
        """
        p, t = float(pressure_pa), float(temperature_k)
        for c in self.components:
            c._validate(p, t)
        if phase not in ("stable", "liquid", "vapor"):
            raise ValueError("phase must be stable, liquid or vapor")
        x = self._composition(mole_fractions)
        a, at, b, aij, bi = self._parameters(t, x)
        A, B = a * p / (R * t) ** 2, b * p / (R * t)
        roots = np.roots([1, B - 1, A - 3 * B * B - 2 * B, -A * B + B * B + B**3])
        states = []
        for z in sorted(float(v.real) for v in roots if abs(v.imag) < 1e-9 and v.real > B):
            v = z * R * t / p
            dpdv = -R * t / (v - b) ** 2 + a * (2 * v + 2 * b) / (v * v + 2 * b * v - b * b) ** 2
            if dpdv >= 0:
                continue
            ratio = math.log1p(2 * math.sqrt(2) * B / (z + (1 - math.sqrt(2)) * B))
            phi = (
                bi / b * (z - 1)
                - math.log(z - B)
                - a / (2 * math.sqrt(2) * b * R * t) * (2 * (aij @ x) / a - bi / b) * ratio
            )
            cp = np.array([c.ideal_cp_j_mol_k for c in self.components])
            tref, pref = (
                self.components[0].reference_temperature_k,
                self.components[0].reference_pressure_pa,
            )
            h = (
                x @ cp * (t - tref)
                + R * t * (z - 1)
                + (t * at - a) / (2 * math.sqrt(2) * b) * ratio
            )
            s = (
                x @ cp * math.log(t / tref)
                - R * math.log(p / pref)
                - R * (x @ np.log(x))
                + R * math.log(z - B)
                + at / (2 * math.sqrt(2) * b) * ratio
            )
            mw = x @ np.array([c.molar_mass_kg_mol for c in self.components])
            states.append(
                {
                    "compressibility_factor": z,
                    "mole_fractions": x.tolist(),
                    "log_fugacity_coefficients": phi.tolist(),
                    "molar_volume_m3_mol": v,
                    "molar_mass_kg_mol": float(mw),
                    "density_kg_m3": mw / v,
                    "molar_enthalpy_j_mol": h,
                    "molar_entropy_j_mol_k": s,
                    "enthalpy_j_kg": h / mw,
                    "entropy_j_kg_k": s / mw,
                }
            )
        if not states:
            raise HydraulicConvergenceError("No mechanically stable mixture PR root")
        best = min(
            range(len(states)), key=lambda i: x @ np.array(states[i]["log_fugacity_coefficients"])
        )
        if phase == "stable" and len(states) > 1:
            g = [x @ np.array(q["log_fugacity_coefficients"]) for q in states]
            if abs(g[0] - g[-1]) < 1e-10:
                raise PropertyModelError(
                    "Degenerate coexistence root: PT does not determine phase fraction"
                )
        i = best if phase == "stable" else (0 if phase == "liquid" else len(states) - 1)
        return dict(
            states[i],
            pressure_pa=p,
            temperature_k=t,
            root_phase="single" if len(states) == 1 else ("liquid" if i == 0 else "vapor"),
        )

    def _wilson(self, p, t):
        return np.array(
            [
                math.log(c.critical_pressure_pa / p)
                + 5.373 * (1 + c.acentric_factor) * (1 - c.critical_temperature_k / t)
                for c in self.components
            ]
        )

    def _tpd(self, p, t, z, d):
        """Minimize dimensionless TPD on the lower homogeneous Gibbs envelope."""
        n = len(z)
        lk = np.clip(self._wilson(p, t), -30, 30)
        starts = [z, softmax(np.log(z) + lk), softmax(np.log(z) - lk)]
        starts += [0.98 * np.eye(n)[i] + 0.02 * z for i in range(n)]
        minima = []

        def objective(u):
            w = softmax(np.r_[u, 0.0])
            state = self.state(p, t, w, "stable")
            delta = np.log(w) + state["log_fugacity_coefficients"] - d
            val = float(w @ delta)
            return val, (w * (delta - val))[:-1]

        for start in starts:
            u = np.log(start[:-1] / start[-1])
            fit = minimize(
                objective,
                np.clip(u, -40, 40),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-40, 40)] * (n - 1),
                options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 250, "maxls": 40},
            )
            value, grad = objective(fit.x)
            if not np.isfinite(value) or np.max(np.abs(grad)) > 2e-6:
                raise HydraulicConvergenceError(
                    "TPD minimization unresolved; refusing phase classification"
                )
            minima.append(
                {
                    "tpd": value,
                    "mole_fractions": softmax(np.r_[fit.x, 0.0]).tolist(),
                    "root": "stable",
                }
            )
        return sorted(minima, key=lambda m: m["tpd"])

    def stability(self, pressure_pa, temperature_k, mole_fractions):
        z = self._composition(mole_fractions)
        feed = self.state(pressure_pa, temperature_k, z)
        d = np.log(z) + feed["log_fugacity_coefficients"]
        minima = self._tpd(pressure_pa, temperature_k, z, d)
        return {
            "stable": minima[0]["tpd"] >= -1e-8,
            "minimum_tpd": minima[0]["tpd"],
            "search": "multistart numerical TPD; not a global certificate",
            "trials": minima,
        }

    def flash_pt(self, pressure_pa, temperature_k, mole_fractions):
        """One or two equilibrium phases. Reject unresolved/third-phase instability."""
        p, t = float(pressure_pa), float(temperature_k)
        z = self._composition(mole_fractions)
        feed = self.state(p, t, z)
        stability = self.stability(p, t, z)
        if stability["stable"]:
            return {
                "pressure_pa": p,
                "temperature_k": t,
                "phase_count": 1,
                "phases": [dict(feed, molar_fraction=1.0)],
                "stability": stability,
                "component_balance_error": 0.0,
                "molar_enthalpy_j_mol": feed["molar_enthalpy_j_mol"],
                "molar_entropy_j_mol_k": feed["molar_entropy_j_mol_k"],
                "density_kg_m3": feed["density_kg_m3"],
            }

        def residual(u):
            k, beta = np.exp(u[:-1]), u[-1]
            x = z / (1 + beta * (k - 1))
            y = k * x
            l = self.state(p, t, x / x.sum(), "liquid")
            v = self.state(p, t, y / y.sum(), "vapor")
            return np.r_[
                u[:-1] + np.array(v["log_fugacity_coefficients"]) - l["log_fugacity_coefficients"],
                np.sum(z * (k - 1) / (1 + beta * (k - 1))),
            ]

        starts = [self._wilson(p, t)]
        for trial in stability["trials"]:
            if trial["tpd"] < -1e-8:
                w = np.array(trial["mole_fractions"])
                starts.extend([np.log(w / z), np.log(z / w)])
        candidates = []
        for lk in starts:
            fit = least_squares(
                residual,
                np.r_[np.clip(lk, -30, 30), 0.5],
                bounds=(np.r_[np.full(len(z), -40), 1e-9], np.r_[np.full(len(z), 40), 1 - 1e-9]),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=250,
            )
            if np.max(np.abs(residual(fit.x))) > 1e-8 or np.max(np.abs(fit.x[:-1])) < 1e-5:
                continue
            k, beta = np.exp(fit.x[:-1]), float(fit.x[-1])
            x = z / (1 + beta * (k - 1))
            y = k * x
            l = self.state(p, t, x / x.sum(), "liquid")
            v = self.state(p, t, y / y.sum(), "vapor")
            gibbs = lambda q: (
                np.array(q["mole_fractions"])
                @ (np.log(q["mole_fractions"]) + q["log_fugacity_coefficients"])
            )
            g = (1 - beta) * gibbs(l) + beta * gibbs(v)
            if g >= gibbs(feed) - 1e-9 or l["molar_volume_m3_mol"] >= v["molar_volume_m3_mol"]:
                continue
            candidates.append((g, beta, l, v))
        if not candidates:
            raise HydraulicConvergenceError("Unstable feed: no resolved nontrivial VLE flash")
        _, beta, liquid, vapor = min(candidates, key=lambda c: c[0])
        d = np.log(liquid["mole_fractions"]) + liquid["log_fugacity_coefficients"]
        post = self._tpd(p, t, z, d)
        if post[0]["tpd"] < -1e-8:
            raise PropertyModelError(
                "Two-phase split still unstable; additional phases unsupported"
            )
        balance = (
            (1 - beta) * np.array(liquid["mole_fractions"])
            + beta * np.array(vapor["mole_fractions"])
            - z
        )
        if np.max(np.abs(balance)) > 1e-8:
            raise HydraulicConvergenceError("Flash component balance failed")

        def average(key):
            return (1 - beta) * liquid[key] + beta * vapor[key]

        return {
            "pressure_pa": p,
            "temperature_k": t,
            "phase_count": 2,
            "vapor_molar_fraction": beta,
            "phases": [dict(liquid, molar_fraction=1 - beta), dict(vapor, molar_fraction=beta)],
            "stability": {"minimum_tpd": post[0]["tpd"], "search": stability["search"]},
            "component_balance_error": float(np.max(np.abs(balance))),
            "log_fugacity_error": float(
                np.max(
                    np.abs(
                        residual(
                            np.r_[
                                np.log(
                                    np.array(vapor["mole_fractions"]) / liquid["mole_fractions"]
                                ),
                                beta,
                            ]
                        )
                    )
                )
            ),
            "molar_enthalpy_j_mol": average("molar_enthalpy_j_mol"),
            "molar_entropy_j_mol_k": average("molar_entropy_j_mol_k"),
            "density_kg_m3": average("molar_mass_kg_mol") / average("molar_volume_m3_mol"),
        }

    def _flash_property(self, pressure_pa, target, mole_fractions, key):
        from scipy.optimize import brentq

        if not math.isfinite(target):
            raise PropertyModelError("Flash target must be finite")
        z = self._composition(mole_fractions)
        low = max(c.min_temperature_k for c in self.components)
        high = min(c.max_temperature_k for c in self.components)
        if low >= high:
            raise PropertyModelError("No common caloric temperature interval")

        def residual(t):
            return self.flash_pt(pressure_pa, t, z)[key] - target

        f_low, f_high = residual(low), residual(high)
        if f_low * f_high > 0:
            raise PropertyModelError("Target outside declared temperature bracket")
        temperature = brentq(residual, low, high, xtol=1e-9, rtol=1e-13)
        result = self.flash_pt(pressure_pa, temperature, z)
        result["property_residual"] = result[key] - target
        if abs(result["property_residual"]) > max(1e-6, abs(target) * 1e-9):
            raise HydraulicConvergenceError("Thermal flash residual exceeds tolerance")
        return result

    def flash_ph(self, pressure_pa, molar_enthalpy_j_mol, mole_fractions):
        return self._flash_property(
            pressure_pa, molar_enthalpy_j_mol, mole_fractions, "molar_enthalpy_j_mol"
        )

    def flash_ps(self, pressure_pa, molar_entropy_j_mol_k, mole_fractions):
        return self._flash_property(
            pressure_pa, molar_entropy_j_mol_k, mole_fractions, "molar_entropy_j_mol_k"
        )
