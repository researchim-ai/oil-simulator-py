"""Rigid, perfectly mixed equilibrium PR vessel with conservative implicit steps.

Component inventories, energy and volume are solved together. This is a CPU
reference, not hydrodynamic separator sizing, water, or a two-fluid model.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .operation import _digest
from .peng_robinson import R
from .thermal_process import ThermalStream, _fluid


@dataclass(frozen=True)
class VesselState:
    component_moles: tuple
    internal_energy_j: float
    pressure_pa: float
    temperature_k: float
    time_s: float
    model_id: str
    wall_temperature_k: float = None
    wall_energy_j: float = None
    wall_model_id: str = None

    def __post_init__(self):
        moles = tuple(_number(x, "component_moles", positive=True) for x in self.component_moles)
        if not 2 <= len(moles) <= 12:
            raise PropertyModelError("Vessel needs 2..12 positive component inventories")
        object.__setattr__(self, "component_moles", moles)
        for key in ("pressure_pa", "temperature_k"):
            _number(getattr(self, key), key, positive=True)
        _number(self.time_s, "time_s", nonnegative=True)
        _number(self.internal_energy_j, "internal_energy_j")
        fields = (self.wall_temperature_k, self.wall_energy_j, self.wall_model_id)
        if any(x is not None for x in fields):
            if any(x is None for x in fields):
                raise ValueError("Wall temperature/energy/model must be present together")
            _number(self.wall_temperature_k, "wall_temperature_k", positive=True)
            _number(self.wall_energy_j, "wall_energy_j")
            if not isinstance(self.wall_model_id, str):
                raise TypeError("wall_model_id must be a string")

    def to_dict(self):
        result = asdict(self)
        result["component_moles"] = list(self.component_moles)
        return result


@dataclass(frozen=True)
class DynamicVessel:
    name: str
    volume_m3: float
    min_pressure_pa: float
    max_pressure_pa: float
    minimum_component_moles: float = 1e-10
    max_fractional_inventory_change: float = 0.1
    max_temperature_change_k: float = 5.0
    min_liquid_volume_fraction: float = None
    max_liquid_volume_fraction: float = None

    def __post_init__(self):
        _name(self.name)
        for key in (
            "volume_m3",
            "min_pressure_pa",
            "max_pressure_pa",
            "minimum_component_moles",
            "max_fractional_inventory_change",
            "max_temperature_change_k",
        ):
            _number(getattr(self, key), key, positive=True)
        for key in ("min_liquid_volume_fraction", "max_liquid_volume_fraction"):
            value = getattr(self, key)
            if value is not None and not 0 <= _number(value, key) <= 1:
                raise ValueError("Liquid volume fraction limit must be in [0,1]")
        if (
            self.min_liquid_volume_fraction is not None
            and self.max_liquid_volume_fraction is not None
            and self.min_liquid_volume_fraction >= self.max_liquid_volume_fraction
        ):
            raise ValueError("Invalid liquid volume fraction interval")
        if self.max_pressure_pa <= self.min_pressure_pa or self.max_fractional_inventory_change > 1:
            raise ValueError("Invalid vessel pressure/inventory-change limits")

    def _values(self, fluid, moles, pressure, temperature):
        total = math.fsum(moles)
        z = np.array(moles) / total
        equilibrium = fluid.flash_pt(pressure, temperature, z)
        volume = total * math.fsum(
            q["molar_fraction"] * q["molar_volume_m3_mol"] for q in equilibrium["phases"]
        )
        energy = total * equilibrium["molar_enthalpy_j_mol"] - pressure * volume
        entropy = total * equilibrium["molar_entropy_j_mol_k"]
        return equilibrium, volume, energy, entropy

    def initialize(self, fluid, pressure_pa, temperature_k, mole_fractions, time_s=0.0):
        _fluid(fluid)
        z = fluid._composition(mole_fractions)
        q = fluid.flash_pt(pressure_pa, temperature_k, z)
        vm = math.fsum(p["molar_fraction"] * p["molar_volume_m3_mol"] for p in q["phases"])
        n = self.volume_m3 / vm
        state = VesselState(
            tuple(n * z),
            n * q["molar_enthalpy_j_mol"] - pressure_pa * self.volume_m3,
            pressure_pa,
            temperature_k,
            time_s,
            _digest(asdict(fluid)),
        )
        self.inspect(state, fluid)
        return state

    def inspect(self, state, fluid):
        _fluid(fluid)
        if state.model_id != _digest(asdict(fluid)) or len(state.component_moles) != len(
            fluid.components
        ):
            raise PropertyModelError("Vessel state model/order mismatch")
        if (
            not self.min_pressure_pa <= state.pressure_pa <= self.max_pressure_pa
            or min(state.component_moles) <= self.minimum_component_moles
        ):
            raise PropertyModelError("Vessel pressure/inventory outside safety domain")
        q, volume, energy, entropy = self._values(
            fluid, state.component_moles, state.pressure_pa, state.temperature_k
        )
        if abs(volume - self.volume_m3) > 1e-8 * self.volume_m3 or abs(
            energy - state.internal_energy_j
        ) > 1e-6 + 1e-8 * max(1, abs(energy)):
            raise ConservationError("Vessel state violates rigid volume or internal energy")
        phase_moles = [sum(state.component_moles) * p["molar_fraction"] for p in q["phases"]]
        fractions = [
            n * p["molar_volume_m3_mol"] / self.volume_m3 for n, p in zip(phase_moles, q["phases"])
        ]
        if (
            self.min_liquid_volume_fraction is not None
            or self.max_liquid_volume_fraction is not None
        ):
            if q["phase_count"] != 2:
                raise PropertyModelError("Liquid level guard requires resolved liquid/vapor phases")
            if (
                self.min_liquid_volume_fraction is not None
                and fractions[0] < self.min_liquid_volume_fraction
                or self.max_liquid_volume_fraction is not None
                and fractions[0] > self.max_liquid_volume_fraction
            ):
                raise PropertyModelError("Vessel liquid volume fraction outside operating limits")
        return {
            "equilibrium": q,
            "phase_moles": phase_moles,
            "phase_volume_fractions": fractions,
            "liquid_volume_fraction": fractions[0] if q["phase_count"] == 2 else None,
            "entropy_j_k": entropy,
            "volume_residual_m3": volume - self.volume_m3,
            "energy_state_residual_j": energy - state.internal_energy_j,
        }

    def _step(self, old, fluid, dt, feeds, withdrawals, heat_w, utility_temperature_k):
        old_info = self.inspect(old, fluid)
        initial = np.array(old.component_moles)
        feed_components = (
            np.sum([s.component_molar_rates_mol_s for s in feeds], axis=0)
            if feeds
            else np.zeros(len(initial))
        )
        feed_energy = math.fsum(s.enthalpy_rate_w for s in feeds)
        feed_entropy = math.fsum(
            s.molar_rate_mol_s * s.state(fluid)["molar_entropy_j_mol_k"]
            for s in feeds
            if s.molar_rate_mol_s
        )
        upper_pressure = min(
            [self.max_pressure_pa] + [s.pressure_pa for s in feeds if s.molar_rate_mol_s]
        )
        if old.pressure_pa > upper_pressure:
            raise PropertyModelError("Passive inlet supply pressure below vessel pressure")
        if upper_pressure <= self.min_pressure_pa:
            raise PropertyModelError("No admissible pressure interval for vessel")
        low_t = max(c.min_temperature_k for c in fluid.components)
        high_t = min(c.max_temperature_k for c in fluid.components)
        nscale = np.maximum(initial, dt * feed_components)
        escale = max(
            1.0,
            abs(old.internal_energy_j),
            sum(initial) * R * old.temperature_k,
            abs(dt * (feed_energy + heat_w)),
        )
        evaluations = 0
        cache = {}

        def evaluate(x):
            nonlocal evaluations
            key = tuple(x)
            if key in cache:
                return cache[key]
            evaluations += 1
            n = np.exp(x[:-2])
            p = math.exp(x[-2])
            t = x[-1]
            q, volume, energy, entropy = self._values(fluid, n, p, t)
            outgoing = np.zeros(len(n))
            hout = 0.0
            sout = 0.0
            for port, rate in withdrawals.items():
                if not rate:
                    continue
                if port == "bulk":
                    z = n / sum(n)
                    h = q["molar_enthalpy_j_mol"]
                    s = q["molar_entropy_j_mol_k"]
                else:
                    index = int(port[-1])
                    if index >= q["phase_count"]:
                        raise PropertyModelError("Requested vessel phase outlet is absent")
                    phase = q["phases"][index]
                    z = np.array(phase["mole_fractions"])
                    h = phase["molar_enthalpy_j_mol"]
                    s = phase["molar_entropy_j_mol_k"]
                outgoing += rate * z
                hout += rate * h
                sout += rate * s
            mass = n - initial - dt * (feed_components - outgoing)
            eres = energy - old.internal_energy_j - dt * (feed_energy - hout + heat_w)
            residual = np.r_[
                mass / nscale, eres / escale, (volume - self.volume_m3) / self.volume_m3
            ]
            cache[key] = (residual, n, p, t, q, energy, entropy, mass, eres, outgoing, hout, sout)
            return cache[key]

        upper_n = initial + dt * feed_components
        lower = np.r_[
            np.full(len(initial), math.log(self.minimum_component_moles)),
            math.log(self.min_pressure_pa),
            low_t,
        ]
        upper = np.r_[np.log(upper_n * (1 + 1e-10)), math.log(upper_pressure), high_t]
        guess = np.r_[np.log(initial), math.log(old.pressure_pa), old.temperature_k]
        fit = least_squares(
            lambda x: evaluate(x)[0],
            guess,
            bounds=(lower, upper),
            xtol=1e-11,
            ftol=1e-11,
            gtol=None,
            max_nfev=100,
        )
        residual, n, p, t, q, energy, entropy, mass, eres, outgoing, hout, sout = evaluate(fit.x)
        if not fit.success or max(abs(residual)) > 1e-9:
            raise HydraulicConvergenceError("Vessel coupled component/energy/volume solve failed")
        if (
            max(abs(n - initial) / initial) > self.max_fractional_inventory_change
            or abs(t - old.temperature_k) > self.max_temperature_change_k
        ):
            raise HydraulicConvergenceError("Vessel state change exceeds step accuracy limits")
        exchange = heat_w / utility_temperature_k if heat_w else 0.0
        entropy_production = (
            entropy - old_info["entropy_j_k"] - dt * (feed_entropy - sout + exchange)
        )
        if entropy_production < -1e-6 - 1e-9 * max(1, abs(entropy)):
            raise ConservationError("Vessel discrete entropy balance is negative")
        if heat_w and (
            heat_w > 0
            and utility_temperature_k < max(t, old.temperature_k)
            or heat_w < 0
            and utility_temperature_k > min(t, old.temperature_k)
        ):
            raise PropertyModelError("Heat reservoir cannot cross vessel endpoint temperature")
        new = VesselState(tuple(n), energy, p, t, old.time_s + dt, old.model_id)
        info = self.inspect(new, fluid)
        return new, {
            "dt_s": dt,
            "component_residuals_mol": mass.tolist(),
            "energy_residual_j": eres,
            "volume_residual_m3": info["volume_residual_m3"],
            "entropy_production_j_k": entropy_production,
            "inlet_component_moles": (dt * feed_components).tolist(),
            "outlet_component_moles": (dt * outgoing).tolist(),
            "inlet_enthalpy_j": dt * feed_energy,
            "outlet_enthalpy_j": dt * hout,
            "heat_j": dt * heat_w,
            "phase_count": q["phase_count"],
            "liquid_volume_fraction": info["liquid_volume_fraction"],
            "nonlinear_evaluations": evaluations,
        }

    def advance(
        self,
        state,
        fluid,
        duration_s,
        feeds=(),
        withdrawals=None,
        heat_w=0.0,
        utility_temperature_k=None,
        initial_dt_s=1.0,
        minimum_dt_s=1e-5,
        max_substeps=10000,
        maximum_dt_s=None,
    ):
        """Constant commands; fully implicit substeps cover the exact interval.

        All state is local. A failed interval never mutates the caller's state.
        Accuracy is first order; state-change guards are not error estimators.
        """
        if state.wall_temperature_k is not None:
            raise PropertyModelError("Wall-bearing states require the coupled network solver")
        duration = _number(duration_s, "duration_s", positive=True)
        dt = _number(initial_dt_s, "initial_dt_s", positive=True)
        maximum = _number(
            maximum_dt_s if maximum_dt_s is not None else dt, "maximum_dt_s", positive=True
        )
        dt = min(dt, maximum)
        minimum = _number(minimum_dt_s, "minimum_dt_s", positive=True)
        _number(heat_w, "heat_w")
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("max_substeps must be a positive integer")
        if heat_w:
            _number(utility_temperature_k, "utility_temperature_k", positive=True)
        feeds = tuple(feeds)
        for s in feeds:
            if not isinstance(s, ThermalStream):
                raise TypeError("Vessel feeds must be ThermalStreams")
            s._validate_fluid(fluid)
            if s.molar_rate_mol_s and s.pressure_pa < state.pressure_pa:
                raise PropertyModelError(
                    "Passive inlet supply pressure below vessel pressure initially"
                )
        withdrawals = dict(withdrawals or {})
        if set(withdrawals) - {"bulk", "phase_0", "phase_1"}:
            raise ValueError("Unknown vessel outlet port")
        for value in withdrawals.values():
            _number(value, "outlet_molar_rate_mol_s", nonnegative=True)
        initial_info = self.inspect(state, fluid)
        if withdrawals.get("phase_1", 0) > 0 and initial_info["equilibrium"]["phase_count"] < 2:
            raise PropertyModelError("Requested vessel phase outlet is absent initially")
        if not math.isfinite(state.time_s + duration) or state.time_s + duration <= state.time_s:
            raise ValueError("Vessel interval cannot advance representable time")
        current = state
        elapsed = 0.0
        records = []
        failures = 0
        while elapsed < duration:
            if len(records) >= max_substeps:
                raise HydraulicConvergenceError("Vessel substep budget exhausted")
            step = min(dt, duration - elapsed)
            try:
                new, record = self._step(
                    current, fluid, step, feeds, withdrawals, heat_w, utility_temperature_k
                )
            except (HydraulicConvergenceError, PropertyModelError) as error:
                failures += 1
                if step / 2 < minimum:
                    raise HydraulicConvergenceError(
                        f"Vessel interval rejected at minimum dt: {error}"
                    ) from error
                dt = step / 2
                continue
            next_elapsed = math.fsum([elapsed, step])
            if next_elapsed <= elapsed or state.time_s + next_elapsed <= current.time_s:
                raise ValueError("Vessel substep cannot advance representable time")
            elapsed = next_elapsed
            # The final timestamp follows the requested interval, independently
            # of floating-point summation of the accepted local step timestamps.
            current = VesselState(
                new.component_moles,
                new.internal_energy_j,
                new.pressure_pa,
                new.temperature_k,
                state.time_s + elapsed,
                new.model_id,
            )
            record["state"] = current.to_dict()
            records.append(record)
            dt = min(maximum, step * 1.5)
        return {
            "state": current.to_dict(),
            "steps": records,
            "failed_steps": failures,
            "duration_s": duration,
            "final": self.inspect(current, fluid),
        }
