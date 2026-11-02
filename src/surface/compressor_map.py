"""Stable-branch dry-gas compressor maps; beta=0 surge, beta=1 choke.

No extrapolation is accepted as a physical operating point. Corrected speed
is in rad/s and corrected mass flow in kg/s at explicit reference p/T.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .gas import IdealGas
from .liquid import ConservationError, EquipmentOperatingError, _number


class CompressorEnvelopeError(EquipmentOperatingError):
    def __init__(self, message, boundary):
        super().__init__(message)
        self.boundary = boundary
        self.last_accepted_state = None


@dataclass(frozen=True)
class CompressorMap:
    corrected_speed_rad_s: tuple
    beta: tuple
    corrected_mass_flow_kg_s: tuple
    pressure_ratio: tuple
    isentropic_efficiency: tuple
    reference_temperature_k: float = 300.0
    reference_pressure_pa: float = 1e5

    def __post_init__(self):
        for key in ("reference_temperature_k", "reference_pressure_pa"):
            _number(getattr(self, key), key, positive=True)
        for key in ("corrected_speed_rad_s", "beta"):
            values = tuple(float(_number(x, key)) for x in getattr(self, key))
            if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
                raise ValueError("Map axes require at least two strictly increasing nodes")
            object.__setattr__(self, key, values)
        if self.corrected_speed_rad_s[0] <= 0 or self.beta[0] != 0 or self.beta[-1] != 1:
            raise ValueError("Require positive speeds and beta from surge=0 to choke=1")
        shape = (len(self.corrected_speed_rad_s), len(self.beta))
        for key in ("corrected_mass_flow_kg_s", "pressure_ratio", "isentropic_efficiency"):
            rows = tuple(
                tuple(float(_number(v, key, positive=True)) for v in row)
                for row in getattr(self, key)
            )
            if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
                raise ValueError("Map tables must match speed x beta dimensions")
            object.__setattr__(self, key, rows)
        if np.any(np.diff(self.corrected_mass_flow_kg_s, axis=1) <= 0):
            raise ValueError("Corrected flow must increase from surge to choke")
        if np.any(np.diff(self.pressure_ratio, axis=1) >= 0) or np.min(self.pressure_ratio) < 1:
            raise ValueError("Pressure ratio must decrease from surge to choke and be >=1")
        if np.max(self.isentropic_efficiency) > 1:
            raise ValueError("Map efficiency must be in (0,1]")

    def to_dict(self):
        return asdict(self)

    def speed_line(self, corrected_speed, strict=True):
        speed = _number(corrected_speed, "corrected_speed_rad_s", nonnegative=True)
        lo, hi = self.corrected_speed_rad_s[0], self.corrected_speed_rad_s[-1]
        tol = 64 * np.finfo(float).eps * max(1, hi)
        if strict and not lo - tol <= speed <= hi + tol:
            raise CompressorEnvelopeError("Corrected speed outside compressor map", "speed")
        speed = min(hi, max(lo, speed))
        i = min(
            max(int(np.searchsorted(self.corrected_speed_rad_s, speed, side="right")) - 1, 0),
            len(self.corrected_speed_rad_s) - 2,
        )
        weight = (speed - self.corrected_speed_rad_s[i]) / (
            self.corrected_speed_rad_s[i + 1] - self.corrected_speed_rad_s[i]
        )
        return tuple(
            (1 - weight) * np.asarray(getattr(self, key)[i])
            + weight * np.asarray(getattr(self, key)[i + 1])
            for key in ("corrected_mass_flow_kg_s", "pressure_ratio", "isentropic_efficiency")
        )

    def operating_point(
        self, speed_rad_s, inlet_pressure_pa, inlet_temperature_k, back_pressure_pa, strict=True
    ):
        speed = _number(speed_rad_s, "speed_rad_s", nonnegative=True)
        pressure = _number(inlet_pressure_pa, "inlet_pressure_pa", positive=True)
        temperature = _number(inlet_temperature_k, "inlet_temperature_k", positive=True)
        back = _number(back_pressure_pa, "back_pressure_pa", positive=True)
        theta = temperature / self.reference_temperature_k
        nc = speed / np.sqrt(theta)
        flows, ratios, efficiencies = self.speed_line(nc, strict)
        requested = max(1.0, back / pressure)
        tol = 64 * np.finfo(float).eps * max(1, float(ratios[0]))
        speed_tol = 64 * np.finfo(float).eps * max(1.0, self.corrected_speed_rad_s[-1])
        if strict and requested > ratios[0] + tol:
            raise CompressorEnvelopeError(
                "Discharge pressure beyond stable surge boundary", "surge"
            )
        # Bounded continuation is used ONLY by internal nonlinear trial states.
        # A strict accepted point must satisfy the original map envelope.
        actual = min(float(ratios[0]), max(float(ratios[-1]), requested))
        beta = float(np.interp(actual, ratios[::-1], self.beta[::-1]))
        corrected_flow = float(np.interp(beta, self.beta, flows))
        eta = float(np.interp(beta, self.beta, efficiencies))
        mass = corrected_flow * (pressure / self.reference_pressure_pa) / np.sqrt(theta)
        return {
            "within_map": bool(
                self.corrected_speed_rad_s[0] - speed_tol
                <= nc
                <= self.corrected_speed_rad_s[-1] + speed_tol
                and requested <= ratios[0] + tol
            ),
            "trial_only": not strict,
            "corrected_speed_rad_s": float(nc),
            "beta": beta,
            "corrected_mass_flow_kg_s": corrected_flow,
            "mass_rate_kg_s": float(mass),
            "pressure_ratio": actual,
            "requested_pressure_ratio": requested,
            "isentropic_efficiency": eta,
            "surge_corrected_mass_flow_kg_s": float(flows[0]),
            "choke_corrected_mass_flow_kg_s": float(flows[-1]),
            "surge_margin": corrected_flow / float(flows[0]) - 1,
            "choke_limited": bool(requested < float(ratios[-1])),
            "discharge_pressure_pa": pressure * actual,
        }


@dataclass(frozen=True)
class MappedGasCompressor:
    map: CompressorMap
    gas: IdealGas
    mechanical_efficiency: float = 0.95
    max_discharge_temperature_k: float = 500.0
    max_shaft_power_w: float | None = None

    def __post_init__(self):
        if not isinstance(self.map, CompressorMap) or not isinstance(self.gas, IdealGas):
            raise TypeError("Expected CompressorMap and IdealGas")
        if not 0 < _number(self.mechanical_efficiency, "mechanical_efficiency") <= 1:
            raise ValueError("Mechanical efficiency must be in (0,1]")
        _number(self.max_discharge_temperature_k, "max_discharge_temperature_k", positive=True)
        if self.max_shaft_power_w is not None:
            _number(self.max_shaft_power_w, "max_shaft_power_w", positive=True)

    def run(
        self, speed_rad_s, inlet_pressure_pa, inlet_temperature_k, back_pressure_pa, strict=True
    ):
        import math

        point = self.map.operating_point(
            speed_rad_s, inlet_pressure_pa, inlet_temperature_k, back_pressure_pa, strict
        )
        gamma = self.gas.heat_capacity_ratio
        delta = (
            inlet_temperature_k
            * math.expm1((gamma - 1) / gamma * math.log(point["pressure_ratio"]))
            / point["isentropic_efficiency"]
        )
        temperature = inlet_temperature_k + delta
        gas_power = point["mass_rate_kg_s"] * self.gas.cp_j_kg_k * delta
        shaft = gas_power / self.mechanical_efficiency
        if strict and temperature > self.max_discharge_temperature_k:
            raise CompressorEnvelopeError("Compressor discharge temperature limit", "temperature")
        if strict and self.max_shaft_power_w is not None and shaft > self.max_shaft_power_w:
            raise CompressorEnvelopeError("Compressor shaft power limit", "shaft_power")
        entropy = self.gas.cp_j_kg_k * math.log(
            temperature / inlet_temperature_k
        ) - self.gas.specific_gas_constant_j_kg_k * math.log(point["pressure_ratio"])
        if entropy < -1e-8:
            raise ConservationError("Mapped compressor decreases gas entropy")
        return {
            **point,
            "discharge_temperature_k": temperature,
            "gas_power_w": gas_power,
            "shaft_power_w": shaft,
            "mechanical_heat_w": shaft - gas_power,
            "compression_entropy_generation_w_k": point["mass_rate_kg_s"] * entropy,
            "discharge_trim_pressure_drop_pa": max(
                0.0, point["discharge_pressure_pa"] - back_pressure_pa
            ),
            "trim_entropy_generation_w_k": point["mass_rate_kg_s"]
            * self.gas.specific_gas_constant_j_kg_k
            * max(0.0, math.log(point["discharge_pressure_pa"] / back_pressure_pa)),
        }
