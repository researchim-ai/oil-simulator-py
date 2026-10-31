"""Dry calorically perfect gas nozzle-equivalent trim, with critical flow.

Plenum stagnation input and receiving pressure; Cd times geometric throat area.
Opening scales area linearly. No real-gas EOS, condensation or IEC valve sizing.
Downstream bulk port is fully mixed/adiabatic, with negligible port velocity.
"""

import math
from dataclasses import asdict, dataclass

from .liquid import EquipmentOperatingError, HydraulicConvergenceError, _name, _number


def ideal_gas_nozzle_flow(
    pressure_pa,
    temperature_k,
    back_pressure_pa,
    gas,
    throat_area_m2,
    discharge_coefficient=1.0,
    opening=1.0,
):
    from .gas import IdealGas

    if not isinstance(gas, IdealGas):
        raise TypeError("Expected IdealGas")
    p = _number(pressure_pa, "pressure_pa", positive=True)
    t = _number(temperature_k, "temperature_k", positive=True)
    pb = _number(back_pressure_pa, "back_pressure_pa", positive=True)
    area = _number(throat_area_m2, "throat_area_m2", positive=True)
    cd = _number(discharge_coefficient, "discharge_coefficient", positive=True)
    a = _number(opening, "opening", nonnegative=True)
    if cd > 1 or a > 1:
        raise ValueError("Require Cd <= 1 and opening <= 1")
    if pb > p:
        raise EquipmentOperatingError("Gas nozzle reverse pressure gradient is unsupported")
    gamma, rg = gas.heat_capacity_ratio, gas.specific_gas_constant_j_kg_k
    exponent = (gamma - 1) / gamma
    log_critical = -gamma / (gamma - 1) * math.log1p((gamma - 1) / 2)
    critical_ratio = math.exp(log_critical)
    ratio = pb / p
    # log1p preserves arbitrarily small positive pressure differences.
    lr = math.log1p((pb - p) / p) if ratio > 0.5 else (math.log(pb) - math.log(p))
    choked = lr <= log_critical
    active_lr = log_critical if choked else lr
    f = (
        2
        * gamma
        / (gamma - 1)
        * math.exp(2 / gamma * active_lr)
        * (-math.expm1(exponent * active_lr))
    )
    root = math.sqrt(max(0.0, f))
    factor = cd * area / math.sqrt(rg * t)
    unit_flow = factor * p * root
    mass = a * unit_flow
    if root == 0:
        dp, db = (0.0, 0.0) if a == 0 else (None, None)
    elif choked:
        dp, db = mass / p, 0.0
    else:
        fp = (
            2
            * math.exp((2 / gamma - 1) * lr)
            * (-1 - (gamma + 1) / (gamma - 1) * math.expm1(exponent * lr))
        )
        dp, db = a * factor * (root - ratio * fp / (2 * root)), a * factor * fp / (2 * root)
    mach = math.sqrt(max(0.0, 2 / (gamma - 1) * math.expm1(-exponent * active_lr)))
    throat_t = t * math.exp(exponent * active_lr)
    throat_p = p * math.exp(active_lr)
    velocity = mach * math.sqrt(gamma * rg * throat_t)
    density = throat_p / (rg * throat_t)
    continuity = mass - cd * area * a * density * velocity
    temperature_drop = -t * math.expm1(exponent * active_lr)
    energy = gas.cp_j_kg_k * temperature_drop - velocity**2 / 2
    return {
        "model": "dry_ideal_gas_nozzle_equivalent",
        "mass_rate_kg_s": mass,
        "opening": a,
        "regime": "closed"
        if a == 0
        else ("no_pressure_drop" if pb == p else ("choked" if choked else "subcritical")),
        "critical_pressure_ratio": critical_ratio,
        "pressure_ratio": ratio,
        "throat_state_defined": mass > 0,
        "throat_pressure_pa": throat_p if mass > 0 else None,
        "throat_temperature_k": throat_t if mass > 0 else None,
        "throat_mach": mach if mass > 0 else None,
        "throat_velocity_m_s": velocity if mass > 0 else None,
        "mass_residual_kg_s": continuity,
        "nozzle_energy_residual_j_kg": energy if mass > 0 else None,
        "pressure_derivative_defined": dp is not None,
        "mass_derivatives": {
            "inlet_pressure_pa": dp,
            "back_pressure_pa": db,
            "temperature_k": -0.5 * mass / t,
            "opening": unit_flow,
            "throat_area_m2": mass / area,
            "discharge_coefficient": mass / cd,
        },
    }


@dataclass(frozen=True)
class GasExportValve:
    name: str
    outlet_pressure_pa: float
    throat_area_m2: float
    discharge_coefficient: float = 1.0
    max_opening: float = 1.0

    def __post_init__(self):
        _name(self.name)
        for key in ("outlet_pressure_pa", "throat_area_m2", "discharge_coefficient"):
            _number(getattr(self, key), key, positive=True)
        _number(self.max_opening, "max_opening", nonnegative=True)
        if self.discharge_coefficient > 1 or self.max_opening > 1:
            raise ValueError("Require Cd <= 1 and max_opening <= 1")

    def run(self, inlet, gas):
        """Required opening for prescribed mass, subject to finite area limit.

        Downstream pressure is specified. This is steady valve control, not a
        fixed-opening imposed-flow solution. Motor/actuator dynamics are absent.
        """
        from .gas import GasState

        if not isinstance(inlet, GasState):
            raise TypeError("Expected GasState")
        if inlet.mass_rate_kg_s == 0:
            return {
                "name": self.name,
                "status": "no_flow",
                "inlet": asdict(inlet),
                "outlet": None,
                "opening": 0.0,
                "capacity_kg_s": None,
                "utilization": 0.0,
                "energy_residual_w": 0.0,
                "mass_residual_kg_s": 0.0,
                "nozzle": None,
            }
        unit = ideal_gas_nozzle_flow(
            inlet.pressure_pa,
            inlet.temperature_k,
            self.outlet_pressure_pa,
            gas,
            self.throat_area_m2,
            self.discharge_coefficient,
            1.0,
        )
        full = unit["mass_rate_kg_s"]
        capacity = full * self.max_opening
        if capacity <= 0 or inlet.mass_rate_kg_s > capacity:
            raise EquipmentOperatingError(
                f"{self.name}: gas valve capacity {capacity:.9g} kg/s below requested {inlet.mass_rate_kg_s:.9g} kg/s"
            )
        opening = inlet.mass_rate_kg_s / full
        flow = ideal_gas_nozzle_flow(
            inlet.pressure_pa,
            inlet.temperature_k,
            self.outlet_pressure_pa,
            gas,
            self.throat_area_m2,
            self.discharge_coefficient,
            opening,
        )
        # The jet mixes adiabatically to a low-velocity bulk port. Perfect-gas
        # enthalpy depends only on T, so outlet bulk T equals inlet stagnation T.
        return {
            "name": self.name,
            "status": "flowing",
            "inlet": asdict(inlet),
            "outlet": asdict(
                GasState(inlet.mass_rate_kg_s, self.outlet_pressure_pa, inlet.temperature_k)
            ),
            "opening": opening,
            "capacity_kg_s": capacity,
            "utilization": inlet.mass_rate_kg_s / capacity,
            "energy_residual_w": 0.0,
            "entropy_generation_w_k": inlet.mass_rate_kg_s
            * gas.specific_gas_constant_j_kg_k
            * math.log(inlet.pressure_pa / self.outlet_pressure_pa),
            "mass_residual_kg_s": inlet.mass_rate_kg_s - flow["mass_rate_kg_s"],
            "opening_derivatives": {
                "mass_rate_kg_s": 1 / full,
                **{
                    k: -opening * v / full
                    for k, v in unit["mass_derivatives"].items()
                    if k
                    in ("inlet_pressure_pa", "back_pressure_pa", "temperature_k", "throat_area_m2")
                    and v is not None
                },
            },
            "nozzle": flow,
        }

    def required_supply_pressure(self, mass_rate_kg_s, temperature_k, gas, opening=None):
        """Unique upstream plenum pressure for positive mass and fixed opening.

        Uses maximum permitted opening by default. No compressor or pipe limits
        are included; callers must enforce those separately.
        """
        from scipy.optimize import brentq

        mass = _number(mass_rate_kg_s, "mass_rate_kg_s", nonnegative=True)
        t = _number(temperature_k, "temperature_k", positive=True)
        a = self.max_opening if opening is None else _number(opening, "opening", nonnegative=True)
        if a > self.max_opening:
            raise ValueError("Requested opening exceeds configured maximum")
        pb = self.outlet_pressure_pa
        reference = ideal_gas_nozzle_flow(
            pb, t, pb, gas, self.throat_area_m2, self.discharge_coefficient, a
        )
        if mass == 0:
            return {
                "status": "no_flow",
                "required_pressure_pa": None if a == 0 else pb,
                "pressure_derivatives": None,
                "nozzle": None,
            }
        if a == 0:
            raise EquipmentOperatingError("Positive gas flow requires an open nozzle")
        transition = pb / reference["critical_pressure_ratio"]
        at_transition = ideal_gas_nozzle_flow(
            transition, t, pb, gas, self.throat_area_m2, self.discharge_coefficient, a
        )
        threshold = at_transition["mass_rate_kg_s"]

        def evaluate(p):
            return ideal_gas_nozzle_flow(
                p, t, pb, gas, self.throat_area_m2, self.discharge_coefficient, a
            )

        if mass >= threshold:
            pressure = transition * (mass / threshold)
        else:
            pressure = brentq(
                lambda p: evaluate(p)["mass_rate_kg_s"] - mass,
                pb,
                transition,
                xtol=1e-7,
                rtol=1e-13,
            )
        result = evaluate(pressure)
        derivatives = result["mass_derivatives"]
        slope = derivatives["inlet_pressure_pa"]
        if slope is None or not math.isfinite(slope) or slope <= 0:
            raise HydraulicConvergenceError("Gas flow is below resolvable pressure precision")
        return {
            "status": "flowing",
            "required_pressure_pa": pressure,
            "nozzle": result,
            "mass_residual_kg_s": result["mass_rate_kg_s"] - mass,
            "pressure_derivatives": {
                "mass_rate_kg_s": 1 / slope,
                **{
                    k: -derivatives[k] / slope
                    for k in (
                        "back_pressure_pa",
                        "temperature_k",
                        "opening",
                        "throat_area_m2",
                        "discharge_coefficient",
                    )
                },
            },
        }

    def required_back_pressure(self, inlet, gas, opening=None):
        """Invert fixed-opening flow; report the nonunique critical plateau.

        When mass equals sonic capacity, all 0 < back pressure <= p_critical
        are admissible. Never return an arbitrary pressure as a unique root.
        """
        from scipy.optimize import brentq

        from .gas import GasState

        if not isinstance(inlet, GasState):
            raise TypeError("Expected GasState")
        a = self.max_opening if opening is None else _number(opening, "opening", nonnegative=True)
        if a > self.max_opening:
            raise ValueError("Requested opening exceeds configured maximum")
        p, t, mass = inlet.pressure_pa, inlet.temperature_k, inlet.mass_rate_kg_s
        equal = ideal_gas_nozzle_flow(
            p, t, p, gas, self.throat_area_m2, self.discharge_coefficient, a
        )
        pc = p * equal["critical_pressure_ratio"]
        capacity = ideal_gas_nozzle_flow(
            p, t, pc, gas, self.throat_area_m2, self.discharge_coefficient, a
        )["mass_rate_kg_s"]
        if a == 0:
            if mass > 0:
                raise EquipmentOperatingError("Positive gas flow requires an open nozzle")
            return {
                "status": "closed",
                "back_pressure_pa": None,
                "max_back_pressure_pa": p,
                "pressure_derivatives": None,
            }
        if mass > capacity * (1 + 1e-12):
            raise EquipmentOperatingError("Requested flow exceeds critical nozzle capacity")
        # This tolerance identifies sonic saturation, not a pressure solve or
        # relaxation of flow-capacity limits. The returned pressure is nonunique.
        if abs(mass - capacity) <= capacity * 1e-12:
            return {
                "status": "critical_plateau",
                "back_pressure_pa": None,
                "max_back_pressure_pa": pc,
                "critical_capacity_kg_s": capacity,
                "capacity_residual_kg_s": mass - capacity,
                "pressure_derivatives": None,
            }
        if mass == 0:
            return {"status": "no_flow", "back_pressure_pa": p, "pressure_derivatives": None}

        def evaluate(pb):
            return ideal_gas_nozzle_flow(
                p, t, pb, gas, self.throat_area_m2, self.discharge_coefficient, a
            )

        back = brentq(
            lambda pb: evaluate(pb)["mass_rate_kg_s"] - mass, pc, p, xtol=1e-7, rtol=1e-13
        )
        result = evaluate(back)
        d = result["mass_derivatives"]
        slope = d["back_pressure_pa"]
        if slope is None or not math.isfinite(slope) or slope >= 0:
            raise HydraulicConvergenceError("Gas back pressure inverse has unresolved sensitivity")
        return {
            "status": "subcritical",
            "back_pressure_pa": back,
            "nozzle": result,
            "mass_residual_kg_s": result["mass_rate_kg_s"] - mass,
            "pressure_derivatives": {
                "mass_rate_kg_s": 1 / slope,
                **{
                    k: -d[k] / slope
                    for k in (
                        "inlet_pressure_pa",
                        "temperature_k",
                        "opening",
                        "throat_area_m2",
                        "discharge_coefficient",
                    )
                },
            },
        }
