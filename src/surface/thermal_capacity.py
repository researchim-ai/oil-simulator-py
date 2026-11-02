"""Shared thermal resources and exact common-throughput control for steady DAGs.

All feed rates scale together. At fixed P/T/eta/split commands the supported
units are extensive in feed rate; phase compositions and specific h/s do not
change. This property does not extend to pressure networks or machine maps.
"""

import copy
import math
from dataclasses import asdict, dataclass

from .liquid import EquipmentOperatingError, _number
from .thermal_flowsheet import run_thermal_flowsheet
from .thermal_process import ThermalStream


@dataclass(frozen=True)
class ThermalResourceLimits:
    max_shaft_power_w: float = None
    max_heating_w: float = None
    max_cooling_w: float = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if value is not None:
                _number(value, key, nonnegative=True)

    def utilization(self, result):
        totals = {
            "shaft_power_w": math.fsum(max(0.0, r["shaft_power_w"]) for r in result["units"]),
            "heating_w": math.fsum(max(0.0, r["heat_duty_w"]) for r in result["units"]),
            "cooling_w": math.fsum(max(0.0, -r["heat_duty_w"]) for r in result["units"]),
        }
        uses = {}
        for metric, actual in totals.items():
            limit = getattr(self, "max_" + metric)
            uses[metric] = {
                "actual": actual,
                "limit": limit,
                "fraction": None if limit is None or limit == 0 else actual / limit,
                "within_limit": limit is None or actual <= limit + max(1e-8, abs(limit) * 1e-10),
            }
        return uses


def period_thermal_units(units, controls):
    """Independent period commands applied to the immutable baseline recipe."""
    result = copy.deepcopy(units)
    if not isinstance(controls, dict):
        raise TypeError("unit_controls must map equipment names to parameter overrides")
    by_name = {u["name"]: u for u in result}
    if len(by_name) != len(result):
        raise ValueError("Duplicate equipment name")
    if set(controls) - set(by_name):
        raise ValueError("Unknown equipment in unit_controls")
    allowed = {
        "heat_exchanger": {
            "outlet_temperature_k",
            "utility_temperature_k",
            "pressure_drop_pa",
            "max_abs_duty_w",
        },
        "pressure_raiser": {
            "outlet_pressure_pa",
            "isentropic_efficiency",
            "max_power_w",
            "max_outlet_temperature_k",
        },
        "separator": {"pressure_pa", "single_phase_outlet"},
        "split": {"fraction"},
        "letdown": {"pressure_pa"},
        "mix": set(),
    }
    for name, patch in controls.items():
        unit = by_name[name]
        if not isinstance(patch, dict) or set(patch) - allowed.get(unit["type"], set()):
            raise ValueError("Unsupported equipment command; topology changes forbidden")
        unit["parameters"].update(copy.deepcopy(patch))
    return result


def run_thermal_capacity(fluid, feeds, units, limits):
    """Choose the largest common feed fraction under extensive power/duty caps.

    Only explicit flow-extensive limits are relaxed for the potential evaluation.
    Phase/domain errors and numerical failures always propagate. Accepted output
    is freshly evaluated with the original equipment caps, never post-scaled.
    """
    if not isinstance(limits, ThermalResourceLimits):
        raise TypeError("Expected ThermalResourceLimits")
    potential_units = copy.deepcopy(units)
    caps = []
    for spec in potential_units:
        key = {"heat_exchanger": "max_abs_duty_w", "pressure_raiser": "max_power_w"}.get(
            spec["type"]
        )
        if key and spec["parameters"].get(key) is not None:
            cap = _number(spec["parameters"][key], key, positive=True)
            caps.append((spec["name"], key, cap))
            spec["parameters"][key] = None
    potential = run_thermal_flowsheet(fluid, feeds, potential_units)
    constraints = []
    for metric, use in limits.utilization(potential).items():
        if use["limit"] is not None and use["actual"] > 0:
            constraints.append((use["limit"] / use["actual"], "shared:" + metric))
    reports = {r["name"]: r for r in potential["units"]}
    for name, key, limit in caps:
        actual = (
            abs(reports[name]["heat_duty_w"])
            if key == "max_abs_duty_w"
            else reports[name]["shaft_power_w"]
        )
        if actual > 0:
            constraints.append((limit / actual, "equipment:" + name + ":" + key))
    fraction = min([1.0] + [x[0] for x in constraints])
    # A tiny inward rounding margin keeps a strict original equipment cap from
    # failing solely due to a changed floating-point summation of scaled rates.
    if 0 < fraction < 1:
        fraction = math.nextafter(fraction * (1 - 1e-12), 0.0)
    selected = {
        name: ThermalStream(
            tuple(n * fraction for n in s.component_molar_rates_mol_s),
            s.pressure_pa,
            s.molar_enthalpy_j_mol,
            s.model_id,
        )
        for name, s in feeds.items()
    }
    accepted = run_thermal_flowsheet(fluid, selected, units)
    utilization = limits.utilization(accepted)
    if not all(r["within_limit"] for r in utilization.values()):
        raise EquipmentOperatingError("Accepted thermal process exceeds shared resources")
    accepted["capacity_control"] = {
        "feed_fraction": fraction,
        "resources": utilization,
        "active_constraints": [
            name for value, name in constraints if value < 1 and abs(value - fraction) <= 1e-9
        ],
        "potential_resources": limits.utilization(potential),
        "unprocessed_component_molar_rates_mol_s": [
            math.fsum(s.component_molar_rates_mol_s[i] * (1 - fraction) for s in feeds.values())
            for i in range(len(fluid.components))
        ],
        "method": "common extensive feed scaling with fresh physical replay",
    }
    return accepted
