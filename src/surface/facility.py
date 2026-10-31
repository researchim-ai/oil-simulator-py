"""Facility capacity audit and conservative separated-water network handoff.

Water is withdrawn through an ideal level/flow controller and let down to the
network inlet pressure. Optional dry ideal-gas conditioning and one export pipe
use prescribed separator gas rates. Reservoir/well feedback, separator inventory,
wet-gas thermodynamics and arbitrary gas networks are not solved here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

from .gas import GasExportSystem
from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    LiquidNetwork,
    NetworkResult,
    _name,
    _number,
)
from .process import ComponentStream, run_separator_train
from .station import PumpStationControl


@dataclass(frozen=True)
class FacilityLimit:
    name: str
    metric: str
    unit: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self):
        for value in (self.name, self.metric, self.unit):
            _name(value)
        if self.minimum is None and self.maximum is None:
            raise ValueError("Limit must have a lower or upper bound")
        for bound in (self.minimum, self.maximum):
            if bound is not None:
                _number(bound, "limit bound")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Minimum exceeds maximum")


def evaluate_limits(limits, measurements):
    """Report every limit; missing/nonfinite measurements are errors, not passes."""
    limits = tuple(limits)
    if len({limit.name for limit in limits}) != len(limits):
        raise ValueError("Duplicate facility limit names")
    report = []
    for limit in limits:
        if limit.metric not in measurements:
            raise ValueError(f"Unknown facility metric: {limit.metric}")
        measurement = measurements[limit.metric]
        if measurement["unit"] != limit.unit:
            raise ValueError(f"Unit mismatch for {limit.metric}")
        value = _number(measurement["value"], limit.metric)
        below = 0.0 if limit.minimum is None else max(0.0, limit.minimum - value)
        above = 0.0 if limit.maximum is None else max(0.0, value - limit.maximum)
        report.append(
            dict(asdict(limit), value=value, violation=below + above, passed=(below + above == 0))
        )
    return {
        "feasible": all(r["passed"] for r in report),
        "limits": report,
        "violations": [r for r in report if not r["passed"]],
    }


class SurfaceFacility:
    def __init__(
        self,
        densities,
        feeds,
        stages,
        water_network,
        water_feed_node,
        limits=(),
        gas_export=None,
        station_control=None,
    ):
        self.densities, self.feeds, self.stages = densities, tuple(feeds), tuple(stages)
        self.water_network, self.water_feed_node = water_network, water_feed_node
        self.limits = tuple(limits)
        if gas_export is not None and not isinstance(gas_export, GasExportSystem):
            raise TypeError("gas_export must be GasExportSystem")
        self.gas_export = gas_export
        if station_control is not None and not isinstance(station_control, PumpStationControl):
            raise TypeError("station_control must be PumpStationControl")
        self.station_control = station_control
        if not self.stages or any(s.densities != densities for s in self.stages):
            raise ValueError("Facility requires separator stages with matching stock densities")
        if water_feed_node not in {n.name for n in water_network.nodes}:
            raise ValueError("Unknown water feed node")
        if sum(n.pressure_pa is not None for n in water_network.nodes) != 1:
            raise ValueError("Water handoff currently requires exactly one pressure boundary")
        for n in water_network.nodes:
            if n.source_mass_rate_kg_s != 0:
                raise ValueError("Facility water network must not contain additional sources")
            if n.name == water_feed_node and n.pressure_pa is not None:
                raise ValueError("Water feed node must not have a fixed pressure")

    @classmethod
    def from_dict(cls, data):
        from .gas import GasExportSystem
        from .process import BlackOilSeparator, StockDensities

        allowed = {
            "stock_densities",
            "feeds",
            "pvt",
            "separator_stages",
            "water_network",
            "water_feed_node",
            "limits",
            "gas_export",
            "pump_station_control",
        }
        if set(data) - allowed:
            raise ValueError(f"Unknown facility fields: {sorted(set(data) - allowed)}")
        densities = StockDensities(**data["stock_densities"])
        return cls(
            densities,
            [ComponentStream(**s) for s in data["feeds"]],
            [
                BlackOilSeparator(pvt=data["pvt"], densities=densities, **stage)
                for stage in data["separator_stages"]
            ],
            LiquidNetwork.from_dict(data["water_network"]),
            data["water_feed_node"],
            [FacilityLimit(**limit) for limit in data.get("limits", [])],
            gas_export=GasExportSystem(**data["gas_export"])
            if data.get("gas_export") is not None
            else None,
            station_control=PumpStationControl(**data["pump_station_control"])
            if data.get("pump_station_control") is not None
            else None,
        )

    def run(
        self,
        initial=None,
        feed_scales=None,
        pump_speed_ratios=None,
        valve_openings=None,
        gas_header_pressure_pa=None,
        gas_cooling_temperature_k=None,
        gas_valve_max_opening=None,
        gas_valve_outlet_pressure_pa=None,
        unavailable_pumps=(),
        previous_running=(),
        station_target_pressure_pa=None,
    ):
        scales = tuple(1.0 for _ in self.feeds) if feed_scales is None else tuple(feed_scales)
        if len(scales) != len(self.feeds):
            raise ValueError("feed_scales must match the number of feeds")
        for value in scales:
            _number(value, "feed scale", nonnegative=True)
            if value > 1:
                raise ValueError("Feed scales must be <= 1; feed values are available production")
        feed = ComponentStream.mix(s.split(scale)[0] for s, scale in zip(self.feeds, scales))
        stages, products, residual = run_separator_train(feed, self.stages)
        water_mass = products["water"].total_mass_rate(self.densities)
        if water_mass <= 0:
            raise ValueError("Water-network handoff requires nonzero separated water")
        rho = stages[0].phase_density_kg_m3["water"]
        if not math.isclose(rho, self.water_network.fluid.density_kg_m3, rel_tol=1e-10):
            raise ValueError("Network water density must match separator water density")
        station = None
        if self.station_control is None:
            if unavailable_pumps or previous_running or station_target_pressure_pa is not None:
                raise ValueError("Station controls require a configured pump station")
            result = self.water_network.solve(
                source_mass_rates_kg_s={self.water_feed_node: water_mass},
                initial=initial,
                pump_speed_ratios=pump_speed_ratios,
                valve_openings=valve_openings,
            )
        else:
            if pump_speed_ratios:
                raise ValueError("Pump station owns pump speed controls")
            control = self.station_control
            if station_target_pressure_pa is not None:
                control = replace(control, target_pressure_pa=station_target_pressure_pa)
            station = control.solve(
                self.water_network,
                initial=initial,
                unavailable=unavailable_pumps,
                previous_running=previous_running,
                source_mass_rates_kg_s={self.water_feed_node: water_mass},
                valve_openings=valve_openings,
            )
            result = station.network
        pressure = result.pressures_pa[self.water_feed_node]
        if pressure > stages[0].pressure_pa:
            raise EquipmentOperatingError("Water handoff requires pressure boost upstream of inlet")
        letdown = water_mass / rho * (stages[0].pressure_pa - pressure)
        measurements = {}

        def record(name, value, unit):
            measurements[name] = {"value": float(value), "unit": unit}

        record("feed_mass_kg_s", feed.total_mass_rate(self.densities), "kg/s")
        record("shaft_power_total_w", math.fsum(result.pump_shaft_power_w.values()), "W")
        record("water_letdown_power_w", letdown, "W")
        record("water_handoff_mass_kg_s", water_mass, "kg/s")
        for stage in stages:
            for phase, value in stage.phase_volume_rates_m3_s.items():
                record(f"separator:{stage.name}:{phase}_m3_s", value, "m3/s")
        for name, value in result.pressures_pa.items():
            record(f"pressure:{name}", value, "Pa")
        for name, value in result.pump_shaft_power_w.items():
            record(f"shaft_power:{name}", value, "W")
        for name, value in result.pump_npsh_margin_m.items():
            record(f"npsh_margin:{name}", value, "m")
        for name, value in result.pump_efficiencies.items():
            record(f"efficiency:{name}", value, "1")
        for name, value in result.pump_head_gains_m.items():
            record(f"pump_head:{name}", value, "m")
        for phase, stream in products.items():
            record(f"product:{phase}:mass_kg_s", stream.total_mass_rate(self.densities), "kg/s")
        balance_scale = max(1.0, feed.total_mass_rate(self.densities))
        if any(abs(v) > 1e-12 * balance_scale for v in residual.values()):
            raise ConservationError("Facility component mass balance failed")
        gas_system = self.gas_export
        if gas_header_pressure_pa is not None or gas_cooling_temperature_k is not None:
            if gas_system is None:
                raise ValueError("Gas controls require a gas export system")
            gas_system = replace(
                gas_system,
                header_pressure_pa=gas_system.header_pressure_pa
                if gas_header_pressure_pa is None
                else gas_header_pressure_pa,
                cooling_temperature_k=gas_system.cooling_temperature_k
                if gas_cooling_temperature_k is None
                else gas_cooling_temperature_k,
            )
        if gas_valve_max_opening is not None or gas_valve_outlet_pressure_pa is not None:
            if gas_system is None or gas_system.export_valve is None:
                raise ValueError("Gas valve controls require an export valve")
            valve = gas_system.export_valve
            gas_system = replace(
                gas_system,
                export_valve=replace(
                    valve,
                    max_opening=valve.max_opening
                    if gas_valve_max_opening is None
                    else gas_valve_max_opening,
                    outlet_pressure_pa=valve.outlet_pressure_pa
                    if gas_valve_outlet_pressure_pa is None
                    else gas_valve_outlet_pressure_pa,
                ),
            )
        gas_result = gas_system.run(stages, self.densities) if gas_system is not None else None
        if gas_result is not None:
            record("gas_drive_power_w", gas_result["drive_power_w"], "W")
            record("gas_cooling_duty_w", gas_result["rejected_heat_w"], "W")
            record("gas_export_mass_kg_s", gas_result["mass_rate_kg_s"], "kg/s")
            record(
                "facility_drive_power_total_w",
                measurements["shaft_power_total_w"]["value"] + gas_result["drive_power_w"],
                "W",
            )
            if gas_result["outlet"] is not None:
                record("gas_export_pressure_pa", gas_result["outlet"]["pressure_pa"], "Pa")
                record("gas_export_temperature_k", gas_result["outlet"]["temperature_k"], "K")
            if gas_result["valve"] is not None:
                trim = gas_result["valve"]
                record("gas_valve_opening", trim["opening"], "1")
                record("gas_valve_capacity_kg_s", trim["capacity_kg_s"], "kg/s")
                record("gas_valve_utilization", trim["utilization"], "1")
                for name, unit in (
                    ("throat_pressure_pa", "Pa"),
                    ("throat_temperature_k", "K"),
                    ("throat_mach", "1"),
                ):
                    record(f"gas_valve_{name}", trim["nozzle"][name], unit)
            expected_gas = products["gas"].total_mass_rate(self.densities)
            if not math.isclose(
                gas_result["mass_rate_kg_s"], expected_gas, rel_tol=1e-12, abs_tol=1e-14
            ):
                raise ConservationError("Gas export component mass balance failed")
            energy_scale = max(1.0, gas_result["shaft_power_w"], gas_result["rejected_heat_w"])
            if abs(gas_result["energy_residual_w"]) > 1e-10 * energy_scale:
                raise ConservationError("Gas export energy balance failed")
        audit = evaluate_limits(self.limits, measurements)
        return {
            "model": (
                "steady_blackoil_and_single_liquid_water_with_dry_ideal_gas_export"
                if gas_result is not None
                else "steady_blackoil_dissolved_gas_and_single_liquid_water"
            ),
            "operating_controls": {
                "feed_scales": list(scales),
                "pump_speed_ratios": dict(pump_speed_ratios or {}),
                "valve_openings": dict(valve_openings or {}),
                "gas_header_pressure_pa": gas_header_pressure_pa,
                "gas_cooling_temperature_k": gas_cooling_temperature_k,
                "gas_valve_max_opening": gas_valve_max_opening,
                "gas_valve_outlet_pressure_pa": gas_valve_outlet_pressure_pa,
                "unavailable_pumps": list(unavailable_pumps),
                "station_target_pressure_pa": station_target_pressure_pa,
            },
            "pump_station": None if station is None else station.to_dict(),
            "gas_export": gas_result,
            "feed": asdict(feed),
            "stages": [r.to_dict() for r in stages],
            "products": {name: asdict(stream) for name, stream in products.items()},
            "component_mass_residuals_kg_s": residual,
            "network": result.to_dict(),
            "measurements": measurements,
            "water_handoff": {
                "model": "prescribed_rate_ideal_letdown",
                "mass_rate_kg_s": water_mass,
                "pressure_drop_pa": stages[0].pressure_pa - pressure,
                "dissipation_w": letdown,
            },
            **audit,
        }

    @staticmethod
    def network_initial(saved):
        if "facility" in saved:
            saved = saved["facility"]
            if saved is None:
                raise ValueError("No feasible facility state in dispatch result")
        return NetworkResult.from_dict(saved["network"])
