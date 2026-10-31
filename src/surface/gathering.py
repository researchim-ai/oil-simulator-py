"""Steady gathering tree → separation → water treatment and dry-gas export.

Prescribed component feeds, NOT reservoir nodal feedback. Network mixing uses
component masses before equilibrium flashing at each local pressure.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .blackoil_control import BlackOilPressureControl
from .blackoil_network import BlackOilProductionNetwork
from .facility import SurfaceFacility, evaluate_limits
from .liquid import ConservationError, PropertyModelError, _number
from .process import ComponentStream


class GatheringFacility:
    def __init__(self, network, facility, feed_nodes, limits=(), pressure_control=None):
        if not isinstance(network, BlackOilProductionNetwork) or not isinstance(
            facility, SurfaceFacility
        ):
            raise TypeError("Expected BlackOilProductionNetwork and SurfaceFacility")
        if network.model.densities != facility.densities:
            raise PropertyModelError("Gathering and separation must share stock densities")
        if len(feed_nodes) != len(facility.feeds) or any(
            n not in network.by_name for n in feed_nodes
        ):
            raise ValueError("Every facility feed needs a known gathering node")
        receiver = network.by_name[network.root]
        if receiver.pressure_pa != facility.stages[0].pressure_pa:
            raise PropertyModelError(
                "Gathering receiving pressure must equal first separator pressure"
            )
        pvt = facility.stages[0].pvt
        for key in ("P", "Bo", "Bw", "Bg", "mu_o", "mu_w", "mu_g", "Rs", "Rv", "oil_rs"):
            if not np.array_equal(getattr(network.model.pvt, key), getattr(pvt, key)):
                raise PropertyModelError("Gathering and first separator must share identical PVT")
        if set(network.model.pvt.oil_tables) != set(pvt.oil_tables):
            raise PropertyModelError("Gathering and separation PVTO tables differ")
        for rs, table in pvt.oil_tables.items():
            if any(
                not np.array_equal(value, network.model.pvt.oil_tables[rs].get(key))
                for key, value in table.items()
            ):
                raise PropertyModelError("Gathering and separation PVTO tables differ")
        if isinstance(pressure_control, dict):
            pressure_control = BlackOilPressureControl(**pressure_control)
        if pressure_control is not None:
            if not isinstance(pressure_control, BlackOilPressureControl):
                raise TypeError("Expected BlackOilPressureControl")
            pressure_control.validate_network(network)
        self.gathering_pressure_control = pressure_control
        self.gathering_network = network
        self.facility = facility
        self.feeds = facility.feeds
        self.densities = facility.densities
        self.feed_nodes = tuple(feed_nodes)
        self.limits = (*facility.limits, *limits)
        if len({l.name for l in self.limits}) != len(self.limits):
            raise ValueError("Duplicate gathering/facility limit names")

    @property
    def water_network(self):
        return self.facility.water_network

    @property
    def gas_export(self):
        return self.facility.gas_export

    @property
    def station_control(self):
        return self.facility.station_control

    def validate_gathering_controls(
        self, gathering_valve_openings=None, gathering_target_pressure_pa=None
    ):
        openings = {} if gathering_valve_openings is None else gathering_valve_openings
        control = self.gathering_pressure_control
        if control is not None and control.valve in openings:
            raise ValueError("Manual gathering opening conflicts with pressure regulator")
        effective = self.gathering_network.with_controls(openings)
        if gathering_target_pressure_pa is not None:
            if control is None:
                raise ValueError("Gathering target requires a pressure regulator")
            control = replace(control, target_pressure_pa=gathering_target_pressure_pa)
        return effective, control

    def run(
        self,
        initial=None,
        feed_scales=None,
        gathering_valve_openings=None,
        gathering_target_pressure_pa=None,
        **controls,
    ):
        effective, regulator = self.validate_gathering_controls(
            gathering_valve_openings, gathering_target_pressure_pa
        )
        scales = tuple(1.0 for _ in self.feeds) if feed_scales is None else tuple(feed_scales)
        if len(scales) != len(self.feeds):
            raise ValueError("Feed scales must match gathering feeds")
        for value in scales:
            if _number(value, "feed scale", nonnegative=True) > 1:
                raise ValueError("Feed scale must be <=1")
        grouped = {}
        for node, feed, scale in zip(self.feed_nodes, self.feeds, scales):
            grouped.setdefault(node, []).append(feed.split(scale)[0])
        sources = {n: ComponentStream.mix(streams) for n, streams in grouped.items()}
        gathering = (
            effective.solve(sources) if regulator is None else regulator.solve(effective, sources)
        )
        result = self.facility.run(initial=initial, feed_scales=scales, **controls)
        total = ComponentStream(**gathering["receiving_stream"])
        actual = ComponentStream(**result["feed"])
        for key, value in total.mass_rates(self.densities).items():
            if not math.isclose(
                value, actual.mass_rates(self.densities)[key], rel_tol=1e-12, abs_tol=1e-14
            ):
                raise ConservationError("Gathering → separator component handoff imbalance")
        # Check the phase split as well as stock component accounting.
        flash = self.gathering_network.model.state(
            self.facility.stages[0].pressure_pa, list(total.mass_rates(self.densities).values())
        )
        for phase, volume in zip(("oil", "water", "gas"), flash["phase_volume_rates_m3_s"]):
            if not math.isclose(
                volume,
                result["stages"][0]["phase_volume_rates_m3_s"][phase],
                rel_tol=1e-10,
                abs_tol=1e-14,
            ):
                raise ConservationError("Gathering → separator phase-volume mismatch")
        measurements = result["measurements"]
        for node, pressure in gathering["node_pressures_pa"].items():
            measurements[f"gathering_pressure:{node}"] = {"value": pressure, "unit": "Pa"}
        measurements["gathering_friction_dissipation_w"] = {
            "value": gathering["friction_dissipation_w"],
            "unit": "W",
        }
        result.update(evaluate_limits(self.limits, measurements))
        result["gathering_network"] = gathering
        result["model"] = "steady_homogeneous_blackoil_gathering_and_facility"
        result["scope"] = "prescribed feeds; no reservoir feedback, no slip or thermal prediction"
        return result

    @staticmethod
    def network_initial(saved):
        return SurfaceFacility.network_initial(saved)
