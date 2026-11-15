"""Independent conservative ODE for the two-pump caloric benchmark.

Uses explicit laminar/pump/valve formulas and donor-cell energy balances,
not the production heat stencil or collocation equations.
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from check_liquid_transient import coupled_model, coupled_rhs

from surface import LiquidHeatNetwork, LiquidTransientNetwork, VesselWall


def case(cells=4, finite=True):
    data = coupled_model(0.02).to_dict()
    data.update(
        temporal_error_control=True,
        relative_tolerance=1e-8,
        pressure_tolerance_pa=0.1,
        flow_tolerance_m3_s=1e-11,
        speed_tolerance_rad_s=1e-6,
    )
    physical = LiquidTransientNetwork.from_dict(data)
    state = physical.initialize(
        {"h": 550000.0, "t": 500000.0}, {"line": 0.002}, {"a": 100.0, "b": 110.0}
    )
    wall = f"line::{cells // 2}"
    heat = LiquidHeatNetwork(
        physical,
        2000.0,
        {"out": 290.0} if finite else {"in": 330.0, "out": 290.0},
        boundary_reservoirs_kg={"in": 1000.0} if finite else {},
        pipe_cells={"line": cells},
        reference_temperature_k=300.0,
        walls={wall: VesselWall(1000.0, 400.0, 50.0, 295.0)},
        pump_heat_to_fluid_fraction=1.0,
        shaft_heat_to_fluid_fraction=0.25,
        motor_heat_to_fluid_fraction=0.5,
        temperature_tolerance_k=1e-6,
        relative_tolerance=1e-8,
    )
    thermal = heat.initialize(state, {"in": 330.0} if finite else {}, {wall: 305.0})
    return physical, heat, state, thermal


class ConservativeReference:
    def __init__(self, heat, motor_torques_nm=(5.0, 7.0), heat_duties_w=None):
        self.heat = heat
        self.motor_torques = np.asarray(motor_torques_nm, dtype=float)
        if self.motor_torques.shape != (2,) or not np.isfinite(self.motor_torques).all():
            raise ValueError("Two finite motor commands required")
        self.duties = dict(heat_duties_w or {})
        if set(self.duties) - set(heat.names) or any(
            not math.isfinite(v) for v in self.duties.values()
        ):
            raise ValueError("Invalid reference heat duty")
        self.names, self.walls = heat.names, heat.wall_names
        self.reservoirs = tuple(sorted(heat.boundary_reservoirs_kg))
        self.index = {n: i for i, n in enumerate(self.names)}

    def encode(self, physical, thermal):
        t, m, tw = (
            dict(thermal.temperatures_k),
            dict(thermal.masses_kg),
            dict(thermal.wall_temperatures_k),
        )
        p, q, w = (
            dict(physical.pressures_pa),
            dict(physical.pipe_volume_rates_m3_s),
            dict(physical.speeds_rad_s),
        )
        return np.r_[
            [p["h"], p["t"], q["line"], w["a"], w["b"]],
            [
                m[n] * self.heat.specific_heat_j_kg_k * (t[n] - self.heat.reference_temperature_k)
                for n in self.names
            ],
            [
                self.heat.walls[n].heat_capacity_j_k * (tw[n] - self.heat.reference_temperature_k)
                for n in self.walls
            ],
            [m[n] for n in self.reservoirs],
        ]

    def decode(self, y):
        h = self.heat
        nr, nf = len(self.reservoirs), len(self.names)
        masses = {
            "h": 1000 * 0.1 * (1 + (y[0] - 500000.0) / 1e7),
            "t": 1000 * 0.1 * (1 + (y[1] - 500000.0) / 1e7),
            **{
                n: 1000 * math.pi * 0.05**2 / 4 * 10 / len(h._paths["line"])
                for n in h._paths["line"]
            },
            **dict(zip(self.reservoirs, y[-nr:] if nr else [])),
        }
        temperatures = {
            **h.boundary_temperatures_k,
            **{
                n: h.reference_temperature_k + y[5 + i] / (masses[n] * h.specific_heat_j_kg_k)
                for i, n in enumerate(self.names)
            },
        }
        walls = {
            n: h.reference_temperature_k + y[5 + nf + i] / h.walls[n].heat_capacity_j_k
            for i, n in enumerate(self.walls)
        }
        return masses, temperatures, walls

    def __call__(self, time, y):
        h = self.heat
        ph, pt, q, wa, wb = y[:5]
        _, temp, wall = self.decode(y)
        speeds = np.array([wa, wb])
        qp = np.sqrt(
            np.maximum(
                0.0,
                1000 * 9.80665 * np.array([20.0, 22.0]) * (speeds / [100.0, 110.0]) ** 2
                - (ph - 400000.0),
            )
            / (1000 * 9.80665 * np.array([1e6, 1.2e6]))
        )
        qv = (
            0.7
            * math.pi
            * 0.02**2
            / 4
            * math.copysign(math.sqrt(2 * abs(pt - 480000.0) / 1000), pt - 480000.0)
        )
        energy = {n: 0.0 for n in self.names}
        dm = {n: 0.0 for n in self.reservoirs}

        def transfer(a, b, volume_rate):
            if volume_rate < 0:
                a, b, volume_rate = b, a, -volume_rate
            power = (
                1000 * volume_rate * h.specific_heat_j_kg_k * (temp[a] - h.reference_temperature_k)
            )
            if a in energy:
                energy[a] -= power
            if b in energy:
                energy[b] += power
            if a in dm:
                dm[a] -= 1000 * volume_rate
            if b in dm:
                dm[b] += 1000 * volume_rate

        transfer("in", "h", qp[0])
        transfer("in", "h", qp[1])
        path = ["h", *h._paths["line"], "t"]
        for a, b in zip(path[:-1], path[1:]):
            transfer(a, b, q)
        transfer("t", "out", qv)
        friction = 128 * 0.1 * 10 / (math.pi * 0.05**4) * q * q
        for n in h._paths["line"]:
            energy[n] += friction / len(h._paths["line"])
        recipient = "out" if qv >= 0 else "t"
        if recipient in energy:
            energy[recipient] += (pt - 480000.0) * qv
        hydraulic_power = (ph - 400000.0) * qp
        shaft_power = (
            hydraulic_power / [0.8, 0.75] + np.array([5.0, 8.0]) * (speeds / [100.0, 110.0]) ** 3
        )
        motor_work = self.motor_torques * speeds
        electric = np.where(motor_work >= 0, motor_work / 0.9, motor_work * 0.7)
        heat = (
            h.pump_heat_to_fluid_fraction * (shaft_power - hydraulic_power)
            + h.shaft_heat_to_fluid_fraction * np.array([0.01, 0.015]) * speeds**2
            + h.motor_heat_to_fluid_fraction * (electric - motor_work)
        )
        energy["h"] += sum(heat)
        for name, power in self.duties.items():
            energy[name] += power
        wall_rates = []
        for n in self.walls:
            w = h.walls[n]
            exchange = w.fluid_ua_w_k * (temp[n] - wall[n])
            energy[n] -= exchange
            wall_rates.append(exchange - w.ambient_ua_w_k * (wall[n] - w.ambient_temperature_k))
        dynamics = coupled_rhs(time, y[:5])
        dynamics[3:] += (self.motor_torques - [5.0, 7.0]) / [0.5, 0.7]
        return np.r_[
            dynamics,
            [energy[n] for n in self.names],
            wall_rates,
            [dm[n] for n in self.reservoirs],
        ]

    def observe(self, y):
        mass, temp, wall = self.decode(y)
        return np.r_[
            y[:5],
            [temp[n] for n in self.names],
            [wall[n] for n in self.walls],
            [mass[n] for n in self.reservoirs],
        ]
