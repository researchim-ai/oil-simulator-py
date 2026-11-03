"""Weakly compressible liquid network with monolithic pump-shaft dynamics.

Linear acoustic storage, pipe inertance, signed steady friction, algebraic
valves/check valves and forward pump maps. Mechanical/acoustic energy is
audited; this is not a thermal EOS, cavitation or a multiphase model.
"""

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.sparse import coo_matrix, lil_matrix

from .actuator import ConnectionActuator
from .dynamic_station import DynamicPumpStation
from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    LiquidNetwork,
    _number,
)
from .liquid_relief import LiquidRelief
from .operation import _digest
from .rotating_drive import RotatingDrive


@dataclass(frozen=True)
class _MovingValve:
    valve: object
    actuator: ConnectionActuator
    position: float
    target: float
    mode: str

    def __getattr__(self, name):
        return getattr(self.valve, name)

    def at(self, dt):
        position = self.actuator.advance(self.position, self.target, dt, self.mode)[
            "final_position"
        ]
        return replace(self.valve, opening=position)

    def after(self, dt):
        return replace(self, position=self.at(dt).opening)


@dataclass(frozen=True)
class LiquidTransientState:
    pressures_pa: tuple
    pipe_volume_rates_m3_s: tuple
    speeds_rad_s: tuple
    unavailable: tuple
    time_s: float
    model_id: str
    tripped: bool = False
    actuator_positions: tuple = ()

    def __post_init__(self):
        for key in ("pressures_pa", "pipe_volume_rates_m3_s", "speeds_rad_s"):
            values = tuple(sorted(dict(getattr(self, key)).items()))
            for _, value in values:
                _number(
                    value, key, positive=key == "pressures_pa", nonnegative=key == "speeds_rad_s"
                )
            object.__setattr__(self, key, values)
        positions = tuple(sorted(dict(self.actuator_positions).items()))
        for _, value in positions:
            ConnectionActuator.position(value)
        object.__setattr__(self, "actuator_positions", positions)
        unavailable = tuple(self.unavailable)
        if len(unavailable) != len(set(unavailable)):
            raise ValueError("Duplicate unavailable pump")
        object.__setattr__(self, "unavailable", unavailable)
        _number(self.time_s, "time_s", nonnegative=True)
        if not isinstance(self.tripped, bool):
            raise TypeError("Trip latch must be boolean")

    def to_dict(self):
        return {
            "pressures_pa": dict(self.pressures_pa),
            "pipe_volume_rates_m3_s": dict(self.pipe_volume_rates_m3_s),
            "speeds_rad_s": dict(self.speeds_rad_s),
            "unavailable": list(self.unavailable),
            "time_s": self.time_s,
            "model_id": self.model_id,
            "tripped": self.tripped,
            "actuator_positions": dict(self.actuator_positions),
        }


class LiquidTransientNetwork:
    def __init__(
        self,
        network,
        storage_volumes_m3,
        bulk_modulus_pa,
        reference_pressure_pa=1e6,
        drives=None,
        nominal_speeds_rad_s=None,
        no_flow_power_w_at_nominal=None,
        max_station_electrical_draw_w=None,
        max_station_electrical_export_w=None,
        maximum_dt_s=0.005,
        minimum_dt_s=1e-8,
        acoustic_courant=0.5,
        max_pressure_pa=1e8,
        max_compression_fraction=0.01,
        max_speed_change_rad_s=2.0,
        max_pressure_change_pa=2e5,
        temporal_error_control=True,
        relative_tolerance=1e-5,
        pressure_tolerance_pa=1.0,
        flow_tolerance_m3_s=1e-9,
        speed_tolerance_rad_s=1e-5,
        relief_valves=None,
        actuated_valves=None,
        backend="torch",
        device="cpu",
    ):
        if not isinstance(network, LiquidNetwork):
            raise TypeError("Expected LiquidNetwork")
        if any(n.elevation_m != 0 or n.source_mass_rate_kg_s != 0 for n in network.nodes):
            raise ValueError("Acoustic reference requires horizontal pressure-driven network")
        if any(not p.open or p.check_valve for p in network.pipes):
            raise ValueError("Inertial pipes must stay open; use a separate algebraic valve")
        if backend not in ("torch", "scipy_reference"):
            raise ValueError("Unknown liquid backend")
        if backend == "scipy_reference" and device != "cpu":
            raise ValueError("SciPy reference backend is CPU-only")
        if backend == "torch":
            from .torch_linear import checked_device

            device = checked_device(device)
        self.backend, self.device = backend, str(device)
        self._torch_bridge = None
        self.network = network
        self.relief_valves = {
            n: LiquidRelief(**v) if isinstance(v, dict) else v
            for n, v in dict(relief_valves or {}).items()
        }
        valves = {v.name: v for v in network.valves}
        if set(self.relief_valves) - set(valves) or any(
            not isinstance(v, LiquidRelief) for v in self.relief_valves.values()
        ):
            raise ValueError("Relief laws must reference existing liquid valves")
        if any(
            not valves[n].open or not valves[n].check_valve or valves[n].opening != 0
            for n in self.relief_valves
        ):
            raise ValueError(
                "Relief branches require open=True, check_valve=True and nominal opening=0"
            )
        self.actuated_valves = {
            n: ConnectionActuator(**v) if isinstance(v, dict) else v
            for n, v in dict(actuated_valves or {}).items()
        }
        if (
            set(self.actuated_valves) - set(valves)
            or set(self.actuated_valves) & set(self.relief_valves)
            or any(not isinstance(a, ConnectionActuator) for a in self.actuated_valves.values())
            or any(not valves[n].open for n in self.actuated_valves)
        ):
            raise ValueError("Actuators require open carrier valves, disjoint from relief laws")
        if not isinstance(temporal_error_control, bool):
            raise TypeError("temporal_error_control must be boolean")
        self.temporal_error_control = temporal_error_control
        self.free = tuple(n.name for n in network.nodes if n.pressure_pa is None)
        if not self.free or set(storage_volumes_m3) != set(self.free):
            raise ValueError("Every free node needs an explicit storage volume")
        self.storage_volumes_m3 = {
            n: _number(storage_volumes_m3[n], "volume", positive=True) for n in self.free
        }
        for key, value in (
            ("bulk_modulus_pa", bulk_modulus_pa),
            ("reference_pressure_pa", reference_pressure_pa),
            ("maximum_dt_s", maximum_dt_s),
            ("minimum_dt_s", minimum_dt_s),
            ("acoustic_courant", acoustic_courant),
            ("max_pressure_pa", max_pressure_pa),
            ("max_compression_fraction", max_compression_fraction),
            ("max_speed_change_rad_s", max_speed_change_rad_s),
            ("max_pressure_change_pa", max_pressure_change_pa),
            ("relative_tolerance", relative_tolerance),
            ("pressure_tolerance_pa", pressure_tolerance_pa),
            ("flow_tolerance_m3_s", flow_tolerance_m3_s),
            ("speed_tolerance_rad_s", speed_tolerance_rad_s),
        ):
            setattr(self, key, _number(value, key, positive=True))
        if minimum_dt_s > maximum_dt_s or acoustic_courant > 1 or max_compression_fraction > 0.05:
            raise ValueError("Invalid acoustic accuracy/domain settings")
        if not network.fluid.vapor_pressure_pa < reference_pressure_pa < max_pressure_pa:
            raise ValueError("Reference pressure outside liquid domain")
        self.names = tuple(p.name for p in network.pumps)
        self.drives = dict(drives or {})
        self.nominal_speeds_rad_s = dict(nominal_speeds_rad_s or {})
        if set(self.drives) != set(self.names) or set(self.nominal_speeds_rad_s) != set(self.names):
            raise ValueError("All pumps require explicit drives and nominal speeds")
        self.no_flow_power_w_at_nominal = dict(no_flow_power_w_at_nominal or {})
        self.max_station_electrical_draw_w = max_station_electrical_draw_w
        self.max_station_electrical_export_w = max_station_electrical_export_w
        self._station = None
        if self.names:
            # Reuse the checked drive allocation, without running the old
            # quasi-steady station's pressure controller/dispatch in a new model.
            self._station = DynamicPumpStation(
                network,
                self.drives,
                self.nominal_speeds_rad_s,
                pressure_node=self.free[0],
                no_flow_power_w_at_nominal=self.no_flow_power_w_at_nominal,
                max_station_electrical_draw_w=max_station_electrical_draw_w,
                max_station_electrical_export_w=max_station_electrical_export_w,
            )
            self.no_flow_power_w_at_nominal = self._station.no_flow_power_w_at_nominal.copy()
        elif self.no_flow_power_w_at_nominal:
            raise ValueError("No-flow power without pumps")
        for value in (max_station_electrical_draw_w, max_station_electrical_export_w):
            if value is not None:
                _number(value, "electrical_capacity", nonnegative=True)
        self._nodes = tuple(n.name for n in network.nodes)
        self._branches = network.pipes + network.pumps + network.valves
        node_index = {n: i for i, n in enumerate(self._nodes)}
        count = len(self._branches)
        self._B = coo_matrix(
            (
                [1.0] * count + [-1.0] * count,
                (
                    [node_index[b.start] for b in self._branches]
                    + [node_index[b.end] for b in self._branches],
                    list(range(count)) * 2,
                ),
            ),
            shape=(len(self._nodes), count),
        ).tocsr()
        self._fi = [node_index[n] for n in self.free]
        self._C = np.asarray([self.storage_volumes_m3[n] / bulk_modulus_pa for n in self.free])
        self._L = np.asarray(
            [
                network.fluid.density_kg_m3 * p.length_m / (math.pi * p.diameter_m**2 / 4)
                for p in network.pipes
            ]
        )
        self._qscale = max(
            [0.001]
            + [p.max_flow_m3_s for p in network.pumps]
            + [math.pi * p.diameter_m**2 / 4 for p in network.pipes]
        )
        self._pscale = max(1e5, reference_pressure_pa)
        self._J = np.asarray([self.drives[n].inertia_kg_m2 for n in self.names])
        self._drag = np.asarray([self.drives[n].viscous_drag_nm_s for n in self.names])
        self._sparsity = self._jacobian_pattern()

    def to_dict(self):
        keys = (
            "backend",
            "device",
            "storage_volumes_m3",
            "bulk_modulus_pa",
            "reference_pressure_pa",
            "nominal_speeds_rad_s",
            "no_flow_power_w_at_nominal",
            "max_station_electrical_draw_w",
            "max_station_electrical_export_w",
            "maximum_dt_s",
            "minimum_dt_s",
            "acoustic_courant",
            "max_pressure_pa",
            "max_compression_fraction",
            "max_speed_change_rad_s",
            "max_pressure_change_pa",
            "temporal_error_control",
            "relative_tolerance",
            "pressure_tolerance_pa",
            "flow_tolerance_m3_s",
            "speed_tolerance_rad_s",
        )
        return {
            "network": self.network.to_dict(),
            "relief_valves": {n: asdict(v) for n, v in self.relief_valves.items()},
            "actuated_valves": {n: asdict(v) for n, v in self.actuated_valves.items()},
            "drives": {n: asdict(d) for n, d in self.drives.items()},
            **{k: getattr(self, k) for k in keys},
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["network"] = LiquidNetwork.from_dict(data["network"])
        data["drives"] = {n: RotatingDrive(**v) for n, v in data.get("drives", {}).items()}
        return cls(**data)

    @property
    def model_id(self):
        return _digest({"model": "weakly_compressible_liquid_v1", **self.to_dict()})

    def _jacobian_pattern(self):
        nf, nb, npump = len(self.free), len(self._branches), len(self.names)
        pattern = lil_matrix((nf + nb + npump, nf + nb + npump), dtype=int)
        for i in range(nf):
            pattern[i, i] = 1
            for e in self._B.getrow(self._fi[i]).indices:
                pattern[i, nf + e] = 1
        for e, branch in enumerate(self._branches):
            for n in (branch.start, branch.end):
                if n in self.free:
                    pattern[nf + e, self.free.index(n)] = 1
            pattern[nf + e, nf + e] = 1
        offset = len(self.network.pipes)
        for i in range(npump):
            pattern[nf + offset + i, nf + nb + i] = 1
            # Group power allocation couples all shaft/load coordinates.
            pattern[nf + nb + i, :nf] = 1
            pattern[nf + nb + i, nf + offset : nf + offset + npump] = 1
            pattern[nf + nb + i, nf + nb :] = 1
        return pattern.tocsr()

    def _controls(self, boundary_pressures_pa=None, valve_openings=None):
        boundary = {n.name: n.pressure_pa for n in self.network.nodes if n.pressure_pa is not None}
        overrides = dict(boundary_pressures_pa or {})
        openings = dict(valve_openings or {})
        if set(openings) & set(self.relief_valves):
            raise ValueError("Pressure determines relief opening; manual override is unsupported")
        if set(overrides) - set(boundary) or set(openings) - {v.name for v in self.network.valves}:
            raise ValueError("Unknown fixed boundary or valve control")
        boundary.update(overrides)
        for p in boundary.values():
            self._pressure(p)
        for value in openings.values():
            _number(value, "opening", nonnegative=True)
            if value > 1:
                raise ValueError("Valve opening exceeds one")
        valves = [replace(v, opening=openings.get(v.name, v.opening)) for v in self.network.valves]
        return boundary, valves

    def _pressure(self, p):
        _number(p, "pressure_pa", positive=True)
        if (
            p <= self.network.fluid.vapor_pressure_pa
            or p > self.max_pressure_pa
            or abs(p - self.reference_pressure_pa) / self.bulk_modulus_pa
            > self.max_compression_fraction
        ):
            raise EquipmentOperatingError(
                "Acoustic pressure/compression domain exceeded; no cavitation model"
            )

    def initialize(
        self,
        pressures_pa=None,
        pipe_volume_rates_m3_s=None,
        speeds_rad_s=None,
        time_s=0.0,
        unavailable=(),
        actuator_positions=None,
    ):
        p, q, w = (
            dict(pressures_pa or {}),
            dict(pipe_volume_rates_m3_s or {}),
            dict(speeds_rad_s or {}),
        )
        if (
            set(p) - set(self.free)
            or set(q) - {b.name for b in self.network.pipes}
            or set(w) - set(self.names)
        ):
            raise ValueError("Unknown initial transient coordinate")
        positions = {
            v.name: v.opening for v in self.network.valves if v.name in self.actuated_valves
        }
        if set(actuator_positions or {}) - set(positions):
            raise ValueError("Unknown initial actuator")
        positions.update(actuator_positions or {})
        state = LiquidTransientState(
            {n: p.get(n, self.reference_pressure_pa) for n in self.free},
            {b.name: q.get(b.name, 0.0) for b in self.network.pipes},
            {n: w.get(n, 0.0) for n in self.names},
            tuple(unavailable),
            time_s,
            self.model_id,
            actuator_positions=positions,
        )
        self._check(state)
        return state

    def _check(self, state):
        if not isinstance(state, LiquidTransientState) or state.model_id != self.model_id:
            raise ValueError("Transient liquid state/model mismatch")
        if (
            set(dict(state.pressures_pa)) != set(self.free)
            or set(dict(state.pipe_volume_rates_m3_s)) != {b.name for b in self.network.pipes}
            or set(dict(state.speeds_rad_s)) != set(self.names)
            or set(state.unavailable) - set(self.names)
            or set(dict(state.actuator_positions)) != set(self.actuated_valves)
        ):
            raise ValueError("Transient coordinate/availability mismatch")
        for _, p in state.pressures_pa:
            self._pressure(p)
        for n, speed in state.speeds_rad_s:
            self.drives[n].initialize(speed, state.time_s)

    def inspect(self, state):
        self._check(state)
        p = np.asarray([dict(state.pressures_pa)[n] for n in self.free])
        q = np.asarray([dict(state.pipe_volume_rates_m3_s)[b.name] for b in self.network.pipes])
        w = np.asarray([dict(state.speeds_rad_s)[n] for n in self.names])
        masses = {
            n: self.network.fluid.density_kg_m3
            * self.storage_volumes_m3[n]
            * (
                1
                + (dict(state.pressures_pa)[n] - self.reference_pressure_pa) / self.bulk_modulus_pa
            )
            for n in self.free
        }
        return {
            "mass_kg": masses,
            "acoustic_energy_j": float(
                0.5 * np.sum(self._C * (p - self.reference_pressure_pa) ** 2)
            ),
            "pipe_kinetic_energy_j": float(0.5 * np.sum(self._L * q * q)),
            "shaft_kinetic_energy_j": float(0.5 * np.sum(self._J * w * w)),
            "wave_speed_m_s": math.sqrt(self.bulk_modulus_pa / self.network.fluid.density_kg_m3),
        }

    def _solve(self, oldp, oldq, oldw, dt, commands, boundary, valves):
        if self.backend == "torch":
            if self._torch_bridge is None:
                from .torch_liquid_bridge import TorchLiquidBridge

                self._torch_bridge = TorchLiquidBridge(self, self.device)
            return self._torch_bridge.solve(oldp, oldq, oldw, dt, commands, boundary, valves)
        valves = [v.at(dt / 2) if isinstance(v, _MovingValve) else v for v in valves]
        nf, npipes, npump = len(self.free), len(self.network.pipes), len(self.names)
        nb = len(self._branches)
        ps, qs, ws = self._pscale, self._qscale, np.maximum(1.0, oldw)
        fixed = np.asarray([boundary.get(n, self.reference_pressure_pa) for n in self._nodes])
        maxw = np.asarray([self.drives[n].max_speed_rad_s for n in self.names])
        initial_torque = (
            self._station._allocate(oldw, oldw, commands, np.zeros(npump), dt)[0]
            if npump
            else np.asarray([])
        )
        held = (oldw == 0) & (initial_torque <= 0)

        def evaluate(x, audit=False):
            endp = x[:nf] * ps
            flows = x[nf : nf + nb] * qs
            endw = x[nf + nb :] * ws
            p = fixed.copy()
            p[self._fi] = (oldp + endp) / 2
            qmid = (oldq + flows[:npipes]) / 2
            allq = np.r_[qmid, flows[npipes:]]
            wm = np.maximum(0.0, (oldw + endw) / 2)
            driving = self._B.T @ p
            equations, losses, loads, pump_power = [], [], [], []
            for i, pipe in enumerate(self.network.pipes):
                loss = pipe.loss_and_gradient(qmid[i], self.network.fluid)[0]
                losses.append(loss * qmid[i])
                equations.append(
                    (self._L[i] * (flows[i] - oldq[i]) - dt * (driving[i] - loss))
                    / (self._L[i] * qs)
                )
            for i, pump in enumerate(self.network.pumps):
                q = flows[npipes + i]
                if wm[i] <= 0:
                    equations.append(q / qs)
                    loads.append(0.0)
                    pump_power.append(0.0)
                    continue
                effective = replace(
                    pump, speed_ratio=float(wm[i] / self.nominal_speeds_rad_s[pump.name])
                )
                loss = effective.loss_and_gradient(q, self.network.fluid)[0]
                blocked = (
                    driving[npipes + i] <= effective.loss_and_gradient(0.0, self.network.fluid)[0]
                )
                equations.append(
                    q / qs if blocked else max((driving[npipes + i] - loss) / ps, -q / qs)
                )
                # Bounded efficiency query is trial-only. Final flow/map and
                # NPSH are checked without extrapolation before acceptance.
                mapped_q = min(effective.max_flow_m3_s * effective.speed_ratio, max(0.0, q))
                eta = effective.efficiency_at(mapped_q)
                # On a blocked branch, a finite-difference trial q must not
                # create P/omega singularities during a near-zero-speed start.
                # Its independent algebraic equation enforces q=0.
                hydraulic = 0.0 if blocked else max(0.0, -driving[npipes + i] * q)
                power = (
                    hydraulic / eta
                    + self.no_flow_power_w_at_nominal[pump.name] * effective.speed_ratio**3
                )
                loads.append(power / wm[i])
                pump_power.append(power)
                if audit:
                    if q < -1e-10 or -driving[npipes + i] * q < -1e-7:
                        raise EquipmentOperatingError(
                            f"{pump.name}: pump reverse/turbine flow unsupported "
                            f"(Q={q:.12g} m3/s, hydraulic_power={-driving[npipes + i] * q:.12g} W, "
                            f"omega={wm[i]:.12g} rad/s, dp={driving[npipes + i]:.12g} Pa)"
                        )
                    effective.operating_state(
                        max(0.0, q), p[self._nodes.index(pump.start)], self.network.fluid
                    )
                    if pump.max_shaft_power_w is not None and power > pump.max_shaft_power_w:
                        raise EquipmentOperatingError("Pump total shaft power exceeded")
            for i, valve in enumerate(valves):
                e = npipes + npump + i
                q = flows[e]
                if valve.name in self.relief_valves:
                    expected = self.relief_valves[valve.name].volume_rate(
                        valve, driving[e], self.network.fluid
                    )
                    equations.append((q - expected) / qs)
                    losses.append(driving[e] * q)
                elif valve.name in self.actuated_valves:
                    # Invert the SAME orifice law. Pressure-loss form divides
                    # by opening squared and becomes singular as an exact
                    # first-order actuator approaches its closed seat.
                    dp = max(0.0, driving[e]) if valve.check_valve else driving[e]
                    expected = (
                        valve.discharge_coefficient
                        * math.pi
                        * valve.diameter_m**2
                        / 4
                        * valve.opening
                        * math.copysign(
                            math.sqrt(2 * abs(dp) / self.network.fluid.density_kg_m3), dp
                        )
                    )
                    equations.append((q - expected) / qs)
                    losses.append(driving[e] * q)
                elif not valve.open or valve.opening == 0:
                    equations.append(q / qs)
                    losses.append(0.0)
                else:
                    loss = valve.loss_and_gradient(q, self.network.fluid)[0]
                    force = (driving[e] - loss) / ps
                    equations.append(max(force, -q / qs) if valve.check_valve else force)
                    losses.append(loss * q)
            torque = np.asarray([])
            fd = fe = 1.0
            if npump:
                torque, fd, fe = self._station._allocate(oldw, endw, commands, loads, dt)
            shaft = (self._J * (endw - oldw) - dt * (torque - loads - self._drag * wm)) / (
                self._J * ws
            )
            # Unilateral hold at rest is a reaction with zero work.
            shaft = np.where(held, endw / ws, shaft)
            continuity = (self._C * (endp - oldp) + dt * (self._B[self._fi] @ allq)) / (
                self._C * ps
            )
            residual = np.r_[continuity, equations, shaft]
            return residual, (
                endp,
                flows[:npipes],
                endw,
                p,
                allq,
                wm,
                torque,
                np.asarray(pump_power),
                losses,
                fd,
                fe,
            )

        def algebraic_guess(pressures, speeds):
            drive = self._B.T @ pressures
            guessed = []
            for i, pump in enumerate(self.network.pumps):
                ratio = max(0.0, float(speeds[i])) / self.nominal_speeds_rad_s[pump.name]
                available = (
                    drive[npipes + i]
                    + self.network.fluid.density_kg_m3 * 9.80665 * pump.shutoff_head_m * ratio**2
                )
                if ratio == 0 or available <= 0:
                    flow = 0.0
                elif pump.head_curve is None:
                    flow = math.sqrt(
                        available
                        / (
                            self.network.fluid.density_kg_m3
                            * 9.80665
                            * pump.curve_coefficient_s2_m5
                        )
                    )
                else:
                    target = -drive[npipes + i] / (
                        self.network.fluid.density_kg_m3 * 9.80665 * ratio**2
                    )
                    flow = ratio * float(
                        np.interp(
                            target, pump.head_curve.values[::-1], pump.head_curve.flow_m3_s[::-1]
                        )
                    )
                guessed.append(flow)
            for i, valve in enumerate(valves):
                dp = float(drive[npipes + npump + i])
                if valve.name in self.relief_valves:
                    flow = self.relief_valves[valve.name].volume_rate(valve, dp, self.network.fluid)
                elif not valve.open or valve.opening == 0 or valve.check_valve and dp <= 0:
                    flow = 0.0
                else:
                    area = math.pi * valve.diameter_m**2 / 4 * valve.opening
                    flow = math.copysign(
                        valve.discharge_coefficient
                        * area
                        * math.sqrt(2 * abs(dp) / self.network.fluid.density_kg_m3),
                        dp,
                    )
                guessed.append(flow)
            return np.asarray(guessed)

        # Zero flow is a singular linearization of a forward quadratic pump
        # just as its check valve reopens. Predict the midpoint head/speed and
        # seed the algebraic flow on the physically indicated branch. These
        # guesses never replace equations or relax final operating limits.
        p_old = fixed.copy()
        p_old[self._fi] = oldp
        old_alg = algebraic_guess(p_old, oldw)
        old_drive = self._B.T @ p_old
        old_load = []
        for i, pump in enumerate(self.network.pumps):
            ratio = oldw[i] / self.nominal_speeds_rad_s[pump.name]
            eta = (
                pump.efficiency_curve.evaluate(min(pump.max_flow_m3_s, old_alg[i] / ratio))[0]
                if ratio > 0 and pump.efficiency_curve is not None
                else pump.efficiency
            )
            power = (
                max(0.0, -old_drive[npipes + i] * old_alg[i]) / eta
                + self.no_flow_power_w_at_nominal[pump.name] * ratio**3
            )
            old_load.append(power / oldw[i] if oldw[i] > 0 else 0.0)
        old_load = np.asarray(old_load)
        prediction_torque = (
            self._station._allocate(oldw, oldw, commands, old_load, dt)[0]
            if npump
            else np.asarray([])
        )
        predictor = oldw + dt * (prediction_torque - old_load - self._drag * oldw) / self._J
        predicted_q = (
            oldq
            + dt
            * np.asarray(
                [
                    old_drive[i] - pipe.loss_and_gradient(oldq[i], self.network.fluid)[0]
                    for i, pipe in enumerate(self.network.pipes)
                ]
            )
            / self._L
        )
        pm_guess = p_old.copy()
        pm_guess[self._fi] = (
            oldp
            - 0.5 * dt * (self._B[self._fi] @ np.r_[0.5 * (oldq + predicted_q), old_alg]) / self._C
        )
        guessed = algebraic_guess(pm_guess, np.maximum(0.0, 0.5 * (oldw + predictor)))
        x0 = np.r_[
            (2 * pm_guess[self._fi] - oldp) / ps, predicted_q / qs, guessed / qs, predictor / ws
        ]
        lower = np.r_[
            np.full(
                nf,
                max(
                    math.nextafter(self.network.fluid.vapor_pressure_pa, math.inf),
                    self.reference_pressure_pa
                    - self.bulk_modulus_pa * self.max_compression_fraction,
                )
                / ps,
            ),
            np.full(nb, -np.inf),
            -maxw / ws,
        ]
        upper = np.r_[
            np.full(
                nf,
                min(
                    self.max_pressure_pa,
                    self.reference_pressure_pa
                    + self.bulk_modulus_pa * self.max_compression_fraction,
                )
                / ps,
            ),
            np.full(nb, np.inf),
            maxw / ws,
        ]
        # Pump branches have physical forward check valves. Enforce their
        # admissible Q>=0 domain during the solve, while pipes and ordinary
        # bidirectional valves retain signed rates. Without this bound, a
        # near-reopening complementarity root can satisfy scaled residuals
        # with a tiny negative Q but fail the unscaled power audit.
        lower[nf + npipes : nf + npipes + npump] = 0.0
        fit = least_squares(
            lambda x: evaluate(x)[0],
            np.clip(x0, lower + 1e-12, upper - 1e-12),
            bounds=(lower, upper),
            jac_sparsity=self._sparsity,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-13,
            max_nfev=100,
            tr_options={"atol": 1e-12, "btol": 1e-12},
        )
        residual, data = evaluate(fit.x)
        if not fit.success or max(abs(residual), default=0) > 1e-8:
            raise HydraulicConvergenceError("Monolithic liquid/shaft solve failed")
        # This exact unilateral constraint must not leave fictitious rotation
        # from an interior-point lower bound when the motor cannot start.
        fit.x[nf + nb :][held] = 0.0
        _, data = evaluate(fit.x)
        # A closed pump check valve imposes the exact algebraic equality
        # Q=0. Sparse least-squares roundoff can leave a tiny negative Q
        # whose product with a large head trips the turbine-power audit.
        # Project ONLY this active equality, never an open-map flow. Then
        # recheck all coupled equations and the usual mass/energy audits.
        driving = self._B.T @ data[3]
        for i, pump in enumerate(self.network.pumps):
            wm = data[5][i]
            blocked = (
                wm <= 0
                or driving[npipes + i]
                <= replace(
                    pump, speed_ratio=float(wm / self.nominal_speeds_rad_s[pump.name])
                ).loss_and_gradient(0.0, self.network.fluid)[0]
            )
            if blocked:
                fit.x[nf + npipes + i] = 0.0
        residual, data = evaluate(fit.x)
        if max(abs(residual), default=0) > 1e-8:
            raise HydraulicConvergenceError("Closed pump constraint projection residual")
        return data, fit.x, evaluate, fit.nfev

    def _step(self, oldp, oldq, oldw, dt, commands, boundary, valves):
        for v in valves:
            if isinstance(v, _MovingValve):
                v.actuator.step(v.position, v.target, dt, v.mode)
        data, x, evaluate, evaluations = self._solve(
            oldp, oldq, oldw, dt, commands, boundary, valves
        )
        candidates = np.flatnonzero(data[2] < -1e-12)
        if len(candidates):
            if any(commands[i] >= 0 for i in candidates):
                raise HydraulicConvergenceError("Shaft overshoots positive equilibrium")

            def event(h):
                if h == 0:
                    return min(oldw[candidates])
                return min(
                    self._solve(oldp, oldq, oldw, h, commands, boundary, valves)[0][2][candidates]
                )

            if self.backend == "torch":
                from .torch_linear import bisect_scalar

                dt = bisect_scalar(event, 0.0, dt, xtol=1e-12, rtol=1e-11)
            else:
                dt = brentq(event, 0.0, dt, xtol=1e-12, rtol=1e-11)
            data, x, evaluate, extra = self._solve(oldp, oldq, oldw, dt, commands, boundary, valves)
            evaluations += extra
            for i in candidates:
                if abs(data[2][i]) < 1e-8:
                    x[len(self.free) + len(self._branches) + i] = 0.0
        # A braking stop may coincide with the requested endpoint to within
        # the event time tolerance. Remove only this temporal roundoff, never
        # a small positive speed during acceleration or free coasting.
        endw, wm, torque, power = data[2], data[5], data[6], data[7]
        loads = np.divide(power, wm, out=np.zeros_like(power), where=wm > 0)
        deceleration = (loads + self._drag * wm - torque) / self._J
        for i in range(len(self.names)):
            if (
                oldw[i] > 0
                and torque[i] < 0
                and endw[i] >= 0
                and deceleration[i] > 0
                and endw[i] / deceleration[i] <= 1e-12
            ):
                x[len(self.free) + len(self._branches) + i] = 0.0
        residual, data = evaluate(x, audit=True)
        if self.backend == "torch":
            return self._torch_bridge.finish_step(dt, data, evaluations, valves)
        p1, q1, w1, pm, qm, wm, torque, power, losses, fd, fe = data
        if (
            min(w1, default=0) < -1e-10
            or max(abs(w1 - oldw), default=0) > self.max_speed_change_rad_s
        ):
            raise HydraulicConvergenceError("Shaft change guard")
        if max(abs(p1 - oldp)) > self.max_pressure_change_pa:
            raise HydraulicConvergenceError("Pressure change guard")
        for p in p1:
            self._pressure(p)
        kinetic = 0.5 * float(np.sum(self._J * (w1 - oldw) * (w1 + oldw)))
        pipe_energy = 0.5 * float(np.sum(self._L * (q1 - oldq) * (q1 + oldq)))
        elastic = 0.5 * float(
            np.sum(self._C * (p1 - oldp) * (p1 + oldp - 2 * self.reference_pressure_pa))
        )
        work = dt * torque * wm
        electric = np.asarray(
            [
                value / self.drives[n].motor_efficiency
                if value >= 0
                else value * self.drives[n].regeneration_efficiency
                for n, value in zip(self.names, work)
            ]
        )
        motor_heat = float(sum(electric - work))
        drag = dt * float(np.sum(self._drag * wm * wm))
        npipes = len(self.network.pipes)
        hydraulic_pump = float(
            sum(
                -driving * q
                for driving, q in zip(
                    (self._B.T @ pm)[npipes : npipes + len(self.names)],
                    qm[npipes : npipes + len(self.names)],
                )
            )
        )
        pump_heat = dt * (float(sum(power)) - hydraulic_pump)
        friction = dt * float(sum(losses))
        outward = self._B @ qm
        mass_transfers = {
            n: dt * self.network.fluid.density_kg_m3 * outward[self._nodes.index(n)]
            for n in boundary
        }
        boundary_work = dt * sum(
            (boundary[n] - self.reference_pressure_pa) * outward[self._nodes.index(n)]
            for n in boundary
        )
        energy_error = (
            sum(electric)
            + boundary_work
            - kinetic
            - pipe_energy
            - elastic
            - motor_heat
            - drag
            - pump_heat
            - friction
        )
        mass_change = self.network.fluid.density_kg_m3 * float(np.sum(self._C * (p1 - oldp)))
        mass_error = mass_change - sum(mass_transfers.values())
        shaft_error = float(sum(work) - dt * sum(power) - drag - kinetic)
        unit_shaft_errors = (
            work
            - dt * power
            - dt * self._drag * wm * wm
            - 0.5 * self._J * (w1 - oldw) * (w1 + oldw)
        )
        local_mass_errors = self.network.fluid.density_kg_m3 * (
            self._C * (p1 - oldp) + dt * (self._B[self._fi] @ qm)
        )
        peak_draw = math.fsum(
            max(0.0, t) * max(a, b) / self.drives[n].motor_efficiency
            for n, t, a, b in zip(self.names, torque, oldw, w1)
        )
        peak_export = math.fsum(
            max(0.0, -t) * max(a, b) * self.drives[n].regeneration_efficiency
            for n, t, a, b in zip(self.names, torque, oldw, w1)
        )
        scale = max(
            1.0,
            sum(
                abs(v)
                for v in (
                    sum(electric),
                    boundary_work,
                    kinetic,
                    pipe_energy,
                    elastic,
                    motor_heat,
                    drag,
                    pump_heat,
                    friction,
                )
            ),
        )
        if (
            not math.isfinite(energy_error)
            or abs(energy_error) > 1e-6 + 1e-7 * scale
            or abs(shaft_error) > 1e-6 + 1e-7 * scale
            or abs(mass_error) > 1e-8
            or max(abs(local_mass_errors), default=0) > 1e-8
            or max(abs(unit_shaft_errors), default=0) > 1e-6 + 1e-7 * scale
            or min(motor_heat, drag, pump_heat, friction) < -1e-7
        ):
            raise ConservationError("Transient liquid mass/mechanical energy audit failed")
        if (
            self.max_station_electrical_draw_w is not None
            and peak_draw > self.max_station_electrical_draw_w * (1 + 1e-10)
        ) or (
            self.max_station_electrical_export_w is not None
            and peak_export > self.max_station_electrical_export_w * (1 + 1e-10)
        ):
            raise ConservationError("Transient station electrical capacity exceeded")
        return (
            p1,
            q1,
            np.maximum(0.0, w1),
            {
                "dt_s": dt,
                "valve_actuators": {
                    v.name: {
                        **v.actuator.advance(v.position, v.target, dt, v.mode),
                        "midpoint_position": v.at(dt / 2).opening,
                    }
                    for v in valves
                    if isinstance(v, _MovingValve)
                },
                "midpoint_pressures_pa": dict(zip(self._nodes, map(float, pm))),
                "midpoint_branch_volume_rates_m3_s": {
                    b.name: float(q) for b, q in zip(self._branches, qm)
                },
                "relief_openings": {
                    v.name: self.relief_valves[v.name].opening(
                        float((self._B.T @ pm)[len(self.network.pipes) + len(self.names) + i])
                    )
                    for i, v in enumerate(valves)
                    if v.name in self.relief_valves
                },
                "motor_torques_nm": dict(zip(self.names, map(float, torque))),
                "pump_shaft_power_w": dict(zip(self.names, map(float, power))),
                "electrical_energy_j": float(sum(electric)),
                "boundary_work_in_j": float(boundary_work),
                "shaft_kinetic_energy_change_j": kinetic,
                "pipe_kinetic_energy_change_j": pipe_energy,
                "acoustic_energy_change_j": elastic,
                "dissipation_heat_j": {
                    "friction": {
                        branch.name: float(dt * value)
                        for branch, value in zip(self.network.pipes + self.network.valves, losses)
                    },
                    "pump": {
                        pump.name: float(
                            dt * (power[i] + (self._B.T @ pm)[npipes + i] * qm[npipes + i])
                        )
                        for i, pump in enumerate(self.network.pumps)
                    },
                    "shaft_drag": dict(zip(self.names, map(float, dt * self._drag * wm * wm))),
                    "motor": dict(zip(self.names, map(float, electric - work))),
                },
                "motor_heat_j": motor_heat,
                "shaft_drag_heat_j": drag,
                "pump_heat_j": pump_heat,
                "friction_heat_j": friction,
                "boundary_mass_transfers_in_kg": {n: float(v) for n, v in mass_transfers.items()},
                "mass_change_kg": mass_change,
                "mass_residual_kg": float(mass_error),
                "energy_residual_j": float(energy_error),
                "shaft_energy_residual_j": shaft_error,
                "unit_shaft_energy_residuals_j": dict(
                    zip(self.names, map(float, unit_shaft_errors))
                ),
                "max_node_mass_residual_kg": float(max(abs(local_mass_errors), default=0)),
                "peak_electrical_draw_w": peak_draw,
                "peak_electrical_export_w": peak_export,
                "draw_allocation_factor": float(fd),
                "export_allocation_factor": float(fe),
                "nonlinear_evaluations": evaluations,
                "max_scaled_equation_residual": float(max(abs(residual), default=0)),
                "final_pressures_pa": dict(zip(self.free, map(float, p1))),
                "final_pipe_volume_rates_m3_s": {
                    b.name: float(q) for b, q in zip(self.network.pipes, q1)
                },
                "final_speeds_rad_s": dict(zip(self.names, map(float, np.maximum(0.0, w1)))),
            },
        )

    def advance(
        self,
        state,
        duration_s,
        motor_torques_nm=None,
        mode="torque",
        boundary_pressures_pa=None,
        valve_openings=None,
        actuator_modes=None,
        unavailable=None,
        reset_trip=False,
        max_substeps=100000,
    ):
        self._check(state)
        duration = _number(duration_s, "duration_s", positive=True)
        if (
            mode not in ("torque", "off", "trip")
            or not isinstance(reset_trip, bool)
            or (mode == "trip" and reset_trip)
        ):
            raise ValueError("Invalid transient mode/reset")
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("Invalid physical step budget")
        if not math.isfinite(state.time_s + duration) or state.time_s + duration <= state.time_s:
            raise ValueError("Transient interval cannot advance time")
        boundary, valves = self._controls(boundary_pressures_pa, valve_openings)
        modes = dict(actuator_modes or {})
        if set(modes) - set(self.actuated_valves) or any(
            m not in ("normal", "fail_safe", "stuck") for m in modes.values()
        ):
            raise ValueError("Unknown actuator/mode")
        positions = dict(state.actuator_positions)
        valves = [
            _MovingValve(
                v,
                self.actuated_valves[v.name],
                positions[v.name],
                v.opening,
                modes.get(v.name, "normal"),
            )
            if v.name in self.actuated_valves
            else v
            for v in valves
        ]
        requested = dict(motor_torques_nm or {})
        unavailable = state.unavailable if unavailable is None else tuple(unavailable)
        if (
            set(requested) - set(self.names)
            or set(unavailable) - set(self.names)
            or len(unavailable) != len(set(unavailable))
        ):
            raise ValueError("Unknown/duplicate drive command")
        for v in requested.values():
            _number(v, "motor_torque_nm")
        tripped = (state.tripped and not reset_trip) or mode == "trip"
        commands = np.asarray(
            [
                requested.get(n, 0.0)
                if mode == "torque" and not tripped and n not in unavailable
                else 0.0
                for n in self.names
            ]
        )
        p = np.asarray([dict(state.pressures_pa)[n] for n in self.free])
        q = np.asarray([dict(state.pipe_volume_rates_m3_s)[b.name] for b in self.network.pipes])
        w = np.asarray([dict(state.speeds_rad_s)[n] for n in self.names])
        speed = math.sqrt(self.bulk_modulus_pa / self.network.fluid.density_kg_m3)
        ceiling = min(
            [self.maximum_dt_s]
            + [self.acoustic_courant * b.length_m / speed for b in self.network.pipes]
        )
        elapsed, steps, rejected = 0.0, [], {}
        step_dt = ceiling
        while elapsed < duration:
            if len(steps) + sum(rejected.values()) >= max_substeps:
                raise HydraulicConvergenceError("Transient liquid substep budget exhausted")
            remaining = duration - elapsed
            dt = remaining if remaining <= step_dt + 8 * math.ulp(duration) else step_dt
            try:
                coarse = self._step(p, q, w, dt, commands, boundary, valves)
                accepted = [coarse]
                error_ratio = None
                # Stops have their own located time. Compare only trajectories
                # over the same interval, not different stopping times.
                if self.temporal_error_control and math.isclose(
                    coarse[3]["dt_s"], dt, rel_tol=1e-12, abs_tol=0.0
                ):
                    a = self._step(p, q, w, dt / 2, commands, boundary, valves)
                    b = self._step(
                        *a[:3],
                        dt / 2,
                        commands,
                        boundary,
                        [
                            v.after(a[3]["dt_s"]) if isinstance(v, _MovingValve) else v
                            for v in valves
                        ],
                    )
                    if not math.isclose(
                        a[3]["dt_s"], dt / 2, rel_tol=1e-12, abs_tol=0.0
                    ) or not math.isclose(b[3]["dt_s"], dt / 2, rel_tol=1e-12, abs_tol=0.0):
                        raise HydraulicConvergenceError("Temporal comparison crosses shaft stop")
                    if self.backend == "torch":
                        import torch

                        errors = []
                        for old, fine, rough, atol in zip(
                            coarse.initial_state,
                            b.device_state,
                            coarse.device_state,
                            (
                                self.pressure_tolerance_pa,
                                self.flow_tolerance_m3_s,
                                self.speed_tolerance_rad_s,
                            ),
                        ):
                            scale = atol + self.relative_tolerance * torch.maximum(
                                old.abs(), fine.abs()
                            )
                            errors.append((fine - rough).abs() / (3 * scale))
                        error_ratio = float(torch.cat(errors).max().detach().cpu())
                        self._torch_bridge.download_batches += 1
                    else:
                        errors = []
                        for old, fine, rough, atol in zip(
                            (p, q, w),
                            b[:3],
                            coarse[:3],
                            (
                                self.pressure_tolerance_pa,
                                self.flow_tolerance_m3_s,
                                self.speed_tolerance_rad_s,
                            ),
                        ):
                            scale = atol + self.relative_tolerance * np.maximum(abs(old), abs(fine))
                            errors.extend(abs(fine - rough) / (3 * scale))
                        error_ratio = float(max(errors, default=0.0))
                    if not math.isfinite(error_ratio) or error_ratio > 1:
                        raise HydraulicConvergenceError("Temporal error control")
                    # Keep the two conservative half steps. Extrapolating the
                    # endpoint would destroy the exact discrete energy ledger.
                    accepted = [a, b]
            except HydraulicConvergenceError as error:
                rejected[str(error)] = rejected.get(str(error), 0) + 1
                if dt / 2 < self.minimum_dt_s:
                    raise
                step_dt = dt / 2
                continue
            macro_dt = math.fsum(item[3]["dt_s"] for item in accepted)
            following = math.fsum((elapsed, macro_dt))
            if following <= elapsed or state.time_s + following <= state.time_s + elapsed:
                raise ValueError("Transient physical step cannot advance time")
            for p1, q1, w1, record in accepted:
                if len(steps) + sum(rejected.values()) >= max_substeps:
                    raise HydraulicConvergenceError("Transient liquid substep budget exhausted")
                steps.append(
                    {
                        "time_s": state.time_s + elapsed,
                        **record,
                        "temporal_error_ratio": error_ratio,
                    }
                )
                elapsed = math.fsum((elapsed, record["dt_s"]))
                p, q, w = p1, q1, w1
                valves = [
                    v.after(record["dt_s"]) if isinstance(v, _MovingValve) else v for v in valves
                ]
            elapsed = min(duration, following)
            growth = (
                1.5
                if error_ratio is None or error_ratio == 0
                else min(1.5, max(0.8, 0.9 * error_ratio ** (-1 / 3)))
            )
            step_dt = min(ceiling, max(self.minimum_dt_s, macro_dt * growth))
        final = LiquidTransientState(
            dict(zip(self.free, map(float, p))),
            {b.name: float(v) for b, v in zip(self.network.pipes, q)},
            dict(zip(self.names, map(float, w))),
            tuple(unavailable),
            state.time_s + duration,
            self.model_id,
            tripped,
            {v.name: v.position for v in valves if isinstance(v, _MovingValve)},
        )
        return {
            "state": final.to_dict(),
            "properties": self.inspect(final),
            "steps": steps,
            "retry_reasons": rejected,
            "failed_steps": sum(rejected.values()),
            "duration_s": duration,
            "requested_mode": mode,
            "effective_mode": "off" if tripped or mode == "off" else "torque",
        }
