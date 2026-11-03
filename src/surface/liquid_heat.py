"""Conservative caloric transport on accepted acoustic liquid trajectories.

Constant cp and hydraulic properties; no thermal expansion, flash or boiling.
Finite suction reservoirs, finite-volume pipe residence and passive wall heat
storage. Mechanical dissipation is recovered once, never added twice.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import splu

from .liquid import ConservationError, EquipmentOperatingError, HydraulicConvergenceError, _number
from .liquid_transient import LiquidTransientNetwork
from .operation import _digest
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class LiquidHeatState:
    temperatures_k: tuple
    wall_temperatures_k: tuple
    masses_kg: tuple
    time_s: float
    model_id: str

    def __post_init__(self):
        for key in ("temperatures_k", "wall_temperatures_k", "masses_kg"):
            values = tuple(sorted(dict(getattr(self, key)).items()))
            for _, v in values:
                _number(v, key, positive=True)
            object.__setattr__(self, key, values)
        _number(self.time_s, "time_s", nonnegative=True)

    def to_dict(self):
        return {
            "temperatures_k": dict(self.temperatures_k),
            "wall_temperatures_k": dict(self.wall_temperatures_k),
            "masses_kg": dict(self.masses_kg),
            "time_s": self.time_s,
            "model_id": self.model_id,
        }


class LiquidHeatNetwork:
    def __init__(
        self,
        physics,
        specific_heat_j_kg_k,
        boundary_temperatures_k,
        boundary_reservoirs_kg=None,
        pipe_cells=None,
        walls=None,
        reference_temperature_k=298.15,
        min_temperature_k=250.0,
        max_temperature_k=450.0,
        pump_heat_to_fluid_fraction=1.0,
        shaft_heat_to_fluid_fraction=0.0,
        motor_heat_to_fluid_fraction=0.0,
        maximum_dt_s=1.0,
        minimum_dt_s=1e-10,
        positivity_courant=0.8,
        maximum_temperature_change_k=1.0,
        temporal_error_control=True,
        temperature_tolerance_k=1e-5,
        relative_tolerance=1e-7,
        store_step_temperatures=True,
        backend=None,
        device=None,
    ):
        if not isinstance(physics, LiquidTransientNetwork):
            raise TypeError("Expected transient liquid physics")
        self.backend = physics.backend if backend is None else backend
        self.device = physics.device if device is None else str(device)
        if self.backend not in ("torch", "scipy_reference"):
            raise ValueError("Unknown caloric backend")
        if self.backend == "torch":
            from .torch_linear import checked_device

            self.device = str(checked_device(self.device))
        if self.backend == "scipy_reference" and self.device != "cpu":
            raise ValueError("SciPy reference backend is CPU-only")
        self._torch_bridge = None
        self.physics = physics
        for key in (
            "specific_heat_j_kg_k",
            "reference_temperature_k",
            "min_temperature_k",
            "max_temperature_k",
            "maximum_dt_s",
            "minimum_dt_s",
            "positivity_courant",
            "maximum_temperature_change_k",
            "temperature_tolerance_k",
            "relative_tolerance",
        ):
            setattr(self, key, _number(locals()[key], key, positive=True))
        if (
            min_temperature_k >= max_temperature_k
            or not min_temperature_k <= reference_temperature_k <= max_temperature_k
            or minimum_dt_s > maximum_dt_s
            or positivity_courant > 1
        ):
            raise ValueError("Invalid caloric domain/integration controls")
        for key in (
            "pump_heat_to_fluid_fraction",
            "shaft_heat_to_fluid_fraction",
            "motor_heat_to_fluid_fraction",
        ):
            value = _number(locals()[key], key, nonnegative=True)
            if value > 1:
                raise ValueError("Heat fraction exceeds one")
            setattr(self, key, value)
        if not isinstance(temporal_error_control, bool):
            raise TypeError("Temporal flag must be boolean")
        self.temporal_error_control = temporal_error_control
        if not isinstance(store_step_temperatures, bool):
            raise TypeError("store_step_temperatures must be boolean")
        self.store_step_temperatures = store_step_temperatures
        boundary = {n.name for n in physics.network.nodes if n.pressure_pa is not None}
        self.boundary_reservoirs_kg = {
            n: _number(m, "reservoir_mass", positive=True)
            for n, m in dict(boundary_reservoirs_kg or {}).items()
        }
        self.boundary_temperatures_k = dict(boundary_temperatures_k)
        if (
            set(self.boundary_reservoirs_kg) & set(self.boundary_temperatures_k)
            or set(self.boundary_reservoirs_kg) | set(self.boundary_temperatures_k) != boundary
        ):
            raise ValueError(
                "Each pressure boundary requires either finite reservoir or prescribed temperature"
            )
        for t in self.boundary_temperatures_k.values():
            self._temperature(t)
        self.pipe_cells = {p.name: 1 for p in physics.network.pipes}
        if set(pipe_cells or {}) - set(self.pipe_cells):
            raise ValueError("Unknown thermal pipe")
        self.pipe_cells.update(pipe_cells or {})
        for n in self.pipe_cells.values():
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError("Positive integer pipe cells required")
        self._paths = {
            p.name: tuple(f"{p.name}::{i}" for i in range(self.pipe_cells[p.name]))
            for p in physics.network.pipes
        }
        self.names = (
            tuple(physics.free)
            + tuple(sorted(self.boundary_reservoirs_kg))
            + tuple(n for p in physics.network.pipes for n in self._paths[p.name])
        )
        thermal_cells = {n for cells in self._paths.values() for n in cells}
        if len(self.names) != len(set(self.names)) or thermal_cells & set(physics._nodes):
            raise ValueError("Thermal cell/node name collision")
        self._index = {n: i for i, n in enumerate(self.names)}
        self.walls = {
            n: VesselWall(**w) if isinstance(w, dict) else w for n, w in dict(walls or {}).items()
        }
        if set(self.walls) - set(self.names) or any(
            not isinstance(w, VesselWall) for w in self.walls.values()
        ):
            raise ValueError("Walls must attach to finite fluid inventories")
        self.wall_names = tuple(sorted(self.walls))
        self._wall_index = {n: len(self.names) + i for i, n in enumerate(self.wall_names)}
        self._pipe_masses = {
            n: physics.network.fluid.density_kg_m3
            * math.pi
            * p.diameter_m**2
            / 4
            * p.length_m
            / len(self._paths[p.name])
            for p in physics.network.pipes
            for n in self._paths[p.name]
        }

    def to_dict(self):
        keys = (
            "backend",
            "device",
            "specific_heat_j_kg_k",
            "boundary_temperatures_k",
            "boundary_reservoirs_kg",
            "pipe_cells",
            "reference_temperature_k",
            "min_temperature_k",
            "max_temperature_k",
            "pump_heat_to_fluid_fraction",
            "shaft_heat_to_fluid_fraction",
            "motor_heat_to_fluid_fraction",
            "maximum_dt_s",
            "minimum_dt_s",
            "positivity_courant",
            "maximum_temperature_change_k",
            "temporal_error_control",
            "temperature_tolerance_k",
            "relative_tolerance",
            "store_step_temperatures",
        )
        return {
            "physics": self.physics.to_dict(),
            "walls": {n: asdict(w) for n, w in self.walls.items()},
            **{k: getattr(self, k) for k in keys},
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["physics"] = LiquidTransientNetwork.from_dict(data["physics"])
        return cls(**data)

    @property
    def model_id(self):
        return _digest({"model": "acoustic_liquid_caloric_transport_v1", **self.to_dict()})

    def _temperature(self, t):
        t = _number(t, "temperature_k", positive=True)
        if not self.min_temperature_k <= t <= self.max_temperature_k:
            raise EquipmentOperatingError(
                "Liquid temperature outside declared caloric domain; no phase-change model"
            )
        return t

    def _free_masses(self, pressures):
        p = self.physics
        rho = p.network.fluid.density_kg_m3
        return {
            n: rho
            * p.storage_volumes_m3[n]
            * (1 + (pressures[n] - p.reference_pressure_pa) / p.bulk_modulus_pa)
            for n in p.free
        }

    def initialize(self, physical, temperatures_k=None, wall_temperatures_k=None):
        self.physics._check(physical)
        initial = dict(temperatures_k or {})
        wall = dict(wall_temperatures_k or {})
        if set(initial) - set(self.names) or set(wall) - set(self.wall_names):
            raise ValueError("Unknown thermal initial coordinate")
        masses = {
            **self._free_masses(dict(physical.pressures_pa)),
            **self.boundary_reservoirs_kg,
            **self._pipe_masses,
        }
        state = LiquidHeatState(
            {n: initial.get(n, self.reference_temperature_k) for n in self.names},
            {n: wall.get(n, initial.get(n, self.reference_temperature_k)) for n in self.wall_names},
            masses,
            physical.time_s,
            self.model_id,
        )
        self._check(state, physical)
        return state

    def _check(self, state, physical=None):
        if not isinstance(state, LiquidHeatState) or state.model_id != self.model_id:
            raise ValueError("Liquid heat state/model mismatch")
        # Build lookup tables ONCE: rebuilding the mass dictionary for
        # every pipe cell made validation quadratic on refined thermal grids.
        masses = dict(state.masses_kg)
        if (
            set(dict(state.temperatures_k)) != set(self.names)
            or set(masses) != set(self.names)
            or set(dict(state.wall_temperatures_k)) != set(self.wall_names)
        ):
            raise ValueError("Thermal coordinate mismatch")
        for _, t in state.temperatures_k:
            self._temperature(t)
        for n, t in state.wall_temperatures_k:
            w = self.walls[n]
            if not w.min_temperature_k <= t <= w.max_temperature_k:
                raise EquipmentOperatingError("Wall temperature outside declared domain")
        for n, m in self._pipe_masses.items():
            if masses[n] != m:
                raise ConservationError("Pipe thermal inventory changed mass")
        if physical is not None:
            self.physics._check(physical)
            if abs(state.time_s - physical.time_s) > 16 * math.ulp(
                max(1.0, state.time_s, physical.time_s)
            ):
                raise ValueError("Thermal/hydraulic clocks differ")
            for n, m in self._free_masses(dict(physical.pressures_pa)).items():
                if abs(masses[n] - m) > 1e-9 + 1e-12 * m:
                    raise ConservationError("Thermal/hydraulic fluid masses differ")

    def inspect(self, state):
        self._check(state)
        t = dict(state.temperatures_k)
        m = dict(state.masses_kg)
        return {
            "fluid_energy_j": math.fsum(
                m[n] * self.specific_heat_j_kg_k * (t[n] - self.reference_temperature_k)
                for n in self.names
            ),
            "wall_energy_j": math.fsum(
                self.walls[n].energy(v) for n, v in state.wall_temperatures_k
            ),
            "fluid_mass_kg": math.fsum(m.values()),
            "temperatures_k": t,
            "wall_temperatures_k": dict(state.wall_temperatures_k),
        }

    def _operator(self, record, boundary, ambient, heat_duties):
        count = len(self.names) + len(self.wall_names)
        cp = self.specific_heat_j_kg_k
        ref = self.reference_temperature_k
        L = lil_matrix((count, count), dtype=float)
        forcing = np.zeros(count)
        sources = np.zeros(count)
        external = []
        for branch in self.physics._branches:
            q = record["midpoint_branch_volume_rates_m3_s"][branch.name]
            rate = abs(q) * self.physics.network.fluid.density_kg_m3 * cp
            path = [branch.start, *self._paths.get(branch.name, ()), branch.end]
            if q < 0:
                path.reverse()
            for start, end in zip(path, path[1:]):
                if start in self._index:
                    L[self._index[start], self._index[start]] -= rate
                if end in self._index:
                    if start in self._index:
                        L[self._index[end], self._index[start]] += rate
                    else:
                        forcing[self._index[end]] += rate * (boundary[start] - ref)
                        external.append((rate * (boundary[start] - ref), None))
                elif start in self._index:
                    external.append((-rate, self._index[start]))
        all_heat = record["dissipation_heat_j"]
        dt = record["dt_s"]
        if set(all_heat) != {"friction", "pump", "shaft_drag", "motor"}:
            raise ValueError("Missing mechanical heat ledger")
        aggregates = {
            "friction": "friction_heat_j",
            "pump": "pump_heat_j",
            "shaft_drag": "shaft_drag_heat_j",
            "motor": "motor_heat_j",
        }
        for name, key in aggregates.items():
            if abs(math.fsum(all_heat[name].values()) - record[key]) > 1e-7 + 1e-10 * abs(
                record[key]
            ):
                raise ConservationError("Per-equipment heat does not match mechanical ledger")
            if any(v < -1e-7 for v in all_heat[name].values()):
                raise ConservationError("Negative equipment dissipation")
        by_name = {b.name: b for b in self.physics._branches}
        if set(all_heat["friction"]) != {
            b.name for b in self.physics.network.pipes + self.physics.network.valves
        } or any(
            set(all_heat[k]) != set(self.physics.names) for k in ("pump", "shaft_drag", "motor")
        ):
            raise ValueError("Mechanical dissipation equipment mismatch")

        def deposit(branch_name, heat):
            branch = by_name[branch_name]
            targets = self._paths.get(branch_name)
            if targets is None:
                targets = (
                    branch.end
                    if record["midpoint_branch_volume_rates_m3_s"][branch_name] >= 0
                    else branch.start,
                )
            for name in targets:
                if name in self._index:
                    sources[self._index[name]] += heat / dt / len(targets)

        for n, h in all_heat["friction"].items():
            deposit(n, h)
        for key, fraction in (
            ("pump", self.pump_heat_to_fluid_fraction),
            ("shaft_drag", self.shaft_heat_to_fluid_fraction),
            ("motor", self.motor_heat_to_fluid_fraction),
        ):
            for n, h in all_heat[key].items():
                deposit(n, h * fraction)
        rejected = math.fsum(record[k] for k in aggregates.values()) - dt * float(sum(sources))
        duty = np.zeros(count)
        for n, v in heat_duties.items():
            duty[self._index[n]] = v
        forcing += sources + duty
        for n, w in self.walls.items():
            i, j = self._index[n], self._wall_index[n]
            L[i, i] -= w.fluid_ua_w_k
            L[i, j] += w.fluid_ua_w_k
            L[j, i] += w.fluid_ua_w_k
            L[j, j] -= w.fluid_ua_w_k + w.ambient_ua_w_k
            forcing[j] += w.ambient_ua_w_k * (ambient[n] - ref)
        return L.tocsc(), forcing, sources, duty, external, rejected

    def _solve(self, t0, m0, m1, dt, L, forcing, external, ambient, cache):
        cp = self.specific_heat_j_kg_k
        nf = len(self.names)
        ref = self.reference_temperature_k
        c0 = np.r_[m0 * cp, [self.walls[n].heat_capacity_j_k for n in self.wall_names]]
        c1 = np.r_[m1 * cp, [self.walls[n].heat_capacity_j_k for n in self.wall_names]]
        theta0 = t0 - ref
        key = (dt, c1.tobytes())
        factors, stats = cache
        if key not in factors:
            if len(factors) >= 4:
                factors.pop(next(iter(factors)))
            factors[key] = splu(diags(c1, format="csc") - dt * 0.5 * L)
            stats[0] += 1
        theta1 = factors[key].solve(c0 * theta0 + dt * 0.5 * (L @ theta0) + dt * forcing)
        t1 = theta1 + ref
        mid = (theta0 + theta1) / 2
        delta = np.r_[
            cp * ((m1 - m0) * theta0[:nf] + m1 * (theta1[:nf] - theta0[:nf])),
            c1[nf:] * (theta1[nf:] - theta0[nf:]),
        ]
        adv = dt * math.fsum(a if i is None else a * mid[i] for a, i in external)
        ambient_heat = dt * math.fsum(
            w.ambient_ua_w_k * (ambient[n] - (mid[self._wall_index[n]] + ref))
            for n, w in self.walls.items()
        )
        local = delta - dt * (L @ mid + forcing)
        return t1, delta, adv, ambient_heat, float(np.max(abs(local), initial=0.0))

    def advance_record(
        self,
        state,
        record,
        boundary_temperatures_k=None,
        ambient_temperatures_k=None,
        heat_duties_w=None,
        max_substeps=100000,
    ):
        if self.backend == "torch":
            if self._torch_bridge is None:
                from .torch_heat_bridge import TorchHeatBridge

                self._torch_bridge = TorchHeatBridge(self, self.device)
            return self._torch_bridge.advance(
                state,
                record,
                boundary_temperatures_k,
                ambient_temperatures_k,
                heat_duties_w,
                max_substeps,
            )
        self._check(state)
        dt = _number(record["dt_s"], "dt_s", positive=True)
        if abs(record["time_s"] - state.time_s) > 32 * math.ulp(
            max(1.0, state.time_s, record["time_s"])
        ):
            raise ValueError("Hydraulic/thermal record time mismatch")
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("Invalid thermal work budget")
        boundary = {**self.boundary_temperatures_k, **dict(boundary_temperatures_k or {})}
        ambient = {n: w.ambient_temperature_k for n, w in self.walls.items()}
        ambient.update(ambient_temperatures_k or {})
        duties = dict(heat_duties_w or {})
        if (
            set(boundary) != set(self.boundary_temperatures_k)
            or set(ambient) != set(self.walls)
            or set(duties) - set(self.names)
        ):
            raise ValueError("Unknown prescribed boundary, wall or finite heat source")
        for t in boundary.values():
            self._temperature(t)
        for t in ambient.values():
            _number(t, "ambient_temperature_k", positive=True)
        for value in duties.values():
            _number(value, "heat_duty_w")
        L, forcing, sources, duty, external, rejected = self._operator(
            record, boundary, ambient, duties
        )
        mstart = dict(state.masses_kg)
        mend = {**mstart, **self._free_masses(record["final_pressures_pa"])}
        for n in self.boundary_reservoirs_kg:
            mend[n] -= record["boundary_mass_transfers_in_kg"][n]
        for mass in mend.values():
            if not math.isfinite(mass) or mass <= 0:
                raise EquipmentOperatingError("Finite thermal reservoir depleted")
        mstart = np.asarray([mstart[n] for n in self.names])
        mend = np.asarray([mend[n] for n in self.names])
        t = np.asarray(
            [dict(state.temperatures_k)[n] for n in self.names]
            + [dict(state.wall_temperatures_k)[n] for n in self.wall_names]
        )
        cache = ({}, [0])
        fraction = 0.0
        hnext = min(dt, self.maximum_dt_s)
        steps = []
        retries = 0
        nf = len(self.names)
        wallcap = np.asarray([self.walls[n].heat_capacity_j_k for n in self.wall_names])
        capacities = np.r_[np.minimum(mstart, mend) * self.specific_heat_j_kg_k, wallcap]
        sinks = -L.diagonal()
        limit = min(
            [self.maximum_dt_s]
            + [self.positivity_courant * c / s for c, s in zip(capacities, sinks) if s > 0]
        )
        while fraction < dt:
            if len(steps) + retries >= max_substeps:
                raise HydraulicConvergenceError("Thermal work budget exhausted")
            remain = dt - fraction
            h = min(hnext, limit, remain)
            if remain <= h + 8 * math.ulp(dt):
                h = remain
            if h < self.minimum_dt_s and remain > self.minimum_dt_s:
                raise HydraulicConvergenceError("Thermal positivity limit below minimum step")
            # Hydraulic accuracy control can emit two conservative halves
            # smaller than one ULP of their absolute timestamp. Their relative
            # durations still carry real work; integrate both, then anchor the
            # containing window to its hydraulic endpoint.
            if fraction + h <= fraction:
                raise ValueError("Thermal relative substep cannot advance time")
            m0 = mstart + (mend - mstart) * (fraction / dt)
            m1 = mstart + (mend - mstart) * ((fraction + h) / dt)
            coarse = self._solve(t, m0, m1, h, L, forcing, external, ambient, cache)
            accepted = [(coarse, m0, m1, h)]
            error = 0.0
            if self.temporal_error_control:
                mm = (m0 + m1) / 2
                a = self._solve(t, m0, mm, h / 2, L, forcing, external, ambient, cache)
                b = self._solve(a[0], mm, m1, h / 2, L, forcing, external, ambient, cache)
                error = float(
                    np.max(
                        abs(b[0] - coarse[0])
                        / (
                            3
                            * (
                                self.temperature_tolerance_k
                                + self.relative_tolerance * np.maximum(abs(t), abs(b[0]))
                            )
                        ),
                        initial=0.0,
                    )
                )
                accepted = [(a, m0, mm, h / 2), (b, mm, m1, h / 2)]
            change = float(np.max(abs(accepted[-1][0][0] - t), initial=0.0))
            guard = min(
                [self.maximum_temperature_change_k]
                + [w.max_temperature_change_k for w in self.walls.values()]
            )
            if not math.isfinite(error) or error > 1 or change > guard:
                retries += 1
                if h / 2 < self.minimum_dt_s:
                    raise HydraulicConvergenceError("Thermal accuracy exhausted minimum step")
                hnext = h / 2
                continue
            following = dt if h == remain else min(dt, math.fsum((fraction, h)))
            for (t1, delta, adv, ambient_heat, local), ma, mb, hd in accepted:
                if len(steps) + retries >= max_substeps:
                    raise HydraulicConvergenceError("Thermal work budget exhausted")
                if (
                    not np.all(np.isfinite(t1[:nf]))
                    or np.any(t1[:nf] < self.min_temperature_k)
                    or np.any(t1[:nf] > self.max_temperature_k)
                ):
                    raise EquipmentOperatingError(
                        "Liquid temperature outside declared caloric domain; no phase-change model"
                    )
                for n, v in zip(self.wall_names, t1[nf:]):
                    w = self.walls[n]
                    if not w.min_temperature_k <= v <= w.max_temperature_k:
                        raise EquipmentOperatingError("Wall temperature outside declared domain")
                absorbed = hd * float(sum(sources))
                external_heat = hd * float(sum(duty))
                residual = math.fsum(delta) - adv - ambient_heat - absorbed - external_heat
                scale = max(
                    1.0,
                    abs(adv)
                    + abs(ambient_heat)
                    + abs(absorbed)
                    + abs(external_heat)
                    + sum(abs(delta)),
                )
                if (
                    not np.all(np.isfinite(t1))
                    or abs(residual) > 1e-7 + 1e-10 * scale
                    or local > 1e-7 + 1e-10 * scale
                ):
                    raise ConservationError("Liquid thermal energy audit failed")
                tm = (t + t1) / 2
                exchange_entropy = hd * math.fsum(
                    w.fluid_ua_w_k
                    * (tm[self._index[n]] - tm[self._wall_index[n]]) ** 2
                    / (tm[self._index[n]] * tm[self._wall_index[n]])
                    + w.ambient_ua_w_k
                    * (tm[self._wall_index[n]] - ambient[n]) ** 2
                    / (tm[self._wall_index[n]] * ambient[n])
                    for n, w in self.walls.items()
                )
                steps.append(
                    {
                        "passive_exchange_entropy_j_k": exchange_entropy,
                        "time_s": state.time_s + fraction,
                        "dt_s": hd,
                        "fluid_energy_change_j": float(sum(delta[:nf])),
                        "wall_energy_change_j": float(sum(delta[nf:])),
                        "external_advective_heat_in_j": adv,
                        "ambient_heat_in_j": ambient_heat,
                        "dissipation_to_inventory_j": absorbed,
                        "equipment_heat_out_j": rejected * (hd / dt),
                        "external_heat_duty_j": external_heat,
                        "energy_residual_j": residual,
                        "max_local_energy_residual_j": local,
                        "temporal_error_ratio": error,
                    }
                )
                if self.store_step_temperatures:
                    steps[-1]["final_temperatures_k"] = dict(zip(self.names, map(float, t1[:nf])))
                    steps[-1]["final_wall_temperatures_k"] = dict(
                        zip(self.wall_names, map(float, t1[nf:]))
                    )
                fraction = math.fsum((fraction, hd))
                t = t1
            fraction = following
            hnext = min(
                self.maximum_dt_s,
                h * min(1.5, max(0.8, 0.9 * error ** (-1 / 3))) if error > 0 else h * 1.5,
            )
        final = LiquidHeatState(
            dict(zip(self.names, map(float, t[:nf]))),
            dict(zip(self.wall_names, map(float, t[nf:]))),
            dict(zip(self.names, map(float, mend))),
            record["time_s"] + dt,
            self.model_id,
        )
        self._check(final)
        thermal_delta = math.fsum(
            s["fluid_energy_change_j"] + s["wall_energy_change_j"] for s in steps
        )
        mech_delta = (
            record["shaft_kinetic_energy_change_j"]
            + record["pipe_kinetic_energy_change_j"]
            + record["acoustic_energy_change_j"]
        )
        total_residual = (
            record["electrical_energy_j"]
            + record["boundary_work_in_j"]
            + math.fsum(
                s["external_advective_heat_in_j"]
                + s["ambient_heat_in_j"]
                + s["external_heat_duty_j"]
                - s["equipment_heat_out_j"]
                for s in steps
            )
            - mech_delta
            - thermal_delta
        )
        if abs(total_residual) > 1e-5 + 1e-9 * max(1, abs(thermal_delta), abs(mech_delta)):
            raise ConservationError("Combined liquid mechanical/caloric energy audit failed")
        return {
            "state": final.to_dict(),
            "steps": steps,
            "failed_steps": retries,
            "combined_energy_residual_j": total_residual,
            "linear_factorizations": cache[1][0],
        }
