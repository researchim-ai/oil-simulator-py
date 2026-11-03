"""Sampled minimum-flow recycle through physical actuated liquid branches.

The controller never adds a source. Bypass flow, valve dissipation and inlet
reservoir exchange remain in the conservative monolithic liquid equations.
"""

import math
from dataclasses import asdict, dataclass

from .liquid import HydraulicConvergenceError, _number
from .liquid_protection import ProtectedLiquidState, ProtectedLiquidStation
from .liquid_relief import inspect_liquid_pumps
from .operation import _digest
from .pid import PIDController, PIDState


@dataclass(frozen=True)
class LiquidRecycleLoop:
    name: str
    valve: str
    pumps: tuple
    controller: PIDController
    minimum_flow_m3_s: float
    min_speed_ratio: float = 0.6
    startup_opening: float = 1.0
    inactive_opening: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "pumps", tuple(self.pumps))
        if not self.name or not self.pumps or len(set(self.pumps)) != len(self.pumps):
            raise ValueError("Recycle loop requires a name and distinct pumps")
        if (
            not isinstance(self.controller, PIDController)
            or self.controller.min_output != 0
            or self.controller.max_output != 1
            or self.controller.direction != 1
        ):
            raise ValueError("Recycle PID must open on positive flow deficit, bounds [0,1]")
        _number(self.minimum_flow_m3_s, "minimum_flow_m3_s", positive=True)
        _number(self.min_speed_ratio, "min_speed_ratio", positive=True)
        for value in (self.startup_opening, self.inactive_opening):
            if not 0 <= _number(value, "opening") <= 1:
                raise ValueError("Recycle opening outside [0,1]")


@dataclass(frozen=True)
class RecyclingLiquidState:
    protected: ProtectedLiquidState
    pids: tuple
    targets: tuple
    clock_origin_s: float
    next_tick: int
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "pids", tuple(sorted(dict(self.pids).items())))
        object.__setattr__(self, "targets", tuple(sorted(dict(self.targets).items())))
        _number(self.clock_origin_s, "clock_origin_s", nonnegative=True)
        if (
            isinstance(self.next_tick, bool)
            or not isinstance(self.next_tick, int)
            or self.next_tick < 0
        ):
            raise ValueError("Invalid recycle clock")

    def to_dict(self):
        return {
            "protected": self.protected.to_dict(),
            "pids": {n: p.to_dict() for n, p in self.pids},
            "targets": dict(self.targets),
            "clock_origin_s": self.clock_origin_s,
            "next_tick": self.next_tick,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["protected"] = ProtectedLiquidState.from_dict(data["protected"])
        data["pids"] = {n: PIDState(**p) for n, p in data["pids"].items()}
        return cls(**data)


class RecyclingLiquidStation:
    def __init__(self, protected, loops, control_interval_s=0.05):
        if not isinstance(protected, ProtectedLiquidStation):
            raise TypeError("Recycle requires a protected liquid station")
        self.protected = protected
        self.loops = tuple(loops)
        self.control_interval_s = _number(control_interval_s, "control_interval_s", positive=True)
        self.physics = protected.controller.physics
        if not self.loops or any(not isinstance(x, LiquidRecycleLoop) for x in self.loops):
            raise ValueError("Nonempty recycle loops required")
        if (
            len({x.name for x in self.loops}) != len(self.loops)
            or len({x.valve for x in self.loops}) != len(self.loops)
            or sum(len(x.pumps) for x in self.loops)
            != len({n for x in self.loops for n in x.pumps})
        ):
            raise ValueError("Duplicate loop, valve or regulated pump")
        valves = {v.name: v for v in self.physics.network.valves}
        pumps = {p.name: p for p in self.physics.network.pumps}
        for loop in self.loops:
            if loop.valve not in self.physics.actuated_valves or set(loop.pumps) - set(pumps):
                raise ValueError("Recycle requires an actual actuated branch and known pumps")
            valve = valves[loop.valve]
            if not valve.check_valve or any(
                valve.start != pumps[n].end or valve.end != pumps[n].start for n in loop.pumps
            ):
                raise ValueError("Recycle must return discharge to suction of its parallel pumps")
            if protected.controller.planning_valve_openings.get(loop.valve) != 0:
                raise ValueError("Steady dispatch must explicitly assume nominal closed recycle")

    def to_dict(self):
        return {
            "protected": self.protected.to_dict(),
            "loops": [asdict(x) for x in self.loops],
            "control_interval_s": self.control_interval_s,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["protected"] = ProtectedLiquidStation.from_dict(data["protected"])
        data["loops"] = [
            LiquidRecycleLoop(**{**x, "controller": PIDController(**x["controller"])})
            for x in data["loops"]
        ]
        return cls(**data)

    @property
    def model_id(self):
        return _digest({"model": "physical_minimum_flow_recycle_v1", **self.to_dict()})

    def initialize(self, protected_initial=None):
        current = self.protected.initialize(**dict(protected_initial or {}))
        positions = dict(current.controlled.physical.actuator_positions)
        pids = {
            x.name: x.controller.initialize(0, positions[x.valve], x.minimum_flow_m3_s)
            for x in self.loops
        }
        return RecyclingLiquidState(
            current,
            pids,
            {x.valve: positions[x.valve] for x in self.loops},
            current.controlled.physical.time_s,
            0,
            self.model_id,
        )

    def _check(self, state):
        if not isinstance(state, RecyclingLiquidState) or state.model_id != self.model_id:
            raise ValueError("Recycle state/model mismatch")
        self.protected._check(state.protected)
        if set(dict(state.pids)) != {x.name for x in self.loops} or set(dict(state.targets)) != {
            x.valve for x in self.loops
        }:
            raise ValueError("Recycle memory coordinates mismatch")
        for loop in self.loops:
            if dict(state.pids)[loop.name].model_id != loop.controller.model_id:
                raise ValueError("Recycle PID mismatch")
            self.physics.actuated_valves[loop.valve].position(dict(state.targets)[loop.valve])
        now = state.protected.controlled.physical.time_s
        next_time = state.clock_origin_s + state.next_tick * self.control_interval_s
        tol = 16 * math.ulp(max(1.0, now, next_time))
        if (
            state.clock_origin_s > now
            or next_time < now - tol
            or next_time > now + self.control_interval_s + tol
        ):
            raise ValueError("Recycle clock inconsistent with physical time")

    def advance(
        self,
        state,
        duration_s,
        recycle_mode="auto",
        recycle_openings=None,
        max_recycle_samples=100000,
        max_substeps=100000,
        max_control_samples=100000,
        max_protection_samples=100000,
        **commands,
    ):
        self._check(state)
        from .liquid_journal import journal_branches

        selected = journal_branches(
            self.physics,
            commands.get("hydraulic_journal", "full"),
            commands.get("hydraulic_journal_branches"),
        )
        if commands.get("hydraulic_journal", "full") == "compact":
            commands["hydraulic_journal_branches"] = tuple(
                dict.fromkeys((*selected, *(x.valve for x in self.loops)))
            )
        duration = _number(duration_s, "duration_s", positive=True)
        for budget in (
            max_recycle_samples,
            max_substeps,
            max_control_samples,
            max_protection_samples,
        ):
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
                raise ValueError("Positive integer work budgets required")
        if recycle_mode not in ("auto", "manual", "off"):
            raise ValueError("Unknown recycle mode")
        manual = dict(recycle_openings or {})
        owned = {x.valve for x in self.loops}
        if set(manual) - owned or owned & set(commands.get("valve_openings") or {}):
            raise ValueError("Unknown recycle valve or conflicting direct valve command")
        for n, value in manual.items():
            self.physics.actuated_valves[n].position(value)
        if recycle_mode == "manual" and set(manual) != owned:
            raise ValueError("Manual recycle requires every loop opening")
        current, tick = state.protected, state.next_tick
        targets, pids = dict(state.targets), dict(state.pids)
        end = current.controlled.physical.time_s + duration
        if not math.isfinite(end) or end <= current.controlled.physical.time_s:
            raise ValueError("Recycle interval cannot advance time")
        records = {
            k: []
            for k in ("steps", "control_samples", "protection_samples", "dispatches", "events")
        }
        samples, retries, first = [], {}, True
        while current.controlled.physical.time_s < end:
            physical = current.controlled.physical
            now = physical.time_s
            next_time = state.clock_origin_s + tick * self.control_interval_s
            tol = 8 * math.ulp(max(1.0, now, next_time, end))
            terminal = 0 < end - now <= tol
            if now >= next_time - tol and not terminal:
                if len(samples) // len(self.loops) >= max_recycle_samples:
                    raise HydraulicConvergenceError("Recycle sample budget exhausted")
                measured = inspect_liquid_pumps(
                    self.physics, physical, commands.get("boundary_pressures_pa")
                )
                requested = set(commands.get("running_pumps", current.controlled.selected))
                if commands.get("mode") == "torque":
                    requested = set(
                        commands.get("running_pumps", commands.get("motor_torques_nm", {}))
                    )
                blocked = set(physical.unavailable) | set(commands.get("unavailable") or ())
                enabled = not physical.tripped and commands.get("mode", "auto") not in (
                    "trip",
                    "off",
                )
                positions = dict(physical.actuator_positions)
                for loop in self.loops:
                    selected = [
                        n for n in loop.pumps if n in requested and n not in blocked and enabled
                    ]
                    armed = [
                        n
                        for n in selected
                        if dict(physical.speeds_rad_s)[n]
                        >= loop.min_speed_ratio * self.physics.nominal_speeds_rad_s[n]
                    ]
                    flow = min((measured[n]["flow_m3_s"] for n in armed), default=0.0)
                    mode = "auto" if armed and recycle_mode == "auto" else "manual"
                    target = (
                        manual[loop.valve]
                        if recycle_mode == "manual"
                        else loop.startup_opening
                        if selected and recycle_mode == "auto"
                        else loop.inactive_opening
                    )
                    update = loop.controller.step(
                        pids[loop.name],
                        loop.minimum_flow_m3_s,
                        flow,
                        self.control_interval_s,
                        mode=mode,
                        manual_output=target,
                        applied_output=positions[loop.valve],
                    )
                    pids[loop.name] = PIDState(**update["state"])
                    targets[loop.valve] = update["output"]
                    samples.append(
                        {
                            "time_s": now,
                            "loop": loop.name,
                            "armed_pumps": armed,
                            "minimum_measured_flow_m3_s": flow,
                            "actual_position": positions[loop.valve],
                            "mode": mode,
                            "target_position": update["output"],
                            "flow_deficit_m3_s": max(0.0, loop.minimum_flow_m3_s - flow)
                            if armed
                            else None,
                        }
                    )
                tick += 1
                next_time = state.clock_origin_s + tick * self.control_interval_s
                if next_time <= now:
                    raise ValueError("Recycle clock cannot advance")
            segment = dict(commands)
            if not first:
                for name in ("reset_trip", "reset_trips", "redispatch"):
                    segment.pop(name, None)
            segment["valve_openings"] = {**dict(segment.get("valve_openings") or {}), **targets}
            remaining = {
                "max_substeps": max_substeps - len(records["steps"]) - sum(retries.values()),
                "max_control_samples": max_control_samples - len(records["control_samples"]),
                "max_protection_samples": max_protection_samples
                - len(records["protection_samples"]),
            }
            if remaining["max_substeps"] < 1:
                raise HydraulicConvergenceError("Recycle physical budget exhausted")
            result = self.protected.advance(
                current,
                (end if terminal else min(end, next_time)) - now,
                **{k: max(1, v) for k, v in remaining.items()},
                **segment,
            )
            for key, budget_key in (
                ("control_samples", "max_control_samples"),
                ("protection_samples", "max_protection_samples"),
            ):
                if len(result[key]) > remaining[budget_key]:
                    raise HydraulicConvergenceError("Recycle nested sample budget exhausted")
            for key, values in records.items():
                values.extend(result[key])
            for reason, count in result["retry_reasons"].items():
                retries[reason] = retries.get(reason, 0) + count
            current = ProtectedLiquidState.from_dict(result["state"])
            first = False
        final = RecyclingLiquidState(
            current, pids, targets, state.clock_origin_s, tick, self.model_id
        )
        self._check(final)
        return {
            "state": final.to_dict(),
            **records,
            "recycle_samples": samples,
            "recycled_volume_m3": {
                x.valve: math.fsum(
                    s["dt_s"] * s["midpoint_branch_volume_rates_m3_s"][x.valve]
                    for s in records["steps"]
                )
                for x in self.loops
            },
            "retry_reasons": retries,
            "failed_steps": sum(retries.values()),
            "latched_trips": result["latched_trips"],
            "properties": result["properties"],
            "pressure_error_pa": result["pressure_error_pa"],
            "duration_s": duration,
        }
