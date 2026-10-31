"""Bounded duty/standby dispatch for a parallel single-liquid pump station.

Enumerate admissible unit subsets, sample safe speed brackets, then solve a
common-speed pressure target. This is steady dispatch, not motor transients or
a proof of global optimality over disconnected unsampled feasible regions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass

from .control import PressureControlResult, PumpPressureControl
from .liquid import EquipmentOperatingError, InvalidPressureError, _name, _number


def station_network_fingerprint(network):
    return hashlib.sha256(
        json.dumps(
            network.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=lambda v: v.item(),
        ).encode()
    ).hexdigest()


class PumpStationOperatingError(EquipmentOperatingError):
    def __init__(self, message, candidates, evaluations):
        super().__init__(message)
        self.candidates = tuple(candidates)
        self.evaluations = evaluations


class _CountedNetwork:
    def __init__(self, network, counter):
        self.network, self.counter = network, counter

    def __getattr__(self, name):
        return getattr(self.network, name)

    def solve(self, **options):
        self.counter[0] += 1
        return self.network.solve(**options)


@dataclass(frozen=True)
class PumpStationResult:
    control: PumpPressureControl
    regulation: PressureControlResult
    running_pumps: tuple
    shaft_power_w: float
    evaluations: int
    candidates: tuple
    model_fingerprint: str
    selection_reason: str
    policy: PumpStationControl

    @property
    def network(self):
        return self.regulation.network

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False, default=lambda v: v.item()))


@dataclass(frozen=True)
class PumpStationControl:
    pumps: tuple
    node: str
    target_pressure_pa: float
    min_speed_ratio: float
    max_speed_ratio: float
    min_running: int = 1
    max_running: int | None = None
    max_station_power_w: float | None = None
    min_npsh_margin_m: float = 0.0
    min_pump_flow_m3_s: float = 0.0
    switch_saving_w: float = 0.0
    pressure_tolerance_pa: float = 0.1
    scan_points: int = 17
    max_configurations: int = 64
    max_control_evaluations: int = 60
    response_direction: str = "auto"

    def __post_init__(self):
        names = tuple(self.pumps)
        if (
            isinstance(self.pumps, str)
            or not names
            or len(names) > 8
            or len(set(names)) != len(names)
        ):
            raise ValueError("Pump station requires 1..8 distinct pump names")
        for name in names:
            _name(name)
        object.__setattr__(self, "pumps", names)
        high = len(names) if self.max_running is None else self.max_running
        object.__setattr__(self, "max_running", high)
        for name in ("min_running", "max_running", "scan_points", "max_configurations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.min_running <= high <= len(names) or self.scan_points < 3:
            raise ValueError("Invalid pump count or speed scan bounds")
        for name in ("min_npsh_margin_m", "min_pump_flow_m3_s", "switch_saving_w"):
            _number(getattr(self, name), name, nonnegative=True)
        if self.max_station_power_w is not None:
            _number(self.max_station_power_w, "max_station_power_w", positive=True)
        self._control(names, self.min_speed_ratio, self.max_speed_ratio)

    def _control(self, names, low, high):
        return PumpPressureControl(
            names[0],
            self.node,
            self.target_pressure_pa,
            low,
            high,
            self.pressure_tolerance_pa,
            self.max_control_evaluations,
            "newton",
            self.response_direction,
            names[1:],
        )

    def _audit(self, result, names):
        controls = result.operating_controls
        if any(
            controls.get("pump_status", {}).get(name, False)
            for name in self.pumps
            if name not in names
        ):
            raise ValueError("Station result enables an unselected pump")
        for name in names:
            speed = controls.get("pump_speed_ratios", {}).get(name)
            if speed is None:
                raise ValueError("Station result misses actual pump speed")
            if not self.min_speed_ratio <= speed <= self.max_speed_ratio:
                raise EquipmentOperatingError(f"{name}: station speed limit exceeded")
            q = result.pump_volume_rates_m3_s[name]
            if q <= 0 or q < self.min_pump_flow_m3_s:
                raise EquipmentOperatingError(f"{name}: station minimum flow not met")
            if self.min_npsh_margin_m > 0:
                if name not in result.pump_npsh_margin_m:
                    raise ValueError("Station NPSH protection requires configured NPSH data")
                if result.pump_npsh_margin_m[name] < self.min_npsh_margin_m:
                    raise EquipmentOperatingError(f"{name}: station NPSH reserve not met")
        power = math.fsum(result.pump_shaft_power_w.get(name, 0.0) for name in names)
        if self.max_station_power_w is not None and power > self.max_station_power_w:
            raise EquipmentOperatingError("Station shaft power limit exceeded")
        return power

    def solve(self, network, initial=None, unavailable=(), previous_running=(), **network_options):
        pumps = {p.name: p for p in network.pumps}
        if set(self.pumps) - set(pumps):
            raise ValueError("Unknown station pump")
        unavailable, previous = tuple(unavailable), tuple(previous_running)
        if (set(unavailable) | set(previous)) - set(self.pumps):
            raise ValueError("Unknown unavailable/previous station pump")
        if len(set(previous)) != len(previous):
            raise ValueError("Duplicate previous running pump")
        if any(not pumps[name].check_valve for name in self.pumps):
            raise ValueError("Parallel station pumps require individual check valves")
        if len({(pumps[n].start, pumps[n].end) for n in self.pumps}) != 1:
            raise ValueError("Station pumps must share suction and discharge nodes")
        if (
            self.node not in network._index
            or network.nodes[network._index[self.node]].pressure_pa is not None
        ):
            raise ValueError("Station pressure target requires a free-pressure node")
        if set(network_options) & {"pump_speed_ratios", "pump_status"}:
            raise ValueError("Station owns pump speed/status for this solve")
        available = tuple(n for n in self.pumps if pumps[n].open and n not in unavailable)
        count = sum(
            math.comb(len(available), k)
            for k in range(self.min_running, min(self.max_running, len(available)) + 1)
        )
        if count > self.max_configurations:
            raise ValueError("Station configuration budget exceeded")
        candidates = []
        solutions = []
        counter = [0]
        for size in range(self.min_running, min(self.max_running, len(available)) + 1):
            for names in itertools.combinations(available, size):
                effective = _CountedNetwork(
                    network.with_controls(pump_status={n: n in names for n in self.pumps}), counter
                )
                trace = []
                safe = []
                # A new duty set or a lower startup speed may make every
                # check valve appear blocked in a warm guess. Start the scan
                # cold; subsequent increasing-speed points reuse safe states.
                guess = None
                for index in range(self.scan_points):
                    speed = self.min_speed_ratio + (
                        self.max_speed_ratio - self.min_speed_ratio
                    ) * index / (self.scan_points - 1)
                    try:
                        result = effective.solve(
                            initial=guess,
                            pump_speed_ratios={n: speed for n in names},
                            **network_options,
                        )
                        self._audit(result, names)
                    except (EquipmentOperatingError, InvalidPressureError) as error:
                        trace.append({"speed_ratio": speed, "safe": False, "reason": str(error)})
                        safe.append(None)
                    else:
                        trace.append({"speed_ratio": speed, "safe": True})
                        safe.append((speed, result))
                        guess = result
                row = {"pumps": list(names), "scan": trace, "status": "no_sampled_target"}
                chosen = None
                for point in safe:
                    if (
                        point is not None
                        and abs(point[1].pressures_pa[self.node] - self.target_pressure_pa)
                        <= self.pressure_tolerance_pa
                    ):
                        c = self._control(names, self.min_speed_ratio, self.max_speed_ratio)
                        chosen = (
                            c,
                            PressureControlResult(
                                point[1],
                                point[0],
                                "target",
                                point[1].pressures_pa[self.node] - self.target_pressure_pa,
                                0,
                            ),
                        )
                        break
                if chosen is None:
                    for lo, hi in zip(safe, safe[1:]):
                        if lo is None or hi is None:
                            continue
                        elo = lo[1].pressures_pa[self.node] - self.target_pressure_pa
                        ehi = hi[1].pressures_pa[self.node] - self.target_pressure_pa
                        if elo * ehi >= 0:
                            continue
                        c = self._control(names, lo[0], hi[0])
                        try:
                            regulated = c.solve(effective, initial=lo[1], **network_options)
                            self._audit(regulated.network, names)
                        except (EquipmentOperatingError, InvalidPressureError) as error:
                            row["reason"] = str(error)
                            continue
                        if regulated.status == "target":
                            chosen = (c, regulated)
                            break
                if chosen is not None:
                    c, regulated = chosen
                    power = self._audit(regulated.network, names)
                    row.update(
                        status="feasible", speed_ratio=regulated.speed_ratio, shaft_power_w=power
                    )
                    # Keep global speed bounds in the returned operating recipe.
                    recipe = self._control(names, self.min_speed_ratio, self.max_speed_ratio)
                    solutions.append((power, names, recipe, regulated))
                candidates.append(row)
        if not solutions:
            raise PumpStationOperatingError(
                "No sampled safe station target; inspect availability, bounds or increase scan_points",
                candidates,
                counter[0],
            )
        solutions.sort(key=lambda x: (x[0], len(x[1]), x[1]))
        best = solutions[0]
        reason = "minimum_sampled_shaft_power"
        previous_solution = next((r for r in solutions if set(r[1]) == set(previous)), None)
        if previous_solution is not None and previous_solution[0] <= best[0] + self.switch_saving_w:
            best = previous_solution
            reason = "retained_by_power_deadband"
        power, names, recipe, regulated = best
        return PumpStationResult(
            recipe,
            regulated,
            names,
            power,
            counter[0],
            tuple(candidates),
            station_network_fingerprint(network),
            reason,
            self,
        )
