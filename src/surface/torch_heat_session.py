"""Persistent caloric state with scalar sensor export at digital control ticks.

A session belongs to one model and is explicitly closed to serialize a restart.
Each window is transactional. Schedule/PID logic stays on the host; it observes
only declared sensors, not a fabricated or lagged full temperature field.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from .liquid import _number


@dataclass(frozen=True)
class TorchHeatObservation:
    temperatures_k: tuple
    wall_temperatures_k: tuple
    time_s: float


class TorchHeatSession:
    def __init__(self, model, state, fluid_sensors=(), wall_sensors=(), execution="auto"):
        if model.backend != "torch":
            raise ValueError("Resident heat sessions require the Torch backend")
        fluid, wall = tuple(fluid_sensors), tuple(wall_sensors)
        if (
            len(set(fluid)) != len(fluid)
            or len(set(wall)) != len(wall)
            or set(fluid) - set(model.names)
            or set(wall) - set(model.wall_names)
        ):
            raise ValueError("Unknown or duplicate thermal sensor")
        if model._torch_bridge is None:
            from .torch_heat_bridge import TorchHeatBridge

            model._torch_bridge = TorchHeatBridge(model, model.device)
        if execution not in ("auto", "eager", "graph"):
            raise ValueError("thermal_execution must be auto, eager or graph")
        if execution == "graph" and model._torch_bridge.kernel.device.type != "cuda":
            raise ValueError("Thermal graph execution requires CUDA")
        self.execution = execution
        self.bridge = model._torch_bridge
        self.closed = False
        self.fluid_sensors = fluid
        self.wall_sensors = wall
        before = self.bridge.transfer_stats()
        self._cursor = self.bridge.begin(state)
        indices = [model._index[n] for n in fluid] + [model._wall_index[n] for n in wall]
        self._indices = self.bridge.upload(np.asarray(indices, dtype=np.int64), dtype=torch.long)
        after = self.bridge.transfer_stats()
        self._transfers = {k: after[k] - before[k] for k in before}

    def _open(self):
        if self.closed:
            raise RuntimeError("Heat session is closed")
        if self.bridge.model.model_id != self.bridge.model_id:
            raise ValueError("Heat model changed during a resident session")

    def _account(self, before):
        after = self.bridge.transfer_stats()
        for name, count in after.items():
            self._transfers[name] += count - before[name]

    def transfer_stats(self):
        return dict(self._transfers)

    @property
    def time_s(self):
        self._open()
        return self._cursor.time_s

    def _observe(self, cursor):
        if self._indices.numel():
            values = self.bridge.download(cursor.temperature.index_select(0, self._indices))
            self.bridge.sensor_download_batches += 1
            self.bridge.sensor_download_bytes += values.nbytes
        else:
            values = np.empty(0)
        split = len(self.fluid_sensors)
        return TorchHeatObservation(
            tuple(zip(self.fluid_sensors, map(float, values[:split]))),
            tuple(zip(self.wall_sensors, map(float, values[split:]))),
            cursor.time_s,
        )

    def observe(self):
        self._open()
        before = self.bridge.transfer_stats()
        try:
            return self._observe(self._cursor)
        finally:
            self._account(before)

    def advance_records(
        self,
        records,
        boundary_temperatures_k=None,
        ambient_temperatures_k=None,
        heat_duties_w=None,
        max_substeps=100000,
    ):
        self._open()
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError("Thermal window requires nonempty hydraulic records")
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("Positive integer thermal work budget required")
        before = self.bridge.transfer_stats()
        try:
            result = self.bridge.advance_records(
                self._cursor,
                records,
                boundary_temperatures_k,
                ambient_temperatures_k,
                heat_duties_w,
                max_substeps,
                _keep_on_device=True,
                _execution=self.execution,
            )
            candidate = result["state"]
            observations = self._observe(candidate)
            # No state is exposed until every record/audit and observation has
            # succeeded. A rejected window can be retried on this same session.
            self._cursor = candidate
            result["state"] = observations
            result["transfer_stats"] = {
                k: v - before[k] for k, v in self.bridge.transfer_stats().items()
            }
            return result
        finally:
            self._account(before)

    def align_time(self, time_s):
        """Synchronize only endpoint roundoff with the hydraulic report clock."""
        self._open()
        time_s = _number(time_s, "time_s", nonnegative=True)
        if abs(time_s - self._cursor.time_s) > 32 * math.ulp(max(1.0, time_s, self._cursor.time_s)):
            raise ValueError("Cannot align an unintegrated thermal interval")
        self._cursor = replace(self._cursor, time_s=time_s)

    def snapshot(self):
        self._open()
        before = self.bridge.transfer_stats()
        try:
            return self.bridge.export(self._cursor)
        finally:
            self._account(before)

    def close(self):
        final = self.snapshot()
        self.closed = True
        self._cursor = None
        return final


class TorchHeatWindow:
    """Streaming trial cursor; publish only after the whole control window succeeds."""

    def __init__(self, session, boundary, ambient, duties, max_substeps):
        session._open()
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("Positive integer thermal work budget required")
        self.session = session
        self.initial = self.cursor = session._cursor
        self.boundary, self.ambient, self.duties = boundary, ambient, duties
        self.budget = max_substeps
        self.steps, self.audits = [], []
        self.failed = self.factors = 0
        self.closed = False

    def push(self, record, payload):
        # Model fingerprints are checked at window begin and before commit.
        # Rehashing the full topology per accepted step adds host work without
        # improving the transactional publication guard.
        if self.session.closed:
            raise RuntimeError("Heat session is closed")
        if self.closed or self.session._cursor is not self.initial:
            raise RuntimeError("Stale or closed thermal window")
        budget = self.budget - len(self.steps) - self.failed
        if budget < 1:
            self.closed = True
            from .liquid import HydraulicConvergenceError

            raise HydraulicConvergenceError("Thermal window work budget exhausted")
        bridge = self.session.bridge
        before = bridge.transfer_stats()
        try:
            if payload is None:
                raise ValueError("Streaming heat requires a hydraulic device record")
            calc = bridge.advance(
                self.cursor,
                record,
                self.boundary,
                self.ambient,
                self.duties,
                budget,
                _keep_on_device=True,
                _hydraulic=payload,
                _execution=self.session.execution,
            )
            self.cursor = calc["state"]
            self.steps.extend(calc["steps"])
            self.failed += calc["failed_steps"]
            self.factors += calc["linear_factorizations"]
            self.audits.append(
                {
                    "time_s": record["time_s"],
                    "dt_s": record["dt_s"],
                    "combined_energy_residual_j": calc["combined_energy_residual_j"],
                }
            )
        except Exception:
            self.closed = True
            raise
        finally:
            self.session._account(before)

    def finish(self):
        session = self.session
        session._open()
        if self.closed or session._cursor is not self.initial or not self.audits:
            raise RuntimeError("Stale, empty or closed thermal window")
        before = session.bridge.transfer_stats()
        try:
            observation = session._observe(self.cursor)
            session._cursor = self.cursor
            self.closed = True
            return {
                "state": observation,
                "steps": self.steps,
                "audits": self.audits,
                "failed_steps": self.failed,
                "linear_factorizations": self.factors,
            }
        finally:
            session._account(before)
