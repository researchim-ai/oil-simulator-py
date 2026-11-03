"""Explicit serialization/adaptivity boundary for the resident Torch heat kernel."""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .liquid import ConservationError, EquipmentOperatingError, HydraulicConvergenceError, _number
from .liquid_heat import LiquidHeatState
from .torch_heat import TorchHeatEquations


@dataclass(frozen=True)
class _ResidentHeatState:
    temperature: torch.Tensor
    mass: torch.Tensor
    time_s: float
    model_id: str


class TorchHeatBridge:
    def __init__(self, model, device):
        self.model = model
        self.kernel = TorchHeatEquations(model, device)
        self.upload_batches = 0
        self.download_batches = 0
        self.upload_bytes = 0
        self.download_bytes = 0
        self.state_upload_batches = 0
        self.state_download_batches = 0
        self.state_upload_bytes = 0
        self.state_download_bytes = 0
        self.history_download_bytes = 0
        self.history_flushes = 0
        self.sensor_download_batches = 0
        self.sensor_download_bytes = 0
        self.device_hydraulic_records = 0
        self.device_hydraulic_bytes = 0
        self.history_buffer_bytes = 16 * 1024 * 1024
        from .torch_heat_graph import TorchHeatLinearGraphs

        self.linear_graphs = TorchHeatLinearGraphs(self.kernel)
        self.model_id = model.model_id
        self.physics_model_id = model.physics.model_id

    def transfer_stats(self):
        # Logical API packets/bytes, excluding one-time geometry construction.
        # On CPU these count serialization traffic, not PCIe transfers.
        names = (
            "device_hydraulic_records",
            "device_hydraulic_bytes",
            "upload_batches",
            "download_batches",
            "upload_bytes",
            "download_bytes",
            "state_upload_batches",
            "state_download_batches",
            "state_upload_bytes",
            "state_download_bytes",
            "history_download_bytes",
            "history_flushes",
            "sensor_download_batches",
            "sensor_download_bytes",
        )
        return {
            **{name: getattr(self, name) for name in names},
            "linear_graph_captures": self.linear_graphs.captures,
            "linear_graph_replays": self.linear_graphs.replays,
            "linear_graph_input_bytes": self.linear_graphs.input_bytes,
            "linear_graph_input_packets": self.linear_graphs.input_packets,
        }

    def upload(self, value, dtype=None):
        self.upload_batches += 1
        self.upload_bytes += value.nbytes
        return torch.as_tensor(value, dtype=dtype or self.kernel.dtype, device=self.kernel.device)

    def download(self, value):
        self.download_batches += 1
        self.download_bytes += value.numel() * value.element_size()
        return value.detach().cpu().numpy()

    def begin(self, state):
        m, e = self.model, self.kernel
        m._check(state)
        if m.model_id != self.model_id:
            raise ValueError("Heat model changed after Torch kernel construction")
        fluid, wall, mass = (
            dict(state.temperatures_k),
            dict(state.wall_temperatures_k),
            dict(state.masses_kg),
        )
        packet = self.upload(
            np.asarray(
                [fluid[n] for n in m.names]
                + [wall[n] for n in m.wall_names]
                + [mass[n] for n in m.names]
            )
        )
        self.state_upload_batches += 1
        self.state_upload_bytes += packet.numel() * packet.element_size()
        return _ResidentHeatState(packet[: e.size], packet[e.size :], state.time_s, self.model_id)

    def export(self, state):
        m, e = self.model, self.kernel
        host = self.download(torch.cat((state.temperature, state.mass)))
        self.state_download_batches += 1
        self.state_download_bytes += host.nbytes
        final = LiquidHeatState(
            dict(zip(m.names, map(float, host[: e.nf]))),
            dict(zip(m.wall_names, map(float, host[e.nf : e.size]))),
            dict(zip(m.names, map(float, host[e.size :]))),
            state.time_s,
            state.model_id,
        )
        m._check(final)
        return final

    def advance_records(
        self,
        state,
        records,
        boundary,
        ambient,
        duties,
        max_substeps,
        *,
        _keep_on_device=False,
        _execution="auto",
    ):
        before = self.transfer_stats()
        cursor = state if isinstance(state, _ResidentHeatState) else self.begin(state)
        if isinstance(state, _ResidentHeatState) and not _keep_on_device:
            raise ValueError("Private resident cursor requires a heat session")
        steps, audits = [], []
        failed = factors = 0
        for record in records:
            budget = max_substeps - len(steps) - failed
            if budget < 1:
                raise HydraulicConvergenceError("Thermal window work budget exhausted")
            calc = self.advance(
                cursor,
                record,
                boundary,
                ambient,
                duties,
                budget,
                _keep_on_device=True,
                _execution=_execution,
            )
            cursor = calc["state"]
            steps.extend(calc["steps"])
            failed += calc["failed_steps"]
            factors += calc["linear_factorizations"]
            audits.append(
                {
                    "time_s": record["time_s"],
                    "dt_s": record["dt_s"],
                    "combined_energy_residual_j": calc["combined_energy_residual_j"],
                }
            )
        final = cursor if _keep_on_device else self.export(cursor)
        return {
            "state": final if _keep_on_device else final.to_dict(),
            "steps": steps,
            "audits": audits,
            "failed_steps": failed,
            "linear_factorizations": factors,
            "transfer_stats": {k: v - before[k] for k, v in self.transfer_stats().items()},
        }

    def advance(
        self,
        state,
        record,
        boundary_temperatures_k=None,
        ambient_temperatures_k=None,
        heat_duties_w=None,
        max_substeps=100000,
        *,
        _keep_on_device=False,
        _hydraulic=None,
        _execution="auto",
    ):
        m = self.model
        e = self.kernel
        if _execution not in ("auto", "eager", "graph"):
            raise ValueError("thermal_execution must be auto, eager or graph")
        if _execution == "graph" and e.device.type != "cuda":
            raise ValueError("Thermal graph execution requires CUDA")
        graph_enabled = e.device.type == "cuda" and _execution != "eager"
        resident = isinstance(state, _ResidentHeatState)
        if resident:
            if not _keep_on_device or state.model_id != self.model_id:
                raise ValueError("Resident heat state/model mismatch")
            if (
                state.temperature.shape != (e.size,)
                or state.mass.shape != (e.nf,)
                or any(
                    v.dtype != e.dtype or v.device != e.device
                    for v in (state.temperature, state.mass)
                )
            ):
                raise ValueError("Resident heat state coordinates/device mismatch")
        else:
            m._check(state)
            if m.model_id != self.model_id:
                raise ValueError("Heat model changed after Torch kernel construction")
        duration = _number(record["dt_s"], "dt_s", positive=True)
        _number(record["time_s"], "time_s", nonnegative=True)
        if abs(record["time_s"] - state.time_s) > 32 * math.ulp(
            max(1.0, state.time_s, record["time_s"])
        ):
            raise ValueError("Hydraulic/thermal record time mismatch")
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("Invalid thermal work budget")
        boundary = {**m.boundary_temperatures_k, **dict(boundary_temperatures_k or {})}
        ambient = {n: w.ambient_temperature_k for n, w in m.walls.items()}
        ambient.update(ambient_temperatures_k or {})
        duties = dict(heat_duties_w or {})
        if (
            set(boundary) != set(m.boundary_temperatures_k)
            or set(ambient) != set(m.walls)
            or set(duties) - set(m.names)
        ):
            raise ValueError("Unknown prescribed boundary, wall or finite heat source")
        for t in boundary.values():
            m._temperature(t)
        for t in ambient.values():
            _number(t, "ambient_temperature_k", positive=True)
        for q in duties.values():
            _number(q, "heat_duty_w")
        physics = m.physics
        compact_input = (
            _hydraulic is not None and getattr(_hydraulic, "journal", "full") == "compact"
        )
        if not compact_input or "dissipation_heat_j" in record:
            heat = record["dissipation_heat_j"]
            if (
                set(heat) != {"friction", "pump", "shaft_drag", "motor"}
                or set(heat["friction"])
                != {b.name for b in physics.network.pipes + physics.network.valves}
                or any(set(heat[k]) != set(physics.names) for k in ("pump", "shaft_drag", "motor"))
            ):
                raise ValueError("Mechanical dissipation equipment mismatch")
        if resident:
            initial_t = initial_m = np.empty(0, dtype=np.float64)
        else:
            temperatures, walls, masses = (
                dict(state.temperatures_k),
                dict(state.wall_temperatures_k),
                dict(state.masses_kg),
            )
            initial_t = np.asarray(
                [temperatures[n] for n in m.names] + [walls[n] for n in m.wall_names]
            )
            initial_m = np.asarray([masses[n] for n in m.names])
        if _hydraulic is None:
            hydraulic_arrays = [
                np.asarray([record["final_pressures_pa"][n] for n in physics.free]),
                np.asarray(
                    [record["midpoint_branch_volume_rates_m3_s"][n] for n in e.branch_names]
                ),
                np.asarray(
                    [
                        heat["friction"][b.name]
                        for b in physics.network.pipes + physics.network.valves
                    ]
                ),
                *[
                    np.asarray([heat[k][n] for n in physics.names])
                    for k in ("pump", "shaft_drag", "motor")
                ],
                np.asarray(
                    [
                        record[k]
                        for k in (
                            "friction_heat_j",
                            "pump_heat_j",
                            "shaft_drag_heat_j",
                            "motor_heat_j",
                        )
                    ]
                ),
            ]
        else:
            from .torch_liquid_interval import DeviceHydraulicStep

            if (
                not isinstance(_hydraulic, DeviceHydraulicStep)
                or _hydraulic.model_id != self.physics_model_id
                or _hydraulic.journal not in ("full", "compact")
            ):
                raise ValueError("Hydraulic device record/model mismatch")
            data, ledger = _hydraulic.data, _hydraulic.ledger
            device_inputs = (
                data[0],
                data[4],
                ledger["friction_unit_heat_j"],
                ledger["pump_unit_heat_j"],
                ledger["drag_unit_heat_j"],
                ledger["motor_unit_heat_j"],
                ledger["boundary_mass_transfers_in_kg"],
            )
            shapes = (
                len(physics.free),
                len(e.branch_names),
                len(physics.network.pipes) + len(physics.network.valves),
                len(physics.names),
                len(physics.names),
                len(physics.names),
                len(physics._nodes),
            )
            if any(
                v.shape != (n,) or v.dtype != e.dtype or v.device != e.device
                for v, n in zip(device_inputs, shapes)
            ):
                raise ValueError("Hydraulic device record coordinates/device mismatch")
            hydraulic_arrays = [np.empty(0)] * 7
        arrays = [
            initial_t,
            initial_m,
            *hydraulic_arrays,
            np.asarray([boundary[n] for n in e.boundary_names]),
            np.asarray([ambient[n] for n in m.wall_names]),
            np.asarray([m._index[n] for n in duties]),
            np.asarray(list(duties.values())),
            (
                np.asarray([record["boundary_mass_transfers_in_kg"][n] for n in e.reservoir_names])
                if _hydraulic is None
                else np.empty(0)
            ),
            np.asarray([duration]),
        ]
        packet = self.upload(np.concatenate(arrays))
        if not resident:
            self.state_upload_batches += 1
            self.state_upload_bytes += initial_t.nbytes + initial_m.nbytes
        chunks = packet.split([len(a) for a in arrays])
        (
            t,
            mstart,
            pfinal,
            flows,
            friction,
            pump,
            shaft,
            motor,
            expected,
            boundary_t,
            ambient_t,
            duty_index,
            duty_values,
            reservoir_transfer,
            physical_dt,
        ) = chunks
        if _hydraulic is not None:
            pfinal, flows, friction, pump, shaft, motor, transfers = device_inputs
            expected = torch.stack(
                [
                    ledger[k]
                    for k in ("friction_heat_j", "pump_heat_j", "shaft_drag_heat_j", "motor_heat_j")
                ]
            )
            reservoir_transfer = transfers.index_select(0, e.reservoir_node)
            self.device_hydraulic_records += 1
            self.device_hydraulic_bytes += sum(v.numel() * v.element_size() for v in device_inputs)
        duty = torch.zeros(e.size, dtype=e.dtype, device=e.device).scatter(
            0, duty_index.to(torch.long), duty_values
        )
        if resident:
            t, mstart = state.temperature, state.mass
        physical_dt = physical_dt[0]
        mend = e.masses_next(mstart, pfinal, flows, physical_dt, reservoir_transfer)
        groups = (friction, pump, shaft, motor)
        sums = torch.stack([g.sum() for g in groups])
        valid_heat = (torch.abs(sums - expected) <= 1e-7 + 1e-10 * expected.abs()).all()
        all_heat = torch.cat(groups)
        valid_heat = valid_heat & torch.isfinite(all_heat).all() & (all_heat >= -1e-7).all()
        heat_vec = e.heat_vector(*groups)
        forcing, absorbed, diagonal, dense = e.compile_operator(
            flows, heat_vec, physical_dt, boundary_t, ambient_t, duty
        )
        cmin = torch.cat((torch.minimum(mstart, mend) * m.specific_heat_j_kg_k, e.wall_capacity))
        limit_tensor = torch.where(
            diagonal < 0, m.positivity_courant * cmin / (-diagonal).clamp_min(1e-300), math.inf
        ).min()
        initial = self.download(
            torch.stack(
                (
                    valid_heat.to(e.dtype),
                    ((mend > 0) & torch.isfinite(mend)).all().to(e.dtype),
                    limit_tensor,
                    (mstart == mend).all().to(e.dtype),
                )
            )
        )
        if not initial[0]:
            raise ConservationError("Per-equipment heat does not match mechanical ledger")
        if not initial[1]:
            raise EquipmentOperatingError("Finite thermal reservoir depleted")
        limit = min(m.maximum_dt_s, float(initial[2]))
        rejected = (expected.sum() - duration * absorbed.sum()) / duration
        factors_cache = {}
        factor_count = 0
        constant_mass = bool(initial[3])

        def linear(t0, m0, m1, h):
            nonlocal factor_count
            ht = physical_dt * (h / duration)
            factors = None
            if dense is not None:
                if constant_mass:
                    if h not in factors_cache:
                        if len(factors_cache) >= 4:
                            factors_cache.pop(next(iter(factors_cache)))
                        factors_cache[h] = e.factor(m1, ht, dense)
                        factor_count += 1
                    factors = factors_cache[h]
                else:
                    factor_count += 1
            linear_step = self.linear_graphs.linear if graph_enabled else e.linear_step
            return linear_step(
                t0, m0, m1, ht, forcing, diagonal, dense, boundary_t, ambient_t, factors
            )

        elapsed = 0.0
        hnext = min(duration, m.maximum_dt_s)
        retries = 0
        gpu_records = []
        times = []
        fields = []
        field_chunks = []
        field_bytes = 0
        while elapsed < duration:
            if len(times) + retries >= max_substeps:
                raise HydraulicConvergenceError("Thermal work budget exhausted")
            remaining = duration - elapsed
            h = min(hnext, limit, remaining)
            if remaining <= h + 8 * math.ulp(duration):
                h = remaining
            if h < m.minimum_dt_s and remaining > m.minimum_dt_s:
                raise HydraulicConvergenceError("Thermal positivity limit below minimum step")
            if elapsed + h <= elapsed:
                raise ValueError("Thermal relative substep cannot advance time")
            m0 = mstart + (mend - mstart) * (elapsed / duration)
            m1 = mstart + (mend - mstart) * ((elapsed + h) / duration)
            rough = linear(t, m0, m1, h)
            if m.temporal_error_control:
                mm = (m0 + m1) / 2
                a = linear(t, m0, mm, h / 2)
                b = linear(a[0], mm, m1, h / 2)
                scale = m.temperature_tolerance_k + m.relative_tolerance * torch.maximum(
                    t.abs(), b[0].abs()
                )
                error = ((b[0] - rough[0]).abs() / (3 * scale)).max()
                accepted = [(a, h / 2), (b, h / 2)]
            else:
                error = torch.zeros((), dtype=e.dtype, device=e.device)
                accepted = [(rough, h)]
            change = (accepted[-1][0][0] - t).abs().max()
            status = self.download(torch.stack((error, change)))
            guard = min(
                [m.maximum_temperature_change_k]
                + [w.max_temperature_change_k for w in m.walls.values()]
            )
            if not math.isfinite(float(status[0])) or status[0] > 1 or status[1] > guard:
                retries += 1
                if h / 2 < m.minimum_dt_s:
                    raise HydraulicConvergenceError("Thermal accuracy exhausted minimum step")
                hnext = h / 2
                continue
            following = duration if h == remaining else min(duration, math.fsum((elapsed, h)))
            for item, hd in accepted:
                if len(times) + retries >= max_substeps:
                    raise HydraulicConvergenceError("Thermal work budget exhausted")
                t1, delta, adv, env, local, _, domain = item
                generated = hd * absorbed.sum()
                external = hd * duty.sum()
                residual = delta.sum() - adv - env - generated - external
                scale = (
                    delta.abs().sum() + adv.abs() + env.abs() + generated.abs() + external.abs()
                ).clamp_min(1.0)
                valid = (
                    domain
                    & torch.isfinite(residual)
                    & (residual.abs() <= 1e-7 + 1e-10 * scale)
                    & (local <= 1e-7 + 1e-10 * scale)
                )
                tm = (t + t1) / 2
                entropy = (
                    hd
                    * (
                        e.ua
                        * (tm[e.wall_i] - tm[e.wall_j]).square()
                        / (tm[e.wall_i] * tm[e.wall_j])
                        + e.ua_ambient
                        * (tm[e.wall_j] - ambient_t).square()
                        / (tm[e.wall_j] * ambient_t)
                    ).sum()
                )
                gpu_records.append(
                    torch.stack(
                        (
                            delta[: e.nf].sum(),
                            delta[e.nf :].sum(),
                            adv,
                            env,
                            generated,
                            hd * rejected,
                            external,
                            residual,
                            local,
                            error,
                            entropy,
                            valid.to(e.dtype),
                        )
                    )
                )
                times.append((state.time_s + elapsed, hd))
                if m.store_step_temperatures:
                    fields.append(t1)
                    field_bytes += t1.numel() * t1.element_size()
                    if field_bytes >= self.history_buffer_bytes:
                        field_chunks.append(self.download(torch.stack(fields)))
                        self.history_download_bytes += field_bytes
                        self.history_flushes += 1
                        fields.clear()
                        field_bytes = 0
                elapsed = math.fsum((elapsed, hd))
                t = t1
            elapsed = following
            ratio = float(status[0])
            hnext = min(
                m.maximum_dt_s,
                h * min(1.5, max(0.8, 0.9 * ratio ** (-1 / 3))) if ratio > 0 else h * 1.5,
            )
        # One packed export of the complete accepted audit log and final field.
        state_fields = () if _keep_on_device else (t, mend)
        packed = torch.cat((*state_fields, torch.stack(gpu_records).flatten(), *fields))
        host = self.download(packed)
        self.history_download_bytes += field_bytes
        offset = 0
        if not _keep_on_device:
            tend = host[: e.size]
            mass = host[e.size : e.size + e.nf]
            offset = e.size + e.nf
            self.state_download_batches += 1
            self.state_download_bytes += offset * host.itemsize
        audit = host[offset : offset + len(times) * 12].reshape(len(times), 12)
        offset += len(times) * 12
        if not np.all(audit[:, -1]):
            raise ConservationError("Liquid thermal energy audit failed")
        keys = (
            "fluid_energy_change_j",
            "wall_energy_change_j",
            "external_advective_heat_in_j",
            "ambient_heat_in_j",
            "dissipation_to_inventory_j",
            "equipment_heat_out_j",
            "external_heat_duty_j",
            "energy_residual_j",
            "max_local_energy_residual_j",
            "temporal_error_ratio",
            "passive_exchange_entropy_j_k",
        )
        if fields:
            field_chunks.append(host[offset:].reshape(len(fields), e.size))
        history = iter(vector for chunk in field_chunks for vector in chunk)
        steps = []
        for (time, hd), row in zip(times, audit):
            item = {"time_s": time, "dt_s": hd, **dict(zip(keys, map(float, row[:-1])))}
            if m.store_step_temperatures:
                vector = next(history)
                item["final_temperatures_k"] = dict(zip(m.names, map(float, vector[: e.nf])))
                item["final_wall_temperatures_k"] = dict(
                    zip(m.wall_names, map(float, vector[e.nf :]))
                )
            steps.append(item)
        if _keep_on_device:
            final = _ResidentHeatState(t, mend, record["time_s"] + duration, self.model_id)
        else:
            final = LiquidHeatState(
                dict(zip(m.names, map(float, tend[: e.nf]))),
                dict(zip(m.wall_names, map(float, tend[e.nf :]))),
                dict(zip(m.names, map(float, mass))),
                record["time_s"] + duration,
                self.model_id,
            )
            m._check(final)
        thermal_delta = math.fsum(
            s["fluid_energy_change_j"] + s["wall_energy_change_j"] for s in steps
        )
        mech = (
            record["shaft_kinetic_energy_change_j"]
            + record["pipe_kinetic_energy_change_j"]
            + record["acoustic_energy_change_j"]
        )
        total = (
            record["electrical_energy_j"]
            + record["boundary_work_in_j"]
            + math.fsum(
                s["external_advective_heat_in_j"]
                + s["ambient_heat_in_j"]
                + s["external_heat_duty_j"]
                - s["equipment_heat_out_j"]
                for s in steps
            )
            - mech
            - thermal_delta
        )
        if abs(total) > 1e-5 + 1e-9 * max(1, abs(thermal_delta), abs(mech)):
            raise ConservationError("Combined liquid mechanical/caloric energy audit failed")
        return {
            "state": final if _keep_on_device else final.to_dict(),
            "steps": steps,
            "failed_steps": retries,
            "combined_energy_residual_j": total,
            "linear_factorizations": factor_count,
        }
