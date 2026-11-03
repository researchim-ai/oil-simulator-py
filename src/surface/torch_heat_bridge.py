"""Explicit serialization/adaptivity boundary for the resident Torch heat kernel."""

import math

import numpy as np
import torch

from .liquid import ConservationError, EquipmentOperatingError, HydraulicConvergenceError, _number
from .liquid_heat import LiquidHeatState
from .torch_heat import TorchHeatEquations


class TorchHeatBridge:
    def __init__(self, model, device):
        self.model = model
        self.kernel = TorchHeatEquations(model, device)
        self.upload_batches = 0
        self.download_batches = 0

    def download(self, value):
        self.download_batches += 1
        return value.detach().cpu().numpy()

    def advance(
        self,
        state,
        record,
        boundary_temperatures_k=None,
        ambient_temperatures_k=None,
        heat_duties_w=None,
        max_substeps=100000,
    ):
        m = self.model
        e = self.kernel
        m._check(state)
        duration = _number(record["dt_s"], "dt_s", positive=True)
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
        heat = record["dissipation_heat_j"]
        physics = m.physics
        if (
            set(heat) != {"friction", "pump", "shaft_drag", "motor"}
            or set(heat["friction"])
            != {b.name for b in physics.network.pipes + physics.network.valves}
            or any(set(heat[k]) != set(physics.names) for k in ("pump", "shaft_drag", "motor"))
        ):
            raise ValueError("Mechanical dissipation equipment mismatch")
        arrays = [
            np.asarray(
                [dict(state.temperatures_k)[n] for n in m.names]
                + [dict(state.wall_temperatures_k)[n] for n in m.wall_names]
            ),
            np.asarray([dict(state.masses_kg)[n] for n in m.names]),
            np.asarray([record["final_pressures_pa"][n] for n in physics.free]),
            np.asarray([record["midpoint_branch_volume_rates_m3_s"][n] for n in e.branch_names]),
            np.asarray(
                [heat["friction"][b.name] for b in physics.network.pipes + physics.network.valves]
            ),
            *[
                np.asarray([heat[k][n] for n in physics.names])
                for k in ("pump", "shaft_drag", "motor")
            ],
            np.asarray(
                [
                    record[k]
                    for k in ("friction_heat_j", "pump_heat_j", "shaft_drag_heat_j", "motor_heat_j")
                ]
            ),
            np.asarray([boundary[n] for n in e.boundary_names]),
            np.asarray([ambient[n] for n in m.wall_names]),
            np.asarray([m._index[n] for n in duties]),
            np.asarray(list(duties.values())),
            np.asarray([record["boundary_mass_transfers_in_kg"][n] for n in e.reservoir_names]),
            np.asarray([duration]),
        ]
        packet = e.tensor(np.concatenate(arrays))
        self.upload_batches += 1
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
        duty = torch.zeros(e.size, dtype=e.dtype, device=e.device).scatter(
            0, duty_index.to(torch.long), duty_values
        )
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
            return e.linear_step(
                t0, m0, m1, ht, forcing, diagonal, dense, boundary_t, ambient_t, factors
            )

        elapsed = 0.0
        hnext = min(duration, m.maximum_dt_s)
        retries = 0
        gpu_records = []
        times = []
        fields = []
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
                elapsed = math.fsum((elapsed, hd))
                t = t1
            elapsed = following
            ratio = float(status[0])
            hnext = min(
                m.maximum_dt_s,
                h * min(1.5, max(0.8, 0.9 * ratio ** (-1 / 3))) if ratio > 0 else h * 1.5,
            )
        # One packed export of the complete accepted audit log and final field.
        packed = torch.cat((t, mend, torch.stack(gpu_records).flatten(), *[f for f in fields]))
        host = self.download(packed)
        offset = 0
        tend = host[: e.size]
        offset += e.size
        mass = host[offset : offset + e.nf]
        offset += e.nf
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
        steps = []
        for i, ((time, hd), row) in enumerate(zip(times, audit)):
            item = {"time_s": time, "dt_s": hd, **dict(zip(keys, map(float, row[:-1])))}
            if m.store_step_temperatures:
                vector = host[offset + i * e.size : offset + (i + 1) * e.size]
                item["final_temperatures_k"] = dict(zip(m.names, map(float, vector[: e.nf])))
                item["final_wall_temperatures_k"] = dict(
                    zip(m.wall_names, map(float, vector[e.nf :]))
                )
            steps.append(item)
        final = LiquidHeatState(
            dict(zip(m.names, map(float, tend[: e.nf]))),
            dict(zip(m.wall_names, map(float, tend[e.nf :]))),
            dict(zip(m.names, map(float, mass))),
            record["time_s"] + duration,
            m.model_id,
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
            "state": final.to_dict(),
            "steps": steps,
            "failed_steps": retries,
            "combined_energy_residual_j": total,
            "linear_factorizations": factor_count,
        }
