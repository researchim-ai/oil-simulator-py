"""Shared-stage conservative heat/stock coupling for liquid Gauss4.

Constant hydraulic properties: caloric transport does not feed back into the
liquid EOS. Thermal error, inventories, walls and ledgers nevertheless take
part in the same trial acceptance. No accepted midpoint surrogate is used.
"""

import math

import torch

from .liquid import _number
from .liquid_collocation import LiquidCollocation
from .liquid_heat import LiquidHeatNetwork, LiquidHeatState
from .liquid_transient import LiquidTransientState
from .torch_heat import TorchHeatEquations
from .torch_krylov import gmres_device
from .torch_linear import TorchCOO


class _HeatCoupling:
    def __init__(self, model, state, boundary, ambient, duties, dense_limit, mass_tolerance):
        self.model = model
        self.mass_tolerance = mass_tolerance
        self.kernel = k = TorchHeatEquations(model, model.device, dense_limit=0)
        self.maximum_dt_s, self.minimum_dt_s = model.maximum_dt_s, model.minimum_dt_s
        self.temperature = k.tensor(
            [dict(state.temperatures_k)[n] for n in model.names]
            + [dict(state.wall_temperatures_k)[n] for n in model.wall_names]
        )
        self.mass = k.tensor([dict(state.masses_kg)[n] for n in model.names])
        self.boundary = k.tensor([boundary[n] for n in k.boundary_names])
        self.ambient = k.tensor([ambient[n] for n in model.wall_names])
        self.duties = k.tensor([duties.get(n, 0.0) for n in model.names] + [0.0] * k.nw)
        rows, cols = k.operator.rows.cpu().tolist(), k.operator.cols.cpu().tolist()
        br, bc = [], []
        for i in range(2):
            for j in range(2):
                br.extend(i * k.size + r for r in rows)
                bc.extend(j * k.size + c for c in cols)
        br.extend(range(2 * k.size))
        bc.extend(range(2 * k.size))
        self.matrix = TorchCOO(br, bc, k.tensor([0.0] * len(br)), 2 * k.size, k.device)
        self.identity = (
            torch.eye(2 * k.size, dtype=k.dtype, device=k.device)
            if 2 * k.size <= dense_limit
            else None
        )
        self.history = []
        self.linear_solves = 0

    def _capacity(self, mass):
        k = self.kernel
        return torch.cat((k.cp * mass, k.wall_capacity))

    def _mass(self, pressure, reservoir_change):
        k = self.kernel
        return k.masses_next(
            self.mass, pressure, k.tensor([]), 0.0, boundary_transfers=-reservoir_change
        )

    def _point(self, hydraulic, y, commands, fixed, openings):
        e, k = hydraulic.e, self.kernel
        rate, _, data = e.instantaneous(y, commands, fixed, openings)
        _, _, w, p, q, _, torque, power, friction, _, _ = data
        work = torque * w
        electric = torch.where(work >= 0, work / e.motor_eta, work * e.regen_eta)
        dp = p[e.start] - p[e.end]
        pump = power + dp[e.npipe : e.npipe + e.npump] * q[e.npipe : e.npipe + e.npump]
        heat = k.heat_vector(friction, pump, e.drag * w.square(), electric - work)
        # This routine assembles an instantaneous linear operator. A one-second
        # units conversion is exact here; no time-discrete state is inverted.
        forcing, absorbed, diagonal, _ = k.compile_operator(
            q, heat, 1.0, self.boundary, self.ambient, self.duties
        )
        return {
            "rate": rate,
            "pressure": data[0],
            "operator": k.operator.values,
            "forcing": forcing,
            "absorbed": absorbed.sum(),
            "diagonal": diagonal,
            "incoming": k.incoming.mv(self.boundary - k.ref).sum(),
            "outgoing": k.outgoing.values,
            "dm": -e.rho * k.incidence.mv(q)[k.reservoir_node],
        }

    def _flux(self, point, theta):
        k = self.kernel
        return k.operator.mv(theta, point["operator"]) + point["forcing"]

    def trial(self, hydraulic, old, new, h, record, commands, fixed, openings):
        k, m = self.kernel, self.model
        self.linear_solves += 1
        stages = record[hydraulic.e.size + 12 :].reshape(2, hydraulic.e.size)
        points = [self._point(hydraulic, y, commands, fixed, openings) for y in stages]
        initial = self._point(hydraulic, old, commands, fixed, openings)
        dm = torch.stack([p["dm"] for p in points])
        stage_mass = [
            self._mass(p["pressure"], h * (hydraulic.a @ dm)[i]) for i, p in enumerate(points)
        ]
        mass = self._mass(new[: hydraulic.e.nf] * hydraulic.e.ps, h * 0.5 * dm.sum(0))
        c0, cs, c1 = (
            self._capacity(self.mass),
            torch.stack([self._capacity(v) for v in stage_mass]),
            self._capacity(mass),
        )
        theta = self.temperature - k.ref
        base_flux = torch.stack([self._flux(p, theta) for p in points])
        rhs = (h * (hydraulic.a @ base_flux) - (cs - c0) * theta) / cs.clamp_min(1e-30)
        chunks = [
            -h * hydraulic.a[i, j] * points[j]["operator"] / cs[i, k.operator.rows].clamp_min(1e-30)
            for i in range(2)
            for j in range(2)
        ]
        raw = torch.cat((*chunks, torch.ones(2 * k.size, dtype=k.dtype, device=k.device)))
        self.matrix.values = raw.index_select(0, self.matrix.order)
        if self.identity is not None:
            delta, info = torch.linalg.solve_ex(
                self.matrix.mv(self.identity).T, rhs.flatten(), check_errors=False
            )
            linear_valid = info == 0
        else:
            delta = gmres_device(self.matrix.mv, rhs.flatten(), self.matrix.diagonal())
            linear_valid = delta.new_tensor(True, dtype=torch.bool)
        linear_residual = (self.matrix.mv(delta) - rhs.flatten()).abs().max()
        delta = delta.reshape(2, k.size)
        temperatures = self.temperature + delta
        flux = torch.stack(
            [base_flux[i] + k.operator.mv(delta[i], p["operator"]) for i, p in enumerate(points)]
        )
        energy = h * 0.5 * flux.sum(0)
        final_delta = (energy - (c1 - c0) * theta) / c1.clamp_min(1e-30)
        temperature = self.temperature + final_delta
        low, auxiliary, _ = hydraulic._embedded(old, h, commands, fixed, openings, initial["rate"])
        middle, last = [self._point(hydraulic, y, commands, fixed, openings) for y in auxiliary]
        r0 = self._flux(initial, theta)
        middle_mass = self._mass(middle["pressure"], 0.5 * h * initial["dm"])
        middle_capacity = self._capacity(middle_mass)
        middle_delta = (0.5 * h * r0 - (middle_capacity - c0) * theta) / middle_capacity.clamp_min(
            1e-30
        )
        middle_flux = self._flux(middle, theta + middle_delta)
        last_mass = self._mass(last["pressure"], h * (-initial["dm"] + 2 * middle["dm"]))
        last_capacity = self._capacity(last_mass)
        last_delta = (
            h * (-r0 + 2 * middle_flux) - (last_capacity - c0) * theta
        ) / last_capacity.clamp_min(1e-30)
        last_flux = self._flux(last, theta + last_delta)
        low_mass = self._mass(
            low[: hydraulic.e.nf] * hydraulic.e.ps,
            h / 6 * (initial["dm"] + 4 * middle["dm"] + last["dm"]),
        )
        low_capacity = self._capacity(low_mass)
        low_temperature = self.temperature + (
            h / 6 * (r0 + 4 * middle_flux + last_flux) - (low_capacity - c0) * theta
        ) / low_capacity.clamp_min(1e-30)
        scale = m.temperature_tolerance_k + m.relative_tolerance * torch.maximum(
            self.temperature.abs(), temperature.abs()
        )
        temperature_error = ((temperature - low_temperature).abs() / scale).max()
        mass_scale = self.mass_tolerance + m.relative_tolerance * torch.maximum(
            self.mass.abs(), mass.abs()
        )
        mass_error = ((mass - low_mass).abs() / mass_scale).max()
        error = torch.maximum(temperature_error, mass_error)
        advective = (
            h
            * 0.5
            * sum(
                p["incoming"] - k.outgoing.mv(temperatures[i] - k.ref, p["outgoing"]).sum()
                for i, p in enumerate(points)
            )
        )
        ambient = (
            h
            * 0.5
            * sum(
                (k.ua_ambient * (self.ambient - temperatures[i, k.wall_j])).sum() for i in range(2)
            )
        )
        absorbed = h * 0.5 * sum(p["absorbed"] for p in points)
        duty = h * self.duties.sum()
        actual_delta = (c1 - c0) * theta + c1 * (temperature - self.temperature)
        local_error = (actual_delta - energy).abs().max()
        thermal_error = actual_delta.sum() - advective - ambient - absorbed - duty
        mechanical = record[hydraulic.e.size : hydraulic.e.size + 6]
        mechanical_change = record[hydraulic.e.size + 6]
        escaped_heat = mechanical[2:].sum() - absorbed
        total_change = mechanical_change + actual_delta.sum()
        total_external = mechanical[:2].sum() + advective + ambient + duty - escaped_heat
        combined_error = total_change - total_external
        audit_scale = (
            actual_delta.abs().sum() + advective.abs() + ambient.abs() + absorbed.abs() + duty.abs()
        )
        limit = 1e-7 + 1e-10 * audit_scale
        all_mass = torch.stack([self.mass, mass, low_mass, middle_mass, last_mass, *stage_mass])
        all_temp = torch.cat(
            (
                temperature[None],
                temperatures,
                (self.temperature + middle_delta)[None],
                (self.temperature + last_delta)[None],
            )
        )
        valid = (
            linear_valid
            & torch.isfinite(all_temp).all()
            & torch.isfinite(error)
            & (all_mass > 0).all()
            & (linear_residual <= 1e-10)
            & (all_temp[:, : k.nf] >= m.min_temperature_k).all()
            & (all_temp[:, : k.nf] <= m.max_temperature_k).all()
            & (all_temp[:, k.nf :] >= k.wall_min).all()
            & (all_temp[:, k.nf :] <= k.wall_max).all()
            & (local_error <= limit)
            & (thermal_error.abs() <= limit)
            & (
                combined_error.abs()
                <= limit + 1e-6 + 1e-7 * (mechanical.abs().sum() + mechanical_change.abs())
            )
            & (escaped_heat >= -1e-7)
        )
        capacities = torch.minimum(c0, torch.minimum(c1, cs.min(0).values))
        slope = (dm[1] - dm[0]) / (hydraulic.c[1] - hydraulic.c[0])
        intercept = (dm[0] * hydraulic.c[1] - dm[1] * hydraulic.c[0]) / (
            hydraulic.c[1] - hydraulic.c[0]
        )
        location = (-intercept / torch.where(slope != 0, slope, torch.ones_like(slope))).clamp(0, 1)
        reservoir_extrema = self.mass[k.reservoir_index] + h * (
            intercept * location + 0.5 * slope * location.square()
        )
        valid = valid & (reservoir_extrema > 0).all()
        positivity = torch.stack(
            [h * (-p["diagonal"]) / capacities.clamp_min(1e-30) for p in points]
        ).max()
        change = (all_temp - self.temperature).abs()
        allowed = k.tensor(
            [m.maximum_temperature_change_k] * k.nf
            + [
                min(m.maximum_temperature_change_k, m.walls[n].max_temperature_change_k)
                for n in m.wall_names
            ]
        )
        valid = (
            valid & (positivity <= m.positivity_courant * (1 + 1e-12)) & (change <= allowed).all()
        )
        # Upper bounds alone can hide a passive overshoot; energy/domain/error
        # checks are accompanied by the inherited thermal Courant restriction.
        metrics = torch.stack((error, valid.to(k.dtype))).detach().cpu().tolist()
        totals = torch.stack(
            (
                actual_delta.sum(),
                advective,
                ambient,
                absorbed,
                duty,
                thermal_error,
                combined_error,
                local_error,
                linear_residual,
                positivity,
                escaped_heat,
                total_change,
                total_external,
                temperature_error,
                mass_error,
            )
        )
        candidate = (
            temperature,
            mass,
            torch.cat(
                (
                    temperature,
                    mass,
                    totals,
                    temperatures.flatten(),
                    torch.stack(stage_mass).flatten(),
                )
            ),
        )
        return candidate, metrics[0], bool(metrics[1])

    def commit(self, candidate, start, end, dt):
        self.temperature, self.mass, record = candidate
        self.history.append((start, end, dt, record))

    def export(self, physical):
        k, m = self.kernel, self.model
        rows = torch.stack([v[3] for v in self.history]).detach().cpu().tolist()
        last = rows[-1]
        final = LiquidHeatState(
            dict(zip(m.names, last[: k.nf])),
            dict(zip(m.wall_names, last[k.nf : k.size])),
            dict(zip(m.names, last[k.size : k.size + k.nf])),
            physical.time_s,
            m.model_id,
        )
        m._check(final, physical)
        steps = []
        for (start, end, dt, _), row in zip(self.history, rows):
            offset = k.size + k.nf
            totals = dict(
                zip(
                    (
                        "thermal_energy_change_j",
                        "advective_energy_in_j",
                        "ambient_heat_in_j",
                        "absorbed_dissipation_j",
                        "external_heat_in_j",
                        "thermal_energy_residual_j",
                        "combined_energy_residual_j",
                        "max_local_energy_residual_j",
                        "linear_residual_k",
                        "thermal_courant",
                        "escaped_dissipation_j",
                        "total_energy_change_j",
                        "total_external_energy_in_j",
                        "temperature_error_ratio",
                        "mass_error_ratio",
                    ),
                    row[offset : offset + 15],
                )
            )
            stages = row[offset + 15 : offset + 15 + 2 * k.size]
            masses = row[offset + 15 + 2 * k.size :]
            steps.append(
                {
                    "time_s": start,
                    "end_time_s": end,
                    "dt_s": dt,
                    "temperatures_k": dict(zip(m.names, row[: k.nf])),
                    "wall_temperatures_k": dict(zip(m.wall_names, row[k.nf : k.size])),
                    "masses_kg": dict(zip(m.names, row[k.size : k.size + k.nf])),
                    "stage_temperatures_k": [stages[: k.size], stages[k.size :]],
                    "stage_masses_kg": [masses[: k.nf], masses[k.nf :]],
                    **totals,
                }
            )
        return final, steps


class ThermalLiquidCollocation:
    def __init__(
        self, heat, *, thermal_dense_limit=128, mass_tolerance_kg=1e-8, **hydraulic_tolerances
    ):
        if not isinstance(heat, LiquidHeatNetwork) or heat.backend != "torch":
            raise TypeError("Expected Torch LiquidHeatNetwork")
        if type(thermal_dense_limit) is not int or thermal_dense_limit < 0:
            raise ValueError("Nonnegative integer thermal dense limit required")
        if heat.device != heat.physics.device:
            raise ValueError("Coupled heat and hydraulics must use the same device")
        self.heat, self.model_id, self.dense_limit = heat, heat.model_id, thermal_dense_limit
        self.mass_tolerance = _number(mass_tolerance_kg, "mass_tolerance_kg", positive=True)
        self.hydraulics = LiquidCollocation(heat.physics, **hydraulic_tolerances)

    @torch.no_grad()
    def advance(
        self,
        physical,
        state,
        duration_s,
        *,
        boundary_temperatures_k=None,
        ambient_temperatures_k=None,
        heat_duties_w=None,
        **hydraulic_controls,
    ):
        m = self.heat
        m._check(state, physical)
        if m.model_id != self.model_id:
            raise ValueError("Coupled heat model changed after construction")
        boundary = {**m.boundary_temperatures_k, **dict(boundary_temperatures_k or {})}
        ambient = {n: w.ambient_temperature_k for n, w in m.walls.items()}
        ambient.update(ambient_temperatures_k or {})
        duties = dict(heat_duties_w or {})
        if (
            set(boundary) != set(m.boundary_temperatures_k)
            or set(ambient) != set(m.walls)
            or set(duties) - set(m.names)
        ):
            raise ValueError("Unknown coupled thermal control")
        for t in boundary.values():
            m._temperature(t)
        for t in ambient.values():
            _number(t, "ambient temperature", positive=True)
        for v in duties.values():
            _number(v, "heat duty")
        coupling = _HeatCoupling(
            m, state, boundary, ambient, duties, self.dense_limit, self.mass_tolerance
        )
        report = self.hydraulics.advance(
            physical, duration_s, _coupling=coupling, **hydraulic_controls
        )
        final_physical = LiquidTransientState(**report["state"])
        final_heat, steps = coupling.export(final_physical)
        return {
            "integrator": "torch_liquid_thermal_gauss4_v1",
            "state": {"hydraulics": report["state"], "heat": final_heat.to_dict()},
            "hydraulics": report,
            "thermal_steps": steps,
            "heat_model": m.to_dict(),
            "initial_heat": state.to_dict(),
            "thermal_linear_solves": coupling.linear_solves,
            "coupling_policy": {
                "shared_trial_acceptance": True,
                "temperature_error_control": True,
                "hydraulic_properties": "constant",
                "thermal_dense_limit": self.dense_limit,
                "mass_tolerance_kg": self.mass_tolerance,
            },
            "thermal_controls": {
                "boundary_temperatures_k": boundary,
                "ambient_temperatures_k": ambient,
                "heat_duties_w": duties,
            },
            "combined_energy_residual_j": math.fsum(s["combined_energy_residual_j"] for s in steps),
        }
