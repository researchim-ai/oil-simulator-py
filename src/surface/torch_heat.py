"""Device-resident conservative caloric liquid kernels (PyTorch float64).

A static signed-flow stencil replaces host sparse assembly. Small systems use
Torch LU; large ones use the positivity-bounded Jacobi contraction, with an
explicit residual audit. Neither path converts a matrix to CPU/SciPy.
"""

from dataclasses import dataclass

import torch

from .torch_linear import TorchCOO, checked_device


@dataclass(frozen=True)
class TorchHeatState:
    temperature: torch.Tensor
    mass: torch.Tensor


class TorchHeatEquations:
    def __init__(self, model, device="cpu", iterations=32, dense_limit=64):
        self.model = model
        self.device = checked_device(device)
        self.dtype = torch.float64
        self.nf = len(model.names)
        self.nw = len(model.wall_names)
        self.size = self.nf + self.nw
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("Positive thermal iteration budget required")
        self.iterations = iterations
        self.dense_limit = dense_limit
        self.cp = model.specific_heat_j_kg_k
        self.ref = model.reference_temperature_k
        self.boundary_names = tuple(sorted(model.boundary_temperatures_k))
        dynamic = {n: i for i, n in enumerate(model.names)}
        external = {n: i for i, n in enumerate(self.boundary_names)}
        branches = model.physics._branches
        self.branch_names = tuple(b.name for b in branches)
        rows = []
        cols = []
        term_b = []
        term_sign = []
        term_factor = []
        forcing_rows = []
        forcing_cols = []
        forcing_b = []
        forcing_sign = []
        outgoing_rows = []
        outgoing_cols = []
        outgoing_b = []
        outgoing_sign = []
        deposit_rows = []
        deposit_cols = []
        deposit_sign = []
        deposit_weight = []
        for e, b in enumerate(branches):
            path = [b.start, *model._paths.get(b.name, ()), b.end]
            for start, end in zip(path, path[1:]):
                for source, dest, sign in ((start, end, 1), (end, start, -1)):
                    if source in dynamic:
                        rows.append(dynamic[source])
                        cols.append(dynamic[source])
                        term_b.append(e)
                        term_sign.append(sign)
                        term_factor.append(-1.0)
                    if dest in dynamic and source in dynamic:
                        rows.append(dynamic[dest])
                        cols.append(dynamic[source])
                        term_b.append(e)
                        term_sign.append(sign)
                        term_factor.append(1.0)
                    elif dest in dynamic:
                        forcing_rows.append(dynamic[dest])
                        forcing_cols.append(external[source])
                        forcing_b.append(e)
                        forcing_sign.append(sign)
                    elif source in dynamic:
                        outgoing_rows.append(dynamic[source])
                        outgoing_cols.append(dynamic[source])
                        outgoing_b.append(e)
                        outgoing_sign.append(sign)
            cells = model._paths.get(b.name)
            if cells:
                for n in cells:
                    deposit_rows.append(dynamic[n])
                    deposit_cols.append(e)
                    deposit_sign.append(0)
                    deposit_weight.append(1 / len(cells))
            else:
                for name, sign in ((b.end, 1), (b.start, -1)):
                    if name in dynamic:
                        deposit_rows.append(dynamic[name])
                        deposit_cols.append(e)
                        deposit_sign.append(sign)
                        deposit_weight.append(1.0)
        self.term_b = self.integer(term_b)
        self.term_sign = self.tensor(term_sign)
        self.term_factor = self.tensor(term_factor)
        self.force_b = self.integer(forcing_b)
        self.force_sign = self.tensor(forcing_sign)
        self.out_b = self.integer(outgoing_b)
        self.out_sign = self.tensor(outgoing_sign)
        self.deposit_b = self.integer(deposit_cols)
        self.deposit_sign = self.tensor(deposit_sign)
        self.deposit_weight = self.tensor(deposit_weight)
        self.wall_i = self.integer([dynamic[n] for n in model.wall_names])
        self.wall_j = self.integer(list(range(self.nf, self.size)))
        self.wall_capacity = self.tensor(
            [model.walls[n].heat_capacity_j_k for n in model.wall_names]
        )
        self.ua = self.tensor([model.walls[n].fluid_ua_w_k for n in model.wall_names])
        self.ua_ambient = self.tensor([model.walls[n].ambient_ua_w_k for n in model.wall_names])
        static = []
        for name in model.wall_names:
            i, j = dynamic[name], model._wall_index[name]
            w = model.walls[name]
            rows.extend((i, i, j, j))
            cols.extend((i, j, i, j))
            static.extend(
                (
                    -w.fluid_ua_w_k,
                    w.fluid_ua_w_k,
                    w.fluid_ua_w_k,
                    -w.fluid_ua_w_k - w.ambient_ua_w_k,
                )
            )
        self.static = self.tensor(static)
        self.operator = TorchCOO(
            rows,
            cols,
            torch.zeros(len(rows), dtype=self.dtype, device=self.device),
            self.size,
            self.device,
        )
        self.incoming = TorchCOO(
            forcing_rows,
            forcing_cols,
            torch.zeros(len(forcing_rows), dtype=self.dtype, device=self.device),
            self.size,
            self.device,
        )
        self.outgoing = TorchCOO(
            outgoing_rows,
            outgoing_cols,
            torch.zeros(len(outgoing_rows), dtype=self.dtype, device=self.device),
            self.size,
            self.device,
        )
        self.deposit = TorchCOO(
            deposit_rows,
            deposit_cols,
            torch.zeros(len(deposit_rows), dtype=self.dtype, device=self.device),
            self.size,
            self.device,
        )
        self.wall_min = self.tensor([model.walls[n].min_temperature_k for n in model.wall_names])
        self.wall_max = self.tensor([model.walls[n].max_temperature_k for n in model.wall_names])
        self.default_boundary = self.tensor(
            [model.boundary_temperatures_k[n] for n in self.boundary_names]
        )
        self.default_ambient = self.tensor(
            [model.walls[n].ambient_temperature_k for n in model.wall_names]
        )
        physical = model.physics
        pn = {n: i for i, n in enumerate(physical._nodes)}
        self.physical_free = self.integer([dynamic[n] for n in physical.free])
        self.free_volume = self.tensor([physical.storage_volumes_m3[n] for n in physical.free])
        self.reservoir_names = tuple(sorted(model.boundary_reservoirs_kg))
        self.reservoir_index = self.integer([dynamic[n] for n in self.reservoir_names])
        self.reservoir_node = self.integer([pn[n] for n in self.reservoir_names])
        incidence_rows = [pn[b.start] for b in branches] + [pn[b.end] for b in branches]
        self.incidence = TorchCOO(
            incidence_rows,
            list(range(len(branches))) * 2,
            self.tensor([1.0] * len(branches) + [-1.0] * len(branches)),
            len(pn),
            self.device,
        )
        self.friction_index = self.integer(
            list(range(len(physical.network.pipes)))
            + list(range(len(physical.network.pipes) + len(physical.names), len(branches)))
        )
        self.pump_index = self.integer(
            list(
                range(
                    len(physical.network.pipes), len(physical.network.pipes) + len(physical.names)
                )
            )
        )
        self.identity = (
            torch.eye(self.size, dtype=self.dtype, device=self.device)
            if self.size <= dense_limit
            else None
        )
        # Auxiliary mirrors are diagnostics, not hidden host array conversions.
        self.host_upload_batches = 0
        self.host_download_batches = 0

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def integer(self, value):
        return torch.tensor(value, dtype=torch.long, device=self.device)

    @staticmethod
    def direction(q, sign):
        return torch.where(sign == 0, True, torch.where(sign > 0, q >= 0, q < 0))

    def compile_operator(self, flows, heat_j, dt, boundary, ambient, duties):
        rate = flows.abs() * self.model.physics.network.fluid.density_kg_m3 * self.cp
        coeff = (
            rate[self.term_b]
            * self.term_factor
            * self.direction(flows[self.term_b], self.term_sign)
        )
        self.operator.values = torch.cat((coeff, self.static)).index_select(0, self.operator.order)
        incoming = rate[self.force_b] * self.direction(flows[self.force_b], self.force_sign)
        self.incoming.values = incoming.index_select(0, self.incoming.order)
        outgoing = rate[self.out_b] * self.direction(flows[self.out_b], self.out_sign)
        self.outgoing.values = outgoing.index_select(0, self.outgoing.order)
        dep = self.deposit_weight * self.direction(flows[self.deposit_b], self.deposit_sign)
        self.deposit.values = dep.index_select(0, self.deposit.order)
        absorbed = self.deposit.mv(heat_j) / dt
        forcing = self.incoming.mv(boundary - self.ref) + absorbed + duties
        forcing = forcing.index_add(0, self.wall_j, self.ua_ambient * (ambient - self.ref))
        diagonal = self.operator.diagonal()
        dense = self.operator.mv(self.identity).T if self.identity is not None else None
        return forcing, absorbed, diagonal, dense

    def masses_next(self, old_mass, end_pressures, flows, dt, boundary_transfers=None):
        physics = self.model.physics
        free_mass = (
            physics.network.fluid.density_kg_m3
            * self.free_volume
            * (1 + (end_pressures - physics.reference_pressure_pa) / physics.bulk_modulus_pa)
        )
        result = old_mass.scatter(0, self.physical_free, free_mass)
        boundary_transfer = (
            dt * physics.network.fluid.density_kg_m3 * self.incidence.mv(flows)[self.reservoir_node]
            if boundary_transfers is None
            else boundary_transfers
        )
        return result.scatter(
            0, self.reservoir_index, old_mass[self.reservoir_index] - boundary_transfer
        )

    def heat_vector(self, friction, pump, shaft, motor):
        result = torch.zeros(len(self.branch_names), dtype=self.dtype, device=self.device)
        result = result.scatter(0, self.friction_index, friction)
        combined = (
            pump * self.model.pump_heat_to_fluid_fraction
            + shaft * self.model.shaft_heat_to_fluid_fraction
            + motor * self.model.motor_heat_to_fluid_fraction
        )
        return result.scatter(0, self.pump_index, combined)

    def factor(self, mass, dt, dense):
        capacity = torch.cat((mass * self.cp, self.wall_capacity))
        return torch.linalg.lu_factor_ex(
            torch.diag(capacity) - dt * 0.5 * dense, check_errors=False
        )

    def linear_step(
        self, t0, m0, m1, dt, forcing, diagonal, dense, boundary, ambient, factors=None
    ):
        c0 = torch.cat((m0 * self.cp, self.wall_capacity))
        c1 = torch.cat((m1 * self.cp, self.wall_capacity))
        theta0 = t0 - self.ref
        b = c0 * theta0 + dt * 0.5 * self.operator.mv(theta0) + dt * forcing
        diag = c1 - dt * 0.5 * diagonal
        if dense is not None:
            if factors is None:
                theta1, info = torch.linalg.solve_ex(
                    torch.diag(c1) - dt * 0.5 * dense, b, check_errors=False
                )
            else:
                lu, pivots, info = factors
                theta1 = torch.linalg.lu_solve(lu, pivots, b[:, None]).squeeze(-1)
        else:
            scaled = dt * 0.5 * self.operator.values / diag[self.operator.rows]
            scaled = torch.where(self.operator.diagonal_mask, 0.0, scaled)
            rhs = b / diag
            theta1 = theta0
            for _ in range(self.iterations):
                theta1 = rhs + self.operator.mv(theta1, scaled)
            info = torch.zeros((), dtype=torch.int32, device=self.device)
        t1 = theta1 + self.ref
        mid = (theta0 + theta1) / 2
        delta = torch.cat(
            (
                self.cp
                * ((m1 - m0) * theta0[: self.nf] + m1 * (theta1[: self.nf] - theta0[: self.nf])),
                self.wall_capacity * (theta1[self.nf :] - theta0[self.nf :]),
            )
        )
        local = delta - dt * (self.operator.mv(mid) + forcing)
        adv = dt * (self.incoming.mv(boundary - self.ref).sum() - self.outgoing.mv(mid).sum())
        env = dt * (self.ua_ambient * (ambient - t1[self.wall_j] / 2 - t0[self.wall_j] / 2)).sum()
        linear_residual = c1 * theta1 - dt * 0.5 * self.operator.mv(theta1) - b
        domain = (
            torch.isfinite(t1).all()
            & (t1[: self.nf] >= self.model.min_temperature_k).all()
            & (t1[: self.nf] <= self.model.max_temperature_k).all()
        )
        domain = (
            domain & (t1[self.nf :] >= self.wall_min).all() & (t1[self.nf :] <= self.wall_max).all()
        )
        # The scaled contraction must be valid; a failed iterative solve never
        # silently becomes a CPU solve or an accepted temperature field.
        valid = domain & (info == 0) & torch.isfinite(local).all()
        return t1, delta, adv, env, local.abs().max(), linear_residual.abs().max(), valid

    def trial(self, state, m1, flows, heat_j, dt, boundary=None, ambient=None, duties=None):
        boundary = self.default_boundary if boundary is None else boundary
        ambient = self.default_ambient if ambient is None else ambient
        duties = (
            torch.zeros(self.size, dtype=self.dtype, device=self.device)
            if duties is None
            else duties
        )
        forcing, absorbed, diagonal, dense = self.compile_operator(
            flows, heat_j, dt, boundary, ambient, duties
        )
        cmin = torch.cat((torch.minimum(state.mass, m1) * self.cp, self.wall_capacity))
        ratio = dt * (-diagonal) / cmin
        positivity = (ratio <= self.model.positivity_courant * (1 + 1e-12)).all() & (m1 > 0).all()
        rough = self.linear_step(
            state.temperature, state.mass, m1, dt, forcing, diagonal, dense, boundary, ambient
        )
        mm = (state.mass + m1) / 2
        a = self.linear_step(
            state.temperature, state.mass, mm, dt / 2, forcing, diagonal, dense, boundary, ambient
        )
        b = self.linear_step(a[0], mm, m1, dt / 2, forcing, diagonal, dense, boundary, ambient)
        scale = self.model.temperature_tolerance_k + self.model.relative_tolerance * torch.maximum(
            state.temperature.abs(), b[0].abs()
        )
        error = ((b[0] - rough[0]).abs() / (3 * scale)).max()
        audits = []
        for item, h in ((a, dt / 2), (b, dt / 2), (rough, dt)):
            delta, adv, env, local = item[1:5]
            generated = h * absorbed.sum()
            external = h * duties.sum()
            total = delta.sum() - adv - env - generated - external
            scale_audit = (
                delta.abs().sum() + adv.abs() + env.abs() + generated.abs() + external.abs()
            ).clamp_min(1.0)
            audits.append(
                item[-1]
                & (local <= 1e-7 + 1e-10 * scale_audit)
                & (total.abs() <= 1e-7 + 1e-10 * scale_audit)
            )
        change_limit = min(
            [self.model.maximum_temperature_change_k]
            + [w.max_temperature_change_k for w in self.model.walls.values()]
        )
        if self.model.temporal_error_control:
            final = b[0]
            valid = positivity & audits[0] & audits[1] & torch.isfinite(error) & (error <= 1)
        else:
            final = rough[0]
            error = torch.zeros_like(error)
            valid = positivity & audits[2]
        valid = valid & ((final - state.temperature).abs().max() <= change_limit)
        return TorchHeatState(final, m1), rough, a, b, error, valid, forcing, absorbed
