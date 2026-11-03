"""Compiled float64 Torch equations for acoustic liquid/shaft networks.

Only static topology and parameter tables are assembled in Python. Residuals,
map interpolation, electrical allocation and Newton linear algebra execute on
one explicitly selected device. Legacy JSON adapters are separate boundaries.
"""

import math

import torch

from .torch_linear import TorchIncidence, TorchNewton, checked_device


class TorchLiquidEquations:
    def __init__(self, physics, device="cpu"):
        self.physics = physics
        self.device = checked_device(device)
        self.dtype = torch.float64
        self.nf = len(physics.free)
        self.npipe = len(physics.network.pipes)
        self.npump = len(physics.names)
        self.nb = len(physics._branches)
        self.nn = len(physics._nodes)
        self.size = self.nf + self.nb + self.npump
        node = {n: i for i, n in enumerate(physics._nodes)}
        self.start = self.integer([node[b.start] for b in physics._branches])
        self.end = self.integer([node[b.end] for b in physics._branches])
        self.incidence_operator = TorchIncidence(
            self.nn,
            [node[b.start] for b in physics._branches],
            [node[b.end] for b in physics._branches],
            self.device,
        )
        self.free = self.integer([node[n] for n in physics.free])
        self.C = self.tensor(
            [physics.storage_volumes_m3[n] / physics.bulk_modulus_pa for n in physics.free]
        )
        self.L = self.tensor(
            [
                physics.network.fluid.density_kg_m3 * p.length_m / (math.pi * p.diameter_m**2 / 4)
                for p in physics.network.pipes
            ]
        )
        self.rho = physics.network.fluid.density_kg_m3
        self.mu = physics.network.fluid.viscosity_pa_s
        self.ps = physics._pscale
        self.qs = physics._qscale
        self.length = self.tensor([p.length_m for p in physics.network.pipes])
        self.diameter = self.tensor([p.diameter_m for p in physics.network.pipes])
        self.roughness = self.tensor([p.roughness_m for p in physics.network.pipes])
        self.minor = self.tensor([p.minor_loss_coefficient for p in physics.network.pipes])
        pumps = physics.network.pumps
        self.shutoff = self.tensor([p.shutoff_head_m for p in pumps])
        self.curvature = self.tensor([p.curve_coefficient_s2_m5 for p in pumps])
        self.efficiency = self.tensor([p.efficiency for p in pumps])
        self.qmax = self.tensor([p.max_flow_m3_s for p in pumps])
        self.nominal = self.tensor([physics.nominal_speeds_rad_s[p.name] for p in pumps])
        self.p0 = self.tensor([physics.no_flow_power_w_at_nominal[p.name] for p in pumps])
        self.maps = {
            key: self.curve_bank([getattr(p, key) for p in pumps])
            for key in ("head_curve", "efficiency_curve", "npsh_curve")
        }
        self.has_head = torch.tensor(
            [p.head_curve is not None for p in pumps], dtype=torch.bool, device=self.device
        )
        self.has_eff = torch.tensor(
            [p.efficiency_curve is not None for p in pumps],
            dtype=torch.bool,
            device=self.device,
        )
        self.has_npsh = torch.tensor(
            [p.npsh_curve is not None for p in pumps], dtype=torch.bool, device=self.device
        )
        self.required_npsh = self.tensor([p.npsh_required_m or 0.0 for p in pumps])
        self.inlet_area = self.tensor(
            [
                math.pi * p.inlet_diameter_m**2 / 4 if p.inlet_diameter_m is not None else math.inf
                for p in pumps
            ]
        )
        self.max_shaft = self.tensor(
            [p.max_shaft_power_w if p.max_shaft_power_w is not None else math.inf for p in pumps]
        )
        drives = [physics.drives[n] for n in physics.names]
        for attr, name, default in [
            ("inertia_kg_m2", "J", None),
            ("viscous_drag_nm_s", "drag", None),
            ("max_speed_rad_s", "maxw", None),
            ("max_motor_torque_nm", "max_motor", None),
            ("max_braking_torque_nm", "max_brake", None),
            ("motor_efficiency", "motor_eta", None),
            ("regeneration_efficiency", "regen_eta", None),
            ("max_electrical_draw_w", "unit_draw", math.inf),
            ("max_electrical_export_w", "unit_export", math.inf),
        ]:
            setattr(
                self,
                name,
                self.tensor(
                    [getattr(d, attr) if getattr(d, attr) is not None else default for d in drives]
                ),
            )
        self.unit_draw_enabled = torch.tensor(
            [d.max_electrical_draw_w is not None for d in drives],
            dtype=torch.bool,
            device=self.device,
        )
        self.unit_export_enabled = torch.tensor(
            [d.max_electrical_export_w is not None for d in drives],
            dtype=torch.bool,
            device=self.device,
        )
        self.unit_draw = self.tensor([d.max_electrical_draw_w or 0.0 for d in drives])
        self.unit_export = self.tensor([d.max_electrical_export_w or 0.0 for d in drives])
        self.station_draw = (
            physics.max_station_electrical_draw_w
            if physics.max_station_electrical_draw_w is not None
            else math.inf
        )
        self.station_export = (
            physics.max_station_electrical_export_w
            if physics.max_station_electrical_export_w is not None
            else math.inf
        )
        valves = physics.network.valves
        self.valve_cv = self.tensor(
            [v.discharge_coefficient * math.pi * v.diameter_m**2 / 4 for v in valves]
        )
        self.valve_open = torch.tensor(
            [v.open for v in valves], dtype=torch.bool, device=self.device
        )
        self.valve_check = torch.tensor(
            [v.check_valve for v in valves], dtype=torch.bool, device=self.device
        )
        self.actuated = torch.tensor(
            [v.name in physics.actuated_valves for v in valves],
            dtype=torch.bool,
            device=self.device,
        )
        self.relief = torch.tensor(
            [v.name in physics.relief_valves for v in valves], dtype=torch.bool, device=self.device
        )
        self.cracking = self.tensor(
            [
                physics.relief_valves[v.name].cracking_pressure_pa
                if v.name in physics.relief_valves
                else 0.0
                for v in valves
            ]
        )
        self.full_lift = self.tensor(
            [
                physics.relief_valves[v.name].full_open_pressure_pa
                if v.name in physics.relief_valves
                else 1.0
                for v in valves
            ]
        )
        self.pmin = max(
            math.nextafter(physics.network.fluid.vapor_pressure_pa, math.inf),
            physics.reference_pressure_pa
            - physics.bulk_modulus_pa * physics.max_compression_fraction,
        )
        self.pmax = min(
            physics.max_pressure_pa,
            physics.reference_pressure_pa
            + physics.bulk_modulus_pa * physics.max_compression_fraction,
        )
        self.newton = TorchNewton(device=self.device)
        # Structural pattern is host topology, not a numerical matrix transfer.
        rows = []
        cols = []
        free_lookup = {node[n]: i for i, n in enumerate(physics.free)}
        for i in range(self.nf):
            rows.append(i)
            cols.append(i)
        for e, b in enumerate(physics._branches):
            for name in (b.start, b.end):
                if node[name] in free_lookup:
                    i = free_lookup[node[name]]
                    rows.extend((i, self.nf + e))
                    cols.extend((self.nf + e, i))
            rows.append(self.nf + e)
            cols.append(self.nf + e)
        for i in range(self.npump):
            rows.append(self.nf + self.npipe + i)
            cols.append(self.nf + self.nb + i)
            for c in (
                list(range(self.nf))
                + list(range(self.nf + self.npipe, self.nf + self.npipe + self.npump))
                + list(range(self.nf + self.nb, self.size))
            ):
                rows.append(self.nf + self.nb + i)
                cols.append(c)
        self.pattern = tuple(zip(*sorted(set(zip(rows, cols)))))
        self.bridge_uploads = 0
        self.bridge_downloads = 0
        self.status_reads = 0
        self._cuda_graph = None

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def integer(self, value):
        return torch.tensor(value, dtype=torch.long, device=self.device)

    def curve_bank(self, curves):
        count = max([2] + [len(c.flow_m3_s) for c in curves if c is not None])
        xs = []
        ys = []
        for curve in curves:
            x = list(curve.flow_m3_s) if curve else [0.0, 1.0]
            y = list(curve.values) if curve else [0.0, 0.0]
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            spacing = max(x[-1], 1.0)
            while len(x) < count:
                x.append(x[-1] + spacing)
                y.append(y[-1] + spacing * slope)
            xs.append(x)
            ys.append(y)
        return self.tensor(xs).reshape(self.npump, count), self.tensor(ys).reshape(
            self.npump, count
        )

    def interpolate(self, bank, query):
        if not self.npump:
            return query
        x, y = bank
        index = (
            torch.searchsorted(x.contiguous(), query[:, None].contiguous(), right=True).squeeze(-1)
            - 1
        )
        index = index.clamp(0, x.shape[1] - 2)
        lo = index[:, None]
        hi = lo + 1
        x0 = x.gather(1, lo).squeeze(1)
        x1 = x.gather(1, hi).squeeze(1)
        y0 = y.gather(1, lo).squeeze(1)
        y1 = y.gather(1, hi).squeeze(1)
        return y0 + (query - x0) * (y1 - y0) / (x1 - x0)

    def incidence(self, rates):
        return self.incidence_operator.mv(rates)

    def pipe_loss(self, q):
        area = math.pi * self.diameter.square() / 4
        re = q.abs() * self.rho * self.diameter / (self.mu * area)
        r = re.clamp_min(2000.0)
        term = self.roughness / (3.7 * self.diameter) + 5.74 * r.pow(-0.9)
        turbulent = 0.25 / torch.log10(term).square()
        r4 = torch.full_like(r, 4000.0)
        a4 = self.roughness / (3.7 * self.diameter) + 5.74 * r4.pow(-0.9)
        log4 = torch.log10(a4)
        f4 = 0.25 / log4.square()
        df4 = -0.5 * (-0.9 * 5.74 * r4.pow(-0.9) / (r4 * a4 * math.log(10))) / log4.pow(3)
        t = ((re - 2000) / 2000).clamp(0, 1)
        f = (
            (2 * t.pow(3) - 3 * t.square() + 1) * 0.032
            + (t.pow(3) - 2 * t.square() + t) * 2000 * (-64 / 2000**2)
            + (-2 * t.pow(3) + 3 * t.square()) * f4
            + (t.pow(3) - t.square()) * 2000 * df4
        )
        f = torch.where(re >= 4000, turbulent, f)
        turbulent_loss = (
            self.rho
            / (2 * area.square())
            * (f * self.length / self.diameter + self.minor)
            * q
            * q.abs()
        )
        laminar = (
            128 * self.mu * self.length / (math.pi * self.diameter.pow(4)) * q
            + self.rho / (2 * area.square()) * self.minor * q * q.abs()
        )
        return torch.where(re <= 2000, laminar, turbulent_loss)

    def allocate(self, old, end, commands, loads, dt):
        peak = torch.maximum(old, end.clamp_min(0))
        mid = ((old + end) / 2).clamp_min(0)
        draw_limit = torch.where(
            self.unit_draw_enabled,
            self.unit_draw * self.motor_eta / peak.clamp_min(1e-30),
            self.max_motor,
        )
        export_limit = torch.where(
            self.regen_eta > 0,
            torch.where(
                self.unit_export_enabled,
                self.unit_export / (peak * self.regen_eta).clamp_min(1e-30),
                self.max_brake,
            ),
            self.max_brake,
        )
        torque = torch.minimum(commands, self.max_motor)
        torque = torch.maximum(torque, -self.max_brake)
        torque = torch.where(
            torque > 0, torch.minimum(torque, draw_limit), torch.maximum(torque, -export_limit)
        )
        speed_limit = self.J * (self.maxw - old) / dt + loads + self.drag * mid
        torque = torch.where(torque > 0, torch.minimum(torque, speed_limit), torque)
        draw = (torque.clamp_min(0) * peak / self.motor_eta).sum()
        export = ((-torque).clamp_min(0) * peak * self.regen_eta).sum()
        fd = (
            torch.ones_like(draw)
            if math.isinf(self.station_draw)
            else torch.clamp(self.station_draw / draw.clamp_min(1e-30), max=1.0)
        )
        fe = (
            torch.ones_like(export)
            if math.isinf(self.station_export)
            else torch.clamp(self.station_export / export.clamp_min(1e-30), max=1.0)
        )
        torque = torque * torch.where(torque > 0, fd, torch.where(self.regen_eta > 0, fe, 1.0))
        return torque, fd, fe

    def pump_quantities(self, q, omega, driving):
        ratio = omega / self.nominal
        safe = ratio.clamp_min(1e-30)
        nominal_q = q.abs() / safe
        head = self.interpolate(self.maps["head_curve"], nominal_q)
        loss = torch.where(
            self.has_head,
            -self.rho * 9.80665 * ratio.square() * head,
            self.rho * 9.80665 * (self.curvature * q * q.abs() - ratio.square() * self.shutoff),
        )
        shutoff = -self.rho * 9.80665 * ratio.square() * self.shutoff
        blocked = (omega <= 0) | (driving <= shutoff)
        eta = torch.where(
            self.has_eff,
            self.interpolate(
                self.maps["efficiency_curve"], nominal_q.clamp_min(0).minimum(self.qmax)
            ),
            self.efficiency,
        )
        hydraulic = torch.where(blocked, 0.0, (-driving * q).clamp_min(0))
        power = torch.where(omega > 0, hydraulic / eta + self.p0 * ratio.pow(3), 0.0)
        loads = power / omega.clamp_min(1e-30)
        return loss, blocked, power, loads, ratio, eta

    @staticmethod
    def signed_root(dp):
        positive = dp.abs() > 0
        return (
            torch.sign(dp)
            * torch.sqrt(torch.where(positive, dp.abs(), torch.ones_like(dp)))
            * positive
        )

    def evaluate(self, x, oldp, oldq, oldw, dt, commands, fixed, openings):
        nf, nb, npip, npump = self.nf, self.nb, self.npipe, self.npump
        ws = oldw.clamp_min(1.0)
        endp = x[:nf] * self.ps
        flows = x[nf : nf + nb] * self.qs
        endw = x[nf + nb :] * ws
        p = fixed.scatter(0, self.free, (oldp + endp) / 2)
        qm = (oldq + flows[:npip]) / 2
        allq = torch.cat((qm, flows[npip:]))
        wm = ((oldw + endw) / 2).clamp_min(0)
        dp = p[self.start] - p[self.end]
        loss_pipe = self.pipe_loss(qm)
        pipe_r = (self.L * (flows[:npip] - oldq) - dt * (dp[:npip] - loss_pipe)) / (
            self.L * self.qs
        )
        loss_pump, blocked, power, loads, _ratio, _eta = self.pump_quantities(
            flows[npip : npip + npump], wm, dp[npip : npip + npump]
        )
        pump_r = torch.where(
            blocked,
            flows[npip : npip + npump] / self.qs,
            torch.maximum(
                (dp[npip : npip + npump] - loss_pump) / self.ps,
                -flows[npip : npip + npump] / self.qs,
            ),
        )
        vd = dp[npip + npump :]
        vq = flows[npip + npump :]
        lift = ((vd - self.cracking) / (self.full_lift - self.cracking)).clamp(0, 1)
        opening = torch.where(self.relief, lift, openings)
        allowed = self.valve_open & (opening > 0)
        driven = torch.where(self.valve_check, vd.clamp_min(0), vd)
        expected = self.valve_cv * opening * self.signed_root(driven) * math.sqrt(2 / self.rho)
        force = (
            vd
            - self.rho / (2 * (self.valve_cv * opening.clamp_min(1e-30)).square()) * vq * vq.abs()
        ) / self.ps
        forward = torch.where(self.valve_check, torch.maximum(force, -vq / self.qs), force)
        inverse = self.actuated | self.relief
        valve_r = torch.where(
            inverse, (vq - expected) / self.qs, torch.where(allowed, forward, vq / self.qs)
        )
        valve_loss = torch.where(
            inverse,
            vd * vq,
            torch.where(
                allowed,
                self.rho
                / (2 * (self.valve_cv * opening.clamp_min(1e-30)).square())
                * vq.square()
                * vq.abs(),
                0.0,
            ),
        )
        torque, fd, fe = self.allocate(oldw, endw, commands, loads, dt)
        initial = self.allocate(oldw, oldw, commands, torch.zeros_like(oldw), dt)[0]
        held = (oldw == 0) & (initial <= 0)
        shaft = (self.J * (endw - oldw) - dt * (torque - loads - self.drag * wm)) / (self.J * ws)
        shaft = torch.where(held, endw / ws, shaft)
        continuity = (self.C * (endp - oldp) + dt * self.incidence(allq)[self.free]) / (
            self.C * self.ps
        )
        residual = torch.cat((continuity, pipe_r, pump_r, valve_r, shaft))
        data = (
            endp,
            flows[:npip],
            endw,
            p,
            allq,
            wm,
            torque,
            power,
            torch.cat((loss_pipe * qm, valve_loss)),
            fd,
            fe,
        )
        return residual, data

    def pump_seed(self, dp, oldw):
        ratio = oldw / self.nominal
        available = dp + self.rho * 9.80665 * self.shutoff * ratio.square()
        quadratic = torch.sqrt(available.clamp_min(0) / (self.rho * 9.80665 * self.curvature))
        if self.npump:
            # Inverse of a strictly decreasing piecewise-linear head table.
            hx, hy = self.maps["head_curve"]
            target = -dp / (self.rho * 9.80665 * ratio.square().clamp_min(1e-30))
            reverse_y = -hy
            query = -target
            idx = (
                torch.searchsorted(
                    reverse_y.contiguous(), query[:, None].contiguous(), right=True
                ).squeeze(1)
                - 1
            ).clamp(0, hx.shape[1] - 2)
            lo = idx[:, None]
            hi = lo + 1
            a = hy.gather(1, lo).squeeze(1)
            b = hy.gather(1, hi).squeeze(1)
            qa = hx.gather(1, lo).squeeze(1)
            qb = hx.gather(1, hi).squeeze(1)
            inverse = (qa + (target - a) * (qb - qa) / (b - a).clamp(max=-1e-30)) * ratio
            qp = torch.where(self.has_head, inverse, quadratic)
            qp = torch.where((available > 0) & (ratio > 0), qp, 0.0)
        else:
            qp = oldw
        return qp

    def algebraic_seed(self, p, oldw, openings):
        dp = p[self.start] - p[self.end]
        qp = self.pump_seed(dp[self.npipe : self.npipe + self.npump], oldw)
        vd = dp[self.npipe + self.npump :]
        lift = ((vd - self.cracking) / (self.full_lift - self.cracking)).clamp(0, 1)
        opening = torch.where(self.relief, lift, openings)
        vd = torch.where(self.valve_check, vd.clamp_min(0), vd)
        qv = torch.where(
            self.valve_open,
            self.valve_cv * opening * self.signed_root(vd) * math.sqrt(2 / self.rho),
            0.0,
        )
        return qp, qv, dp

    def pump_sensors(self, suction, discharge, omega):
        dp = suction - discharge
        flow = self.pump_seed(dp, omega)
        ratio = omega / self.nominal
        limit_loss = self.pump_quantities(self.qmax * ratio, omega, dp)[0]
        outside = (omega > 0) & (dp > limit_loss)
        required = torch.maximum(
            self.required_npsh,
            ratio.square()
            * self.interpolate(self.maps["npsh_curve"], flow / ratio.clamp_min(1e-30)),
        )
        available = (suction - self.physics.network.fluid.vapor_pressure_pa) / (
            self.rho * 9.80665
        ) + (flow / self.inlet_area).square() / (2 * 9.80665)
        bad_npsh = (omega > 0) & (available < required)
        return flow, ratio, available - required, outside, bad_npsh

    def guess(self, oldp, oldq, oldw, dt, commands, fixed, openings):
        p = fixed.scatter(0, self.free, oldp)
        qp, qv, dp = self.algebraic_seed(p, oldw, openings)
        _, _, _, loads, _, _ = self.pump_quantities(
            qp, oldw, dp[self.npipe : self.npipe + self.npump]
        )
        torque = self.allocate(oldw, oldw, commands, loads, dt)[0]
        w1 = oldw + dt * (torque - loads - self.drag * oldw) / self.J
        q1 = oldq + dt * (dp[: self.npipe] - self.pipe_loss(oldq)) / self.L
        pm = (
            oldp
            - 0.5 * dt * self.incidence(torch.cat(((oldq + q1) / 2, qp, qv)))[self.free] / self.C
        )
        qp, qv, _ = self.algebraic_seed(
            fixed.scatter(0, self.free, pm), ((oldw + w1) / 2).clamp_min(0), openings
        )
        # The predicted midpoint seed avoids singular q=0 reopening slopes.
        return torch.cat(
            (
                (2 * pm - oldp) / self.ps,
                q1 / self.qs,
                qp / self.qs,
                qv / self.qs,
                w1 / oldw.clamp_min(1.0),
            )
        )

    def _solve_impl(self, oldp, oldq, oldw, dt, commands, fixed, openings, initial=None):
        ws = oldw.clamp_min(1.0)
        lower = torch.cat(
            (
                torch.full_like(oldp, self.pmin / self.ps),
                torch.full((self.nb,), -math.inf, dtype=self.dtype, device=self.device),
                -self.maxw / ws,
            )
        )
        upper = torch.cat(
            (
                torch.full_like(oldp, self.pmax / self.ps),
                torch.full((self.nb,), math.inf, dtype=self.dtype, device=self.device),
                self.maxw / ws,
            )
        )
        lower[self.nf + self.npipe : self.nf + self.npipe + self.npump] = 0.0
        fn = lambda x: self.evaluate(x, oldp, oldq, oldw, dt, commands, fixed, openings)[0]
        x0 = (
            self.guess(oldp, oldq, oldw, dt, commands, fixed, openings)
            if initial is None
            else initial
        )
        x, residual, success, evaluations = self.newton.solve(fn, x0, lower, upper, self.pattern)
        _, data = self.evaluate(x, oldp, oldq, oldw, dt, commands, fixed, openings)
        drive = data[3][self.start] - data[3][self.end]
        _, blocked, _, _, _, _ = self.pump_quantities(
            data[4][self.npipe : self.npipe + self.npump],
            data[5],
            drive[self.npipe : self.npipe + self.npump],
        )
        initial = self.allocate(oldw, oldw, commands, torch.zeros_like(oldw), dt)[0]
        held = (oldw == 0) & (initial <= 0)
        x = torch.cat(
            (
                x[: self.nf + self.npipe],
                torch.where(
                    blocked, 0.0, x[self.nf + self.npipe : self.nf + self.npipe + self.npump]
                ),
                x[self.nf + self.npipe + self.npump : self.nf + self.nb],
                torch.where(held, 0.0, x[self.nf + self.nb :]),
            )
        )
        residual, data = self.evaluate(x, oldp, oldq, oldw, dt, commands, fixed, openings)
        success = success & (residual.abs().max() <= 1e-8)
        return x, residual, data, success, evaluations

    def solve(self, oldp, oldq, oldw, dt, commands, fixed, openings):
        args = (oldp, oldq, oldw, dt, commands, fixed, openings)
        if any(t.dtype != self.dtype or t.device != self.device for t in args):
            raise ValueError("Torch liquid inputs must be float64 on the configured device")
        if self.device.type != "cuda":
            return self._solve_impl(*args)
        initial = self.guess(*args)
        r0, data0 = self.evaluate(initial, *args)
        norm = float(r0.abs().max().detach().cpu())
        self.status_reads += 1
        if norm <= 2e-11:
            return initial, r0, data0, torch.isfinite(initial).all(), 1
        with torch.cuda.device(self.device):
            if self._cuda_graph is None:
                self._graph_inputs = tuple(t.clone() for t in args)
                self._graph_initial = initial.clone()
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                old_budget = self.newton.maximum_iterations
                self.newton.maximum_iterations = 2
                try:
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            self._solve_impl(*self._graph_inputs, initial=self._graph_initial)
                    torch.cuda.current_stream(self.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        self._graph_output = self._solve_impl(
                            *self._graph_inputs, initial=self._graph_initial
                        )
                    self._cuda_graph = graph
                finally:
                    self.newton.maximum_iterations = old_budget
            for dst, src in zip(self._graph_inputs, args):
                dst.copy_(src)
            self._graph_initial.copy_(initial)
            evaluations = 1
            for _ in range(8):
                self._cuda_graph.replay()
                output = self._graph_output
                evaluations += output[-1]
                valid = bool((output[3] & (output[1].abs().max() <= 2e-11)).detach().cpu())
                self.status_reads += 1
                if valid:
                    return self._snapshot(output, evaluations)
                self._graph_initial.copy_(output[0])
            return self._snapshot(self._graph_output, evaluations)

    @staticmethod
    def _snapshot(output, evaluations):
        # CUDA Graph output storage is overwritten by the next replay. Public
        # results must own their buffers, including the nested physical fields.
        x, residual, data, success, _ = output
        return (
            x.clone(),
            residual.clone(),
            tuple(t.clone() for t in data),
            (success & (residual.abs().max() <= 2e-11)).clone(),
            evaluations,
        )

    def mechanical_ledger(self, oldp, oldq, oldw, dt, data, residual):
        p1, q1, w1, pm, qm, wm, torque, power, losses, fd, fe = data
        kinetic = 0.5 * (self.J * (w1 - oldw) * (w1 + oldw)).sum()
        pipe_energy = 0.5 * (self.L * (q1 - oldq) * (q1 + oldq)).sum()
        elastic = (
            0.5
            * (self.C * (p1 - oldp) * (p1 + oldp - 2 * self.physics.reference_pressure_pa)).sum()
        )
        work = dt * torque * wm
        electric = torch.where(work >= 0, work / self.motor_eta, work * self.regen_eta)
        motor_units = electric - work
        drag_units = dt * self.drag * wm.square()
        dp = pm[self.start] - pm[self.end]
        pump_units = dt * (
            power
            + dp[self.npipe : self.npipe + self.npump] * qm[self.npipe : self.npipe + self.npump]
        )
        fric_units = dt * losses
        outward = self.incidence(qm)
        boundary_mask = torch.ones(self.nn, dtype=self.dtype, device=self.device).scatter(
            0, self.free, 0.0
        )
        transfers = dt * self.rho * outward * boundary_mask
        boundary_work = (
            dt * ((pm - self.physics.reference_pressure_pa) * outward * boundary_mask).sum()
        )
        mass_change = self.rho * (self.C * (p1 - oldp)).sum()
        shaft_units = work - dt * power - drag_units - 0.5 * self.J * (w1 - oldw) * (w1 + oldw)
        local_mass = self.rho * (self.C * (p1 - oldp) + dt * outward[self.free])
        motor = motor_units.sum()
        drag = drag_units.sum()
        pump = pump_units.sum()
        friction = fric_units.sum()
        error = (
            electric.sum()
            + boundary_work
            - kinetic
            - pipe_energy
            - elastic
            - motor
            - drag
            - pump
            - friction
        )
        scale = (
            torch.stack(
                (
                    electric.sum(),
                    boundary_work,
                    kinetic,
                    pipe_energy,
                    elastic,
                    motor,
                    drag,
                    pump,
                    friction,
                )
            )
            .abs()
            .sum()
            .clamp_min(1.0)
        )
        peak = torch.maximum(oldw, w1)
        draw = (torque.clamp_min(0) * peak / self.motor_eta).sum()
        export = ((-torque).clamp_min(0) * peak * self.regen_eta).sum()
        shaft_error = shaft_units.sum()
        mass_error = mass_change - transfers.sum()
        max_mass = local_mass.abs().max()
        max_unit = (
            torch.max(shaft_units.abs(), dim=0).values if self.npump else torch.zeros_like(error)
        )
        valid = (
            torch.isfinite(error)
            & (error.abs() <= 1e-6 + 1e-7 * scale)
            & (shaft_error.abs() <= 1e-6 + 1e-7 * scale)
        )
        valid = (
            valid
            & (mass_error.abs() <= 1e-8)
            & (max_mass <= 1e-8)
            & (max_unit <= 1e-6 + 1e-7 * scale)
        )
        valid = valid & (torch.stack((motor, drag, pump, friction)) >= -1e-7).all()
        valid = (
            valid
            & (draw <= self.station_draw * (1 + 1e-10))
            & (export <= self.station_export * (1 + 1e-10))
        )
        return {
            "shaft_kinetic_energy_change_j": kinetic,
            "pipe_kinetic_energy_change_j": pipe_energy,
            "acoustic_energy_change_j": elastic,
            "electrical_energy_j": electric.sum(),
            "boundary_work_in_j": boundary_work,
            "motor_heat_j": motor,
            "shaft_drag_heat_j": drag,
            "pump_heat_j": pump,
            "friction_heat_j": friction,
            "mass_change_kg": mass_change,
            "mass_residual_kg": mass_error,
            "energy_residual_j": error,
            "shaft_energy_residual_j": shaft_error,
            "max_node_mass_residual_kg": max_mass,
            "peak_electrical_draw_w": draw,
            "peak_electrical_export_w": export,
            "draw_allocation_factor": fd,
            "export_allocation_factor": fe,
            "max_scaled_equation_residual": residual.abs().max(),
            "unit_shaft_energy_residuals_j": shaft_units,
            "boundary_mass_transfers_in_kg": transfers,
            "friction_unit_heat_j": fric_units,
            "pump_unit_heat_j": pump_units,
            "drag_unit_heat_j": drag_units,
            "motor_unit_heat_j": motor_units,
            "max_speed_change_rad_s": (w1 - oldw).abs().max()
            if self.npump
            else torch.zeros_like(error),
            "minimum_speed_rad_s": w1.min() if self.npump else torch.zeros_like(error),
            "max_pressure_change_pa": (p1 - oldp).abs().max(),
            "conservation_valid": valid.to(self.dtype),
        }
