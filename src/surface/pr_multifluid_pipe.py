"""Phase-preserving PR multifluid pipe with path transport and local sources.

Horizontal constant-area cells; periodic or closed ends. Own N/P/E for every
phase, transported compositions, independent pressures, temperatures and speeds.
No interphase mass transfer, phase creation/deletion or regime map.
"""

import math
from dataclasses import dataclass, replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_interphase_relaxation import PRInterphaseRelaxation, PRRelaxationPhase, PRRelaxationState
from .pr_multifluid_flux import path_fluctuations
from .time_interval import interval_step_s, split_step_s
from .torch_linear import TorchNewton


@dataclass(frozen=True)
class PRMultifluidPipeState:
    cells: tuple
    time_s: float
    next_step_s: float
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "cells", tuple(self.cells))
        _number(self.time_s, "pipe clock", nonnegative=True)
        _number(self.next_step_s, "next pipe step", positive=True)

    def to_dict(self):
        return {
            "cells": [s.to_dict() for s in self.cells],
            "time_s": self.time_s,
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"cells", "time_s", "next_step_s", "model_id"}:
            raise ValueError("Unknown multifluid checkpoint fields")
        return cls(
            tuple(PRRelaxationState.from_dict(s) for s in data["cells"]),
            data["time_s"],
            data["next_step_s"],
            data["model_id"],
        )


class PRMultifluidPipe:
    def __init__(
        self,
        sources,
        lengths_m,
        diameter_m,
        *,
        boundary_mode="periodic",
        wave_speed_bound_m_s=4000.0,
        minimum_phase_fraction=1e-6,
        dense_limit=128,
        maximum_unknowns=100000,
    ):
        self.sources = tuple(sources)
        if not self.sources or any(not isinstance(s, PRInterphaseRelaxation) for s in self.sources):
            raise TypeError("One local PR source model per cell required")
        self.lengths = tuple(_number(v, "cell length", positive=True) for v in lengths_m)
        if len(self.lengths) != len(self.sources):
            raise ValueError("Cell/source grid differs")
        self.diameter = _number(diameter_m, "pipe diameter", positive=True)
        self.area = math.pi * self.diameter**2 / 4
        self.volumes = tuple(self.area * l for l in self.lengths)
        self.fluid = self.sources[0].fluid
        self.phase_roots = self.sources[0].phase_roots
        if any(
            s.fluid.model_id != self.fluid.model_id
            or s.fluid.device != self.fluid.device
            or s.phase_roots != self.phase_roots
            for s in self.sources
        ):
            raise ValueError("Common fluid, device and ordered phase roots required")
        if boundary_mode not in ("periodic", "walls"):
            raise ValueError("Explicit periodic or closed ends required")
        self.boundary_mode = boundary_mode
        self.wave_speed = _number(wave_speed_bound_m_s, "wave-speed envelope", positive=True)
        self.minimum_phase_fraction = _number(
            minimum_phase_fraction, "minimum phase fraction", positive=True
        )
        if self.minimum_phase_fraction >= 1 / len(self.phase_roots):
            raise ValueError("Strictly interior phase-fraction domain required")
        for key, value in (("dense_limit", dense_limit), ("maximum_unknowns", maximum_unknowns)):
            if type(value) is not int or value < 1:
                raise ValueError("Positive integer solver budget required")
            setattr(self, key, value)
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_multifluid_pipe_v1",
            "sources": [s.to_dict() for s in self.sources],
            "lengths_m": self.lengths,
            "diameter_m": self.diameter,
            "boundary_mode": self.boundary_mode,
            "wave_speed_bound_m_s": self.wave_speed,
            "minimum_phase_fraction": self.minimum_phase_fraction,
            "dense_limit": self.dense_limit,
            "maximum_unknowns": self.maximum_unknowns,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        raw = dict(data)
        if raw.pop("model", None) != "pr_multifluid_pipe_v1":
            raise ValueError("Unknown multifluid pipe model")
        raw["sources"] = [
            PRInterphaseRelaxation.from_dict(s, device=device) for s in raw["sources"]
        ]
        return cls(**raw)

    def initialize(self, cells):
        if len(cells) != len(self.sources):
            raise ValueError("Initial pipe grid differs")
        state = PRMultifluidPipeState(
            tuple(s.initialize(p) for s, p in zip(self.sources, cells)), 0.0, 1e-5, self.model_id
        )
        self.inspect(state)
        return state

    def _thermo(self, n, p, t):
        f = self.fluid
        amount = n.sum(-1)
        x = n / amount.clamp_min(1e-300)[..., None]
        thermo = [
            torch.vmap(
                lambda pp, tt, xx, tag=tag: f.tensor_homogeneous(
                    pp, tt, xx, tag, derivative_order=2
                )
            )(p[:, k], t[:, k], x[:, k])
            for k, tag in enumerate(self.phase_roots)
        ]
        return {
            "volume": torch.stack(
                [amount[:, k] * q["molar_volume_m3_mol"] for k, q in enumerate(thermo)], dim=-1
            ),
            "internal": torch.stack(
                [amount[:, k] * q["molar_internal_energy_j_mol"] for k, q in enumerate(thermo)],
                dim=-1,
            ),
            "entropy": torch.stack(
                [amount[:, k] * q["molar_entropy_j_mol_k"] for k, q in enumerate(thermo)], dim=-1
            ),
            "sound": torch.stack([q["frozen_sound_speed_m_s"] for q in thermo], dim=-1),
            "thermo_valid": torch.stack([q["valid"] for q in thermo]).all() & (amount > 0).all(),
        }

    def _arrays(self, state):
        cache = getattr(self, "_state_cache", None)
        if cache is not None and cache[0] is state:
            return cache[1]
        f = self.fluid
        data = {
            key: f.tensor([[getattr(p, field) for p in c.phases] for c in state.cells])
            for key, field in (
                ("n", "component_moles"),
                ("pressure", "pressure_pa"),
                ("temperature", "temperature_k"),
                ("stored_volume", "volume_m3"),
                ("momentum", "momentum_kg_m_s"),
                ("energy", "total_energy_j"),
            )
        }
        data["mass"] = (data["n"] * f.molar_masses).sum(-1)
        data["velocity"] = data["momentum"] / data["mass"]
        data.update(self._thermo(data["n"], data["pressure"], data["temperature"]))
        self._state_cache = (state, data)
        return data

    def inspect(self, state):
        if (
            not isinstance(state, PRMultifluidPipeState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
            or len(state.cells) != len(self.sources)
        ):
            raise PropertyModelError("Multifluid pipe state/model differs")
        for source, cell in zip(self.sources, state.cells):
            if not isinstance(cell, PRRelaxationState) or any(
                not isinstance(p, PRRelaxationPhase) for p in cell.phases
            ):
                raise PropertyModelError("Frozen typed phase stocks required")
            if (
                cell.time_s != state.time_s
                or cell.model_id != source.model_id
                or len(cell.phases) != len(self.phase_roots)
            ):
                raise PropertyModelError("Multifluid cell identity/clock differs")
            if any(len(p.component_moles) != len(self.fluid.names) for p in cell.phases):
                raise PropertyModelError("Multifluid component registry differs")
        d = self._arrays(state)
        f = self.fluid
        volume = f.tensor(self.volumes)
        alpha = d["stored_volume"] / volume[:, None]
        expected = d["internal"] + 0.5 * d["momentum"].square() / d["mass"]
        valid = (
            d["thermo_valid"]
            & torch.isfinite(d["n"]).all()
            & (d["n"] >= 0).all()
            & (d["pressure"] >= f.min_pressure_pa).all()
            & (d["pressure"] <= f.max_pressure_pa).all()
            & (d["temperature"] >= f.min_temperature_k).all()
            & (d["temperature"] <= f.max_temperature_k).all()
            & ((d["volume"] - d["stored_volume"]).abs() <= 1e-9 * d["stored_volume"]).all()
            & ((expected - d["energy"]).abs() <= 1e-9 * d["energy"].abs().clamp_min(1.0)).all()
            & ((d["stored_volume"].sum(-1) - volume).abs() <= 1e-12 + 1e-9 * volume).all()
            & (alpha >= self.minimum_phase_fraction).all()
            & (d["velocity"].abs() + d["sound"] <= self.wave_speed).all()
            & (
                (
                    torch.abs(
                        (d["velocity"][:, :1] - d["velocity"][:, 1:]).abs() - d["sound"][:, 1:]
                    )
                    > 1e-6
                ).all()
            )
        )
        if not bool(valid.cpu()):
            raise PropertyModelError("Multifluid EOS, phase fraction or wave envelope violated")

    def _problem(self, state, dt):
        self.inspect(state)
        f = self.fluid
        count, nph, nc = len(self.sources), len(self.phase_roots), len(f.names)
        width = nc + 3
        old = self._arrays(state)
        nold = old["n"]
        old_momentum, old_energy = old["momentum"], old["energy"]
        cell_volume = f.tensor(self.volumes)
        cell_length = f.tensor(self.lengths)
        old_volume = old["stored_volume"]
        nscale = torch.maximum(nold, nold.sum(-1, keepdim=True) * 1e-6)
        oldmass = (nold * f.molar_masses).sum(-1)
        old_velocity = old_momentum / oldmass
        old_kinetic = 0.5 * oldmass * old_velocity.square()
        # Component-specific caloric offsets cancel reference-dependent terms
        # through the corresponding component conservation equations.
        anchor = 0.5 * (f.min_temperature_k + f.max_temperature_k)
        eref = f.cp * (anchor - f.tref)
        old_relative = old_energy - (nold * eref).sum(-1)
        scales_energy = old_relative.abs() + old_kinetic + oldmass * 1e4
        scales_momentum = (oldmass * self.wave_speed).clamp_min(1e-12)
        initial = torch.cat(
            (
                old["pressure"].log()[..., None],
                (old["temperature"] / f.tref)[..., None],
                (old_velocity / self.wave_speed)[..., None],
                nold / nscale,
            ),
            dim=-1,
        ).flatten()
        lower = (
            f.tensor(
                [math.log(f.min_pressure_pa), f.min_temperature_k / f.tref, -0.9, *([0.0] * nc)]
            )
            .expand(count, nph, width)
            .reshape(-1)
            .clone()
        )
        upper = (
            f.tensor(
                [math.log(f.max_pressure_pa), f.max_temperature_k / f.tref, 0.9, *([1e12] * nc)]
            )
            .expand(count, nph, width)
            .reshape(-1)
            .clone()
        )
        if len(initial) > self.maximum_unknowns:
            raise ValueError("Multifluid pipe exceeds unknown budget")
        grid = torch.arange(count, device=f.device)
        next_grid = torch.roll(grid, -1)
        interior_grid = grid[:-1]

        supported = torch.tensor(
            [i in getattr(self, "supported_cells", ()) for i in range(count)],
            device=f.device,
            dtype=torch.bool,
        )[:, None]

        def decode(y):
            w = y.reshape(count, nph, width)
            p, t, u = w[..., 0].exp(), w[..., 1] * f.tref, w[..., 2] * self.wave_speed
            velocity_unknown = u
            u = torch.where(supported, torch.zeros_like(u), u)
            n = w[..., 3:] * nscale
            amount = n.sum(-1)
            thermodynamics = self._thermo(n, p, t)
            volume, internal, entropy, sound = (
                thermodynamics[k] for k in ("volume", "internal", "entropy", "sound")
            )
            mass = (n * f.molar_masses).sum(-1)
            momentum = mass * u
            energy = internal + 0.5 * mass * u.square()
            alpha = volume / cell_volume[:, None]
            conserved = torch.cat(
                (
                    n / cell_volume[:, None, None],
                    (momentum / cell_volume[:, None])[..., None],
                    (energy / cell_volume[:, None])[..., None],
                ),
                dim=-1,
            )
            flux = torch.cat(
                (
                    n / cell_volume[:, None, None] * u[..., None],
                    (momentum / cell_volume[:, None] * u + alpha * p)[..., None],
                    ((energy / cell_volume[:, None] + alpha * p) * u)[..., None],
                ),
                dim=-1,
            )
            valid = (
                thermodynamics["thermo_valid"]
                & torch.isfinite(conserved).all()
                & (amount > 0).all()
                & (alpha >= self.minimum_phase_fraction).all()
                & (alpha <= 1 - self.minimum_phase_fraction).all()
                & (u.abs() + sound <= self.wave_speed).all()
            )
            # The reference-speed field must avoid sonic resonance with another
            # phase's acoustic characteristic in this bounded formulation.
            valid = valid & ((torch.abs((u[:, :1] - u[:, 1:]).abs() - sound[:, 1:]) > 1e-6).all())
            return {
                "pressure": p,
                "temperature": t,
                "velocity": u,
                "velocity_unknown": velocity_unknown,
                "n": n,
                "momentum": momentum,
                "energy": energy,
                "alpha": alpha,
                "conserved": conserved,
                "flux": flux,
                "entropy": entropy,
                "sound": sound,
                "valid": valid,
            }

        def part(data, index):
            return {
                k: data[k][index] for k in ("pressure", "velocity", "alpha", "conserved", "flux")
            }

        def ghost(packet):
            # A rigid stationary wall reverses each phase's normal velocity.
            c = packet["conserved"]
            ff = packet["flux"]
            return {
                **packet,
                "velocity": -packet["velocity"],
                "conserved": torch.cat((c[..., :-2], -c[..., -2:-1], c[..., -1:]), dim=-1),
                "flux": torch.cat((-ff[..., :-2], ff[..., -2:-1], -ff[..., -1:]), dim=-1),
            }

        def evaluate(y):
            d = decode(y)
            wall_flux = []
            if hasattr(self, "_network_operator"):
                incoming, face, wall_impulse = self._network_operator(d, part, dt)
            elif self.boundary_mode == "periodic":
                left = part(d, grid)
                right = part(d, next_grid)
                minus, plus, face = path_fluctuations(left, right, self.wave_speed)
                incoming = torch.roll(plus, 1, dims=0) + minus
                wall_impulse = y.new_zeros(())
            else:
                index = interior_grid
                minus, plus, face = path_fluctuations(
                    part(d, index), part(d, index + 1), self.wave_speed
                )
                lp, rp = part(d, 0), part(d, -1)

                # At a wall alpha is continuous; using per-phase acoustic bounds
                # avoids overestimating a liquid rarefaction by a gas/other-phase bound.
                def wall_packet(left, right, speeds):
                    jump = right["conserved"] - left["conserved"]
                    physical = right["flux"] - left["flux"]
                    mm_phase = 0.5 * (physical - speeds[..., None] * jump)
                    pp_phase = 0.5 * (physical + speeds[..., None] * jump)
                    z = y.new_zeros(nph - 1)
                    mm = torch.cat((mm_phase.flatten(), z))
                    pp = torch.cat((pp_phase.flatten(), z))
                    flux = 0.5 * (left["flux"] + right["flux"]) - 0.5 * speeds[..., None] * jump
                    return mm, pp, flux

                _lm, lp_inc, lf = wall_packet(ghost(lp), lp, d["velocity"][0].abs() + d["sound"][0])
                rm_inc, _rp, rf = wall_packet(
                    rp, ghost(rp), d["velocity"][-1].abs() + d["sound"][-1]
                )
                incoming = torch.cat((lp_inc[None, :], plus), dim=0) + torch.cat(
                    (minus, rm_inc[None, :]), dim=0
                )
                wall_impulse = dt * self.area * (lf[:, -2].sum() - rf[:, -2].sum())
                wall_flux = [lf, rf]
                d["valid"] = d["valid"] & (lf[:, -2] > 0).all() & (rf[:, -2] > 0).all()
            phase_rate = incoming[:, : nph * (nc + 2)].reshape(count, nph, nc + 2) * self.area
            rn = d["n"] - nold + dt * phase_rate[..., :nc]
            rm = d["momentum"] - old_momentum + dt * phase_rate[..., -2]
            relative = d["energy"] - (d["n"] * eref).sum(-1)
            relative_rate = phase_rate[..., -1] - (phase_rate[..., :nc] * eref).sum(-1)
            re = relative - old_relative + dt * relative_rate
            ra = (
                d["alpha"][:, 1:]
                - old_volume[:, 1:] / cell_volume[:, None]
                + dt / cell_length[:, None] * incoming[:, nph * (nc + 2) :]
            )
            rv = d["alpha"].sum(-1) - 1
            # Cell-local rows and unknowns support a nearest-neighbour sparse graph.
            rows = torch.cat(
                (
                    rn.reshape(count, -1) / nscale.reshape(count, -1),
                    torch.where(
                        supported, d["velocity_unknown"] / self.wave_speed, rm / scales_momentum
                    ),
                    re / scales_energy,
                    ra,
                    rv[:, None],
                ),
                dim=-1,
            )
            residual = rows.flatten()
            return residual, {
                **d,
                "faces": face,
                "incoming": incoming,
                "wall_flux": wall_flux,
                "wall_impulse": wall_impulse,
                "phase_rate": phase_rate,
                "residual": residual,
                "external_work_j": d.get("external_work_j", y.new_zeros(())),
            }

        cellwidth = nph * width
        edges = set()
        for i in range(count):
            neighbours = {i}
            if hasattr(self, "graph_neighbours"):
                neighbours |= self.graph_neighbours[i]
            elif self.boundary_mode == "periodic":
                neighbours |= {(i - 1) % count, (i + 1) % count}
            else:
                neighbours |= {j for j in (i - 1, i + 1) if 0 <= j < count}
            edges.update(
                (r, c)
                for j in neighbours
                for r in range(i * cellwidth, (i + 1) * cellwidth)
                for c in range(j * cellwidth, (j + 1) * cellwidth)
            )
        pairs = sorted(edges)
        evaluate.sparsity = ([r for r, _ in pairs], [c for _, c in pairs])
        evaluate.linear_blocks = [
            list(range(i * cellwidth, (i + 1) * cellwidth)) for i in range(count)
        ]
        return initial, lower, upper, evaluate

    def transport_step(self, state, dt_s):
        dt = _number(dt_s, "transport step", positive=True)
        if not math.isfinite(state.time_s + dt) or state.time_s + dt <= state.time_s:
            raise ValueError("Step below clock resolution")
        y, lo, hi, evaluate = self._problem(state, dt)
        old_entropy = self.entropy(state)
        solver = TorchNewton(self.fluid.device, maximum_iterations=40, dense_limit=self.dense_limit)
        solver.host_convergence = True
        with torch.no_grad():
            y, _residual, ok, count = solver.solve(
                lambda z: evaluate(z)[0],
                y,
                lo,
                hi,
                pattern=evaluate.sparsity if len(y) > self.dense_limit else None,
                blocks=evaluate.linear_blocks,
                tolerance=2e-13,
            )
            r, d = evaluate(y)
            if not ok or not bool(d["valid"].cpu()) or float(r.abs().max().cpu()) > 1e-10:
                raise HydraulicConvergenceError(
                    "Multifluid path transport Newton failed or left closure domain"
                )
            raw = {
                k: d[k].cpu().tolist()
                for k in ("n", "alpha", "pressure", "temperature", "momentum", "energy")
            }
            cells = []
            for i, source in enumerate(self.sources):
                phases = tuple(
                    PRRelaxationPhase(
                        tuple(raw["n"][i][k]),
                        raw["alpha"][i][k] * self.volumes[i],
                        raw["pressure"][i][k],
                        raw["temperature"][i][k],
                        raw["momentum"][i][k],
                        raw["energy"][i][k],
                    )
                    for k in range(len(self.phase_roots))
                )
                cells.append(
                    PRRelaxationState(
                        phases, state.time_s + dt, source.model_id, state.cells[i].next_step_s
                    )
                )
            new = PRMultifluidPipeState(
                tuple(cells), state.time_s + dt, state.next_step_s, self.model_id
            )
            self.inspect(new)
            entropy = float(d["entropy"].sum().cpu()) - old_entropy
            if entropy < -1e-7:
                raise ConservationError("Multifluid transport entropy decreased")
            wall = d["wall_impulse"].detach().cpu().tolist()
            work = float(d.get("external_work_j", d["energy"].new_zeros(())).detach().cpu())
            ledger = self.ledger(state, new, wall, work)
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "wall_impulse_kg_m_s": wall,
            "external_work_j": work,
            "conservation_residuals": ledger,
            "entropy_change_j_k": entropy,
            "face_fluxes": d["faces"]["flux"].cpu().tolist(),
            "phase_path_integrals": d["faces"]["phase_path"].cpu().tolist(),
            "alpha_path_integrals": d["faces"]["alpha_path"].cpu().tolist(),
            "wall_fluxes": [x.cpu().tolist() for x in d["wall_flux"]],
            "evaluations": count,
        }

    def _combine_impulses(self, transport, reports):
        return transport + math.fsum(r["wall_impulse_kg_m_s"] for r in reports)

    def _sum_impulses(self, values):
        return math.fsum(values)

    def entropy(self, state):
        return float(self._arrays(state)["entropy"].sum().cpu())

    def _stored_energy(self, state):
        return math.fsum(p.total_energy_j for c in state.cells for p in c.phases)

    def ledger(self, old, new, wall_impulse, external_work_j=0.0):
        nn = len(self.fluid.names)
        oldp = [p for c in old.cells for p in c.phases]
        newp = [p for c in new.cells for p in c.phases]
        dn = [
            math.fsum(p.component_moles[k] for p in newp)
            - math.fsum(p.component_moles[k] for p in oldp)
            for k in range(nn)
        ]
        de = self._stored_energy(new) - self._stored_energy(old) - external_work_j
        dp = (
            math.fsum(p.momentum_kg_m_s for p in newp)
            - math.fsum(p.momentum_kg_m_s for p in oldp)
            - wall_impulse
        )
        for k in range(len(self.phase_roots)):
            for j in range(nn):
                drift = math.fsum(c.phases[k].component_moles[j] for c in new.cells) - math.fsum(
                    c.phases[k].component_moles[j] for c in old.cells
                )
                if abs(drift) > 1e-8:
                    raise ConservationError("Unmodelled interphase component transfer")
        if (
            max(map(abs, dn)) > 1e-8
            or abs(de) > 1e-7 + 1e-10 * math.fsum(abs(p.total_energy_j) for p in oldp)
            or abs(dp) > 1e-8
        ):
            raise ConservationError("Multifluid pipe component/E/momentum ledger failed")
        return dn + [de, dp]

    def step(self, state, dt_s):
        """First-order Lie composition, one physical clock interval.

        Both subflows span dt; they are not consecutive physical time intervals.
        Source inputs reuse transport stocks and the original subflow clock.
        """
        dt = _number(dt_s, "multifluid step", positive=True)
        moved, transport = self.transport_step(state, dt)
        source_inputs = [replace(c, time_s=state.time_s) for c in moved.cells]
        cells, reports = [], []
        for source, cell in zip(self.sources, source_inputs):
            end, report = source.step(cell, dt)
            cells.append(end)
            reports.append(report)
        new = PRMultifluidPipeState(
            tuple(cells), state.time_s + dt, state.next_step_s, self.model_id
        )
        self.inspect(new)
        wall = self._combine_impulses(transport["wall_impulse_kg_m_s"], reports)
        entropy = transport["entropy_change_j_k"] + math.fsum(
            r["entropy_change_j_k"] for r in reports
        )
        if entropy < -1e-7:
            raise ConservationError("Composed multifluid entropy decreased")
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "transport": transport,
            "source_inputs": [s.to_dict() for s in source_inputs],
            "sources": reports,
            "wall_impulse_kg_m_s": wall,
            "entropy_change_j_k": entropy,
            "external_work_j": transport["external_work_j"],
            "conservation_residuals": self.ledger(state, new, wall, transport["external_work_j"]),
        }

    def advance(
        self,
        state,
        duration_s,
        *,
        relative_tolerance=1e-3,
        component_tolerance_mol=1e-10,
        velocity_tolerance_m_s=1e-4,
        maximum_step_s=0.001,
        minimum_step_s=1e-10,
        maximum_attempts=1000,
    ):
        self.inspect(state)
        duration = _number(duration_s, "multifluid interval", positive=True)
        tolerance = _number(relative_tolerance, "relative tolerance", positive=True)
        component_tolerance = _number(component_tolerance_mol, "component tolerance", positive=True)
        velocity_tolerance = _number(velocity_tolerance_m_s, "velocity tolerance", positive=True)
        maximum = _number(maximum_step_s, "maximum step", positive=True)
        minimum = _number(minimum_step_s, "minimum step", positive=True)
        if maximum < minimum or type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Ordered step bounds and positive budget required")
        end = state.time_s + duration
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Interval below clock resolution")
        initial = state
        steps = []
        dt = min(duration, maximum, state.next_step_s)
        for attempt in range(maximum_attempts):
            dt = interval_step_s(state.time_s, end, dt)
            a, b = split_step_s(state.time_s, dt)
            try:
                coarse, _ = self.step(state, dt)
                half, ra = self.step(state, a)
                fine, rb = self.step(half, b)
                error = 0.0
                for ca, cb in zip(coarse.cells, fine.cells):
                    for pa, pb in zip(ca.phases, cb.phases):
                        for x, y in zip(pa.component_moles, pb.component_moles):
                            error = max(
                                error,
                                abs(x - y)
                                / (component_tolerance + tolerance * max(abs(x), abs(y))),
                            )
                        for x, y, floor in (
                            (pa.pressure_pa, pb.pressure_pa, 1.0),
                            (pa.temperature_k, pb.temperature_k, 1.0),
                            (pa.volume_m3, pb.volume_m3, 1e-12),
                        ):
                            error = max(
                                error, abs(x - y) / (tolerance * max(floor, abs(x), abs(y)))
                            )
                        ma = sum(
                            n * c.molar_mass_kg_mol
                            for n, c in zip(pa.component_moles, self.fluid.components)
                        )
                        mb = sum(
                            n * c.molar_mass_kg_mol
                            for n, c in zip(pb.component_moles, self.fluid.components)
                        )
                        va, vb = pa.momentum_kg_m_s / ma, pb.momentum_kg_m_s / mb
                        error = max(
                            error,
                            abs(va - vb) / (velocity_tolerance + tolerance * max(abs(va), abs(vb))),
                        )
                if error > 1:
                    raise HydraulicConvergenceError("Multifluid temporal error exceeded")
            except (HydraulicConvergenceError, PropertyModelError, ConservationError):
                dt /= 2
                if dt < minimum:
                    raise
                continue
            nextdt = min(maximum, max(minimum, dt * (1.5 if error < 0.25 else 1.0)))
            state = replace(fine, next_step_s=nextdt)
            steps.extend((ra, rb))
            dt = nextdt
            if state.time_s >= end:
                return {
                    "state": state.to_dict(),
                    "steps": steps,
                    "attempts": attempt + 1,
                    "conservation_residuals": self.ledger(
                        initial,
                        state,
                        self._sum_impulses(r["wall_impulse_kg_m_s"] for r in steps),
                        math.fsum(r.get("external_work_j", 0.0) for r in steps),
                    ),
                }
        raise HydraulicConvergenceError("Multifluid interval budget exhausted")
