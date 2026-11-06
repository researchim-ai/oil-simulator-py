"""Closed horizontal two-vessel / thermal Euler pipe system.

Finite stirred single-phase PR vessels, stationary supports and equal-area
HLLC ports. Vessel momentum is absorbed by fixed supports (zero wall work).
This is a zero-gravity validation model, not a stratified nozzle replacement.
"""

import math
from dataclasses import asdict, dataclass, replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_momentum_thermal import PRThermalMomentumPipe, PRThermalMomentumState
from .torch_linear import TorchNewton
from .vessel_wall import VesselWall


def hllc_port(left, right, alpha):
    """Subsonic HLLC flux, entropy flux and positivity diagnostics.

    Tuples contain (conserved density, physical flux, p, v, rho, entropy density).
    Species use the contact donor; there is no separate species diffusion term.
    """
    ul, fl, pl, vl, rl, sl = left
    ur, fr, pr, vr, rr, sr = right
    al, ar = rl * (-alpha - vl), rr * (alpha - vr)
    contact = (pr - pl + al * vl - ar * vr) / (al - ar)
    pressure = pl + al * (contact - vl)

    def star(u, flux, p, v, rho, entropy, speed):
        ratio = (speed - v) / (speed - contact)
        change = (contact - v) / (speed - contact)
        increment = torch.cat(
            (
                u[:-2] * change,
                (rho * speed * change).reshape(1),
                ((u[-1] + p + rho * (speed - v) * contact) * change).reshape(1),
            )
        )
        return flux + speed * increment, entropy * ratio * contact, ratio

    lf, ls, lr = star(ul, fl, pl, vl, rl, sl, -alpha)
    rf, rs, rrstar = star(ur, fr, pr, vr, rr, sr, alpha)
    flux = torch.where(contact >= 0, lf, rf)
    entropy = torch.where(contact >= 0, ls, rs)
    valid = (pressure > 0) & (lr > 0) & (rrstar > 0) & (contact.abs() < alpha)
    return flux, entropy, contact, pressure, valid


@dataclass(frozen=True)
class PRMomentumVesselState:
    component_moles: tuple
    internal_energy_j: float
    solution_variables: tuple
    wall_temperature_k: float

    def __post_init__(self):
        object.__setattr__(self, "component_moles", tuple(self.component_moles))
        object.__setattr__(self, "solution_variables", tuple(self.solution_variables))
        for n in self.component_moles:
            _number(n, "vessel inventory", nonnegative=True)
        for y in self.solution_variables:
            _number(y, "vessel primitive")
        _number(self.internal_energy_j, "vessel internal energy")
        _number(self.wall_temperature_k, "vessel wall temperature", positive=True)


@dataclass(frozen=True)
class PRMomentumVesselNetworkState:
    pipe: PRThermalMomentumState
    vessels: tuple
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "vessels", tuple(self.vessels))

    @property
    def time_s(self):
        return self.pipe.time_s

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Unknown momentum vessel network state fields")
        return cls(
            PRThermalMomentumState.from_dict(data["pipe"]),
            tuple(PRMomentumVesselState(**v) for v in data["vessels"]),
            data["model_id"],
        )


class PRMomentumVesselNetwork:
    """Two finite stirred reservoirs, one horizontal pipe, binary end isolation."""

    def __init__(self, pipe, *, vessel_volumes_m3, vessel_walls):
        if not isinstance(pipe, PRThermalMomentumPipe) or pipe.pipe.boundary_mode != "ends":
            raise TypeError("Requires thermal momentum pipe with end boundaries")
        if len(vessel_volumes_m3) != 2 or len(vessel_walls) != 2:
            raise ValueError("Exactly two endpoint vessels required")
        if pipe.pipe.to_dict()["model"] != "single_phase_pr_momentum_v1":
            raise ValueError("Inclined pipes require gravitational vessel coupling")
        self.pipe = pipe
        self.volumes = tuple(_number(v, "vessel volume", positive=True) for v in vessel_volumes_m3)
        self.walls = tuple(vessel_walls)
        if any(not isinstance(w, VesselWall) for w in self.walls):
            raise TypeError("VesselWall required for each reservoir")
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "single_phase_pr_momentum_vessels_v1",
            "pipe": self.pipe.to_dict(),
            "vessel_volumes_m3": self.volumes,
            "vessel_walls": [asdict(w) for w in self.walls],
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        if (
            set(data) != {"model", "pipe", "vessel_volumes_m3", "vessel_walls"}
            or data["model"] != "single_phase_pr_momentum_vessels_v1"
        ):
            raise ValueError("Unknown momentum vessel network model")
        return cls(
            PRThermalMomentumPipe.from_dict(data["pipe"], device=device),
            vessel_volumes_m3=data["vessel_volumes_m3"],
            vessel_walls=[VesselWall(**w) for w in data["vessel_walls"]],
        )

    def _decode(self, y, active):
        f = self.pipe.pipe.fluid
        c = len(active)
        rows = y.reshape(2, c + 1)
        pressure, temperature = rows[:, 0].exp(), rows[:, c] * f.tref
        xa = torch.softmax(torch.cat((rows[:, 1:c], rows.new_zeros((2, 1))), -1), -1)
        x = rows.new_zeros((2, len(f.names))).index_copy(-1, active, xa)
        q = torch.vmap(lambda pp, tt, xx: f.tensor_homogeneous(pp, tt, xx, "stable"))(
            pressure, temperature, x
        )
        n = x / q["molar_volume_m3_mol"][:, None]
        e = q["molar_internal_energy_j_mol"] / q["molar_volume_m3_mol"]
        u = torch.cat((n, pressure.new_zeros((2, 1)), e[:, None]), -1)
        flux = torch.cat((torch.zeros_like(n), pressure[:, None], pressure.new_zeros((2, 1))), -1)
        return {
            "p": pressure,
            "t": temperature,
            "x": x,
            "q": q,
            "u": u,
            "flux": flux,
            "entropy": q["molar_entropy_j_mol_k"] / q["molar_volume_m3_mol"],
        }

    def initialize(
        self, pipe_state, *, pressure_pa, temperature_k, mole_fractions, wall_temperature_k
    ):
        p, f = self.pipe.pipe, self.pipe.pipe.fluid
        self.pipe.inspect(pipe_state)
        if any(
            len(v) != 2 for v in (pressure_pa, temperature_k, mole_fractions, wall_temperature_k)
        ):
            raise ValueError("Two endpoint initial values required")
        active = p._support(pipe_state.pipe.component_moles)
        rows = []
        for pressure, temperature, x in zip(pressure_pa, temperature_k, mole_fractions):
            _number(pressure, "vessel pressure", positive=True)
            _number(temperature, "vessel temperature", positive=True)
            if (
                len(x) != len(f.names)
                or any(not math.isfinite(z) or z < 0 for z in x)
                or abs(math.fsum(x) - 1) > 1e-10
            ):
                raise ValueError("Invalid vessel composition")
            if [i for i, z in enumerate(x) if z > 0] != active.cpu().tolist():
                raise PropertyModelError("Vessels and pipe require common positive support")
            xa = f.tensor(x)[active]
            rows.append(
                torch.cat(
                    (
                        f.tensor([pressure]).log(),
                        (xa[:-1] / xa[-1]).log(),
                        f.tensor([temperature / f.tref]),
                    )
                )
            )
        y = torch.cat(rows)
        d = self._decode(y, active)
        vessels = tuple(
            PRMomentumVesselState(
                tuple((d["u"][i, :-2] * v).cpu().tolist()),
                float((d["u"][i, -1] * v).cpu()),
                tuple(rows[i].cpu().tolist()),
                wall_temperature_k[i],
            )
            for i, v in enumerate(self.volumes)
        )
        state = PRMomentumVesselNetworkState(pipe_state, vessels, self.model_id)
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, PRMomentumVesselNetworkState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Momentum vessel network model/state mismatch")
        self.pipe.inspect(state.pipe)
        p, f = self.pipe.pipe, self.pipe.pipe.fluid
        active = p._support(state.pipe.pipe.component_moles)
        if len(state.vessels) != 2 or any(
            len(s.solution_variables) != len(active) + 1 or len(s.component_moles) != len(f.names)
            for s in state.vessels
        ):
            raise ValueError("Vessel restart dimensions differ")
        d = self._decode(f.tensor([s.solution_variables for s in state.vessels]).flatten(), active)
        for i, (s, v, w) in enumerate(zip(state.vessels, self.volumes, self.walls)):
            if not w.min_temperature_k <= s.wall_temperature_k <= w.max_temperature_k:
                raise PropertyModelError("Vessel wall outside domain")
            if not bool((torch.isfinite(d["u"][i]).all() & d["q"]["valid"][i]).cpu()):
                raise PropertyModelError("Vessel EOS rejected")
            pressure, temperature = float(d["p"][i].cpu()), float(d["t"][i].cpu())
            if (
                not f.min_pressure_pa <= pressure <= f.max_pressure_pa
                or not f.min_temperature_k <= temperature <= f.max_temperature_k
            ):
                raise PropertyModelError("Vessel primitive outside domain")
            if p._stable(pressure, temperature, d["x"][i]) > p.wave_speed_bound_m_s:
                raise PropertyModelError("Vessel acoustic bound exceeded")
            n = f.tensor(s.component_moles)
            if not bool(
                ((n > 0) == (f.tensor(state.pipe.pipe.component_moles[0]) > 0)).all().cpu()
            ):
                raise PropertyModelError("Vessel component support differs")
            if not bool(
                ((d["u"][i, :-2] * v - n).abs() <= 1e-10 + 1e-9 * n.abs()).all().cpu()
            ) or abs(float((d["u"][i, -1] * v).cpu()) - s.internal_energy_j) > 1e-6 + 1e-9 * max(
                1, abs(s.internal_energy_j)
            ):
                raise ConservationError("Vessel primitive/inventory mismatch")

    def _problem(self, state, dt, openings):
        p, f = self.pipe.pipe, self.pipe.pipe.fluid
        active = p._support(state.pipe.pipe.component_moles)
        c, k = len(active), len(state.pipe.pipe.solution_variables)
        ry = f.tensor([s.solution_variables for s in state.vessels]).flatten()
        rlo = f.tensor(
            [math.log(f.min_pressure_pa)] + [-100.0] * (c - 1) + [f.min_temperature_k / f.tref]
        ).repeat(2)
        rhi = f.tensor(
            [math.log(f.max_pressure_pa)] + [100.0] * (c - 1) + [f.max_temperature_k / f.tref]
        ).repeat(2)
        ns = [f.tensor(s.component_moles) for s in state.vessels]
        nscale = [n.clamp_min(n.sum() * 1e-12) for n in ns]
        escale = [
            max(1, abs(s.internal_energy_j), float(n.sum().cpu()) * 8.31446261815324 * f.tref)
            for n, s in zip(ns, state.vessels)
        ]

        def faces(pd, prepared, rd):
            ports = []
            base, entropy = p._faces(pd, (None, None))
            ends, send = [], []
            for side, index in enumerate((0, -1)):
                if not openings[side]:
                    ends.append(base[index])
                    send.append(entropy[index])
                    ports.append(
                        (
                            base[index],
                            entropy[index],
                            pd["p"].new_zeros(()),
                            pd["p"][index],
                            pd["p"].new_ones((), dtype=torch.bool),
                        )
                    )
                    continue
                vessel = (
                    rd["u"][side],
                    rd["flux"][side],
                    rd["p"][side],
                    pd["p"].new_zeros(()),
                    rd["q"]["density_kg_m3"][side],
                    rd["entropy"][side],
                )
                pipe = (
                    pd["u"][index],
                    pd["flux"][index],
                    pd["p"][index],
                    pd["v"][index],
                    pd["q"]["density_kg_m3"][index],
                    pd["entropy"][index],
                )
                port = (
                    hllc_port(vessel, pipe, p.wave_speed_bound_m_s)
                    if side == 0
                    else hllc_port(pipe, vessel, p.wave_speed_bound_m_s)
                )
                ports.append(port)
                ends.append(port[0])
                send.append(port[1])
            return (
                torch.cat((ends[0][None], base[1:-1], ends[1][None]), 0),
                torch.cat((send[0].reshape(1), entropy[1:-1], send[1].reshape(1))),
                ports,
            )

        py, pl, ph, pe = self.pipe._problem(state.pipe, dt, (None, None), face_operator=faces)

        def evaluate(y):
            rd = self._decode(y[k:], active)
            pr, result = pe(y[:k], rd)
            ports = result["face_details"]
            residuals, targets, walls, heats, ambients = [], [], [], [], []
            for i, (old, volume, w) in enumerate(zip(state.vessels, self.volumes, self.walls)):
                flux = (
                    ports[i][0] * p.area * (-1 if i == 0 else 1)
                    if openings[i]
                    else torch.zeros_like(ports[i][0])
                )
                tw = (
                    w.heat_capacity_j_k * old.wall_temperature_k
                    + dt
                    * (w.fluid_ua_w_k * rd["t"][i] + w.ambient_ua_w_k * w.ambient_temperature_k)
                ) / (w.heat_capacity_j_k + dt * (w.fluid_ua_w_k + w.ambient_ua_w_k))
                heat = dt * w.fluid_ua_w_k * (tw - rd["t"][i])
                ambient = dt * w.ambient_ua_w_k * (w.ambient_temperature_k - tw)
                tn = ns[i] + dt * flux[:-2]
                te = old.internal_energy_j + dt * flux[-1] + heat
                residuals.append(
                    torch.cat(
                        (
                            (rd["u"][i, :-2] * volume - tn)[active] / nscale[i][active],
                            ((rd["u"][i, -1] * volume - te) / escale[i]).reshape(1),
                        )
                    )
                )
                targets.append((tn, te))
                walls.append(tw)
                heats.append(heat)
                ambients.append(ambient)
            return torch.cat((pr, *residuals)), {
                "pipe": result,
                "reservoirs": rd,
                "targets": targets,
                "walls": walls,
                "heats": heats,
                "ambients": ambients,
                "ports": ports,
            }

        return torch.cat((py, ry)), torch.cat((pl, rlo)), torch.cat((ph, rhi)), evaluate, k

    def _step(self, state, dt, openings):
        p, f = self.pipe.pipe, self.pipe.pipe.fluid
        y, lo, hi, evaluate, k = self._problem(state, dt, openings)
        if len(y) > 768:
            raise ValueError("Coupled momentum network exceeds dense Newton budget")
        solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=768)
        solver.host_convergence = True
        with torch.no_grad():
            y, _, ok, evaluations = solver.solve(
                lambda z: evaluate(z)[0], y, lo, hi, tolerance=3e-12
            )
            r, data = evaluate(y)
        if not bool((ok & torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("Coupled finite vessel/momentum Newton rejected")
        if not all(bool(port[4].cpu()) for port in data["ports"]):
            raise PropertyModelError("HLLC port outside positive subsonic domain")
        pipe, report = self.pipe._accept_step(
            state.pipe, dt, (None, None), y[:k], r, data["pipe"], evaluations
        )
        rows = y[k:].reshape(2, -1)
        vessels = []
        active = p._support(state.pipe.pipe.component_moles)
        for i, (old, w) in enumerate(zip(state.vessels, self.walls)):
            tw = float(data["walls"][i].cpu())
            if abs(tw - old.wall_temperature_k) > w.max_temperature_change_k:
                raise HydraulicConvergenceError("Vessel wall step limit exceeded")
            n, e = data["targets"][i]
            if not bool(
                (torch.isfinite(n).all() & torch.isfinite(e) & (n[active] > 0).all()).cpu()
            ):
                raise HydraulicConvergenceError("Coupled step exhausts vessel component")
            vessels.append(
                PRMomentumVesselState(
                    tuple(n.cpu().tolist()), float(e.cpu()), tuple(rows[i].cpu().tolist()), tw
                )
            )
        new = PRMomentumVesselNetworkState(pipe, vessels, self.model_id)
        self.inspect(new)
        ledger, ambient = self._ledger(state, new, [report], data["ambients"])
        active = p._support(state.pipe.pipe.component_moles)
        oldrd = self._decode(
            f.tensor([v.solution_variables for v in state.vessels]).flatten(), active
        )
        entropy = report["entropy_production_j_k"]
        for i, (old, final, volume, w) in enumerate(
            zip(state.vessels, new.vessels, self.volumes, self.walls)
        ):
            entropy += float(
                ((data["reservoirs"]["entropy"][i] - oldrd["entropy"][i]) * volume).cpu()
            )
            entropy += w.heat_capacity_j_k * math.log(
                final.wall_temperature_k / old.wall_temperature_k
            )
            entropy -= float(data["ambients"][i].cpu()) / w.ambient_temperature_k
            if openings[i]:
                entropy += float((dt * p.area * data["ports"][i][1] * (1 if i == 0 else -1)).cpu())
        if entropy < -1e-7:
            raise ConservationError("Coupled vessel/pipe total entropy rejected")
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "pipe": report,
            "vessel_ambient_heat_j": [float(z.cpu()) for z in data["ambients"]],
            "port_contact_velocity_m_s": [float(z[2].cpu()) for z in data["ports"]],
            "port_pressure_pa": [float(z[3].cpu()) for z in data["ports"]],
            "vessel_support_impulse_kg_m_s": [
                float((dt * p.area * z[0][-2] * (1 if i == 0 else -1)).cpu())
                if openings[i]
                else 0.0
                for i, z in enumerate(data["ports"])
            ],
            "conservation_residuals": ledger,
            "ambient_heat_j": ambient,
            "entropy_production_j_k": entropy,
            "max_scaled_residual": float(r.abs().max().cpu()),
        }

    def _ledger(self, before, after, pipe_reports, vessel_ambient):
        p, w = self.pipe.pipe, self.pipe.wall
        n = [
            math.fsum(row[i] for row in after.pipe.pipe.component_moles)
            - math.fsum(row[i] for row in before.pipe.pipe.component_moles)
            + math.fsum(
                b.component_moles[i] - a.component_moles[i]
                for a, b in zip(before.vessels, after.vessels)
            )
            for i in range(len(p.fluid.names))
        ]
        de = math.fsum(after.pipe.pipe.total_energy_j) - math.fsum(before.pipe.pipe.total_energy_j)
        de += (
            w.heat_capacity_j_k
            / p.cells
            * math.fsum(
                b - a for a, b in zip(before.pipe.wall_temperature_k, after.pipe.wall_temperature_k)
            )
        )
        de += math.fsum(
            b.internal_energy_j
            - a.internal_energy_j
            + wall.heat_capacity_j_k * (b.wall_temperature_k - a.wall_temperature_k)
            for a, b, wall in zip(before.vessels, after.vessels, self.walls)
        )
        ambient = math.fsum(q for r in pipe_reports for q in r["ambient_heat_j"]) + math.fsum(
            float(q) for q in vessel_ambient
        )
        residual = n + [de - ambient]
        if any(
            abs(x)
            > 1e-8
            + 1e-9 * math.fsum(row[i] for row in before.pipe.pipe.component_moles)
            + 1e-9 * math.fsum(s.component_moles[i] for s in before.vessels)
            for i, x in enumerate(n)
        ) or abs(residual[-1]) > 1e-5 + 1e-9 * max(1, abs(ambient), abs(de)):
            raise ConservationError("Closed vessel/pipe component or total energy ledger rejected")
        return residual, ambient

    def _error(self, coarse, fine):
        p = self.pipe.pipe
        error = p._error(coarse.pipe.pipe, fine.pipe.pipe)
        error = max(
            error,
            max(
                abs(a - b) / p.temperature_tolerance_k
                for a, b in zip(coarse.pipe.wall_temperature_k, fine.pipe.wall_temperature_k)
            ),
        )
        for a, b in zip(coarse.vessels, fine.vessels):
            error = max(
                error,
                abs(a.wall_temperature_k - b.wall_temperature_k) / p.temperature_tolerance_k,
                max(
                    abs(x - y) / (1e-10 + p.relative_tolerance * max(abs(x), abs(y)))
                    for x, y in zip(a.component_moles, b.component_moles)
                ),
                abs(a.internal_energy_j - b.internal_energy_j)
                / (
                    1e-6
                    + p.relative_tolerance
                    * max(1, abs(a.internal_energy_j), abs(b.internal_energy_j))
                ),
                abs(a.solution_variables[-1] - b.solution_variables[-1])
                * p.fluid.tref
                / p.temperature_tolerance_k,
                abs(math.exp(a.solution_variables[0]) - math.exp(b.solution_variables[0]))
                / (1 + p.relative_tolerance * math.exp(a.solution_variables[0])),
            )
        return error

    def _retime(self, state, time, nextdt):
        return replace(
            state,
            pipe=replace(
                state.pipe, pipe=replace(state.pipe.pipe, time_s=time, next_step_s=nextdt)
            ),
        )

    def advance(self, state, duration_s, *, openings=(True, True)):
        self.inspect(state)
        if (
            not isinstance(openings, (tuple, list))
            or len(openings) != 2
            or any(type(v) is not bool for v in openings)
        ):
            raise ValueError("Ports require two binary isolation states")
        p = self.pipe.pipe
        end = state.time_s + _number(duration_s, "duration", positive=True)
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Unrepresentable coupled interval")
        current, steps, attempts = state, [], 0
        dt = min(current.pipe.pipe.next_step_s, duration_s)
        while current.time_s < end:
            attempts += 1
            if attempts > p.maximum_attempts:
                raise HydraulicConvergenceError("Coupled momentum integration budget exhausted")
            dt = min(dt, end - current.time_s)
            if current.time_s + dt / 2 <= current.time_s:
                raise ValueError("Substep below floating time resolution")
            try:
                coarse, _ = self._step(current, dt, openings)
                half, one = self._step(current, dt / 2, openings)
                fine, two = self._step(half, dt / 2, openings)
                error = self._error(coarse, fine)
                if error > 1:
                    raise HydraulicConvergenceError("Coupled momentum temporal error exceeded")
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                dt /= 2
                if dt < p.minimum_step_s:
                    raise HydraulicConvergenceError(
                        "Coupled momentum minimum step exhausted"
                    ) from exc
                continue
            nextdt = min(p.maximum_step_s, max(p.minimum_step_s, dt * (1.5 if error < 0.25 else 1)))
            time = end if dt >= end - current.time_s else min(fine.time_s, end)
            current = self._retime(fine, time, nextdt)
            steps.extend((one, two))
            dt = nextdt
        residual, ambient = self._ledger(
            state,
            current,
            [s["pipe"] for s in steps],
            [q for s in steps for q in s["vessel_ambient_heat_j"]],
        )
        return {
            "state": current.to_dict(),
            "steps": steps,
            "attempts": attempts,
            "openings": list(openings),
            "conservation_residuals": residual,
            "ambient_heat_j": ambient,
            "entropy_production_j_k": math.fsum(s["entropy_production_j_k"] for s in steps),
        }
