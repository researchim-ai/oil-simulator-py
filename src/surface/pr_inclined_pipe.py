"""Well-balanced single-phase PR Euler pipe on a straight inclined centreline.

Two hydrostatic reconstructions per cell are joint Newton unknowns, not a
frozen reference profile. Stocks store U+K+PE relative to the left endpoint.
"""

import math

import torch

from .darcy_hydraulics import darcy_pressure_loss
from .equilibrium_properties import EquilibriumFluid
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .peng_robinson import R
from .pr_momentum_pipe import PRMomentumPipe, PRMomentumState
from .torch_linear import TorchNewton


def solve_chemical_nodes(f, initial, temperatures, chemical_targets, active):
    c = len(active)
    count = len(initial)
    if count * c > 768:
        raise ValueError("Hydrostatic reconstruction exceeds dense initialization budget")

    def evaluate(y):
        rows = y.reshape(count, c)
        pressure = rows[:, 0].exp()
        xa = torch.softmax(torch.cat((rows[:, 1:], rows.new_zeros((count, 1))), -1), -1)
        x = rows.new_zeros((count, len(f.names))).index_copy(-1, active, xa)
        q = torch.vmap(lambda pp, tt, xx: f.tensor_homogeneous(pp, tt, xx, "stable"))(
            pressure, temperatures, x
        )
        mu = (
            torch.log(x[:, active].clamp_min(1e-300))
            + q["log_fugacity_coefficients"][:, active]
            + torch.log(pressure / f.pref)[:, None]
        )
        return (mu - chemical_targets).flatten()

    lo = f.tensor([math.log(f.min_pressure_pa)] + [-100.0] * (c - 1)).repeat(count)
    hi = f.tensor([math.log(f.max_pressure_pa)] + [100.0] * (c - 1)).repeat(count)
    solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=768)
    solver.host_convergence = True
    with torch.no_grad():
        y, r, ok, _ = solver.solve(evaluate, initial.flatten(), lo, hi, tolerance=3e-12)
    if not bool((ok & torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()):
        raise HydraulicConvergenceError("Local PR hydrostatic reconstruction initialization failed")
    return y.reshape(count, c)


class PRInclinedMomentumPipe(PRMomentumPipe):
    def __init__(
        self,
        fluid,
        *,
        elevation_change_m,
        reference_elevation_m=0.0,
        gravity_m_s2=9.80665,
        **options,
    ):
        self.elevation_change_m = _number(elevation_change_m, "elevation change")
        self.reference_elevation_m = _number(reference_elevation_m, "reference elevation")
        self.gravity_m_s2 = _number(gravity_m_s2, "gravity", nonnegative=True)
        super().__init__(fluid, **options)
        if abs(self.elevation_change_m) > self.length_m:
            raise ValueError("Elevation change exceeds straight pipe length")
        if self.boundary_mode == "periodic" and self.elevation_change_m:
            raise ValueError("Inclined endpoints cannot be periodic")
        self._zc = fluid.tensor(
            [(i + 0.5) * self.elevation_change_m / self.cells for i in range(self.cells)]
        )
        self._zf = fluid.tensor(
            [i * self.elevation_change_m / self.cells for i in range(self.cells + 1)]
        )

    def to_dict(self):
        return {
            **super().to_dict(),
            "model": "inclined_single_phase_pr_momentum_v1",
            "elevation_change_m": self.elevation_change_m,
            "reference_elevation_m": self.reference_elevation_m,
            "gravity_m_s2": self.gravity_m_s2,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "inclined_single_phase_pr_momentum_v1":
            raise ValueError("Unknown inclined pipe model")
        data["fluid"] = EquilibriumFluid.from_dict(data["fluid"], device=device)
        return cls(**data)

    def _bounds(self, c):
        lo, hi = super()._bounds(c)
        f = self.fluid
        return (
            torch.cat(
                (
                    lo,
                    f.tensor([math.log(f.min_pressure_pa)] + [-100.0] * (c - 1)).repeat(
                        2 * self.cells
                    ),
                )
            ),
            torch.cat(
                (
                    hi,
                    f.tensor([math.log(f.max_pressure_pa)] + [100.0] * (c - 1)).repeat(
                        2 * self.cells
                    ),
                )
            ),
        )

    def _nodes(self, rows, t, v, z, active):
        f = self.fluid
        count = len(rows)
        p = rows[:, 0].exp()
        xa = torch.softmax(torch.cat((rows[:, 1:], rows.new_zeros((count, 1))), -1), -1)
        x = rows.new_zeros((count, len(f.names))).index_copy(-1, active, xa)
        q = torch.vmap(lambda pp, tt, xx: f.tensor_homogeneous(pp, tt, xx, "stable"))(p, t, x)
        n = x / q["molar_volume_m3_mol"][:, None]
        rho = q["density_kg_m3"]
        internal = q["molar_internal_energy_j_mol"] / q["molar_volume_m3_mol"]
        kinetic = 0.5 * rho * v.square()
        potential = rho * self.gravity_m_s2 * z
        u = torch.cat((n, (rho * v)[:, None], (internal + kinetic + potential)[:, None]), -1)
        flux = torch.cat(
            (n * v[:, None], (rho * v.square() + p)[:, None], ((u[:, -1] + p) * v)[:, None]), -1
        )
        mu = (
            torch.log(x[:, active].clamp_min(1e-300))
            + q["log_fugacity_coefficients"][:, active]
            + torch.log(p / f.pref)[:, None]
        )
        return {
            "p": p,
            "t": t,
            "x": x,
            "v": v,
            "q": q,
            "u": u,
            "flux": flux,
            "entropy": q["molar_entropy_j_mol_k"] / q["molar_volume_m3_mol"],
            "internal": internal,
            "kinetic": kinetic,
            "potential": potential,
            "mu": mu,
        }

    def _decode(self, y, active):
        c = len(active)
        body_size = self.cells * (c + 2)
        rows = y[:body_size].reshape(self.cells, c + 2)
        body = self._nodes(
            rows[:, :c],
            rows[:, c] * self.fluid.tref,
            rows[:, c + 1] * self.maximum_velocity_m_s,
            self._zc,
            active,
        )
        local = y[body_size:].reshape(2 * self.cells, c)
        zz = torch.stack((self._zf[:-1], self._zf[1:]), -1).flatten()
        faces = self._nodes(
            local, body["t"].repeat_interleave(2), body["v"].repeat_interleave(2), zz, active
        )
        heights = zz - self._zc.repeat_interleave(2)
        hydro = (
            faces["mu"]
            - body["mu"].repeat_interleave(2, 0)
            + self.fluid.molar_masses[active][None, :]
            * self.gravity_m_s2
            * heights[:, None]
            / (R * faces["t"][:, None])
        )
        body["reconstruction"] = faces
        body["hydrostatic_residual"] = hydro.flatten()
        return body

    def _port_data(self, data, side):
        return data["reconstruction"], 0 if side == 0 else -1

    def _faces(self, data, prepared):
        d = data["reconstruction"]
        alpha = self.wave_speed_bound_m_s
        # Each shared interface sees two hydrostatic states at the SAME elevation.
        ul, ur = d["u"][1:-1:2], d["u"][2::2]
        fl, fr = d["flux"][1:-1:2], d["flux"][2::2]
        sl, sr = d["entropy"][1:-1:2], d["entropy"][2::2]
        vl, vr = d["v"][1:-1:2], d["v"][2::2]
        flux = (fl + fr) / 2 - alpha * (ur - ul) / 2
        sf = (sl * vl + sr * vr) / 2 - alpha * (sr - sl) / 2
        if self.boundary_mode == "periodic":
            a, b = d["u"][-1], d["u"][0]
            af, bf = d["flux"][-1], d["flux"][0]
            gs = (d["entropy"][-1] * d["v"][-1] + d["entropy"][0] * d["v"][0]) / 2 - alpha * (
                d["entropy"][0] - d["entropy"][-1]
            ) / 2
            end = (af + bf) / 2 - alpha * (b - a) / 2
            return torch.cat((end[None], flux, end[None])), torch.cat(
                (gs.reshape(1), sf, gs.reshape(1))
            )
        ends = []
        sentropy = []
        f = self.fluid
        for side, (index, bound) in enumerate(zip((0, -1), prepared)):
            interior = d["u"][index]
            v = d["v"][index]
            p = d["p"][index]
            s = d["entropy"][index]
            if bound is None:
                ghost = torch.cat((interior[:-2], (-interior[-2]).reshape(1), interior[-1:]))
                gv = -v
                gp = p
                gs = s
            else:
                spec, q, _ = bound
                gv = v
                gp = f.tensor(spec.pressure_pa)
                rho = q["density_kg_m3"]
                z = self._zf[0 if side == 0 else -1]
                e = (
                    q["molar_internal_energy_j_mol"] / q["molar_volume_m3_mol"]
                    + 0.5 * rho * gv * gv
                    + rho * self.gravity_m_s2 * z
                )
                ghost = torch.cat(
                    (
                        f.tensor(spec.mole_fractions) / q["molar_volume_m3_mol"],
                        (rho * gv).reshape(1),
                        e.reshape(1),
                    )
                )
                gs = q["molar_entropy_j_mol_k"] / q["molar_volume_m3_mol"]
            gf = torch.cat(
                (
                    ghost[:-2] * gv,
                    (ghost[-2] * gv + gp).reshape(1),
                    ((ghost[-1] + gp) * gv).reshape(1),
                )
            )
            if side == 0:
                end = (gf + d["flux"][index]) / 2 - alpha * (interior - ghost) / 2
                se = (gs * gv + s * v) / 2 - alpha * (s - gs) / 2
            else:
                end = (d["flux"][index] + gf) / 2 - alpha * (ghost - interior) / 2
                se = (s * v + gs * gv) / 2 - alpha * (gs - s) / 2
            ends.append(end)
            sentropy.append(se)
        return torch.cat((ends[0][None], flux, ends[1][None])), torch.cat(
            (sentropy[0].reshape(1), sf, sentropy[1].reshape(1))
        )

    def _problem(self, state, dt, prepared, *, face_operator=None):
        f = self.fluid
        active = self._support(state.component_moles)
        old = self._conserved(state)
        scale = self._scales(old)
        initial = f.tensor(state.solution_variables)
        lo, hi = self._bounds(len(active))
        selected = torch.cat((active, active.new_tensor([len(f.names), len(f.names) + 1])))

        def evaluate(y, context=None):
            data = self._decode(y, active)
            details = None
            if face_operator is None:
                flux, sflux = self._faces(data, prepared)
            else:
                flux, sflux, details = face_operator(data, prepared, context)
            drag = torch.zeros_like(data["v"])
            if self.dynamic_viscosity_pa_s:
                drag = darcy_pressure_loss(
                    data["v"],
                    data["q"]["density_kg_m3"],
                    1.0,
                    self.diameter_m,
                    self.dynamic_viscosity_pa_s,
                    self.roughness_m,
                )[0]
            pp = data["reconstruction"]["p"].reshape(self.cells, 2)
            gravity = self.area * (pp[:, 1] - pp[:, 0])
            momentum = gravity - self.volume * drag
            source = torch.cat(
                (
                    torch.zeros_like(old[:, :-2]),
                    momentum[:, None],
                    torch.zeros_like(momentum)[:, None],
                ),
                -1,
            )
            target = old + dt * (self.area * (flux[:-1] - flux[1:]) + source)
            r = (data["u"] * self.volume - target) / scale
            return torch.cat((r[:, selected].flatten(), data["hydrostatic_residual"])), {
                "data": data,
                "target": target,
                "flux": flux,
                "entropy_flux": sflux,
                "drag": drag,
                "gravity_force_n": gravity,
                "face_details": details,
            }

        return initial, lo, hi, evaluate

    def _external_impulse(self, result, dt):
        return dt * result["gravity_force_n"].sum()

    def _solve_nodes(self, initial, temperatures, chemical_targets, active):
        return solve_chemical_nodes(self.fluid, initial, temperatures, chemical_targets, active)

    def _make_state(self, body, local, active):
        y = torch.cat((body.flatten(), local.flatten()))
        if len(y) > 768:
            raise ValueError("Inclined pipe exceeds 768 joint unknowns including reconstructions")
        d = self._decode(y, active)
        u = d["u"] * self.volume
        state = PRMomentumState(
            tuple(map(tuple, u[:, : len(self.fluid.names)].cpu().tolist())),
            tuple(u[:, -2].cpu().tolist()),
            tuple(u[:, -1].cpu().tolist()),
            tuple(y.cpu().tolist()),
            0.0,
            self.initial_step_s,
            self.model_id,
        )
        self.inspect(state)
        return state

    def initialize(self, pressure_pa, temperature_k, mole_fractions, velocity_m_s=0.0):
        # Reject an oversized joint system before expensive phase checks.
        composition = self.fluid.tensor(mole_fractions)
        if composition.ndim == 1:
            composition = composition.expand(self.cells, -1)
        active = self._support(composition)
        if self.cells * (3 * len(active) + 2) > 768:
            raise ValueError("Inclined pipe exceeds joint unknown budget")
        # Reuse input validation without pretending a horizontal state is a restart.
        config = self.to_dict()
        for key in (
            "model",
            "fluid",
            "elevation_change_m",
            "reference_elevation_m",
            "gravity_m_s2",
        ):
            config.pop(key)
        base = PRMomentumPipe(self.fluid, **config)
        s = base.initialize(pressure_pa, temperature_k, mole_fractions, velocity_m_s)
        active = self._support(s.component_moles)
        f = self.fluid
        c = len(active)
        if self.cells * (3 * c + 2) > 768:
            raise ValueError("Inclined pipe exceeds joint unknown budget")
        body = f.tensor(s.solution_variables).reshape(self.cells, c + 2)
        d = base._decode(body.flatten(), active)
        mu = (
            torch.log(d["x"][:, active].clamp_min(1e-300))
            + d["q"]["log_fugacity_coefficients"][:, active]
            + torch.log(d["p"] / f.pref)[:, None]
        )
        heights = torch.stack((self._zf[:-1] - self._zc, self._zf[1:] - self._zc), -1).flatten()
        temps = d["t"].repeat_interleave(2)
        target = mu.repeat_interleave(2, 0) - f.molar_masses[active][
            None, :
        ] * self.gravity_m_s2 * heights[:, None] / (R * temps[:, None])
        local = self._solve_nodes(body[:, :c].repeat_interleave(2, 0), temps, target, active)
        return self._make_state(body, local, active)

    def initialize_equilibrium(self, pressure_pa, temperature_k, mole_fractions):
        """Prescribe P/T/x at the LEFT endpoint and initialize its whole profile."""
        base = self.initialize(pressure_pa, temperature_k, mole_fractions)
        f = self.fluid
        active = self._support(base.component_moles)
        c = len(active)
        body = f.tensor(base.solution_variables[: self.cells * (c + 2)]).reshape(self.cells, c + 2)
        anchor = self._nodes(
            body[:1, :c], body[:1, c] * f.tref, body.new_zeros(1), body.new_zeros(1), active
        )
        zz = torch.cat((self._zc, torch.stack((self._zf[:-1], self._zf[1:]), -1).flatten()))
        target = anchor["mu"].expand(len(zz), c) - f.molar_masses[active][
            None, :
        ] * self.gravity_m_s2 * zz[:, None] / (R * temperature_k)
        initial = body[:1, :c].expand(len(zz), c)
        rows = self._solve_nodes(initial, f.tensor([temperature_k] * len(zz)), target, active)
        body = torch.cat((rows[: self.cells], body[:, c:]), -1)
        return self._make_state(body, rows[self.cells :], active)

    def inspect(self, state):
        if not isinstance(state, PRMomentumState) or state.model_id != self.model_id:
            raise PropertyModelError("Inclined pipe model/state differs")
        from .operation import _digest

        if _digest(self.to_dict()) != self.model_id:
            raise PropertyModelError("Inclined pipe parameters changed")
        key = _digest(state.to_dict())
        if key in self._verified:
            return
        f = self.fluid
        active = self._support(state.component_moles)
        c = len(active)
        y = f.tensor(state.solution_variables)
        if (
            len(y) != self.cells * (3 * c + 2)
            or len(y) > 768
            or len(state.momentum_kg_m_s) != self.cells
            or len(state.total_energy_j) != self.cells
        ):
            raise ValueError("Inclined pipe reconstruction dimensions differ")
        if not self.minimum_step_s <= state.next_step_s <= self.maximum_step_s:
            raise ValueError("Inclined restart step outside domain")
        lo, hi = self._bounds(c)
        if not bool(((y >= lo) & (y <= hi)).all().cpu()):
            raise PropertyModelError("Inclined primitive outside domain")
        d = self._decode(y, active)
        stock = self._conserved(state)
        if not bool(
            (
                torch.isfinite(d["u"]).all()
                & (((d["u"] * self.volume - stock).abs() / self._scales(stock)).max() < 1e-9)
                & (d["hydrostatic_residual"].abs().max() < 1e-9)
            ).cpu()
        ):
            raise ConservationError("Inclined EOS, PE or face reconstruction differs from restart")
        self._validate_data(d)
        self._validate_data(d["reconstruction"])
        self._remember(state)

    def describe(self, state):
        result = super().describe(state)
        d = self._decode(
            self.fluid.tensor(state.solution_variables), self._support(state.component_moles)
        )
        result["potential_energy_j"] = (self.volume * d["potential"]).cpu().tolist()
        result["relative_elevation_m"] = self._zc.cpu().tolist()
        return result
