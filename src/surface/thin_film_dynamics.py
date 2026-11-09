"""Isothermal long-wave film on a periodic planar wall patch, in SI units.

Discrete-gradient free energy and frozen positive face mobility give a
checkable work/dissipation identity. Adsorption thickness is an explicit
material length, not an invisible numerical inventory floor. This model
resolves capillary/wetting modes, not inertial turbulent annular waves.
"""

import math
from dataclasses import asdict, dataclass

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest


@dataclass(frozen=True)
class ThinFilmState:
    thickness_m: tuple
    shape: tuple
    time_s: float
    dissipated_energy_j: float
    external_work_j: float
    model_id: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            **{**data, "thickness_m": tuple(data["thickness_m"]), "shape": tuple(data["shape"])}
        )


class ThinFilmDynamics:
    """One/two dimensional resolved wall film; no PR-network bridge implied."""

    def __init__(
        self,
        shape,
        lengths_m,
        surface_tension_n_m,
        viscosity_pa_s,
        density_kg_m3,
        adsorption_thickness_m,
        equilibrium_contact_angle_deg,
        slip_length_m,
        *,
        transverse_width_m=1.0,
        maximum_slope=0.2,
        normal_gravity_m_s2=0.0,
        maximum_unknowns=512,
        device="cpu",
    ):
        self.shape = tuple(shape)
        if not 1 <= len(self.shape) <= 2 or any(type(n) is not int or n < 4 for n in self.shape):
            raise ValueError("Periodic wall patch needs one/two axes with >=4 cells each")
        if (
            type(maximum_unknowns) is not int
            or maximum_unknowns < 4
            or math.prod(self.shape) > maximum_unknowns
        ):
            raise ValueError("Explicit dense-solve size bound exceeded")
        self.lengths = tuple(_number(v, "patch length", positive=True) for v in lengths_m)
        if len(self.lengths) != len(self.shape):
            raise ValueError("One length per wall coordinate required")
        self.sigma = _number(surface_tension_n_m, "surface tension", positive=True)
        self.mu = _number(viscosity_pa_s, "viscosity", positive=True)
        self.rho = _number(density_kg_m3, "density", positive=True)
        self.adsorption = _number(
            adsorption_thickness_m, "material adsorption length", positive=True
        )
        self.angle = _number(equilibrium_contact_angle_deg, "contact angle", nonnegative=True)
        if self.angle > 20:
            raise ValueError("Long-wave wetting potential requires contact angle <=20 degrees")
        self.slip = _number(slip_length_m, "Navier slip length", nonnegative=True)
        self.width = _number(transverse_width_m, "transverse width", positive=True)
        if len(self.shape) == 2 and self.width != 1.0:
            raise ValueError("2D patch has explicit area; do not apply a second width")
        self.slope = _number(maximum_slope, "maximum slope", positive=True)
        if self.slope > 0.2:
            raise ValueError("Long-wave slope envelope <=.2 required")
        self.normal_g = _number(normal_gravity_m_s2, "normal gravity")
        self.maximum_unknowns = maximum_unknowns
        self.device = torch.device(device)
        self.spacing = tuple(l / n for l, n in zip(self.lengths, self.shape))
        self.cell_area = math.prod(self.spacing) * (self.width if len(self.shape) == 1 else 1)
        self.wetting_scale = 10 / 3 * self.sigma * (1 - math.cos(math.radians(self.angle)))
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "periodic_long_wave_film_v1",
            "shape": self.shape,
            "lengths_m": self.lengths,
            "surface_tension_n_m": self.sigma,
            "viscosity_pa_s": self.mu,
            "density_kg_m3": self.rho,
            "adsorption_thickness_m": self.adsorption,
            "equilibrium_contact_angle_deg": self.angle,
            "slip_length_m": self.slip,
            "transverse_width_m": self.width,
            "maximum_slope": self.slope,
            "normal_gravity_m_s2": self.normal_g,
            "maximum_unknowns": self.maximum_unknowns,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        d = dict(data)
        if d.pop("model", None) != "periodic_long_wave_film_v1":
            raise ValueError("Unknown film model")
        return cls(**d, device=device)

    def tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float64, device=self.device)

    def gradient(self, h, axis):
        return (torch.roll(h, -1, axis) - h) / self.spacing[axis]

    def divergence(self, flux, axis):
        return (flux - torch.roll(flux, 1, axis)) / self.spacing[axis]

    def laplacian(self, h):
        return sum(self.divergence(self.gradient(h, k), k) for k in range(len(self.shape)))

    def potential(self, h):
        r = self.adsorption / h
        return (
            self.wetting_scale * (-r.square() / 2 + r**5 / 5)
            + self.rho * self.normal_g * h.square() / 2
        )

    def potential_divided_difference(self, a, b):
        # Polynomial identity avoids cancellation and the a==b removable pole.
        ra, rb = self.adsorption / a, self.adsorption / b
        fifth = sum(ra**k * rb ** (4 - k) for k in range(5))
        return (
            self.wetting_scale / self.adsorption * ra * rb * ((ra + rb) / 2 - fifth / 5)
            + self.rho * self.normal_g * (a + b) / 2
        )

    def free_energy(self, h):
        density = self.potential(h) + self.sigma / 2 * sum(
            self.gradient(h, k).square() for k in range(len(self.shape))
        )
        return density.sum() * self.cell_area

    def _validate_h(self, h):
        if tuple(h.shape) != self.shape or not bool((torch.isfinite(h) & (h > 0)).all().cpu()):
            raise PropertyModelError("Positive finite physical film thickness required")
        if any(
            not bool((self.gradient(h, k).abs() <= self.slope).all().cpu())
            for k in range(len(self.shape))
        ):
            raise PropertyModelError("Film outside long-wave slope envelope")

    def initialize(self, thickness_m):
        h = self.tensor(thickness_m).reshape(self.shape)
        self._validate_h(h)
        state = ThinFilmState(
            tuple(h.flatten().cpu().tolist()), self.shape, 0.0, 0.0, 0.0, self.model_id
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            state.model_id != self.model_id
            or tuple(state.shape) != self.shape
            or _digest(self.to_dict()) != self.model_id
        ):
            raise ValueError("Film checkpoint identity differs")
        _number(state.time_s, "film clock", nonnegative=True)
        _number(state.dissipated_energy_j, "film dissipation", nonnegative=True)
        _number(state.external_work_j, "film work")
        h = self.tensor(state.thickness_m).reshape(self.shape)
        self._validate_h(h)
        values = {
            "liquid_volume_m3": float((h.sum() * self.cell_area).cpu()),
            "free_energy_j": float(self.free_energy(h).cpu()),
            "minimum_thickness_m": float(h.min().cpu()),
            "maximum_thickness_m": float(h.max().cpu()),
        }
        if not all(math.isfinite(v) for v in values.values()):
            raise PropertyModelError("Nonfinite derived film inventory or free energy")
        return values

    def step(self, state, dt_s, *, tangential_gravity_m_s2=None, interface_shear_pa=None):
        self.inspect(state)
        dt = _number(dt_s, "film step", positive=True)
        end_time = state.time_s + dt
        if not math.isfinite(end_time) or end_time <= state.time_s:
            raise ValueError("Film clock cannot represent this interval")
        dt = end_time - state.time_s
        ndim = len(self.shape)
        gravity = tuple(tangential_gravity_m_s2 or (0.0,) * ndim)
        shear = tuple(interface_shear_pa or (0.0,) * ndim)
        if len(gravity) != ndim or len(shear) != ndim:
            raise ValueError("One driving component per wall coordinate required")
        for v in (*gravity, *shear):
            _number(v, "film driving")
        old = self.tensor(state.thickness_m).reshape(self.shape)
        # Gravity/shear flux is explicit; capillary/wetting pressure is implicit.
        # Bound the explicit characteristic speed independently of positivity.
        courant = sum(
            dt
            / self.spacing[k]
            * (
                (
                    (old.square() + 2 * self.slip * old) * self.rho * gravity[k]
                    + (old + self.slip) * shear[k]
                )
                / self.mu
            )
            .abs()
            .max()
            for k in range(ndim)
        )
        if not bool((courant <= 0.25).cpu()):
            raise HydraulicConvergenceError("Explicit film driving Courant bound exceeded")
        scale = old.mean()
        mobility = old**3 / (3 * self.mu) + self.slip * old.square() / self.mu
        shear_mobility = old.square() / (2 * self.mu) + self.slip * old / self.mu
        mf = [(mobility + torch.roll(mobility, -1, k)) / 2 for k in range(ndim)]
        sf = [(shear_mobility + torch.roll(shear_mobility, -1, k)) / 2 for k in range(ndim)]
        interface_mobility = (old + self.slip) / self.mu
        cf = [(interface_mobility + torch.roll(interface_mobility, -1, k)) / 2 for k in range(ndim)]
        forcing = [
            mf[k] * self.rho * gravity[k]
            + (shear_mobility + torch.roll(shear_mobility, -1, k)) / 2 * shear[k]
            for k in range(ndim)
        ]

        def evaluate(z):
            new = z.reshape(self.shape) * scale
            chemical = -self.sigma * self.laplacian(
                (new + old) / 2
            ) + self.potential_divided_difference(new, old)
            gradients = [self.gradient(chemical, k) for k in range(ndim)]
            fluxes = [-mf[k] * gradients[k] + forcing[k] for k in range(ndim)]
            residual = (
                new - old + dt * sum(self.divergence(fluxes[k], k) for k in range(ndim))
            ) / scale
            return residual.flatten(), new, chemical, gradients, fluxes

        z = (old / scale).flatten().clone()
        tolerance = 2e-12
        iterations = 0
        for iterations in range(31):
            r, *_ = evaluate(z)
            norm = float(r.abs().max().cpu())
            if norm <= tolerance:
                break
            if iterations == 30 or not math.isfinite(norm):
                raise HydraulicConvergenceError("Film discrete-gradient Newton failed; reduce step")
            jacobian = torch.func.jacfwd(lambda w: evaluate(w)[0])(z)
            try:
                delta = torch.linalg.solve(jacobian, -r)
            except RuntimeError as e:
                raise HydraulicConvergenceError("Film linear solve failed") from e
            accepted = False
            for power in range(24):
                trial = z + delta * 2.0 ** (-power)
                if bool((torch.isfinite(trial) & (trial > 0)).all().cpu()):
                    rr = evaluate(trial)[0]
                    if float(rr.abs().max().cpu()) < norm:
                        z, accepted = trial, True
                        break
            if not accepted:
                raise HydraulicConvergenceError("Positive film Newton line search failed")
        r, new, _chemical, gradients, fluxes = evaluate(z)
        self._validate_h(new)
        forces = [self.rho * gravity[k] - gradients[k] for k in range(ndim)]
        interface_velocities = [sf[k] * forces[k] + cf[k] * shear[k] for k in range(ndim)]
        # Positive Schur form of the integrated Stokes dissipation. Averaging
        # two wall heights adds a weighted variance to the local Schur term.
        local_schur = old * (old + 4 * self.slip) / (4 * self.mu * (old + 3 * self.slip))
        ratio = shear_mobility / mobility
        dissipation_density = []
        for k in range(ndim):
            rf = sf[k] / mf[k]
            schur = (local_schur + torch.roll(local_schur, -1, k)) / 2 + (
                mobility * (ratio - rf).square()
                + torch.roll(mobility, -1, k) * (torch.roll(ratio, -1, k) - rf).square()
            ) / 2
            dissipation_density.append(
                mf[k] * (forces[k] + rf * shear[k]).square() + schur * shear[k] ** 2
            )
        dissipation = dt * self.cell_area * sum(d.sum() for d in dissipation_density)
        shear_work = (
            dt
            * self.cell_area
            * sum((shear[k] * interface_velocities[k]).sum() for k in range(ndim))
        )
        gravity_work = (
            dt
            * self.cell_area
            * sum((self.rho * gravity[k] * fluxes[k]).sum() for k in range(ndim))
        )
        work = shear_work + gravity_work
        energy_change = self.free_energy(new) - self.free_energy(old)
        defect = energy_change + dissipation - work
        budget = (
            self.free_energy(new).abs() + self.free_energy(old).abs() + dissipation + work.abs()
        )
        if not bool(
            (
                torch.isfinite(defect)
                & torch.isfinite(budget)
                & torch.isfinite(dissipation)
                & torch.isfinite(work)
            ).cpu()
        ):
            raise ConservationError("Nonfinite film work or dissipation budget")
        if abs(float(defect.cpu())) > 1e-20 + 2e-9 * float(budget.cpu()):
            raise ConservationError("Film free-energy/work/dissipation balance failed")
        volume_defect = (new.sum() - old.sum()) * self.cell_area
        if abs(float(volume_defect.cpu())) > 1e-25 + 1e-11 * float(
            (old.sum() * self.cell_area).cpu()
        ):
            raise ConservationError("Film volume balance failed")
        end = ThinFilmState(
            tuple(new.flatten().cpu().tolist()),
            self.shape,
            state.time_s + dt,
            state.dissipated_energy_j + float(dissipation.cpu()),
            state.external_work_j + float(work.cpu()),
            self.model_id,
        )
        return end, {
            "dt_s": dt,
            "newton_iterations": iterations,
            "volume_residual_m3": float(volume_defect.cpu()),
            "free_energy_change_j": float(energy_change.cpu()),
            "dissipation_j": float(dissipation.cpu()),
            "external_work_j": float(work.cpu()),
            "interface_work_j": float(shear_work.cpu()),
            "gravity_work_j": float(gravity_work.cpu()),
            "interface_velocity_m_s": [u.cpu().tolist() for u in interface_velocities],
            "face_dissipation_j": [
                (dt * self.cell_area * d).cpu().tolist() for d in dissipation_density
            ],
            "energy_residual_j": float(defect.cpu()),
            "face_volume_flux_m2_s": [q.cpu().tolist() for q in fluxes],
        }

    def advance(
        self,
        state,
        duration_s,
        *,
        initial_step_s,
        minimum_step_s=1e-12,
        maximum_steps=10000,
        **driving,
    ):
        duration = _number(duration_s, "film interval", positive=True)
        dt = _number(initial_step_s, "initial step", positive=True)
        minimum = _number(minimum_step_s, "minimum step", positive=True)
        end_time = state.time_s + duration
        if not math.isfinite(end_time) or end_time <= state.time_s:
            raise ValueError("Film clock cannot represent this report interval")
        reports, rejected = [], 0
        while state.time_s < end_time:
            if len(reports) + rejected >= maximum_steps:
                raise HydraulicConvergenceError("Film interval attempt limit")
            trial = min(dt, end_time - state.time_s)
            try:
                new, report = self.step(state, trial, **driving)
            except (HydraulicConvergenceError, PropertyModelError, ConservationError):
                dt = trial / 2
                rejected += 1
                if dt < minimum:
                    raise
                continue
            report["state"] = new.to_dict()
            reports.append(report)
            state = new
            dt = min(initial_step_s, trial * 1.5)
        return state, {"steps": reports, "rejected_steps": rejected}
