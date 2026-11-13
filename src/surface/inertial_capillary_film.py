"""Joint height/momentum capillary film, staggered finite volumes in Torch.

Nonlinear constant-density shallow film, plug momentum profile and optional
Poiseuille/Navier wall drag. Periodic planar, energetically neutral wetting;
prescribed tangential gravity and interface shear have signed work ledgers.
Exterior pressure prescribed, no gas inertia. This is a mechanical kernel,
not a compressible PR/network coupling or a turbulent three-phase closure.
"""

import math
from dataclasses import asdict, dataclass

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .planar_film import resistance_response
from .thin_film_dynamics import ThinFilmDynamics


@dataclass(frozen=True)
class InertialCapillaryState:
    thickness_m: tuple
    face_velocity_m_s: tuple
    time_s: float
    physical_heat_j: float
    numerical_heat_j: float
    model_id: str
    external_work_j: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "thickness_m", tuple(self.thickness_m))
        object.__setattr__(self, "face_velocity_m_s", tuple(self.face_velocity_m_s))

    def to_dict(self):
        result = asdict(self)
        if self.external_work_j == 0.0:
            result.pop("external_work_j")
        return result

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class InertialCapillaryFilm:
    def __init__(
        self,
        surface,
        *,
        viscous_drag,
        maximum_unknowns=256,
        maximum_velocity_m_s=10.0,
        time_scheme="auto",
        device="cpu",
    ):
        self.surface = ThinFilmDynamics.from_dict(surface, device=device)
        s = self.surface
        if len(s.shape) != 1 or s.angle != 0.0:
            raise ValueError("Joint inertial kernel requires a periodic 1D energy-neutral film")
        if type(viscous_drag) is not bool:
            raise ValueError("Explicit viscous_drag boolean required")
        if (
            type(maximum_unknowns) is not int
            or maximum_unknowns < 8
            or 2 * s.shape[0] > maximum_unknowns
        ):
            raise ValueError("Joint dense solve size exceeds configured bound")
        self.drag = viscous_drag
        self.maximum_unknowns = maximum_unknowns
        self.maximum_velocity = _number(maximum_velocity_m_s, "velocity envelope", positive=True)
        if time_scheme not in ("auto", "roe", "backward"):
            raise ValueError("Known joint time scheme required")
        self.time_scheme = time_scheme
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "inertial_capillary_film_v1",
            "surface": self.surface.to_dict(),
            "viscous_drag": self.drag,
            "maximum_unknowns": self.maximum_unknowns,
            "maximum_velocity_m_s": self.maximum_velocity,
            "time_scheme": self.time_scheme,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        raw = dict(data)
        if raw.pop("model", None) != "inertial_capillary_film_v1":
            raise ValueError("Unknown inertial capillary model")
        return cls(**raw, device=device)

    @staticmethod
    def face(h):
        return (h + torch.roll(h, -1, 0)) / 2

    @staticmethod
    def center(v):
        return (v + torch.roll(v, 1, 0)) / 2

    def initialize(self, thickness_m, face_velocity_m_s=None, *, time_s=0.0):
        s = self.surface
        h = s.tensor(thickness_m)
        u = s.tensor([0.0] * s.shape[0] if face_velocity_m_s is None else face_velocity_m_s)
        state = InertialCapillaryState(
            tuple(h.cpu().tolist()), tuple(u.cpu().tolist()), time_s, 0.0, 0.0, self.model_id
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if state.model_id != self.model_id or _digest(self.to_dict()) != self.model_id:
            raise ValueError("Joint capillary checkpoint identity differs")
        s = self.surface
        h = s.tensor(state.thickness_m)
        u = s.tensor(state.face_velocity_m_s)
        s._validate_h(h)
        if tuple(u.shape) != s.shape or not bool(
            (torch.isfinite(u) & (u.abs() <= self.maximum_velocity)).all().cpu()
        ):
            raise PropertyModelError("Invalid mechanical face velocity")
        for value, name in (
            (state.time_s, "clock"),
            (state.physical_heat_j, "physical heat"),
            (state.numerical_heat_j, "numerical heat"),
        ):
            _number(value, name, nonnegative=True)
        _number(state.external_work_j, "signed external mechanical work")
        values = {
            "liquid_volume_m3": float((h.sum() * s.cell_area).cpu()),
            "momentum_kg_m_s": float((s.rho * s.cell_area * (self.face(h) * u).sum()).cpu()),
            "kinetic_energy_j": float(
                (0.5 * s.rho * s.cell_area * (self.face(h) * u.square()).sum()).cpu()
            ),
            "potential_energy_j": float(s.free_energy(h).cpu()),
        }
        if not all(math.isfinite(v) for v in values.values()):
            raise PropertyModelError("Nonfinite derived capillary inventory or energy")
        return values

    def step(
        self, state, dt_s, *, normal_load=None, tangential_gravity_m_s2=0.0, interface_shear_pa=0.0
    ):
        self.inspect(state)
        dt = _number(dt_s, "joint capillary step", positive=True)
        end_time = state.time_s + dt
        if not math.isfinite(end_time) or end_time <= state.time_s:
            raise ValueError("Joint capillary clock cannot represent this interval")
        dt = end_time - state.time_s
        gravity = _number(tangential_gravity_m_s2, "tangential gravity")
        shear = _number(interface_shear_pa, "prescribed interface shear")
        s = self.surface
        h = self.face(s.tensor(state.thickness_m))
        stiff = self.drag and bool((dt * s.mu / (s.rho * h * (h / 3 + s.slip)) > 2).any().cpu())
        backward = self.time_scheme == "backward" or (self.time_scheme == "auto" and stiff)
        new, report = self._step(state, dt, backward, normal_load, gravity, shear)
        if self.time_scheme == "auto" and self.drag and not backward:
            mid = (h + self.face(s.tensor(new.thickness_m))) / 2
            if bool((dt * s.mu / (s.rho * mid * (mid / 3 + s.slip)) > 2).any().cpu()):
                return self._step(state, dt, True, normal_load, gravity, shear)
        return new, report

    def _step(self, state, dt_s, backward, normal_load=None, gravity=0.0, shear=0.0):
        before = self.inspect(state)
        dt = _number(dt_s, "joint capillary step", positive=True)
        s = self.surface
        h0 = s.tensor(state.thickness_m)
        u0 = s.tensor(state.face_velocity_m_s)
        hf0 = self.face(h0)
        n = len(h0)
        hs = h0.mean()
        mu0 = -s.sigma * s.laplacian(h0) + s.rho * s.normal_g * h0
        load0 = h0 * 0 if normal_load is None else normal_load.pressure_and_energy(h0, h0 * 0)[0]
        kick = dt * s.gradient(mu0 + load0, 0).abs().max() / s.rho
        if gravity != 0 or shear != 0:
            kick = torch.maximum(kick, dt * (abs(gravity) + 1.5 * abs(shear) / (s.rho * hf0.min())))
        us = torch.maximum(torch.maximum(u0.abs().max(), kick), s.tensor(1e-12))

        def evaluate(z):
            dh = z[:n] * hs
            du = z[n:] * us
            h = h0 + dh
            u = u0 + du
            hm = h0 + dh / 2
            hf = self.face(hm)
            if backward:
                ut = u
            else:
                root_new = torch.sqrt(self.face(h))
                ut = u0 + root_new / (root_new + torch.sqrt(hf0)) * du
            mass_flux = hf * ut
            adv_flux = self.center(mass_flux) * self.center(ut)
            chemical = mu0 - s.sigma / 2 * s.laplacian(dh) + s.rho * s.normal_g * dh / 2
            if normal_load is not None:
                chemical = chemical + normal_load.pressure_and_energy(h0, dh)[0]
            traction = hf * s.gradient(chemical, 0)
            if self.drag:
                eta, beta, resistance = resistance_response(hf, s.mu, s.slip)
            else:
                eta, beta, resistance = hf * 0 + 1, hf * 0, hf * 0
            forcing = s.rho * hf * gravity + eta * shear
            rm = (dh + dt * s.divergence(mass_flux, 0)) / hs
            rp = (
                self.face(dh) * u
                + hf0 * du
                + dt * s.gradient(adv_flux, 0)
                + dt / s.rho * (traction + resistance * ut - forcing)
            ) / (hs * us)
            return (
                torch.cat((rm, rp)),
                h,
                u,
                mass_flux,
                adv_flux,
                traction,
                resistance,
                ut,
                eta,
                beta,
            )

        z = torch.zeros(2 * n, dtype=h0.dtype, device=h0.device)
        iterations = 0
        for iterations in range(31):
            r, *_ = evaluate(z)
            norm = float(r.abs().max().cpu())
            if norm <= 2e-12:
                break
            if iterations == 30 or not math.isfinite(norm):
                raise HydraulicConvergenceError("Joint capillary Newton failed")
            try:
                delta = torch.linalg.solve(torch.func.jacfwd(lambda w: evaluate(w)[0])(z), -r)
            except RuntimeError as e:
                raise HydraulicConvergenceError("Joint capillary linear solve failed") from e
            accepted = False
            for power in range(24):
                trial = z + delta * 2.0 ** (-power)
                if (
                    bool((torch.isfinite(trial).all() & (h0 + trial[:n] * hs > 0).all()).cpu())
                    and float(evaluate(trial)[0].abs().max().cpu()) < norm
                ):
                    z = trial
                    accepted = True
                    break
            if not accepted:
                raise HydraulicConvergenceError("Positive joint capillary line search failed")
        _, h, u, mass_flux, adv_flux, traction, resistance, ut, eta, beta = evaluate(z)
        hf = self.face(h0 + (h - h0) / 2)
        interface_velocity = eta * ut + beta * shear
        face_interface_work = dt * s.cell_area * shear * interface_velocity
        face_gravity_work = dt * s.cell_area * s.rho * hf * gravity * ut
        face_external_work = face_interface_work + face_gravity_work
        external_work = face_external_work.sum()
        s._validate_h(h)
        physical = dt * s.cell_area * (resistance * ut.square() + beta * shear**2).sum()
        numerical = (
            0.5 * s.rho * s.cell_area * (hf0 * (z[n:] * us).square()).sum()
            if backward
            else physical * 0
        )
        if not bool(
            (
                torch.isfinite(physical)
                & torch.isfinite(numerical)
                & (physical >= 0)
                & (numerical >= 0)
            ).cpu()
        ):
            raise ConservationError("Nonfinite capillary heat budget")
        end = InertialCapillaryState(
            tuple(h.cpu().tolist()),
            tuple(u.cpu().tolist()),
            state.time_s + dt,
            state.physical_heat_j + float(physical.cpu()),
            state.numerical_heat_j + float(numerical.cpu()),
            self.model_id,
            state.external_work_j + float(external_work.cpu()),
        )
        after = self.inspect(end)
        de = (
            after["kinetic_energy_j"]
            - before["kinetic_energy_j"]
            + after["potential_energy_j"]
            - before["potential_energy_j"]
        )
        load_pressure, load_energy = (
            (h0 * 0, h0 * 0)
            if normal_load is None
            else normal_load.pressure_and_energy(h0, z[:n] * hs)
        )
        if not bool((torch.isfinite(load_pressure) & torch.isfinite(load_energy)).all().cpu()):
            raise ConservationError("Nonfinite normal pressure work")
        load_impulse = (
            -dt
            * s.cell_area
            * (self.face(h0 + z[:n] * hs / 2) * s.gradient(load_pressure, 0)).sum()
        )
        energy_residual = de + float(
            (physical + numerical + load_energy.sum() - external_work).cpu()
        )
        face_wall_impulse = dt * s.cell_area * (-resistance * ut + (eta - 1) * shear)
        face_interface_impulse = torch.full_like(hf, dt * s.cell_area * shear)
        face_gravity_impulse = dt * s.cell_area * s.rho * hf * gravity
        impulse = face_wall_impulse.sum()
        momentum_residual = (
            after["momentum_kg_m_s"]
            - before["momentum_kg_m_s"]
            - float(
                (
                    impulse
                    + load_impulse
                    + face_interface_impulse.sum()
                    + face_gravity_impulse.sum()
                ).cpu()
            )
        )
        budget = (
            abs(before["kinetic_energy_j"])
            + abs(after["kinetic_energy_j"])
            + abs(before["potential_energy_j"])
            + abs(after["potential_energy_j"])
            + float((physical + numerical).cpu())
            + float(load_energy.abs().sum().cpu())
            + float(face_external_work.abs().sum().cpu())
        )
        if abs(energy_residual) > 1e-22 + 2e-9 * budget:
            raise ConservationError("Joint capillary energy/work balance failed")
        momentum_scale = (
            abs(before["momentum_kg_m_s"])
            + abs(after["momentum_kg_m_s"])
            + float((impulse.abs() + load_impulse.abs()).cpu())
            + float((face_interface_impulse.abs().sum() + face_gravity_impulse.abs().sum()).cpu())
        )
        if abs(momentum_residual) > 1e-22 + 2e-10 * momentum_scale:
            raise ConservationError("Joint capillary momentum balance failed")
        if (
            abs(after["liquid_volume_m3"] - before["liquid_volume_m3"])
            > 1e-25 + 1e-11 * before["liquid_volume_m3"]
        ):
            raise ConservationError("Joint capillary volume balance failed")
        # Local energy flux from the staggered product rules. The last term
        # converts dual-cell kinetic energy to the material-cell partition;
        # no cumulative/global residual is redistributed into this flux.
        dh = z[:n] * hs
        du = z[n:] * us
        gradient_mid = s.gradient(h0, 0) + s.gradient(dh, 0) / 2
        chemical_mid = mu0 - s.sigma / 2 * s.laplacian(dh) + s.rho * s.normal_g * dh / 2
        chemical_mid = chemical_mid + load_pressure
        kinetic_center = 0.5 * s.rho * torch.roll(ut, 1, 0) * ut * self.center(mass_flux)
        shift_change = (
            0.125
            * s.rho
            * s.spacing[0] ** 2
            * (s.gradient(h0, 0) * du * (2 * u0 + du) + s.gradient(dh, 0) * u.square())
        )
        energy_flux = (
            self.face(kinetic_center)
            + self.face(chemical_mid) * mass_flux
            - s.sigma * gradient_mid * self.face(dh) / dt
            + shift_change / dt
        )

        def cell_energy(hh, uu):
            gg = s.gradient(hh, 0)
            return (
                0.25 * s.rho * hh * (uu.square() + torch.roll(uu, 1, 0).square())
                + 0.25 * s.sigma * (gg.square() + torch.roll(gg, 1, 0).square())
                + 0.5 * s.rho * s.normal_g * hh.square()
            )

        face_physical = dt * s.cell_area * (resistance * ut.square() + beta * shear**2)
        face_numerical = (
            0.5 * s.rho * s.cell_area * hf0 * du.square() if backward else face_physical * 0
        )
        cell_physical = self.center(face_physical)
        cell_numerical = self.center(face_numerical)
        cell_external_work = self.center(face_external_work)
        ce0, ce1 = cell_energy(h0, u0), cell_energy(h, u)
        local_residual = (
            s.cell_area * (ce1 - ce0 + dt * s.divergence(energy_flux, 0))
            + cell_physical
            + cell_numerical
            + load_energy
            - cell_external_work
        )
        local_budget = (
            s.cell_area * (ce1.abs() + ce0.abs())
            + dt * s.width * (energy_flux.abs() + torch.roll(energy_flux, 1, 0).abs())
            + cell_physical
            + cell_numerical
            + load_energy.abs()
            + cell_external_work.abs()
        )
        if not bool((local_residual.abs() <= 1e-23 + 2e-9 * local_budget).all().cpu()):
            raise ConservationError("Joint local capillary energy flux balance failed")
        momentum_shift = (
            0.25 * s.rho * s.spacing[0] ** 2 * (s.gradient(h0, 0) * du + s.gradient(dh, 0) * u)
        )
        material_momentum_flux = s.rho * self.face(adv_flux) + momentum_shift / dt
        cell_force = self.center(-s.cell_area * traction)
        cell_wall_impulse = self.center(face_wall_impulse)
        cell_external_impulse = self.center(face_interface_impulse + face_gravity_impulse)
        material_m0 = s.rho * s.cell_area * h0 * self.center(u0)
        material_m1 = s.rho * s.cell_area * h * self.center(u)
        local_momentum_residual = (
            material_m1
            - material_m0
            + dt * s.cell_area * s.divergence(material_momentum_flux, 0)
            - dt * cell_force
            - cell_wall_impulse
            - cell_external_impulse
        )
        momentum_budget = (
            material_m1.abs()
            + material_m0.abs()
            + dt
            * s.width
            * (material_momentum_flux.abs() + torch.roll(material_momentum_flux, 1, 0).abs())
            + dt * cell_force.abs()
            + cell_wall_impulse.abs()
            + cell_external_impulse.abs()
        )
        if not bool((local_momentum_residual.abs() <= 1e-23 + 2e-9 * momentum_budget).all().cpu()):
            raise ConservationError("Material-cell capillary impulse flux balance failed")
        return end, {
            "dt_s": dt,
            "newton_iterations": iterations,
            "time_scheme": "backward" if backward else "roe",
            "face_time_velocity_m_s": ut.cpu().tolist(),
            "state": end.to_dict(),
            "face_volume_flux_m2_s": mass_flux.cpu().tolist(),
            "face_mechanical_energy_flux_w": (energy_flux * s.width).cpu().tolist(),
            "cell_physical_heat_j": cell_physical.cpu().tolist(),
            "cell_numerical_heat_j": cell_numerical.cpu().tolist(),
            "cell_external_work_j": cell_external_work.cpu().tolist(),
            "cell_external_impulse_kg_m_s": cell_external_impulse.cpu().tolist(),
            "external_work_j": float(external_work.cpu()),
            "gravity_work_j": float(face_gravity_work.sum().cpu()),
            "interface_work_j": float(face_interface_work.sum().cpu()),
            "interface_velocity_m_s": interface_velocity.cpu().tolist(),
            "gravity_impulse_kg_m_s": float(face_gravity_impulse.sum().cpu()),
            "interface_impulse_kg_m_s": float(face_interface_impulse.sum().cpu()),
            "cell_energy_residual_j": local_residual.cpu().tolist(),
            "face_material_momentum_flux_n": (material_momentum_flux * s.width).cpu().tolist(),
            "cell_potential_force_n": cell_force.cpu().tolist(),
            "cell_wall_impulse_kg_m_s": cell_wall_impulse.cpu().tolist(),
            "cell_momentum_residual_kg_m_s": local_momentum_residual.cpu().tolist(),
            "momentum_flux_m3_s2": adv_flux.cpu().tolist(),
            "capillary_force_n": (-s.cell_area * traction).cpu().tolist(),
            "wall_impulse_kg_m_s": float(impulse.cpu()),
            "normal_load_impulse_kg_m_s": float(load_impulse.cpu()),
            "cell_normal_load_potential_change_j": load_energy.cpu().tolist(),
            "cell_normal_pressure_pa": load_pressure.cpu().tolist(),
            "physical_heat_j": float(physical.cpu()),
            "numerical_heat_j": float(numerical.cpu()),
            "kinetic_energy_change_j": after["kinetic_energy_j"] - before["kinetic_energy_j"],
            "potential_energy_change_j": after["potential_energy_j"] - before["potential_energy_j"],
            "energy_residual_j": energy_residual,
            "momentum_residual_kg_m_s": momentum_residual,
        }

    def advance(
        self,
        state,
        duration_s,
        *,
        initial_step_s,
        minimum_step_s=1e-12,
        maximum_attempts=10000,
        tangential_gravity_m_s2=0.0,
        interface_shear_pa=0.0,
    ):
        self.inspect(state)
        duration = _number(duration_s, "joint interval", positive=True)
        maximum = _number(initial_step_s, "joint maximum step", positive=True)
        minimum = _number(minimum_step_s, "joint minimum step", positive=True)
        if minimum > maximum or type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Ordered steps and integer attempt bound required")
        target = state.time_s + duration
        if not math.isfinite(target) or target <= state.time_s:
            raise ValueError("Joint capillary clock cannot represent this report interval")
        dt = maximum
        reports = []
        rejected = 0
        while state.time_s < target:
            if len(reports) + rejected >= maximum_attempts:
                raise HydraulicConvergenceError("Joint capillary attempt limit")
            trial = min(dt, target - state.time_s)
            try:
                new, r = self.step(
                    state,
                    trial,
                    tangential_gravity_m_s2=tangential_gravity_m_s2,
                    interface_shear_pa=interface_shear_pa,
                )
            except (HydraulicConvergenceError, PropertyModelError, ConservationError):
                dt = trial / 2
                rejected += 1
                if dt < minimum:
                    raise
                continue
            state = new
            reports.append(r)
            dt = min(maximum, trial * 1.5)
        return state, {"steps": reports, "rejected_steps": rejected}
