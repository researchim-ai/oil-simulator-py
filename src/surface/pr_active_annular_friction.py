"""Opt-in fully wetted laminar annulus with a gas core, frozen-coefficient BE.

The interface velocity is reconstructed from the radial Stokes profile; it is
NOT the mean liquid velocity. Eliminating that profile gives a symmetric
positive resistance matrix in the two mean velocities. Interface work is
transferred between phases, wall work is zero, and lost kinetic energy heats
the fluids. No claim of experimental qualification of the Wallis correlation.
"""

import math
from dataclasses import replace

import torch

from .annular_film import interfacial_conductance
from .liquid import ConservationError, PropertyModelError, _number
from .pr_active_pipe_friction import darcy_factor
from .pr_homogeneous_uv import recover_homogeneous_phases
from .pr_stock_balances import entropy_allowance

MODEL = "laminar_annular_wallis_interface_v1"


class PRActiveAnnularFriction:
    def __init__(self, viscosity_pa_s, gas_slot, liquid_slot, roughness_m=0.0, wetting=None):
        if gas_slot == liquid_slot or set(viscosity_pa_s) != {gas_slot, liquid_slot}:
            raise ValueError("Annular closure requires distinct gas/liquid viscosity slots")
        self.viscosity = {
            n: _number(v, "phase viscosity", positive=True)
            for n, v in sorted(viscosity_pa_s.items())
        }
        self.gas_slot, self.liquid_slot = gas_slot, liquid_slot
        self.roughness = _number(roughness_m, "wall roughness", nonnegative=True)
        from .film_wetting import FilmWettingCriterion

        self.wetting = None if wetting is None else FilmWettingCriterion(**wetting)

    @classmethod
    def from_dict(cls, data):
        raw = dict(data)
        if raw.pop("model", None) != MODEL:
            raise ValueError("Unknown annular closure")
        return cls(**raw)

    def to_dict(self):
        return {
            "model": MODEL,
            "gas_slot": self.gas_slot,
            "liquid_slot": self.liquid_slot,
            "viscosity_pa_s": self.viscosity,
            "roughness_m": self.roughness,
            **({"wetting": self.wetting.to_dict()} if self.wetting is not None else {}),
        }

    def validate_pipe(self, pipe):
        roots = {s["name"]: s["root"] for s in pipe.slots}
        if roots != {self.gas_slot: "vapor", self.liquid_slot: "liquid"}:
            raise ValueError("Annular closure requires declared gas vapor and liquid roots")
        if pipe.elevations is None or any(
            abs(b - a) < 0.95 * length
            for a, b, length in zip(pipe.elevations, pipe.elevations[1:], pipe.lengths)
        ):
            raise ValueError("Annular closure is restricted to surveyed near-vertical segments")

    def apply(self, pipe, state, dt_s):
        dt = _number(dt_s, "friction step", positive=True)
        self.validate_pipe(pipe)
        old = pipe.inspect(state)
        f = pipe.fluid
        flat = [(i, name, p) for i, cell in enumerate(state.cells) for name, p in cell]
        locations = {(i, name): j for j, (i, name, _) in enumerate(flat)}
        n = f.tensor([p.component_moles for _, _, p in flat])
        volume = f.tensor([p.volume_m3 for _, _, p in flat])
        mass = (n * f.molar_masses).sum(-1)
        momentum = f.tensor([p.momentum_kg_m_s for _, _, p in flat])
        energy = f.tensor([p.total_energy_j for _, _, p in flat])
        velocity = momentum / mass
        new_velocity = velocity.clone()
        new_energy = energy.clone()
        physical_heat = torch.zeros_like(mass)
        interface_work = torch.zeros_like(mass)
        paired = [i for i, cell in enumerate(state.cells) if len(cell) == 2]
        diagnostics = []
        wetting_records = []
        if paired:
            gi = torch.tensor([locations[i, self.gas_slot] for i in paired], device=f.device)
            li = torch.tensor([locations[i, self.liquid_slot] for i in paired], device=f.device)
            alpha = volume[li] / f.tensor([pipe.volumes[i] for i in paired])
            rg, rl = mass[gi] / volume[gi], mass[li] / volume[li]
            mu = self.viscosity[self.liquid_slot]
            film_re = (
                mass[li]
                / f.tensor([pipe.lengths[i] for i in paired])
                * velocity[li].abs()
                / (math.pi * pipe.diameter * mu)
            )
            if not bool(((alpha > 0) & (alpha <= 0.3) & (rg < rl) & (film_re <= 400)).all().cpu()):
                raise PropertyModelError(
                    "Annular state outside alpha_l (0,.3], density or laminar film domain"
                )
            effective, eta, beta, c, perimeter, gas_re = interfacial_conductance(
                alpha,
                pipe.diameter,
                mu,
                self.viscosity[self.gas_slot],
                rg,
                velocity[gi],
                velocity[li],
            )
            if not bool(((beta > 0) & (c > 0) & torch.isfinite(eta)).all().cpu()):
                raise ConservationError("Nonpassive annular resistance")
            lengths = f.tensor([pipe.lengths[i] for i in paired])
            # tau = effective*(ug-eta*ul), coefficients frozen at old slip.
            k = lengths * perimeter * effective
            wall = lengths * c
            mg, ml = mass[gi], mass[li]
            share = dt * k / (mg + dt * k)
            # Positive Schur form avoids subtracting two large stiff terms.
            ul = (ml * velocity[li] + eta * mg * share * velocity[gi]) / (
                ml + dt * wall + eta.square() * mg * share
            )
            ug = mg / (mg + dt * k) * velocity[gi] + share * eta * ul
            film_re_after = ml / lengths * ul.abs() / (math.pi * pipe.diameter * mu)
            if not bool((torch.isfinite(film_re_after) & (film_re_after <= 400)).all().cpu()):
                raise PropertyModelError(
                    "Annular trial leaves laminar film domain Gamma/mu<=400; reduce step or use a qualified regime model"
                )
            tau = effective * (ug - eta * ul)
            ui = eta * ul + beta * tau
            if self.wetting is not None:
                diagnostic = self.wetting.diagnose(alpha, pipe.diameter, mu, rl, ul, tau)
                if self.wetting.require_full_wetting and not bool(
                    diagnostic["full_wetting_supported"].all().cpu()
                ):
                    raise PropertyModelError(
                        "Full annular wetting is not supported by the declared contact angles/profile"
                    )
                data = {k: v.cpu().tolist() for k, v in diagnostic.items()}
                wetting_records = [
                    {"cell": i, **{k: v[j] for k, v in data.items()}} for j, i in enumerate(paired)
                ]
            power = lengths * perimeter * tau * ui
            new_velocity[gi], new_velocity[li] = ug, ul
            new_energy[gi], new_energy[li] = energy[gi] - dt * power, energy[li] + dt * power
            interface_work[gi], interface_work[li] = -dt * power, dt * power
            physical_heat[gi] = dt * lengths * perimeter * tau * (ug - ui)
            physical_heat[li] = dt * (
                wall * ul.square() + lengths * perimeter * beta * tau.square()
            )
            packed = (
                torch.stack(
                    (alpha, eta, beta, c, effective, tau, ui, film_re, gas_re, film_re_after), -1
                )
                .cpu()
                .tolist()
            )
            diagnostics = [
                dict(
                    zip(
                        (
                            "cell",
                            "alpha_l",
                            "eta",
                            "beta",
                            "wall_resistance_per_m",
                            "effective_shear",
                            "tau_pa",
                            "interface_velocity_m_s",
                            "film_reynolds",
                            "gas_driving_reynolds",
                            "film_reynolds_after",
                        ),
                        (i, *row),
                    )
                )
                for i, row in zip(paired, packed)
            ]
        single = [j for j, (i, _, _) in enumerate(flat) if len(state.cells[i]) == 1]
        if single:
            idx = torch.tensor(single, device=f.device)
            mu = f.tensor([self.viscosity[flat[j][1]] for j in single])
            rho = mass[idx] / volume[idx]
            u = velocity[idx]
            reynolds = rho * u.abs() * pipe.diameter / mu
            factor = darcy_factor(reynolds, self.roughness / pipe.diameter)
            turbulent = 2 * u / (1 + torch.sqrt(1 + 2 * dt * factor / pipe.diameter * u.abs()))
            laminar = u / (1 + dt * 32 * mu / (rho * pipe.diameter**2))
            un = torch.where(reynolds <= 2000, laminar, turbulent)
            new_velocity[idx] = un
            # Remove BE temporal dissipation to report actual drag work.
            physical_heat[idx] = mass[idx] * un * (u - un)
        new_momentum = mass * new_velocity
        temporal = 0.5 * mass * (new_velocity - velocity).square()
        heat = (
            new_energy - energy + 0.5 * mass * (velocity - new_velocity) * (velocity + new_velocity)
        )
        allowance = 1e-10 + 1e-10 * (physical_heat.abs() + temporal)
        if not bool(
            (
                torch.isfinite(heat).all()
                & (physical_heat >= -allowance).all()
                & (
                    (heat - physical_heat - temporal).abs() <= allowance + 1e-10 * energy.abs()
                ).all()
            ).cpu()
        ):
            raise ConservationError("Annular viscous heating or interface energy balance failed")
        phases = recover_homogeneous_phases(
            f,
            n,
            volume,
            new_momentum,
            new_energy,
            [pipe.slots[pipe.names.index(name)]["root"] for _, name, _ in flat],
            [p.temperature_k for _, _, p in flat],
        )
        cells = [[] for _ in state.cells]
        for (i, name, _), p in zip(flat, phases):
            cells[i].append((name, p))
        new = replace(state, cells=tuple(tuple(cell) for cell in cells))
        current = pipe.inspect(new)
        production = current["entropy"] - old["entropy"]
        if production < -entropy_allowance(current["entropy"], old["entropy"]):
            raise ConservationError("Annular source decreased total entropy")
        impulse = float((new_momentum - momentum).sum().cpu())
        return new, {
            "model": MODEL,
            **({"wetting": wetting_records} if self.wetting is not None else {}),
            "state": new.to_dict(),
            "dt_s": dt,
            "wall_impulse_kg_m_s": impulse,
            "entropy_production_j_k": production,
            "active_locations": [(i, name) for i, name, _ in flat],
            "dissipation_j": heat.cpu().tolist(),
            "physical_dissipation_j": physical_heat.cpu().tolist(),
            "temporal_dissipation_j": temporal.cpu().tolist(),
            "interface_work_j": interface_work.cpu().tolist(),
            "annular_cells": diagnostics,
            "conservation_residuals": pipe.ledger(state, new, impulse),
        }
