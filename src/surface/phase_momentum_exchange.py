"""Batched finite phase inertia with symmetric drag and stationary walls.

This is a local source operator, not an algebraic slip correlation or a pipe
transport discretization. Masses are fixed during this mechanical subproblem.
"""

from dataclasses import dataclass, field

import torch

from .liquid import _number


def symmetric_coefficients(values, count, name):
    rows = tuple(tuple(_number(v, name, nonnegative=True) for v in row) for row in values)
    if len(rows) != count or any(len(r) != count for r in rows):
        raise ValueError(f"{name} must have one row/column per phase")
    if any(rows[i][i] != 0 for i in range(count)) or any(
        rows[i][j] != rows[j][i] for i in range(count) for j in range(count)
    ):
        raise ValueError(f"{name} must be symmetric with zero diagonal")
    return rows


@dataclass(frozen=True)
class PhaseMomentumExchange:
    pair_drag_kg_s: tuple
    wall_drag_kg_s: tuple
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        count = len(self.wall_drag_kg_s)
        if count not in (2, 3):
            raise ValueError("Two or three finite phases required")
        object.__setattr__(
            self, "pair_drag_kg_s", symmetric_coefficients(self.pair_drag_kg_s, count, "pair drag")
        )
        object.__setattr__(
            self,
            "wall_drag_kg_s",
            tuple(_number(v, "wall drag", nonnegative=True) for v in self.wall_drag_kg_s),
        )

    @staticmethod
    def coordinates(masses, velocity):
        mean = (masses * velocity).sum(-1, keepdim=True) / masses.sum(-1, keepdim=True)
        return torch.cat((mean, velocity[..., 1:] - velocity[..., :1]), dim=-1)

    def tensors(self, reference):
        key = (reference.device, reference.dtype)
        if key not in self._cache:
            n = len(self.wall_drag_kg_s)
            self._cache[key] = (
                reference.new_tensor(self.pair_drag_kg_s),
                reference.new_tensor(self.wall_drag_kg_s),
                torch.eye(n, device=reference.device, dtype=reference.dtype)[:, 1:],
            )
        return self._cache[key]

    def basis(self, masses):
        n = masses.shape[-1]
        if n != len(self.wall_drag_kg_s):
            raise ValueError("Momentum phase registry differs")
        ident = self.tensors(masses)[2]
        return ident - (masses[..., None, 1:] / masses.sum(-1)[..., None, None])

    def kinematics(self, masses, coordinates):
        fluctuation = (self.basis(masses) @ coordinates[..., 1:, None]).squeeze(-1)
        return coordinates[..., :1] + fluctuation, fluctuation

    def terms(self, masses, old_velocity, velocity, dt, *, coordinates=None):
        """No host reads; supports arbitrary batch axes before the phase axis."""
        n = len(self.wall_drag_kg_s)
        if (
            masses.shape != old_velocity.shape
            or masses.shape != velocity.shape
            or masses.shape[-1] != n
        ):
            raise ValueError("Matching (..., phases) inertia coordinates required")
        drag, wall, _ = self.tensors(velocity)
        if coordinates is None:
            coordinates = self.coordinates(masses, velocity)
        velocity, fluctuation = self.kinematics(masses, coordinates)
        old_coordinates = self.coordinates(masses, old_velocity)
        _, old_fluctuation = self.kinematics(masses, old_coordinates)
        difference = fluctuation[..., :, None] - fluctuation[..., None, :]
        change = coordinates[..., :1] - old_coordinates[..., :1] + fluctuation - old_fluctuation
        pair_force = -drag * difference
        wall_force = -wall * velocity
        impulse = dt * (pair_force.sum(-1) + wall_force)
        physical = dt * (0.5 * (drag * difference.square()).sum(-1) + wall * velocity.square())
        numerical = 0.5 * masses * change.square()
        residual = masses * change - impulse
        return {
            "momentum_residual": residual,
            "pair_force_n": pair_force,
            "wall_force_n": wall_force,
            "physical_dissipation_j": physical,
            "temporal_dissipation_j": numerical,
            "internal_energy_gain_j": physical + numerical,
            "velocity_coordinates": coordinates,
            "valid": (
                torch.isfinite(masses).all(-1)
                & (masses > 0).all(-1)
                & torch.isfinite(old_velocity).all(-1)
                & torch.isfinite(velocity).all(-1)
                & torch.isfinite(residual).all(-1)
                & torch.isfinite(torch.as_tensor(dt, device=masses.device, dtype=masses.dtype))
                & (torch.as_tensor(dt, device=masses.device, dtype=masses.dtype) > 0)
            ),
        }

    def solve(self, masses, old_velocity, dt):
        """Backward Euler. No iteration, SciPy, or implicit CPU fallback.

        In a wall-free cell one row is the exact barycentric momentum
        constraint, avoiding the stiff Laplacian's constant-velocity null mode.
        """
        if masses.shape != old_velocity.shape or masses.shape[-1] != len(self.wall_drag_kg_s):
            raise ValueError("Matching (..., phases) masses and velocities required")
        if (
            masses.dtype != torch.float64
            or old_velocity.dtype != masses.dtype
            or old_velocity.device != masses.device
        ):
            raise ValueError("Matching float64 tensors on one device required")
        k, w, _ = self.tensors(masses)
        basis = self.basis(masses)
        # Keep relative velocities as independent coordinates. Adding a tiny
        # slip to a large bulk speed must not erase the finite drag impulse.
        lb = (k[..., :, :, None] * (basis[..., :, None, :] - basis[..., None, :, :])).sum(-2)
        sub = masses[..., :, None] * basis + dt * (lb + w[..., :, None] * basis)
        first = masses + dt * w
        matrix = torch.cat((first[..., :, None], sub), dim=-1)
        old_coordinates = self.coordinates(masses, old_velocity)
        _, old_fluctuation = self.kinematics(masses, old_coordinates)
        right = masses * old_fluctuation - dt * w * old_coordinates[..., :1]
        global_row = torch.cat(
            (masses.sum(-1, keepdim=True) + dt * w.sum(), dt * (w[..., :, None] * basis).sum(-2)),
            dim=-1,
        )
        matrix = torch.cat((global_row[..., None, :], matrix[..., 1:, :]), dim=-2)
        right = torch.cat((-dt * w.sum() * old_coordinates[..., :1], right[..., 1:]), dim=-1)
        scale = matrix.abs().amax(-1).clamp_min(torch.finfo(masses.dtype).tiny)
        coordinates = torch.linalg.solve(matrix / scale[..., :, None], right / scale)
        coordinates = torch.cat(
            (coordinates[..., :1] + old_coordinates[..., :1], coordinates[..., 1:]), dim=-1
        )
        velocity, _ = self.kinematics(masses, coordinates)
        return velocity, self.terms(masses, old_velocity, velocity, dt, coordinates=coordinates)
