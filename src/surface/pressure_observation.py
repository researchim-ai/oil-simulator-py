"""Pressure between static taps, distinct from distributed friction.

Constant-density, steady, one-direction measurement operator. The caller must
specify the two tap areas and the differential impulse-line hydrostatic term.
This is not a compressible network or a model of liquid injection losses.
"""

import math
from dataclasses import dataclass

import torch

from .liquid import _number
from .pr_active_pipe_friction import darcy_factor


@dataclass(frozen=True)
class PressureTapSpan:
    length_m: float
    diameter_m: float
    inlet_area_m2: float
    outlet_area_m2: float
    elevation_change_m: float
    minor_loss_coefficient: float
    impulse_line_difference_pa: float
    roughness_m: float = 0.0
    gravity_m_s2: float = 9.80665

    def __post_init__(self):
        for field in ("length_m", "diameter_m", "inlet_area_m2", "outlet_area_m2", "gravity_m_s2"):
            _number(getattr(self, field), field, positive=True)
        for field in ("minor_loss_coefficient", "roughness_m"):
            _number(getattr(self, field), field, nonnegative=True)
        for field in ("elevation_change_m", "impulse_line_difference_pa"):
            _number(getattr(self, field), field)
        if self.roughness_m / self.diameter_m > 0.05:
            raise ValueError("Relative roughness exceeds .05")

    def predict(
        self, volume_flow_m3_s, density_kg_m3, viscosity_pa_s, *, distributed_friction_pa=None
    ):
        """Return p_in-p_out and sensor reading; K uses velocity in diameter_m.

        distributed_friction_pa may supply an independently computed wet gas
        friction loss. It must exclude gas hydrostatics, acceleration and local
        losses. Impulse-line difference is added to p_in-p_out, with sign given
        by the actual line geometry/fluid. Zero must be explicitly specified.
        """
        ref = next(
            (
                x
                for x in (volume_flow_m3_s, density_kg_m3, viscosity_pa_s, distributed_friction_pa)
                if isinstance(x, torch.Tensor)
            ),
            None,
        )
        device = ref.device if ref is not None else "cpu"
        q, rho, mu = torch.broadcast_tensors(
            *[
                torch.as_tensor(v, dtype=torch.float64, device=device)
                for v in (volume_flow_m3_s, density_kg_m3, viscosity_pa_s)
            ]
        )
        if not bool(
            (
                torch.isfinite(q)
                & torch.isfinite(rho)
                & torch.isfinite(mu)
                & (q >= 0)
                & (rho > 0)
                & (mu > 0)
            )
            .all()
            .cpu()
        ):
            raise ValueError("Finite nonnegative forward flow and positive properties required")
        d = self.diameter_m
        area = math.pi * d * d / 4
        u = q / area
        re = rho * u * d / mu
        if distributed_friction_pa is None:
            factor = darcy_factor(re, self.roughness_m / d)
            # Exact Poiseuille expression keeps the zero-flow derivative finite.
            friction = torch.where(
                re <= 2000,
                128 * mu * self.length_m * q / (math.pi * d**4),
                factor * self.length_m / d * rho * u.square() / 2,
            )
        else:
            friction = torch.as_tensor(distributed_friction_pa, dtype=q.dtype, device=q.device)
            friction, q = torch.broadcast_tensors(friction, q)
            if not bool((torch.isfinite(friction) & (friction >= 0)).all().cpu()):
                raise ValueError("Distributed friction must be finite and nonnegative")
        minor = self.minor_loss_coefficient * rho * u.square() / 2
        hydro = rho * self.gravity_m_s2 * self.elevation_change_m
        kinetic = rho * ((q / self.outlet_area_m2).square() - (q / self.inlet_area_m2).square()) / 2
        static = friction + minor + hydro + kinetic
        result = {
            "distributed_friction_pa": friction,
            "minor_losses_pa": minor,
            "hydrostatic_pa": hydro,
            "kinetic_change_pa": kinetic,
            "static_tap_difference_pa": static,
            "sensor_difference_pa": static + self.impulse_line_difference_pa,
        }
        if not bool(torch.stack([torch.isfinite(v).all() for v in result.values()]).all().cpu()):
            raise ValueError("Pressure observation exceeds finite representable domain")
        return result
