"""Explicit dissipative signed pump closure; coefficients require calibration.

Positive head raises pressure left-to-right; positive torque is shaft load.
This reduced closure is not a measured universal pump or compressor map.
"""

from dataclasses import dataclass

import torch

from .liquid import _number

G = 9.80665


@dataclass(frozen=True)
class SignedPumpMap:
    reference_speed_rad_s: float
    maximum_speed_rad_s: float
    maximum_flow_m3_s: float
    euler_head_coefficient_m_s2: float
    flow_loss_coefficient_s2_m5: float
    viscous_torque_coefficient_m4_s: float = 0.0
    linear_head_coefficient_m_s: float = 0.0

    def __post_init__(self):
        for key in ("reference_speed_rad_s", "maximum_speed_rad_s", "maximum_flow_m3_s"):
            _number(getattr(self, key), key, positive=True)
        for key in (
            "euler_head_coefficient_m_s2",
            "flow_loss_coefficient_s2_m5",
            "viscous_torque_coefficient_m4_s",
            "linear_head_coefficient_m_s",
        ):
            _number(getattr(self, key), key, nonnegative=True)
        if self.reference_speed_rad_s > self.maximum_speed_rad_s:
            raise ValueError("Reference speed exceeds signed map envelope")

    def evaluate(self, flow, speed, density):
        """Torch-only head, load torque, shaft/hydraulic/dissipated power, validity.

        h = (a*|omega| + c)*omega - b*q*|q|
        T = rho*g*((a*|omega| + c)*q + d*omega).
        Thus T*omega-rho*g*h*q = rho*g*(d*omega²+b*|q|³) >= 0.
        The optional linear coupling permits finite fluid-driven torque at rest.
        """
        omega = torch.as_tensor(speed, dtype=flow.dtype, device=flow.device)
        rho = torch.as_tensor(density, dtype=flow.dtype, device=flow.device)
        coupling = self.euler_head_coefficient_m_s2 * omega.abs() + self.linear_head_coefficient_m_s
        head = coupling * omega - self.flow_loss_coefficient_s2_m5 * flow * flow.abs()
        torque = rho * G * (coupling * flow + self.viscous_torque_coefficient_m4_s * omega)
        hydraulic = rho * G * head * flow
        shaft = torque * omega
        dissipation = (
            rho
            * G
            * (
                self.viscous_torque_coefficient_m4_s * omega.square()
                + self.flow_loss_coefficient_s2_m5 * flow.abs().pow(3)
            )
        )
        valid = (
            (flow.abs() <= self.maximum_flow_m3_s)
            & (omega.abs() <= self.maximum_speed_rad_s)
            & (rho > 0)
            & torch.isfinite(head)
            & torch.isfinite(torque)
            & torch.isfinite(shaft)
            & torch.isfinite(hydraulic)
            & torch.isfinite(dissipation)
        )
        return head, torque, shaft, hydraulic, dissipation, valid
