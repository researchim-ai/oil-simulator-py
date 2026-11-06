"""Explicit dissipative Euler gas map, in specific J/kg (no gravity factor).

A bounded quasi-steady dry-gas constitutive model, not a surge model or an
extension of a measured forward compressor card. Windage belongs to the drive.
"""

import math
from dataclasses import dataclass

import torch

from .liquid import _number
from .peng_robinson import R


@dataclass(frozen=True)
class SignedGasMap:
    reference_speed_rad_s: float
    maximum_speed_rad_s: float
    maximum_flow_m3_s: float
    euler_coefficient: float
    linear_coupling: float
    flow_loss_coefficient: float

    def __post_init__(self):
        for key in ("reference_speed_rad_s", "maximum_speed_rad_s", "maximum_flow_m3_s"):
            _number(getattr(self, key), key, positive=True)
        for key in ("euler_coefficient", "linear_coupling", "flow_loss_coefficient"):
            _number(getattr(self, key), key, nonnegative=True)

        if self.reference_speed_rad_s > self.maximum_speed_rad_s:
            raise ValueError("Reference speed exceeds signed map envelope")

    def evaluate(self, flow, speed, density):
        speed = torch.as_tensor(speed, dtype=flow.dtype, device=flow.device)
        density = torch.as_tensor(density, dtype=flow.dtype, device=flow.device)
        coupling = self.euler_coefficient * speed.abs() + self.linear_coupling
        head = coupling * speed - self.flow_loss_coefficient * flow * flow.abs()
        torque = density * coupling * flow
        power = torque * speed
        hydraulic = density * flow * head
        dissipation = density * self.flow_loss_coefficient * flow.abs().pow(3)
        valid = (
            (flow.abs() <= self.maximum_flow_m3_s)
            & (speed.abs() <= self.maximum_speed_rad_s)
            & (density > 0)
            & torch.isfinite(head)
            & torch.isfinite(power)
            & torch.isfinite(dissipation)
        )
        return head, torque, power, hydraulic, dissipation, valid


def prepare_signed(bundle, unit, a, b, offset, enabled, state):
    f = bundle.model.fluid
    old = state.nodes[a][1]
    other = state.nodes[b][1]
    speed = dict(getattr(state, "rotors", ())).get(unit.name, unit.speed_rad_s)
    # A zero-Q seed makes parallel identical quadratic losses rank deficient.
    # Use only the explicit signed map and an ideal-gas PS estimate to seed
    # the branch; the actual closure remains the real-PR Newton equations.
    flow = 0.0
    if enabled and hasattr(old, "pressure_pa") and unit.signed_map.flow_loss_coefficient > 0:
        x = [v / sum(old.component_moles) for v in old.component_moles]
        cp = sum(v * c.ideal_cp_j_mol_k for v, c in zip(x, f.components))
        mw = sum(v * c.molar_mass_kg_mol for v, c in zip(x, f.components))
        estimate = (
            cp / mw * old.temperature_k * ((other.pressure_pa / old.pressure_pa) ** (R / cp) - 1)
        )
        euler = (
            unit.signed_map.euler_coefficient * abs(speed) + unit.signed_map.linear_coupling
        ) * speed
        delta = euler - estimate
        flow = math.copysign(
            min(
                unit.signed_map.maximum_flow_m3_s * 0.9,
                math.sqrt(abs(delta) / unit.signed_map.flow_loss_coefficient),
            ),
            delta,
        )
    donor = old if flow >= 0 else other
    bundle.initial.append(
        f.tensor([flow, donor.temperature_k / f.tref, donor.temperature_k / f.tref])
    )
    bundle.lower.append(
        f.tensor(
            [
                -unit.signed_map.maximum_flow_m3_s,
                f.min_temperature_k / f.tref,
                f.min_temperature_k / f.tref,
            ]
        )
    )
    bundle.upper.append(
        f.tensor(
            [
                unit.signed_map.maximum_flow_m3_s,
                f.max_temperature_k / f.tref,
                f.max_temperature_k / f.tref,
            ]
        )
    )
    return (
        unit,
        a,
        b,
        offset,
        enabled,
        None,
        None,
        None,
        f.tensor(unit.reference_mole_fractions),
        max(1.0, old.temperature_k * R),
    )


def evaluate_signed(bundle, entry, y, data, shaft_speeds):
    unit, a, b, offset, enabled, _, _, _, reference, scale = entry
    f = bundle.model.fluid
    flow, tis, tout = y[offset], y[offset + 1] * f.tref, y[offset + 2] * f.tref
    # Each donor branch is smooth away from Q=0. A parent solver may freeze
    # donor_sign and retry the other branch when a composition jump destabilizes Newton.
    frozen = bundle.donor_signs.get(unit.name)
    forward = flow >= 0 if frozen is None else flow.new_tensor(frozen > 0, dtype=torch.bool)
    sign = torch.where(forward, flow.new_tensor(1.0), flow.new_tensor(-1.0))
    left, right = data[a], data[b]

    def choose(key):
        return torch.where(forward, left[key], right[key])

    x = torch.where(forward, left["x"][0], right["x"][0])
    hin = torch.where(
        forward, left["q"]["molar_enthalpy_j_mol"][0], right["q"]["molar_enthalpy_j_mol"][0]
    )
    sin = torch.where(
        forward, left["q"]["molar_entropy_j_mol_k"][0], right["q"]["molar_entropy_j_mol_k"][0]
    )
    rho = torch.where(forward, left["q"]["density_kg_m3"][0], right["q"]["density_kg_m3"][0])
    mw = (x * f.molar_masses).sum()
    speed = (shaft_speeds or {}).get(unit.name, unit.speed_rad_s)
    head, torque, power, _hydraulic, dissipation, valid = unit.signed_map.evaluate(flow, speed, rho)
    receiver = torch.where(forward, right["p"], left["p"])
    stage = receiver
    qi = f.tensor_homogeneous(stage, tis, x, "vapor")
    qo = f.tensor_homogeneous(receiver, tout, x, "vapor")
    coupling = (
        unit.signed_map.euler_coefficient
        * torch.as_tensor(speed, dtype=y.dtype, device=y.device).abs()
        + unit.signed_map.linear_coupling
    )
    actual_work = sign * coupling * speed
    row = torch.stack(
        (
            (qi["molar_enthalpy_j_mol"] - hin - mw * sign * head) / scale,
            (qi["molar_entropy_j_mol_k"] - sin) / R,
            (qo["molar_enthalpy_j_mol"] - hin - mw * actual_work) / scale,
        )
    )
    if not enabled:
        zero = flow * 0
        row = torch.stack((flow, (tis - choose("t")) / f.tref, (tout - choose("t")) / f.tref))
        return (
            {},
            row,
            {
                "volume_rate_m3_s": zero,
                "mass_rate_kg_s": zero,
                "shaft_power_w": zero,
                "load_torque_nm": zero,
                "entropy_production_w_k": zero,
                "isentropic_temperature_k": tis,
                "delivery_temperature_k": tout,
                "stage_pressure_pa": stage,
                "isentropic_head_j_kg": zero,
                "dissipation_w": zero,
            },
            [],
            y.new_ones((), dtype=torch.bool),
        )
    molar = rho * flow / mw
    hout = qo["molar_enthalpy_j_mol"]
    # Signed fluxes at physical ends; only the receiving end carries delivery h.
    ha = torch.where(forward, hin, hout)
    hb = torch.where(forward, hout, hin)
    phia = bundle.model.gravity_m_s2 * (
        bundle.model.nodes[a].elevation_m - bundle.model.reference_elevation_m
    )
    phib = bundle.model.gravity_m_s2 * (
        bundle.model.nodes[b].elevation_m - bundle.model.reference_elevation_m
    )

    def flux(h, phi):
        return torch.cat((molar * x, y.new_zeros(1), (molar * (h + mw * phi)).reshape(1)))

    entropy = molar.abs() * (qo["molar_entropy_j_mol_k"] - sin)
    donor_single = torch.where(
        forward,
        y.new_tensor(len(left["phase_n"]) == 1, dtype=torch.bool),
        y.new_tensor(len(right["phase_n"]) == 1, dtype=torch.bool),
    )
    z = torch.where(
        forward, left["q"]["compressibility_factor"][0], right["q"]["compressibility_factor"][0]
    )
    valid = (
        valid
        & (sign * flow >= 0)
        & donor_single
        & ((x - reference).abs().sum() <= unit.maximum_composition_departure)
        & (z >= unit.minimum_compressibility_factor)
        & qi["valid"]
        & qo["valid"]
        & (qi["compressibility_factor"] >= unit.minimum_compressibility_factor)
        & (qo["compressibility_factor"] >= unit.minimum_compressibility_factor)
        & (entropy >= -1e-8)
    )
    if unit.maximum_shaft_power_w is not None:
        valid = valid & (power.abs() <= unit.maximum_shaft_power_w)
    record = {
        "speed_rad_s": torch.as_tensor(speed, dtype=y.dtype, device=y.device),
        "volume_rate_m3_s": flow,
        "mass_rate_kg_s": rho * flow,
        "shaft_power_w": power,
        "load_torque_nm": torque,
        "entropy_production_w_k": entropy,
        "isentropic_temperature_k": tis,
        "delivery_temperature_k": tout,
        "stage_pressure_pa": stage,
        "isentropic_head_j_kg": head,
        "dissipation_w": dissipation,
    }
    donor_p = choose("p")
    return (
        {a: -flux(ha, phia), b: flux(hb, phib)},
        row,
        record,
        [(donor_p, choose("t"), stage, tis, x), (stage, tis, receiver, tout, x)],
        valid,
    )
