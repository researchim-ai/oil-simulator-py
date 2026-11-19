"""Explicit low-Mach, adiabatic, zero-holdup Darcy links between stirred vessels.

This is a reduced model, not an automatic replacement for transient pipes.
Finite endpoint inventories remain in the common conservative PR solve.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .darcy_hydraulics import darcy_pressure_loss
from .liquid import EquipmentOperatingError, _name, _number


@dataclass(frozen=True)
class PRDarcyConnection:
    name: str
    source: str
    target: str
    length_m: float
    diameter_m: float
    dynamic_viscosity_pa_s: float
    roughness_m: float = 0.0
    local_loss_coefficient: float = 1.0
    maximum_velocity_m_s: float = 100.0
    maximum_pressure_drop_fraction: float = 0.03
    maximum_mach_number: float = 0.1
    allow_reverse: bool = True
    open: bool = True

    def __post_init__(self):
        for key in ("name", "source", "target"):
            _name(getattr(self, key))
        if self.source == self.target:
            raise ValueError("Darcy link requires distinct finite vessels")
        for key in ("length_m", "diameter_m", "dynamic_viscosity_pa_s", "maximum_velocity_m_s"):
            _number(getattr(self, key), key, positive=True)
        _number(self.roughness_m, "roughness", nonnegative=True)
        _number(self.local_loss_coefficient, "local loss coefficient", nonnegative=True)
        if self.roughness_m >= self.diameter_m / 2:
            raise ValueError("Invalid Darcy roughness")
        if not 0 < self.maximum_pressure_drop_fraction <= 0.05:
            raise ValueError("Darcy low-pressure-drop envelope must be <= 5%")
        if not 0 < self.maximum_mach_number <= 0.1:
            raise ValueError("Darcy low-Mach envelope must be <= 0.1")
        if type(self.allow_reverse) is not bool or type(self.open) is not bool:
            raise ValueError("Explicit Darcy isolation and reverse modes required")

    def velocity(self, pressure_drop, density):
        # Poiseuille plus the local velocity-head loss is a quadratic with
        # a cancellation-free inverse. Classify by its own resulting Re;
        # transition/turbulent requests retain the original friction solve.
        linear = 32 * self.dynamic_viscosity_pa_s * self.length_m / self.diameter_m**2
        laminar = (
            2
            * pressure_drop
            / (
                linear
                + torch.sqrt(
                    linear**2 + 2 * self.local_loss_coefficient * density * pressure_drop.abs()
                )
            )
        )
        eligible = (
            density.detach() * laminar.detach().abs() * self.diameter_m
            <= 2000 * self.dynamic_viscosity_pa_s
        ) & (laminar.detach().abs() <= self.maximum_velocity_m_s)
        # vmap carries different primal branches in one tensor. Its generic
        # path remains fully tensorized; ordinary and AD calls can avoid all
        # root iterations when their primals are laminar.
        try:
            all_laminar = bool(eligible.all().cpu())
        except RuntimeError:
            all_laminar = False
        if all_laminar:
            return laminar

        def loss(v, rho):
            distributed = darcy_pressure_loss(
                v,
                rho,
                self.length_m,
                self.diameter_m,
                self.dynamic_viscosity_pa_s,
                self.roughness_m,
            )[0]
            # Ideal entry and a fully mixed receiving vessel: one exit
            # velocity head in the same plug-flow convention as the PR pipe.
            return distributed + 0.5 * self.local_loss_coefficient * rho * v * v.abs()

        # Bounded device solve of the existing friction law. Its comparisons
        # are not differentiated; the implicit correction supplies exact
        # first derivatives at the converged root, including density changes.
        dp, rho = pressure_drop.detach(), density.detach()
        low = torch.zeros_like(dp)
        high = torch.full_like(dp, self.maximum_velocity_m_s)
        for _ in range(44):
            middle = (low + high) / 2
            below = loss(middle, rho) < dp.abs()
            low, high = torch.where(below, middle, low), torch.where(below, high, middle)
        root = ((low + high) / 2 * dp.sign()).detach()
        value, slope = torch.func.jvp(lambda v: loss(v, density), (root,), (torch.ones_like(root),))
        return root + (pressure_drop - value) / slope.clamp_min(1e-30)


class DarcyMaterial:
    external_ports = ()

    def __init__(self, graph, state, openings, external=None):
        self.graph, self.external = graph, external
        t, f = graph.transport, graph.transport.fluid
        self.entries = [
            (c, t.names.index(c.source), t.names.index(c.target))
            for c in graph.darcy_connections
            if openings[c.name]
        ]
        self.internal_edges = [(a, b) for _, a, b in self.entries]
        self.bidirectional_edges = [(a, b) for c, a, b in self.entries if c.allow_reverse]
        self.directed_edges = [(a, b) for c, a, b in self.entries if not c.allow_reverse]
        if external is not None:
            self.external_ports = external.external_ports
        supports = [{i for i, v in enumerate(s.component_moles) if v > 0} for _, s in state.nodes]
        if external is not None:
            supports = [s | set(a.cpu().tolist()) for s, a in zip(supports, external.active)]
        for _c, a, b in self.entries:
            for i in (a, b):
                if len(state.nodes[i][1].phase_roots) != 1:
                    raise EquipmentOperatingError("Darcy link requires single-phase inlet vessels")
            left, right = state.nodes[a][1], state.nodes[b][1]
            dp = left.pressure_pa - right.pressure_pa
            if not _c.allow_reverse:
                dp = max(0.0, dp)
            if abs(dp) > _c.maximum_pressure_drop_fraction * min(
                left.pressure_pa, right.pressure_pa
            ):
                raise EquipmentOperatingError(
                    "Darcy link initial pressure drop exceeds its envelope"
                )
        # Reuse transport support propagation, then add the reduced links.
        changed = True
        while changed:
            previous = [set(s) for s in supports]
            fake = SimpleNamespace(
                nodes=tuple(
                    (
                        name,
                        SimpleNamespace(
                            component_moles=[float(j in support) for j in range(len(f.names))]
                        ),
                    )
                    for name, support in zip(t.names, supports)
                )
            )
            supports = [set(a.cpu().tolist()) for a in t._active(fake, openings)]
            for a, b in self.bidirectional_edges:
                supports[a] = supports[b] = supports[a] | supports[b]
            for a, b in self.directed_edges:
                supports[b] |= supports[a]
            changed = supports != previous
        self.active = [torch.tensor(sorted(s), dtype=torch.long, device=f.device) for s in supports]
        amounts = (
            external.ordinary_guesses
            if external is not None
            else [f.tensor(s.component_moles) for _, s in state.nodes]
        )
        self.ordinary_guesses = [
            n.index_copy(0, active, n[active].clamp_min(n.sum() * 1e-10))
            for n, active in zip(amounts, self.active)
        ]

    def exchange(self, data):
        t, f = self.graph.transport, self.graph.transport.fluid
        nc = len(f.names)
        base = (
            self.external.exchange(data)
            if self.external
            else {
                "net": [d["p"].new_zeros(nc + 2) for d in data],
                "valid": data[0]["p"].new_ones((), dtype=torch.bool),
                "incoming_entropy": data[0]["p"] * 0,
            }
        )
        net, valid, records = list(base["net"]), base["valid"], {}
        for c, a, b in self.entries:
            left, right = data[a], data[b]
            dp = left["p"] - right["p"]
            if not c.allow_reverse:
                dp = dp.clamp_min(0)
            forward = dp >= 0
            rho = torch.where(forward, left["rho"], right["rho"])
            velocity = c.velocity(dp, rho)
            volume_rate = math.pi * c.diameter_m**2 / 4 * velocity
            n_density = torch.where(
                forward, left["n"] / t.nodes[a].volume_m3, right["n"] / t.nodes[b].volume_m3
            )
            energy_density = torch.where(
                forward,
                (left["internal"] + left["p"] * t.nodes[a].volume_m3 + left["potential"])
                / t.nodes[a].volume_m3,
                (right["internal"] + right["p"] * t.nodes[b].volume_m3 + right["potential"])
                / t.nodes[b].volume_m3,
            )
            components, energy = volume_rate * n_density, volume_rate * energy_density
            packet = torch.cat((components, energy.new_zeros(1), energy.reshape(1)))
            net[a], net[b] = net[a] - packet, net[b] + packet
            sound = torch.where(
                forward,
                left["q"]["frozen_sound_speed_m_s"][0],
                right["q"]["frozen_sound_speed_m_s"][0],
            )
            valid = valid & (len(left["phase_n"]) == 1) & (len(right["phase_n"]) == 1)
            valid = valid & (velocity.abs() <= c.maximum_velocity_m_s)
            valid = valid & (velocity.abs() <= c.maximum_mach_number * sound)
            valid = valid & (
                dp.abs() <= c.maximum_pressure_drop_fraction * torch.minimum(left["p"], right["p"])
            )
            records[c.name] = {
                "volume_rate_m3_s": volume_rate,
                "velocity_m_s": velocity,
                "component_rates_mol_s": components,
                "total_energy_rate_w": energy,
                "reynolds_number": rho * velocity.abs() * c.diameter_m / c.dynamic_viscosity_pa_s,
            }
        return {**base, "net": net, "valid": valid, "darcy_records": records}

    def prepare(self, y, lo, hi, base, offsets, dt, pattern):
        def evaluate(z):
            return base(z, external_fluxes=self.exchange)

        evaluate.linear_blocks, evaluate.maximum_update = base.linear_blocks, base.maximum_update
        if pattern is not None:
            support = set(zip(*pattern))
            for a, b in self.internal_edges:
                for i, j in ((a, b), (b, a)):
                    support.update(
                        (r, c)
                        for r in range(offsets[i], offsets[i + 1])
                        for c in range(offsets[j], offsets[j + 1])
                    )
            pattern = tuple(map(list, zip(*sorted(support))))
        return y, lo, hi, evaluate, pattern

    def accept(self, y, result):
        base = self.external.accept(y, result) if self.external else {}
        records = result["exchange"]["darcy_records"]
        f = self.graph.transport.fluid
        for c, a, b in self.entries:
            volume_rate = float(records[c.name]["volume_rate_m3_s"].detach().cpu())
            if volume_rate == 0:
                records[c.name]["delivery_temperature_k"] = None
                continue
            forward = volume_rate > 0
            donor, receiver = (
                result["data"][a if forward else b],
                result["data"][b if forward else a],
            )
            h = (donor["phase_n"] * donor["q"]["molar_enthalpy_j_mol"]).sum() / donor[
                "phase_n"
            ].sum()
            pressure, enthalpy, temperature = (
                torch.stack((receiver["p"], h, donor["t"])).detach().cpu().tolist()
            )
            q = f.flash_ph(
                pressure, enthalpy, donor["n"] / donor["n"].sum(), temperature_guess_k=temperature
            )
            if q["phase_count"] != 1:
                raise EquipmentOperatingError("Darcy delivery crosses a phase boundary")
            records[c.name]["delivery_temperature_k"] = q["temperature_k"]
        return {
            **base,
            "darcy_connections": {
                name: {k: None if v is None else v.detach().cpu().tolist() for k, v in row.items()}
                for name, row in records.items()
            },
        }

    def ledger(self, record, dt):
        return (
            self.external.ledger(record, dt)
            if self.external
            else {
                "material_component_moles": [0.0] * len(self.graph.transport.fluid.names),
                "material_energy_j": 0.0,
            }
        )
