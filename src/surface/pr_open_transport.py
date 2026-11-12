"""Open material boundaries for the existing implicit PR pipe/equipment graph.

Ports connect real, supported, stirred finite vessels, not axial pipe ghosts.
External pressure characteristics are explicitly empirical linear laws. The
same end-state N/H/S flux participates in Newton, local and global ledgers.
"""

import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch

from .component_energy import ComponentEnergyStream
from .equilibrium_cache import cached_flashes
from .liquid import PropertyModelError, _name, _number
from .operation import _digest
from .pr_equilibrium_transport import PREquilibriumTransportNetwork
from .stratified_pr_flow import ReceiverFeedPressureError


@dataclass(frozen=True)
class PRTransportFeed:
    name: str
    node: str
    stream: ComponentEnergyStream

    def __post_init__(self):
        _name(self.name)
        _name(self.node)
        if not isinstance(self.stream, ComponentEnergyStream):
            raise TypeError("Explicit component/energy stream required")

    def to_dict(self):
        return {"name": self.name, "node": self.node, "stream": self.stream.to_dict()}


@dataclass(frozen=True)
class PRTransportPressurePort:
    name: str
    node: str
    pressure_pa: float
    temperature_k: float
    mole_fractions: tuple
    volume_conductance_m3_s_pa: float
    allow_reverse: bool = False
    opening: float = 1.0
    maximum_pressure_drop_fraction: float = 0.05

    def __post_init__(self):
        _name(self.name)
        _name(self.node)
        _number(self.pressure_pa, "external pressure", positive=True)
        _number(self.temperature_k, "external temperature", positive=True)
        _number(self.volume_conductance_m3_s_pa, "port conductance", nonnegative=True)
        object.__setattr__(self, "mole_fractions", tuple(self.mole_fractions))
        z = self.mole_fractions
        if not z or any(_number(x, "mole fraction", nonnegative=True) > 1 for x in z):
            raise ValueError("Explicit boundary composition required")
        if abs(math.fsum(z) - 1.0) > 1e-12:
            raise ValueError("Boundary fractions must sum to one")
        if type(self.allow_reverse) is not bool:
            raise TypeError("Explicit boolean reverse-flow policy required")
        if not 0 <= _number(self.opening, "port opening") <= 1:
            raise ValueError("Port opening outside [0,1]")
        if not 0 < _number(self.maximum_pressure_drop_fraction, "pressure drop domain") <= 0.05:
            raise ValueError("Linear port requires an explicit low-dP domain <=5%")


class PROpenTransportNetwork:
    def __init__(self, transport, feeds=(), pressure_ports=()):
        # Do not clone or override an unknown graph subclass's physical operator.
        if type(transport) is not PREquilibriumTransportNetwork:
            raise TypeError("Expected ordinary PR finite-volume transport graph")
        self.transport = transport
        self.fluid = transport.fluid
        self.feeds = tuple(sorted(feeds, key=lambda v: v.name))
        self.pressure_ports = tuple(sorted(pressure_ports, key=lambda v: v.name))
        if any(not isinstance(v, PRTransportFeed) for v in self.feeds) or any(
            not isinstance(v, PRTransportPressurePort) for v in self.pressure_ports
        ):
            raise TypeError("Explicit feed/pressure-port types required")
        names = [v.name for v in (*self.feeds, *self.pressure_ports)]
        if len(names) != len(set(names)) or set(names) & {v.name for v in transport.switches}:
            raise ValueError("Distinct boundary and equipment names required")
        for port in (*self.feeds, *self.pressure_ports):
            if port.node not in transport.names:
                raise ValueError("Unknown boundary vessel")
            node = transport.nodes[transport.names.index(port.node)]
            if node.kind != "vessel" or node.geometry is not None or node.drift_flux is not None:
                raise ValueError(
                    "Open ports require a supported stirred vessel; resolved apertures need their own coupling"
                )
        for feed in self.feeds:
            stream = feed.stream
            stream.validate_fluid(self.fluid)
            node = transport.nodes[transport.names.index(feed.node)]
            if (
                stream.elevation_m != node.elevation_m
                or stream.gravity_m_s2 != transport.gravity_m_s2
            ):
                raise ValueError("Feed and physical node elevation/gravity differ")
            if stream.velocity_m_s:
                raise ValueError("Stirred boundary requires a stagnated energy stream")
        for port in self.pressure_ports:
            if len(port.mole_fractions) != len(self.fluid.names):
                raise ValueError("Boundary component registry differs")
            self.fluid._validate(
                self.fluid.tensor(port.pressure_pa),
                self.fluid.tensor(port.temperature_k),
                self.fluid.tensor(port.mole_fractions),
            )
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_open_transport_v1",
            "transport": self.transport.to_dict(),
            "feeds": [v.to_dict() for v in self.feeds],
            "pressure_ports": [asdict(v) for v in self.pressure_ports],
        }

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        if (
            set(raw) != {"model", "transport", "feeds", "pressure_ports"}
            or raw["model"] != "pr_open_transport_v1"
        ):
            raise ValueError("Unknown open PR graph specification")
        return cls(
            PREquilibriumTransportNetwork.from_dict(raw["transport"], device=device),
            [
                PRTransportFeed(v["name"], v["node"], ComponentEnergyStream(**v["stream"]))
                for v in raw["feeds"]
            ],
            [PRTransportPressurePort(**v) for v in raw["pressure_ports"]],
        )

    def inspect(self, state):
        if _digest(self.to_dict()) != self.model_id:
            raise PropertyModelError("Open boundary model changed")
        self.transport.inspect(state)

    def advance(self, state, duration_s, *, port_openings=None, **controls):
        if _digest(self.to_dict()) != self.model_id:
            raise PropertyModelError("Open boundary model changed")
        if "_coupling_factory" in controls:
            raise ValueError("Boundary coupling factory is owned by the open graph")
        commands = {v.name: v.opening for v in self.pressure_ports}
        if set(port_openings or {}) - set(commands):
            raise ValueError("Unknown external port opening")
        for name, value in (port_openings or {}).items():
            if not 0 <= _number(value, "port opening") <= 1:
                raise ValueError("Port opening outside [0,1]")
            commands[name] = value
        self.transport.inspect(state)
        # Immutable packets and maintained PT states are resolved once per call.
        # The transaction cache never leaks into the next control interval.
        with cached_flashes(self.fluid):
            prepared = _BoundaryProperties(self, commands)
            result = self.transport.advance(
                state,
                duration_s,
                _coupling_factory=lambda s, dt, o: _OpenCoupling(self, prepared, s, dt, o),
                **controls,
            )
        nc = len(self.fluid.names)
        result.update(
            {
                "open_model_id": self.model_id,
                "port_openings": commands,
                "boundary_component_moles": [
                    math.fsum(s["exchange_component_moles"][j] for s in result["steps"])
                    for j in range(nc)
                ],
                "boundary_energy_j": math.fsum(s["exchange_energy_j"] for s in result["steps"]),
                "scope": "Implicit distributed PR pipes/equipment with material boundaries on supported stirred vessels. Empirical low-dP pressure ports; no resolved gravity-column boundary or reservoir energy equation.",
            }
        )
        return result


class _BoundaryProperties:
    def __init__(self, model, commands):
        t, f = model.transport, model.fluid
        self.commands = commands
        self.feeds, self.reservoirs = [], []
        for feed in model.feeds:
            stream = feed.stream
            state = stream.equilibrate(f)
            n = f.tensor(stream.component_molar_rates_mol_s)
            mass = (n * f.molar_masses).sum()
            energy = (
                f.tensor(stream.total_energy_rate_w)
                - mass * t.gravity_m_s2 * t.reference_elevation_m
            )
            entropy = n.sum() * state["molar_entropy_j_mol_k"] if state is not None else n.sum()
            self.feeds.append((t.names.index(feed.node), feed, n, energy, entropy))
        for port in model.pressure_ports:
            # The receiving external PT state is used only if flow reverses.
            if commands[port.name] and port.allow_reverse and port.volume_conductance_m3_s_pa:
                q = f.flash_pt(port.pressure_pa, port.temperature_k, port.mole_fractions)
                density = f.tensor(port.mole_fractions) / q["molar_volume_m3_mol"]
                node = t.nodes[t.names.index(port.node)]
                potential = (
                    (density * f.molar_masses).sum()
                    * t.gravity_m_s2
                    * (node.elevation_m - t.reference_elevation_m)
                )
                e = q["molar_enthalpy_j_mol"] / q["molar_volume_m3_mol"] + potential
                s = q["molar_entropy_j_mol_k"] / q["molar_volume_m3_mol"]
                self.reservoirs.append((density, e, s))
            else:
                self.reservoirs.append(None)


class _OpenCoupling:
    external_ports = ()

    def __init__(self, model, prepared, state, dt, openings):
        self.model, self.prepared = model, prepared
        t, f = model.transport, model.fluid
        amounts = [f.tensor(s.component_moles) for _, s in state.nodes]
        support = [n.clone() for n in amounts]
        for i, _feed, n, _e, _s in prepared.feeds:
            amounts[i] = amounts[i] + dt * n
            support[i] = support[i] + n
        for port, reservoir in zip(model.pressure_ports, prepared.reservoirs):
            if reservoir is not None:
                i = t.names.index(port.node)
                support[i] = support[i] + reservoir[0]
        fake = SimpleNamespace(
            nodes=tuple(
                (name, SimpleNamespace(component_moles=n)) for name, n in zip(t.names, support)
            )
        )
        self.active = t._active(fake, openings)
        # Positive amounts for newly reachable support are guesses, never stocks.
        self.ordinary_guesses = []
        for n, active in zip(amounts, self.active):
            self.ordinary_guesses.append(
                n.index_copy(0, active, n[active].clamp_min(n.sum() * 1e-10))
            )

    def exchange(self, data):
        m, b = self.model, self.prepared
        t, f = m.transport, m.fluid
        nc = len(f.names)
        net = [d["p"].new_zeros(nc + 2) for d in data]
        entropy = data[0]["p"] * 0
        valid = torch.ones((), dtype=torch.bool, device=f.device)
        records = []
        for i, feed, n, energy, s in b.feeds:
            net[i] = net[i] + torch.cat((n, energy.new_zeros(1), energy.reshape(1)))
            entropy = entropy + s
            records.append((feed.name, "feed", n, energy, s, energy * 0))
        for port, reservoir in zip(m.pressure_ports, b.reservoirs):
            i = t.names.index(port.node)
            d, node = data[i], t.nodes[i]
            delta = d["p"] - port.pressure_pa
            volume = delta * (b.commands[port.name] * port.volume_conductance_m3_s_pa)
            if not port.allow_reverse:
                volume = volume.clamp_min(0)
            forward_n = d["n"] / node.volume_m3
            forward_e = (d["internal"] + d["p"] * node.volume_m3 + d["potential"]) / node.volume_m3
            forward_s = d["entropy"] / node.volume_m3
            if reservoir is None:
                density, energy_density, entropy_density = forward_n, forward_e, forward_s
            else:
                density = torch.where(volume >= 0, forward_n, reservoir[0])
                energy_density = torch.where(volume >= 0, forward_e, reservoir[1])
                entropy_density = torch.where(volume >= 0, forward_s, reservoir[2])
            n, energy, s = volume * density, volume * energy_density, volume * entropy_density
            net[i] = net[i] - torch.cat((n, energy.new_zeros(1), energy.reshape(1)))
            entropy = entropy - s
            valid = valid & (
                (volume == 0)
                | (
                    delta.abs()
                    <= port.maximum_pressure_drop_fraction
                    * torch.maximum(d["p"], d["p"].new_tensor(port.pressure_pa))
                )
            )
            records.append((port.name, "outward", n, energy, s, volume))
        return {
            "net": net,
            "incoming_entropy": entropy,
            "valid": valid,
            "boundary_records": records,
        }

    def prepare(self, y, lo, hi, base, offsets, dt, pattern):
        def evaluate(z):
            return base(z, external_fluxes=self.exchange)

        evaluate.linear_blocks = base.linear_blocks
        evaluate.maximum_update = base.maximum_update
        return y, lo, hi, evaluate, pattern

    def ledger(self, record, dt):
        nc = len(self.model.fluid.names)
        rows = record["ports"].values()
        n = [
            dt
            * math.fsum(
                (1 if r["orientation"] == "feed" else -1) * r["component_rates_mol_s"][j]
                for r in rows
            )
            for j in range(nc)
        ]
        e = dt * math.fsum(
            (1 if r["orientation"] == "feed" else -1) * r["total_energy_rate_w"] for r in rows
        )
        return {"material_component_moles": n, "material_energy_j": e}

    def accept(self, y, result):
        for i, feed, n, _e, _s in self.prepared.feeds:
            if (
                feed.stream.molar_rate_mol_s
                and float(result["data"][i]["p"].cpu()) > feed.stream.pressure_pa
            ):
                raise ReceiverFeedPressureError(
                    feed.stream.pressure_pa, float(result["data"][i]["p"].cpu())
                )
        records = result["exchange"]["boundary_records"]
        return {
            "closure": "end_state_stirred_material_boundaries",
            "model_id": self.model.model_id,
            "ports": {
                name: {
                    "orientation": orientation,
                    "component_rates_mol_s": n.cpu().tolist(),
                    "total_energy_rate_w": float(e.cpu()),
                    "entropy_rate_w_k": float(s.cpu()),
                    "volume_rate_m3_s": float(v.cpu()),
                }
                for name, orientation, n, e, s, v in records
            },
        }
