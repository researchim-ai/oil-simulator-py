"""Simultaneous finite PR graph with multiple mixed-profile pipe branches."""

import math
from dataclasses import dataclass, replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .operation import _digest
from .pr_equilibrium_transport import PREquilibriumTransportNetwork, PREquilibriumTransportState
from .pr_layered_plant import LayeredPlantState, PRLayeredPlantNetwork, _LayeredCoupling
from .pr_layered_transport import LayeredTransportState, PRLayeredTransportNetwork


@dataclass(frozen=True)
class ProfileGraphState:
    transport: PREquilibriumTransportState
    profiles: tuple[tuple[str, LayeredTransportState], ...]
    model_id: str

    @property
    def time_s(self):
        return self.transport.time_s

    def to_dict(self):
        return {
            "transport": self.transport.to_dict(),
            "profiles": {n: s.to_dict() for n, s in self.profiles},
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"transport", "profiles", "model_id"}:
            raise ValueError("Unknown profile graph state fields")
        return cls(
            PREquilibriumTransportState.from_dict(data["transport"]),
            tuple(
                (n, LayeredTransportState.from_dict(s)) for n, s in sorted(data["profiles"].items())
            ),
            data["model_id"],
        )


class PRProfileGraphNetwork(PRLayeredPlantNetwork):
    """Each named branch has two real finite-vessel endpoints or closed ends.

    A branch uses the native shared-velocity, transverse-equilibrium closure.
    Vessel junctions may connect any number of branches. There is one ordinary
    graph evaluation and one evaluation per native branch in each residual.
    """

    def __init__(self, transport, profiles, *, maximum_head_fraction=1e-3):
        if not isinstance(transport, PREquilibriumTransportNetwork) or not profiles:
            raise ValueError("Finite ordinary graph and nonempty profile registry required")
        self.transport = transport
        self.maximum_head_fraction = _number(maximum_head_fraction, "head bound", positive=True)
        self.bridges = {}
        for name, (pipe, ends) in sorted(profiles.items()):
            _name(name)
            self.bridges[name] = PRLayeredPlantNetwork(
                transport, pipe, ends, maximum_head_fraction=self.maximum_head_fraction
            )
        self.names = tuple(self.bridges)
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_profile_graph_v1",
            "transport": self.transport.to_dict(),
            "profiles": {
                n: {"pipe": b.layered.to_dict(), "boundary_nodes": b.boundary_nodes}
                for n, b in self.bridges.items()
            },
            "maximum_head_fraction": self.maximum_head_fraction,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        if set(data) != {"model", "transport", "profiles", "maximum_head_fraction"}:
            raise ValueError("Unknown profile graph model fields")
        if data["model"] != "pr_profile_graph_v1":
            raise ValueError("Unknown profile graph model")
        profiles = {}
        for name, spec in data["profiles"].items():
            if set(spec) != {"pipe", "boundary_nodes"}:
                raise ValueError("Unknown profile branch fields")
            profiles[name] = (
                PRLayeredTransportNetwork.from_dict(spec["pipe"], device=device),
                spec["boundary_nodes"],
            )
        return cls(
            PREquilibriumTransportNetwork.from_dict(data["transport"], device=device),
            profiles,
            maximum_head_fraction=data["maximum_head_fraction"],
        )

    def initialize(self, transport_state, profile_states):
        if set(profile_states) != set(self.names):
            raise ValueError("Profile initialization registry differs")
        state = ProfileGraphState(
            transport_state, tuple((n, profile_states[n]) for n in self.names), self.model_id
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, ProfileGraphState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
            or tuple(n for n, _ in state.profiles) != self.names
        ):
            raise PropertyModelError("Profile graph state/model registry differs")
        self.transport.inspect(state.transport)
        for n, s in state.profiles:
            if s.time_s != state.time_s:
                raise PropertyModelError("Profile graph clocks differ")
            self.bridges[n].layered.inspect(s)

    def _stocks(self, state):
        first = self.bridges[self.names[0]]
        n, e = first._stocks(
            LayeredPlantState(state.transport, state.profiles[0][1], first.model_id)
        )
        f = self.transport.fluid
        for name, s in state.profiles[1:]:
            for cell in s.cells:
                nn = f.tensor(cell.profile.component_moles)
                n = n + nn
                e += cell.total_energy_j - float((nn * f.molar_masses).sum().cpu()) * (
                    self.bridges[name].layered.gravity * self.transport.reference_elevation_m
                )
        return n, e

    def _suggested_step(self, state):
        return min(state.transport.next_step_s, *(s.next_step_s for _, s in state.profiles))

    def _temporal_error(self, coarse, fine, relative_tolerance, interface_tolerance_m):
        return max(
            self.transport._error(coarse.transport, fine.transport),
            *(
                self.bridges[n].layered._error(
                    a,
                    b,
                    relative_tolerance=relative_tolerance,
                    interface_tolerance_m=interface_tolerance_m,
                )
                for (n, a), (_, b) in zip(coarse.profiles, fine.profiles, strict=True)
            ),
        )

    def _adaptive_state(self, fine, nextdt):
        return replace(
            fine,
            transport=replace(fine.transport, next_step_s=nextdt),
            profiles=tuple((n, replace(s, next_step_s=nextdt)) for n, s in fine.profiles),
        )

    def step(
        self,
        state,
        dt_s,
        *,
        openings=None,
        heat_w=None,
        utility_temperature_k=None,
        loss_coefficients=None,
        drive_torque_nm=None,
        restriction_openings=None,
        profile_heat_w=None,
        profile_utility_temperature_k=None,
        profile_openings=None,
    ):
        self.inspect(state)
        dt = _number(dt_s, "time step", positive=True)
        if state.time_s + dt <= state.time_s:
            raise ValueError("Step below clock resolution")
        t = self.transport
        opened = dict(t.default_openings)
        if openings is not None:
            if set(openings) - set(opened) or any(type(v) is not bool for v in openings.values()):
                raise ValueError("Unknown/nonboolean graph opening")
            opened.update(openings)
        heat, utility, losses = (
            dict(heat_w or {}),
            dict(utility_temperature_k or {}),
            dict(loss_coefficients or {}),
        )
        if (
            set(heat) - set(t.names)
            or set(utility) - set(t.names)
            or set(losses) - {n.name for n in t.nodes if n.kind == "pipe_cell"}
        ):
            raise ValueError("Unknown heat, utility or resistance node")
        hh = [_number(heat.get(n, 0.0), "heat rate") for n in t.names]
        uu = [
            _number(utility.get(n, s.temperature_k), "utility temperature", positive=True)
            for n, (_, s) in zip(t.names, state.transport.nodes)
        ]
        if any(q and n not in utility for n, q in zip(t.names, hh)):
            raise ValueError("Nonzero heat requires explicit utility temperature")
        losses = {n: _number(v, "loss coefficient", nonnegative=True) for n, v in losses.items()}
        ph, pu, ends = {}, {}, {}
        profile_heat_w = dict(profile_heat_w or {})
        profile_utility_temperature_k = dict(profile_utility_temperature_k or {})
        profile_openings = dict(profile_openings or {})
        if (set(profile_heat_w) | set(profile_utility_temperature_k) | set(profile_openings)) - set(
            self.names
        ):
            raise ValueError("Unknown commanded profile branch")
        for n, bridge in self.bridges.items():
            ph[n] = bridge.layered._heat(profile_heat_w.get(n))
            pu[n] = (
                None
                if n not in profile_utility_temperature_k
                else tuple(
                    _number(v, "profile utility temperature", positive=True)
                    for v in profile_utility_temperature_k[n]
                )
            )
            if (any(ph[n]) and pu[n] is None) or (pu[n] is not None and len(pu[n]) != len(ph[n])):
                raise ValueError("Profile heat requires utility temperatures per cell")
            flags = profile_openings.get(n, (True, True))
            if len(flags) != 2 or any(type(v) is not bool for v in flags):
                raise ValueError("Two boolean profile endpoint openings required")
            ends[n] = tuple(e if on else None for e, on in zip(bridge.boundary_nodes, flags))
        seeds = {}
        for attempt in range(3):
            coupling = _ProfileGraphCoupling(self, state, dt, opened, ph, pu, ends, seeds)
            try:
                ordinary, report = t._step(
                    state.transport,
                    dt,
                    opened,
                    hh,
                    uu,
                    losses,
                    _coupling=coupling,
                    drive_torque_nm=drive_torque_nm,
                    restriction_openings=restriction_openings,
                )
                break
            except (HydraulicConvergenceError, PropertyModelError):
                if attempt == 2:
                    raise
                for n, child in coupling.children.items():
                    if child.last_data is None:
                        raise
                    seeds[n] = child.model.layered.predict_seeds(
                        child.state.layered, dt, child.last_data, heat_w=ph[n]
                    )
        linked = report["coupling"]
        new = self.initialize(
            ordinary,
            {
                n: LayeredTransportState.from_dict(linked["profiles"][n]["state"])
                for n in self.names
            },
        )
        heat_j = (
            report["ambient_heat_j"]
            + report["external_heat_j"]
            + dt * math.fsum(q for values in ph.values() for q in values)
        )
        work = report["external_work_j"]
        entropy = report["entropy_production_j_k"] + math.fsum(
            v["entropy_production_j_k"] for v in linked["profiles"].values()
        )
        if entropy < -1e-6:
            raise ConservationError("Profile graph entropy ledger rejected")
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "transport": report,
            "profiles": linked["profiles"],
            "boundary_nodes": ends,
            "vessel_support_force_n": linked["vessel_support_force_n"],
            "total_heat_j": heat_j,
            "total_work_j": work,
            "conservation_residuals": self._ledger(state, new, heat_j + work),
            "entropy_production_j_k": entropy,
            "topology_retries": attempt,
        }


class _ProfileGraphCoupling:
    def __init__(self, model, state, dt, openings, heat, utility, ends, seeds):
        self.model = model
        t = model.transport
        self.children = {}
        ports = {
            n: tuple((t.names.index(e), None) for e in ends[n] if e is not None)
            for n in model.names
        }
        self.external_ports = tuple(p for pp in ports.values() for p in pp)
        support = [set(v.cpu().tolist()) for v in t._active(state.transport, openings)]
        native = {
            n: {i for i, v in enumerate(s.cells[0].profile.component_moles) if v > 0}
            for n, s in state.profiles
        }
        for n, pp in ports.items():
            for i, _ in pp:
                support[i] |= native[n]
        edges = [(a, b, True) for l, (a, b) in zip(t.links, t.ends) if openings[l.name]]
        edges += [
            (
                t.names.index(u.source),
                t.names.index(u.target),
                getattr(u, "signed_map", None) is not None,
            )
            for u in (*t.compressors, *t.pump_connections)
            if openings[u.name]
        ]
        edges += [
            (t.names.index(u.source), t.names.index(u.target), True)
            for u in t.restrictions
            if openings[u.name]
        ]
        for pp in ports.values():
            points = [i for i, _ in pp]
            edges += [(a, b, True) for a in points for b in points]
        for _ in t.nodes:
            for a, b, both in edges:
                support[b] |= support[a]
                if both:
                    support[a] |= support[b]
        for n, pp in ports.items():
            if any(support[i] - native[n] for i, _ in pp):
                raise PropertyModelError(
                    "Cross-branch inlet introduces an unsupported native component"
                )
        self.active = [
            torch.tensor(sorted(s), device=t.fluid.device, dtype=torch.long) for s in support
        ]
        self.ordinary_guesses = [
            t.fluid.tensor(s.component_moles) for _, s in state.transport.nodes
        ]
        for name, s in state.profiles:
            b = model.bridges[name]
            bridge = (
                b
                if ends[name] == b.boundary_nodes
                else PRLayeredPlantNetwork(
                    t, b.layered, ends[name], maximum_head_fraction=model.maximum_head_fraction
                )
            )
            self.children[name] = _LayeredCoupling(
                bridge,
                LayeredPlantState(state.transport, s, bridge.model_id),
                dt,
                openings,
                heat[name],
                utility[name],
                seeds.get(name),
                shared_active=self.active,
                shared_guesses=self.ordinary_guesses,
            )

    def prepare(self, y, lo, hi, base, offsets, dt, pattern):
        self.cut = len(y)
        self.slices = {}
        yy, ll, hh = [y], [lo], [hi]
        edges, blocks = set(), []
        end = self.cut
        for name, child in self.children.items():
            cy, cl, ch, _ce, _cp = child.prepare(y, lo, hi, base, offsets, dt, pattern)
            joint = child.joint
            size = len(cy) - self.cut
            self.slices[name] = slice(end, end + size)
            yy.append(cy[self.cut :])
            ll.append(cl[self.cut :])
            hh.append(ch[self.cut :])
            shift = end - self.cut
            edges.update(
                (r if r < self.cut else r + shift, c if c < self.cut else c + shift)
                for r, c in zip(*joint.sparsity)
            )
            if not blocks:
                blocks.extend(base.linear_blocks)
            blocks.extend(
                [[v + end for v in block] for block in joint.profile_evaluate.linear_blocks]
            )
            end += size
        if end > self.model.transport.maximum_unknowns:
            raise ValueError("Profile graph exceeds maximum_unknowns")
        for child in self.children.values():
            if end > child.model.layered.max_unknowns:
                raise ValueError("Profile graph exceeds native maximum_unknowns")

        def evaluate(z):
            def callback(ordinary_data):
                values = {
                    n: child.joint.vessel_exchange(z[self.slices[n]], ordinary_data)
                    for n, child in self.children.items()
                }
                return {
                    "net": [
                        sum(
                            (v["net"][i] for v in values.values()),
                            z.new_zeros(len(self.model.transport.fluid.names) + 2),
                        )
                        for i in range(len(self.model.transport.nodes))
                    ],
                    "valid": torch.stack([v["valid"] for v in values.values()]).all(),
                    "incoming_entropy": sum(
                        (v["incoming_entropy"] for v in values.values()), z.new_zeros(())
                    ),
                    "profiles": values,
                }

            residual, data = base(z[: self.cut], external_fluxes=callback)
            profiles = data["exchange"]["profiles"]
            data["profiles_data"] = profiles
            return torch.cat((residual, *(v["profile_residual"] for v in profiles.values()))), data

        evaluate.linear_blocks = blocks
        evaluate.maximum_update = torch.cat(
            (base.maximum_update, y.new_full((end - self.cut,), math.inf))
        )
        pairs = sorted(edges)
        with torch.no_grad():
            initial = evaluate(torch.cat(yy))[1]["profiles_data"]
            for n, c in self.children.items():
                c.last_data = initial[n]["profile"]
        return (
            torch.cat(yy),
            torch.cat(ll),
            torch.cat(hh),
            evaluate,
            ([r for r, _ in pairs], [c for _, c in pairs])
            if end > self.model.transport.dense_limit
            else None,
        )

    def accept(self, y, result):
        profiles = {}
        reactions = {}
        for n, child in self.children.items():
            data = result["profiles_data"][n]
            profiles[n] = child.accept(
                torch.cat((y[: self.cut], y[self.slices[n]])),
                {
                    "layered_data": data["profile"],
                    "vessel_support_force_n": data["vessel_support_force_n"],
                },
            )
            reactions[n] = profiles[n]["vessel_support_force_n"]
        return {"profiles": profiles, "vessel_support_force_n": reactions}
