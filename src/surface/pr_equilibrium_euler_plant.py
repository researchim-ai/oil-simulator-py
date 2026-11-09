"""Conservative first-order IMEX bridge to existing finite-vessel equipment.

Euler faces are explicit and CFL-limited. The same accepted N/E transfer is
an external load in the existing equipment graph Newton. It is not a single
monolithic pressure solve. Connected endpoints are real stirred vessels;
resolved separator nozzles remain handled by the existing graph connections.
"""

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from .pr_equilibrium_euler_pipe import PREquilibriumEulerPipe, PREquilibriumEulerPipeState
from .pr_equilibrium_transport import PREquilibriumTransportNetwork, PREquilibriumTransportState
from .pr_interphase_relaxation import PRRelaxationPhase
from .pr_phase_equilibration import PRPhaseEquilibriumState


@dataclass(frozen=True)
class PREquilibriumEulerPlantState:
    transport: PREquilibriumTransportState
    pipes: tuple
    next_step_s: float
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "pipes", tuple((name, state) for name, state in self.pipes))
        _number(self.next_step_s, "plant step", positive=True)

    @property
    def time_s(self):
        return self.transport.time_s

    def to_dict(self):
        return {
            "transport": self.transport.to_dict(),
            "pipes": {n: p.to_dict() for n, p in self.pipes},
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"transport", "pipes", "next_step_s", "model_id"}:
            raise ValueError("Unknown Euler plant checkpoint fields")
        return cls(
            PREquilibriumTransportState.from_dict(data["transport"]),
            tuple(
                (
                    n,
                    (
                        PRActiveMultifluidState
                        if "bulk_stocks" not in p
                        else PREquilibriumEulerPipeState
                    ).from_dict(p),
                )
                for n, p in sorted(data["pipes"].items())
            ),
            data["next_step_s"],
            data["model_id"],
        )


class _KnownFaceExchange:
    def __init__(self, transport, state, dt, openings, net, entropy):
        self.external_ports = ()
        self.net = net
        self.entropy = entropy
        self.ordinary_guesses = [
            transport.fluid.tensor(s.component_moles) + dt * v[:-2]
            for (_, s), v in zip(state.nodes, net)
        ]
        if any(not bool((v >= 0).all().cpu()) for v in self.ordinary_guesses):
            raise PropertyModelError("Interface flux exhausts a finite vessel component")
        # Logical support descriptors only, never published as physical states.
        support = SimpleNamespace(
            nodes=tuple(
                (
                    name,
                    SimpleNamespace(
                        component_moles=(
                            transport.fluid.tensor(s.component_moles) + dt * v[:-2].clamp_min(0)
                        )
                        .cpu()
                        .tolist()
                    ),
                )
                for (name, s), v in zip(state.nodes, net)
            )
        )
        self.active = transport._active(support, openings)

    def prepare(self, y, lo, hi, base, offsets, dt, pattern):
        def exchange(data):
            return {
                "net": self.net,
                "valid": y.new_ones((), dtype=torch.bool),
                "incoming_entropy": y.new_tensor(self.entropy),
            }

        def evaluate(z):
            return base(z, external_fluxes=exchange)

        evaluate.linear_blocks = base.linear_blocks
        evaluate.maximum_update = base.maximum_update
        return y, lo, hi, evaluate, pattern

    def accept(self, y, result):
        return {"closure": "explicit_shared_Euler_faces"}


class PREquilibriumEulerPlant:
    def __init__(self, transport, pipes):
        if not isinstance(transport, PREquilibriumTransportNetwork) or not pipes:
            raise TypeError("Finite PR equipment graph and nonempty Euler pipe registry required")
        if transport.gravity_m_s2 != 0 or any(n.kind != "vessel" for n in transport.nodes):
            raise ValueError("Initial IMEX bridge requires zero-gravity supported vessel nodes")
        self.transport = transport
        self.fluid = transport.fluid
        self.pipes = {}
        for name, entry in sorted(pipes.items()):
            if (
                not isinstance(name, str)
                or not name
                or set(entry) != {"pipe", "boundary_nodes", "direction"}
            ):
                raise ValueError("Named pipe, two boundary nodes and direction required")
            pipe = entry["pipe"]
            if isinstance(pipe, PRActiveMultifluidPipe) and pipe.gravity != 0:
                raise ValueError(
                    "Use the gravitational active surface adapter for surveyed gravity"
                )
            ends = tuple(entry["boundary_nodes"])
            direction = tuple(float(v) for v in entry["direction"])
            if (
                not isinstance(pipe, (PREquilibriumEulerPipe, PRActiveMultifluidPipe))
                or pipe.boundary_mode != "walls"
                or pipe.fluid.model_id != self.fluid.model_id
                or pipe.fluid.device != self.fluid.device
            ):
                raise ValueError("Closed/open Euler endpoints require the same PR model and device")
            if (
                len(ends) != 2
                or all(n is None for n in ends)
                or any(n is not None and n not in transport.names for n in ends)
            ):
                raise ValueError("At least one known finite-vessel endpoint required")
            if any(
                transport.nodes[transport.names.index(n)].geometry is not None
                for n in ends
                if n is not None
            ):
                raise ValueError(
                    "Connect Euler faces through mixed finite plenums, not unsampled separator profiles"
                )
            if (
                len(direction) != 3
                or not all(math.isfinite(v) for v in direction)
                or abs(sum(v * v for v in direction) - 1) > 1e-12
                or direction[2] != 0
            ):
                raise ValueError("Horizontal unit pipe direction required")
            self.pipes[name] = {"pipe": pipe, "boundary_nodes": ends, "direction": direction}
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_active_multifluid_plant_v1"
            if any(isinstance(e["pipe"], PRActiveMultifluidPipe) for e in self.pipes.values())
            else "pr_equilibrium_euler_plant_v1",
            "transport": self.transport.to_dict(),
            "pipes": {
                n: {
                    "pipe": e["pipe"].to_dict(),
                    "boundary_nodes": e["boundary_nodes"],
                    "direction": e["direction"],
                }
                for n, e in self.pipes.items()
            },
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        if set(data) != {"model", "transport", "pipes"} or data["model"] not in (
            "pr_equilibrium_euler_plant_v1",
            "pr_active_multifluid_plant_v1",
        ):
            raise ValueError("Unknown Euler plant model")
        return cls(
            PREquilibriumTransportNetwork.from_dict(data["transport"], device=device),
            {
                n: {
                    **e,
                    "pipe": (
                        PRActiveMultifluidPipe
                        if e["pipe"]["model"]
                        in (
                            "pr_active_multifluid_pipe_v1",
                            "pr_active_multifluid_pipe_v2",
                            "pr_active_multifluid_pipe_v3",
                        )
                        else PREquilibriumEulerPipe
                    ).from_dict(e["pipe"], device=device),
                }
                for n, e in data["pipes"].items()
            },
        )

    @property
    def maximum_step_s(self):
        return min(
            self.transport.maximum_step_s, *(e["pipe"].maximum_step_s for e in self.pipes.values())
        )

    def initialize(self, transport_state, pipe_states):
        state = PREquilibriumEulerPlantState(
            transport_state, tuple(sorted(pipe_states.items())), self.maximum_step_s, self.model_id
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, PREquilibriumEulerPlantState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
            or tuple(n for n, _ in state.pipes) != tuple(self.pipes)
        ):
            raise PropertyModelError("Euler plant model/state differs")
        self.transport.inspect(state.transport)
        for name, pipe_state in state.pipes:
            if pipe_state.time_s != state.time_s:
                raise PropertyModelError("Euler plant submodels have different clocks")
            self.pipes[name]["pipe"].inspect(pipe_state)

    def ledger(self, old, new, external_energy, external_impulse):
        nc = len(self.fluid.names)
        r = [
            math.fsum(
                b.component_moles[j] - a.component_moles[j]
                for (_, a), (_, b) in zip(old.transport.nodes, new.transport.nodes)
            )
            for j in range(nc)
        ]
        de = math.fsum(
            b.total_energy_j
            - a.total_energy_j
            + (
                node.wall.heat_capacity_j_k * (b.wall_temperature_k - a.wall_temperature_k)
                if node.wall
                else 0.0
            )
            for node, (_, a), (_, b) in zip(
                self.transport.nodes, old.transport.nodes, new.transport.nodes
            )
        )
        de += self.transport.rotor_energy(new.transport) - self.transport.rotor_energy(
            old.transport
        )
        momentum = [0.0, 0.0, 0.0]
        for (name, a), (_, b) in zip(old.pipes, new.pipes):
            for va, vb in zip(a.bulk_stocks, b.bulk_stocks):
                for j in range(nc):
                    r[j] += vb[j] - va[j]
                de += vb[-1] - va[-1]
                for j in range(3):
                    momentum[j] += (vb[-2] - va[-2]) * self.pipes[name]["direction"][j]
        de -= external_energy
        momentum = [v - external_impulse[j] for j, v in enumerate(momentum)]
        if max(map(abs, r)) > 1e-8 or abs(de) > 1e-5 or max(map(abs, momentum)) > 1e-8:
            raise ConservationError("Euler plant component/energy/vector momentum ledger failed")
        return r + [de] + momentum

    def _active_boundary(self, pipe, pipe_cell, node_state):
        phases = []
        for amounts, root in zip(node_state.phase_component_moles, node_state.phase_roots):
            amount = math.fsum(amounts)
            q = self.fluid.homogeneous(
                node_state.pressure_pa,
                node_state.temperature_k,
                [v / amount for v in amounts],
                root,
            )
            phases.append(
                PRRelaxationPhase(
                    tuple(amounts),
                    amount * float(q["molar_volume_m3_mol"].cpu()),
                    node_state.pressure_pa,
                    node_state.temperature_k,
                    0.0,
                    amount * float(q["molar_internal_energy_j_mol"].cpu()),
                )
            )
        # Reuse marker/root lineage selection; no equilibrium projection of pipe phases.
        target = PRPhaseEquilibriumState(
            tuple(phases), tuple(node_state.phase_roots), "boundary", 0.0
        )
        return dict(
            pipe._assign_source(
                pipe_cell, target, {"phase_pairs": [(None, j, 1.0) for j in range(len(phases))]}
            )
        )

    def step(
        self,
        state,
        dt_s,
        *,
        openings=None,
        pipe_heat_w=None,
        pipe_heat_temperatures_k=None,
        graph_heat_w=None,
        graph_utility_temperature_k=None,
        drive_torque_nm=None,
        restriction_openings=None,
    ):
        self.inspect(state)
        dt = _number(dt_s, "plant step", positive=True)
        end = state.time_s + dt
        dt = end - state.time_s
        if dt <= 0 or dt > self.maximum_step_s * (1 + 1e-12):
            raise ValueError("Plant step violates its physical clock or CFL limit")
        commands = dict(self.transport.default_openings)
        if openings:
            if set(openings) - set(commands) or any(type(v) is not bool for v in openings.values()):
                raise ValueError("Unknown/nonboolean equipment opening")
            commands.update(openings)
        pipe_heat_w = pipe_heat_w or {}
        pipe_heat_temperatures_k = pipe_heat_temperatures_k or {}
        if (set(pipe_heat_w) | set(pipe_heat_temperatures_k)) - set(self.pipes):
            raise ValueError("Unknown heated pipe")
        graph_heat_w = graph_heat_w or {}
        graph_utility_temperature_k = graph_utility_temperature_k or {}
        if (set(graph_heat_w) | set(graph_utility_temperature_k)) - set(self.transport.names):
            raise ValueError("Unknown heated vessel")
        heat = [_number(graph_heat_w.get(n, 0.0), "vessel heat") for n in self.transport.names]
        utility = [
            _number(
                graph_utility_temperature_k.get(n, dict(state.transport.nodes)[n].temperature_k),
                "utility temperature",
                positive=True,
            )
            for n in self.transport.names
        ]
        if any(
            v and n not in graph_utility_temperature_k for n, v in zip(self.transport.names, heat)
        ):
            raise ValueError("Vessel heat needs a reservoir temperature")
        graph_net = [
            self.fluid.tensor([0.0] * (len(self.fluid.names) + 2)) for _ in self.transport.nodes
        ]
        graph_entropy = 0.0
        new_pipes = []
        reports = {}
        interfaces = []
        graph_data = {
            i: self.transport._state_data(i, s) for i, (_, s) in enumerate(state.transport.nodes)
        }
        for name, old in state.pipes:
            entry = self.pipes[name]
            pipe = entry["pipe"]
            if isinstance(pipe, PRActiveMultifluidPipe):
                boundaries = [
                    None
                    if node is None
                    else self._active_boundary(
                        pipe, old.cells[0 if side == 0 else -1], dict(state.transport.nodes)[node]
                    )
                    for side, node in enumerate(entry["boundary_nodes"])
                ]
                new, r = pipe.step(
                    old,
                    dt,
                    boundary_cells=boundaries,
                    heat_rates_w=pipe_heat_w.get(name),
                    heat_temperatures_k=pipe_heat_temperatures_k.get(name),
                )
                for side, node in enumerate(entry["boundary_nodes"]):
                    if node is None:
                        continue
                    sign = 1.0 if side == 0 else -1.0
                    fan = self.fluid.tensor(r["face_fluxes"][0 if side == 0 else -1]).sum(0)
                    index = self.transport.names.index(node)
                    graph_flow = -sign * pipe.area * fan
                    graph_net[index] = graph_net[index] + torch.cat(
                        (graph_flow[:-2], graph_flow.new_zeros(1), graph_flow[-1:])
                    )
                    sf = r["boundary_entropy_fluxes"][side]
                    graph_entropy -= sign * pipe.area * sf
                    interfaces.append(
                        {
                            "pipe": name,
                            "side": side,
                            "node": node,
                            "flux": fan.cpu().tolist(),
                            "entropy_flux_w_m2_k": sf,
                            "area_m2": pipe.area,
                        }
                    )
                r = {
                    **r,
                    "wall_impulse_kg_m_s": r["wall_impulse_kg_m_s"] + r["external_transfer"][-1],
                    "external_heat_j": r.get("external_heat_j", 0.0),
                }
                new_pipes.append((name, new))
                reports[name] = r
                continue
            packet = pipe.inspect(old)
            boundary = [None, None]
            entropy_flux = [0.0, 0.0]
            for side, node_name in enumerate(entry["boundary_nodes"]):
                if node_name is None:
                    continue
                index = self.transport.names.index(node_name)
                d = graph_data[index]
                node = self.transport.nodes[index]
                if not bool((d["q"]["frozen_sound_speed_m_s"] <= pipe.wave_speed).all().cpu()):
                    raise PropertyModelError("Vessel acoustic envelope exceeds interface bound")
                vessel = torch.cat(
                    (
                        d["n"] / node.volume_m3,
                        d["p"].new_zeros(1),
                        (d["internal"] / node.volume_m3).reshape(1),
                    )
                )
                vessel_flux = vessel * 0
                vessel_flux[-2] = d["p"]
                ci = 0 if side == 0 else -1
                cell = packet["conserved"][ci]
                cell_flux = packet["flux"][ci]
                cell_entropy = pipe.equilibration.inspect(old.cells[ci]) / pipe.volumes[ci]
                cell_velocity = float(
                    (cell[-2] / (cell[:-2] * self.fluid.molar_masses).sum()).cpu()
                )
                vessel_entropy = float((d["entropy"] / node.volume_m3).cpu())
                left, right, lf, rf = (
                    (vessel, cell, vessel_flux, cell_flux)
                    if side == 0
                    else (cell, vessel, cell_flux, vessel_flux)
                )
                fan = 0.5 * (lf + rf) - 0.5 * pipe.wave_speed * (right - left)
                sl, sr = (
                    (vessel_entropy, cell_entropy) if side == 0 else (cell_entropy, vessel_entropy)
                )
                entropy_rate = 0.5 * cell_entropy * cell_velocity - 0.5 * pipe.wave_speed * (
                    sr - sl
                )
                boundary[side] = fan
                entropy_flux[side] = entropy_rate
                sign = 1.0 if side == 0 else -1.0
                graph_flow = -sign * pipe.area * fan
                graph_net[index] = graph_net[index] + torch.cat(
                    (graph_flow[:-2], graph_flow.new_zeros(1), graph_flow[-1:])
                )
                graph_entropy -= sign * pipe.area * entropy_rate
                interfaces.append(
                    {
                        "pipe": name,
                        "side": side,
                        "node": node_name,
                        "flux": fan.detach().cpu().tolist(),
                        "entropy_flux_w_m2_k": entropy_rate,
                        "area_m2": pipe.area,
                    }
                )
            new, r = pipe.step(
                old,
                dt,
                heat_rates_w=pipe_heat_w.get(name),
                heat_temperatures_k=pipe_heat_temperatures_k.get(name),
                _boundary_fluxes=boundary,
                _boundary_entropy_fluxes=entropy_flux,
            )
            new_pipes.append((name, new))
            reports[name] = r
        coupling = _KnownFaceExchange(
            self.transport, state.transport, dt, commands, graph_net, graph_entropy
        )
        graph, gr = self.transport._step(
            state.transport,
            dt,
            commands,
            heat,
            utility,
            _coupling=coupling,
            drive_torque_nm=drive_torque_nm,
            restriction_openings=restriction_openings,
        )
        new = PREquilibriumEulerPlantState(
            graph, tuple(new_pipes), state.next_step_s, self.model_id
        )
        self.inspect(new)
        external = (
            gr["ambient_heat_j"]
            + gr["external_heat_j"]
            + gr["external_work_j"]
            + math.fsum(r["external_heat_j"] for r in reports.values())
        )
        impulse = [
            math.fsum(
                reports[n]["wall_impulse_kg_m_s"] * e["direction"][j] for n, e in self.pipes.items()
            )
            for j in range(3)
        ]
        entropy = gr["entropy_production_j_k"] + math.fsum(
            v["entropy_production_j_k"] for v in reports.values()
        )
        if entropy < -1e-6 - 1e-7 * len(reports):
            raise ConservationError("Euler plant total entropy balance failed")
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "pipes": reports,
            "transport": gr,
            "interfaces": interfaces,
            "external_energy_j": external,
            "external_impulse_kg_m_s": impulse,
            "entropy_production_j_k": entropy,
            "conservation_residuals": self.ledger(state, new, external, impulse),
        }

    def advance(
        self,
        state,
        duration_s,
        *,
        maximum_step_s=None,
        minimum_step_s=1e-12,
        maximum_attempts=10000,
        **commands,
    ):
        self.inspect(state)
        duration = _number(duration_s, "plant interval", positive=True)
        maximum = (
            self.maximum_step_s
            if maximum_step_s is None
            else min(
                self.maximum_step_s, _number(maximum_step_s, "plant maximum step", positive=True)
            )
        )
        minimum = _number(minimum_step_s, "plant minimum step", positive=True)
        if type(maximum_attempts) is not int or maximum_attempts < 1 or minimum > maximum:
            raise ValueError("Invalid plant integration budget")
        start = state
        end = state.time_s + duration
        dt = min(state.next_step_s, maximum)
        steps = []
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Plant interval below clock resolution")
        for attempt in range(maximum_attempts):
            proposed = min(dt, maximum, end - state.time_s)
            target = min(end, state.time_s + proposed)
            if target - state.time_s > proposed:
                target = math.nextafter(target, state.time_s)
            dt = target - state.time_s
            if dt <= 0:
                raise ValueError("Plant CFL below clock resolution")
            try:
                new, r = self.step(state, dt, **commands)
            except (HydraulicConvergenceError, ConservationError, PropertyModelError):
                dt *= 0.5
                if dt < minimum:
                    raise
                continue
            steps.append(r)
            dt = min(maximum, dt * 1.25)
            state = replace(new, next_step_s=dt)
            if state.time_s >= end:
                energy = math.fsum(r["external_energy_j"] for r in steps)
                impulse = [
                    math.fsum(r["external_impulse_kg_m_s"][j] for r in steps) for j in range(3)
                ]
                return {
                    "state": state.to_dict(),
                    "steps": steps,
                    "attempts": attempt + 1,
                    "conservation_residuals": self.ledger(start, state, energy, impulse),
                }
        raise HydraulicConvergenceError("Euler plant interval budget exhausted")
