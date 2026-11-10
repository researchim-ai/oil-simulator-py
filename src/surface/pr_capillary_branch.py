"""Finite-inventory ideal mixing chambers coupled to PR capillary pipe cells.

Each junction has its own N/V/E and phase temperatures. Its mean momentum is
constrained to zero: port impulse is absorbed by the stationary chamber,
while incoming kinetic energy remains in its caloric inventory. A common
strip-height graph energy is differentiated jointly over pipes and chambers.
This is a lumped chamber model, not resolved three-dimensional junction CFD.
"""

import math
from dataclasses import asdict, dataclass, replace

import torch

from .liquid import ConservationError, _number
from .operation import _digest
from .pr_active_multifluid_flux import fluctuations
from .pr_active_multifluid_pipe import PRActiveMultifluidPipe, PRActiveMultifluidState
from .pr_capillary_axial import PRCapillaryAxialPipe
from .pr_capillary_pressure import PRCapillaryPressure
from .pr_interphase_relaxation import PRRelaxationPhase


class PRCapillaryGraphPressure(PRCapillaryPressure):
    def __init__(self, pipe, edges, **parameters):
        self.surface_edges = tuple(tuple(e) for e in edges)
        if any(
            len(e) != 2 or e[0] == e[1] or min(e) < 0 or max(e) >= len(pipe.lengths)
            for e in self.surface_edges
        ) or len({tuple(sorted(e)) for e in self.surface_edges}) != len(self.surface_edges):
            raise ValueError("Unique physical chamber/pipe surface edges required")
        super().__init__(pipe, **parameters)
        if any(h is not None for h in self.boundary_heights):
            raise ValueError("Graph boundary heights need explicit external ports")
        self.set_geometry(self.surface_edges)
        self.distances = pipe.fluid.tensor(
            [(pipe.lengths[i] + pipe.lengths[j]) / 2 for i, j in self.surface_edges]
        )
        self.weights = self.sigma * self.width / self.distances
        self.height_offsets = pipe.fluid.tensor([0.0] * len(self.surface_edges))
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            **super().to_dict(),
            "model": "pr_capillary_graph_pressure_v1",
            "surface_edges": [list(e) for e in self.surface_edges],
        }


class _BranchSolver(PRCapillaryAxialPipe):
    def __init__(self, source, faces, pinned, spec):
        self.source, self.pipe = source, source.pipe
        self.spec = spec
        self.maximum_unknowns = spec["maximum_unknowns"]
        self.volume_diffusion = spec.get("volume_diffusion", "acoustic")
        self.linear_solver = spec.get("linear_solver", "auto")
        self.boundary_geometry = "extrapolated_face"
        self.boundaries = [None, None]
        self.boundary_packets = [None, None]
        self.boundary_entropy = [None, None]
        self.faces = tuple(faces)
        f = source.pipe.fluid
        self.left = f.tensor([e[0] for e in faces]).long()
        self.right = f.tensor([e[1] for e in faces]).long()
        self.wall_left = torch.tensor([e[2] == 0 for e in faces], device=f.device)
        self.wall_right = torch.tensor([e[2] == 1 for e in faces], device=f.device)
        self.internal = ~(self.wall_left | self.wall_right)
        self.directions = f.tensor([e[3] for e in faces])
        self.pinned_cells = torch.tensor(pinned, device=f.device)
        lengths = source.pipe.lengths
        distances = [(lengths[e[0]] + lengths[e[1]]) / 2 for e in faces]
        self.face_speed = f.tensor([source.pipe.wave_speed * max(distances) / d for d in distances])
        # A sum of momenta in differently oriented pipes is not a global scalar.
        # Their local equations and the vector chamber-support balance are separate.
        self.ledger_indices = tuple(range(len(f.names) + 1))
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return self.spec

    def source_terms(self, packet):
        q = packet["conserved"]
        f = self.pipe.fluid
        nc = len(f.names)
        numbers = torch.zeros_like(q[..., :nc])
        energy = torch.zeros_like(q[..., -1])
        for index, phase, rate, composition, enthalpy, _ in self.inlets:
            target = torch.tensor([index], device=f.device)
            mask = f.tensor([float(j == phase) for j in range(2)])
            row = rate * composition[None, None, :] * mask[None, :, None]
            numbers = numbers.index_add(0, target, row)
            erow = rate * enthalpy * mask[None, :]
            energy = energy.index_add(0, target, erow)
        for index, phase, rate in self.withdrawals:
            density = q[index, phase, :nc].sum()
            composition = q[index, phase, :nc] / density
            enthalpy = (
                q[index, phase, -1]
                + packet["pressure"][index, phase] * packet["end_alpha"][index, phase]
            ) / density
            target = torch.tensor([index], device=f.device)
            mask = f.tensor([float(j == phase) for j in range(2)])
            row = -rate * composition[None, None, :] * mask[None, :, None]
            numbers = numbers.index_add(0, target, row)
            erow = -rate * enthalpy * mask[None, :]
            energy = energy.index_add(0, target, erow)
        for index, phase, power, _ in self.heaters:
            target = torch.tensor([index], device=f.device)
            row = power * f.tensor([[float(j == phase) for j in range(2)]])
            energy = energy.index_add(0, target, row)
        return numbers, energy

    def extra_incoming_entropy(self, before, after, dt):
        f = self.pipe.fluid
        entropy = sum(rate * sm for _, _, rate, _, _, sm in self.inlets)
        for index, phase, rate in self.withdrawals:
            stored = dict(after.cells[index])[self.pipe.names[phase]]
            total = math.fsum(stored.component_moles)
            eos = f.homogeneous(
                stored.pressure_pa,
                stored.temperature_k,
                [v / total for v in stored.component_moles],
                self.pipe.slots[phase]["root"],
            )
            entropy -= rate * float(eos["molar_entropy_j_mol_k"].cpu())
        entropy += sum(power / temperature for _, _, power, temperature in self.heaters)
        return dt * entropy

    def _face_fields(self, packet):
        p = self.pipe
        left = {k: v[self.left] for k, v in packet.items()}
        right = {k: v[self.right] for k, v in packet.items()}

        def reflect(rows, mask):
            sign = torch.where(mask, -1.0, 1.0)[:, None]
            rows["velocity"] = rows["velocity"] * sign
            q = rows["conserved"]
            rows["conserved"] = torch.cat(
                (q[..., :-2], (q[..., -2] * sign)[..., None], q[..., -1:]), -1
            )
            q = rows["flux"]
            rows["flux"] = torch.cat(
                (q[..., :-2] * sign[..., None], q[..., -2:-1], (q[..., -1] * sign)[..., None]), -1
            )

        reflect(left, self.wall_left)
        reflect(right, self.wall_right)
        _, _, _, _, ff = fluctuations(left, right, p.wave_speed)
        intrinsic_l, intrinsic_r = self.intrinsic_pair(left, right)
        jump = intrinsic_r * right["alpha"][..., None] - intrinsic_l * left["alpha"][..., None]
        if self.volume_diffusion == "none":
            jump = (left["alpha"] + right["alpha"])[..., None] / 2 * (intrinsic_r - intrinsic_l)
        flux = (left["flux"] + right["flux"]) / 2 - self.face_speed[:, None, None] / 2 * jump
        return left, right, ff, flux

    def _spatial(self, packet, h0, h1, *, potential_override=None):
        hmid = (h0 + h1) / 2
        p, s = self.pipe, self.source
        left, right, ff, flux = self._face_fields(packet)
        delta = torch.zeros_like(packet["conserved"])
        delta = delta.index_add(0, self.left, flux + ff["phase_path"] / 2)
        delta = delta.index_add(0, self.right, -flux + ff["phase_path"] / 2)
        source_n, source_e = self.source_terms(packet)
        delta = (
            delta
            - torch.cat((source_n, torch.zeros_like(source_e)[..., None], source_e[..., None]), -1)
            / p.area
        )
        # A wall ghost is not an owned cell: remove its fluctuation contribution.
        delta = delta.index_add(
            0,
            self.left,
            torch.where(self.wall_left[:, None, None], -(flux + ff["phase_path"] / 2), 0.0),
        )
        delta = delta.index_add(
            0,
            self.right,
            torch.where(self.wall_right[:, None, None], flux - ff["phase_path"] / 2, 0.0),
        )
        path = ff["alpha_path"][:, 0]
        contribution = -p.area / 2 * path
        contribution = torch.where(self.internal, contribution, 0.0)
        physical = (
            torch.zeros_like(hmid)
            .index_add(0, self.left, contribution)
            .index_add(0, self.right, contribution)
        )
        diffusion = p.area * self.face_speed / 2 * (right["alpha"][:, 0] - left["alpha"][:, 0])
        numerical = (
            torch.zeros_like(hmid)
            .index_add(0, self.left, diffusion)
            .index_add(0, self.right, -diffusion)
        )
        if self.volume_diffusion == "none":
            numerical = torch.zeros_like(numerical)
        potential = (
            s.discrete_pressure(h0, h1) if potential_override is None else potential_override
        )
        al, ar = left["alpha"], right["alpha"]
        coeff_l, coeff_r = (2 * al + ar) / 6, (al + 2 * ar) / 6
        factor = p.area / 2 * (potential[self.left] + potential[self.right]) * (ar[:, 0] - al[:, 0])
        factor = torch.where(self.internal, factor, 0.0)
        force_l = factor[:, None] * coeff_l
        force_r = factor[:, None] * coeff_r
        force = (
            torch.zeros_like(packet["velocity"])
            .index_add(0, self.left, force_l)
            .index_add(0, self.right, force_r)
        )
        heat = -potential * numerical
        wall = (
            p.area
            * flux[:, :, -2].sum(-1)
            * (self.wall_left.to(flux.dtype) - self.wall_right.to(flux.dtype))
        ).sum()
        return (
            delta,
            physical,
            numerical,
            force,
            heat,
            wall,
            torch.cat((source_n.sum((0, 1)), flux.new_zeros(1), source_e.sum().reshape(1))),
            wall * 0,
            wall * 0,
            flux.new_ones((0, 2)),
        )

    def momentum_report(self, before, after):
        p, s, f = self.pipe, self.source, self.pipe.fluid
        old, new = p.inspect(before), p.inspect(after)
        packet, h0, h1 = self.state_packet(before, after)
        q = packet["conserved"]
        left, right, ff, faces = self._face_fields(packet)
        potential = s.discrete_pressure(h0, h1)
        al, ar = left["alpha"], right["alpha"]
        coefficient = (
            p.area / 2 * (potential[self.left] + potential[self.right]) * (ar[:, 0] - al[:, 0])
        )
        coefficient = torch.where(self.internal, coefficient, 0.0)
        force_l = coefficient[:, None] * (2 * al + ar) / 6
        force_r = coefficient[:, None] * (al + 2 * ar) / 6
        reaction_l = p.area * (faces[:, :, -2] + ff["phase_path"][:, :, -2] / 2) - force_l
        reaction_r = p.area * (-faces[:, :, -2] + ff["phase_path"][:, :, -2] / 2) - force_r
        reaction_l = torch.where(
            (self.pinned_cells[self.left] & ~self.wall_left)[:, None], reaction_l, 0.0
        )
        reaction_r = torch.where(
            (self.pinned_cells[self.right] & ~self.wall_right)[:, None], reaction_r, 0.0
        )
        reaction = torch.zeros((len(p.lengths), 2, 3), dtype=q.dtype, device=f.device)
        reaction = reaction.index_add(
            0, self.left, reaction_l[:, :, None] * self.directions[:, None, :]
        )
        reaction = reaction.index_add(
            0, self.right, reaction_r[:, :, None] * self.directions[:, None, :]
        )
        dt = after.time_s - before.time_s
        wall = (
            p.area
            * faces[:, :, -2].sum(-1)
            * (self.wall_left.to(q.dtype) - self.wall_right.to(q.dtype))
        )[:, None] * self.directions
        surface = (force_l + force_r).sum(-1)[:, None] * self.directions
        volumes = f.tensor(p.volumes)
        initial = (
            (old["conserved"][:, :, -2] * volumes[:, None]).sum(-1)[:, None] * self.body_axes
        ).sum(0)
        final = (
            (new["conserved"][:, :, -2] * volumes[:, None]).sum(-1)[:, None] * self.body_axes
        ).sum(0)
        wall_impulse = dt * wall.sum(0)
        surface_impulse = dt * surface.sum(0)
        reactions = dt * reaction.sum(1)
        defect = final - initial - wall_impulse - surface_impulse - reactions.sum(0)
        scale = (
            initial.abs()
            + final.abs()
            + wall_impulse.abs()
            + surface_impulse.abs()
            + reactions.abs().sum(0)
        )
        if bool((defect.abs() > 2e-10 * scale + 1e-22).any().cpu()):
            raise ConservationError("Branch pipe/chamber-support vector momentum balance failed")
        return {
            "fluid_momentum_before_kg_m_s": initial.cpu().tolist(),
            "fluid_momentum_after_kg_m_s": final.cpu().tolist(),
            "wall_impulse_vector_kg_m_s": wall_impulse.cpu().tolist(),
            "surface_force_resultant_impulse_kg_m_s": surface_impulse.cpu().tolist(),
            "junction_support_impulses_kg_m_s": {
                name: reactions[i].cpu().tolist() for name, i in self.junction_indices.items()
            },
            "vector_momentum_residual_kg_m_s": defect.cpu().tolist(),
        }


@dataclass(frozen=True)
class PRCapillaryBranchState:
    pipes: tuple
    junctions: tuple
    time_s: float
    next_step_s: float
    model_id: str

    def to_dict(self):
        return {
            "pipes": {name: state.to_dict() for name, state in self.pipes},
            "junctions": {
                name: {n: asdict(p) for n, p in phases} for name, phases in self.junctions
            },
            "time_s": self.time_s,
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, raw):
        if set(raw) != {"pipes", "junctions", "time_s", "next_step_s", "model_id"}:
            raise ValueError("Explicit finite-chamber graph checkpoint required")
        return cls(
            tuple(
                (n, PRActiveMultifluidState.from_dict(v)) for n, v in sorted(raw["pipes"].items())
            ),
            tuple(
                (n, tuple((k, PRRelaxationPhase.from_dict(p)) for k, p in v.items()))
                for n, v in sorted(raw["junctions"].items())
            ),
            raw["time_s"],
            raw["next_step_s"],
            raw["model_id"],
        )


class PRCapillaryBranchNetwork:
    def __init__(
        self,
        pipes,
        junction_volumes_m3,
        parameters,
        *,
        maximum_unknowns=192,
        sources=None,
        openings=None,
        volume_diffusion="acoustic",
        linear_solver="auto",
    ):
        if type(maximum_unknowns) is not int or maximum_unknowns <= 0:
            raise ValueError("Positive dense unknown budget required")
        if not pipes or not junction_volumes_m3:
            raise ValueError("Physical pipes and finite mixing chamber volumes required")
        self.pipes = dict(sorted(pipes.items()))
        self.junction_volumes = {
            n: _number(v, "junction volume", positive=True)
            for n, v in sorted(junction_volumes_m3.items())
        }
        first = next(iter(self.pipes.values()))["pipe"]
        self.fluid = first.fluid
        self.parameters = dict(parameters)
        self.maximum_unknowns = maximum_unknowns
        if linear_solver not in ("auto", "dense", "sparse"):
            raise ValueError("Capillary linear solver must be auto, dense or sparse")
        self.linear_solver = linear_solver
        self.offsets = {}
        lengths = []
        pinned = []
        self.axes = []
        cursor = 0
        for name, entry in self.pipes.items():
            if set(entry) != {"pipe", "boundary_nodes", "axis"}:
                raise ValueError("Pipe, two endpoint bindings and physical unit axis required")
            p = entry["pipe"]
            axis = tuple(entry["axis"])
            if (
                len(axis) != 3
                or not all(math.isfinite(x) for x in axis)
                or abs(math.fsum(x * x for x in axis) - 1) > 1e-12
                or abs(axis[2]) > 1e-12
            ):
                raise ValueError("Finite unit pipe axis required")
            if (
                p.boundary_mode != "walls"
                or p.gravity
                or p.fluid.model_id != self.fluid.model_id
                or p.fluid.device != self.fluid.device
                or p.diameter != first.diameter
                or p.slots != first.slots
                or p.wave_speed != first.wave_speed
                or p.chemical is not None
                or p.friction is not None
            ):
                raise ValueError(
                    "Common PR section/registry/wave bound and no other pipe split required"
                )
            ends = tuple(entry["boundary_nodes"])
            if len(ends) != 2 or any(
                n is not None and n not in self.junction_volumes for n in ends
            ):
                raise ValueError(
                    "Two known finite-junction or reflecting endpoint bindings required"
                )
            self.offsets[name] = (cursor, cursor + len(p.lengths))
            cursor += len(p.lengths)
            lengths.extend(p.lengths)
            pinned.extend([False] * len(p.lengths))
            self.axes.extend([axis] * len(p.lengths))
        self.junction_indices = {n: cursor + i for i, n in enumerate(self.junction_volumes)}
        lengths.extend(v / first.area for v in self.junction_volumes.values())
        pinned.extend([True] * len(self.junction_volumes))
        self.axes.extend([(0.0, 0.0, 0.0)] * len(self.junction_volumes))
        port_names = {
            f"{name}:{side}"
            for name, entry in self.pipes.items()
            for side, node in enumerate(entry["boundary_nodes"])
            if node is not None
        }
        commands = {} if openings is None else dict(openings)
        if set(commands) - port_names or any(
            type(value) is not bool for value in commands.values()
        ):
            raise ValueError("Known junction ports and binary openings required")
        self.openings = {key: commands.get(key, True) for key in sorted(port_names)}
        faces = []
        connected = set()
        for name, entry in self.pipes.items():
            start, end = self.offsets[name]
            axis = tuple(entry["axis"])
            faces.extend((i, i + 1, -1, axis) for i in range(start, end - 1))
            for side, node in enumerate(entry["boundary_nodes"]):
                cell = start if side == 0 else end - 1
                if node is None:
                    faces.append((cell, cell, side, axis))
                else:
                    connected.add(node)
                    j = self.junction_indices[node]
                    if self.openings[f"{name}:{side}"]:
                        faces.append((j, cell, -1, axis) if side == 0 else (cell, j, -1, axis))
                    else:
                        faces.append((cell, cell, side, axis))
                        faces.append((j, j, 1 - side, axis))
        if connected != set(self.junction_volumes):
            raise ValueError("Every finite junction must bind an actual pipe port")
        viewpipe = PRActiveMultifluidPipe(
            self.fluid, first.slots, lengths, first.diameter, wave_speed_bound_m_s=first.wave_speed
        )
        edges = [(a, b) for a, b, wall, _ in faces if wall == -1]
        self.sources = {}
        inlet_data = []
        sink_data = []
        heat_data = []
        for name, config in sorted((sources or {}).items()):
            if name not in self.junction_indices or set(config) - {
                "inlets",
                "withdrawals",
                "heaters",
            }:
                raise ValueError("External streams/heaters require a known finite junction")
            index = self.junction_indices[name]
            normalized = {"inlets": [], "withdrawals": [], "heaters": []}
            for item in config.get("inlets", []):
                if (
                    set(item) != {"phase", "molar_rate_mol_s", "state"}
                    or item["phase"] not in first.names
                ):
                    raise ValueError("Explicit named phase, molar rate and inlet PR state required")
                rate = _number(item["molar_rate_mol_s"], "inlet rate", positive=True)
                phase = (
                    PRRelaxationPhase.from_dict(item["state"])
                    if isinstance(item["state"], dict)
                    else item["state"]
                )
                if phase.momentum_kg_m_s != 0:
                    raise ValueError("Lumped external stream requires zero mean inlet momentum")
                viewpipe._boundary_packet({item["phase"]: phase})
                j = first.names.index(item["phase"])
                amount = math.fsum(phase.component_moles)
                x = self.fluid.tensor([v / amount for v in phase.component_moles])
                eos = self.fluid.homogeneous(
                    phase.pressure_pa, phase.temperature_k, x, first.slots[j]["root"]
                )
                enthalpy = (phase.total_energy_j + phase.pressure_pa * phase.volume_m3) / amount
                sm = float(eos["molar_entropy_j_mol_k"].cpu())
                inlet_data.append((index, j, rate, x, enthalpy, sm))
                normalized["inlets"].append(
                    {"phase": item["phase"], "molar_rate_mol_s": rate, "state": asdict(phase)}
                )
            for item in config.get("withdrawals", []):
                if set(item) != {"phase", "molar_rate_mol_s"} or item["phase"] not in first.names:
                    raise ValueError("Explicit named phase and molar withdrawal rate required")
                rate = _number(item["molar_rate_mol_s"], "withdrawal rate", positive=True)
                sink_data.append((index, first.names.index(item["phase"]), rate))
                normalized["withdrawals"].append({"phase": item["phase"], "molar_rate_mol_s": rate})
            for item in config.get("heaters", []):
                if (
                    set(item) != {"phase", "power_w", "temperature_k"}
                    or item["phase"] not in first.names
                ):
                    raise ValueError("Explicit phase heater power and temperature required")
                power = _number(item["power_w"], "phase heater power")
                temp = _number(item["temperature_k"], "heater temperature", positive=True)
                heat_data.append((index, first.names.index(item["phase"]), power, temp))
                normalized["heaters"].append(
                    {"phase": item["phase"], "power_w": power, "temperature_k": temp}
                )
            self.sources[name] = normalized
        if volume_diffusion not in ("acoustic", "none"):
            raise ValueError("Explicit acoustic/none volume diffusion required")
        self.volume_diffusion = volume_diffusion
        self.source = PRCapillaryGraphPressure(viewpipe, edges, **parameters)
        if (
            self.source.wetting is not None
            and self.source.wetting.scale > 0
            and volume_diffusion != "none"
        ):
            raise ValueError("Nonconvex material wetting requires volume_diffusion='none'")
        self.parameters = {
            k: v
            for k, v in self.source.to_dict().items()
            if k not in ("model", "pipe_model_id", "surface_edges")
        }
        self.solver = _BranchSolver(self.source, faces, pinned, self.to_dict())
        self.solver.inlets = inlet_data
        self.solver.withdrawals = sink_data
        self.solver.heaters = heat_data
        self.solver.body_axes = self.fluid.tensor(self.axes)
        self.solver.junction_indices = self.junction_indices
        self.model_id = self.solver.model_id

    def to_dict(self):
        result = {
            "model": "pr_capillary_branch_network_v1",
            "pipes": {
                n: {
                    "pipe": v["pipe"].to_dict(),
                    "boundary_nodes": list(v["boundary_nodes"]),
                    "axis": list(v["axis"]),
                }
                for n, v in self.pipes.items()
            },
            "junction_volumes_m3": self.junction_volumes,
            "parameters": self.parameters,
            "maximum_unknowns": self.maximum_unknowns,
            "transport_state": "midpoint_intrinsic",
        }

        if self.volume_diffusion != "acoustic":
            result["volume_diffusion"] = self.volume_diffusion
        if self.linear_solver != "auto":
            result["linear_solver"] = self.linear_solver
        if self.sources:
            result["sources"] = self.sources
        if any(not value for value in self.openings.values()):
            result["openings"] = self.openings
        return result

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        if (
            set(raw)
            - {"sources", "openings", "volume_diffusion", "linear_solver", "transport_state"}
            != {"model", "pipes", "junction_volumes_m3", "parameters", "maximum_unknowns"}
            or raw["model"] != "pr_capillary_branch_network_v1"
            or raw.get("transport_state", "midpoint_intrinsic") != "midpoint_intrinsic"
        ):
            raise ValueError("Explicit finite-chamber PR graph model required")
        pipes = {
            n: {**v, "pipe": PRActiveMultifluidPipe.from_dict(v["pipe"], device=device)}
            for n, v in raw["pipes"].items()
        }
        return cls(
            pipes,
            raw["junction_volumes_m3"],
            raw["parameters"],
            maximum_unknowns=raw["maximum_unknowns"],
            sources=raw.get("sources"),
            openings=raw.get("openings"),
            volume_diffusion=raw.get("volume_diffusion", "acoustic"),
            linear_solver=raw.get("linear_solver", "auto"),
        )

    def _view(self, state):
        if _digest(self.to_dict()) != self.model_id:
            raise ValueError("Changed physical finite-junction graph configuration")
        if (
            state.model_id != self.model_id
            or set(dict(state.pipes)) != set(self.pipes)
            or set(dict(state.junctions)) != set(self.junction_volumes)
        ):
            raise ValueError("Finite-chamber graph registry/identity differs")
        if (
            state.time_s < 0
            or not math.isfinite(state.time_s)
            or state.next_step_s <= 0
            or not math.isfinite(state.next_step_s)
        ):
            raise ValueError("Finite-chamber graph clock/step invalid")
        if tuple(n for n, _ in state.pipes) != tuple(self.pipes) or tuple(
            n for n, _ in state.junctions
        ) != tuple(self.junction_volumes):
            raise ValueError("Canonical named branch inventory order required")
        owned = []
        for name, st in state.pipes:
            self.pipes[name]["pipe"].inspect(st)
            if st.time_s != state.time_s:
                raise ValueError("Pipe and graph clocks differ")
            owned.extend(st.cells)
        for _, phases in state.junctions:
            if any(p.momentum_kg_m_s != 0 for _, p in phases):
                raise ValueError("Ideal mixing chamber has zero mean momentum")
            owned.append(phases)
        view = PRActiveMultifluidState(tuple(owned), state.time_s, self.source.pipe.model_id)
        self.source.pipe.inspect(view)
        return view

    def initialize(self, pipe_states, junction_phases, *, time_s=0.0, next_step_s=None):
        state = PRCapillaryBranchState(
            tuple(sorted(pipe_states.items())),
            tuple((n, tuple(v.items())) for n, v in sorted(junction_phases.items())),
            time_s,
            self.source.pipe.maximum_step_s * 0.1 if next_step_s is None else next_step_s,
            self.model_id,
        )
        self._view(state)
        return state

    def switch_ports(self, state, commands):
        """Ideal reversible binary isolation with an explicit surface-work port.

        This is an instantaneous boundary-condition event, not resolved valve
        mechanics, contact-angle hysteresis or a finite-speed actuator model.
        No fluid heat is invented to hide the change of surface geometry.
        """
        initial = self._view(state)
        if set(commands) - set(self.openings) or any(
            type(v) is not bool for v in commands.values()
        ):
            raise ValueError("Known junction ports and binary commands required")
        openings = {**self.openings, **commands}
        model = type(self)(
            self.pipes,
            self.junction_volumes,
            self.parameters,
            maximum_unknowns=self.maximum_unknowns,
            sources=self.sources,
            openings=openings,
            volume_diffusion=self.volume_diffusion,
            linear_solver=self.linear_solver,
        )
        end = replace(state, model_id=model.model_id)
        final = model._view(end)
        before_energy = self.source.surface_energy(initial)
        after_energy = model.source.surface_energy(final)
        work = after_energy - before_energy
        oldstock = self.source.pipe._totals(initial)
        newstock = model.source.pipe._totals(final)
        if initial.cells != final.cells or initial.time_s != final.time_s:
            raise ConservationError("Isolation event changed physical phase inventory")
        return (
            model,
            end,
            {
                "state": end.to_dict(),
                "model": model.to_dict(),
                "old_model_id": self.model_id,
                "new_model_id": model.model_id,
                "commands": dict(commands),
                "clock_advanced": False,
                "surface_energy_before_j": before_energy,
                "surface_energy_after_j": after_energy,
                "external_actuator_work_j": work,
                "component_residuals_mol": [
                    b - a for a, b in zip(oldstock[:-2], newstock[:-2], strict=True)
                ],
                "bulk_energy_residual_j": newstock[-2] - oldstock[-2],
                "augmented_energy_residual_j": math.fsum(
                    [newstock[-2] - oldstock[-2], after_energy, -before_energy, -work]
                ),
                "qualification": "Ideal reversible isolation work on declared graph strip energy; no meniscus/wetting/actuator calibration",
            },
        )

    def advance(self, state, duration_s, *, normal_relaxation=True, **options):
        initial = self._view(state)
        options.setdefault("initial_step_s", state.next_step_s)
        final, record = self.solver.advance(
            initial, duration_s, normal_relaxation=normal_relaxation, **options
        )
        previous = initial
        for step in record["steps"]:
            axial = step["axial"] if normal_relaxation else step
            current = PRActiveMultifluidState.from_dict(axial["state"])
            axial["momentum_balance"] = self.solver.momentum_report(previous, current)
            axial["conservation_field_order"] = [
                *self.fluid.names,
                "total_bulk_plus_surface_energy_j",
            ]
            axial["sum_local_wall_impulses_kg_m_s"] = axial.pop("wall_impulse_kg_m_s")
            axial["force_basis"] = (
                "Pipe axial components; stationary chamber auxiliaries are port sums; physical chamber reactions are vectors"
            )
            previous = PRActiveMultifluidState.from_dict(step["state"])
        pipes = tuple(
            (n, replace(old, cells=final.cells[slice(*self.offsets[n])], time_s=final.time_s))
            for n, old in state.pipes
        )
        junctions = tuple((n, final.cells[i]) for n, i in self.junction_indices.items())
        end = PRCapillaryBranchState(
            pipes, junctions, final.time_s, record["next_step_s"], self.model_id
        )
        self._view(end)
        return end, {
            "state": end.to_dict(),
            "body_initial": initial.to_dict(),
            "advance": record,
            "normal_relaxation": normal_relaxation,
            "qualification": "Finite stationary ideal mixing chambers, common PR strip section, positive phase/component support; not junction CFD or calibrated equipment",
        }
