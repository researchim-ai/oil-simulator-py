"""Monolithic Torch PR vessels with conservative, upwind inter-vessel transfer.

Lumped common-elevation vessels and empirical linear connections. No line
inventory, gravity head, choking, entrainment or thermal reservoir is inferred.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import torch

from .component_energy import ComponentEnergyStream
from .compositional_separator import CompositionalSeparator, CompositionalSeparatorState
from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    PropertyModelError,
    _name,
    _number,
)
from .operation import _digest
from .phase_geometry import EquilibriumLevelControl, pickup_fractions, validate_pickup
from .torch_linear import TorchNewton


@dataclass(frozen=True)
class CompositionalConnection:
    name: str
    source: str
    target: str
    volume_conductance_m3_s_pa: float
    allow_reverse: bool = True
    opening: float = 1.0
    maximum_pressure_drop_fraction: float = 0.05
    source_selection: str = "bulk"
    target_selection: str = "bulk"
    source_component: str | None = None
    target_component: str | None = None
    minimum_mole_fraction: float = 0.5
    source_height_m: float | None = None
    source_diameter_m: float | None = None
    target_height_m: float | None = None
    target_diameter_m: float | None = None
    level_control: EquilibriumLevelControl | None = None
    control_side: str = "source"

    def __post_init__(self):
        for name in (self.name, self.source, self.target):
            _name(name)
        if self.source == self.target:
            raise ValueError("Connection cannot be a self loop")
        if not isinstance(self.allow_reverse, bool):
            raise TypeError("allow_reverse must be boolean")
        _number(self.volume_conductance_m3_s_pa, "conductance", nonnegative=True)
        for key in ("opening", "minimum_mole_fraction"):
            if not 0 <= _number(getattr(self, key), key) <= 1:
                raise ValueError(f"{key} outside [0,1]")
        if not 0 < _number(self.maximum_pressure_drop_fraction, "pressure drop") <= 1:
            raise ValueError("Invalid declared connection pressure-drop domain")
        for end in ("source", "target"):
            selection = getattr(self, end + "_selection")
            component = getattr(self, end + "_component")
            if selection not in ("bulk", "densest", "lightest", "component_rich"):
                raise ValueError("Unknown connection phase selection")
            if (component is not None) != (selection == "component_rich"):
                raise ValueError("component_rich requires a named component")
            if component is not None:
                _name(component)
            height, diameter = getattr(self, end + "_height_m"), getattr(self, end + "_diameter_m")
            if (height is None) != (diameter is None):
                raise ValueError("Pickup needs height and diameter")
            if height is not None:
                _number(height, "height", nonnegative=True)
                _number(diameter, "diameter", positive=True)
        if self.control_side not in ("source", "target"):
            raise ValueError("Unknown level control side")
        if self.level_control is not None and not isinstance(
            self.level_control, EquilibriumLevelControl
        ):
            raise TypeError("Expected EquilibriumLevelControl")


@dataclass(frozen=True)
class CompositionalNetworkState:
    vessels: tuple[tuple[str, CompositionalSeparatorState], ...]
    next_dt_s: float
    model_id: str

    def __post_init__(self):
        if not isinstance(self.vessels, tuple) or not self.vessels:
            raise TypeError("Network state requires immutable named vessels")
        names = [name for name, _ in self.vessels]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate network state vessel")
        for name, state in self.vessels:
            _name(name)
            if not isinstance(state, CompositionalSeparatorState):
                raise TypeError("Invalid network vessel state")
        _number(self.next_dt_s, "next_dt_s", positive=True)
        if not isinstance(self.model_id, str) or len(self.model_id) != 64:
            raise ValueError("Missing network identity")

    def to_dict(self):
        return {
            "vessels": {n: s.to_dict() for n, s in self.vessels},
            "next_dt_s": self.next_dt_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"vessels", "next_dt_s", "model_id"}:
            raise ValueError("Unknown network state fields")
        return cls(
            tuple(
                (n, CompositionalSeparatorState.from_dict(s))
                for n, s in sorted(data["vessels"].items())
            ),
            data["next_dt_s"],
            data["model_id"],
        )


class CompositionalNetwork:
    def __init__(
        self,
        vessels,
        connections=(),
        *,
        maximum_dt_s=0.1,
        minimum_dt_s=1e-7,
        relative_tolerance=1e-4,
        temperature_tolerance_k=0.001,
        temporal_error_control=True,
        maximum_branch_attempts=24,
    ):
        if not isinstance(vessels, dict) or not vessels:
            raise ValueError("Named finite vessels required")
        self.names = tuple(sorted(vessels))
        for n in self.names:
            _name(n)
            if not isinstance(vessels[n], CompositionalSeparator):
                raise TypeError("Network requires CompositionalSeparator vessels")
        self.vessels = tuple(vessels[n] for n in self.names)
        self.fluid = self.vessels[0].fluid
        reference = self.vessels[0]
        for v in self.vessels:
            if v.fluid.model_id != self.fluid.model_id or v.fluid.device != self.fluid.device:
                raise PropertyModelError("Network requires identical EOS, registry and device")
            if (v.gravity_m_s2, v.elevation_m) != (reference.gravity_m_s2, reference.elevation_m):
                raise PropertyModelError(
                    "Network requires one elevation/gravity datum; head is not modeled"
                )
        self.connections = tuple(connections)
        if not all(isinstance(c, CompositionalConnection) for c in self.connections):
            raise TypeError("Invalid connection")
        if len({c.name for c in self.connections}) != len(self.connections):
            raise ValueError("Duplicate connection name")
        self.ends = []
        self.adjacency = [[] for _ in self.names]
        for e, c in enumerate(self.connections):
            if c.source not in vessels or c.target not in vessels:
                raise ValueError("Unknown connection endpoint")
            for component in (c.source_component, c.target_component):
                if component is not None and component not in self.fluid.names:
                    raise ValueError("Unknown connection component")
            a, b = self.names.index(c.source), self.names.index(c.target)
            for side, index in (("source", a), ("target", b)):
                validate_pickup(
                    self.vessels[index].geometry,
                    getattr(c, side + "_height_m"),
                    getattr(c, side + "_diameter_m"),
                )
            control_index = a if c.control_side == "source" else b
            self.vessels[control_index]._validate_level_control(c.level_control)
            self.ends.append((a, b))
            self.adjacency[a].append((e, -1))
            self.adjacency[b].append((e, 1))
        for key, value in (
            ("maximum_dt_s", maximum_dt_s),
            ("minimum_dt_s", minimum_dt_s),
            ("relative_tolerance", relative_tolerance),
            ("temperature_tolerance_k", temperature_tolerance_k),
        ):
            setattr(self, key, _number(value, key, positive=True))
        if minimum_dt_s > maximum_dt_s:
            raise ValueError("Invalid network timestep range")
        if not isinstance(temporal_error_control, bool):
            raise TypeError("temporal_error_control must be boolean")
        if (
            isinstance(maximum_branch_attempts, bool)
            or not isinstance(maximum_branch_attempts, int)
            or maximum_branch_attempts < 1
        ):
            raise ValueError("Positive branch budget required")
        self.temporal_error_control = temporal_error_control
        self.maximum_branch_attempts = maximum_branch_attempts
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "vessels": {n: v.to_dict() for n, v in zip(self.names, self.vessels)},
            "connections": [asdict(c) for c in self.connections],
            **{
                k: getattr(self, k)
                for k in (
                    "maximum_dt_s",
                    "minimum_dt_s",
                    "relative_tolerance",
                    "temperature_tolerance_k",
                    "temporal_error_control",
                    "maximum_branch_attempts",
                )
            },
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        data["vessels"] = {
            n: CompositionalSeparator.from_dict(v, device=device)
            for n, v in data["vessels"].items()
        }
        connections = []
        for connection in data.get("connections", []):
            connection = dict(connection)
            if connection.get("level_control") is not None:
                connection["level_control"] = EquilibriumLevelControl(**connection["level_control"])
            connections.append(CompositionalConnection(**connection))
        data["connections"] = connections
        return cls(**data)

    def initialize(self, initial):
        if set(initial) != set(self.names):
            raise ValueError("Initial conditions must name every vessel")
        state = CompositionalNetworkState(
            tuple((n, v.initialize(**initial[n])) for n, v in zip(self.names, self.vessels)),
            self.maximum_dt_s,
            self.model_id,
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if not isinstance(state, CompositionalNetworkState) or state.model_id != self.model_id:
            raise PropertyModelError("Network state/model mismatch")
        if _digest(self.to_dict()) != self.model_id:
            raise PropertyModelError("Network configuration changed after construction")
        if tuple(n for n, _ in state.vessels) != self.names:
            raise ValueError("Network state topology mismatch")
        if not self.minimum_dt_s <= state.next_dt_s <= self.maximum_dt_s:
            raise ValueError("Invalid network carried timestep")
        if len({s.inventory.time_s for _, s in state.vessels}) != 1:
            raise ValueError("Vessel clocks differ")
        seeds = []
        for v, (_, s) in zip(self.vessels, state.vessels):
            q = v.inspect(s)
            seeds.append(
                {
                    **q,
                    "old_entropy_j_k": self.fluid.tensor(sum(s.inventory.component_moles))
                    * q["molar_entropy_j_mol_k"],
                }
            )
        return seeds

    def _donor(self, c, end, v, data, dp, opening):
        _, _, p, _, _, phase_n, x, q, *_ = data
        layout = v._layout(phase_n, q)
        height = getattr(c, end + "_height_m")
        holdup = (
            phase_n * q["molar_volume_m3_mol"] / v.volume_m3
            if height is None
            else pickup_fractions(layout, height, getattr(c, end + "_diameter_m"))
        )
        selection = getattr(c, end + "_selection")
        mask = torch.ones_like(holdup)
        if selection in ("densest", "lightest"):
            index = (
                q["density_kg_m3"].argmax()
                if selection == "densest"
                else q["density_kg_m3"].argmin()
            )
            mask = torch.nn.functional.one_hot(index, len(phase_n)).to(phase_n.dtype)
        elif selection == "component_rich":
            column = x[:, self.fluid.names.index(getattr(c, end + "_component"))]
            mask = torch.nn.functional.one_hot(column.argmax(), len(phase_n)).to(phase_n.dtype) * (
                column.max() >= c.minimum_mole_fraction
            )
        vd = c.volume_conductance_m3_s_pa * opening * dp * holdup * mask
        nd = vd / q["molar_volume_m3_mol"]
        component = (nd[:, None] * x).sum(0)
        energy = (nd * q["molar_enthalpy_j_mol"]).sum() + (
            component * self.fluid.molar_masses
        ).sum() * v.gravity_m_s2 * v.elevation_m
        entropy = (nd * q["molar_entropy_j_mol_k"]).sum()
        valid = (vd.sum() == 0) | (dp / p <= c.maximum_pressure_drop_fraction)
        if layout is not None:
            valid = valid & layout["density_order_resolved"]
        return torch.cat(
            (component, energy.reshape(1), entropy.reshape(1), vd.sum().reshape(1))
        ), valid

    def _connection_opening(self, c, data, openings):
        if c.name in openings:
            return openings[c.name]
        if c.level_control is None:
            return self.fluid.tensor(c.opening)
        index = self.names.index(c.source if c.control_side == "source" else c.target)
        values = data[index]
        return c.level_control.value(
            self.vessels[index]._layout(values[5], values[7]), values[6], self.fluid.names
        )[0]

    def _transfers(self, data, openings):
        nc = len(self.fluid.names)
        rows, valid = [], self.fluid.tensor(True).to(torch.bool)
        for c, (a, b) in zip(self.connections, self.ends):
            dp = data[a][2] - data[b][2]
            opening = self._connection_opening(c, data, openings)
            forward, vf = self._donor(
                c, "source", self.vessels[a], data[a], dp.clamp_min(0), opening
            )
            reverse, vr = self._donor(
                c,
                "target",
                self.vessels[b],
                data[b],
                (-dp).clamp_min(0),
                opening * int(c.allow_reverse),
            )
            rows.append(forward - reverse)
            valid = valid & vf & vr
        net = []
        for adjacency in self.adjacency:
            net.append(
                torch.stack([rows[e] * sign for e, sign in adjacency]).sum(0)
                if adjacency
                else self.fluid.tensor([0.0] * (nc + 3))
            )
        return net, rows, valid

    def _active(self, states, feeds, openings):
        masks = [
            self.fluid.tensor(s.inventory.component_moles) + feed[0]
            for s, feed in zip(states, feeds)
        ]
        masks = [x > 0 for x in masks]
        # Possible transport over a whole implicit step, including chains and
        # reverse flow. Closed or zero-conductance connections do not spread it.
        for _ in self.names:
            for c, (a, b) in zip(self.connections, self.ends):
                if (
                    c.volume_conductance_m3_s_pa
                    and float(
                        openings.get(
                            c.name,
                            self.fluid.tensor(1.0 if c.level_control is not None else c.opening),
                        ).cpu()
                    )
                    > 0
                ):
                    masks[b] = masks[b] | masks[a]
                    if c.allow_reverse:
                        masks[a] = masks[a] | masks[b]
        return [torch.nonzero(mask).flatten() for mask in masks]

    def _transport_guesses(self, states, dt, commands):
        """Positive frozen-pressure bulk predictor; never an accepted balance.

        Finite Jacobi sweeps seed newly arriving components through chains.
        No convergence is assumed: the full PR Newton must solve afterwards.
        """
        rhs = [
            self.fluid.tensor(s.inventory.component_moles) + dt * c[0][0]
            for s, c in zip(states, commands[0])
        ]
        diagonal = [self.fluid.tensor(1.0) for _ in states]
        incoming = [[] for _ in states]
        for c, (a, b) in zip(self.connections, self.ends):
            dp = states[a].inventory.pressure_pa - states[b].inventory.pressure_pa
            if dp < 0:
                if not c.allow_reverse:
                    continue
                a, b = b, a
            opening = commands[1].get(
                c.name, self.fluid.tensor(1.0 if c.level_control is not None else c.opening)
            )
            rate = dt * c.volume_conductance_m3_s_pa * opening * abs(dp) / self.vessels[a].volume_m3
            diagonal[a] = diagonal[a] + rate
            incoming[b].append((a, rate))
        guess = rhs
        for _ in range(min(16, max(2, len(states)))):
            guess = [
                (r + sum((k * guess[j] for j, k in edges), torch.zeros_like(r))) / d
                for r, d, edges in zip(rhs, diagonal, incoming)
            ]
        return guess

    def _problem(self, states, branches, dt, commands, active):
        guesses = self._transport_guesses(states, dt, commands)
        problems = [
            v._branch_step(
                s,
                seed,
                keep,
                dt,
                *command,
                _capture=True,
                _network_active=indices,
                _network_guess=guess,
            )
            for v, s, (seed, keep), command, indices, guess in zip(
                self.vessels, states, branches, commands[0], active, guesses
            )
        ]
        for p, guess in zip(problems, guesses):
            p["guess"] = guess
        sizes = [p["initial"].numel() for p in problems]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)

        def evaluate(y):
            blocks = [p["evaluate"](y[offsets[i] : offsets[i + 1]]) for i, p in enumerate(problems)]
            net, rows, valid = self._transfers([d for _, d in blocks], commands[1])
            residual = []
            nc = len(self.fluid.names)
            for p, (r, _), incoming in zip(problems, blocks, net):
                c = len(p["active"])
                erow = p["energy_row"]
                residual.append(
                    torch.cat(
                        (
                            r[:c] - dt * incoming[p["active"]] / p["nscale"],
                            r[c:erow],
                            (r[erow] - dt * incoming[nc] / p["energy_scale"]).reshape(1),
                            r[erow + 1 :],
                        )
                    )
                )
            return torch.cat(residual), (blocks, net, rows, valid)

        # Each residual block depends only on itself and graph neighbours.
        neighbours = [{i} for i in range(len(sizes))]
        for a, b in self.ends:
            neighbours[a].add(b)
            neighbours[b].add(a)
        pattern = ([], [])
        for i, adjacent in enumerate(neighbours):
            for j in sorted(adjacent):
                for row in range(offsets[i], offsets[i + 1]):
                    for col in range(offsets[j], offsets[j + 1]):
                        pattern[0].append(row)
                        pattern[1].append(col)
        return problems, offsets, evaluate, pattern

    def _step(self, states, seeds, dt, commands):
        active = self._active(states, [c[0] for c in commands[0]], commands[1])
        queue = [tuple((q, tuple(range(q["phase_count"]))) for q in seeds)]
        seen, errors, predictors = set(), [], set()
        nc = len(self.fluid.names)
        while queue and len(seen) < self.maximum_branch_attempts:
            branches = queue.pop(0)
            signature = tuple(
                (
                    keep,
                    tuple(
                        round(float(v), 11)
                        for p in seed["phases"]
                        for v in p["mole_fractions"].cpu().tolist()
                    ),
                    round(float(seed["pressure_pa"].cpu()), 5),
                    round(float(seed["temperature_k"].cpu()), 9),
                )
                for seed, keep in branches
            )
            if signature in seen:
                continue
            seen.add(signature)
            problems, offsets, evaluate, pattern = self._problem(
                states, branches, dt, commands, active
            )
            solver = TorchNewton(self.fluid.device, maximum_iterations=30)
            solver.host_convergence = True
            with torch.no_grad():
                y, r, ok, evaluations = solver.solve(
                    lambda z, evaluate=evaluate: evaluate(z)[0],
                    torch.cat([p["initial"] for p in problems]),
                    torch.cat([p["lower"] for p in problems]),
                    torch.cat([p["upper"] for p in problems]),
                    pattern=pattern,
                    tolerance=2e-13,
                )
                r, (_blocks, net, transfers, valid) = evaluate(y)
            solved, replacements = [], []
            if bool((ok & valid & torch.isfinite(r).all() & (r.abs().max() < 1e-10)).cpu()):
                for i, (v, s, (seed, keep), command) in enumerate(
                    zip(self.vessels, states, branches, commands[0])
                ):
                    incoming = (net[i][:nc].detach(), net[i][nc].detach(), net[i][nc + 1].detach())
                    try:
                        accepted, post = v._branch_step(
                            s,
                            seed,
                            keep,
                            dt,
                            *command,
                            _solution=(y[offsets[i] : offsets[i + 1]], ok, evaluations),
                            _network_incoming=incoming,
                            _network_active=active[i],
                            _network_guess=problems[i]["guess"],
                        )
                        if accepted is None:
                            replacements.append((i, post))
                        else:
                            solved.append(accepted)
                    except (HydraulicConvergenceError, PropertyModelError) as exc:
                        errors.append(str(exc))
                if len(solved) == len(states):
                    reports = {n: x[2] for n, x in zip(self.names, solved)}
                    links = {
                        c.name: {
                            "component_molar_rates_mol_s": row[:nc],
                            "energy_rate_w": row[nc],
                            "entropy_rate_w_k": row[nc + 1],
                            "donor_volume_rate_m3_s": row[nc + 2],
                        }
                        for c, row in zip(self.connections, [x.cpu().tolist() for x in transfers])
                    }
                    final_data = [d for _, d in _blocks]
                    for c, (a, b) in zip(self.connections, self.ends):
                        links[c.name]["opening"] = float(
                            self._connection_opening(c, final_data, commands[1]).cpu()
                        )
                        for side, index in (("source", a), ("target", b)):
                            height = getattr(c, side + "_height_m")
                            if height is not None:
                                d = final_data[index]
                                links[c.name][side + "_pickup_area_fractions"] = (
                                    self.vessels[index]
                                    ._pickup(d[5], d[7], height, getattr(c, side + "_diameter_m"))
                                    .cpu()
                                    .tolist()
                                )
                        if c.level_control is not None:
                            index = a if c.control_side == "source" else b
                            d = final_data[index]
                            _, level, present = c.level_control.value(
                                self.vessels[index]._layout(d[5], d[7]), d[6], self.fluid.names
                            )
                            links[c.name]["controlled_level_m"] = float(level.cpu())
                            links[c.name]["controlled_phase_present"] = bool(present.cpu())
                    return (
                        [x[0] for x in solved],
                        [x[1] for x in solved],
                        {
                            "dt_s": dt,
                            "vessels": reports,
                            "connections": links,
                            "phase_branch_attempts": len(seen),
                            "nonlinear_evaluations": evaluations,
                            "max_scaled_residual": float(r.abs().max().cpu()),
                        },
                    )
            else:
                errors.append("Coupled Newton or connection operating domain rejected")
            for i, post in replacements:
                changed = list(branches)
                changed[i] = (
                    {**post, "old_entropy_j_k": seeds[i]["old_entropy_j_k"]},
                    tuple(range(post["phase_count"])),
                )
                queue.append(tuple(changed))
            for i, (seed, keep) in enumerate(branches):
                if len(keep) > 1:
                    for sub in itertools.combinations(keep, len(keep) - 1):
                        changed = list(branches)
                        changed[i] = (seed, sub)
                        queue.append(tuple(changed))
                # Pure-fluid phase birth requires conserved energy/volume,
                # not an ambiguous saturation PT flash. Predictor only.
                if i not in predictors and len(active[i]) == 1:
                    predictors.add(i)
                    try:
                        p = problems[i]
                        _, olddata = p["evaluate"](p["initial"])
                        all_initial = [q["evaluate"](q["initial"])[1] for q in problems]
                        inflow = self._transfers(all_initial, commands[1])[0][i]
                        feed, heat, _, _ = commands[0][i]
                        n0 = self.fluid.tensor(states[i].inventory.component_moles)
                        npred = n0 + dt * (feed[0] + inflow[:nc] - olddata[11])
                        v = self.vessels[i]
                        exchange = (
                            v.wall.fluid_ua_w_k
                            * (states[i].wall_temperature_k - states[i].inventory.temperature_k)
                            if v.wall
                            else 0
                        )
                        potential = (
                            ((npred - n0) * self.fluid.molar_masses).sum()
                            * v.gravity_m_s2
                            * v.elevation_m
                        )
                        u = (
                            states[i].inventory.internal_energy_j
                            + dt * (feed[1] + inflow[nc] - olddata[12] + heat + exchange)
                            - potential
                        )
                        post = self.fluid.flash_uv(
                            npred,
                            float(u.cpu()),
                            v.volume_m3,
                            pressure_guess_pa=states[i].inventory.pressure_pa,
                            temperature_guess_k=states[i].inventory.temperature_k,
                        )
                        changed = list(branches)
                        changed[i] = (
                            {**post, "old_entropy_j_k": seeds[i]["old_entropy_j_k"]},
                            tuple(range(post["phase_count"])),
                        )
                        queue.append(tuple(changed))
                    except (PropertyModelError, HydraulicConvergenceError) as exc:
                        errors.append(str(exc))
        raise HydraulicConvergenceError(
            "Coupled PR phase solve failed: " + (errors[-1] if errors else "branch budget")
        )

    def prepare(
        self,
        state,
        *,
        feeds=None,
        heat_w=None,
        utility_temperature_k=None,
        outlet_openings=None,
        connection_openings=None,
        _check_feed_pressure=True,
    ):
        self.inspect(state)
        mappings = [dict(x or {}) for x in (feeds, heat_w, utility_temperature_k, outlet_openings)]
        if any(set(x) - set(self.names) for x in mappings):
            raise ValueError("Unknown command vessel")
        feeds, heat_w, utility_temperature_k, outlet_openings = mappings
        commands = []
        for n, v, (_, s) in zip(self.names, self.vessels, state.vessels):
            fn = self.fluid.tensor([0.0] * len(self.fluid.names))
            fe, fs, minp = self.fluid.tensor(0.0), self.fluid.tensor(0.0), math.inf
            for stream in feeds.get(n, ()):
                if not isinstance(stream, ComponentEnergyStream):
                    raise TypeError("Network feed requires ComponentEnergyStream")
                stream.validate_fluid(self.fluid)
                if stream.gravity_m_s2 != v.gravity_m_s2:
                    raise PropertyModelError("Feed gravity datum differs")
                if stream.molar_rate_mol_s:
                    if _check_feed_pressure and stream.pressure_pa < s.inventory.pressure_pa:
                        raise EquipmentOperatingError("Feed pressure below vessel pressure")
                    q = stream.equilibrate(
                        self.fluid, temperature_guess_k=s.inventory.temperature_k
                    )
                    fs = fs + stream.molar_rate_mol_s * q["molar_entropy_j_mol_k"]
                    minp = min(minp, stream.pressure_pa)
                fn = fn + self.fluid.tensor(stream.component_molar_rates_mol_s)
                fe = fe + stream.total_energy_rate_w
            heat = _number(heat_w.get(n, 0.0), "heat_w")
            if heat and n not in utility_temperature_k:
                raise ValueError("Nonzero heat requires utility temperature")
            utility = _number(
                utility_temperature_k.get(n, s.inventory.temperature_k),
                "utility temperature",
                positive=True,
            )
            opening = dict(outlet_openings.get(n, {}))
            if set(opening) - {p.name for p in v.outlets}:
                raise ValueError("Unknown external outlet")
            for value in opening.values():
                if not 0 <= _number(value, "opening") <= 1:
                    raise ValueError("Opening outside [0,1]")
            commands.append(
                (
                    (fn, fe, fs, self.fluid.tensor(minp)),
                    heat,
                    self.fluid.tensor(utility),
                    {k: self.fluid.tensor(x) for k, x in opening.items()},
                )
            )
        opening = dict(connection_openings or {})
        if set(opening) - {c.name for c in self.connections}:
            raise ValueError("Unknown connection opening")
        for value in opening.values():
            if not 0 <= _number(value, "opening") <= 1:
                raise ValueError("Connection opening outside [0,1]")
        return commands, {k: self.fluid.tensor(x) for k, x in opening.items()}

    def _error(self, coarse, fine):
        error = 0.0
        for a, b in zip(coarse, fine):
            na, nb = (
                self.fluid.tensor(a.inventory.component_moles),
                self.fluid.tensor(b.inventory.component_moles),
            )
            scale = torch.maximum(na.abs(), nb.abs()).clamp_min(nb.sum() * 1e-8)
            error = max(
                error,
                float(((na - nb).abs() / (1e-10 + self.relative_tolerance * scale)).max().cpu()),
            )
            error = max(
                error,
                abs(a.inventory.pressure_pa - b.inventory.pressure_pa)
                / (
                    1
                    + self.relative_tolerance
                    * max(abs(a.inventory.pressure_pa), abs(b.inventory.pressure_pa))
                ),
            )
            for ta, tb in (
                (a.inventory.temperature_k, b.inventory.temperature_k),
                (a.wall_temperature_k, b.wall_temperature_k),
            ):
                if ta is not None:
                    error = max(
                        error,
                        abs(ta - tb)
                        / (
                            self.temperature_tolerance_k
                            + self.relative_tolerance * max(abs(ta), abs(tb))
                        ),
                    )
        return error

    def _totals(self, states):
        n = self.fluid.tensor([0.0] * len(self.fluid.names))
        energy = 0.0
        for v, s in zip(self.vessels, states):
            n = n + self.fluid.tensor(s.inventory.component_moles)
            energy += s.inventory.total_energy_j(self.fluid)
            if v.wall:
                energy += v.wall.energy(s.wall_temperature_k)
        return n, energy

    def advance(
        self,
        state,
        duration_s,
        *,
        feeds=None,
        heat_w=None,
        utility_temperature_k=None,
        outlet_openings=None,
        connection_openings=None,
        max_substeps=10000,
    ):
        seeds = self.inspect(state)
        duration = _number(duration_s, "duration_s", positive=True)
        t0 = state.vessels[0][1].inventory.time_s
        if not math.isfinite(t0 + duration) or t0 + duration <= t0:
            raise ValueError("Network clock cannot advance")
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("Positive work budget required")
        commands = self.prepare(
            state,
            feeds=feeds,
            heat_w=heat_w,
            utility_temperature_k=utility_temperature_k,
            outlet_openings=outlet_openings,
            connection_openings=connection_openings,
        )
        states = [s for _, s in state.vessels]
        initial_n, initial_e = self._totals(states)
        active = self._active(states, [c[0] for c in commands[0]], commands[1])
        branches = tuple((q, tuple(range(q["phase_count"]))) for q in seeds)
        problems, _, _, _ = self._problem(states, branches, self.minimum_dt_s, commands, active)
        initial_data = [p["evaluate"](p["initial"])[1] for p in problems]
        if not bool(self._transfers(initial_data, commands[1])[-1].cpu()) or any(
            not bool(d[15].cpu()) for d in initial_data
        ):
            raise EquipmentOperatingError("Initial connection/outlet pressure-drop domain exceeded")
        steps, reasons = [], {}
        elapsed, hnext = 0.0, state.next_dt_s
        while elapsed < duration:
            if len(steps) + sum(reasons.values()) >= max_substeps:
                raise HydraulicConvergenceError("Network work budget exhausted")
            h = min(hnext, duration - elapsed)
            if elapsed + h <= elapsed:
                raise HydraulicConvergenceError("Network substep cannot advance")
            try:
                coarse = self._step(states, seeds, h, commands)
                accepted, error = [coarse], 0.0
                if self.temporal_error_control:
                    a = self._step(states, seeds, h / 2, commands)
                    b = self._step(a[0], a[1], h / 2, commands)
                    error = self._error(coarse[0], b[0])
                    if not math.isfinite(error) or error > 1:
                        raise HydraulicConvergenceError("Network temporal error control")
                    accepted = [a, b]
            except HydraulicConvergenceError as exc:
                reasons[str(exc)] = reasons.get(str(exc), 0) + 1
                if h / 2 < self.minimum_dt_s:
                    raise
                hnext = h / 2
                continue
            for states, seeds, record in accepted:
                if len(steps) + sum(reasons.values()) >= max_substeps:
                    raise HydraulicConvergenceError("Network work budget exhausted")
                steps.append({**record, "time_s": t0 + elapsed, "temporal_error_ratio": error})
                elapsed = math.fsum((elapsed, record["dt_s"]))
            if abs(duration - elapsed) <= 16 * math.ulp(max(1.0, duration)):
                elapsed = duration
            hnext = min(
                self.maximum_dt_s,
                max(
                    self.minimum_dt_s,
                    h * min(1.5, max(0.5, 0.9 * error**-0.5)) if error else h * 1.5,
                ),
            )
        # Set one exact common report clock, retaining per-vessel identities.
        from dataclasses import replace

        states = [replace(s, inventory=replace(s.inventory, time_s=t0 + duration)) for s in states]
        final = CompositionalNetworkState(tuple(zip(self.names, states)), hnext, self.model_id)
        final_n, final_e = self._totals(states)
        external_n = self.fluid.tensor([0.0] * len(self.fluid.names))
        external_e = 0.0
        for step in steps:
            dt = step["dt_s"]
            for n, command in zip(self.names, commands[0]):
                body = step["vessels"][n]
                external_n = external_n + dt * (
                    command[0][0] - self.fluid.tensor(body["outlet_component_rates_mol_s"])
                )
                external_e += dt * (
                    float(command[0][1].cpu())
                    + command[1]
                    + body["ambient_heat_w"]
                    - body["outlet_energy_rate_w"]
                )
        rn = final_n - initial_n - external_n
        re = final_e - initial_e - external_e
        if float(rn.abs().max().cpu()) > 1e-8 + 1e-9 * float(external_n.abs().sum().cpu()) or abs(
            re
        ) > 1e-4 + 1e-9 * max(1.0, abs(external_e), abs(final_e - initial_e)):
            raise ConservationError("Coupled network global component/energy balance rejected")
        return {
            "state": final.to_dict(),
            "steps": steps,
            "failed_steps": sum(reasons.values()),
            "retry_reasons": reasons,
            "component_residuals_mol": rn.cpu().tolist(),
            "energy_residual_j": re,
            "external_component_transfer_mol": external_n.cpu().tolist(),
            "external_energy_transfer_j": external_e,
        }
