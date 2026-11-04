"""Conservative monolithic gravitational PR vessels with finite direct connectors.

Connectors have no stored mass, inertia or choking. Both apertures coincide
in absolute elevation and size; no unmodelled lift or pipe inventory is added.
Each vessel currently requires the same positive component support.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .darcy_hydraulics import darcy_pressure_loss
from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    PropertyModelError,
    _name,
    _number,
)
from .operation import _digest
from .peng_robinson import R
from .phase_geometry import validate_pickup
from .stratified_pr_flow import StratifiedPRFlow
from .stratified_pr_thermal import StratifiedThermalState
from .three_phase_properties import disk_area_moment
from .torch_linear import TorchNewton


@dataclass(frozen=True)
class StratifiedConnection:
    name: str
    source: str
    target: str
    source_height_m: float
    target_height_m: float
    diameter_m: float
    volume_conductance_m3_s_pa: float
    opening: float = 1.0
    allow_reverse: bool = True
    maximum_pressure_drop_fraction: float = 0.05
    hydraulic_model: str = "linear"
    length_m: float | None = None
    dynamic_viscosity_pa_s: float | None = None
    roughness_m: float = 0.0
    maximum_velocity_m_s: float | None = None

    def to_dict(self):
        data = asdict(self)
        if self.hydraulic_model == "linear":
            for key in (
                "hydraulic_model",
                "length_m",
                "dynamic_viscosity_pa_s",
                "roughness_m",
                "maximum_velocity_m_s",
            ):
                data.pop(key)
        return data

    def __post_init__(self):
        for key in ("name", "source", "target"):
            _name(getattr(self, key))
        if self.source == self.target:
            raise ValueError("Connection cannot join a vessel to itself")
        for key in ("source_height_m", "target_height_m", "volume_conductance_m3_s_pa"):
            _number(getattr(self, key), key, nonnegative=True)
        _number(self.diameter_m, "diameter", positive=True)
        if not 0 <= _number(self.opening, "opening") <= 1:
            raise ValueError("Opening outside [0,1]")
        if not isinstance(self.allow_reverse, bool):
            raise TypeError("allow_reverse must be boolean")
        if not 0 < _number(self.maximum_pressure_drop_fraction, "pressure drop fraction") <= 1:
            raise ValueError("Invalid connection pressure-drop domain")
        if self.hydraulic_model not in ("linear", "darcy_weisbach"):
            raise ValueError("Unknown hydraulic model")
        if self.hydraulic_model == "linear":
            if (
                self.length_m is not None
                or self.dynamic_viscosity_pa_s is not None
                or self.roughness_m != 0
                or self.maximum_velocity_m_s is not None
            ):
                raise ValueError("Linear connector cannot silently ignore Darcy parameters")
        else:
            for key in ("length_m", "dynamic_viscosity_pa_s", "maximum_velocity_m_s"):
                _number(getattr(self, key), key, positive=True)
            if (
                not 0
                <= _number(self.roughness_m, "roughness", nonnegative=True)
                < self.diameter_m / 2
            ):
                raise ValueError("Roughness outside pipe geometry")
            if self.volume_conductance_m3_s_pa != 0 or self.opening not in (0.0, 1.0):
                raise ValueError(
                    "Darcy connector requires zero empirical conductance and binary isolation"
                )


@dataclass(frozen=True)
class StratifiedNetworkState:
    vessels: tuple
    time_s: float
    next_step_s: float
    model_id: str

    def to_dict(self):
        return {
            "vessels": {name: s.to_dict() for name, s in self.vessels},
            "time_s": self.time_s,
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"vessels", "time_s", "next_step_s", "model_id"}:
            raise ValueError("Unknown network state fields")
        return cls(
            tuple(
                (n, StratifiedThermalState.from_dict(s)) for n, s in sorted(data["vessels"].items())
            ),
            data["time_s"],
            data["next_step_s"],
            data["model_id"],
        )


class StratifiedPRNetwork:
    def __init__(
        self,
        vessels,
        connections=(),
        *,
        initial_step_s=0.01,
        minimum_step_s=1e-6,
        maximum_step_s=1.0,
        relative_tolerance=1e-5,
        temperature_tolerance_k=0.002,
        maximum_attempts=1000,
    ):
        if not isinstance(vessels, dict) or not vessels:
            raise TypeError("Nonempty named vessels required")
        self.vessels = dict(sorted(vessels.items()))
        for name, vessel in self.vessels.items():
            _name(name)
            if not isinstance(vessel, StratifiedPRFlow):
                raise TypeError("Expected StratifiedPRFlow vessels")
        first = next(iter(self.vessels.values())).column
        self.fluid = first.fluid
        for vessel in self.vessels.values():
            c = vessel.column
            if (
                c.fluid.model_id != self.fluid.model_id
                or c.fluid.device != self.fluid.device
                or c.gravity_m_s2 != first.gravity_m_s2
            ):
                raise ValueError("Network requires common EOS, device, references and gravity")
        self.connections = tuple(connections)
        if any(not isinstance(p, StratifiedConnection) for p in self.connections):
            raise TypeError("Expected StratifiedConnection")
        if len({p.name for p in self.connections}) != len(self.connections):
            raise ValueError("Duplicate connection name")
        for p in self.connections:
            if not isinstance(p, StratifiedConnection):
                raise TypeError("Expected StratifiedConnection")
            if p.source not in self.vessels or p.target not in self.vessels:
                raise ValueError("Connection references an unknown vessel")
            a, b = self.vessels[p.source].column, self.vessels[p.target].column
            validate_pickup(a.geometry, p.source_height_m, p.diameter_m)
            validate_pickup(b.geometry, p.target_height_m, p.diameter_m)
            if abs(a.elevation_m + p.source_height_m - b.elevation_m - p.target_height_m) > 1e-9:
                raise ValueError(
                    "Direct connector apertures must coincide in absolute height; explicit pipe required"
                )
        for key, value in locals().copy().items():
            if key in (
                "initial_step_s",
                "minimum_step_s",
                "maximum_step_s",
                "relative_tolerance",
                "temperature_tolerance_k",
            ):
                setattr(self, key, _number(value, key, positive=True))
        if not minimum_step_s <= initial_step_s <= maximum_step_s:
            raise ValueError("Invalid network step interval")
        if (
            isinstance(maximum_attempts, bool)
            or not isinstance(maximum_attempts, int)
            or maximum_attempts < 1
        ):
            raise ValueError("Positive integer attempt budget required")
        self.maximum_attempts = maximum_attempts
        self.start_order = max(v.column.quadrature_order for v in self.vessels.values())
        self.max_order = min(v.column.maximum_quadrature_order for v in self.vessels.values())
        if self.start_order >= self.max_order:
            raise ValueError("Network requires a shared quadrature-refinement interval")
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "stratified_pr_network_v1",
            "vessels": {n: v.to_dict() for n, v in self.vessels.items()},
            "connections": [p.to_dict() for p in self.connections],
            **{
                k: getattr(self, k)
                for k in (
                    "initial_step_s",
                    "minimum_step_s",
                    "maximum_step_s",
                    "relative_tolerance",
                    "temperature_tolerance_k",
                    "maximum_attempts",
                )
            },
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "stratified_pr_network_v1":
            raise ValueError("Unknown network model")
        data["vessels"] = {
            n: StratifiedPRFlow.from_dict(v, device=device) for n, v in data["vessels"].items()
        }
        data["connections"] = [StratifiedConnection(**p) for p in data["connections"]]
        return cls(**data)

    def initialize(self, states):
        if set(states) != set(self.vessels):
            raise ValueError("Initial vessel names differ")
        t = next(iter(states.values())).time_s
        state = StratifiedNetworkState(
            tuple(sorted(states.items())), t, self.initial_step_s, self.model_id
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, StratifiedNetworkState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Network state/model mismatch")
        _number(state.time_s, "time", nonnegative=True)
        _number(state.next_step_s, "next step", positive=True)
        if not self.minimum_step_s <= state.next_step_s <= self.maximum_step_s:
            raise ValueError("Network restart step outside domain")
        if not isinstance(state.vessels, tuple) or tuple(n for n, _ in state.vessels) != tuple(
            self.vessels
        ):
            raise ValueError("Network vessel order/names differ")
        support = None
        for name, s in state.vessels:
            self.vessels[name].inspect(s)
            if s.time_s != state.time_s:
                raise ValueError("Network vessel times differ")
            active = tuple(i for i, n in enumerate(s.column.component_moles) if n > 0)
            if support is not None and active != support:
                raise PropertyModelError(
                    "Connected vessels currently require identical positive component support"
                )
            support = active
            if any(
                v.stream.component_molar_rates_mol_s[i] > 0
                for v in self.vessels[name].feeds
                for i in range(len(self.fluid.names))
                if i not in active
            ):
                raise PropertyModelError(
                    "Network feed activates an absent component; unsupported active-set change"
                )

    def _commands(self, openings):
        openings = {} if openings is None else dict(openings)
        if set(openings) - {p.name for p in self.connections}:
            raise ValueError("Unknown connection opening")
        for value in openings.values():
            if not 0 <= _number(value, "opening") <= 1:
                raise ValueError("Opening outside [0,1]")
        commands = {p.name: openings.get(p.name, p.opening) for p in self.connections}
        if any(
            p.hydraulic_model == "darcy_weisbach" and commands[p.name] not in (0.0, 1.0)
            for p in self.connections
        ):
            raise ValueError("Darcy faces support binary isolation; add a separate control valve")
        return commands

    def _seeds(self, states):
        f = self.fluid
        return {
            name: (
                [f.tensor(x) for x in s.column.phase_seeds],
                list(s.column.phase_roots),
                f.tensor(s.column.interface_heights_m),
                math.exp(s.column.solution_variables[0]),
                s.column.temperature_k,
                f.tensor(s.column.chemical_constants),
            )
            for name, s in states.items()
        }

    def _patch(self, ba, bb, ia, ib, connection, order):
        a, b = self.vessels[connection.source].column, self.vessels[connection.target].column
        center = a.elevation_m + connection.source_height_m
        radius = connection.diameter_m / 2
        lo = torch.maximum(
            torch.maximum(ba[ia] + a.elevation_m, bb[ib] + b.elevation_m),
            ba.new_tensor(center - radius),
        )
        hi = torch.minimum(
            torch.minimum(ba[ia + 1] + a.elevation_m, bb[ib + 1] + b.elevation_m),
            ba.new_tensor(center + radius),
        )
        hi = torch.maximum(hi, lo)

        def angle(h):
            z = (h - center) / radius
            return torch.where(
                z <= -1,
                z.new_tensor(-math.pi / 2),
                torch.where(
                    z >= 1, z.new_tensor(math.pi / 2), torch.asin(z.clamp(-1 + 1e-12, 1 - 1e-12))
                ),
            )

        low, high = angle(lo), angle(hi)
        nodes, weights = a._rules[order]
        theta = (low + high) / 2 + (high - low) * nodes / 2
        fraction = (
            disk_area_moment(hi - center + radius, connection.diameter_m)[0]
            - disk_area_moment(lo - center + radius, connection.diameter_m)[0]
        ) / (math.pi * radius**2)
        raw = weights * torch.cos(theta) ** 2
        area = fraction * raw / raw.sum().clamp_min(1e-30)
        z = center + radius * torch.sin(theta)
        ha = torch.where(fraction > 0, z - a.elevation_m, (ba[ia] + ba[ia + 1]) / 2)
        hb = torch.where(fraction > 0, z - b.elevation_m, (bb[ib] + bb[ib + 1]) / 2)
        return ha, hb, area, z

    def _problem(self, states, seeds, dt, order, openings):
        f = self.fluid
        problems = {}
        offsets = {}
        initials = []
        lowers = []
        uppers = []
        offset = 0
        for name, v in self.vessels.items():
            problem = v._problem(states[name], dt, seeds[name], order)
            problems[name] = problem
            y, lo, hi = problem[:3]
            offsets[name] = (offset, offset + len(y))
            offset += len(y)
            initials.append(y)
            lowers.append(lo)
            uppers.append(hi)
        segments = []
        for connection in self.connections:
            a, b = connection.source, connection.target
            for ia, roota in enumerate(seeds[a][1]):
                for ib, rootb in enumerate(seeds[b][1]):
                    ends = []
                    for name, phase, tag in ((a, ia, roota), (b, ib, rootb)):
                        active = problems[name][4]
                        c = len(active)
                        z = seeds[name][0][phase][active].clamp_min(1e-30)
                        row = torch.cat(
                            (f.tensor([math.log(seeds[name][3])]), torch.log(z[:-1] / z[-1]))
                        )
                        y = row.repeat(order)
                        lo = torch.cat(
                            (f.tensor([math.log(f.min_pressure_pa)]), f.tensor([-100.0] * (c - 1)))
                        ).repeat(order)
                        hi = torch.cat(
                            (f.tensor([math.log(f.max_pressure_pa)]), f.tensor([100.0] * (c - 1)))
                        ).repeat(order)
                        ends.append((offset, offset + len(y), name, tag))
                        offset += len(y)
                        initials.append(y)
                        lowers.append(lo)
                        uppers.append(hi)
                    segments.append((connection, ia, ib, ends))
        velocity_offsets = {}
        pressure_scales = {}
        for connection in self.connections:
            if connection.hydraulic_model == "darcy_weisbach":
                velocity_offsets[connection.name] = offset
                offset += 1
                initials.append(f.tensor([0.0]))
                lowers.append(
                    f.tensor(
                        [-connection.maximum_velocity_m_s if connection.allow_reverse else 0.0]
                    )
                )
                uppers.append(f.tensor([connection.maximum_velocity_m_s]))
                pressure_scales[connection.name] = max(
                    1.0,
                    abs(seeds[connection.source][3] - seeds[connection.target][3]),
                    1e-3 * max(seeds[connection.source][3], seeds[connection.target][3]),
                )
        if offset > 768:
            raise ValueError("Network exceeds declared 768-unknown dense budget")
        initial, lower, upper = map(torch.cat, (initials, lowers, uppers))

        def evaluate(y):
            residuals = {}
            data = {}
            bounds = {}
            dn = {}
            de = {}
            ds = {}
            chemical = []
            for name, problem in problems.items():
                left, right = offsets[name]
                residuals[name], data[name] = problem[3](y[left:right])
                d = data[name]["data"]
                c = self.vessels[name].column
                bounds[name] = torch.cat(
                    (
                        d["temperature"].new_zeros(1),
                        d["interfaces"],
                        d["temperature"].new_tensor([c.geometry.height_m]),
                    )
                )
                dn[name] = torch.zeros_like(d["component_moles"])
                de[name] = d["temperature"].new_zeros(())
                ds[name] = d["temperature"].new_zeros(())
            transfers = []
            hydraulic = {
                p.name: {
                    "pressure_source": y.new_zeros(()),
                    "pressure_target": y.new_zeros(()),
                    "density": y.new_zeros(()),
                }
                for p in self.connections
                if p.hydraulic_model == "darcy_weisbach"
            }
            for connection, ia, ib, ends in segments:
                a, b = connection.source, connection.target
                ha, hb, area, z = self._patch(bounds[a], bounds[b], ia, ib, connection, order)
                points = []
                for (left, right, name, tag), height in zip(ends, (ha, hb)):
                    active = problems[name][4]
                    d = data[name]["data"]
                    c = self.vessels[name].column
                    t = d["temperature"]
                    p, x, q = c._decode_nodes(
                        y[left:right].reshape(order, len(active)), active, t, tag
                    )
                    mu = (
                        torch.log(x[:, active].clamp_min(1e-300))
                        + q["log_fugacity_coefficients"][:, active]
                        + torch.log(p / f.pref)[:, None]
                        + c.gravity_m_s2 * height[:, None] * f.molar_masses[active] / (R * t)
                        - d["constants"]
                    )
                    chemical.append(mu.flatten())
                    points.append((p, x, q, height))
                pa, xa, qa, _ = points[0]
                pb, xb, qb, _ = points[1]
                dp = pa - pb
                if connection.hydraulic_model == "darcy_weisbach":
                    velocity = y[velocity_offsets[connection.name]]
                    signed = (
                        math.pi
                        * connection.diameter_m**2
                        / 4
                        * velocity
                        * area
                        * openings[connection.name]
                    )
                    forward = (velocity >= 0).expand_as(dp)
                    hydraulics = hydraulic[connection.name]
                    hydraulics["pressure_source"] = (
                        hydraulics["pressure_source"] + (pa * area).sum()
                    )
                    hydraulics["pressure_target"] = (
                        hydraulics["pressure_target"] + (pb * area).sum()
                    )
                    hydraulics["density"] = (
                        hydraulics["density"]
                        + (
                            torch.where(forward, qa["density_kg_m3"], qb["density_kg_m3"]) * area
                        ).sum()
                    )
                else:
                    signed = (
                        connection.volume_conductance_m3_s_pa
                        * openings[connection.name]
                        * area
                        * (dp if connection.allow_reverse else dp.clamp_min(0))
                    )
                    forward = dp >= 0
                x = torch.where(forward[:, None], xa, xb)
                vm = torch.where(forward, qa["molar_volume_m3_mol"], qb["molar_volume_m3_mol"])
                h = torch.where(forward, qa["molar_enthalpy_j_mol"], qb["molar_enthalpy_j_mol"])
                s = torch.where(forward, qa["molar_entropy_j_mol_k"], qb["molar_entropy_j_mol_k"])
                rate = signed / vm
                flux = (rate[:, None] * x).sum(0)
                energy = (
                    rate
                    * (h + (x * f.molar_masses).sum(-1) * self.vessels[a].column.gravity_m_s2 * z)
                ).sum()
                entropy = (rate * s).sum()
                dn[a] = dn[a] - flux
                dn[b] = dn[b] + flux
                de[a] = de[a] - energy
                de[b] = de[b] + energy
                ds[a] = ds[a] - entropy
                ds[b] = ds[b] + entropy
                donor_p = torch.where(forward, pa, pb)
                valid = (
                    qa["valid"].all()
                    & qb["valid"].all()
                    & (
                        (dp.abs() / donor_p <= connection.maximum_pressure_drop_fraction + 1e-12)
                        | (signed == 0)
                    ).all()
                )
                transfers.append(
                    {
                        "name": connection.name,
                        "source_phase": ia,
                        "target_phase": ib,
                        "component_rates": flux,
                        "total_energy_rate": energy,
                        "entropy_rate": entropy,
                        "volume_rate": signed.sum(),
                        "area": area.sum(),
                        "valid": valid,
                        "points": points,
                    }
                )
            momentum = []
            for connection in self.connections:
                if connection.hydraulic_model == "darcy_weisbach":
                    h = hydraulic[connection.name]
                    velocity = y[velocity_offsets[connection.name]]
                    pressure = h["pressure_source"] - h["pressure_target"]
                    loss, re = darcy_pressure_loss(
                        velocity,
                        h["density"],
                        connection.length_m,
                        connection.diameter_m,
                        connection.dynamic_viscosity_pa_s,
                        connection.roughness_m,
                    )
                    drive = pressure if connection.allow_reverse else pressure.clamp_min(0)
                    residual = (
                        (drive - loss) / pressure_scales[connection.name]
                        if openings[connection.name]
                        else velocity / connection.maximum_velocity_m_s
                    )
                    momentum.append(residual.reshape(1))
                    h.update(
                        velocity=velocity, pressure_drop=pressure, friction_loss=loss, reynolds=re
                    )
            rows = []
            for name, problem in problems.items():
                active = problem[4]
                c = len(active)
                m = len(seeds[name][1])
                body = (m * order + 2 * (m - 1)) * c + c + m
                start = body - c - 1
                n = f.tensor(states[name].column.component_moles)
                nref = n + dt * problem[6]
                nscale = nref[active].clamp_min(nref.sum() * 1e-12)
                col = self.vessels[name].column
                erel = (
                    states[name].column.total_energy_j
                    - float((n * f.molar_masses).sum().cpu()) * col.gravity_m_s2 * col.elevation_m
                )
                escale = max(1.0, abs(erel), float(nref.sum().cpu()) * R * seeds[name][4])
                relative_in = (
                    de[name]
                    - (dn[name] * f.molar_masses).sum() * col.gravity_m_s2 * col.elevation_m
                )
                r = residuals[name]
                rows.append(
                    torch.cat(
                        (
                            r[:start],
                            r[start : start + c] - dt * dn[name][active] / nscale,
                            (r[start + c] - dt * relative_in / escale).reshape(1),
                            r[start + c + 1 :],
                        )
                    )
                )
            return torch.cat((*rows, *chemical, *momentum)), {
                "vessels": data,
                "component_net": dn,
                "energy_net": de,
                "entropy_net": ds,
                "transfers": transfers,
                "hydraulic": hydraulic,
            }

        return initial, lower, upper, evaluate, problems

    def _balance_seeds(self, states, seeds, dt, data, problems):
        """Retry topology using trial balance targets, never publish trial stocks."""
        f = self.fluid
        updated = {}
        for name, v in self.vessels.items():
            old = states[name].column
            d = data["vessels"][name]
            col = v.column
            if not all(bool(q["valid"].all().cpu()) for q in d["data"]["properties"]):
                raise HydraulicConvergenceError("Invalid trial EOS cannot seed phase topology")
            n = f.tensor(old.component_moles) + dt * (
                problems[name][6] - d["out_n"] + data["component_net"][name]
            )
            if not bool((n[problems[name][4]] > 0).all().cpu()):
                raise HydraulicConvergenceError("Phase predictor exhausts an active component")
            massout = float((d["out_n"] * f.molar_masses).sum().cpu())
            out_e = float(d["out_e"].cpu()) + massout * col.gravity_m_s2 * col.elevation_m
            energy = old.total_energy_j + dt * (
                problems[name][7]
                - out_e
                + float(d["heat"].cpu())
                + float(data["energy_net"][name].cpu())
            )
            t = float(d["data"]["temperature"].cpu())
            p = float(d["data"]["pressures"][0][0].cpu())
            updated[name] = col._seed(n, energy, p, t)
        return updated

    def _step(self, state, dt, openings):
        states = dict(state.vessels)
        seeds = self._seeds(states)
        f = self.fluid
        # Explicit transport is only a branch predictor; accepted stocks come
        # exclusively from the final simultaneous balance.
        guess, _, _, predictor, predictor_problems = self._problem(
            states, seeds, dt, self.start_order, openings
        )
        with torch.no_grad():
            predicted = predictor(guess)[1]
        for name, v in self.vessels.items():
            old = states[name].column
            d = predicted["vessels"][name]
            col = v.column
            n = f.tensor(old.component_moles) + dt * (
                predictor_problems[name][6] - d["out_n"] + predicted["component_net"][name]
            )
            active = predictor_problems[name][4]
            if not bool((n[active] > 0).all().cpu()):
                raise HydraulicConvergenceError("Network predictor exhausts active component")
            massout = float((d["out_n"] * f.molar_masses).sum().cpu())
            out_e = float(d["out_e"].cpu()) + massout * col.gravity_m_s2 * col.elevation_m
            cv = max(
                1e-12,
                sum(
                    nn * max(1.0, c.ideal_cp_j_mol_k - R)
                    for nn, c in zip(old.component_moles, f.components)
                ),
            )
            energy = (
                old.total_energy_j
                + dt
                * (predictor_problems[name][7] - out_e + float(predicted["energy_net"][name].cpu()))
                + dt * float(d["heat"].cpu()) / (1 + dt * v.wall.fluid_ua_w_k / cv)
            )
            seeds[name] = col._seed(n, energy, seeds[name][3], old.temperature_k)
        previous = None
        order = self.start_order
        topology_attempts = 0
        solver_attempts = []
        while order <= self.max_order:
            initial, lower, upper, evaluate, problems = self._problem(
                states, seeds, dt, order, openings
            )
            solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=768)
            solver.host_convergence = True
            with torch.no_grad():
                y, r, ok, evaluations = solver.solve(
                    lambda z, evaluate=evaluate: evaluate(z)[0],
                    initial,
                    lower,
                    upper,
                    tolerance=3e-12,
                )
                r, data = evaluate(y)
            solver_attempts.append(
                {
                    "quadrature_order": order,
                    "phase_counts": {n: len(seed[1]) for n, seed in seeds.items()},
                    "nonlinear_evaluations": evaluations,
                    "max_scaled_residual": float(r.abs().max().cpu()),
                }
            )
            if not bool((ok & torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()):
                topology_attempts += 1
                if topology_attempts <= 4 and bool(torch.isfinite(r).all().cpu()):
                    updated = self._balance_seeds(states, seeds, dt, data, problems)
                    if any(
                        len(updated[n][1]) != len(seeds[n][1]) or updated[n][1] != seeds[n][1]
                        for n in seeds
                    ):
                        seeds = updated
                        previous = None
                        continue
                raise HydraulicConvergenceError(
                    f"Coupled stratified network Newton rejected: residual={float(r.abs().max().cpu()):.3g}"
                )
            for transfer in data["transfers"]:
                if not bool(transfer["valid"].cpu()):
                    raise EquipmentOperatingError("Connector outside EOS/pressure-drop domain")
            candidates = {}
            restart_topology = False
            reports = {}
            next_seeds = {}
            feed_e = out_e = ambient_e = entropy_boundary = 0.0
            component_boundary = torch.zeros(len(f.names), dtype=torch.float64, device=f.device)
            for name, v in self.vessels.items():
                col, w = v.column, v.wall
                old = states[name]
                d = data["vessels"][name]
                q = d["data"]
                active = problems[name][4]
                n = f.tensor(old.column.component_moles) + dt * (
                    problems[name][6] - d["out_n"] + data["component_net"][name]
                )
                if not bool((n[active] > 0).all().cpu()):
                    raise HydraulicConvergenceError("Network exhausts active component")
                massout = float((d["out_n"] * f.molar_masses).sum().cpu())
                energyout = float(d["out_e"].cpu()) + massout * col.gravity_m_s2 * col.elevation_m
                heat, ambient, tw = (float(z.cpu()) for z in (d["heat"], d["ambient"], d["wall"]))
                energy = old.column.total_energy_j + dt * (
                    problems[name][7] - energyout + heat + float(data["energy_net"][name].cpu())
                )
                constants = torch.zeros_like(n).index_copy(0, active, q["constants"])
                seed = (
                    [x[0] for x in q["compositions"][: len(seeds[name][1])]],
                    seeds[name][1],
                    q["interfaces"],
                    float(q["pressures"][0][0].cpu()),
                    float(q["temperature"].cpu()),
                    constants,
                )
                try:
                    candidate, report = col._accept(n, energy, seed, order, d["column_variables"])
                except PropertyModelError:
                    topology_attempts += 1
                    if topology_attempts > 4:
                        raise HydraulicConvergenceError(
                            "Network phase-topology retry budget exhausted"
                        ) from None
                    seeds[name] = col._seed(n, energy, seed[3], seed[4])
                    previous = None
                    restart_topology = True
                    break
                if (
                    abs(candidate.temperature_k - seed[4]) > 1e-6
                    or max(
                        abs(a - b)
                        for a, b in zip(candidate.chemical_constants, constants.cpu().tolist())
                    )
                    > 1e-7
                    or any(
                        abs(h - k) > 1e-8 * col.geometry.height_m
                        for h, k in zip(
                            candidate.interface_heights_m, q["interfaces"].cpu().tolist()
                        )
                    )
                ):
                    raise ConservationError(
                        "Validated vessel differs from joint connection solution"
                    )
                if (
                    not w.min_temperature_k <= tw <= w.max_temperature_k
                    or abs(tw - old.wall_temperature_k) > w.max_temperature_change_k
                ):
                    raise HydraulicConvergenceError("Network wall step/domain limit exceeded")
                for port in d["ports"]:
                    if not bool(port["valid"].cpu()):
                        raise EquipmentOperatingError("External outlet outside domain")
                    for h, area, p, x, properties in port["pieces"]:
                        for j in range(order):
                            col._check_local(
                                float(p[j].cpu()),
                                candidate.temperature_k,
                                x[j],
                                {k: z[j] for k, z in properties.items()},
                            )
                for feed in v.feeds:
                    if feed.stream.molar_rate_mol_s and col.sample(
                        candidate, feed.height_m, side="below"
                    )["pressure_pa"] > feed.stream.pressure_pa * (1 + 1e-7):
                        raise EquipmentOperatingError("Network feed pressure below receiver")
                incoming_s = math.fsum(
                    feed.stream.molar_rate_mol_s
                    * float(
                        feed.stream.equilibrate(f, temperature_guess_k=old.column.temperature_k)[
                            "molar_entropy_j_mol_k"
                        ].cpu()
                    )
                    for feed in v.feeds
                    if feed.stream.molar_rate_mol_s
                )
                entropy_boundary += (
                    incoming_s - float(d["out_s"].cpu()) + ambient / w.ambient_temperature_k
                )
                feed_e += problems[name][7]
                out_e += energyout
                ambient_e += ambient
                component_boundary += problems[name][6] - d["out_n"]
                candidates[name] = StratifiedThermalState(
                    candidate, tw, state.time_s + dt, old.next_step_s, v.model_id
                )
                reports[name] = report
                next_seeds[name] = seed
            if restart_topology:
                continue
            # Validate all connector-side states, including dry phase probes, on their own EOS branches.
            for transfer in data["transfers"]:
                connection = next(p for p in self.connections if p.name == transfer["name"])
                for name, (p, x, q, height) in zip(
                    (connection.source, connection.target), transfer["points"]
                ):
                    col = self.vessels[name].column
                    for j in range(order):
                        col._check_local(
                            float(p[j].cpu()),
                            candidates[name].column.temperature_k,
                            x[j],
                            {k: v[j] for k, v in q.items()},
                        )
            error = math.inf
            if previous is not None:
                old_candidates, old_reports, old_fluxes, old_external = previous
                error = self._error(candidates, old_candidates)
                for name, a in reports.items():
                    b = old_reports[name]
                    tol = self.vessels[name].column.relative_tolerance
                    error = max(
                        error,
                        abs(a["relative_potential_energy_j"] - b["relative_potential_energy_j"])
                        / (1e-5 + tol * max(1, abs(a["relative_potential_energy_j"]))),
                    )
                    error = max(
                        error,
                        abs(a["mean_pressure_pa"] - b["mean_pressure_pa"])
                        / (1 + tol * a["mean_pressure_pa"]),
                        float(
                            (
                                (
                                    f.tensor(a["phase_component_moles"])
                                    - f.tensor(b["phase_component_moles"])
                                ).abs()
                                / (1e-9 + tol * sum(candidates[name].column.component_moles))
                            )
                            .max()
                            .cpu()
                        ),
                    )
                for name, d in data["vessels"].items():
                    for port, oldport in zip(d["ports"], old_external[name]["ports"]):
                        a, b = port["component_rates"], oldport["component_rates"]
                        error = max(
                            error,
                            float(
                                (
                                    (a - b).abs()
                                    / (
                                        1e-12
                                        + self.relative_tolerance * torch.maximum(a.abs(), b.abs())
                                    )
                                )
                                .max()
                                .cpu()
                            ),
                        )
                for a, b in zip(data["transfers"], old_fluxes):
                    ea, eb = (
                        float(a["total_energy_rate"].cpu()),
                        float(b["total_energy_rate"].cpu()),
                    )
                    error = max(
                        error,
                        abs(ea - eb) / (1e-7 + self.relative_tolerance * max(abs(ea), abs(eb))),
                    )
                    error = max(
                        error,
                        float(
                            (
                                (a["component_rates"] - b["component_rates"]).abs()
                                / (
                                    1e-12
                                    + self.relative_tolerance
                                    * torch.maximum(
                                        a["component_rates"].abs(), b["component_rates"].abs()
                                    )
                                )
                            )
                            .max()
                            .cpu()
                        ),
                    )
                if error <= 1:
                    old_n, old_e, old_s = self._totals(states)
                    new_n, new_e, new_s = self._totals(candidates)
                    nr = new_n - old_n - dt * component_boundary
                    er = new_e - old_e - dt * (feed_e - out_e + ambient_e)
                    production = new_s - old_s - dt * entropy_boundary
                    if float(nr.abs().max().cpu()) > 1e-8 + 1e-9 * float(new_n.sum().cpu()) or abs(
                        er
                    ) > 1e-5 + 1e-9 * max(1, abs(dt * feed_e), abs(dt * out_e), abs(new_e - old_e)):
                        raise ConservationError("Network component/energy ledger rejected")
                    if production < -1e-6 - 1e-9 * max(1, abs(old_s), abs(new_s)):
                        raise ConservationError("Network entropy production rejected")
                    result = StratifiedNetworkState(
                        tuple(candidates.items()),
                        state.time_s + dt,
                        state.next_step_s,
                        self.model_id,
                    )
                    transfers = []
                    for connection in self.connections:
                        parts = [p for p in data["transfers"] if p["name"] == connection.name]
                        transfers.append(
                            {
                                "name": connection.name,
                                "hydraulic_model": connection.hydraulic_model,
                                "hydraulics": {
                                    key: float(value.cpu())
                                    for key, value in data["hydraulic"]
                                    .get(connection.name, {})
                                    .items()
                                },
                                "component_rates_mol_s": sum(
                                    (p["component_rates"] for p in parts), torch.zeros_like(new_n)
                                )
                                .cpu()
                                .tolist(),
                                "total_energy_rate_w": math.fsum(
                                    float(p["total_energy_rate"].cpu()) for p in parts
                                ),
                                "volume_rate_m3_s": math.fsum(
                                    float(p["volume_rate"].cpu()) for p in parts
                                ),
                                "phase_pair_areas": [
                                    {
                                        "source_phase": p["source_phase"],
                                        "target_phase": p["target_phase"],
                                        "fraction": float(p["area"].cpu()),
                                    }
                                    for p in parts
                                ],
                            }
                        )
                    return result, {
                        "dt_s": dt,
                        "phase_topology_retries": topology_attempts,
                        "solver_attempts": solver_attempts,
                        "state": result.to_dict(),
                        "transfers": transfers,
                        "openings": openings,
                        "component_boundary_moles": (dt * component_boundary).cpu().tolist(),
                        "feed_energy_j": dt * feed_e,
                        "outlet_energy_j": dt * out_e,
                        "ambient_heat_j": dt * ambient_e,
                        "component_residuals_mol": nr.cpu().tolist(),
                        "energy_residual_j": er,
                        "entropy_production_j_k": production,
                        "quadrature_error_ratio": error,
                    }
            previous = candidates, reports, data["transfers"], data["vessels"]
            seeds = next_seeds
            order *= 2
        raise HydraulicConvergenceError("Network quadrature budget exhausted")

    def _error(self, a, b):
        error = 0.0
        f = self.fluid
        for name in self.vessels:
            x, y = a[name], b[name]
            col = self.vessels[name].column
            if len(x.column.phase_roots) != len(y.column.phase_roots):
                return math.inf
            nx, ny = f.tensor(x.column.component_moles), f.tensor(y.column.component_moles)
            scale = max(
                1,
                abs(
                    x.column.total_energy_j
                    - float((nx * f.molar_masses).sum().cpu()) * col.gravity_m_s2 * col.elevation_m
                ),
            )
            error = max(
                error,
                abs(x.column.temperature_k - y.column.temperature_k) / self.temperature_tolerance_k,
                abs(x.wall_temperature_k - y.wall_temperature_k) / self.temperature_tolerance_k,
                float(((nx - ny).abs() / (1e-9 + self.relative_tolerance * nx.sum())).max().cpu()),
                abs(x.column.total_energy_j - y.column.total_energy_j)
                / (1e-5 + self.relative_tolerance * scale),
            )
            for h, k in zip(x.column.interface_heights_m, y.column.interface_heights_m):
                error = max(
                    error, abs(h - k) / (1e-8 + self.relative_tolerance * col.geometry.height_m)
                )
            dx, dy = self.vessels[name]._data(x.column), self.vessels[name]._data(y.column)

            def mean_p(d, col=col):
                return (
                    float(
                        torch.stack([(w * p).sum() for w, p in zip(d["weights"], d["pressures"])])
                        .sum()
                        .cpu()
                    )
                    / col.geometry.volume_m3
                )

            px, py = mean_p(dx), mean_p(dy)
            error = max(error, abs(px - py) / (1 + self.relative_tolerance * max(px, py)))
        return error

    def _totals(self, states):
        f = self.fluid
        n = torch.zeros(len(f.names), dtype=torch.float64, device=f.device)
        energy = entropy = 0.0
        for name, s in states.items():
            n += f.tensor(s.column.component_moles)
            w = self.vessels[name].wall
            energy += s.column.total_energy_j + w.energy(s.wall_temperature_k)
            entropy += float(self.vessels[name]._data(s.column)["entropy_j_k"].cpu()) + w.entropy(
                s.wall_temperature_k
            )
        return n, energy, entropy

    def advance(self, state, duration_s, *, openings=None):
        self.inspect(state)
        commands = self._commands(openings)
        duration = _number(duration_s, "duration", positive=True)
        target = state.time_s + duration
        if not math.isfinite(target) or target <= state.time_s:
            raise ValueError("Unrepresentable network end time")
        current = state
        dt = min(state.next_step_s, duration)
        history = []
        attempts = 0
        while current.time_s < target:
            attempts += 1
            if attempts > self.maximum_attempts:
                raise HydraulicConvergenceError("Network integration budget exhausted")
            dt = min(dt, target - current.time_s)
            try:
                coarse, _ = self._step(current, dt, commands)
                half, one = self._step(current, dt / 2, commands)
                fine, two = self._step(half, dt / 2, commands)
                error = self._error(dict(coarse.vessels), dict(fine.vessels))
                if error > 1:
                    raise HydraulicConvergenceError("Network time error exceeds tolerance")
            except (HydraulicConvergenceError, PropertyModelError, ConservationError) as exc:
                dt *= 0.5
                if dt < self.minimum_step_s:
                    raise HydraulicConvergenceError("Network minimum step exhausted") from exc
                continue
            history.extend((one, two))
            nextdt = min(
                self.maximum_step_s, max(self.minimum_step_s, dt * (1.5 if error < 0.25 else 1.0))
            )
            time = target if dt >= target - current.time_s else min(fine.time_s, target)
            children = tuple(
                (
                    name,
                    StratifiedThermalState(
                        s.column, s.wall_temperature_k, time, s.next_step_s, s.model_id
                    ),
                )
                for name, s in fine.vessels
            )
            current = StratifiedNetworkState(children, time, nextdt, self.model_id)
            dt = nextdt
        old_n, old_e, _old_s = self._totals(dict(state.vessels))
        new_n, new_e, _new_s = self._totals(dict(current.vessels))
        boundary = sum(
            (self.fluid.tensor(s["component_boundary_moles"]) for s in history),
            torch.zeros_like(old_n),
        )
        ambient = math.fsum(s["ambient_heat_j"] for s in history)
        feed = math.fsum(s["feed_energy_j"] for s in history)
        out = math.fsum(s["outlet_energy_j"] for s in history)
        nr = new_n - old_n - boundary
        er = new_e - old_e - ambient - feed + out
        if float(nr.abs().max().cpu()) > 1e-8 + 1e-9 * float(new_n.sum().cpu()) or abs(
            er
        ) > 1e-5 + 1e-9 * max(1, abs(feed), abs(out), abs(ambient), abs(new_e - old_e)):
            raise ConservationError("Network interval ledger rejected")
        return {
            "state": current.to_dict(),
            "steps": history,
            "attempts": attempts,
            "component_residuals_mol": nr.cpu().tolist(),
            "energy_residual_j": er,
            "entropy_production_j_k": math.fsum(s["entropy_production_j_k"] for s in history),
            "ambient_heat_j": ambient,
            "feed_energy_j": feed,
            "outlet_energy_j": out,
        }
