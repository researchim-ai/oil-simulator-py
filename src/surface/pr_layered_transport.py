"""Bounded, monolithic inclined profile-cell Euler transport.

No phase slip, layer inertia, transverse momentum, lateral ports or area changes.
The instantaneous gravity-equilibrium profile is an algebraic closure, not a
resolved interface dynamics model. Phase topology is fixed within a Newton; balance-predicted retries can change it.
All phase-pair intersections, including zero-area patches, remain in Newton. Sources use the *current* profile.
"""

import math
from dataclasses import asdict, dataclass, replace

import torch

from .darcy_hydraulics import darcy_pressure_loss
from .inclined_cylinder import InclinedCylinderGeometry
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .peng_robinson import R
from .pr_momentum_vessels import hllc_port
from .stratified_pr import StratifiedPRColumn, StratifiedPRState
from .torch_linear import TorchNewton


@dataclass(frozen=True)
class LayeredCellState:
    profile: StratifiedPRState
    momentum_kg_m_s: float
    total_energy_j: float

    def __post_init__(self):
        _number(self.momentum_kg_m_s, "axial momentum")
        _number(self.total_energy_j, "total energy")

    @classmethod
    def from_dict(cls, data):
        return cls(
            StratifiedPRState.from_dict(data["profile"]),
            data["momentum_kg_m_s"],
            data["total_energy_j"],
        )


@dataclass(frozen=True)
class LayeredTransportState:
    cells: tuple
    time_s: float
    model_id: str
    next_step_s: float = 1e-4

    def __post_init__(self):
        _number(self.next_step_s, "next step", positive=True)
        object.__setattr__(self, "cells", tuple(self.cells))
        _number(self.time_s, "time", nonnegative=True)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Unknown layered state fields")
        return cls(
            tuple(LayeredCellState.from_dict(c) for c in data["cells"]),
            data["time_s"],
            data["model_id"],
            data["next_step_s"],
        )


class PRLayeredTransportNetwork:
    """Straight coaxial constant-D closed chain, one M*u per profile cell.

    All profile, face P/x and velocity variables share one Torch Newton.
    ``wave_speed_bound_m_s`` is a caller-supplied subsonic HLLC bound, checked
    against homogeneous frozen-phase acoustic speeds at accepted face traces.
    No ordinary transport graph bridge is supplied by this release kernel.
    """

    def __init__(
        self,
        columns,
        *,
        face_order=8,
        wave_speed_bound_m_s=2000.0,
        slip=False,
        lateral_ports=(),
        dense_limit=256,
        max_unknowns=16384,
        dynamic_viscosity_pa_s=0.0,
        roughness_m=0.0,
    ):
        self.columns = tuple(columns)
        if not self.columns or any(not isinstance(c, StratifiedPRColumn) for c in self.columns):
            raise TypeError("Profile columns required")
        if slip or lateral_ports:
            raise ValueError("Shared-velocity end-face kernel rejects slip and lateral ports")
        if face_order not in (2, 4, 8, 16):
            raise ValueError("Face order must be 2/4/8/16")
        for name, value in (("dense_limit", dense_limit), ("max_unknowns", max_unknowns)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        self.face_order = face_order
        self.alpha = _number(wave_speed_bound_m_s, "wave bound", positive=True)
        first = self.columns[0]
        self.fluid = first.fluid
        self.gravity = first.gravity_m_s2
        self.datum = first.elevation_m  # numerical energy datum only
        self.angle = (
            first.geometry.inclination_rad
            if isinstance(first.geometry, InclinedCylinderGeometry)
            else None
        )
        if self.angle is None or abs(self.angle) >= math.pi / 2 - 1e-8:
            raise ValueError("Nonvertical InclinedCylinderGeometry required")
        self.diameter = first.geometry.diameter_m
        self.dynamic_viscosity_pa_s = _number(
            dynamic_viscosity_pa_s, "effective mixture viscosity", nonnegative=True
        )
        self.roughness_m = _number(roughness_m, "roughness", nonnegative=True)
        if self.roughness_m >= self.diameter / 2:
            raise ValueError("Roughness must be less than D/2")
        self.cosine = math.cos(self.angle)
        for i, col in enumerate(self.columns):
            g = col.geometry
            if (
                not isinstance(g, InclinedCylinderGeometry)
                or g.diameter_m != self.diameter
                or g.inclination_rad != self.angle
            ):
                raise ValueError("Straight coaxial constant diameter cells required")
            if (
                col.fluid.model_id != self.fluid.model_id
                or col.fluid.device != self.fluid.device
                or col.gravity_m_s2 != self.gravity
            ):
                raise ValueError("Common EOS, device and gravity required")
            if i:
                prev = self.columns[i - 1]
                expected = prev.elevation_m + (prev.geometry.length_m + g.length_m) / 2 * math.sin(
                    self.angle
                )
                # Local bottom = centre elevation minus half vertical envelope.
                expected += (prev.geometry.height_m - g.height_m) / 2
                if abs(col.elevation_m - expected) > 1e-8:
                    raise ValueError("Cell end disks are not coaxial and coincident")
        self.order = max(c.quadrature_order for c in self.columns)
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_layered_transport_v1",
            "columns": [c.to_dict() for c in self.columns],
            "face_order": self.face_order,
            "wave_speed_bound_m_s": self.alpha,
            "dense_limit": self.dense_limit,
            "max_unknowns": self.max_unknowns,
            "dynamic_viscosity_pa_s": self.dynamic_viscosity_pa_s,
            "roughness_m": self.roughness_m,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "pr_layered_transport_v1":
            raise ValueError("Unknown layered transport model")
        return cls(
            [StratifiedPRColumn.from_dict(c, device=device) for c in data.pop("columns")], **data
        )

    @staticmethod
    def _seed(col, state):
        f = col.fluid
        return (
            [f.tensor(x) for x in state.phase_seeds],
            list(state.phase_roots),
            f.tensor(state.interface_heights_m),
            math.exp(state.solution_variables[0]),
            state.temperature_k,
            f.tensor(state.chemical_constants),
        )

    def _mass(self, profile):
        return sum(
            n * c.molar_mass_kg_mol for n, c in zip(profile.component_moles, self.fluid.components)
        )

    def initialize(self, profiles, velocities_m_s=None):
        profiles = tuple(profiles)
        velocities = (0.0,) * len(profiles) if velocities_m_s is None else tuple(velocities_m_s)
        if len(profiles) != len(self.columns) or len(velocities) != len(profiles):
            raise ValueError("One profile and velocity per cell required")
        cells = []
        for p, v in zip(profiles, velocities):
            _number(v, "velocity")
            mass = self._mass(p)
            cells.append(LayeredCellState(p, mass * v, p.total_energy_j + 0.5 * mass * v * v))
        state = LayeredTransportState(cells, 0.0, self.model_id)
        self.inspect(state)
        return state

    def initialize_equilibrium(self, pressure_pa, temperature_k, mole_fractions):
        """Restrict one global gravity profile to actual cell cut geometries.

        Initialization may solve equilibrium. Time-step residuals never call
        this method, flash, sample, equilibrate, or another Newton solver.
        """
        cfg = self.columns[0].to_dict()
        cfg["geometry"] = {
            **cfg["geometry"],
            "length_m": sum(c.geometry.length_m for c in self.columns),
        }
        cfg["elevation_m"] = min(c.elevation_m for c in self.columns)
        reference = StratifiedPRColumn.from_dict(cfg, device=self.fluid.device)
        profile = StratifiedPRState.from_dict(
            reference.initialize(pressure_pa, temperature_k, mole_fractions)["state"]
        )
        return self.initialize(self.restrict_equilibrium(reference, profile))

    def restrict_equilibrium(self, reference, profile):
        """Initialize cells from the same absolute chemical-potential profile."""
        reference.inspect(profile)
        if (
            reference.fluid.model_id != self.fluid.model_id
            or reference.gravity_m_s2 != self.gravity
        ):
            raise ValueError("Reference EOS/gravity mismatch")
        f = self.fluid
        active = torch.nonzero(f.tensor(profile.component_moles) > 0).flatten()
        nc = len(active)
        global_bounds = [
            reference.elevation_m,
            *(reference.elevation_m + h for h in profile.interface_heights_m),
            reference.elevation_m + reference.geometry.height_m,
        ]
        states = []
        for col in self.columns:
            low, high = col.elevation_m, col.elevation_m + col.geometry.height_m
            if low < global_bounds[0] - 1e-8 or high > global_bounds[-1] + 1e-8:
                raise ValueError("Reference profile does not cover cell")
            phases = [
                i
                for i in range(len(profile.phase_roots))
                if min(high, global_bounds[i + 1]) > max(low, global_bounds[i]) + 1e-12
            ]
            interfaces = f.tensor([global_bounds[i + 1] - low for i in phases[:-1]])
            constants = f.tensor(profile.chemical_constants) - (
                f.molar_masses
                * self.gravity
                * (low - reference.elevation_m)
                / (R * profile.temperature_k)
            )
            seed = (
                [f.tensor(profile.phase_seeds[i]) for i in phases],
                [profile.phase_roots[i] for i in phases],
                interfaces,
                math.exp(profile.solution_variables[0]),
                profile.temperature_k,
                constants,
            )
            n0 = f.tensor(profile.component_moles) * (
                col.geometry.volume_m3 / reference.geometry.volume_m3
            )
            problem = col._problem(n0, profile.total_energy_j, seed, self.order)

            # At fixed lambda,T the chemical/interface equations determine
            # inventories and U+PE; no conservation target is imposed here.
            def residual(y, problem=problem, constants=constants):
                r, d = problem[3](y)
                return torch.cat(
                    (
                        r[: -nc - 1],
                        d["constants"] - constants[active],
                        (d["temperature"] / f.tref - profile.temperature_k / f.tref).reshape(1),
                    )
                )

            solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=768)
            solver.host_convergence = True
            with torch.no_grad():
                y, _, ok, count = solver.solve(residual, *problem[:3], tolerance=3e-12)
                if not bool(ok.cpu()):
                    raise HydraulicConvergenceError("Global profile restriction rejected")
                d = problem[3](y)[1]
                n = d["component_moles"]
                energy = d["internal_energy_j"] + d["relative_potential_energy_j"]
                energy += (n * f.molar_masses).sum() * self.gravity * col.elevation_m
                accepted, _ = col._accept(
                    n, float(energy.cpu()), seed, self.order, y, evaluations=count
                )
            states.append(accepted)
        return tuple(states)

    def inspect(self, state):
        if (
            not isinstance(state, LayeredTransportState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
            or len(state.cells) != len(self.columns)
        ):
            raise PropertyModelError("Layered model/state mismatch")
        supports = []
        for col, cell in zip(self.columns, state.cells):
            col.inspect(cell.profile)
            mass = self._mass(cell.profile)
            if mass <= 0 or abs(cell.momentum_kg_m_s / mass) >= self.alpha:
                raise PropertyModelError("Cell mass or velocity outside closure domain")
            expected = cell.profile.total_energy_j + 0.5 * cell.momentum_kg_m_s**2 / mass
            if abs(expected - cell.total_energy_j) > 1e-7 + 1e-10 * abs(expected):
                raise PropertyModelError("Profile U+PE and total E/K disagree")
            supports.append(tuple(n > 0 for n in cell.profile.component_moles))
        if any(s != supports[0] for s in supports):
            raise PropertyModelError("Common active component support required")

    def _disk_bottom(self, i, side):
        col = self.columns[i]
        return (
            col.elevation_m
            + col.geometry.height_m / 2
            + side * col.geometry.length_m / 2 * math.sin(self.angle)
            - self.diameter * self.cosine / 2
        )

    def _bounds(self, i, data):
        col = self.columns[i]
        return torch.cat(
            (
                self.fluid.tensor([col.elevation_m]),
                data["interfaces"] + col.elevation_m,
                self.fluid.tensor([col.elevation_m + col.geometry.height_m]),
            )
        )

    def _patch(self, bottom, a, b):
        """Gauss angle quadrature on disk strips; dA=2*r²*cos²(phi)dphi."""
        f = self.fluid
        span = self.diameter * self.cosine

        # At fixed disk caps asin is constant; avoid infinite endpoint AD slopes.
        def angle(h):
            x = 2 * (h - bottom) / span - 1
            regular = torch.asin(x.clamp(-1 + 1e-14, 1 - 1e-14))
            return torch.where(
                x <= -1, f.tensor(-math.pi / 2), torch.where(x >= 1, f.tensor(math.pi / 2), regular)
            )

        aa, bb = angle(a), angle(b)
        nodes, weights = self.columns[0]._rules[self.face_order]
        phi = (aa + bb) / 2 + (bb - aa) / 2 * nodes
        h = bottom + span / 2 * (1 + torch.sin(phi))
        w = weights * (bb - aa) / 2 * self.diameter**2 / 2 * torch.cos(phi) ** 2
        radius = self.diameter / 2
        delta = bb - aa
        ds = torch.where(
            delta.abs() < 1e-3,
            delta**3 / 6 - delta**5 / 120 + delta**7 / 5040,
            delta - torch.sin(delta),
        )
        area = radius**2 * (ds + 2 * torch.cos((aa + bb) / 2).square() * torch.sin(delta))
        w = w * area / w.sum().clamp_min(1e-300)
        # Exact area and first vertical moment protect small gravity forces
        # from subtraction of two large end-pressure integrals.
        local = h - bottom
        moment = self.cosine * (
            radius * area + 2 * radius**3 / 3 * (torch.cos(aa) ** 3 - torch.cos(bb) ** 3)
        )
        mean = (w * local).sum() / area.clamp_min(1e-300)
        variance = (w * (local - mean).square()).sum()
        correction = (moment - area * mean) * (local - mean) / variance.clamp_min(1e-300)
        # Tiny intersections need no moment correction: analytic moment
        # subtraction loses precision there; Gauss is already locally exact.
        w = w * (1 + torch.where(delta > 1e-3, correction, torch.zeros_like(correction)))
        return h, w

    def _layout(self, state, seeds=None):
        """Enumerate all phase pairs; live intersections form the interface union."""
        roots = (
            [s[1] for s in seeds]
            if seeds is not None
            else [c.profile.phase_roots for c in state.cells]
        )
        patches = []
        for i, cell in enumerate(state.cells):
            for side in (-1, 1):
                for phase in range(len(roots[i])):
                    patches.append(("local", i, side, phase, -1))
        for i in range(len(self.columns) - 1):
            for pa in range(len(roots[i])):
                for pb in range(len(roots[i + 1])):
                    patches.append(("shared", i, 1, pa, pb))
        return patches

    def _heat(self, heat_w):
        if heat_w is None:
            return (0.0,) * len(self.columns)
        if isinstance(heat_w, (float, int)):
            heat_w = (heat_w,) * len(self.columns)
        heat = tuple(_number(v, "cell heat") for v in heat_w)
        if len(heat) != len(self.columns):
            raise ValueError("One heat rate per cell required")
        return heat

    def _problem(
        self, state, dt, *, heat_w=None, seed_overrides=None, external_ends=(False, False)
    ):
        self.inspect(state)
        heat = self._heat(heat_w)
        overrides = seed_overrides or {}
        f = self.fluid
        active = torch.nonzero(f.tensor(state.cells[0].profile.component_moles) > 0).flatten()
        nc = len(active)
        problems, initials, lows, highs, slices, seeds = [], [], [], [], [], []
        offset = 0
        for i, (col, cell) in enumerate(zip(self.columns, state.cells)):
            p = cell.profile
            seed = overrides.get(i, self._seed(col, p))
            problem = col._problem(f.tensor(p.component_moles), p.total_energy_j, seed, self.order)
            initial = (
                f.tensor(p.solution_variables)
                if p.quadrature_order == self.order and i not in overrides
                else problem[0]
            )
            slices.append(slice(offset, offset + len(initial)))
            offset += len(initial)
            initials.append(initial)
            lows.append(problem[1])
            highs.append(problem[2])
            problems.append(problem)
            seeds.append(seed)
        velocity_slice = slice(offset, offset + len(state.cells))
        initials.append(f.tensor([s.momentum_kg_m_s / self._mass(s.profile) for s in state.cells]))
        lows.append(f.tensor([-self.alpha * 0.9] * len(state.cells)))
        highs.append(f.tensor([self.alpha * 0.9] * len(state.cells)))
        offset += len(state.cells)
        layout = self._layout(state, seeds)
        trace_slices = []
        for kind, i, side, pa, pb in layout:
            indices = [(i, pa)] if kind == "local" else [(i, pa), (i + 1, pb)]
            block = []
            for j, phase in indices:
                p = state.cells[j].profile
                x = seeds[j][0][phase][active].clamp_min(1e-30)
                row = torch.cat((f.tensor([math.log(seeds[j][3])]), (x[:-1] / x[-1]).log()))
                block.append(row.repeat(self.face_order))
            initial = torch.cat(block)
            trace_slices.append(slice(offset, offset + len(initial)))
            offset += len(initial)
            initials.append(initial)
            lows.append(
                f.tensor([math.log(f.min_pressure_pa)] + [-100.0] * (nc - 1)).repeat(
                    len(indices) * self.face_order
                )
            )
            highs.append(
                f.tensor([math.log(f.max_pressure_pa)] + [100.0] * (nc - 1)).repeat(
                    len(indices) * self.face_order
                )
            )
        if offset > self.max_unknowns:
            raise ValueError("Layered joint Newton exceeds configured max_unknowns")
        # Datum-relative energy equations avoid loss of accuracy under elevation shifts.
        nold = [f.tensor(s.profile.component_moles) for s in state.cells]
        nscale = [n[active].clamp_min(n.sum() * 1e-12) for n in nold]
        escale = [
            max(
                1.0,
                abs(
                    s.profile.total_energy_j - self._mass(s.profile) * self.gravity * c.elevation_m
                ),
                sum(s.profile.component_moles) * R * s.profile.temperature_k,
            )
            for c, s in zip(self.columns, state.cells)
        ]

        def evaluate(y, external_fluxes=None):
            raw, bodies = zip(*(p[3](y[sl]) for p, sl in zip(problems, slices)))
            velocities = y[velocity_slice]
            masses = [(d["component_moles"] * f.molar_masses).sum() for d in bodies]
            kinetic = [0.5 * m * v * v for m, v in zip(masses, velocities)]
            pressure = [[y.new_zeros(()), y.new_zeros(())] for _ in bodies]
            wall_pressure = [[y.new_zeros(()), y.new_zeros(())] for _ in bodies]
            wall_faces = []
            rates = [y.new_zeros(len(f.names) + 2) for _ in bodies]
            chemical, valid, shared, diagnostics, boundary = [], [], [], [], []
            bounds = [self._bounds(i, d) for i, d in enumerate(bodies)]
            coverage = [y.new_zeros(()) for _ in range(len(bodies) - 1)]
            for spec, sl in zip(layout, trace_slices):
                kind, i, side, pa, pb = spec
                bottom = self._disk_bottom(i, side)
                a = torch.maximum(bounds[i][pa], y.new_tensor(bottom))
                b = torch.minimum(
                    bounds[i][pa + 1], y.new_tensor(bottom + self.diameter * self.cosine)
                )
                if kind == "shared":
                    a = torch.maximum(a, bounds[i + 1][pb])
                    b = torch.minimum(b, bounds[i + 1][pb + 1])
                # Invalid topology is rejected after Newton, not repaired by a flash.
                a = a.clamp(bottom, bottom + self.diameter * self.cosine)
                b = b.clamp(bottom, bottom + self.diameter * self.cosine)
                h, weights = self._patch(bottom, a, torch.maximum(a, b))
                valid.append(torch.isfinite(weights).all() & (weights >= 0).all())
                indices = [(i, pa)] if kind == "local" else [(i, pa), (i + 1, pb)]
                traces = []
                rows = y[sl].reshape(len(indices), self.face_order, nc)
                for k, (j, phase) in enumerate(indices):
                    d, col = bodies[j], self.columns[j]
                    p, x, q = col._decode_nodes(
                        rows[k], active, d["temperature"], seeds[j][1][phase]
                    )
                    mu = (
                        x[:, active].clamp_min(1e-300).log()
                        + q["log_fugacity_coefficients"][:, active]
                        + (p / f.pref).log()[:, None]
                    )
                    chemical.append(
                        (
                            mu
                            + self.gravity
                            * (h - col.elevation_m)[:, None]
                            * f.molar_masses[active]
                            / (R * d["temperature"])
                            - d["constants"]
                        ).flatten()
                    )
                    valid.append(
                        q["valid"].all() & (q["frozen_sound_speed_m_s"] <= self.alpha).all()
                    )
                    diagnostics.append((spec, j, weights, q["valid"], q["frozen_sound_speed_m_s"]))
                    rho, volume = q["density_kg_m3"], q["molar_volume_m3_mol"]
                    u = velocities[j]
                    energy = q["molar_internal_energy_j_mol"] / volume + rho * (
                        0.5 * u * u + self.gravity * (h - self.datum)
                    )
                    conserved = torch.cat(
                        (x / volume[:, None], (rho * u)[:, None], energy[:, None]), dim=1
                    )
                    flux = torch.cat(
                        (
                            x / volume[:, None] * u,
                            (rho * u * u + p)[:, None],
                            ((energy + p) * u)[:, None],
                        ),
                        dim=1,
                    )
                    entropy = q["molar_entropy_j_mol_k"] / volume
                    traces.append((conserved, flux, p, u.expand_as(p), rho, entropy))
                if kind == "local":
                    if (i == 0 and side == -1) or (i == len(bodies) - 1 and side == 1):
                        boundary.append(
                            {
                                "cell": i,
                                "side": side,
                                "phase": pa,
                                "heights": h,
                                "weights": weights,
                                "packet": traces[0],
                            }
                        )
                    pressure[i][0 if side == -1 else 1] = (
                        pressure[i][0 if side == -1 else 1] + (weights * traces[0][2]).sum()
                    )
                    exterior = (i == 0 and side == -1) or (i == len(bodies) - 1 and side == 1)
                    if exterior and not external_ends[0 if side == -1 else 1]:
                        packet = traces[0]
                        conserved, flux, p, velocity, rho, entropy = packet
                        ghost = (
                            torch.cat(
                                (conserved[:, :-2], -conserved[:, -2:-1], conserved[:, -1:]), dim=1
                            ),
                            torch.cat((-flux[:, :-2], flux[:, -2:-1], -flux[:, -1:]), dim=1),
                            p,
                            -velocity,
                            rho,
                            entropy,
                        )
                        left, right = (ghost, packet) if side == -1 else (packet, ghost)
                        fan = torch.vmap(lambda *args: hllc_port(args[:6], args[6:], self.alpha))(
                            *left, *right
                        )
                        valid.append(fan[4].all())
                        # Reflection has zero contact speed and no N/E or
                        # entropy transport. Retain only its momentum traction,
                        # eliminating cancellation roundoff in the zero fluxes.
                        traction = (weights * fan[0][:, -2]).sum()
                        wall_pressure[i][0 if side == -1 else 1] += traction
                        wall_faces.append((spec, h, weights, fan))
                else:
                    # Exactly one fan at each common quadrature point, then opposite scatter.
                    face = torch.vmap(lambda *args: hllc_port(args[:6], args[6:], self.alpha))(
                        *traces[0], *traces[1]
                    )
                    flow = (weights[:, None] * face[0]).sum(0)
                    rates[i] = rates[i] - flow
                    rates[i + 1] = rates[i + 1] + flow
                    valid.append(face[4].all())
                    shared.append((spec, h, weights, face, traces))
                    coverage[i] = coverage[i] + weights.sum()
            exchange = None
            if external_fluxes is not None:
                exchange = external_fluxes(
                    {"bodies": bodies, "velocities": velocities, "boundary": boundary}
                )
                rates = [a + b for a, b in zip(rates, exchange["net"])]
                valid.append(exchange["valid"])
            elif any(external_ends):
                raise ValueError("Open profile ends require an external_fluxes callback")
            residuals, sources, momentumrows, friction_forces = [], [], [], []
            for i, (d, r, cell, col) in enumerate(zip(bodies, raw, state.cells, self.columns)):
                source = pressure[i][1] - pressure[i][0]
                sources.append(source)
                # Wall traction p_left A, -p_right A; no mass or energy wall flux.
                wall = wall_pressure[i][0] - wall_pressure[i][1]
                nr = (
                    d["component_moles"][active] - nold[i][active] - dt * rates[i][active]
                ) / nscale[i]
                # Use cell-local datum so the column energy row has the same scaling.
                massflow = (rates[i][:-2] * f.molar_masses).sum()
                oldmass = self._mass(cell.profile)
                old_relative = (
                    cell.profile.total_energy_j - oldmass * self.gravity * col.elevation_m
                )
                er = (
                    d["internal_energy_j"]
                    + d["relative_potential_energy_j"]
                    - old_relative
                    + kinetic[i]
                    - 0.5 * cell.momentum_kg_m_s**2 / oldmass
                    - dt
                    * (
                        rates[i][-1]
                        + massflow * self.gravity * (self.datum - col.elevation_m)
                        + heat[i]
                    )
                ) / escale[i]
                friction = y.new_zeros(())
                if self.dynamic_viscosity_pa_s > 0:
                    volume = col.geometry.volume_m3
                    loss, _ = darcy_pressure_loss(
                        velocities[i],
                        masses[i] / volume,
                        1.0,
                        self.diameter,
                        self.dynamic_viscosity_pa_s,
                        self.roughness_m,
                    )
                    friction = -loss * volume
                friction_forces.append(friction)
                momentum_scale = max(1.0, self._mass(cell.profile) * self.alpha)
                mr = (
                    masses[i] * velocities[i]
                    - cell.momentum_kg_m_s
                    - dt * (rates[i][-2] + wall + source + friction)
                ) / momentum_scale
                residuals.append(torch.cat((r[: -nc - 1], nr, er.reshape(1))))
                momentumrows.append(mr)
            result = torch.cat((*residuals, torch.stack(momentumrows), *chemical))
            return result, {
                "residual": result,
                "bodies": bodies,
                "velocities": velocities,
                "masses": masses,
                "kinetic": kinetic,
                "rates": rates,
                "pressure": pressure,
                "wall_pressure": wall_pressure,
                "wall_faces": wall_faces,
                "sources": sources,
                "friction_force_n": friction_forces,
                "valid": valid,
                "shared": shared,
                "coverage": coverage,
                "slices": slices,
                "seeds": seeds,
                "diagnostics": diagnostics,
                "boundary": boundary,
                "exchange": exchange,
            }

        body_blocks = [
            list(range(sl.start, sl.stop)) + [velocity_slice.start + i]
            for i, sl in enumerate(slices)
        ]
        trace_blocks = [list(range(sl.start, sl.stop)) for sl in trace_slices]
        blocks = body_blocks + trace_blocks
        if any(len(b) > 256 for b in blocks):
            raise ValueError("Profile/face physical block exceeds 256 unknowns")
        edges = set()
        # Conservative structural support: do not infer zeros from a resting Jacobian.
        for i, rows in enumerate(body_blocks):
            deps = {
                v
                for j in range(max(0, i - 1), min(len(body_blocks), i + 2))
                for v in body_blocks[j]
            }
            for spec, block in zip(layout, trace_blocks):
                kind, j, _, _, _ = spec
                if j == i or (kind == "shared" and j + 1 == i):
                    deps.update(block)
            edges.update((r, c) for r in rows for c in deps)
        for spec, rows in zip(layout, trace_blocks):
            kind, i, _, _, _ = spec
            deps = set(rows + body_blocks[i])
            if kind == "shared":
                deps.update(body_blocks[i + 1])
            edges.update((r, c) for r in rows for c in deps)
        pairs = sorted(edges)
        evaluate.sparsity = ([p[0] for p in pairs], [p[1] for p in pairs])
        evaluate.linear_blocks = blocks
        evaluate.profile_slices = slices
        evaluate.velocity_slice = velocity_slice
        evaluate.trace_slices = trace_slices
        return torch.cat(initials), torch.cat(lows), torch.cat(highs), evaluate

    def _candidate(self, state, dt_s, *, heat_w=None, seed_overrides=None, trial=None):
        dt = _number(dt_s, "step", positive=True)
        heat = self._heat(heat_w)
        y, lo, hi, evaluate = self._problem(state, dt, heat_w=heat, seed_overrides=seed_overrides)
        solver = TorchNewton(self.fluid.device, maximum_iterations=40, dense_limit=self.dense_limit)
        solver.host_convergence = True
        with torch.no_grad():
            y, _, ok, evaluations = solver.solve(
                lambda z: evaluate(z)[0],
                y,
                lo,
                hi,
                tolerance=3e-12,
                pattern=evaluate.sparsity if len(y) > self.dense_limit else None,
                blocks=evaluate.linear_blocks,
            )
            r, d = evaluate(y)
        if trial is not None:
            trial[:] = [d]
        if not bool((ok & torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("Layered joint profile/face/Euler Newton rejected")
        return self.accept_solved(state, dt, y, d, heat_w=heat, evaluations=evaluations)

    def accept_solved(
        self, state, dt, y, d, heat_w=None, external_ends=(False, False), *, evaluations=0
    ):
        """Validate and publish the supplied joint iterate, without a new solve.

        y is the native slice; d is joint_data['profile'] (or native evaluate
        data). Open-end N/E/momentum transfers are subtracted in global ledgers.
        Heat has no utility-temperature closure here: entropy production is
        reported/checked for adiabatic steps only. Parent owns heat entropy.
        """
        dt = _number(dt, "step", positive=True)
        heat = self._heat(heat_w)
        self.inspect(state)
        r = d["residual"]
        if not bool((torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("Supplied layered joint iterate is not converged")
        if not all(bool(v.cpu()) for v in d["valid"]):
            raise PropertyModelError(
                "Layered face topology, EOS or subsonic fan rejected: "
                + str(
                    [
                        (spec, j, w.tolist(), v.tolist(), c.tolist())
                        for spec, j, w, v, c in d["diagnostics"]
                        if not bool((v.all() & (c <= self.alpha).all() & (w >= 0).all()).cpu())
                    ]
                )
            )
        area = math.pi * self.diameter**2 / 4
        if any(abs(float(a.cpu()) - area) > 1e-10 * area for a in d["coverage"]):
            raise PropertyModelError("Shared disk phase intersections do not cover face")
        gravity_force = [-float(m.cpu()) * self.gravity * math.sin(self.angle) for m in d["masses"]]
        source_errors = [float(s.cpu()) - g for s, g in zip(d["sources"], gravity_force)]
        if any(abs(e) > 1e-7 + 1e-4 * abs(g) for e, g in zip(source_errors, gravity_force)):
            raise PropertyModelError(
                "Local pressure source/volume gravity quadrature mismatch; refine order"
            )
        cells, reports = [], []
        for i, (col, old, body) in enumerate(zip(self.columns, state.cells, d["bodies"])):
            n = self.fluid.tensor(old.profile.component_moles) + dt * d["rates"][i][:-2]
            massflow = (d["rates"][i][:-2] * self.fluid.molar_masses).sum()
            oldmass = self._mass(old.profile)
            old_relative = old.profile.total_energy_j - oldmass * self.gravity * col.elevation_m
            relative = (
                old_relative
                + 0.5 * old.momentum_kg_m_s**2 / oldmass
                - d["kinetic"][i]
                + dt
                * (
                    d["rates"][i][-1]
                    + massflow * self.gravity * (self.datum - col.elevation_m)
                    + heat[i]
                )
            )
            energy = relative + (n * self.fluid.molar_masses).sum() * self.gravity * col.elevation_m
            p, report = col._accept(
                n,
                float(energy.cpu()),
                d["seeds"][i],
                self.order,
                y[d["slices"][i]],
                evaluations=evaluations,
            )
            cells.append(
                LayeredCellState(
                    p,
                    float((d["masses"][i] * d["velocities"][i]).cpu()),
                    float((energy + d["kinetic"][i]).cpu()),
                )
            )
            reports.append(report)
        new = LayeredTransportState(cells, state.time_s + dt, self.model_id, state.next_step_s)
        self.inspect(new)
        if any(external_ends) and d["exchange"] is None:
            raise ValueError("Open-end acceptance requires decoded boundary transfers")
        external = (
            torch.stack(d["exchange"]["net"]).sum(0)
            if d["exchange"] is not None
            else y.new_zeros(len(self.fluid.names) + 2)
        )
        nr = [
            math.fsum(
                b.profile.component_moles[k] - a.profile.component_moles[k]
                for a, b in zip(state.cells, new.cells)
            )
            - dt * float(external[k].cpu())
            for k in range(len(self.fluid.names))
        ]
        er = math.fsum(
            (b.total_energy_j - self._mass(b.profile) * self.gravity * self.datum)
            - (a.total_energy_j - self._mass(a.profile) * self.gravity * self.datum)
            for a, b in zip(state.cells, new.cells)
        )
        er -= dt * (math.fsum(heat) + float(external[-1].cpu()))
        wall_force = float((d["wall_pressure"][0][0] - d["wall_pressure"][-1][1]).cpu())
        momentum_error = math.fsum(
            b.momentum_kg_m_s - a.momentum_kg_m_s for a, b in zip(state.cells, new.cells)
        ) - dt * (
            wall_force
            + math.fsum(float(q.cpu()) for q in d["sources"])
            + math.fsum(float(q.cpu()) for q in d["friction_force_n"])
            + float(external[-2].cpu())
        )
        if max(map(abs, nr)) > 1e-8 or abs(er) > 1e-5 or abs(momentum_error) > 1e-7:
            raise ConservationError("Layered boundary-corrected N/E/momentum ledger rejected")
        old_entropy = 0.0
        for col, cell in zip(self.columns, state.cells):
            p = cell.profile
            data = col._problem(
                self.fluid.tensor(p.component_moles),
                p.total_energy_j,
                self._seed(col, p),
                p.quadrature_order,
            )[3](self.fluid.tensor(p.solution_variables))[1]
            old_entropy += float(data["entropy_j_k"].cpu())
        entropy_change = (
            math.fsum(float(body["entropy_j_k"].cpu()) for body in d["bodies"]) - old_entropy
        )
        incoming_entropy = (
            float(d["exchange"].get("incoming_entropy", y.new_zeros(())).cpu())
            if d["exchange"] is not None
            else 0.0
        )
        production = entropy_change - dt * incoming_entropy if not any(heat) else None
        if production is not None and production < -1e-6 - 1e-10 * abs(old_entropy):
            raise ConservationError("Adiabatic layered entropy ledger rejected")
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "max_scaled_residual": float(r.abs().max().cpu()),
            "source_frozen": False,
            "columns": reports,
            "heat_j": [dt * q for q in heat],
            "source_n": [float(s.cpu()) for s in d["sources"]],
            "friction_force_n": [float(s.cpu()) for s in d["friction_force_n"]],
            "gravity_mass_n": gravity_force,
            "source_quadrature_error_n": source_errors,
            "component_residuals_mol": nr,
            "energy_residual_j": er,
            "momentum_residual_kg_m_s": momentum_error,
            "wall_force_n": wall_force,
            "wall_pressure": [[float(p.cpu()) for p in row] for row in d["wall_pressure"]],
            "pressure": [[float(p.cpu()) for p in row] for row in d["pressure"]],
            "wall_fluxes": [
                torch.cat(
                    (
                        y.new_zeros(len(self.fluid.names)),
                        ((weights * fan[0][:, -2]).sum()).reshape(1),
                        y.new_zeros(1),
                    )
                )
                .cpu()
                .tolist()
                for _, _, weights, fan in d["wall_faces"]
            ],
            "shared_fluxes": [
                (w[:, None] * fan[0]).sum(0).cpu().tolist() for _, _, w, fan, _ in d["shared"]
            ],
            "cell_net_rates": [q.cpu().tolist() for q in d["rates"]],
            "boundary_net_rates": external.cpu().tolist(),
            "entropy_change_j_k": entropy_change,
            "incoming_entropy_j_k": dt * incoming_entropy,
            "entropy_production_j_k": production,
            "heat_entropy_validated": not any(heat),
            "face_patch_count": len(d["shared"]),
            "evaluations": evaluations,
            "face_contact_velocity_m_s": [
                fan[2].cpu().tolist()
                for _, _, w, fan, _ in d["shared"]
                if float(w.sum().cpu()) > 1e-14
            ],
            "face_pressure_jump_pa": [
                (tr[1][2] - tr[0][2]).cpu().tolist()
                for _, _, w, _, tr in d["shared"]
                if float(w.sum().cpu()) > 1e-14
            ],
        }

    def step(self, state, dt_s, *, heat_w=None, maximum_topology_retries=2):
        """Try the old topology, then seed *new joint solves* from balance targets."""
        dt = _number(dt_s, "step", positive=True)
        heat = self._heat(heat_w)
        seeds = None
        for attempt in range(maximum_topology_retries + 1):
            trial = []
            try:
                new, report = self._candidate(
                    state, dt, heat_w=heat, seed_overrides=seeds, trial=trial
                )
                report["topology_retries"] = attempt
                report["phase_counts"] = [len(c.profile.phase_roots) for c in new.cells]
                return new, report
            except (HydraulicConvergenceError, PropertyModelError):
                if attempt == maximum_topology_retries or not trial:
                    raise
                seeds = self.predict_seeds(state, dt, trial[0], heat_w=heat)

    def predict_seeds(self, state, dt, data, *, heat_w=None):
        """Balance-target phase seeds for a *new* Newton, never accepted stocks.

        The parent coupled driver may use this on rejected decoded native data.
        Active component support is fixed; adding new components is unsupported.
        """
        heat = self._heat(heat_w)
        seeds = {}
        for i, (col, old, body) in enumerate(zip(self.columns, state.cells, data["bodies"])):
            n = self.fluid.tensor(old.profile.component_moles) + dt * data["rates"][i][:-2]
            active = self.fluid.tensor(old.profile.component_moles) > 0
            if not bool(
                (torch.isfinite(n).all() & (n[active] > 0).all() & (n[~active] == 0).all()).cpu()
            ):
                raise HydraulicConvergenceError(
                    "Topology predictor exhausts or changes active support"
                )
            mass = float((n * self.fluid.molar_masses).sum().cpu())
            momentum = float((data["masses"][i] * data["velocities"][i]).cpu())
            flowmass = float((data["rates"][i][:-2] * self.fluid.molar_masses).sum().cpu())
            energy = (
                old.total_energy_j
                + dt
                * (
                    float(data["rates"][i][-1].cpu())
                    + flowmass * self.gravity * self.datum
                    + heat[i]
                )
                - 0.5 * momentum**2 / mass
            )
            seeds[i] = col._seed(
                n, energy, float(body["pressures"][0][0].cpu()), float(body["temperature"].cpu())
            )
        return seeds

    def _error(self, coarse, fine, *, relative_tolerance, interface_tolerance_m):
        error = 0.0
        for col, a, b in zip(self.columns, coarse.cells, fine.cells):
            mass = self._mass(b.profile)
            for x, y in zip(a.profile.component_moles, b.profile.component_moles):
                error = max(error, abs(x - y) / (1e-10 + relative_tolerance * max(abs(x), abs(y))))
            ea = a.total_energy_j - self._mass(a.profile) * self.gravity * self.datum
            eb = b.total_energy_j - mass * self.gravity * self.datum
            escale = max(
                1.0, abs(ea), abs(eb), sum(b.profile.component_moles) * R * b.profile.temperature_k
            )
            error = max(
                error,
                abs(ea - eb) / (1e-6 + relative_tolerance * escale),
                abs(a.momentum_kg_m_s - b.momentum_kg_m_s) / (1e-9 + relative_tolerance * mass),
            )
            if len(a.profile.phase_roots) != len(b.profile.phase_roots):
                return math.inf
            for x, y in zip(a.profile.interface_heights_m, b.profile.interface_heights_m):
                error = max(error, abs(x - y) / interface_tolerance_m)
        return error

    def advance(
        self,
        state,
        duration_s,
        *,
        heat_w=None,
        maximum_step_s=1.0,
        minimum_step_s=1e-9,
        relative_tolerance=1e-4,
        interface_tolerance_m=1e-5,
        maximum_attempts=1000,
    ):
        """Adaptive coarse/two-half backward Euler; checkpoint stores next step."""
        self.inspect(state)
        duration = _number(duration_s, "duration", positive=True)
        maximum = _number(maximum_step_s, "maximum step", positive=True)
        minimum = _number(minimum_step_s, "minimum step", positive=True)
        _number(relative_tolerance, "relative tolerance", positive=True)
        _number(interface_tolerance_m, "interface tolerance", positive=True)
        target = state.time_s + duration
        dt = min(state.next_step_s, maximum, duration)
        reports = []
        for _ in range(maximum_attempts):
            remaining = target - state.time_s
            if remaining <= max(1e-15, duration * 1e-13):
                return replace(state, time_s=target), reports
            dt = min(dt, remaining, maximum)
            try:
                coarse, _ = self.step(state, dt, heat_w=heat_w)
                half, r1 = self.step(state, dt / 2, heat_w=heat_w)
                fine, r2 = self.step(half, dt / 2, heat_w=heat_w)
                error = self._error(
                    coarse,
                    fine,
                    relative_tolerance=relative_tolerance,
                    interface_tolerance_m=interface_tolerance_m,
                )
            except (HydraulicConvergenceError, PropertyModelError):
                error = math.inf
            if error <= 1:
                factor = min(2.0, max(0.5, 0.9 / max(error, 1e-6) ** 0.5))
                nextdt = min(maximum, max(minimum, dt * factor))
                state = replace(fine, next_step_s=nextdt)
                reports.append(
                    {"dt_s": dt, "error": error, "halves": [r1, r2], "source_frozen": False}
                )
                dt = nextdt
            else:
                dt *= 0.5
                if dt < minimum:
                    raise HydraulicConvergenceError("Adaptive layered step exhausted minimum step")
        raise HydraulicConvergenceError("Adaptive layered step exhausted attempts")

    def joint_vessel_problem(
        self,
        ordinary,
        ordinary_problem,
        state,
        dt_s,
        *,
        boundary_nodes,
        heat_w=None,
        seed_overrides=None,
        maximum_head_fraction=1e-3,
    ):
        """Append this chain to an existing ordinary graph residual, with no solve.

        ``ordinary_problem`` is the tuple returned by ordinary._problem, built
        with external_ports for the connected finite vessels. Returned evaluate
        calls its external_fluxes callback and appends all profile/trace rows.
        Endpoint names (left,right), with None for a closed wall, bind *vessels*
        only. Acceptance, ordinary topology retries and plant ledgers belong to
        the parent graph driver; this method does not publish a coupled state.
        """
        if len(boundary_nodes) != 2:
            raise ValueError("Two end bindings required")
        if (
            ordinary.fluid.model_id != self.fluid.model_id
            or ordinary.fluid.device != self.fluid.device
        ):
            raise ValueError("Bridge EOS/device mismatch")
        if ordinary.gravity_m_s2 != self.gravity:
            raise ValueError("Bridge gravity mismatch")
        guard = _number(maximum_head_fraction, "head guard", positive=True)
        if guard > 0.01:
            raise ValueError("Uniform vessel requires head fraction <= .01")
        indices = []
        for side, name in zip((-1, 1), boundary_nodes):
            if name is None:
                indices.append(None)
                continue
            index = ordinary.names.index(name)
            node = ordinary.nodes[index]
            if node.kind != "vessel" or node.drift_flux is not None:
                raise ValueError(
                    "Bridge accepts ordinary HEM finite vessels only; reject pipe/slip"
                )
            cell = 0 if side == -1 else len(self.columns) - 1
            center = self._disk_bottom(cell, side) + self.diameter * self.cosine / 2
            if abs(node.elevation_m - center) > 1e-8:
                raise ValueError("Vessel datum must match connected disk center")
            indices.append(index)
        py, pl, ph, pe = self._problem(
            state,
            dt_s,
            heat_w=heat_w,
            seed_overrides=seed_overrides,
            external_ends=tuple(i is not None for i in indices),
        )
        oy, ol, oh, oe = ordinary_problem[:4]
        cut = len(oy)
        if cut + len(py) > self.max_unknowns:
            raise ValueError("Coupled problem exceeds max_unknowns")

        def vessel_exchange(y, ordinary_data):
            def exchange(profile_data):
                net = [y.new_zeros(len(self.fluid.names) + 2) for _ in self.columns]
                onet = [y.new_zeros(len(self.fluid.names) + 2) for _ in ordinary.nodes]
                valid = y.new_ones((), dtype=torch.bool)
                reactions = [y.new_zeros(()) for _ in ordinary.nodes]
                records = []
                for patch in profile_data["boundary"]:
                    side, ci = patch["side"], patch["cell"]
                    index = indices[0 if side == -1 else 1]
                    if index is None:
                        continue
                    d = ordinary_data[index]
                    node = ordinary.nodes[index]
                    rho = d["mass"] / node.volume_m3
                    valid = (
                        valid
                        & (rho * self.gravity * self.diameter / d["p"] <= guard)
                        & d["q"]["valid"].all()
                        & (d["q"]["frozen_sound_speed_m_s"] <= self.alpha).all()
                    )
                    v = d["p"] * 0  # supported finite-vessel HEM packet, never drift velocities
                    h, w = patch["heights"], patch["weights"]
                    p = d["p"].expand_as(h)
                    species = (d["n"] / node.volume_m3).expand(len(h), -1)
                    energy = d["internal"] / node.volume_m3 + rho * self.gravity * (h - self.datum)
                    conserved = torch.cat(
                        (species, torch.zeros_like(h)[:, None], energy[:, None]), dim=1
                    )
                    flux = torch.cat(
                        (torch.zeros_like(species), p[:, None], torch.zeros_like(h)[:, None]),
                        dim=1,
                    )
                    vessel = (
                        conserved,
                        flux,
                        p,
                        v.expand_as(h),
                        rho.expand_as(h),
                        (d["entropy"] / node.volume_m3).expand_as(h),
                    )
                    left, right = (
                        (vessel, patch["packet"]) if side == -1 else (patch["packet"], vessel)
                    )
                    fan = torch.vmap(lambda *args: hllc_port(args[:6], args[6:], self.alpha))(
                        *left, *right
                    )
                    flow = (w[:, None] * fan[0]).sum(0) * (-side)
                    net[ci] = net[ci] + flow
                    # Opposite N/E transfer. The vessel support absorbs
                    # momentum explicitly; its stock momentum stays zero.
                    other = -flow
                    massrate = (other[:-2] * self.fluid.molar_masses).sum()
                    energy_other = other[-1] + massrate * self.gravity * (
                        self.datum - ordinary.reference_elevation_m
                    )
                    onet[index] = onet[index] + torch.cat(
                        (other[:-2], other[-2].new_zeros(1), energy_other.reshape(1))
                    )
                    reactions[index] = reactions[index] + other[-2]
                    valid = valid & fan[4].all()
                    records.append(
                        {
                            "cell": ci,
                            "node": index,
                            "side": side,
                            "flow": flow,
                            "weights": w,
                            "fan": fan,
                            "entropy_rate": side * (w * fan[1]).sum(),
                        }
                    )
                ordinary_incoming = sum((rec["entropy_rate"] for rec in records), y.new_zeros(()))
                return {
                    "net": net,
                    "ordinary_net": onet,
                    "valid": valid,
                    "reactions_n": reactions,
                    "records": records,
                    "incoming_entropy": -ordinary_incoming,
                }

            pr, pd = pe(y, external_fluxes=exchange)
            ex = pd["exchange"]
            return {
                "net": ex["ordinary_net"],
                "valid": ex["valid"] & torch.stack(pd["valid"]).all(),
                "incoming_entropy": -ex["incoming_entropy"],
                "profile_residual": pr,
                "profile": pd,
                "vessel_support_force_n": ex["reactions_n"],
            }

        def evaluate(y):
            def callback(ordinary_data):
                return vessel_exchange(y[cut:], ordinary_data)

            r, d = oe(y[:cut], external_fluxes=callback)
            return torch.cat((r, d["exchange"]["profile_residual"])), {
                "ordinary": d,
                "profile": d["exchange"]["profile"],
                "vessel_support_force_n": d["exchange"]["vessel_support_force_n"],
            }

        # Preserve the parent's graph support; only raw standalone callers
        # without metadata use a conservative dense ordinary subblock.
        ordinary_pattern = getattr(oe, "sparsity", None)
        edges = (
            set(zip(*ordinary_pattern))
            if ordinary_pattern is not None
            else {(r, c) for r in range(cut) for c in range(cut)}
        )
        edges.update((r + cut, c + cut) for r, c in zip(*pe.sparsity))
        offsets = ordinary_problem[4]
        layout = self._layout(
            state,
            seed_overrides
            and [
                seed_overrides.get(i, self._seed(c, cell.profile))
                for i, (c, cell) in enumerate(zip(self.columns, state.cells))
            ],
        )
        for side, index in zip((-1, 1), indices):
            if index is None:
                continue
            ci = 0 if side == -1 else len(self.columns) - 1
            native_vars = set(pe.linear_blocks[ci])
            for sl, spec in zip(pe.trace_slices, layout):
                if spec[0] == "local" and spec[1] == ci and spec[2] == side:
                    native_vars.update(range(sl.start, sl.stop))
            ordinary_vars = range(offsets[index], offsets[index + 1])
            edges.update((r, c + cut) for r in ordinary_vars for c in native_vars)
            edges.update((r + cut, c) for r in native_vars for c in ordinary_vars)
        pairs = sorted(edges)
        evaluate.sparsity = ([p[0] for p in pairs], [p[1] for p in pairs])
        evaluate.linear_blocks = list(oe.linear_blocks) + [
            [v + cut for v in block] for block in pe.linear_blocks
        ]
        evaluate.ordinary_size = cut
        evaluate.profile_evaluate = pe
        evaluate.vessel_exchange = vessel_exchange
        evaluate.maximum_update = torch.cat(
            (
                getattr(oe, "maximum_update", oy.new_full(oy.shape, math.inf)),
                py.new_full(py.shape, math.inf),
            )
        )
        return torch.cat((oy, py)), torch.cat((ol, pl)), torch.cat((oh, ph)), evaluate
