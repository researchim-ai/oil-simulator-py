"""Closed multifluid graphs with finite supported mixing junctions.

All phases remain present; no phase transfer, equipment or external
boundary conditions. Junction supports absorb momentum without removing energy.
"""

import copy
import math
from dataclasses import asdict

import torch

from .liquid import ConservationError, PropertyModelError, _number
from .operation import _digest
from .pr_interphase_relaxation import PRInterphaseRelaxation
from .pr_multifluid_flux import path_fluctuations
from .pr_multifluid_pipe import PRMultifluidPipe
from .pr_pump_four_quadrant import SignedPumpMap


class PRMultifluidNetwork(PRMultifluidPipe):
    def __init__(
        self,
        sources,
        volumes_m3,
        cell_names,
        faces,
        supported_nodes,
        *,
        elevations_m=None,
        gravity_m_s2=9.80665,
        hydrostatic_reference=None,
        pumps=(),
        **policy,
    ):
        self.pumps = tuple(copy.deepcopy(pumps))
        self.reference_specs = copy.deepcopy(hydrostatic_reference)
        if self.pumps and self.reference_specs is not None:
            raise ValueError("Driven equipment cannot be subtracted as a hydrostatic reference")
        self._reference_operator = None
        if hydrostatic_reference is not None and elevations_m is None:
            raise ValueError("Hydrostatic reference requires explicit elevations")
        if "boundary_mode" in policy:
            raise ValueError("Network topology replaces pipe boundary modes")
        self.cell_names = tuple(cell_names)
        if not self.cell_names or any(not isinstance(n, str) or not n for n in self.cell_names):
            raise ValueError("Nonempty named cells required")
        if len(set(self.cell_names)) != len(self.cell_names):
            raise ValueError("Duplicate cell names")
        self.elevations = (
            None
            if elevations_m is None
            else tuple(_number(z, "cell elevation") for z in elevations_m)
        )
        self.gravity = _number(gravity_m_s2, "gravity", nonnegative=True)
        if self.elevations is not None and len(self.elevations) != len(self.cell_names):
            raise ValueError("Cell/elevation grid differs")
        self.supported_nodes = tuple(supported_nodes)
        if len(set(self.supported_nodes)) != len(self.supported_nodes) or not set(
            self.supported_nodes
        ) <= set(self.cell_names):
            raise ValueError("Unknown or duplicate supported junction")
        lookup = {n: i for i, n in enumerate(self.cell_names)}
        self.supported_cells = frozenset(lookup[n] for n in self.supported_nodes)
        self.faces = []
        incidence = [[] for _ in self.cell_names]
        self.graph_neighbours = [set() for _ in self.cell_names]
        for face in faces:
            if set(face) != {"source", "target", "area_m2", "direction"}:
                raise ValueError("Explicit face endpoints, area and direction required")
            if face["source"] not in lookup or face["target"] not in lookup:
                raise ValueError("Unknown face endpoint")
            a, b = lookup[face["source"]], lookup[face["target"]]
            direction = tuple(float(x) for x in face["direction"])
            area = _number(face["area_m2"], "face area", positive=True)
            if (
                len(direction) != 3
                or not all(math.isfinite(x) for x in direction)
                or abs(sum(x * x for x in direction) - 1) > 1e-12
            ):
                raise ValueError("Unit face direction required")
            if self.elevations is None and direction[2] != 0:
                raise ValueError("Inclined faces require explicit cell elevations")
            if a == b or (a in self.supported_cells and b in self.supported_cells):
                raise ValueError("Each connection needs an inertial pipe cell")
            self.faces.append(
                {
                    "source": face["source"],
                    "target": face["target"],
                    "area_m2": area,
                    "direction": direction,
                }
            )
            incidence[a].append((len(self.faces) - 1, 0))
            incidence[b].append((len(self.faces) - 1, 1))
            self.graph_neighbours[a].add(b)
            self.graph_neighbours[b].add(a)
        self.cell_directions = []
        for i, ports in enumerate(incidence):
            if i in self.supported_cells:
                if not ports:
                    raise ValueError("Disconnected junction")
                self.cell_directions.append((0.0, 0.0, 0.0))
                continue
            if len(ports) != 2 or {side for _, side in ports} != {0, 1}:
                raise ValueError("Inertial cells require one incoming and one outgoing face")
            f, g = (self.faces[j] for j, _ in ports)
            if f["area_m2"] != g["area_m2"] or f["direction"] != g["direction"]:
                raise ValueError("Turning or branching requires a supported junction")
            self.cell_directions.append(f["direction"])
        volumes_m3 = tuple(_number(v, "cell volume", positive=True) for v in volumes_m3)
        if len(volumes_m3) != len(self.cell_names):
            raise ValueError("Cell/volume grid differs")
        self.face_elevations = []
        for face in self.faces:
            a, b = lookup[face["source"]], lookup[face["target"]]
            za, zb = (
                (0.0, 0.0) if self.elevations is None else (self.elevations[a], self.elevations[b])
            )
            za += (
                0.0
                if a in self.supported_cells
                else 0.5 * volumes_m3[a] / face["area_m2"] * face["direction"][2]
            )
            zb -= (
                0.0
                if b in self.supported_cells
                else 0.5 * volumes_m3[b] / face["area_m2"] * face["direction"][2]
            )
            if not math.isclose(za, zb, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("Face elevations disagree with cell volumes and inclination")
            self.face_elevations.append(0.5 * (za + zb))
        self._incidence = incidence
        # Unit reference area normalizes the control-volume residual. Actual
        # physical face areas multiply every fluctuation in the graph operator.
        super().__init__(sources, volumes_m3, 2 / math.sqrt(math.pi), **policy)
        if len(self.cell_names) != len(self.sources):
            raise ValueError("Named cell/source grid differs")
        self.area = 1.0
        self.volumes = self.lengths
        self.model_id = _digest(self.to_dict())
        device = self.fluid.device
        self._left = torch.tensor([lookup[f["source"]] for f in self.faces], device=device)
        self._right = torch.tensor([lookup[f["target"]] for f in self.faces], device=device)
        self._areas = torch.tensor(
            [f["area_m2"] for f in self.faces], dtype=torch.float64, device=device
        )
        self._directions = torch.tensor(
            [f["direction"] for f in self.faces], dtype=torch.float64, device=device
        )
        width = max(map(len, incidence))
        self._gather = torch.tensor(
            [
                [2 * j + side for j, side in ports] + [2 * len(self.faces)] * (width - len(ports))
                for ports in incidence
            ],
            device=device,
        )
        self._supported_left = torch.tensor(
            [lookup[f["source"]] in self.supported_cells for f in self.faces], device=device
        )
        self._supported_right = torch.tensor(
            [lookup[f["target"]] in self.supported_cells for f in self.faces], device=device
        )
        self._cell_directions = self.fluid.tensor(self.cell_directions)
        nc = len(self.fluid.names)
        self._energy_slot = self.fluid.tensor([0.0] * (nc + 1) + [1.0])
        self._momentum_slot = self.fluid.tensor([0.0] * nc + [1.0, 0.0])
        self._port_dz = self.fluid.tensor(
            [
                [
                    0.0
                    if f["source"] in self.supported_nodes
                    else 0.5 * self.volumes[lookup[f["source"]]] / f["area_m2"] * f["direction"][2],
                    0.0
                    if f["target"] in self.supported_nodes
                    else 0.5 * self.volumes[lookup[f["target"]]] / f["area_m2"] * f["direction"][2],
                ]
                for f in self.faces
            ]
        )

        self._pumps = []
        names = set()
        for pump in self.pumps:
            if (
                set(pump)
                != {"name", "cell", "phase", "map", "speed_rad_s", "minimum_density_kg_m3"}
                or not isinstance(pump["name"], str)
                or not pump["name"]
                or pump["name"] in names
            ):
                raise ValueError(
                    "Unique named pump with explicit cell, phase, map and speed required"
                )
            names.add(pump["name"])
            if pump["cell"] not in lookup or lookup[pump["cell"]] in self.supported_cells:
                raise ValueError("Pump needs an inertial cell")
            k = pump["phase"]
            if (
                type(k) is not int
                or not 0 <= k < len(self.phase_roots)
                or self.phase_roots[k] != "liquid"
            ):
                raise ValueError("Phase-selective pump needs an explicit liquid branch")
            mapping = SignedPumpMap(**pump["map"])
            speed = _number(pump["speed_rad_s"], "pump speed")
            if abs(speed) > mapping.maximum_speed_rad_s:
                raise ValueError("Pump speed outside map")
            i = lookup[pump["cell"]]
            area = next(f["area_m2"] for f in self.faces if f["source"] == pump["cell"])
            selector = self.fluid.tensor(
                [
                    [float(a == i and b == k) for b in range(len(self.phase_roots))]
                    for a in range(len(self.sources))
                ]
            )
            minimum_density = _number(
                pump["minimum_density_kg_m3"], "pump minimum density", positive=True
            )
            self._pumps.append((i, k, mapping, speed, area, selector, minimum_density))
        if self.reference_specs is not None:
            self._prepare_hydrostatic_reference()

    def _prepare_hydrostatic_reference(self):
        reference = self.initialize(self.reference_specs)
        arrays = self._arrays(reference)
        fraction = arrays["stored_volume"] / self.fluid.tensor(self.volumes)[:, None]
        composition = arrays["n"] / arrays["n"].sum(-1, keepdim=True)
        temperature = arrays["temperature"]
        mass = arrays["mass"]
        gibbs = (
            arrays["internal"]
            + arrays["pressure"] * arrays["stored_volume"]
            - temperature * arrays["entropy"]
        ) / mass
        potential = self.fluid.tensor(
            [self.gravity * (z - self.elevations[0]) for z in self.elevations]
        )[:, None]
        balance = gibbs + potential
        valid = (
            (arrays["velocity"] == 0).all()
            & ((fraction - fraction[:1]).abs() < 1e-10).all()
            & ((composition - composition[:1]).abs() < 1e-10).all()
            & ((temperature - temperature[:1]).abs() < 1e-8).all()
            & ((balance - balance[:1]).abs() < 1e-5).all()
        )
        if not bool(valid.cpu()):
            raise ValueError(
                "Reference must be isothermal, constant-composition/fraction PR hydrostatics at rest"
            )
        for source, cell in zip(self.sources, reference.cells):
            if any(
                abs(cell.phases[i].pressure_pa - cell.phases[j].pressure_pa) > 1e-6
                for i, j, _ in source.pressure_edges
            ):
                raise ValueError("Reference is not stationary under pressure exchange")
            if any(
                abs(cell.phases[i].temperature_k - cell.phases[j].temperature_k) > 1e-8
                for i, j, _ in source.thermal_edges
            ):
                raise ValueError("Reference is not stationary under heat exchange")
        y, _, _, evaluate = self._problem(reference, 1.0)
        with torch.no_grad():
            _, data = evaluate(y)
            if not bool(data["valid"].cpu()):
                raise ValueError("Reference leaves the reconstructed port domain")
            self._reference_operator = (
                data["incoming"].detach(),
                data["wall_impulse"].detach(),
                {k: v.detach() for k, v in data["faces"].items()},
            )

    def to_dict(self):
        result = {
            "model": "pr_multifluid_network_v1",
            "sources": [s.to_dict() for s in self.sources],
            "volumes_m3": self.lengths,
            "cell_names": self.cell_names,
            "faces": self.faces,
            "supported_nodes": self.supported_nodes,
            "wave_speed_bound_m_s": self.wave_speed,
            "minimum_phase_fraction": self.minimum_phase_fraction,
            "dense_limit": self.dense_limit,
            "maximum_unknowns": self.maximum_unknowns,
        }
        if self.elevations is not None:
            result.update(
                model="pr_multifluid_network_v2",
                elevations_m=self.elevations,
                gravity_m_s2=self.gravity,
            )
        if self.reference_specs is not None:
            result.update(
                model="pr_multifluid_network_v3", hydrostatic_reference=self.reference_specs
            )
        if self.pumps:
            result.update(
                model="pr_multifluid_network_v4",
                pumps=[{**p, "map": asdict(SignedPumpMap(**p["map"]))} for p in self.pumps],
            )
        return result

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        raw = dict(data)
        version = raw.pop("model", None)
        if version not in (
            "pr_multifluid_network_v1",
            "pr_multifluid_network_v2",
            "pr_multifluid_network_v3",
            "pr_multifluid_network_v4",
        ):
            raise ValueError("Unknown multifluid network")
        if version == "pr_multifluid_network_v1" and (
            "elevations_m" in raw or "gravity_m_s2" in raw
        ):
            raise ValueError("Gravity geometry requires v2 model")
        if version in ("pr_multifluid_network_v2", "pr_multifluid_network_v3") and (
            not {"elevations_m", "gravity_m_s2"} <= raw.keys() or raw["elevations_m"] is None
        ):
            raise ValueError("Gravity model requires elevations and acceleration")
        if (version == "pr_multifluid_network_v3") != (
            raw.get("hydrostatic_reference") is not None
        ):
            raise ValueError("Hydrostatic reference requires explicit v3 schema")
        if (version == "pr_multifluid_network_v4") != bool(raw.get("pumps")):
            raise ValueError("Driven equipment requires explicit v4 schema")
        raw["sources"] = [
            PRInterphaseRelaxation.from_dict(s, device=device) for s in raw["sources"]
        ]
        return cls(**raw)

    def inspect(self, state):
        result = super().inspect(state)
        if any(p.momentum_kg_m_s != 0 for i in self.supported_cells for p in state.cells[i].phases):
            raise PropertyModelError("Supported junction velocities must be zero")
        return result

    def _network_operator(self, d, part, dt):
        left, right = part(d, self._left), part(d, self._right)
        minus, plus, face = path_fluctuations(left, right, self.wave_speed)
        nph, nc = len(self.phase_roots), len(self.fluid.names)
        if self.elevations is not None:
            # Reconstruct hydrostatic pressure traction to the actual ports.
            # Junction centres lie at their ports; inertial centres lie halfway
            # along finite segments. Ignoring these half lengths gives a wrong
            # gravity/pressure balance at a finite vessel interface.
            rho_left = (left["conserved"][..., :nc] * self.fluid.molar_masses).sum(-1)
            rho_right = (right["conserved"][..., :nc] * self.fluid.molar_masses).sum(-1)
            delta_left = -self.gravity * rho_left * self._port_dz[:, :1]
            delta_right = self.gravity * rho_right * self._port_dz[:, 1:]
            traction = 0.5 * (delta_left + delta_right)
            velocity = 0.5 * (left["velocity"] + right["velocity"])
            correction = (
                traction[..., None] * self._momentum_slot
                + (traction * velocity)[..., None] * self._energy_slot
            )
            extra = torch.cat(
                (correction.flatten(1), torch.zeros_like(minus[:, nph * (nc + 2) :])), dim=-1
            )
            minus, plus = minus + extra, plus - extra
            face = {**face, "flux": face["flux"] + correction}
            pleft = left["pressure"] + delta_left / left["alpha"]
            pright = right["pressure"] + delta_right / right["alpha"]
            d["valid"] = (
                d["valid"]
                & (pleft >= self.fluid.min_pressure_pa).all()
                & (pright >= self.fluid.min_pressure_pa).all()
                & (pleft <= self.fluid.max_pressure_pa).all()
                & (pright <= self.fluid.max_pressure_pa).all()
            )
        weighted = torch.stack((minus, plus), dim=1) * self._areas[:, None, None]
        phase_weighted = weighted[..., : nph * (nc + 2)].reshape(-1, 2, nph, nc + 2)
        mm = phase_weighted[..., -2].sum(-1)
        reaction = (mm[:, 0] * self._supported_left + mm[:, 1] * self._supported_right)[
            :, None
        ] * self._directions
        geometry = (
            -self._areas[:, None]
            * (right["flux"][..., -2].sum(-1) - left["flux"][..., -2].sum(-1))[:, None]
            * self._directions
        )
        if self.elevations is not None:
            # The same numerical species flux transports gravitational potential.
            # This includes Rusanov diffusion; using rho*u here would lose energy.
            mass_flux = (face["flux"][..., :nc] * self.fluid.molar_masses).sum(-1)
            work = (
                self.gravity
                * self._areas[:, None, None]
                * self._port_dz[:, :, None]
                * mass_flux[:, None, :]
            )
            phase_weighted = phase_weighted + work[..., None] * self._energy_slot
            weighted = torch.cat(
                (phase_weighted.flatten(2), weighted[..., nph * (nc + 2) :]), dim=-1
            )
        flat = weighted.flatten(0, 1)
        padded = torch.cat((flat, flat.new_zeros((1, flat.shape[1]))))
        incoming = padded[self._gather].sum(1)
        impulse = dt * (reaction + geometry).sum(0)
        if self.elevations is not None:
            mass = (d["n"] * self.fluid.molar_masses).sum(-1)
            gravity_load = self.gravity * mass * self._cell_directions[:, 2:3]
            phase = incoming[:, : nph * (nc + 2)].reshape(-1, nph, nc + 2)
            phase = phase + gravity_load[..., None] * self._momentum_slot
            incoming = torch.cat((phase.flatten(1), incoming[:, nph * (nc + 2) :]), dim=-1)
            # Includes wall-normal support reactions; supported junction weights
            # cancel at their supports. Only the tangent load accelerates fluid.
            impulse = impulse - dt * (gravity_load.sum(-1)[:, None] * self._cell_directions).sum(0)
        if self._reference_operator is not None:
            reference_rate, reference_impulse, reference_face = self._reference_operator
            incoming = incoming - reference_rate
            impulse = impulse - dt * reference_impulse
            face = {k: v - reference_face[k] for k, v in face.items()}
        work = impulse.new_zeros(())
        for i, k, mapping, speed, area, selector, minimum_density in self._pumps:
            rho = (d["n"][i, k] * self.fluid.molar_masses).sum() / (
                d["alpha"][i, k] * self.volumes[i]
            )
            flow = d["alpha"][i, k] * area * d["velocity"][i, k]
            head, _, shaft, _, _, valid = mapping.evaluate(flow, speed, rho)
            force = rho * 9.80665 * head * d["alpha"][i, k] * area
            phase = incoming[:, : nph * (nc + 2)].reshape(-1, nph, nc + 2)
            phase = phase - selector[..., None] * (
                force * self._momentum_slot + shaft * self._energy_slot
            )
            incoming = torch.cat((phase.flatten(1), incoming[:, nph * (nc + 2) :]), dim=-1)
            impulse = impulse + dt * force * self._cell_directions[i]
            work = work + dt * shaft
            d["valid"] = d["valid"] & valid & (rho >= minimum_density)
        d["external_work_j"] = work
        return incoming, face, impulse

    def _stored_energy(self, state):
        if self.elevations is None:
            return super()._stored_energy(state)
        mw = tuple(c.molar_mass_kg_mol for c in self.fluid.components)
        return math.fsum(
            p.total_energy_j
            + self.gravity
            * (z - self.elevations[0])
            * math.fsum(n * w for n, w in zip(p.component_moles, mw))
            for cell, z in zip(state.cells, self.elevations)
            for p in cell.phases
        )

    def _sum_impulses(self, values):
        values = list(values)
        return [math.fsum(v[j] for v in values) for j in range(3)]

    def _combine_impulses(self, transport, reports):
        return self._sum_impulses(
            [transport]
            + [
                [r["wall_impulse_kg_m_s"] * x for x in direction]
                for r, direction in zip(reports, self.cell_directions)
            ]
        )

    def ledger(self, old, new, wall_impulse, external_work_j=0.0):
        # Reuse strict component, energy and phase-stock checks with the scalar
        # momentum difference removed; the graph checks physical vector momentum.
        scalar = math.fsum(p.momentum_kg_m_s for c in new.cells for p in c.phases) - math.fsum(
            p.momentum_kg_m_s for c in old.cells for p in c.phases
        )
        result = super().ledger(old, new, scalar, external_work_j)[:-1]
        vector = [
            math.fsum(
                (
                    math.fsum(p.momentum_kg_m_s for p in b.phases)
                    - math.fsum(p.momentum_kg_m_s for p in a.phases)
                )
                * direction[j]
                for a, b, direction in zip(old.cells, new.cells, self.cell_directions)
            )
            - wall_impulse[j]
            for j in range(3)
        ]
        if max(map(abs, vector)) > 1e-8:
            raise ConservationError("Multifluid graph vector momentum ledger failed")
        return result + vector
