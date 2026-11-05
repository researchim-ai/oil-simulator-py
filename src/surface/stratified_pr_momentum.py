"""Monolithic gravitational PR columns / horizontal thermal Euler pipe.

Single-phase columns in sedimentation equilibrium; centre-height, finite-area
1-D ports. A reconstructed inclined pipe can connect different elevations.
No resolved transverse port profile or free interface.
All energies use the same PR reference and explicit gravitational datum.
"""

import math
from dataclasses import asdict, dataclass

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .peng_robinson import R
from .pr_inclined_pipe import PRInclinedMomentumPipe
from .pr_momentum_thermal import PRThermalMomentumPipe, PRThermalMomentumState
from .pr_momentum_vessels import PRMomentumVesselNetwork, hllc_port
from .stratified_pr import StratifiedPRColumn
from .stratified_pr_thermal import StratifiedThermalState
from .torch_linear import TorchNewton
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class StratifiedMomentumState:
    pipe: PRThermalMomentumState
    vessels: tuple
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "vessels", tuple(self.vessels))

    @property
    def time_s(self):
        return self.pipe.time_s

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Unknown stratified momentum state fields")
        return cls(
            PRThermalMomentumState.from_dict(data["pipe"]),
            tuple(StratifiedThermalState.from_dict(v) for v in data["vessels"]),
            data["model_id"],
        )


class StratifiedPRMomentumNetwork(PRMomentumVesselNetwork):
    """Two equilibrium columns with one straight inertial pipe.

    Reuses adaptive interval integration, not the homogeneous vessel closure.
    Port heights refer to column bases and must match pipe endpoint elevations.
    """

    def __init__(
        self, pipe, *, columns, vessel_walls, port_heights_m, maximum_transverse_head_fraction=1e-3
    ):
        if not isinstance(pipe, PRThermalMomentumPipe) or pipe.pipe.boundary_mode != "ends":
            raise TypeError("Thermal momentum pipe with end boundaries required")
        if len(columns) != 2 or any(not isinstance(c, StratifiedPRColumn) for c in columns):
            raise TypeError("Two StratifiedPRColumns required")
        if len(vessel_walls) != 2 or any(not isinstance(w, VesselWall) for w in vessel_walls):
            raise TypeError("Two VesselWalls required")
        if len(port_heights_m) != 2:
            raise ValueError("Two port heights required")
        self.pipe, self.columns, self.walls = pipe, tuple(columns), tuple(vessel_walls)
        self.heights = tuple(_number(h, "port height", nonnegative=True) for h in port_heights_m)
        self.maximum_transverse_head_fraction = _number(
            maximum_transverse_head_fraction, "transverse head fraction", positive=True
        )
        if maximum_transverse_head_fraction > 0.01:
            raise ValueError("Centre-height port approximation requires head fraction <= 0.01")
        p, f = pipe.pipe, pipe.pipe.fluid
        self.gravity = self.columns[0].gravity_m_s2
        self.elevation = self.columns[0].elevation_m + self.heights[0]
        self.end_elevations = (self.elevation, self.elevation)
        if isinstance(p, PRInclinedMomentumPipe):
            if (
                p.gravity_m_s2 != self.gravity
                or abs(p.reference_elevation_m - self.elevation) > 1e-9
            ):
                raise ValueError("Inclined pipe datum or gravity differs from columns")
            self.end_elevations = (self.elevation, self.elevation + p.elevation_change_m)
        for c, h, endpoint in zip(self.columns, self.heights, self.end_elevations):
            if c.fluid.model_id != f.model_id or c.fluid.device != f.device:
                raise PropertyModelError("Column/pipe EOS or device differs")
            if c.gravity_m_s2 != self.gravity or abs(c.elevation_m + h - endpoint) > 1e-9:
                raise ValueError(
                    "Pipe requires equal absolute elevations at each endpoint and common gravity"
                )
            if h < p.diameter_m / 2 or h + p.diameter_m / 2 > c.geometry.height_m:
                raise ValueError("Port does not fit within column height")
            if p.diameter_m > min(c.geometry.diameter_m, c.geometry.length_m):
                raise ValueError("Port diameter exceeds available column cross-section")
        self.start_order = max(c.quadrature_order for c in self.columns)
        self.max_order = min(c.maximum_quadrature_order for c in self.columns)
        if self.start_order >= self.max_order:
            raise ValueError("Common quadrature refinement range required")
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "stratified_pr_momentum_v1",
            "pipe": self.pipe.to_dict(),
            "columns": [c.to_dict() for c in self.columns],
            "vessel_walls": [asdict(w) for w in self.walls],
            "port_heights_m": self.heights,
            "maximum_transverse_head_fraction": self.maximum_transverse_head_fraction,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "stratified_pr_momentum_v1":
            raise ValueError("Unknown stratified momentum model")
        data["pipe"] = PRThermalMomentumPipe.from_dict(data["pipe"], device=device)
        data["columns"] = [StratifiedPRColumn.from_dict(c, device=device) for c in data["columns"]]
        data["vessel_walls"] = [VesselWall(**w) for w in data["vessel_walls"]]
        return cls(**data)

    def initialize(self, pipe_state, vessel_states):
        states = tuple(vessel_states)
        if len(states) != 2:
            raise ValueError("Two column states required")
        state = StratifiedMomentumState(pipe_state, states, self.model_id)
        self.inspect(state)
        return state

    def initialize_equilibrium(
        self, pressure_pa, temperature_k, mole_fractions, **wall_temperatures
    ):
        from .pr_hydrostatic_initialization import initialize_network_equilibrium

        return initialize_network_equilibrium(
            self, pressure_pa, temperature_k, mole_fractions, **wall_temperatures
        )

    def inspect(self, state):
        if (
            not isinstance(state, StratifiedMomentumState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Stratified momentum model/state mismatch")
        self.pipe.inspect(state.pipe)
        support = [n > 0 for n in state.pipe.pipe.component_moles[0]]
        if len(state.vessels) != 2:
            raise ValueError("Two column states required")
        for s, c, w in zip(state.vessels, self.columns, self.walls):
            if not isinstance(s, StratifiedThermalState):
                raise TypeError("StratifiedThermalState required")
            c.inspect(s.column)
            if len(s.column.phase_roots) != 1:
                raise PropertyModelError("Inertial column ports currently require one stable phase")
            if [n > 0 for n in s.column.component_moles] != support:
                raise PropertyModelError("Column/pipe active component support differs")
            if (
                s.model_id != self.model_id
                or abs(s.time_s - state.time_s) > 1e-12
                or s.next_step_s != state.pipe.pipe.next_step_s
            ):
                raise PropertyModelError("Column clock or coupled identity differs")
            if (
                not w.min_temperature_k <= s.wall_temperature_k <= w.max_temperature_k
                or not math.isfinite(s.wall_temperature_k)
            ):
                raise PropertyModelError("Column wall outside domain")

    def initial_vessel_state(
        self, index, column_state, wall_temperature_k, *, time_s=0.0, next_step_s=None
    ):
        """Attach an existing conserved gravitational profile without reinitializing it."""
        _number(wall_temperature_k, "column wall temperature", positive=True)
        if isinstance(index, bool) or index not in (0, 1):
            raise ValueError("Column index must be 0 or 1")
        self.columns[index].inspect(column_state)
        return StratifiedThermalState(
            column_state,
            wall_temperature_k,
            time_s,
            self.pipe.pipe.initial_step_s if next_step_s is None else next_step_s,
            self.model_id,
        )

    @staticmethod
    def _seed(column, state):
        f = column.fluid
        return (
            [f.tensor(x) for x in state.phase_seeds],
            list(state.phase_roots),
            f.tensor(state.interface_heights_m),
            math.exp(state.solution_variables[0]),
            state.temperature_k,
            f.tensor(state.chemical_constants),
        )

    def _problem(self, state, dt, openings, order):
        p, f = self.pipe.pipe, self.pipe.pipe.fluid
        active = p._support(state.pipe.pipe.component_moles)
        ccount = len(active)
        problems = []
        sizes = []
        seeds = []
        ns = []
        nscales = []
        escales = []
        port_initial = []
        for col, s, h in zip(self.columns, state.vessels, self.heights):
            old = s.column
            n = f.tensor(old.component_moles)
            seed = self._seed(col, old)
            problem = col._problem(n, old.total_energy_j, seed, order)
            problems.append(problem)
            sizes.append(len(problem[0]))
            seeds.append(seed)
            ns.append(n)
            nscales.append(n[active].clamp_min(n.sum() * 1e-12))
            relative = (
                old.total_energy_j
                - float((n * f.molar_masses).sum().cpu()) * self.gravity * col.elevation_m
            )
            escales.append(max(1, abs(relative), float(n.sum().cpu()) * R * old.temperature_k))
            sample = col.sample(old, h)
            x = f.tensor(sample["mole_fractions"])[active]
            port_initial.append(
                torch.cat((f.tensor([sample["pressure_pa"]]).log(), (x[:-1] / x[-1]).log()))
            )
        k = len(state.pipe.pipe.solution_variables)
        offsets = (k, k + sizes[0], k + sum(sizes))

        def faces(pd, prepared, ports):
            ff, ss = p._faces(pd, (None, None))
            ends = []
            send = []
            details = []
            for i, j in enumerate((0, -1)):
                if not openings[i]:
                    port = (
                        ff[j],
                        ss[j],
                        pd["p"].new_zeros(()),
                        pd["p"][j],
                        pd["p"].new_ones((), dtype=torch.bool),
                    )
                else:
                    rd = ports[i]
                    q = rd["q"]
                    pp = rd["p"]
                    entropy = q["molar_entropy_j_mol_k"] / q["molar_volume_m3_mol"]
                    end, endpoint_index = p._port_data(pd, i)
                    potential = (
                        q["density_kg_m3"]
                        * self.gravity
                        * (self.end_elevations[i] - self.elevation)
                    )
                    u = torch.cat(
                        (
                            rd["x"] / q["molar_volume_m3_mol"],
                            pp.new_zeros(1),
                            (
                                q["molar_internal_energy_j_mol"] / q["molar_volume_m3_mol"]
                                + potential
                            ).reshape(1),
                        )
                    )
                    flux = torch.cat((torch.zeros_like(u[:-2]), pp.reshape(1), pp.new_zeros(1)))
                    vessel = (u, flux, pp, pp.new_zeros(()), q["density_kg_m3"], entropy)
                    pipe = (
                        end["u"][endpoint_index],
                        end["flux"][endpoint_index],
                        end["p"][endpoint_index],
                        end["v"][endpoint_index],
                        end["q"]["density_kg_m3"][endpoint_index],
                        end["entropy"][endpoint_index],
                    )
                    port = (
                        hllc_port(vessel, pipe, p.wave_speed_bound_m_s)
                        if i == 0
                        else hllc_port(pipe, vessel, p.wave_speed_bound_m_s)
                    )
                ends.append(port[0])
                send.append(port[1])
                details.append(port)
            return (
                torch.cat((ends[0][None], ff[1:-1], ends[1][None])),
                torch.cat((send[0].reshape(1), ss[1:-1], send[1].reshape(1))),
                details,
            )

        py, pl, ph, pe = self.pipe._problem(state.pipe, dt, (None, None), face_operator=faces)
        portlo = f.tensor([math.log(f.min_pressure_pa)] + [-100.0] * (ccount - 1)).repeat(2)
        porthi = f.tensor([math.log(f.max_pressure_pa)] + [100.0] * (ccount - 1)).repeat(2)
        initial_bodies = [
            f.tensor(s.column.solution_variables)
            if s.column.quadrature_order == order
            else problem[0]
            for s, problem in zip(state.vessels, problems)
        ]

        def evaluate(y):
            body = []
            raw = []
            ports = []
            chemical = []
            local = y[offsets[2] :].reshape(2, ccount)
            for i, (col, h, problem) in enumerate(zip(self.columns, self.heights, problems)):
                r, d = problem[3](y[offsets[i] : offsets[i + 1]])
                pp, xx, qq = col._decode_nodes(
                    local[i : i + 1], active, d["temperature"], seeds[i][1][0]
                )
                q = {key: value[0] for key, value in qq.items()}
                pressure = pp[0]
                x = xx[0]
                mu = (
                    torch.log(x[active].clamp_min(1e-300))
                    + q["log_fugacity_coefficients"][active]
                    + torch.log(pressure / f.pref)
                )
                chemical.append(
                    mu
                    + f.molar_masses[active] * self.gravity * h / (R * d["temperature"])
                    - d["constants"]
                )
                ports.append({"p": pressure, "x": x, "t": d["temperature"], "q": q})
                raw.append(r)
                body.append(d)
            pr, result = pe(y[:k], ports)
            face = result["face_details"]
            residuals = []
            targets = []
            walls = []
            heats = []
            ambients = []
            for i, (s, col, w, h) in enumerate(
                zip(state.vessels, self.columns, self.walls, self.heights)
            ):
                flow = (
                    face[i][0] * p.area * (-1 if i == 0 else 1)
                    if openings[i]
                    else torch.zeros_like(face[i][0])
                )
                mn = (flow[:-2] * f.molar_masses).sum()
                tw = (
                    w.heat_capacity_j_k * s.wall_temperature_k
                    + dt
                    * (
                        w.fluid_ua_w_k * body[i]["temperature"]
                        + w.ambient_ua_w_k * w.ambient_temperature_k
                    )
                ) / (w.heat_capacity_j_k + dt * (w.fluid_ua_w_k + w.ambient_ua_w_k))
                heat = dt * w.fluid_ua_w_k * (tw - body[i]["temperature"])
                ambient = dt * w.ambient_ua_w_k * (w.ambient_temperature_k - tw)
                n = ns[i] + dt * flow[:-2]
                energy = (
                    s.column.total_energy_j
                    + dt * (flow[-1] + mn * self.gravity * self.elevation)
                    + heat
                )
                r = raw[i]
                mass = r[-ccount - 1 : -1] - dt * flow[active] / nscales[i]
                energy_r = (
                    r[-1]
                    - (
                        dt * (flow[-1] + mn * self.gravity * (self.elevation - col.elevation_m))
                        + heat
                    )
                    / escales[i]
                )
                residuals.append(torch.cat((r[: -ccount - 1], mass, energy_r.reshape(1))))
                targets.append((n, energy))
                walls.append(tw)
                heats.append(heat)
                ambients.append(ambient)
            return torch.cat((pr, *residuals, *chemical)), {
                "pipe": result,
                "bodies": body,
                "ports": ports,
                "faces": face,
                "targets": targets,
                "walls": walls,
                "heats": heats,
                "ambients": ambients,
            }

        return (
            torch.cat((py, *initial_bodies, *port_initial)),
            torch.cat((pl, *(q[1] for q in problems), portlo)),
            torch.cat((ph, *(q[2] for q in problems), porthi)),
            evaluate,
            offsets,
            seeds,
            active,
        )

    def _candidate(self, state, dt, openings, order):
        p, f = self.pipe.pipe, self.pipe.pipe.fluid
        y, lo, hi, evaluate, offsets, seeds, active = self._problem(state, dt, openings, order)
        if len(y) > 768:
            raise ValueError("Stratified momentum joint Newton exceeds 768 unknowns")
        solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=768)
        solver.host_convergence = True
        with torch.no_grad():
            y, _, ok, evaluations = solver.solve(
                lambda z: evaluate(z)[0], y, lo, hi, tolerance=3e-12
            )
            r, d = evaluate(y)
        if not bool((ok & torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("Gravitational column/momentum Newton rejected")
        for i, port in enumerate(d["ports"]):
            q = port["q"]
            if not bool((q["valid"] & d["faces"][i][4]).cpu()):
                raise PropertyModelError("Column port EOS or HLLC fan rejected")
            sound = p._stable(float(port["p"].cpu()), float(port["t"].cpu()), port["x"])
            if sound > p.wave_speed_bound_m_s:
                raise PropertyModelError("Column port acoustic bound exceeded")
            fraction = float((q["density_kg_m3"] * self.gravity * p.diameter_m / port["p"]).cpu())
            if fraction > self.maximum_transverse_head_fraction:
                raise PropertyModelError("Port requires a resolved transverse hydrostatic model")
        pipe, report = self.pipe._accept_step(
            state.pipe, dt, (None, None), y[: offsets[0]], r, d["pipe"], evaluations
        )
        vessels = []
        column_reports = []
        for i, (col, old, w) in enumerate(zip(self.columns, state.vessels, self.walls)):
            n, e = d["targets"][i]
            data = d["bodies"][i]
            if not bool(
                (torch.isfinite(n).all() & torch.isfinite(e) & (n[active] > 0).all()).cpu()
            ):
                raise HydraulicConvergenceError("Gravitational column exhausts component")
            constants = torch.zeros_like(n).index_copy(0, active, data["constants"])
            seed = (
                [data["compositions"][0][0]],
                seeds[i][1],
                data["interfaces"],
                float(data["pressures"][0][0].cpu()),
                float(data["temperature"].cpu()),
                constants,
            )
            candidate, cr = col._accept(
                n,
                float(e.cpu()),
                seed,
                order,
                y[offsets[i] : offsets[i + 1]],
                evaluations=evaluations,
            )
            tw = float(d["walls"][i].cpu())
            if (
                not w.min_temperature_k <= tw <= w.max_temperature_k
                or abs(tw - old.wall_temperature_k) > w.max_temperature_change_k
            ):
                raise HydraulicConvergenceError("Gravitational column wall domain/step exceeded")
            vessels.append(
                StratifiedThermalState(
                    candidate, tw, pipe.time_s, pipe.pipe.next_step_s, self.model_id
                )
            )
            column_reports.append(cr)
        new = StratifiedMomentumState(pipe, vessels, self.model_id)
        self.inspect(new)
        residual, ambient = self._ledger(state, new, [report], d["ambients"])
        entropy = report["entropy_production_j_k"]
        for i, (old, final, col, w) in enumerate(
            zip(state.vessels, new.vessels, self.columns, self.walls)
        ):
            seed = self._seed(col, old.column)
            olddata = col._problem(
                f.tensor(old.column.component_moles),
                old.column.total_energy_j,
                seed,
                old.column.quadrature_order,
            )[3](f.tensor(old.column.solution_variables))[1]
            entropy += float((d["bodies"][i]["entropy_j_k"] - olddata["entropy_j_k"]).cpu())
            entropy += (
                w.heat_capacity_j_k * math.log(final.wall_temperature_k / old.wall_temperature_k)
                - float(d["ambients"][i].cpu()) / w.ambient_temperature_k
            )
            if openings[i]:
                entropy += float((dt * p.area * d["faces"][i][1] * (1 if i == 0 else -1)).cpu())
        # Roundoff allowance for entropy differences of finite equilibrium columns.
        if entropy < -1e-7:
            raise ConservationError("Gravitational column/pipe entropy rejected")
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "pipe": report,
            "columns": column_reports,
            "quadrature_order": order,
            "vessel_ambient_heat_j": [float(q.cpu()) for q in d["ambients"]],
            "port_pressure_pa": [float(q["p"].cpu()) for q in d["ports"]],
            "port_mole_fractions": [q["x"].cpu().tolist() for q in d["ports"]],
            "port_contact_velocity_m_s": [float(q[2].cpu()) for q in d["faces"]],
            "vessel_support_impulse_kg_m_s": [
                float((dt * p.area * q[0][-2] * (1 if i == 0 else -1)).cpu())
                if openings[i]
                else 0.0
                for i, q in enumerate(d["faces"])
            ],
            "port_energy_rates_with_pe_w": [
                float(
                    (
                        p.area
                        * (
                            q[0][-1]
                            + (q[0][:-2] * f.molar_masses).sum() * self.gravity * self.elevation
                        )
                    ).cpu()
                )
                if openings[i]
                else 0.0
                for i, q in enumerate(d["faces"])
            ],
            "conservation_residuals": residual,
            "ambient_heat_j": ambient,
            "entropy_production_j_k": entropy,
            "max_scaled_residual": float(r.abs().max().cpu()),
        }

    def _quadrature_error(self, a, b, ra, rb):
        error = self.pipe.pipe._error(a.pipe.pipe, b.pipe.pipe)
        for i, (x, y, c) in enumerate(zip(a.vessels, b.vessels, self.columns)):
            side = "left_flux" if i == 0 else "right_flux"
            # Small dt must not hide an unresolved port rate behind tiny stock changes.
            for j, (u, v) in enumerate(zip(ra["pipe"][side], rb["pipe"][side])):
                floor = 1e-10 if j < len(self.pipe.pipe.fluid.names) else 1e-6
                error = max(
                    error, abs(u - v) / (floor + c.relative_tolerance * max(abs(u), abs(v)))
                )
            error = max(
                error,
                abs(ra["port_contact_velocity_m_s"][i] - rb["port_contact_velocity_m_s"][i])
                / (
                    1e-7
                    + c.relative_tolerance
                    * max(
                        abs(ra["port_contact_velocity_m_s"][i]),
                        abs(rb["port_contact_velocity_m_s"][i]),
                    )
                ),
                abs(x.wall_temperature_k - y.wall_temperature_k)
                / self.pipe.pipe.temperature_tolerance_k,
                abs(x.column.temperature_k - y.column.temperature_k)
                / (c.temperature_tolerance_k + c.relative_tolerance * y.column.temperature_k),
                abs(ra["port_pressure_pa"][i] - rb["port_pressure_pa"][i])
                / (1 + c.relative_tolerance * rb["port_pressure_pa"][i]),
                max(
                    abs(u - v) / (1e-10 + c.relative_tolerance * max(abs(u), abs(v)))
                    for u, v in zip(x.column.component_moles, y.column.component_moles)
                ),
                max(
                    abs(u - v) / (1e-8 + c.relative_tolerance)
                    for u, v in zip(ra["port_mole_fractions"][i], rb["port_mole_fractions"][i])
                ),
                abs(
                    ra["columns"][i]["relative_potential_energy_j"]
                    - rb["columns"][i]["relative_potential_energy_j"]
                )
                / (
                    1e-5
                    + c.relative_tolerance
                    * max(1, abs(rb["columns"][i]["relative_potential_energy_j"]))
                ),
            )
        return error

    def _step(self, state, dt, openings):
        order = self.start_order
        previous = None
        while order <= self.max_order:
            candidate, report = self._candidate(state, dt, openings, order)
            if previous is not None:
                error = self._quadrature_error(previous[0], candidate, previous[1], report)
                if error <= 1:
                    report["quadrature_error"] = error
                    return candidate, report
            previous = (candidate, report)
            order *= 2
        raise HydraulicConvergenceError("Coupled gravitational port quadrature budget exhausted")

    def _relative_energy(self, s, index):
        c = s.column
        mass = math.fsum(
            n * comp.molar_mass_kg_mol
            for n, comp in zip(c.component_moles, self.pipe.pipe.fluid.components)
        )
        return c.total_energy_j - mass * self.gravity * self.elevation

    def _ledger(self, before, after, pipe_reports, vessel_ambient):
        p, w = self.pipe.pipe, self.pipe.wall
        n = [
            math.fsum(row[i] for row in after.pipe.pipe.component_moles)
            - math.fsum(row[i] for row in before.pipe.pipe.component_moles)
            + math.fsum(
                b.column.component_moles[i] - a.column.component_moles[i]
                for a, b in zip(before.vessels, after.vessels)
            )
            for i in range(len(p.fluid.names))
        ]
        de = math.fsum(after.pipe.pipe.total_energy_j) - math.fsum(before.pipe.pipe.total_energy_j)
        de += (
            w.heat_capacity_j_k
            / p.cells
            * math.fsum(
                b - a for a, b in zip(before.pipe.wall_temperature_k, after.pipe.wall_temperature_k)
            )
        )
        de += math.fsum(
            self._relative_energy(b, i)
            - self._relative_energy(a, i)
            + wall.heat_capacity_j_k * (b.wall_temperature_k - a.wall_temperature_k)
            for i, (a, b, wall) in enumerate(zip(before.vessels, after.vessels, self.walls))
        )
        ambient = math.fsum(q for r in pipe_reports for q in r["ambient_heat_j"]) + math.fsum(
            float(q) for q in vessel_ambient
        )
        residual = n + [de - ambient]
        if any(
            abs(x) > 1e-8 + 1e-9 * math.fsum(s.column.component_moles[i] for s in before.vessels)
            for i, x in enumerate(n)
        ) or abs(residual[-1]) > 1e-5 + 1e-9 * max(1, abs(de), abs(ambient)):
            raise ConservationError(
                "Gravitational vessel/pipe N or datum-relative total energy rejected"
            )
        return residual, ambient

    def _error(self, a, b):
        p = self.pipe.pipe
        error = p._error(a.pipe.pipe, b.pipe.pipe)
        error = max(
            error,
            max(
                abs(x - y) / p.temperature_tolerance_k
                for x, y in zip(a.pipe.wall_temperature_k, b.pipe.wall_temperature_k)
            ),
        )
        for i, (x, y) in enumerate(zip(a.vessels, b.vessels)):
            error = max(
                error,
                abs(x.wall_temperature_k - y.wall_temperature_k) / p.temperature_tolerance_k,
                abs(x.column.temperature_k - y.column.temperature_k) / p.temperature_tolerance_k,
                max(
                    abs(u - v) / (1e-10 + p.relative_tolerance * max(abs(u), abs(v)))
                    for u, v in zip(x.column.component_moles, y.column.component_moles)
                ),
                abs(self._relative_energy(x, i) - self._relative_energy(y, i))
                / (
                    1e-6
                    + p.relative_tolerance
                    * max(1, abs(self._relative_energy(x, i)), abs(self._relative_energy(y, i)))
                ),
            )
        return error

    def _retime(self, state, time, nextdt):
        # Called by shared adaptive driver; all child clocks are carried together.
        from dataclasses import replace

        return replace(
            state,
            pipe=replace(
                state.pipe, pipe=replace(state.pipe.pipe, time_s=time, next_step_s=nextdt)
            ),
            vessels=tuple(replace(v, time_s=time, next_step_s=nextdt) for v in state.vessels),
        )
