"""Conserved, isothermal sedimentation equilibrium of finite PR columns.

Local fugacities satisfy ln(f_i/p_ref)+M_i*g*h/(R*T)=lambda_i.
Interfaces, pressure/composition profiles and T solve one Torch system with
global N and U+PE constraints. No dynamic uniform-pressure vessel is patched.
Capillarity, metastable barriers, sedimentation kinetics and flows are absent.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .component_energy import ComponentEnergyStream
from .equilibrium_properties import EquilibriumFluid
from .inclined_cylinder import InclinedCylinderGeometry, geometry_from_dict
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .peng_robinson import R
from .separator_geometry import SeparatorGeometry
from .three_phase_properties import disk_area_moment, level_tensor
from .torch_linear import TorchNewton


@dataclass(frozen=True)
class StratifiedPRState:
    component_moles: tuple
    total_energy_j: float
    temperature_k: float
    chemical_constants: tuple
    interface_heights_m: tuple
    phase_seeds: tuple
    phase_roots: tuple
    quadrature_order: int
    solution_variables: tuple
    model_id: str

    def __post_init__(self):
        for key in ("component_moles", "chemical_constants", "interface_heights_m"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        object.__setattr__(self, "phase_seeds", tuple(tuple(x) for x in self.phase_seeds))
        object.__setattr__(self, "phase_roots", tuple(self.phase_roots))
        object.__setattr__(self, "solution_variables", tuple(self.solution_variables))
        if not self.solution_variables:
            raise ValueError("Empty column collocation solution")
        for v in self.solution_variables:
            _number(v, "collocation variable")
        if not self.component_moles or not any(n > 0 for n in self.component_moles):
            raise ValueError("Positive column inventory required")
        for n in self.component_moles:
            _number(n, "component inventory", nonnegative=True)
        _number(self.total_energy_j, "total energy")
        _number(self.temperature_k, "temperature", positive=True)
        for v in self.chemical_constants:
            _number(v, "chemical constant")
        for v in self.interface_heights_m:
            _number(v, "interface height", positive=True)
        if (
            len(self.phase_roots) != len(self.phase_seeds)
            or len(self.interface_heights_m) != len(self.phase_roots) - 1
        ):
            raise ValueError("Column phase topology mismatch")
        if any(tag not in ("stable", "liquid", "vapor") for tag in self.phase_roots):
            raise ValueError("Invalid column root")
        for x in self.phase_seeds:
            if len(x) != len(self.component_moles):
                raise ValueError("Phase seed registry differs")
            for v in x:
                _number(v, "phase mole fraction", nonnegative=True)
            if abs(math.fsum(x) - 1) > 1e-10:
                raise ValueError("Phase seed is not normalized")
        if isinstance(self.quadrature_order, bool) or self.quadrature_order not in (2, 4, 8, 16):
            raise ValueError("Unsupported quadrature order")
        if not isinstance(self.model_id, str) or len(self.model_id) != 64:
            raise ValueError("Missing column identity")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Unknown column state fields")
        return cls(**data)


class StratifiedPRColumn:
    def __init__(
        self,
        fluid,
        geometry,
        *,
        gravity_m_s2=9.80665,
        elevation_m=0.0,
        quadrature_order=4,
        maximum_quadrature_order=16,
        relative_tolerance=1e-6,
        temperature_tolerance_k=1e-5,
    ):
        if not isinstance(fluid, EquilibriumFluid):
            raise TypeError("Expected EquilibriumFluid")
        if not isinstance(geometry, SeparatorGeometry):
            raise TypeError("Expected SeparatorGeometry")
        self.fluid, self.geometry = fluid, geometry
        self.gravity_m_s2 = _number(gravity_m_s2, "gravity", nonnegative=True)
        self.elevation_m = _number(elevation_m, "elevation")
        for key, v in (
            ("quadrature_order", quadrature_order),
            ("maximum_quadrature_order", maximum_quadrature_order),
        ):
            if isinstance(v, bool) or v not in (2, 4, 8, 16):
                raise ValueError("Quadrature order must be 2/4/8/16")
            setattr(self, key, v)
        if quadrature_order >= maximum_quadrature_order:
            raise ValueError("Quadrature refinement requires a higher maximum order")
        self.relative_tolerance = _number(relative_tolerance, "relative tolerance", positive=True)
        self.temperature_tolerance_k = _number(
            temperature_tolerance_k, "temperature tolerance", positive=True
        )
        self.model_id = _digest({"model": "stratified_pr_uv_v1", **self.to_dict()})
        self._verified = {}
        self._samples = {}
        self._rules = {}
        for order in (2, 4, 8, 16):
            nodes, weights = np.polynomial.legendre.leggauss(order)
            self._rules[order] = (fluid.tensor(nodes), fluid.tensor(weights))
        self._zero = fluid.tensor(0.0)
        self._height = fluid.tensor(geometry.height_m)
        self._angle_low = fluid.tensor(-math.pi / 2)
        self._angle_high = fluid.tensor(math.pi / 2)

    def to_dict(self):
        return {
            "fluid": self.fluid.to_dict(),
            "geometry": asdict(self.geometry),
            **{
                k: getattr(self, k)
                for k in (
                    "gravity_m_s2",
                    "elevation_m",
                    "quadrature_order",
                    "maximum_quadrature_order",
                    "relative_tolerance",
                    "temperature_tolerance_k",
                )
            },
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        data["fluid"] = EquilibriumFluid.from_dict(data["fluid"], device=device)
        data["geometry"] = geometry_from_dict(data["geometry"])
        return cls(**data)

    def _identity(self):
        if self.model_id != _digest({"model": "stratified_pr_uv_v1", **self.to_dict()}):
            raise PropertyModelError("Column configuration changed")

    def _volume(self, height):
        if isinstance(self.geometry, InclinedCylinderGeometry):
            return self.geometry.volume_tensor(height)
        if self.geometry.orientation == "vertical":
            return height * self.geometry.volume_m3 / self.geometry.height_m
        return disk_area_moment(height, self.geometry.diameter_m)[0] * self.geometry.length_m

    def _moment(self, height):
        if isinstance(self.geometry, InclinedCylinderGeometry):
            return self.geometry.moment_tensor(height)
        if self.geometry.orientation == "vertical":
            return height**2 * self.geometry.volume_m3 / (2 * self.geometry.height_m)
        return disk_area_moment(height, self.geometry.diameter_m)[1] * self.geometry.length_m

    def _grid(self, interfaces, order):
        g = self.geometry
        bounds = torch.cat((self._zero.reshape(1), interfaces, self._height.reshape(1)))
        points, weights = self._rules[order]
        heights = []
        volumes = []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            volume = self._volume(b) - self._volume(a)
            if isinstance(g, InclinedCylinderGeometry):
                va, vb = self._volume(a), self._volume(b)
                h = g.level_tensor(va + (points + 1) / 2 * (vb - va))
                w = weights / 2 * volume
                mean = (w * h).sum() / volume.clamp_min(1e-300)
                variance = (w * (h - mean).square()).sum()
                exact_moment = self._moment(b) - self._moment(a)
                w = w * (
                    1 + (exact_moment - volume * mean) * (h - mean) / variance.clamp_min(1e-300)
                )
            elif g.orientation == "vertical":
                h = (a + b) / 2 + (b - a) / 2 * points
                w = weights / 2 * volume
            else:
                angle_a = self._angle_low if i == 0 else torch.asin(2 * a / g.height_m - 1)
                angle_b = (
                    self._angle_high if i == len(bounds) - 2 else torch.asin(2 * b / g.height_m - 1)
                )
                angle = (angle_a + angle_b) / 2 + (angle_b - angle_a) / 2 * points
                h = g.diameter_m / 2 * (1 + torch.sin(angle))
                w = weights * torch.cos(angle) ** 2
                w = w / w.sum() * volume
            heights.append(h)
            volumes.append(w)
        return heights, volumes

    def _decode_nodes(self, y, active, temperature, tag):
        p = torch.exp(y[:, 0])
        logits = torch.cat((y[:, 1:], y.new_zeros((len(y), 1))), dim=-1)
        xa = torch.softmax(logits, dim=-1)
        x = y.new_zeros((len(y), len(self.fluid.names))).index_copy(-1, active, xa)
        q = torch.vmap(lambda pp, xx: self.fluid.tensor_homogeneous(pp, temperature, xx, tag))(p, x)
        return p, x, q

    def _problem(self, n, energy, seed, order, *, _network_active=None, _network_guess=None):
        f = self.fluid
        g = self.geometry
        active = torch.nonzero(n > 0).flatten() if _network_active is None else _network_active
        c = len(active)
        phase_x, roots, interfaces, pguess, tguess, constants = seed
        m = len(roots)
        if not 1 <= m <= f.max_phases:
            raise PropertyModelError("Unsupported column phase topology")
        count = m * order + 2 * (m - 1)
        if count * c + c + m > 768:
            raise ValueError("Column collocation exceeds declared 768-unknown dense limit")
        initial_rows = []
        segments = []
        offset = 0
        heights, _weights = self._grid(interfaces, order)
        # Include both homogeneous sides of each material interface as unknowns.
        for i in range(m):
            z = phase_x[i][active].clamp_min(1e-30)
            row = torch.cat((torch.log(f.tensor(pguess)).reshape(1), torch.log(z[:-1] / z[-1])))
            initial_rows.append(row.expand(order, c))
            segments.append((offset, offset + order, i, heights[i]))
            offset += order
        for i in range(m - 1):
            for j in (i, i + 1):
                z = phase_x[j][active].clamp_min(1e-30)
                row = torch.cat((torch.log(f.tensor(pguess)).reshape(1), torch.log(z[:-1] / z[-1])))
                initial_rows.append(row.reshape(1, c))
                segments.append((offset, offset + 1, j, interfaces[i].reshape(1)))
                offset += 1
        fractions = (
            torch.diff(torch.cat((f.tensor([0.0]), interfaces, f.tensor([g.height_m]))))
            / g.height_m
        )
        layer_logits = torch.log(fractions[:-1] / fractions[-1])
        initial = torch.cat(
            (
                torch.cat(initial_rows).flatten(),
                constants[active],
                f.tensor([tguess / f.tref]),
                layer_logits,
            )
        )
        lower = torch.cat(
            (
                torch.cat(
                    (f.tensor([math.log(f.min_pressure_pa)]), f.tensor([-100.0] * (c - 1)))
                ).repeat(count),
                f.tensor([-1000.0] * c + [f.min_temperature_k / f.tref] + [-30.0] * (m - 1)),
            )
        )
        upper = torch.cat(
            (
                torch.cat(
                    (f.tensor([math.log(f.max_pressure_pa)]), f.tensor([100.0] * (c - 1)))
                ).repeat(count),
                f.tensor([1000.0] * c + [f.max_temperature_k / f.tref] + [30.0] * (m - 1)),
            )
        )
        e_relative = (
            energy - float((n * f.molar_masses).sum().cpu()) * self.gravity_m_s2 * self.elevation_m
        )
        escale = max(1.0, abs(e_relative), float(n.sum().cpu()) * R * tguess)
        scale_n = n if _network_guess is None else _network_guess
        nscale = scale_n[active].clamp_min(scale_n.sum() * 1e-12)

        def evaluate(y):
            local = y[: count * c].reshape(count, c)
            lam = y[count * c : count * c + c]
            temperature = y[count * c + c] * f.tref
            frac = torch.softmax(torch.cat((y[count * c + c + 1 :], y.new_zeros(1))), dim=0)
            boundaries = torch.cumsum(frac, dim=0)[:-1] * g.height_m
            hh, vw = self._grid(boundaries, order)
            node_heights = list(hh) + [
                boundaries[i].reshape(1) for i in range(m - 1) for _ in range(2)
            ]
            rows = []
            properties = []
            pressures = []
            xs = []
            for (a, b, phase, _), height in zip(segments, node_heights):
                p, x, q = self._decode_nodes(local[a:b], active, temperature, roots[phase])
                mu = (
                    torch.log(x[:, active].clamp_min(1e-300))
                    + q["log_fugacity_coefficients"][:, active]
                    + torch.log(p / f.pref)[:, None]
                )
                chemical = (
                    mu
                    + self.gravity_m_s2
                    * height[:, None]
                    * f.molar_masses[active]
                    / (R * temperature)
                    - lam
                )
                rows.append(chemical.flatten())
                properties.append(q)
                pressures.append(p)
                xs.append(x)
            totals = torch.stack(
                [
                    (vw[i][:, None] * xs[i] / properties[i]["molar_volume_m3_mol"][:, None]).sum(0)
                    for i in range(m)
                ]
            ).sum(0)
            internal = torch.stack(
                [
                    (
                        vw[i]
                        * properties[i]["molar_internal_energy_j_mol"]
                        / properties[i]["molar_volume_m3_mol"]
                    ).sum()
                    for i in range(m)
                ]
            ).sum()
            pe_relative = torch.stack(
                [
                    (vw[i] * properties[i]["density_kg_m3"] * self.gravity_m_s2 * hh[i]).sum()
                    for i in range(m)
                ]
            ).sum()
            entropy = torch.stack(
                [
                    (
                        vw[i]
                        * properties[i]["molar_entropy_j_mol_k"]
                        / properties[i]["molar_volume_m3_mol"]
                    ).sum()
                    for i in range(m)
                ]
            ).sum()
            continuity = [
                torch.log(pressures[m + 2 * i] / pressures[m + 2 * i + 1]).flatten()
                for i in range(m - 1)
            ]
            residual = torch.cat(
                (
                    *rows,
                    *continuity,
                    (totals[active] - n[active]) / nscale,
                    ((internal + pe_relative - e_relative) / escale).reshape(1),
                )
            )
            return residual, {
                "properties": properties,
                "pressures": pressures,
                "compositions": xs,
                "heights": hh,
                "weights": vw,
                "interfaces": boundaries,
                "temperature": temperature,
                "constants": lam,
                "component_moles": totals,
                "internal_energy_j": internal,
                "relative_potential_energy_j": pe_relative,
                "entropy_j_k": entropy,
                "phase_component_moles": torch.stack(
                    [
                        (
                            vw[i][:, None] * xs[i] / properties[i]["molar_volume_m3_mol"][:, None]
                        ).sum(0)
                        for i in range(m)
                    ]
                ),
            }

        return initial, lower, upper, evaluate, active, roots

    def _check_local(self, p, t, x, q):
        stable = self.fluid.flash_pt(p, t, x, _allow_pure_coexistence=True)
        active = x > 0
        phase = stable["phases"][0]
        delta = (
            torch.log(x[active])
            + q["log_fugacity_coefficients"][active]
            - torch.log(phase["mole_fractions"][active])
            - phase["log_fugacity_coefficients"][active]
        )
        if float(delta.abs().max().cpu()) > 2e-7:
            raise PropertyModelError("Column requires a different locally stable phase topology")

    def _solve(self, n, energy, seed, order):
        initial, lower, upper, evaluate, _active, _roots = self._problem(n, energy, seed, order)
        solver = TorchNewton(self.fluid.device, maximum_iterations=40, dense_limit=768)
        solver.host_convergence = True
        with torch.no_grad():
            y, _r, ok, evaluations = solver.solve(
                lambda z: evaluate(z)[0], initial, lower, upper, tolerance=3e-12
            )
        return self._accept(n, energy, seed, order, y, evaluations=evaluations, solver_success=ok)

    def _accept(self, n, energy, seed, order, y, *, evaluations=0, solver_success=True):
        """Validate a joint collocation iterate without moving it in another Newton."""
        initial, lower, upper, evaluate, active, roots = self._problem(n, energy, seed, order)
        if y.shape != initial.shape:
            raise ConservationError("Accepted column collocation dimensions differ")
        with torch.no_grad():
            r, data = evaluate(y)
        valid = (
            solver_success
            & torch.isfinite(r).all()
            & (r.abs().max() < 1e-9)
            & (y >= lower).all()
            & (y <= upper).all()
        )
        for q in data["properties"]:
            valid = valid & q["valid"].all()
        for weights in data["weights"]:
            valid = valid & torch.isfinite(weights).all() & (weights > 0).all()
        if not bool(valid.cpu()):
            raise HydraulicConvergenceError("Stratified PR UV Newton rejected")
        m = len(roots)
        temp = float(data["temperature"].cpu())
        for p, x, q in zip(data["pressures"], data["compositions"], data["properties"]):
            for j in range(len(p)):
                self._check_local(float(p[j].cpu()), temp, x[j], {k: v[j] for k, v in q.items()})
        self._check_topology(data, roots)
        constants = self.fluid.tensor([0.0] * len(self.fluid.names)).index_copy(
            0, active, data["constants"]
        )
        phase_x = tuple(tuple(x[0].cpu().tolist()) for x in data["compositions"][:m])
        state = StratifiedPRState(
            tuple(n.cpu().tolist()),
            energy,
            temp,
            tuple(constants.cpu().tolist()),
            tuple(data["interfaces"].cpu().tolist()),
            phase_x,
            tuple(roots),
            order,
            tuple(y.cpu().tolist()),
            self.model_id,
        )
        mass_error = data["component_moles"] - n
        e_rel = data["internal_energy_j"] + data["relative_potential_energy_j"]
        target_rel = (
            energy
            - float((n * self.fluid.molar_masses).sum().cpu())
            * self.gravity_m_s2
            * self.elevation_m
        )
        if float(mass_error.abs().max().cpu()) > 1e-8 + 1e-9 * float(n.sum().cpu()) or abs(
            float(e_rel.cpu()) - target_rel
        ) > 1e-5 + 1e-9 * max(1, abs(target_rel)):
            raise ConservationError("Column conserved inventory rejected")
        report = {
            "state": state.to_dict(),
            "component_residuals_mol": mass_error.cpu().tolist(),
            "energy_residual_j": float(e_rel.cpu()) - target_rel,
            "internal_energy_j": float(data["internal_energy_j"].cpu()),
            "potential_energy_j": float(data["relative_potential_energy_j"].cpu())
            + float((n * self.fluid.molar_masses).sum().cpu())
            * self.gravity_m_s2
            * self.elevation_m,
            "entropy_j_k": float(data["entropy_j_k"].cpu()),
            "relative_potential_energy_j": float(data["relative_potential_energy_j"].cpu()),
            "mean_pressure_pa": float(
                torch.stack([(data["weights"][i] * data["pressures"][i]).sum() for i in range(m)])
                .sum()
                .cpu()
            )
            / self.geometry.volume_m3,
            "phase_component_moles": data["phase_component_moles"].cpu().tolist(),
            "max_scaled_residual": float(r.abs().max().cpu()),
            "nonlinear_evaluations": evaluations,
            "layers": [
                {
                    "height_m": data["heights"][i].cpu().tolist(),
                    "volume_weights_m3": data["weights"][i].cpu().tolist(),
                    "pressure_pa": data["pressures"][i].cpu().tolist(),
                    "mole_fractions": data["compositions"][i].cpu().tolist(),
                    "density_kg_m3": data["properties"][i]["density_kg_m3"].cpu().tolist(),
                    "molar_internal_energy_j_mol": data["properties"][i][
                        "molar_internal_energy_j_mol"
                    ]
                    .cpu()
                    .tolist(),
                }
                for i in range(m)
            ],
        }
        self._remember(state)
        return state, report

    def _remember(self, state):
        if len(self._verified) >= 8:
            self._verified.pop(next(iter(self._verified)))
        self._verified[_digest(state.to_dict())] = True

    def _check_topology(self, data, roots):
        bounds = torch.cat((self._zero.reshape(1), data["interfaces"], self._height.reshape(1)))
        volumes = self._volume(bounds[1:]) - self._volume(bounds[:-1])
        if not bool(
            (
                torch.isfinite(volumes).all() & (volumes > self.geometry.volume_m3 * 1e-12).all()
            ).cpu()
        ):
            raise PropertyModelError(
                "Column layer degenerates; a different phase topology is required"
            )
        for i in range(len(roots) - 1):
            below = data["properties"][len(roots) + 2 * i]["density_kg_m3"][0]
            above = data["properties"][len(roots) + 2 * i + 1]["density_kg_m3"][0]
            if float((below - above).cpu()) <= 1e-10 * max(1, float(below.cpu())):
                raise PropertyModelError("Gravitational phase stacking is not density stable")

    def inspect(self, state):
        self._identity()
        if not isinstance(state, StratifiedPRState) or state.model_id != self.model_id:
            raise PropertyModelError("Column state/model mismatch")
        if _digest(state.to_dict()) in self._verified:
            return
        f = self.fluid
        if len(state.component_moles) != len(f.names) or len(state.chemical_constants) != len(
            f.names
        ):
            raise PropertyModelError("Column registry mismatch")
        if not self.quadrature_order <= state.quadrature_order <= self.maximum_quadrature_order:
            raise ValueError("State quadrature outside model domain")
        heights = state.interface_heights_m
        if any(not 0 < h < self.geometry.height_m for h in heights) or any(
            b <= a for a, b in zip(heights, heights[1:])
        ):
            raise PropertyModelError("Invalid column interfaces")
        if not f.min_temperature_k <= state.temperature_k <= f.max_temperature_k:
            raise PropertyModelError("Column temperature outside EOS domain")
        n = f.tensor(state.component_moles)
        seed = (
            [f.tensor(x) for x in state.phase_seeds],
            list(state.phase_roots),
            f.tensor(heights),
            math.exp(state.solution_variables[0]),
            state.temperature_k,
            f.tensor(state.chemical_constants),
        )
        initial, lower, upper, evaluate, active, roots = self._problem(
            n, state.total_energy_j, seed, state.quadrature_order
        )
        y = f.tensor(state.solution_variables)
        if y.shape != initial.shape:
            raise PropertyModelError("Column collocation dimensions differ")
        with torch.no_grad():
            r, data = evaluate(y)
        valid = (
            torch.isfinite(r).all()
            & (r.abs().max() < 1e-9)
            & (y >= lower).all()
            & (y <= upper).all()
        )
        for q in data["properties"]:
            valid = valid & q["valid"].all()
        for weights in data["weights"]:
            valid = valid & torch.isfinite(weights).all() & (weights > 0).all()
        if not bool(valid.cpu()):
            raise ConservationError("Serialized column violates equilibrium/inventory")
        if (
            abs(float(data["temperature"].cpu()) - state.temperature_k) > 1e-9
            or float(
                (data["constants"] - f.tensor(state.chemical_constants)[active]).abs().max().cpu()
            )
            > 1e-10
            or (
                len(heights)
                and float((data["interfaces"] - f.tensor(heights)).abs().max().cpu())
                > 1e-9 * self.geometry.height_m
            )
        ):
            raise ConservationError("Column metadata differs from collocation solution")
        for i, x in enumerate(data["compositions"][: len(roots)]):
            if float((x[0] - f.tensor(state.phase_seeds[i])).abs().max().cpu()) > 1e-10:
                raise ConservationError("Column phase seed differs from solved profile")
        for p, x, q in zip(data["pressures"], data["compositions"], data["properties"]):
            for j in range(len(p)):
                self._check_local(
                    float(p[j].cpu()), state.temperature_k, x[j], {k: v[j] for k, v in q.items()}
                )
        self._check_topology(data, roots)
        self._remember(state)

    def _seed(self, n, energy, p, t):
        mass = float((n * self.fluid.molar_masses).sum().cpu())
        u = energy - mass * self.gravity_m_s2 * (self.elevation_m + self.geometry.height_m / 2)
        q = self.fluid.flash_uv(
            n, u, self.geometry.volume_m3, pressure_guess_pa=p, temperature_guess_k=t
        )
        return self._seed_from_flash(q, n.sum())

    def _seed_from_flash(self, q, total):
        """Build a profile guess from a resolved flash, without changing stocks."""
        phases = sorted(q["phases"], key=lambda v: -float(v["density_kg_m3"].cpu()))
        vol = torch.stack([total * v["molar_fraction"] * v["molar_volume_m3_mol"] for v in phases])
        heights = (
            torch.stack([level_tensor(v, self.geometry) for v in torch.cumsum(vol, 0)[:-1]])
            if len(phases) > 1
            else self.fluid.tensor([])
        )
        roots = []
        for phase in phases:
            a = self.fluid.tensor_homogeneous(
                q["pressure_pa"], q["temperature_k"], phase["mole_fractions"], "liquid"
            )
            b = self.fluid.tensor_homogeneous(
                q["pressure_pa"], q["temperature_k"], phase["mole_fractions"], "vapor"
            )
            roots.append(
                "liquid"
                if abs(float((a["compressibility_factor"] - phase["compressibility_factor"]).cpu()))
                <= abs(float((b["compressibility_factor"] - phase["compressibility_factor"]).cpu()))
                else "vapor"
            )
        if len(phases) == 1:
            roots = ["stable"]
        ref = phases[0]
        x = ref["mole_fractions"]
        refheight = float(heights[0].cpu()) if len(heights) else self.geometry.height_m / 2
        constants = (
            torch.log(x.clamp_min(1e-300))
            + ref["log_fugacity_coefficients"]
            + torch.log(q["pressure_pa"] / self.fluid.pref)
            + self.fluid.molar_masses * self.gravity_m_s2 * refheight / (R * q["temperature_k"])
        )
        return (
            [v["mole_fractions"] for v in phases],
            roots,
            heights,
            float(q["pressure_pa"].cpu()),
            float(q["temperature_k"].cpu()),
            constants,
        )

    def equilibrate(
        self, component_moles, total_energy_j, *, pressure_guess_pa, temperature_guess_k
    ):
        self._identity()
        n = self.fluid.tensor(component_moles)
        if n.shape != (len(self.fluid.names),) or not bool(
            (
                torch.isfinite(n).all() & (n >= 0).all() & torch.isfinite(n.sum()) & (n.sum() > 0)
            ).cpu()
        ):
            raise ValueError("Invalid column component inventory")
        energy = _number(total_energy_j, "total energy")
        p = _number(pressure_guess_pa, "pressure guess", positive=True)
        t = _number(temperature_guess_k, "temperature guess", positive=True)
        seed = self._seed(n, energy, p, t)
        previous = None
        attempts = []
        order = self.quadrature_order
        while order <= self.maximum_quadrature_order:
            state, report = self._solve(n, energy, seed, order)
            attempts.append(
                {"order": order, "nonlinear_evaluations": report["nonlinear_evaluations"]}
            )
            if previous is not None:
                old = StratifiedPRState.from_dict(previous["state"])
                error = max(
                    abs(state.temperature_k - old.temperature_k)
                    / (
                        self.temperature_tolerance_k
                        + self.relative_tolerance * max(state.temperature_k, old.temperature_k)
                    ),
                    abs(
                        report["relative_potential_energy_j"]
                        - previous["relative_potential_energy_j"]
                    )
                    / (
                        1e-5
                        + self.relative_tolerance
                        * max(
                            1,
                            abs(report["relative_potential_energy_j"]),
                            abs(previous["relative_potential_energy_j"]),
                        )
                    ),
                )
                error = max(
                    error,
                    abs(report["mean_pressure_pa"] - previous["mean_pressure_pa"])
                    / (
                        1
                        + self.relative_tolerance
                        * max(abs(report["mean_pressure_pa"]), abs(previous["mean_pressure_pa"]))
                    ),
                )
                for a, b in zip(state.interface_heights_m, old.interface_heights_m):
                    error = max(
                        error,
                        abs(a - b) / (1e-8 + self.relative_tolerance * self.geometry.height_m),
                    )
                a = self.fluid.tensor(report["phase_component_moles"])
                b = self.fluid.tensor(previous["phase_component_moles"])
                error = max(
                    error,
                    float(((a - b).abs() / (1e-9 + self.relative_tolerance * n.sum())).max().cpu()),
                )
                if error <= 1:
                    return {
                        **report,
                        "quadrature_error_ratio": error,
                        "quadrature_attempts": attempts,
                    }
            previous = report
            # Conserved UV problem remains unchanged; only resolution changes.
            seed = (
                [self.fluid.tensor(x) for x in state.phase_seeds],
                list(state.phase_roots),
                self.fluid.tensor(state.interface_heights_m),
                report["layers"][0]["pressure_pa"][0],
                state.temperature_k,
                self.fluid.tensor(state.chemical_constants),
            )
            order *= 2
        raise HydraulicConvergenceError("Column quadrature convergence not resolved within budget")

    def initialize(self, pressure_pa, temperature_k, mole_fractions):
        self._identity()
        f = self.fluid
        q = f.flash_pt(pressure_pa, temperature_k, mole_fractions)
        total = self.geometry.volume_m3 / float(q["molar_volume_m3_mol"].cpu())
        n = total * q["mole_fractions"]
        phases = sorted(q["phases"], key=lambda v: -float(v["density_kg_m3"].cpu()))
        volume = f.tensor(0.0)
        moment = f.tensor(0.0)
        potential = f.tensor(0.0)
        for phase in phases:
            dv = total * phase["molar_fraction"] * phase["molar_volume_m3_mol"]
            top = level_tensor(volume + dv, self.geometry)
            nextmoment = self._moment(top)
            potential = potential + phase["density_kg_m3"] * self.gravity_m_s2 * (
                dv * self.elevation_m + nextmoment - moment
            )
            volume = volume + dv
            moment = nextmoment
        energy = total * float(q["molar_internal_energy_j_mol"].cpu()) + float(potential.cpu())
        result = self.equilibrate(
            n, energy, pressure_guess_pa=pressure_pa, temperature_guess_k=temperature_k
        )
        initial_entropy = total * float(q["molar_entropy_j_mol_k"].cpu())
        if result["entropy_j_k"] < initial_entropy - 1e-6 - 1e-10 * max(1, abs(initial_entropy)):
            raise ConservationError("Column equilibrium lowered entropy at fixed N/E/V")
        return {
            **result,
            "initial_uniform_entropy_j_k": initial_entropy,
            "entropy_gain_j_k": result["entropy_j_k"] - initial_entropy,
            "initial_uniform_state": {
                "pressure_pa": pressure_pa,
                "temperature_k": temperature_k,
                "phase_moles": [total * float(p["molar_fraction"].cpu()) for p in phases],
                "phase_mole_fractions": [p["mole_fractions"].cpu().tolist() for p in phases],
                "phase_compressibility_factors": [
                    float(p["compressibility_factor"].cpu()) for p in phases
                ],
                "internal_energy_j": total * float(q["molar_internal_energy_j_mol"].cpu()),
                "potential_energy_j": float(potential.cpu()),
            },
        }

    def sample(self, state, height_m, *, side=None):
        self.inspect(state)
        height = _number(height_m, "sample height", nonnegative=True)
        if height > self.geometry.height_m:
            raise ValueError("Sample outside column")
        if side not in (None, "below", "above"):
            raise ValueError("Unknown interface side")
        cache_key = (_digest(state.to_dict()), height, side)
        if cache_key in self._samples:
            return copy.deepcopy(self._samples[cache_key])
        matches = [
            i
            for i, h in enumerate(state.interface_heights_m)
            if abs(height - h) <= 1e-10 * max(1, self.geometry.height_m)
        ]
        if matches and side is None:
            raise ValueError("Interface sample needs below/above side")
        index = (
            (matches[0] if side == "below" else matches[0] + 1)
            if matches
            else sum(height > h for h in state.interface_heights_m)
        )
        f = self.fluid
        n = f.tensor(state.component_moles)
        active = torch.nonzero(n > 0).flatten()
        c = len(active)
        x0 = f.tensor(state.phase_seeds[index])[active].clamp_min(1e-30)
        # Fugacity constants determine P, not an interpolated hydrostatic guess.
        p0 = math.exp(state.solution_variables[index * state.quadrature_order * c])
        initial = torch.cat(
            (
                f.tensor([math.log(min(f.max_pressure_pa, max(f.min_pressure_pa, p0)))]),
                torch.log(x0[:-1] / x0[-1]),
            )
        )
        t = f.tensor(state.temperature_k)
        lam = f.tensor(state.chemical_constants)[active]

        def evaluate(y):
            p, x, q = self._decode_nodes(y.reshape(1, c), active, t, state.phase_roots[index])
            mu = (
                torch.log(x[0, active].clamp_min(1e-300))
                + q["log_fugacity_coefficients"][0, active]
                + torch.log(p[0] / f.pref)
            )
            return mu + f.molar_masses[active] * self.gravity_m_s2 * height / (R * t) - lam

        solver = TorchNewton(f.device, maximum_iterations=40)
        solver.host_convergence = True
        with torch.no_grad():
            y, r, ok, _ = solver.solve(
                evaluate,
                initial,
                f.tensor([math.log(f.min_pressure_pa)] + [-100.0] * (c - 1)),
                f.tensor([math.log(f.max_pressure_pa)] + [100.0] * (c - 1)),
                tolerance=3e-12,
            )
            p, x, q = self._decode_nodes(y.reshape(1, c), active, t, state.phase_roots[index])
        if not bool((ok & q["valid"].all() & (r.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("Column local sample rejected")
        self._check_local(
            float(p[0].cpu()), state.temperature_k, x[0], {k: v[0] for k, v in q.items()}
        )
        result = {
            "height_m": height,
            "elevation_m": self.elevation_m + height,
            "phase_index": index,
            "pressure_pa": float(p[0].cpu()),
            "temperature_k": state.temperature_k,
            "mole_fractions": x[0].cpu().tolist(),
            **{
                key: float(q[key][0].cpu())
                for key in (
                    "density_kg_m3",
                    "molar_enthalpy_j_mol",
                    "molar_entropy_j_mol_k",
                    "molar_volume_m3_mol",
                )
            },
            "max_chemical_residual": float(r.abs().max().cpu()),
        }
        if len(self._samples) >= 64:
            self._samples.pop(next(iter(self._samples)))
        self._samples[cache_key] = copy.deepcopy(result)
        return result

    def stream_at(self, state, height_m, molar_rate_mol_s, *, side=None):
        rate = _number(molar_rate_mol_s, "molar rate", nonnegative=True)
        sample = self.sample(state, height_m, side=side)
        components = tuple(rate * x for x in sample["mole_fractions"])
        mass = math.fsum(n * c.molar_mass_kg_mol for n, c in zip(components, self.fluid.components))
        energy = (
            rate * sample["molar_enthalpy_j_mol"] + mass * self.gravity_m_s2 * sample["elevation_m"]
        )
        return ComponentEnergyStream(
            component_molar_rates_mol_s=components,
            total_energy_rate_w=energy,
            pressure_pa=sample["pressure_pa"],
            model_id=self.fluid.model_id,
            elevation_m=sample["elevation_m"],
            gravity_m_s2=self.gravity_m_s2,
        )
