"""Quasi-steady incompressible-water distribution tree with exact tangents.

Reuses validated pipe/pump/valve laws. Equipment property evaluation is on the
host; pressure/rate tangents and the coupled well solve use Torch float64.
No SciPy hydraulic solve, empirical reservoir source or water inventory is used.
"""

import math
from dataclasses import replace

import torch

from .liquid import G, Liquid, Node, Pipe, PropertyModelError, Pump, ThrottleValve


class WaterInjectionTree:
    def __init__(self, spec):
        if set(spec) - {"fluid", "nodes", "pipes", "pumps", "valves"}:
            raise ValueError("Unknown injection-tree fields")
        self.fluid = Liquid(**spec["fluid"])
        self.nodes = tuple(Node(**n) for n in spec["nodes"])
        self.branches = (
            tuple(Pipe(**b) for b in spec.get("pipes", []))
            + tuple(Pump(**b) for b in spec.get("pumps", []))
            + tuple(ThrottleValve(**b) for b in spec.get("valves", []))
        )
        self.by_name = {n.name: n for n in self.nodes}
        if len(self.by_name) != len(self.nodes) or len({b.name for b in self.branches}) != len(
            self.branches
        ):
            raise ValueError("Duplicate injection node/branch names")
        roots = [n for n in self.nodes if n.pressure_pa is not None]
        if len(roots) != 1 or any(n.source_mass_rate_kg_s for n in self.nodes):
            raise ValueError("Injection tree needs one pressure supply and no hidden node sources")
        self.root = roots[0].name
        if roots[0].pressure_pa < self.fluid.vapor_pressure_pa:
            raise PropertyModelError("Injection supply is below water vapor pressure")
        self.children = {n: [] for n in self.by_name}
        incoming = {}
        for b in self.branches:
            if (
                b.start not in self.by_name
                or b.end not in self.by_name
                or b.end in incoming
                or b.end == self.root
            ):
                raise ValueError("Injection distribution must be a directed tree")
            if not b.open or isinstance(b, ThrottleValve) and b.opening <= 0:
                raise ValueError(
                    "Use injection_enabled for whole-skid isolation; closed internal branches unsupported"
                )
            if isinstance(b, Pipe) and b.length_m < abs(
                self.by_name[b.end].elevation_m - self.by_name[b.start].elevation_m
            ):
                raise ValueError("Injection pipe length is smaller than elevation difference")
            incoming[b.end] = b
            self.children[b.start].append(b)
        self.order, paths = [], {self.root: []}
        queue = [self.root]
        for n in queue:
            for branch in self.children[n]:
                if branch.end in paths:
                    raise ValueError("Injection tree cycle")
                index = len(self.order)
                self.order.append(branch)
                paths[branch.end] = paths[n] + [index]
                queue.append(branch.end)
        if set(paths) != set(self.by_name):
            raise ValueError("Disconnected injection tree")
        self.paths = paths
        self.leaves = tuple(n for n in queue if not self.children[n])
        self._path_matrices = {}
        if not self.leaves or self.root in self.leaves:
            raise ValueError("Injection tree needs delivery leaves")

    def evaluate(self, mass_rates, device="cpu", *, speed_ratios=None):
        if set(mass_rates) != set(self.leaves):
            raise ValueError("Every injection leaf must have an actual delivered mass rate")
        speeds = speed_ratios or {}
        pumps = {b.name for b in self.order if isinstance(b, Pump)}
        if set(speeds) - pumps:
            raise ValueError("Unknown injection pump speed control")
        if any(not math.isfinite(q) or q < -1e-10 for q in mass_rates.values()):
            raise PropertyModelError("Injection tree does not admit reverse well flow")
        rho = self.fluid.density_kg_m3
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        paths = self._path_matrices.get(dev)
        if paths is None:
            paths = torch.tensor(
                [[float(e in self.paths[n]) for e in range(len(self.order))] for n in self.leaves],
                dtype=torch.float64, device=dev,
            )
            self._path_matrices[dev] = paths
        rates = torch.tensor(
            [mass_rates[n] for n in self.leaves], dtype=torch.float64, device=dev
        )
        flows = (paths.T @ rates / rho).detach().cpu().tolist()
        pressure = {self.root: self.by_name[self.root].pressure_pa}
        losses, slopes, branches, violations = [], [], {}, []
        margins = {}
        pressure_scale = max(pressure[self.root], 1e5)
        shaft = 0.0
        for branch, q in zip(self.order, flows):
            if isinstance(branch, Pump):
                branch = replace(branch, speed_ratio=speeds.get(branch.name, branch.speed_ratio))
            loss, slope = branch.loss_and_gradient(q, self.fluid)
            losses.append(loss)
            slopes.append(slope)
            pressure[branch.end] = (
                pressure[branch.start]
                - loss
                - rho
                * G
                * (self.by_name[branch.end].elevation_m - self.by_name[branch.start].elevation_m)
            )
            item = {"flow_m3_s": q, "mass_rate_kg_s": q * rho, "pressure_loss_pa": loss}
            if isinstance(branch, Pump):
                try:
                    hydraulic, power, npsh = branch.operating_state(
                        q, pressure[branch.start], self.fluid
                    )
                    shaft += power
                    item.update(
                        hydraulic_power_w=hydraulic, shaft_power_w=power, npsh_available_m=npsh
                    )
                except ValueError as error:
                    # A finite extrapolated trial curve is allowed in Newton;
                    # the dispatch must reject it before accepting the FIM step.
                    violations.append({"equipment": branch.name, "reason": str(error)})
                # Signed proposal facets do not alter operating_state or its
                # admissible maps. Out-of-domain properties stay opaque and
                # force the caller's safeguarded fallback.
                margins[f"pump:{branch.name}:flow"] = q / (branch.speed_ratio * branch.max_flow_m3_s) - 1
                try:
                    power_value = item.get("shaft_power_w")
                    if power_value is None:
                        power_value = -q * loss / branch.efficiency_at(q)
                    if branch.max_shaft_power_w is not None:
                        margins[f"pump:{branch.name}:power"] = power_value / branch.max_shaft_power_w - 1
                    required = branch.required_npsh_at(q)
                    if required is not None:
                        available = item.get("npsh_available_m")
                        if available is None:
                            velocity = q / (math.pi * branch.inlet_diameter_m**2 / 4)
                            available = (pressure[branch.start] - self.fluid.vapor_pressure_pa) / (rho * G) + velocity**2 / (2 * G)
                        margins[f"pump:{branch.name}:npsh"] = (required - available) / max(required, 1.)
                except ValueError:
                    margins[f"pump:{branch.name}:map_domain"] = None
            if pressure[branch.end] < self.fluid.vapor_pressure_pa:
                violations.append({"node": branch.end, "reason": "below water vapor pressure"})
            branches[branch.name] = item
            margins[f"node:{branch.end}:vapor_pressure"] = (
                self.fluid.vapor_pressure_pa - pressure[branch.end]
            ) / pressure_scale
        tangent = -(paths * rates.new_tensor(slopes)[None, :]) @ paths.T / rho
        return {
            "model": "incompressible_water_injection_tree_v1",
            "leaves": self.leaves,
            "node_pressures_pa": pressure,
            "branches": branches,
            "pressure_derivatives_pa_per_kg_s": tangent,
            "supply_mass_rate_kg_s": math.fsum(mass_rates.values()),
            "shaft_power_w": shaft,
            "feasible": not violations,
            "violations": violations,
            "constraint_margins": margins,
            "scope": "Quasi-steady water tree, prescribed source pressure, no surface storage, slip, thermal equation or calibrated equipment claim",
        }
