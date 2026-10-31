"""Steady, isothermal, incompressible single-liquid hydraulics in SI.

Elevation is positive upwards; pressure is absolute; sources inject mass.
Darcy-Weisbach uses 64/Re, Swamee-Jain, and a C1 cubic transition between
Re=2000 and 4000. Unknown pressures and signed pipe rates are solved together.
There is no gas, multiphase holdup, temperature solve, or automatic cavitation cap.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from numbers import Real

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .curves import EquipmentCurve

G = 9.80665


def _number(value, name, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name}: expected a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name}: invalid sign")
    return float(value)


def _name(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("Names must be nonempty strings without surrounding whitespace")


class HydraulicConvergenceError(RuntimeError):
    """The hydraulic equations did not converge; no passing result is returned."""


class InvalidPressureError(ValueError):
    """The incompressible liquid solution violates its pressure domain."""


@dataclass(frozen=True)
class Liquid:
    density_kg_m3: float
    viscosity_pa_s: float
    vapor_pressure_pa: float = 0.0

    def __post_init__(self):
        _number(self.density_kg_m3, "density_kg_m3", positive=True)
        _number(self.viscosity_pa_s, "viscosity_pa_s", positive=True)
        _number(self.vapor_pressure_pa, "vapor_pressure_pa", nonnegative=True)


@dataclass(frozen=True)
class Node:
    name: str
    elevation_m: float = 0.0
    pressure_pa: float | None = None
    source_mass_rate_kg_s: float = 0.0

    def __post_init__(self):
        _name(self.name)
        _number(self.elevation_m, "elevation_m")
        _number(self.source_mass_rate_kg_s, "source_mass_rate_kg_s")
        if self.pressure_pa is not None:
            _number(self.pressure_pa, "pressure_pa", positive=True)


@dataclass(frozen=True)
class Pipe:
    name: str
    start: str
    end: str
    length_m: float
    diameter_m: float
    roughness_m: float = 0.0
    minor_loss_coefficient: float = 0.0
    open: bool = True
    check_valve: bool = False

    def __post_init__(self):
        for name in (self.name, self.start, self.end):
            _name(name)
        if self.start == self.end:
            raise ValueError(f"{self.name}: pipe endpoints must differ")
        _number(self.length_m, "length_m", positive=True)
        _number(self.diameter_m, "diameter_m", positive=True)
        _number(self.roughness_m, "roughness_m", nonnegative=True)
        _number(self.minor_loss_coefficient, "minor_loss_coefficient", nonnegative=True)
        if self.roughness_m / self.diameter_m > 0.05:
            raise ValueError(f"{self.name}: relative roughness exceeds 0.05")
        if not isinstance(self.check_valve, bool):
            raise TypeError("check_valve must be boolean")
        if not isinstance(self.open, bool):
            raise TypeError(f"{self.name}: open must be boolean")

    def loss_and_gradient(self, q_m3_s, liquid):
        """Signed dissipative pressure loss and d(loss)/dq, including q=0."""
        q = _number(q_m3_s, "q_m3_s")
        rho, mu = liquid.density_kg_m3, liquid.viscosity_pa_s
        d, length, k = self.diameter_m, self.length_m, self.minor_loss_coefficient
        area = math.pi * d * d / 4
        re_per_q = rho * d / (mu * area)
        re = abs(q) * re_per_q
        c = rho / (2 * area * area)
        laminar = 128 * mu * length / (math.pi * d**4)
        if re <= 2000:
            return laminar * q + c * k * q * abs(q), laminar + 2 * c * k * abs(q)

        def turbulent(r):
            term = 5.74 * r ** (-0.9)
            a = self.roughness_m / (3.7 * d) + term
            loga = math.log10(a)
            f = 0.25 / loga**2
            df = -0.5 * (-0.9 * term / (r * a * math.log(10))) / loga**3
            return f, df

        if re >= 4000:
            f, df = turbulent(re)
        else:
            # Cubic Hermite form of the continuous transition in EPANET 2.2.
            t = (re - 2000) / 2000
            a, da = 0.032, -64 / 2000**2
            b, db = turbulent(4000.0)
            f = (
                (2 * t**3 - 3 * t * t + 1) * a
                + (t**3 - 2 * t * t + t) * 2000 * da
                + (-2 * t**3 + 3 * t * t) * b
                + (t**3 - t * t) * 2000 * db
            )
            df = (
                (6 * t * t - 6 * t) * a / 2000
                + (3 * t * t - 4 * t + 1) * da
                + (-6 * t * t + 6 * t) * b / 2000
                + (3 * t * t - 2 * t) * db
            )
        loss = c * (f * length / d + k) * q * abs(q)
        gradient = c * abs(q) * (2 * (f * length / d + k) + length / d * df * re)
        return loss, gradient


class EquipmentOperatingError(ValueError):
    """A converged hydraulic state is outside an equipment operating envelope."""


class PropertyModelError(EquipmentOperatingError):
    """The property model cannot represent the supplied physical state."""


class ConservationError(HydraulicConvergenceError):
    """A computed state fails mass or energy conservation; never a capacity bound."""


@dataclass(frozen=True)
class Pump:
    """Forward-only fixed-speed centrifugal pump with a bounded quadratic curve.

    H(q, speed) = speed**2 * shutoff_head_m - curve_coefficient_s2_m5 * q**2.
    Numerical continuation to negative q is only for Newton; reverse operation
    is rejected unless an enabled check valve blocks it. This is not a turbine.
    """

    name: str
    start: str
    end: str
    shutoff_head_m: float
    curve_coefficient_s2_m5: float
    max_flow_m3_s: float
    efficiency: float = 0.75
    speed_ratio: float = 1.0
    max_shaft_power_w: float | None = None
    npsh_required_m: float | None = None
    inlet_diameter_m: float | None = None
    open: bool = True
    check_valve: bool = False

    head_curve: EquipmentCurve | None = None
    efficiency_curve: EquipmentCurve | None = None
    npsh_curve: EquipmentCurve | None = None

    def __post_init__(self):
        for name in (self.name, self.start, self.end):
            _name(name)
        if self.start == self.end:
            raise ValueError(f"{self.name}: pump endpoints must differ")
        for name in (
            "shutoff_head_m",
            "curve_coefficient_s2_m5",
            "max_flow_m3_s",
            "efficiency",
            "speed_ratio",
        ):
            _number(getattr(self, name), name, positive=True)
        if self.efficiency > 1:
            raise ValueError("Pump efficiency must not exceed 1")
        if (
            self.head_curve is None
            and self.curve_coefficient_s2_m5 * self.max_flow_m3_s**2 > self.shutoff_head_m
        ):
            raise ValueError("Pump maximum flow extends into negative head")
        for name in ("max_shaft_power_w", "inlet_diameter_m"):
            if getattr(self, name) is not None:
                _number(getattr(self, name), name, positive=True)
        if self.npsh_required_m is not None:
            _number(self.npsh_required_m, "npsh_required_m", nonnegative=True)
            if self.inlet_diameter_m is None:
                raise ValueError("NPSH check requires inlet_diameter_m")
        for name in ("head_curve", "efficiency_curve", "npsh_curve"):
            curve = getattr(self, name)
            if isinstance(curve, dict):
                curve = EquipmentCurve(**curve)
                object.__setattr__(self, name, curve)
            if curve is None:
                continue
            if not isinstance(curve, EquipmentCurve):
                raise TypeError(f"{name} must be an EquipmentCurve")
            if curve.flow_m3_s[0] != 0 or curve.flow_m3_s[-1] < self.max_flow_m3_s:
                raise ValueError(f"{name} must cover zero through nominal maximum flow")
            if name == "head_curve":
                if any(v < 0 for v in curve.values) or any(
                    b >= a for a, b in zip(curve.values, curve.values[1:])
                ):
                    raise ValueError("Pump head curve must be nonnegative and strictly decreasing")
                if not math.isclose(curve.values[0], self.shutoff_head_m, rel_tol=1e-12):
                    raise ValueError("Head curve shutoff must match shutoff_head_m")
            elif name == "efficiency_curve" and any(not 0 < v <= 1 for v in curve.values):
                raise ValueError("Efficiency map must be in (0, 1]")
            elif name == "npsh_curve":
                if any(v < 0 for v in curve.values) or self.inlet_diameter_m is None:
                    raise ValueError("NPSH map requires nonnegative values and inlet_diameter_m")
        if not isinstance(self.check_valve, bool):
            raise TypeError("check_valve must be boolean")
        if not isinstance(self.open, bool):
            raise TypeError("Pump open must be boolean")

    def loss_and_gradient(self, q_m3_s, liquid):
        q = _number(q_m3_s, "q_m3_s")
        coefficient = liquid.density_kg_m3 * G
        if self.head_curve is not None:
            head, slope = self.head_curve.evaluate(abs(q) / self.speed_ratio, extrapolate=True)
            sign = 1 if q >= 0 else -1
            return (
                coefficient
                * self.speed_ratio**2
                * (-self.shutoff_head_m + sign * (self.shutoff_head_m - head)),
                -coefficient * self.speed_ratio * slope,
            )
        return (
            coefficient
            * (
                self.curve_coefficient_s2_m5 * q * abs(q)
                - self.speed_ratio**2 * self.shutoff_head_m
            ),
            coefficient * 2 * self.curve_coefficient_s2_m5 * abs(q),
        )

    def efficiency_at(self, q):
        return (
            self.efficiency_curve.evaluate(q / self.speed_ratio)[0]
            if self.efficiency_curve is not None
            else self.efficiency
        )

    def loss_speed_gradient(self, q_m3_s, liquid):
        """Partial d(pressure loss)/d(speed) at fixed volume flow."""
        q = _number(q_m3_s, "q_m3_s")
        coefficient = liquid.density_kg_m3 * G
        if self.head_curve is None:
            return -2 * coefficient * self.speed_ratio * self.shutoff_head_m
        head, slope = self.head_curve.evaluate(abs(q) / self.speed_ratio, extrapolate=True)
        sign = 1 if q >= 0 else -1
        base = -self.shutoff_head_m + sign * (self.shutoff_head_m - head)
        return coefficient * (2 * self.speed_ratio * base + sign * abs(q) * slope)

    def required_npsh_at(self, q):
        required = self.npsh_required_m
        if self.npsh_curve is not None:
            # Approximate affinity model for the nominal-speed NPSH map.
            mapped = self.speed_ratio**2 * self.npsh_curve.evaluate(q / self.speed_ratio)[0]
            required = max(required or 0.0, mapped)
        return required

    def operating_state(self, q, suction_pressure_pa, liquid):
        if q < 0 or q > self.speed_ratio * self.max_flow_m3_s:
            raise EquipmentOperatingError(f"{self.name}: flow outside forward pump curve")
        hydraulic = -q * self.loss_and_gradient(q, liquid)[0]
        shaft = hydraulic / self.efficiency_at(q)
        if self.max_shaft_power_w is not None and shaft > self.max_shaft_power_w:
            raise EquipmentOperatingError(f"{self.name}: shaft power limit exceeded")
        npsh = None
        if self.inlet_diameter_m is not None:
            velocity = q / (math.pi * self.inlet_diameter_m**2 / 4)
            npsh = (suction_pressure_pa - liquid.vapor_pressure_pa) / (
                liquid.density_kg_m3 * G
            ) + velocity**2 / (2 * G)
        required = self.required_npsh_at(q)
        if required is not None and npsh < required:
            raise EquipmentOperatingError(f"{self.name}: insufficient NPSH")
        return hydraulic, shaft, npsh


@dataclass(frozen=True)
class ThrottleValve:
    """Nonflashing incompressible orifice; opening is fractional flow area."""

    name: str
    start: str
    end: str
    diameter_m: float
    discharge_coefficient: float = 0.62
    opening: float = 1.0
    open: bool = True
    check_valve: bool = False

    def __post_init__(self):
        for name in (self.name, self.start, self.end):
            _name(name)
        if self.start == self.end:
            raise ValueError("Valve endpoints must differ")
        _number(self.diameter_m, "diameter_m", positive=True)
        _number(self.discharge_coefficient, "discharge_coefficient", positive=True)
        _number(self.opening, "opening", nonnegative=True)
        if self.opening > 1 or self.discharge_coefficient > 1:
            raise ValueError("Opening and discharge coefficient must not exceed 1")
        if not isinstance(self.open, bool) or not isinstance(self.check_valve, bool):
            raise TypeError("Valve status must be boolean")

    def loss_and_gradient(self, q_m3_s, liquid):
        q = _number(q_m3_s, "q_m3_s")
        if self.opening == 0:
            raise ValueError("Closed valve has no flow equation")
        area = math.pi * self.diameter_m**2 / 4 * self.opening
        coefficient = liquid.density_kg_m3 / (2 * (self.discharge_coefficient * area) ** 2)
        return coefficient * q * abs(q), 2 * coefficient * abs(q)


@dataclass(frozen=True)
class NetworkResult:
    pressures_pa: dict
    pipe_mass_rates_kg_s: dict
    pipe_volume_rates_m3_s: dict
    boundary_mass_rates_kg_s: dict
    node_mass_residuals_kg_s: dict
    pipe_equation_residuals_pa: dict
    pipe_dissipation_w: dict
    power_balance_error_w: float
    iterations: int
    valve_mass_rates_kg_s: dict = field(default_factory=dict)
    valve_volume_rates_m3_s: dict = field(default_factory=dict)
    valve_equation_residuals_pa: dict = field(default_factory=dict)
    valve_dissipation_w: dict = field(default_factory=dict)
    pump_mass_rates_kg_s: dict = field(default_factory=dict)
    pump_volume_rates_m3_s: dict = field(default_factory=dict)
    pump_equation_residuals_pa: dict = field(default_factory=dict)
    pump_hydraulic_power_w: dict = field(default_factory=dict)
    pump_shaft_power_w: dict = field(default_factory=dict)
    pump_npsh_available_m: dict = field(default_factory=dict)
    pump_npsh_required_m: dict = field(default_factory=dict)
    pump_npsh_margin_m: dict = field(default_factory=dict)
    pump_efficiencies: dict = field(default_factory=dict)
    pump_head_gains_m: dict = field(default_factory=dict)
    branch_states: dict = field(default_factory=dict)
    check_valve_driving_pressure_pa: dict = field(default_factory=dict)
    operating_controls: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class LiquidNetwork:
    """Arbitrary pipe graph with at least one pressure boundary per component.

    Positive source and pressure-boundary rates supply the network. Pipe
    rates are positive start->end. Fixed-pressure nodes can exchange the
    mass necessary to hold pressure; those rates are returned explicitly.
    """

    def __init__(self, fluid, nodes, pipes, pumps=(), valves=()):
        if not isinstance(fluid, Liquid):
            raise TypeError("This network requires one incompressible Liquid")
        self.fluid, self.nodes, self.pipes = fluid, tuple(nodes), tuple(pipes)
        if not self.nodes or not all(isinstance(n, Node) for n in self.nodes):
            raise ValueError("Provide at least one Node")
        if not all(isinstance(p, Pipe) for p in self.pipes):
            raise ValueError("All branches must be Pipe objects")
        self.pumps = tuple(pumps)
        if not all(isinstance(p, Pump) for p in self.pumps):
            raise ValueError("All pumps must be Pump objects")
        self.valves = tuple(valves)
        if not all(isinstance(v, ThrottleValve) for v in self.valves):
            raise TypeError("All valves must be ThrottleValve objects")
        self._branches = self.pipes + self.pumps + self.valves
        self._index = {n.name: i for i, n in enumerate(self.nodes)}
        if len(self._index) != len(self.nodes):
            raise ValueError("Duplicate node names")
        if len({p.name for p in self._branches}) != len(self._branches):
            raise ValueError("Duplicate branch names")
        # Geometry and endpoints remain valid when closed branches reopen.
        for p in self._branches:
            if p.start not in self._index or p.end not in self._index:
                raise ValueError(f"{p.name}: unknown endpoint")
            a, b = self._index[p.start], self._index[p.end]
            if (
                isinstance(p, Pipe)
                and abs(self.nodes[a].elevation_m - self.nodes[b].elevation_m) > p.length_m + 1e-9
            ):
                raise ValueError(f"{p.name}: length is less than its elevation difference")
        self._active = tuple(
            p
            for p in self._branches
            if p.open and not (isinstance(p, ThrottleValve) and p.opening == 0)
        )
        rows, cols, values = [], [], []
        for e, p in enumerate(self._active):
            a, b = self._index[p.start], self._index[p.end]
            rows.extend((a, b))
            cols.extend((e, e))
            values.extend((1.0, -1.0))
        self._B = sp.csc_matrix((values, (rows, cols)), shape=(len(self.nodes), len(self._active)))
        graph = abs(self._B) @ abs(self._B.T)
        _, self._components = sp.csgraph.connected_components(graph, directed=False)

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {"model", "fluid", "nodes", "pipes", "pumps", "valves"}
        if unknown or data.get("model", "incompressible_liquid") != "incompressible_liquid":
            raise ValueError(f"Unsupported surface model or fields: {sorted(unknown)}")
        return cls(
            Liquid(**data["fluid"]),
            [Node(**n) for n in data["nodes"]],
            [Pipe(**p) for p in data["pipes"]],
            [Pump(**p) for p in data.get("pumps", [])],
            [ThrottleValve(**v) for v in data.get("valves", [])],
        )

    def to_dict(self):
        return {
            "model": "incompressible_liquid",
            "fluid": asdict(self.fluid),
            "nodes": [asdict(n) for n in self.nodes],
            "pipes": [asdict(p) for p in self.pipes],
            "pumps": [asdict(p) for p in self.pumps],
            "valves": [asdict(v) for v in self.valves],
        }

    def with_controls(self, pump_speed_ratios=None, pump_status=None, valve_openings=None):
        """Return an effective immutable equipment configuration."""
        speeds, status, openings = pump_speed_ratios or {}, pump_status or {}, valve_openings or {}
        names = {p.name for p in self.pumps}
        unknown = (set(speeds) | set(status)) - names
        if unknown:
            raise ValueError(f"Unknown pump controls: {sorted(unknown)}")
        unknown = set(openings) - {v.name for v in self.valves}
        if unknown:
            raise ValueError(f"Unknown valve controls: {sorted(unknown)}")
        pumps = [replace(p, speed_ratio=speeds.get(p.name, p.speed_ratio),
                         open=status.get(p.name, p.open)) for p in self.pumps]
        valves = [replace(v, opening=openings.get(v.name, v.opening)) for v in self.valves]
        return LiquidNetwork(self.fluid, self.nodes, self.pipes, pumps, valves)

    def _equipment_controls(self):
        return {"pump_speed_ratios": {p.name: p.speed_ratio for p in self.pumps},
                "pump_status": {p.name: p.open for p in self.pumps},
                "valve_openings": {v.name: v.opening for v in self.valves}}

    def pressure_sensitivities(self, result, source_nodes, pressure_nodes, *,
                               regulated_pump=None, regulated_node=None):
        """d(node pressure)/d(source mass rate), with the current active set fixed.

        Recorded operating controls recover the effective configuration. With
        regulation the speed response is included as an extra unknown. At an
        active-set transition the derivative is one-sided, not a smooth map.
        """
        if (regulated_pump is None) != (regulated_node is None):
            raise ValueError("A regulated pump and node must be supplied together")
        if result.operating_controls and result.operating_controls != self._equipment_controls():
            return self.with_controls(**result.operating_controls).pressure_sensitivities(
                result, source_nodes, pressure_nodes, regulated_pump=regulated_pump,
                regulated_node=regulated_node)
        return self._pressure_tangent(result, source_nodes, pressure_nodes,
                                      regulated_pump, regulated_node)

    def pump_speed_sensitivities(self, result, pump_names, pressure_nodes):
        """d(node pressure)/d(pump speed), for the recorded equipment state."""
        if result.operating_controls and result.operating_controls != self._equipment_controls():
            return self.with_controls(**result.operating_controls).pump_speed_sensitivities(
                result, pump_names, pressure_nodes)
        return self._pressure_tangent(result, (), pressure_nodes, speed_pumps=tuple(pump_names))

    def _pressure_tangent(self, result, source_nodes, pressure_nodes,
                          regulated_pump=None, regulated_node=None, speed_pumps=()):
        source_nodes, pressure_nodes = tuple(source_nodes), tuple(pressure_nodes)
        for name in source_nodes + pressure_nodes:
            if name not in self._index:
                raise ValueError(f"Unknown sensitivity node: {name}")
        rho = self.fluid.density_kg_m3
        fixed = np.array([n.name in result.boundary_mass_rates_kg_s for n in self.nodes])
        if any(fixed[self._index[name]] for name in source_nodes):
            raise ValueError("Source sensitivity requires a free-pressure node")
        free = np.flatnonzero(~fixed)
        Bf = self._B[free].tocsc()
        rates = dict(
            result.pipe_volume_rates_m3_s,
            **result.pump_volume_rates_m3_s,
            **result.valve_volume_rates_m3_s,
        )
        q = np.array([rates[p.name] for p in self._active])
        heads = np.array(
            [
                _number(result.pressures_pa[n.name], "sensitivity pressure", positive=True)
                + rho * G * n.elevation_m
                for n in self.nodes
            ]
        )
        losses = np.array(
            [p.loss_and_gradient(float(v), self.fluid) for p, v in zip(self._active, q)]
        ).reshape(-1, 2)
        blocked = np.array([result.branch_states.get(p.name) == "blocked" for p in self._active], dtype=bool)
        driving = self._B.T @ heads - losses[:, 0]
        if (
            np.max(np.abs(driving[~blocked]), initial=0) > 1e-3
            or np.max(driving[blocked], initial=0) > 1e-3
        ):
            raise ValueError("Sensitivity state does not satisfy this network configuration")
        ps = max(1e5, float(np.abs(heads).max(initial=0)))
        qs = max(1e-6, float(np.abs(q).max(initial=0)))
        ms = rho * qs
        tangent = np.where(blocked, 1.0, losses[:, 1] * qs / ps)
        jac = sp.bmat(
            [
                [sp.diags((~blocked).astype(float)) @ Bf.T, -sp.diags(tangent)],
                [sp.csc_matrix((len(free), len(free))), Bf * (rho * qs / ms)],
            ],
            format="csc",
        )
        count = len(speed_pumps) if speed_pumps else len(source_nodes)
        rhs = np.zeros((len(self._active) + len(free), count))
        free_position = {int(node): i for i, node in enumerate(free)}
        for col, name in enumerate(source_nodes):
            rhs[len(self._active) + free_position[self._index[name]], col] = 1 / ms
        pump_positions = {p.name: i for i, p in enumerate(self._active) if isinstance(p, Pump)}
        for col, name in enumerate(speed_pumps):
            if name not in {p.name for p in self.pumps}:
                raise ValueError(f"Unknown sensitivity pump: {name}")
            if name in pump_positions:
                pos = pump_positions[name]
                if not blocked[pos]:
                    rhs[pos, col] = self._active[pos].loss_speed_gradient(q[pos], self.fluid) / ps
        if regulated_pump is not None:
            regulated_names = ((regulated_pump,) if isinstance(regulated_pump, str)
                               else tuple(regulated_pump))
            if not regulated_names or len(set(regulated_names)) != len(regulated_names):
                raise ValueError("Regulated pump group must be nonempty and distinct")
            if set(regulated_names) - set(pump_positions) or regulated_node not in self._index:
                raise ValueError("Unknown/closed regulated pump or node")
            node = self._index[regulated_node]
            if fixed[node]:
                raise ValueError("Regulated sensitivity node must be free")
            positions = [pump_positions[name] for name in regulated_names]
            if all(blocked[pos] for pos in positions):
                raise HydraulicConvergenceError("Blocked pump cannot regulate pressure")
            column = np.zeros((jac.shape[0], 1))
            for pos in positions:
                if not blocked[pos]:
                    column[pos, 0] = -self._active[pos].loss_speed_gradient(q[pos], self.fluid) / ps
            row = np.zeros((1, jac.shape[1]))
            row[0, free_position[node]] = 1.0
            jac = sp.bmat([[jac, sp.csc_matrix(column)],
                           [sp.csc_matrix(row), sp.csc_matrix((1, 1))]], format="csc")
            rhs = np.vstack([rhs, np.zeros((1, count))])
        if jac.shape[0] == 0:
            return np.zeros((len(pressure_nodes), count))
        try:
            solution = splu(jac).solve(rhs)
        except RuntimeError as error:
            raise HydraulicConvergenceError("Singular network sensitivity system") from error
        if not np.isfinite(solution).all():
            raise HydraulicConvergenceError("Nonfinite network sensitivity")
        derivative = np.zeros((len(self.nodes), count))
        derivative[free] = ps * solution[: len(free)]
        return derivative[[self._index[name] for name in pressure_nodes]]

    def solve(
        self,
        source_mass_rates_kg_s=None,
        boundary_pressures_pa=None,
        initial=None,
        pressure_tolerance_pa=1e-3,
        mass_tolerance_kg_s=1e-8,
        max_iterations=60,
        pump_speed_ratios=None,
        pump_status=None,
        valve_openings=None,
    ):
        """Solve without mutating the model or the previous result.

        Sources and pressures may be overridden by node name at each call.
        A pressure override of None releases that pressure boundary. This is
        the interface for subsequent well/facility coupling, not a coupled
        reservoir solve in this first single-liquid implementation.
        """
        if pump_speed_ratios or pump_status or valve_openings:
            return self.with_controls(pump_speed_ratios, pump_status, valve_openings).solve(
                source_mass_rates_kg_s=source_mass_rates_kg_s,
                boundary_pressures_pa=boundary_pressures_pa, initial=initial,
                pressure_tolerance_pa=pressure_tolerance_pa,
                mass_tolerance_kg_s=mass_tolerance_kg_s, max_iterations=max_iterations)
        ptol = _number(pressure_tolerance_pa, "pressure_tolerance_pa", positive=True)
        mtol = _number(mass_tolerance_kg_s, "mass_tolerance_kg_s", positive=True)
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        rho = self.fluid.density_kg_m3
        z = np.array([n.elevation_m for n in self.nodes])
        sources = np.array([n.source_mass_rate_kg_s for n in self.nodes], dtype=float)
        pressures = np.array(
            [np.nan if n.pressure_pa is None else n.pressure_pa for n in self.nodes], dtype=float
        )
        for mapping, target, pressure in [
            (source_mass_rates_kg_s, sources, False),
            (boundary_pressures_pa, pressures, True),
        ]:
            for name, value in (mapping or {}).items():
                if name not in self._index:
                    raise ValueError(f"Unknown boundary node: {name}")
                target[self._index[name]] = (
                    np.nan
                    if pressure and value is None
                    else _number(value, name, positive=pressure)
                )
        fixed = np.isfinite(pressures)
        if np.any(pressures[fixed] <= self.fluid.vapor_pressure_pa):
            raise InvalidPressureError("Pressure boundary is at or below vapor pressure")
        for component in np.unique(self._components):
            if not fixed[self._components == component].any():
                names = [
                    n.name for i, n in enumerate(self.nodes) if self._components[i] == component
                ]
                raise ValueError(f"No pressure boundary in connected component: {names}")
        free = np.flatnonzero(~fixed)
        Bf, B = self._B[free].tocsc(), self._B
        H = np.where(fixed, pressures + rho * G * z, 0.0)

        def initial_conductance(branch):
            if isinstance(branch, Pipe):
                return (
                    math.pi
                    * branch.diameter_m**4
                    / (128 * self.fluid.viscosity_pa_s * branch.length_m)
                )
            if isinstance(branch, Pump):
                return branch.max_flow_m3_s / (rho * G * branch.speed_ratio * branch.shutoff_head_m)
            area = (
                math.pi * branch.diameter_m**2 / 4 * branch.opening * branch.discharge_coefficient
            )
            return area * math.sqrt(2 / (rho * 1e5))

        if initial is None:
            conductance = np.array([initial_conductance(p) for p in self._active])
            offsets = np.array([p.loss_and_gradient(0.0, self.fluid)[0] for p in self._active])
            if len(free):
                A = Bf @ sp.diags(conductance) @ Bf.T
                H[free] = splu(A.tocsc()).solve(
                    sources[free] / rho - Bf @ (conductance * (B.T @ H - offsets))
                )
            q = conductance * (B.T @ H - offsets)
        else:
            if set(initial.pressures_pa) != set(self._index) or set(
                dict(
                    initial.pipe_mass_rates_kg_s,
                    **initial.pump_mass_rates_kg_s,
                    **initial.valve_mass_rates_kg_s,
                )
            ) != {p.name for p in self._branches}:
                raise ValueError("Initial result does not match network node/pipe names")
            H[free] = [
                _number(initial.pressures_pa[self.nodes[i].name], "initial pressure", positive=True)
                + rho * G * z[i]
                for i in free
            ]
            q = np.array(
                [
                    _number(
                        (
                            initial.pipe_mass_rates_kg_s
                            if isinstance(p, Pipe)
                            else initial.pump_mass_rates_kg_s
                            if isinstance(p, Pump)
                            else initial.valve_mass_rates_kg_s
                        )[p.name],
                        "initial rate",
                    )
                    / rho
                    for p in self._active
                ]
            )
        pscale = max(1e5, float(np.max(np.abs(H))))
        qscale = max(
            1e-8,
            float(np.abs(sources).sum()) / rho,
            max(
                (
                    math.pi * p.diameter_m**2 / 4
                    if not isinstance(p, Pump)
                    else p.speed_ratio * p.max_flow_m3_s
                    for p in self._active
                ),
                default=0.0,
            ),
        )
        mscale = max(1.0, rho * qscale)

        protected = np.array([p.check_valve for p in self._active], dtype=bool)

        def residual(head, rate):
            losses = np.array(
                [p.loss_and_gradient(float(v), self.fluid) for p, v in zip(self._active, rate)]
            ).reshape(-1, 2)
            driving = B.T @ head - losses[:, 0]
            # Complementarity: q >= 0, driving <= 0, q*driving = 0.
            # max(driving/pscale, -q/qscale)=0 admits a shut non-return valve
            # without introducing artificial leakage or large resistances.
            blocked = protected & (-rate / qscale >= driving / pscale)
            re = np.where(blocked, -rate * pscale / qscale, driving)
            rm = rho * (Bf @ rate) - sources[free]
            return losses, re, rm, np.concatenate((re / pscale, rm / mscale)), blocked

        for iteration in range(max_iterations + 1):
            losses, re, rm, r, blocked = residual(H, q)
            if not np.isfinite(r).all():
                raise HydraulicConvergenceError("Nonfinite hydraulic residual")
            if np.max(np.abs(re), initial=0.0) <= ptol and np.max(np.abs(rm), initial=0.0) <= mtol:
                if np.any(blocked & (q != 0)):
                    q[blocked] = 0.0
                    losses, re, rm, r, blocked = residual(H, q)
                if (
                    np.max(np.abs(re), initial=0.0) <= ptol
                    and np.max(np.abs(rm), initial=0.0) <= mtol
                ):
                    break
            if iteration == max_iterations:
                raise HydraulicConvergenceError(
                    f"Iteration limit: pressure={abs(re).max(initial=0.0)} Pa, "
                    f"mass={abs(rm).max(initial=0.0)} kg/s"
                )
            tangent = losses[:, 1].copy()
            for e, branch in enumerate(self._active):
                if tangent[e] == 0 and not blocked[e]:
                    # Exact quadratic/orifice residual; startup secant only.
                    tangent[e] = 1 / initial_conductance(branch)
            J = sp.bmat(
                [
                    [
                        sp.diags((~blocked).astype(float)) @ Bf.T,
                        -sp.diags(np.where(blocked, 1.0, tangent * qscale / pscale)),
                    ],
                    [sp.csc_matrix((len(free), len(free))), Bf * (rho * qscale / mscale)],
                ],
                format="csc",
            )
            try:
                step = splu(J).solve(-r)
            except RuntimeError as error:
                raise HydraulicConvergenceError("Singular hydraulic Jacobian") from error
            if not np.isfinite(step).all():
                raise HydraulicConvergenceError("Nonfinite hydraulic update")
            norm = np.linalg.norm(r)
            alpha = 1.0
            for _ in range(35):
                trial_H = H.copy()
                trial_H[free] += alpha * pscale * step[: len(free)]
                trial_q = q + alpha * qscale * step[len(free) :]
                _, _, _, trial_r, _ = residual(trial_H, trial_q)
                if (
                    np.isfinite(trial_r).all()
                    and np.linalg.norm(trial_r) <= (1 - 1e-4 * alpha) * norm
                ):
                    H, q = trial_H, trial_q
                    break
                alpha *= 0.5
            else:
                raise HydraulicConvergenceError("Hydraulic line search failed")
        if blocked.any():
            live = B[:, ~blocked]
            _, components = sp.csgraph.connected_components(abs(live) @ abs(live.T), directed=False)
            for component in np.unique(components):
                if not fixed[components == component].any():
                    raise EquipmentOperatingError(
                        "Pressure is undetermined in a component isolated by non-return valves"
                    )
        solved_pressures = H - rho * G * z
        solved_pressures[fixed] = pressures[fixed]
        pressures = solved_pressures
        if np.any(pressures <= self.fluid.vapor_pressure_pa):
            bad = [
                n.name
                for i, n in enumerate(self.nodes)
                if pressures[i] <= self.fluid.vapor_pressure_pa
            ]
            raise InvalidPressureError(f"Liquid pressure is at or below vapor pressure: {bad}")
        boundary = np.where(fixed, rho * (B @ q) - sources, 0.0)
        mass_residual = rho * (B @ q) - sources - boundary
        mass_rates = dict.fromkeys((p.name for p in self.pipes), 0.0)
        mass_rates.update(
            {p.name: float(v * rho) for p, v in zip(self._active, q) if isinstance(p, Pipe)}
        )
        pump_rates = dict.fromkeys((p.name for p in self.pumps), 0.0)
        hydraulic, shaft, npsh = {}, {}, {}
        required_npsh, npsh_margin, efficiencies, head_gains = {}, {}, {}, {}
        states = {p.name: "closed" for p in self._branches}
        valve_driving = {}
        raw_driving = B.T @ H - losses[:, 0]
        for p, v, shut, driving in zip(self._active, q, blocked, raw_driving):
            states[p.name] = "blocked" if shut else "open"
            if p.check_valve:
                valve_driving[p.name] = float(driving)
            if isinstance(p, Pump) and not shut:
                power, drive, available = p.operating_state(
                    float(v), pressures[self._index[p.start]], self.fluid
                )
                pump_rates[p.name] = float(v * rho)
                hydraulic[p.name], shaft[p.name] = float(power), float(drive)
                efficiencies[p.name] = float(p.efficiency_at(float(v)))
                head_gains[p.name] = float(
                    -p.loss_and_gradient(float(v), self.fluid)[0] / (rho * G)
                )
                required = p.required_npsh_at(float(v))
                if available is not None:
                    npsh[p.name] = float(available)
                if required is not None:
                    required_npsh[p.name] = float(required)
                    npsh_margin[p.name] = float(available - required)
        dissipation = {
            p.name: float(v * loss)
            for p, v, loss in zip(self._active, q, losses[:, 0])
            if isinstance(p, Pipe)
        }
        valve_rates = dict.fromkeys((v.name for v in self.valves), 0.0)
        valve_rates.update(
            {
                p.name: float(v * rho)
                for p, v in zip(self._active, q)
                if isinstance(p, ThrottleValve)
            }
        )
        valve_dissipation = {
            p.name: float(v * loss)
            for p, v, loss in zip(self._active, q, losses[:, 0])
            if isinstance(p, ThrottleValve)
        }
        return NetworkResult(
            pressures_pa={n.name: float(v) for n, v in zip(self.nodes, pressures)},
            pipe_mass_rates_kg_s=mass_rates,
            pipe_volume_rates_m3_s={name: v / rho for name, v in mass_rates.items()},
            boundary_mass_rates_kg_s={
                n.name: float(boundary[i]) for i, n in enumerate(self.nodes) if fixed[i]
            },
            node_mass_residuals_kg_s={n.name: float(v) for n, v in zip(self.nodes, mass_residual)},
            pipe_equation_residuals_pa={
                p.name: float(v) for p, v in zip(self._active, re) if isinstance(p, Pipe)
            },
            pipe_dissipation_w=dissipation,
            power_balance_error_w=float(
                H @ (sources + boundary) / rho
                + sum(hydraulic.values())
                - sum(dissipation.values())
                - sum(valve_dissipation.values())
            ),
            iterations=iteration,
            valve_mass_rates_kg_s=valve_rates,
            valve_volume_rates_m3_s={name: v / rho for name, v in valve_rates.items()},
            valve_equation_residuals_pa={
                p.name: float(v) for p, v in zip(self._active, re) if isinstance(p, ThrottleValve)
            },
            valve_dissipation_w=valve_dissipation,
            pump_mass_rates_kg_s=pump_rates,
            pump_volume_rates_m3_s={name: v / rho for name, v in pump_rates.items()},
            pump_equation_residuals_pa={
                p.name: float(v) for p, v in zip(self._active, re) if isinstance(p, Pump)
            },
            pump_hydraulic_power_w=hydraulic,
            pump_shaft_power_w=shaft,
            pump_npsh_available_m=npsh,
            pump_npsh_required_m=required_npsh,
            pump_npsh_margin_m=npsh_margin,
            pump_efficiencies=efficiencies,
            pump_head_gains_m=head_gains,
            branch_states=states,
            check_valve_driving_pressure_pa=valve_driving,
            operating_controls=self._equipment_controls(),
        )
