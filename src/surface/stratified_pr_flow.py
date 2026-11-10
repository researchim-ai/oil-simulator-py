"""Implicit quasi-static PR vessel with finite circular withdrawals.

Local aperture P/x, phase interfaces, fluid N/U+PE and wall heat exchange
share one Torch Newton system. Outlets are low-pressure-drop linear valves,
not choked nozzles. Prescribed feeds are boundary conditions, not pipe models.
"""

import math
from dataclasses import asdict, dataclass

import torch

from .component_energy import ComponentEnergyStream
from .compositional_separator import EquilibriumOutlet
from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    PropertyModelError,
    _number,
)
from .peng_robinson import R
from .phase_geometry import EquilibriumLevelControl, validate_pickup
from .stratified_pr import StratifiedPRColumn
from .stratified_pr_thermal import StratifiedPRThermal, StratifiedThermalState
from .three_phase_properties import disk_band_area
from .torch_linear import TorchNewton
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class StratifiedFeed:
    height_m: float
    stream: ComponentEnergyStream

    def __post_init__(self):
        _number(self.height_m, "feed height", nonnegative=True)
        if not isinstance(self.stream, ComponentEnergyStream):
            raise TypeError("Expected ComponentEnergyStream")


class StratifiedPRFlow(StratifiedPRThermal):
    def __init__(self, column, wall, *, outlets=(), feeds=(), **integration):
        self.outlets, self.feeds = tuple(outlets), tuple(feeds)
        names = []
        for p in self.outlets:
            if not isinstance(p, EquilibriumOutlet) or p.height_m is None:
                raise TypeError("Finite-aperture EquilibriumOutlet required")
            validate_pickup(column.geometry, p.height_m, p.diameter_m)
            names.append(p.name)
            for component in (p.component, p.level_control.component if p.level_control else None):
                if component is not None and component not in column.fluid.names:
                    raise ValueError("Unknown outlet/control component")
            if p.level_control and p.level_control.setpoint_m > column.geometry.height_m:
                raise ValueError("Level setpoint above vessel")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate outlet names")
        for feed in self.feeds:
            if not isinstance(feed, StratifiedFeed):
                raise TypeError("Expected StratifiedFeed")
            feed.stream.validate_fluid(column.fluid)
            if feed.height_m > column.geometry.height_m:
                raise ValueError("Feed above vessel")
            if (
                abs(feed.stream.elevation_m - column.elevation_m - feed.height_m) > 1e-9
                or feed.stream.gravity_m_s2 != column.gravity_m_s2
            ):
                raise ValueError("Feed elevation/gravity differs; transport must be explicit")
        super().__init__(column, wall, **integration)

    def to_dict(self):
        data = super().to_dict()
        data.update(
            model="stratified_pr_flow_v1",
            outlets=[asdict(p) for p in self.outlets],
            feeds=[asdict(feed) for feed in self.feeds],
        )
        return data

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "stratified_pr_flow_v1":
            raise ValueError("Unknown flow model")
        data["column"] = StratifiedPRColumn.from_dict(data["column"], device=device)
        data["wall"] = VesselWall(**data["wall"])
        outlets = []
        for raw in data["outlets"]:
            raw = dict(raw)
            if raw.get("level_control") is not None:
                raw["level_control"] = EquilibriumLevelControl(**raw["level_control"])
            outlets.append(EquilibriumOutlet(**raw))
        data["outlets"] = outlets
        data["feeds"] = [
            StratifiedFeed(v["height_m"], ComponentEnergyStream(**v["stream"]))
            for v in data["feeds"]
        ]
        return cls(**data)

    def _aperture(self, boundaries, phase, port, order):
        """Angle quadrature, normalized to exact occupied area (including dry caps)."""
        radius = port.diameter_m / 2
        low = torch.maximum(boundaries[phase], boundaries.new_tensor(port.height_m - radius))
        high = torch.minimum(boundaries[phase + 1], boundaries.new_tensor(port.height_m + radius))
        high = torch.maximum(high, low)

        def angle(h):
            z = (h - port.height_m) / radius
            interior = torch.asin(z.clamp(-1 + 1e-12, 1 - 1e-12))
            return torch.where(
                z <= -1,
                z.new_tensor(-math.pi / 2),
                torch.where(z >= 1, z.new_tensor(math.pi / 2), interior),
            )

        a, b = angle(low), angle(high)
        nodes, weights = self.column._rules[order]
        theta = (a + b) / 2 + (b - a) * nodes / 2
        mirror = low >= port.height_m
        band_low = torch.where(mirror, port.height_m + radius - high, low - port.height_m + radius)
        band_high = torch.where(mirror, port.height_m + radius - low, high - port.height_m + radius)
        fraction = disk_band_area(band_low, band_high, port.diameter_m) / (math.pi * radius**2)
        raw = weights * torch.cos(theta) ** 2
        ww = fraction * raw / raw.sum().clamp_min(1e-30)
        hh = torch.where(
            fraction > 0,
            port.height_m + radius * torch.sin(theta),
            (boundaries[phase] + boundaries[phase + 1]) / 2,
        )
        return hh, ww

    def _problem(self, state, dt, seed, order):
        c, f, w = self.column, self.column.fluid, self.wall
        old = state.column
        n0 = f.tensor(old.component_moles)
        feed_n = sum(
            (f.tensor(v.stream.component_molar_rates_mol_s) for v in self.feeds),
            torch.zeros_like(n0),
        )
        feed_e = math.fsum(v.stream.total_energy_rate_w for v in self.feeds)
        feed_mass = float((feed_n * f.molar_masses).sum().cpu())
        nref = n0 + dt * feed_n
        anchor = old.total_energy_j + dt * feed_mass * c.gravity_m_s2 * c.elevation_m
        initial, lower, upper, base, active, roots = c._problem(nref, anchor, seed, order)
        count, size = len(active), len(initial)
        nscale = nref[active].clamp_min(nref.sum() * 1e-12)
        relative_old = (
            old.total_energy_j
            - float((n0 * f.molar_masses).sum().cpu()) * c.gravity_m_s2 * c.elevation_m
        )
        escale = max(1.0, abs(relative_old), float(nref.sum().cpu()) * R * seed[4])
        local = []
        for _ in self.outlets:
            for x in seed[0]:
                z = x[active].clamp_min(1e-30)
                row = torch.cat((f.tensor([math.log(seed[3])]), torch.log(z[:-1] / z[-1])))
                local.append(row.repeat(order))
        if local:
            initial = torch.cat((initial, *local))
            low = torch.cat(
                (f.tensor([math.log(f.min_pressure_pa)]), f.tensor([-100.0] * (count - 1)))
            )
            high = torch.cat(
                (f.tensor([math.log(f.max_pressure_pa)]), f.tensor([100.0] * (count - 1)))
            )
            lower = torch.cat((lower, low.repeat(order * len(roots) * len(self.outlets))))
            upper = torch.cat((upper, high.repeat(order * len(roots) * len(self.outlets))))
        if len(initial) > 768:
            raise ValueError("Flow collocation exceeds 768-unknown limit")

        def evaluate(y):
            r, data = base(y[:size])
            t = data["temperature"]
            tw = (
                w.heat_capacity_j_k * state.wall_temperature_k
                + dt * (w.fluid_ua_w_k * t + w.ambient_ua_w_k * w.ambient_temperature_k)
            ) / (w.heat_capacity_j_k + dt * (w.fluid_ua_w_k + w.ambient_ua_w_k))
            heat = w.fluid_ua_w_k * (tw - t)
            ambient = w.ambient_ua_w_k * (w.ambient_temperature_k - tw)
            bounds = torch.cat(
                (t.new_zeros(1), data["interfaces"], t.new_tensor([c.geometry.height_m]))
            )
            phase_n = data["phase_component_moles"]
            mean_x = phase_n / phase_n.sum(-1, keepdim=True).clamp_min(1e-30)
            density = (phase_n * f.molar_masses).sum(-1) / torch.stack(
                [c._volume(bounds[i + 1]) - c._volume(bounds[i]) for i in range(len(roots))]
            ).clamp_min(1e-30)
            out_n, out_e, out_s = torch.zeros_like(n0), t.new_zeros(()), t.new_zeros(())
            chemical, ports = [], []
            offset = size
            for port in self.outlets:
                pieces = []
                for phase, root in enumerate(roots):
                    h, area = self._aperture(bounds, phase, port, order)
                    local_y = y[offset : offset + order * count].reshape(order, count)
                    offset += order * count
                    p, x, q = c._decode_nodes(local_y, active, t, root)
                    mu = (
                        torch.log(x[:, active].clamp_min(1e-300))
                        + q["log_fugacity_coefficients"][:, active]
                        + torch.log(p / f.pref)[:, None]
                    )
                    chemical.append(
                        (
                            mu
                            + c.gravity_m_s2 * h[:, None] * f.molar_masses[active] / (R * t)
                            - data["constants"]
                        ).flatten()
                    )
                    pieces.append((h, area, p, x, q))
                mean_p = sum((p * area).sum() for h, area, p, x, q in pieces)
                opening = t.new_tensor(port.opening)
                level = t.new_zeros(())
                if port.pressure_setpoint_pa is not None:
                    opening = (
                        port.opening_bias
                        + port.opening_gain_per_pa * (mean_p - port.pressure_setpoint_pa)
                    ).clamp(0, 1)
                if port.level_control is not None:
                    opening, level, _ = port.level_control.value(
                        {"phase_tops_m": bounds[1:]}, mean_x, f.names
                    )
                pn, pe, ps, volume = (
                    torch.zeros_like(n0),
                    t.new_zeros(()),
                    t.new_zeros(()),
                    t.new_zeros(()),
                )
                valid = t.new_tensor(True, dtype=torch.bool)
                for phase, (h, area, p, x, q) in enumerate(pieces):
                    selected = t.new_tensor(True, dtype=torch.bool)
                    if port.selection in ("densest", "lightest"):
                        index = (
                            density.argmax() if port.selection == "densest" else density.argmin()
                        )
                        selected = index == phase
                    if port.selection == "component_rich":
                        fraction = mean_x[:, f.names.index(port.component)]
                        selected = (
                            (fraction.argmax() == phase)
                            & (fraction[phase] >= port.minimum_mole_fraction)
                            & (fraction[phase] > 0)
                        )
                    vdot = (
                        port.volume_conductance_m3_s_pa
                        * opening
                        * area
                        * (p - port.back_pressure_pa).clamp_min(0)
                        * selected
                    )
                    rate = vdot / q["molar_volume_m3_mol"]
                    pn = pn + (rate[:, None] * x).sum(0)
                    pe = (
                        pe
                        + (
                            rate
                            * (
                                q["molar_enthalpy_j_mol"]
                                + (x * f.molar_masses).sum(-1) * c.gravity_m_s2 * h
                            )
                        ).sum()
                    )
                    ps = ps + (rate * q["molar_entropy_j_mol_k"]).sum()
                    volume = volume + vdot.sum()
                    valid = (
                        valid
                        & q["valid"].all()
                        & (
                            (
                                (p - port.back_pressure_pa) / p
                                <= port.maximum_pressure_drop_fraction + 1e-12
                            )
                            | (vdot == 0)
                        ).all()
                    )
                out_n, out_e, out_s = out_n + pn, out_e + pe, out_s + ps
                ports.append(
                    {
                        "pieces": pieces,
                        "component_rates": pn,
                        "relative_energy_rate": pe,
                        "entropy_rate": ps,
                        "volume_rate": volume,
                        "opening": opening,
                        "level": level,
                        "valid": valid,
                    }
                )
            feed_relative = feed_e - feed_mass * c.gravity_m_s2 * c.elevation_m
            residual = torch.cat(
                (
                    r[: -count - 1],
                    r[-count - 1 : -1] + dt * out_n[active] / nscale,
                    (r[-1] - dt * (feed_relative - out_e + heat) / escale).reshape(1),
                    *chemical,
                )
            )
            return residual, {
                "data": data,
                "column_variables": y[:size],
                "wall": tw,
                "heat": heat,
                "ambient": ambient,
                "out_n": out_n,
                "out_e": out_e,
                "out_s": out_s,
                "ports": ports,
            }

        return initial, lower, upper, evaluate, active, roots, feed_n, feed_e

    def _step(self, state, dt):
        c, f, w = self.column, self.column.fluid, self.wall
        old = state.column
        feed_n = sum(
            (f.tensor(v.stream.component_molar_rates_mol_s) for v in self.feeds),
            f.tensor([0.0] * len(f.names)),
        )
        feed_e = math.fsum(v.stream.total_energy_rate_w for v in self.feeds)
        # Old topology supplies an inexpensive flow predictor, never accepted stocks.
        old_seed = (
            [f.tensor(x) for x in old.phase_seeds],
            list(old.phase_roots),
            f.tensor(old.interface_heights_m),
            math.exp(old.solution_variables[0]),
            old.temperature_k,
            f.tensor(old.chemical_constants),
        )
        guess, _, _, predictor, *_ = self._problem(state, dt, old_seed, c.quadrature_order)
        with torch.no_grad():
            estimate = predictor(guess)[1]
        predicted_n = f.tensor(old.component_moles) + dt * (feed_n - estimate["out_n"])
        if not bool((predicted_n >= 0).all().cpu()) or not bool((predicted_n.sum() > 0).cpu()):
            raise HydraulicConvergenceError("Predictor exhausts component inventory")
        mass_out = float((estimate["out_n"] * f.molar_masses).sum().cpu())
        estimated_out_e = float(estimate["out_e"].cpu()) + mass_out * c.gravity_m_s2 * c.elevation_m
        cv = max(
            1e-12,
            sum(
                n * max(1.0, component.ideal_cp_j_mol_k - R)
                for n, component in zip(old.component_moles, f.components)
            ),
        )
        # Stable heat predictor; the final residual uses the undamped implicit heat.
        predicted_e = (
            old.total_energy_j
            + dt * (feed_e - estimated_out_e)
            + dt * float(estimate["heat"].cpu()) / (1 + dt * w.fluid_ua_w_k / cv)
        )
        seed = c._seed(
            predicted_n, predicted_e, math.exp(old.solution_variables[0]), old.temperature_k
        )
        incoming_s = math.fsum(
            v.stream.molar_rate_mol_s
            * float(
                v.stream.equilibrate(f, temperature_guess_k=old.temperature_k)[
                    "molar_entropy_j_mol_k"
                ].cpu()
            )
            for v in self.feeds
            if v.stream.molar_rate_mol_s
        )
        previous = None
        order = c.quadrature_order
        topology_attempts = 0
        while order <= c.maximum_quadrature_order:
            initial, lower, upper, evaluate, active, roots, feed_n, feed_e = self._problem(
                state, dt, seed, order
            )
            solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=768)
            solver.host_convergence = True
            with torch.no_grad():
                y, r, ok, _ = solver.solve(
                    lambda z, evaluate=evaluate: evaluate(z)[0],
                    initial,
                    lower,
                    upper,
                    tolerance=3e-12,
                )
                r, result = evaluate(y)
            if not bool((ok & torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()):
                raise HydraulicConvergenceError("Implicit flow column Newton rejected")
            data = result["data"]
            for port in result["ports"]:
                if not bool(port["valid"].cpu()):
                    raise EquipmentOperatingError("Outlet outside low-pressure-drop/EOS domain")
            n = f.tensor(old.component_moles) + dt * (feed_n - result["out_n"])
            if not bool((n >= 0).all().cpu()) or not bool((n.sum() > 0).cpu()):
                raise HydraulicConvergenceError("Outlet exhausts component inventory")
            tw = float(result["wall"].cpu())
            if (
                not w.min_temperature_k <= tw <= w.max_temperature_k
                or abs(tw - state.wall_temperature_k) > w.max_temperature_change_k
            ):
                raise HydraulicConvergenceError("Wall step/domain limit exceeded")
            out_mass = float((result["out_n"] * f.molar_masses).sum().cpu())
            out_e = float(result["out_e"].cpu()) + out_mass * c.gravity_m_s2 * c.elevation_m
            heat, ambient = float(result["heat"].cpu()), float(result["ambient"].cpu())
            energy = old.total_energy_j + dt * (feed_e - out_e + heat)
            constants = torch.zeros_like(n).index_copy(0, active, data["constants"])
            seed = (
                [x[0] for x in data["compositions"][: len(roots)]],
                roots,
                data["interfaces"],
                float(data["pressures"][0][0].cpu()),
                float(data["temperature"].cpu()),
                constants,
            )
            try:
                candidate, report = c._solve(n, energy, seed, order)
            except PropertyModelError:
                topology_attempts += 1
                if topology_attempts > 4:
                    raise HydraulicConvergenceError(
                        "Flow phase-topology retry budget exhausted"
                    ) from None
                seed = c._seed(n, energy, seed[3], seed[4])
                previous = None
                continue
            for port in result["ports"]:
                for h, area, p, x, q in port["pieces"]:
                    for j in range(order):
                        c._check_local(
                            float(p[j].cpu()),
                            float(data["temperature"].cpu()),
                            x[j],
                            {k: v[j] for k, v in q.items()},
                        )
            if abs(candidate.temperature_k - float(data["temperature"].cpu())) > 1e-6:
                raise ConservationError("Flow/equilibrium temperatures differ")
            if max(
                abs(a - b) for a, b in zip(candidate.chemical_constants, constants.cpu().tolist())
            ) > 1e-7 or any(
                abs(a - b) > 1e-8 * c.geometry.height_m
                for a, b in zip(candidate.interface_heights_m, data["interfaces"].cpu().tolist())
            ):
                raise ConservationError("Accepted profile differs from implicit port solution")
            # Fixed feed carries its own enthalpy/kinetic/PE. No unmodelled pump work.
            for feed in self.feeds:
                if feed.stream.molar_rate_mol_s:
                    local = c.sample(candidate, feed.height_m, side="below")
                    if (
                        local["pressure_pa"]
                        > feed.stream.pressure_pa + 1e-7 * feed.stream.pressure_pa
                    ):
                        raise EquipmentOperatingError(
                            "Feed pressure below receiver; explicit pump required"
                        )
            if previous is not None:
                prior, prior_report, prior_out, prior_e, prior_ports = previous
                if len(prior.phase_roots) != len(candidate.phase_roots):
                    raise HydraulicConvergenceError("Flow quadrature topology changed")
                error = max(
                    abs(candidate.temperature_k - prior.temperature_k)
                    / (c.temperature_tolerance_k + c.relative_tolerance * candidate.temperature_k),
                    abs(report["mean_pressure_pa"] - prior_report["mean_pressure_pa"])
                    / (1 + c.relative_tolerance * report["mean_pressure_pa"]),
                    float(
                        (
                            dt
                            * (result["out_n"] - prior_out).abs()
                            / (1e-9 + c.relative_tolerance * n.sum())
                        )
                        .max()
                        .cpu()
                    ),
                    abs(dt * (float(result["out_e"].cpu()) - prior_e))
                    / (
                        1e-5
                        + c.relative_tolerance
                        * max(
                            1,
                            abs(
                                energy
                                - float((n * f.molar_masses).sum().cpu())
                                * c.gravity_m_s2
                                * c.elevation_m
                            ),
                        )
                    ),
                )
                error = max(
                    error,
                    abs(
                        report["relative_potential_energy_j"]
                        - prior_report["relative_potential_energy_j"]
                    )
                    / (
                        1e-5
                        + c.relative_tolerance
                        * max(1.0, abs(report["relative_potential_energy_j"]))
                    ),
                )
                for a, b in zip(candidate.interface_heights_m, prior.interface_heights_m):
                    error = max(
                        error, abs(a - b) / (1e-8 + c.relative_tolerance * c.geometry.height_m)
                    )
                error = max(
                    error,
                    float(
                        (
                            (
                                f.tensor(report["phase_component_moles"])
                                - f.tensor(prior_report["phase_component_moles"])
                            ).abs()
                            / (1e-9 + c.relative_tolerance * n.sum())
                        )
                        .max()
                        .cpu()
                    ),
                )
                for current_port, prior_port in zip(result["ports"], prior_ports):
                    rate, old_rate = current_port["component_rates"], prior_port["component_rates"]
                    error = max(
                        error,
                        float(
                            (
                                (rate - old_rate).abs()
                                / (
                                    1e-14
                                    + c.relative_tolerance
                                    * torch.maximum(rate.abs(), old_rate.abs())
                                )
                            )
                            .max()
                            .cpu()
                        ),
                    )
                if error <= 1:
                    entropy_old = float(self._data(old)["entropy_j_k"].cpu())
                    entropy = (
                        report["entropy_j_k"]
                        - entropy_old
                        + w.heat_capacity_j_k * math.log(tw / state.wall_temperature_k)
                        - dt
                        * (
                            incoming_s
                            - float(result["out_s"].cpu())
                            + ambient / w.ambient_temperature_k
                        )
                    )
                    if entropy < -1e-6 - 1e-9 * max(
                        1, abs(entropy_old), abs(report["entropy_j_k"])
                    ):
                        raise ConservationError("Open vessel entropy balance rejected")
                    imbalance = (
                        candidate.total_energy_j
                        - old.total_energy_j
                        + w.heat_capacity_j_k * (tw - state.wall_temperature_k)
                        - dt * (feed_e - out_e + ambient)
                    )
                    if abs(imbalance) > 1e-5 + 1e-9 * max(
                        1, abs(dt * feed_e), abs(dt * out_e), abs(dt * ambient)
                    ):
                        raise ConservationError("Open vessel energy balance rejected")
                    return StratifiedThermalState(
                        candidate, tw, state.time_s + dt, state.next_step_s, self.model_id
                    ), {
                        "dt_s": dt,
                        "column_state": candidate.to_dict(),
                        "wall_temperature_k": tw,
                        "ambient_heat_j": dt * ambient,
                        "fluid_heat_j": dt * heat,
                        "energy_residual_j": imbalance,
                        "entropy_production_j_k": entropy,
                        "feed_energy_j": dt * feed_e,
                        "outlet_energy_j": dt * out_e,
                        "feed_component_moles": (dt * feed_n).cpu().tolist(),
                        "outlet_component_moles": (dt * result["out_n"]).cpu().tolist(),
                        "quadrature_error_ratio": error,
                        "phase_count": len(roots),
                        "outlets": [
                            {
                                "name": p.name,
                                "opening": float(v["opening"].cpu()),
                                "volume_rate_m3_s": float(v["volume_rate"].cpu()),
                                "component_rates_mol_s": v["component_rates"].cpu().tolist(),
                                "total_energy_rate_w": float(v["relative_energy_rate"].cpu())
                                + float((v["component_rates"] * f.molar_masses).sum().cpu())
                                * c.gravity_m_s2
                                * c.elevation_m,
                                "phase_area_fractions": [
                                    float(piece[1].sum().cpu()) for piece in v["pieces"]
                                ],
                            }
                            for p, v in zip(self.outlets, result["ports"])
                        ],
                    }
            previous = (
                candidate,
                report,
                result["out_n"],
                float(result["out_e"].cpu()),
                result["ports"],
            )
            order *= 2
        raise HydraulicConvergenceError("Flow quadrature budget exhausted")
