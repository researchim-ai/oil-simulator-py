"""Conservative end-intrinsic traces with genuinely zero owned phase stocks.

The positive-support equations are those of PRCapillaryAxialPipe. Density,
temperature and velocity are finite transport traces, including an outgoing
phase that disappears at the endpoint. Owned N/V/M/E are volume times those
traces: zero volume owns exactly zero stock, never a residual film.
"""

import math
from dataclasses import asdict

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_active_multifluid_pipe import PRActiveMultifluidState
from .pr_batched_homogeneous import homogeneous_batch
from .pr_capillary_linear import CapillaryLinearSystem
from .pr_homogeneous_uv import caloric_tv, recover_homogeneous_phases
from .pr_interphase_relaxation import PRRelaxationPhase


def active_step(model, state, dt_s, *, phase_controls=None):
    p, s, f = model.pipe, model.source, model.pipe.fluid
    if s.wetting is not None and s.wetting.scale > 0 and s.wetting.dry_extension is None:
        raise ValueError("Active dry states require an explicit finite dry-wall material extension")
    if hasattr(model, "faces"):
        raise ValueError("Active intrinsic traces currently require a physical line")
    before = p.inspect(state)
    if _digest(model.to_dict()) != model.model_id or _digest(s.to_dict()) != s.model_id:
        raise ValueError("Changed active capillary constitutive model")
    initial_surface = s.surface_energy(state)
    dt = _number(dt_s, "active capillary interval", positive=True)
    clock = state.time_s + dt
    dt = clock - state.time_s
    if not math.isfinite(clock) or dt <= 0:
        raise ValueError("Active capillary interval below clock resolution")
    n, nc = len(state.cells), len(f.names)
    width = 2 * nc + 5
    if n * width > model.maximum_unknowns:
        raise ValueError("Active capillary unknown budget exceeded")
    names = (s.liquid, s.gas)
    cells = [dict(row) for row in state.cells]
    if any(set(row) - set(names) for row in cells):
        raise ValueError("Exactly the declared liquid/gas phase registry required")
    volume = f.tensor(p.volumes)
    old_n = f.tensor(
        [
            [row[name].component_moles if name in row else [0.0] * nc for name in names]
            for row in cells
        ]
    )
    old_v = f.tensor(
        [[row[name].volume_m3 if name in row else 0.0 for name in names] for row in cells]
    )
    old_m = f.tensor(
        [[row[name].momentum_kg_m_s if name in row else 0.0 for name in names] for row in cells]
    )
    old_e = f.tensor(
        [[row[name].total_energy_j if name in row else 0.0 for name in names] for row in cells]
    )
    alpha0 = old_v / volume[:, None]
    h0 = old_v[:, 0] / s.area
    controls, normalized = {}, {}
    donors = {
        name: [(i, row[name]) for i, row in enumerate(cells) if name in row] for name in names
    }
    for side, row in enumerate(model.boundaries):
        if row is not None:
            for name, phase in row.items():
                donors[name].append((-1 if side == 0 else n, phase))
    for key, command in ({} if phase_controls is None else phase_controls).items():
        i = int(key)
        if str(i) != str(key):
            raise ValueError("Phase controller cell names must be integer indices")
        if (
            i in controls
            or not 0 <= i < n
            or set(command) != {"inlet_slot", "inlet_phase", "target_liquid_fraction"}
        ):
            raise ValueError("One explicit phase-exchange controller per known cell required")
        name = command["inlet_slot"]
        if name not in names:
            raise ValueError("Unknown side-port inlet phase")
        phase = command["inlet_phase"]
        if not isinstance(phase, PRRelaxationPhase):
            phase = PRRelaxationPhase.from_dict(phase)
        p._boundary_packet({name: phase})
        target = _number(command["target_liquid_fraction"], "controlled fraction", nonnegative=True)
        if target > 1:
            raise ValueError("Controlled fraction must lie in [0,1]")
        j = names.index(name)
        if (1 if j == 0 else -1) * (target - float(alpha0[i, 0].cpu())) < 0:
            raise ValueError("Inlet direction differs from requested phase displacement")
        rho = f.tensor(phase.component_moles) / phase.volume_m3
        q = f.homogeneous(
            phase.pressure_pa, phase.temperature_k, rho / rho.sum(), p.slots[j]["root"]
        )
        controls[i] = (
            j,
            target,
            rho,
            phase.momentum_kg_m_s / phase.volume_m3,
            (phase.total_energy_j / phase.volume_m3 + phase.pressure_pa),
            float((q["molar_entropy_j_mol_k"] * rho.sum()).cpu()),
        )
        normalized[str(i)] = {**command, "inlet_phase": asdict(phase)}
        donors[name].append((i, phase))
    present = [bool(donors[name]) for name in names]
    proto = []
    for i, row in enumerate(cells):
        proto.append(
            [
                row.get(name)
                or (min(donors[name], key=lambda item: abs(item[0] - i))[1] if present[j] else None)
                for j, name in enumerate(names)
            ]
        )
    rho0 = f.tensor(
        [
            [
                [phase.component_moles[k] / phase.volume_m3 for k in range(nc)]
                if phase is not None
                else [0.0] * nc
                for phase in row
            ]
            for row in proto
        ]
    )
    t0 = f.tensor(
        [[phase.temperature_k if phase is not None else f.tref for phase in row] for row in proto]
    )
    mw = f.molar_masses
    u0 = f.tensor(
        [
            [
                phase.momentum_kg_m_s
                / math.fsum(
                    a * b.molar_mass_kg_mol for a, b in zip(phase.component_moles, f.components)
                )
                if phase is not None
                else 0.0
                for phase in row
            ]
            for row in proto
        ]
    )
    donor_density = f.tensor(
        [
            [
                max(
                    (phase.component_moles[k] / phase.volume_m3 for _, phase in donors[name]),
                    default=0.0,
                )
                for k in range(nc)
            ]
            for name in names
        ]
    )
    component_support = donor_density > 0
    rscale = torch.maximum(rho0, donor_density[None] * 1e-4)
    uscale = torch.maximum(u0.abs(), f.tensor(1.0))
    mixing = min(0.1, dt * p.wave_speed / min(p.lengths))
    guess = (1 - mixing) * alpha0[:, 0] + mixing * alpha0[:, 0].mean()
    for i, (_j, target, *_) in controls.items():
        guess[i] = target
    fractions = torch.stack((guess, 1 - guess), -1)
    time_fraction = torch.maximum(alpha0, fractions).clamp_min(1e-6)  # equation units only
    nscale = torch.where(
        component_support[None], rscale * volume[:, None, None] * time_fraction[..., None], 1.0
    )
    mscale = (rscale * mw).sum(-1) * volume[:, None] * time_fraction * uscale
    mscale = torch.where(mscale > 0, mscale, 1.0)
    escale = torch.maximum(
        old_e.abs(), rscale.sum(-1) * volume[:, None] * time_fraction * 8.31446261815324 * t0
    ).clamp_min(1e-30)
    controlled = torch.tensor([i in controls for i in range(n)], device=f.device)
    targets = f.tensor([controls[i][1] if i in controls else 0.0 for i in range(n)])
    z = f.tensor([0.0] * (n * width)).reshape(n, width)
    z[:, 2 * nc] = guess
    for i, (_j, target, *_) in controls.items():
        z[i, 2 * nc] = abs(target - float(alpha0[i, 0].cpu()))
    z = z.flatten()

    def evaluate(point):
        rows = point.reshape(n, width)
        rho = rho0 + rscale * rows[:, : 2 * nc].reshape(n, 2, nc)
        rho = torch.where(component_support[None], rho, 0.0)
        u = u0 + uscale * rows[:, 2 * nc + 1 : 2 * nc + 3]
        temp = t0 * torch.exp(rows[:, -2:])
        a = torch.where(controlled, targets, rows[:, 2 * nc])
        if not present[0]:
            a = a * 0
        if not present[1]:
            a = a * 0 + 1
        alpha = torch.stack((a, 1 - a), -1)
        vm = volume[:, None] * alpha
        amount = rho.sum(-1)
        mass = (rho * mw).sum(-1)
        pressure = []
        internal = []
        stability = []
        cv = []
        for j in range(2):
            if present[j]:
                safe = amount[:, j].clamp_min(1e-100)
                eos = caloric_tv(f, temp[:, j], 1 / safe, rho[:, j] / safe[:, None])
                pressure.append(eos["p"])
                internal.append(amount[:, j] * eos["u"])
                stability.append(eos["dpdv"])
                cv.append(eos["cv"])
            else:
                zero = amount[:, j] * 0
                pressure.append(zero)
                internal.append(zero)
                stability.append(zero - 1)
                cv.append(zero + 1)
        pressure = torch.stack(pressure, -1)
        internal = torch.stack(internal, -1)
        eden = internal + 0.5 * mass * u.square()
        numbers = vm[..., None] * rho
        momenta = vm * mass * u
        energy = vm * eden
        amid = (alpha + alpha0) / 2
        q = amid[..., None] * torch.cat((rho, (mass * u)[..., None], eden[..., None]), -1)
        flux = torch.cat(
            (
                q[..., :nc] * u[..., None],
                (q[..., -2] * u + amid * pressure)[..., None],
                ((q[..., -1] + amid * pressure) * u)[..., None],
            ),
            -1,
        )
        packet = {
            "alpha": amid,
            "end_alpha": amid,
            "conserved": q,
            "velocity": u,
            "temperature": temp,
            "pressure": pressure,
            "pressure_precision": pressure * 0,
            "flux": flux,
        }
        spatial = model._spatial(packet, h0, vm[:, 0] / s.area)
        (
            delta,
            physical,
            diff,
            force,
            capheat,
            _wall,
            _external,
            _capwork,
            _capimpulse,
            _boundary_alpha,
        ) = spatial
        source_n = torch.zeros_like(numbers)
        source_m = torch.zeros_like(momenta)
        source_e = torch.zeros_like(energy)
        streams = q.new_zeros(nc + 2)
        actuator = q.sum() * 0
        source_vol = physical * 0
        port_data = []
        chemical = s.discrete_pressure(h0, vm[:, 0] / s.area)
        for i, (j, target, rin, minlet, hinlet, sinlet) in controls.items():
            rate = rows[i, 2 * nc] * volume[i] / dt
            out = 1 - j
            qvl = rate if j == 0 else -rate
            inj = rate * rin
            withdraw = rate * rho[i, out]
            source_n = source_n.index_add(
                0,
                torch.tensor([i], device=f.device),
                torch.stack((inj if j == 0 else -withdraw, -withdraw if j == 0 else inj))[None],
            )
            dm = rate * minlet - rate * mass[i, out] * u[i, out]
            dh = rate * hinlet - rate * (eden[i, out] + pressure[i, out])
            phase_m = torch.stack(
                (
                    rate * minlet if j == 0 else -rate * mass[i, out] * u[i, out],
                    -rate * mass[i, out] * u[i, out] if j == 0 else rate * minlet,
                )
            )
            phase_e = torch.stack(
                (
                    rate * hinlet if j == 0 else -rate * (eden[i, out] + pressure[i, out]),
                    -rate * (eden[i, out] + pressure[i, out]) if j == 0 else rate * hinlet,
                )
            )
            qv = torch.stack((qvl, -qvl))
            source_m = source_m.index_add(0, torch.tensor([i], device=f.device), phase_m[None])
            source_e = source_e.index_add(
                0, torch.tensor([i], device=f.device), (phase_e - pressure[i] * qv)[None]
            )
            streams = streams + dt * torch.cat((inj - withdraw, dm.reshape(1), dh.reshape(1)))
            work = dt * qvl * (chemical[i] - pressure[i, 0] + pressure[i, 1])
            actuator = actuator + work
            source_vol = source_vol.index_add(0, torch.tensor([i], device=f.device), qvl.reshape(1))
            port_data.append((i, rate, work, qv, sinlet))
        rates = delta * p.area
        rn = (numbers - old_n + dt * (rates[..., :nc] - source_n)) / nscale
        rm = (momenta - old_m + dt * (rates[..., -2] - force - source_m)) / mscale
        heat_share = torch.where(
            (amid > 0).all(-1)[:, None], f.tensor([s.share, 1 - s.share]), (amid > 0).to(amid.dtype)
        )
        re = (
            energy
            - old_e
            + dt * (rates[..., -1] - u * force - capheat[:, None] * heat_share - source_e)
        ) / escale
        rv = (vm[:, 0] - old_v[:, 0] - dt * (physical + diff + source_vol)) / volume
        physical_res = torch.cat((rn.reshape(n, 2 * nc), rv[:, None], rm, re), -1).flatten()
        active = amid > 0
        rn = torch.where(
            active[..., None] & component_support[None], rn, rows[:, : 2 * nc].reshape(n, 2, nc)
        )
        rm = torch.where(active, rm, rows[:, 2 * nc + 1 : 2 * nc + 3])
        re = torch.where(active, re, rows[:, -2:])
        if not all(present):
            rv = torch.where(controlled, rv, rows[:, 2 * nc] - (0.0 if not present[0] else 1.0))
        residual = torch.cat((rn.reshape(n, 2 * nc), rv[:, None], rm, re), -1).flatten()
        valid = (
            torch.isfinite(residual).all()
            & torch.isfinite(physical_res).all()
            & (rho >= 0).all()
            & (alpha >= 0).all()
            & (alpha <= 1).all()
            & (torch.stack(stability, -1) < 0).all()
            & (torch.stack(cv, -1) > 0).all()
        )
        for j in range(2):
            if present[j]:
                valid = (
                    valid
                    & (amount[:, j] > 0).all()
                    & (temp[:, j] >= f.min_temperature_k).all()
                    & (temp[:, j] <= f.max_temperature_k).all()
                    & (pressure[:, j] >= f.min_pressure_pa).all()
                    & (pressure[:, j] <= f.max_pressure_pa).all()
                )
        if controls:
            valid = valid & (rows[controlled, 2 * nc] >= 0).all()
        valid = (
            valid
            & (
                (s.height_jump(vm[:, 0] / s.area) + s.height_offsets).abs() / s.distances
                <= s.maximum_slope
            ).all()
        )
        return (
            residual,
            physical_res,
            valid,
            numbers,
            vm,
            momenta,
            energy,
            temp,
            pressure,
            rho,
            u,
            packet,
            spatial,
            streams,
            actuator,
            port_data,
        )

    if getattr(model, "_linear", None) is None:
        model._linear = CapillaryLinearSystem(
            [list(range(i * width, (i + 1) * width)) for i in range(n)],
            s.geometry_edges,
            f.device,
            radius=2,
            mode=model.linear_solver,
        )
    linear = model._linear
    linear.solves = linear.krylov_restarts = 0
    linear.maximum_relative_residual = 0.0
    for iteration in range(41):
        data = evaluate(z)
        norm = float(torch.maximum(data[0].abs().max(), data[1].abs().max()).cpu())
        if not bool(data[2].cpu()) or not math.isfinite(norm):
            raise PropertyModelError("Inadmissible active capillary trace")
        if norm <= 2e-12:
            break
        if iteration == 40:
            raise HydraulicConvergenceError("Active capillary Newton exhausted")
        change = linear.solve(lambda value: evaluate(value)[0], z, data[0])
        for power in range(24):
            trial = (z + change * 2.0 ** (-power)).reshape(n, width)
            trial[:, 2 * nc] = torch.where(
                controlled, trial[:, 2 * nc].clamp_min(0), trial[:, 2 * nc].clamp(0, 1)
            )
            trial = trial.flatten()
            test = evaluate(trial)
            if (
                bool(test[2].cpu())
                and float(torch.maximum(test[0].abs().max(), test[1].abs().max()).cpu()) < norm
            ):
                z = trial
                break
        else:
            raise HydraulicConvergenceError("Active capillary line search failed")
    (
        _,
        _,
        _,
        nn,
        vv,
        mm,
        ee,
        tt,
        pp,
        rho,
        velocity,
        packet,
        spatial,
        streams,
        actuator,
        port_data,
    ) = evaluate(z)
    locations = [(i, j) for i in range(n) for j in range(2) if float(vv[i, j].cpu()) > 0]
    phases = recover_homogeneous_phases(
        f,
        [nn[i, j].cpu().tolist() for i, j in locations],
        [float(vv[i, j].cpu()) for i, j in locations],
        [float(mm[i, j].cpu()) for i, j in locations],
        [float(ee[i, j].cpu()) for i, j in locations],
        [p.slots[j]["root"] for i, j in locations],
        [float(tt[i, j].cpu()) for i, j in locations],
    )
    rows = [[] for _ in range(n)]
    for (i, j), phase in zip(locations, phases):
        rows[i].append((names[j], phase))
    end = PRActiveMultifluidState(tuple(tuple(row) for row in rows), clock, p.model_id)
    after = p.inspect(end)
    surface = s.surface_energy(end)
    oldtot, newtot = p._totals(state), p._totals(end)
    transfer = (dt * spatial[6]).cpu().tolist()
    side_transfer = streams.cpu().tolist()
    capwork = float((dt * spatial[7]).cpu())
    work = float(actuator.cpu())
    wall = float((dt * spatial[5]).cpu())
    capimp = float((dt * spatial[8]).cpu())
    if s.wetting is not None and s.wetting.scale > 0 and not any(model.boundaries):
        capimp = float((dt * spatial[3].sum()).cpu())
    defect = [newtot[j] - oldtot[j] - transfer[j] - side_transfer[j] for j in range(nc)]
    defect.extend(
        (
            newtot[-2]
            - oldtot[-2]
            + surface
            - initial_surface
            - transfer[-1]
            - side_transfer[-1]
            - capwork
            - work,
            newtot[-1] - oldtot[-1] - wall - transfer[-2] - side_transfer[-2] - capimp,
        )
    )
    allowances = [2e-10 * (abs(a) + abs(b)) + 1e-22 for a, b in zip(oldtot, newtot)]
    face_left, face_right, ff, face_flux = model._face_fields(packet)
    allowances[-1] += (
        128
        * torch.finfo(nn.dtype).eps
        * (
            abs(wall)
            + abs(transfer[-2])
            + abs(side_transfer[-2])
            + float(mm.abs().sum().cpu())
            + dt
            * (
                p.area
                * float(
                    (
                        2 * face_flux[..., -2].abs().sum() + ff["phase_path"][..., -2].abs().sum()
                    ).cpu()
                )
                + float(spatial[3].abs().sum().cpu())
            )
        )
    )
    if any(abs(v) > limit for v, limit in zip(defect, allowances)):
        raise ConservationError("Active capillary N/M/(E+F) balance failed")
    trace_locations = [
        (i, j)
        for i in range(n)
        for j in range(2)
        if present[j]
        and (
            float(old_v[i, j].cpu()) + float(vv[i, j].cpu()) > 0
            or (i in controls and controls[i][0] != j)
        )
    ]
    trace_entropy = torch.zeros((n, 2), device=f.device, dtype=rho.dtype)
    if trace_locations:
        trace_q = homogeneous_batch(
            f,
            [float(pp[i, j].cpu()) for i, j in trace_locations],
            [float(tt[i, j].cpu()) for i, j in trace_locations],
            [(rho[i, j] / rho[i, j].sum()).cpu().tolist() for i, j in trace_locations],
            [p.slots[j]["root"] for i, j in trace_locations],
        )
        for k, (i, j) in enumerate(trace_locations):
            expected = 1 / rho[i, j].sum()
            if abs(float((trace_q["molar_volume_m3_mol"][k] - expected).cpu())) > 1e-9 * float(
                expected.cpu()
            ):
                raise PropertyModelError("Active outflow trace changed its declared PR branch")
            trace_entropy[i, j] = rho[i, j].sum() * trace_q["molar_entropy_j_mol_k"][k]
    entropy_in = 0.0
    ports = [None, None]
    potential = s.discrete_pressure(h0, vv[:, 0] / s.area)
    amid = packet["alpha"]
    for side, boundary in enumerate(model.boundaries):
        if boundary is None:
            continue
        index = 0 if side == 0 else -1
        sign = 1 if side == 0 else -1
        body = {k: v[:1] if side == 0 else v[-1:] for k, v in packet.items()}
        ghost = model.boundary_packet(side, body["alpha"], body["velocity"])
        bs = amid[index] * trace_entropy[index]
        gs = ghost["entropy_density"][0]
        entropy_jump = (bs.sum() - gs) if side == 0 else (gs - bs.sum())
        if model.volume_diffusion == "none":
            gi = torch.where(
                model.boundary_packets[side]["alpha"][0] > 0,
                model.boundary_entropy[side][0],
                trace_entropy[index],
            )
            entropy_jump = (
                (body["alpha"][0] + ghost["alpha"][0]) / 2 * (trace_entropy[index] - gi)
            ).sum() * sign
        sf = (
            (bs * velocity[index]).sum() + ghost["entropy_flux"][0]
        ) / 2 - p.wave_speed / 2 * entropy_jump
        entropy_in += sign * dt * p.area * float(sf.cpu())
        aa, bb = face_left["alpha"][index], face_right["alpha"][index]
        outside = (2 * aa + bb) / 6 if side == 0 else (aa + 2 * bb) / 6
        outside = torch.where(model.boundary_packets[side]["alpha"][0] > 0, outside, 0.0)
        w = (
            -dt
            * p.area
            / 2
            * potential[index]
            * (bb[0] - aa[0])
            * (outside * ghost["velocity"][0]).sum()
        )
        ports[side] = {
            "transfer_n_m_e": (sign * dt * p.area * face_flux[index].sum(0)).cpu().tolist(),
            "capillary_work_j": float(w.cpu()),
        }
    model._active_port_cache = (_digest(state.to_dict()), _digest(end.to_dict()), tuple(ports))
    boundary_entropy_in = entropy_in
    control_report = []
    for i, rate, w, qv, sinlet in port_data:
        j = controls[i][0]
        out = 1 - j
        if float(rate.cpu()) == 0.0 or not present[out]:
            s_out = 0.0
        else:
            s_out = float(trace_entropy[i, out].cpu())
        entropy_in += dt * float(rate.cpu()) * (sinlet - s_out)
        control_report.append(
            {
                "cell": i,
                "volume_rate_m3_s": float(rate.cpu()),
                "phase_volume_transfer_m3": (dt * qv).cpu().tolist(),
                "actuator_work_j": float(w.cpu()),
            }
        )
    material_impulse = 0.0
    if s.wetting is not None:
        material_pressure = s.wetting.divided_difference(h0, vv[:, 0] / s.area)
        material_impulse = float(
            (
                dt
                * model._spatial(
                    packet, h0, vv[:, 0] / s.area, potential_override=material_pressure
                )[3].sum()
            ).cpu()
        )
    production = after["entropy"] - before["entropy"] - entropy_in
    if production < -256 * (math.ulp(before["entropy"]) + math.ulp(after["entropy"])) - 1e-20:
        raise ConservationError(f"Active capillary entropy decreased: {production:.17g} J/K")
    return end, {
        "state": end.to_dict(),
        "model_id": model.model_id,
        "source_model_id": s.model_id,
        "source_model": s.to_dict(),
        "normal_sources_applied": False,
        "material_wall_impulse_kg_m_s": material_impulse,
        "capillary_force_n": spatial[3].cpu().tolist(),
        "capillary_diffusion_heat_j": (dt * spatial[4]).cpu().tolist(),
        "dt_s": dt,
        "newton_iterations": iteration,
        "linear_solver": linear.report(),
        "active_phase_support": True,
        "phase_controls": normalized,
        "controls": control_report,
        "primitive_traces": {
            "component_densities_mol_m3": rho.cpu().tolist(),
            "velocity_m_s": velocity.cpu().tolist(),
            "temperature_k": tt.cpu().tolist(),
            "pressure_pa": pp.cpu().tolist(),
        },
        "external_transfer_n_m_e": transfer,
        "sideport_transfer_n_m_e": side_transfer,
        "sideport_actuator_work_j": work,
        "external_capillary_work_j": capwork,
        "wall_impulse_kg_m_s": wall,
        "external_capillary_impulse_kg_m_s": capimp,
        "surface_energy_before_j": initial_surface,
        "surface_energy_after_j": surface,
        "conservation_residuals": defect,
        "conservation_allowances": allowances,
        "entropy_production_j_k": production,
        "incoming_entropy_j_k": entropy_in,
        "sideport_incoming_entropy_j_k": entropy_in - boundary_entropy_in,
        "port_transfers": ports,
        "maximum_scaled_residual": norm,
        "qualification": "End-intrinsic active capillary traces; owned zero phases omitted exactly; transport and controlled balanced side ports; no spontaneous bulk nucleation closure",
    }
