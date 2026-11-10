"""Joint implicit two-phase PR/capillary transport on a uniform closed line.

Fixed positive two-phase/component support, zero gravity. Physical BN paths
use midpoint geometry and end velocities/calorics. The discrete-adjoint
capillary force is paired with that same volume path. Midpoint geometric
acoustic diffusion releases a derived nonnegative capillary heat, not a
bulk-energy residual correction. Full N/M/(E+F) and entropy are audited.
"""

import math
from dataclasses import asdict

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_active_multifluid_flux import fluctuations
from .pr_active_multifluid_pipe import PRActiveMultifluidState
from .pr_capillary_linear import CapillaryLinearSystem
from .pr_capillary_pressure import PRCapillaryPressure
from .pr_homogeneous_uv import caloric_tv, recover_homogeneous_phases


class PRCapillaryAxialPipe:
    def __init__(
        self,
        source,
        *,
        maximum_unknowns=192,
        boundaries=None,
        volume_diffusion="acoustic",
        linear_solver="auto",
        boundary_geometry="extrapolated_face",
    ):
        if (
            not isinstance(source, PRCapillaryPressure)
            or source.to_dict()["model"] != "pr_capillary_pressure_v1"
        ):
            raise TypeError("Resolved PR capillary source required")
        self.source, self.pipe = source, source.pipe
        if linear_solver not in ("auto", "dense", "sparse"):
            raise ValueError("Capillary linear solver must be auto, dense or sparse")
        self.linear_solver = linear_solver
        if boundary_geometry not in ("extrapolated_face", "reservoir_cell"):
            raise ValueError(
                "Explicit extrapolated_face or reservoir_cell boundary geometry required"
            )
        self.boundary_geometry = boundary_geometry
        p = self.pipe
        if (
            p.boundary_mode not in ("periodic", "walls")
            or p.gravity != 0
            or len(p.names) != 2
            or tuple(p.names) != (source.liquid, source.gas)
            or any(dx != p.lengths[0] for dx in p.lengths)
            or p.chemical is not None
            or p.friction is not None
        ):
            raise ValueError(
                "Uniform periodic or reflecting two-phase line without gravity/other split sources required"
            )
        if type(maximum_unknowns) is not int or maximum_unknowns < 1:
            raise ValueError("Explicit positive unknown budget required")
        rows = (None, None) if boundaries is None else tuple(boundaries)
        if len(rows) != 2 or (p.boundary_mode == "periodic" and any(b is not None for b in rows)):
            raise ValueError("Two maintained face states require nonperiodic geometry")
        self.boundaries, self.boundary_packets, self.boundary_entropy = [], [], []
        heights = []
        for row in rows:
            if row is None:
                self.boundaries.append(None)
                self.boundary_packets.append(None)
                self.boundary_entropy.append(None)
                heights.append(None)
                continue
            packet, phases = p._boundary_packet(row)
            if set(phases) != set(p.names) and boundary_geometry != "reservoir_cell":
                raise ValueError("Absent boundary phases require reservoir_cell geometry")
            if set(phases) != set(p.names) and volume_diffusion != "acoustic":
                raise ValueError("Absent boundary phases require acoustic volume transport")
            height = float(packet["alpha"][0, 0].cpu()) * p.area / source.width
            heights.append(height)
            self.boundaries.append(phases)
            self.boundary_packets.append(packet)
            entropy = []
            for name, root in zip(p.names, ("liquid", "vapor")):
                if name not in phases:
                    entropy.append(0.0)
                    continue
                phase = phases[name]
                amount = math.fsum(phase.component_moles)
                eos = p.fluid.homogeneous(
                    phase.pressure_pa,
                    phase.temperature_k,
                    [v / amount for v in phase.component_moles],
                    root,
                )
                entropy.append(float(eos["molar_entropy_j_mol_k"].cpu()) * amount / phase.volume_m3)
            self.boundary_entropy.append(p.fluid.tensor([entropy]))
        if any(h is not None for h in heights):
            for prescribed, computed in zip(source.boundary_heights, heights, strict=True):
                if prescribed is not None and (
                    computed is None or abs(prescribed - computed) > 32 * math.ulp(computed)
                ):
                    raise ValueError("Face inventory and capillary geometry differ")
            params = {
                k: v for k, v in source.to_dict().items() if k not in ("model", "pipe_model_id")
            }
            params["boundary_heights_m"] = heights
            if boundary_geometry == "reservoir_cell":
                params["boundary_distances_m"] = [
                    None if h is None else p.lengths[0 if side == 0 else -1]
                    for side, h in enumerate(heights)
                ]
            self.source = PRCapillaryPressure(p, **params)
        elif any(h is not None for h in source.boundary_heights):
            raise ValueError(
                "Fixed face geometry requires maintained thermodynamic boundary states"
            )
        if volume_diffusion not in ("acoustic", "none"):
            raise ValueError("Explicit acoustic/none volume diffusion required")
        if source.wetting is not None and source.wetting.scale > 0 and volume_diffusion != "none":
            raise ValueError("Nonconvex material wetting requires volume_diffusion='none'")
        self.volume_diffusion = volume_diffusion
        self.maximum_unknowns = maximum_unknowns
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        result = {
            "model": "pr_capillary_axial_v1",
            "pipe": self.pipe.to_dict(),
            "source": self.source.to_dict(),
            "maximum_unknowns": self.maximum_unknowns,
            "transport_state": "midpoint_intrinsic",
        }

        if any(b is not None for b in self.boundaries):
            result["boundaries"] = [
                None if row is None else {n: asdict(v) for n, v in row.items()}
                for row in self.boundaries
            ]
        if self.volume_diffusion != "acoustic":
            result["volume_diffusion"] = self.volume_diffusion
        if self.linear_solver != "auto":
            result["linear_solver"] = self.linear_solver
        if self.boundary_geometry != "extrapolated_face":
            result["boundary_geometry"] = self.boundary_geometry
        return result

    def boundary_packet(self, side, adjacent_alpha, adjacent_velocity=None):
        reference = self.boundary_packets[side]
        if reference is None:
            return None
        alpha = (
            reference["alpha"]
            if self.boundary_geometry == "reservoir_cell"
            else 2 * reference["alpha"] - adjacent_alpha
        )
        conserved = (
            reference["conserved"]
            * (alpha / torch.where(reference["alpha"] > 0, reference["alpha"], 1))[..., None]
        )
        velocity, pressure = reference["velocity"], reference["pressure"]
        if adjacent_velocity is not None:
            velocity = torch.where(reference["alpha"] > 0, velocity, adjacent_velocity)
        flux = torch.cat(
            (
                conserved[..., :-2] * velocity[..., None],
                (conserved[..., -2] * velocity + alpha * pressure)[..., None],
                ((conserved[..., -1] + alpha * pressure) * velocity)[..., None],
            ),
            -1,
        )
        entropy = alpha * self.boundary_entropy[side]
        return {
            **reference,
            "velocity": velocity,
            "alpha": alpha,
            "end_alpha": alpha,
            "conserved": conserved,
            "flux": flux,
            "entropy_density": entropy.sum(-1),
            "entropy_flux": (entropy * velocity).sum(-1),
        }

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        from .pr_active_multifluid_pipe import PRActiveMultifluidPipe

        if (
            set(raw)
            - {
                "model",
                "pipe",
                "source",
                "maximum_unknowns",
                "boundaries",
                "volume_diffusion",
                "linear_solver",
                "transport_state",
                "boundary_geometry",
            }
            or not {"model", "pipe", "source", "maximum_unknowns"} <= set(raw)
            or raw["model"] != "pr_capillary_axial_v1"
            or raw.get("transport_state", "midpoint_intrinsic") != "midpoint_intrinsic"
        ):
            raise ValueError("Explicit joint PR capillary axial model required")
        pipe = PRActiveMultifluidPipe.from_dict(raw["pipe"], device=device)
        source = PRCapillaryPressure.from_dict(pipe, raw["source"])
        return cls(
            source,
            maximum_unknowns=raw["maximum_unknowns"],
            boundaries=raw.get("boundaries"),
            linear_solver=raw.get("linear_solver", "auto"),
            boundary_geometry=raw.get("boundary_geometry", "extrapolated_face"),
            volume_diffusion=raw.get("volume_diffusion", "acoustic"),
        )

    def _face_fields(self, packet):
        """Physical traces and fluxes shared by Newton and roundoff accounting."""
        p = self.pipe
        c = p.wave_speed
        velocity, amid = packet["velocity"], packet["alpha"]
        periodic = p.boundary_mode == "periodic"
        if periodic:
            left = packet
            right = {k: torch.roll(v, -1, 0) for k, v in packet.items()}
        else:
            left, right = {}, {}
            maintained = [
                self.boundary_packet(0, amid[:1], velocity[:1]),
                self.boundary_packet(1, amid[-1:], velocity[-1:]),
            ]
            for key, value in packet.items():
                gl, gr = value[:1].clone(), value[-1:].clone()
                if key == "velocity":
                    gl, gr = -gl, -gr
                elif key == "conserved":
                    gl[..., -2] *= -1
                    gr[..., -2] *= -1
                elif key == "flux":
                    gl[..., :-2] *= -1
                    gr[..., :-2] *= -1
                    gl[..., -1] *= -1
                    gr[..., -1] *= -1
                if maintained[0] is not None:
                    gl = maintained[0][key]
                if maintained[1] is not None:
                    gr = maintained[1][key]
                left[key] = torch.cat((gl, value), 0)
                right[key] = torch.cat((value, gr), 0)
        _, _, _, _, ff = fluctuations(left, right, c)
        # Always use the declared acoustic viscosity, with no contact switch.
        intrinsic_l, intrinsic_r = self.intrinsic_pair(left, right)
        jump = intrinsic_r * right["alpha"][..., None] - intrinsic_l * left["alpha"][..., None]
        if self.volume_diffusion == "none":
            jump = (left["alpha"] + right["alpha"])[..., None] / 2 * (intrinsic_r - intrinsic_l)
        flux = (left["flux"] + right["flux"]) / 2 - c / 2 * jump
        return left, right, ff, flux

    def _spatial(self, packet, h0, h1, *, potential_override=None):
        p, s, f = self.pipe, self.source, self.pipe.fluid
        n, nc = len(p.lengths), len(f.names)
        area, c = p.area, p.wave_speed
        velocity, amid, conserved = packet["velocity"], packet["alpha"], packet["conserved"]
        periodic = p.boundary_mode == "periodic"
        left, right, ff, flux = self._face_fields(packet)
        if not periodic:
            maintained = [
                self.boundary_packet(0, amid[:1], velocity[:1]),
                self.boundary_packet(1, amid[-1:], velocity[-1:]),
            ]
        if periodic:
            delta = (
                flux
                - torch.roll(flux, 1, 0)
                + (ff["phase_path"] + torch.roll(ff["phase_path"], 1, 0)) / 2
            )
            wall_rate = flux.sum() * 0
        else:
            delta = flux[1:] - flux[:-1] + (ff["phase_path"][1:] + ff["phase_path"][:-1]) / 2
            wall_rate = area * (flux[0, :, -2].sum() - flux[-1, :, -2].sum())
        i = torch.arange(n if periodic else n - 1, device=f.device)
        j = (i + 1) % n
        al, ar = amid[i], amid[j]
        da = ar[:, 0] - al[:, 0]
        cl, cr = (2 * al + ar) / 6, (al + 2 * ar) / 6
        uface = (cl * velocity[i] + cr * velocity[j]).sum(-1)
        path = da * uface
        if periodic:
            physical_dv_rate = -area / 2 * (path + torch.roll(path, 1, 0))
        else:
            endpoints = torch.zeros(n + 1, dtype=path.dtype, device=f.device).index_copy(
                0, i + 1, path
            )
            for side, external in enumerate(maintained):
                if external is not None:
                    fi = 0 if side == 0 else -1
                    aa, bb = left["alpha"][fi], right["alpha"][fi]
                    vl, vr = left["velocity"][fi], right["velocity"][fi]
                    integral = (((2 * aa + bb) / 6) * vl + ((aa + 2 * bb) / 6) * vr).sum()
                    endpoints[fi] = (bb[0] - aa[0]) * integral
            physical_dv_rate = -area / 2 * (endpoints[1:] + endpoints[:-1])
        diffusion = area * c / 2 * (right["alpha"][:, 0] - left["alpha"][:, 0])
        diff_dv_rate = (
            diffusion - torch.roll(diffusion, 1, 0) if periodic else diffusion[1:] - diffusion[:-1]
        )
        if self.volume_diffusion == "none":
            diff_dv_rate = torch.zeros_like(h0)
        potential = (
            s.discrete_pressure(h0, h1) if potential_override is None else potential_override
        )
        conjugate = area / 2 * (potential[i] + potential[j]) * da
        force = (
            torch.zeros_like(velocity)
            .index_add(0, i, conjugate[:, None] * cl)
            .index_add(0, j, conjugate[:, None] * cr)
        )
        cap_external_work = potential.sum() * 0
        boundary_alpha = torch.ones((0, 2), dtype=amid.dtype, device=f.device)
        if not periodic:
            boundary_alpha = (
                torch.cat([r["alpha"] for r in maintained if r is not None], 0)
                if any(r is not None for r in maintained)
                else boundary_alpha
            )
            for side, external in enumerate(maintained):
                if external is None:
                    continue
                fi, ci = (0, 0) if side == 0 else (-1, n - 1)
                aa, bb = left["alpha"][fi], right["alpha"][fi]
                coeff_left, coeff_right = (2 * aa + bb) / 6, (aa + 2 * bb) / 6
                factor = area / 2 * potential[ci] * (bb[0] - aa[0])
                inside = coeff_right if side == 0 else coeff_left
                outside = coeff_left if side == 0 else coeff_right
                absent = self.boundary_packets[side]["alpha"][0] == 0
                inside = inside + torch.where(absent, outside, 0.0)
                outside = torch.where(absent, 0.0, outside)
                extra = torch.zeros_like(force).index_add(
                    0, torch.tensor([ci], device=f.device), (factor * inside)[None, :]
                )
                force = force + extra
                cap_external_work = (
                    cap_external_work - factor * (outside * external["velocity"][0]).sum()
                )
        external_flux = conserved.new_zeros(nc + 2)
        if not periodic:
            wall_rate = conserved.sum() * 0
            for side in (0, 1):
                fi = 0 if side == 0 else -1
                sign = 1 if side == 0 else -1
                if self.boundaries[side] is None:
                    wall_rate = wall_rate + sign * area * flux[fi, :, -2].sum()
                else:
                    external_flux = external_flux + sign * area * flux[fi].sum(0)
        cap_external_impulse = (
            force.sum() if any(b is not None for b in self.boundaries) else force.sum() * 0
        )
        cap_heat = -potential * diff_dv_rate
        return (
            delta,
            physical_dv_rate,
            diff_dv_rate,
            force,
            cap_heat,
            wall_rate,
            external_flux,
            cap_external_work,
            cap_external_impulse,
            boundary_alpha,
        )

    @staticmethod
    def intrinsic_pair(left, right):
        """No EOS/stock in an absent phase; intensive traces copy the real side."""
        la, ra = left["end_alpha"] > 0, right["end_alpha"] > 0
        ll = left["conserved"] / torch.where(la, left["end_alpha"], 1)[..., None]
        rr = right["conserved"] / torch.where(ra, right["end_alpha"], 1)[..., None]
        return torch.where(la[..., None], ll, rr), torch.where(ra[..., None], rr, ll)

    @staticmethod
    def midpoint_packet(old, new):
        alpha = (old["alpha"] + new["alpha"]) / 2
        q, u, pressure = new["conserved"], new["velocity"], new["pressure"]
        transported = q / new["alpha"][..., None] * alpha[..., None]
        flux = torch.cat(
            (
                transported[..., :-2] * u[..., None],
                (transported[..., -2] * u + alpha * pressure)[..., None],
                ((transported[..., -1] + alpha * pressure) * u)[..., None],
            ),
            -1,
        )
        packet = {key: new[key] for key in ("conserved", "pressure", "temperature", "velocity")}
        packet.update(
            alpha=alpha,
            end_alpha=new["alpha"],
            flux=flux,
            pressure_precision=torch.zeros_like(pressure),
        )
        return packet

    def state_packet(self, before, after):
        p = self.pipe
        packet = self.midpoint_packet(p.inspect(before), p.inspect(after))
        h0 = (
            p.fluid.tensor([dict(c)[self.source.liquid].volume_m3 for c in before.cells])
            / self.source.area
        )
        h1 = (
            p.fluid.tensor([dict(c)[self.source.liquid].volume_m3 for c in after.cells])
            / self.source.area
        )
        return packet, h0, h1

    def port_transfers(self, before, after):
        """Accepted transfers into this pipe, separately at each real endpoint.

        Reconstruct the same midpoint-volume/end-caloric traces as Newton.
        These are flux integrals, never a partition of the global residual.
        Wall endpoints have no material or capillary-work port.
        """
        if self.pipe.boundary_mode == "periodic":
            return (None, None)
        dt = after.time_s - before.time_s
        if dt <= 0:
            raise ValueError("Accepted positive pipe interval required")
        packet, h0, h1 = self.state_packet(before, after)
        potential = self.source.discrete_pressure(h0, h1)
        rows = []
        for side, reference in enumerate(self.boundary_packets):
            if reference is None:
                rows.append(None)
                continue
            ci = 0 if side == 0 else -1
            body = {k: v[:1] if side == 0 else v[-1:] for k, v in packet.items()}
            ghost = self.boundary_packet(side, body["alpha"], body["velocity"])
            left, right = (ghost, body) if side == 0 else (body, ghost)
            il, ir = self.intrinsic_pair(left, right)
            jump = ir * right["alpha"][..., None] - il * left["alpha"][..., None]
            if self.volume_diffusion == "none":
                jump = (left["alpha"] + right["alpha"])[..., None] / 2 * (ir - il)
            flux = (left["flux"] + right["flux"]) / 2 - self.pipe.wave_speed / 2 * jump
            sign = 1 if side == 0 else -1
            transfer = sign * dt * self.pipe.area * flux[0].sum(0)
            aa, bb = left["alpha"][0], right["alpha"][0]
            outside = (2 * aa + bb) / 6 if side == 0 else (aa + 2 * bb) / 6
            outside = torch.where(reference["alpha"][0] > 0, outside, 0.0)
            work = (
                -dt
                * self.pipe.area
                / 2
                * potential[ci]
                * (bb[0] - aa[0])
                * (outside * ghost["velocity"][0]).sum()
            )
            rows.append(
                {
                    "transfer_n_m_e": transfer.cpu().tolist(),
                    "capillary_work_j": float(work.cpu()),
                }
            )
        return tuple(rows)

    def step(self, state, dt_s):
        p, s, f = self.pipe, self.source, self.pipe.fluid
        initial = p.inspect(state)
        if _digest(s.to_dict()) != s.model_id or _digest(self.to_dict()) != self.model_id:
            raise ValueError("Changed capillary axial constitutive model")
        s.surface_energy(state)
        dt = _number(dt_s, "joint capillary axial interval", positive=True)
        end = state.time_s + dt
        dt = end - state.time_s
        if not math.isfinite(end) or dt <= 0:
            raise ValueError("Capillary axial interval below clock resolution")
        n, nc = len(state.cells), len(f.names)
        width = 2 * nc + 5
        if n * width > self.maximum_unknowns:
            raise ValueError("Joint capillary axial dense unknown budget exceeded")
        names = (s.liquid, s.gas)
        if any(set(dict(c)) != set(names) for c in state.cells):
            raise ValueError("Positive liquid/gas topology required")
        cells = [[dict(c)[name] for name in names] for c in state.cells]
        nn0 = f.tensor([[v.component_moles for v in c] for c in cells])
        if not bool((nn0 > 0).all().cpu()):
            raise ValueError("Positive common component support required")
        vv0 = f.tensor([[v.volume_m3 for v in c] for c in cells])
        mm0 = f.tensor([[v.momentum_kg_m_s for v in c] for c in cells])
        tt0 = f.tensor([[v.temperature_k for v in c] for c in cells])
        ee0 = f.tensor([[v.total_energy_j for v in c] for c in cells])
        volume = f.tensor(p.volumes)
        alpha0 = vv0 / volume[:, None]
        for side, ref in enumerate(self.boundary_packets):
            adjacent = alpha0[:1] if side == 0 else alpha0[-1:]
            if (
                ref is not None
                and self.boundary_geometry == "extrapolated_face"
                and not bool((2 * ref["alpha"] - adjacent > 0).all().cpu())
            ):
                raise ValueError("Initial interface incompatible with prescribed face height")
        h0 = vv0[:, 0] / s.area
        mw = f.tensor([c.molar_mass_kg_mol for c in f.components])
        mass0 = (nn0 * mw).sum(-1)
        mscale = mass0 * torch.maximum((mm0 / mass0).abs(), f.tensor(1.0))
        escale = torch.maximum(ee0.abs(), nn0.sum(-1) * 8.31446261815324 * tt0)
        logit0 = torch.log(alpha0[:, 0] / alpha0[:, 1])
        area, c = p.area, p.wave_speed

        def evaluate(z):
            rows = z.reshape(n, width)
            numbers = nn0 * torch.exp(rows[:, : 2 * nc].reshape(n, 2, nc))
            a = torch.sigmoid(logit0 + rows[:, 2 * nc])
            volumes = volume[:, None] * torch.stack((a, 1 - a), -1)
            momenta = mm0 + mscale * rows[:, 2 * nc + 1 : 2 * nc + 3]
            if hasattr(self, "pinned_cells"):
                momenta = torch.where(
                    self.pinned_cells[:, None], torch.zeros_like(momenta), momenta
                )
            temperature = tt0 * torch.exp(rows[:, -2:])
            amount = numbers.sum(-1)
            mass = (numbers * mw).sum(-1)
            velocity = momenta / mass
            caloric = caloric_tv(f, temperature, volumes / amount, numbers / amount[..., None])
            pressure = caloric["p"]
            energy = amount * caloric["u"] + 0.5 * momenta * velocity
            conserved = (
                torch.cat((numbers, momenta[..., None], energy[..., None]), -1)
                / volume[:, None, None]
            )
            amid = (volumes / volume[:, None] + alpha0) / 2
            transported = conserved / (volumes / volume[:, None])[..., None] * amid[..., None]
            physical_flux = torch.cat(
                (
                    transported[..., :-2] * velocity[..., None],
                    (transported[..., -2] * velocity + amid * pressure)[..., None],
                    ((transported[..., -1] + amid * pressure) * velocity)[..., None],
                ),
                -1,
            )
            packet = {
                "alpha": amid,
                "end_alpha": volumes / volume[:, None],
                "pressure": pressure,
                "temperature": temperature,
                "velocity": velocity,
                "conserved": conserved,
                "flux": physical_flux,
                "pressure_precision": torch.zeros_like(pressure),
            }
            (
                delta,
                physical_dv_rate,
                diff_dv_rate,
                force,
                cap_heat,
                wall_rate,
                external_flux,
                cap_external_work,
                cap_external_impulse,
                boundary_alpha,
            ) = self._spatial(packet, h0, volumes[:, 0] / s.area)
            heat = cap_heat[:, None] * f.tensor([s.share, 1 - s.share])
            rates = torch.cat(
                (
                    delta[..., :-2] * area,
                    (delta[..., -2] * area - force)[..., None],
                    (delta[..., -1] * area - velocity * force - heat)[..., None],
                ),
                -1,
            )
            rn = (numbers - nn0 + dt * rates[..., :-2]) / nn0
            rv = (volumes[:, 0] - vv0[:, 0] - dt * (physical_dv_rate + diff_dv_rate)) / volume
            rm = (momenta - mm0 + dt * rates[..., -2]) / mscale
            if hasattr(self, "pinned_cells"):
                rm = torch.where(self.pinned_cells[:, None], rows[:, 2 * nc + 1 : 2 * nc + 3], rm)
            re = (energy - ee0 + dt * rates[..., -1]) / escale
            residual = torch.cat((rn.reshape(n, 2 * nc), rv[:, None], rm, re), -1).reshape(-1)
            return (
                residual,
                numbers,
                volumes,
                momenta,
                energy,
                temperature,
                pressure,
                caloric["dpdv"],
                force,
                cap_heat,
                wall_rate,
                external_flux,
                cap_external_work,
                cap_external_impulse,
                boundary_alpha,
            )

        def admissible(result):
            _, numbers, volumes, _, _, temperature, pressure, dpdv, *_ = result
            boundary_alpha = result[-1]
            return bool(
                (
                    torch.isfinite(torch.cat([x.reshape(-1) for x in result])).all()
                    & (
                        (boundary_alpha >= 0).all()
                        if self.boundary_geometry == "reservoir_cell"
                        else (boundary_alpha > 0).all()
                    )
                    & (numbers > 0).all()
                    & (volumes > 0).all()
                    & (temperature >= f.min_temperature_k).all()
                    & (temperature <= f.max_temperature_k).all()
                    & (pressure >= f.min_pressure_pa).all()
                    & (pressure <= f.max_pressure_pa).all()
                    & (dpdv < 0).all()
                ).cpu()
            )

        z = f.tensor([0.0] * (n * width))
        if getattr(self, "_linear", None) is None:
            edges = set(s.geometry_edges)
            if hasattr(self, "faces"):
                edges.update((a, b) for a, b, _wall, _axis in self.faces if a != b)
            self._linear = CapillaryLinearSystem(
                [list(range(i * width, (i + 1) * width)) for i in range(n)],
                edges,
                f.device,
                radius=2,
                mode=self.linear_solver,
            )
        linear = self._linear
        linear.solves, linear.maximum_relative_residual = 0, 0.0
        linear.krylov_restarts = 0
        for iteration in range(31):
            result = evaluate(z)
            norm = float(result[0].abs().max().cpu())
            if not admissible(result):
                raise PropertyModelError("Inadmissible joint capillary PR iterate")
            if norm <= 2e-12:
                break
            if iteration == 30:
                raise HydraulicConvergenceError("Joint capillary axial Newton failed")
            # Differentiation/programming errors are not time-step failures.
            # Only an invalid/singular linear system belongs to adaptive retry.
            correction = linear.solve(lambda w: evaluate(w)[0], z, result[0])
            accepted = False
            for power in range(24):
                trial = z + correction * 2.0 ** (-power)
                test = evaluate(trial)
                if admissible(test) and float(test[0].abs().max().cpu()) < norm:
                    z = trial
                    accepted = True
                    break
            if not accepted:
                raise HydraulicConvergenceError("Joint capillary axial line search failed")
        result = evaluate(z)
        (
            _,
            numbers,
            volumes,
            momenta,
            energy,
            temperature,
            _,
            _,
            force,
            heat,
            wall_rate,
            external_flux,
            cap_work,
            cap_impulse,
            _,
        ) = result
        phases = recover_homogeneous_phases(
            f,
            numbers.reshape(-1, nc).cpu().tolist(),
            volumes.reshape(-1).cpu().tolist(),
            momenta.reshape(-1).cpu().tolist(),
            energy.reshape(-1).cpu().tolist(),
            [root for _ in range(n) for root in ("liquid", "vapor")],
            temperature.reshape(-1).cpu().tolist(),
        )
        new = PRActiveMultifluidState(
            tuple(
                tuple((name, phases[2 * i + j]) for j, name in enumerate(names)) for i in range(n)
            ),
            end,
            p.model_id,
        )
        current = p.inspect(new)
        oldf, newf = s.surface_energy(state), s.surface_energy(new)
        oldtot, newtot = p._totals(state), p._totals(new)
        oldtot[-2] += oldf
        newtot[-2] += newf
        defect = [b - a for a, b in zip(oldtot, newtot, strict=True)]
        wall_impulse = dt * float(wall_rate.cpu())
        transfer = (dt * external_flux).cpu().tolist()
        cap_work = float((dt * cap_work).cpu())
        cap_impulse = float((dt * cap_impulse).cpu())
        for j in range(nc):
            defect[j] -= transfer[j]
        defect[-2] -= transfer[-1] + cap_work
        material_impulse = 0.0
        material_resultant_fraction = 0.0
        if s.wetting is not None and s.wetting.scale > 0:
            packet, wall_h0, wall_h1 = self.state_packet(state, new)
            wall_pressure = s.wetting.divided_difference(wall_h0, wall_h1)
            wall_force = self._spatial(packet, wall_h0, wall_h1, potential_override=wall_pressure)[
                3
            ]
            net = float(wall_force.sum().cpu())
            absolute = float(wall_force.abs().sum().cpu())
            material_impulse = dt * net
            material_resultant_fraction = abs(net) / absolute if absolute > 0 else 0.0
            if not any(b is not None for b in self.boundaries):
                cap_impulse = material_impulse
        defect[-1] -= wall_impulse + transfer[-2] + cap_impulse
        ledger_indices = getattr(self, "ledger_indices", tuple(range(len(defect))))
        defect = [defect[j] for j in ledger_indices]
        oldtot = [oldtot[j] for j in ledger_indices]
        newtot = [newtot[j] for j in ledger_indices]
        # Nearly opposed pressure impulses can leave zero total momentum.
        # Their arithmetic error scales with the actual flux/path operands,
        # not that cancelled total. This is a forward-roundoff bound only;
        # the nonlinear equations still use their original 2e-12 tolerance.
        packet = self.midpoint_packet(initial, current)
        _, _, face_fields, face_flux = self._face_fields(packet)
        momentum_roundoff = (
            64
            * torch.finfo(numbers.dtype).eps
            * (
                float((mm0.abs().sum() + momenta.abs().sum()).cpu())
                + abs(wall_impulse)
                + abs(transfer[-2])
                + abs(cap_impulse)
                + dt
                * (
                    area
                    * float(
                        (
                            2 * face_flux[..., -2].abs().sum()
                            + face_fields["phase_path"][..., -2].abs().sum()
                        ).cpu()
                    )
                    + float(force.abs().sum().cpu())
                )
            )
        )
        for j, value in enumerate(defect):
            scale = abs(oldtot[j]) + abs(newtot[j])
            limit = 2e-10 * scale + 1e-22
            if ledger_indices[j] == nc + 1:
                limit += momentum_roundoff
            if not math.isfinite(value) or abs(value) > limit:
                raise ConservationError(
                    f"Joint capillary axial augmented inventory failed: ledger index "
                    f"{ledger_indices[j]}, defect={value:.17g}, limit={limit:.17g}"
                )
        incoming_entropy = 0.0
        final_alpha = (volumes / volume[:, None] + alpha0) / 2
        for side, ref in enumerate(self.boundary_packets):
            if ref is None:
                continue
            ci = 0 if side == 0 else -1
            ghost = self.boundary_packet(
                side,
                final_alpha[:1] if side == 0 else final_alpha[-1:],
                current["velocity"][:1] if side == 0 else current["velocity"][-1:],
            )
            body_alpha, ghost_alpha = final_alpha[ci], ghost["alpha"][0]
            intrinsic_entropy = current["phase_entropy_density"][ci] / current["alpha"][ci]
            body_entropy = intrinsic_entropy * body_alpha
            si = body_entropy.sum()
            fi = (body_entropy * current["velocity"][ci]).sum()
            sg, fg = ghost["entropy_density"][0], ghost["entropy_flux"][0]
            entropy_jump = (si - sg) if side == 0 else (sg - si)
            if self.volume_diffusion == "none":
                ghost_intrinsic = torch.where(
                    ref["alpha"][0] > 0, self.boundary_entropy[side][0], intrinsic_entropy
                )
                difference = intrinsic_entropy - ghost_intrinsic
                entropy_jump = ((body_alpha + ghost_alpha) / 2 * difference).sum() * (
                    1 if side == 0 else -1
                )
            entropy_flux = (fg + fi) / 2 - c / 2 * entropy_jump
            incoming_entropy += (1 if side == 0 else -1) * dt * area * float(entropy_flux.cpu())
        if hasattr(self, "extra_incoming_entropy"):
            incoming_entropy += self.extra_incoming_entropy(state, new, dt)
        production = current["entropy"] - initial["entropy"] - incoming_entropy
        entropy_limit = 128 * (math.ulp(current["entropy"]) + math.ulp(initial["entropy"])) + 1e-22
        if production < -entropy_limit:
            raise ConservationError("Joint capillary axial entropy decreased")
        if bool((heat < -128 * torch.finfo(heat.dtype).eps * heat.abs().max()).any().cpu()):
            raise ConservationError("Negative derived capillary diffusion heat")
        return new, {
            "model_id": self.model_id,
            "state": new.to_dict(),
            "dt_s": dt,
            "newton_iterations": iteration,
            "linear_solver": linear.report(),
            "normal_sources_applied": False,
            "source_model": s.to_dict(),
            "source_model_id": s.model_id,
            "wall_impulse_kg_m_s": wall_impulse,
            "momentum_roundoff_allowance_kg_m_s": momentum_roundoff,
            "external_transfer_n_m_e": transfer,
            "external_capillary_work_j": cap_work,
            "external_capillary_impulse_kg_m_s": cap_impulse,
            "material_wall_impulse_kg_m_s": material_impulse,
            "pure_surface_boundary_impulse_kg_m_s": cap_impulse - material_impulse,
            "material_wall_resultant_fraction": material_resultant_fraction,
            "incoming_entropy_j_k": incoming_entropy,
            "capillary_force_n": force.cpu().tolist(),
            "capillary_diffusion_heat_j": (dt * heat).cpu().tolist(),
            "surface_energy_before_j": oldf,
            "surface_energy_after_j": newf,
            "conservation_residuals": defect,
            "entropy_production_j_k": production,
            "maximum_scaled_residual": float(result[0].abs().max().cpu()),
            "qualification": "Uniform positive two-phase PR capillary line with periodic, reflecting or maintained faces; no phase-topology-change or field qualification; material wall resultant includes spatial truncation error",
        }

    def step_coupled(self, state, dt_s):
        """First-order axial/normal split, both on the same real PR stocks/F."""
        transport, axial = self.step(state, dt_s)
        final, normal = self.source.apply(transport, axial["dt_s"])
        return final, {
            "state": final.to_dict(),
            "dt_s": axial["dt_s"],
            "newton_iterations": axial["newton_iterations"] + normal["newton_iterations"],
            "axial": axial,
            "normal": normal,
            "integration": "first_order_axial_then_normal_split",
        }

    def advance(
        self,
        state,
        duration_s,
        *,
        initial_step_s=None,
        minimum_step_s=1e-20,
        maximum_attempts=10000,
        normal_relaxation=False,
    ):
        """Reach the report clock exactly; failed trials never mutate stocks."""
        if type(normal_relaxation) is not bool:
            raise ValueError("Explicit boolean normal_relaxation required")
        duration = _number(duration_s, "capillary axial report duration", positive=True)
        target = state.time_s + duration
        if not math.isfinite(target) or target <= state.time_s:
            raise ValueError("Capillary axial report clock cannot represent interval")
        minimum = _number(minimum_step_s, "minimum capillary axial step", positive=True)
        dt = _number(
            0.1 * self.pipe.maximum_step_s if initial_step_s is None else initial_step_s,
            "initial capillary axial step",
            positive=True,
        )
        if type(maximum_attempts) is not int or maximum_attempts <= 0:
            raise ValueError("Positive capillary axial attempt budget required")
        self.pipe.inspect(state)
        current, steps, rejected = state, [], []
        for _ in range(maximum_attempts):
            if current.time_s == target:
                return current, {
                    "state": current.to_dict(),
                    "steps": steps,
                    "rejected": rejected,
                    "next_step_s": dt,
                    "duration_s": target - state.time_s,
                }
            attempt = min(dt, target - current.time_s)
            if attempt < minimum or current.time_s + attempt <= current.time_s:
                raise HydraulicConvergenceError("Capillary axial minimum step/clock reached")
            try:
                trial, record = (self.step_coupled if normal_relaxation else self.step)(
                    current, attempt
                )
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                rejected.append({"time_s": current.time_s, "dt_s": attempt, "reason": str(exc)})
                dt = attempt / 2
                continue
            current = trial
            steps.append(record)
            dt = attempt * (1.3 if record["newton_iterations"] <= 4 else 0.8)
            if current.time_s == target:
                return current, {
                    "state": current.to_dict(),
                    "steps": steps,
                    "rejected": rejected,
                    "next_step_s": dt,
                    "duration_s": target - state.time_s,
                }
        raise HydraulicConvergenceError("Capillary axial report attempt budget exhausted")
