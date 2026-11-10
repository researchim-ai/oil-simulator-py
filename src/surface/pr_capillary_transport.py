"""Discrete-gradient capillary transport work on fixed covered PR topology.

This constitutive operator consumes actual integrated liquid-volume fluxes.
It supplies capillary work/energy ports for a future joint momentum solve;
it does not retrofit those forces into the old BN/Rusanov integrator.
No total-energy residual is redistributed into bulk stocks.
"""

import math

import torch

from .liquid import ConservationError, _number
from .operation import _digest
from .pr_capillary_pressure import PRCapillaryPressure


class PRCapillaryTransportWork:
    def __init__(self, source):
        if (
            not isinstance(source, PRCapillaryPressure)
            or source.to_dict()["model"] != "pr_capillary_pressure_v1"
        ):
            raise TypeError("Resolved PR capillary constitutive source required")
        if any(h is not None for h in source.boundary_heights):
            raise ValueError("Fixed face geometry requires external capillary work ports")
        self.source = source
        self.pipe = source.pipe

    def evaluate(self, before, after, integrated_liquid_flux_m3, *, volume_source_m3=None):
        """Return integrated work/ports for one actual volume update.

        Periodic face i joins cell i to i+1, including the last-to-first face.
        Wall-mode faces run left to right, with n+1 faces and sealed ends.
        Volume sources describe independently known normal/chemical exchange;
        they are never inferred from a balance defect. Their capillary work
        must be supplied by that source, not counted twice in axial work.
        """
        s, pipe = self.source, self.pipe
        if _digest(s.to_dict()) != s.model_id:
            raise ValueError("Changed capillary transport constitutive model")
        pipe.inspect(before)
        pipe.inspect(after)
        s.surface_energy(before)
        s.surface_energy(after)
        dt = _number(after.time_s - before.time_s, "capillary transport interval", positive=True)
        f = pipe.fluid
        n = len(pipe.lengths)
        periodic = pipe.boundary_mode == "periodic"
        count = n if periodic else n + 1
        q = f.tensor(integrated_liquid_flux_m3)
        supply = f.tensor([0.0] * n if volume_source_m3 is None else volume_source_m3)
        if q.shape != (count,) or supply.shape != (n,):
            raise ValueError(
                "One integrated flux per face and independent source per cell required"
            )
        if not bool(torch.isfinite(q).all().cpu()) or not bool(torch.isfinite(supply).all().cpu()):
            raise ValueError("Finite volume fluxes and sources required")
        if not periodic and bool((q[[0, -1]] != 0).any().cpu()):
            raise ValueError("Open capillary ends require exterior surface geometry")
        volumes = [
            f.tensor([dict(c)[s.liquid].volume_m3 for c in v.cells]) for v in (before, after)
        ]
        dv = volumes[1] - volumes[0]
        divergence = torch.roll(q, 1) - q if periodic else q[:-1] - q[1:]
        defect = dv - divergence - supply
        scale = volumes[0].abs() + volumes[1].abs() + divergence.abs() + supply.abs()
        allowance = 128 * torch.finfo(q.dtype).eps * scale
        if bool((defect.abs() > allowance).any().cpu()):
            raise ConservationError("Actual liquid-volume update differs from supplied flux/source")
        h0, h1 = [v / s.area for v in volumes]
        mid = (h0 + h1) / 2
        mu = s.discrete_pressure(h0, h1)
        # All faces are evaluated together, including continuation faces.
        i = torch.arange(n if periodic else n - 1, device=f.device)
        j = (i + 1) % n
        face = i if periodic else i + 1
        e0 = 0.5 * s.weights * s.height_jump(h0).square()
        e1 = 0.5 * s.weights * s.height_jump(h1).square()
        assigned0 = torch.zeros_like(mu).index_add(0, i, e0 / 2).index_add(0, j, e0 / 2)
        assigned1 = torch.zeros_like(mu).index_add(0, i, e1 / 2).index_add(0, j, e1 / 2)
        if s.wetting is not None:
            assigned0 = assigned0 + s.area * s.wetting.potential(h0)
            assigned1 = assigned1 + s.area * s.wetting.potential(h1)
        # Pressure difference is conjugate to actual integrated volume q.
        work = -q[face] * (mu[j] - mu[i])
        axial_work = torch.zeros_like(mu).index_add(0, i, work / 2).index_add(0, j, work / 2)
        shape = s.weights * (mid[j] - mid[i]) * ((h1[i] - h0[i]) + (h1[j] - h0[j])) / 2
        transfer = (mu[i] + mu[j]) * q[face] / 2 - shape
        face_work = torch.zeros_like(q).index_copy(0, face, work)
        shape_work = torch.zeros_like(q).index_copy(0, face, shape)
        energy_transfer = torch.zeros_like(q).index_copy(0, face, transfer)
        normal_work = -mu * supply
        incoming = (
            torch.roll(energy_transfer, 1) - energy_transfer
            if periodic
            else energy_transfer[:-1] - energy_transfer[1:]
        )
        residual = assigned1 - assigned0 + axial_work + normal_work - incoming
        energy_scale = (
            assigned0.abs()
            + assigned1.abs()
            + axial_work.abs()
            + normal_work.abs()
            + incoming.abs()
            + mu.abs() * scale
        )
        if bool(
            (residual.abs() > 512 * torch.finfo(q.dtype).eps * energy_scale + 1e-30).any().cpu()
        ):
            raise ConservationError("Local axial capillary energy/work identity failed")
        total = float((assigned1 - assigned0 + axial_work + normal_work).sum().cpu())
        if not math.isfinite(total):
            raise ConservationError("Nonfinite axial capillary work")
        return {
            "model": "pr_capillary_transport_work_v1",
            "source_model_id": s.model_id,
            "source_model": s.to_dict(),
            "dt_s": dt,
            "capillary_potential_pa": mu.cpu().tolist(),
            "integrated_liquid_flux_m3": q.cpu().tolist(),
            "volume_source_m3": supply.cpu().tolist(),
            "surface_energy_before_j": assigned0.cpu().tolist(),
            "surface_energy_after_j": assigned1.cpu().tolist(),
            "axial_bulk_work_j": axial_work.cpu().tolist(),
            "normal_bulk_work_j": normal_work.cpu().tolist(),
            "face_bulk_work_j": face_work.cpu().tolist(),
            "face_shape_work_into_left_j": shape_work.cpu().tolist(),
            "face_surface_energy_transfer_j": energy_transfer.cpu().tolist(),
            "face_surface_energy_flux_w": (energy_transfer / dt).cpu().tolist(),
            "volume_residual_m3": defect.cpu().tolist(),
            "local_energy_residual_j": residual.cpu().tolist(),
            "closed_energy_residual_j": total,
            "qualification": "Fixed-topology constitutive work ports; joint axial momentum/thermodynamic evolution is not implemented by this operator",
        }
