"""Scalar finite-volume audit of capillary axial work, no runtime force helpers."""

import math

import numpy as np
from check_pr_capillary_pressure import close

from surface.operation import _digest


def audit(source, before, after, report):
    pipe = source.pipe
    if report["model"] != "pr_capillary_transport_work_v1" or (
        report["source_model_id"] != source.model_id
        or _digest(report["source_model"]) != _digest(source.to_dict())
    ):
        raise ValueError("Capillary transport model differs")
    dt = after.time_s - before.time_s
    if not math.isfinite(dt) or dt <= 0 or report["dt_s"] != dt:
        raise ValueError("Capillary transport clock differs")
    lengths = np.array(pipe.lengths)
    area = source.width * lengths
    n = len(lengths)
    periodic = pipe.boundary_mode == "periodic"
    states = [before, after]
    for state in states:
        if state.model_id != pipe.model_id or len(state.cells) != n:
            raise ValueError("Capillary transport geometry differs")
        if any(set(dict(c)) != {source.liquid, source.gas} for c in state.cells):
            raise ValueError("Capillary transport topology differs")
    volumes = [
        np.array([dict(c)[source.liquid].volume_m3 for c in state.cells]) for state in states
    ]
    h0, h1 = [v / area for v in volumes]
    q, supply = np.array(report["integrated_liquid_flux_m3"]), np.array(report["volume_source_m3"])
    if q.shape != ((n,) if periodic else (n + 1,)) or supply.shape != (n,):
        raise ValueError("Invalid transport face/source dimensions")
    if not np.isfinite(q).all() or not np.isfinite(supply).all():
        raise ValueError("Nonfinite transport volume update")
    if not periodic and (q[0] != 0 or q[-1] != 0):
        raise ValueError("Missing exterior capillary geometry")
    divergence = np.roll(q, 1) - q if periodic else q[:-1] - q[1:]
    volume_residual = volumes[1] - volumes[0] - divergence - supply
    scale = abs(volumes[0]) + abs(volumes[1]) + abs(divergence) + abs(supply)
    close(
        volume_residual,
        np.zeros(n),
        "actual volume flux",
        atol=128 * np.finfo(float).eps * scale,
        rtol=0,
    )
    edges = list(zip(range(n - 1), range(1, n))) + ([(n - 1, 0)] if periodic else [])
    energies = [np.zeros(n), np.zeros(n)]
    mu = np.zeros(n)
    shapes = np.zeros(len(q))
    mid = (h0 + h1) / 2
    for i, j in edges:
        distance = (lengths[i] + lengths[j]) / 2
        if max(abs(h0[j] - h0[i]), abs(h1[j] - h1[i])) > source.maximum_slope * distance:
            raise ValueError("Unsupported capillary slope")
        weight = source.sigma * source.width / distance
        mu[i] += weight * (mid[i] - mid[j]) / area[i]
        mu[j] += weight * (mid[j] - mid[i]) / area[j]
        for h, e in zip((h0, h1), energies):
            value = 0.25 * weight * (h[j] - h[i]) ** 2
            e[i] += value
            e[j] += value
        face = i if periodic else i + 1
        shapes[face] = weight * (mid[j] - mid[i]) * ((h1[i] - h0[i]) + (h1[j] - h0[j])) / 2
    if source.wetting is not None:
        from reference_wetting_potential import potential as wall_energy
        from reference_wetting_potential import pressure as wall_pressure

        params = source.wetting.to_dict()
        mu += wall_pressure(params, source.sigma, h0, h1)
        energies[0] += area * wall_energy(params, source.sigma, h0)
        energies[1] += area * wall_energy(params, source.sigma, h1)
    work = np.zeros(n)
    face_work = np.zeros(len(q))
    transfer = np.zeros(len(q))
    for i, j in edges:
        face = i if periodic else i + 1
        face_work[face] = -q[face] * (mu[j] - mu[i])
        work[i] += face_work[face] / 2
        work[j] += face_work[face] / 2
        transfer[face] = (mu[i] + mu[j]) * q[face] / 2 - shapes[face]
    normal = -mu * supply
    incoming = np.roll(transfer, 1) - transfer if periodic else transfer[:-1] - transfer[1:]
    residual = energies[1] - energies[0] + work + normal - incoming
    expected = {
        "capillary_potential_pa": mu,
        "surface_energy_before_j": energies[0],
        "surface_energy_after_j": energies[1],
        "axial_bulk_work_j": work,
        "normal_bulk_work_j": normal,
        "face_bulk_work_j": face_work,
        "face_shape_work_into_left_j": shapes,
        "face_surface_energy_transfer_j": transfer,
        "face_surface_energy_flux_w": transfer / dt,
        "volume_residual_m3": volume_residual,
        "local_energy_residual_j": residual,
        "closed_energy_residual_j": math.fsum(energies[1] - energies[0] + work + normal),
    }
    for name, value in expected.items():
        tolerance = 1e-10 if name == "capillary_potential_pa" else 1e-25
        if name == "volume_residual_m3":
            tolerance = 128 * np.finfo(float).eps * scale
        close(report[name], value, name, atol=tolerance, rtol=1e-9)
    close(residual, np.zeros(n), "local surface transport identity", atol=1e-25, rtol=0)
    return {
        "accepted": True,
        "cells": n,
        "faces": len(q),
        "qualification": "Constitutive geometry/work audit, not a joint momentum/EOS evolution validation",
    }
