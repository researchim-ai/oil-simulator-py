"""Independent analytic Laplace-pressure limit of the axial work operator."""

from dataclasses import replace

import numpy as np
from test_surface_pr_capillary_pressure import chemical_case

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_capillary_pressure import PRCapillaryPressure
from surface.pr_capillary_transport import PRCapillaryTransportWork


def mode(cells):
    base, seed = chemical_case()
    length, width, depth = 1e-4, 1e-3, 20e-6
    dx = length / cells
    raw = base.pipe.to_dict()
    raw["lengths_m"] = [dx] * cells
    pipe = PRActiveMultifluidPipe.from_dict(raw)
    heights = 3e-6 + 2e-8 * np.cos(2 * np.pi * (np.arange(cells) + 0.5) / cells)
    rows = []
    for h in heights:
        row = {}
        for name, p in seed.cells[0]:
            volume = width * dx * (h if name == base.liquid else depth - h)
            ratio = volume / p.volume_m3
            row[name] = replace(
                p,
                component_moles=tuple(n * ratio for n in p.component_moles),
                volume_m3=volume,
                momentum_kg_m_s=p.momentum_kg_m_s * ratio,
                total_energy_j=p.total_energy_j * ratio,
            )
        rows.append(row)
    old = pipe.initialize(rows)
    after = replace(old, time_s=0.001)
    params = {k: v for k, v in base.to_dict().items() if k not in ("model", "pipe_model_id")}
    params["maximum_unknowns"] = (3 + len(base.active)) * cells
    source = PRCapillaryPressure(pipe, **params)
    report = PRCapillaryTransportWork(source).evaluate(old, after, [0.0] * cells)
    return source, heights, report


def test_laplace_pressure_fourier_symbol_and_second_order_continuum_limit():
    errors = []
    for cells in (16, 32, 64):
        source, h, report = mode(cells)
        dx = source.pipe.lengths[0]
        wave = 2 * np.pi / (cells * dx)
        analytic = source.sigma * wave**2 * (h - 3e-6)
        exact_discrete = analytic * (np.sin(wave * dx / 2) / (wave * dx / 2)) ** 2
        actual = np.array(report["capillary_potential_pa"])
        np.testing.assert_allclose(actual, exact_discrete, atol=1e-10, rtol=1e-9)
        errors.append(np.linalg.norm(actual - analytic) / np.linalg.norm(analytic))
        assert max(abs(x) for x in report["local_energy_residual_j"]) == 0
    assert 3.9 < errors[0] / errors[1] < 4.1
    assert 3.9 < errors[1] / errors[2] < 4.1
