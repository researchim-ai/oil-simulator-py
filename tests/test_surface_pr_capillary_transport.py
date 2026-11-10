"""Axial surface-energy ports from actual volume fluxes, not bulk-energy repair."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_surface_pr_capillary_pressure import chemical_case

from surface.liquid import ConservationError
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_capillary_pressure import PRCapillaryPressure
from surface.pr_capillary_transport import PRCapillaryTransportWork


def case(device="cpu", *, periodic=True, sources=False, reverse=False):
    source, seed = chemical_case(device)
    raw = source.pipe.to_dict()
    raw["boundary_mode"] = "periodic" if periodic else "walls"
    raw["lengths_m"] = [1e-5, 1.3e-5, 0.8e-5, 1.1e-5]
    pipe = PRActiveMultifluidPipe.from_dict(raw, device=device)
    cells = []
    for cell, volume in zip(seed.cells, pipe.volumes):
        ratio = volume / sum(p.volume_m3 for _, p in cell)
        cells.append(
            {
                name: replace(
                    p,
                    component_moles=tuple(n * ratio for n in p.component_moles),
                    volume_m3=p.volume_m3 * ratio,
                    momentum_kg_m_s=p.momentum_kg_m_s * ratio,
                    total_energy_j=p.total_energy_j * ratio,
                )
                for name, p in cell
            }
        )
    old = pipe.initialize(cells)
    params = {k: v for k, v in source.to_dict().items() if k not in ("model", "pipe_model_id")}
    source = PRCapillaryPressure(pipe, **params)
    q = np.array([2e-16, -1e-16, 3e-16, -2e-16] if periodic else [0.0, 2e-16, -1e-16, 3e-16, 0.0])
    if reverse:
        q = -q
    supply = np.array([1e-17, -2e-17, 3e-17, 0.0] if sources else [0.0] * 4)
    dv = (np.roll(q, 1) - q if periodic else q[:-1] - q[1:]) + supply
    cells = []
    for cell, change in zip(old.cells, dv):
        row = {}
        for name, p in cell:
            volume = p.volume_m3 + (change if name == source.liquid else -change)
            ratio = volume / p.volume_m3
            row[name] = replace(
                p,
                component_moles=tuple(n * ratio for n in p.component_moles),
                volume_m3=volume,
                momentum_kg_m_s=p.momentum_kg_m_s * ratio,
                total_energy_j=p.total_energy_j * ratio,
            )
        cells.append(row)
    after = pipe.initialize(cells, time_s=old.time_s + 0.001)
    return source, old, after, q, supply


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("sources", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_local_work_and_energy_flux_follow_discrete_variational_identity(
    periodic, sources, reverse
):
    source, old, after, q, supply = case(periodic=periodic, sources=sources, reverse=reverse)
    from check_pr_capillary_transport import audit

    r = PRCapillaryTransportWork(source).evaluate(old, after, q, volume_source_m3=supply)
    assert audit(source, old, after, r)["accepted"]
    lengths = np.array(source.pipe.lengths)
    area = source.width * lengths
    h0, h1 = [
        np.array([dict(c)[source.liquid].volume_m3 for c in s.cells]) / area for s in (old, after)
    ]
    edges = list(zip(range(3), range(1, 4))) + ([(3, 0)] if periodic else [])

    def energy(v):
        h = v / area
        return sum(
            0.5 * source.sigma * source.width / ((lengths[i] + lengths[j]) / 2) * (h[j] - h[i]) ** 2
            for i, j in edges
        )

    midv = (h0 + h1) * area / 2
    for i in range(4):
        delta = np.zeros(4)
        delta[i] = 1e-19
        gradient = (energy(midv + delta) - energy(midv - delta)) / (2e-19)
        assert r["capillary_potential_pa"][i] == pytest.approx(gradient, rel=2e-8, abs=1e-10)
    df = energy(h1 * area) - energy(h0 * area)
    work = sum(r["axial_bulk_work_j"]) + sum(r["normal_bulk_work_j"])
    assert df + work == pytest.approx(0.0, abs=1e-25)
    np.testing.assert_allclose(r["local_energy_residual_j"], 0.0, atol=1e-25, rtol=0)
    np.testing.assert_allclose(
        np.array(r["face_surface_energy_transfer_j"]) / 0.001,
        r["face_surface_energy_flux_w"],
        atol=1e-25,
    )
    if sources:
        assert any(x != 0 for x in r["normal_bulk_work_j"])
    assert any(x != 0 for x in r["axial_bulk_work_j"])
    assert any(x != 0 for x in r["face_shape_work_into_left_j"])


def test_volume_flux_is_checked_and_never_inferred_from_missing_energy():
    source, old, after, q, supply = case(sources=True)
    snapshot = (old.to_dict(), after.to_dict())
    op = PRCapillaryTransportWork(source)
    with pytest.raises(ConservationError, match="volume update"):
        op.evaluate(old, after, q)
    wrong = q.copy()
    wrong[0] += 1e-17
    with pytest.raises(ConservationError, match="volume update"):
        op.evaluate(old, after, wrong, volume_source_m3=supply)
    with pytest.raises(ValueError, match="Finite"):
        op.evaluate(old, after, [float("nan")] * 4)
    with pytest.raises(ValueError, match="per face"):
        op.evaluate(old, after, [0.0] * 3)
    assert snapshot == (old.to_dict(), after.to_dict())


def test_exterior_capillary_geometry_and_positive_interval_are_required():
    source, old, after, q, _supply = case(periodic=False)
    op = PRCapillaryTransportWork(source)
    q[0] = 1e-16
    with pytest.raises(ValueError, match="exterior surface geometry"):
        op.evaluate(old, after, q)
    with pytest.raises(ValueError, match="interval"):
        op.evaluate(old, old, q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_discrete_capillary_ports_match_cpu():
    reports = []
    for device in ("cpu", "cuda"):
        s, old, after, q, supply = case(device, sources=True)
        reports.append(PRCapillaryTransportWork(s).evaluate(old, after, q, volume_source_m3=supply))
    for field in (
        "capillary_potential_pa",
        "axial_bulk_work_j",
        "normal_bulk_work_j",
        "face_surface_energy_transfer_j",
        "face_shape_work_into_left_j",
    ):
        np.testing.assert_allclose(reports[0][field], reports[1][field], rtol=1e-9, atol=1e-24)


def test_graph_ports_preserve_surface_energy_when_line_is_segmented():
    from surface.pr_active_surface import PRActivePipeContinuation, PRActiveSurfaceNetwork

    source, old, after, q, supply = case(sources=True)
    pipes = {}
    for name, offset in (("a", 0), ("b", 2)):
        raw = source.pipe.to_dict()
        raw["boundary_mode"] = "walls"
        raw["lengths_m"] = list(source.pipe.lengths[offset : offset + 2])
        raw["face_elevations_m"] = [0.0] * 3
        pipes[name] = PRActiveMultifluidPipe.from_dict(raw)
    graph = PRActiveSurfaceNetwork(
        {
            "a": {"pipe": pipes["a"], "boundary_nodes": ("return", "join")},
            "b": {"pipe": pipes["b"], "boundary_nodes": ("join", "return")},
        },
        {},
        [],
        continuations={
            "join": PRActivePipeContinuation("a", "b"),
            "return": PRActivePipeContinuation("b", "a"),
        },
    )
    states = []
    for view in (old, after):
        states.append(
            graph.initialize(
                {},
                {
                    name: pipe.initialize(
                        [dict(c) for c in view.cells[offset : offset + 2]], time_s=view.time_s
                    )
                    for (name, pipe), offset in zip(pipes.items(), (0, 2))
                },
                time_s=view.time_s,
            )
        )
    params = {k: v for k, v in source.to_dict().items() if k not in ("model", "pipe_model_id")}
    report = graph.capillary_chain_transport_work(
        *states, ["a", "b"], params, q, volume_source_m3=supply
    )
    whole = PRCapillaryTransportWork(source).evaluate(old, after, q, volume_source_m3=supply)
    assert report["face_surface_energy_transfer_j"] == whole["face_surface_energy_transfer_j"]
    assert set(report["continuations"]) == {"join", "return"}
    for port in report["continuations"].values():
        assert port["surface_energy_into_left_j"] == -port["surface_energy_into_right_j"]
    for segment in report["segments"].values():
        defect = (
            segment["surface_energy_after_j"]
            - segment["surface_energy_before_j"]
            + segment["axial_bulk_work_j"]
            + segment["normal_bulk_work_j"]
            - segment["surface_energy_in_j"]
            + segment["surface_energy_out_j"]
        )
        assert abs(defect) < 1e-25
    with pytest.raises(ValueError, match="Closed continuation"):
        graph.capillary_chain_transport_work(
            *states, ["a", "b"], params, q, volume_source_m3=supply, port_openings={"join": False}
        )


@pytest.mark.parametrize(
    "field",
    [
        "axial_bulk_work_j",
        "face_surface_energy_transfer_j",
        "normal_bulk_work_j",
        "face_shape_work_into_left_j",
    ],
)
def test_independent_audit_rejects_corrupt_work_ports(field):
    from copy import deepcopy

    from check_pr_capillary_transport import audit

    source, old, after, q, supply = case(sources=True)
    report = PRCapillaryTransportWork(source).evaluate(old, after, q, volume_source_m3=supply)
    bad = deepcopy(report)
    bad[field][0] += 1e-18
    with pytest.raises(ValueError, match=field):
        audit(source, old, after, bad)


def test_changed_constitutive_model_is_rejected():
    source, old, after, q, supply = case()
    op = PRCapillaryTransportWork(source)
    source.sigma *= 2
    with pytest.raises(ValueError, match="Changed"):
        op.evaluate(old, after, q, volume_source_m3=supply)
