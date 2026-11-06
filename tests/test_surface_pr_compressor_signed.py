"""Synthetic explicit map tests; no inference from a measured compressor card."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.pr_compressor_gasmap import SignedGasMap
from surface.pr_compressor_unit import CompressorBundle, PRCompressorUnit


def fixture(flow=0.01, speed=100.0, composition=(0.99, 0.01)):
    f = binary()
    card = SignedGasMap(100.0, 200.0, 0.1, 0.1, 1.0, 1e5)
    unit = PRCompressorUnit("gas", "a", "b", None, speed, composition, signed_map=card)
    x = f.tensor(composition)
    pin, tin = f.tensor(1e5), f.tensor(350.0)
    qin = f.tensor_homogeneous(pin, tin, x, "vapor")
    mw = (x * f.molar_masses).sum()
    rho = qin["density_kg_m3"]
    head, _, _, _, _, _ = card.evaluate(f.tensor(flow), speed, rho)
    sign = 1 if flow >= 0 else -1
    # Independently construct a PS receiver and PH delivery at prescribed Q.
    z = f.tensor([1.0, 1.0])

    def ps(v):
        q = f.tensor_homogeneous(pin * v[0], tin * v[1], x, "vapor")
        return torch.stack(
            (
                (q["molar_enthalpy_j_mol"] - qin["molar_enthalpy_j_mol"] - mw * sign * head)
                / 1000.0,
                (q["molar_entropy_j_mol_k"] - qin["molar_entropy_j_mol_k"]) / 10.0,
            )
        )

    for _ in range(8):
        z = z - torch.linalg.solve(torch.func.jacfwd(ps)(z), ps(z))
    pout, tis = pin * z[0], tin * z[1]
    tout = tis.clone()
    work = sign * (card.euler_coefficient * abs(speed) + card.linear_coupling) * speed
    for _ in range(8):

        def ph(t):
            return (
                f.tensor_homogeneous(pout, t, x, "vapor")["molar_enthalpy_j_mol"]
                - qin["molar_enthalpy_j_mol"]
                - mw * work
            )

        tout = tout - ph(tout) / torch.func.jacfwd(ph)(tout)

    def node(p, t, xx):
        q = f.tensor_homogeneous(p, t, xx, "vapor")
        return {
            "p": p,
            "t": t,
            "x": xx[None],
            "phase_n": f.tensor([1.0]),
            "q": {k: v[None] for k, v in q.items()},
        }

    donor = node(pin, tin, x)
    receiver = node(pout, tin, f.tensor(tuple(reversed(composition))))
    data = [donor, receiver] if sign > 0 else [receiver, donor]
    model = SimpleNamespace(
        fluid=f,
        compressors=[unit],
        names=("a", "b"),
        gravity_m_s2=0.0,
        reference_elevation_m=0.0,
        nodes=[SimpleNamespace(elevation_m=0.0)] * 2,
    )
    old = SimpleNamespace(temperature_k=350.0)
    state = SimpleNamespace(nodes=[("a", old), ("b", old)])
    bundle = CompressorBundle(model, state, {"gas": True}, 0)
    y = f.tensor([flow, tis / f.tref, tout / f.tref])
    return bundle, y, data


@pytest.mark.parametrize("flow", [0.01, -0.01])
@pytest.mark.parametrize("speed", [100.0, -100.0, 0.0])
def test_signed_ps_ph_physical_end_conservation_jacobian_and_audit(flow, speed, monkeypatch):
    bundle, y, data = fixture(flow, speed)
    net, rows, records, traces, valid = bundle.evaluate(y, data)
    assert valid
    assert rows[0].abs().max() < 1e-10
    point = records["gas"]
    torch.testing.assert_close((net[0] + net[1])[-1], point["shaft_power_w"], atol=1e-9, rtol=1e-10)
    torch.testing.assert_close(
        (net[0] + net[1])[:-1], torch.zeros_like(net[0][:-1]), atol=1e-12, rtol=0
    )
    assert point["entropy_production_w_k"] >= 0
    bundle.validate(traces)
    jac = torch.func.jacfwd(lambda z: bundle.evaluate(z, data)[1][0])(y)
    d = y.new_tensor([0.01, 0.02, -0.03])
    eps = 1e-5
    fd = (bundle.evaluate(y + eps * d, data)[1][0] - bundle.evaluate(y - eps * d, data)[1][0]) / (
        2 * eps
    )
    torch.testing.assert_close(jac @ d, fd, atol=1e-8, rtol=1e-5)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from reference_pr_compressors import audit_compressors

    cfg = {
        "compressors": [asdict(bundle.model.compressors[0])],
        "nodes": [{"name": n, "elevation_m": 0.0} for n in ("a", "b")],
        "gravity_m_s2": 0.0,
        "reference_elevation_m": 0.0,
    }
    reference_data = {
        n: {"p": float(q["p"]), "t": float(q["t"]), "x": q["x"].numpy(), "pn": [1.0]}
        for n, q in zip(("a", "b"), data)
    }
    saved = bundle.serialize(records)
    independent, power = audit_compressors(
        cfg, reference_data, {"gas": True}, saved, bundle.model.fluid
    )
    assert power == pytest.approx(float(point["shaft_power_w"]), rel=1e-8, abs=1e-9)
    for i, n in enumerate(("a", "b")):
        assert independent[n] == pytest.approx(net[i].numpy(), rel=1e-8, abs=1e-8)
    saved["gas"]["shaft_power_w"] += 1.0
    with pytest.raises(ValueError):
        audit_compressors(cfg, reference_data, {"gas": True}, saved, bundle.model.fluid)


def test_zero_flow_nonzero_shutoff_head_and_envelopes():
    bundle, y, data = fixture(0.0, 100.0)
    net, rows, records, _, valid = bundle.evaluate(y, data)
    assert valid and rows[0].abs().max() < 1e-10
    assert all(v.abs().max() == 0 for v in net.values())
    assert records["gas"]["shaft_power_w"] == 0
    assert records["gas"]["isentropic_head_j_kg"] > 0
    assert records["gas"]["load_torque_nm"] == 0
    for flow, speed in ((0.11, 0.0), (-0.11, 0.0), (0.0, 201.0), (0.0, -201.0)):
        assert not bundle.model.compressors[0].signed_map.evaluate(y.new_tensor(flow), speed, 1.0)[
            -1
        ]


def test_explicit_configuration_and_reference_guard():
    bundle, y, data = fixture(-0.01)
    unit = bundle.model.compressors[0]
    assert PRCompressorUnit.from_dict(json.loads(json.dumps(asdict(unit)))) == unit
    wrong = replace(unit, reference_mole_fractions=(0.01, 0.99))
    bundle.entries[0] = (
        *bundle.entries[0][:8],
        y.new_tensor(wrong.reference_mole_fractions),
        bundle.entries[0][-1],
    )
    assert not bundle.evaluate(y, data)[-1]
    with pytest.raises(ValueError):
        SignedGasMap(201.0, 200.0, 0.1, 0.1, 1.0, 1.0)


def test_dissipative_identity_all_quadrants():
    card = SignedGasMap(100.0, 200.0, 0.1, 0.1, 1.0, 1e5)
    for q in (-0.1, 0.0, 0.1):
        for w in (-200.0, 0.0, 200.0):
            _head, _torque, power, hydraulic, dissipation, valid = card.evaluate(
                torch.tensor(q, dtype=torch.float64), w, 2.0
            )
            assert valid
            assert float(power - hydraulic) == pytest.approx(float(dissipation), abs=1e-10)
            assert torch.isfinite(
                torch.func.jacfwd(lambda v, w=w: card.evaluate(v, w, 2.0)[0])(torch.tensor(q))
            )


def test_synthetic_config_isolation_and_frozen_donor():
    root = Path(__file__).resolve().parents[1]
    unit = PRCompressorUnit.from_dict(
        json.loads((root / "configs/pr_compressor_signed_unit.json").read_text())
    )
    assert unit.card is None and unit.signed_map.reference_speed_rad_s == 100.0
    bundle, y, data = fixture(-0.01)
    bundle.donor_signs = {"gas": -1}
    assert bundle.evaluate(y, data)[-1]
    bundle.donor_signs = {"gas": 1}
    assert not bundle.evaluate(y, data)[-1]
    entry = bundle.entries[0]
    bundle.entries[0] = (*entry[:4], False, *entry[5:])
    y[0] = 0.0
    y[1:] = data[0]["t"] / bundle.model.fluid.tref
    net, rows, records, traces, valid = bundle.evaluate(y, data)
    assert valid and net == {} and traces == []
    assert rows[0].abs().max() == 0
    assert records["gas"]["shaft_power_w"] == 0


def test_complete_signed_compressor_facility_runner_and_independent_audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    config = json.loads((root / "configs/surface_pr_signed_compressor.json").read_text())
    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    operating = report["intervals"][0]["steps"][0]["compressors"]
    assert operating["compressor"]["volume_rate_m3_s"] < 0
    assert operating["compressor"]["speed_rad_s"] < 0
    assert operating["forward_compressor"]["volume_rate_m3_s"] > 0
    assert operating["forward_compressor"]["speed_rad_s"] > 0
    for step in report["intervals"][1]["steps"]:
        assert all(point["shaft_power_w"] == 0 for point in step["compressors"].values())
