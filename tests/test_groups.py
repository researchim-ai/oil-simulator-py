import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.groups import GroupTree, allocate, group_targets  # noqa: E402

SPE9_DIR = os.path.join(os.path.dirname(__file__), "data", "spe9")


def test_allocate_redistributes_shortfall():
    s = allocate(100.0, guide=[1, 1, 2], cap=[10, 100, 100])
    assert s[0] == pytest.approx(10.0)
    assert s.sum() == pytest.approx(100.0)
    assert s[2] == pytest.approx(2 * s[1])
    assert allocate(1000.0, [1, 1], [10, 20]).tolist() == pytest.approx([10, 20])


def test_group_targets_hierarchy_and_own_limits():
    tree = GroupTree({"A": "FIELD", "B": "FIELD"}, {"P1": "A", "P2": "A", "P3": "B"})
    wells = [dict(name=n, type="producer", open=True, grup=False, phase="oil", value=v, inj_phase=None)
             for n, v in (("P1", 50.0), ("P2", 50.0), ("P3", 5.0))]
    pot = {n: {"oil": 100.0, "water": 0.0, "gas": 0.0, "resv": 100.0} for n in ("P1", "P2", "P3")}
    cur = {n: {"oil": 0.0, "water": 0.0, "gas": 0.0} for n in pot}
    # группа A не нарушена (50+50 ≤ 120) — режимы скважин не трогаются
    assert group_targets(tree, [dict(group="A", kind="prod", phase="oil", target=120.0)], wells, pot, cur) == {}
    # месторождение 60: P3 ограничена собственной целью 5, остальное делится поровну
    new = group_targets(tree, [dict(group="FIELD", kind="prod", phase="oil", target=60.0)], wells, pot, cur)
    assert new["P3"][1] == pytest.approx(5.0)
    assert new["P1"][1] == pytest.approx(27.5) and new["P2"][1] == pytest.approx(27.5)


def test_group_iteration_can_restore_nominal_well_targets():
    from types import SimpleNamespace
    import torch
    from simulator.blackoil_fim import BlackOilWells, DAY

    tree = GroupTree({}, {"P1": "FIELD", "P2": "FIELD"})
    constraint = dict(group="FIELD", kind="prod", phase="oil", target=60.0 * DAY)
    wells = [SimpleNamespace(name=n, type="producer", surface_phase="oil", grup=False,
                             rate_limits={}) for n in ("P1", "P2")]
    q = torch.tensor([[100., 0., 0.], [100., 0., 0.]], dtype=torch.float64)
    a = torch.tensor([[1., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    nominal = [("rate", 50.0), ("rate", 50.0)]
    fake = SimpleNamespace(groups=(tree, [constraint]), conn_well=torch.arange(2), n_w=2,
                           device="cpu", rho_sc=torch.ones(3), wells=wells, p_bh=torch.zeros(2),
                           is_inj=torch.zeros(2, dtype=torch.bool), _resv_idx=set(), liftopt=None,
                           has_thp=lambda i: False, bhp_limit=lambda w: 0.0,
                           _resv_weights=lambda v: torch.ones_like(q), conn_a=a.clone(),
                           step_targets=list(nominal), _group_base_targets=nominal, _group_base_a=a)
    BlackOilWells._apply_groups(fake, None, q, torch.zeros_like(q))
    assert [val for _, val in fake.step_targets] == pytest.approx([30., 30.])
    constraint["target"] = 120.0 * DAY
    BlackOilWells._apply_groups(fake, None, q, torch.zeros_like(q))
    assert fake.step_targets == nominal


def test_network_capacity_uses_pressure_control_until_rate_limit_binds():
    from types import SimpleNamespace
    import torch
    from simulator.blackoil_fim import BlackOilWells

    fake = SimpleNamespace(network=None, _Q=torch.ones(1, 3), _lin=object(), n_w=1,
                           wells=[SimpleNamespace(name="P")], step_targets=[("rate", 10.0)],
                           step_mode=["rate"], switches=0, well_switches=[0],
                           bhp_limit=lambda w: 80.0, has_thp=lambda i: True,
                           _vfp_bhp=lambda i, q: 100.0, _sgn=lambda i: 1.0,
                           _thp_stable=lambda i, lim: (100.0, 10.0), _group_capacity_limited={0})
    pb, q = torch.tensor([100.0]), torch.tensor([10.01])
    BlackOilWells.update_controls(fake, pb, q)
    assert fake.step_mode == ["thp_limit"]
    BlackOilWells.update_controls(fake, pb, q)
    assert fake.step_mode == ["thp_limit"]
    # Теперь ограничение группы ниже потенциала: реальную уставку дебита превышать нельзя.
    fake._group_capacity_limited.clear()
    fake.step_targets = [("rate", 9.0)]
    BlackOilWells.update_controls(fake, pb, q)
    assert fake.step_mode == ["rate"]


def _spe9_with_groups(tmp_path, extra):
    text = open(os.path.join(SPE9_DIR, "SPE9_CP.DATA")).read()
    for inc in ("'SPE9.GRDECL'", "PERMVALUES.DATA"):
        text = text.replace(inc, "'" + os.path.join(SPE9_DIR, inc.strip("'")) + "'")
    text = text.replace("SUMMARY\n", "SUMMARY\nGOPR\n 'FIELD' /\nGWIR\n 'FIELD' /\n", 1)
    i = text.index("TSTEP")
    path = tmp_path / "SPE9_GROUPS.DATA"
    path.write_text(text[:i] + extra + text[i:])
    return str(path)


@pytest.fixture(scope="module")
def spe9_groups(tmp_path_factory):
    from simulator.eclipse_run import run_deck
    extra = "GCONPROD\n 'FIELD' 'ORAT' 20000 /\n/\nGCONINJE\n 'FIELD' 'WATER' 'RATE' 500 /\n/\n"
    s, _ = run_deck(_spe9_with_groups(tmp_path_factory.mktemp("g"), extra), max_days=100)
    return s


def test_spe9_field_oil_target_is_met(spe9_groups):
    s = spe9_groups
    assert np.allclose(s["FOPR"][1:], 20000.0, rtol=1e-4)
    assert np.allclose(s["GOPR:FIELD"][1:], s["FOPR"][1:])
    # без группы каждая из 25 скважин дала бы 1500 — доли распределены по потенциалам, а не поровну
    wopr = np.array([s[k][-1] for k in s if k.startswith("WOPR:PRODU")])
    assert wopr.max() <= 1500.0 * (1 + 1e-6) and wopr.std() > 1.0


def test_spe9_field_water_injection_target_is_met(spe9_groups):
    assert np.allclose(spe9_groups["GWIR:FIELD"][1:], 500.0, rtol=1e-4)
