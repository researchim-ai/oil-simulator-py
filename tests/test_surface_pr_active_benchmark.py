"""The scaling benchmark must advance a real coupled, two-phase network."""

from benchmark_pr_active_surface import build, independent_totals


def test_benchmark_contains_connected_multiphase_columns_pump_and_finite_walls():
    model, old = build(8, 4, "cpu")
    assert len(old.pipes) == 4 and len(old.columns) == 2
    assert len(old.equipment.rotors) == 1
    assert sum(len(cell) for _, p in old.pipes for cell in p.cells) == 16
    a = independent_totals(model, old)
    new, r = model.step(old, 0.25 * model.maximum_step_s)
    b = independent_totals(model, new)
    assert max(abs(y - x) for x, y in zip(a[0], b[0])) < 1e-7
    assert abs(b[1] - a[1] - r["external_energy_j"]) < 1e-5
    assert dict(new.equipment.rotors) != dict(old.equipment.rotors)
