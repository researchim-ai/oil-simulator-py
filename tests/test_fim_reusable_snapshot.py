"""A rollback snapshot must survive mutable well data in successive trials."""
from types import SimpleNamespace

import torch

from simulator.fim_transaction import _StepState


def test_repeated_restore_isolates_wells_segments_and_nested_cumulatives():
    well = SimpleNamespace(control_value=10., cum_prod={'oil': {'mass': 2.}})
    segment = {'head': torch.tensor([3.])}
    wells = SimpleNamespace(wells=[well], p_bh=torch.tensor([4.]),
                            msw={'w': segment}, switches={'a': [1]})
    fluid = SimpleNamespace(pressure=torch.tensor([10.], dtype=torch.float64),
                            s_w=torch.tensor([.2]), s_o=torch.tensor([.8]))
    solver = SimpleNamespace(fluid=fluid, wells=wells)
    snapshot = _StepState(solver)
    storage = snapshot.old['pressure'].data_ptr()
    for _ in range(3):
        snapshot.restore(reusable=True)
        assert snapshot.old['pressure'].data_ptr() == storage
        assert fluid.pressure.item() == 10.
        assert wells.p_bh.item() == 4.
        assert segment['head'].item() == 3.
        assert wells.switches == {'a': [1]}
        assert well.cum_prod == {'oil': {'mass': 2.}}
        assert not hasattr(wells, 'step_targets')
        fluid.pressure.add_(100.)
        wells.p_bh.add_(5.)
        segment['head'].add_(7.)
        wells.switches['a'].append(9)
        well.cum_prod['oil']['mass'] = 100.
        well.control_value = 0.
        wells.step_targets = torch.ones(1)
    snapshot.restore()
    assert fluid.pressure.item() == 10.
    assert wells.p_bh.item() == 4.
    assert well.control_value == 10.
    assert well.cum_prod == {'oil': {'mass': 2.}}
