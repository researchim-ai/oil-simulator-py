"""Rejected FIM trials cannot change the starting state of the retry."""
import copy

import pytest
import torch

from test_blackoil_fim import BASE, _objects
from simulator.simulation import Simulator


def _model(model='blackoil', debug=None):
    cfg = copy.deepcopy(BASE)
    cfg['reservoir'].update(dimensions=[1, 1, 2], grid_size=[30., 30., 100.])
    cfg['simulation']['fim_model'] = model
    cfg['fluid']['s_g'] = .1 if model == 'blackoil' else 0.
    cfg['wells'] = [dict(name='P', type='producer', i=0, j=0, k=0, radius=.1,
                         perforations=[[0,0,0], [0,0,1]], control_type='rate',
                         control_value=20., surface_phase='oil', bhp_min=1.)]
    if debug:
        cfg['simulation']['debug'] = debug
    res, fluid, wells = _objects(cfg)
    fluid.pressure[:, 0, :] = torch.tensor([20e6, 22e6], dtype=torch.float64)
    fluid.s_w[:, 0, :] = torch.tensor([.3, .6], dtype=torch.float64)
    return Simulator(res, fluid, wells, cfg['simulation'], device=torch.device('cpu'))


def _fields(sim):
    return {key: getattr(sim.fluid, key).clone() for key in
            ('pressure', 's_w', 's_o', 's_g', 'rs', 'rv')
            if torch.is_tensor(getattr(sim.fluid, key, None))}


@pytest.mark.parametrize('model', ['blackoil', 'oil_water'])
@pytest.mark.parametrize('failure', ['linear', 'iterations', 'nan'])
def test_rejected_trial_matches_clean_retry(monkeypatch, model, failure):
    sim, clean = _model(model), _model(model)
    solver, wells = sim.fim_solver, sim.fim_solver.wells
    before = _fields(sim)
    heads = getattr(wells, 'conn_head', wells.rho_wb).clone()
    with monkeypatch.context() as patch:
        if failure == 'linear':
            patch.setattr(solver, 'solve_linear', lambda *a, **kw: None)
        elif failure == 'iterations':
            patch.setattr(solver, 'max_newton', 0)
        else:
            name = 'assemble_bo' if model == 'blackoil' else 'assemble_system'
            assemble = getattr(solver, name)

            def nan_residual(*a, **kw):
                result = assemble(*a, **kw)
                result[0][0] = float('nan')
                return result

            patch.setattr(solver, name, nan_residual)
        assert not solver.step(.5 * 86400)
    assert wells.p_bh is None
    assert wells.step_targets is None
    assert solver._p_bh_cur is None
    assert not hasattr(solver, '_step_initial')
    torch.testing.assert_close(getattr(wells, 'conn_head', wells.rho_wb), heads, rtol=0, atol=0)
    for key, value in before.items():
        torch.testing.assert_close(getattr(sim.fluid, key), value, rtol=0, atol=0)
    assert solver.step(.001 * 86400)
    assert clean.fim_solver.step(.001 * 86400)
    for key, value in _fields(clean).items():
        torch.testing.assert_close(getattr(sim.fluid, key), value, rtol=0, atol=0)
    torch.testing.assert_close(wells.p_bh, clean.fim_solver.wells.p_bh, rtol=0, atol=0)


@pytest.mark.parametrize('model', ['blackoil', 'oil_water'])
def test_exception_during_commit_restores_accepted_state(monkeypatch, tmp_path, model):
    sim = _model(model, dict(enabled=True, console=False, dump_on_failure=False,
                             output_dir=str(tmp_path), run_id=model))
    solver = sim.fim_solver
    assert solver.step(.001 * 86400)
    fields = _fields(sim)
    pressure_reference = sim.fluid.pressure
    well = solver.wells.wells[0]
    rates = dict(well.last_surface_rates)
    cumulative = dict(getattr(well, 'cum_surface', {}))
    cumulative_reference = getattr(well, 'cum_surface', None)
    pb = solver.wells.p_bh.clone()
    storage = solver._store_well_results

    def failing_commit(*a, **kw):
        storage(*a, **kw)
        raise RuntimeError('commit interrupted')

    try:
        with monkeypatch.context() as patch:
            patch.setattr(solver, '_store_well_results', failing_commit)
            with pytest.raises(RuntimeError, match='commit interrupted'):
                solver.step(.001 * 86400)
        assert sim.fluid.pressure is pressure_reference
        assert not hasattr(solver, '_step_initial')
        for key, value in fields.items():
            torch.testing.assert_close(getattr(sim.fluid, key), value, rtol=0, atol=0)
        torch.testing.assert_close(solver.wells.p_bh, pb, rtol=0, atol=0)
        assert well.last_surface_rates == rates
        if cumulative_reference is not None:
            assert well.cum_surface is cumulative_reference
            assert cumulative_reference == cumulative
        assert solver.debug.summary['steps'] == 2
        assert solver.debug.summary['failed_steps'] == 1
        assert solver.step(.001 * 86400)
    finally:
        solver.debug.close()


def test_aborted_well_setup_restores_lift_network_and_segment_state(monkeypatch):
    sim = _model()
    solver, wells = sim.fim_solver, sim.fim_solver.wells
    well = wells.wells[0]
    well.alq, well.thp_limit = 10., 2.
    wells.alq_opt, wells._lift_t = {'P': 10.}, 0.
    wells.node_p = {'N': 2e6}
    wells.reinit_bhp = {0}
    static = torch.tensor([0, 1])
    wells.msw = {0: dict(head=None, conns=static)}

    def aborted_setup(*a, **kw):
        well.alq, well.thp_limit = 99., 8.
        wells.alq_opt, wells._lift_t = {'P': 99.}, 5.
        wells.node_p = {'N': 8e6}
        wells.reinit_bhp.clear()
        wells.msw[0].update(head=torch.ones(2), p_seg=torch.ones(2), rho=torch.ones(2))
        raise RuntimeError('setup interrupted')

    monkeypatch.setattr(wells, 'begin_step', aborted_setup)
    with pytest.raises(RuntimeError, match='setup interrupted'):
        solver.step(864.)
    assert (well.alq, well.thp_limit) == (10., 2.)
    assert wells.alq_opt == {'P': 10.} and wells._lift_t == 0.
    assert wells.node_p == {'N': 2e6} and wells.reinit_bhp == {0}
    assert wells.msw[0]['head'] is None
    assert 'p_seg' not in wells.msw[0] and 'rho' not in wells.msw[0]
    assert wells.msw[0]['conns'] is static
