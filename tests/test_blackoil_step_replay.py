"""The reproducer must preserve execution policy and solver acceptance."""
import importlib.util
from pathlib import Path

import pytest
import torch

from test_fim_facility_coupling import build


def test_replay_matches_small_independent_model(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/replay_blackoil_step.py'
    spec = importlib.util.spec_from_file_location('step_replay', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solver = build(device='cpu').solver
    execution = dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                     warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                     threads=torch.get_num_threads(),
                     matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_tf32=torch.backends.cudnn.allow_tf32)
    state = dict(solver=solver, dt=60., current_time=0., execution=execution,
                 source_hashes={})
    snapshot = tmp_path / 'step.pt'
    torch.save(state, snapshot)
    expected = solver.step(60., current_time=0.)
    result = module.replay(snapshot, tmp_path / 'replay')
    assert result['status'] == ('accepted' if expected else 'rejected')
    assert expected
    assert result['newton_corrections'] == solver.last_newton_iters
    assert result['execution'] == execution
    assert result['criteria'][-1]['unrelaxed_converged']
    state['source_hashes'] = {'src/simulator/blackoil_fim.py': 'invalid'}
    torch.save(state, snapshot)
    with pytest.raises(ValueError, match='sources differ'):
        module.replay(snapshot, tmp_path / 'mismatch')
