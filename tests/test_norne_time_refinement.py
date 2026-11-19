from copy import deepcopy
import numpy as np
import pytest

from check_norne_time_refinement import check_nested, protocol
from check_fim_time_refinement import difference_order


def grids():
    coarse = np.array([[0., .3, 1., 4.], [.3, .7, 1., 5.]])
    result = [coarse]
    for _ in range(2):
        previous = result[-1]
        refined = np.repeat(previous, 2, axis=0)
        refined[:, 1] /= 2
        refined[1::2, 0] += refined[1::2, 1]
        result.append(refined)
    return result


def test_nested_nonuniform_steps_preserve_events():
    check_nested(grids())


@pytest.mark.parametrize('kind', ['gap', 'rejected', 'not_halved', 'missing', 'nan'])
def test_invalid_refinement_is_rejected(kind):
    g = grids()
    if kind == 'gap': g[2][1, 0] += .01
    if kind == 'rejected': g[1][0, 2] = 0
    if kind == 'not_halved': g[1][0, 1] *= .9
    if kind == 'missing': g[1] = g[1][:-1]
    if kind == 'nan': g[0][0, 1] = np.nan
    with pytest.raises(ValueError):
        check_nested(g)


def test_only_grid_input_may_differ(tmp_path):
    manifests = []
    for i in range(3):
        grid = str(tmp_path / f'{i}.npz')
        manifests.append(dict(sources={'solver.py': 'same'}, launcher_sha256='same',
                              effective_settings={'tol_cnv': 1e-4},
                              arguments={'grid': grid}, inputs={grid: str(i), 'deck': 'same'}))
    protocol(manifests)
    for key in ('sources', 'effective_settings', 'inputs'):
        changed = deepcopy(manifests)
        changed[1][key]['changed'] = 'different'
        with pytest.raises(ValueError, match='Different'):
            protocol(changed)


def test_nonuniform_backward_euler_has_first_order_refinement():
    # Independent ODE y'=-y, exact endpoint exp(-1). No simulator invoked.
    solutions = [np.prod(1/(1+g[:, 1])) for g in grids()]
    errors = [abs(y-np.exp(-1)) for y in solutions]
    assert errors[2] < errors[1] < errors[0]
    result = difference_order(*[[y] for y in solutions])
    assert result['decreases'] and .6 < result['observed_order'] < 1.2
