"""Среда DeckEnv: множитель 1 повторяет run_deck, управление дебитом действует, reset и старт с чекпоинта."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
pytest.importorskip("gymnasium")

from deck_env import DeckEnv  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")
N = 4


def _roll(env, action):
    oil = 0.0
    for _ in range(N):
        obs, r, done, _, info = env.step(action)
        assert "failed" not in info, info
        oil += info["oil_m3"]
    return obs, oil, info["t"]


def test_unit_action_reproduces_run_deck():
    env = DeckEnv(SPE1)
    obs0, _ = env.reset()
    assert obs0.shape == env.observation_space.shape and np.isfinite(obs0).all()
    obs, oil, t = _roll(env, np.ones(env.action_space.shape))
    s, info = run_deck(SPE1, max_days=t)
    assert info["convergence"]["tol_cnv"] == 1e-3
    assert info["convergence"]["relaxed_max_pv_fraction"] == 0
    liq = info["case"].u["liq"]     # м³ на единицу объёма жидкости deck
    k = env.well_names.index("PROD")
    assert np.isclose(obs[4 * k], s["FOPR"][-1] * liq, rtol=1e-6)
    assert np.isclose(oil, float((s["FOPR"][1:] * np.diff(s["TIME"])).sum()) * liq, rtol=2e-2)


def test_rate_action_and_reset():
    env = DeckEnv(SPE1)
    env.reset()
    _, oil_full, _ = _roll(env, np.ones(env.action_space.shape))
    obs0, _ = env.reset()
    a = np.ones(env.action_space.shape)
    a[env.well_names.index("PROD")] = 0.5
    _, oil_half, _ = _roll(env, a)
    assert 0.3 * oil_full < oil_half < 0.7 * oil_full
    obs_again, _ = env.reset()
    assert np.array_equal(obs0, obs_again)


def test_start_from_checkpoint(tmp_path):
    ck = str(tmp_path / "spe1.ckpt")
    s, _ = run_deck(SPE1, max_days=100, checkpoint=ck)
    env = DeckEnv(SPE1, start=ck)
    obs, info = env.reset()
    assert np.isclose(info["t"], s["TIME"][-1])
    obs, r, done, _, info = env.step(np.ones(env.action_space.shape))
    assert info["t"] > s["TIME"][-1] and np.isfinite(r)


@pytest.mark.parametrize("failure", [True, False])
def test_terminal_info_has_consistent_volume_keys(failure):
    env = DeckEnv.__new__(DeckEnv)
    env.t, env.fail_penalty = 12.0, -100.0
    env._cum = lambda: np.zeros(4)
    env._obs = lambda: np.zeros(1)

    def stopped():
        if failure:
            raise RuntimeError("linear solver failed")
        return
        yield

    env._gen = stopped()
    _, reward, done, _, info = env.step(np.ones(1))
    assert done and info["t"] == 12.0
    assert all(info[k] == 0 for k in ("oil_m3", "water_m3", "water_inj_m3", "gas_inj_m3"))
    assert ("failed" in info) == failure
    assert reward == (-100.0 if failure else 0.0)
