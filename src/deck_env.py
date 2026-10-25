"""
Среда gymnasium поверх deck Eclipse (black-oil FIM): агент на каждом отчётном шаге масштабирует дебиты скважин
относительно расписания deck и получает экономический результат шага.

    env = DeckEnv("tests/data/spe1/SPE1CASE1.DATA")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(np.ones(env.action_space.shape))

Действие — множитель a ∈ [0, 2] к уставке дебита каждой скважины на текущем шаге (скважины на BHP-контроле,
закрытые и под групповым управлением не меняются; для исторического RESV масштабируются и исторические дебиты,
по которым считается цель). Наблюдение — по скважинам дебиты нефти, воды, газа (м³/сут на поверхности, закачка —
с минусом) и забой (МПа), затем среднее давление пласта (МПа) и доля пройденного времени расписания.
Награда за шаг — (oil_price·Qo − water_cost·Qw − inj_cost·Qw_inj − gas_inj_cost·Qg_inj) / reward_scale,
объёмы в м³. reset(options={"start": путь}) — с чекпоинта run_deck(checkpoint=...), иначе с начала deck.
"""
import contextlib
import io

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from simulator.eclipse_run import iter_deck



class DeckEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, deck, device=None, start=None, max_steps=None, oil_price=1.0, water_cost=0.1,
                 inj_cost=0.05, gas_inj_cost=0.0, reward_scale=1e4, fail_penalty=-100.0, quiet=True):
        super().__init__()
        self.deck, self.start, self.max_steps, self.quiet = deck, start, max_steps, quiet
        self.device = device or torch.device("cpu")
        self.prices = (oil_price, water_cost, inj_cost, gas_inj_cost)
        self.reward_scale, self.fail_penalty = reward_scale, fail_penalty
        self._gen = None
        self._begin()
        n_w = len(self.ctx["wm"].wells)
        self.action_space = spaces.Box(0.0, 2.0, shape=(n_w,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4 * n_w + 2,), dtype=np.float32)

    @property
    def well_names(self):
        return [w.name for w in self.ctx["wm"].wells]

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "start" in options:
            self.start = options["start"]
        self._begin()
        return self._obs(), {"t": self.t}

    def step(self, action):
        self._action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), 0.0, 2.0)
        before = self._cum()
        try:
            _, t, _ = next(self._gen)
        except StopIteration:
            return self._obs(), 0.0, True, False, {"t": self.t, "end_of_schedule": True}
        except RuntimeError as ex:
            return self._obs(), float(self.fail_penalty), True, False, {"t": self.t, "failed": str(ex)[:300]}
        self.t, self.n_steps = t, self.n_steps + 1
        d = self._cum() - before
        po, cw, ci, cg = self.prices
        reward = (po * d[0] - cw * d[1] - ci * d[2] - cg * d[3]) / self.reward_scale
        done = t >= self._t_end - 1e-9 or (self.max_steps is not None and self.n_steps >= self.max_steps)
        info = {"t": t, "oil_m3": d[0], "water_m3": d[1], "water_inj_m3": d[2], "gas_inj_m3": d[3]}
        return self._obs(), float(reward), bool(done), False, info

    def close(self):
        if self._gen is not None:
            self._gen.close()
            self._gen = None

    # ------------------------------------------------------------------ внутреннее
    def _begin(self):
        self.close()
        self._action = None
        self.n_steps = 0
        self.ctx = {}
        out = io.StringIO()
        with contextlib.redirect_stdout(out) if self.quiet else contextlib.nullcontext():
            self._gen = iter_deck(self.deck, device=self.device, quiet=self.quiet, resume=self.start,
                                  control_hook=self._hook, context=self.ctx, yield_start=True)
            _, self.t, _ = next(self._gen)
        self._t_end = float(sum(self.ctx["case"].report_days))

    def _hook(self, n, t, wm, solver):
        a = self._action
        if a is None:
            return
        for k, w in enumerate(wm.wells):
            if w.control_type in ("bhp", "shut") or getattr(w, "grup", False):
                continue
            w.control_value = float(w.control_value) * float(a[k])
            if getattr(w, "hist_rates", None) is not None:
                w.hist_rates = tuple(float(x) * float(a[k]) for x in w.hist_rates)

    def _cum(self):
        out = np.zeros(4)
        for w in self.ctx["wm"].wells:
            cp = getattr(w, "cum_prod", None) or {}
            ci = getattr(w, "cum_inj", None) or {}
            out += (cp.get("oil", 0.0), cp.get("water", 0.0), ci.get("water", 0.0), ci.get("gas", 0.0))
        return out

    def _obs(self):
        rows = []
        for w in self.ctx["wm"].wells:
            q = getattr(w, "last_surface_rates", None) or {}
            bhp = getattr(w, "bhp_mpa", None)
            rows += [q.get("oil", 0.0), q.get("water", 0.0), q.get("gas", 0.0),
                     0.0 if bhp is None else float(bhp)]
        p = float(self.ctx["fluid"].pressure.double().mean()) / 1e6
        rows += [p, self.t / max(self._t_end, 1e-9)]
        return np.asarray(rows, dtype=np.float32)
