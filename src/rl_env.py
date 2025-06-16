import json
from typing import Tuple, Dict

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from simulator.reservoir import Reservoir
from simulator.fluid import Fluid
from simulator.well import WellManager
from simulator.simulation import Simulator


class ReservoirEnv(gym.Env):
    """Gymnasium-совместимый энвайронмент для обучения RL-агентов управлению пластом."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        config_path: str,
        max_steps: int = 365,
        device: torch.device | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.config_path = config_path
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Создаём объекты симуляции по конфигу ---
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.reservoir = Reservoir(cfg["reservoir"], device=self.device)
        self.well_manager = WellManager(cfg["wells"], self.reservoir)
        self.fluid = Fluid(cfg["fluid"], self.reservoir, device=self.device)
        self.simulator = Simulator(
            reservoir=self.reservoir,
            fluid=self.fluid,
            well_manager=self.well_manager,
            sim_params=cfg.get("simulation", {}),
            device=self.device,
        )

        # --- Экспозиция spaces ---
        n_wells = len(self.well_manager.wells)
        # Действие: изменение дебита / давления в диапазоне [-1,1] для каждого скважины
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_wells,), dtype=np.float32)

        # Наблюдение: давления и насыщенности, нормализованные
        nx, ny, nz = self.reservoir.dimensions
        obs_len = nx * ny * nz * 2  # давление + насыщенность
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_len,), dtype=np.float32)

        # --- Внутренние состояния ---
        self._step_count = 0
        self._prev_oil_mass = self._calc_oil_mass().item()

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        # Перечитываем конфиг для жёсткого ресета
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Переинициализируем объекты
        self.reservoir = Reservoir(cfg["reservoir"], device=self.device)
        self.well_manager = WellManager(cfg["wells"], self.reservoir)
        self.fluid = Fluid(cfg["fluid"], self.reservoir, device=self.device)
        self.simulator = Simulator(
            reservoir=self.reservoir,
            fluid=self.fluid,
            well_manager=self.well_manager,
            sim_params=cfg.get("simulation", {}),
            device=self.device,
        )

        self._step_count = 0
        self._prev_oil_mass = self._calc_oil_mass().item()
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._apply_action(action)

        # Выполняем один временной шаг симулятора
        dt_days = self.simulator.sim_params.get("time_step_days", 1.0)
        dt_sec = dt_days * 24 * 3600
        success = self.simulator.run_step(dt_sec)
        if not success:
            # Если решатель не сошёлся — раннее завершение эпизода с большим штрафом
            obs = self._get_observation()
            return obs, -1e3, True, False, {"converged": False}

        self._step_count += 1
        obs = self._get_observation()
        reward = self._calc_reward()
        terminated = self._step_count >= self.max_steps
        truncated = False
        info = {"converged": True}
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        if mode != "human":
            raise NotImplementedError
        # Простейший вывод среднего давления и добытой нефти
        p_mean = float(self.fluid.pressure.mean() / 1e6)
        print(f"Среднее давление: {p_mean:.2f} МПа, шаг {self._step_count}")

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        p = self.fluid.pressure.cpu().numpy().flatten() / 1e7  # нормируем к десяткам МПа
        sw = self.fluid.s_w.cpu().numpy().flatten()
        obs = np.concatenate([p, sw]).astype(np.float32)
        return obs

    def _apply_action(self, action: np.ndarray):
        # Масштабируем действие к диапазону управления каждой скважиной
        for a, well in zip(action, self.well_manager.wells):
            if well.control_type == "rate":
                base_rate = well.control_value  # м3/сут
                delta = base_rate * 0.2  # изменение до ±20%
                well.control_value = float(base_rate + delta * a)
            elif well.control_type == "bhp":
                base_bhp = well.control_value  # МПа
                delta = 5.0  # ±5 МПа
                well.control_value = float(base_bhp + delta * a)
            # Прочие типы управления можно добавить позже

    def _calc_oil_mass(self) -> torch.Tensor:
        phi = self.reservoir.porosity
        s_o = self.fluid.s_o
        rho_o = self.fluid.rho_o
        cell_vol = self.reservoir.cell_volume
        return torch.sum(phi * s_o * rho_o * cell_vol)

    def _calc_reward(self) -> float:
        """Вознаграждение = добытая нефть (положительно) минус штраф за обводненность"""
        oil_mass = self._calc_oil_mass().item()
        produced_oil = self._prev_oil_mass - oil_mass
        self._prev_oil_mass = oil_mass
        # В дальнейшем можно добавить штраф за воду; пока только нефть
        return produced_oil / 1e6  # масштабирование для стабильности RL 