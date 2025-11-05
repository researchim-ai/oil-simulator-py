import json
import numpy as np


class PVTTable:
    """
    Простой загрузчик/интерполятор PVT-таблиц (Black-Oil) из JSON.

    Ожидаемые поля JSON:
      pressure_MPa: [P]
      Bo, Bw, Bg: []
      mu_o_cP, mu_w_cP, mu_g_cP: []
      Rs_m3m3: []
      units: { pressure: MPa, ... }
    """

    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.P = np.asarray(data["pressure_MPa"], dtype=np.float64)
        self.Bo = np.asarray(data.get("Bo", []), dtype=np.float64)
        self.Bw = np.asarray(data.get("Bw", []), dtype=np.float64)
        self.Bg = np.asarray(data.get("Bg", []), dtype=np.float64)
        self.mu_o = np.asarray(data.get("mu_o_cP", []), dtype=np.float64)
        self.mu_w = np.asarray(data.get("mu_w_cP", []), dtype=np.float64)
        self.mu_g = np.asarray(data.get("mu_g_cP", []), dtype=np.float64)
        self.Rs = np.asarray(data.get("Rs_m3m3", []), dtype=np.float64)

        # Базовые проверки
        n = len(self.P)
        for arr, name in [
            (self.Bo, "Bo"), (self.Bw, "Bw"), (self.Bg, "Bg"),
            (self.mu_o, "mu_o_cP"), (self.mu_w, "mu_w_cP"), (self.mu_g, "mu_g_cP"),
            (self.Rs, "Rs_m3m3")
        ]:
            if len(arr) != n:
                raise ValueError(f"PVT: длины столбцов не совпадают (pressure vs {name})")

    def _interp(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        # Линейная интерполяция с экстраполяцией по краям
        return np.interp(x, xp, fp, left=fp[0], right=fp[-1])

    def eval(self, P_MPa: np.ndarray) -> dict:
        """Возвращает словарь массивов Bo,Bw,Bg, mu_o,mu_w,mu_g, Rs при заданном давлении (MPa)."""
        p = P_MPa.astype(np.float64)
        return {
            "Bo": self._interp(p, self.P, self.Bo),
            "Bw": self._interp(p, self.P, self.Bw),
            "Bg": self._interp(p, self.P, self.Bg),
            "mu_o": self._interp(p, self.P, self.mu_o),
            "mu_w": self._interp(p, self.P, self.mu_w),
            "mu_g": self._interp(p, self.P, self.mu_g),
            "Rs": self._interp(p, self.P, self.Rs),
        }


