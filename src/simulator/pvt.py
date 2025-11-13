import json
import pathlib
import numpy as np
from typing import Dict, Any
from .deck import load_pvt_from_deck


class PVTTable:
    """
    Загрузчик/интерполятор PVT-таблиц (Black-Oil).

    Поддерживаемые форматы:
      * JSON с полями pressure_MPa, Bo, Bw, Bg, mu_o_cP, mu_w_cP, mu_g_cP, Rs_m3m3, Rv_m3m3
      * Deck-файлы (Eclipse/CMG) с секциями PVTO/PVTW/PVDG.
      * Прямой словарь с эквивалентными полями.
    """

    def __init__(self, source: Any):
        if isinstance(source, dict):
            data = source
        else:
            path = pathlib.Path(source)
            if not path.exists():
                raise FileNotFoundError(f"PVT: файл '{source}' не найден")
            if path.suffix.lower() in {".json", ".jsn"}:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = load_pvt_from_deck(str(path))

        self.P = np.asarray(data["pressure_MPa"], dtype=np.float64)
        self.Bo = np.asarray(data.get("Bo", []), dtype=np.float64)
        self.Bw = np.asarray(data.get("Bw", []), dtype=np.float64)
        self.Bg = np.asarray(data.get("Bg", []), dtype=np.float64)
        self.mu_o = np.asarray(data.get("mu_o_cP", []), dtype=np.float64)
        self.mu_w = np.asarray(data.get("mu_w_cP", []), dtype=np.float64)
        self.mu_g = np.asarray(data.get("mu_g_cP", []), dtype=np.float64)
        self.Rs = np.asarray(data.get("Rs_m3m3", []), dtype=np.float64)
        rv_raw = data.get("Rv_m3m3")
        if rv_raw is None:
            self.Rv = np.zeros_like(self.Rs, dtype=np.float64)
        else:
            self.Rv = np.asarray(rv_raw, dtype=np.float64)

        self.extended = data.get("extended", {})
        oil_ext = self.extended.get("oil", {})
        self.oil_rs = np.asarray(oil_ext.get("rs_values", []), dtype=np.float64) if oil_ext else np.array([], dtype=np.float64)
        self.oil_tables: Dict[float, Dict[str, np.ndarray]] = {}
        for rs, table in (oil_ext.get("tables", {}) or {}).items():
            rs_val = float(rs)
            self.oil_tables[rs_val] = {
                "pressure_MPa": np.asarray(table["pressure_MPa"], dtype=np.float64),
                "Bo": np.asarray(table["Bo"], dtype=np.float64),
                "mu_o_cP": np.asarray(table["mu_o_cP"], dtype=np.float64),
                "Rv_m3m3": np.asarray(table.get("Rv_m3m3", np.zeros(len(table["pressure_MPa"]))), dtype=np.float64),
            }
        self.polymer_tables = self.extended.get("polymer", {})
        self.surfactant_tables = self.extended.get("surfactant", {})
        self.thermal_tables = self.extended.get("thermal", {})
        self.metadata = self.extended.get("meta", {})

        n = len(self.P)
        for arr, name in [
            (self.Bo, "Bo"), (self.Bw, "Bw"), (self.Bg, "Bg"),
            (self.mu_o, "mu_o_cP"), (self.mu_w, "mu_w_cP"), (self.mu_g, "mu_g_cP"),
            (self.Rs, "Rs_m3m3")
        ]:
            if len(arr) != n:
                raise ValueError(f"PVT: длины столбцов не совпадают (pressure vs {name})")
        if len(self.Rv) != n:
            raise ValueError("PVT: длины столбцов не совпадают (pressure vs Rv_m3m3)")

    def _interp(self, x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
        # Линейная интерполяция с экстраполяцией по краям
        return np.interp(x, xp, fp, left=fp[0], right=fp[-1])

    def eval(self, P_MPa: np.ndarray) -> dict:
        """Возвращает словарь массивов Bo,Bw,Bg, mu_o,mu_w,mu_g, Rs,Rv при заданном давлении (MPa)."""
        p = P_MPa.astype(np.float64)
        return {
            "Bo": self._interp(p, self.P, self.Bo),
            "Bw": self._interp(p, self.P, self.Bw),
            "Bg": self._interp(p, self.P, self.Bg),
            "mu_o": self._interp(p, self.P, self.mu_o),
            "mu_w": self._interp(p, self.P, self.mu_w),
            "mu_g": self._interp(p, self.P, self.mu_g),
            "Rs": self._interp(p, self.P, self.Rs),
            "Rv": self._interp(p, self.P, self.Rv),
        }

    def eval_oil(self, P_MPa: np.ndarray, Rs_m3m3: np.ndarray | None = None) -> Dict[str, np.ndarray]:
        """
        Возвращает свойства нефти с учётом растворённого газа (Rs).
        Если таблиц PVTO несколько (по Rs), выполняем би-линейную интерполяцию.
        """
        p = np.asarray(P_MPa, dtype=np.float64)
        if Rs_m3m3 is None or self.oil_rs.size <= 1 or not self.oil_tables:
            base = self.eval(p)
            return {
                "Bo": base["Bo"],
                "mu_o": base["mu_o"],
                "Rv": base["Rv"],
                "Rs": base["Rs"],
            }

        rs = np.asarray(Rs_m3m3, dtype=np.float64)
        flat_p = p.reshape(-1)
        flat_rs = rs.reshape(-1)

        rs_values = self.oil_rs
        result_bo = np.empty_like(flat_p)
        result_mu = np.empty_like(flat_p)
        result_rv = np.empty_like(flat_p)

        for idx, (pi, rsi) in enumerate(zip(flat_p, flat_rs)):
            # определяем интервал по Rs
            upper_idx = np.searchsorted(rs_values, rsi, side="right")
            lower_idx = max(0, upper_idx - 1)
            upper_idx = min(upper_idx, len(rs_values) - 1)
            lower_val = rs_values[lower_idx]
            upper_val = rs_values[upper_idx]

            table_lower = self.oil_tables.get(lower_val)
            table_upper = self.oil_tables.get(upper_val)

            def _interp_table(table: Dict[str, np.ndarray], key: str, pressure: float) -> float:
                return float(np.interp(
                    pressure,
                    table["pressure_MPa"],
                    table[key],
                    left=table[key][0],
                    right=table[key][-1],
                ))

            lo_bo = _interp_table(table_lower, "Bo", pi)
            lo_mu = _interp_table(table_lower, "mu_o_cP", pi)
            lo_rv = _interp_table(table_lower, "Rv_m3m3", pi)

            if upper_val == lower_val:
                result_bo[idx] = lo_bo
                result_mu[idx] = lo_mu
                result_rv[idx] = lo_rv
                continue

            hi_bo = _interp_table(table_upper, "Bo", pi)
            hi_mu = _interp_table(table_upper, "mu_o_cP", pi)
            hi_rv = _interp_table(table_upper, "Rv_m3m3", pi)

            weight = (rsi - lower_val) / (upper_val - lower_val)
            weight = np.clip(weight, 0.0, 1.0)

            result_bo[idx] = lo_bo + (hi_bo - lo_bo) * weight
            result_mu[idx] = lo_mu + (hi_mu - lo_mu) * weight
            result_rv[idx] = lo_rv + (hi_rv - lo_rv) * weight

        reshape = p.shape
        return {
            "Bo": result_bo.reshape(reshape),
            "mu_o": result_mu.reshape(reshape),
            "Rv": result_rv.reshape(reshape),
            "Rs": rs,
        }

    def get_polymer_viscosity_multiplier(self, concentration: float) -> float:
        table = self.polymer_tables.get("viscosity")
        if not table:
            return 1.0
        conc = np.asarray(table["concentration"], dtype=np.float64)
        mult = np.asarray(table["multiplier"], dtype=np.float64)
        return float(np.interp(concentration, conc, mult, left=mult[0], right=mult[-1]))

    def get_polymer_rock_multiplier(self, concentration: float) -> float:
        table = self.polymer_tables.get("rock")
        if not table:
            return 1.0
        conc = np.asarray(table["concentration"], dtype=np.float64)
        mult = np.asarray(table["multiplier"], dtype=np.float64)
        return float(np.interp(concentration, conc, mult, left=mult[0], right=mult[-1]))

    def get_polymer_adsorption(self, concentration: float) -> float:
        table = self.polymer_tables.get("adsorption")
        if not table:
            return 0.0
        conc = np.asarray(table["concentration"], dtype=np.float64)
        ads = np.asarray(table["adsorption"], dtype=np.float64)
        return float(np.interp(concentration, conc, ads, left=ads[0], right=ads[-1]))


