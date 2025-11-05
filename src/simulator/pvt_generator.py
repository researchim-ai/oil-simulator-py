import argparse
import os
import json
import math
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PVTParams:
    p_min_mpa: float = 5.0
    p_max_mpa: float = 60.0
    n_points: int = 50
    temperature_c: float = 90.0
    oil_api: float = 35.0
    gas_sg: float = 0.7  # относительная плотность газа (воздух=1)
    salinity_gpl: float = 100.0
    bubble_point_mpa: float = 20.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def generate_black_oil_pvt(params: PVTParams) -> Dict:
    P = _linspace(params.p_min_mpa, params.p_max_mpa, params.n_points)

    # Оценка максимального Rs (м3/м3) — синтетически, на основе API/SG
    rs_max = 120.0 + 2.0 * (params.oil_api - 30.0) + 50.0 * (params.gas_sg - 0.7)
    rs_max = _clip(rs_max, 50.0, 250.0)

    # Вспомогательные параметры для сжимаемостей (на MPa^-1)
    c_o = 3.0e-4
    c_w = 4.0e-4

    Bo = []
    Bw = []
    Bg = []
    mu_o = []
    mu_w = []
    mu_g = []
    Rs = []

    # Базовые значения вязкостей при p ~ p_min
    mu_od = 2.5  # dead oil, cP (синтетика)
    mu_w_base = 0.5
    mu_g_base = 0.02

    # Факторы по давлению
    for p in P:
        # Rs: растет до Pb, дальше плато
        if p <= params.bubble_point_mpa:
            rs = rs_max * (p / params.bubble_point_mpa) ** 0.85
        else:
            rs = rs_max
        Rs.append(rs)

        # Bo: до Pb падает от ~1.2 к ~Bo(Pb), выше Pb — слабая компрессия
        if p <= params.bubble_point_mpa:
            bo_pb = 1.10  # типичное значение у Pb
            bo = 1.20 - 0.10 * (p / params.bubble_point_mpa) ** 0.6
            bo = max(bo, bo_pb)
        else:
            delta = p - params.bubble_point_mpa
            bo = 1.10 * math.exp(-c_o * delta)
        Bo.append(bo)

        # Bw: слабая компрессия с давлением
        bw = 1.02 * math.exp(-c_w * (p - params.p_min_mpa))
        Bw.append(bw)

        # Bg: ~ 0.1 / P (MPa) — уменьшается с ростом давления
        bg = 0.1 / max(p, 1e-6)
        bg = _clip(bg, 0.001, 0.05)
        Bg.append(bg)

        # μo: ниже Pb понижается с ростом Rs, выше Pb — слегка растет с P
        if p <= params.bubble_point_mpa:
            red = 0.5 * (rs / rs_max)  # 0..0.5
            muo = max(0.5, mu_od * (0.8 - red))
        else:
            mu_pb = max(0.5, mu_od * (0.8 - 0.5))  # на Pb
            muo = mu_pb * (1.0 + 0.01 * (p - params.bubble_point_mpa) / 10.0)
        mu_o.append(muo)

        # μw: слегка растет с давлением
        muw = mu_w_base * (1.0 + 0.02 * (p - params.p_min_mpa) / max(params.p_max_mpa - params.p_min_mpa, 1e-6))
        mu_w.append(muw)

        # μg: растет с давлением (синтетически)
        mug = mu_g_base * (1.0 + 0.5 * (p / params.p_max_mpa))
        mu_g.append(mug)

    return {
        "units": {
            "pressure": "MPa",
            "Bo,Bw,Bg": "dimensionless",
            "mu": "cP",
            "Rs": "m3/m3"
        },
        "pressure_MPa": P,
        "Bo": Bo,
        "Bw": Bw,
        "Bg": Bg,
        "mu_o_cP": mu_o,
        "mu_w_cP": mu_w,
        "mu_g_cP": mu_g,
        "Rs_m3m3": Rs,
        "meta": {
            "temperature_C": params.temperature_c,
            "oil_API": params.oil_api,
            "gas_SG": params.gas_sg,
            "salinity_gpl": params.salinity_gpl,
            "bubble_point_MPa": params.bubble_point_mpa
        }
    }


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic Black-Oil PVT tables (JSON)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--p-min", type=float, default=5.0)
    ap.add_argument("--p-max", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--temp", type=float, default=90.0, help="Reservoir temperature, C")
    ap.add_argument("--api", type=float, default=35.0)
    ap.add_argument("--gas-sg", type=float, default=0.7)
    ap.add_argument("--salinity", type=float, default=100.0, help="g/L")
    ap.add_argument("--pb", type=float, default=20.0, help="Bubble point, MPa")
    args = ap.parse_args()

    params = PVTParams(
        p_min_mpa=args.p_min,
        p_max_mpa=args.p_max,
        n_points=args.n,
        temperature_c=args.temp,
        oil_api=args.api,
        gas_sg=args.gas_sg,
        salinity_gpl=args.salinity,
        bubble_point_mpa=args.pb,
    )

    data = generate_black_oil_pvt(params)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"PVT JSON saved to {args.out}")


if __name__ == "__main__":
    main()


