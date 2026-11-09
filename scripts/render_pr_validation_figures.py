"""Render scientific figures from saved measurements and scaling summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "benchmarks/results")
    p.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((args.results / "multiphase_experiments.json").read_text())
    scaling = json.loads((args.results / "pr_active_surface_scaling.json").read_text())
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5.3), layout="constrained")
    cases = experiment["roughness_sensitivity"]["0.0002"]["cases"]
    x = np.array([r["measured_wet_increment_pa"] for r in cases])
    y = np.array([r["predicted_wet_increment_pa"] for r in cases])
    line = np.linspace(0, 42, 100)
    ax.fill_between(
        line,
        np.maximum(0, line - 4),
        line + 4,
        color="#dddddd",
        label="Только приборная граница ±4 Па",
    )
    ax.plot(line, line, color="#888888", linestyle="--", linewidth=1, label="Совпадение 1:1")
    ax.scatter(x, y, color="#235b8e", s=30, label="Объёмное смачивание, k=0,2 мм")
    ax.set(
        xlabel="Измеренная прибавка к сухой трубе, Па",
        ylabel="Прогноз прибавки давления, Па",
        xlim=(0, 42),
        ylim=(0, 42),
        title="Плёночный противоток: замыкание не воспроизводит эксперимент",
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.supxlabel(
        "Источник: Hagedorn et al., DOI 10.25835/vcwbeqix; 20 стационарных окон.\nБез подгонки. Это проверка замыкания, не сквозной PR-расчёт воды.",
        fontsize=8,
    )
    fig.savefig(args.output / "multiphase_experimental_challenge.svg")
    fig.savefig("/tmp/multiphase_experimental_challenge.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
    rows = scaling["cases"]
    sizes = [r["cells"] for r in rows]
    axes[0].plot(
        sizes,
        [r["cpu"]["mean_step_seconds"] for r in rows],
        "o-",
        color="#333333",
        label="CPU: 1 поток",
    )
    axes[0].plot(
        sizes,
        [r["cuda"]["mean_step_seconds"] for r in rows],
        "s-",
        color="#235b8e",
        label="RTX 3090 Ti",
    )
    axes[0].set(
        xlabel="Число ячеек труб",
        ylabel="Среднее время связанного шага, с",
        xscale="log",
        title="Масштабирование настоящей связанной сети",
    )
    axes[0].set_xticks(sizes, labels=[str(n) for n in sizes])
    axes[0].legend()
    keys = [
        "pipes_and_apertures_inclusive",
        "column_inventory_recovery_inclusive",
        "column_thermal_newton_inclusive",
        "equipment_newton_inclusive",
        "whole_network_inspect_inclusive",
    ]
    labels = ["Трубы и патрубки", "Колонны: N/E", "Колонны: тепло", "Оборудование", "Проверка сети"]
    last = rows[-1]
    positions = np.arange(len(keys))
    height = 0.36
    for offset, (device, color, label) in enumerate(
        [("cpu", "#333333", "CPU: 1 поток"), ("cuda", "#235b8e", "RTX 3090 Ti")]
    ):
        values = [
            last[device]["timings_inclusive"][key]["seconds"] / len(last[device]["step_seconds"])
            for key in keys
        ]
        axes[1].barh(
            positions + (offset - 0.5) * height, values, height=height, color=color, label=label
        )
    axes[1].set_yticks(positions, labels=labels)
    axes[1].invert_yaxis()
    axes[1].set(
        xlabel="Время стадии на шаг, с", title=f"Распределение времени: {last['cells']} ячеек"
    )
    axes[1].legend()
    fig.supxlabel(
        "8 ветвей, 2 колонны со стенками, насос и ротор; float64. Среднее 2 шагов по 0,125 мс.\nСинхронизация CUDA на границах шага. Это короткий бенчмарк; вложенные стадии не суммируются.",
        fontsize=8,
    )
    fig.savefig(args.output / "pr_active_surface_scaling.svg")
    fig.savefig("/tmp/pr_active_surface_scaling.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
