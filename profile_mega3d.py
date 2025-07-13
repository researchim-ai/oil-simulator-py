#!/usr/bin/env python
"""profile_mega3d.py — утилита для профилирования большого 3-D расчёта.

Запуск примера:
    python profile_mega3d.py \
        --config configs/large_3d.json \
        --backend geo \
        --smoother chebyshev \
        --output profile_mega3d.prof

После завершения можно проанализировать результаты командой, например:
    snakeviz profile_mega3d.prof
или
    python -m pstats profile_mega3d.prof
"""
from __future__ import annotations
import argparse
import cProfile
import pstats
import sys
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Помещаем корень проекта и папку src в sys.path, чтобы гарантировать корректные
# импорты вне зависимости от места запуска.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Импорт после настройки путей
from src import main as sim_main  # type: ignore  # noqa: E402


def _build_argv(args: argparse.Namespace) -> list[str]:
    """Сформировать sys.argv для симулятора из аргументов скрипта."""
    argv = ["sim"]  # фиктивное имя программы
    argv.extend(["--config", str(args.config)])
    if args.steps is not None:
        argv.extend(["--steps", str(args.steps)])
    if args.backend is not None:
        argv.extend(["--backend", args.backend])
    if args.smoother is not None:
        argv.extend(["--smoother", args.smoother])
    return argv


def profile_run(args: argparse.Namespace) -> None:
    """Запустить симуляцию под cProfile."""
    # Подготавливаем argv для внутреннего CLI симулятора
    sys.argv = _build_argv(args)

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        sim_main.main()
    finally:
        profiler.disable()
        profiler.dump_stats(str(args.output))
        print(
            f"\n\u2705 Профиль сохранён в {args.output}. "
            "Для анализа используйте snakeviz, pyprof2calltree или pstats."
        )
        # Также выводим топ-30 по времени сунмарно
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative").print_stats(30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Профилирование Mega-3D расчёта")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/large_3d.json"),
        help="Файл конфигурации симулятора",
    )
    parser.add_argument("--steps", type=int, default=None, help="Ограничить число шагов")
    parser.add_argument("--backend", type=str, default="geo", help="Backend CPR/AMG")
    parser.add_argument(
        "--smoother",
        type=str,
        default="chebyshev",
        help="Сглаживатель Geo-AMG: jacobi, l1gs, chebyshev",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile_mega3d.prof"),
        help="Файл для сохранения профиля cProfile",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args_ns = parse_args()
    profile_run(args_ns) 