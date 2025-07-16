import os
from pathlib import Path

import numpy as np

try:
    import h5py  # type: ignore
    _H5PY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _H5PY_AVAILABLE = False


def save_to_hdf5(reservoir, fluid, filename: str = "snapshot.h5"):
    """Сохраняет текущие состояния давления и насыщенностей в HDF5.

    Пишем минимальный объём данных: давление (Па), Sw, So, Sg (если есть).
    Если h5py не установлен, выводим предупреждение и ничего не делаем,
    чтобы не срывать основной расчёт.

    Parameters
    ----------
    reservoir : Reservoir
        Экземпляр пласта – используем размеры сетки, проницаемости и пр.
    fluid : Fluid
        Экземпляр флюидов с полями pressure, s_w, s_o, s_g (опц.).
    filename : str, optional
        Путь к файлу (будет перезаписан).  Если указан только базовый
        файл без каталога, помещаем его в подкаталог ``results``.
    """
    if not _H5PY_AVAILABLE:
        print("[WARN] h5py не найден – HDF5-снапшоты пропущены.")
        return

    # Создаём каталоги, если нужно
    filepath = Path(filename)
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)

    # Снимаем данные на CPU и в NumPy
    p = fluid.pressure.detach().cpu().numpy()
    sw = fluid.s_w.detach().cpu().numpy()
    so = fluid.s_o.detach().cpu().numpy()

    has_g = hasattr(fluid, "s_g")
    sg = fluid.s_g.detach().cpu().numpy() if has_g else None

    with h5py.File(filepath.as_posix(), "w") as f:
        f.create_dataset("pressure_pa", data=p, compression="gzip", compression_opts=1)
        f.create_dataset("s_w", data=sw, compression="gzip", compression_opts=1)
        f.create_dataset("s_o", data=so, compression="gzip", compression_opts=1)
        if has_g:
            f.create_dataset("s_g", data=sg, compression="gzip", compression_opts=1)

    print(f"Снапшот сохранён в {filepath}") 