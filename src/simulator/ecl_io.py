"""
Чтение бинарных файлов Eclipse/OPM (SMSPEC + UNSMRY) без внешних зависимостей.

Формат: последовательность Fortran-записей (big-endian). Каждый массив — заголовок
[8 символов имени, int32 число элементов, 4 символа типа] и блоки данных
(до 1000 чисел или 105 строк в блоке).
"""
import struct
from pathlib import Path

import numpy as np

_DTYPES = {"INTE": ">i4", "REAL": ">f4", "DOUB": ">f8", "LOGI": ">i4"}


def read_ecl_arrays(path):
    """Возвращает список (имя, numpy-массив или список строк) в порядке записи."""
    data = Path(path).read_bytes()
    pos, out = 0, []

    def record():
        nonlocal pos
        (n,) = struct.unpack(">i", data[pos:pos + 4])
        payload = data[pos + 4:pos + 4 + n]
        pos += n + 8
        return payload

    while pos < len(data):
        head = record()
        name = head[:8].decode("ascii").strip()
        (count,) = struct.unpack(">i", head[8:12])
        typ = head[12:16].decode("ascii")
        if typ == "MESS" or count == 0:
            out.append((name, []))
            continue
        if typ == "CHAR" or typ.startswith("C0"):
            width = 8 if typ == "CHAR" else int(typ[1:])
            items = []
            while len(items) < count:
                buf = record()
                items += [buf[i:i + width].decode("ascii", "replace").strip() for i in range(0, len(buf), width)]
            out.append((name, items))
        else:
            dt = np.dtype(_DTYPES[typ])
            chunks, got = [], 0
            while got < count:
                arr = np.frombuffer(record(), dtype=dt)
                chunks.append(arr)
                got += arr.size
            out.append((name, np.concatenate(chunks).astype(dt.newbyteorder("="))))
    return out


class EclSummary:
    """Сводка: ``s.get('WBHP', 'PROD')``, ``s.get('BPR', num=1)``, ``s.days``."""

    def __init__(self, case):
        case = str(case)
        base = case[:-7] if case.upper().endswith(".SMSPEC") else case
        spec = dict()
        for name, arr in read_ecl_arrays(base + ".SMSPEC"):
            spec.setdefault(name, arr)
        self.keywords = spec["KEYWORDS"]
        self.wgnames = spec.get("WGNAMES") or spec.get("NAMES") or [""] * len(self.keywords)
        self.nums = list(spec.get("NUMS", np.zeros(len(self.keywords), dtype=int)))
        self.units = spec.get("UNITS", [""] * len(self.keywords))
        dims = spec["DIMENS"]
        self.dims = (int(dims[1]), int(dims[2]), int(dims[3]))
        rows = [arr for name, arr in read_ecl_arrays(base + ".UNSMRY") if name == "PARAMS"]
        self.values = np.vstack(rows).astype(np.float64)
        self.days = self.get("TIME")

    def _index(self, kw, name=None, num=None):
        for i, k in enumerate(self.keywords):
            if k != kw:
                continue
            if name is not None and self.wgnames[i] != name:
                continue
            if num is not None and int(self.nums[i]) != int(num):
                continue
            return i
        raise KeyError(f"{kw} {name or ''} {num or ''} нет в сводке")

    def get(self, kw, name=None, num=None):
        return self.values[:, self._index(kw, name, num)]

    def unit(self, kw, name=None, num=None):
        return self.units[self._index(kw, name, num)]

    def cell_num(self, i, j, k):
        """Номер ячейки (1-based i, j, k), как в NUMS для BPR/BGSAT."""
        nx, ny, _ = self.dims
        return i + (j - 1) * nx + (k - 1) * nx * ny
