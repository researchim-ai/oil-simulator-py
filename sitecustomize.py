"""Bootstrap sys.path при запуске из произвольных подпапок.

Добавляет корень репозитория и ./src в PYTHONPATH.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_root = _os.path.dirname(__file__)
_src = _os.path.join(_root, "src")

for _p in (_root, _src):
    if _p not in _sys.path:
        _sys.path.insert(0, _p) 