"""sitecustomize автоматически импортируется Python-ом при старте.

Гарантирует, что корень репозитория *и* каталог ``src`` попадают в
``sys.path`` независимо от текущего каталога запуска, чтобы импорты вида
``import simulator``/``solver``/``linear_gpu`` работали корректно.
"""
from __future__ import annotations

import os as _os
import sys as _sys

# Абсолютный путь к корню репозитория (родитель каталога tests)
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
_src = _os.path.join(_root, "src")

for _p in (_root, _src):
    if _p not in _sys.path:
        _sys.path.insert(0, _p) 