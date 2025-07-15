# Namespace wrapper for src/solver package without explicit PYTHONPATH tweaks.

import os as _os
import pkgutil as _pkgutil

_root = _os.path.dirname(__file__)
_impl_path = _os.path.abspath(_os.path.join(_root, _os.pardir, "src", "solver"))

__path__ = _pkgutil.extend_path(__path__, __name__)
if _impl_path not in __path__:
    __path__.append(_impl_path) 