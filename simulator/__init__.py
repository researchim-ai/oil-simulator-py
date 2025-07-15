# Namespace wrapper for src/simulator to allow `import simulator.*` without
# explicit PYTHONPATH tweaks.  This file lives at repository root and
# complements the real package located at `src/simulator`.

import os as _os
import pkgutil as _pkgutil

# Extend this package's search path to include the implementation package
_root = _os.path.dirname(__file__)
# Реальный пакет находится на уровень выше: <repo>/src/simulator
_impl_path = _os.path.abspath(_os.path.join(_root, _os.pardir, "src", "simulator"))

__path__ = _pkgutil.extend_path(__path__, __name__)
if _impl_path not in __path__:
    __path__.append(_impl_path)

# Nothing else to do: submodules (reservoir, fluid, ...) will be discovered
# in `src/simulator` transparently. 