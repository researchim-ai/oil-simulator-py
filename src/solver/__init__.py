# solver package
try:
    from .normalizer import VariableScaler
except ImportError:  # fallback if class is named Normalizer
    from .normalizer import Normalizer as VariableScaler
try:
    from .normalizer import Normalizer  # optional explicit export
except Exception:
    pass 