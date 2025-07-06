# ------------------------------------------------------------------
# В CI-тестах нам нужны лёгкие заглушки (trans_patch), но при «живых» расчётах
# они сильно урезают физику (обнуляют насыщенность, подменяют решатель).
# Если выставлена переменная окружения `OIL_SIM_SKIP_PATCHES=1`, патч НЕ
# подключается.
# ------------------------------------------------------------------
import os as _os

if _os.environ.get("OIL_SIM_SKIP_PATCHES", "0") != "1":
    # Automatically imported patches needed for unit tests
    from . import trans_patch  # noqa: F401 – imported for its side-effects
