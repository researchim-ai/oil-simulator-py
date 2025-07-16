import math
import pytest

from src.utils import PIDController


def test_anti_windup():
    """Интеграл не должен выходить за пределы integral_limit."""
    ctrl = PIDController(ki=1.0, integral_limit=5.0)
    for _ in range(100):
        ctrl.update(10.0)  # большой постоянный error
    assert abs(ctrl.integral) <= 5.0 + 1e-9, "Anti-windup ограничение нарушено"


def test_derivative_filter():
    """Проверяем, что low-pass фильтр производной работает (alpha < 1)."""
    alpha = 0.2
    ctrl = PIDController(kd=1.0, derivative_alpha=alpha)
    ctrl.update(0.0)  # первый вызов, derivative=0
    ctrl.update(10.0)  # скачок ошибки
    expected_derivative = alpha * 10.0 + (1 - alpha) * 0.0  # 2.0
    assert math.isclose(ctrl.last_derivative, expected_derivative, rel_tol=1e-6)


def test_max_scale_change():
    """Scale не должен изменяться больше чем на max_scale_change раз за шаг."""
    ctrl = PIDController(kp=1.0, ki=0.0, kd=0.0, max_scale_change=1.2, scale_max=100.0)
    scale1 = ctrl.update(10.0)  # без лимита было бы огромным
    assert scale1 <= 1.2 + 1e-9
    # второй вызов с тем же error – scale должно быть ровно 1.2, т.к. prev_scale=scale1
    scale2 = ctrl.update(10.0)
    assert scale2 <= 1.2 + 1e-9 