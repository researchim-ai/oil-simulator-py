"""Representable final steps for nested coarse/two-half integrations."""

import math


def interval_step_s(time_s, end_s, proposed_s):
    """Integrate the entire remaining interval if only roundoff separates it.

    Snapping the *step before solving* includes the final representable time
    in the balances. An absolute epsilon would silently skip small intervals;
    use only a few ULPs of the actual clocks and proposed step instead.
    """
    remaining = end_s - time_s
    step = min(proposed_s, remaining)
    rounding = 8 * max(math.ulp(time_s), math.ulp(end_s), math.ulp(step))
    if 0 <= remaining - step <= rounding:
        return remaining
    actual = (time_s + step) - time_s
    if actual <= 0:
        raise ValueError("Substep below time resolution")
    return min(actual, remaining)


def split_step_s(time_s, step_s):
    """Two positive clock-representable parts, including an odd final ULP.

    Equal floating durations need not produce equal advances at a large epoch.
    Their sum must cover exactly the same interval as the coarse solve.
    """
    end = time_s + step_s
    middle = time_s + step_s / 2
    first, second = middle - time_s, end - middle
    if first <= 0 or second <= 0:
        raise ValueError("Nested substep below time resolution")
    return first, second
