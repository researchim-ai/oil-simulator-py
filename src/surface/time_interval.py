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
    rounding = 4 * max(math.ulp(time_s), math.ulp(end_s), math.ulp(step))
    return remaining if 0 <= remaining - step <= rounding else step
