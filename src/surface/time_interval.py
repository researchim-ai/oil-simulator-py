"""Representable final steps for nested coarse/two-half integrations."""

import math


def interval_step_s(time_s, end_s, proposed_s, *, maximum_s=math.inf):
    """Integrate the entire remaining interval if only roundoff separates it.

    Snapping the *step before solving* includes the final representable time
    in the balances. An absolute epsilon would silently skip small intervals;
    use only a few ULPs of the actual clocks and proposed step instead.
    """
    if math.isnan(maximum_s) or maximum_s <= 0:
        raise ValueError("Positive maximum step required")
    remaining = end_s - time_s
    step = min(proposed_s, remaining, maximum_s)
    rounding = 8 * max(math.ulp(time_s), math.ulp(end_s), math.ulp(step))
    # Unit conversion of a maximum expressed in days can shift it by an
    # ULP. Allow only ULPs of that maximum, never a large-epoch clock ULP,
    # otherwise a genuinely oversized final step could be accepted.
    cap_roundoff = maximum_s if math.isinf(maximum_s) else maximum_s + 8 * math.ulp(maximum_s)
    if remaining <= cap_roundoff and 0 <= remaining - step <= rounding:
        return remaining
    actual = (time_s + step) - time_s
    if actual > maximum_s:
        actual = math.nextafter(time_s + step, time_s) - time_s
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
