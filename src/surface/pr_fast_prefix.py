"""Reuse qualified local integration prefixes inside one public transaction.

Only complete accepted coarse/two-half pairs are reused. No interpolation,
rounded thermodynamic keys, extrapolated physical state or skipped error test.
The endpoint-clipped last pair is excluded unless the endpoint matches exactly.
"""

import math
from dataclasses import replace

from .equilibrium_cache import _MODEL_TENSORS, FlashCache, _copy, _key, _signature
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
from .operation import _digest
from .pr_equilibrium_transport import PREquilibriumTransportState
from .time_interval import split_step_s

_MAX_ENTRIES = 4
_MAX_RECORDS = 1024


def advance_fast(model, initial, duration, commands):
    cache = getattr(model.fluid, "_active_flash_cache", None)
    if not isinstance(cache, FlashCache):
        return model.advance(initial, duration, **commands)
    model.inspect(initial)
    try:
        key = (
            model.model_id,
            _signature(model.fluid),
            _digest(initial.to_dict()),
            _key(commands),
            model._analytic_single_phase_characteristics,
            model._batch_characteristics,
            model._batch_stability,
        )
    except (TypeError, RuntimeError):
        return model.advance(initial, duration, **commands)
    end = initial.time_s + duration
    saved = cache.transport_prefixes.get(key)
    report = None
    if saved is not None and math.isfinite(end) and end > initial.time_s:
        cache.transport_prefixes.move_to_end(key)
        old = saved["report"]
        steps = old["steps"]
        if end == old["state"]["time_s"]:
            report = _copy(old)
            report["reused_macrosteps"] = len(steps) // 2
            report["executed_attempts"] = 0
            report["attempts"] = len(steps) // 2
        else:
            # The last old pair may have been truncated by its own reporting
            # boundary. Earlier pairs follow the same adaptive path regardless
            # of the requested end. An individual half is NOT independently
            # temporally qualified and can never be reused on its own.
            for count in range(len(steps) - 2, 0, -2):
                raw = steps[count - 1]["state"]
                clock = raw["time_s"]
                if clock > end:
                    continue
                try:
                    if clock < end:
                        split_step_s(clock, end - clock)
                except ValueError:
                    continue
                # The next pair stores the proposal selected after THIS pair;
                # the raw endpoint itself still carries its incoming proposal.
                proposal = steps[count]["state"]["next_step_s"]
                current = replace(PREquilibriumTransportState.from_dict(raw), next_step_s=proposal)
                try:
                    if clock < end:
                        suffix = model.advance(current, end - clock, **commands)
                        suffix_steps, attempts = suffix["steps"], suffix["attempts"]
                        final = PREquilibriumTransportState.from_dict(suffix["state"])
                    else:
                        suffix_steps, attempts, final = [], 0, current
                    if final.time_s != end:
                        raise ConservationError("Cached prefix does not cover the exact interval")
                    physical = [*_copy(steps[:count]), *suffix_steps]
                    heat = math.fsum(s["ambient_heat_j"] + s["external_heat_j"] for s in physical)
                    work = math.fsum(s["external_work_j"] for s in physical)
                    residual = model._ledger(initial, final, heat + work)
                    report = {
                        **{k: _copy(old[k]) for k in ("openings", "loss_coefficients")},
                        "state": final.to_dict(),
                        "steps": physical,
                        "attempts": count // 2 + attempts,
                        "executed_attempts": attempts,
                        "reused_macrosteps": count // 2,
                        "conservation_residuals": residual,
                        "entropy_production_j_k": math.fsum(
                            s["entropy_production_j_k"] for s in physical
                        ),
                        "total_heat_j": heat,
                        "total_work_j": work,
                    }
                except (ConservationError, PropertyModelError, HydraulicConvergenceError):
                    # Reuse is optional; an original solve still has to satisfy
                    # all gates. Never hide a physical failure by loosening it.
                    cache.transport_prefixes.pop(key, None)
                break
    if report is None:
        report = model.advance(initial, duration, **commands)
        report["executed_attempts"] = report["attempts"]
        report["reused_macrosteps"] = 0
    else:
        cache.transport_prefix_hits += 1
        cache.transport_prefix_steps += report["reused_macrosteps"]
    # Keep the longest trajectory for this exact initial state and commands,
    # with bounded memory. Pin parameters so replacement cannot reuse tensor
    # identities while an old numerical signature is still in this cache.
    if len(report["steps"]) <= _MAX_RECORDS and (
        saved is None or report["state"]["time_s"] > saved["report"]["state"]["time_s"]
    ):
        cache.transport_prefixes[key] = {
            "report": _copy(report),
            "parameters": tuple(getattr(model.fluid, name) for name in _MODEL_TENSORS),
        }
        cache.transport_prefixes.move_to_end(key)
        while len(cache.transport_prefixes) > _MAX_ENTRIES:
            cache.transport_prefixes.popitem(last=False)
    return report
