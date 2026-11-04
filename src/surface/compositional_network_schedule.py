"""Source-bound coupled PR schedules; restart physically replays the prefix."""

import copy
import hashlib
from pathlib import Path

from .component_energy import ComponentEnergyStream
from .compositional_network import CompositionalNetworkState
from .compositional_separator_schedule import SOURCES as SEPARATOR_SOURCES
from .liquid import ConservationError, _number
from .operation import _digest
from .torch_backend import runtime_fingerprint

SOURCES = tuple(
    dict.fromkeys(
        ("compositional_network.py", "compositional_network_schedule.py") + SEPARATOR_SOURCES
    )
)


def run_compositional_network_schedule(system, initial, periods, checkpoint=None, stop_after=None):
    system.inspect(initial)
    root = Path(__file__).resolve().parent
    hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in SOURCES}
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Nonempty network periods required")
    periods = copy.deepcopy(periods)
    commands = []
    for period in periods:
        if (
            not isinstance(period, dict)
            or "duration_s" not in period
            or set(period)
            - {
                "duration_s",
                "feeds",
                "heat_w",
                "utility_temperature_k",
                "outlet_openings",
                "connection_openings",
                "max_substeps",
            }
        ):
            raise ValueError("Unknown or missing network period field")
        _number(period["duration_s"], "duration_s", positive=True)
        budget = period.get("max_substeps", 10000)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("Invalid network work budget")
        raw_feeds = period.get("feeds", {})
        if not isinstance(raw_feeds, dict) or set(raw_feeds) - set(system.names):
            raise ValueError("Unknown feed vessel")
        streams, canonical = {}, {}
        for name, specs in raw_feeds.items():
            if not isinstance(specs, (list, tuple)):
                raise TypeError("Feed vessel needs a list of streams")
            streams[name], canonical[name] = [], []
            v = system.vessels[system.names.index(name)]
            for spec in specs:
                if isinstance(spec, ComponentEnergyStream):
                    stream = spec
                    raw = {"stream": stream.to_dict()}
                elif isinstance(spec, dict) and set(spec) == {"stream"}:
                    stream = ComponentEnergyStream.from_dict(spec["stream"])
                    raw = copy.deepcopy(spec)
                elif isinstance(spec, dict) and set(spec) == {"from_pt"}:
                    args = dict(spec["from_pt"])
                    args.setdefault("gravity_m_s2", v.gravity_m_s2)
                    args.setdefault("elevation_m", v.elevation_m)
                    stream = ComponentEnergyStream.from_pt(system.fluid, **args)
                    raw = copy.deepcopy(spec)
                else:
                    raise ValueError("Feed needs exactly stream or from_pt")
                streams[name].append(stream)
                canonical[name].append(raw)
        period["feeds"] = canonical
        command = {**period, "feeds": streams}
        # Validate all future command names and thermodynamic boundaries before
        # any publication. Actual feed pressure is checked at its real time.
        system.prepare(
            initial,
            **{k: v for k, v in command.items() if k not in {"duration_s", "max_substeps"}},
            _check_feed_pressure=False,
        )
        commands.append(command)
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0
    ):
        raise ValueError("stop_after must be a nonnegative integer")
    fingerprint = _digest(
        {
            "system": system.to_dict(),
            "initial": initial.to_dict(),
            "periods": periods,
            "device": str(system.fluid.device),
            "runtime": runtime_fingerprint(),
            "sources": hashes,
        }
    )
    saved = {
        "fingerprint": fingerprint,
        "next_period": 0,
        "state": initial.to_dict(),
        "records": [],
    }
    current = initial
    if checkpoint is not None:
        candidate = copy.deepcopy(checkpoint)
        if set(candidate) != set(saved) | {"checksum"}:
            raise ValueError("Invalid network checkpoint fields")
        checksum = candidate.pop("checksum")
        if checksum != _digest(candidate) or candidate["fingerprint"] != fingerprint:
            raise ValueError("Network checkpoint checksum/fingerprint mismatch")
        index = candidate["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(candidate["records"]) != index
        ):
            raise ValueError("Invalid network checkpoint index")
        for i, record in enumerate(candidate["records"]):
            replay = {
                "period": i,
                "initial_state": current.to_dict(),
                **system.advance(current, **commands[i]),
            }
            if _digest(replay) != _digest(record):
                raise ConservationError("Network checkpoint physical replay mismatch")
            current = CompositionalNetworkState.from_dict(replay["state"])
        if current.to_dict() != candidate["state"]:
            raise ConservationError("Network checkpoint final state mismatch")
        saved = candidate
    start = saved["next_period"]
    end = len(periods) if stop_after is None else min(len(periods), start + stop_after)
    for i in range(start, end):
        record = {
            "period": i,
            "initial_state": current.to_dict(),
            **system.advance(current, **commands[i]),
        }
        saved["records"].append(record)
        saved["next_period"] = i + 1
        saved["state"] = copy.deepcopy(record["state"])
        current = CompositionalNetworkState.from_dict(record["state"])
    if hashes != {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in SOURCES}:
        raise ConservationError("Sources changed during network calculation")
    return {**saved, "checksum": _digest(saved)}
