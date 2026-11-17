"""Host-process telemetry; no CUDA synchronization or allocator side effects."""

import math
import os
import sys
from pathlib import Path


def process_memory():
    resident = swapped = peak = None
    if sys.platform.startswith("linux"):
        try:
            resident = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
                "SC_PAGE_SIZE"
            )
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmSwap:"):
                    swapped = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            pass
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak *= 1024
    except (ImportError, OSError, ValueError):
        pass
    return {
        "process_rss_bytes": resident,
        "process_swap_bytes": swapped,
        "process_lifetime_peak_rss_bytes": peak,
    }


def enforce_host_memory(limit_mib, memory):
    """Reactive guard at startup/accepted stages, not a hard allocation limiter."""
    if limit_mib is None:
        return
    if isinstance(limit_mib, bool) or not math.isfinite(limit_mib) or limit_mib <= 0:
        raise ValueError("Positive finite host memory limit required")
    rss, swap = memory["process_rss_bytes"], memory["process_swap_bytes"]
    if rss is None or swap is None:
        raise ValueError("Host memory guard requires resident and swap telemetry")
    if rss + swap > limit_mib * 1024**2:
        raise MemoryError(
            f"Host memory budget exceeded: RSS+swap={(rss + swap) / 1024**2:.1f} MiB > {limit_mib:g} MiB"
        )
