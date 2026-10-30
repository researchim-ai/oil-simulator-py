"""Fixed-rate pressure reconstruction for the existing Eclipse VFP network.

This preserves the legacy surface-volume/ALQ convention. It is a table
model, distinct from the single-liquid Darcy-Weisbach network.
"""

from collections import deque

import torch


def evaluate_vfp_network(network, tables, own_rates, zero):
    """Evaluate the existing production tree without recursive depth truncation.

    own_rates[node] is an ordered sequence of (surface phase rates, lift gas
    rate or None). Children are summed before local wells, as in the original
    BlackOilWells implementation. Boundary pressures are absolute Pa.
    """
    branches, nodes = network["branches"], network["nodes"]
    names = dict.fromkeys([*nodes, *branches, *(b["up"] for b in branches.values())])
    children = {n: [] for n in names}
    indegree = dict.fromkeys(names, 0)
    for down, branch in branches.items():
        children[branch["up"]].append(down)
        indegree[down] += 1
    queue = deque(n for n in names if indegree[n] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(names):
        raise ValueError("Cyclic VFP network: a production tree is required")
    totals = {}
    for node in reversed(order):
        rate = zero.clone()
        for child in children[node]:
            rate = rate + totals[child]
        for well_rate, lift_rate in own_rates.get(node, ()):
            rate = rate + well_rate
            if lift_rate is not None:
                rate[2] = rate[2] + lift_rate
        totals[node] = rate
    pressures = {}
    for node in order:
        branch = branches.get(node)
        if branch is None:
            p = nodes.get(node, {}).get("p")
        else:
            upstream = pressures[branch["up"]]
            table = tables.get(branch["vfp"])
            if branch["vfp"] == 9999:
                p = upstream
            elif upstream is None or table is None:
                p = None
            else:
                rate = totals[node]
                with torch.no_grad():
                    p = float(table.bhp(rate[0:1], rate[1:2], rate[2:3], upstream, branch["alq"]))
        pressures[node] = p
    return pressures
