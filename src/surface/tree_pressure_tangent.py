"""Compact exact tangent of a frozen directed production-tree solution.

Subtree component sums use Euler intervals; pressure recurrences use binary
ancestor composition. No node-by-source or node-by-control matrix is stored.
Only first derivatives of the already solved local pipe laws are represented.
"""

from __future__ import annotations

import copy

import torch

from .liquid import _name, _number


class TreePressureTangent:
    def __init__(self, spec, *, device="cpu"):
        required = {
            "model",
            "node_names",
            "parent_names",
            "source_names",
            "pressure_coefficients",
            "mass_coefficients",
            "valve_names",
            "valve_nodes",
            "valve_coefficients",
        }
        if set(spec) != required or spec["model"] != "blackoil_tree_pressure_tangent_v1":
            raise ValueError("Expected compact production-tree tangent schema")
        self.node_names = tuple(spec["node_names"])
        self.source_names = tuple(spec["source_names"])
        self.valve_names = tuple(spec["valve_names"])
        for registry in (self.node_names, self.source_names, self.valve_names):
            for name in registry:
                _name(name)
            if len(set(registry)) != len(registry):
                raise ValueError("Unique tangent registries required")
        n = len(self.node_names)
        if (
            not n
            or len(spec["parent_names"]) != n
            or len(spec["pressure_coefficients"]) != n
            or len(spec["mass_coefficients"]) != n
        ):
            raise ValueError("Exact node coefficient dimensions required")
        if len(spec["valve_nodes"]) != len(self.valve_names) or len(
            spec["valve_coefficients"]
        ) != len(self.valve_names):
            raise ValueError("Exact valve coefficient dimensions required")
        index = {name: i for i, name in enumerate(self.node_names)}
        if set(self.source_names) - index.keys() or set(spec["valve_nodes"]) - index.keys():
            raise ValueError("Unknown tangent source or valve node")
        roots = [i for i, p in enumerate(spec["parent_names"]) if p is None]
        if len(roots) != 1:
            raise ValueError("One pressure boundary required")
        root = roots[0]
        children = [[] for _ in range(n)]
        parents = []
        for i, name in enumerate(spec["parent_names"]):
            if name is None:
                parents.append(i)
            elif name not in index or index[name] == i:
                raise ValueError("Known distinct tree parent required")
            else:
                parents.append(index[name])
                children[index[name]].append(i)
        order, ends, seen, depths = [], {}, set(), {}
        stack = [(root, False, 0)]
        while stack:
            i, finished, depth = stack.pop()
            if finished:
                ends[i] = len(order)
                continue
            if i in seen:
                raise ValueError("Tangent topology contains a cycle")
            seen.add(i)
            order.append(i)
            depths[i] = depth
            stack.append((i, True, depth))
            stack.extend((child, False, depth + 1) for child in reversed(children[i]))
        if len(order) != n:
            raise ValueError("Disconnected tangent topology or cycle")
        a = [_number(v, "pressure coefficient") for v in spec["pressure_coefficients"]]
        b = []
        for row in spec["mass_coefficients"]:
            if len(row) != 3:
                raise ValueError("Three conserved stock-component coefficients required")
            b.append([_number(v, "mass coefficient") for v in row])
        c = [_number(v, "valve coefficient") for v in spec["valve_coefficients"]]
        if a[root] != 0 or any(b[root]) or self.node_names[root] in spec["valve_nodes"]:
            raise ValueError("Fixed receiving node has no local flow/valve law")
        position = {i: p for p, i in enumerate(order)}
        self.node_index = index
        self.device = torch.device(device)
        integer = lambda values: torch.tensor(values, dtype=torch.long, device=self.device)
        scalar = lambda values: torch.tensor(values, dtype=torch.float64, device=self.device)
        self._output_to_preorder = integer([position[i] for i in range(n)])
        self._preorder_to_output = integer(order)
        self._source = integer([position[index[name]] for name in self.source_names])
        self._valve = integer([position[index[name]] for name in spec["valve_nodes"]])
        self._b = scalar([b[i] for i in order])
        self.device = self._b.device
        self._c = scalar(c)
        # Disjoint balanced interval sums avoid subtracting large unrelated
        # prefixes: a small independent branch must survive beside a huge one.
        self._sums = []
        left, right, size = list(range(n)), [ends[i] for i in order], n
        while True:
            ln, lp, rn, rp = [], [], [], []
            for i in range(n):
                if left[i] < right[i] and left[i] % 2:
                    ln.append(i)
                    lp.append(left[i])
                    left[i] += 1
                if left[i] < right[i] and right[i] % 2:
                    right[i] -= 1
                    rn.append(i)
                    rp.append(right[i])
                left[i] //= 2
                right[i] //= 2
            self._sums.append((size, integer(ln), integer(lp), integer(rn), integer(rp)))
            if size == 1:
                break
            size = (size + 1) // 2
        self._stages = []
        ancestor = integer([position[parents[i]] for i in order])
        weight = scalar([a[i] for i in order])
        for _ in range(max(depths.values()).bit_length()):
            self._stages.append((ancestor, weight))
            weight, ancestor = weight * weight[ancestor], ancestor[ancestor]
        self._spec = copy.deepcopy(spec)

    def to_dict(self):
        return copy.deepcopy(self._spec)

    def _batch(self, value, prefix):
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.float64
            or value.device != self.device
        ):
            raise ValueError("Tangent inputs must be float64 tensors on its device")
        vector = value.ndim == len(prefix)
        if (
            value.ndim not in (len(prefix), len(prefix) + 1)
            or tuple(value.shape[: len(prefix)]) != prefix
        ):
            raise ValueError("Incorrect tangent input dimensions")
        return value.unsqueeze(-1) if vector else value, vector

    def apply(self, rates, *, receiving_pressure=None, valve_openings=None):
        rates, vector = self._batch(rates, (len(self.source_names), 3))
        n, k = len(self.node_names), rates.shape[-1]
        own = rates.new_zeros(n, 3, k).index_add(0, self._source, rates)
        subtree = torch.zeros_like(own)
        values = own
        for size, ln, lp, rn, rp in self._sums:
            subtree = subtree.index_add(0, ln, values[lp]).index_add(0, rn, values[rp])
            if size > 1:
                if size % 2:
                    values = torch.cat((values, values.new_zeros(1, 3, k)), dim=0)
                values = values.reshape(-1, 2, 3, k).sum(1)
        pressure = (self._b[:, :, None] * subtree).sum(1)
        if receiving_pressure is not None:
            boundary = torch.as_tensor(receiving_pressure, dtype=rates.dtype, device=self.device)
            if boundary.numel() not in (1, k):
                raise ValueError("Boundary tangent must match the RHS batch")
            pressure = torch.cat((boundary.reshape(-1).expand(k)[None, :], pressure[1:]), dim=0)
        if valve_openings is not None:
            valves, _ = self._batch(valve_openings, (len(self.valve_names),))
            if valves.shape[-1] != k:
                raise ValueError("Valve tangent must match the RHS batch")
            pressure = pressure.index_add(0, self._valve, self._c[:, None] * valves)
        for ancestor, weight in self._stages:
            pressure = pressure + weight[:, None] * pressure[ancestor]
        result = pressure[self._output_to_preorder]
        return result[:, 0] if vector else result

    def matvec(self, rates):
        return self.apply(rates)

    def own_source_coefficients(self):
        """Diagonal node/source component rows without a dense identity seed."""
        pressure = self._b
        for ancestor, weight in self._stages:
            pressure = pressure + weight[:, None] * pressure[ancestor]
        return pressure[self._output_to_preorder]

    def adjoint(self, pressure_weights):
        weights, vector = self._batch(pressure_weights, (len(self.node_names),))
        z = weights[self._preorder_to_output]
        for ancestor, weight in reversed(self._stages):
            z = z + torch.zeros_like(z).index_add(0, ancestor, weight[:, None] * z)
        mass = self._b[:, :, None] * z[:, None, :]
        parent = None
        for size, ln, lp, rn, rp in reversed(self._sums):
            gradient = mass.new_zeros(size, 3, mass.shape[-1])
            gradient = gradient.index_add(0, lp, mass[ln]).index_add(0, rp, mass[rn])
            if parent is not None:
                gradient = gradient + parent.repeat_interleave(2, dim=0)[:size]
            parent = gradient
        sources = parent[self._source]
        valves = self._c[:, None] * z[self._valve]
        boundary = z[0]
        return {
            "component_mass_rates": sources[:, :, 0] if vector else sources,
            "receiving_pressure": boundary[0] if vector else boundary,
            "valve_openings": valves[:, 0] if vector else valves,
        }

    def rmatvec(self, pressure_weights):
        return self.adjoint(pressure_weights)["component_mass_rates"]

    def jacobian_rows(
        self, names, *, maximum_rhs_entries=1_048_576, maximum_output_entries=1_048_576
    ):
        names = tuple(names)
        if set(names) - self.node_index.keys():
            raise ValueError("Unknown requested tangent row")
        n = len(self.node_names)
        if type(maximum_rhs_entries) is not int or maximum_rhs_entries < n:
            raise ValueError("RHS budget must fit one tree vector")
        size = len(names) * len(self.source_names) * 3
        if type(maximum_output_entries) is not int or maximum_output_entries < size:
            raise ValueError("Requested dense output exceeds explicit budget")
        if not names:
            return self._b.new_empty(0, len(self.source_names), 3)
        rows = []
        chunk = max(1, maximum_rhs_entries // n)
        for begin in range(0, len(names), chunk):
            selected = names[begin : begin + chunk]
            seed = self._b.new_zeros(n, len(selected))
            indices = torch.tensor([self.node_index[name] for name in selected], device=self.device)
            seed[indices, torch.arange(len(selected), device=self.device)] = 1
            rows.append(self.rmatvec(seed).permute(2, 0, 1))
        return torch.cat(rows)
