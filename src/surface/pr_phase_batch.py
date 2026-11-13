"""Exact grouped phase EOS calls for a compiled transport Newton branch.

Only static topology/root selections are compiled on the host. Trial stocks,
P/T/composition and forward/reverse derivatives remain on the fluid device.
"""

import torch


class PhaseEOSGroups:
    """Shared exact root grouping for transport and column quadrature."""

    def __init__(self, fluid, tags):
        self.fluid = fluid
        groups = {}
        tags = tuple(tags)
        for position, tag in enumerate(tags):
            groups.setdefault(tag, []).append(position)
        self.groups = tuple(
            (tag, torch.tensor(indices, dtype=torch.long, device=fluid.device))
            for tag, indices in groups.items()
        )
        ordering = [i for indices in groups.values() for i in indices]
        inverse = [0] * len(tags)
        for i, original in enumerate(ordering):
            inverse[original] = i
        self.inverse = torch.tensor(inverse, dtype=torch.long, device=fluid.device)

    def properties(self, p, t, x):
        grouped = []
        for tag, indices in self.groups:

            def eos(pp, tt, xx, tag=tag):
                return self.fluid.tensor_homogeneous(pp, tt, xx, tag)

            if t.ndim == 0:
                # A column has ONE T. Preserve its shared thermodynamic
                # coefficients and the outer AD tangent, rather than making
                # redundant copies of attraction matrices for every point.
                q = torch.vmap(eos, in_dims=(0, None, 0))(p[indices], t, x[indices])
            else:
                q = torch.vmap(eos)(p[indices], t[indices], x[indices])
            grouped.append(q)
        return {
            key: torch.cat([q[key] for q in grouped]).index_select(0, self.inverse)
            for key in grouped[0]
        }


class PhaseEOSBatch(PhaseEOSGroups):
    def __init__(self, fluid, entries):
        self.entries = tuple(entries)
        self.slices = []
        position = 0
        for _offset, _active, tags in self.entries:
            self.slices.append(slice(position, position + len(tags)))
            position += len(tags)
        super().__init__(fluid, [tag for _offset, _active, tags in self.entries for tag in tags])

    def evaluate(self, y):
        fluid = self.fluid
        amounts, compositions, pressures, temperatures = [], [], [], []
        for offset, active, tags in self.entries:
            m, c = len(tags), len(active)
            stocks = y[offset : offset + m * c].exp().reshape(m, c)
            total = stocks.sum(-1)
            x = y.new_zeros((m, len(fluid.names))).index_copy(-1, active, stocks / total[:, None])
            amounts.append(total)
            compositions.append(x)
            pressures.append(y[offset + m * c].exp().expand(m))
            temperatures.append((y[offset + m * c + 1] * fluid.tref).expand(m))
        p, t, x = torch.cat(pressures), torch.cat(temperatures), torch.cat(compositions)
        properties = self.properties(p, t, x)
        return [
            (n, xx, {key: value[region] for key, value in properties.items()})
            for n, xx, region in zip(amounts, compositions, self.slices)
        ]
