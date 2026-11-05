"""Zero-wall-capacity thermal contacts between finite PR control volumes.

UA is explicit input, not an inferred heat-exchanger rating. Both sides retain
their own inventory, pressure and composition. No material crosses the wall.
"""

from dataclasses import dataclass

from .liquid import _name, _number


@dataclass(frozen=True)
class PRHeatLink:
    name: str
    source: str
    target: str
    conductance_w_k: float

    def __post_init__(self):
        for value in (self.name, self.source, self.target):
            _name(value)
        if self.source == self.target:
            raise ValueError("Thermal contact needs distinct finite bodies")
        _number(self.conductance_w_k, "thermal conductance", positive=True)


def thermal_exchange(links, temperatures, openings):
    """Device-only differentiable heat rates; positive rate is source to target."""
    rates = {}
    records = {}
    for link in links:
        ta, tb = temperatures[link.source], temperatures[link.target]
        ua = link.conductance_w_k if openings[link.name] else 0.0
        difference = ta - tb
        rate = ua * difference
        rates[link.source] = rates.get(link.source, 0) - rate
        rates[link.target] = rates.get(link.target, 0) + rate
        records[link.name] = {
            "heat_rate_w": rate,
            "entropy_production_w_k": ua * difference.square() / (ta * tb),
        }
    return rates, records


def thermal_records(records):
    return {
        name: {key: float(value.detach().cpu()) for key, value in row.items()}
        for name, row in records.items()
    }
