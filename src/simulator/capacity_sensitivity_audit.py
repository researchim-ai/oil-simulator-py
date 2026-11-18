"""Offline accounting for proposal-only sensitivities; no Torch dependency."""

import math


def audit_capacity_sensitivity(coupling):
    if coupling.get('search_method') != 'sensitivity_safeguarded':
        return 0, 0
    work = coupling.get('sensitivity_work')
    if not isinstance(work, dict):
        raise TypeError('Sensitivity work report required')
    for key in ('linear_solves', 'plant_evaluations', 'proposals'):
        if type(work.get(key)) is not int or work[key] < 0:
            raise ValueError('Invalid sensitivity work counter')
    linear = plant = proposals = 0
    trace = coupling['trace']
    for index, row in enumerate(trace):
        sensitivity = row.get('capacity_sensitivity')
        if sensitivity is not None:
            if 'certified_from_trace_index' in row:
                raise ValueError('Sensitivity requires a directly solved source')
            nl, np = sensitivity.get('linear_solves'), sensitivity.get('plant_evaluations')
            slopes = sensitivity.get('derivatives')
            if type(nl) is not int or nl not in (0, 1) or type(np) is not int or not 0 <= np <= 4*nl:
                raise ValueError('Invalid sensitivity evaluation accounting')
            if not isinstance(slopes, dict):
                raise TypeError('Sensitivity derivatives must be a facet mapping')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                   for v in slopes.values()):
                raise ValueError('Invalid increasing facet derivative')
            if slopes and (nl != 1 or np != 4 or sensitivity.get('reason') is not None):
                raise ValueError('Unchecked plant derivative was used')
            linear += nl
            plant += np
        proposal = row.get('capacity_proposal')
        if proposal is not None:
            low, high = proposal['low_fraction'], proposal['high_fraction']
            sources = [p for p in trace[:index] if p['fraction'] in (low, high)]
            if (not low < row['fraction'] < high
                    or not any(p['fraction'] == low and p['feasible'] for p in sources)
                    or not any(p['fraction'] == high and not p['feasible'] for p in sources)
                    or not any((p.get('capacity_sensitivity') or {}).get('derivatives') for p in sources)):
                raise ValueError('Sensitivity proposal lacks solved bracket/source endpoints')
            proposals += 1
    if (linear, plant, proposals) != (work['linear_solves'], work['plant_evaluations'], work['proposals']):
        raise ValueError('Sensitivity counters differ from actual trace work')
    return linear, plant
