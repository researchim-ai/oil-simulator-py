"""Exact inactive-set reuse inside one old-state/dt/controls capacity search.

Only an already solved INFEASIBLE state is reused. Every well row must remain
independent of its rate cap. The deliberately conservative domain leaves a
factor-two gap to each switching threshold. No extrapolated flows are used.
"""

import math

DAY = 86400.


def rate_cap_switch_value(rate, bhp, pressure_target):
    return rate + max(abs(rate), 0.01 / DAY) / max(pressure_target, 1e5) * (bhp - pressure_target)


def minimum_invariant_fraction(proof):
    if proof.get('version') != 1:
        raise ValueError('Unsupported inactive-cap certificate')
    source = proof['source_fraction']
    if isinstance(source, bool) or not math.isfinite(source) or not 0 < source <= 1:
        raise ValueError('Invalid inactive-cap source fraction')
    wells = proof['wells']
    if not isinstance(wells, dict) or not wells:
        raise ValueError('Inactive-cap certificate requires every well')
    threshold = 0.
    for row in wells.values():
        if row['mode'] not in ('surface', 'bhp_limit'):
            raise ValueError('A rate-controlled row cannot be reused')
        values = [row[k] for k in ('rate_m3_s', 'cap_m3_s', 'bhp_pa', 'network_pressure_pa', 'floor_pa')]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise ValueError('Nonfinite inactive-cap certificate')
        rate, cap, bhp, network, floor = values
        if min(rate, cap, bhp, network, floor) <= 0:
            raise ValueError('Positive producing state required')
        target = max(network, floor)
        expected = 'surface' if network >= floor else 'bhp_limit'
        shadow = rate_cap_switch_value(rate, bhp, target)
        if row['mode'] != expected or shadow >= cap:
            raise ValueError('Inactive-cap source has an inconsistent active set')
        threshold = max(threshold, max(0., shadow) / cap * source)
    return max(2 * threshold, 32 * math.ulp(source))


def make_inactive_cap_certificate(report, targets, base_caps, floors, fraction):
    wells = report['wells']
    if set(wells) != set(base_caps) or set(wells) != set(targets):
        return None
    rows = {}
    for name, row in wells.items():
        control, cap = targets[name]
        expected = base_caps[name] * fraction / DAY
        if control != 'rate' or cap != expected:
            return None
        rows[name] = {
            'mode': row['mode'], 'rate_m3_s': row['standard_control_rate_m3_s'],
            'cap_m3_s': cap, 'bhp_pa': row['bhp_pa'],
            'network_pressure_pa': row['network_reference_pressure_pa'], 'floor_pa': floors[name],
        }
    proof = {'version': 1, 'source_fraction': fraction, 'wells': rows}
    try:
        minimum = minimum_invariant_fraction(proof)
    except ValueError:
        return None
    if minimum >= fraction:
        return None
    proof['minimum_fraction'] = minimum
    return proof


def certificate_covers(proof, fraction):
    return minimum_invariant_fraction(proof) <= fraction <= proof['source_fraction']


def audit_inactive_cap_trace(coupling, config_wells, enabled):
    """Validate same-trial proof references and count actual reservoir solves."""
    trace = coupling['trace']
    expected = {w['name']: w for w in config_wells}
    reused = 0
    for index, row in enumerate(trace):
        proof = row.get('inactive_cap_certificate')
        if proof is not None:
            if not enabled or row['feasible'] or 'certified_from_trace_index' in row:
                raise ValueError('Invalid inactive-cap source trial')
            if proof['source_fraction'] != row['fraction'] or set(proof['wells']) != set(expected):
                raise ValueError('Inactive-cap source registry/fraction mismatch')
            minimum = minimum_invariant_fraction(proof)
            if proof['minimum_fraction'] != minimum:
                raise ValueError('Inactive-cap lower bound mismatch')
            for name, bounds in proof['wells'].items():
                well = expected[name]
                cap = well['control_value'] * row['fraction'] / DAY
                floor = (well.get('bhp_min') if well.get('bhp_min') is not None else .101325) * 1e6
                if bounds['cap_m3_s'] != cap or bounds['floor_pa'] != floor:
                    raise ValueError('Inactive-cap source controls mismatch')
        if 'certified_from_trace_index' not in row:
            continue
        source_index = row['certified_from_trace_index']
        if not enabled or type(source_index) is not int or not 0 <= source_index < index or row['feasible']:
            raise ValueError('Invalid inactive-cap reuse reference')
        source = trace[source_index]
        proof = source.get('inactive_cap_certificate')
        if proof is None or source['feasible'] or 'certified_from_trace_index' in source:
            raise ValueError('Inactive-cap reuse requires a directly solved infeasible source')
        if not certificate_covers(proof, row['fraction']):
            raise ValueError('Reused fraction lies outside the inactive-cap domain')
        for key in ('reason', 'normalized_margin', 'constraint_margins'):
            if row.get(key) != source.get(key):
                raise ValueError('Reused plant result differs from its solved source')
        reused += 1
    actual = len(trace) - reused
    for key, value in (('reservoir_evaluations', actual), ('certified_evaluations', reused)):
        if key in coupling and (type(coupling[key]) is not int or coupling[key] != value):
            raise ValueError('Inactive-cap evaluation accounting mismatch')
        if enabled and key not in coupling:
            raise ValueError('Missing inactive-cap evaluation accounting')
    return actual
