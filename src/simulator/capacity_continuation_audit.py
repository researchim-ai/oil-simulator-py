"""Audit local-branch continuation and mandatory full-scan evidence."""

def audit_capacity_continuation(report, control, previous=None):
    if control.get('search_method') != 'continuation_safeguarded':
        return
    if report.get('search_method') != 'continuation_safeguarded':
        raise ValueError('Continuation policy differs from execution')
    interval=control.get('continuation_scan_interval',16)
    if type(interval) is not int or interval<1:
        raise ValueError('Invalid continuation scan interval')
    record=report.get('capacity_continuation') or {}
    used,age=record.get('used'),record.get('age')
    if (type(used) is not bool or type(age) is not int or not 0<=age<interval
            or record.get('scan_interval')!=interval):
        raise ValueError('Missing or invalid continuation evidence')
    if used:
        before=None if previous is None else previous.get('capacity_continuation',{}).get('age',0)
        if (previous is None or before is None or age!=before+1 or record.get('reason')!='local_bracket'
                or previous.get('operator_controls')!=report.get('operator_controls')
                or previous.get('requested_rate_caps_m3_day')!=report.get('requested_rate_caps_m3_day')):
            raise ValueError('Continuation reused stale controls or exceeded scan cadence')
        if not control.get('min_fraction',.001)<previous['fraction']<1:
            raise ValueError('Continuation requires a previously constrained branch')
    else:
        if age!=0 or record.get('reason') not in ('full_scan','periodic_scan','probe_fallback'):
            raise ValueError('Invalid full-scan continuation reset')
        samples={r['fraction']:r for r in report['trace']}
        if 1. not in samples:
            raise ValueError('Full scan omitted unrestricted trial')
        if report['fraction']!=1.:
            minimum=control.get('min_fraction',.001);count=control.get('scan_points',9)
            expected=[minimum+(1-minimum)*i/(count-1) for i in range(count-1)]
            if any(x not in samples for x in expected):
                raise ValueError('Full scan omitted domain samples')
