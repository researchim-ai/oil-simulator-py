# ruff: noqa: SIM102
"""Frozen pre-packet control and RESV reference for exact comparison."""
DAY = 86400.0

def legacy_update_controls(self, p_bh, q_surf):
    """Переключение дебит ↔ BHP ↔ THP: работает самое жёсткое из ограничений (как в Eclipse/OPM)."""
    msgs = []
    self.pbh_override = {}
    if self.network is not None and self._Q is not None and self._net_iter < self.network["balan"]["iters"]:
        if self._apply_network(self._Q):
            self._net_iter += 1
            msgs.append("сеть: " + ", ".join(f"{k} {v / 1e5:.2f} бар" for k, v in self.node_p.items()
                                              if v is not None))
    capped = self.switches >= 4 * max(1, self.n_w)
    for w_idx, w in enumerate(self.wells):
        ctrl, val = self.step_targets[w_idx]
        if ctrl != 'rate':
            continue
        lim = self.bhp_limit(w)
        pb = float(p_bh[w_idx])
        mode = self.step_mode[w_idx]
        if mode == 'stop':
            continue
        thp_b = None
        if self.has_thp(w_idx) and self._Q is not None:
            thp_b = self._vfp_bhp(w_idx, self._Q[w_idx])
        # sg·p: у добывающей ограничения снизу, у нагнетательной — сверху; работает самое жёсткое
        sg = self._sgn(w_idx)
        thp_harder = thp_b is not None and sg * thp_b > sg * lim + 1e-9 * abs(lim)
        new = None
        if mode == 'rate' and thp_b is not None and sg > 0 and getattr(self, "_lin", None) is not None:
            # дебит уставки сравнивается с потенциалом на устойчивой ветви THP, а не с BHP_VFP(текущий дебит):
            # на неустойчивой (левой) ветви кривая лифта выше при меньших дебитах
            st = self._thp_stable(w_idx, lim)
            if st is None and pb < self._vfp_bhp(w_idx, self._Q[w_idx]):
                new = 'stop'
            elif st is not None and st[1] is not None and (
                    st[1] < val * (1 - 1e-9) or w_idx in getattr(self, "_group_capacity_limited", ())):
                new = 'thp_limit'
                self.pbh_override[w_idx] = st[0]
            elif pb < lim:
                new = 'bhp_limit'
        elif mode == 'rate':
            eff = thp_b if thp_harder else lim
            if sg * pb < sg * eff:
                new = 'thp_limit' if thp_harder else 'bhp_limit'
        else:
            q = float(q_surf[w_idx])
            if q > val * (1 + 1e-9) and w_idx not in getattr(self, "_group_capacity_limited", ()):
                new = 'rate'
            elif mode == 'bhp_limit' and thp_harder:
                new = 'thp_limit'
            elif mode == 'thp_limit' and (thp_b is None or sg * pb < sg * lim):
                new = 'bhp_limit'
        # не больше 3 переключений скважины за шаг (как max_number_of_well_switches в OPM); возврат на дебит
        # разрешён всегда: решение на ограничении с дебитом выше уставки недопустимо
        exhausted = capped or self.well_switches[w_idx] >= 3
        if new is not None and (not exhausted or new == 'rate'):
            self.step_mode[w_idx] = new
            self.switches += 1
            self.well_switches[w_idx] += 1
            msgs.append(f"{w.name}: {mode}→{new} (p_bh {pb/1e6:.3f} МПа, дебит {float(q_surf[w_idx])*DAY:.2f})")
    return msgs

def legacy_update_resv_averages(self, p, sw):
    """RateConverter state, weighted by current hydrocarbon pore volume."""
    pv_current = self._pv(p.view(-1))
    hc = pv_current * (1.0 - sw.view(-1)).clamp_min(0.0)
    if float(hc.sum()) <= 0:
        hc = pv_current
    hc_sum = hc.sum().clamp_min(1e-30)
    self.wells.p_avg = float((hc * p.view(-1)).sum() / hc_sum)
    if self.pvt.has_pvt:
        # RateConverter averages all three quantities over the SAME
        # hydrocarbon pore volume, not separate oil/gas phase volumes.
        self.wells.rs_avg = float((hc * self.fluid.rs.reshape(-1).double()).sum() / hc_sum)
        rv = getattr(self.fluid, "rv", None)
        self.wells.rv_avg = float((hc * rv.reshape(-1).double()).sum() / hc_sum) \
            if self.wet and rv is not None else None

