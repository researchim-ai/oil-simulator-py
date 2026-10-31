"""Frozen convergence contract before packing GPU diagnostic transfers."""

import torch

from simulator.blackoil_fim import I_B
from simulator.fim_wells import DAY


def legacy_convergence(self, R, dt, pv, wsys, p_bh, v=None):
    Rb = R.view(-1, 3)
    b_avg = (1.0 / v[I_B : I_B + 3]).mean(dim=1)
    scale = dt * b_avg / self.rho_sc
    cnv_cell = scale * Rb.abs() / pv.unsqueeze(1)
    cnv = cnv_cell.max(dim=0).values
    mb = scale * Rb.sum(dim=0).abs() / pv.sum()
    bad = cnv_cell.max(dim=1).values > self.tol_cnv
    # ослабление — после max_strict_iter итераций или раньше, если Ньютон застрял (невязка не убывает)
    prev = getattr(self, "_prev_norm", None)
    norm = max(max(c / self.tol_cnv, m / self.tol_mb) for c, m in zip(cnv.tolist(), mb.tolist()))
    stagnant = prev is not None and norm > 0.99 * prev
    relaxed = (
        (getattr(self, "_newton_iter", 0) >= self.max_strict_iter or stagnant)
        and bool(bad.any())
        and float(pv[bad].sum() / pv.sum()) <= self.relaxed_max_pv_fraction
    )
    tol_cnv = self.tol_cnv_relaxed if relaxed else self.tol_cnv
    tol_mb = self.tol_mb_relaxed if relaxed else self.tol_mb
    well_ok, well_err = True, 0.0
    if wsys is not None:
        rate = wsys["is_rate"]
        # нулевая уставка (группа без отбора) — абсолютная невязка в долях 0.01 ст. м³/сут
        rel = wsys["r_w"].abs() / torch.clamp(wsys["target"].abs(), min=0.01 / DAY)
        rate_err = torch.where(rate, rel, torch.zeros_like(rel))
        bhp_err = torch.where(rate, torch.zeros_like(rel), wsys["r_w"].abs())
        # таблица VFP кусочно-линейна: у узлов Ньютон по THP колеблется на 0.01–0.05 бар — допуск 0.01 бар,
        # с 4-й итерации 0.1 бар (как ослабленный CNV)
        tol_thp = 1e3 if getattr(self, "_newton_iter", 0) < 4 else 1e4
        tol_p = torch.where(
            wsys.get("is_thp", torch.zeros_like(rate)),
            torch.full_like(rel, max(self.tol_well_bhp, tol_thp)),
            torch.full_like(rel, self.tol_well_bhp),
        )
        well_err = float(rate_err.max()) if bool(rate.any()) else 0.0
        well_ok = bool((rate_err <= self.tol_well_rate).all() and (bhp_err <= tol_p).all())
        stopped = wsys.get("is_stopped")
        if stopped is not None and bool(stopped.any()):
            # A zero total surface rate alone must never hide oil/gas
            # cancellation. Every stopped-well component must be zero.
            component_err = wsys["component_rates"][stopped].abs() / (0.01 / DAY)
            well_err = max(well_err, float(component_err.max()))
            well_ok = well_ok and bool((component_err <= self.tol_well_rate).all())
    cnv_l, mb_l = cnv.tolist(), mb.tolist()
    ok = max(cnv_l) <= tol_cnv and max(mb_l) <= tol_mb and well_ok
    return ok, {
        "cnv": cnv_l,
        "mb": mb_l,
        "well_rate_rel_err": well_err,
        "cnv_relaxed": relaxed,
        "tol_cnv_used": tol_cnv,
        "tol_mb_used": tol_mb,
        "unrelaxed_converged": max(cnv_l) <= self.tol_cnv and max(mb_l) <= self.tol_mb and well_ok,
        "cnv_bad_pv_fraction": float(pv[bad].sum() / pv.sum()),
    }


def legacy_store_well_results(self, wsys, p_bh, dt):
    rho = self.pvt.rho_sc
    for w_idx, w in enumerate(self.wells.wells):
        m = self.wells.conn_well == w_idx
        q = wsys["q"][m].sum(dim=0)
        q_o, q_w, q_g = (float(q[i]) / rho[i] * DAY for i in range(3))
        w.last_net_connection_rates = {"oil": q_o, "water": q_w, "gas": q_g}
        if getattr(w, "stopped", False):
            # Prescribed zero surface boundary; keep the actual connection
            # imbalance above for diagnostics and check it in convergence.
            q_o = q_w = q_g = 0.0
        w.bhp_mpa = float(p_bh[w_idx]) / 1e6
        tab = self.wells._lift(w_idx)
        if tab is not None:
            Q = self.wells._sgn(w_idx) * q / self.wells.rho_sc
            with torch.no_grad():
                thp = tab.thp_from_bhp(
                    p_bh[w_idx : w_idx + 1] - self.wells._vfp_hydro(w_idx, tab),
                    Q[0:1],
                    Q[1:2],
                    Q[2:3],
                    float(getattr(w, "alq", 0.0) or 0.0),
                )
            w.thp_mpa = float(thp) / 1e6
        w.last_surface_rates = {"oil": q_o, "water": q_w, "gas": q_g, "liquid": q_o + q_w}
        # добыча без перетоков (перфорации с обратным потоком в пласт не вычитаются, как в Eclipse)
        qp = wsys["q"][m].clamp_min(0.0).sum(dim=0)
        w.last_surface_prod = {
            ph: float(qp[i]) / rho[i] * DAY for i, ph in enumerate(("oil", "water", "gas"))
        }
        # накопленные поверхностные объёмы, м³ (знак как у дебитов: + отбор); отдельно — за периоды
        # добычи (cum_prod) и закачки (cum_inj, + закачка) у скважин, меняющих тип
        zero = {"oil": 0.0, "water": 0.0, "gas": 0.0}
        cum = getattr(w, "cum_surface", None) or dict(zero)
        cp = getattr(w, "cum_prod", None) or dict(zero)
        ci = getattr(w, "cum_inj", None) or dict(zero)
        e = float(getattr(w, "eff", 1.0) or 1.0)
        for ph, val in (("oil", q_o * e), ("water", q_w * e), ("gas", q_g * e)):
            cum[ph] += val * dt / DAY
            if w.type == "producer":
                cp[ph] += val * dt / DAY
            else:
                ci[ph] -= val * dt / DAY
        w.cum_surface, w.cum_prod, w.cum_inj = cum, cp, ci
