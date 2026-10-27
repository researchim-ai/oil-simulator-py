"""Графики страницы web/gaslift.html: GASLIFT-04 (opm-tests), наш солвер против OPM Flow до 213 сут."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from simulator.ecl_io import EclSummary  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

D = os.path.join(ROOT, "tests", "data", "gaslift")
OUT = os.path.join(ROOT, "web", "assets")
T_END = 213

BG, INK, MUTED, GRID = "#14110e", "#f3ecdf", "#b3a394", "#342c26"
OURS, REF = "#e39a3c", "#7ec4de"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 11,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})

s, _ = run_deck(os.path.join(D, "GASLIFT-04.DATA"), device=torch.device("cpu"), max_days=T_END)
s = {k: np.asarray(v, dtype=float) for k, v in s.items()}
r = EclSummary(os.path.join(D, "opm", "GASLIFT-04"))
tr = np.asarray(r.get("TIME"))
m = tr <= T_END + 1e-6


def opm(key):
    kw, *n = key.split(":")
    return np.asarray(r.get(kw, n[0] if n else None))[m]


def ours(key):
    return np.nan_to_num(s[key])


panels = [("FOPR", "Нефть месторождения, м³/сут", 1.0),
          ("FGLIR", "Газлифт месторождения, тыс. м³/сут", 1e-3),
          ("WGLIR:OP-A01", "Газлифт OP-A01, тыс. м³/сут", 1e-3),
          ("WGLIR:OP-B01", "Газлифт OP-B01, тыс. м³/сут", 1e-3)]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
for ax, (k, title, sc) in zip(axes.flat, panels):
    a, b = opm(k), ours(k)[1:]
    ax.step(np.r_[0, tr[m]], np.r_[a[0], a] * sc, where="pre", color=REF, lw=1.9, label="OPM Flow")
    ax.step(np.r_[0, s["TIME"][1:]], np.r_[b[0], b] * sc, where="pre", color=OURS, lw=1.6, ls=(0, (3.2, 1.6)), label="Наш солвер")
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=8)
    ax.set_xlim(0, T_END)
    ax.set_ylim(bottom=0)
    if k == "FOPR":
        ax.set_ylim(9000, 10300)
    for sp in ax.spines:
        ax.spines[sp].set_color(GRID)
for ax in axes[1]:
    ax.set_xlabel("сутки")
axes[0, 0].legend(frameon=False, loc="lower right", fontsize=10, labelcolor=MUTED)
fig.savefig(os.path.join(OUT, "gaslift_rates.png"), dpi=150)


def total(t, q):
    t = np.concatenate([[0.0], t])
    return float(np.sum(np.diff(t) * q))


t0 = s["TIME"][1:]
for k in ("FOPR", "FGLIR"):
    a, b = total(t0, ours(k)[1:]), total(tr[m], opm(k))
    print(f"{k}: наш {a:.4g}, OPM {b:.4g}, {100 * (a / b - 1):+.1f}%")
