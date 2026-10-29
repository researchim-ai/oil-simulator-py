"""Рисунки страницы web/norne.html по результатам scripts/run_norne.py.

    NORNE_DIR=путь/к/opm-tests/norne NORNE_OUT=results/norne python scripts/norne_figures.py

Сравнение с Eclipse 2014.2 из NORNE_DIR/ECL.2014.2; рисунки пишутся в web/assets (NORNE_FIGS — другой каталог).
Объёмные виды строятся на CPU через Matplotlib без OpenGL. Сводные цифры печатаются в stdout в JSON.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from simulator.ecl_io import EclSummary, read_ecl_arrays

ROOT = os.environ.get("NORNE_OUT", "results/norne")
ECL = os.path.join(os.environ.get("NORNE_DIR", "opm-tests/norne"), "ECL.2014.2", "NORNE_ATW2013")
OUT = os.environ.get("NORNE_FIGS", "web/assets")
os.makedirs(OUT, exist_ok=True)

for p in (
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
):
    font_manager.fontManager.addfont(p)

BG, INK, MUTED, GRID = "#14110e", "#f3ecdf", "#b3a394", "#342c26"
OURS, ECLC, HIST = "#e39a3c", "#7ec4de", "#9a8e80"
plt.rcParams.update({
    "font.family": "Noto Sans",
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 11,
    "axes.unicode_minus": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "medium",
})

s = dict(np.load(f"{ROOT}/summary.npz"))
F = dict(np.load(f"{ROOT}/fields.npz"))
sub = np.load(f"{ROOT}/substeps.npz")["data"]
meta = json.load(open(f"{ROOT}/meta.json"))
e = EclSummary(ECL)
t, te = s["TIME"], e.days
yr = lambda d: 1997 + 11 / 12 + np.asarray(d, dtype=float) / 365.25


def ecl(kw, name=None):
    try:
        return e.get(kw, name)
    except KeyError:
        return None


def at_ecl(kw, name=None):
    v = ecl(kw, name)
    if v is None:
        return None
    return np.interp(t, te, v)


def style_ax(ax):
    ax.tick_params(length=0, pad=4)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


# ---------------------------------------------------------------- rates
panels = [
    ("FOPR", "Нефть, м³/сут", 1.0),
    ("FWPR", "Вода, м³/сут", 1.0),
    ("FGPR", "Газ, тыс. м³/сут", 1e-3),
    ("FWIR", "Закачка воды, м³/сут", 1.0),
    ("FGIR", "Закачка газа, тыс. м³/сут", 1e-3),
    ("FWCT", "Обводнённость", 1.0),
]
fig, axes = plt.subplots(3, 2, figsize=(12.4, 8.6), dpi=150)
for ax, (k, title, sc) in zip(axes.flat, panels):
    if ecl(k + "H") is not None:
        ax.plot(yr(te), ecl(k + "H") * sc, color=HIST, lw=0, marker="o", ms=1.5, alpha=0.55, label="История")
    ax.plot(yr(te), ecl(k) * sc, color=ECLC, lw=1.8, label="Eclipse 2014.2", solid_capstyle="round")
    ax.plot(yr(t), s[k] * sc, color=OURS, lw=1.35, ls=(0, (3.2, 1.6)), label="Наш солвер")
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=8)
    ax.set_ylim(bottom=0)
    style_ax(ax)
axes[0, 0].legend(frameon=False, loc="upper right", fontsize=10, labelcolor=MUTED)
fig.tight_layout(pad=1.1)
fig.savefig(f"{OUT}/rates.png")
plt.close()

# ---------------------------------------------------------------- cumulative + pressure
fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), dpi=150)
specs = [
    ("FOPT", "FOPR", "Нефть, млн м³", 1e-6),
    ("FWPT", "FWPR", "Вода, млн м³", 1e-6),
    ("FGPT", "FGPR", "Газ, млрд м³", 1e-9),
]
for ax, (k, rk, title, sc) in zip(axes.flat[:3], specs):
    if ecl(k + "H") is not None:
        ax.plot(yr(te), ecl(k + "H") * sc, color=HIST, lw=1.15, ls=(0, (1, 1.4)), label="История")
    ax.plot(yr(te), ecl(k) * sc, color=ECLC, lw=1.9, label="Eclipse")
    ax.plot(yr(t), s[k] * sc, color=OURS, lw=1.6, ls=(0, (3.2, 1.6)), label="Наш солвер")
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=8)
    style_ax(ax)
g = dict((n, a) for n, a in read_ecl_arrays(ECL + ".EGRID") if n in ("COORD", "ZCORN", "ACTNUM"))
nx, ny, nz = meta["dims"]
act = g["ACTNUM"].reshape(nz, ny, nx).transpose(2, 1, 0) > 0
pv_c = F["pv"] * act
hc = [pv_c * (1.0 - F["sw"][n]) for n in range(len(F["t"]))]
fpr = np.array([float((F["p"][n] * hc[n]).sum() / hc[n].sum()) for n in range(len(F["t"]))])
ax = axes.flat[3]
ax.plot(yr(te), ecl("FPR"), color=ECLC, lw=1.9, label="Eclipse, FPR")
ax.plot(yr(F["t"]), fpr, color=OURS, marker="o", ms=4.5, lw=1.3, ls=(0, (3.2, 1.6)), label="Наш, среднее по HCPV")
ax.set_title("Давление, бар", loc="left", fontsize=13, color=INK, pad=8)
style_ax(ax)
axes[0, 0].legend(frameon=False, fontsize=10, labelcolor=MUTED)
ax.legend(frameon=False, fontsize=10, labelcolor=MUTED)
fig.tight_layout(pad=1.1)
fig.savefig(f"{OUT}/cumulative.png")
plt.close()

# hero sparkline
def svg_series(xs, ys, w, h, pad=2):
    x0, x1, y0, y1 = xs.min(), xs.max(), min(ys[0].min(), ys[1].min()), max(ys[0].max(), ys[1].max())
    def xy(x, y):
        X = pad + (x - x0) / (x1 - x0) * (w - 2 * pad)
        Y = pad + (y1 - y) / (y1 - y0) * (h - 2 * pad)
        return X, Y
    paths = []
    for y in ys:
        pts = [xy(x, v) for x, v in zip(xs, y)]
        d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
        paths.append(d)
    return paths

m = np.isfinite(s["FOPT"])
p_ours, p_ecl = svg_series(yr(t[m]), [s["FOPT"][m] / 1e6, at_ecl("FOPT")[m] / 1e6], 640, 180)
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 180" fill="none">
<path d="{p_ecl}" stroke="#7ec4de" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>
<path d="{p_ours}" stroke="#e39a3c" stroke-width="2.2" stroke-dasharray="5 3.5" stroke-linejoin="round" stroke-linecap="round"/>
</svg>'''
open(f"{OUT}/hero_fopt.svg", "w").write(svg)

# ---------------------------------------------------------------- wells
prods = sorted(
    {k.split(":")[1] for k in s if k.startswith("WOPR:")},
    key=lambda w: -float(np.nan_to_num(ecl("WOPT", w)).max()) if ecl("WOPT", w) is not None else 0,
)
top = [w for w in prods if ecl("WOPR", w) is not None][:12]
for fname, kw, sc in (
    ("wells_oil.png", "WOPR", 1.0),
    ("wells_bhp.png", "WBHP", 1.0),
    ("wells_water.png", "WWPR", 1.0),
    ("wells_gas.png", "WGPR", 1e-3),
):
    fig, axes = plt.subplots(4, 3, figsize=(12.4, 9.4), dpi=140, sharex=True)
    for ax, w in zip(axes.flat, top):
        h = ecl(kw + "H", w)
        if h is not None and kw != "WBHP":
            ax.plot(yr(te), h * sc, color=HIST, lw=0, marker="o", ms=1.15, alpha=0.5)
        ax.plot(yr(te), ecl(kw, w) * sc, color=ECLC, lw=1.35)
        if f"{kw}:{w}" in s:
            ax.plot(yr(t), s[f"{kw}:{w}"] * sc, color=OURS, lw=1.15, ls=(0, (3, 1.5)))
        ax.set_title(w, loc="left", fontsize=12, color=INK, fontweight="medium", pad=4)
        ax.set_ylim(bottom=0)
        style_ax(ax)
        ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.7)
    fig.savefig(f"{OUT}/{fname}")
    plt.close()

# ---------------------------------------------------------------- maps
coord = g["COORD"].reshape(ny + 1, nx + 1, 6)
zc = g["ZCORN"].reshape(2 * nz, 2 * ny, 2 * nx)
X = coord[..., 0].T
Y = coord[..., 1].T
picks = [0, len(F["t"]) // 3, 2 * len(F["t"]) // 3, len(F["t"]) - 1]


def col_avg(a, w=None):
    w = pv_c if w is None else w * act
    num, den = (a * w).sum(2), w.sum(2)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


for fname, key, cmap, vmax in (
    ("map_so.png", "so", "YlOrBr", 0.9),
    ("map_sw.png", "sw", "Blues", 1.0),
    ("map_sg.png", "sg", "Greens", 0.6),
    ("map_p.png", "p", "magma", None),
):
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 4.15), dpi=150)
    vals = [col_avg(F[key][n]) for n in picks]
    lo = 0.0 if key != "p" else np.nanmin(vals)
    hi = np.nanmax(vals) if vmax is None else vmax
    on = np.isfinite(vals[0])
    xs = np.concatenate([X[:-1, :-1][on], X[1:, 1:][on]])
    ys = np.concatenate([Y[:-1, :-1][on], Y[1:, 1:][on]])
    for ax, n, v in zip(axes, picks, vals):
        im = ax.pcolormesh(X, Y, np.ma.masked_invalid(v), cmap=cmap, vmin=lo, vmax=hi, shading="flat")
        ax.set_aspect("equal")
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{yr(F['t'][n]):.1f}", fontsize=12, color=INK, pad=6)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.82, pad=0.012, fraction=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9, colors=MUTED)
    fig.savefig(f"{OUT}/{fname}", bbox_inches="tight", pad_inches=0.15)
    plt.close()

# ---------------------------------------------------------------- solver
ok = sub[sub[:, 2] > 0]
bad = sub[sub[:, 2] == 0]
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.55), dpi=150)
ax = axes[0]
ax.semilogy(yr(ok[:, 0] + ok[:, 1]), ok[:, 1], color=OURS, lw=0, marker=".", ms=3.2, alpha=0.9)
if len(bad):
    ax.semilogy(yr(bad[:, 0]), np.maximum(bad[:, 1], 1e-3), color="#e07a5f", lw=0, marker="x", ms=5, label="отказ шага")
    ax.legend(frameon=False, fontsize=10, labelcolor=MUTED)
ax.set_title("Шаг, сутки", loc="left", fontsize=13, color=INK, pad=8)
style_ax(ax)
ax = axes[1]
ax.hist(ok[:, 3], bins=np.arange(0, ok[:, 3].max() + 2) - 0.5, color=OURS, edgecolor=BG)
ax.set_title("Итераций Ньютона", loc="left", fontsize=13, color=INK, pad=8)
style_ax(ax)
ax = axes[2]
progress = np.maximum.accumulate(np.where(sub[:, 2] > 0, sub[:, 0] + sub[:, 1], sub[:, 0]))
ax.plot(yr(progress), np.cumsum(sub[:, 4]) / 60, color=OURS, lw=1.8)
ax.set_title("Время подшагов, мин", loc="left", fontsize=13, color=INK, pad=8)
style_ax(ax)
fig.tight_layout(pad=1.0)
fig.savefig(f"{OUT}/solver.png")
plt.close()

# ---------------------------------------------------------------- numbers
def cum(rate):
    r = np.nan_to_num(rate)
    return np.concatenate([[0.0], np.cumsum(r[1:] * np.diff(t))])


rows = []
report_path = os.path.join(ROOT, "comparison.json")
report = json.load(open(report_path)) if os.path.exists(report_path) else {}
for k, rk in (("FOPT", "FOPR"), ("FWPT", "FWPR"), ("FGPT", "FGPR"), ("FWIT", "FWIR"), ("FGIT", "FGIR")):
    audited = report.get("field_cumulative", {}).get(k)
    ours = float(audited["ours"] if audited else (s[k][-1] if k in s else cum(s[rk])[-1]))
    ev = float(at_ecl(k)[-1])
    rows.append({"k": k, "ours": ours, "ecl": ev, "pct": 100 * (ours - ev) / ev})
rate_err = {}
for k in ("FOPR", "FWPR", "FGPR", "FWIR", "FGIR"):
    ev = at_ecl(k)
    msk = np.isfinite(s[k]) & np.isfinite(ev) & (t > 1)
    rate_err[k] = 100 * float(np.abs(s[k][msk] - ev[msk]).sum() / np.abs(ev[msk]).sum())
wells = []
for w in top:
    ev = float(at_ecl("WOPT", w)[-1])
    ours = float(s[f"WOPT:{w}"][-1]) if f"WOPT:{w}" in s else float("nan")
    wells.append({"w": w, "ours": ours, "ecl": ev, "pct": 100 * (ours - ev) / ev})
# ошибки накопленной нефти на нескольких датах
marks = {}
for day in (365, 1000, 2000, 3312):
    i = int(np.argmin(np.abs(t - day)))
    ev = float(at_ecl("FOPT")[i])
    marks[str(day)] = {"t": float(t[i]), "pct": 100 * (float(s["FOPT"][i]) - ev) / ev}
numbers = {"cum": rows, "rate": rate_err, "wells": wells, "marks": marks,
                  "n_act": int(act.sum()), "fpr_end_ours": float(fpr[-1]),
                  "fpr_end_ecl": float(np.interp(t[-1], te, ecl("FPR")))}
with open(os.path.join(OUT, "norne-numbers.json"), "w") as stream:
    json.dump(numbers, stream, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps(numbers, ensure_ascii=False, indent=2))

# ---------------------------------------------------------------- 3D: CPU renderer, no OpenGL/GPU context
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize

pts = np.empty((2 * nx, 2 * ny, 2 * nz, 3))
for di in (0, 1):
    for dj in (0, 1):
        pil = coord[dj:dj + ny, di:di + nx].transpose(1, 0, 2)
        x1, y1, z1, x2, y2, z2 = (pil[..., q][..., None] for q in range(6))
        for dk in (0, 1):
            z = zc[dk::2, dj::2, di::2].transpose(2, 1, 0)
            f = np.where(np.abs(z2-z1) > 1e-9, (z-z1)/np.where(np.abs(z2-z1) > 1e-9, z2-z1, 1), 0)
            pts[di::2, dj::2, dk::2] = np.stack([x1+f*(x2-x1), y1+f*(y2-y1), -5*z], -1)

# Keep exterior faces and separated faces across faults; omit shared interiors.
faces, indices, lighting = [], [], []
for axis in range(3):
    others = [a for a in range(3) if a != axis]
    def vertices(side):
        corners = []
        for a, b in [(0,0), (1,0), (1,1), (0,1)]:
            bits = [0,0,0]; bits[axis] = side; bits[others[0]] = a; bits[others[1]] = b
            corners.append(pts[bits[0]::2, bits[1]::2, bits[2]::2])
        return np.stack(corners, -2)
    for side in (0,1):
        shift = 1 if side == 0 else -1
        neighbor = np.roll(act, shift, axis=axis)
        boundary = [slice(None)]*3; boundary[axis] = 0 if side == 0 else -1
        neighbor[tuple(boundary)] = False
        vv = vertices(side)
        other = np.roll(vertices(1-side), shift, axis=axis)
        separated = np.max(np.abs(vv-other), axis=(-2,-1)) > 1e-3
        mask = act & (~neighbor | separated)
        faces.append(vv[mask]); indices.append(np.argwhere(mask))
        lighting.append(np.full(int(mask.sum()), [.70,.88,1.0][axis] * (1.0 if side else .91)))
faces, indices, lighting = np.concatenate(faces), np.concatenate(indices), np.concatenate(lighting)
lo, hi = faces.min((0,1)), faces.max((0,1))
center = (lo+hi)/2
faces -= center
extent = hi-lo
shots = []
for key, cmap, clim in (("so","YlOrBr",(0,.9)), ("sw","Blues",(0,1)), ("sg","Greens",(0,.6))):
    for tag, n in (("start",0), ("end",len(F["t"])-1)):
        values = F[key][n][tuple(indices.T)]
        colors = plt.get_cmap(cmap)(Normalize(*clim)(values))
        colors[:,:3] *= lighting[:,None]
        fig = plt.figure(figsize=(10,7), dpi=140)
        ax = fig.add_axes([0,0,1,1], projection="3d", computed_zorder=False)
        ax.set_facecolor(BG); ax.set_axis_off(); ax.set_proj_type("ortho")
        ax.add_collection3d(Poly3DCollection(faces, facecolors=colors, edgecolors="none", linewidths=0, zsort="average"))
        ax.set_xlim(-extent[0]/2,extent[0]/2); ax.set_ylim(-extent[1]/2,extent[1]/2); ax.set_zlim(-extent[2]/2,extent[2]/2)
        ax.set_box_aspect(extent.copy()); ax.view_init(elev=24,azim=-55)
        path = f"{OUT}/3d_{key}_{tag}.png"
        fig.savefig(path,facecolor=BG); plt.close(fig); shots.append(path)
ims = [np.array(Image.open(p).convert("RGB")) for p in shots]
bg = np.array([0x14,0x11,0x0e]); boxes = []
for im in ims:
    yy, xx = np.where(np.abs(im.astype(np.int16)-bg).sum(-1)>12)
    boxes.append((xx.min(),yy.min(),xx.max(),yy.max()))
x0=max(0,min(b[0] for b in boxes)-28); y0=max(0,min(b[1] for b in boxes)-22)
x1=min(ims[0].shape[1],max(b[2] for b in boxes)+28); y1=min(ims[0].shape[0],max(b[3] for b in boxes)+22)
for path, im in zip(shots,ims):
    Image.fromarray(im[y0:y1,x0:x1]).save(path,optimize=True)
print("3D CPU:",len(faces),"faces;",x1-x0,y1-y0)
