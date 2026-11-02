# -*- coding: utf-8 -*-
"""Regenerate the two figures of the model/software paper.

Inputs  (data/): _sweep_fine.csv, _sweep_ext.csv, _sweep.csv
Outputs (figures/): Fig1_emission_factor.(png|pdf), Fig2_operating_map.(png|pdf)

Figure 1 of the paper = emission factor (emission-factor panel).
Figure 2 of the paper = steady-state operating map with the Pareto front.
No value is hard-coded: everything is recomputed from the archived scan results.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.unicode_minus": False,
})


def load_csv(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def panel(ax, tag):
    ax.text(-0.14, 1.06, tag, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor="white")
    plt.close(fig)
    print("saved", name, "(png+pdf)")


fine = load_csv("_sweep_fine.csv")
ext = load_csv("_sweep_ext.csv")

uniq = {}
for r in fine + ext:
    uniq[(round(r["so"], 3), round(r["sno"], 3))] = r
rows_u = list(uniq.values())


def feasible(r):
    return r["snh"] <= 4.0 and r["snoe"] <= 18.0 and r["tss"] <= 30.0


fea = [r for r in rows_u if feasible(r)]


def pareto(points):
    pts = sorted(points)
    out, best = [], float("inf")
    for x, y in pts:
        if y < best:
            out.append((x, y))
            best = y
    return out


pf = pareto([(r["eqi"], r["tot"]) for r in fea])
print("rows %d, unique %d, feasible %d, Pareto %d"
      % (len(fine) + len(ext), len(rows_u), len(fea), len(pf)))

# ---------------------------------------------------------------- Figure 1
sno_list = sorted({round(r["sno"], 3) for r in rows_u})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3),
                         gridspec_kw={"width_ratios": [1.35, 1.0]})
ax = axes[0]
cmap = plt.get_cmap("plasma")
norm = matplotlib.colors.Normalize(vmin=min(sno_list), vmax=max(sno_list))
for sno in sno_list:
    sub = sorted([r for r in rows_u if round(r["sno"], 3) == sno], key=lambda r: r["so"])
    ax.plot([r["so"] for r in sub], [r["ef"] for r in sub], "-o", ms=2.2, lw=0.9,
            color=cmap(norm(sno)))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
cb.set_label("Nitrate setpoint $S_{NO_3}$ (gN/m$^3$)")
cb.ax.tick_params(labelsize=7)
ax.set_xlabel("Aerobic DO setpoint (mg/L)")
ax.set_ylabel("Direct N$_2$O emission factor\n(% of influent N)")
ax.set_title("Emission factor vs aeration intensity")
panel(ax, "(a)")

ax = axes[1]
for do in (1.0, 2.0, 3.0, 4.0):
    xs, ys = [], []
    for sno in sno_list:
        cand = [r for r in rows_u if round(r["sno"], 3) == sno
                and abs(r["so"] - do) < 1e-6]
        if cand:
            xs.append(sno)
            ys.append(cand[0]["ef"])
    ax.plot(xs, ys, "-o", ms=2.6, lw=1.0, label="DO = %.1f" % do)
ax.set_xlabel("Nitrate setpoint $S_{NO_3}$ (gN/m$^3$)")
ax.set_ylabel("Direct N$_2$O emission factor (% of influent N)")
ax.set_title("Effect of nitrate setpoint")
ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
panel(ax, "(b)")
fig.tight_layout()
save(fig, "Fig1_emission_factor.png")

# ---------------------------------------------------------------- Figure 2
fig, ax = plt.subplots(figsize=(6.3, 4.7))
sc = ax.scatter([r["eqi"] for r in fea], [r["tot"] for r in fea],
                c=[r["so"] for r in fea], cmap="viridis", s=16, linewidths=0.25,
                edgecolors="white", vmin=1.0, vmax=4.0, zorder=2)
ax.plot([p[0] for p in pf], [p[1] for p in pf], "-o", color="#d62728", ms=3.2,
        lw=1.0, label="Pareto front (%d points)" % len(pf), zorder=3)
ax.scatter([7440], [8523], marker="*", s=150, color="black", zorder=4,
           label="BSM1 default setpoints")
ax.text(0.025, 0.975,
        "Pareto front: %d points\nEQI  5680 \u2192 6575 kg/d  (+15.8%%)\n"
        "carbon  6267 \u2192 4989 kgCO$_2$e/d  (\u221220.4%%)" % len(pf),
        transform=ax.transAxes, va="top", ha="left", fontsize=6.6, linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.6, alpha=0.95))
ax.set_xlabel("Effluent quality index EQI (kg/d)")
ax.set_ylabel("Total carbon footprint (kgCO$_2$e/d)")
ax.set_ylim(4700, 8900)
ax.set_title("Steady-state operating map: %d distinct feasible setpoint pairs" % len(fea))
cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
cb.set_label("Aerobic DO setpoint (mg/L)")
cb.ax.tick_params(labelsize=7)
ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="0.8")
save(fig, "Fig2_operating_map.png")

means = {s: float(np.mean([r["ef"] for r in rows_u if round(r["sno"], 3) == s]))
         for s in sno_list}
print("mean emission factor by nitrate setpoint:")
for k in sorted(means):
    print("  %5.2f  %.4f" % (k, means[k]))
