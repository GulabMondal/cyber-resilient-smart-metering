"""Regenerates every results figure (fig2-fig15) from results/ produced by experiments.py."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

R = json.load(open("results/results.json"))
E = np.load("results/energy_series.npz")
OUT = "figs/"
import os
os.makedirs(OUT, exist_ok=True)

# reference categorical palette (fixed order) + neutral ink
C1, C2, C3, C4, C5, C6 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASE, OPT = C2, C1        # baseline = orange, optimised = blue

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 1.6,
})


def save(fig, name):
    fig.savefig(OUT + name, dpi=300)
    plt.close(fig)


days = np.arange(1, 61)
b, o = slice(0, 30), slice(30, 60)

# ---- Fig 2: daily energy
dk = E["daily_kwh"]
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(days[b], dk[b], "-o", ms=3, color=BASE, label="Baseline period")
ax.plot(days[o], dk[o], "-o", ms=3, color=OPT, label="Optimisation period")
for sl, c in ((b, BASE), (o, OPT)):
    ax.hlines(dk[sl].mean(), days[sl][0], days[sl][-1], colors=c, linestyles="--", lw=1)
ax.text(2, dk[b].mean() + 12, f"mean {dk[b].mean():,.0f} kWh", color=INK2, fontsize=8)
ax.text(32, dk[o].mean() + 12, f"mean {dk[o].mean():,.0f} kWh", color=INK2, fontsize=8)
ax.axvline(30.5, color=INK2, lw=0.8, ls=":")
ax.set_xlabel("Day"); ax.set_ylabel("Daily energy (kWh)")
ax.legend(loc="lower left", ncol=2)
save(fig, "fig2.png")

# ---- Fig 3: idle energy
di = E["idle_kwh"]
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(days[b], di[b], color=BASE, width=0.8, label="Baseline period")
ax.bar(days[o], di[o], color=OPT, width=0.8, label="Optimisation period")
ax.set_xlabel("Day"); ax.set_ylabel("Idle energy loss (kWh/day)")
ax.legend(loc="upper right")
save(fig, "fig3.png")

# ---- Fig 4: apparent power profile (representative day), 1-min averages
sb = E["S_base_day"].reshape(-1, 12).mean(1); so = E["S_opt_day"].reshape(-1, 12).mean(1)
h = np.arange(len(sb)) / 60
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(h, sb, color=BASE, label="Baseline: synchronised start, uncorrected PF")
ax.plot(h, so, color=OPT, label="Optimised: staggered start, corrected PF")
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
ax.set_xlabel("Hour of day"); ax.set_ylabel("Apparent power (kVA, 1-min mean)")
ax.legend(loc="upper right", fontsize=8)
save(fig, "fig4.png")

# ---- Fig 5: SEC per machine
pm = R["energy"]["per_machine"]
x = np.arange(len(pm)); w = 0.36
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(x - w / 2, [m["base_sec"] for m in pm], w, color=BASE, label="Baseline")
ax.bar(x + w / 2, [m["opt_sec"] for m in pm], w, color=OPT, label="Optimised")
for i, m in enumerate(pm):
    ax.text(i, max(m["base_sec"], m["opt_sec"]) + 0.015, f"−{m['sec_impr_pct']:.1f}%", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels([m["name"] for m in pm])
ax.set_ylabel("SEC (kWh/kg, all energy)"); ax.set_ylim(0, max(m["base_sec"] for m in pm) * 1.2)
ax.legend(loc="upper left", ncol=2)
save(fig, "fig5.png")

# ---- Fig 6: daily PF
pf = E["pf_day"]
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot(days[b], pf[b], "-o", ms=3, color=BASE, label="Baseline period")
ax.plot(days[o], pf[o], "-o", ms=3, color=OPT, label="Optimisation period")
ax.axhline(0.90, color=INK2, lw=0.8, ls="--"); ax.text(1, 0.903, "0.90 penalty threshold", fontsize=8, color=INK2)
ax.set_xlabel("Day"); ax.set_ylabel("Energy-weighted PF")
ax.legend(loc="center right")
save(fig, "fig6.png")

# ---- Fig 7: detection rates
rows = R["detection"]["rows"]
names = [k for k in rows]
rates = [rows[k]["rate_pct"] for k in names]
order = np.argsort(rates)
fig, ax = plt.subplots(figsize=(7, 3.6))
cols = [C3 if rows[names[i]]["rate_pct"] >= 99.95 else (C4 if rows[names[i]]["rate_pct"] >= 90 else C2) for i in order]
ax.barh(np.arange(len(order)), [rates[i] for i in order], color=cols, height=0.65)
for j, i in enumerate(order):
    ax.text(rates[i] + 0.8, j, f"{rates[i]:.1f}%", va="center", fontsize=8, color=INK)
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([names[i] for i in order], fontsize=8)
ax.set_xlim(0, 112); ax.set_xlabel("Event detection rate (%)")
ax.grid(axis="y", visible=False)
save(fig, "fig7.png")

# ---- Fig 8: confusion matrix (row-normalised), RF chronological
cm = np.array(R["ml_chrono"]["Random Forest"]["cm"], float)
cmn = cm / cm.sum(1, keepdims=True)
short = ["Normal", "Energy\nineff.", "Elec.\nhealth", "Cyber\n(network)", "Source-side\nintegrity", "Combined"]
fig, ax = plt.subplots(figsize=(5.6, 4.6))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cmn[i, j]*100:.1f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=6.5,
                color="white" if cmn[i, j] > 0.6 else INK)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(short, fontsize=7); ax.set_yticklabels(short, fontsize=7)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised share")
save(fig, "fig8.png")

# ---- Fig 9: ROC (one-vs-rest), RF chronological
pp = np.load("results/rf_proba_chrono.npy"); yt = np.load("results/y_test_chrono.npy")
fig, ax = plt.subplots(figsize=(5, 4))
cols6 = [C1, C2, C3, C4, C5, C6]
for i in range(6):
    fpr, tpr, _ = roc_curve((yt == i).astype(int), pp[:, i])
    ax.plot(fpr, tpr, color=cols6[i], lw=1.4,
            label=f"{short[i].replace(chr(10), ' ')} (AUC {R['ml_chrono']['Random Forest']['auc_ovr'][i]:.3f})")
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(fontsize=7, loc="lower right")
save(fig, "fig9.png")

# ---- Fig 10: feature importance (top 15)
fi = R["feature_importance"][:15][::-1]
fig, ax = plt.subplots(figsize=(6, 4))
cyber = {"n_rx", "n_acc", "f_tag_fail", "f_seq_fail", "f_fresh_fail", "max_age", "gap_s", "n_spoof"}
phys = {"phys_resid", "feeder_resid", "feeder_resid30"}
cc = [C2 if f in cyber else (C3 if f in phys else C1) for f, _ in fi]
ax.barh(range(len(fi)), [v for _, v in fi], color=cc, height=0.65)
ax.set_yticks(range(len(fi))); ax.set_yticklabels([f for f, _ in fi], fontsize=8)
ax.set_xlabel("Mean decrease in impurity"); ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C1, label="Energy"), Patch(color=C2, label="Cyber"), Patch(color=C3, label="Physics / feeder")],
          loc="lower right", fontsize=8)
save(fig, "fig10.png")

# ---- Fig 11: ablation
ab = R["ablation"]
labels = list(ab)
fig, ax = plt.subplots(figsize=(7, 3.4))
y_ = np.arange(len(labels)); w = 0.38
ax.barh(y_ - w / 2, [ab[k]["acc"] * 100 for k in labels], w, color=C1, label="Accuracy (%)")
ax.barh(y_ + w / 2, [ab[k]["f1"] * 100 for k in labels], w, color=C4, label="Macro-F1 (×100)")
for j, k in enumerate(labels):
    ax.text(ab[k]["acc"] * 100 + 0.5, j - w / 2, f"{ab[k]['acc']*100:.1f}", va="center", fontsize=7)
    ax.text(ab[k]["f1"] * 100 + 0.5, j + w / 2, f"{ab[k]['f1']*100:.1f}", va="center", fontsize=7)
ax.set_yticks(y_); ax.set_yticklabels(labels, fontsize=7.5); ax.set_xlim(0, 108)
ax.legend(loc="lower right", fontsize=8); ax.grid(axis="y", visible=False)
save(fig, "fig11.png")

# ---- Fig 12: annual savings breakdown (baseline tariff, demand Rs 400)
r = R["roi"]["8.5_400"]
cats = [("Idle energy", r["idle"]), ("Process efficiency (SEC)", r["efficiency"]),
        ("Demand: PF correction", r["pf_demand"]), ("Demand: staggered start", r["stagger_demand"])]
fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.barh(range(4), [v / 1e5 for _, v in cats][::-1], color=[C1, C3, C4, C2][::-1], height=0.6)
for j, (n, v) in enumerate(cats[::-1]):
    ax.text(v / 1e5 + 0.05, j, f"₹{v/1e5:.2f} lakh", va="center", fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels([n for n, _ in cats][::-1], fontsize=8)
ax.set_xlabel("Annual saving (₹ lakh)"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(v for _, v in cats) / 1e5 * 1.3)
save(fig, "fig12.png")

# ---- Fig 13: cumulative cash flow incl. maintenance (three tariffs)
fig, ax = plt.subplots(figsize=(6.5, 3))
mo = np.arange(0, 61)
for key, c, lab in (("7.5_400", C2, "₹7.5/kWh"), ("8.5_400", C1, "₹8.5/kWh"), ("9.5_400", C3, "₹9.5/kWh")):
    net = R["roi"][key]["net"]
    ax.plot(mo, (-16.9e5 + net / 12 * mo) / 1e5, color=c, label=f"{lab} (payback {R['roi'][key]['payback_months']:.1f} mo)")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("Months after deployment"); ax.set_ylabel("Cumulative net cash flow (₹ lakh)")
ax.legend(fontsize=8, loc="upper left")
save(fig, "fig13.png")

# ---- Fig 14: source-side FDI detection vs magnitude
fm = np.load("results/fdi_magnitude.npy")
bins = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15])
mid = (bins[:-1] + bins[1:]) / 2 * 100
fig, ax = plt.subplots(figsize=(6, 3))
for phys_flag, c, lab in ((1.0, C1, "Physics-consistent (P, Q, S, I scaled)"), (0.0, C2, "Power-only manipulation")):
    sel = fm[:, 1] == phys_flag
    rate = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = sel & (fm[:, 0] >= lo) & (fm[:, 0] < hi)
        rate.append(100 * fm[s, 2].mean() if s.sum() else np.nan)
    ax.plot(mid, rate, "-o", ms=4, color=c, label=lab)
ax.set_xlabel("Under-reporting magnitude δ (%)"); ax.set_ylabel("Detection rate (%)")
ax.set_ylim(-3, 105); ax.legend(fontsize=8, loc="lower right")
save(fig, "fig14.png")

# ---- Fig 15: network degradation robustness
nl = R["robustness"]["network"]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
lx = [f"{d['loss']*100:g}%\n{d['bursts_per_day']:g}/day" for d in nl]
ax[0].plot(range(len(nl)), [d["acc"] * 100 for d in nl], "-o", color=C1, label="Accuracy")
ax[0].plot(range(len(nl)), [d["f1"] * 100 for d in nl], "-o", color=C4, label="Macro-F1 ×100")
ax[0].set_xticks(range(len(nl))); ax[0].set_xticklabels(lx, fontsize=7); ax[0].legend(fontsize=7)
ax[0].set_xlabel("Packet loss / EMI bursts per machine-day"); ax[0].set_ylabel("%")
ax[1].plot(range(len(nl)), [d["false_alarm"] * 100 for d in nl], "-o", color=C2)
ax[1].set_xticks(range(len(nl))); ax[1].set_xticklabels(lx, fontsize=7)
ax[1].set_xlabel("Packet loss / EMI bursts per machine-day"); ax[1].set_ylabel("False alarms on normal windows (%)")
fig.tight_layout()
save(fig, "fig15.png")
print("figures written")
