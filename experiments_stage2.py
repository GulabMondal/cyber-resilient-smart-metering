"""Second stage: network robustness, seeds, per-unit LOMO and ROI.
Runs after experiments.py (reads results/results.json); kept separate so that
only one 4-million-record dataset is held in memory at a time."""
import copy, json, math, time, gc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import sim.config as C
from sim.config import DT, SPD, DAYS, BASE_DAYS, MACHINES
from sim.dataset import build
from sim.features import FEATURES
from experiments_common import smote_fit, energy_stats, tr_days, te_days, log
R = json.load(open("results/results.json"))
save = lambda: json.dump(R, open("results/results.json", "w"), indent=1, default=float)
rf_best = R["hyperparameters"]["RF"]
b, o = slice(0, BASE_DAYS), slice(BASE_DAYS, DAYS)
rob = R["robustness"]
D = build(C.SEED, verbose=False, with_store=False)
X, y, day, mach = D["X"], D["y"], D["day"], D["machine"]
tr, te = tr_days(day), te_days(day)
del D; gc.collect()

# (a2) per-unit normalised features for cross-machine transfer
PU = ["P_mean", "P_std", "S_max", "Q_mean", "P_mean_prev3", "S_max_prev3"]
Xpu = X.copy()
rated = np.array([m.rated_kw for m in MACHINES])[mach]
for k in PU:
    j = FEATURES.index(k)
    Xpu[:, j] = np.where(Xpu[:, j] >= 0, Xpu[:, j] / rated, Xpu[:, j])
lomo_pu = {}
for mi in range(len(MACHINES)):
    a = tr & (mach != mi); bb = te & (mach == mi)
    Xs, ys, _, _ = smote_fit(Xpu[a], y[a], 30000)
    m = RandomForestClassifier(**rf_best, n_jobs=2, random_state=0).fit(Xs, ys)
    yp = m.predict(Xpu[bb])
    lomo_pu[MACHINES[mi].name] = dict(acc=accuracy_score(y[bb], yp), f1=f1_score(y[bb], yp, average="macro"))
rob["lomo_per_unit"] = lomo_pu
log("lomo per-unit", lomo_pu)
save()

# (b) degraded network: train on nominal data, test on noisier links
rf_nom = RandomForestClassifier(**rf_best, n_jobs=2, random_state=0).fit(*smote_fit(X[tr], y[tr], 30000)[:2])
del X; gc.collect()
net_lv = []
for loss, burst in [(0.005, 2.0), (0.02, 6.0), (0.05, 12.0), (0.10, 24.0)]:
    netp = copy.deepcopy(C.NET); netp.loss_rate = loss; netp.burst_per_day = burst
    Dn = build(C.SEED, net=netp, verbose=False, with_store=False)
    yp = rf_nom.predict(Dn["X"][te])
    yt_n = Dn["y"][te]
    del Dn; gc.collect()
    normal = yt_n == 0
    net_lv.append(dict(loss=loss, bursts_per_day=burst, acc=accuracy_score(yt_n, yp),
                       f1=f1_score(yt_n, yp, average="macro"),
                       false_alarm=float((yp[normal] != 0).mean())))
    log("  net", net_lv[-1])
rob["network"] = net_lv
save()

R["robustness"] = rob
save()
log("stage 2 done")
