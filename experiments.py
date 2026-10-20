"""Runs every experiment reported in the paper and writes results/*.json.

    sh run_all.sh   # runs experiments.py, experiments_stage2.py, experiments_stage3.py, figures.py (~1.5 h)

Sections refer to the revised manuscript.
"""
import copy
import json
import math
import os
import pickle
import time
import warnings

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             precision_recall_fscore_support, confusion_matrix,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import sim.config as C
from sim.config import DT, SPD, DAYS, BASE_DAYS, MACHINES
from sim.dataset import build
from sim.plant import simulate_plant, IDLE, OFF
from sim.features import FEATURES, CYBER_FEATS, PHYS_FEATS, ENERGY_FEATS, CONTEXT_FEATS, CLASSES
from sim import security as S

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)
R = {}
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def save():
    json.dump(R, open("results/results.json", "w"), indent=1, default=float)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ============================================================ data
log("building main dataset (seed", C.SEED, ")")
D = build(C.SEED, with_store=True)
X, y, day, mach, code_w = D["X"], D["y"], D["day"], D["machine"], D["code_w"]
plant, aux = D["plant"], D["aux"]
R["dataset"] = dict(records_per_machine=int(DAYS * SPD), machines=len(MACHINES), days=DAYS,
                    total_records=int(DAYS * SPD * len(MACHINES)), windows=int(len(y)),
                    window_class_counts={CLASSES[i]: int((y == i).sum()) for i in range(6)},
                    n_features=len(FEATURES))

# record-level ground-truth counts
rec = dict(normal=0, energy=0, health=0, cyber=0, integrity=0, combined=0)
for mi, s in enumerate(plant):
    c = D["codes"][mi]
    net = np.isin(c, S.NETWORK_ATTACKS) & (c != S.A_COMB)
    src = c == S.A_SRC
    comb = (c == S.A_COMB) | ((net | src) & (s["health"] | s["ineff"]))
    rec["combined"] += int(comb.sum())
    rec["cyber"] += int((net & ~comb).sum())
    rec["integrity"] += int((src & ~comb).sum())
    rest = ~(net | src | comb)
    rec["health"] += int((rest & s["health"]).sum())
    rec["energy"] += int((rest & ~s["health"] & s["ineff"]).sum())
    rec["normal"] += int((rest & ~s["health"] & ~s["ineff"]).sum())
R["dataset"]["record_class_counts"] = rec
ev_counts = {}
for ev in D["events"]:
    ev_counts[S.A_NAMES[ev["kind"]]] = ev_counts.get(S.A_NAMES[ev["kind"]], 0) + 1
ev_counts["Database tampering"] = len(D["db"])
R["dataset"]["attack_events"] = ev_counts
R["dataset"]["attack_events_total"] = int(sum(ev_counts.values()))
save()

# ============================================================ E1 energy
log("E1 energy results")


def energy_stats(pl):
    P = sum(s["P"] for s in pl); Sx = sum(s["S"] for s in pl)
    dk = (P * DT / 3600).reshape(DAYS, SPD).sum(1)
    idle = sum(np.where(s["state"] == IDLE, s["P"], 0) for s in pl)
    di = (idle * DT / 3600).reshape(DAYS, SPD).sum(1)
    c = np.cumsum(np.r_[0, Sx]); w = 360
    roll = (c[w:] - c[:-w]) / w
    md = np.array([roll[d * SPD:(d + 1) * SPD - w].max() for d in range(DAYS)])
    inst = Sx.reshape(DAYS, SPD).max(1)
    on = [s["state"] != OFF for s in pl]
    pf_day = np.array([sum(s["P"][d*SPD:(d+1)*SPD][o[d*SPD:(d+1)*SPD]].sum() for s, o in zip(pl, on)) /
                       sum(s["S"][d*SPD:(d+1)*SPD][o[d*SPD:(d+1)*SPD]].sum() for s, o in zip(pl, on))
                       for d in range(DAYS)])
    return dict(daily_kwh=dk, idle_kwh=di, md_kva=md, inst_peak=inst, pf_day=pf_day, S=Sx)


E = energy_stats(plant)
b, o = slice(0, BASE_DAYS), slice(BASE_DAYS, DAYS)
dk = E["daily_kwh"]
t, p = stats.ttest_ind(dk[b], dk[o], equal_var=False)
s1, s2 = dk[b].std(ddof=1), dk[o].std(ddof=1)
dfw = (s1**2/30 + s2**2/30)**2 / ((s1**2/30)**2/29 + (s2**2/30)**2/29)
d_c = (dk[b].mean() - dk[o].mean()) / math.sqrt((s1**2 + s2**2) / 2)
mw = stats.mannwhitneyu(dk[b], dk[o], alternative="greater")
# bootstrap CI of the percentage reduction
rng = np.random.default_rng(1)
boots = [1 - rng.choice(dk[o], 30).mean() / rng.choice(dk[b], 30).mean() for _ in range(10000)]
R["energy"] = dict(
    base_mean=dk[b].mean(), base_sd=s1, opt_mean=dk[o].mean(), opt_sd=s2,
    reduction_pct=100 * (1 - dk[o].mean() / dk[b].mean()),
    reduction_ci95=[100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)],
    welch_t=t, welch_df=dfw, welch_p=p, cohens_d=d_c, mannwhitney_p=mw.pvalue,
    idle_base=E["idle_kwh"][b].mean(), idle_opt=E["idle_kwh"][o].mean(),
    idle_reduction_pct=100 * (1 - E["idle_kwh"][o].mean() / E["idle_kwh"][b].mean()),
    md_base=E["md_kva"][b].mean(), md_opt=E["md_kva"][o].mean(),
    md_base_max=E["md_kva"][b].max(), md_opt_max=E["md_kva"][o].max(),
    md_reduction_pct=100 * (1 - E["md_kva"][o].mean() / E["md_kva"][b].mean()),
    inst_peak_base=E["inst_peak"][b].mean(), inst_peak_opt=E["inst_peak"][o].mean(),
    pf_base=E["pf_day"][b].mean(), pf_opt=E["pf_day"][o].mean())
pm = []
for mi, s in enumerate(plant):
    e = (s["P"] * DT / 3600).reshape(DAYS, SPD); oo = s["out_kg"].reshape(DAYS, SPD)
    pm.append(dict(name=MACHINES[mi].name, base_kwh=e[b].sum(), opt_kwh=e[o].sum(),
                   base_sec=e[b].sum() / oo[b].sum(), opt_sec=e[o].sum() / oo[o].sum()))
    pm[-1]["sec_impr_pct"] = 100 * (1 - pm[-1]["opt_sec"] / pm[-1]["base_sec"])
R["energy"]["per_machine"] = pm

# counterfactuals for maximum-demand decomposition
cf = {}
for name, mod in [("pf_only", dict(stagger_min=0.0)), ("stagger_only", dict(pf_corrected=0.0))]:
    saved = copy.deepcopy(C.PLANT)
    for k, v in mod.items():
        setattr(C.PLANT, k, v)
    pl2, _ = simulate_plant(np.random.default_rng(C.SEED))
    for k in mod:
        setattr(C.PLANT, k, getattr(saved, k))
    _m = energy_stats(pl2)["md_kva"]
    cf[name] = _m[o].mean()
    cf[name + "_max"] = _m[o].max()
R["energy"]["md_cf_pf_only"] = cf["pf_only"]
R["energy"]["md_cf_stagger_only"] = cf["stagger_only"]
# billed maximum demand = highest 30-min demand in the 30-day billing period
R["energy"]["md_cf_pf_only_max"] = cf["pf_only_max"]
R["energy"]["md_cf_stagger_only_max"] = cf["stagger_only_max"]
np.savez("results/energy_series.npz", daily_kwh=dk, idle_kwh=E["idle_kwh"], md_kva=E["md_kva"],
         pf_day=E["pf_day"],
         S_base_day=E["S"][(5) * SPD:(6) * SPD], S_opt_day=E["S"][(35) * SPD:(36) * SPD])
save()
log("energy:", {k: round(v, 3) for k, v in R["energy"].items() if isinstance(v, float)})

# ============================================================ E2 detection
log("E2 attack detection")
NET = C.NET
tr_days = lambda d: ((d >= 0) & (d <= 20)) | ((d >= 30) & (d <= 50))
te_days = lambda d: ((d >= 22) & (d <= 29)) | ((d >= 52) & (d <= 59))

# feeder CUSUM threshold chosen on TRAINING days with no source-side attack active
fc = D["feeder_resid30"]
wday = np.arange(len(fc)) // 1440
src_any = np.zeros(len(fc), bool)
for ev in D["events"]:
    if ev["kind"] in (S.A_SRC, S.A_POISON):
        src_any[ev["st"] // 12: ev["en"] // 12 + 1] = True
clean_tr = tr_days(wday) & ~src_any
H_CUSUM = float(np.quantile(fc[clean_tr], 0.995))
PERSIST = 20     # windows (minutes) the statistic must stay above H
PHYS_TH = 0.02
R["thresholds"] = dict(feeder_resid30_h=H_CUSUM, feeder_persist_min=PERSIST, phys_resid=PHYS_TH,
                       freshness_s=NET.freshness_s, heartbeat_s=NET.heartbeat_tol_s)

slot_alert = []
for mi, rx in enumerate(D["rxs"]):
    pres = rx["present"]
    det = pres & (~rx["tag_ok"] | ~rx["fresh"] | rx["seq_fail"])
    if len(rx["spoof_slot"]):
        det[rx["spoof_slot"]] = True
    # heartbeat gap
    acc = rx["accepted"]
    gap = np.zeros(len(acc)); last = -1
    for i in range(len(acc)):
        if acc[i]:
            last = i
        gap[i] = (i - last) * DT if last >= 0 else 0
    avail = gap > NET.heartbeat_tol_s
    v = rx["rx_vals"]
    Iavg = (v["I1"] + v["I2"] + v["I3"]) / 3
    phys = np.abs(v["P"] - np.sqrt(3) * v["V"] * Iavg * v["PF"] / 1000) / np.maximum(v["P"], 0.5)
    phys_flag = acc & (np.nan_to_num(phys) > PHYS_TH)
    slot_alert.append(dict(crypto=det, avail=avail, phys=phys_flag))
_c = np.convolve((fc > H_CUSUM).astype(int), np.ones(PERSIST, int), "full")[:len(fc)]
feeder_alarm = _c >= PERSIST

det_rows = {}
fdi_mag = []
grace = 12   # one minute after the event
for ev in D["events"]:
    k, mi, st, en = ev["kind"], ev["mi"], ev["st"], ev["en"]
    a = slot_alert[mi]
    sl = slice(st, min(en + grace, len(a["crypto"])))
    if k in (S.A_MITM, S.A_REPLAY, S.A_SPOOF, S.A_TS, S.A_DELAY, S.A_COMB):
        hit = a["crypto"][sl].any()
    elif k == S.A_DOS:
        hit = a["avail"][sl].any()
    elif k == S.A_SRC:
        hit = a["phys"][sl].any() or feeder_alarm[st // 12: en // 12 + 1].any()
        hidden = ev["delta"] * plant[mi]["P"][st:en].mean()
        fdi_mag.append((ev["delta"], ev["physical"], bool(hit), en - st, hidden))
    elif k == S.A_POISON:
        w = feeder_alarm[st // 12: en // 12 + 1]
        hit = w.any()
        ev["delay_days"] = float(np.argmax(w) / 1440) if hit else None
    name = S.A_NAMES[k]
    r = det_rows.setdefault(name, dict(events=0, detected=0))
    r["events"] += 1; r["detected"] += int(hit)
    if k == S.A_POISON:
        r.setdefault("delays", []).append(ev.get("delay_days"))
db = D["db"]
det_rows["Database tampering (chain only)"] = dict(events=len(db), detected=int(sum(e["chain"] for e in db)))
det_rows["Database tampering (chain + tag audit)"] = dict(events=len(db), detected=int(sum(e["chain_or_tag"] for e in db)))

# false positive rates: fraction of attack-free, test-period windows raising each alert type
attack_free = np.ones(len(y), bool)
attack_free &= (code_w == 0)
fp = {}
for nm, key in [("crypto", "crypto"), ("avail", "avail"), ("phys", "phys")]:
    flags = np.concatenate([a[key][: (len(a[key]) // 12) * 12].reshape(-1, 12).any(1) for a in slot_alert])
    fp[nm] = float(flags[attack_free & te_days(day)].mean())
fp["feeder"] = float(feeder_alarm[te_days(wday) & ~src_any].mean())
_st = np.flatnonzero(feeder_alarm[1:] & ~feeder_alarm[:-1]) + 1
_cl = te_days(wday) & ~src_any
fp["feeder_false_episodes_per_day"] = float(sum(_cl[s] for s in _st) / (_cl.sum() / 1440))
for name, r in det_rows.items():
    r["rate_pct"] = 100 * r["detected"] / r["events"]
R["detection"] = dict(rows=det_rows, fpr_window=fp)
fdi_mag = np.array(fdi_mag, dtype=float)
np.save("results/fdi_magnitude.npy", fdi_mag)
save()
log("detection:", {k: round(v["rate_pct"], 2) for k, v in det_rows.items()}, fp)

# ============================================================ E3 ML
log("E3 machine learning")
tr = tr_days(day); te = te_days(day)
n_te = int(te.sum())
R["split"] = dict(train_days="0-20 and 30-50", gap_days="21 and 51", test_days="22-29 and 52-59",
                  n_train=int(tr.sum()), n_test=n_te)


def smote_fit(Xa, ya, target=30000, seed=0):
    cnt = np.bincount(ya, minlength=6)
    strat = {c: max(int(cnt[c]), target) for c in range(6) if 0 < cnt[c] < target}
    if not strat:
        return Xa, ya, cnt, cnt
    Xs, ys = SMOTE(sampling_strategy=strat, random_state=seed, k_neighbors=5).fit_resample(Xa, ya)
    return Xs, ys, cnt, np.bincount(ys, minlength=6)


def evaluate(yt, yp, proba=None):
    acc = accuracy_score(yt, yp)
    lo, hi = wilson(int((yt == yp).sum()), len(yt))
    out = dict(acc=acc, acc_ci=[lo, hi], f1=f1_score(yt, yp, average="macro"),
               prec=precision_score(yt, yp, average="macro", zero_division=0))
    pr, rc, f, sup = precision_recall_fscore_support(yt, yp, labels=range(6), zero_division=0)
    out["per_class"] = [dict(cls=CLASSES[i], p=pr[i], r=rc[i], f1=f[i], n=int(sup[i])) for i in range(6)]
    out["cm"] = confusion_matrix(yt, yp, labels=range(6)).tolist()
    if proba is not None:
        out["auc_ovr"] = [roc_auc_score((yt == i).astype(int), proba[:, i]) for i in range(6)]
        out["auc_macro"] = float(np.mean(out["auc_ovr"]))
    return out


# --- forward-chaining CV inside the training period for tuning (RF, XGB) ---
def fc_folds(d):
    # three folds per period: train on days < k, validate on next 4 days (gap 1 day)
    folds = []
    for k in (9, 14, 19):
        trf = ((d >= 0) & (d < k)) | ((d >= 30) & (d < 30 + k))
        vaf = ((d > k) & (d <= k + 4)) | ((d > 30 + k) & (d <= 30 + k + 4))
        folds.append((trf, vaf))
    return folds


sub = np.random.default_rng(3).random(len(y)) < 0.35      # tuning subsample
folds = fc_folds(day)
R["tuning"] = {}


def tune(make, grid, name):
    best, res = None, []
    for g in grid:
        scores = []
        for trf, vaf in folds:
            a, bb = trf & sub, vaf & sub
            Xs, ys, _, _ = smote_fit(X[a], y[a], target=10000)
            mdl = make(g).fit(Xs, ys)
            scores.append(f1_score(y[bb], mdl.predict(X[bb]), average="macro"))
        res.append(dict(params=g, f1=float(np.mean(scores)), sd=float(np.std(scores))))
        if best is None or res[-1]["f1"] > best[1]:
            best = (g, res[-1]["f1"])
        log(f"  tune {name} {g} -> {res[-1]['f1']:.4f}")
    R["tuning"][name] = res
    return best[0]


rf_grid = [dict(n_estimators=n, max_depth=dp, min_samples_leaf=1) for n in (100, 150, 300) for dp in (10, 20, 40)]
rf_best = tune(lambda g: RandomForestClassifier(**g, n_jobs=2, random_state=0), rf_grid, "RF")
xgb_grid = [dict(n_estimators=n, max_depth=dp, learning_rate=lr, subsample=0.8, colsample_bytree=0.8)
            for n in (200, 400) for dp in (4, 8) for lr in (0.1, 0.3)]
xgb_best = tune(lambda g: XGBClassifier(**g, tree_method="hist", n_jobs=2, random_state=0, verbosity=0),
                xgb_grid, "XGBoost")
save()

# --- class distribution and SMOTE on the full training set ---
Xtr_s, ytr_s, cnt_before, cnt_after = smote_fit(X[tr], y[tr], target=30000)
R["class_distribution"] = dict(train_before=cnt_before.tolist(), train_after=cnt_after.tolist(),
                               test=np.bincount(y[te], minlength=6).tolist())

models = {}
res_chrono, res_random, timing = {}, {}, {}


def run_model(name, fit_pred, split):
    t = time.time()
    yp, proba, extra = fit_pred(split)
    el = time.time() - t
    return yp, proba, el, extra


# random 80/20 split (as in v1) for comparison
rnd = np.random.default_rng(7).random(len(y)) < 0.8
rtr, rte = rnd, ~rnd


def rf_fp(split):
    a, bb = split
    Xs, ys, _, _ = smote_fit(X[a], y[a], 30000)
    m = RandomForestClassifier(**rf_best, n_jobs=2, random_state=0)
    t = time.time(); m.fit(Xs, ys); tt = time.time() - t
    t = time.time(); pp = m.predict_proba(X[bb]); ti = (time.time() - t) / bb.sum()
    return pp.argmax(1), pp, dict(train_s=tt, infer_ms=ti * 1000, model=m)


def xgb_fp(split):
    a, bb = split
    Xs, ys, _, _ = smote_fit(X[a], y[a], 30000)
    m = XGBClassifier(**xgb_best, tree_method="hist", n_jobs=2, random_state=0, verbosity=0)
    t = time.time(); m.fit(Xs, ys); tt = time.time() - t
    t = time.time(); pp = m.predict_proba(X[bb]); ti = (time.time() - t) / bb.sum()
    return pp.argmax(1), pp, dict(train_s=tt, infer_ms=ti * 1000)


SVM_N = 40000
svm_grid = [dict(C=c, gamma=g) for c in (1, 10, 100) for g in ("scale", 0.05)]


def svm_fp(split):
    a, bb = split
    rs = np.random.default_rng(5)
    ia = np.flatnonzero(a)
    ia = rs.choice(ia, min(SVM_N, len(ia)), replace=False)
    sc = StandardScaler().fit(X[ia])
    Xs, ys, _, _ = smote_fit(sc.transform(X[ia]), y[ia], 4000)
    # tune on last forward-chaining fold (inside the subsample)
    best, bf = None, -1
    trf, vaf = folds[-1]
    it = np.flatnonzero(trf & a); iv = np.flatnonzero(vaf & a)
    it = rs.choice(it, min(15000, len(it)), replace=False); iv = rs.choice(iv, min(8000, len(iv)), replace=False)
    for g in svm_grid:
        m = SVC(kernel="rbf", **g).fit(sc.transform(X[it]), y[it])
        f = f1_score(y[iv], m.predict(sc.transform(X[iv])), average="macro")
        if f > bf:
            best, bf = g, f
    R["tuning"]["SVM"] = dict(best=best, f1=bf, grid=svm_grid, train_subsample=SVM_N)
    m = SVC(kernel="rbf", **best)
    t = time.time(); m.fit(Xs, ys); tt = time.time() - t
    t = time.time(); yp = m.predict(sc.transform(X[bb])); ti = (time.time() - t) / bb.sum()
    return yp, None, dict(train_s=tt, infer_ms=ti * 1000, params=best)


def rules_from_flags(Xa):
    """Algorithm 1 + 2 + 3 deterministic mapping used by IF and the ablation."""
    f = {k: Xa[:, i] for i, k in enumerate(FEATURES)}
    cyber = (f["f_tag_fail"] > 0) | (f["f_fresh_fail"] > 0) | (f["f_seq_fail"] > 0) | \
            (f["n_spoof"] > 0) | (f["gap_s"] > C.NET.heartbeat_tol_s)
    integ = (f["phys_resid"] > PHYS_TH) | (f["feeder_resid30"] > H_CUSUM)
    energy = (f["idle_frac"] > 0.5) | (f["sec_ratio"] > 1.12)
    health = (f["imb_max"] > 0.10) | (f["THD_mean"] > 0.12) | ((f["PF_mean"] < 0.82) & (f["PF_mean"] > 0.5) & (f["P_mean"] > 5))
    return cyber, integ, energy, health


def alg3(cyber, integ, energy, health):
    yp = np.zeros(len(cyber), np.int8)
    yp[energy] = 1
    yp[health] = 2
    yp[cyber] = 3
    yp[integ & ~cyber] = 4
    yp[(cyber | integ) & (energy | health)] = 5
    return yp


def if_fp(split):
    a, bb = split
    sc = StandardScaler().fit(X[a])
    best, bf = None, -1
    trf, vaf = folds[-1]
    for cont in (0.1, 0.2, 0.3):
        m = IsolationForest(n_estimators=200, contamination=cont, random_state=0, n_jobs=2).fit(sc.transform(X[trf & sub]))
        an = m.predict(sc.transform(X[vaf])) == -1
        c, i, e, h = rules_from_flags(X[vaf])
        yp = np.where(an, alg3(c, i, e, h), 0)
        # anomalies not explained by any rule -> energy inefficiency (most frequent anomaly)
        yp = np.where(an & (yp == 0), 1, yp)
        f = f1_score(y[vaf], yp, average="macro")
        if f > bf:
            best, bf = cont, f
    R["tuning"]["IsolationForest"] = dict(best_contamination=best, f1=bf)
    m = IsolationForest(n_estimators=200, contamination=best, random_state=0, n_jobs=2)
    t = time.time(); m.fit(sc.transform(X[a])); tt = time.time() - t
    t = time.time()
    an = m.predict(sc.transform(X[bb])) == -1
    c, i, e, h = rules_from_flags(X[bb])
    yp = np.where(an, alg3(c, i, e, h), 0)
    yp = np.where(an & (yp == 0), 1, yp)
    ti = (time.time() - t) / bb.sum()
    return yp, None, dict(train_s=tt, infer_ms=ti * 1000, contamination=best)


# ---- LSTM (sequence of the last 10 windows of the same machine) ----
import torch
import torch.nn as nn
torch.set_num_threads(2)
SEQ = 10


def make_seq_index(mask):
    idx = np.flatnonzero(mask)
    ok = []
    for i in idx:
        if i - SEQ + 1 >= 0 and mach[i - SEQ + 1] == mach[i]:
            ok.append(i)
    return np.array(ok)


class LSTMNet(nn.Module):
    def __init__(self, nf, hid):
        super().__init__()
        self.l = nn.LSTM(nf, hid, batch_first=True)
        self.o = nn.Linear(hid, 6)

    def forward(self, x):
        h, _ = self.l(x)
        return self.o(h[:, -1])


def lstm_fp(split, hid_grid=(32, 64), epochs=6):
    a, bb = split
    sc = StandardScaler().fit(X[a])
    Xn = torch.tensor(sc.transform(X), dtype=torch.float32)
    ia, ib = make_seq_index(a), make_seq_index(bb)
    w = torch.tensor(np.clip(np.bincount(y[ia], minlength=6).max() / np.maximum(np.bincount(y[ia], minlength=6), 1), 1, 50) ** 0.5, dtype=torch.float32)
    yt = torch.tensor(y.astype(np.int64))
    offs = torch.arange(-SEQ + 1, 1)

    def batches(ii, bs, shuffle):
        ii = np.random.default_rng(0).permutation(ii) if shuffle else ii
        for k in range(0, len(ii), bs):
            j = torch.tensor(ii[k:k + bs])
            yield Xn[j[:, None] + offs], yt[j]

    def train(ii, hid):
        torch.manual_seed(0)
        net = LSTMNet(X.shape[1], hid)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lf = nn.CrossEntropyLoss(weight=w)
        for ep in range(epochs):
            net.train()
            for xb, yb in batches(ii, 256, True):
                opt.zero_grad(); loss = lf(net(xb), yb); loss.backward(); opt.step()
        return net

    def predict(net, ii):
        net.eval(); out = []
        with torch.no_grad():
            for xb, _ in batches(ii, 2048, False):
                out.append(torch.softmax(net(xb), 1).numpy())
        return np.concatenate(out)

    trf, vaf = folds[-1]
    best, bf = None, -1
    for hid in hid_grid:
        net = train(make_seq_index(trf & sub & a), hid)
        iv = make_seq_index(vaf)
        f = f1_score(y[iv], predict(net, iv).argmax(1), average="macro")
        log(f"  tune LSTM hidden={hid} -> {f:.4f}")
        if f > bf:
            best, bf = hid, f
    R["tuning"]["LSTM"] = dict(best_hidden=best, f1=bf, window=SEQ, epochs=epochs, lr=1e-3, batch=256)
    t = time.time(); net = train(ia, best); tt = time.time() - t
    t = time.time(); pp = predict(net, ib); ti = (time.time() - t) / len(ib)
    # windows without full history (first 9 minutes of the test block) -> predicted normal
    full = np.zeros((bb.sum(), 6)); full[:, 0] = 1
    pos = {v: k for k, v in enumerate(np.flatnonzero(bb))}
    for k, i in enumerate(ib):
        full[pos[i]] = pp[k]
    return full.argmax(1), full, dict(train_s=tt, infer_ms=ti * 1000, hidden=best)


MODELS = [("Random Forest", rf_fp), ("XGBoost", xgb_fp), ("LSTM", lstm_fp), ("SVM", svm_fp), ("Isolation Forest", if_fp)]
for name, fn in MODELS:
    for tag, split, store in (("chrono", (tr, te), res_chrono), ("random", (rtr, rte), res_random)):
        log(f"model {name} [{tag}]")
        yp, pp, el, extra = run_model(name, fn, split)
        yt = y[split[1]]
        store[name] = evaluate(yt, yp, pp)
        store[name].update({k: v for k, v in extra.items() if k != "model"})
        if name == "Random Forest" and tag == "chrono":
            rf_model = extra["model"]
            R["feature_importance"] = sorted(
                [(FEATURES[i], float(v)) for i, v in enumerate(rf_model.feature_importances_)],
                key=lambda t: -t[1])
            np.save("results/rf_proba_chrono.npy", pp)
            np.save("results/y_test_chrono.npy", yt)
        log(f"   acc={store[name]['acc']:.4f} f1={store[name]['f1']:.4f}")
        R["ml_chrono"], R["ml_random"] = res_chrono, res_random
        save()

R["hyperparameters"] = dict(RF=rf_best, XGBoost=xgb_best)
save()

# blocked CV accuracy for the selected RF (forward-chaining folds, full data in folds)
cv = []
for trf, vaf in folds:
    Xs, ys, _, _ = smote_fit(X[trf], y[trf], 30000)
    m = RandomForestClassifier(**rf_best, n_jobs=2, random_state=0).fit(Xs, ys)
    cv.append(accuracy_score(y[vaf], m.predict(X[vaf])))
R["rf_blocked_cv"] = dict(acc=cv, mean=float(np.mean(cv)), sd=float(np.std(cv)))
save()

# ============================================================ E4 ablation
log("E4 ablation")
abl = {}
c, i_, e, h = rules_from_flags(X[te])
yt = y[te]
abl["Energy rules only (Alg. 1)"] = evaluate(yt, alg3(np.zeros_like(c), np.zeros_like(c), e, h))
# rules + Z-score: per-machine z on P_mean, S_max, THD_mean fitted on training windows
zf = np.zeros(te.sum(), bool)
for k in ("P_mean", "S_max", "THD_mean", "imb_max"):
    j = FEATURES.index(k)
    for mi in range(len(MACHINES)):
        mt = tr & (mach == mi) & (X[:, FEATURES.index("producing")] == 1)
        mu, sd = X[mt, j].mean(), X[mt, j].std() + 1e-9
        sel = mach[te] == mi
        zf[sel] |= np.abs((X[te][sel, j] - mu) / sd) > 3
abl["Rules + Z-score"] = evaluate(yt, alg3(np.zeros_like(c), np.zeros_like(c), e, h | zf))
abl["Cyber validation only (Alg. 2 + 3)"] = evaluate(yt, alg3(c, np.zeros_like(c), np.zeros_like(c), np.zeros_like(c)))
abl["Rules + crypto + physics (no ML)"] = evaluate(yt, alg3(c, i_, e, h | zf))


def rf_subset(cols):
    j = [FEATURES.index(k) for k in cols]
    Xs, ys, _, _ = smote_fit(X[tr][:, j], y[tr], 30000)
    m = RandomForestClassifier(**rf_best, n_jobs=2, random_state=0).fit(Xs, ys)
    return evaluate(yt, m.predict(X[te][:, j]))


abl["ML, energy features only (no cyber/physics)"] = rf_subset(ENERGY_FEATS + CONTEXT_FEATS)
abl["ML, energy + cyber (no physics/feeder)"] = rf_subset(ENERGY_FEATS + CONTEXT_FEATS + CYBER_FEATS)
abl["Full hybrid framework"] = res_chrono["Random Forest"]
R["ablation"] = {k: dict(acc=v["acc"], f1=v["f1"]) for k, v in abl.items()}
save()
log("ablation", R["ablation"])

# ============================================================ E5 robustness
log("E5 robustness")
rob = {}
# (a) leave-one-machine-out (train 3 machines/train days, test held-out machine/test days)
lomo = {}
for mi in range(len(MACHINES)):
    a = tr & (mach != mi); bb = te & (mach == mi)
    Xs, ys, _, _ = smote_fit(X[a], y[a], 30000)
    m = RandomForestClassifier(**rf_best, n_jobs=2, random_state=0).fit(Xs, ys)
    yp = m.predict(X[bb])
    lomo[MACHINES[mi].name] = dict(acc=accuracy_score(y[bb], yp), f1=f1_score(y[bb], yp, average="macro"))
rob["lomo"] = lomo
log("lomo", lomo)
save()

R["robustness"] = rob
save()
log("stage 1 done")
