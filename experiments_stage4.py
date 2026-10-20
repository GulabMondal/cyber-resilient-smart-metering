"""Fourth stage: matched-conditions classifier comparison, server-side
verification cost, dataset export, trained-model export and checksums.

Runs after experiments_stage3.py and updates results/results.json.
"""
import gc
import hashlib
import hmac
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import sim.config as C
from sim.dataset import build
from sim.features import FEATURES, CLASSES
from sim import security as S
from experiments_common import smote_fit, tr_days, te_days, log

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
R = json.load(open("results/results.json"))
save = lambda: json.dump(R, open("results/results.json", "w"), indent=1, default=float)

D = build(C.SEED, verbose=False, with_store=False)
X, y, day, mach = D["X"], D["y"], D["day"], D["machine"]
tr, te = tr_days(day), te_days(day)

# ------------------------------------------------------------------ checksums
def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

R["checksums"] = dict(X=sha(X.astype(np.float64)), y=sha(y.astype(np.int8)),
                      machine=sha(mach.astype(np.int64)), n_windows=int(len(y)))
save()
log("checksums", R["checksums"])

# ------------------------------------------------------------------ dataset export
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(X, columns=FEATURES)
df.insert(0, "machine", [C.MACHINES[m].name for m in mach])
df.insert(1, "day", day + 1)
df.insert(2, "minute_of_day", np.tile(np.arange(1440), len(C.MACHINES) * C.DAYS))
df["label"] = y
df["label_name"] = [CLASSES[i] for i in y]
df["split"] = np.where(tr, "train", np.where(te, "test", "gap"))
df.to_csv("data/windows_features.csv.gz", index=False, float_format="%.6g", compression="gzip")
ev = []
for e in D["events"]:
    ev.append(dict(attack=S.A_NAMES[e["kind"]], machine=C.MACHINES[e["mi"]].name,
                   start_s=e["st"] * C.DT, end_s=e["en"] * C.DT,
                   start_day=e["st"] // C.SPD + 1,
                   **{k: v for k, v in e.items() if k not in ("kind", "mi", "st", "en", "delay_days")}))
pd.DataFrame(ev).to_csv("data/attack_events.csv", index=False)
log("exported", os.path.getsize("data/windows_features.csv.gz") // 1024, "KB")

# ------------------------------------------------------------------ trained model
rf_best = R["hyperparameters"]["RF"]
Xs, ys, _, _ = smote_fit(X[tr], y[tr], 30000)
rf = RandomForestClassifier(**rf_best, n_jobs=2, random_state=0).fit(Xs, ys)
pp = rf.predict_proba(X[te])
same = np.allclose(pp, np.load("results/rf_proba_chrono.npy"))
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest_chrono.joblib", compress=3)
R["model_export"] = dict(file="models/random_forest_chrono.joblib",
                         size_kb=os.path.getsize("models/random_forest_chrono.joblib") // 1024,
                         predictions_identical_to_stage1=bool(same),
                         test_accuracy=float(accuracy_score(y[te], pp.argmax(1))))
save()
log("model export", R["model_export"])
del Xs, ys; gc.collect()

# ------------------------------------------------------------------ server-side cost
key = os.urandom(32)
msg = b"HM-1|1760860000123|123456|" + b"9" * 120
n = 200000
t = time.perf_counter()
tag = hmac.new(key, msg, hashlib.sha256).digest()
for _ in range(n):
    hmac.compare_digest(hmac.new(key, msg, hashlib.sha256).digest(), tag)
t_hmac = (time.perf_counter() - t) / n
h = b"\x00" * 32
t = time.perf_counter()
for _ in range(n):
    h = hashlib.sha256(msg + tag + h).digest()
t_chain = (time.perf_counter() - t) / n
R["server_cost"] = dict(hmac_verify_us=t_hmac * 1e6, chain_append_us=t_chain * 1e6,
                        records_per_s=1 / (t_hmac + t_chain), python="CPython, single core",
                        message_bytes=len(msg))
save()
log("server cost", R["server_cost"])

# ------------------------------------------------------------------ matched-conditions comparison
# Identical conditions for all five models: same 40,000-window training subsample,
# same SMOTE target, same tuning fold and validation subsample, same test set.
log("matched-conditions comparison")
rs = np.random.default_rng(42)
idx_tr = rs.choice(np.flatnonzero(tr), 40000, replace=False)
trf = ((day >= 0) & (day < 19)) | ((day >= 30) & (day < 49))
vaf = ((day > 19) & (day <= 20)) | ((day > 49) & (day <= 50))
it = rs.choice(np.flatnonzero(trf), 15000, replace=False)
iv = rs.choice(np.flatnonzero(vaf), 8000, replace=False)
TARGET = 4000
sc = StandardScaler().fit(X[idx_tr])
Xn = sc.transform(X)

SEQ = 10
Xt = torch.tensor(Xn, dtype=torch.float32)
yt = torch.tensor(y.astype(np.int64))
offs = torch.arange(-SEQ + 1, 1)


def has_hist(ii):
    return np.array([i for i in ii if i - SEQ + 1 >= 0 and mach[i - SEQ + 1] == mach[i]])


class Net(nn.Module):
    def __init__(self, nf, hid):
        super().__init__(); self.l = nn.LSTM(nf, hid, batch_first=True); self.o = nn.Linear(hid, 6)
    def forward(self, x):
        h, _ = self.l(x); return self.o(h[:, -1])


def lstm_train(ii, hid, epochs=15):
    torch.manual_seed(0)
    net = Net(X.shape[1], hid); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    cnt = np.bincount(y[ii], minlength=6)
    w = torch.tensor(np.clip(cnt.max() / np.maximum(cnt, 1), 1, 50) ** 0.5, dtype=torch.float32)
    lf = nn.CrossEntropyLoss(weight=w)
    for ep in range(epochs):
        p = np.random.default_rng(ep).permutation(ii)
        for k in range(0, len(p), 256):
            j = torch.tensor(p[k:k + 256]); opt.zero_grad()
            loss = lf(net(Xt[j[:, None] + offs]), yt[j]); loss.backward(); opt.step()
    return net


def lstm_pred(net, ii):
    out = []
    with torch.no_grad():
        for k in range(0, len(ii), 4096):
            j = torch.tensor(ii[k:k + 4096]); out.append(net(Xt[j[:, None] + offs]).argmax(1).numpy())
    return np.concatenate(out)


def rules(Xa):
    f = {k: Xa[:, i] for i, k in enumerate(FEATURES)}
    th = R["thresholds"]
    cyber = (f["f_tag_fail"] > 0) | (f["f_fresh_fail"] > 0) | (f["f_seq_fail"] > 0) | (f["n_spoof"] > 0) | (f["gap_s"] > th["heartbeat_s"])
    integ = (f["phys_resid"] > th["phys_resid"]) | (f["feeder_resid30"] > th["feeder_resid30_h"])
    energy = (f["idle_frac"] > 0.5) | (f["sec_ratio"] > 1.12)
    health = (f["imb_max"] > 0.10) | (f["THD_mean"] > 0.12) | ((f["PF_mean"] < 0.82) & (f["PF_mean"] > 0.5) & (f["P_mean"] > 5))
    yp = np.zeros(len(cyber), np.int8)
    yp[energy] = 1; yp[health] = 2; yp[cyber] = 3; yp[integ & ~cyber] = 4; yp[(cyber | integ) & (energy | health)] = 5
    return yp


models = {
    "Random Forest": ([dict(n_estimators=n, max_depth=d) for n in (100, 300) for d in (10, 20, 40)],
                      lambda g: RandomForestClassifier(**g, n_jobs=2, random_state=0), False),
    "XGBoost": ([dict(n_estimators=n, max_depth=d, learning_rate=lr) for n in (200, 400) for d in (4, 8) for lr in (0.1, 0.3)],
                lambda g: XGBClassifier(**g, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2, random_state=0, verbosity=0), False),
    "SVM": ([dict(C=c, gamma=g) for c in (1, 10, 100) for g in ("scale", 0.05)],
            lambda g: SVC(kernel="rbf", **g), True),
}
res = {}
for name, (grid, make, scaled) in models.items():
    Xa = Xn if scaled else X
    best, bf = None, -1
    Xs_t, ys_t, _, _ = smote_fit(Xa[it], y[it], 1500)
    for g in grid:
        f = f1_score(y[iv], make(g).fit(Xs_t, ys_t).predict(Xa[iv]), average="macro")
        if f > bf:
            best, bf = g, f
    Xs, ys, _, _ = smote_fit(Xa[idx_tr], y[idx_tr], TARGET)
    t = time.time(); m = make(best).fit(Xs, ys); tt = time.time() - t
    yp = m.predict(Xa[te])
    res[name] = dict(params=best, val_f1=bf, acc=accuracy_score(y[te], yp), f1=f1_score(y[te], yp, average="macro"), train_s=tt)
    log(name, res[name])

# LSTM: same subsample (windows with 10-window history), same tuning fold
ia, itt, ivv, ib = has_hist(idx_tr), has_hist(it), has_hist(iv), has_hist(np.flatnonzero(te))
best, bf = None, -1
for hid in (32, 64):
    f = f1_score(y[ivv], lstm_pred(lstm_train(itt, hid), ivv), average="macro")
    if f > bf:
        best, bf = hid, f
t = time.time(); net = lstm_train(ia, best); tt = time.time() - t
pr = np.zeros(int(te.sum()), np.int64)
pos = {v: k for k, v in enumerate(np.flatnonzero(te))}
pl = lstm_pred(net, ib)
for k, i in enumerate(ib):
    pr[pos[i]] = pl[k]
res["LSTM"] = dict(params=dict(hidden=best, epochs=15), val_f1=bf, acc=accuracy_score(y[te], pr),
                   f1=f1_score(y[te], pr, average="macro"), train_s=tt)
log("LSTM", res["LSTM"])

# Isolation Forest + Algorithm 3 rules, same subsample
best, bf = None, -1
for cont in (0.1, 0.2, 0.3):
    m = IsolationForest(n_estimators=200, contamination=cont, random_state=0, n_jobs=2).fit(Xn[it])
    an = m.predict(Xn[iv]) == -1
    yp = np.where(an, rules(X[iv]), 0); yp = np.where(an & (yp == 0), 1, yp)
    f = f1_score(y[iv], yp, average="macro")
    if f > bf:
        best, bf = cont, f
t = time.time(); m = IsolationForest(n_estimators=200, contamination=best, random_state=0, n_jobs=2).fit(Xn[idx_tr]); tt = time.time() - t
an = m.predict(Xn[te]) == -1
yp = np.where(an, rules(X[te]), 0); yp = np.where(an & (yp == 0), 1, yp)
res["Isolation Forest"] = dict(params=dict(contamination=best), val_f1=bf, acc=accuracy_score(y[te], yp),
                               f1=f1_score(y[te], yp, average="macro"), train_s=tt)
log("IF", res["Isolation Forest"])
R["ml_matched"] = dict(results=res, protocol=dict(train_subsample=40000, smote_target=TARGET, tuning_train=15000,
                                                  tuning_val=8000, tuning_fold="train days 1-19 / 31-49, validate days 21 / 51",
                                                  lstm_epochs=15, seed=42))
save()
log("stage 4 done")
