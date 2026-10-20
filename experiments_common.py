import math, time
import numpy as np
from imblearn.over_sampling import SMOTE
import sim.config as C
from sim.config import DT, SPD, DAYS
from sim.plant import IDLE, OFF
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
tr_days = lambda d: ((d >= 0) & (d <= 20)) | ((d >= 30) & (d <= 50))
te_days = lambda d: ((d >= 22) & (d <= 29)) | ((d >= 52) & (d <= 59))
def smote_fit(Xa, ya, target=30000, seed=0):
    cnt = np.bincount(ya, minlength=6)
    strat = {c: max(int(cnt[c]), target) for c in range(6) if 0 < cnt[c] < target}
    if not strat:
        return Xa, ya, cnt, cnt
    Xs, ys = SMOTE(sampling_strategy=strat, random_state=seed, k_neighbors=5).fit_resample(Xa, ya)
    return Xs, ys, cnt, np.bincount(ys, minlength=6)


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


