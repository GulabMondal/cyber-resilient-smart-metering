"""Quick independent verification (about 5 minutes, 8 GB RAM).

    python verify.py

1. Regenerates the full dataset from the published seed and checks it against
   the SHA-256 checksums in results/results.json.
2. Reloads the published trained model and recomputes its test accuracy and
   macro-F1 on the chronological test split.
3. Recomputes the energy statistics and the financial analysis from the saved
   simulation series (independently of the pipeline code) and compares them
   with the published values.
Prints PASS/FAIL for every check and exits with status 1 on any failure.
"""
import hashlib
import json
import math
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import sim.config as C
from sim.dataset import build

R = json.load(open("results/results.json"))
E = R["energy"]
fails = 0


def check(name, got, want, tol=1e-6):
    global fails
    ok = (got == want) if isinstance(want, str) else abs(got - want) <= tol * max(1, abs(want))
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r}, published {want!r}")


sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

print("Regenerating dataset from seed", C.SEED, "...")
D = build(C.SEED, verbose=False, with_store=False)
X, y, day = D["X"], D["y"], D["day"]
check("records", int(C.DAYS * C.SPD * len(C.MACHINES)), R["dataset"]["total_records"])
check("feature matrix SHA-256", sha(X.astype(np.float64)), R["checksums"]["X"])
check("label vector SHA-256", sha(y.astype(np.int8)), R["checksums"]["y"])

te = ((day >= 22) & (day <= 29)) | ((day >= 52) & (day <= 59))
rf = joblib.load("models/random_forest_chrono.joblib")
yp = rf.predict(X[te])
check("RF chronological accuracy", accuracy_score(y[te], yp), R["ml_chrono"]["Random Forest"]["acc"])
check("RF chronological macro-F1", f1_score(y[te], yp, average="macro"), R["ml_chrono"]["Random Forest"]["f1"])

# independent recomputation from saved series
S = np.load("results/energy_series.npz")
dk, di, md, pf = S["daily_kwh"], S["idle_kwh"], S["md_kva"], S["pf_day"]
b, o = slice(0, 30), slice(30, 60)
check("daily energy reduction %", 100 * (1 - dk[o].mean() / dk[b].mean()), E["reduction_pct"])
check("idle energy reduction %", 100 * (1 - di[o].mean() / di[b].mean()), E["idle_reduction_pct"])
check("billed max demand, baseline (kVA)", md[b].max(), E["md_base_max"])
check("billed max demand, optimised (kVA)", md[o].max(), E["md_opt_max"])
check("power factor, optimised", pf[o].mean(), E["pf_opt"])
s1, s2 = dk[b].std(ddof=1), dk[o].std(ddof=1)
t = (dk[b].mean() - dk[o].mean()) / math.sqrt(s1 ** 2 / 30 + s2 ** 2 / 30)
check("Welch t", t, E["welch_t"])

# financial analysis from first principles (baseline: Rs 8.5/kWh, Rs 400/kVA/month)
tariff, dem = 8.5, 400.0
saving_kwh = (dk[b].mean() - dk[o].mean()) * 365
demand = (md[b].max() - md[o].max()) * dem * 12
gross = saving_kwh * tariff + demand
net = gross - C.MAINT_FRAC * C.CAPEX_INR
npv = sum(net / (1 + C.DISCOUNT) ** k for k in range(1, C.LIFE_YEARS + 1)) - C.CAPEX_INR
r = R["roi"]["8.5_400"]
check("gross annual saving (Rs)", gross, r["gross"])
check("payback (months)", 12 * C.CAPEX_INR / net, r["payback_months"])
check("5-year NPV (Rs)", npv, r["npv"])

# ---------------------------------------------------------------- internal consistency
cm = np.array(R["ml_chrono"]["Random Forest"]["cm"], float)
check("accuracy from confusion matrix", np.trace(cm) / cm.sum(), R["ml_chrono"]["Random Forest"]["acc"])
check("test windows in confusion matrix", float(cm.sum()), float(R["split"]["n_test"]))
for i, c in enumerate(R["ml_chrono"]["Random Forest"]["per_class"]):
    check(f"recall class {i} from confusion matrix", cm[i, i] / cm[i].sum(), c["r"])
    check(f"precision class {i} from confusion matrix", cm[i, i] / cm[:, i].sum(), c["p"])
check("record classes sum to total", float(sum(R["dataset"]["record_class_counts"].values())), float(R["dataset"]["total_records"]))
check("window classes sum to total", float(sum(R["dataset"]["window_class_counts"].values())), float(R["dataset"]["windows"]))
check("test class counts sum", float(sum(R["class_distribution"]["test"])), float(R["split"]["n_test"]))

# ---------------------------------------------------------------- variability quoted in Section 9.3
from sim.plant import simulate_plant, PROD
pl, _ = simulate_plant(np.random.default_rng(C.SEED))
cv1, cvd = [], []
for s in pl:
    P = s["P"].reshape(-1, 12).mean(1); st = s["state"].reshape(-1, 12)
    pr = (st == PROD).all(1)
    cv1.append(100 * P[pr].std() / P[pr].mean())
    d = (s["P"] * C.DT / 3600).reshape(C.DAYS, -1).sum(1)[:30]
    cvd.append(100 * d.std(ddof=1) / d.mean())
tot = sum((s["P"] * C.DT / 3600).reshape(C.DAYS, -1).sum(1) for s in pl)[:30]
print(f"INFO  one-minute CV of producing machines: {min(cv1):.1f}-{max(cv1):.1f}% (paper: 12-14%)")
print(f"INFO  daily CV per machine: {min(cvd):.1f}-{max(cvd):.1f}% (paper: 6-8%)")
print(f"INFO  daily CV of plant total: {100*tot.std(ddof=1)/tot.mean():.2f}% (paper: 3.1%)")
fails += not (12 <= min(cv1) and max(cv1) <= 14 and 6 <= min(cvd) and max(cvd) <= 8 and abs(100*tot.std(ddof=1)/tot.mean() - 3.1) < 0.05)

print("\nALL CHECKS PASSED" if fails == 0 else f"\n{fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
