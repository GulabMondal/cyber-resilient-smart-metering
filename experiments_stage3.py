"""Third stage: network robustness, seeds, per-unit LOMO and ROI.
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
# leave-one-machine-out with absolute features (stage 1, logged before the stage-1 process stopped)


# (c) independent seeds (full pipeline regenerated)
seeds = []
for sd in (11, 22, 33):
    Ds = build(sd, verbose=False, with_store=False)
    trs, tes = tr_days(Ds["day"]), te_days(Ds["day"])
    Xs, ys, _, _ = smote_fit(Ds["X"][trs], Ds["y"][trs], 30000)
    m = RandomForestClassifier(**rf_best, n_jobs=2, random_state=0).fit(Xs, ys)
    yp = m.predict(Ds["X"][tes])
    Es = energy_stats(Ds["plant"])
    seeds.append(dict(seed=sd, acc=accuracy_score(Ds["y"][tes], yp), f1=f1_score(Ds["y"][tes], yp, average="macro"),
                      energy_red=100 * (1 - Es["daily_kwh"][o].mean() / Es["daily_kwh"][b].mean())))
    del Ds, Es; gc.collect()
    log("  seed", seeds[-1])
rob["seeds"] = seeds
R["robustness"] = rob
save()

# ============================================================ E6 ROI
log("E6 ROI")
en = R["energy"]
kwh_day = en["base_mean"] - en["opt_mean"]
idle_day = en["idle_base"] - en["idle_opt"]
eff_day = kwh_day - idle_day
# Billing uses the highest 30-min demand of each 30-day period (one billing cycle each)
md_tot = en["md_base_max"] - en["md_opt_max"]
md_pf = en["md_base_max"] - en["md_cf_pf_only_max"]     # PF effect alone (no stagger)
roi = {}
for tariff in (7.5, 8.5, 9.5):
    for dem in (300.0, 400.0, 500.0):
        idle_s = idle_day * 365 * tariff
        eff_s = eff_day * 365 * tariff
        dem_total = md_tot * dem * 12
        pf_s = min(md_pf, md_tot) * dem * 12
        stag_s = dem_total - pf_s
        gross = idle_s + eff_s + dem_total
        net_ = gross - C.MAINT_FRAC * C.CAPEX_INR
        af = sum(1 / (1 + C.DISCOUNT) ** k for k in range(1, C.LIFE_YEARS + 1))
        roi[f"{tariff}_{int(dem)}"] = dict(tariff=tariff, demand=dem, idle=idle_s, efficiency=eff_s,
                                            pf_demand=pf_s, stagger_demand=stag_s, gross=gross, net=net_,
                                            payback_months=12 * C.CAPEX_INR / net_,
                                            npv=net_ * af - C.CAPEX_INR)
R["roi"] = roi
save()
log("stage 2 done")
