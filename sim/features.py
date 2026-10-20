"""1-minute feature windows, plant-level feeder balance and ground-truth labels."""
import numpy as np
from .config import DT, SPD, MACHINES, PLANT, NET
from .security import (A_NONE, A_SRC, A_POISON, A_COMB, NETWORK_ATTACKS)

W = 12                       # samples per 1-minute window
CLASSES = ["Normal operation", "Energy inefficiency", "Electrical health event",
           "Cyber-attack (network)", "Source-side integrity attack",
           "Combined cyber-energy anomaly"]
ENERGY_FEATS = ["P_mean", "P_std", "S_max", "Q_mean", "PF_mean", "V_mean", "THD_mean",
                "imb_max", "sec_ratio", "idle_frac", "producing"]
PHYS_FEATS = ["phys_resid", "feeder_resid", "feeder_resid30"]
CYBER_FEATS = ["n_rx", "n_acc", "f_tag_fail", "f_seq_fail", "f_fresh_fail",
               "max_age", "gap_s", "n_spoof"]
CONTEXT_BASE = ["P_mean", "S_max", "imb_max", "THD_mean", "PF_mean", "idle_frac", "sec_ratio"]
CONTEXT_FEATS = [f"{k}_prev3" for k in CONTEXT_BASE]   # causal: previous 3 windows
FEATURES = ENERGY_FEATS + PHYS_FEATS + CYBER_FEATS + CONTEXT_FEATS
SQ3 = np.sqrt(3.0)


def _win(x, fn):
    n = len(x) // W
    return fn(x[: n * W].reshape(n, W), axis=1)


def _nanwin(x, fn, fill):
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = _win(x, fn)
    return np.where(np.isnan(r), fill, r)


def feeder_balance(plant, rxs, aux, rng):
    """Plant incomer meter (signed, trusted) vs. sum of accepted machine reports.

    The residual is only evaluated over slots in which every machine delivered
    an authentic, fresh record (otherwise imputation error would dominate).
    Statistic: 30-min mean residual minus the median of the same quantity at
    the same time of day over the previous 7 days (tracks the daily profile
    and slow drift of the unmetered auxiliary load).
    """
    import pandas as pd
    true_total = sum(s["P"] for s in plant) + aux
    feeder = true_total * (1 + 0.002 * rng.standard_normal(len(aux)))
    rep = np.zeros_like(feeder)
    complete = np.ones(len(aux), bool)
    for rx in rxs:
        complete &= rx["accepted"]
        rep += np.nan_to_num(rx["rx_vals"]["P"], nan=0.0)
    resid = np.where(complete, feeder - rep - PLANT.aux_load_kw, np.nan)
    rs = pd.Series(resid)
    n = len(resid) // W
    r_w = rs.values[: n * W].reshape(n, W)
    cnt = np.sum(~np.isnan(r_w), 1)
    with np.errstate(all="ignore"):
        r_w = np.where(cnt >= W // 2, np.nanmean(r_w, 1), np.nan)
    f_w = _win(feeder, np.mean)
    rel = np.nan_to_num(r_w / np.maximum(f_w, 1.0), nan=0.0)
    s = pd.Series(r_w)
    r30 = s.rolling(30, min_periods=10).mean()
    # seasonal baseline: same time of day over the previous 7 days (30-min smoothed)
    r30v = r30.values
    lagged = np.full((7, len(r30v)), np.nan)
    for k in range(1, 8):
        lagged[k - 1, k * 1440:] = r30v[:-k * 1440]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = np.nanmedian(lagged, 0)
    stat = np.nan_to_num(r30v - base, nan=0.0)
    return rel, stat


def machine_windows(mi, sim, rx, mes_out, feeder_rel, feeder_resid30, codes):
    m = MACHINES[mi]
    v = rx["rx_vals"]; acc = rx["accepted"]; pres = rx["present"]
    nanif = lambda x: np.where(acc, x, np.nan)
    P = nanif(v["P"]); S = nanif(v["S"]); Q = nanif(v["Q"]); PF = nanif(v["PF"])
    V = nanif(v["V"]); THD = nanif(v["THD"])
    I = np.stack([nanif(v["I1"]), nanif(v["I2"]), nanif(v["I3"])], 1)
    Iavg = I.mean(1)
    imb = np.abs(I - Iavg[:, None]).max(1) / np.maximum(Iavg, 1e-6)
    phys = np.abs(P - SQ3 * V * Iavg * PF / 1000) / np.maximum(P, 0.5)

    f = {}
    f["P_mean"] = _nanwin(P, np.nanmean, -1)
    f["P_std"] = _nanwin(P, np.nanstd, -1)
    f["S_max"] = _nanwin(S, np.nanmax, -1)
    f["Q_mean"] = _nanwin(Q, np.nanmean, -1)
    f["PF_mean"] = _nanwin(PF, np.nanmean, -1)
    f["V_mean"] = _nanwin(V, np.nanmean, -1)
    f["THD_mean"] = _nanwin(THD, np.nanmean, -1)
    f["imb_max"] = _nanwin(imb, np.nanmax, -1)
    e_rep = _nanwin(P * DT / 3600, np.nansum, 0)
    n_acc = _win(acc.astype(float), np.sum)
    e_rep = e_rep * W / np.maximum(n_acc, 1)                 # scale for missing samples
    out = _win(mes_out, np.sum)
    f["producing"] = (out > 0).astype(float)
    f["sec_ratio"] = np.where(out > 0, e_rep / np.maximum(out * m.sec_base, 1e-9), 0.0)
    # Algorithm 1 idle rule: not producing (MES) but drawing more than standby
    idle_s = (mes_out == 0) & (np.nan_to_num(P, nan=0) > (PLANT.standby_frac * 1.8) * m.rated_kw)
    f["idle_frac"] = _win(idle_s.astype(float), np.mean)
    f["phys_resid"] = _nanwin(phys, np.nanmean, -1)
    f["feeder_resid"] = feeder_rel
    f["feeder_resid30"] = feeder_resid30
    f["n_rx"] = _win(pres.astype(float), np.sum)
    f["n_acc"] = n_acc
    f["f_tag_fail"] = _win((pres & ~rx["tag_ok"]).astype(float), np.sum) / np.maximum(f["n_rx"], 1)
    f["f_seq_fail"] = _win(rx["seq_fail"].astype(float), np.sum) / np.maximum(f["n_rx"], 1)
    f["f_fresh_fail"] = _win((pres & ~rx["fresh"]).astype(float), np.sum) / np.maximum(f["n_rx"], 1)
    f["max_age"] = np.clip(_nanwin(np.where(pres, np.abs(rx["age"]), np.nan), np.nanmax, 60), 0, 3600)
    # longest gap between accepted packets inside the window (s)
    gap = np.zeros(len(acc))
    last = -1
    for i in range(len(acc)):
        if acc[i]:
            last = i
        gap[i] = (i - last) * DT if last >= 0 else 0
    f["gap_s"] = _win(gap, np.max)
    nsp = np.zeros(len(acc))
    if len(rx["spoof_slot"]):
        np.add.at(nsp, rx["spoof_slot"], 1)
    f["n_spoof"] = _win(nsp, np.sum)
    # causal context: last valid value over the previous 3 windows (no look-ahead)
    for k in CONTEXT_BASE:
        x = f[k].copy()
        valid = f["n_acc"] > 0
        idx = np.where(valid, np.arange(len(x)), -1)
        np.maximum.accumulate(idx, out=idx)
        held = np.where(idx >= 0, x[np.maximum(idx, 0)], -1)
        prev = np.full(len(x), -1.0)
        for lag in (1, 2, 3):
            sh = np.r_[np.full(lag, -1.0), held[:-lag]]
            prev = np.maximum(prev, sh) if k in ("S_max", "imb_max", "THD_mean") else np.where(prev == -1, sh, prev)
        f[f"{k}_prev3"] = prev
    X = np.stack([f[k] for k in FEATURES], 1)

    # ---- ground truth label per window ----
    code_w = _win(codes, np.max)
    net_att = np.isin(code_w, NETWORK_ATTACKS) & (code_w != A_COMB)
    src_att = code_w == A_SRC
    comb = code_w == A_COMB
    health = _win(sim["health"].astype(float), np.mean) >= 0.5
    ineff = _win(sim["ineff"].astype(float), np.mean) >= 0.5
    y = np.zeros(len(code_w), np.int8)
    y[ineff] = 1
    y[health] = 2
    y[net_att] = 3
    y[src_att] = 4
    y[(net_att | src_att) & (health | ineff)] = 5
    y[comb] = 5
    return X, y, code_w
