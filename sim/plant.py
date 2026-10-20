"""Monte Carlo simulator of four industrial heat machines at 5-s resolution.

Produces the *true* physical quantities for every machine and sample, plus the
ground-truth operating state and event labels.  The cyber layer (attacks,
packets, validation) is applied afterwards in ``security.py``.
"""
import numpy as np
from .config import (DT, SPD, DAYS, BASE_DAYS, V_LL, MACHINES, PLANT, EVENTS)

OFF, HEATUP, PROD, IDLE, STANDBY = 0, 1, 2, 3, 4
SQ3 = np.sqrt(3.0)


def _m2s(minutes):
    return int(round(minutes * 60 / DT))


def _smooth(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def simulate_machine(mi, rng):
    """Simulate one machine for DAYS days. Returns dict of arrays (len N)."""
    m, p = MACHINES[mi], PLANT
    N = DAYS * SPD
    state = np.zeros(N, np.int8)
    load = np.zeros(N)            # fraction of rating actually drawn (true kW / rated)
    out_kg = np.zeros(N)          # production output per sample
    ineff = np.zeros(N, bool)     # ground truth: avoidable energy use
    idle_run = np.zeros(N)        # minutes spent continuously idle (for labels)
    inrush = np.zeros(N)          # extra kVA from motor/heater inrush

    # smooth production-load fluctuation: Weibull(k=1.5) scaled, 1-min smoothing
    w = rng.weibull(p.load_weibull_k, N) * p.load_weibull_scale
    prod_load = p.load_mean + _smooth(w, 12)

    for d in range(DAYS):
        opt = d >= BASE_DAYS
        base = d * SPD
        day = np.full(SPD, OFF, np.int8)
        jitter = rng.uniform(0, 90)                         # seconds
        offset_min = mi * p.stagger_min if opt else 0.0
        t0 = _m2s(m.shift_start_h * 60 + offset_min) + int(jitter / DT)
        t_heat_end = t0 + _m2s(p.heatup_min)
        t_end = t0 + _m2s(p.shift_hours * 60)
        day[t0:t_heat_end] = HEATUP
        day[t_heat_end:t_end] = PROD
        prod_load[base:base + SPD] *= max(0.6, 1 + p.demand_sigma * rng.standard_normal())

        # idle intervals inside the shift
        idles = [(_m2s(p.lunch_h * 60), _m2s(p.lunch_min)),
                 (_m2s(p.shiftchange_h * 60), _m2s(p.shiftchange_min))]
        for _ in range(rng.poisson(p.random_idle_per_day)):
            st = rng.integers(t_heat_end, t_end - _m2s(10))
            idles.append((st, _m2s(rng.exponential(p.random_idle_mean_min))))
        if rng.random() < p.overrun_prob:                    # left on after shift
            idles.append((t_end, _m2s(rng.uniform(*p.overrun_min))))
        resp = _m2s(p.alert_response_min)
        for st, du in idles:
            en = min(st + du, SPD)
            if opt and du > resp and rng.random() < p.response_prob:
                day[st:st + resp] = IDLE
                # operator acts on the idle alert: standby (or off after shift)
                day[st + resp:en] = OFF if st >= t_end else STANDBY
            else:
                day[st:en] = IDLE
        state[base:base + SPD] = day

        # inrush at start of heat-up (decaying exponential in kVA)
        peak = rng.uniform(*p.inrush_mult) - 0.95
        n_in = _m2s(4)
        tt = np.arange(n_in) * DT
        inrush[base + t0: base + t0 + n_in] = peak * np.exp(-tt / p.inrush_tau_s)

    # loads by state
    load[state == OFF] = p.off_leak_kw / m.rated_kw
    load[state == HEATUP] = 0.95
    load[state == PROD] = np.clip(prod_load[state == PROD], 0.25, 1.0)
    load[state == IDLE] = p.idle_frac * (1 + 0.03 * rng.standard_normal((state == IDLE).sum()))
    load[state == STANDBY] = p.standby_frac

    # production output is fixed by demand; SEC improvement lowers energy needed
    opt_mask = np.repeat(np.arange(DAYS) >= BASE_DAYS, SPD)
    kwh_nominal = load * m.rated_kw * DT / 3600
    out_kg[state == PROD] = kwh_nominal[state == PROD] / m.sec_base
    load = np.where((state == PROD) & opt_mask, load * m.sec_gain, load)

    # continuous idle duration -> avoidable-idle label (beyond alert threshold)
    run = 0
    is_idle = state == IDLE
    for i in range(N):
        run = run + 1 if is_idle[i] else 0
        idle_run[i] = run * DT / 60
    ineff |= is_idle & (idle_run > p.alert_response_min)

    return dict(state=state, load=load, out_kg=out_kg, ineff=ineff,
                inrush=inrush, opt=opt_mask)


def _episodes(rng, n, dur_min, N, mask=None):
    """Random non-overlapping-ish episodes; returns boolean mask."""
    res = np.zeros(N, bool)
    lo, hi = dur_min
    allowed = np.flatnonzero(mask) if mask is not None else None
    for _ in range(n):
        du = _m2s(rng.uniform(lo, hi))
        if allowed is not None and len(allowed):
            st = int(rng.choice(allowed))
        else:
            st = int(rng.integers(0, N - du))
        res[st:st + du] = True
    return res


def add_electrics(mi, sim, rng):
    """Voltage, currents, PF, THD and electrical-health events."""
    m, p, e = MACHINES[mi], PLANT, EVENTS
    N = len(sim["state"])
    state, load, opt = sim["state"], sim["load"], sim["opt"]
    running = (state == PROD) | (state == HEATUP)
    days = np.repeat(np.arange(DAYS), SPD)

    # power factor (load dependent, drifting in baseline, corrected later)
    pf_full = m.pf_base - p.pf_drift_per_day * days
    lf = np.clip(load, 0, 1)
    pf = np.where(running, pf_full - p.pf_load_slope * (1 - lf), p.pf_idle)
    pf = np.where(state == OFF, 0.55, pf)
    pf_corr = np.where(running, p.pf_corrected, 0.93)
    pf_corr = np.where(state == OFF, 0.55, pf_corr)
    if p.pf_corrected > 0:
        pf = np.where(opt, pf_corr, pf)
    pf = pf + 0.008 * rng.standard_normal(N)

    # electrical-health episodes (only while energised)
    on = state != OFF
    imb_ev = _episodes(rng, e.imbalance_n, e.imbalance_min, N, running)
    thd_ev = _episodes(rng, e.thd_n, e.thd_min, N, running)
    ovl_ev = _episodes(rng, e.overload_n, e.overload_min, N, state == PROD)
    pfd_ev = _episodes(rng, e.pfdrop_n, e.pfdrop_min, N, running & opt)
    sec_ev = _episodes(rng, e.secfault_n, e.secfault_min, N, state == PROD)

    load = load.copy()
    load[ovl_ev] = rng.uniform(1.25, 1.4)
    load[sec_ev] *= rng.uniform(1.15, 1.30)
    pf[pfd_ev] = 0.80 + 0.01 * rng.standard_normal(pfd_ev.sum())
    pf = np.clip(pf, 0.3, 0.995)

    P = load * m.rated_kw                                        # kW (true)
    S = P / pf + sim["inrush"] * m.rated_kw / max(m.pf_base, 0.5)
    Q = np.sqrt(np.maximum(S ** 2 - P ** 2, 0))
    V = V_LL * (1 + p.v_sigma * _smooth(rng.standard_normal(N), 60) * 4)
    I_avg = S * 1000 / (SQ3 * V)
    imb = np.where(imb_ev, rng.uniform(0.12, 0.20), p.imb_normal) * np.abs(1 + 0.2 * rng.standard_normal(N))
    ph = rng.standard_normal((N, 3)); ph -= ph.mean(1, keepdims=True)
    ph /= np.maximum(np.abs(ph).max(1, keepdims=True), 1e-9)
    I = I_avg[:, None] * (1 + imb[:, None] * ph)
    thd = rng.uniform(*p.thd_normal) + 0.006 * rng.standard_normal(N)
    thd = np.where(thd_ev, rng.uniform(0.15, 0.25), thd)
    pf_meas = np.clip(P / np.maximum(S, 1e-6), 0, 1)

    health = (imb_ev | thd_ev | ovl_ev | pfd_ev) & on
    sim.update(P=P, S=S, Q=Q, V=V, I=I, pf=pf_meas, thd=thd,
               health=health, ovl=ovl_ev & on, secfault=sec_ev & on)
    sim["ineff"] = sim["ineff"] | (sec_ev & on)
    return sim


def simulate_plant(rng):
    plant = []
    for mi in range(len(MACHINES)):
        sim = simulate_machine(mi, rng)
        plant.append(add_electrics(mi, sim, rng))
    # unmetered auxiliary load (lighting, HVAC, compressors): daily profile + slow variation
    t = np.arange(DAYS * SPD) * DT / 3600.0
    daily = 0.10 * np.clip(np.sin(2 * np.pi * (t % 24 - 6) / 24), 0, None)
    slow = _smooth(rng.standard_normal(DAYS * SPD), 720) * np.sqrt(720) * PLANT.aux_sigma
    aux = PLANT.aux_load_kw * (1 + daily + slow)
    return plant, aux
