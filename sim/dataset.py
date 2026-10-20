"""End-to-end dataset generation: plant -> signing -> network/attacks -> validation
-> storage -> features.  ``build(seed)`` returns everything the experiments need.
"""
import time
import numpy as np
from .config import SEED, DT, SPD, DAYS, MACHINES, ATTACK, NET
from .plant import simulate_plant
from . import security as sec
from .features import feeder_balance, machine_windows


def build(seed=SEED, net=NET, attack=ATTACK, verbose=True, with_store=True):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    plant, aux = simulate_plant(rng)
    N = len(aux)
    keys = sec.device_keys(rng)
    codes, events = sec.schedule_attacks(
        rng, len(MACHINES), N,
        [s["health"] for s in plant], [s["ovl"] for s in plant], attack,
        run_masks=[np.isin(s["state"], (1, 2)) for s in plant])
    rxs, stores = [], []
    for mi, s in enumerate(plant):
        vals = sec.reported_values(s, rng)
        vals = sec.apply_source_attacks(vals, mi, events)
        rx = sec.transmit(mi, vals, keys[mi], codes[mi], events, rng, net)
        rxs.append(rx)
        if with_store:
            stores.append(sec.build_store(rx))
        if verbose:
            print(f"  {MACHINES[mi].name}: signed/verified {N:,} records ({time.time()-t0:.0f}s)")
    rel, cus = feeder_balance(plant, rxs, aux, rng)
    Xs, ys, cws, mids, days = [], [], [], [], []
    for mi, (s, rx) in enumerate(zip(plant, rxs)):
        X, y, cw = machine_windows(mi, s, rx, s["out_kg"], rel, cus, codes[mi])
        Xs.append(X); ys.append(y); cws.append(cw)
        mids.append(np.full(len(y), mi)); days.append(np.arange(len(y)) // 1440)
    db = None
    if with_store:
        db = []
        for mi in range(len(MACHINES)):
            db += sec.db_tamper_and_audit(stores[mi], keys[mi], attack.db_tamper // len(MACHINES), rng)
    if verbose:
        print(f"  dataset built in {time.time()-t0:.0f}s")
    return dict(plant=plant, aux=aux, rxs=rxs, codes=codes, events=events,
                X=np.concatenate(Xs), y=np.concatenate(ys), code_w=np.concatenate(cws),
                machine=np.concatenate(mids), day=np.concatenate(days),
                feeder_rel=rel, feeder_resid30=cus, db=db, stores=stores)
