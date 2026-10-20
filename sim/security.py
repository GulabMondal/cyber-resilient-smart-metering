"""Edge signing, network, attack injection, Algorithm 2 validation and the
hash-chained store.

Cryptography is real: every record is signed with HMAC-SHA256 under a
per-device 256-bit key and verified by the server; stored rows are linked with
SHA-256.  Attacks are applied to the packet stream exactly as an adversary
without the key could apply them.
"""
import hashlib
import hmac
import numpy as np
from .config import DT, SPD, DAYS, NET, ATTACK, MACHINES

SQ3 = np.sqrt(3.0)
FIELDS = ("P", "Q", "S", "V", "I1", "I2", "I3", "PF", "THD", "E")

# attack type codes (per slot)
A_NONE, A_MITM, A_REPLAY, A_SPOOF, A_TS, A_DELAY, A_DOS, A_SRC, A_COMB, A_POISON = range(10)
A_NAMES = {A_MITM: "MITM FDI", A_REPLAY: "Replay", A_SPOOF: "Spoofing",
           A_TS: "Timestamp manipulation", A_DELAY: "Delay", A_DOS: "DoS / missed heartbeat",
           A_SRC: "Source-side FDI", A_COMB: "Replay masking overload", A_POISON: "Slow poisoning"}
NETWORK_ATTACKS = (A_MITM, A_REPLAY, A_SPOOF, A_TS, A_DELAY, A_DOS, A_COMB)


def device_keys(rng):
    return [rng.bytes(32) for _ in MACHINES]


def _payload(vals, i):
    return "|".join(f"{vals[f][i]:.3f}" for f in FIELDS)


def _msg(dev_id, ts_ms, seq, payload):
    return f"{dev_id}|{ts_ms}|{seq}|{payload}".encode()


def _tag(key, dev_id, ts_ms, seq, payload):
    return hmac.new(key, _msg(dev_id, ts_ms, seq, payload), hashlib.sha256).digest()


# --------------------------------------------------------------------------
def reported_values(sim, rng):
    """Values the metering IC reports (0.2 % measurement noise)."""
    n = len(sim["P"])
    noise = lambda: 1 + 0.002 * rng.standard_normal(n)
    v = dict(P=sim["P"] * noise(), Q=sim["Q"] * noise(), S=sim["S"] * noise(),
             V=sim["V"] * noise(), I1=sim["I"][:, 0] * noise(),
             I2=sim["I"][:, 1] * noise(), I3=sim["I"][:, 2] * noise(),
             PF=np.clip(sim["pf"] * noise(), 0, 1), THD=sim["thd"] * noise())
    v["E"] = np.cumsum(sim["P"] * DT / 3600)
    return v


def schedule_attacks(rng, n_machines, N, health_masks, ovl_masks, cfg=ATTACK, run_masks=None):
    """Place attack events. Returns per-machine slot codes and an event list."""
    codes = [np.zeros(N, np.int8) for _ in range(n_machines)]
    events = []
    m2s = lambda mins: max(1, int(round(mins * 60 / DT)))

    def place(kind, count, dur_fn, mask_fn=None):
        placed = 0
        tries = 0
        while placed < count and tries < count * 50:
            tries += 1
            mi = int(rng.integers(n_machines))
            du = dur_fn()
            if mask_fn is not None:
                cand = np.flatnonzero(mask_fn(mi))
                if len(cand) == 0:
                    continue
                st = int(rng.choice(cand))
            else:
                st = int(rng.integers(SPD, N - du - 1))   # never on day 0
            en = min(st + du, N - 1)
            if codes[mi][st:en].any():
                continue
            # signed-false-data events never overlap in time across machines
            if kind == A_SRC and any(c[st:en].__eq__(A_SRC).any() for c in codes):
                continue
            codes[mi][st:en] = kind
            ev = dict(kind=kind, mi=mi, st=st, en=en)
            events.append(ev)
            placed += 1
        return events

    c = cfg
    place(A_COMB, c.combined, lambda: m2s(rng.uniform(5, 15)), lambda mi: ovl_masks[mi])
    place(A_MITM, c.mitm_fdi, lambda: m2s(rng.uniform(*c.dur_min)))
    place(A_REPLAY, c.replay, lambda: m2s(rng.uniform(1, 60)))
    place(A_SPOOF, c.spoof, lambda: m2s(rng.uniform(1, 10)))
    place(A_TS, c.timestamp, lambda: m2s(rng.uniform(*c.dur_min)))
    place(A_DELAY, c.delay, lambda: m2s(rng.uniform(*c.dur_min)))
    place(A_DOS, c.dos, lambda: max(1, int(rng.uniform(*c.dos_s) / DT)))
    # signed false data is placed while the machine is running (when there is energy to hide)
    place(A_SRC, c.source_fdi, lambda: m2s(rng.uniform(*c.src_fdi_min)),
          (lambda mi: run_masks[mi]) if run_masks is not None else None)
    # per-event parameters
    for ev in events:
        k = ev["kind"]
        if k == A_MITM:
            ev["scale"] = rng.uniform(*c.mitm_scale)
        elif k in (A_REPLAY, A_COMB):
            ev["age"] = int(rng.uniform(*c.replay_age_h) * 3600 / DT)
        elif k == A_SPOOF:
            ev["unknown_id"] = bool(rng.random() < 0.5)
        elif k == A_TS:
            ev["shift_s"] = float(rng.choice([-1, 1]) * rng.uniform(30, 3600))
        elif k == A_DELAY:
            ev["delay_s"] = float(rng.uniform(*c.delay_s))
        elif k == A_SRC:
            ev["delta"] = float(rng.uniform(*c.src_fdi_scale))
            ev["physical"] = bool(rng.random() < c.src_fdi_physical_frac)
    # slow poisoning: two per machine-pair, starting in the baseline or optimisation period
    for j in range(c.poisoning):
        mi = j % n_machines
        st = int(rng.integers(3, DAYS - 22)) * SPD
        days = int(rng.integers(*c.poison_days))
        en = st + days * SPD
        if codes[mi][st:en].any() and False:
            pass
        events.append(dict(kind=A_POISON, mi=mi, st=st, en=en, rate=c.poison_rate_per_day))
    return codes, events


def apply_source_attacks(vals, mi, events):
    """Signed false data: applied on the device BEFORE signing."""
    v = {k: a.copy() for k, a in vals.items()}
    for ev in events:
        if ev["mi"] != mi:
            continue
        sl = slice(ev["st"], ev["en"])
        if ev["kind"] == A_SRC:
            f = 1 - ev["delta"]
            keys = ("P", "Q", "S", "I1", "I2", "I3") if ev["physical"] else ("P",)
            for k in keys:
                v[k][sl] *= f
        elif ev["kind"] == A_POISON:
            n = ev["en"] - ev["st"]
            f = 1 - ev["rate"] * np.arange(n) * DT / 86400
            for k in ("P", "Q", "S", "I1", "I2", "I3"):
                v[k][sl] *= f
    # cumulative energy register follows the (falsified) power
    v["E"] = np.cumsum(v["P"] * DT / 3600)
    return v


def transmit(mi, vals, key, codes, events, rng, net=NET):
    """Device signing + network + in-transit attacks + Algorithm 2.

    Returns a dict of per-slot arrays describing what the server saw, plus a
    list of injected (spoofed) packets.
    """
    N = len(vals["P"])
    dev_id = MACHINES[mi].name
    t_true = np.arange(N) * DT * 1.0
    clock = net.clock_sigma_s * rng.standard_normal(N)
    ts = np.round((t_true + clock) * 1000).astype(np.int64)        # device ms
    seq = np.arange(1, N + 1, dtype=np.int64)
    payload = [_payload(vals, i) for i in range(N)]
    tags = [_tag(key, dev_id, int(ts[i]), int(seq[i]), payload[i]) for i in range(N)]

    # network: latency, benign loss and EMI bursts (retransmission delay / loss)
    lat = rng.uniform(*net.latency_ms, N) / 1000
    present = rng.random(N) >= net.loss_rate
    n_b = rng.poisson(net.burst_per_day * DAYS)
    burst = np.zeros(N, bool)
    for _ in range(n_b):
        st = int(rng.integers(0, N - 20)); du = int(rng.uniform(*net.burst_s) / DT) + 1
        burst[st:st + du] = True
    nb = burst.sum()
    lat[burst] += rng.uniform(*net.burst_delay_s, nb)
    present[burst] &= rng.random(nb) >= 0.3
    arrival = t_true + lat

    rx_ts = ts.copy(); rx_seq = seq.copy(); rx_pay = list(payload); rx_tag = list(tags)
    rx_id_known = np.ones(N, bool)
    modified = np.zeros(N, bool)
    spoof_pk = []      # (slot, id_known, tag_ok) injected packets

    for ev in events:
        if ev["mi"] != mi:
            continue
        k, st, en = ev["kind"], ev["st"], ev["en"]
        if k == A_MITM:
            for i in range(st, en):
                v = {f: vals[f][i] for f in FIELDS}
                v["P"] *= ev["scale"]; v["E"] *= ev["scale"]
                rx_pay[i] = "|".join(f"{v[f]:.3f}" for f in FIELDS)
            modified[st:en] = True
        elif k in (A_REPLAY, A_COMB):
            src = np.arange(st, en) - ev["age"]
            for i, j in zip(range(st, en), src):
                rx_ts[i], rx_seq[i], rx_pay[i], rx_tag[i] = ts[j], seq[j], payload[j], tags[j]
            present[st:en] = True
        elif k == A_TS:
            rx_ts[st:en] = ts[st:en] + int(ev["shift_s"] * 1000)
        elif k == A_DELAY:
            arrival[st:en] += ev["delay_s"]
        elif k == A_DOS:
            present[st:en] = False
        elif k == A_SPOOF:
            for i in range(st, en):
                spoof_pk.append((i, not ev["unknown_id"]))

    # ---------------- Algorithm 2 (server) ----------------
    tag_ok = np.zeros(N, bool)
    for i in np.flatnonzero(present):
        t = _tag(key, dev_id, int(rx_ts[i]), int(rx_seq[i]), rx_pay[i])
        tag_ok[i] = hmac.compare_digest(t, rx_tag[i])
    fresh = np.abs(arrival - rx_ts / 1000.0) <= net.freshness_s
    seq_ok = np.zeros(N, bool)
    accepted = np.zeros(N, bool)
    last = 0
    for i in np.flatnonzero(present):
        if not (tag_ok[i] and fresh[i]):
            continue
        if rx_seq[i] > last:
            seq_ok[i] = True; accepted[i] = True; last = rx_seq[i]
    # sequence failures are counted only for authentic & fresh packets
    seq_fail = present & tag_ok & fresh & ~seq_ok
    # spoofed packets: either unknown ID or a forged (random) tag -> always rejected
    sp_slot = np.array([s for s, _ in spoof_pk], dtype=np.int64)
    sp_known = np.array([kn for _, kn in spoof_pk], dtype=bool)
    sp_tag_ok = np.zeros(len(spoof_pk), bool)
    for n, (s, kn) in enumerate(spoof_pk):
        if kn:   # attacker guesses a tag without the key
            forged = rng.bytes(32)
            t = _tag(key, dev_id, int(ts[s]), int(seq[s]) + 10**7, payload[s])
            sp_tag_ok[n] = hmac.compare_digest(t, forged)

    # values as seen by the server (decoded from the received payload)
    rx_vals = {f: np.full(N, np.nan) for f in FIELDS}
    idx = np.flatnonzero(present)
    dec = np.array([[float(x) for x in rx_pay[i].split("|")] for i in idx]) if len(idx) else np.zeros((0, len(FIELDS)))
    for j, f in enumerate(FIELDS):
        rx_vals[f][idx] = dec[:, j]

    return dict(present=present, tag_ok=tag_ok, fresh=fresh, seq_fail=seq_fail,
                accepted=accepted, arrival=arrival, rx_ts=rx_ts, age=arrival - rx_ts / 1000.0,
                rx_vals=rx_vals, tags=rx_tag, payload=rx_pay, seq=rx_seq,
                spoof_slot=sp_slot, spoof_known=sp_known, spoof_tag_ok=sp_tag_ok,
                dev_id=dev_id)


# --------------------------------------------------------------------------
def build_store(rx):
    """Accepted rows with their device tags, linked by a SHA-256 chain."""
    idx = np.flatnonzero(rx["accepted"])
    rows = [f"{rx['dev_id']}|{int(rx['rx_ts'][i])}|{int(rx['seq'][i])}|{rx['payload'][i]}" for i in idx]
    tags = [rx["tags"][i] for i in idx]
    chain = []
    h = b"\x00" * 32
    for r, t in zip(rows, tags):
        h = hashlib.sha256(r.encode() + t + h).digest()
        chain.append(h)
    return dict(idx=idx, rows=rows, tags=tags, chain=chain)


def db_tamper_and_audit(store, key, n_events, rng, recompute_frac=0.5):
    """Attacker on the DB host (no device keys) edits stored rows.

    In half of the events the attacker also recomputes the hash chain.  The
    audit checks (a) the chain and (b) every stored device tag.
    Returns per-event detection by chain-only and by chain+tag audit.
    """
    rows, tags = list(store["rows"]), store["tags"]
    chain = list(store["chain"])
    n = len(rows)
    out = []
    for _ in range(n_events):
        st = int(rng.integers(1, n - 60)); k = int(rng.integers(1, 50))
        recompute = rng.random() < recompute_frac
        tampered = list(range(st, st + k))
        new_rows = {}
        for i in tampered:
            parts = rows[i].split("|")
            parts[3] = f"{float(parts[3]) * rng.uniform(0.6, 0.95):.3f}"   # lower kW
            new_rows[i] = "|".join(parts)
        # chain check over the affected region
        h = chain[st - 1]
        chain_break = False
        stored_next = chain[st + k] if st + k < n else None
        hh = h
        for i in range(st, st + k):
            hh = hashlib.sha256(new_rows[i].encode() + tags[i] + hh).digest()
        if recompute:
            # attacker rewrites the chain from st onward, so links verify
            chain_break = False
        else:
            # original chain entries are kept; recomputation differs at st
            chain_break = hh != chain[st + k - 1]
        # tag audit with the server-side key store
        tag_break = False
        for i in tampered:
            dev, t, s, pay = new_rows[i].split("|", 3)
            t2 = hmac.new(key, f"{dev}|{t}|{s}|{pay}".encode(), hashlib.sha256).digest()
            if not hmac.compare_digest(t2, tags[i]):
                tag_break = True
                break
        out.append(dict(recompute=recompute, rows=k, chain=chain_break, chain_or_tag=chain_break or tag_break))
    return out
