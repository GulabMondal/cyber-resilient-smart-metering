"""Draws Fig. 1 (seven-layer architecture) to match the revised design."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2 = "#1d1d1b", "#52514e"
FILL = {"phys": "#e9e6df", "sense": "#cdeee2", "meter": "#cfe2f7", "comm": "#ddd8f5",
        "db": "#fde7b8", "ai": "#f6d3c6", "dash": "#d6e8c4", "side": "#ffffff"}
EDGE = {"phys": "#a8a49a", "sense": "#3fae86", "meter": "#4b8fd9", "comm": "#7f72d6",
        "db": "#d9a227", "ai": "#d9734e", "dash": "#6fa443", "side": "#8a8984"}

fig, ax = plt.subplots(figsize=(10, 12.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 122); ax.axis("off")
plt.rcParams["font.family"] = "DejaVu Sans"


def box(x, y, w, h, kind, title, lines, tsize=11.5, lsize=9.2, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                       fc=FILL[kind], ec=EDGE[kind], lw=1.4, ls="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=tsize, weight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.3 - i * 3.2, ln, ha="center", va="center", fontsize=lsize, color=INK2)


def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx if lx is not None else x1 + 1.2, ly if ly is not None else (y1 + y2) / 2, label,
                fontsize=8.3, color=INK2, va="center", style="italic")


X, W = 2, 70
box(X, 109, W, 11, "phys", "Layer 1 — Industrial machines / feeder",
    ["Heat machines · motors · compressors · process lines"])
box(X, 93, W, 11, "sense", "Layer 2 — Sensing & signal conditioning",
    ["1000/5 A CTs (R·Y·B) · burden resistors · RC anti-aliasing filters"])
box(X, 72.5, W, 15.5, "meter", "Layer 3 — ATM90E36 metering & ESP32-S3 acquisition",
    ["8 kHz sampling · V · I · P · Q · S · PF · Hz · THD · kWh",
     "Device ID · timestamp · sequence no. · HMAC-SHA256 tag",
     "(hardware HMAC unit, per-device eFuse key) · Secure Boot V2"])
box(X, 56, W, 11.5, "comm", "Layer 4 — Secure communication",
    ["MQTT over TLS 1.3 · plant LAN (Wi-Fi / Ethernet) · optional VPN"])
box(X, 37.5, W, 13.5, "db", "Layer 5 — Energy database & tamper-evident log",
    ["Rows stored with device tags · H$_n$ = SHA256(Data$_n$ ‖ T$_n$ ‖ H$_{n-1}$)",
     "Daily signed Merkle checkpoint stored off-host"])
# layer 6 with two sub-boxes
p = FancyBboxPatch((X, 13), W, 19.5, boxstyle="round,pad=0.4,rounding_size=1.6",
                   fc=FILL["ai"], ec=EDGE["ai"], lw=1.4)
ax.add_patch(p)
ax.text(X + W / 2, 30, "Layer 6 — Analytics, validation & classification", ha="center", va="center",
        fontsize=11.5, weight="bold", color=INK)
for (sx, t, l1, l2) in ((X + 2, "Energy analytics (Alg. 1)", "idle · SEC · demand", "PF · imbalance · THD"),
                        (X + 36, "Cyber & physics validation (Alg. 2)", "registry · tag · sequence · freshness",
                         "P–V–I–PF check · feeder balance")):
    q = FancyBboxPatch((sx, 19.5), 32, 7.6, boxstyle="round,pad=0.3,rounding_size=1.2",
                       fc="#ffffff", ec=EDGE["ai"], lw=1.0)
    ax.add_patch(q)
    ax.text(sx + 16, 25.2, t, ha="center", fontsize=8.6, weight="bold", color=INK)
    ax.text(sx + 16, 22.6, l1, ha="center", fontsize=8.4, color=INK2)
    ax.text(sx + 16, 20.6, l2, ha="center", fontsize=8.4, color=INK2)
ax.text(X + W / 2, 15.6, "Random Forest cyber-energy classifier (6 classes) · Alg. 3 severity routing",
        ha="center", fontsize=9.2, color=INK)
box(X, 0.5, W, 9, "dash", "Layer 7 — Energy & security dashboard",
    ["Alerts · recommendations · savings & ROI · ISO 50001 EnPI · incident log"])

# vertical flow arrows
cx = X + W / 2
for (y1, y2, lab) in ((108.6, 104.4, "3-phase supply"), (92.6, 88.4, "analogue signals"),
                      (72.1, 67.9, "HMAC-tagged record every 5 s"), (55.6, 51.4, "records accepted by Alg. 2"),
                      (37.1, 32.9, "trusted time series"), (12.6, 9.9, "classified events")):
    arrow(cx, y1, cx, y2, lab, lx=cx + 1.5)

# side inputs
SX, SW = 77, 21
box(SX, 37, SW, 12, "side", "Key store", ["per-device keys", "separate host"], tsize=10, lsize=8.6, dashed=True)
box(SX, 21, SW, 12, "side", "Feeder meter", ["authenticated plant", "incomer total"], tsize=10, lsize=8.6, dashed=True)
box(SX, 5, SW, 12, "side", "MES", ["production output", "per machine"], tsize=10, lsize=8.6, dashed=True)
for y in (43, 27, 11):
    arrow(SX - 0.4, y, X + W + 0.5, min(max(y, 16), 29))
ax.text(SX + SW / 2, 51.5, "Independent inputs", ha="center", fontsize=9, color=INK2, weight="bold")

fig.savefig("figs/fig1.png", dpi=300, bbox_inches="tight", facecolor="white")
print("fig1 written")
