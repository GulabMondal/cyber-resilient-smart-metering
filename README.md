# Cyber-resilient smart energy metering — simulator, dataset and analysis pipeline

Code and data for the paper *"A Cyber-Resilient AI-Enabled Smart Energy Metering Framework for Industrial Energy Saving, Load Optimization, and Trustworthy Energy Data Analytics"* (Discover Internet of Things).

Repository: https://github.com/GulabMondal/cyber-resilient-smart-metering

One command regenerates the dataset from the published seed and reproduces every number, table and figure in Sections 9–10 of the paper.

```bash
pip install -r requirements.txt      # exact versions used for the paper
python verify.py                     # ~5 min: regenerates the data, checks SHA-256 checksums,
                                     # reloads the trained model, recomputes headline results and ROI
sh run_all.sh                        # ~2 h on 2 CPU cores, 8 GB RAM: reruns every experiment
```

Outputs: `results/results.json` (all reported numbers), `results/*.npz|npy` (series used in figures), `figs/fig1.png … fig15.png`.

## Contents

| Path | What |
|---|---|
| `data/windows_features.csv.gz` | 345,600 one-minute windows: machine, day, minute, 29 features, label, train/test/gap split |
| `data/attack_events.csv` | Every injected attack event with machine, start, end and parameters |
| `models/random_forest_chrono.joblib` | Trained Random Forest (chronological split) used for the reported results |
| `results/results.json` | Every number reported in the paper, written by the pipeline |
| `results/verification_report.txt` | Output of `verify.py` (31 checks, all passing) |
| `firmware/security_config/` | ESP-IDF Secure Boot V2 / flash-encryption configuration and eFuse verification procedure |
| `firmware/esp32s3_crypto_bench/` | Benchmark sketch for HMAC, AES-GCM and TLS on an ESP32-S3 |

The 4,147,200 record-level samples (5-s resolution) are not stored; `sim.dataset.build()` regenerates them deterministically from the seed.

## What is simulated

| Layer | File | What it does |
|---|---|---|
| Configuration | `sim/config.py` | Every parameter and the master seed (`SEED = 20261019`) |
| Plant | `sim/plant.py` | 4 heat machines (45–60 kW), 60 days at 5-s resolution: shifts, heat-up, Weibull production load, idle behaviour, inrush, PF, imbalance, THD, health and inefficiency episodes, optimisation scenario, auxiliary feeder load |
| Edge, network, attacks, validation | `sim/security.py` | Real HMAC-SHA256 signing with a distinct 256-bit key per device; latency, loss and EMI bursts; MITM, replay, spoofing, timestamp, delay, DoS, signed (source-side) FDI, slow poisoning; Algorithm 2 validation; SHA-256 hash-chained store; database tampering with and without chain recomputation |
| Features and labels | `sim/features.py` | 1-minute windows, 29 features, feeder energy balance, six ground-truth classes |
| Experiments | `experiments.py`, `experiments_stage2.py`, `experiments_stage3.py` | Energy statistics, event-level detection, five classifiers on chronological and random splits, ablation, robustness (LOMO, degraded networks, seeds), ROI |
| Figures | `figures.py` | All result figures |

Every record in the dataset (4,147,200 in the main run) is actually signed on the "device" and verified by the "server"; cryptographic detection results are therefore produced by real cryptographic operations, not assumed.

## Leakage control

The classifier is evaluated with a chronological split (training days 1–21 and 31–51, test days 23–30 and 53–60, gap days 22 and 52). SMOTE is applied only to training data, inside each cross-validation fold during tuning. A random 80/20 split is also reported for comparison with the original submission.

## Edge firmware benchmark

`firmware/esp32s3_crypto_bench/` measures on an ESP32-S3 the cost of hardware and software HMAC-SHA256, AES-128-GCM, a mutually authenticated TLS handshake and session resumption (Table 5 of the paper). Open in Arduino IDE 2.x (board "ESP32S3 Dev Module"), set Wi-Fi, broker and certificates, flash, and read the results block on the serial monitor at 115200 baud.

## Licence and citation

MIT licence. Please cite the paper (see `CITATION.cff`).
