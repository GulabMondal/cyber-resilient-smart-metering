#!/bin/sh
# Reproduces every number and figure in the paper (about 2 h on 2 cores, 8 GB RAM).
set -e
python3 -u experiments.py          # dataset, energy, detection, ML benchmark, ablation, LOMO
python3 -u experiments_stage2.py   # per-unit LOMO, network degradation
python3 -u experiments_stage3.py   # independent seeds, ROI
python3 -u experiments_stage4.py   # matched-conditions comparison, server cost, data/model export, checksums
python3 figures.py                 # figs/fig2.png ... figs/fig15.png
python3 fig1_architecture.py       # figs/fig1.png
