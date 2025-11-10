# -*- coding: utf-8 -*-
IMPLEMENTED = True

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

np.random.seed(99)
OUTDIR = "outputs/E8/exp4"
os.makedirs(OUTDIR, exist_ok=True)
N_NODES = 60

def ring_positions(n_nodes):
    th = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x = np.cos(th)
    y = np.sin(th)
    return np.stack([x, y], axis=1)

def path_order_by_angle(P):
    ctr = P.mean(axis=0)
    U = P - ctr
    ang = np.arctan2(U[:,1], U[:,0])
    return np.argsort(ang)

def sp_path_smoothness(P, order):
    D = pdist(P, metric='euclidean')
    mean_global = float(D.mean())
    Pord = P[order]
    dstep = np.linalg.norm(Pord[1:] - Pord[:-1], axis=1).mean()
    sp = 1.0 - (dstep / (mean_global + 1e-12))
    return float(np.clip(sp, 0.0, 1.0))

P = ring_positions(N_NODES)
order_good = path_order_by_angle(P)     # phi on
order_bad = np.random.permutation(N_NODES)  # phi off

SP_good = sp_path_smoothness(P, order_good)
SP_bad = sp_path_smoothness(P, order_bad)

df = pd.DataFrame([
    {"Route": "angle-ordered (phi on)", "SP": SP_good},
    {"Route": "shuffled (phi off)", "SP": SP_bad}
])

print("=== E8 EXP4 Phi Ablation ===")
print(df.to_string(index=False))
df.to_csv(f"{OUTDIR}/E8_exp4_metrics.csv", index=False)

plt.figure(figsize=(5, 3.5))
plt.bar(df["Route"], df["SP"])
plt.ylim(0, 1)
plt.ylabel("SP (smoothness)")
plt.title("E8-Exp4: Route Order Ablation on Same Positions")
plt.grid(alpha=0.3, linestyle=":")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp4_phi_ablation.png", dpi=200, bbox_inches="tight")
plt.show()
