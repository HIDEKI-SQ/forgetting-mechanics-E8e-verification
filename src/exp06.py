# -*- coding: utf-8 -*-
IMPLEMENTED = True

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

np.random.seed(2024)
OUTDIR = "outputs/E8/exp6"
os.makedirs(OUTDIR, exist_ok=True)
N_LIST = [20, 40, 80, 160]
SEM_DIM = 8
N_CLUSTERS = 3
CLUSTER_STD = 0.6

def generate_semantics(n, dim=8, k=3, std=0.6, seed=None):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, size=(k, dim))
    labels = rng.integers(0, k, size=n)
    S = centers[labels] + rng.normal(0, std, size=(n, dim))
    return S

def ring_positions(n):
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([np.cos(th), np.sin(th)], axis=1)

def random_positions(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 2))

def order_by_angle(P):
    ctr = P.mean(axis=0)
    U = P - ctr
    ang = np.arctan2(U[:,1], U[:,0])
    return np.argsort(ang)

def sp(P, order):
    D = pdist(P)
    mg = float(D.mean())
    Pord = P[order]
    dstep = np.linalg.norm(Pord[1:] - Pord[:-1], axis=1).mean()
    return float(np.clip(1.0 - dstep / (mg + 1e-12), 0, 1))

def gen(P, bins=10):
    H, *_ = np.histogram2d(P[:,0], P[:,1], bins=bins, range=[[-1.2,1.2],[-1.2,1.2]])
    p = H.ravel().astype(float)
    p = p / (p.sum() + 1e-12)
    nz = p[p > 0]
    Hs = -np.sum(nz * np.log(nz + 1e-12))
    Hmax = math.log(len(p))
    return float(Hs / (Hmax + 1e-12))

def vs(S, P):
    d_sem = pdist(S)
    d_spa = pdist(P)
    rho, _ = spearmanr(d_sem, d_spa)
    return float(np.clip(rho, -1, 1))

rows = []
for n in N_LIST:
    S = generate_semantics(n, dim=SEM_DIM, k=N_CLUSTERS, std=CLUSTER_STD, seed=100+n)
    # spatial
    P1 = ring_positions(n)
    o1 = order_by_angle(P1)
    # random
    P2 = random_positions(n, seed=200+n)
    o2 = order_by_angle(P2)
    rows.append({"N": n, "Condition": "spatial", "SP": sp(P1, o1), "GEN": gen(P1), "VS": vs(S, P1)})
    rows.append({"N": n, "Condition": "random", "SP": sp(P2, o2), "GEN": gen(P2), "VS": vs(S, P2)})

df = pd.DataFrame(rows)
print("=== E8 EXP6 Scale ===")
print(df.round(3).to_string(index=False))
df.to_csv(f"{OUTDIR}/E8_exp6_metrics.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for j, m in enumerate(["SP", "GEN", "VS"]):
    for cond, color in [("spatial", "C0"), ("random", "C1")]:
        sub = df[df["Condition"] == cond]
        ax[j].plot(sub["N"], sub[m], marker="o", label=cond if j == 0 else None)
    ax[j].set_title(m)
    ax[j].grid(alpha=0.3, linestyle=":")
    if m == "VS":
        ax[j].axhline(0, color="k", linewidth=1, alpha=0.3)
ax[0].legend()
fig.suptitle("E8-Exp6: Scaling with N", y=1.03)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp6_scale.png", dpi=200, bbox_inches="tight")
plt.show()
