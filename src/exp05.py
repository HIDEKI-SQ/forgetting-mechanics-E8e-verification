# -*- coding: utf-8 -*-
IMPLEMENTED = True

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

np.random.seed(123)
OUTDIR = "outputs/E8/exp5"
os.makedirs(OUTDIR, exist_ok=True)
N_NODES = 60
SEM_DIM = 8
N_CLUSTERS = 3
CLUSTER_STD = 0.6
SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]

def generate_semantics(n_nodes, dim=8, n_clusters=3, cluster_std=0.6):
    rng = np.random.default_rng(2025)
    centers = rng.normal(0, 1.0, size=(n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n_nodes)
    S = centers[labels] + rng.normal(0, cluster_std, size=(n_nodes, dim))
    return S

def ring_positions(n_nodes):
    th = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    return np.stack([np.cos(th), np.sin(th)], axis=1)

def add_noise(P, sigma):
    rng = np.random.default_rng(2026)
    return P + rng.normal(0, sigma, size=P.shape)

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
    return float(np.clip(1.0 - dstep / (mean_global + 1e-12), 0, 1))

def gen_uniformity(P, bins=8):
    H, *_ = np.histogram2d(P[:,0], P[:,1], bins=bins, range=[[-1.2,1.2],[-1.2,1.2]])
    p = H.ravel().astype(float)
    p = p / (p.sum() + 1e-12)
    nz = p[p > 0]
    Hs = -np.sum(nz * np.log(nz + 1e-12))
    Hmax = math.log(len(p))
    return float(Hs / (Hmax + 1e-12))

def vs_spearman(S, P):
    d_sem = pdist(S, metric='euclidean')
    d_spa = pdist(P, metric='euclidean')
    rho, _ = spearmanr(d_sem, d_spa)
    return float(np.clip(rho, -1.0, 1.0))

S = generate_semantics(N_NODES, dim=SEM_DIM, n_clusters=N_CLUSTERS, cluster_std=CLUSTER_STD)
P0 = ring_positions(N_NODES)

rows = []
for sigma in SIGMAS:
    P = add_noise(P0, sigma)
    order = path_order_by_angle(P)
    SP = sp_path_smoothness(P, order)
    GEN = gen_uniformity(P)
    VS = vs_spearman(S, P)
    rows.append({"sigma": sigma, "SP": SP, "GEN": GEN, "VS": VS})

df = pd.DataFrame(rows)
print("=== E8 EXP5 Noise Robustness ===")
print(df.round(3).to_string(index=False))
df.to_csv(f"{OUTDIR}/E8_exp5_metrics.csv", index=False)

plt.figure(figsize=(8, 3.6))
for m in ["SP", "GEN", "VS"]:
    plt.plot(df["sigma"], df[m], marker="o", label=m)
plt.axhline(0, color="k", linewidth=1, alpha=0.3)
plt.legend()
plt.xlabel("jitter sigma")
plt.title("E8-Exp5: Robustness to Spatial Jitter")
plt.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp5_noise_curves.png", dpi=200, bbox_inches="tight")
plt.show()
