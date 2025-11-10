# -*- coding: utf-8 -*-
IMPLEMENTED = True

import os, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

np.random.seed(42)
OUTDIR = "outputs/E8/exp1"
os.makedirs(OUTDIR, exist_ok=True)
N_NODES = 60
SEM_DIM = 8
N_CLUSTERS = 3
CLUSTER_STD = 0.6

def generate_semantics(n_nodes, dim=8, n_clusters=3, cluster_std=0.6, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    centers = rng.normal(0, 1.0, size=(n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n_nodes)
    S = centers[labels] + rng.normal(0, cluster_std, size=(n_nodes, dim))
    return S, labels

def ring_positions(n_nodes, radius=1.0):
    theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y], axis=1)

def random_positions(n_nodes, scale=1.0):
    rng = np.random.default_rng(None)
    return rng.uniform(-scale, scale, size=(n_nodes, 2))

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

def gen_uniformity(P, bins=8):
    H, *_ = np.histogram2d(P[:,0], P[:,1], bins=bins, range=[[-1.2,1.2],[-1.2,1.2]])
    p = H.ravel().astype(float)
    p = p / (p.sum() + 1e-12)
    nz = p[p > 0]
    Hs = -np.sum(nz * np.log(nz + 1e-12))
    Hmax = math.log(len(p))
    return float(Hs / (Hmax + 1e-12))

def vs_spearman_semantic_spatial(S, P):
    d_sem = pdist(S, metric='euclidean')
    d_spa = pdist(P, metric='euclidean')
    rho, _ = spearmanr(d_sem, d_spa)
    return float(np.clip(rho, -1.0, 1.0))

S, labels = generate_semantics(N_NODES, dim=SEM_DIM, n_clusters=N_CLUSTERS, cluster_std=CLUSTER_STD)
P_spatial = ring_positions(N_NODES)
order_spatial = np.arange(N_NODES)
P_random = random_positions(N_NODES)
order_random = path_order_by_angle(P_random)

SP_spa = sp_path_smoothness(P_spatial, order_spatial)
GEN_spa = gen_uniformity(P_spatial)
VS_spa = vs_spearman_semantic_spatial(S, P_spatial)

SP_rnd = sp_path_smoothness(P_random, order_random)
GEN_rnd = gen_uniformity(P_random)
VS_rnd = vs_spearman_semantic_spatial(S, P_random)

df = pd.DataFrame([
    {"Condition": "spatial", "SP": SP_spa, "GEN": GEN_spa, "VS": VS_spa},
    {"Condition": "random", "SP": SP_rnd, "GEN": GEN_rnd, "VS": VS_rnd},
])
print("=== E8 EXP1 Metrics ===")
print(df.to_string(index=False))
df.to_csv(f"{OUTDIR}/E8_exp1_metrics.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
for j, (metric, ylim) in enumerate([("SP", (0, 1)), ("GEN", (0, 1)), ("VS", (-0.3, 0.3))]):
    ax[j].bar(["spatial", "random"], [df.loc[0, metric], df.loc[1, metric]])
    ax[j].set_title(metric)
    ax[j].set_ylim(ylim)
    ax[j].grid(alpha=0.3, linestyle=":")
fig.suptitle("E8-Exp1: Spatial vs Random (SP/GEN/VS)", y=1.03, fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp1_baseline.png", dpi=200, bbox_inches="tight")
plt.show()

