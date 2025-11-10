# -*- coding: utf-8 -*-
IMPLEMENTED = True

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

OUTDIR = "outputs/E8/exp2"
os.makedirs(OUTDIR, exist_ok=True)
N_NODES = 60
SEM_DIM = 8
N_CLUSTERS = 3
CLUSTER_STD = 0.6
N_SEEDS = 30

def generate_semantics(n_nodes, dim=8, n_clusters=3, cluster_std=0.6, seed=None):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, size=(n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n_nodes)
    S = centers[labels] + rng.normal(0, cluster_std, size=(n_nodes, dim))
    return S, labels

def ring_positions(n_nodes, radius=1.0):
    theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y], axis=1)

def grid_positions(n_nodes, width=1.2):
    g = int(np.round(np.sqrt(n_nodes)))
    xs = np.linspace(-width, width, g)
    ys = np.linspace(-width, width, g)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    if len(P) >= n_nodes:
        return P[:n_nodes]
    else:
        pad = np.zeros((n_nodes-len(P),2))
        return np.vstack([P, pad])

def spiral_positions(n_nodes, a=0.05, b=0.20):
    theta = np.linspace(0, 4*np.pi, n_nodes)
    r = a + b*theta
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    P = np.stack([x,y], axis=1)
    P = P / (np.max(np.linalg.norm(P, axis=1)) + 1e-12)
    return P

def twolayer_ring_positions(n_nodes, r1=1.0, r2=0.6):
    n1 = n_nodes//2
    n2 = n_nodes - n1
    th1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    th2 = np.linspace(0, 2*np.pi, n2, endpoint=False)
    P1 = np.stack([r1*np.cos(th1), r1*np.sin(th1)], axis=1)
    P2 = np.stack([r2*np.cos(th2), r2*np.sin(th2)], axis=1)
    return np.vstack([P1, P2])

def random_positions(n_nodes, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n_nodes, 2))

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
    p = p/(p.sum()+1e-12)
    nz = p[p>0]
    Hs = -np.sum(nz*np.log(nz+1e-12))
    Hmax = math.log(len(p))
    return float(Hs/(Hmax+1e-12))

def vs_spearman_semantic_spatial(S, P):
    d_sem = pdist(S, metric='euclidean')
    d_spa = pdist(P, metric='euclidean')
    rho, _ = spearmanr(d_sem, d_spa)
    return float(np.clip(rho, -1.0, 1.0))

layouts = {
    "Ring": ring_positions,
    "Grid": grid_positions,
    "Spiral": spiral_positions,
    "TwoLayer": twolayer_ring_positions,
    "Random": None,
}

rows = []
for seed in range(N_SEEDS):
    S,_ = generate_semantics(N_NODES, dim=SEM_DIM, n_clusters=N_CLUSTERS, cluster_std=CLUSTER_STD, seed=seed)
    for name, func in layouts.items():
        if name == "Random":
            P = random_positions(N_NODES, seed=seed+1000)
            order = path_order_by_angle(P)
        else:
            P = func(N_NODES)
            order = path_order_by_angle(P)
        SP = sp_path_smoothness(P, order)
        GEN = gen_uniformity(P)
        VS = vs_spearman_semantic_spatial(S, P)
        rows.append({"layout":name,"seed":seed,"SP":SP,"GEN":GEN,"VS":VS})

df = pd.DataFrame(rows)
print(df.groupby("layout")[["SP","GEN","VS"]].mean().round(3))
df.to_csv(f"{OUTDIR}/E8_exp2_metrics.csv", index=False)

fig, axes = plt.subplots(1,3, figsize=(12,3.6))
for j,metric in enumerate(["SP","GEN","VS"]):
    data = [df[df["layout"]==name][metric].values for name in layouts.keys()]
    axes[j].boxplot(data, labels=list(layouts.keys()), showmeans=True)
    axes[j].set_title(metric)
    if metric=="VS":
        axes[j].axhline(0, color="k", linewidth=1, alpha=0.4)
    axes[j].grid(alpha=0.3, linestyle=":")
fig.suptitle("E8-Exp2: Layout Variants across Seeds", y=1.03)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp2_boxplots.png", dpi=200, bbox_inches="tight")
plt.show()
