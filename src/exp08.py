# -*- coding: utf-8 -*-
IMPLEMENTED = True

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr

np.random.seed(20251105)
OUTDIR = "outputs/E8/exp8"
os.makedirs(OUTDIR, exist_ok=True)

def make_grid(n_side=8):
    N = n_side * n_side
    coords = []
    edges = set()
    for i in range(n_side):
        for j in range(n_side):
            idx = i * n_side + j
            coords.append([i, j])
            if i + 1 < n_side:
                edges.add(tuple(sorted((idx, (i + 1) * n_side + j))))
            if j + 1 < n_side:
                edges.add(tuple(sorted((idx, i * n_side + (j + 1)))))
    coords = np.array(coords, float)
    coords = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-9)
    return coords, edges

def greedy_route(coords):
    N = len(coords)
    remaining = list(range(N))
    start = 0
    order = [start]
    remaining.remove(start)
    while remaining:
        last = order[-1]
        rem = np.array(remaining)
        d = np.linalg.norm(coords[rem] - coords[last], axis=1)
        nxt = rem[np.argmin(d)]
        order.append(int(nxt))
        remaining.remove(int(nxt))
    return order

def path_edges(order):
    return {tuple(sorted((order[k], order[k + 1]))) for k in range(len(order) - 1)}

def sp_from_path(bp_edges, path_edges_set):
    return len(bp_edges & path_edges_set) / max(1, len(path_edges_set))

def vs_from_values(values, order):
    ranks = np.empty(len(order))
    ranks[order] = np.arange(len(order))
    rho, _ = spearmanr(values, -ranks)
    return float(abs(rho))

def gen_from_path(coords, order, n_rand=30):
    N = len(coords)
    def path_len(ordr):
        return float(np.sum(np.linalg.norm(coords[ordr[1:]] - coords[ordr[:-1]], axis=1)))
    L = path_len(order)
    base = list(range(N))
    Lr = []
    for _ in range(n_rand):
        np.random.shuffle(base)
        Lr.append(path_len(base))
    Lr = float(np.mean(Lr))
    D = squareform(pdist(coords))
    mst = minimum_spanning_tree(D).toarray()
    Lmst = float(mst[mst > 0].sum())
    Lmin = 1.2 * Lmst
    gen = (Lr - L) / max(1e-9, (Lr - Lmin))
    return float(np.clip(gen, 0, 1))

coords0, bp_edges = make_grid(n_side=8)
values = np.random.rand(len(coords0))

S = np.array([[1.0, 0.5], [0.0, 1.0]])
coords_shear = coords0 @ S.T
coords_shear = (coords_shear - coords_shear.min(0)) / (coords_shear.max(0) - coords_shear.min(0) + 1e-9)

rows = []
for name, C in [("isometry", coords0), ("shear", coords_shear)]:
    order = greedy_route(C)
    sp = sp_from_path(bp_edges, path_edges(order))
    vs = vs_from_values(values, order)
    gen = gen_from_path(C, order)
    rows.append((name, sp, gen, vs))

df = pd.DataFrame(rows, columns=["Condition", "SP", "GEN", "VS"])
print(df)
df.to_csv(f"{OUTDIR}/E8_exp08_metrics.csv", index=False)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
x = np.arange(len(df))
w = 0.28
ax.bar(x - w, df["SP"], width=w, label="SP")
ax.bar(x, df["GEN"], width=w, label="GEN")
ax.bar(x + w, df["VS"], width=w, label="VS")
ax.set_xticks(x)
ax.set_xticklabels(df["Condition"])
ax.set_ylim(0, 1)
ax.set_title("EXP08: Topology vs Metric Distortion")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp08_metric_distortion.png", dpi=200)
plt.show()
