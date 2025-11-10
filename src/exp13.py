# -*- coding: utf-8 -*-
IMPLEMENTED = True

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr

np.random.seed(20251105)
OUTDIR = "outputs/E8/exp13"
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

def greedy_value_gated(coords, values, lam=0.0):
    N = len(coords)
    rem = list(range(N))
    start = 0
    order = [start]
    rem.remove(start)
    D = cdist(coords, coords)
    while rem:
        last = order[-1]
        rr = np.array(rem)
        cost = D[last, rr] - lam * values[rr]
        nxt = int(rr[np.argmin(cost)])
        order.append(nxt)
        rem.remove(nxt)
    return order

def path_edges(order):
    return {tuple(sorted((order[k], order[k + 1]))) for k in range(len(order) - 1)}

def sp(bp, pe):
    return len(bp & pe) / max(1, len(pe))

def vs(values, order):
    ranks = np.empty(len(order))
    ranks[order] = np.arange(len(order))
    rho, _ = spearmanr(values, -ranks)
    return float(abs(rho))

def gen(C, order, n_rand=20):
    def L(ordr):
        return float(np.sum(np.linalg.norm(C[ordr[1:]] - C[ordr[:-1]], axis=1)))
    Lp = L(order)
    base = list(range(len(C)))
    Lr = []
    for _ in range(n_rand):
        np.random.shuffle(base)
        Lr.append(L(base))
    Lr = float(np.mean(Lr))
    D = squareform(pdist(C))
    mst = minimum_spanning_tree(D).toarray()
    Lmin = 1.2 * float(mst[mst > 0].sum())
    return float(np.clip((Lr - Lp) / max(1e-9, (Lr - Lmin)), 0, 1))

coords0, bp = make_grid(8)
values = np.random.rand(len(coords0))
lams = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

rows = []
for lam in lams:
    order = greedy_value_gated(coords0, values, lam=lam)
    rows.append((lam, sp(bp, path_edges(order)), gen(coords0, order), vs(values, order)))

df = pd.DataFrame(rows, columns=["lambda", "SP", "GEN", "VS"])
print(df)
df.to_csv(f"{OUTDIR}/E8_exp13_metrics.csv", index=False)

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.plot(df["lambda"], df["SP"], marker="o", label="SP")
ax.plot(df["lambda"], df["VS"], marker="s", label="VS")
ax.plot(df["lambda"], df["GEN"], marker="^", label="GEN")
ax.set_xlabel("lambda (value weight)")
ax.set_ylim(0, 1)
ax.set_title("EXP13: Value-gated Retrieval")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp13_lambda.png", dpi=200)
plt.show()
