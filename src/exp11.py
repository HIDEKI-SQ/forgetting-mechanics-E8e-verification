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
OUTDIR = "outputs/E8/exp11"
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

def greedy_route(C):
    N = len(C)
    rem = list(range(N))
    start = 0
    order = [start]
    rem.remove(start)
    while rem:
        last = order[-1]
        rr = np.array(rem)
        d = np.linalg.norm(C[rr] - C[last], axis=1)
        nxt = int(rr[np.argmin(d)])
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

coords0, bp0 = make_grid(8)
N = len(coords0)
values = np.random.rand(N)

def node_drop(coords, bp, drop_frac):
    keep = np.sort(np.random.choice(len(coords), size=int((1 - drop_frac) * len(coords)), replace=False))
    idx_map = {int(k): i for i, k in enumerate(keep)}
    C = coords[keep]
    bp_kept = {e for e in bp if (e[0] in idx_map and e[1] in idx_map)}
    bp_kept = {tuple(sorted((idx_map[e[0]], idx_map[e[1]]))) for e in bp_kept}
    return C, bp_kept, keep

def edge_rewire(bp, rewire_frac, N):
    bp_list = list(bp)
    M = len(bp_list)
    k = int(rewire_frac * M)
    keep_idx = np.setdiff1d(np.arange(M), np.random.choice(M, size=k, replace=False))
    new_edges = set([bp_list[i] for i in keep_idx])
    attempts = 0
    while len(new_edges) < M and attempts < 10 * M:
        a, b = np.random.choice(N, 2, replace=False)
        e = tuple(sorted((int(a), int(b))))
        if e not in new_edges:
            new_edges.add(e)
        attempts += 1
    return new_edges

drop_list = [0.0, 0.1, 0.2, 0.3, 0.4]
rewire_list = [0.0, 0.1, 0.2, 0.3, 0.4]
rows_drop = []
for p in drop_list:
    C, bpK, keep = node_drop(coords0, bp0, p)
    vals = values[keep]
    order = greedy_route(C)
    rows_drop.append(("node_drop", p, sp(bpK, path_edges(order)), gen(C, order), vs(vals, order)))

rows_rew = []
for q in rewire_list:
    bpR = edge_rewire(bp0, q, N)
    C = coords0.copy()
    order = greedy_route(C)
    rows_rew.append(("edge_rewire", q, sp(bpR, path_edges(order)), gen(C, order), vs(values, order)))

dfD = pd.DataFrame(rows_drop, columns=["Condition", "level", "SP", "GEN", "VS"])
dfR = pd.DataFrame(rows_rew, columns=["Condition", "level", "SP", "GEN", "VS"])
dfD.to_csv(f"{OUTDIR}/E8_exp11_node_drop.csv", index=False)
dfR.to_csv(f"{OUTDIR}/E8_exp11_edge_rewire.csv", index=False)
print(dfD)
print(dfR)

fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
axs[0].plot(dfD["level"], dfD["SP"], marker='o', label="SP")
axs[0].plot(dfD["level"], dfD["VS"], marker='^', label="VS")
axs[0].set_title("EXP11: Node Drop")
axs[0].set_xlabel("drop fraction")
axs[0].set_ylim(0, 1)
axs[0].grid(alpha=0.3)
axs[0].legend()

axs[1].plot(dfR["level"], dfR["SP"], marker='o', label="SP")
axs[1].plot(dfR["level"], dfR["VS"], marker='^', label="VS")
axs[1].set_title("EXP11: Edge Rewire")
axs[1].set_xlabel("rewire fraction")
axs[1].grid(alpha=0.3)
axs[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp11_drop_rewire.png", dpi=200)
plt.show()
