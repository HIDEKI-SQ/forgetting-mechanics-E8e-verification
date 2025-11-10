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
OUTDIR = "outputs/E8/exp12"
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

def detour(order, frac=0.1):
    N = len(order)
    out = [order[0]]
    for i in range(1, N):
        out.append(order[i])
        if np.random.rand() < frac and i + 1 < N:
            out.append(order[i - 1])
    comp = [out[0]]
    for x in out[1:]:
        if x != comp[-1]:
            comp.append(x)
    seen = set()
    uniq = []
    for x in comp:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def teleport(order, frac=0.1):
    N = len(order)
    pool = list(order)
    out = [pool[0]]
    used = {pool[0]}
    rng = np.random.default_rng(20251105)
    for _ in range(1, N):
        if rng.random() < frac:
            candidates = [x for x in pool if x not in used]
            if candidates:
                nxt = rng.choice(candidates)
                out.append(int(nxt))
                used.add(int(nxt))
        else:
            candidates = [x for x in pool if x not in used]
            if candidates:
                out.append(candidates[0])
                used.add(candidates[0])
    return out

def shuffle_blocks(order, frac=0.1, block=4):
    N = len(order)
    out = order.copy()
    idx = list(range(0, N - block, block))
    k = int(frac * len(idx))
    pick = np.random.choice(idx, size=max(0, k), replace=False)
    for s in pick:
        out[s:s + block] = out[s:s + block][::-1]
    return out

coords0, bp = make_grid(8)
values = np.random.rand(len(coords0))
base_order = greedy_route(coords0)
levels = [0.0, 0.05, 0.1, 0.2, 0.3]

rows = []
for r in levels:
    o = detour(base_order, frac=r)
    rows.append(("detour", r, sp(bp, path_edges(o)), gen(coords0, o), vs(values, o)))
for k in levels:
    o = teleport(base_order, frac=k)
    rows.append(("teleport", k, sp(bp, path_edges(o)), gen(coords0, o), vs(values, o)))
for s in levels:
    o = shuffle_blocks(base_order, frac=s, block=4)
    rows.append(("shuffle", s, sp(bp, path_edges(o)), gen(coords0, o), vs(values, o)))

df = pd.DataFrame(rows, columns=["Perturb", "level", "SP", "GEN", "VS"])
df.to_csv(f"{OUTDIR}/E8_exp12_metrics.csv", index=False)
print(df.head())

fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for name, sty in [("detour", "o"), ("teleport", "s"), ("shuffle", "^")]:
    sub = df[df["Perturb"] == name].sort_values("level")
    axs[0].plot(sub["level"], sub["SP"], marker=sty, label=f"{name} - SP")
    axs[1].plot(sub["level"], sub["VS"], marker=sty, label=f"{name} - VS")
axs[0].set_title("EXP12: SP vs perturbation level")
axs[0].set_xlabel("level")
axs[0].set_ylim(0, 1)
axs[0].grid(alpha=0.3)
axs[0].legend()
axs[1].set_title("EXP12: VS vs perturbation level")
axs[1].set_xlabel("level")
axs[1].set_ylim(0, 1)
axs[1].grid(alpha=0.3)
axs[1].legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp12_perturbations.png", dpi=200)
plt.show()
