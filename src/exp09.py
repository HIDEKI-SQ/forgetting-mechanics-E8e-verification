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
OUTDIR = "outputs/E8/exp9"
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

def gen(C, order, n_rand=30):
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
N = len(coords0)
values = np.linspace(0, 1, N)

def place_colocation(values, coords):
    order = np.argsort(values)
    C = coords.copy()
    C_sorted = C[np.argsort(np.lexsort((C[:, 1], C[:, 0])))]
    out = np.zeros_like(C)
    out[order] = C_sorted
    return out

def place_anti(values, coords):
    order_hi = np.argsort(values)[::-1]
    order_lo = np.argsort(values)
    inter = []
    for a, b in zip(order_hi, order_lo):
        inter += [a, b]
    # 重複除去（順序保持）
    seen = set()
    uniq = []
    for x in inter:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    C = coords.copy()
    C_sorted = C[np.argsort(np.lexsort((C[:, 1], C[:, 0])))]
    out = np.zeros_like(C)
    out[uniq[:len(C)]] = C_sorted
    return out

def place_neutral(coords):
    return coords.copy()

rows = []
for name, C in [
    ("co_location", place_colocation(values, coords0)),
    ("neutral", place_neutral(coords0)),
    ("anti_location", place_anti(values, coords0)),
]:
    order = greedy_route(C)
    rows.append((name, sp(bp, path_edges(order)), gen(C, order), vs(values, order)))

df = pd.DataFrame(rows, columns=["Condition", "SP", "GEN", "VS"])
print(df)
df.to_csv(f"{OUTDIR}/E8_exp09_metrics.csv", index=False)

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
x = np.arange(len(df))
w = 0.28
ax.bar(x - w, df["SP"], width=w, label="SP")
ax.bar(x, df["GEN"], width=w, label="GEN")
ax.bar(x + w, df["VS"], width=w, label="VS")
ax.set_xticks(x)
ax.set_xticklabels(df["Condition"], rotation=10)
ax.set_ylim(0, 1)
ax.set_title("EXP09: Value Co-location vs Anti-location")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp09_colocation.png", dpi=200)
plt.show()
