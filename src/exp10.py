# -*- coding: utf-8 -*-
IMPLEMENTED = True

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
OUTDIR = "outputs/E8/exp10"
os.makedirs(OUTDIR, exist_ok=True)

def ring_coords(n):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.c_[np.cos(t), np.sin(t)]

def greedy_route(coords):
    ang = np.arctan2(coords[:,1], coords[:,0])
    return np.argsort(ang)

def path_edges(order):
    return list(zip(order, np.roll(order, -1)))

def sp(coords, edges):
    P = coords
    angles = []
    for (i, j), (j2, k) in zip(edges, edges[1:] + edges[:1]):
        v1 = P[j] - P[i]
        v2 = P[k] - P[j]
        c = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1, 1)
        ang = np.arccos(c) / np.pi
        angles.append(ang)
    return 1.0 - float(np.mean(angles))

def gen(coords, order):
    P = coords[order]
    d = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
    d = d / np.max(d)
    return 1.0 - float(np.std(d))

def vs(values, order):
    idx = np.arange(len(order))
    v = values[order]
    v = (v - v.mean()) / (v.std() + 1e-8)
    idx = (idx - idx.mean()) / (idx.std() + 1e-8)
    return float(abs(np.corrcoef(idx, v)[0,1]))

def apply_overlap(coords, frac):
    C = coords.copy()
    N = len(C)
    pairs = int(np.floor(frac * N / 2))
    if pairs == 0:
        return C
    perm = np.random.permutation(N)
    A, B = perm[:pairs], perm[pairs:2*pairs]
    jitter = 1e-3 * np.random.randn(pairs, 2)
    C[A] = C[B] + jitter
    return C

N = 36
coords0 = ring_coords(N)
values = np.linspace(-1, 1, N)
base_order = greedy_route(coords0)
fracs = [0.0, 0.25, 0.5, 0.75]

rows = []
for f in fracs:
    C = apply_overlap(coords0, frac=f)
    order = greedy_route(C)
    rows.append((f, sp(C, path_edges(order)), gen(C, order), vs(values, order)))

df = pd.DataFrame(rows, columns=["overlap_frac", "SP", "GEN", "VS"])
print("=== EXP10 Metrics ===")
print(df.to_string(index=False))
df.to_csv(f"{OUTDIR}/E8_exp10_metrics.csv", index=False)

plt.figure(figsize=(7, 4))
w = 0.25
x = np.arange(len(fracs))
plt.bar(x - w, df["SP"], width=w, label="SP")
plt.bar(x, df["GEN"], width=w, label="GEN")
plt.bar(x + w, df["VS"], width=w, label="VS")
plt.xticks(x, fracs)
plt.ylim(0, 1.05)
plt.title("EXP10: Overlap / Interference Stress")
plt.xlabel("overlap fraction")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp10_overlap.png", dpi=200)
plt.show()
