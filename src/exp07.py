# -*- coding: utf-8 -*-
IMPLEMENTED = True

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

np.random.seed(7)
OUTDIR = "outputs/E8/exp7"
os.makedirs(OUTDIR, exist_ok=True)

def ring_coords(n):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.c_[np.cos(t), np.sin(t)]

def pair_corr(X, Y):
    iu = np.triu_indices_from(X, k=1)
    return spearmanr(X[iu], Y[iu]).correlation

def mantel_rho(X, Y, nperm=500):
    iu = np.triu_indices_from(X, k=1)
    x, y = X[iu], Y[iu]
    rho_obs = spearmanr(x, y).correlation
    cnt = 0
    for _ in range(nperm):
        p = np.random.permutation(len(y))
        if spearmanr(x, y[p]).correlation >= rho_obs:
            cnt += 1
    pval = (cnt + 1) / (nperm + 1)
    return rho_obs, pval

def hsic_rbf(X, Y, sigma=None):
    iu = np.triu_indices_from(X, k=1)
    x, y = X[iu], Y[iu]
    if sigma is None:
        sigma = np.median(x)
    Kx = np.exp(-(x[:, None] - x[None, :])**2 / (2 * sigma**2))
    Ky = np.exp(-(y[:, None] - y[None, :])**2 / (2 * sigma**2))
    H = np.eye(len(x)) - np.ones((len(x), len(x))) / len(x)
    HSIC = np.trace(H @ Kx @ H @ Ky) / (len(x) - 1)**2
    return HSIC

def RV_coeff(X, Y):
    iu = np.triu_indices_from(X, k=1)
    x, y = X[iu], Y[iu]
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y) + 1e-12))

N = 60
coords_spatial = ring_coords(N)
coords_random = np.random.randn(N, 2)
coords_random /= np.linalg.norm(coords_random, axis=1, keepdims=True)
phi = np.linspace(0, 2*np.pi, N, endpoint=False)
sem_coords = np.c_[np.cos(phi), np.sin(phi)]

D_space_sp = squareform(pdist(coords_spatial))
D_space_rd = squareform(pdist(coords_random))
D_sem = squareform(pdist(sem_coords))

rows = []
for cond, D in [("spatial", D_space_sp), ("random", D_space_rd)]:
    rho_s = pair_corr(D, D_sem)
    rho_m, p_m = mantel_rho(D, D_sem, nperm=300)
    hs = hsic_rbf(D, D_sem)
    rv = RV_coeff(D, D_sem)
    rows.append((cond, rho_s, rho_m, p_m, hs, rv))

df = pd.DataFrame(rows, columns=["Condition", "Spearman", "Mantel_rho", "Mantel_p", "HSIC", "RV"])
print("=== EXP07 Independence Metrics ===")
print(df.to_string(index=False))
df.to_csv(f"{OUTDIR}/E8_exp07_metrics.csv", index=False)

plt.figure(figsize=(6, 4))
x = np.arange(2)
W = 0.35
plt.bar(x - W/2, np.abs(df["Spearman"]), width=W, label="|Spearman|")
plt.bar(x + W/2, np.abs(df["RV"]), width=W, label="|RV|")
plt.xticks(x, df["Condition"])
plt.ylim(0, 0.3)
plt.title("EXP07: Independence audit (smaller = more independent)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp07_independence.png", dpi=200)
plt.show()
