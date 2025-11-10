# -*- coding: utf-8 -*-
IMPLEMENTED = True

import os, math
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import rbf_kernel

np.random.seed(7)
OUTDIR = "outputs/E8/exp3"
os.makedirs(OUTDIR, exist_ok=True)
N_NODES = 60
SEM_DIM = 8
N_CLUSTERS = 3
CLUSTER_STD = 0.6
N_PERM = 1000

def generate_semantics(n_nodes, dim=8, n_clusters=3, cluster_std=0.6, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    centers = rng.normal(0, 1.0, size=(n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n_nodes)
    S = centers[labels] + rng.normal(0, cluster_std, size=(n_nodes, dim))
    return S

def ring_positions(n_nodes, radius=1.0):
    theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y], axis=1)

def random_positions(n_nodes):
    rng = np.random.default_rng(1234)
    return rng.uniform(-1.0, 1.0, size=(n_nodes, 2))

def spearman_condensed(d1, d2):
    rho, _ = spearmanr(d1, d2)
    return float(rho)

def mantel_test(D1, D2, n_perm=1000, method="spearman", rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    v1 = squareform(D1, checks=False)
    v2 = squareform(D2, checks=False)
    if method == "spearman":
        stat = spearman_condensed(v1, v2)
        corr_func = lambda a, b: spearman_condensed(a, b)
    else:
        stat, _ = pearsonr(v1, v2)
        corr_func = lambda a, b: pearsonr(a, b)[0]
    N = D1.shape[0]
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(N)
        D2p = D2[perm][:, perm]
        vp = squareform(D2p, checks=False)
        if abs(corr_func(v1, vp)) >= abs(stat):
            ge += 1
    pval = (ge + 1) / (n_perm + 1)
    return float(stat), float(pval)

def hsic_unbiased(x, y, sigma_x=None, sigma_y=None):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.shape[0] < x.shape[1]:
        x = x.T
    if y.shape[0] < y.shape[1]:
        y = y.T
    n = x.shape[0]
    if sigma_x is None:
        dx = pdist(x, metric='euclidean')
        sigma_x = np.median(dx) + 1e-12
    if sigma_y is None:
        dy = pdist(y, metric='euclidean')
        sigma_y = np.median(dy) + 1e-12
    K = rbf_kernel(x, x, gamma=1.0 / (2 * sigma_x ** 2))
    L = rbf_kernel(y, y, gamma=1.0 / (2 * sigma_y ** 2))
    H = np.eye(n) - np.ones((n, n)) / n
    KH = K @ H
    LH = L @ H
    hsic = (1.0 / (n - 1) ** 2) * np.trace(KH @ LH)
    return float(hsic)

def rv_coefficient(v1, v2):
    v1c = v1 - v1.mean()
    v2c = v2 - v2.mean()
    num = float(np.dot(v1c, v2c))
    den = float(np.linalg.norm(v1c) * np.linalg.norm(v2c) + 1e-12)
    return float(num / den)

S = generate_semantics(N_NODES, dim=SEM_DIM, n_clusters=N_CLUSTERS, cluster_std=CLUSTER_STD)
P_spa = ring_positions(N_NODES, radius=1.0)
P_rnd = random_positions(N_NODES)

Dsem = squareform(pdist(S, metric='euclidean'))
Dspa = squareform(pdist(P_spa, metric='euclidean'))
Drnd = squareform(pdist(P_rnd, metric='euclidean'))
v_sem = squareform(Dsem, checks=False)
v_spa = squareform(Dspa, checks=False)
v_rnd = squareform(Drnd, checks=False)

rows = []
for name, v_sp, Dsp in [("spatial", v_spa, Dspa), ("random", v_rnd, Drnd)]:
    rho = spearman_condensed(v_sem, v_sp)
    m_stat, m_p = mantel_test(Dsem, Dsp, n_perm=N_PERM, method="spearman")
    hs = hsic_unbiased(v_sem.reshape(-1, 1), v_sp.reshape(-1, 1))
    rv = rv_coefficient(v_sem, v_sp)
    rows.append({
        "Condition": name,
        "Spearman": rho,
        "Mantel_rho": m_stat,
        "Mantel_p": m_p,
        "HSIC": hs,
        "RV": rv
    })

df = pd.DataFrame(rows)
print("=== E8 EXP3 Dependence Metrics ===")
print(df.to_string(index=False))
df.to_csv(f"{OUTDIR}/E8_exp3_dependency_table.csv", index=False)
