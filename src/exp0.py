# -*- coding: utf-8 -*-
# E8e 実験コード（exp0）
# このファイルは Colab で実際に使用した実験コードをそのまま保存しています。
# 検証目的のため、コードの動作や改行・空白は変更しません。

IMPLEMENTED = True

# --- 以下、Colabで実行した内容をそのまま貼り付け ---
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- replace simulate: robust VS (semantic↔spatial distance correlation) ---
def simulate_bp_space_robust(n_nodes=50, spatial=True, seed=42):
    rng = np.random.default_rng(seed)
    if spatial:
        x = np.linspace(0, 1, n_nodes)
        y = np.sin(2*np.pi*x)*0.2 + 0.5
    else:
        x, y = rng.random(n_nodes), rng.random(n_nodes)

    cat = rng.integers(0, 5, size=n_nodes)
    strength = rng.normal(0, 1, size=n_nodes)
    S = np.stack([cat, strength], axis=1)

    D_spatial = squareform(pdist(np.stack([x,y], axis=1), metric='euclidean'))
    D_semantic = squareform(pdist(S, metric='euclidean'))

    sp = float(np.clip(0.6 + 0.3*np.exp(-np.var(np.diff(x,2))), 0, 1))
    gen = float(np.clip(0.4 + 0.4*(1 - np.std(y)), 0, 1))
    rho, _ = spearmanr(D_semantic.ravel(), D_spatial.ravel())
    vs = float(np.clip(rho, -1, 1))
    return dict(SP=sp, GEN=gen, VS=vs)

def run_conditions(n_seeds=30):
    rows=[]
    for cond in ['spatial','random']:
        vals = {'SP':[], 'GEN':[], 'VS':[]}
        for s in range(n_seeds):
            res = simulate_bp_space_robust(spatial=(cond=='spatial'), seed=100+s)
            for k in vals:
                vals[k].append(res[k])
        rows.append(dict(Condition=cond, SP=np.mean(vals['SP']),
                         GEN=np.mean(vals['GEN']), VS=np.mean(vals['VS'])))
    return rows

rows = run_conditions()
df = pd.DataFrame(rows)
print(df)

os.makedirs("outputs/E8", exist_ok=True)
df.to_csv("outputs/E8/E8_metrics.csv", index=False)

plt.figure(figsize=(7,5))
bar_w=0.25; X=np.arange(len(df))
plt.bar(X-bar_w, df['SP'], width=bar_w, label='SP')
plt.bar(X, df['GEN'], width=bar_w, label='GEN')
plt.bar(X+bar_w, np.abs(df['VS']), width=bar_w, label='|VS|')
plt.xticks(X, df['Condition']); plt.ylim(0,1)
plt.ylabel('Normalized Value')
plt.title('E8: Spatialization vs Randomization')
plt.legend(frameon=False); plt.grid(axis='y', ls=':', alpha=.4)
plt.tight_layout()
plt.savefig("outputs/E8/fig_E8_exp0.png", dpi=300, bbox_inches='tight')
plt.show()
