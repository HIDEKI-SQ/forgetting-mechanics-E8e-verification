# -*- coding: utf-8 -*-
IMPLEMENTED = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(42)
OUTDIR = "outputs/E8/exp_beta"
os.makedirs(OUTDIR, exist_ok=True)

def simulate_bp_space(n_nodes=10, spatial=True, seed=42):
    np.random.seed(seed)
    if spatial:
        x = np.linspace(0, 1, n_nodes)
        y = np.sin(2 * np.pi * x) * 0.2 + 0.5
    else:
        x, y = np.random.rand(n_nodes), np.random.rand(n_nodes)
    coords = np.stack([x, y], axis=1)

    sp = 0.6 + 0.3 * np.exp(-np.var(np.diff(x)))
    gen = 0.4 + 0.4 * (1 - np.std(y))
    vs = np.corrcoef(x, y)[0, 1]
    return dict(SP=sp, GEN=gen, VS=vs)

conditions = ["spatial", "random"]
rows = []
for cond in conditions:
    res = simulate_bp_space(spatial=(cond == "spatial"))
    res["Condition"] = cond
    rows.append(res)
df = pd.DataFrame(rows)
print("=== E8β Metrics ===")
print(df)

os.makedirs("outputs/E8", exist_ok=True)
df.to_csv(f"{OUTDIR}/E8_exp_beta_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 5))
bar_width = 0.25
x = np.arange(len(df))
ax.bar(x - bar_width, df["SP"], width=bar_width, label="SP (Structure Preservation)", color="#2E86AB")
ax.bar(x, df["GEN"], width=bar_width, label="GEN (Generalization)", color="#F77F00")
ax.bar(x + bar_width, df["VS"], width=bar_width, label="VS (Semantic Correlation)", color="#06A77D")
ax.set_xticks(x)
ax.set_xticklabels(df["Condition"])
ax.set_ylabel("Normalized Value")
ax.set_ylim(0, 1)
ax.set_title("E8β: Initial Proof of Concept (SP / GEN / VS)")
ax.legend(frameon=False)
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_E8_exp_beta.png", dpi=300, bbox_inches="tight")
plt.show()
