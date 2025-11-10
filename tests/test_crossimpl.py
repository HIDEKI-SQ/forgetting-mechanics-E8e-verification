# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

def _vs_ravel(X, Y):
    Dx = squareform(pdist(X))
    Dy = squareform(pdist(Y))
    return spearmanr(Dx.ravel(), Dy.ravel()).correlation

def _vs_condensed(X, Y):
    dx = pdist(X); dy = pdist(Y)
    return spearmanr(dx, dy).correlation

def test_ravel_vs_condensed_delta_small():
    # 合成データで差分が小さいことを確認（設計上の許容差）
    rng = np.random.default_rng(42)
    X = rng.normal(0,1,(50,2))
    Y = rng.normal(0,1,(50,2))
    v1 = _vs_ravel(X,Y)
    v2 = _vs_condensed(X,Y)
    assert abs(v1 - v2) < 0.05  # 実務上の許容差
