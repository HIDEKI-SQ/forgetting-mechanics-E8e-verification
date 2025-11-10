# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

def test_empirical_properties_basics():
    # O1: ランダム独立 → Spearman ≈ 0（合成データで傾向確認）
    rng = np.random.default_rng(0)
    X = rng.normal(0,1,(60,2))
    Y = rng.normal(0,1,(60,2))
    r,_ = spearmanr(pdist(X), pdist(Y))
    assert abs(r) < 0.2  # 傾向レベルのゆるい検査（E8e: 忠実保存）
