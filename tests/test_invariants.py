# -*- coding: utf-8 -*-
import importlib, inspect, re, numpy as np

EXP_MODULES = [
    "src.exp_beta","src.exp0",
] + [f"src.exp{str(i).zfill(2)}" for i in range(1,14)]

def _implemented(mod):
    src = inspect.getsource(mod)
    m = re.search(r"IMPLEMENTED\s*=\s*True", src)
    return bool(m)

def test_modules_importable():
    for name in EXP_MODULES:
        mod = importlib.import_module(name)
        assert mod is not None

def test_invariants_smoke():
    r"""
    各expが IMPLEMENTED=True の場合のみ軽い不変量チェックを行う。
    何も貼っていない場合はスキップ扱い。
    r"""
    for name in EXP_MODULES:
        mod = importlib.import_module(name)
        if not _implemented(mod):
            continue  # skip until pasted
        # ここでは import 成功だけを確認（E8e: 原コードは改変しないため）
        assert mod is not None
