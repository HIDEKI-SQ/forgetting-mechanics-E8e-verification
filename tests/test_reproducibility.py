# -*- coding: utf-8 -*-
import importlib, inspect, re, os, glob, hashlib

OUTDIR = "outputs"

def list_outputs():
    if not os.path.isdir(OUTDIR):
        return []
    files = []
    for ext in ("*.csv","*.png","*.json"):
        files += glob.glob(os.path.join(OUTDIR, "**", ext), recursive=True)
    return sorted(files)

def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def test_outputs_signature_smoke():
    # 生成物があれば、ハッシュが計算できる（可読性のため一覧化）
    files = list_outputs()
    for p in files:
        _ = sha256(p)  # 例外が起きないことだけ確認
