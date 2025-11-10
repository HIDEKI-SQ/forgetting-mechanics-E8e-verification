# E8e — Code Validity Verification (Skeleton)

[![E8e Verification](https://github.com/HIDEKI-SQ/forgetting-mechanics-E8e-verification/actions/workflows/test.yml/badge.svg)](../../actions)


このリポジトリは **E8e: コード妥当性の検証** のための最小骨格です。  
**ポイント**：実験コードは *そのまま* `src/` に貼り付け、テストは *外側* の `tests/` が行います（原コードは変更しません）。

## ディレクトリ構成
```
E8e-verification/
├── src/                   # Colabで実際に使った15本のコードを「そのまま」貼付
│   ├── exp_beta.py
│   ├── exp0.py
│   ├── exp01.py … exp13.py
│   └── __init__.py
├── tests/                 # 妥当性検査（外付けハーネス）
│   ├── test_invariants.py
│   ├── test_reproducibility.py
│   ├── test_crossimpl.py
│   └── test_empirical.py
├── outputs/               # 実行生成物（CSV/PNG等）
├── scripts/               # 補助スクリプト
│   └── run_all.sh
├── notebooks/             # 任意（検証ノートを置く）
├── requirements.txt
├── pyproject.toml         # 任意（使わなければ削除可）
├── LICENSE
└── .github/workflows/test.yml
```

## 使い方（最小）
1. `src/` の各ファイルに **Colabの実験コードをそのまま貼り付け**（exp_beta〜exp13）。
2.（任意）`src/exp*.py` の先頭に `IMPLEMENTED = True` を1行追記すると、該当実験のテストが有効化されます。
3. 依存をインストールしてテストを実行：
```bash
pip install -r requirements.txt
pytest -q
```
4. CI（GitHub Actions）は push 時に自動で走ります。

## テスト方針（概要）
- **不変量**：数値の有限性・範囲、距離ベクトルの長さ一致など。
- **再現性**：seed/乱数の固定があるとき、同一出力になるか。
- **交差検証**：`squareform(...).ravel()` と `pdist(...)` の Spearman が同傾向か。
- **経験的一致**：O1〜O4 に対応する“符号・傾向”が崩れていないか（厳密な閾値は課さない）。

> 注：この骨格は **E8e（忠実保存）** 用です。改良版は **E8f** として別ブランチ/別リポジトリで育ててください。
