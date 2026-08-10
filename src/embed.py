"""
コメントを文埋め込み（sentence embedding）ベクトルに変換する。

クラスタリング（KMeans等）や次元圧縮（UMAP等）による可視化、
「似た言い回しのコメントが複数トピックにまたがって出現するか」といった
探索的分析は、このベクトルを使って notebooks/ 側で行う想定。
本スクリプトはベクトル生成までを担当する。

使い方:
    python src/embed.py

出力:
    data/processed/embeddings.npy       -- (N, D) の埋め込み行列
    data/processed/embeddings_ids.csv   -- 各行がどの comment_id に対応するかの対応表
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ANALYSIS_ROOT / "data" / "processed" / "comments.csv"
EMBEDDINGS_PATH = ANALYSIS_ROOT / "data" / "processed" / "embeddings.npy"
IDS_PATH = ANALYSIS_ROOT / "data" / "processed" / "embeddings_ids.csv"

# 多言語対応・日本語の実績が比較的多いモデル。
# ローカルCPUでも動くが、件数が多い場合はGPUがあると大幅に速い。
MODEL_NAME = "intfloat/multilingual-e5-large"


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH} がありません。先に preprocess.py を実行してください。")

    df = pd.read_csv(INPUT_PATH)
    print(f"{len(df)} 件のコメントを埋め込みます（モデル: {MODEL_NAME}）")

    model = SentenceTransformer(MODEL_NAME)
    # multilingual-e5 系は "query: " / "passage: " プレフィックスを付けると精度が上がる
    texts = [f"passage: {t}" for t in df["text_clean"].fillna("").tolist()]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    df[["comment_id", "topic", "video_id"]].to_csv(IDS_PATH, index=False, encoding="utf-8-sig")

    print(f"書き出し: {EMBEDDINGS_PATH} (shape={embeddings.shape})")
    print(f"書き出し: {IDS_PATH}")


if __name__ == "__main__":
    main()
