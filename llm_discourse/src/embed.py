"""
コメントを文埋め込み（sentence embedding）ベクトルに変換する。

クラスタリング（KMeans等）や次元圧縮（UMAP等）による可視化、
「似た言い回しのコメントが複数トピックにまたがって出現するか」といった
探索的分析は、このベクトルを使って notebooks/ 側で行う想定。
本スクリプトはベクトル生成までを担当する。

使い方:
    python src/embed.py                                   # comments.csv 全件
    python src/embed.py --input data/processed/comments_sample.csv \
        --output-prefix sample                             # label_ollama.py と同じサンプルのみ
    python src/embed.py --device cpu                        # GPUを使わずCPUで実行
                                                              # （label_ollama.pyがGPUを使用中の場合）

出力（--output-prefixを指定した場合はファイル名に "{prefix}_" が付く）:
    data/processed/embeddings.npy       -- (N, D) の埋め込み行列
    data/processed/embeddings_ids.csv   -- 各行がどの comment_id に対応するかの対応表
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = ANALYSIS_ROOT / "data" / "processed" / "comments.csv"
RESULTS_DIR = ANALYSIS_ROOT / "data" / "processed"

# 多言語対応・日本語の実績が比較的多いモデル。
# ローカルCPUでも動くが、件数が多い場合はGPUがあると大幅に速い。
MODEL_NAME = "intfloat/multilingual-e5-large"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH,
                         help="入力CSVのパス（省略時はcomments.csv全件）")
    parser.add_argument("--output-prefix", default=None,
                         help="出力ファイル名の接頭辞（例: sample -> sample_embeddings.npy）")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="計算デバイス。省略時はsentence-transformersが自動選択（GPUがあれば使う）。"
                              "label_ollama.py等がGPUを使用中の場合はcpuを明示すること。")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} がありません。先に preprocess.py を実行してください。")

    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    embeddings_path = RESULTS_DIR / f"{prefix}embeddings.npy"
    ids_path = RESULTS_DIR / f"{prefix}embeddings_ids.csv"

    df = pd.read_csv(args.input)
    print(f"{len(df)} 件のコメントを埋め込みます（モデル: {MODEL_NAME}, device={args.device or 'auto'}）")

    model = SentenceTransformer(MODEL_NAME, device=args.device)
    # multilingual-e5 系は "query: " / "passage: " プレフィックスを付けると精度が上がる
    texts = [f"passage: {t}" for t in df["text_clean"].fillna("").tolist()]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    df[["comment_id", "topic", "video_id"]].to_csv(ids_path, index=False, encoding="utf-8-sig")

    print(f"書き出し: {embeddings_path} (shape={embeddings.shape})")
    print(f"書き出し: {ids_path}")


if __name__ == "__main__":
    main()
