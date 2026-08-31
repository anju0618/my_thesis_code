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

差分インクリメント:
    既存の出力ファイル（{prefix}embeddings.npy / {prefix}embeddings_ids.csv）が
    存在する場合，そこに含まれる comment_id のうち今回の入力にも含まれるものは
    再利用し，新規の comment_id のみをエンコードする（label_ollama.py の
    comment_id単位スキップ，youtube_collect.py の動画単位スキップと同じ方針）。
    逆に，既存の埋め込みのうち今回の入力に含まれなくなった comment_id
    （例：comments_sample.csv をサンプルサイズ変更等で再抽出し，一部の
    コメントが入れ替わった場合）は破棄する——*_ids.csv が常に埋め込み行列と
    1対1対応する状態を保ち，分析側（analyze_coding.py）が古いサンプル構成の
    埋め込みを誤って使ってしまう事故を防ぐため。

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

    # 差分インクリメント：既存の埋め込みのうち今回の入力にも残っている
    # comment_id は再利用し，新規分のみをエンコードする。
    reused_embeddings = None
    reused_ids_df = None
    if embeddings_path.exists() and ids_path.exists():
        existing_embeddings = np.load(embeddings_path)
        existing_ids_df = pd.read_csv(ids_path)
        if len(existing_ids_df) != len(existing_embeddings):
            print(f"[WARN] 既存の{ids_path.name}（{len(existing_ids_df)}件）と"
                  f"{embeddings_path.name}（{len(existing_embeddings)}件）の件数が"
                  "一致しないため，差分インクリメントを行わず全件再計算します。")
        else:
            input_ids = set(df["comment_id"])
            keep_mask = existing_ids_df["comment_id"].isin(input_ids).values
            n_dropped = (~keep_mask).sum()
            reused_ids_df = existing_ids_df[keep_mask].reset_index(drop=True)
            reused_embeddings = existing_embeddings[keep_mask]
            if n_dropped:
                print(f"既存の埋め込みのうち，現在の入力{args.input.name}に"
                      f"含まれなくなった{n_dropped}件を破棄")

    if reused_ids_df is not None:
        todo_df = df[~df["comment_id"].isin(set(reused_ids_df["comment_id"]))].reset_index(drop=True)
    else:
        todo_df = df

    n_reused = len(reused_ids_df) if reused_ids_df is not None else 0
    print(f"入力: {len(df)}件 / 既存の埋め込みを再利用: {n_reused}件 / "
          f"今回新規にエンコード: {len(todo_df)}件"
          f"（モデル: {MODEL_NAME}, device={args.device or 'auto'}）")

    if len(todo_df) == 0:
        print("新規にエンコードすべきコメントはありません。既存の埋め込みをそのまま維持します。")
        return

    model = SentenceTransformer(MODEL_NAME, device=args.device)
    # multilingual-e5 系は "query: " / "passage: " プレフィックスを付けると精度が上がる
    texts = [f"passage: {t}" for t in todo_df["text_clean"].fillna("").tolist()]
    new_embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    new_ids_df = todo_df[["comment_id", "topic", "video_id"]].reset_index(drop=True)

    if reused_embeddings is not None and len(reused_embeddings) > 0:
        embeddings = np.concatenate([reused_embeddings, new_embeddings], axis=0)
        ids_df = pd.concat([reused_ids_df, new_ids_df], ignore_index=True)
    else:
        embeddings = new_embeddings
        ids_df = new_ids_df

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    ids_df.to_csv(ids_path, index=False, encoding="utf-8-sig")

    print(f"書き出し: {embeddings_path} (shape={embeddings.shape})")
    print(f"書き出し: {ids_path}")


if __name__ == "__main__":
    main()
