"""
data/processed/comments.csv から、各トピックにつき指定件数を層化ランダムサンプリングし、
data/processed/comments_sample.csv に書き出す。

全件（約66,000件）をローカルLLM（qwen2.5:7b-instruct）でコーディングすると計算時間が
非現実的（実測約7.4秒/件 → 全件で約136時間）なため、卒論第4部「分析2」では
トピックごとの代表サンプルを対象にコーディングする。サンプルサイズ・乱数シードは
卒論本文の方法論記述にそのまま転記できるよう、ここに明記する。

使い方:
    python src/sample_comments.py --per-topic 2000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ANALYSIS_ROOT / "data" / "processed" / "comments.csv"
OUTPUT_PATH = ANALYSIS_ROOT / "data" / "processed" / "comments_sample.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-topic", type=int, default=2000, help="トピックごとのサンプルサイズ")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（再現性のため固定）")
    args = parser.parse_args()

    df = pd.read_csv(INPUT_PATH)

    parts = []
    for topic, group in df.groupby("topic"):
        n = min(args.per_topic, len(group))
        parts.append(group.sample(n=n, random_state=args.seed))
    sample = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"サンプル件数: {len(sample)} 件（seed={args.seed}）")
    print(sample["topic"].value_counts())
    print(f"出力: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
