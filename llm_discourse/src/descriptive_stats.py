"""
機械学習・埋め込みを一切介さない，最もシンプルな記述統計分析。

analyze_coding.py はカイ二乗検定（blame_target×topic）とクラスタリングを
行っているが，本スクリプトは (1) コメント投稿時期別の各変数出現率の推移
（トピックごとの盛り上がり時期との対応を素朴に確認する）と (2) 3変数
（people_vs_elite / economic_resentment / theft_of_enjoyment）それぞれに
ついてのトピック間カイ二乗検定を，シンプルなクロス集計として行う。

使い方:
    uv run src/descriptive_stats.py

出力:
    results/descriptive_timetrend.png       -- 月別出現率の推移（トピック別）
    results/descriptive_crosstab_summary.txt -- クロス集計＋カイ二乗検定
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Yu Gothic"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
LABELED_PATH = ANALYSIS_ROOT / "data" / "processed" / "labeled_comments.jsonl"
SAMPLE_PATH = ANALYSIS_ROOT / "data" / "processed" / "comments_sample.csv"
RESULTS_DIR = ANALYSIS_ROOT / "results"

BOOL_VARS = ["people_vs_elite", "economic_resentment", "theft_of_enjoyment"]
TOPIC_LABELS_JA = {
    "zaimusho_demo": "財務省デモ",
    "sanseito": "参政党",
    "immigration": "不法移民",
}


def load_labeled_with_date() -> pd.DataFrame:
    records = []
    with open(LABELED_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("parse_error"):
                continue
            records.append(d)
    df = pd.DataFrame(records)

    sample = pd.read_csv(SAMPLE_PATH)[["comment_id", "published_at"]]
    sample_ids = set(sample["comment_id"])
    df = df[df["comment_id"].isin(sample_ids)]
    df = df[df["is_spam_or_offtopic"] != True]  # noqa: E712
    df = df.merge(sample, on="comment_id", how="left")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["month"] = df["published_at"].dt.to_period("M").dt.to_timestamp()
    print(f"分析対象: {len(df)}件（投稿日時が判明: {df['published_at'].notna().sum()}件）")
    return df


def time_trend(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(BOOL_VARS), 1, figsize=(10, 10), sharex=True)
    for ax, var in zip(axes, BOOL_VARS):
        for topic in TOPIC_LABELS_JA:
            sub = df[df["topic"] == topic].dropna(subset=["month", var])
            if sub.empty:
                continue
            monthly = sub.groupby("month")[var].agg(["mean", "count"])
            monthly = monthly[monthly["count"] >= 10]
            if monthly.empty:
                continue
            ax.plot(monthly.index, monthly["mean"] * 100, marker="o", markersize=3,
                     label=TOPIC_LABELS_JA[topic])
        ax.set_ylabel(f"{var}\n出現率(%)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[-1].set_xlabel("投稿月（コメントが10件未満の月は非表示）")
    fig.suptitle("コメント投稿時期別の各変数出現率の推移（トピック別）")
    fig.tight_layout()
    out_path = RESULTS_DIR / "descriptive_timetrend.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"書き出し: {out_path}")


def crosstab_chi2(df: pd.DataFrame, out_lines: list[str]) -> None:
    out_lines.append(f"\n{'=' * 78}")
    out_lines.append("トピック × 各変数のクロス集計・カイ二乗検定（機械学習を介さない直接集計）")
    out_lines.append(f"{'=' * 78}")
    for var in BOOL_VARS:
        sub = df.dropna(subset=[var, "topic"])
        ct = pd.crosstab(sub["topic"], sub[var])
        out_lines.append(f"\n--- {var} ---")
        out_lines.append(ct.to_string())
        pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
        out_lines.append("\n該当率(%):")
        out_lines.append(pct.to_string())
        if ct.shape[1] >= 2 and (ct.values.sum(axis=0) > 0).all():
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            n = ct.values.sum()
            cramers_v = np.sqrt((chi2 / n) / (min(ct.shape) - 1))
            out_lines.append(f"カイ二乗統計量={chi2:.2f}  自由度={dof}  p値={p:.6f}"
                              f"{'  *有意*' if p < 0.05 else '  非有意'}  Cramér's V={cramers_v:.4f}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labeled_with_date()

    print("時系列トレンドを描画中...")
    time_trend(df)

    out_lines: list[str] = [
        "YouTubeコメントのシンプルな記述統計（機械学習・埋め込みを介さない直接集計）",
    ]
    crosstab_chi2(df, out_lines)

    summary_path = RESULTS_DIR / "descriptive_crosstab_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n書き出し: {summary_path}")


if __name__ == "__main__":
    main()
