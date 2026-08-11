"""
ISSP「Role of Government」（ZA4747）日本サブセットの回帰分析。

旧版（Role of Government.r）のPythonへの再実装。
インターネット普及率および調査年が，政治的有効性感覚の欠如・公務員信頼・
政治家の汚職認識に与える影響をOLSで検証する。

使い方:
    uv run scripts/issp_analysis.py

出力:
    results/issp_summary.txt   -- 各モデルの回帰結果（統計サマリ）
    results/issp_trend.png     -- 日本の政治意識の時系列変化グラフ
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Yu Gothic"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "issp_role_of_government" / "Role of Government_ISSP.sav"
RESULTS_DIR = ROOT / "results"

JAPAN_COUNTRY_CODE = 392

# ISSP調査年に対応する日本のインターネット普及率（総務省統計，代理変数）
INTERNET_RATE = {1996: 9.2, 2006: 72.6, 2016: 83.5}


def load_japan_data() -> pd.DataFrame:
    df, meta = pyreadstat.read_sav(DATA_PATH)
    df_jp = df[df["country"] == JAPAN_COUNTRY_CODE][
        ["year_sdno", "v60", "v61", "v65", "v66", "v73", "AGE", "SEX", "DEGREE"]
    ].rename(
        columns={
            "year_sdno": "year",
            "v60": "pol_interest",
            "v61": "no_say",
            "v65": "trust_mps",
            "v66": "trust_civil",
            "v73": "corruption",
            "AGE": "age",
            "SEX": "sex",
            "DEGREE": "education",
        }
    )
    df_jp["year"] = df_jp["year"].astype(int)
    df_jp["internet_rate"] = df_jp["year"].map(INTERNET_RATE)
    df_jp["sex"] = df_jp["sex"].astype("category")
    df_jp["education"] = df_jp["education"].astype("category")
    df_jp["year_factor"] = df_jp["year"].astype("category")
    return df_jp


def fit_and_report(formula: str, data: pd.DataFrame, label: str, out_lines: list[str]) -> None:
    model = smf.ols(formula, data=data, missing="drop").fit()
    out_lines.append(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    out_lines.append(f"formula: {formula}")
    out_lines.append(f"N = {int(model.nobs)}")
    out_lines.append(model.summary().as_text())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_japan_data()
    print(f"日本サブセット: {len(df)}件 (年別: {df['year'].value_counts().sort_index().to_dict()})")

    out_lines: list[str] = []

    # --- 調査年ダミーによるモデル（旧 model_1/model_2/model_3/model_4） ---
    fit_and_report(
        "no_say ~ C(year_factor) + age + C(sex) + C(education) + pol_interest",
        df,
        "モデル1: 政治的有効性感覚の欠如 (v61) ~ 調査年ダミー",
        out_lines,
    )
    fit_and_report(
        "trust_civil ~ C(year_factor) + age + C(sex) + C(education) + pol_interest",
        df,
        "モデル2: 公務員への信頼 (v66) ~ 調査年ダミー",
        out_lines,
    )
    fit_and_report(
        "corruption ~ C(year_factor) + age + C(sex) + C(education) + pol_interest",
        df[df["year"] != 1996],  # v73は1996年に調査なし
        "モデル3: 政治家の汚職認識 (v73) ~ 調査年ダミー（1996年除外）",
        out_lines,
    )
    fit_and_report(
        "trust_mps ~ C(year_factor) + age + C(sex) + C(education) + pol_interest",
        df,
        "モデル4: 議員への信頼 (v65) ~ 調査年ダミー",
        out_lines,
    )

    # --- インターネット普及率によるモデル（旧 model_*_internet） ---
    fit_and_report(
        "no_say ~ internet_rate + age + C(sex) + C(education) + pol_interest",
        df,
        "モデル1-internet: 政治的有効性感覚の欠如 (v61) ~ インターネット普及率",
        out_lines,
    )
    fit_and_report(
        "trust_civil ~ internet_rate + age + C(sex) + C(education) + pol_interest",
        df,
        "モデル2-internet: 公務員への信頼 (v66) ~ インターネット普及率",
        out_lines,
    )
    fit_and_report(
        "corruption ~ internet_rate + age + C(sex) + C(education) + pol_interest",
        df[df["year"] != 1996],
        "モデル3-internet: 政治家の汚職認識 (v73) ~ インターネット普及率（1996年除外）",
        out_lines,
    )
    fit_and_report(
        "trust_mps ~ internet_rate + age + C(sex) + C(education) + pol_interest",
        df,
        "モデル4-internet: 議員への信頼 (v65) ~ インターネット普及率",
        out_lines,
    )

    summary_path = RESULTS_DIR / "issp_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"書き出し: {summary_path}")

    # --- トレンドグラフ ---
    trend = df.groupby("year")[["no_say", "trust_civil", "corruption", "pol_interest"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {
        "no_say": "政治的有効性感覚の欠如 (v61)",
        "trust_civil": "公務員への信頼 (v66)",
        "corruption": "政治家の汚職認識 (v73)",
        "pol_interest": "政治への関心 (v60)",
    }
    for col, label in labels.items():
        ax.plot(trend.index, trend[col], marker="o", label=label)
    ax.set_title("日本の政治意識の時系列変化 (ISSP, 1996-2016年)")
    ax.set_xlabel("調査年")
    ax.set_ylabel("平均値 (尺度)")
    ax.set_xticks([1996, 2006, 2016])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    plot_path = RESULTS_DIR / "issp_trend.png"
    fig.savefig(plot_path, dpi=150)
    print(f"書き出し: {plot_path}")


if __name__ == "__main__":
    main()
