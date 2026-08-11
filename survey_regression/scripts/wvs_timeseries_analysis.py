"""
WVS Time Series（1981-2022，Wave 1-7統合）日本データの分析。

ISSPよりも長い射程（1981-2019年，日本の実施年）で，制度への信頼の
長期トレンドを可視化するとともに，メディア利用頻度（SNS・インターネット・
テレビニュース・新聞）の変数が利用可能なWave 6-7（2010年・2019年）の
個人単位データを用いて，issp_ml_analysis.pyと同じ枠組み（OLS対Random
Forest比較，SHAP分析）でメディア利用が制度への信頼に与える影響を検証する。

使い方:
    uv run scripts/wvs_timeseries_analysis.py

出力:
    results/wvs_trend.png                    -- 1981-2019年の信頼度の長期トレンド
    results/wvs_media_ml_comparison.txt      -- OLS vs RFの性能比較（Wave6-7）
    results/wvs_shap_summary_{target}.png    -- SHAP要約プロット
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyreadstat
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Yu Gothic"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "wvs_time_series" / "WVS_Time_Series_1981-2022_spss_v5_0.sav"
RESULTS_DIR = ROOT / "results"

JAPAN_COUNTRY_CODE = 392

# 生データの列名 -> 分かりやすい名前
RAW_COLS = [
    "S003", "S002VS", "S020",
    "E069_07", "E069_08", "E069_11", "E069_12",  # 信頼: 議会/公務員/政府/政党
    "E253B", "E262B", "E258B", "E248B",            # メディア利用: SNS/インターネット/TV/新聞
    "X001", "X003", "X025R",                         # 性別/年齢/学歴（X025Rはwave4-7で利用可能）
]

TRUST_TARGETS = {
    "trust_parl": "議会への信頼 (E069_07)",
    "trust_civil": "公務員への信頼 (E069_08)",
}

CONTROL_FEATURES = ["age", "sex", "education"]

# E253B（SNS利用）はWave 7（2019年）でのみ調査されており，Wave 6（2010年）には
# 存在しない。internet_use/tv_news_use/newspaper_useはWave 6-7両方で調査されて
# いる。この非対称性のため，分析を2種類に分ける。
POOLED_MEDIA_FEATURES = ["internet_use", "tv_news_use", "newspaper_use"]  # Wave 6+7
SNS_MEDIA_FEATURES = ["sns_use", "internet_use", "tv_news_use", "newspaper_use"]  # Wave 7のみ


def load_japan_data() -> pd.DataFrame:
    df, _ = pyreadstat.read_sav(DATA_PATH, usecols=RAW_COLS)
    jp = df[df["S003"] == JAPAN_COUNTRY_CODE].copy()
    jp = jp.rename(
        columns={
            "S002VS": "wave", "S020": "year",
            "E069_07": "trust_parl_raw", "E069_08": "trust_civil_raw",
            "E069_11": "trust_gov_raw", "E069_12": "trust_parties_raw",
            "E253B": "sns_raw", "E262B": "internet_raw",
            "E258B": "tv_news_raw", "E248B": "newspaper_raw",
            "X001": "sex", "X003": "age", "X025R": "education",
        }
    )
    # 信頼度: 1(great deal)-4(none at all) -> 反転して「高いほど信頼が高い」に統一
    for src, dst in [("trust_parl_raw", "trust_parl"), ("trust_civil_raw", "trust_civil"),
                      ("trust_gov_raw", "trust_gov"), ("trust_parties_raw", "trust_parties")]:
        jp[dst] = 5 - jp[src]
    # メディア利用: 1(daily)-5(never) -> 反転して「高いほど利用頻度が高い」に統一
    for src, dst in [("sns_raw", "sns_use"), ("internet_raw", "internet_use"),
                      ("tv_news_raw", "tv_news_use"), ("newspaper_raw", "newspaper_use")]:
        jp[dst] = 6 - jp[src]
    jp["year"] = jp["year"].astype(int)
    return jp


def plot_long_run_trend(df: pd.DataFrame) -> None:
    trend = df.groupby("year")[["trust_parl", "trust_civil", "trust_gov", "trust_parties"]].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = {
        "trust_parl": "議会への信頼", "trust_civil": "公務員への信頼",
        "trust_gov": "政府への信頼", "trust_parties": "政党への信頼",
    }
    for col, label in labels.items():
        ax.plot(trend.index, trend[col], marker="o", label=label)
    ax.set_title("日本における制度への信頼の長期トレンド (WVS, 1981-2019年)")
    ax.set_xlabel("調査年")
    ax.set_ylabel("平均値 (1-4，高いほど信頼が高い)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    plot_path = RESULTS_DIR / "wvs_trend.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"書き出し: {plot_path}")
    print(trend.round(2).to_string())


def compare_ols_vs_rf(
    df: pd.DataFrame, target: str, media_features: list[str], label: str, out_lines: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    features = media_features + CONTROL_FEATURES
    data = df[[*features, target]].dropna()
    X, y = data[features], data[target]

    if len(data) < 20:
        out_lines.append(f"\n[{label}] {target}: サンプルサイズ不足 (N={len(data)}) のためスキップ")
        return data, features

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    ols_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    rf_scores = cross_val_score(
        RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=20, random_state=0),
        X, y, cv=cv, scoring="r2",
    )

    out_lines.append(f"\n{'=' * 70}")
    out_lines.append(f"[{label}] 目的変数: {target} ({TRUST_TARGETS.get(target, target)})  N={len(data)}")
    out_lines.append(f"説明変数: {features}")
    out_lines.append(f"{'=' * 70}")
    out_lines.append(f"OLS          5-fold CV R^2: 平均={ols_scores.mean():.4f}  (各fold: {np.round(ols_scores, 3).tolist()})")
    out_lines.append(f"RandomForest 5-fold CV R^2: 平均={rf_scores.mean():.4f}  (各fold: {np.round(rf_scores, 3).tolist()})")

    # OLS係数(参考): media変数の符号・有意性を素朴に確認
    import statsmodels.api as sm
    X_sm = sm.add_constant(X)
    ols_full = sm.OLS(y, X_sm).fit()
    out_lines.append("\nOLS全データでの係数（参考）:")
    out_lines.append(ols_full.summary().tables[1].as_text())

    return data, features


def shap_analysis(data: pd.DataFrame, features: list[str], target: str, suffix: str) -> None:
    X, y = data[features], data[target]
    model = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=20, random_state=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    name_jp = {
        "sns_use": "SNS利用頻度", "internet_use": "インターネット利用頻度",
        "tv_news_use": "テレビニュース利用頻度", "newspaper_use": "新聞利用頻度",
        "age": "年齢", "sex": "性別", "education": "学歴",
    }
    X_jp = X.rename(columns=name_jp)

    plt.figure()
    shap.summary_plot(shap_values, X_jp, show=False)
    plt.title(f"SHAP要約[{suffix}]: {target} ({TRUST_TARGETS.get(target, target)})への各要因の寄与")
    plt.tight_layout()
    out_path = RESULTS_DIR / f"wvs_shap_summary_{target}_{suffix}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"書き出し: {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_japan_data()
    print(f"日本サブセット全体: {len(df)}件 (waves: {sorted(df['wave'].unique())})")

    plot_long_run_trend(df)

    out_lines: list[str] = [
        "WVS Time Series 日本データ: メディア利用 -> 制度への信頼",
    ]

    # (A) Wave 6+7 プール（SNS利用を除く3変数）: N大きい
    print("\n=== (A) Wave 6+7 プール分析（internet/TV/新聞） ===")
    pooled_df = df[df["wave"].isin([6, 7])]
    out_lines.append("\n\n### (A) Wave 6+7 プール分析（internet/TV/新聞，SNS変数は除く） ###")
    for target in ["trust_parl", "trust_civil"]:
        print(f"[{target}] OLS vs RandomForest 比較 (pooled)...")
        data, features = compare_ols_vs_rf(pooled_df, target, POOLED_MEDIA_FEATURES, "pooled", out_lines)
        if len(data) >= 20:
            print(f"[{target}] SHAP分析 (pooled)...")
            shap_analysis(data, features, target, "pooled")

    # (B) Wave 7 のみ（SNS利用を含む4変数）: SNS効果を直接検証
    print("\n=== (B) Wave 7 (2019年) のみ分析（SNSを含む） ===")
    wave7_df = df[df["wave"] == 7]
    out_lines.append("\n\n### (B) Wave 7 (2019年) のみ分析（SNS利用を含む4変数） ###")
    for target in ["trust_parl", "trust_civil"]:
        print(f"[{target}] OLS vs RandomForest 比較 (wave7-only)...")
        data, features = compare_ols_vs_rf(wave7_df, target, SNS_MEDIA_FEATURES, "wave7only", out_lines)
        if len(data) >= 20:
            print(f"[{target}] SHAP分析 (wave7-only)...")
            shap_analysis(data, features, target, "wave7only")

    summary_path = RESULTS_DIR / "wvs_media_ml_comparison.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n書き出し: {summary_path}")


if __name__ == "__main__":
    main()
