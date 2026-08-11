"""
WVS Time Series（1981-2022，Wave 1-7統合）を用いた，日本と海外諸国の比較分析。

wvs_timeseries_analysis.py（日本単体）を拡張し，以下の2種類の比較を行う。

  (A) 制度への信頼の長期トレンドを，国ごとに重ね描きする。
  (B) Wave 7（2017-2022年，国により実施年が異なる）の個人単位データを用い，
      「SNS利用頻度 -> 制度への信頼」の効果（OLS係数・p値，OLS対Random Forestの
      予測性能）を国ごとに算出し，1つの比較表にまとめる。日本で観察された
      「SNS利用は信頼に有意な効果を持たない」という結果が，日本固有の現象か，
      他国でも共通して見られるのかを検証する。

対象国はデフォルトで日本・アメリカ・韓国・ドイツ・イギリスの5か国
（COUNTRIES辞書を編集すれば追加・変更できる。WVS Wave 7でSNS利用変数
E253Bが利用可能な国は全65か国あるので，他国を追加したい場合は
`pyreadstat`でS003の値ラベルを確認して国コードを調べればよい）。

使い方:
    uv run scripts/wvs_cross_country_analysis.py

出力:
    results/wvs_cross_country_trend_{target}.png       -- 国別の長期トレンド重ね描き
    results/wvs_cross_country_sns_effect_summary.txt   -- 国別のSNS効果比較表
    results/wvs_cross_country_shap_{country}_{target}.png -- 国別のSHAP要約プロット
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
import statsmodels.api as sm
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

# 比較対象国: WVS S003コード。値ラベルはpyreadstatのmeta.variable_value_labelsで確認済み
# （Germanyは統一後の276を使用。統一前の西独/東独separate codeは900/901で別扱い、未使用）。
COUNTRIES = {
    "Japan": 392,
    "United States": 840,
    "South Korea": 410,
    "Germany": 276,
    "United Kingdom": 826,
}

RAW_COLS = [
    "S003", "S002VS", "S020",
    "E069_07", "E069_08", "E069_11", "E069_12",
    "E253B", "E262B", "E258B", "E248B",
    "X001", "X003", "X025R",
]

TRUST_TARGETS = {
    "trust_parl": "議会への信頼 (E069_07)",
    "trust_civil": "公務員への信頼 (E069_08)",
}
CONTROL_FEATURES = ["age", "sex", "education"]
SNS_MEDIA_FEATURES = ["sns_use", "internet_use", "tv_news_use", "newspaper_use"]


def load_data() -> pd.DataFrame:
    df, _ = pyreadstat.read_sav(DATA_PATH, usecols=RAW_COLS)
    df = df[df["S003"].isin(COUNTRIES.values())].copy()
    code_to_name = {v: k for k, v in COUNTRIES.items()}
    df["country"] = df["S003"].map(code_to_name)
    df = df.rename(
        columns={
            "S002VS": "wave", "S020": "year",
            "E069_07": "trust_parl_raw", "E069_08": "trust_civil_raw",
            "E069_11": "trust_gov_raw", "E069_12": "trust_parties_raw",
            "E253B": "sns_raw", "E262B": "internet_raw",
            "E258B": "tv_news_raw", "E248B": "newspaper_raw",
            "X001": "sex", "X003": "age", "X025R": "education",
        }
    )
    for src, dst in [("trust_parl_raw", "trust_parl"), ("trust_civil_raw", "trust_civil"),
                      ("trust_gov_raw", "trust_gov"), ("trust_parties_raw", "trust_parties")]:
        df[dst] = 5 - df[src]
    for src, dst in [("sns_raw", "sns_use"), ("internet_raw", "internet_use"),
                      ("tv_news_raw", "tv_news_use"), ("newspaper_raw", "newspaper_use")]:
        df[dst] = 6 - df[src]
    df["year"] = df["year"].astype(int)
    return df


def plot_cross_country_trend(df: pd.DataFrame, target: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for country in COUNTRIES:
        sub = df[df["country"] == country]
        trend = sub.groupby("year")[target].mean().dropna()
        if trend.empty:
            continue
        ax.plot(trend.index, trend.values, marker="o", label=country)
    ax.set_title(f"{TRUST_TARGETS.get(target, target)}の国際比較 (WVS, 1981-2022年)")
    ax.set_xlabel("調査年")
    ax.set_ylabel("平均値 (1-4，高いほど信頼が高い)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    out_path = RESULTS_DIR / f"wvs_cross_country_trend_{target}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"書き出し: {out_path}")


def sns_effect_for_country(df: pd.DataFrame, country: str, target: str, out_lines: list[str]) -> None:
    wave7 = df[(df["country"] == country) & (df["wave"] == 7)]
    features = SNS_MEDIA_FEATURES + CONTROL_FEATURES
    data = wave7[[*features, target]].dropna()

    if len(data) < 20:
        out_lines.append(f"{country:16s} {target:12s}  サンプルサイズ不足 (N={len(data)}) のためスキップ")
        return

    X, y = data[features], data[target]

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    ols_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    rf_scores = cross_val_score(
        RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=20, random_state=0),
        X, y, cv=cv, scoring="r2",
    )

    X_sm = sm.add_constant(X)
    ols_full = sm.OLS(y, X_sm).fit()
    sns_coef = ols_full.params.get("sns_use", float("nan"))
    sns_p = ols_full.pvalues.get("sns_use", float("nan"))
    sig_mark = "*" if sns_p < 0.05 else " "

    out_lines.append(
        f"{country:16s} {target:12s}  N={len(data):5d}  "
        f"SNS係数={sns_coef:+.4f} p={sns_p:.4f}{sig_mark}  "
        f"OLS-R2={ols_scores.mean():.4f}  RF-R2={rf_scores.mean():.4f}"
    )

    # SHAP（国名にスペースを含む場合はアンダースコアに置換してファイル名に使う）
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
    plt.title(f"SHAP要約[{country}]: {target} ({TRUST_TARGETS.get(target, target)})への各要因の寄与")
    plt.tight_layout()
    country_slug = country.replace(" ", "_")
    out_path = RESULTS_DIR / f"wvs_cross_country_shap_{country_slug}_{target}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"書き出し: {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"対象国合計: {len(df)}件")
    for country in COUNTRIES:
        n = len(df[df["country"] == country])
        waves = sorted(df[df["country"] == country]["wave"].dropna().unique().astype(int).tolist())
        print(f"  {country:16s} N={n:6d}  waves={waves}")

    print("\n=== (A) 制度への信頼の長期トレンド：国際比較 ===")
    for target in ["trust_parl", "trust_civil"]:
        plot_cross_country_trend(df, target)

    print("\n=== (B) Wave 7: SNS利用 -> 信頼 の効果比較 ===")
    out_lines: list[str] = [
        "WVS Time Series 国際比較: Wave 7における「SNS利用頻度 -> 制度への信頼」の効果",
        "（p<0.05のものに*を付与。OLS-R2/RF-R2は5分割交差検証の平均R^2）",
        "=" * 100,
    ]
    for target in ["trust_parl", "trust_civil"]:
        out_lines.append(f"\n--- 目的変数: {target} ({TRUST_TARGETS.get(target, target)}) ---")
        for country in COUNTRIES:
            print(f"[{country}] {target} を分析中...")
            sns_effect_for_country(df, country, target, out_lines)

    summary_path = RESULTS_DIR / "wvs_cross_country_sns_effect_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n書き出し: {summary_path}")
    print("\n" + "\n".join(out_lines))


if __name__ == "__main__":
    main()
