"""
ISSP「Role of Government」日本データに対する機械学習分析。

単純なOLS回帰（issp_analysis.py）に代えて，Random Forestによる予測モデルと
SHAP値による特徴量の寄与度分析，および対象者の政治的態度プロファイルに
基づくクラスタリングを行う。非線形性・変数間の交互作用を捉えられるかを
線形回帰と比較しつつ検証する。

使い方:
    uv run scripts/issp_ml_analysis.py

出力:
    results/ml_model_comparison.txt   -- OLS vs Random ForestのCV性能比較
    results/shap_summary_{target}.png -- 各目的変数のSHAP要約プロット
    results/shap_dependence_year_{target}.png -- 「調査年」のSHAP依存プロット
    results/cluster_profile.png       -- 政治的態度クラスタの年別構成比の推移
    results/cluster_summary.txt       -- 各クラスタの特徴量平均値
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
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Yu Gothic"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "issp_role_of_government" / "Role of Government_ISSP.sav"
RESULTS_DIR = ROOT / "results"

JAPAN_COUNTRY_CODE = 392
INTERNET_RATE = {1996: 9.2, 2006: 72.6, 2016: 83.5}

FEATURE_COLS = [
    "year", "internet_rate", "age", "sex", "education", "pol_interest",
    "is_unemployed", "is_not_in_labor_force",
]
TARGETS = {
    "no_say": "政治的有効性感覚の欠如",
    "trust_civil": "公務員への信頼",
    "corruption": "政治家の汚職認識",
}


def load_japan_data() -> pd.DataFrame:
    df, _ = pyreadstat.read_sav(DATA_PATH)
    df_jp = df[df["country"] == JAPAN_COUNTRY_CODE][
        ["year_sdno", "v60", "v61", "v66", "v73", "AGE", "SEX", "DEGREE", "WORKYN"]
    ].rename(
        columns={
            "year_sdno": "year",
            "v60": "pol_interest",
            "v61": "no_say",
            "v66": "trust_civil",
            "v73": "corruption",
            "AGE": "age",
            "SEX": "sex",
            "DEGREE": "education",
        }
    )
    df_jp["year"] = df_jp["year"].astype(int)
    df_jp["internet_rate"] = df_jp["year"].map(INTERNET_RATE)
    # WORKYN: 1=Employed, 2=Unemployed, 3=Not in labour force（ISSP全3時点で共通調査）。
    # 「経済的破壊の当事者性」を測る変数が年齢・学歴等の人口統計学的変数のみで
    # あるという限界（academic_review 2026-08-11, 3.1節）を受け，雇用状態を
    # 追加した。Employedを基準カテゴリとする2つのダミー変数に変換する。
    df_jp["is_unemployed"] = (df_jp["WORKYN"] == 2).astype(float)
    df_jp["is_not_in_labor_force"] = (df_jp["WORKYN"] == 3).astype(float)
    df_jp.loc[df_jp["WORKYN"].isna(), ["is_unemployed", "is_not_in_labor_force"]] = pd.NA
    return df_jp


def compare_ols_vs_rf(df: pd.DataFrame, target: str, out_lines: list[str]) -> None:
    data = df[[*FEATURE_COLS, target]].dropna()
    X, y = data[FEATURE_COLS], data[target]

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    ols_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    rf_scores = cross_val_score(
        RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=20, random_state=0),
        X, y, cv=cv, scoring="r2",
    )

    out_lines.append(f"\n{'=' * 70}")
    out_lines.append(f"目的変数: {target} ({TARGETS[target]})  N={len(data)}")
    out_lines.append(f"{'=' * 70}")
    out_lines.append(f"OLS          5-fold CV R^2: 平均={ols_scores.mean():.4f}  (各fold: {np.round(ols_scores, 3).tolist()})")
    out_lines.append(f"RandomForest 5-fold CV R^2: 平均={rf_scores.mean():.4f}  (各fold: {np.round(rf_scores, 3).tolist()})")
    diff = rf_scores.mean() - ols_scores.mean()
    out_lines.append(f"差分 (RF - OLS): {diff:+.4f}")


def shap_analysis(df: pd.DataFrame, target: str) -> None:
    data = df[[*FEATURE_COLS, target]].dropna()
    X, y = data[FEATURE_COLS], data[target]

    model = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=20, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    feature_names_jp = {
        "year": "調査年", "internet_rate": "インターネット普及率",
        "age": "年齢", "sex": "性別", "education": "学歴", "pol_interest": "政治への関心",
    }
    X_jp = X.rename(columns=feature_names_jp)

    plt.figure()
    shap.summary_plot(shap_values, X_jp, show=False)
    plt.title(f"SHAP要約: {target} ({TARGETS[target]})への各要因の寄与")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"shap_summary_{target}.png", dpi=150)
    plt.close()

    plt.figure()
    shap.dependence_plot("調査年", shap_values, X_jp, interaction_index="政治への関心", show=False)
    plt.title(f"SHAP依存プロット: 調査年 → {target}（色=政治への関心）")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"shap_dependence_year_{target}.png", dpi=150)
    plt.close()

    print(f"  SHAP出力: shap_summary_{target}.png, shap_dependence_year_{target}.png")


def choose_k(X_scaled: np.ndarray, out_lines: list[str], k_range: range = range(2, 7)) -> int:
    """エルボー法（inertia）とシルエット係数によりKを選定する。

    academic_review 2026-08-11 (1.2節) の指摘——付録がエルボー法/シルエット
    係数によるK選定を説明しているにもかかわらず実装ではK=3が決め打ちに
    なっていた不一致——を解消するため，実際に両指標を計算して記録する。
    """
    from sklearn.metrics import silhouette_score

    out_lines.append(f"\n{'=' * 70}\nK-means: クラスタ数の選定（エルボー法・シルエット係数）\n{'=' * 70}")
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        out_lines.append(f"  K={k}: inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

    best_k_by_silhouette = list(k_range)[int(np.argmax(silhouettes))]
    out_lines.append(f"シルエット係数が最大となるK: {best_k_by_silhouette}"
                      f"（値={max(silhouettes):.4f}）")
    out_lines.append("エルボー法は目視判断が必要なため，inertiaの値を上記に記録した"
                      "（急激な減少が緩やかになる『肘』の位置を確認する）。")
    return best_k_by_silhouette


def cluster_analysis(df: pd.DataFrame, out_lines: list[str]) -> None:
    cluster_vars = ["no_say", "trust_civil", "corruption", "pol_interest"]
    data = df.dropna(subset=cluster_vars).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[cluster_vars])

    best_k_by_silhouette = choose_k(X_scaled, out_lines)

    # 論文本文（第4部）の記述との対応・年別構成比の解釈可能性を優先し，
    # 本稿では引き続きK=3を採用する。シルエット係数上の最適値との異同は
    # choose_k()の出力（cluster_summary.txt）に記録済みであり，両者が
    # 一致しない場合はその旨を限界として本文に明記すること。
    k = 3
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    data["cluster"] = km.fit_predict(X_scaled)

    out_lines.append(
        f"\n{'=' * 70}\nクラスタ分析 (K-means, k={k}, N={len(data)}；"
        f"シルエット係数最適値はK={best_k_by_silhouette}）\n{'=' * 70}"
    )
    cluster_means = data.groupby("cluster")[cluster_vars].mean().round(2)
    out_lines.append(cluster_means.to_string())

    # クラスタの解釈用ラベル（値の大小関係から自動命名するのは危険なため，
    # 平均値を見て手動で解釈することを前提に，ここでは番号のまま集計する）
    trend = data.groupby(["year", "cluster"]).size().unstack(fill_value=0)
    trend_pct = trend.div(trend.sum(axis=1), axis=0) * 100
    out_lines.append("\n年別クラスタ構成比(%):")
    out_lines.append(trend_pct.round(1).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    trend_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_title("政治的態度クラスタの年別構成比の推移 (ISSP, K-means)")
    ax.set_xlabel("調査年")
    ax.set_ylabel("構成比 (%)")
    ax.legend(title="クラスタ番号", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=k)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cluster_profile.png", dpi=150)
    plt.close(fig)

    summary_path = RESULTS_DIR / "cluster_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"  クラスタ分析出力: cluster_profile.png, cluster_summary.txt")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_japan_data()
    print(f"日本サブセット: {len(df)}件")

    out_lines: list[str] = []
    for target in TARGETS:
        print(f"[{target}] OLS vs RandomForest 比較...")
        compare_ols_vs_rf(df, target, out_lines)
        print(f"[{target}] SHAP分析...")
        shap_analysis(df, target)

    comparison_path = RESULTS_DIR / "ml_model_comparison.txt"
    comparison_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"書き出し: {comparison_path}")

    print("クラスタ分析...")
    cluster_analysis(df, [])


if __name__ == "__main__":
    main()
