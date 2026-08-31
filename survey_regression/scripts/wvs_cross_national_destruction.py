"""
日米独クロスナショナル比較（本格版）。

wvs_cross_country_analysis.pyはメディア利用（SNS利用頻度）を主要な説明変数に
していたが，本スクリプトはRicciの理論の核である「経済的破壊の当事者性
（失業・非労働力化・低所得）が政治的不信を生むか」という関係そのものを，
日本・アメリカ・ドイツの3か国で比較する。

  (A) 3か国をプールした回帰に「経済的破壊指標 × 国」の交互作用項を入れ，
      効果が国によって異なるか（交互作用項の有意性）を検定する。
  (B) 国別に別々の回帰を推定し，係数・p値を1つの比較表にまとめる
      （Ricci理論が予測する「経済的破壊→不信」が米独では成立し，
      日本では成立しない，という対照を直接確認する）。

使い方:
    uv run scripts/wvs_cross_national_destruction.py

出力:
    results/wvs_cross_national_destruction_summary.txt
    results/wvs_cross_national_interaction_{target}.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Yu Gothic"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "wvs_time_series" / "WVS_Time_Series_1981-2022_spss_v5_0.sav"
RESULTS_DIR = ROOT / "results"

COUNTRIES = {"Japan": 392, "United States": 840, "Germany": 276}

RAW_COLS = [
    "S003", "S002VS", "S020",
    "E069_07", "E069_08",
    "X001", "X003", "X025R", "X028", "X047R_WVS",
]
TRUST_TARGETS = {
    "trust_parl": "議会への信頼 (E069_07)",
    "trust_civil": "公務員への信頼 (E069_08)",
}


def load_data() -> pd.DataFrame:
    df, _ = pyreadstat.read_sav(DATA_PATH, usecols=RAW_COLS)
    df = df[df["S003"].isin(COUNTRIES.values())].copy()
    code_to_name = {v: k for k, v in COUNTRIES.items()}
    df["country"] = df["S003"].map(code_to_name)
    df = df.rename(
        columns={
            "S002VS": "wave", "S020": "year",
            "E069_07": "trust_parl_raw", "E069_08": "trust_civil_raw",
            "X001": "sex", "X003": "age", "X025R": "education",
            "X028": "employment_raw", "X047R_WVS": "income_level_raw",
        }
    )
    for src, dst in [("trust_parl_raw", "trust_parl"), ("trust_civil_raw", "trust_civil")]:
        df[dst] = 5 - df[src]
    df["is_unemployed"] = (df["employment_raw"] == 7).astype(float)
    df["is_not_in_labor_force"] = df["employment_raw"].isin([4, 5, 6, 8]).astype(float)
    df.loc[df["employment_raw"] < 0, ["is_unemployed", "is_not_in_labor_force"]] = np.nan
    df.loc[df["employment_raw"].isna(), ["is_unemployed", "is_not_in_labor_force"]] = np.nan
    df["income_level"] = df["income_level_raw"].where(df["income_level_raw"] > 0)
    df["year"] = df["year"].astype(int)
    df = df[df["wave"].isin([6, 7])]
    return df


def pooled_interaction_model(df: pd.DataFrame, target: str, out_lines: list[str]) -> None:
    data = df.dropna(subset=[target, "income_level", "is_unemployed", "is_not_in_labor_force",
                              "age", "sex", "education", "country"]).copy()
    data["country"] = pd.Categorical(data["country"], categories=list(COUNTRIES.keys()))

    formula = (
        f"{target} ~ income_level * C(country) + is_unemployed * C(country) "
        f"+ is_not_in_labor_force * C(country) + age + C(sex) + C(education)"
    )
    model = smf.ols(formula, data=data, missing="drop").fit()

    out_lines.append(f"\n{'=' * 90}")
    out_lines.append(f"(A) プール回帰＋交互作用項: 目的変数={target} ({TRUST_TARGETS[target]})  N={len(data)}")
    out_lines.append(f"式: {formula}")
    out_lines.append(f"{'=' * 90}")
    out_lines.append(model.summary().tables[1].as_text())

    interaction_terms = [p for p in model.params.index if ":C(country)" in p]
    out_lines.append("\n交互作用項（『経済的破壊指標の効果が国によって異なるか』の検定）:")
    for t in interaction_terms:
        out_lines.append(f"  {t}: coef={model.params[t]:+.4f}  p={model.pvalues[t]:.4f}"
                          f"{'  *有意*' if model.pvalues[t] < 0.05 else ''}")


def per_country_models(df: pd.DataFrame, target: str, out_lines: list[str]) -> dict[str, tuple[float, float]]:
    out_lines.append(f"\n(B) 国別回帰: 目的変数={target} ({TRUST_TARGETS[target]})")
    out_lines.append(f"{'国':16s} {'N':>6s}  {'income_level係数':>18s} {'p値':>8s}  "
                      f"{'is_unemployed係数':>18s} {'p値':>8s}")
    results: dict[str, tuple[float, float]] = {}
    for country in COUNTRIES:
        sub = df[df["country"] == country]
        data = sub.dropna(subset=[target, "income_level", "is_unemployed", "is_not_in_labor_force",
                                   "age", "sex", "education"])
        if len(data) < 30:
            out_lines.append(f"{country:16s}  サンプルサイズ不足 (N={len(data)})")
            continue
        formula = f"{target} ~ income_level + is_unemployed + is_not_in_labor_force + age + C(sex) + C(education)"
        model = smf.ols(formula, data=data, missing="drop").fit()
        inc_coef = model.params.get("income_level", np.nan)
        inc_p = model.pvalues.get("income_level", np.nan)
        unemp_coef = model.params.get("is_unemployed", np.nan)
        unemp_p = model.pvalues.get("is_unemployed", np.nan)
        results[country] = (inc_coef, inc_p)
        out_lines.append(
            f"{country:16s} {len(data):6d}  {inc_coef:+18.4f} {inc_p:8.4f}  "
            f"{unemp_coef:+18.4f} {unemp_p:8.4f}"
        )
    return results


def plot_comparison(results: dict[str, tuple[float, float]], target: str) -> None:
    if not results:
        return
    countries = list(results.keys())
    coefs = [results[c][0] for c in countries]
    colors = ["tab:red" if results[c][1] < 0.05 else "tab:gray" for c in countries]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(countries, coefs, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"主観的所得水準 -> {TRUST_TARGETS[target]}\n国別OLS係数（赤=p<0.05）")
    ax.set_ylabel("係数（income_level, 高いほど高所得）")
    fig.tight_layout()
    out_path = RESULTS_DIR / f"wvs_cross_national_interaction_{target}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"書き出し: {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"対象国合計(Wave6+7): {len(df)}件")
    for country in COUNTRIES:
        n = len(df[df["country"] == country])
        print(f"  {country:16s} N={n:6d}")

    out_lines: list[str] = [
        "日米独クロスナショナル比較: 経済的破壊の当事者性（主観的所得水準・雇用状態）が",
        "議会/公務員への信頼に与える効果。WVS Time Series Wave 6+7（国により実施年が異なる）。",
        "Ricci理論の予測: 経済的破壊の効果は米独で(負に)有意，日本では非有意 or 弱い、という",
        "対照が観察されるか。",
    ]

    for target in TRUST_TARGETS:
        pooled_interaction_model(df, target, out_lines)
        results = per_country_models(df, target, out_lines)
        plot_comparison(results, target)

    summary_path = RESULTS_DIR / "wvs_cross_national_destruction_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n書き出し: {summary_path}")


if __name__ == "__main__":
    main()
