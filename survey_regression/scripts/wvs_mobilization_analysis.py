"""
Ricci因果連鎖の「後半」――怨恨・エリート不信が実際の動員行動（デモ参加）に
転化するか――を直接検証する分析。

これまでの分析（issp_ml_analysis.py, wvs_timeseries_analysis.py,
mediation_analysis.py）はすべて「政治的態度（信頼・疎外感）」を目的変数に
してきたが，Ricciの理論が最終的に説明しようとしているのは態度ではなく
「ポピュリズムの動員エネルギー」という行動的な帰結である。WVSにある
E027（デモ参加経験: 1=したことがある/2=するかもしれない/3=絶対にしない）
を「動員」の直接的な代理変数として用い，

    X（経済的破壊の当事者性） -> M（政府への信頼） -> Y（デモ参加経験）

という2段階の関係を，日本・アメリカ・ドイツで比較する。Yが二値変数のため，
mediation_analysis.py と同じ積の分解（a*bのブートストラップ）ではなく，
2つの回帰（a経路: OLS, b/c'経路: ロジスティック回帰）を別々に報告する，
より単純で解釈しやすい方法を採る（二値アウトカムでのa*b分解は係数の
スケールが異なり解釈が難しいため，本分析ではあえて簡略化している）。

使い方:
    uv run scripts/wvs_mobilization_analysis.py

出力:
    results/wvs_mobilization_summary.txt
    results/wvs_mobilization_rates_by_country.png
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
    "S003", "S002VS", "S020", "E025", "E026", "E027", "E028", "E069_11",
    "X001", "X003", "X025R", "X028", "X047R_WVS",
]
CONTROLS = ["age", "C(sex)", "C(education)"]


def load_data() -> pd.DataFrame:
    df, _ = pyreadstat.read_sav(DATA_PATH, usecols=RAW_COLS)
    df = df[df["S003"].isin(COUNTRIES.values())].copy()
    code_to_name = {v: k for k, v in COUNTRIES.items()}
    df["country"] = df["S003"].map(code_to_name)
    df = df.rename(
        columns={
            "S002VS": "wave", "S020": "year", "E069_11": "trust_gov_raw",
            "X001": "sex", "X003": "age", "X025R": "education",
            "X028": "employment_raw", "X047R_WVS": "income_level_raw",
        }
    )
    df["trust_gov"] = 5 - df["trust_gov_raw"]
    df["is_unemployed"] = (df["employment_raw"] == 7).astype(float)
    df["is_not_in_labor_force"] = df["employment_raw"].isin([4, 5, 6, 8]).astype(float)
    df.loc[df["employment_raw"] < 0, ["is_unemployed", "is_not_in_labor_force"]] = np.nan
    df.loc[df["employment_raw"].isna(), ["is_unemployed", "is_not_in_labor_force"]] = np.nan
    df["income_level"] = df["income_level_raw"].where(df["income_level_raw"] > 0)
    # E027: 1=したことがある, 2=するかもしれない, 3=絶対にしない。負値は欠損コード。
    df["protest_done"] = np.where(df["E027"] == 1, 1.0, np.where(df["E027"].isin([2, 3]), 0.0, np.nan))
    df["protest_open"] = np.where(df["E027"].isin([1, 2]), 1.0, np.where(df["E027"] == 3, 0.0, np.nan))

    # 頑健性チェック用の複合指標: 署名(E025)・不買運動(E026)・デモ(E027)・スト(E028)の
    # 4種類のうち「実際に行ったことがある」項目が1つでもあれば1とする
    # （単一設問E027だけに依存しない，より広い『政治的動員行動』の代理変数）。
    action_items = ["E025", "E026", "E027", "E028"]
    done_flags = []
    valid_flags = []
    for item in action_items:
        done_flags.append((df[item] == 1))
        valid_flags.append(df[item].isin([1, 2, 3]))
    any_valid = pd.concat(valid_flags, axis=1).any(axis=1)
    any_done = pd.concat(done_flags, axis=1).any(axis=1)
    df["action_index_done"] = np.where(any_valid, any_done.astype(float), np.nan)

    df["year"] = df["year"].astype(int)
    df = df[df["wave"].isin([6, 7])]
    return df


def analyze_country(df: pd.DataFrame, country: str, out_lines: list[str]) -> dict:
    sub = df[df["country"] == country]
    ctrl_str = " + ".join(CONTROLS)
    needed = ["trust_gov", "protest_done", "income_level", "is_unemployed",
              "is_not_in_labor_force", "age", "sex", "education"]
    data = sub.dropna(subset=needed)

    out_lines.append(f"\n{'=' * 78}")
    out_lines.append(f"国: {country}  N={len(data)}")
    out_lines.append(f"デモ参加経験あり(protest_done=1)の割合: {data['protest_done'].mean() * 100:.1f}%")
    out_lines.append(f"{'=' * 78}")

    if len(data) < 50:
        out_lines.append("サンプルサイズ不足のためスキップ")
        return {}

    # 経路a: 経済的破壊 -> 政府信頼
    a_formula = f"trust_gov ~ income_level + is_unemployed + is_not_in_labor_force + {ctrl_str}"
    a_model = smf.ols(a_formula, data=data, missing="drop").fit()
    out_lines.append("\n[経路a] trust_gov ~ 経済的破壊指標 + controls (OLS)")
    for v in ["income_level", "is_unemployed", "is_not_in_labor_force"]:
        out_lines.append(f"  {v:22s}: coef={a_model.params.get(v, np.nan):+.4f}  p={a_model.pvalues.get(v, np.nan):.4f}"
                          f"{'  *有意*' if a_model.pvalues.get(v, 1) < 0.05 else ''}")

    # 経路b/c': 政府信頼・経済的破壊 -> デモ参加経験 (ロジスティック回帰)
    bc_formula = f"protest_done ~ trust_gov + income_level + is_unemployed + is_not_in_labor_force + {ctrl_str}"
    bc_model = smf.logit(bc_formula, data=data, missing="drop").fit(disp=0)
    out_lines.append("\n[経路b/c'] protest_done ~ trust_gov + 経済的破壊指標 + controls (ロジスティック回帰, オッズ比)")
    for v in ["trust_gov", "income_level", "is_unemployed", "is_not_in_labor_force"]:
        coef = bc_model.params.get(v, np.nan)
        p = bc_model.pvalues.get(v, np.nan)
        or_val = np.exp(coef)
        flag = ""
        if v in ("is_unemployed", "is_not_in_labor_force") and abs(coef) > 10:
            cell_n = int(data[v].sum())
            cell_events = int(data.loc[data[v] == 1, "protest_done"].sum())
            flag = (f"  [注: N={cell_n}中 protest_done=1が{cell_events}件のみで完全分離"
                    f"（quasi-complete separation）が生じており，係数が発散した数値的アーティファクト。"
                    f"実質的な効果として解釈しない]")
        out_lines.append(f"  {v:22s}: coef={coef:+.4f}  OR={or_val:.4f}  p={p:.4f}"
                          f"{'  *有意*' if p < 0.05 else ''}{flag}")

    trust_or = np.exp(bc_model.params.get("trust_gov", np.nan))
    trust_p = bc_model.pvalues.get("trust_gov", np.nan)
    out_lines.append(
        f"\n解釈: trust_gov のオッズ比={trust_or:.4f} (p={trust_p:.4f})。"
        f"{'1未満で有意なら「政府を信頼しない人ほどデモ参加経験あり」というRicci理論と整合的。' if trust_or < 1 else ''}"
    )

    return {"protest_rate": data["protest_done"].mean() * 100, "trust_or": trust_or, "trust_p": trust_p}


def analyze_country_robustness(df: pd.DataFrame, country: str, out_lines: list[str]) -> None:
    """単一設問(E027)ではなく，署名・不買運動・デモ・ストの複合指標での頑健性チェック。"""
    sub = df[df["country"] == country]
    ctrl_str = " + ".join(CONTROLS)
    needed = ["trust_gov", "action_index_done", "income_level", "is_unemployed",
              "is_not_in_labor_force", "age", "sex", "education"]
    data = sub.dropna(subset=needed)

    out_lines.append(f"\n[頑健性チェック: 複合政治的行動指標（署名/不買運動/デモ/ストのいずれか実施経験）]")
    out_lines.append(f"該当率: {data['action_index_done'].mean() * 100:.1f}%  (N={len(data)})")
    if len(data) < 50:
        out_lines.append("サンプルサイズ不足のためスキップ")
        return

    formula = f"action_index_done ~ trust_gov + income_level + is_unemployed + is_not_in_labor_force + {ctrl_str}"
    try:
        model = smf.logit(formula, data=data, missing="drop").fit(disp=0)
    except Exception as e:  # noqa: BLE001
        out_lines.append(f"モデル推定失敗: {e}")
        return
    coef = model.params.get("trust_gov", np.nan)
    p = model.pvalues.get("trust_gov", np.nan)
    or_val = np.exp(coef)
    out_lines.append(f"  trust_gov: coef={coef:+.4f}  OR={or_val:.4f}  p={p:.4f}"
                      f"{'  *有意*' if p < 0.05 else ''}")
    out_lines.append(
        "  解釈: E027単独の結果と方向・有意性が一致すれば，単一設問の測定誤差による"
        "アーティファクトではないことの追加的な裏付けとなる。"
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"対象国合計(Wave6+7): {len(df)}件")

    out_lines: list[str] = [
        "Ricci因果連鎖の『後半』（怨恨・不信 -> 動員行動）の直接検証",
        "X=経済的破壊指標(income_level/is_unemployed/is_not_in_labor_force) -> ",
        "M=trust_gov（政府への信頼） -> Y=protest_done（デモ参加経験，WVS E027）",
        "WVS Time Series Wave 6+7，日本・アメリカ・ドイツ比較",
    ]

    results = {}
    for country in COUNTRIES:
        print(f"[{country}] 分析中...")
        results[country] = analyze_country(df, country, out_lines)
        analyze_country_robustness(df, country, out_lines)

    summary_path = RESULTS_DIR / "wvs_mobilization_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"書き出し: {summary_path}")

    valid = {k: v for k, v in results.items() if v}
    if valid:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(valid.keys()), [v["protest_rate"] for v in valid.values()], color="steelblue")
        ax.set_title("デモ参加経験ありの割合（国別，WVS Wave6+7）")
        ax.set_ylabel("割合(%)")
        fig.tight_layout()
        out_path = RESULTS_DIR / "wvs_mobilization_rates_by_country.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"書き出し: {out_path}")


if __name__ == "__main__":
    main()
