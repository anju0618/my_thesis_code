"""
Ricciの因果連鎖（経済的破壊 -> 怨恨・政治的疎外感 -> エリートへの不信）を，
SHAPのような間接的な特徴量重要度ではなく，媒介分析（mediation analysis /
Baron & Kenny 1986の枠組み + ブートストラップ法による間接効果の検定）で
直接検証する。

ISSP「Role of Government」日本データを使う。ISSPには「怨恨」そのものを
測る設問がないため，本スクリプトでは
    X（経済的破壊の当事者性） = 雇用状態（is_unemployed / is_not_in_labor_force）
    M（政治的疎外感，怨恨の代理変数） = no_say（v61: 「自分のような人間には
        政府のすることに発言力がない」への同意度）
    Y（エリートへの不信・反エリート認知） = trust_civil（公務員への信頼の低さ）
        および corruption（政治家は汚職している，という認知）
という代理変数を用いる。M=no_sayを「怨恨」の直接的な測定とみなすのは
理論的な飛躍であり，この限界は出力に明記する。

手法: 3本のOLS回帰（a経路: M~X, b/c'経路: Y~X+M, 総効果c: Y~X）を
それぞれcontrols（年齢・性別・学歴・調査年）付きで推定し，間接効果
a*bの有意性をパーセンタイル・ブートストラップ（resample=2000）で検定する。

使い方:
    uv run scripts/mediation_analysis.py

出力:
    results/mediation_summary.txt
    results/mediation_diagram_{x}_{y}.png -- a, b, c' の係数を書き込んだ簡易パス図
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
DATA_PATH = ROOT / "data" / "issp_role_of_government" / "Role of Government_ISSP.sav"
RESULTS_DIR = ROOT / "results"

JAPAN_COUNTRY_CODE = 392
N_BOOT = 1000
RANDOM_SEED = 0

X_VARS = {
    "is_unemployed": "完全失業（対 就業者）",
    "is_not_in_labor_force": "非労働力人口（対 就業者）",
}
M_VAR = "no_say"
M_LABEL = "政治的疎外感（no_say: 発言力の欠如）"
Y_VARS = {
    "trust_civil": "公務員への信頼（低いほどエリート不信が強い）",
    "corruption": "政治家の汚職認知（高いほど反エリート的）",
}
CONTROLS = ["age", "C(sex)", "C(education)", "C(year)"]


def load_japan_data() -> pd.DataFrame:
    df, _ = pyreadstat.read_sav(DATA_PATH)
    df_jp = df[df["country"] == JAPAN_COUNTRY_CODE][
        ["year_sdno", "v61", "v66", "v73", "AGE", "SEX", "DEGREE", "WORKYN"]
    ].rename(
        columns={
            "year_sdno": "year",
            "v61": "no_say",
            "v66": "trust_civil",
            "v73": "corruption",
            "AGE": "age",
            "SEX": "sex",
            "DEGREE": "education",
        }
    )
    df_jp["year"] = df_jp["year"].astype(int)
    df_jp["is_unemployed"] = (df_jp["WORKYN"] == 2).astype(float)
    df_jp["is_not_in_labor_force"] = (df_jp["WORKYN"] == 3).astype(float)
    df_jp.loc[df_jp["WORKYN"].isna(), ["is_unemployed", "is_not_in_labor_force"]] = np.nan
    return df_jp


def fit_ols(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data, missing="drop").fit()


def mediation_once(data: pd.DataFrame, x: str, m: str, y: str, controls: list[str]) -> tuple[float, float, float]:
    """1回分の a, b, c' を返す（ブートストラップの1反復として使う）。"""
    ctrl_str = " + ".join(controls)
    a_model = fit_ols(f"{m} ~ {x} + {ctrl_str}", data)
    a = a_model.params.get(x, np.nan)
    b_model = fit_ols(f"{y} ~ {x} + {m} + {ctrl_str}", data)
    b = b_model.params.get(m, np.nan)
    c_prime = b_model.params.get(x, np.nan)
    return a, b, c_prime


def bootstrap_indirect(data: pd.DataFrame, x: str, m: str, y: str, controls: list[str],
                        n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(data)
    indirect = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = data.iloc[idx]
        try:
            a, b, _ = mediation_once(boot, x, m, y, controls)
            indirect[i] = a * b
        except Exception:
            indirect[i] = np.nan
    return indirect


def run_mediation(df: pd.DataFrame, x: str, m: str, y: str, controls: list[str],
                   out_lines: list[str]) -> None:
    needed = [x, m, y, "age", "sex", "education", "year"]
    data = df.dropna(subset=needed).copy()
    ctrl_str = " + ".join(controls)

    out_lines.append(f"\n{'=' * 78}")
    out_lines.append(f"媒介分析: X={x} ({X_VARS[x]}) -> M={m} ({M_LABEL}) -> Y={y} ({Y_VARS[y]})  N={len(data)}")
    out_lines.append(f"{'=' * 78}")

    total_model = fit_ols(f"{y} ~ {x} + {ctrl_str}", data)
    c_total = total_model.params.get(x, np.nan)
    c_total_p = total_model.pvalues.get(x, np.nan)

    a_model = fit_ols(f"{m} ~ {x} + {ctrl_str}", data)
    a = a_model.params.get(x, np.nan)
    a_p = a_model.pvalues.get(x, np.nan)

    full_model = fit_ols(f"{y} ~ {x} + {m} + {ctrl_str}", data)
    b = full_model.params.get(m, np.nan)
    b_p = full_model.pvalues.get(m, np.nan)
    c_prime = full_model.params.get(x, np.nan)
    c_prime_p = full_model.pvalues.get(x, np.nan)

    indirect_point = a * b
    boot = bootstrap_indirect(data, x, m, y, controls)
    boot = boot[~np.isnan(boot)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    sig = "有意（95%CIが0を含まない）" if (ci_lo > 0 or ci_hi < 0) else "非有意（95%CIが0を含む）"

    out_lines.append(f"総効果 c  (X -> Y):            {c_total:+.4f}  p={c_total_p:.4f}")
    out_lines.append(f"経路 a    (X -> M):            {a:+.4f}  p={a_p:.4f}")
    out_lines.append(f"経路 b    (M -> Y, X統制済):    {b:+.4f}  p={b_p:.4f}")
    out_lines.append(f"直接効果 c' (X -> Y, M統制済):  {c_prime:+.4f}  p={c_prime_p:.4f}")
    out_lines.append(f"間接効果 a*b (点推定):          {indirect_point:+.4f}")
    out_lines.append(f"間接効果のブートストラップ95%CI ({len(boot)}/{N_BOOT}回成功): "
                      f"[{ci_lo:+.4f}, {ci_hi:+.4f}]  -> {sig}")
    if abs(c_total) > 1e-9:
        out_lines.append(f"媒介比率（間接効果/総効果）: {indirect_point / c_total * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.axis("off")
    ax.text(0.05, 0.8, f"X: {X_VARS[x]}", ha="left", fontsize=11, bbox=dict(boxstyle="round", fc="lightblue"))
    ax.text(0.42, 0.8, f"M: {M_LABEL}", ha="left", fontsize=11, bbox=dict(boxstyle="round", fc="lightyellow"))
    ax.text(0.75, 0.8, f"Y: {Y_VARS[y]}", ha="left", fontsize=11, bbox=dict(boxstyle="round", fc="lightgreen"))
    ax.annotate("", xy=(0.42, 0.85), xytext=(0.22, 0.85), arrowprops=dict(arrowstyle="->"))
    ax.text(0.28, 0.9, f"a={a:+.3f}{'*' if a_p < .05 else ''}", fontsize=9)
    ax.annotate("", xy=(0.75, 0.85), xytext=(0.55, 0.85), arrowprops=dict(arrowstyle="->"))
    ax.text(0.6, 0.9, f"b={b:+.3f}{'*' if b_p < .05 else ''}", fontsize=9)
    ax.annotate("", xy=(0.75, 0.6), xytext=(0.1, 0.6), arrowprops=dict(arrowstyle="->", linestyle="dashed"))
    ax.text(0.35, 0.5, f"c'={c_prime:+.3f}{'*' if c_prime_p < .05 else ''} （直接効果，Mを統制）", fontsize=9)
    ax.set_title(f"媒介分析パス図: {x} -> {m} -> {y}\n間接効果={indirect_point:+.3f} 95%CI=[{ci_lo:+.3f},{ci_hi:+.3f}] ({sig})",
                 fontsize=10)
    fig.tight_layout()
    out_path = RESULTS_DIR / f"mediation_diagram_{x}_{y}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    out_lines.append(f"書き出し: {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_japan_data()
    print(f"日本サブセット: {len(df)}件")

    out_lines: list[str] = [
        "Ricci因果連鎖の媒介分析（Baron&Kenny 1986の枠組み + ブートストラップ間接効果検定）",
        "ISSP「Role of Government」日本データ（1996/2006/2016年）",
        "",
        "【重要な限界】ISSPには『怨恨（resentment）』を直接測定する設問が存在しない。",
        "M=no_say（『自分のような人間には政府のすることに発言力がない』への同意度）を",
        "怨恨・政治的疎外感の代理変数として用いるが，これは理論的な飛躍を含む",
        "近似であり，Ricciの言う『怨恨』そのものではない点に留意が必要。",
        "controls: age, C(sex), C(education), C(year)",
    ]

    for x in X_VARS:
        for y in Y_VARS:
            print(f"[{x} -> {M_VAR} -> {y}] 媒介分析実行中（ブートストラップ{N_BOOT}回）...")
            run_mediation(df, x, M_VAR, y, CONTROLS, out_lines)

    summary_path = RESULTS_DIR / "mediation_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n書き出し: {summary_path}")


if __name__ == "__main__":
    main()
