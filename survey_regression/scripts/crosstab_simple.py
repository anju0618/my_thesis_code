"""
機械学習を一切介さない，一番シンプルな記述統計・検定による分析。

ISSP日本データの雇用状態（就業/失業/非労働力）ごとに，政治的疎外感
（no_say）・公務員への信頼（trust_civil）・汚職認知（corruption）の
平均値と95%信頼区間を素朴に比較し，一元配置分散分析（ANOVA）で群間差の
有意性を検定する。SHAPやRandom Forestのような『ブラックボックス』を
経由せず，「経済的破壊の当事者かどうかで政治意識が実際に違うのか」を
最も直接的な形で確認するための分析。

使い方:
    uv run scripts/crosstab_simple.py

出力:
    results/crosstab_simple_summary.txt
    results/crosstab_simple_{target}.png -- 雇用状態別の平均値＋95%CIの棒グラフ
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats

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
WORKYN_LABELS = {1: "就業者", 2: "完全失業者", 3: "非労働力人口"}
TARGETS = {
    "no_say": "政治的疎外感（no_say，高いほど『発言力がない』と感じる）",
    "trust_civil": "公務員への信頼（高いほど信頼）",
    "corruption": "政治家の汚職認知（高いほど汚職していると思う）",
}


def load_japan_data() -> pd.DataFrame:
    df, _ = pyreadstat.read_sav(DATA_PATH)
    df_jp = df[df["country"] == JAPAN_COUNTRY_CODE][
        ["year_sdno", "v61", "v66", "v73", "WORKYN"]
    ].rename(columns={"year_sdno": "year", "v61": "no_say", "v66": "trust_civil", "v73": "corruption"})
    df_jp["year"] = df_jp["year"].astype(int)
    df_jp["employment_group"] = df_jp["WORKYN"].map(WORKYN_LABELS)
    return df_jp


def mean_ci(x: pd.Series, z: float = 1.96) -> tuple[float, float, float, int]:
    x = x.dropna()
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, m - z * se, m + z * se, n


def analyze_target(df: pd.DataFrame, target: str, out_lines: list[str]) -> None:
    data = df.dropna(subset=[target, "employment_group"])
    groups = [data.loc[data["employment_group"] == g, target] for g in WORKYN_LABELS.values()
              if g in data["employment_group"].values]
    group_names = [g for g in WORKYN_LABELS.values() if g in data["employment_group"].values]

    out_lines.append(f"\n{'=' * 70}")
    out_lines.append(f"目的変数: {target} ({TARGETS[target]})  N={len(data)}")
    out_lines.append(f"{'=' * 70}")

    means, los, his = [], [], []
    for name, g in zip(group_names, groups):
        m, lo, hi, n = mean_ci(g)
        means.append(m); los.append(lo); his.append(hi)
        out_lines.append(f"  {name:10s}: 平均={m:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]  N={n}")

    f_stat, p_val = stats.f_oneway(*groups)
    out_lines.append(f"一元配置分散分析（ANOVA）: F={f_stat:.3f}  p={p_val:.4f}"
                      f"{'  *有意（p<0.05）*' if p_val < 0.05 else '  非有意'}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    errs = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
    ax.bar(group_names, means, yerr=errs, capsize=6, color="steelblue")
    ax.set_title(f"雇用状態別の{target}\n(平均値±95%CI, ANOVA p={p_val:.4f})")
    ax.set_ylabel(target)
    fig.tight_layout()
    out_path = RESULTS_DIR / f"crosstab_simple_{target}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    out_lines.append(f"書き出し: {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_japan_data()
    print(f"日本サブセット: {len(df)}件")

    out_lines: list[str] = [
        "雇用状態（就業者/完全失業者/非労働力人口）ごとの政治意識の単純比較",
        "（ANOVA + 95%信頼区間，機械学習を介さない最もシンプルな検定）",
        "ISSP「Role of Government」日本データ（1996/2006/2016年プール）",
    ]
    for target in TARGETS:
        analyze_target(df, target, out_lines)

    summary_path = RESULTS_DIR / "crosstab_simple_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n書き出し: {summary_path}")


if __name__ == "__main__":
    main()
