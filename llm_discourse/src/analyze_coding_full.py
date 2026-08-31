"""
構造化コーディング結果（labeled_comments.jsonl）の集計・検定・探索的分析
——全8トピック版。

analyze_coding.py（財務省デモ・参政党・不法移民の3トピック，卒論本文
sections/04_empirical.tex が既に参照している確定版）に，2026年8月に追加で
LLMコーディングした5トピック（氷河期世代・年収の壁・高市政権・大阪維新の会・
れいわ新選組）を加えた全8トピック（計約48,000件）で同じ3種類の分析を
やり直したもの。既存の3トピック版の結果・卒論本文の記述には一切影響しない
（analyze_coding.pyは変更していない）。

新たに追加した2トピック（大阪維新の会・れいわ新選組）は，Mudde &
Rovira Kaltwasser (2017) の「ポピュリズムは左右どちらにも付着しうる薄い
イデオロギー」というテーゼを検証するため，右派ポピュリズム的言説とされる
参政党と，左派ポピュリズム的言説とされるれいわ新選組を対比する目的で
追加したもの。氷河期世代・年収の壁は，Ricciの言う「経済的破壊の当事者」
そのものが語り手となる言説を捉える目的で追加した。

使い方:
    uv run src/analyze_coding_full.py

出力:
    results/coding_full_summary_{件数}.txt
    results/coding_full_blame_target_heatmap_{件数}.png
    results/coding_full_cluster_{件数}.png
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
SAMPLE_PATH = ANALYSIS_ROOT / "data" / "processed" / "comments_sample_all8.csv"
EMBEDDINGS_PATH = ANALYSIS_ROOT / "data" / "processed" / "final_embeddings.npy"
EMBEDDINGS_IDS_PATH = ANALYSIS_ROOT / "data" / "processed" / "final_embeddings_ids.csv"
RESULTS_DIR = ANALYSIS_ROOT / "results"

BOOL_VARS = ["people_vs_elite", "economic_resentment", "theft_of_enjoyment"]
TOPIC_LABELS_JA = {
    "zaimusho_demo": "財務省デモ",
    "sanseito": "参政党",
    "immigration": "不法移民",
    "ice_age_precarity": "氷河期世代",
    "income_wall": "年収の壁",
    "takaichi_admin": "高市政権",
    "ishin_osaka": "大阪維新の会",
    "reiwa_shinsengumi": "れいわ新選組",
}


def load_labeled() -> pd.DataFrame:
    records = []
    n_parse_error = 0
    with open(LABELED_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("parse_error"):
                n_parse_error += 1
                continue
            records.append(d)
    df = pd.DataFrame(records)
    print(f"読み込み: {len(df) + n_parse_error}件中 parse_error {n_parse_error}件を除外，"
          f"{len(df)}件を分析対象とする")

    n_before_sample_filter = len(df)
    sample_ids = set(pd.read_csv(SAMPLE_PATH)["comment_id"])
    df = df[df["comment_id"].isin(sample_ids)]
    n_excluded = n_before_sample_filter - len(df)
    if n_excluded:
        print(f"最終サンプル（comments_sample_all8.csv, N={len(sample_ids)}）に含まれない"
              f"{n_excluded}件を除外，{len(df)}件が残る")

    n_before = len(df)
    df = df[df["is_spam_or_offtopic"] != True]  # noqa: E712
    print(f"is_spam_or_offtopic除外: {n_before - len(df)}件除外，{len(df)}件が残る")

    if "blame_target" in df.columns:
        n_normalized = 0

        def _normalize_bt(v):
            nonlocal n_normalized
            if not isinstance(v, str):
                return v
            if v == "foreigners":
                n_normalized += 1
                return "immigrants"
            if "|" in v:
                n_normalized += 1
                return v.split("|")[0]
            return v

        df["blame_target"] = df["blame_target"].apply(_normalize_bt)
        if n_normalized:
            print(f"blame_targetの表記ゆれ・複合値を正規化: {n_normalized}件")
    return df


def wilson_score_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return p_hat, center - margin, center + margin


def occurrence_rates(df: pd.DataFrame, out_lines: list[str]) -> None:
    out_lines.append(f"\n{'=' * 70}\n1. 各トピックにおける変数の出現率（Wilsonスコア区間，95%）\n{'=' * 70}")
    for topic in TOPIC_LABELS_JA:
        if topic not in df["topic"].unique():
            continue
        sub = df[df["topic"] == topic]
        out_lines.append(f"\n--- {topic} ({TOPIC_LABELS_JA.get(topic, topic)}), N={len(sub)} ---")
        for var in BOOL_VARS:
            valid = sub[var].dropna()
            n = len(valid)
            k = int(valid.sum()) if n > 0 else 0
            p_hat, lo, hi = wilson_score_interval(k, n)
            n_null = len(sub) - n
            out_lines.append(
                f"  {var:22s}: {k:4d}/{n:4d} = {p_hat * 100:5.1f}%  "
                f"[{lo * 100:5.1f}%, {hi * 100:5.1f}%]  (欠測 {n_null}件)"
            )

    out_lines.append(f"\n{'=' * 70}\n1b. 左右ポピュリズム対比: 参政党（右派） vs れいわ新選組（左派）\n{'=' * 70}")
    out_lines.append(
        "Mudde & Rovira Kaltwasser (2017) の「ポピュリズムは薄いイデオロギーで左右どちらにも"
        "付着しうる」というテーゼが正しければ，people_vs_elite（人民対エリート図式）の出現率は"
        "両党で同水準に近いはずである。"
    )
    for var in BOOL_VARS:
        r = df[df["topic"] == "sanseito"][var].dropna()
        l = df[df["topic"] == "reiwa_shinsengumi"][var].dropna()
        if len(r) == 0 or len(l) == 0:
            continue
        ct = pd.DataFrame({
            "count": [int(r.sum()), int(l.sum())],
            "n": [len(r), len(l)],
        }, index=["sanseito", "reiwa_shinsengumi"])
        contingency = np.array([[ct.loc["sanseito", "count"], ct.loc["sanseito", "n"] - ct.loc["sanseito", "count"]],
                                 [ct.loc["reiwa_shinsengumi", "count"], ct.loc["reiwa_shinsengumi", "n"] - ct.loc["reiwa_shinsengumi", "count"]]])
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        out_lines.append(
            f"  {var:22s}: 参政党={ct.loc['sanseito', 'count'] / ct.loc['sanseito', 'n'] * 100:5.1f}%  "
            f"れいわ={ct.loc['reiwa_shinsengumi', 'count'] / ct.loc['reiwa_shinsengumi', 'n'] * 100:5.1f}%  "
            f"カイ二乗 chi2={chi2:.2f} p={p:.6f}{'  *有意差あり*' if p < 0.05 else '  有意差なし（同水準）'}"
        )


def chi_square_blame_target(df: pd.DataFrame, out_lines: list[str]) -> None:
    out_lines.append(f"\n{'=' * 70}\n2. blame_targetの分布：トピック間のカイ二乗検定（全8トピック）\n{'=' * 70}")
    sub = df.dropna(subset=["blame_target"])
    sub = sub[sub["blame_target"] != "none"]
    ct = pd.crosstab(sub["topic"], sub["blame_target"])
    out_lines.append("\n分割表（度数）:")
    out_lines.append(ct.to_string())

    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = np.sqrt((chi2 / n) / (min(ct.shape) - 1))
    out_lines.append(f"\nカイ二乗統計量: {chi2:.2f}, 自由度: {dof}, p値: {p:.6f}")
    out_lines.append(f"Cramér's V: {cramers_v:.4f}")

    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    row_totals = ct.sum(axis=1)
    col_totals = ct.sum(axis=0)
    resid = (ct - expected_df) / np.sqrt(
        expected_df
        * (1 - row_totals.values.reshape(-1, 1) / n)
        * (1 - col_totals.values.reshape(1, -1) / n)
    )
    out_lines.append("\n標準化残差（|値|が大きいほど期待値からの乖離が大きい）:")
    out_lines.append(resid.round(2).to_string())

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(resid.values, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    ax.set_xticks(range(len(resid.columns)))
    ax.set_xticklabels(resid.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(resid.index)))
    ax.set_yticklabels([TOPIC_LABELS_JA.get(t, t) for t in resid.index])
    for i in range(resid.shape[0]):
        for j in range(resid.shape[1]):
            ax.text(j, i, f"{resid.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="標準化残差")
    ax.set_title(f"非難の矛先(blame_target)×トピック 標準化残差（全8トピック, N={n}）")
    fig.tight_layout()
    out_path = RESULTS_DIR / f"coding_full_blame_target_heatmap_{len(df)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    out_lines.append(f"\n書き出し: {out_path}")


def embedding_cluster(df: pd.DataFrame, out_lines: list[str]) -> None:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    out_lines.append(f"\n{'=' * 70}\n3. 文埋め込みによる探索的クラスタリング（全8トピック）\n{'=' * 70}")

    if not EMBEDDINGS_PATH.exists():
        out_lines.append("embeddings.npyが見つからないためスキップ（先にembed.pyを実行すること）")
        return

    embeddings = np.load(EMBEDDINGS_PATH)
    ids = pd.read_csv(EMBEDDINGS_IDS_PATH)
    ids["emb_idx"] = range(len(ids))

    merged = df.merge(ids[["comment_id", "emb_idx"]], on="comment_id", how="inner")
    coverage_note = (
        "（カバー率100%）" if len(merged) == len(df)
        else f"（埋め込みが{len(df) - len(merged)}件不足）"
    )
    out_lines.append(
        f"ラベリング済み{len(df)}件のうち，埋め込みが存在する{len(merged)}件を"
        f"クラスタリング対象とする{coverage_note}。"
    )
    if len(merged) < 20:
        out_lines.append("対象件数が少なすぎるためクラスタリングをスキップ")
        return

    X = embeddings[merged["emb_idx"].values]

    k_range = range(2, 12)
    sils = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=0)
        sils.append(sil)
        out_lines.append(f"  K={k}: silhouette={sil:.4f}")
    best_k = list(k_range)[int(np.argmax(sils))]
    out_lines.append(f"シルエット係数最大のK: {best_k}")

    km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    merged["cluster"] = km.fit_predict(X)

    out_lines.append(f"\nクラスタ×トピックの分布 (K={best_k}):")
    out_lines.append(pd.crosstab(merged["cluster"], merged["topic"]).rename(columns=TOPIC_LABELS_JA).to_string())

    out_lines.append("\nクラスタ×主要変数の該当率:")
    for var in BOOL_VARS:
        rates = merged.groupby("cluster")[var].mean().round(3)
        out_lines.append(f"  {var}: {rates.to_dict()}")

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=merged["cluster"], cmap="tab10", s=6, alpha=0.5)
    ax.set_title(f"文埋め込みクラスタ（PCA 2次元投影, K={best_k}, N={len(merged)}, 全8トピック）")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend1 = ax.legend(*scatter.legend_elements(), title="クラスタ", loc="upper right", fontsize=8)
    ax.add_artist(legend1)
    fig.tight_layout()
    out_path = RESULTS_DIR / f"coding_full_cluster_{len(df)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    out_lines.append(f"\n書き出し: {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labeled()

    n_sample_total = len(pd.read_csv(SAMPLE_PATH))
    out_lines: list[str] = [
        f"構造化コーディング結果の集計（全8トピック版, is_spam_or_offtopic除外後の分析対象: "
        f"{len(df)}件，目標サンプル{n_sample_total}件中）",
    ]

    occurrence_rates(df, out_lines)
    chi_square_blame_target(df, out_lines)
    embedding_cluster(df, out_lines)

    summary_path = RESULTS_DIR / f"coding_full_summary_{len(df)}.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n書き出し: {summary_path}")


if __name__ == "__main__":
    main()
