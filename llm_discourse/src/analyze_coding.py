"""
構造化コーディング結果（labeled_comments.jsonl）の集計・検定・探索的分析。

sections/04_empirical.tex の「分析の進め方」節（subsec:formalization）で
予告した4種類の分析のうち，人手検証（Cohenのkappa）を除く3種類を実施する：
  1. 各トピックにおける people_vs_elite / economic_resentment /
     theft_of_enjoyment の出現率と，Wilsonスコア信頼区間
     （sections/90_appendix_math.tex の定式化に対応，中心を再補正した
     正しい版を実装する）。
  2. blame_target の分布がトピック間で異なるかのカイ二乗検定・Cramér's V・
     標準化残差（Fujishiro et al. 2020のベンチマーク検証）。
  3. 文埋め込み（embed.py で作成済み）を用いた K-means 探索的クラスタリング
     （シルエット係数で K を選定，issp_ml_analysis.py と同じ方針）。

このスクリプトは labeled_comments.jsonl が完成していない時点でも実行できる
（コーディング中のサブセットに対する暫定集計として使う）。実行のたびに
現時点のコーディング件数を明記した出力を残すため，最終版とは別に日付入りの
ファイル名で結果を保存する。

使い方:
    uv run src/analyze_coding.py

出力:
    results/coding_summary_{件数}.txt   -- 出現率・カイ二乗検定・クラスタ分析の要約
    results/coding_blame_target_heatmap_{件数}.png -- トピック×非難対象の残差ヒートマップ
    results/coding_cluster_{件数}.png   -- 埋め込みクラスタのUMAP風2次元散布図（PCA使用）
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
# final_embeddings.npy/csv はコーディング済み全件（comments_labeled_subset.csv）に
# 対して embed.py --device cpu を再実行して作成したもので，旧sample_embeddings.npy
# （初期6,000件サンプルのみ）より広い範囲をカバーする。
EMBEDDINGS_PATH = ANALYSIS_ROOT / "data" / "processed" / "final_embeddings.npy"
EMBEDDINGS_IDS_PATH = ANALYSIS_ROOT / "data" / "processed" / "final_embeddings_ids.csv"
RESULTS_DIR = ANALYSIS_ROOT / "results"

BOOL_VARS = ["people_vs_elite", "economic_resentment", "theft_of_enjoyment"]
TOPIC_LABELS_JA = {
    "zaimusho_demo": "財務省デモ",
    "sanseito": "参政党",
    "immigration": "不法移民",
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

    # is_spam_or_offtopic=True のコメントは除外（コーディングスキーム通り）
    n_before = len(df)
    df = df[df["is_spam_or_offtopic"] != True]  # noqa: E712
    print(f"is_spam_or_offtopic除外: {n_before - len(df)}件除外，{len(df)}件が残る")

    # blame_target の表記ゆれ・複合値を正規化する。
    # コーディングスキーム（config/coding_scheme.yaml）が定める7分類
    # （media/government/specific_party/foreign_country/immigrants/
    # elites_general/none）以外の値がLLM出力に稀に混入する
    # （例："foreigners"という表記ゆれ，"immigrants|elites_general"の
    # ようなパイプ区切りの複合値）。前者は正規のカテゴリ名に補正し，
    # 後者は最初の要素を採用する（件数はごく少数，2026-08-12時点で
    # 全5,849件中18件＝0.3%）。
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
    """正しいWilsonスコア区間（sections/90_appendix_math.texの修正版に対応）。

    中心を p_hat ではなく p_tilde = (p_hat + z^2/2n) / (1 + z^2/n) に補正し，
    マージン項 [z/(1+z^2/n)] * sqrt(p_hat(1-p_hat)/n + z^2/4n^2) を1回だけ加減する。
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return p_hat, center - margin, center + margin


def occurrence_rates(df: pd.DataFrame, out_lines: list[str]) -> None:
    out_lines.append(f"\n{'=' * 70}\n1. 各トピックにおける変数の出現率（Wilsonスコア区間，95%）\n{'=' * 70}")
    for topic in df["topic"].unique():
        sub = df[df["topic"] == topic]
        out_lines.append(f"\n--- {topic} ({TOPIC_LABELS_JA.get(topic, topic)}), N={len(sub)} ---")
        for var in BOOL_VARS:
            valid = sub[var].dropna()
            n = len(valid)
            k = int(valid.sum()) if n > 0 else 0
            p_hat, lo, hi = wilson_score_interval(k, n)
            n_null = len(sub) - n
            out_lines.append(
                f"  {var:22s}: {k:4d}/{n:4d} = {p_hat*100:5.1f}%  "
                f"[{lo*100:5.1f}%, {hi*100:5.1f}%]  (欠測 {n_null}件)"
            )


def chi_square_blame_target(df: pd.DataFrame, out_lines: list[str]) -> None:
    out_lines.append(f"\n{'=' * 70}\n2. blame_targetの分布：トピック間のカイ二乗検定\n{'=' * 70}")
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

    # 標準化残差
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

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(resid.values, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    ax.set_xticks(range(len(resid.columns)))
    ax.set_xticklabels(resid.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(resid.index)))
    ax.set_yticklabels([TOPIC_LABELS_JA.get(t, t) for t in resid.index])
    for i in range(resid.shape[0]):
        for j in range(resid.shape[1]):
            ax.text(j, i, f"{resid.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="標準化残差")
    ax.set_title(f"非難の矛先(blame_target)×トピック 標準化残差 (N={n}, 暫定集計)")
    fig.tight_layout()
    out_path = RESULTS_DIR / f"coding_blame_target_heatmap_{len(df)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    out_lines.append(f"\n書き出し: {out_path}")


def embedding_cluster(df: pd.DataFrame, out_lines: list[str]) -> None:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    out_lines.append(f"\n{'=' * 70}\n3. 文埋め込みによる探索的クラスタリング\n{'=' * 70}")

    if not EMBEDDINGS_PATH.exists():
        out_lines.append("embeddings.npyが見つからないためスキップ（先にembed.pyを実行すること）")
        return

    embeddings = np.load(EMBEDDINGS_PATH)
    ids = pd.read_csv(EMBEDDINGS_IDS_PATH)
    ids["emb_idx"] = range(len(ids))

    merged = df.merge(ids[["comment_id", "emb_idx"]], on="comment_id", how="inner")
    out_lines.append(
        f"ラベリング済み{len(df)}件のうち，埋め込みが存在する{len(merged)}件を"
        f"クラスタリング対象とする（埋め込みはsample_embeddings.npy作成時点の"
        f"6,000件サンプルに基づくため，その後拡大したサンプルの全件はカバーしない。"
        f"最終版ではembed.pyを最新のcomments_sample.csvに対して再実行すること）。"
    )
    if len(merged) < 20:
        out_lines.append("対象件数が少なすぎるためクラスタリングをスキップ")
        return

    X = embeddings[merged["emb_idx"].values]

    k_range = range(2, 8)
    sils = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=0)
        sils.append(sil)
        out_lines.append(f"  K={k}: silhouette={sil:.4f}")
    best_k = list(k_range)[int(np.argmax(sils))]
    out_lines.append(f"シルエット係数最大のK: {best_k}")

    km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    merged["cluster"] = km.fit_predict(X)

    out_lines.append(f"\nクラスタ×トピックの分布 (K={best_k}):")
    out_lines.append(pd.crosstab(merged["cluster"], merged["topic"]).to_string())

    out_lines.append(f"\nクラスタ×主要変数の該当率:")
    for var in [*BOOL_VARS]:
        rates = merged.groupby("cluster")[var].mean().round(3)
        out_lines.append(f"  {var}: {rates.to_dict()}")

    # 代表コメント（各クラスタの重心に最も近い3件）をrationale_jaと共に記録
    out_lines.append("\n各クラスタの代表コメント（重心に最も近い3件，rationale_ja付き）:")
    for c in range(best_k):
        mask = merged["cluster"] == c
        idx_in_cluster = np.where(mask.values)[0]
        Xc = X[idx_in_cluster]
        centroid = Xc.mean(axis=0)
        dists = np.linalg.norm(Xc - centroid, axis=1)
        top3 = idx_in_cluster[np.argsort(dists)[:3]]
        out_lines.append(f"\n  クラスタ{c} (N={mask.sum()}):")
        for i in top3:
            row = merged.iloc[i]
            text_preview = str(row.get("rationale_ja", ""))[:80]
            out_lines.append(
                f"    topic={row['topic']}, blame_target={row.get('blame_target')}, "
                f"tone={row.get('emotional_tone')}: {text_preview}"
            )

    # PCA 2次元プロット
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=merged["cluster"], cmap="tab10", s=8, alpha=0.6)
    ax.set_title(f"文埋め込みクラスタ（PCA 2次元投影, K={best_k}, N={len(merged)}, 暫定集計）")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend1 = ax.legend(*scatter.legend_elements(), title="クラスタ", loc="upper right", fontsize=8)
    ax.add_artist(legend1)
    fig.tight_layout()
    out_path = RESULTS_DIR / f"coding_cluster_{len(df)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    out_lines.append(f"\n書き出し: {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labeled()

    out_lines: list[str] = [
        f"構造化コーディング結果の暫定集計（実行時点でのラベリング済み件数: {len(df)}件）",
        "※ コーディング処理は本稿執筆時点で継続中であり，これは最終結果ではない。",
    ]

    occurrence_rates(df, out_lines)
    chi_square_blame_target(df, out_lines)
    embedding_cluster(df, out_lines)

    summary_path = RESULTS_DIR / f"coding_summary_{len(df)}.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n書き出し: {summary_path}")


if __name__ == "__main__":
    main()
