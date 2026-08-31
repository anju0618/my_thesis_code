"""
8トピック全件（comments.csv，約72,000件，LLMコーディング未実施の3トピック
ice_age_precarity/income_wall/takaichi_admin を含む）を対象にした，
ローカルLLMコーディングを一切必要としない3種類の分析。

label_ollama.py によるLLM構造化コーディング（qwen2.5:7b-instruct，1件あたり
約7秒）は新規3トピック計18,000件だけでも完了に半日以上かかるため，コーディング
待ちの間に「LLMを介さない独立測定」だけで先に確認できることをまとめて行う。

  1. 辞書ベース感情スコア（dict_sentiment.pyと同じ簡略実装）のトピック別比較。
     ice_age_precarity（就職氷河期世代の貧困）はRicciの言う『経済的破壊の
     当事者』そのものが語り手であるトピックであり，他トピックより明確に
     ネガティブなトーンが観察されるかを確認する。
  2. コメント投稿量の月別推移（トピック別）。各トピックの言説がいつ盛り上がった
     かを，LLMコーディングなしで素朴に可視化する。
  3. LDAトピックモデル（全8トピック，topic_model_lda.pyと同じ手法）。

使い方:
    uv run src/eight_topic_comparison.py

出力:
    results/eight_topic_dict_sentiment.png / .txt
    results/eight_topic_volume_trend.png
    results/eight_topic_lda_summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import fugashi
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Yu Gothic"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dict_sentiment import SimpleDictSentiment  # noqa: E402

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
COMMENTS_PATH = ANALYSIS_ROOT / "data" / "processed" / "comments.csv"
RESULTS_DIR = ANALYSIS_ROOT / "results"

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
TARGET_POS = {"名詞", "動詞", "形容詞"}
STOPWORDS = {
    "する", "いる", "ある", "なる", "こと", "もの", "これ", "それ", "あれ",
    "よう", "ため", "の", "な", "れる", "られる", "思う", "言う", "見る",
    "くる", "いく", "いう", "そう", "何", "人", "自分", "的", "たち",
}
N_TOPICS_LDA = 15


def load_comments() -> pd.DataFrame:
    df = pd.read_csv(COMMENTS_PATH)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df[df["text_clean"].notna() & (df["text_clean"].str.len() >= 4)]
    print(f"読み込み: {len(df)}件, トピック内訳:\n{df['topic'].value_counts()}")
    return df


def dict_sentiment_by_topic(df: pd.DataFrame, out_lines: list[str]) -> None:
    print("辞書ベース感情スコアを計算中（全8トピック，時間がかかる場合あり）...")
    analyzer = SimpleDictSentiment()
    scores = np.empty(len(df))
    texts = df["text_clean"].tolist()
    for i, t in enumerate(texts):
        s, _, _ = analyzer.score(t)
        scores[i] = s
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(texts)} 件処理済み")
    df = df.copy()
    df["dict_score"] = scores

    out_lines.append(f"\n{'=' * 78}\n1. 辞書ベース感情スコアのトピック別比較（全件, N={len(df)}）\n{'=' * 78}")
    groups, names, means, cis = [], [], [], []
    for topic in TOPIC_LABELS_JA:
        sub = df[df["topic"] == topic]["dict_score"]
        if sub.empty:
            continue
        groups.append(sub)
        names.append(TOPIC_LABELS_JA[topic])
        m = sub.mean()
        se = sub.std(ddof=1) / np.sqrt(len(sub))
        means.append(m)
        cis.append(1.96 * se)
        out_lines.append(f"  {TOPIC_LABELS_JA[topic]:10s}: N={len(sub):6d}  平均={m:+.4f}  "
                          f"95%CI=[{m - 1.96 * se:+.4f}, {m + 1.96 * se:+.4f}]")

    f_stat, p_val = stats.f_oneway(*groups)
    out_lines.append(f"一元配置分散分析（トピック間で辞書スコアが異なるか）: F={f_stat:.3f} p={p_val:.6f}"
                      f"{'  *有意*' if p_val < 0.05 else '  非有意'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:red" if m < 0 else "tab:blue" for m in means]
    ax.bar(names, means, yerr=cis, capsize=5, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"トピック別 辞書ベース感情スコア（平均±95%CI）\nANOVA p={p_val:.4f}")
    ax.set_ylabel("辞書スコア（-1〜+1，負=ネガティブ）")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out_path = RESULTS_DIR / "eight_topic_dict_sentiment.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    out_lines.append(f"書き出し: {out_path}")

    summary_path = RESULTS_DIR / "eight_topic_dict_sentiment.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")


def volume_trend(df: pd.DataFrame) -> None:
    df = df.dropna(subset=["published_at"]).copy()
    df["month"] = df["published_at"].dt.to_period("M").dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 6))
    for topic in TOPIC_LABELS_JA:
        sub = df[df["topic"] == topic]
        monthly = sub.groupby("month").size()
        if monthly.empty:
            continue
        ax.plot(monthly.index, monthly.values, marker="o", markersize=3, label=TOPIC_LABELS_JA[topic])
    ax.set_title("トピック別コメント投稿量の月別推移（全8トピック，LLMコーディング不要の記述統計）")
    ax.set_xlabel("投稿月")
    ax.set_ylabel("コメント件数")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = RESULTS_DIR / "eight_topic_volume_trend.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"書き出し: {out_path}")


def tokenize_ja(tagger: fugashi.Tagger, text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    tokens = []
    for w in tagger(text):
        pos1 = w.feature.pos1
        lemma = w.feature.lemma or w.surface
        if pos1 in TARGET_POS and len(lemma) > 1 and lemma not in STOPWORDS:
            tokens.append(lemma)
    return tokens


def lda_all_topics(df: pd.DataFrame) -> None:
    print("形態素解析中（全8トピック，LDA用）...")
    tagger = fugashi.Tagger()
    texts = df["text_clean"].tolist()
    tokenized = [" ".join(tokenize_ja(tagger, t)) for t in texts]
    df = df.reset_index(drop=True)

    vectorizer = CountVectorizer(min_df=20, max_df=0.4, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(tokenized)
    print(f"語彙数: {len(vectorizer.get_feature_names_out())}, 文書数: {X.shape[0]}")

    print(f"LDA学習中（トピック数={N_TOPICS_LDA}）...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS_LDA, random_state=0, learning_method="online",
        max_iter=15, n_jobs=-1,
    )
    doc_topic = lda.fit_transform(X)
    df["lda_topic"] = doc_topic.argmax(axis=1)

    feature_names = vectorizer.get_feature_names_out()
    out_lines = [
        f"LDAトピックモデル（K={N_TOPICS_LDA}）: 全8トピック統合コーポラ（N={len(df)}）",
        "",
        "各トピックの上位語:",
    ]
    for t in range(N_TOPICS_LDA):
        top_idx = lda.components_[t].argsort()[::-1][:15]
        top_words = [feature_names[i] for i in top_idx]
        n_docs = (df["lda_topic"] == t).sum()
        out_lines.append(f"  LDAトピック{t:2d} (N={n_docs:5d}): {' / '.join(top_words)}")

    out_lines.append(f"\n{'=' * 78}\nLDAトピック × 収集トピック（8トピック）のクロス集計\n{'=' * 78}")
    ct = pd.crosstab(df["lda_topic"], df["topic"])
    out_lines.append(ct.rename(columns=TOPIC_LABELS_JA).to_string())

    summary_path = RESULTS_DIR / "eight_topic_lda_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"書き出し: {summary_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_comments()

    out_lines: list[str] = [
        "8トピック全件比較（LLMコーディング不要の分析: 辞書ベース感情，投稿量トレンド，LDA）",
        f"対象: comments.csv 全件 N={len(df)}（ice_age_precarity / income_wall / "
        "takaichi_admin はまだLLMコーディング未完了のため，本分析はLLMラベルに依存しない"
        "手法のみで実施）",
    ]

    dict_sentiment_by_topic(df, out_lines)
    print("投稿量トレンドを描画中...")
    volume_trend(df)
    lda_all_topics(df)


if __name__ == "__main__":
    main()
