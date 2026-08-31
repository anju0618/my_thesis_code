"""
LDA（潜在ディリクレ配分法）による，解釈しやすい古典的トピックモデル分析。

embed.py + analyze_coding.py の文埋め込み(multilingual-e5-large)によるK-means
クラスタリングは，各クラスタの意味を「代表コメントを人間が読んで解釈する」
必要がある点でブラックボックス性が残る。LDAは各トピックが「どの単語の
出現確率が高いか」という形でそのまま出力されるため，追加の解釈作業なしに
トピックの内容を確認できる，より透明性の高い代替手法として実施する。

形態素解析には fugashi（MeCab, unidic-lite辞書）を用い，名詞・動詞・
形容詞の基本形のみを抽出してBag-of-Wordsを構成する。

使い方:
    uv run src/topic_model_lda.py

出力:
    results/lda_topics_summary.txt  -- 各トピックの上位語，トピック×既存ラベルのクロス集計
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fugashi
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
LABELED_PATH = ANALYSIS_ROOT / "data" / "processed" / "labeled_comments.jsonl"
SAMPLE_PATH = ANALYSIS_ROOT / "data" / "processed" / "comments_sample.csv"
RESULTS_DIR = ANALYSIS_ROOT / "results"

TARGET_POS = {"名詞", "動詞", "形容詞"}
N_TOPICS = 12
N_TOP_WORDS = 15

STOPWORDS = {
    "する", "いる", "ある", "なる", "こと", "もの", "これ", "それ", "あれ",
    "よう", "ため", "の", "な", "れる", "られる", "思う", "言う", "見る",
    "くる", "いく", "いう", "そう", "何", "人", "自分", "的", "たち",
}


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


def load_labeled_with_text() -> pd.DataFrame:
    records = []
    with open(LABELED_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("parse_error"):
                continue
            records.append(d)
    df = pd.DataFrame(records)

    sample = pd.read_csv(SAMPLE_PATH)[["comment_id", "text_clean"]]
    sample_ids = set(sample["comment_id"])
    df = df[df["comment_id"].isin(sample_ids)]
    df = df[df["is_spam_or_offtopic"] != True]  # noqa: E712
    df = df.merge(sample, on="comment_id", how="left")
    df = df[df["text_clean"].notna() & (df["text_clean"].str.len() >= 4)]
    print(f"分析対象: {len(df)}件")
    return df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labeled_with_text()

    print("形態素解析（名詞・動詞・形容詞の抽出）中...")
    tagger = fugashi.Tagger()
    tokenized = [" ".join(tokenize_ja(tagger, t)) for t in df["text_clean"].tolist()]
    df = df.reset_index(drop=True)

    vectorizer = CountVectorizer(min_df=10, max_df=0.4, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(tokenized)
    print(f"語彙数: {len(vectorizer.get_feature_names_out())}, 文書数: {X.shape[0]}")

    print(f"LDA学習中（トピック数={N_TOPICS}）...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS, random_state=0, learning_method="online",
        max_iter=20, n_jobs=-1,
    )
    doc_topic = lda.fit_transform(X)
    df["lda_topic"] = doc_topic.argmax(axis=1)

    feature_names = vectorizer.get_feature_names_out()
    out_lines: list[str] = [
        f"LDAトピックモデル（K={N_TOPICS}）による探索的分析（埋め込みクラスタリングの代替手法）",
        f"対象: {len(df)}件（形態素解析後，語彙数{len(feature_names)}）",
        "",
        "各トピックの上位語（確率の高い順）:",
    ]
    for t in range(N_TOPICS):
        top_idx = lda.components_[t].argsort()[::-1][:N_TOP_WORDS]
        top_words = [feature_names[i] for i in top_idx]
        n_docs = (df["lda_topic"] == t).sum()
        out_lines.append(f"  トピック{t:2d} (N={n_docs:5d}): {' / '.join(top_words)}")

    out_lines.append(f"\n{'=' * 78}\nLDAトピック × 収集トピック（財務省デモ/参政党/不法移民）のクロス集計\n{'=' * 78}")
    out_lines.append(pd.crosstab(df["lda_topic"], df["topic"]).to_string())

    if "blame_target" in df.columns:
        out_lines.append(f"\n{'=' * 78}\nLDAトピック × blame_target（LLMコーディング）のクロス集計\n{'=' * 78}")
        out_lines.append(pd.crosstab(df["lda_topic"], df["blame_target"]).to_string())

    for var in ["people_vs_elite", "economic_resentment", "theft_of_enjoyment"]:
        if var in df.columns:
            out_lines.append(f"\nLDAトピックごとの{var}該当率:")
            rates = df.groupby("lda_topic")[var].mean().round(3)
            out_lines.append(rates.to_string())

    summary_path = RESULTS_DIR / "lda_topics_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n書き出し: {summary_path}")


if __name__ == "__main__":
    main()
