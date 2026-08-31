"""
辞書ベース感情分析による，LLM構造化コーディングの独立した頑健性検証。

analyze_coding.py で報告した通り，economic_resentment（kappa=-0.033）・
theft_of_enjoyment（kappa=0.000）の人手検証によるコーダー内信頼性はほぼ
ゼロであり，これら2変数に基づく結論は暫定的なものにとどまる
（CLAUDE.md, sections/04_empirical.tex参照）。本スクリプトはLLMに一切
依存しない別の測定手法——東北大学 乾・岡崎研究室の日本語評価極性辞書
（Takamura, Inui & Okumura 2005; および用言表現辞書，高村・乾・奥村ほかの
後続版）による語彙ベースの極性判定——で同じコメント群を独立に測定し，
LLMコーディング結果とどの程度対応するかを確認する（三角測量による
頑健性チェックであり，κの低さそのものを解消するものではない）。

【実装上の注記】上記辞書はPythonパッケージ oseti が内部で使っているものを
そのまま流用する。oseti本体は文分割ライブラリ bunkai が transformers の
新しいバージョンとAPI非互換を起こし本環境ではimportに失敗するため，
oseti既存の辞書ファイル（pn_noun.json: 名詞極性辞書, pn_wago.json:
用言表現極性辞書）のみを読み込み，fugashi（MeCab, unidic-lite辞書）で
得た各形態素の基本形（lemma）を単純に辞書照合する簡略版を実装した。
oseti本来のアルゴリズム（文分割・隣接語との複合表現照合・否定辞による
極性反転）は再現していない，より単純なbag-of-words式の照合である点に
留意が必要（本スクリプトは頑健性チェックのための補助分析であり，
分析2の主結果ではない）。

使い方:
    uv run src/dict_sentiment.py

出力:
    results/dict_sentiment_summary.txt
    results/dict_sentiment_by_tone.png -- LLMのemotional_tone別，辞書スコアの分布
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import fugashi
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
SAMPLE_PATH = ANALYSIS_ROOT / "data" / "processed" / "comments_sample.csv"
RESULTS_DIR = ANALYSIS_ROOT / "results"

BOOL_VARS = ["people_vs_elite", "economic_resentment", "theft_of_enjoyment"]
TONE_ORDER = ["neutral_factual", "resignation", "fear", "contempt", "anger"]


def find_oseti_dict_dir() -> Path:
    spec = importlib.util.find_spec("oseti")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("osetiパッケージが見つからない（uv add osetiを確認）")
    return Path(list(spec.submodule_search_locations)[0]) / "dic"


def load_polarity_dicts() -> tuple[dict, dict]:
    dic_dir = find_oseti_dict_dir()
    with open(dic_dir / "pn_noun.json", encoding="utf-8") as f:
        word_dict = json.load(f)
    with open(dic_dir / "pn_wago.json", encoding="utf-8") as f:
        wago_dict = json.load(f)
    return word_dict, wago_dict


class SimpleDictSentiment:
    def __init__(self) -> None:
        self.word_dict, self.wago_dict = load_polarity_dicts()
        self.tagger = fugashi.Tagger()

    def score(self, text: str) -> tuple[float, int, int]:
        if not isinstance(text, str) or not text.strip():
            return 0.0, 0, 0
        pos, neg = 0, 0
        for w in self.tagger(text):
            lemma = w.feature.lemma or w.surface
            if lemma in self.word_dict:
                if self.word_dict[lemma] == "p":
                    pos += 1
                else:
                    neg += 1
            elif lemma in self.wago_dict:
                if self.wago_dict[lemma].startswith("ポジ"):
                    pos += 1
                else:
                    neg += 1
        total = pos + neg
        norm_score = (pos - neg) / total if total > 0 else 0.0
        return norm_score, pos, neg


def load_labeled_with_text() -> pd.DataFrame:
    records = []
    with open(LABELED_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("parse_error"):
                continue
            records.append(d)
    df = pd.DataFrame(records)

    sample = pd.read_csv(SAMPLE_PATH)[["comment_id", "text_clean", "published_at"]]
    sample_ids = set(sample["comment_id"])
    df = df[df["comment_id"].isin(sample_ids)]
    df = df[df["is_spam_or_offtopic"] != True]  # noqa: E712
    df = df.merge(sample, on="comment_id", how="left")
    print(f"分析対象: {len(df)}件（テキスト結合後）")
    return df


def run_scoring(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SimpleDictSentiment()
    scores, poss, negs = [], [], []
    n = len(df)
    for i, text in enumerate(df["text_clean"].tolist()):
        s, p, ng = analyzer.score(text)
        scores.append(s)
        poss.append(p)
        negs.append(ng)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n} 件処理済み")
    df = df.copy()
    df["dict_score"] = scores
    df["dict_pos"] = poss
    df["dict_neg"] = negs
    df["dict_has_hit"] = (df["dict_pos"] + df["dict_neg"]) > 0
    return df


def analyze(df: pd.DataFrame, out_lines: list[str]) -> None:
    out_lines.append(f"\n{'=' * 78}")
    out_lines.append(f"辞書ベース感情スコアの基本統計 (N={len(df)})")
    out_lines.append(f"{'=' * 78}")
    out_lines.append(f"辞書ヒットあり: {df['dict_has_hit'].sum()}件 "
                      f"({df['dict_has_hit'].mean() * 100:.1f}%)")
    out_lines.append(f"平均スコア: {df['dict_score'].mean():+.4f}  "
                      f"（-1=完全にネガティブ，+1=完全にポジティブ，0=ヒットなし含む）")

    out_lines.append(f"\n{'=' * 78}")
    out_lines.append("1. emotional_tone（LLMコーディング）別の辞書スコア分布")
    out_lines.append(f"{'=' * 78}")
    present_tones = [t for t in TONE_ORDER if t in df["emotional_tone"].unique()]
    groups = []
    for tone in present_tones:
        sub = df[df["emotional_tone"] == tone]["dict_score"]
        groups.append(sub)
        out_lines.append(f"  {tone:18s}: N={len(sub):5d}  平均={sub.mean():+.4f}  "
                          f"標準偏差={sub.std():.4f}")
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        out_lines.append(f"一元配置分散分析（tone間で辞書スコアが異なるか）: F={f_stat:.3f} p={p_val:.6f}"
                          f"{'  *有意*' if p_val < 0.05 else '  非有意'}")
        out_lines.append("解釈: 有意かつanger/contemptで平均が最も低ければ，辞書ベース測定は"
                          "LLMのemotional_tone判定と整合的（独立手法による裏付け）と言える。")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([g.dropna() for g in groups], tick_labels=present_tones, showmeans=True)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("LLMのemotional_tone別，辞書ベース感情スコアの分布")
    ax.set_ylabel("辞書スコア（-1〜+1）")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out_path = RESULTS_DIR / "dict_sentiment_by_tone.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    out_lines.append(f"書き出し: {out_path}")

    out_lines.append(f"\n{'=' * 78}")
    out_lines.append("2. LLMの3変数（people_vs_elite等）と辞書スコアの対応")
    out_lines.append(f"{'=' * 78}")
    for var in BOOL_VARS:
        sub = df.dropna(subset=[var])
        true_scores = sub[sub[var] == True]["dict_score"]  # noqa: E712
        false_scores = sub[sub[var] == False]["dict_score"]  # noqa: E712
        if len(true_scores) < 5 or len(false_scores) < 5:
            out_lines.append(f"  {var}: サンプルサイズ不足のためスキップ")
            continue
        t_stat, p_val = stats.ttest_ind(true_scores, false_scores, equal_var=False)
        out_lines.append(
            f"  {var:22s}: True群平均={true_scores.mean():+.4f} (N={len(true_scores)}), "
            f"False群平均={false_scores.mean():+.4f} (N={len(false_scores)}), "
            f"Welchのt検定 t={t_stat:.3f} p={p_val:.4f}"
            f"{'  *有意*' if p_val < 0.05 else '  非有意'}"
        )
    out_lines.append(
        "\n解釈の注意: economic_resentment/theft_of_enjoymentは人手検証でのカッパ係数が"
        "ほぼゼロ（信頼性が低い）と判明済みの変数であるため，ここで辞書スコアとの対応が"
        "弱い/非有意であっても，『理論的関係が存在しない』ことの証拠ではなく，"
        "『LLMコーディング自体の測定誤差が大きい』ことの追加的な傍証と解釈すべきである。"
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labeled_with_text()
    print("辞書ベース感情スコアを計算中...")
    df = run_scoring(df)

    out_lines: list[str] = [
        "辞書ベース感情分析（日本語評価極性辞書の簡略実装）による",
        "LLM構造化コーディング結果の独立検証（頑健性チェック，補助分析）",
    ]
    analyze(df, out_lines)

    summary_path = RESULTS_DIR / "dict_sentiment_summary.txt"
    summary_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n書き出し: {summary_path}")


if __name__ == "__main__":
    main()
