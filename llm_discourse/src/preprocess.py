"""
data/raw/{topic}/*.json を読み込み、コメントを1行1件のテーブルに整形して
data/processed/comments.csv に書き出す前処理スクリプト。

行うこと:
    - 全トピック・全動画のコメントを1つのテーブルに統合（topic, video_id列を付与）
    - 重複コメント（同一 comment_id）の除去
    - 明らかなスパム・短すぎるコメントの除去（is_spam_or_offtopic の粗いヒューリスティック候補）
    - URLの除去、絵文字・記号の連続の正規化（内容は残す。分析対象なので過度な除去はしない）

やらないこと（意図的に）:
    - 高度な言語検出やスパム分類は label_ollama.py 側（LLM判定）に任せる。
      ここでは「明らかに空・短すぎる・URLのみ」等の機械的な足切りだけ行う。

使い方:
    python src/preprocess.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ANALYSIS_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ANALYSIS_ROOT / "data" / "processed"

URL_RE = re.compile(r"https?://\S+")
MIN_TEXT_LENGTH = 5  # これ未満の文字数のコメントは分析対象から除外


def load_all_raw() -> pd.DataFrame:
    records = []
    if not RAW_DATA_DIR.exists():
        return pd.DataFrame()

    for topic_dir in sorted(RAW_DATA_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        topic = topic_dir.name
        for json_path in sorted(topic_dir.glob("*.json")):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            video = data["video"]
            for c in data["comments"]:
                records.append(
                    {
                        "topic": topic,
                        "video_id": video["video_id"],
                        "video_title": video["title"],
                        "channel_title": video["channel_title"],
                        "comment_id": c["comment_id"],
                        "parent_id": c["parent_id"],
                        "text": c["text"],
                        "author": c["author"],
                        "like_count": c["like_count"],
                        "published_at": c["published_at"],
                    }
                )
    return pd.DataFrame.from_records(records)


def clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.drop_duplicates(subset="comment_id").copy()
    df["text_clean"] = df["text"].map(clean_text)
    df = df[df["text_clean"].str.len() >= MIN_TEXT_LENGTH]
    df = df.reset_index(drop=True)
    return df


def main() -> None:
    df_raw = load_all_raw()
    print(f"読み込んだ生コメント数: {len(df_raw)}")

    if df_raw.empty:
        print(
            "data/raw/ にデータがありません。先に youtube_collect.py を実行してください。"
        )
        return

    df_clean = preprocess(df_raw)
    print(f"前処理後のコメント数: {len(df_clean)}")
    print(df_clean.groupby("topic").size())

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "comments.csv"
    df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"書き出し: {out_path}")


if __name__ == "__main__":
    main()
