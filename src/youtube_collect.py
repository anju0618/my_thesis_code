"""
YouTube Data API v3 を使って、config/topics.yaml で定義したトピックごとに
動画を検索し、各動画のコメント（トップレベル＋任意で返信）を収集する。

使い方:
    python src/youtube_collect.py                  # topics.yaml の全トピックを収集
    python src/youtube_collect.py --topic sanseito  # 特定トピックのみ
    python src/youtube_collect.py --topic sanseito --dry-run  # 動画一覧だけ確認（コメント取得なし）

出力:
    data/raw/{topic}/{video_id}.json
    1動画1ファイル。動画メタデータ + コメント一覧（トップレベル・返信を1つのフラットな
    リストにまとめ、返信には parent_id を付与する）を保存する。

必要な環境変数（.env）:
    YOUTUBE_API_KEY=...

注意（クォータ）:
    YouTube Data API v3 の無料クォータは1日10,000ユニット。
    search.list は1回100ユニット、commentThreads.list / comments.list は1回1ユニット。
    動画数を絞った検索（search）を多用するとすぐにクォータを消費するので、
    最初はある程度 video_ids を手動で選んでから収集する運用を推奨する。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Windowsのコンソール既定コードページ（cp932等）では動画タイトルに含まれる絵文字や
# 一部記号（例: ‼）が表現できずUnicodeEncodeErrorでクラッシュすることがあるため、
# 標準出力/エラー出力をUTF-8に強制する（表示が乱れることはあってもクラッシュはしない）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ANALYSIS_ROOT / "config" / "topics.yaml"
RAW_DATA_DIR = ANALYSIS_ROOT / "data" / "raw"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_client():
    load_dotenv(ANALYSIS_ROOT / ".env")
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        sys.exit(
            "YOUTUBE_API_KEY が設定されていません。"
            f"{ANALYSIS_ROOT / '.env.example'} を参考に {ANALYSIS_ROOT / '.env'} を作成してください。"
        )
    return build("youtube", "v3", developerKey=api_key)


def search_video_ids(youtube, query: str, max_results: int, language_hint: str) -> list[str]:
    video_ids: list[str] = []
    page_token = None
    while len(video_ids) < max_results:
        request = youtube.search().list(
            part="id",
            q=query,
            type="video",
            relevanceLanguage=language_hint,
            maxResults=min(50, max_results - len(video_ids)),
            pageToken=page_token,
        )
        try:
            response = request.execute()
        except HttpError as e:
            print(f"  [WARN] 検索失敗 (query={query!r}): {e}", file=sys.stderr)
            break
        video_ids.extend(item["id"]["videoId"] for item in response.get("items", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return video_ids[:max_results]


def fetch_video_metadata(youtube, video_id: str) -> dict | None:
    try:
        response = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    except HttpError as e:
        print(f"  [WARN] 動画メタデータ取得失敗 (video_id={video_id}): {e}", file=sys.stderr)
        return None
    items = response.get("items", [])
    if not items:
        return None
    item = items[0]
    snippet = item["snippet"]
    stats = item.get("statistics", {})
    return {
        "video_id": video_id,
        "title": snippet.get("title"),
        "channel_title": snippet.get("channelTitle"),
        "published_at": snippet.get("publishedAt"),
        "view_count": stats.get("viewCount"),
        "comment_count": stats.get("commentCount"),
    }


def fetch_comments(youtube, video_id: str, max_comments: int, include_replies: bool) -> list[dict]:
    comments: list[dict] = []
    page_token = None

    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=page_token,
                textFormat="plainText",
                order="relevance",
            )
            response = request.execute()
        except HttpError as e:
            reason = getattr(e, "reason", str(e))
            print(f"  [WARN] コメント取得失敗 (video_id={video_id}): {reason}", file=sys.stderr)
            break

        for thread in response.get("items", []):
            top_snippet = thread["snippet"]["topLevelComment"]["snippet"]
            top_id = thread["snippet"]["topLevelComment"]["id"]
            comments.append(_snippet_to_record(top_id, top_snippet, parent_id=None))

            if include_replies and thread["snippet"].get("totalReplyCount", 0) > 0:
                for reply in thread.get("replies", {}).get("comments", []):
                    comments.append(
                        _snippet_to_record(reply["id"], reply["snippet"], parent_id=top_id)
                    )

        page_token = response.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.1)  # 念のための軽いレート制御

    return comments[:max_comments]


def _snippet_to_record(comment_id: str, snippet: dict, parent_id: str | None) -> dict:
    return {
        "comment_id": comment_id,
        "parent_id": parent_id,
        "text": snippet.get("textOriginal", ""),
        "author": snippet.get("authorDisplayName"),
        "like_count": snippet.get("likeCount"),
        "published_at": snippet.get("publishedAt"),
        "updated_at": snippet.get("updatedAt"),
    }


def collect_topic(
    youtube, topic_key: str, topic_conf: dict, defaults: dict, dry_run: bool, ids_only: bool
) -> None:
    print(f"\n=== トピック: {topic_key} ({topic_conf.get('label_ja')}) ===")
    out_dir = RAW_DATA_DIR / topic_key
    out_dir.mkdir(parents=True, exist_ok=True)

    max_videos = topic_conf.get("max_videos_per_query", defaults["max_videos_per_query"])
    max_comments = topic_conf.get("max_comments_per_video", defaults["max_comments_per_video"])
    include_replies = topic_conf.get("include_replies", defaults["include_replies"])
    language_hint = topic_conf.get("language_hint", defaults["language_hint"])

    video_ids: set[str] = set(topic_conf.get("video_ids") or [])

    if ids_only:
        print(f"  --ids-only 指定: video_ids の {len(video_ids)} 件のみ処理します（検索はスキップ）")
    else:
        for query in topic_conf.get("search_queries") or []:
            print(f"  検索: {query!r}")
            found = search_video_ids(youtube, query, max_videos, language_hint)
            print(f"    -> {len(found)} 件の動画")
            video_ids.update(found)

    print(f"  合計 {len(video_ids)} 件の動画を処理します")

    if dry_run:
        for vid in sorted(video_ids):
            print(f"    [dry-run] {vid}  https://www.youtube.com/watch?v={vid}")
        return

    for vid in sorted(video_ids):
        out_path = out_dir / f"{vid}.json"
        if out_path.exists():
            print(f"  [SKIP] {vid} は取得済み")
            continue

        meta = fetch_video_metadata(youtube, vid)
        if meta is None:
            continue

        comments = fetch_comments(youtube, vid, max_comments, include_replies)
        print(f"  [OK] {vid}: {meta['title']!r} -- {len(comments)} 件のコメント")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"video": meta, "comments": comments}, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", help="収集するトピックキー（省略時は全トピック）")
    parser.add_argument(
        "--dry-run", action="store_true", help="動画一覧の確認のみ行い、コメントは取得しない"
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="topics.yaml の video_ids のみ処理し、search_queries による検索はスキップする",
    )
    args = parser.parse_args()

    config = load_config()
    defaults = config["defaults"]
    topics = config["topics"]

    if args.topic:
        if args.topic not in topics:
            sys.exit(f"未定義のトピック: {args.topic}. 定義済み: {list(topics)}")
        topics = {args.topic: topics[args.topic]}

    youtube = get_client()
    for topic_key, topic_conf in topics.items():
        collect_topic(youtube, topic_key, topic_conf, defaults, args.dry_run, args.ids_only)


if __name__ == "__main__":
    main()
