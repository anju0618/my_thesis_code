"""
config/coding_scheme.yaml で定義した理論的変数を、ローカルLLM（Ollama）を使って
コメント1件ごとに構造化コーディングする。

事前準備:
    1. Ollama をインストール (https://ollama.com/download)
    2. 日本語が扱えるモデルを取得しておく。例:
           ollama pull qwen2.5:7b-instruct
       （マシンスペックに応じて 7b / 14b / 32b 等を選ぶ。VRAMが厳しい場合は
        量子化版 (例: qwen2.5:7b-instruct-q4_K_M) も検討する）
    3. .env の OLLAMA_MODEL を使うモデル名に合わせる（未設定ならデフォルト値を使用）

使い方:
    python src/label_ollama.py --limit 50    # まず50件だけ試してプロンプトの精度を確認
    python src/label_ollama.py               # data/processed/comments.csv 全件をラベリング

出力:
    data/processed/labeled_comments.jsonl
        1行1コメントのJSON Lines。既存の comment_id はスキップするので、
        中断しても再実行すれば続きから処理される。

注意:
    ローカルLLMのJSON出力は完璧ではない。パースに失敗した場合は
    parse_error=True として記録し、処理は止めずに次へ進む。
    後で `df[df.parse_error].sample(20)` のように失敗例を確認し、
    プロンプトやモデルの調整、あるいは失敗分の手動コーディングを検討すること。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import ollama
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv
from tqdm import tqdm

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
SCHEME_PATH = ANALYSIS_ROOT / "config" / "coding_scheme.yaml"
INPUT_PATH = ANALYSIS_ROOT / "data" / "processed" / "comments.csv"
OUTPUT_PATH = ANALYSIS_ROOT / "data" / "processed" / "labeled_comments.jsonl"

DEFAULT_MODEL = "qwen2.5:7b-instruct"


def load_scheme() -> dict:
    with open(SCHEME_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system_prompt(scheme: dict) -> str:
    lines = [
        "あなたは日本の政治・社会に関するSNSコメントを研究目的で分析するアシスタントです。",
        "以下の変数定義に従って、与えられたコメント1件を厳密にJSON形式でコーディングしてください。",
        "判断に迷う場合は、より保守的な（＝該当なしに近い）値を選んでください。",
        "出力はJSONオブジェクトのみとし、説明文やコードブロック記法は含めないでください。",
        "",
        "# 変数定義",
    ]
    for var in scheme["variables"]:
        lines.append(f"\n## {var['name']} ({var['type']})")
        if "definition_ja" in var:
            lines.append(var["definition_ja"].strip())
        if "options" in var:
            lines.append("選択肢: " + " | ".join(var["options"]))
        if "positive_examples" in var:
            lines.append("該当する例: " + " / ".join(var["positive_examples"]))
        if "negative_examples" in var:
            lines.append("該当しない例: " + " / ".join(var["negative_examples"]))

    lines.append("\n# 出力JSONスキーマ")
    lines.append(scheme["output_schema_note"].strip())
    return "\n".join(lines)


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["comment_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def label_comment(model: str, system_prompt: str, text: str) -> dict:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"コメント: {text}"},
        ],
        format="json",
        options={"temperature": 0.0},
    )
    content = response["message"]["content"]
    try:
        parsed = json.loads(content)
        parsed["parse_error"] = False
    except json.JSONDecodeError:
        parsed = {"parse_error": True, "raw_output": content}
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="処理するコメント数の上限（動作確認用）")
    parser.add_argument("--model", default=None, help="使用するOllamaモデル名（省略時は.envまたはデフォルト）")
    args = parser.parse_args()

    load_dotenv(ANALYSIS_ROOT / ".env")
    model = args.model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

    if not INPUT_PATH.exists():
        sys.exit(f"{INPUT_PATH} がありません。先に preprocess.py を実行してください。")

    df = pd.read_csv(INPUT_PATH)
    if args.limit:
        df = df.head(args.limit)

    scheme = load_scheme()
    system_prompt = build_system_prompt(scheme)

    done_ids = load_existing_ids(OUTPUT_PATH)
    todo = df[~df["comment_id"].isin(done_ids)]
    print(f"対象: {len(df)} 件 / 既に処理済み: {len(done_ids)} 件 / 今回処理: {len(todo)} 件")
    print(f"使用モデル: {model}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for _, row in tqdm(todo.iterrows(), total=len(todo)):
            try:
                labels = label_comment(model, system_prompt, row["text_clean"])
            except Exception as e:  # noqa: BLE001 -- 1件の失敗で全体を止めない
                labels = {"parse_error": True, "raw_output": str(e)}

            record = {
                "comment_id": row["comment_id"],
                "topic": row["topic"],
                "video_id": row["video_id"],
                **labels,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
