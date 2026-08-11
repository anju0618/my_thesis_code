# 分析2：YouTubeコメントのLLM構造化コーディング（第4部用）

卒論第4部「実証分析」における「分析2」（ミクロ・言説レベルの分析）のための、
YouTubeコメントデータ収集〜構造化コーディング〜分析のパイプライン。理論編
（第1〜3部）で導入した概念を、実際の日本語コメントデータに対して検証・拡張
することを目的とする（詳しい経緯はセッションのやり取り、および `notes/` 配下の
各文献レジュメを参照）。

もう一方の「分析1」（マクロ・アンケート調査レベルの分析：WVS/ISSP）は
`../survey_regression/` を参照。両者の位置づけは `../README.md` および
`sections/04_empirical.tex` を参照のこと。

## 研究デザインの要旨

- **対象トピック**（`config/topics.yaml`）: 財務省デモ・参政党・不法移民の3件を
  初期セットとする。いずれも「人民 vs エリート」「経済的怨恨」「享楽の盗取」
  といった理論編の概念が観察されやすいと想定される事例。
- **データソース**: YouTubeコメント（YouTube Data API v3, 無料・公式）。
  X（Twitter）はAPI制限のため、Yahoo!ニュースは公式APIが無いため、
  どちらも現実的な収集手段がなく採用しない（詳細は `notes/fujishiro2020.md`
  の「使うときの注意・限界」も参照——ベンチマーク論文のFujishiro et al.(2020)
  もTwitterのみを使っており、CrowdTangleは2024年に廃止済み）。
- **コーディングスキーム**（`config/coding_scheme.yaml`）: 理論編の各概念を
  コメント単位の変数に操作化したもの。各変数がどの文献のどのページに基づくかを
  明記してあるので、卒論第4部の方法論記述にそのまま転用できる。
  - `people_vs_elite` — Mudde & Rovira Kaltwasser (2017), pp.5–6
  - `economic_resentment` — Ricci (2020), pp.viii–ix, p.52
  - `theft_of_enjoyment` — Kawamura & Iwabuchi (2022)
  - `blame_target` — Fujishiro et al. (2020), p.315（「リベラルvs排外主義」軸の検証）
  - `emotional_tone` — Törnberg / Bennett & Livingston (2020) の「怒りの物語」

## セットアップ

[uv](https://docs.astral.sh/uv/) で仮想環境と依存関係を管理する。

```bash
cd analysis/llm_discourse
uv sync                 # pyproject.toml / uv.lock から .venv を作成・依存関係インストール
cp .env.example .env    # その後 .env を編集してAPIキー等を設定
```

以降のコマンドはすべて `uv run <コマンド>` の形で実行する（`.venv`をactivateしなくても
自動的にその仮想環境内で実行される）。新しいパッケージを追加する場合は
`uv add <パッケージ名>` を使う（`pip install`は使わない——`pyproject.toml`/`uv.lock`が
ズレる）。

### Windows ⇔ Ubuntu（42等）を行き来する場合

`uv run ...` の中身はOSに依存しないので、`Makefile`（後述）を含めてこのリポジトリの
中身自体はWindowsでもUbuntuでも一切変更不要。マシンを変えるたびに必要なのは
以下の2つだけ:

1. **`uv`本体のインストール**（両OS共通の1行インストールスクリプトがある。
   https://docs.astral.sh/uv/getting-started/installation/ 参照）。
   `.venv`自体はOS間で使い回せない（バイナリがOS依存）ので、新しいマシンでは
   `uv sync` をやり直せば良い（`uv.lock`はコミットしてあるので毎回同じ依存関係が
   再現される）。
2. **`make`本体のインストール**
   - Ubuntu（42など）: 大抵は最初から入っている。無ければ `sudo apt install make`。
   - Windows: 未インストールなら `winget install ezwinports.make` を実行
     （このセッションでは対話的なストア同意プロンプトが必要で自動化できなかったので、
     ターミナルで自分で実行すること）。Git Bashが入っていれば`make`が使う`sh`経由で
     `rm`/`find`等のUNIXコマンドがそのまま動く。

`.env`（APIキー）は`.gitignore`対象なので、マシンを変えたら`.env.example`から
作り直す必要がある。

## Makefile（`uv run ...`を毎回打つ代わりに使える）

下記「パイプラインの実行順序」と同じことを `make <target>` でも実行できる
（例: `make collect TOPIC=sanseito`, `make label-test LIMIT=50`）。
ターゲット一覧は `make help` を参照。`make`が無い場合は下のセクションの
生の`uv run`コマンドをそのまま使えばよい（両方とも常に同じ動作をする）。

### YouTube Data API v3 のキー取得
`.env.example` 内のコメントを参照。Google Cloud Consoleで無料のAPIキーを発行する。
1日あたり10,000ユニットのクォータがあり、`search.list`が100ユニット/回、
`commentThreads.list`が1ユニット/回。動画検索を多用するとすぐ枯渇するので、
最初は少数のクエリで試し、良い動画を見つけたら`topics.yaml`の`video_ids`に
直接追加していく運用を推奨する。

### Ollama（ローカルLLM）のセットアップ
1. https://ollama.com/download からインストール
2. 日本語が扱えるモデルを取得: `ollama pull qwen2.5:7b-instruct`
   （マシンのVRAM/メモリに応じて7b/14b/32bや量子化版を選ぶ）
3. `.env`の`OLLAMA_MODEL`を使うモデル名に合わせる

## パイプラインの実行順序

```bash
# 1. YouTubeからコメント収集（まずdry-runで動画一覧だけ確認するのを推奨）
uv run src/youtube_collect.py --topic sanseito --dry-run
uv run src/youtube_collect.py --topic sanseito

# 2. 前処理（全トピックまとめて1つのCSVに）
uv run src/preprocess.py

# 3. LLMによる構造化コーディング（まず少数件で試す）
uv run src/label_ollama.py --limit 50
uv run src/label_ollama.py   # 問題なければ全件

# 4. （任意）埋め込みベクトル生成——クラスタリング等はnotebooks/で
uv run src/embed.py

# notebooks/ を使う場合
uv run jupyter lab
```

## ディレクトリ構成

```
analysis/llm_discourse/
  pyproject.toml / uv.lock  # uvが管理する依存関係定義
  Makefile                    # uv run のショートカット（Windows/Ubuntu共通）
  config/
    topics.yaml          # 収集対象トピック・検索クエリ・動画ID
    coding_scheme.yaml    # LLMコーディングの変数定義（理論編との対応表）
  src/
    youtube_collect.py    # YouTube API でコメント収集
    preprocess.py          # クリーニング・統合
    label_ollama.py         # ローカルLLMで構造化コーディング
    embed.py                 # 文埋め込みベクトル生成
  data/
    raw/{topic}/{video_id}.json    # 生データ（1動画1ファイル）
    processed/comments.csv          # 前処理後の統合テーブル
    processed/labeled_comments.jsonl # LLMコーディング結果
    processed/embeddings.npy         # 埋め込みベクトル
  notebooks/               # 探索的分析・可視化（クラスタリング、集計、図表作成）
  results/                  # 卒論に転載する図表の最終版
```

`data/` 配下は `.gitignore` でリポジトリ管理対象外にしている
（YouTubeコメントは個人が投稿した公開データではあるが、著者名等を含む生データを
そのままリポジトリに残さない方が望ましいため）。

## 今後の作業（未着手）

- `topics.yaml`の`video_ids`を実際に動画を見て選定し、充実させる。
- `label_ollama.py`の出力を少量サンプルして人手で確認し（コーダー内信頼性の
  簡易チェック）、必要ならプロンプト・変数定義を調整する。
- `notebooks/`で、トピック別・変数別の集計、`blame_target`の分布（Fujishiro et al.
  の「保守vsリベラルでなく排外主義vsリベラル」という主張の検証）、埋め込み
  ベクトルによるクラスタリング等を行う。
- 卒論第4部（`sections/04_empirical.tex`）に、ここでの方法論・結果を反映させる。
