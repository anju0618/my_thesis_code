# 実証分析パイプライン用 Makefile
#
# Windows / Ubuntu 両対応の考え方:
#   実行しているのは全部 `uv run ...` で、これはOS間で全く同じコマンド。
#   make自体がOSに依存する部分（rm, find等）を使っているだけなので、
#   両方のマシンに make と uv さえ入っていれば、このMakefile自体は
#   一切変更せずそのまま使い回せる。
#
#   Windows: 未インストールなら `winget install ezwinports.make` などで導入
#            （Git Bash が入っていれば sh 経由でrm/find等はそのまま動く）
#   Ubuntu (42等): 大抵は最初から make が入っている。無ければ `sudo apt install make`
#   uv:     https://docs.astral.sh/uv/getting-started/installation/
#            （どちらのOSも同じインストールスクリプトで導入できる）

TOPIC ?= sanseito
LIMIT ?= 50

.PHONY: help install collect collect-dry collect-ids collect-all preprocess label label-test embed notebook pipeline clean clean-cache

help:
	@echo "使い方: make <target> [TOPIC=sanseito] [LIMIT=50]"
	@echo ""
	@echo "  install       uvで依存関係をインストール（uv sync）"
	@echo "  collect-dry   TOPICの動画一覧だけ確認（コメント取得なし）"
	@echo "  collect       TOPICのコメントを収集"
	@echo "  collect-all   topics.yaml の全トピックを収集"
	@echo "  preprocess    生データをクリーニングして1つのCSVに統合"
	@echo "  label-test    LIMIT件だけLLMコーディングを試す（デフォルト50件）"
	@echo "  label         全件をLLMコーディング"
	@echo "  collect-ids   TOPICのvideo_idsのみ収集（検索なし）"
	@echo "  embed         埋め込みベクトルを生成"
	@echo "  pipeline      preprocess -> label -> embed を一気に実行"
	@echo "  notebook      Jupyter Lab を起動"
	@echo "  clean         data/raw, data/processed の中身を削除（.gitkeepは残す）"
	@echo "  clean-cache   __pycache__ 等を削除"

install:
	uv sync

collect-dry:
	uv run src/youtube_collect.py --topic $(TOPIC) --dry-run

collect:
	uv run src/youtube_collect.py --topic $(TOPIC)

collect-ids:
	uv run src/youtube_collect.py --topic $(TOPIC) --ids-only

collect-all:
	uv run src/youtube_collect.py

preprocess:
	uv run src/preprocess.py

label-test:
	uv run src/label_ollama.py --limit $(LIMIT)

label:
	uv run src/label_ollama.py

embed:
	uv run src/embed.py

pipeline: preprocess label embed

notebook:
	uv run jupyter lab

clean:
	find data/raw -mindepth 1 ! -name '.gitkeep' -delete
	find data/processed -mindepth 1 ! -name '.gitkeep' -delete

clean-cache:
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
