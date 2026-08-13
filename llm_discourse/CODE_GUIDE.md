# 分析2 コード解説（`analysis/llm_discourse/`）

このドキュメントは，分析2（YouTubeコメントのLLM構造化コーディング）で
使っている `src/` 内のPythonコードを1関数ずつ詳しく解説する内部資料。
卒論本文には含めない，コードの中身を先生に説明するための資料。

---

## 0. 全体像：パイプラインの流れ

分析2は6本のスクリプトが，データを順番に加工していくパイプラインに
なっている。

```
① src/youtube_collect.py   YouTube上のコメントを収集
        ↓  data/raw/{topic}/{video_id}.json （動画1本＝1ファイル）
② src/preprocess.py        重複除去・クレンジング・短文除外
        ↓  data/processed/comments.csv （全38,214件，1行1コメント）
③ src/sample_comments.py   トピックごと層化ランダムサンプリング
        ↓  data/processed/comments_sample.csv （最大18,000件）
④ src/label_ollama.py      ローカルLLMによる6変数の構造化コーディング
        ↓  data/processed/labeled_comments.jsonl （1行1コメントのJSON）
⑤ src/embed.py             文埋め込みベクトルへの変換（④と並行して実行可）
        ↓  data/processed/*_embeddings.npy, *_embeddings_ids.csv
⑥ src/analyze_coding.py    ④・⑤の結果を集計・検定・可視化
        ↓  results/coding_summary_*.txt, *.png
```

①→②→③→④→⑥という順が基本の流れで，⑤（埋め込み計算）はいつでも
独立に実行できる（④のLLMコーディングとはまったく別の処理）。
それぞれ`analysis/llm_discourse/`直下で

```
uv run src/preprocess.py
```

のように実行する。`config/topics.yaml`（収集するトピック・検索クエリ・
動画IDの定義）と`config/coding_scheme.yaml`（LLMに判定させる6変数の
定義）という2つの設定ファイルが，①と④の挙動を決めている。

---

## 1. `src/youtube_collect.py` — YouTubeコメントの収集

### 目的
YouTube Data API v3を使い，`config/topics.yaml`で定義した3トピック
（財務省デモ・参政党・不法移民）について，動画を検索してコメントを
一括取得する。

### `search_video_ids(youtube, query, max_results, language_hint)`
```python
request = youtube.search().list(
    part="id", q=query, type="video",
    relevanceLanguage=language_hint, maxResults=..., pageToken=page_token,
)
```
YouTube Data APIの`search.list`エンドポイントで，検索クエリ（例：
「財務省 解体デモ」）にヒットする動画IDを取得する。`pageToken`を
使ってページネーション（1回のリクエストで最大50件までしか返らない
ため，`nextPageToken`をたどって複数回リクエストする）を実装している。

### `fetch_comments(youtube, video_id, max_comments, include_replies)`
```python
request = youtube.commentThreads().list(
    part="snippet,replies", videoId=video_id, ...,
    textFormat="plainText", order="relevance",
)
```
指定した動画のコメントスレッド（トップレベルコメント＋その返信）を
取得する。ここで重要なのが`order="relevance"`という指定——投稿された
時系列順（`order="time"`）ではなく，YouTube独自の「関連性」アルゴリズム
（実質的には高評価数・返信数などのエンゲージメント指標に基づく
ランキング）順にコメントが返ってくる。これにより，取得できるのは
各動画で**最もエンゲージメントの高い上位**コメントに事実上限定される。
これは卒論本文でも明記している標本抽出上の重要な限界（感情的・
対立的な言説へ偏る可能性）の直接の原因になっているコードである。

`include_replies=True`の場合，各トップレベルコメントの返信も
`parent_id`を付けて平坦なリストに追加する（返信がスレッド構造のまま
だと後の分析で扱いにくいため，1件＝1レコードのフラットな形にしている）。

### `collect_topic(...)`
```python
out_path = out_dir / f"{vid}.json"
if out_path.exists():
    print(f"  [SKIP] {vid} は取得済み")
    continue
```
動画ごとに既にJSONファイルが存在すればスキップする，冪等性
（同じコマンドを何度実行しても安全）を持たせた実装。API呼び出しの
クォータ制限（1日10,000ユニット）を無駄遣いしないための工夫。

---

## 2. `src/preprocess.py` — 前処理

### 目的
`data/raw/`以下に散らばった動画ごとのJSONファイルをすべて読み込み，
1つのテーブル（CSV）にまとめたうえで，重複除去とテキストクレンジング
を行う。

### `load_all_raw()`
```python
for topic_dir in sorted(RAW_DATA_DIR.iterdir()):
    for json_path in sorted(topic_dir.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for c in data["comments"]:
            records.append({"topic": topic, "video_id": ..., "comment_id": c["comment_id"], ...})
```
全トピック・全動画のJSONファイルを走査し，コメント1件ごとに辞書を
作って1つのリストにまとめる。`topic`列と`video_id`列をここで付与する
ことで，後段の分析で「どのトピックのコメントか」を追跡できるように
している。

### `clean_text(text)`
```python
text = URL_RE.sub("", text)              # URLを除去
text = re.sub(r"\s+", " ", text).strip()  # 連続空白・改行を1つの半角スペースに正規化
```
機械的なテキストクレンジング。絵文字や記号自体は残す——LLMコーディング
段階で感情トーン等の判定材料になりうるため，過度な除去はしない方針
（docstringに明記）。

### `preprocess(df)`
```python
df = df.drop_duplicates(subset="comment_id").copy()
df["text_clean"] = df["text"].map(clean_text)
df = df[df["text_clean"].str.len() >= MIN_TEXT_LENGTH]  # MIN_TEXT_LENGTH = 5
```
1. `comment_id`基準で重複除去（同じ動画を複数回収集した場合などに
   備える）。
2. クレンジング後の文字数が5文字未満（絵文字のみ・「www」のみ等）
   のコメントを除外。
3. スパムかどうかの本格的な判定はここでは**あえて行わない**——
   ④のLLMコーディング側の`is_spam_or_offtopic`変数に委ねる設計に
   なっている（機械的なルールだけでスパム判定すると誤判定が多い
   ため）。

---

## 3. `src/sample_comments.py` — 層化ランダムサンプリング

### 目的
前処理後の全コメント（数万件）は，ローカルLLMで全件コーディングする
には計算時間が非現実的（後述の通り136時間超）なため，トピックごとに
一定数を無作為抽出したサンプルを作る。

### `main()`
```python
for topic, group in df.groupby("topic"):
    n = min(args.per_topic, len(group))
    parts.append(group.sample(n=n, random_state=args.seed))
sample = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
```
- `df.groupby("topic")`でトピックごとにグループ化し，各グループから
  `group.sample(n=n, random_state=args.seed)`で指定件数（デフォルト
  2,000件，実際には`--per-topic 6000`で実行）を無作為抽出する。
  これが**層化抽出（stratified sampling）**——トピックという層ごとに
  独立にサンプリングすることで，どのトピックも同じ割合でサンプルに
  含まれることを保証する（単純な無作為抽出だと，件数の多いトピックが
  サンプルを支配してしまう可能性がある）。
- `random_state=args.seed`（デフォルト42）で乱数シードを固定して
  いるため，同じコマンドを何度実行しても毎回まったく同じサンプルが
  再現される——これが卒論本文で「乱数シード42で固定，再現可能」と
  書いている根拠。
- 最後の`.sample(frac=1, ...)`は「全体の100%を無作為に並べ替える」
  という意味で，トピックごとに固まっていたサンプルの行順をシャッフル
  している（後段のLLMコーディングを`comment_id`順ではなく無作為の
  順序で進めるため，途中で打ち切っても特定トピックに偏らない）。

---

## 4. `src/label_ollama.py` — ローカルLLMによる構造化コーディング

### 目的
サンプリングされたコメント1件ごとに，ローカルLLM（Ollama上の
qwen2.5:7b-instruct）を使って`config/coding_scheme.yaml`で定義した
6変数を判定させる，分析2の中核となるスクリプト。

### `build_system_prompt(scheme)`
```python
for var in scheme["variables"]:
    lines.append(f"\n## {var['name']} ({var['type']})")
    if "definition_ja" in var:
        lines.append(var["definition_ja"].strip())
    if "options" in var:
        lines.append("選択肢: " + " | ".join(var["options"]))
    if "positive_examples" in var:
        lines.append("該当する例: " + ...)
```
`config/coding_scheme.yaml`に書かれた6変数の定義・選択肢・正例／負例
を読み込み，LLMに渡す「システムプロンプト」（LLMへの指示文）を
自動組み立てする関数。コーディングスキーム（何を判定させるか）と
プロンプトの文面が同じYAMLファイル1つで一元管理されているため，
変数の定義を変えたい場合はYAMLを直すだけでよい設計になっている。

### `label_comment(model, system_prompt, text)`
```python
response = ollama.chat(
    model=model,
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": f"コメント: {text}"}],
    format="json",
    options={"temperature": 0.0},
)
```
- `ollama.chat()`は，ローカルで動いているOllamaサーバー（別プロセス，
  `localhost:11434`）にHTTPリクエストを送り，指定したモデルに
  チャット形式で応答させるPythonクライアント関数。
- `format="json"`は，Ollama側にJSON形式での出力を強制させるオプション
  （自由な文章ではなく構造化されたJSONだけを返すよう制約する）。
- `options={"temperature": 0.0}`——**temperature**はLLMが次の単語を
  選ぶ際のランダムさを制御するパラメータで，0にすると常に最も確率の
  高い選択をする決定的な挙動になる。同じコメントに対して常に同じ
  判定結果が返ることを保証する（再現性のため）ための設定。
- 返ってきた文字列を`json.loads()`でパースし，失敗した場合は
  `parse_error=True`として記録して処理を止めずに次へ進む（1件の
  パース失敗で全体が止まらないようにする例外処理）。

### `load_existing_ids(path)` と再開可能性
```python
ids = set()
with open(path, encoding="utf-8") as f:
    for line in f:
        ids.add(json.loads(line)["comment_id"])
```
出力ファイル（`labeled_comments.jsonl`）を1行ずつ読み，既に処理済みの
`comment_id`の集合を作る。`main()`側で
```python
todo = df[~df["comment_id"].isin(done_ids)]
```
と，まだ処理していない行だけに絞り込んでいる。これにより，処理を
途中で止めても（PCのシャットダウン等），再実行すれば処理済み分を
スキップして続きから再開できる——18,000件全体を1回で処理しきる
必要がなく，何日かに分けて少しずつ処理を進められる設計になっている。

### 出力形式（JSON Lines）
```python
with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
    for _, row in tqdm(todo.iterrows(), total=len(todo)):
        ...
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
```
- `"a"`（追記モード）でファイルを開き，1件処理するごとに1行ずつ
  JSON文字列を追記する（**JSON Lines**形式：1行＝1つの独立したJSON
  オブジェクト）。
- `out_f.flush()`を毎回呼ぶことで，途中でプロセスが強制終了しても
  それまでに処理した分はディスクに確実に書き込まれているようにして
  いる（バッファに溜めたまま失われることを防ぐ）。

---

## 5. `src/embed.py` — 文埋め込みベクトルへの変換

### 目的
各コメントを，意味的な近さを数値ベクトルで表現する「文埋め込み
（sentence embedding）」に変換する。LLMコーディング（④）とは独立に
実行できる，探索的クラスタリング用の前処理。

### コード解説
```python
model = SentenceTransformer(MODEL_NAME, device=args.device)
texts = [f"passage: {t}" for t in df["text_clean"].fillna("").tolist()]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
```
- `MODEL_NAME = "intfloat/multilingual-e5-large"`——多言語対応・
  日本語での実績が比較的多い文埋め込みモデル。文章を入力すると，
  意味的な内容を圧縮した固定長の数値ベクトル（1,024次元）を返す。
  意味が近い文章ほど，ベクトル空間上で近い位置に来るように学習
  されている。
- `f"passage: {t}"`——multilingual-e5系のモデルは，検索用途を想定して
  「検索クエリ側には`query: `，検索対象の文書側には`passage: `という
  接頭辞を付けると精度が上がる」という仕様になっている。ここでは
  各コメントを「検索される側の文書」として扱っているため`passage:`
  を使っている。
- `normalize_embeddings=True`——各ベクトルの長さ（ノルム）を1に
  揃える正規化。これにより，後でコサイン類似度やユークリッド距離を
  使ったクラスタリング（K-means等）が計算しやすくなる。
- `--device cpu`オプション（`main.py`側の`argparse`で定義）は，
  `label_ollama.py`がGPUを使用中の場合に埋め込み計算をCPU側に
  逃がすためのもの（このPCのGPUはVRAM 6GBと限られているため，
  同時に2つのGPU処理を走らせるとメモリ不足になる）。

出力は`embeddings.npy`（NumPy形式のベクトル行列，1行が1コメントの
ベクトル）と，どの行がどの`comment_id`に対応するかを記録する
`embeddings_ids.csv`の2ファイル。

---

## 6. `src/analyze_coding.py` — 集計・検定・可視化

### 目的
④のLLMコーディング結果と⑤の埋め込みを読み込み，卒論本文で報告する
3種類の分析（出現率，カイ二乗検定，探索的クラスタリング）を行う。
`labeled_comments.jsonl`が完成していない途中の時点でも実行でき，
実行するたびにその時点の件数を付けたファイル名で結果を保存する
（`coding_summary_7144.txt`のように，末尾の数字がその時点での
ラベル済み件数）。

### `load_labeled()`
```python
for line in f:
    d = json.loads(line)
    if d.get("parse_error"):
        n_parse_error += 1
        continue
    records.append(d)
```
JSON Linesファイルを1行ずつ読み，`parse_error=True`（LLMの出力が
JSONとしてパースできなかった行）を除外してDataFrameを作る。続けて
```python
df = df[df["is_spam_or_offtopic"] != True]
```
でスパム／話題逸脱と判定されたコメントも分析対象から除外する
（コーディングスキームの定義通り）。

さらに，LLM出力に稀に混入する表記ゆれ（`"foreigners"`という
本来存在しないカテゴリ名）やパイプ区切りの複合値
（`"immigrants|elites_general"`）を正規のカテゴリ名に補正する
`_normalize_bt()`という内部関数も定義している——これはLLM出力が
必ずしも指定した選択肢どおりに完璧には返ってこないという，
LLMコーディング特有の後処理の必要性を示す部分でもある。

### `wilson_score_interval(k, n, z=1.96)` — 信頼区間の計算
```python
p_hat = k / n
denom = 1 + z**2 / n
center = (p_hat + z**2 / (2 * n)) / denom
margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
return p_hat, center - margin, center + margin
```
ある変数（例：`people_vs_elite`）が「該当する」と判定された件数
$k$，全体の件数$n$から，出現率$\hat{p}=k/n$の**95%信頼区間**を計算する。
単純な正規近似（$\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$）は，
$n$が小さい・$\hat p$が0や1に近い場合に区間が不自然になる（負の
確率になる等）弱点があるため，**Wilsonスコア区間**という，中心を
$\hat p$からわずかにずらして補正するより正確な方法を使っている
（`economic_resentment`のように出現率が数%と低い変数を扱う本稿では
特に重要な補正）。数式の理論的な導出は`sections/90_appendix_math.tex`
にまとめてある。

### `occurrence_rates(df, out_lines)`
トピックごとに`BOOL_VARS = ["people_vs_elite", "economic_resentment",
"theft_of_enjoyment"]`の3変数について，`valid = sub[var].dropna()`
で欠損（LLMが判断を保留した`null`）を除いた有効件数の中で該当率を
計算し，Wilsonスコア区間とともに出力する。

### `chi_square_blame_target(df, out_lines)` — カイ二乗検定
```python
ct = pd.crosstab(sub["topic"], sub["blame_target"])
chi2, p, dof, expected = stats.chi2_contingency(ct)
cramers_v = np.sqrt((chi2 / n) / (min(ct.shape) - 1))
```
- `pd.crosstab()`で「トピック×非難対象」の分割表（クロス集計表）を
  作る。
- `stats.chi2_contingency()`（scipyの関数）で**カイ二乗検定**を行う。
  これは「トピックと非難対象がもし無関係だったら期待される件数」と
  「実際の観測件数」のズレを統計的に検定する手法で，$p$値が小さいほど
  「トピックによって非難対象の分布は統計的に有意に異なる」と言える。
- **Cramér's V**は，カイ二乗検定の結果を「関連の強さ」として0〜1に
  正規化した効果量の指標（$p$値は「関連があるかないか」しか教えて
  くれないが，Cramér's Vは「その関連がどれくらい強いか」を表す）。
- 続く**標準化残差**（standardized residuals）の計算は，どのセル
  （例：「財務省デモ×elites_general」）が特に期待値より多い／少ない
  かを個別に見るための指標で，値が正に大きいほど「そのトピックで
  その非難対象が期待以上に多く出現している」ことを意味する。これを
  ヒートマップ（`coding_blame_target_heatmap_*.png`）として可視化
  している。

### `embedding_cluster(df, out_lines)`
```python
merged = df.merge(ids[["comment_id", "emb_idx"]], on="comment_id", how="inner")
X = embeddings[merged["emb_idx"].values]
for k in k_range:
    km = KMeans(n_clusters=k, ...)
    sil = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=0)
```
- LLMコーディング済みのコメント（`df`）と，⑤で計算した埋め込み
  ベクトル（`embeddings`）を`comment_id`で紐付ける。これにより，
  「同じコメントに対して，理論駆動の6変数コーディングと，データ
  駆動の埋め込みクラスタリングの両方の結果を突き合わせる」ことが
  できる設計になっている。
- `issp_ml_analysis.py`と同じ考え方でシルエット係数最大の$K$を選び，
  各クラスタの`people_vs_elite`等の該当率を比較する。
- ```python
  centroid = Xc.mean(axis=0)
  dists = np.linalg.norm(Xc - centroid, axis=1)
  top3 = idx_in_cluster[np.argsort(dists)[:3]]
  ```
  各クラスタの「重心に最も近い3件」を探し，そのコメントの
  `rationale_ja`（LLMが判定理由として書いた1文）とともに出力する。
  これにより，各クラスタが実際にどんな内容のコメント群なのかを，
  数値だけでなく具体的な文章で確認できるようにしている。
- 最後に**PCA（主成分分析）**で1,024次元のベクトルを2次元に圧縮し，
  散布図（`coding_cluster_*.png`）として可視化する。PCAは，
  高次元データの中で最も分散が大きい方向（＝データの違いを最もよく
  説明する方向）を2つ選んで平面に投影する次元削減手法。

---

## 7. 使われている統計・機械学習手法のまとめ

| 手法 | 使用箇所 | 何のために使うか |
|---|---|---|
| ローカルLLMによる構造化コーディング | `label_ollama.py` | 自由記述コメントを理論的に定義した変数へ操作化する（内容分析のLLM版） |
| Wilsonスコア区間 | `analyze_coding.py` | 出現率の推定に対する95%信頼区間を，低出現率でも正確に計算する |
| カイ二乗検定 | `analyze_coding.py` | トピックと非難対象の分布に統計的な関連があるかを検定する |
| Cramér's V | `analyze_coding.py` | カイ二乗検定で見つかった関連の「強さ」を0〜1で定量化する |
| 標準化残差 | `analyze_coding.py` | クロス集計表のどのセルが期待値から特に乖離しているかを特定する |
| 文埋め込み（multilingual-e5-large） | `embed.py` | テキストの意味的な近さを数値ベクトルとして表現する |
| K-means クラスタリング | `analyze_coding.py` | 埋め込みベクトルをもとにコメントを教師なしでグループ分けする |
| シルエット係数 | `analyze_coding.py` | クラスタ数$K$の妥当性を定量的に評価する |
| PCA（主成分分析） | `analyze_coding.py` | 高次元の埋め込みベクトルを2次元に圧縮して可視化する |

Cohenの$\kappa$係数（コーダー内信頼性チェック）は本ディレクトリの
スクリプトとしては自動化されておらず，著者が手動で57件を判定し
Pythonの対話環境（あるいはJupyter）で計算したものである。定式化は
`sections/90_appendix_math.tex`の「コーダー内信頼性」節を参照。
