# 分析1 コード解説（`analysis/survey_regression/`）

このドキュメントは，分析1（ISSP・WVSアンケート調査データの実証分析）で
使っている `scripts/` 内のPythonコードを1関数ずつ詳しく解説する内部資料。
「なぜこの手法を使うのか」「このコードは具体的に何をしているのか」を
先生に説明できるようにするための解説であり，卒論本文には含めない。

---

## 0. 全体像：4本のスクリプトの関係

```
scripts/issp_analysis.py          … ISSPデータ：OLS回帰のみ（一番シンプル）
scripts/issp_ml_analysis.py       … ISSPデータ：OLS対Random Forest比較 + SHAP + クラスタリング
scripts/wvs_timeseries_analysis.py … WVS日本データ：長期トレンド + メディア利用の効果（OLS対RF + SHAP）
scripts/wvs_cross_country_analysis.py … WVSデータ：5か国比較版（wvs_timeseries_analysis.pyの拡張）
```

依存関係としては，`issp_analysis.py` → `issp_ml_analysis.py` の順で「まず単純な
OLSで何が見えるか確認し，その後Random Forest・SHAP・クラスタリングという
より柔軟な手法で同じデータを掘り下げる」という流れになっている。WVS側も
`wvs_timeseries_analysis.py`（日本単体）→`wvs_cross_country_analysis.py`
（5か国に拡張）という順で発展している。4本とも独立に実行可能で，
`results/` に出力する（既存ファイルは上書き）。

実行方法はすべて共通で，`analysis/survey_regression/` 直下で

```
uv run scripts/issp_analysis.py
```

のように実行する（`uv run` はこのディレクトリの `pyproject.toml` に
書かれた依存パッケージが入った専用のPython環境で実行してくれるコマンド）。

`data/` 以下にはISSP・WVSの生データ（`.sav`形式，SPSS用バイナリファイル）
が置いてある。これを読むのに使っているのが `pyreadstat` というライブラリ。

`scripts/*.R` の3本（`issp_role_of_government.R`, `wvs_wave7.R`,
`wvs_time_series_template.R`）は，Python版を書く前に最初に書いたR言語の
試作スクリプトで，ローカルの個人フォルダ（`C:/under my control/...`）に
パスがハードコードされており，現在は使っていない。`issp_analysis.py`の
冒頭コメントに「旧版（Role of Government.r）のPythonへの再実装」と
明記してある通り，最終的な結果はすべてPython版から出ている。卒論の
結果の根拠としてはPython版だけを見ればよい。

---

## 1. `scripts/issp_analysis.py` — ISSPデータのOLS回帰

### 目的
ISSP「Role of Government」調査（1996・2006・2016年，日本）を使い，
「調査年」または「インターネット普及率」が，政治的有効性感覚の欠如・
公務員信頼・政治家の汚職認識・議員信頼という4つの政治意識指標に
与える影響を，最も基本的な重回帰分析（OLS: 最小二乗法）で確認する。

### `load_japan_data()`
```python
df, meta = pyreadstat.read_sav(DATA_PATH)
df_jp = df[df["country"] == JAPAN_COUNTRY_CODE][...]
```
- `pyreadstat.read_sav()` でSPSS形式（`.sav`）のISSP生データを読み込む。
  ISSPは各変数がアルファベット+数字の変数名（`v61`, `v66`など）で
  管理されているコードブック形式のデータなので，まず日本
  （`country == 392`，ISO 3166数値国コード）のみを抽出し，
  変数名を人間に読める名前（`no_say`, `trust_civil`など）に
  `rename()` で付け替えている。
- `INTERNET_RATE = {1996: 9.2, 2006: 72.6, 2016: 83.5}` は総務省統計に
  基づく，日本のインターネット普及率（%）。ISSPは個人のインターネット
  利用頻度そのものは聞いていないため，「調査が行われた年の全国普及率」
  を，個人ごとに同じ値を割り当てるマクロな代理変数として使っている
  （これは「調査年そのものの代理に近い」という限界が卒論本文にも
  明記されている）。
- `.astype("category")` で性別・学歴・調査年をカテゴリ型に変換している。
  これは後で `C(sex)` のように書くと，OLSの数式（formula）の中で
  自動的にダミー変数化してくれるようにするため。

### `fit_and_report(formula, data, label, out_lines)`
```python
model = smf.ols(formula, data=data, missing="drop").fit()
```
- `statsmodels` の `smf.ols()` は，R言語のような「数式文字列」
  （例：`"no_say ~ C(year_factor) + age + C(sex) + C(education) + pol_interest"`）
  でOLS回帰を指定できる関数。`~`の左が目的変数（Y），右が説明変数（X）。
  `C(...)` はその変数をカテゴリ変数として扱う（ダミー変数化する）という
  statsmodels特有の記法。
- `missing="drop"` は，説明変数・目的変数のどれかが欠損（NaN）している
  行を自動的に除外するオプション。
- `model.summary()` が，係数・標準誤差・t値・p値・R²などを含む
  統計サマリを整形されたテキストで返してくれる。これをそのまま
  `results/issp_summary.txt` に書き出している。

### `main()`
4つの目的変数（`no_say`, `trust_civil`, `corruption`, `trust_mps`）
それぞれについて，「調査年ダミー版」と「インターネット普及率版」の
2種類の式で計8回 `fit_and_report()` を呼んでいる。`corruption`
（v73，汚職認識）だけは1996年に調査されていない設問なので，
`df[df["year"] != 1996]` で1996年を除外している。

最後に，4変数の年ごとの平均値を `groupby("year").mean()` で集計し，
折れ線グラフ（`issp_trend.png`）として保存している。

---

## 2. `scripts/issp_ml_analysis.py` — OLS対Random Forest・SHAP・クラスタリング

### 目的
`issp_analysis.py` の単純なOLSでは「線形の関係」しか捉えられない。
非線形性・変数同士の交互作用（例えば「政治的関心が高い人だけ，
インターネット普及率の効果が違う」といったパターン）を拾えるかを，
Random Forestという機械学習モデルとの比較で検証する。加えて，
Ricciの「経済的破壊の当事者性」という理論的概念を，人口統計学的
変数だけでなく雇用状態（`WORKYN`）という変数でも捉えられるように
拡張している。

### `load_japan_data()`
`issp_analysis.py` とほぼ同じだが，`WORKYN`（1=就業，2=完全失業，
3=非労働力人口）を読み込み，
```python
df_jp["is_unemployed"] = (df_jp["WORKYN"] == 2).astype(float)
df_jp["is_not_in_labor_force"] = (df_jp["WORKYN"] == 3).astype(float)
```
で「就業者を基準（0,0）とする2つのダミー変数」に変換している点が
異なる。欠損（`WORKYN`がNaN）の行は両ダミーとも `pd.NA` にしている
（0にしてしまうと「就業者」と区別がつかなくなるため）。

### `compare_ols_vs_rf(df, target, out_lines)` — OLSとRFの予測性能比較
```python
cv = KFold(n_splits=5, shuffle=True, random_state=0)
ols_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
rf_scores = cross_val_score(RandomForestRegressor(...), X, y, cv=cv, scoring="r2")
```
- **5分割交差検証（5-fold cross validation）**：データを5つに分割し，
  4つで学習・残り1つで予測精度（R²）を測るのを5回繰り返して平均を取る
  手法。「学習に使ったデータでしか良い成績を出せていないだけ」という
  過学習の問題を避け，未知データに対する予測力を公平に評価できる。
- `LinearRegression()` がOLS（線形回帰）モデル，
  `RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=20)`
  がRandom Forestモデル。`n_estimators=500`は「500本の決定木を作って
  多数決（平均）を取る」という意味，`max_depth=5`は各木の深さを5階層
  までに制限（深すぎると過学習しやすいため），`min_samples_leaf=20`は
  「末端の葉に最低20サンプルなければ分岐しない」という制約（同様に
  過学習防止）。
- 出力される `R²`（決定係数）は「モデルが目的変数の分散のうち何%を
  説明できたか」を表す指標で，1に近いほど当てはまりが良い。

### `shap_analysis(df, target)` — SHAP値による寄与度の可視化
```python
model = RandomForestRegressor(...).fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X_jp, show=False)
```
- Random Forestを実データ全体で学習し直し，`shap.TreeExplainer`で
  各説明変数が「個々の予測値をどれだけ押し上げた/押し下げたか」を
  ゲーム理論のShapley値に基づいて分解する。`summary_plot`は，
  横軸がSHAP値（寄与の大きさ），色が元の変数の値（赤=高い，青=低い）
  の散布図で，「年齢が高いほど予測値が上がる／下がる」といった
  傾向を一目で見られるようにしたグラフ。単なる回帰係数と違い，
  非線形な関係（「中くらいの年齢だけ効果が強い」等）も表現できる。
- `dependence_plot`は，特定の1変数（ここでは「調査年」）に絞って，
  その値とSHAP値の関係を，別の変数（「政治への関心」）で色分けして
  見る補助的なプロット。

### `choose_k(X_scaled, out_lines, k_range=range(2,7))` — クラスタ数Kの選定
```python
for k in k_range:
    km = KMeans(n_clusters=k, ...)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
```
- **K-means**は，データを$K$個のグループ（クラスタ）に自動分類する
  教師なし学習手法。各点を最も近い「重心」に割り当て，重心を
  更新することを繰り返して収束させる。
- `inertia_`（慣性）は「各点と自分のクラスタの重心との距離の
  二乗和」で，$K$を増やすほど必ず小さくなる（クラスタが細かくなる
  ため）。そのため単独では最適な$K$を決められず，「急激に下がる
  勾配が緩やかになる地点（エルボー＝肘）」を目視で探す補助情報として
  使う。
- **シルエット係数（silhouette score）**は，各点について「自分の
  クラスタ内の点との近さ」対「一番近い別クラスタの点との近さ」を
  比較し，-1〜1の範囲でクラスタの分離の良さを定量化する指標。
  1に近いほど「クラスタ同士がはっきり分かれている」ことを意味する。
  このスクリプトでは，シルエット係数が最大になる$K$を自動的に選び，
  結果を `cluster_summary.txt` に記録する。

### `cluster_analysis(df, out_lines)`
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[cluster_vars])
```
- クラスタリングの前に**標準化（StandardScaler）**を行う。これは
  各変数を「平均0，標準偏差1」に変換する処理で，これをしないと
  スケールの大きい変数（例えば1〜10の変数）がスケールの小さい変数
  （例えば0〜1の変数）より距離計算で不当に重視されてしまう。
- `choose_k()`でシルエット係数最適値を確認した上で，コード内では
  **あえて$K=3$を採用**している（コメントに理由が明記されている：
  卒論本文の年別構成比の解釈のしやすさを優先し，シルエット係数の
  最適値との異同は出力ファイルに記録して透明性を保つ，という方針）。
  実際に`cluster_summary.txt`を見ると$K=3$のシルエット係数
  （0.255）が$K=2$〜$6$の中で最大でもあるため，両者は一致している。
- 最後に，`groupby(["year","cluster"]).size()`で「各年に各クラスタが
  何%を占めるか」を集計し，積み上げ棒グラフ（`cluster_profile.png`）
  として可視化している。

---

## 3. `scripts/wvs_timeseries_analysis.py` — WVS日本データ（長期トレンド＋メディア利用効果）

### 目的
WVS Time Series（1981〜2022年，日本は1981〜2019年）を使い，(A)
制度への信頼の超長期トレンドを可視化し，(B) メディア利用頻度
（インターネット・テレビ・新聞・SNS）が信頼に与える効果を，
ISSP版と同じOLS対RF・SHAPの枠組みで検証する。

### `load_japan_data()`
- WVSは変数名がすべて `E069_07`のような記号のため，`RAW_COLS`で
  必要な列だけ指定して読み込み（メモリ・読み込み時間の節約），
  分かりやすい名前に`rename()`している。
- 信頼度の変数は元データで「1=とても信頼する 〜 4=全く信頼しない」
  という**値が大きいほど信頼が低い**逆向きの尺度になっているため，
  `jp[dst] = 5 - jp[src]` で反転し，「大きいほど信頼が高い」に
  統一している（メディア利用頻度も同様に`6 - 値`で反転）。
- `X028`（雇用状態，8カテゴリ）を`is_unemployed`
  （7=Unemployed）・`is_not_in_labor_force`（4=Retired, 5=Housewife,
  6=Students, 8=Other）の2ダミーに変換。ISSP版の`WORKYN`と対応する
  設計にしている。
- `X047R_WVS`（主観的所得水準：1=低 2=中 3=高）はISSPには存在しない
  変数で，「経済的破壊の当事者性」をより直接的に測る変数として
  追加している（コメントに明記の通り，査読メモ`academic_review`
  での指摘に対応したもの）。

### メディア利用変数の非対称性への対応
`E253B`（SNS利用）はWave 7（2019年）でのみ調査されており，Wave 6
（2010年）には存在しない。そのため
```python
POOLED_MEDIA_FEATURES = ["internet_use", "tv_news_use", "newspaper_use"]  # Wave6+7
SNS_MEDIA_FEATURES = ["sns_use", "internet_use", "tv_news_use", "newspaper_use"]  # Wave7のみ
```
と2つの変数セットを用意し，`main()`内で(A)Wave6+7をプールした
分析（SNSなし，サンプル数が多い）と(B)Wave7限定の分析（SNSあり，
サンプル数は少ない）の両方を実行している。これが，卒論本文で
「(A)プール分析」「(B)Wave7限定分析」と呼んでいるものの実装。

### `compare_ols_vs_rf()` / `shap_analysis()`
`issp_ml_analysis.py`とほぼ同じ処理（5分割交差検証でOLS対RFの
R²比較，TreeExplainerでSHAP要約プロット）。ここでは追加で
```python
X_sm = sm.add_constant(X)
ols_full = sm.OLS(y, X_sm).fit()
out_lines.append(ols_full.summary().tables[1].as_text())
```
と，`statsmodels`の`sm.OLS`で全データに対するOLSも別途走らせ，
各説明変数の係数・p値を出力している（`cross_val_score`は予測性能
の数値だけを返すため，「SNS利用の係数がいくつでp値は」という
個別の統計的検定結果を見るには別途フルデータでOLSを推定する
必要があるため）。

---

## 4. `scripts/wvs_cross_country_analysis.py` — 5か国比較版

### 目的
`wvs_timeseries_analysis.py`のWave 7限定分析（日本のみ）を，
米国・韓国・ドイツ・英国にも同じ変数構成で適用し，「SNS利用が
制度への信頼に対して非有意」という日本の結果が，日本固有の現象か
どうかを検証する。

### `COUNTRIES`辞書
```python
COUNTRIES = {"Japan": 392, "United States": 840, "South Korea": 410,
             "Germany": 276, "United Kingdom": 826}
```
WVSのS003（国コード）は ISO 3166-1 numeric に準拠。ドイツは統一後
コード（276）を使用し，統一前の西独/東独separate code（900/901）は
使っていない，とコメントに明記されている。

### `sns_effect_for_country(df, country, target, out_lines)`
`wvs_timeseries_analysis.py`のWave7限定分析とほぼ同じ処理を，
`country`でループしながら1か国ずつ実行する関数。
```python
if len(data) < 20:
    out_lines.append(f"... サンプルサイズ不足 (N={len(data)}) のためスキップ")
    return
```
のように，国によってはサンプルサイズが極端に小さい場合に分析を
スキップする安全策も入っている。SHAP要約プロットも国ごとに
`wvs_cross_country_shap_{国名}_{target}.png`として保存する。

---

## 5. 使われている統計・機械学習手法のまとめ

| 手法 | 使用箇所 | 何のために使うか |
|---|---|---|
| OLS（最小二乗法） | 全スクリプト | 説明変数と目的変数の線形関係を推定する最も基本的な回帰分析 |
| Random Forest回帰 | `issp_ml_analysis.py`, `wvs_timeseries_analysis.py`, `wvs_cross_country_analysis.py` | 非線形性・変数間の交互作用を捉えられるか，OLSと比較検証する |
| 5分割交差検証 | 同上 | 未知データへの予測力を過学習なく公平に評価する |
| SHAP（SHapley値） | 同上 | Random Forestのブラックボックスな予測を，各変数の寄与度に分解して解釈可能にする |
| K-means クラスタリング | `issp_ml_analysis.py` | 回答者を似た政治的態度パターンでグループ分けする教師なし学習 |
| シルエット係数 | 同上 | クラスタ数$K$の妥当性を定量的に評価する |
| 標準化（StandardScaler） | 同上 | スケールの異なる変数を公平にクラスタリングに使えるようにする |

すべて`sklearn`（scikit-learn，Pythonの標準的な機械学習ライブラリ）・
`statsmodels`（統計モデリング）・`shap`という，学術・実務の両方で
広く使われているオープンソースライブラリで実装されている。
