# 分析1：既存アンケート調査データによる機械学習分析（Python）

ISSP（International Social Survey Programme）およびWVS（World Values
Survey）Time Seriesの日本データを用い，Random Forest回帰・SHAP・
K-meansクラスタリングにより，政治的信頼・効力感を規定する要因を検証する。
Pythonを使用（`uv`で環境管理）。卒論第4部「分析1」として
`sections/04_empirical.tex` に結果を反映済み。

**2026-08-11: R（`.R`）ベースの旧版OLS分析から，Python（`.py`）ベースの
機械学習分析へ全面的に作り直した。** さらに同日中に，当初読み込めなかった
WVS Wave 7の個票データについて，WVS公式サイトから**SPSS形式**で
Wave 1--7統合の「Time Series」データセットを取得し直すことで解決し，
1981--2019年の長期トレンド分析と，SNS利用頻度を含む個人単位のメディア
利用効果分析を追加した（下記「WVSデータの入手経緯」参照）。

## セットアップ

```bash
cd analysis/survey_regression
uv sync   # pyproject.toml / uv.lock から .venv を作成・依存関係インストール
```

## 使用データ

| データセット | 対象 | 状態 |
|---|---|---|
| `data/issp_role_of_government/Role of Government_ISSP.sav` | ISSP「Role of Government」（ZA4747），日本，1996/2006/2016年 | **使用中**（`pyreadstat`で正常に読み込み可能，$N=4{,}091$） |
| `data/wvs_time_series/WVS_Time_Series_1981-2022_spss_v5_0.sav` | WVS Time Series（1981–2022，Wave 1–7統合，v3.0），全世界443,488件・1046変数 | **使用中**（`pyreadstat`で正常に読み込み可能。日本サブセット$N=9{,}523$，全7波） |
| `data/wvs_wave7/`（コードブックPDFのみ） | WVS Wave 7単体の`.rdata`は削除済み | 不使用（Time Seriesに統合されているため個票データ自体は不要になった） |
| `data/auxiliary/*.csv` | 総務省統計等 | インターネット普及率の代理変数の参照値として使用中（ISSP分析） |

いずれも個票データであり，サイズが大きい（wvs_time_seriesの`.sav`のみで
約611MB）ため`.gitignore`でリポジトリ管理対象外にしている。

### WVSデータの入手経緯

当初WVS Wave 7単体の`.rdata`ファイルを試したが，R独自の「ASCII形式」
シリアライズの中に正しくエスケープされていない生のUTF-8バイト列が
含まれており，Pythonの主要な`.rdata`読み込みライブラリ2種（`pyreadr`,
`rdata`）のいずれでも読み込めなかった（詳細な技術的経緯は
`sections/90_appendix_math.tex`の付録参照）。パッチによる強行突破は
データ破損のリスクがあるため断念し，代わりにWVS公式サイトで配布されて
いる**SPSS形式**のWave 1--7統合Time Seriesデータセットを取得し直した
ところ，問題なく読み込めた。結果として，Wave 7単体（2019年，$N \approx
1{,}350$）よりも長い1981--2022年の時系列と，より大きなサンプルサイズが
得られた。

**教訓**: 同じデータセットでも配布形式（`.rdata`か`.sav`か）によって
読み込みの成否が左右されうる。特定の形式で問題が生じた場合，まず配布元が
提供する別形式を確認する方が，ファイル自体の修復を試みるより効率的である
ことが多い。

## スクリプト

- `scripts/issp_ml_analysis.py`: ISSP日本データの機械学習分析。OLS対
  Random Forestの5分割交差検証によるR²比較，SHAP値による特徴量寄与の
  可視化，$K$-meansクラスタリングによる政治的態度プロファイルの分類と
  年次推移の可視化を行う。
- `scripts/wvs_timeseries_analysis.py`: WVS Time Series日本データの分析。
  (1)1981--2019年の制度への信頼の長期トレンド，(2)Wave 6+7プールでの
  メディア利用（インターネット/テレビ/新聞）効果分析，(3)Wave 7限定での
  SNS利用を含む4変数の効果分析（OLS対Random Forest比較・SHAP）を行う。
- `scripts/issp_analysis.py`: ISSP日本データのOLS回帰分析（旧`Role of
  Government.r`のPython再実装，比較用ベースライン）。
- `scripts/issp_role_of_government.R` / `scripts/wvs_wave7.R` /
  `scripts/wvs_time_series_template.R`: 旧版のRスクリプト（参考保持，
  不使用）。

## 実行方法

```bash
uv run scripts/issp_ml_analysis.py        # ISSP機械学習分析
uv run scripts/wvs_timeseries_analysis.py  # WVS長期トレンド + メディア利用分析
uv run scripts/issp_analysis.py            # ISSP OLS回帰分析（比較用ベースライン）
```

## 主要な結果

### ISSP（1996–2016年）
- **OLS対Random Forest**: 政治的有効性感覚の欠如・公務員信頼・汚職認識の
  いずれについても，Random ForestはOLSよりわずかに高い予測性能（5分割CV
  R²）を示した（公務員信頼: OLS 0.029 → RF 0.047）。ただしいずれもR²は
  0.06未満にとどまる。
- **クラスタリング**: 「効力感・信頼・汚職認識・関心のいずれも低い」
  クラスタが2006年の25.8%から2016年の35.2%へと増加。

### WVS Time Series（1981–2019年）
- **長期トレンド**: 議会・公務員・政府・政党への信頼はいずれも1980年代から
  2000年代半ばにかけて緩やかに低下し，2005年前後に底を打った後，2019年
  には系列開始以来の最高水準まで回復した。
- **メディア利用効果（Wave 7限定，SNS利用を含む，$N\approx1{,}170$）**:
  **SNS利用頻度は議会信頼（$p=0.747$）・公務員信頼（$p=0.997$）のいずれに
  対しても統計的に有意な効果を持たなかった**。対照的に，新聞利用頻度は
  両方の信頼指標に対して統計的に有意な正の効果を持ち（$p=0.019$，
  $p<0.001$），テレビニュース利用頻度も議会信頼に対して有意（$p=0.002$）
  だった。

結果一式は`results/`内のテキストサマリ（`ml_model_comparison.txt`,
`cluster_summary.txt`, `wvs_media_ml_comparison.txt`）・画像
（`shap_summary_*.png`, `cluster_profile.png`, `wvs_trend.png`,
`wvs_shap_summary_*.png`）を参照。旧版OLS分析の結果（HTML表・PNG）も
同ディレクトリに残してある。

## 統計的限界

1. **内生性**: メディア利用・政治的関心と政治的信頼の間の逆因果を
   識別できていない。
2. **除外変数バイアス**: 所得・居住地・支持政党等の重要な統制変数が未投入。
3. **Wave 7限定のSNS分析のサンプルサイズ**: $N\approx1{,}170$と相対的に
   小さく，2019年単一時点の横断面データであるため，SNS利用効果の時代的
   変化を検証できていない。

これらの限界は，卒論本文でも明記した上で，「機械学習かつ個人単位の
SNS利用変数を直接投入してもなお，マクロな調査データでは政治的疎外の
要因を十分に説明できない」ことを，ミクロな言説データ（分析2）による
補足的検証の根拠として位置づける。

## 今後の課題

- 所得・支持政党等の統制変数を追加投入する。
- WVS Time Seriesの全波データを用いたパネル的・マルチレベル分析への拡張。
- `setwd()`の絶対パス依存が残る旧Rスクリプトの整理（現在は不使用のため
  優先度低）。
