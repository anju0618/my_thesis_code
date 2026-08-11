# install.packages("haven")
setwd("C:/under my control/明治大学/ゼミ/卒論/データ/Role of Government")
library(haven)
library(dplyr)
D <- read_sav("Role of Government_ISSP.sav")

D_JPN <- D %>%
  filter(country == 392) %>%
  select(
    year = year_sdno,  # 調査年
    no_say = v61,       # 政治的有効性感覚の欠如
    trust_civil = v66,  # 公務員への信頼
    corruption = v73,   # 政治家の汚職認識
    pol_interest = v60, # 政治への関心
    age = AGE,          # 年齢
    sex = SEX,          # 性別
    education = DEGREE  # 学歴
  )
D_JPN
print(head(D_JPN))

result <- D_JPN %>%
  group_by(year) %>%
  summarise(
    avg_no_say = mean(no_say, na.rm = TRUE),
    avg_trust_civil = mean(trust_civil, na.rm = TRUE),
    avg_corruption = mean(corruption, na.rm = TRUE),
    avg_pol_interest = mean(pol_interest, na.rm = TRUE)
  )
print(result)

result2 <- D_JPN %>%
  group_by(year) %>%
  summarise(
    avg_pol_interest = mean(pol_interest, na.rm = TRUE)
  )

# パッケージを読み込む
library(ggplot2)
library(tidyr) # データの整形（縦持ち）に使います

# --------------------------------------------------------
# ステップ6：グラフ作成
# --------------------------------------------------------
# ggplot2は「縦長」のデータを好むため、resultの3つの平均値列を1列にまとめます
result_long <- result %>%
  pivot_longer(
    cols = c(avg_no_say, avg_trust_civil, avg_corruption, avg_pol_interest), # 1列にまとめたい列
    names_to = "variable",  # 新しい「変数名」列
    values_to = "average"   # 新しい「平均値」列
  )

# グラフの作成
# (yearを数値として扱うために as.numeric() を使います)
trend_plot <- ggplot(result_long, 
                     aes(x = as.numeric(year), y = average, color = variable, group = variable)) +
  geom_line(linewidth = 1.5) +  # 折れ線グラフ
  geom_point(size = 3) +     # 点
  labs(
    title = "日本の政治意識の時系列変化 (1996-2016年)",
    subtitle = "データソース: ISSP Role of Government (ZA4747)",
    x = "調査年",
    y = "平均値 (尺度 1-5)",
    color = "質問項目" # 凡例のタイトル
  ) +
  scale_color_manual(labels = c(
    "avg_corruption" = "政治家の汚職認識 (v73)",
    "avg_no_say" = "政治的有効性感覚の欠如 (v61)",
    "avg_trust_civil" = "公務員への信頼 (v66)",
    "avg_pol_interest" = "政治への関心 (v60)"
  ),
  values = c(
    "avg_corruption" = "#E41A1C", # 赤
    "avg_no_say" = "#377EB8",     # 青
    "avg_trust_civil" = "#4DAF4A", # 緑
    "avg_pol_interest" = "#FFFF00"
  )) +
  scale_x_continuous(breaks = c(1996, 2006, 2016)) + # X軸の目盛りを年に合わせる
  theme_minimal() + # シンプルなデザイン
  theme(legend.position = "bottom") # 凡例を下に

# グラフを表示
print(trend_plot)

# --------------------------------------------------------
# ステップ7：グラフをファイルとして保存
# --------------------------------------------------------
# 卒論のWordファイルに貼り付けるための画像ファイルとして保存します
# ggsave("issp_japan_trend_plot.png", plot = trend_plot, width = 10, height = 6)

# 回帰分析の結果を「卒論の表」のように美しく出力するパッケージ
# install.packages("gtsummary")
library(gtsummary)
library(gt)
# このコマンドでロックファイルを削除します
unlink("C:/Users/joker/AppData/Local/R/win-library/4.4/00LOCK")

D_JPN_clean <- D_JPN %>%
  mutate(
    # as.numeric() で「ラベル」を外し、数値として扱えるようにする
    year_num = as.numeric(year),
    no_say_num = as.numeric(no_say),
    trust_civil_num = as.numeric(trust_civil),
    corruption_num = as.numeric(corruption),
    pol_interest_num = as.numeric(pol_interest),
    age_num = as.numeric(age),
    
    # as.factor() で「カテゴリ変数（ダミー変数）」として扱う
    # これで「1996年と比べて」という比較が可能になる
    year_factor = as.factor(year),
    sex_factor = as.factor(sex),
    education_factor = as.factor(education)
  ) %>%
  # 回帰分析は欠損値(NA)があると実行できないため、分析に使う変数のNAを削除
  # (na.omit()はリストワイズ削除と呼ばれます)
  select(
    year_factor, no_say_num, trust_civil_num, corruption_num, 
    pol_interest_num, age_num, sex_factor, education_factor
  ) %>%
  na.omit()

# --------------------------------------------------------
# ステップ3：モデルの実行（lm = 線形モデル）
# --------------------------------------------------------

# モデル1：政治的有効性感覚（v61: no_say）への影響
# no_say_num ~ year_factor + [統制変数...]
# (1996年を基準として、2006年と2016年の影響を見ます)
model_1 <- lm(no_say_num ~ year_factor + age_num + sex_factor + education_factor + pol_interest_num, 
              data = D_JPN_clean)

# モデル2：公務員への信頼（v66: trust_civil）への影響
model_2 <- lm(trust_civil_num ~ year_factor + age_num + sex_factor + education_factor + pol_interest_num, 
              data = D_JPN_clean)

# モデル3：汚職認識（v73: corruption）への影響
# v73は1996年に調査がない(NA)ため、1996年を除外したデータで分析
D_JPN_corr <- D_JPN_clean %>% filter(year_factor != 1996)

model_3 <- lm(corruption_num ~ year_factor + age_num + sex_factor + education_factor + pol_interest_num, 
              data = D_JPN_corr)

# --------------------------------------------------------
# ステップ4：分析結果を表にする（gtsummary）
# --------------------------------------------------------
# これで卒論にそのまま貼れるような、非常に美しい表が作成されます

# まず、モデル1の表を作成
table_1 <- tbl_regression(
  model_1,
  exponentiate = FALSE,
  pvalue_fun = ~style_pvalue(.x, digits = 3)
)

# 次に、モデル2の表を作成
table_2 <- tbl_regression(
  model_2,
  exponentiate = FALSE,
  pvalue_fun = ~style_pvalue(.x, digits = 3)
)

# tbl_merge() を使って2つの表を横に結合します
table1_2_merged <- tbl_merge(
  tbls = list(table_1, table_2),
  tab_spanner = c("**モデル1: 政治的有効性感覚(v61)**", "**モデル2: 公務員信頼(v66)**")
)

# RStudioのViewerで結合した表（table1_2_merged）を表示
print(table1_2_merged)

# 結合した表をHTMLファイルとして保存
table1_2_merged %>%
  as_gt() %>%
  gtsave(file = "regression_table_1_2.html")

print("regression_table_1_2.html を保存しました。")


# --- モデル3（汚職認識）の表を作成 ---
table_3 <- tbl_regression(
  model_3,
  exponentiate = FALSE,
  pvalue_fun = ~style_pvalue(.x, digits = 3)
)

# RStudioのViewerで表3を表示
print(table_3)

# 表3をHTMLファイルとして保存
table_3 %>%
  as_gt() %>%
  gtsave(file = "regression_table_3.html")

print("regression_table_3.html を保存しました。")
D


# --- ステップ1：分析用データの再作成（v65とv60を追加） ---
# v65: 議員への信頼（公約）
# v60: 政治的関心
D_JPN_clean_v2 <- D %>%
  filter(country == 392) %>%
  select(
    year = year_sdno,
    no_say = v61,
    trust_civil = v66,  # 比較用のモデル2
    trust_mps = v65,    # ★新しいモデル4のY
    corruption = v73,   # 比較用のモデル3
    pol_interest = v60, # 統制変数
    age = AGE,
    sex = SEX,
    education = DEGREE
  ) %>%
  mutate(
    year_num = as.numeric(year),
    no_say_num = as.numeric(no_say),
    trust_civil_num = as.numeric(trust_civil),
    trust_mps_num = as.numeric(trust_mps), # v65を数値化
    corruption_num = as.numeric(corruption),
    pol_interest_num = as.numeric(pol_interest),
    age_num = as.numeric(age),
    year_factor = as.factor(year),
    sex_factor = as.factor(sex),
    education_factor = as.factor(education)
  )

# --- ステップ2：新しいモデル（model_4）の作成 ---
# v65(trust_mps_num)は1996, 2006, 2016年の全てに存在します
model_4_data <- D_JPN_clean_v2 %>%
  select(year_factor, trust_mps_num, age_num, sex_factor, education_factor, pol_interest_num) %>%
  na.omit() # 欠損値を含む行を削除

model_4 <- lm(trust_mps_num ~ year_factor + age_num + sex_factor + education_factor + pol_interest_num, 
              data = model_4_data)

summarise(model_4_data)
model_4_data

# --- ステップ3：モデル4の結果を表にする ---
table_4 <- tbl_regression(
  model_4,
  exponentiate = FALSE,
  pvalue_fun = ~style_pvalue(.x, digits = 3),
  model_label = "モデル4: 政治家(議員)への信頼 (v65)"
)

# RStudioのViewerに表示
print(table_4)
table_4 %>%
  as_gt() %>%
  gtsave(file = "regression_table_4.html")


# install.packages("tibble") # もし 'tibble' がなければ
library(tibble)

# 1. インターネット普及率のルックアップテーブルを作成
# ISSPの調査年（1996, 2006, 2016）に
# CSVの最も近い年の値を割り当てます

# 1996年: CSVの1997年(9.2%)の値を代理変数として使用
# 2006年: CSVの2006年(72.6%)の値を使用
# 2016年: CSVの2016年(83.5%)の値を使用
internet_lookup <- tibble(
  year = c(1996, 2006, 2016),
  internet_rate = c(9.2, 72.6, 83.5) 
)

print("--- インターネット普及率データ ---")
print(internet_lookup)

D_JPN <- D %>%
  filter(country == 392) %>%
  select(
    year = year_sdno, # 調査年
    no_say = v61, # 政治的有効性感覚の欠如
    trust_civil = v66, # 公務員への信頼
    corruption = v73, # 政治家の汚職認識
    pol_interest = v60, # 政治への関心
    age = AGE, # 年齢
    sex = SEX, # 性別
    education = DEGREE # 学歴
  )

# ★ここが重要：作成した D_JPN に internet_lookup を 'year' をキーにして結合
D_JPN <- D_JPN %>%
  left_join(internet_lookup, by = "year")

# 結合できたか確認（末尾に internet_rate 列が追加されているはず）
print("--- 普及率を結合したデータ（D_JPN） ---")
print(head(D_JPN))

D_JPN_clean <- D_JPN %>%
  mutate(
    # as.numeric() で「ラベル」を外し、数値として扱えるようにする
    year_num = as.numeric(year),
    internet_rate_num = as.numeric(internet_rate), # ★インターネット普及率を追加
    no_say_num = as.numeric(no_say),
    trust_civil_num = as.numeric(trust_civil),
    corruption_num = as.numeric(corruption),
    pol_interest_num = as.numeric(pol_interest),
    age_num = as.numeric(age),
    
    # as.factor() で「カテゴリ変数（ダミー変数）」として扱う
    year_factor = as.factor(year), # ★モデル3の除外用に残す
    sex_factor = as.factor(sex),
    education_factor = as.factor(education)
  ) %>%
  # 回帰分析は欠損値(NA)があると実行できないため、分析に使う変数のNAを削除
  select(
    year_factor, internet_rate_num, # ★インターネット普及率を追加
    no_say_num, trust_civil_num, corruption_num,
    pol_interest_num, age_num, sex_factor, education_factor
  ) %>%
  na.omit()

print("--- 分析用データ D_JPN_clean ---")
print(head(D_JPN_clean))


# --------------------------------------------------------
# ステップ3：モデルの実行（インターネット普及率を使用）
# --------------------------------------------------------
# 従属変数 ~ internet_rate_num + [統制変数...]
# という形で、インターネット普及率が1%上がると、
# 政治意識（Y）が何ポイント変化するかを検証します。

# モデル1：政治的有効性感覚（v61: no_say）への影響
model_1_internet <- lm(no_say_num ~ internet_rate_num + age_num + sex_factor + education_factor + pol_interest_num, 
                       data = D_JPN_clean)

# モデル2：公務員への信頼（v66: trust_civil）への影響
model_2_internet <- lm(trust_civil_num ~ internet_rate_num + age_num + sex_factor + education_factor + pol_interest_num, 
                       data = D_JPN_clean)

# モデル3：汚職認識（v73: corruption）への影響
# v73は1996年に調査がないため、1996年を除外したデータで分析
# (このフィルタリングは元のコードのままです)
D_JPN_corr <- D_JPN_clean %>% filter(year_factor != 1996)

model_3_internet <- lm(corruption_num ~ internet_rate_num + age_num + sex_factor + education_factor + pol_interest_num, 
                       data = D_JPN_corr)

# --------------------------------------------------------
# ステップ4：分析結果を表にする（gtsummary）
# --------------------------------------------------------
# install.packages("gtsummary") # 実行済みのはず
# install.packages("gt")        # 実行済みのはず
library(gtsummary)
library(gt)

# --- モデル1とモデル2を結合した表 ---

# まず、モデル1の表を作成
table_1_new <- tbl_regression(
  model_1_internet,
  pvalue_fun = ~style_pvalue(.x, digits = 3)
)

# 次に、モデル2の表を作成
table_2_new <- tbl_regression(
  model_2_internet,
  pvalue_fun = ~style_pvalue(.x, digits = 3)
)

# tbl_merge() を使って2つの表を横に結合します
table1_2_merged_new <- tbl_merge(
  tbls = list(table_1_new, table_2_new),
  tab_spanner = c("**モデル1: 政治的有効性感覚(v61)**", "**モデル2: 公務員信頼(v66)**")
)

# RStudioのViewerで結合した表を表示
print(table1_2_merged_new)

# 結合した表をHTMLファイルとして保存
# (ファイル名は _internet をつけて区別します)
table1_2_merged_new %>%
  as_gt() %>%
  gtsave(file = "regression_table_1_2_internet.html")

print("regression_table_1_2_internet.html を保存しました。")


# --- モデル3（汚職認識）の表を作成 ---
table_3_new <- tbl_regression(
  model_3_internet,
  pvalue_fun = ~style_pvalue(.x, digits = 3)
)

# RStudioのViewerで表3を表示
print(table_3_new)

# 表3をHTMLファイルとして保存
table_3_new %>%
  as_gt() %>%
  gtsave(file = "regression_table_3_internet.html")

print("regression_table_3_internet.html を保存しました。")



# --- ステップ1：分析用データの再作成（v65とv60を追加、ネット普及率も） ---
D_JPN_clean_v2 <- D %>%
  filter(country == 392) %>%
  select(
    year = year_sdno,
    no_say = v61,
    trust_civil = v66,
    trust_mps = v65,    # ★モデル4のY
    corruption = v73,
    pol_interest = v60, # 統制変数
    age = AGE,
    sex = SEX,
    education = DEGREE
  ) %>%
  # ★ D_JPN と同じく、ここで internet_lookup を結合
  left_join(internet_lookup, by = "year") %>%
  mutate(
    year_num = as.numeric(year),
    internet_rate_num = as.numeric(internet_rate), # ★ネット普及率
    no_say_num = as.numeric(no_say),
    trust_civil_num = as.numeric(trust_civil),
    trust_mps_num = as.numeric(trust_mps), # v65を数値化
    corruption_num = as.numeric(corruption),
    pol_interest_num = as.numeric(pol_interest),
    age_num = as.numeric(age),
    year_factor = as.factor(year),
    sex_factor = as.factor(sex),
    education_factor = as.factor(education)
  )

# --- ステップ2：新しいモデル（model_4）の作成 ---
# v65(trust_mps_num)は1996, 2006, 2016年の全てに存在します
model_4_data <- D_JPN_clean_v2 %>%
  # ★ year_factor の代わりに internet_rate_num を select
  select(internet_rate_num, trust_mps_num, age_num, sex_factor, education_factor, pol_interest_num) %>%
  na.omit() # 欠損値を含む行を削除

# ★ 説明変数を internet_rate_num に変更
model_4_internet <- lm(trust_mps_num ~ internet_rate_num + age_num + sex_factor + education_factor + pol_interest_num, 
                       data = model_4_data)

# --- ステップ3：モデル4の結果を表にする ---
table_4_new <- tbl_regression(
  model_4_internet,
  exponentiate = FALSE,
  pvalue_fun = ~style_pvalue(.x, digits = 3),
  model_label = "モデル4: 政治家(議員)への信頼 (v65)"
)

# RStudioのViewerに表示
print(table_4_new)

# HTMLファイルとして保存
table_4_new %>%
  as_gt() %>%
  gtsave(file = "regression_table_4_internet.html")

print("regression_table_4_internet.html を保存しました。")


# --------------------------------------------------------
# 卒論用：グラフと表の保存
# --------------------------------------------------------
library(ggplot2)
library(gt)

# 1. ISSPの時系列グラフ (trend_plot) を保存
# ※ trend_plot オブジェクトが残っている前提です。
# もし消えていたら、ISSPのグラフ作成コードを再実行してください。
if (exists("trend_plot")) {
  ggsave("issp_trend_plot.png", plot = trend_plot, width = 10, height = 6)
  print("issp_trend_plot.png を保存しました")
} else {
  print("trend_plot が見つかりません。ISSPのグラフ作成コードを再実行してください")
}

# 2. WVSの回帰分析表 (table_wvs_merged) の保存
# 【重要】表を「画像」として保存するのが一番確実です。
# RStudioのViewerウィンドウにある「Export」ボタン -> 「Save as Image」をクリックし、
# "wvs_regression_table.png" という名前で保存してください。

# もしくは、以下のコードでLaTeX形式として保存することも可能です（上級者向け）
if (exists("table_wvs_merged")) {
  table_wvs_merged %>%
    as_gt() %>%
    gtsave("wvs_regression_table.tex")
  print("wvs_regression_table.tex を保存しました（LaTeXコード版）")
}
