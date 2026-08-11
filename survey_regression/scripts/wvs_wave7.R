setwd("C:/under my control/明治大学/ゼミ/卒論/データ/World Values Survey Wave 7 (2017-2022)")
load("WVS_Cross-National_Wave_7_v6_0.rdata")
ls()
D =  `WVS_Cross-National_Wave_7_v6_0`
D
str(D)
head(D)
summary(D)

library(dplyr)
D_JPN <- D %>%
  filter(B_COUNTRY == 392) %>%
  select(
    # 従属変数 (Y)： 政治信頼
    trust_parties = Q72,   # 政党への信頼
    trust_parl = Q73,      # 議会への信頼
    trust_civil = Q74,     # 公務員への信頼
    
    # 独立変数 (X)： メディア利用 (論文の核心)
    info_sns = Q207,       # SNSからの情報入手頻度
    info_internet = Q206,  # インターネットからの情報入手頻度
    info_tv = Q202,        # テレビニュースからの情報入手頻度
    
    # 統制変数 (個人属性)
    sex = Q260,            # 性別
    age = Q262,            # 年齢
    education = Q275       # 学歴
  )
# 先頭6行を表示
print("--- 日本データ(D_WVS_JPN)の先頭 ---")
head(D_JPN)

# 各変数の要約統計量（最小値、最大値、欠損値など）を表示
print("--- 日本データ(D_WVS_JPN)の要約 ---")
summary(D_JPN)

D_JPN_clean <- D_JPN %>%
  mutate(
    # --- 1. 変数の数値化と「尺度反転」 ---
    
    # 政治信頼 (1-4 -> 4-1)
    # (例: 1 "Very much" -> 4, 4 "Not at all" -> 1)
    # こうすることで「数値が大きいほど、信頼も高い」に揃えます。
    trust_parties_num = 5 - as.numeric(trust_parties),
    trust_parl_num    = 5 - as.numeric(trust_parl),
    trust_civil_num   = 5 - as.numeric(trust_civil),
    
    # メディア利用 (1-5 -> 4-0)
    # (例: 1 "Daily" -> 4, 5 "Never" -> 0)
    # こうすることで「数値が大きいほど、利用頻度も高い」に揃えます。
    info_sns_num      = 5 - as.numeric(info_sns),
    info_internet_num = 5 - as.numeric(info_internet),
    info_tv_num       = 5 - as.numeric(info_tv),
    
    # 統制変数 (個人属性)
    age_num           = as.numeric(age),
    sex_factor        = as.factor(sex),
    education_num     = as.numeric(education) # 学歴も数値として扱います
  ) %>%
  
  # --- 2. 欠損値（マイナスの値）の除外 ---
  #
  # 反転させた結果、有効なデータが入っている行「だけ」を
  # フィルタリング（抽出）します。
  filter(
    # 信頼 (有効な範囲は 1, 2, 3, 4)
    trust_parties_num %in% 1:4,
    trust_parl_num    %in% 1:4,
    trust_civil_num   %in% 1:4,
    
    # メディア利用 (有効な範囲は 0, 1, 2, 3, 4)
    info_sns_num      %in% 0:4,
    info_internet_num %in% 0:4,
    info_tv_num       %in% 0:4,
    
    # 学歴 (有効な範囲は 0以上)
    education_num >= 0
    
    # (age と sex は summary で問題なかったので、除外不要)
  )

# --------------------------------------------------------
# ステップ5：クリーニング結果の確認
# --------------------------------------------------------
# これが分析の「最終確認」です。
print("--- クリーニング後のデータ要約 (D_JPN_clean) ---")
summary(D_JPN_clean)


library(gtsummary)
library(gt)
# Y（政治信頼） ~ X（メディア利用） + 統制変数

# モデル1：政党信頼 (trust_parties_num) への影響
# 「info_sns_num（SNS利用）」が「trust_parties_num（政党信頼）」に
# どのような影響を与えるか（年齢・性別・学歴の影響を除去した上で）
model_1_wvs <- lm(trust_parties_num ~ info_sns_num + info_internet_num + info_tv_num + 
                    age_num + sex_factor + education_num, 
                  data = D_JPN_clean)

# モデル2：議会信頼 (trust_parl_num) への影響
model_2_wvs <- lm(trust_parl_num ~ info_sns_num + info_internet_num + info_tv_num + 
                    age_num + sex_factor + education_num, 
                  data = D_JPN_clean)

# モデル3：公務員信頼 (trust_civil_num) への影響
model_3_wvs <- lm(trust_civil_num ~ info_sns_num + info_internet_num + info_tv_num + 
                    age_num + sex_factor + education_num, 
                  data = D_JPN_clean)


# --------------------------------------------------------
# ステップ7：分析結果を表にする (gtsummary)
# --------------------------------------------------------
# 3つのモデルの結果を、1つの表にまとめます
table_wvs_merged <- tbl_merge(
  tbls = list(
    # gtsummaryでモデル1を表に
    tbl_regression(model_1_wvs, pvalue_fun = ~style_pvalue(.x, digits = 3)),
    # gtsummaryでモデル2を表に
    tbl_regression(model_2_wvs, pvalue_fun = ~style_pvalue(.x, digits = 3)),
    # gtsummaryでモデル3を表に
    tbl_regression(model_3_wvs, pvalue_fun = ~style_pvalue(.x, digits = 3))
  ),
  # 表の列名を指定
  tab_spanner = c("**モデル1: 政党信頼 (Q72)**", 
                  "**モデル2: 議会信頼 (Q73)**", 
                  "**モデル3: 公務員信頼 (Q74)**")
)

# RStudioのViewerで表を表示
print(table_wvs_merged)

# --------------------------------------------------------
# ステップ8：表をHTMLファイルとして保存
# --------------------------------------------------------
# このHTMLファイルをWordに貼り付ければ、卒論の表が完成します
table_wvs_merged %>%
  as_gt() %>%
  gtsave(file = "regression_table_wvs_japan.html")

print("--- regression_table_wvs_japan.html を保存しました ---")
