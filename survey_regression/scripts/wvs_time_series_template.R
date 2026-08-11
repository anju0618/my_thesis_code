# --------------------------------------------------------
# 卒論 第4章（発展）：WVS時系列データの分析準備
# --------------------------------------------------------
setwd("C:/under my control/明治大学/ゼミ/卒論/データ/F00011932-WVS_Time_Series_1981-2022_Rdata_v5_0")
library(dplyr)

# ========================================================
# 1. 時系列分析 (Wave 5:2005年 - Wave 7:2019年)
# ========================================================
# データ作成とクリーニング
D_Final <- D %>%
  filter(S003 == 392) %>%  # 日本のみ抽出
  select(
    wave = S002VS,
    year = S020,
    
    # --- 従属変数: 政党信頼 ---
    trust_parties = E069_12,
    
    # --- 独立変数: インターネット/SNS ---
    net_old_w5    = E253,   # Wave 5用 (0:Not used, 1:Used)
    net_social_w7 = E253B,  # Wave 7用 (1:Daily ... 5:Never)
    
    # --- 統制変数 ---
    interest_pol = E023,    # 政治的関心 (ニュース関心の代用)
    age          = X003,
    sex          = X001,
    educ         = X025,
    inc          = X047_WVS # 所得
  ) %>%
  mutate(
    # --- A. 従属変数の処理 (数値反転) ---
    # 1:A great deal -> 4:None なので、5から引いて反転 (1->4, 4->1)
    trust_parties_n = 5 - as.numeric(trust_parties),
    
    # --- B. インターネット変数の統合 (最重要) ---
    # Wave 7のSNS変数を 0/1 に変換 (1〜4:利用あり=1, 5:利用なし=0)
    net_social_dummy = case_when(
      net_social_w7 <= 4 ~ 1, # Daily, Weekly, Monthly, Less than monthly
      net_social_w7 == 5 ~ 0, # Never
      TRUE ~ NA_real_
    ),
    
    # Wave 5の変数と合体 (coalesce: net_socialがあれば使い、なければnet_oldを使う)
    source_net_unified = coalesce(net_social_dummy, as.numeric(net_old_w5)),
    
    # --- C. その他の変数処理 ---
    # 政治関心 (1:Very -> 4:Not at all なので反転)
    interest_pol_n = 5 - as.numeric(interest_pol),
    
    female = ifelse(sex == 2, 1, 0), # 女性ダミー
    educ_n = as.numeric(educ),
    income_n = as.numeric(inc)
  ) %>%
  # Wave 5 (2005年) 以降に限定
  filter(wave >= 5)

# 最終確認: データがちゃんと残っているか？
print("--- Waveごとのデータ数とインターネット変数の欠損なし数 ---")
table(D_Final$wave, !is.na(D_Final$source_net_unified))

# 分析実行 (Wave固定効果モデル)
# as.factor(wave) を入れることで、「時代の違い」をコントロール
model_final_trend <- lm(trust_parties_n ~ source_net_unified + interest_pol_n + 
                          age + female + educ_n + income_n + as.factor(wave), 
                        data = D_Final)

# 結果の表示
summary(model_final_trend)




library(dplyr)
library(ggplot2) # グラフ描画用

# 1. データ作成（Wave 1 - 7 全期間用）
D_Long <- D %>%
  filter(S003 == 392) %>%  # 日本のみ
  select(
    wave = S002VS,
    year = S020,
    
    # --- 従属変数：政治信頼 ---
    # 政党(E069_12)はWave 1にないので、議会(E069_07)を使用
    trust_parl    = E069_07,
    
    # --- 独立変数：長期的価値観・メディア ---
    trust_press   = E069_04,   # マスコミへの信頼
    interest_pol  = E023,      # 政治への関心
    post_mat      = Y002,      # 脱物質主義 (1:Materialist, 2:Mixed, 3:Post-mat)
    
    # --- 統制変数 ---
    age = X003,
    sex = X001,
    inc = X047_WVS,     # 所得
    educ = X025         # 学歴 (※Wave 1等で欠損の可能性あるが一旦含める)
  ) %>%
  mutate(
    # --- リコーディング（数値を扱いやすく反転） ---
    
    # 信頼 (1:Great deal ... 4:None) -> (1:低い ... 4:高い)
    trust_parl_n  = 5 - as.numeric(trust_parl),
    trust_press_n = 5 - as.numeric(trust_press),
    
    # 政治関心 (1:Very ... 4:Not at all) -> (1:低い ... 4:高い)
    interest_pol_n = 5 - as.numeric(interest_pol),
    
    # 脱物質主義 (そのままでOK: 1->3で高くなる)
    post_mat_n = as.numeric(post_mat),
    
    # 統制変数
    female = ifelse(sex == 2, 1, 0),
    age_n = as.numeric(age),
    educ_n = as.numeric(educ),
    income_n = as.numeric(inc)
  ) %>%
  # 欠損値の除外（全期間データなので慎重に）
  filter(
    trust_parl_n >= 1,
    trust_press_n >= 1,
    interest_pol_n >= 1,
    post_mat_n >= 1
    # 学歴(educ)は欠損が多い場合があるので、ここでのfilterから一旦外しても良い
  )

# 2. 確認：各Waveのデータ数
print("--- Waveごとのサンプル数 ---")
table(D_Long$wave)


# 3. 分析モデル構築
# モデルA: 全期間プール分析（時代ダミー入り）
# 「40年間全体を通して、何が政治信頼を決めているのか？」
model_long_A <- lm(trust_parl_n ~ trust_press_n + interest_pol_n + post_mat_n + 
                     age_n + female + income_n + as.factor(wave), 
                   data = D_Long)

# モデルB: 交互作用（時代×マスコミ信頼）
# 「マスコミを信じることの意味は、昔と今で変わったのか？」
# マスコミ信頼 * Wave の交互作用を見ます
model_long_B <- lm(trust_parl_n ~ trust_press_n * as.factor(wave) + 
                     interest_pol_n + post_mat_n + age_n + female + income_n, 
                   data = D_Long)

# 結果の表示
print("--- モデルA: 全体傾向 ---")
summary(model_long_A)

print("--- モデルB: マスコミと政治信頼の関係の変化（交互作用） ---")
# 係数が多いので、交互作用項（:as.factor(wave)）に注目してください
summary(model_long_B)


# 4. （おまけ）グラフで可視化
# 「マスコミ信頼」と「議会信頼」の相関が、時代とともにどう変化したか？
# これが出ると非常にカッコいい図になります
library(ggplot2)
plot_trend <- D_Long %>%
  filter(!is.na(wave)) %>%
  ggplot(aes(x = trust_press_n, y = trust_parl_n)) +
  geom_smooth(method = "lm", aes(color = as.factor(wave)), se = FALSE) +
  labs(title = "マスコミ信頼と議会信頼の相関の変化 (Wave 1-7)",
       x = "マスコミへの信頼 (高)", y = "議会への信頼 (高)",
       color = "Wave") +
  theme_minimal()

# グラフ保存
ggsave("trend_trust_press_parl.png", plot_trend)
print(plot_trend)


library(ggplot2)
library(sjPlot) # 相互作用の可視化に便利なパッケージ
library(sjmisc)

# もしパッケージがなければインストールしてください
# install.packages("sjPlot")
# install.packages("sjmisc")

# テーマ設定（論文用に見やすく）
theme_set(theme_classic(base_size = 12))

# 交互作用プロットの作成
# 「マスコミ信頼(X)」が「議会信頼(Y)」に与える影響が、「Wave(色)」によってどう違うか？
plot_model(model_long_B, type = "pred", terms = c("trust_press_n", "wave")) +
  labs(
    title = "マスコミへの信頼が議会信頼に与える影響の推移 (1981-2019)",
    y = "議会への信頼 (予測値)",
    x = "マスコミへの信頼 (1:低 〜 4:高)",
    color = "調査年 (Wave)"
  ) +
  theme(legend.position = "bottom")

