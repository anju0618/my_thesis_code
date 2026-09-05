# Empirical Analysis (Part 4): Two Complementary Analyses

*[日本語](README.md)*

Part 4 ("Empirical Analysis") of the thesis consists of two analyses that differ in nature.

## Analysis 1: Machine-learning analysis of existing survey data (`survey_regression/`)

A macro, survey-data-level analysis (Python) using individual-level data from the ISSP
(International Social Survey Programme, Japan, 1996/2006/2016) and the WVS Time Series
(1981–2022, Waves 1–7 pooled, Japan subsample $N=9{,}523$). Random Forest regression, SHAP,
and $K$-means clustering are used to examine what drives political trust and efficacy. In
particular, the WVS Wave 7 (2019) data allow SNS (social media) usage frequency itself to be
entered directly as an explanatory variable, and no statistically significant effect was found.
This low predictive power — even with a flexible machine-learning method and an individual-level
SNS variable — is used as empirical support for the claim, made in Part 4 §1
(`subsec:methodology`), that "macro attitudinal indicators alone cannot capture the reality of
political polarization and alienation." See `survey_regression/README.en.md` for details.

## Analysis 2: Micro-level analysis via LLM-based structured coding of YouTube comments (`llm_discourse/`)

A micro, discourse-level analysis designed to directly capture affective/narrative polarization,
which is hard to detect with the survey-based indicators used in Analysis 1. Concepts introduced
in the theoretical part of the thesis (people-vs-elite, economic resentment, theft of enjoyment,
etc.) are operationalized as structured variables and coded on raw YouTube comment text using a
local LLM. See `llm_discourse/README.en.md` for details.

## How the two analyses relate

In Analysis 1 (macro survey data), even flexible machine-learning methods (Random Forest, SHAP,
clustering) — more flexible than plain OLS — could explain only a small fraction of the variance
in political trust and efficacy using demographic variables alone. This is simply because Analysis
1 captures only **static attitude/trust indicators**; populist mobilization, the thesis argues, is
better understood as occurring at the **level of concrete discourse and narrative** (the layer
Analysis 2 targets). The two analyses are positioned as a contrasting pair that empirically
supports this theoretical claim (theory chapters, Parts 1–3 of the thesis).

```
analysis/
  README.md / README.en.md        # this file
  survey_regression/              # Analysis 1: ML analysis of ISSP/WVS survey data (Python)
  llm_discourse/                  # Analysis 2: LLM-based structured coding of YouTube comments (Python)
```
