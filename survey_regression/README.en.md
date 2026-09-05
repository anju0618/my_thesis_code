# Analysis 1: Machine-Learning Analysis of Existing Survey Data (Python)

*[日本語](README.md)*

Using Japanese data from the ISSP (International Social Survey Programme) and the WVS (World
Values Survey) Time Series, this analysis examines the factors that shape political trust and
political efficacy via Random Forest regression, SHAP, and $K$-means clustering. Implemented in
Python (environment managed with `uv`). Results are reflected in the thesis as "Analysis 1" of
Part 4 (`sections/04_empirical.tex`).

**2026-08-11: Fully rebuilt from the old R-based (`.R`) OLS analysis to a Python-based (`.py`)
machine-learning analysis.** On the same day, the WVS Wave 7 individual-level data — which
originally could not be loaded — was fixed by re-obtaining the pooled Wave 1–7 "Time Series"
dataset in **SPSS format** from the official WVS site; this made it possible to add a 1981–2019
long-run trend analysis and an individual-level media-use effects analysis that includes SNS
usage frequency (see "How the WVS data was obtained" below).

## Setup

```bash
cd analysis/survey_regression
uv sync   # creates .venv and installs dependencies from pyproject.toml / uv.lock
```

## Data used

| Dataset | Coverage | Status |
|---|---|---|
| `data/issp_role_of_government/Role of Government_ISSP.sav` | ISSP "Role of Government" (ZA4747), Japan, 1996/2006/2016 | **In use** (loads correctly with `pyreadstat`, $N=4{,}091$) |
| `data/wvs_time_series/WVS_Time_Series_1981-2022_spss_v5_0.sav` | WVS Time Series (1981–2022, Waves 1–7 pooled, v3.0), 443,488 cases worldwide / 1,046 variables | **In use** (loads correctly with `pyreadstat`; Japan subsample $N=9{,}523$, all 7 waves) |
| `data/wvs_wave7/` (codebook PDF only) | The standalone Wave 7 `.rdata` file has been deleted | Not used (no longer needed since it is subsumed by the Time Series file) |
| `data/auxiliary/*.csv` | Statistics Bureau of Japan (MIC) etc. | Used as a proxy reference for internet penetration rate (ISSP analysis) |

All of these are individual-level microdata and are large (the `wvs_time_series` `.sav` file alone
is about 611 MB), so they are excluded from the repository via `.gitignore`.

### How the WVS data was obtained

The standalone WVS Wave 7 `.rdata` file was tried first, but its R-specific "ASCII format"
serialization contained raw UTF-8 byte sequences that were not correctly escaped, and it could not
be loaded by either of the two main Python `.rdata`-reading libraries (`pyreadr`, `rdata`) (see the
appendix in `sections/90_appendix_math.tex` for the detailed technical background). Since patching
the file by force risked data corruption, this approach was abandoned; instead, the **SPSS-format**
pooled Wave 1–7 Time Series dataset distributed on the official WVS site was obtained, and it
loaded without issue. As a result, a longer 1981–2022 time series and a larger sample size were
obtained than would have been available from Wave 7 alone (2019, $N \approx 1{,}350$).

**Lesson learned**: even for the same dataset, whether loading succeeds can depend on the
distribution format (`.rdata` vs. `.sav`). When a particular format causes problems, checking
whether the data provider offers another format is often more efficient than trying to repair the
file itself.

## Scripts

- `scripts/issp_ml_analysis.py`: Machine-learning analysis of the Japanese ISSP data. Compares OLS
  and Random Forest via 5-fold cross-validated R², visualizes feature contributions with SHAP
  values, and classifies political-attitude profiles via $K$-means clustering, visualizing their
  year-over-year change.
- `scripts/wvs_timeseries_analysis.py`: Analysis of the Japanese WVS Time Series data. Covers
  (1) the 1981–2019 long-run trend in institutional trust, (2) an analysis of media-use effects
  (internet/TV/newspaper) pooling Waves 6+7, and (3) an analysis of four variables including SNS
  usage restricted to Wave 7 (OLS vs. Random Forest comparison, SHAP).
- `scripts/wvs_cross_country_analysis.py`: **Cross-national comparison** using the WVS Time Series
  (added 2026-08-11, to check whether the findings for Japan are Japan-specific or shared
  internationally). Covers Japan, the United States, South Korea, Germany, and the United Kingdom
  (configurable via the `COUNTRIES` dictionary; 60 other countries have usable SNS-usage variables
  in Wave 7). (1) Overlays the long-run trend in trust in parliament/civil service by country, and
  (2) computes and tabulates the "SNS use → trust" effect (OLS coefficient/p-value, OLS-vs-RF
  comparison, SHAP) by country, restricted to Wave 7. **Result**: in Japan, the United States, and
  Germany, SNS usage frequency was non-significant for both parliamentary and civil-service trust
  (South Korea was also close to non-significant for parliamentary trust, $p=0.070$), broadly
  consistent with the Japan finding; **only the United Kingdom showed a statistically significant
  negative effect on parliamentary trust** (coefficient $-0.0224$, $p=0.028$). This shows that the
  Japanese finding — "SNS use has no significant effect on trust" — replicates in many countries
  but is not universal; whether to reflect this in the thesis body is undecided (to be considered).
  See `results/wvs_cross_country_sns_effect_summary.txt` and
  `results/wvs_cross_country_trend_*.png`, `results/wvs_cross_country_shap_*.png` for details.
- `scripts/issp_analysis.py`: OLS regression analysis of the Japanese ISSP data (a Python
  reimplementation of the old `Role of Government.r`, kept as a comparison baseline).
- `scripts/mediation_analysis.py` (added 2026-08-28): Directly tests Ricci's causal chain
  (economic destruction → resentment → elite distrust) via formal mediation analysis — a stepwise
  regression framework (total effect, path a, path b, direct effect) combined with a bootstrap test
  of the indirect effect (2,000 resamples) — rather than an indirect proxy such as SHAP feature
  importance. X = employment status (unemployed / not in the labor force), M = `no_say` (a sense of
  political powerlessness, used as a proxy for resentment since the ISSP contains no direct measure
  of it), Y = trust in civil servants / perceived corruption.
- `scripts/wvs_cross_national_destruction.py` (added 2026-08-28): Whereas
  `wvs_cross_country_analysis.py` treats SNS use as the main explanatory variable, this script
  compares Japan, the US, and Germany on "exposure to economic destruction" itself (subjective
  income level and employment status) — the core construct in Ricci's theory. It runs both a pooled
  regression with interaction terms and separate per-country regressions.
- `scripts/crosstab_simple.py` (added 2026-08-28): A comparison of political attitudes by
  employment status (means ± 95% confidence intervals, one-way ANOVA) that involves no machine
  learning at all.
- `scripts/wvs_mobilization_analysis.py` (added 2026-08-28): Directly tests the "second half" of
  Ricci's causal chain — whether resentment/distrust actually translates into mobilization
  (protest participation) — using WVS protest-participation experience (E027). To avoid relying on
  a single survey item, a robustness check is also run using a composite indicator (signing a
  petition, boycotting, demonstrating, or striking).
- `scripts/issp_role_of_government.R` / `scripts/wvs_wave7.R` /
  `scripts/wvs_time_series_template.R`: legacy R scripts (kept for reference; no longer used).

## How to run

```bash
uv run scripts/issp_ml_analysis.py        # ISSP machine-learning analysis
uv run scripts/wvs_timeseries_analysis.py  # WVS long-run trend + media-use analysis
uv run scripts/issp_analysis.py            # ISSP OLS regression (comparison baseline)
```

## Main results

### ISSP (1996–2016)

- **OLS vs. Random Forest**: for lack of political efficacy, trust in civil servants, and
  perceived corruption alike, Random Forest showed slightly higher predictive performance
  (5-fold CV R²) than OLS (trust in civil servants: OLS 0.029 → RF 0.047). Both, however, stayed
  below R² = 0.06.
- **Clustering**: the cluster characterized by low efficacy, low trust, low corruption awareness,
  and low political interest grew from 25.8% (2006) to 35.2% (2016).

### WVS Time Series (1981–2019)

- **Long-run trend**: trust in parliament, the civil service, government, and political parties
  all declined gradually from the 1980s to the mid-2000s, bottomed out around 2005, and then
  recovered by 2019 to the highest level since the series began.
- **Media-use effects (Wave 7 only, including SNS use, $N \approx 1{,}170$)**: **SNS usage
  frequency had no statistically significant effect on either trust in parliament ($p=0.747$) or
  trust in civil servants ($p=0.997$)**. By contrast, newspaper usage frequency had a
  statistically significant positive effect on both trust measures ($p=0.019$, $p<0.001$), and TV
  news usage frequency was also significant for trust in parliament ($p=0.002$).

The full set of results is in the text summaries in `results/` (`ml_model_comparison.txt`,
`cluster_summary.txt`, `wvs_media_ml_comparison.txt`) and the images (`shap_summary_*.png`,
`cluster_profile.png`, `wvs_trend.png`, `wvs_shap_summary_*.png`). Results from the earlier OLS
analysis (HTML tables, PNGs) are also kept in the same directory.

### Additional analyses (2026-08-28 to 29): directly testing Ricci's causal chain

Ricci's causal chain itself — which SHAP and clustering could not test directly — was tested
head-on using formal mediation analysis and a mobilization analysis.

- **Mediation analysis (ISSP)**: the indirect effect of economic destruction (unemployed / not in
  the labor force) → political alienation (`no_say`) → elite distrust (trust in civil servants /
  perceived corruption) was non-significant in all four combinations tested — the bootstrap 95% CI
  included zero in every case. In particular, path a (economic destruction → alienation) itself was
  not significant in any combination ($p=0.48$–$0.97$), so **the starting point of Ricci's causal
  chain is not observed in this thesis's data**. That said, the total/direct effect of "not in the
  labor force" on trust in civil servants / perceived corruption is itself significant ($p<0.001$,
  $p=0.029$), meaning that a link between economic marginalization and institutional distrust does
  exist — it just cannot be explained via the alienation pathway Ricci posits. See
  `results/mediation_summary.txt` for details.
- **Japan–US–Germany cross-national comparison (economic-destruction indicators)**: the interaction
  between subjective income level and country is statistically significant for both parliamentary
  and civil-service trust when comparing Japan against the US and Germany (US: $p<0.001$, Germany:
  $p=0.001$, for parliamentary trust). However, income level is also significant for trust in civil
  servants within Japan alone ($p<0.001$), so it is not simply the case that "economic destruction
  is unrelated to institutional distrust in Japan." See
  `results/wvs_cross_national_destruction_summary.txt` for details.
- **Direct test of distrust → mobilization**: in a logistic regression with WVS protest-
  participation experience (E027) as the outcome, the odds ratio for government distrust was
  significant in Japan, the US, and Germany alike (Japan: OR=0.777, $p=0.047$) — consistent with a
  relationship where stronger distrust is associated with having participated in a protest.
  **However, when robustness was checked using a composite indicator of signing a petition,
  boycotting, demonstrating, or striking, this relationship became non-significant in Japan**
  (OR=0.993, $p=0.908$; in the US and Germany it remained robustly significant with the composite
  indicator too, $p<0.001$). The single-item (E027) result is not robust for Japan, so this thesis
  adopts the more conservative conclusion that **the "exit" side of Ricci's causal chain (distrust
  → mobilization) is likewise not supported by this thesis's data**. See
  `results/wvs_mobilization_summary.txt` for details.
- **Simple comparison by employment status (ANOVA)**: `no_say`, trust in civil servants, and
  perceived corruption all show statistically significant differences across employed / fully
  unemployed / not-in-labor-force groups ($p<0.01$), but the direction does not always match the
  theoretical prediction (`no_say` is slightly higher for the fully unemployed than for the
  employed, but is actually lower for those not in the labor force). See
  `results/crosstab_simple_summary.txt` for details.

## Statistical limitations

1. **Endogeneity**: reverse causation between media use / political interest and political trust
   is not identified.
2. **Omitted-variable bias**: important controls such as income, place of residence, and party
   support are not included.
3. **Sample size of the Wave-7-only SNS analysis**: at $N \approx 1{,}170$, this is relatively
   small, and since it is cross-sectional data from a single time point (2019), it cannot test
   whether the effect of SNS use has changed over time.

Even accounting for these limitations, this thesis treats the finding that "macro survey data
still cannot adequately explain the sources of political alienation, even after directly entering
a machine-learning method and an individual-level SNS-use variable" as grounds for a supplementary,
micro-level verification using discourse data (Analysis 2).

## Future work

- Add income, party support, and other controls.
- Extend to panel/multilevel analysis using the full set of WVS Time Series waves.
- Clean up the legacy R scripts, which still depend on absolute paths via `setwd()` (low priority
  since they are no longer used).
