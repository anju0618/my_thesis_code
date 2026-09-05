# Analysis 2: LLM-Based Structured Coding of YouTube Comments (for Part 4)

*[日本語](README.md)*

A data collection → structured coding → analysis pipeline for "Analysis 2" (the micro,
discourse-level analysis) in Part 4 ("Empirical Analysis") of the thesis. The goal is to test and
extend, against real Japanese-language comment data, the concepts introduced in the theoretical
chapters (Parts 1–3) of the thesis (see the session history and the literature notes under
`notes/` in the main thesis repository for the detailed background).

The other analysis, "Analysis 1" (the macro, survey-data-level analysis using WVS/ISSP), lives in
`../survey_regression/`. See `../README.en.md` and `sections/04_empirical.tex` (in the main thesis
repository) for how the two are positioned relative to each other.

## Research design summary

- **Target topics** (`config/topics.yaml`): the initial set of three topics — the Ministry of
  Finance protests, Sanseito (a Japanese political party), and the undocumented-immigration
  debate — was chosen because each is expected to make it easy to observe the concepts from the
  theory chapters ("people vs. elite," "economic resentment," "theft of enjoyment"). On 2026-08-28,
  five more topics were added, expanding the set to eight (roughly 96,000 comments collected in
  total, of which 6,000 per topic — 48,000 in total — have been LLM-coded):
  - `ice_age_precarity` (the "employment ice age" generation / precarious non-regular employment)
    and `income_wall` (the "income wall" tax/benefit threshold) — intended to capture discourse in
    which the people Ricci would call "subjects of economic destruction" speak in their own voice.
  - `takaichi_admin` (the Takaichi administration / criticism of the LDP) — intended to capture
    the most recent political developments.
  - `ishin_osaka` (Osaka Ishin no Kai) and `reiwa_shinsengumi` (Reiwa Shinsengumi) — intended to
    test Mudde & Rovira Kaltwasser's (2017) thesis that "populism is a thin-centered ideology that
    can attach to either the left or the right," via a direct comparison of right-populist
    discourse (Sanseito) and left-populist discourse (Reiwa).
- **Data source**: YouTube comments (YouTube Data API v3, free and official). X (Twitter) was
  excluded due to API restrictions, and Yahoo! News was excluded because it has no official API;
  neither offers a practically feasible collection method (see also the "Caveats and limitations"
  section of `notes/fujishiro2020.md` in the main thesis repository — the benchmark paper,
  Fujishiro et al. (2020), also relies solely on Twitter, and CrowdTangle, the tool it used, was
  discontinued in 2024).
- **Coding scheme** (`config/coding_scheme.yaml`): operationalizes each theoretical concept as a
  comment-level variable. Each variable records which paper and page(s) it is based on, so this
  can be reused directly in the methodology section of Part 4 of the thesis.
  - `people_vs_elite` — Mudde & Rovira Kaltwasser (2017), pp. 5–6
  - `economic_resentment` — Ricci (2020), pp. viii–ix, p. 52
  - `theft_of_enjoyment` — Kawamura & Iwabuchi (2022)
  - `blame_target` — Fujishiro et al. (2020), p. 315 (testing the "liberal vs. xenophobic" axis)
  - `emotional_tone` — the "narrative of anger" in Törnberg / Bennett & Livingston (2020)

## Setup

Environment and dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
cd analysis/llm_discourse
uv sync                 # creates .venv and installs dependencies from pyproject.toml / uv.lock
cp .env.example .env    # then edit .env to set your API key etc.
```

All commands below are run as `uv run <command>` (this automatically runs inside the `.venv`
without needing to activate it). To add a new package, use `uv add <package>` (do not use
`pip install`, which would desynchronize `pyproject.toml`/`uv.lock`).

### Moving between Windows and Ubuntu (e.g., a 42 campus machine)

Since everything invoked via `uv run ...` is OS-independent, nothing in this repository — including
the `Makefile` described below — needs to change between Windows and Ubuntu. Only two things are
needed each time you switch machines:

1. **Install `uv` itself** (there is a one-line installer for both OSes; see
   https://docs.astral.sh/uv/getting-started/installation/). The `.venv` directory itself cannot be
   shared across OSes (its binaries are OS-specific), so just re-run `uv sync` on the new machine
   (since `uv.lock` is committed, the exact same dependency versions will be reproduced every time).
2. **Install `make`**
   - Ubuntu (e.g., 42): usually preinstalled; otherwise `sudo apt install make`.
   - Windows: if not installed, run `winget install ezwinports.make` (this required an interactive
     Microsoft Store consent prompt that could not be automated in one particular session, so run it
     yourself in a terminal). If Git Bash is installed, `make` will use its `sh`, so UNIX commands
     such as `rm`/`find` work as-is.

`.env` (which holds API keys) is excluded via `.gitignore`, so it needs to be recreated from
`.env.example` whenever you switch machines.

## Makefile (an alternative to typing `uv run ...` every time)

Everything in the "Pipeline execution order" section below can also be run via `make <target>`
(e.g., `make collect TOPIC=sanseito`, `make label-test LIMIT=50`). See `make help` for the full
list of targets. If `make` is not available, just use the raw `uv run` commands in the section
below directly — both always behave identically.

### Getting a YouTube Data API v3 key

See the comments inside `.env.example`. Issue a free API key via the Google Cloud Console. The
daily quota is 10,000 units; `search.list` costs 100 units per call and `commentThreads.list`
costs 1 unit per call. Heavy use of video search burns through the quota quickly, so it is
recommended to start with a small number of queries, and once good videos are found, add them
directly to `video_ids` in `topics.yaml`.

### Setting up Ollama (local LLM)

1. Install from https://ollama.com/download
2. Pull a model that can handle Japanese: `ollama pull qwen2.5:7b-instruct`
   (choose a 7b/14b/32b model or a quantized variant depending on your machine's VRAM/memory)
3. Set `OLLAMA_MODEL` in `.env` to match the model you are using

## Pipeline execution order

```bash
# 1. Collect comments from YouTube (a dry run to check the video list first is recommended)
uv run src/youtube_collect.py --topic sanseito --dry-run
uv run src/youtube_collect.py --topic sanseito

# 2. Preprocessing (merges all topics into a single CSV)
uv run src/preprocess.py

# 3. Structured coding via LLM (try a small number first)
uv run src/label_ollama.py --limit 50
uv run src/label_ollama.py   # run on all cases once things look right

# 4. (optional) generate embedding vectors — clustering etc. lives under notebooks/
uv run src/embed.py

# to use notebooks/
uv run jupyter lab
```

### Additional analysis scripts (added 2026-08-28 to 31, directly under `src/`)

- `analyze_coding.py`: computes prevalence rates, chi-square tests, and embedding-based clustering
  for the three confirmed topics — the Ministry of Finance protests, Sanseito, and undocumented
  immigration (the version referenced by the thesis body, `sections/04_empirical.tex`).
  **Left unchanged.**
- `analyze_coding_full.py`: the same analysis extended to all eight topics (about 46,000 cases). A
  separate script that does not affect the results of the existing three-topic version. Adds a
  direct comparison between Sanseito and Reiwa Shinsengumi (§1b).
- `dict_sentiment.py`: an independent check using dictionary-based sentiment analysis that does not
  rely on an LLM at all (a lightweight implementation, via `fugashi`, of the Japanese polarity
  dictionary from Tohoku University's Inui–Okazaki Lab). This serves as a triangulating robustness
  check in response to the near-zero inter-rater kappa found for `economic_resentment` and
  `theft_of_enjoyment` in manual verification.
- `descriptive_stats.py`: variable-prevalence trends over time by posting date, and
  topic-by-variable cross-tabulations with chi-square tests — none of it involving machine
  learning.
- `topic_model_lda.py`: a classical topic-model analysis via LDA (Latent Dirichlet Allocation),
  which is easier to interpret than embedding-based clustering.
- `eight_topic_comparison.py`: runs the three analyses that do not require LLM coding (dictionary-
  based sentiment, posting-volume trends, LDA) on the raw eight-topic corpus (about 96,000 cases).
  Can be run even before LLM coding is complete.

## Directory layout

```
analysis/llm_discourse/
  pyproject.toml / uv.lock  # dependency definitions managed by uv
  Makefile                    # shortcuts for uv run (same on Windows/Ubuntu)
  config/
    topics.yaml          # topics to collect, search queries, video IDs
    coding_scheme.yaml    # variable definitions for LLM coding (mapped to the theory chapters)
  src/
    youtube_collect.py    # collects comments via the YouTube API
    preprocess.py          # cleaning and merging
    label_ollama.py         # structured coding via a local LLM
    embed.py                 # generates sentence-embedding vectors
    analyze_coding.py        # aggregation, tests, clustering (three-topic confirmed version)
    analyze_coding_full.py   # same, for the extended eight-topic version
    dict_sentiment.py        # dictionary-based sentiment analysis (independent, LLM-free check)
    descriptive_stats.py     # descriptive statistics with no machine learning (time series, cross-tabs)
    topic_model_lda.py       # LDA topic model (three-topic version)
    eight_topic_comparison.py # runs the three LLM-free analyses on the full eight-topic raw corpus
  data/
    raw/{topic}/{video_id}.json    # raw data (one file per video)
    processed/comments.csv          # merged table after preprocessing
    processed/labeled_comments.jsonl # LLM coding results
    processed/embeddings.npy         # embedding vectors
  notebooks/               # exploratory analysis and visualization (clustering, aggregation, figures)
  results/                  # final versions of figures/tables reproduced in the thesis
```

Everything under `data/` is excluded from the repository via `.gitignore` (although YouTube
comments are data publicly posted by individuals, it is preferable not to keep the raw data —
which includes things like commenter display names — directly in the repository).

## Main results (additional analyses, 2026-08-28 to 31)

### Prevalence across all eight topics (`analyze_coding_full.py`, N=45,856)

`people_vs_elite` is highest by far for the Ministry of Finance protests (28.3%), with the other
topics ranging from 5.5% (undocumented immigration) to 11.8% (the Takaichi administration).
`economic_resentment` was around 2–4% for the original three topics, but **rose clearly for the
topics that are directly about economic hardship — the ice-age generation (5.8%) and the income
wall (5.4%)** — suggesting that looking only at overtly political topics may have understated the
signal this coding scheme can pick up. The pattern of `theft_of_enjoyment` being strongly
concentrated in the undocumented-immigration topic (6.8%) remains unchanged.

### Sanseito vs. Reiwa Shinsengumi: a direct comparison of right- and left-wing populism

The prevalence of `people_vs_elite` (the people-vs-elite framing) is **not statistically
different** between Sanseito (10.4%) and Reiwa (10.7%) ($\chi^2=0.19$, $p=0.660$) — a result
consistent with Mudde & Rovira Kaltwasser's (2017) thesis that "the core frame of populism is a
thin-centered ideology that can attach to either the left or the right." By contrast,
`economic_resentment` (Reiwa 2.0% vs. Sanseito 0.9%, $p<0.001$) and `theft_of_enjoyment` (Sanseito
0.8% vs. Reiwa 0.1%, $p<0.001$) are clearly asymmetric — while the core frame is ideologically
neutral, the **content** of the accompanying grievance (xenophobic victimhood vs. dissatisfaction
oriented toward economic redistribution) diverges between left and right in a way that is
theoretically coherent.

### Distribution of `blame_target` (all eight topics, chi-square test)

$\chi^2=10267.76$, df=63, $p<0.001$, Cramér's V=0.29. The Ministry of Finance protests are
strongly associated with "elites in general" (standardized residual +43.1); undocumented
immigration with "immigrants" (+66.9) and "foreign countries" (+36.6); and Sanseito, Reiwa, and
Ishin all show a spike in "a specific party" (presumably referring to their own party). **The
ice-age-generation topic reacts strongly to "elites in general" (+16.4) and "government" (+8.6) but
does not react to "a specific party"** (−14.0) — a pattern suggesting dissatisfaction with the
system or society as a whole, rather than with any particular party.

### Dictionary-based sentiment analysis (an independent, LLM-free check, `dict_sentiment.py`)

The correspondence with `emotional_tone` is clear (the dictionary score is lower for
anger/contempt/fear, $p<0.001$). However, the difference in dictionary score between comments coded
as `economic_resentment` and those not is non-significant ($p=0.064$) — consistent with the
near-zero inter-rater kappa found in manual verification ($\kappa=-0.033$), and suggestive, from a
different angle, of the sizable measurement error in this variable. `theft_of_enjoyment` does show
a significant difference ($p<0.001$). Comparing dictionary scores across all eight topics, Ishin
(+0.140) and Reiwa (+0.142) come out at almost the same level, while undocumented immigration is
the only topic with a negative score (−0.030) — revealing a methodological limitation: this naive
dictionary method has almost no power to distinguish left from right ideologically, since
vocabulary expressing support/endorsement pushes the score up by a similar amount for both camps.

### Limits of embedding-based clustering when pooling all eight topics

The silhouette score was 0.255 (at K=3) for the three-topic version, but dropped to at most 0.045
(at K=2) once expanded to eight topics. When a variety of political topics are mixed into a single
embedding space, semantic separation via $K$-means barely functions at all — a limitation showing
that sentence-embedding clustering is well suited to capturing fine-grained narrative patterns
within a topic, but poorly suited to discovering global structure that cuts across topics.

## Future work (not yet started)

- Curate `video_ids` in `topics.yaml` by actually watching candidate videos and selecting them.
- Manually check a small sample of `label_ollama.py`'s output (a lightweight check of inter-rater
  reliability) and adjust the prompt / variable definitions as needed.
- In `notebooks/`, perform per-topic / per-variable aggregation, examine the distribution of
  `blame_target` (testing Fujishiro et al.'s claim that the divide is "xenophobic vs. liberal"
  rather than "conservative vs. liberal"), and run embedding-based clustering.
- Reflect the methodology and results developed here in Part 4 of the thesis
  (`sections/04_empirical.tex`).
