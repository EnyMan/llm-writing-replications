# Reinhart et al. (2025): *Do LLMs write like humans?*, replicated and extended

Paper: Reinhart, Markey, Laudenbach, Pantusen, Yurko, Weinberg & Brown, arXiv [2410.16107](https://arxiv.org/abs/2410.16107).
Report: https://enyman.github.io/llm-writing-replications/reinhart2025/

**Task.** A model gets the first chunk of a human text with the paper's prompt ("In the same style, tone, and diction of
the following text, complete the next 500 words, …") and writes a continuation. Its 67 Biber features (pybiber) are
compared with the human second chunk of the same text: random-forest classification, paired Wilcoxon + Bonferroni,
and paired Cohen's d per feature, as in the paper. We use the 900 doc IDs of HAP-E mini (150 per genre, six genres).

**Results in short.** The paper's six models reproduce closely (r = 0.992 between our and the paper's d over 396
feature × model cells). Every 2026 model is still easy to tell from human writing (pairwise classifier accuracy
≥ 89%), but all sit closer to the human text than GPT-4o did (mean |d| 0.12 to 0.38, GPT-4o 0.41).

## Steps (run from this folder: `cd reinhart2025`)

| # | Command | Cost | Output |
|---|---|---|---|
| 1 | `uv run python get_data.py` | ~150 MB download | `data/` (HAP-E text, mini, Biber), `data/mini_parallel.parquet`, `claude_kit/prompts.jsonl` |
| 2 | `uv run python validate_biber.py 120` | minutes | checks that pybiber matches the published HAP-E features (median r ≈ 0.99) |
| 3 | `uv run python generate.py` *(or fetch the released outputs)* | API money: ≈ $15 for 3 + 3b together | `out/<model>.jsonl` |
| 3b | `uv run python generate.py 150 $(cat models_extra.txt)` | NanoGPT + OpenRouter; Hemmingway-1 needs Ollama | finetunes and extra models |
| 3c | `claude_kit/` (see its README) | Claude subscription | `out/claude-*@{minimal,default}.jsonl` |
| 4 | `uv run python tag_all.py` | the slow step (spaCy on every text; cached per source) | `data/features.parquet`, `results/cleaning.csv` |
| 5 | `uv run python analysis.py` then `uv run python analysis.py all` | minutes | `results/*_paper.csv`, `results/*_all.csv` |
| 6 | `uv run python compare_paper.py` | seconds | `results/replication_vs_paper.csv` (vs `paper_table_s6.csv`) |
| 7 | `uv run python compare_conditions.py`, `uv run python vocab.py`, `uv run python calibrate.py` | minutes | condition pairs, vocabulary, prompt calibration |
| 8 | `uv run python build_report_data.py` | seconds | `report/*.json` (the report's data) |
| 9 | `cd .. && uv run python site/build.py` | seconds | `docs/` |

The released outputs include `data/features.parquet`, so steps 1-4 can be skipped when you only want to rerun the
statistics: fetch the outputs, then start at step 5.

## Details that matter

- **Generation settings.** Reasoning is off where a model allows it; models that cannot switch it off get minimal effort
  (`REASONING` in `generate.py`). Reasoning models given "generate exactly 500 words" recount words until the token limit,
  so two models (`NOCOUNT_MODELS`) get one extra sentence telling them not to count; `calibrate.py` shows the sentence
  does not shift style (0 of 67 features significant on two other models). Their files carry `@nocount`.
- **Filtering** as in the paper's SI: responses under 100 words are dropped, plus refusals and API errors; a trailing
  note about the word count is stripped (`tag_all.py`, counts in `results/cleaning.csv`).
- **Tagger.** pybiber 0.3.1 with spaCy `en_core_web_sm`, `biber(tokens, force_ttr=True, strict_be_main_verb=False)`,
  which matches the published features; it needs `polars<2`.
- **Model names.** `nanogpt:<id>` goes through NanoGPT, `ollama:<name>` through a local Ollama server, anything else
  through OpenRouter (`../llm.py`). `google/gemini-3.8-flash:floor` is OpenRouter's cheapest-provider routing.
- **Claude.** The Claude rows come from Claude Code's headless mode (`claude -p`) under two conditions: a minimal
  one-line system prompt and Claude Code's default prompt (`claude_kit/README.md`).
- `out/_discarded/` holds pilot and superseded runs; they count toward the budget but are not analysed.

## Files

`generate.py` generation · `build_mini.py` working set · `features.py` Biber tagging · `tag_all.py` tag every source ·
`analysis.py` classifiers and effect sizes · `compare_paper.py` our d vs the paper's Table S6 · `compare_conditions.py`
paired comparisons (Claude conditions, finetunes vs parents) · `vocab.py` over- and under-used words ·
`build_report_data.py` report data · `paper_table_s6.csv` the paper's Table S6 effect sizes, transcribed.
