# Clark, Martin & Mahowald (2026): *Why is AI-generated language so hard to understand?*, replicated

Paper: arXiv [2610.29162](https://arxiv.org/abs/2610.29162). Report: https://enyman.github.io/llm-writing-replications/clark2026/

The paper measures eight features linked to reading difficulty in model text and compares them with human writing in
three settings. Its data was not released, so everything here is rebuilt from the appendix; the report's method notes
list every choice we had to make.

| Setting | Models write | Human corpus | n-gram reference |
|---|---|---|---|
| Essays | 10 essays × 7 PERSUADE 2.0 assignments, the paper's system prompt | 2,934 top-scored student essays (PERSUADE 2.0) | KenLM trigram on all of English Wikipedia (2.88 B words) |
| Chat | 10 twelve-turn conversations with a simulated user (Claude Sonnet 4.6) | 500k-word OpenSubtitles sample | OpenSubtitles v2018 minus the sample (2.43 B tokens) |
| Coding | Up to 5 Pi coding-agent sessions on small Python repos (Docker), 11 user turns each | 500k-word StackExchange sample | StackExchange subset minus the sample (484 M tokens) |

Measures (`features.py`, `jev.py`, `ngrams.py`): object relatives with and without a relativizer, stacked nouns,
dependency length (standardized to Wikipedia's sentence-length mix), denominal verbs, inanimate subjects with animate
verbs, never-seen word pairs, and trigram perplexity.

**Results in short.** Our models land on the same side of the human corpus as the paper's in 26 of 32 essay checks,
12 of 16 chat checks and 10 of 12 coding checks (`report/agreement.json`).

## Requirements

- `uv sync --extra gpu` at the repo root (spaCy `en_core_web_trf` on a CUDA GPU; CPU works but is much slower).
- **WSL Ubuntu** (22.04 used) for two Linux-only tools, called from the Git Bash scripts:
  - **KenLM** built at `/opt/kenlm` (`apt install build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev`,
    clone https://github.com/kpu/kenlm, `cmake` + `make`) plus its Python module for the system `python3`
    (`pip install https://github.com/kpu/kenlm/archive/master.zip`). Models go to `/root/lm/`.
  - **Torchcast Decision 4B** (the yes/no judge for the two classifier-based measures) in a venv at `/opt/tcvenv`
    with Python 3.10, `torch` (cu128), `transformers` 5.20, `fla-core` 0.6 and `accelerate`. The model's bundled
    `torchcast_decision` package is used directly (`decide_local.py`). `build_causal_conv.sh` builds the optional
    `causal-conv1d` kernel; without it the pipelines set `TC_ALLOW_SLOW=1`. Keep `TC_BATCH_TOKENS=8192`: larger
    batches fill a 24 GB GPU and get slower.
- **Docker** for the coding setting: `docker build -t clark-pi:0.84.4 clark2026/code` (the Pi agent, pinned).
- Disk: ~10 GB model, ~45 GB corpora and token files, KenLM binaries 0.5-5 GB each.

## Steps (from the repo root, in Git Bash)

1. **Data:** `uv run python clark2026/get_data.py all` (or name the steps: `persuade wikipedia opensubtitles stackexchange torchcast`).
2. **Model outputs:** `uv run python scripts/fetch_outputs.py` for ours, or generate new ones (API money; keys in `.env`):
   - essays: `uv run python clark2026/gen_essays.py 10 openai/gpt-5.6-luna z-ai/glm-5 moonshotai/kimi-k2.5 meta-llama/llama-3.1-8b-instruct`
   - chat: `uv run python clark2026/gen_chat.py 10 nanogpt:openai/gpt-5.6-luna ...`
   - coding: `SIM_VIA=openrouter uv run python clark2026/gen_code.py 5 openai/gpt-5.6-luna z-ai/glm-5 moonshotai/kimi-k2.5`
     (watch for agents stuck in edit loops: `clark2026/loop_watchdog.sh` stops those containers).
   - Budgets count what is already in `out/`: essays and chat share `BUDGET` (`gen_essays.py`, $10), coding has
     `CODE_BUDGET`. With the released outputs present, raise them (or generate into an empty `out/`).
3. **n-gram references** (one-off, hours, WSL): `bash clark2026/build_wiki_lm.sh`, the OpenSubtitles step inside
   `run_chat_pipeline.sh`, and `bash clark2026/build_se_lm.sh`.
4. **Pipelines** (parse with spaCy, judge with Torchcast, score n-grams, compare with the paper):
   `bash clark2026/run_pipeline.sh` (essays), `bash clark2026/run_chat_pipeline.sh`, `bash clark2026/run_code_pipeline.sh`.
   Each writes `out/comparison_<setting>.csv`.
5. **Report data:** `uv run python clark2026/build_report.py` → `report/values.json`, `report/agreement.json`;
   then `uv run python site/build.py`.

Steps 4-5 alone reproduce the published numbers from the released outputs; parsing and Torchcast together take a few GPU hours.

## Details that matter

- **Paper values** (`paper_values.csv`) were read off the paper's vector figure by `extract_figure.py`
  (needs the arXiv source's `combined_compact.pdf`).
- **Classifier thresholds.** The paper used Jev 1.13 with unpublished questions. We ask Torchcast our own
  definitional questions (`questions.json`, version `v3`) and set each acceptance threshold so the student essays
  match the paper's human rate (denominal 0.346, inanimate 0.712); the same thresholds are used in chat and coding.
- **Held-out references.** Each human sample is excluded from its own n-gram reference (`subs_tools.py heldout`,
  `se_tools.py tokens`); scoring a sample against a reference that contains it makes it look far too predictable.
- **Coding sessions.** One Pi process per user turn, resuming one session file; a turn's reply is the last message with
  no tool call after it. Turns without a reply get "ok, keep going" instead of a simulated-user message. Provider
  errors back off and retry from the session state before the turn. Sessions that loop or fail are discarded
  (`out/_discarded/` in the release); the report lists them.
- **Routes.** `gen_code.py` sends GLM 5, Qwen 3.8 27B and Mistral through OpenRouter (GLM 5 through NanoGPT wrote its
  tool calls as plain text), Hemmingway-1 through local Ollama (`hemmingway-1-96k`: `num_ctx 98304`, with
  `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`), everything else through NanoGPT.
- **Settings switch:** scripts shared between settings read `CLARK_SETTING=essay|chat|coding`.
