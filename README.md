# LLM writing replications

Two replications of recent studies on how AI-written text differs from human writing, each extended to current models.
The reports are on the project site: **https://enyman.github.io/llm-writing-replications/**

| Folder | Paper | What we did |
|---|---|---|
| [`reinhart2025/`](reinhart2025/) | Reinhart et al. (2025), *Do LLMs write like humans? Variation in grammatical and rhetorical styles*, arXiv [2410.16107](https://arxiv.org/abs/2410.16107) | Reproduced the paper's Biber-feature effect sizes on the HAP-E corpus (r = 0.99), then ran the same continuation task with 25 current models, Claude through Claude Code, and "human-sounding" finetunes. |
| [`clark2026/`](clark2026/) | Clark, Martin & Mahowald (2026), *Why is AI-generated language so hard to understand?*, arXiv [2610.29162](https://arxiv.org/abs/2610.29162) | Rebuilt the eight difficulty measures and all three settings (essays, chat, agentic coding) from the appendix; our models land on the same side of the human corpus as the paper's in 48 of 60 checks. |

These are independent replications, not affiliated with the papers' authors. The papers themselves are not included; read them on arXiv.

> **Made with AI assistance.** This project was built with the help of AI: the code, the analyses, the report pages and
> this documentation were written largely by Claude Code (Anthropic), working under the direction and review of the
> repository's author, who chose the studies, models, budgets and methodological decisions. AI-written code and text can
> contain mistakes; the checks in [`AGENTS.md`](AGENTS.md) and the released outputs are there so anyone can verify the results.

## What is in the repository

- **Code** for every step: generation, feature extraction, statistics, report data, and the site (`site/`, built into `docs/`).
- **Result tables** behind both reports (`reinhart2025/results/`, `reinhart2025/report/`, `clark2026/out/*.csv`, `clark2026/report/`).
- **Model outputs** (all continuations, essays, chats and coding-agent transcripts) are attached to the
  [GitHub release](https://github.com/EnyMan/llm-writing-replications/releases), not stored in git:
  `uv run python scripts/fetch_outputs.py` puts them where the scripts expect them.
- **Third-party corpora** are not redistributed; each project's `get_data.py` downloads them from their sources.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+. Everything was run on Windows 11 with Git Bash and an
RTX 4090; parts of the Clark pipeline run in WSL Ubuntu (see [`clark2026/README.md`](clark2026/README.md)).

```bash
uv sync --extra gpu                      # or plain `uv sync` without a CUDA GPU (slower spaCy parsing)
uv run python scripts/fetch_outputs.py   # model outputs from the release (no API spend needed)
cp .env.example .env                     # only if you want to generate new outputs: add API keys
```

Then follow the README of the project you are interested in. Each lists its steps in order, which scripts are expensive
(API money, GPU hours, disk), and how to rebuild the report data and the site.

**Using a coding agent?** Point it at [`AGENTS.md`](AGENTS.md): it describes the layout, the order of the
pipelines, what costs money, and the checks that tell you a rerun matches the published numbers.

## Costs of the original runs

| | API spend | Other |
|---|---|---|
| Reinhart replication + extension | ≈ $15.70 (OpenRouter, NanoGPT) | Claude via a Claude Code subscription; Hemmingway-1 locally (Ollama) |
| Clark replication: essays / chat / coding | ≈ $0.70 / $6.80 / $19.10 | ~10 GB Torchcast model and ~45 GB of corpora; several GPU hours (spaCy, Torchcast, local Hemmingway-1) |

Rerunning the analyses from the released outputs costs nothing in API credits.

## Licences

- Code: [MIT](LICENSE).
- Generated outputs, result tables, report text and figures: [CC BY 4.0](LICENSE-DATA.md).
- Third-party data keeps its own licence: HAP-E (MIT), PERSUADE 2.0 (CC BY 4.0), StackExchange (CC BY-SA),
  Wikipedia (CC BY-SA), OpenSubtitles via OPUS (see opus.nlpl.eu). Model outputs are also subject to the respective providers' terms.

## Citing

If you use this work, cite the original papers, and link to this repository for the replication itself.
