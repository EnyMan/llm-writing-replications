# Guide for coding agents

You are helping someone rerun or extend two replications of studies on AI-written text. Read `README.md` first, then the
README of the project you work on (`reinhart2025/README.md`, `clark2026/README.md`). This file adds what an agent needs.

## Ground rules

- **Ask before spending money.** Scripts that call model APIs (`reinhart2025/generate.py`, `clark2026/gen_essays.py`,
  `gen_chat.py`, `gen_code.py`, `jev.py run`) spend the user's OpenRouter/NanoGPT credit. Each has a budget constant
  (`BUDGET`, `CODE_BUDGET`) that stops it; confirm the amount with the user before raising it or starting a run.
- **Prefer the released outputs.** `uv run python scripts/fetch_outputs.py` restores every model output we analysed, so the
  statistics can be rerun without generating anything.
- **Never commit** `.env`, `data/`, `models/`, parsed corpora (`clark2026/parsed*/`), or Torchcast/n-gram intermediates:
  they hold API keys or third-party text. `.gitignore` covers them; keep it that way.
- Long jobs (parsing, Torchcast, KenLM builds, coding sessions) run for hours: start them in the background and write
  logs straight to a file. A `grep` between the job and its log buffers output and makes the job look stuck.

## Layout

```
llm.py                  shared model client (OpenRouter / NanoGPT / Ollama), reads .env at the repo root
reinhart2025/           Reinhart et al. (2025); scripts run from inside this folder (paths are relative to it)
clark2026/              Clark et al. (2026); scripts run from the repo root (paths are relative to clark2026/)
site/                   landing page, shared CSS and the small `dash` runtime; site/build.py writes docs/
docs/                   the built GitHub Pages site (do not edit by hand; rebuild it)
scripts/                fetch_outputs.py (restore released outputs), pack_outputs.py (make the release zips)
```

## Checks that a rerun matches the published results

| Check | Command | Expected |
|---|---|---|
| Reinhart effect sizes vs the paper | `cd reinhart2025 && uv run python compare_paper.py` | `396 feature x model cells; overall r = 0.992` |
| Reinhart report data | `cd reinhart2025 && uv run python build_report_data.py` then `git diff --stat report/` | no changes |
| Clark report data | `uv run python clark2026/build_report.py` then `git diff --stat clark2026/report/` | no changes |
| Clark agreement with the paper | read `clark2026/report/agreement.json` | same-side 26/32 essays, 12/16 chat, 10/12 coding |
| PERSUADE baseline | `uv run python clark2026/get_data.py persuade` | asserts exactly 2,934 essays |

## Platform notes

- Developed on Windows 11 + Git Bash. The Clark shell scripts call WSL Ubuntu (`wsl -d Ubuntu -u root`) for KenLM and
  Torchcast and translate paths with `wslpath`; prefix `wsl` calls with `MSYS_NO_PATHCONV=1` in Git Bash. On Linux,
  run the commands inside those `wsl ... bash -c "..."` strings directly.
- spaCy's transformer parser uses the GPU through CuPy (`uv sync --extra gpu`, CUDA 12).
- The coding setting needs Docker and the `clark-pi:0.84.4` image (`clark2026/code/Dockerfile`).

## Extending to new models

1. Generate with the project's generation script (model ids: plain = OpenRouter, `nanogpt:<id>`, `ollama:<name>`).
2. Rerun the project's pipeline: new sources are picked up incrementally, already-processed ones are cached.
3. Add a display label (`LABELS` in `clark2026/build_report.py`, or the label map in `reinhart2025/build_report_data.py`),
   rebuild the report data, run `site/build.py`, and update any sentence in `report/index.html` that states a count.
