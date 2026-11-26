#!/bin/bash
# StackExchange human sample + KenLM trigram (the coding n-gram reference). Run from Git Bash at the repo root.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
# the same folder as WSL sees it (KenLM and Torchcast run inside WSL Ubuntu)
C=$(MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- wslpath -a "$(cd clark2026 && pwd -W)")
[ -f clark2026/data/stackexchange/sample.jsonl ] || uv run python clark2026/se_tools.py sample
uv run python clark2026/se_tools.py tokens | MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash -c "mkdir -p /root/lm /root/lmtmp && cd /root/lm && /opt/kenlm/build/bin/lmplz -o 3 -S 60% -T /root/lmtmp > se3.arpa 2> se_lmplz.log && /opt/kenlm/build/bin/build_binary -q 8 trie se3.arpa se3.binary >> se_lmplz.log 2>&1 && rm se3.arpa && ls -la /root/lm"
echo SE_LM_DONE
