#!/bin/bash
# Coding-setting processing once the Pi sessions, the StackExchange sample and se3.binary are in. Run from Git Bash.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
# the same folder as WSL sees it (KenLM and Torchcast run inside WSL Ubuntu)
C=$(MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- wslpath -a "$(cd clark2026 && pwd -W)")
log() { echo "[$(date +%H:%M:%S)] $*"; }
export CLARK_SETTING=coding
log "parse coding replies + StackExchange sample"
uv run python -u clark2026/parse_chat.py 2>&1 | grep -v -iE "warn|dlpack" || true
log "candidates + Torchcast v3"
uv run python clark2026/jev.py export
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash -c "cd $C && TC_ALLOW_SLOW=1 USE_HUB_KERNELS=0 TC_BATCH_TOKENS=8192 /opt/tcvenv/bin/python decide_local.py models/torchcast-decision-4b out/candidates_code.jsonl questions.json v3 out/tc_code_v3.jsonl 2>&1 | grep -E 'to judge|done'"
uv run python clark2026/jev.py calibrated clark2026/out/tc_code_v3.jsonl clark2026/out/jev_rates_code.csv clark2026/out/tc_full_v3.jsonl 2>&1 | grep -v -i warn
log "n-grams against StackExchange"
uv run python clark2026/ngram_run.py export_code
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash -c "cd $C && python3 kenlm_score.py /root/lm/se3.binary out/ngram_inputs_code out/ngram_rates_code.csv 2>&1 | tail -1"
log "compare"
uv run python clark2026/compare.py
log "CODE_PIPELINE_DONE"
