#!/bin/bash
# Essay processing after new essays arrive. Run from the repo root in Git Bash.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
# the same folder as WSL sees it (KenLM and Torchcast run inside WSL Ubuntu)
C=$(MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- wslpath -a "$(cd clark2026 && pwd -W)")
log() { echo "[$(date +%H:%M:%S)] $*"; }
rm -f clark2026/parsed/wikipedia.parquet            # re-parse Wikipedia from the pinned shard
log "parse"; uv run python -u clark2026/parse_essays.py 2>&1 | grep -v -iE "warn|dlpack"
log "export candidates"; uv run python clark2026/jev.py export
log "torchcast v3 (new candidates only)"
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash -c "cd $C && TC_ALLOW_SLOW=1 USE_HUB_KERNELS=0 TC_BATCH_TOKENS=8192 /opt/tcvenv/bin/python decide_local.py models/torchcast-decision-4b out/candidates.jsonl questions.json v3 out/tc_full_v3.jsonl 2>&1 | grep -E 'to judge|done'"
log "calibrated rates"; uv run python clark2026/jev.py calibrated clark2026/out/tc_full_v3.jsonl clark2026/out/jev_rates.csv 2>&1 | grep -v -i warn
log "n-grams"; uv run python clark2026/ngram_run.py export 2>&1 | grep -v -i warn
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash -c "cd $C && python3 kenlm_score.py /root/lm/wiki3.binary out/ngram_inputs out/ngram_rates.csv 2>&1 | tail -1"
log "compare + report data"; uv run python clark2026/compare.py > clark2026/out/compare.txt 2>&1; uv run python clark2026/build_report.py
log "PIPELINE_DONE"
