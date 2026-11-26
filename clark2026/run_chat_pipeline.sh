#!/bin/bash
# Chat-setting processing once the conversations and OpenSubtitles are in. Run from Git Bash.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
# the same folder as WSL sees it (KenLM and Torchcast run inside WSL Ubuntu)
C=$(MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- wslpath -a "$(cd clark2026 && pwd -W)")
log() { echo "[$(date +%H:%M:%S)] $*"; }
W="MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root --"
SUBS=$C/data/opensubtitles

if [ ! -f clark2026/data/opensubtitles/sample.txt ]; then
  log "OpenSubtitles: 500k-word human sample"
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- python3 $C/subs_tools.py sample $SUBS/en.txt.gz $SUBS/sample.txt
fi
if ! MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- test -f /root/lm/subs3.binary; then
  log "OpenSubtitles: KenLM trigram without the human sample (the chat n-gram reference)"
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash -c "mkdir -p /root/lm /root/lmtmp && cd /root/lm && python3 $C/subs_tools.py heldout $SUBS/en.txt.gz | /opt/kenlm/build/bin/lmplz -o 3 -S 60% -T /root/lmtmp > subs3.arpa 2> subs_lmplz.log && /opt/kenlm/build/bin/build_binary -q 8 trie subs3.arpa subs3.binary >> subs_lmplz.log 2>&1 && rm subs3.arpa && ls -la /root/lm"
fi
log "parse chat replies + OpenSubtitles sample"
uv run python -u clark2026/parse_chat.py 2>&1 | grep -v -iE "warn|dlpack" || true
log "candidates + Torchcast v3"
CLARK_SETTING=chat uv run python clark2026/jev.py export
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash -c "cd $C && TC_ALLOW_SLOW=1 USE_HUB_KERNELS=0 TC_BATCH_TOKENS=8192 /opt/tcvenv/bin/python decide_local.py models/torchcast-decision-4b out/candidates_chat.jsonl questions.json v3 out/tc_chat_v3.jsonl 2>&1 | grep -E 'to judge|done'"
CLARK_SETTING=chat uv run python clark2026/jev.py calibrated clark2026/out/tc_chat_v3.jsonl clark2026/out/jev_rates_chat.csv clark2026/out/tc_full_v3.jsonl 2>&1 | grep -v -i warn
log "n-grams against OpenSubtitles"
uv run python clark2026/ngram_run.py export_chat
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash -c "cd $C && python3 kenlm_score.py /root/lm/subs3.binary out/ngram_inputs_chat out/ngram_rates_chat.csv 2>&1 | tail -1"
log "compare"
CLARK_SETTING=chat uv run python clark2026/compare.py
log "CHAT_PIPELINE_DONE"
