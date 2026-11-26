#!/bin/bash
# Essay n-gram reference: tokenize all of English Wikipedia, then a KenLM trigram in WSL -> /root/lm/wiki3.binary
# (2.88 billion words; lmplz needs ~60% of WSL memory and ~100 GB of temp space). Run from Git Bash.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
C=$(MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- wslpath -a "$(cd clark2026 && pwd -W)")
[ -f clark2026/data/wiki_tokens.txt.gz ] || uv run python clark2026/wiki_corpus.py
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash -c "mkdir -p /root/lm /root/lmtmp && cd /root/lm && zcat $C/data/wiki_tokens.txt.gz | /opt/kenlm/build/bin/lmplz -o 3 -S 60% -T /root/lmtmp > wiki3.arpa 2> wiki_lmplz.log && /opt/kenlm/build/bin/build_binary -q 8 trie wiki3.arpa wiki3.binary >> wiki_lmplz.log 2>&1 && rm wiki3.arpa && ls -la /root/lm"
echo WIKI_LM_DONE
