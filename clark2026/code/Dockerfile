# Coding-setting sandbox for the Clark et al. (2026) replication: one container per agent session.
# Pi (the paper's harness, pinned) + Python for the repositories. Built as clark-pi:0.84.4.
FROM node:24-bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends bash ca-certificates git ripgrep python3 python3-venv python3-pip build-essential \
  && rm -rf /var/lib/apt/lists/*
# --ignore-scripts: no install hooks run (as in Pi's own containerization docs)
RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent@0.84.4

WORKDIR /workspace
CMD ["sleep", "infinity"]
