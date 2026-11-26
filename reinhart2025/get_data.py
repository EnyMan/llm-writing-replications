"""Download the HAP-E corpora (Reinhart et al., MIT licence) from Hugging Face and build the working set.

  data/human-ai-parallel-corpus/          full text corpus (human chunks 1 and 2 + the paper's six LLMs)
  data/human-ai-parallel-corpus-mini/     the 900-document mini subset whose doc IDs we use
  data/human-ai-parallel-corpus-biber/    the published Biber features (validate_biber.py compares against them)
Then runs build_mini.py (-> data/mini_parallel.parquet) and writes claude_kit/prompts.jsonl (the 900 chunk-1 texts
the `claude -p` kit continues).
usage (from reinhart2025/): uv run python get_data.py
"""
import json, os, subprocess, sys
import pandas as pd
from huggingface_hub import snapshot_download

for name in ["human-ai-parallel-corpus", "human-ai-parallel-corpus-mini", "human-ai-parallel-corpus-biber"]:
    snapshot_download(f"browndw/{name}", repo_type="dataset", local_dir=os.path.join("data", name))
    print("downloaded", name)
subprocess.run([sys.executable, "build_mini.py"], check=True)
df = pd.read_parquet("data/mini_parallel.parquet")
docs = df[df.source == "human-chunk-1"].sort_values("doc_id")
with open(os.path.join("claude_kit", "prompts.jsonl"), "w", encoding="utf-8") as f:
    for r in docs.itertuples():
        f.write(json.dumps({"doc_id": r.doc_id, "text": r.text}, ensure_ascii=False) + "\n")
print(f"{len(docs)} prompts -> claude_kit/prompts.jsonl")
