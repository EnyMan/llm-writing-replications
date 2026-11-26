"""Download the StackExchange reference for the coding setting (HuggingFaceTB/stackexchange_2025_md, Markdown).
A seeded random 18 of Stack Overflow's 177 parquet files (~9 GB) + softwareengineering, codereview, unix in full."""
import json, os, random
from huggingface_hub import HfApi, hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "HuggingFaceTB/stackexchange_2025_md"
files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
so = sorted(f for f in files if f.startswith("stackoverflow.com/"))
pick = sorted(random.Random(0).sample(so, 18))
pick += sorted(f for f in files if f.split("/")[0] in ("softwareengineering.stackexchange.com", "codereview.stackexchange.com", "unix.stackexchange.com"))
json.dump(pick, open(os.path.join(HERE, "data", "stackexchange_files.json"), "w"), indent=1)
for i, f in enumerate(pick):
    hf_hub_download(REPO, f, repo_type="dataset", local_dir=os.path.join(HERE, "data", "stackexchange"))
    print(f"{i + 1}/{len(pick)} {f}", flush=True)
print("done")
