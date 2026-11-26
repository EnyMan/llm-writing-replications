"""Download the released model outputs and extract them where the scripts expect them.

  reinhart2025-outputs.zip -> reinhart2025/out/ and reinhart2025/data/features.parquet
  clark2026-outputs.zip    -> clark2026/out/{essays,chat,code,_discarded}/
usage (repo root): uv run python scripts/fetch_outputs.py [reinhart2025|clark2026 ...] [--tag v1.0.0]
Uses the public release URL; if that fails (e.g. a private fork), falls back to `gh release download`.
"""
import os, subprocess, sys, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "EnyMan/llm-writing-replications"
args = sys.argv[1:]
tag = args[args.index("--tag") + 1] if "--tag" in args else "v1.0.0"
names = [a for a in args if not a.startswith("--") and a != tag] or ["reinhart2025", "clark2026"]

os.makedirs(os.path.join(ROOT, "release"), exist_ok=True)
for name in names:
    asset = f"{name}-outputs.zip"
    path = os.path.join(ROOT, "release", asset)
    if not os.path.exists(path):
        url = f"https://github.com/{REPO}/releases/download/{tag}/{asset}"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"{url}: {e}; trying gh")
            subprocess.run(["gh", "release", "download", tag, "-R", REPO, "-p", asset, "-D", os.path.dirname(path)], check=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(ROOT)
        print(f"{asset}: {len(z.namelist())} files extracted")
