"""Pack the model outputs into the release zips (release/*.zip), for `gh release upload`.

  reinhart2025-outputs.zip  reinhart2025/out/**/*.jsonl (incl. _discarded/) + reinhart2025/data/features.parquet
  clark2026-outputs.zip     clark2026/out/{essays,chat,code,_discarded}/**/*.jsonl
Paths inside the zips are relative to the repo root, so scripts/fetch_outputs.py extracts them in place.
usage (repo root): uv run python scripts/pack_outputs.py
"""
import glob, os, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKS = {
    "reinhart2025-outputs.zip": ["reinhart2025/out/**/*.jsonl", "reinhart2025/data/features.parquet"],
    "clark2026-outputs.zip": [f"clark2026/out/{d}/**/*.jsonl" for d in ("essays", "chat", "code", "_discarded")],
}

os.makedirs(os.path.join(ROOT, "release"), exist_ok=True)
for name, patterns in PACKS.items():
    files = sorted({f for p in patterns for f in glob.glob(os.path.join(ROOT, p), recursive=True)})
    path = os.path.join(ROOT, "release", name)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(f, os.path.relpath(f, ROOT).replace(os.sep, "/"))
    print(f"{name}: {len(files)} files, {os.path.getsize(path) / 1e6:.1f} MB")
