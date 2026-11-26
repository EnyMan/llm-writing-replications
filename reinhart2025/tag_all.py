"""Tag every source with Biber features -> data/features.parquet (incremental: skips cached sources).

Sources: the paper's human chunk 2 + 6 LLMs (data/mini_parallel.parquet) and our models (out/*.jsonl).
New outputs are filtered as in the paper (SI): responses < 100 words dropped; we also drop refusals and
API errors, and strip a trailing meta note about the prompt ("*Note: exactly 500 words...*") before tagging.
Counts per source go to results/cleaning.csv.
usage: uv run python tag_all.py [--paper-only]
"""
import glob, json, os, re, sys
import pandas as pd

CACHE = "data/features.parquet"
REFUSAL = re.compile(r"^\W*(?:I can.?t|I cannot|I.?m sorry|Sorry, there was no response|I.?m unable|I won.?t|"
                     r"I.?m not (?:able|going)|I.?ll (?:stop|pass|leave)|I (?:do not|don.?t) (?:feel|think I should))", re.I)
# last paragraph that talks about the prompt itself: "*Note: ... 500 words / continuation / does not necessarily end*"
NOTE = re.compile(r"\n\s*(?:---\s*)?[*(\[]*\s*(?:Note|Word count|\(?This continuation|The continuation|\[?End)\b"
                  r"[^\n]*(?:word|continu|necessarily|end)[^\n]*\s*$", re.I)
CLEANING = []


def sources():
    df = pd.read_parquet("data/mini_parallel.parquet")
    for src, d in df[df.source != "human-chunk-1"].groupby("source"):
        yield src, d[["doc_id", "text"]]
    if "--paper-only" in sys.argv:
        return
    for f in sorted(glob.glob("out/*.jsonl")):
        name = os.path.basename(f).removesuffix(".jsonl")
        d = pd.DataFrame(map(json.loads, open(f, encoding="utf-8")))
        err = d.get("is_error", pd.Series(False, index=d.index)).fillna(False).astype(bool) | d.text.str.startswith("API Error")
        d = d[~err & (d.text.str.strip() != "")].drop_duplicates("doc_id", keep="last")
        if len(d) < 800:  # calibration / pilot / unfinished files
            continue
        note = d.text.str.contains(NOTE)
        d = d.assign(text=d.text.str.replace(NOTE, "", regex=True).str.strip())
        ref = d.text.str.match(REFUSAL)
        short = ~ref & (d.text.str.split().str.len() < 100)  # paper: drop < 100 words
        CLEANING.append({"source": name, "texts": len(d), "refusals": int(ref.sum()), "short": int(short.sum()),
                         "meta_notes_stripped": int(note.sum())})
        print(f"{name}: {len(d)} docs, dropping {ref.sum()} refusals + {short.sum()} short; stripped {note.sum()} notes")
        yield name, d.loc[~ref & ~short, ["doc_id", "text"]]


if __name__ == "__main__":
    from features import tag
    cache = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
    have = set(cache.source) if len(cache) else set()
    for src, d in sources():
        if src in have:
            continue
        print(f"tagging {src} ({len(d)} docs)", flush=True)
        f = tag(d).reset_index().assign(source=src)
        cache = pd.concat([cache, f], ignore_index=True)
        cache.to_parquet(CACHE, index=False)
    print(cache.groupby("source").size())
    pd.DataFrame(CLEANING).to_csv("results/cleaning.csv", index=False)
