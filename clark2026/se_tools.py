"""StackExchange helpers for the coding setting (data/stackexchange, HuggingFaceTB/stackexchange_2025_md ThreadText).

  sample   seeded pass over every thread: keep each with probability P, shuffle, trim at ~500k prose words ->
           data/stackexchange/sample.jsonl ({"id", "text"} per thread: question + answers, comments dropped)
  tokens   every thread's prose except the sample's (features.clean_chat, code and placeholders out) -> lowercased tokens on stdout, for lmplz
usage (repo root): uv run python clark2026/se_tools.py sample | tokens
"""
import glob, json, os, random, re, sys
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import clean_chat, CODE_WORD  # noqa: E402

FILES = sorted(glob.glob(os.path.join(HERE, "data", "stackexchange", "*", "*.parquet")))
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")  # as subs_tools / ngrams


def thread_text(t):
    """ThreadText minus the dump's own scaffolding: asked-by and tags lines, comment blocks (quoted '>' lines)."""
    out = []
    for line in t.split("\n"):
        s = line.strip()
        if s.startswith(("(Asked by", "**Tags:**", ">")) or re.match(r"^## \d+ answers?$", s):
            continue
        out.append(line)
    return "\n".join(out)


def threads():
    for f in FILES:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=["Id", "ThreadText"], batch_size=20000):
            for i, t in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if t:
                    yield f"{os.path.basename(os.path.dirname(f))}:{i}", t


def prose_words(text):
    return sum(len(s.split()) for s, head in clean_chat(text) if not head)


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        TARGET, P = 500_000, 1 / 900  # 2.67M threads -> ~3,000 kept (~190 prose words each), spread over every file
        rng = random.Random(0)
        kept = [(i, thread_text(t)) for i, t in threads() if rng.random() < P]
        rng.shuffle(kept)
        n, out = 0, []
        for i, t in kept:
            out.append({"id": i, "text": t})
            n += prose_words(t)
            if n >= TARGET:
                break
        with open(os.path.join(HERE, "data", "stackexchange", "sample.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in out)
        print(f"kept {len(kept)} threads; wrote {len(out)} threads / {n} prose words")
    elif sys.argv[1] == "tokens":
        out = sys.stdout
        code = CODE_WORD.lower()
        sample = os.path.join(HERE, "data", "stackexchange", "sample.jsonl")
        held_out = {r["id"] for r in map(json.loads, open(sample, encoding="utf-8"))}  # the human sample stays out of its own reference
        for i, t in threads():
            if i in held_out:
                continue
            for s, head in clean_chat(thread_text(t)):
                if head:
                    continue
                for piece in s.lower().replace("’", "'").split(code):  # a placeholder breaks the n-gram chain
                    toks = TOKEN.findall(piece)
                    if toks:
                        out.write(" ".join(toks) + "\n")
