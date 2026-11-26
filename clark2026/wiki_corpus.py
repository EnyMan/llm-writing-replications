"""Tokenize all of English Wikipedia (wikimedia/wikipedia 20231101.en) for the KenLM reference model.

One sentence per line, lowercased word tokens (ngrams.tokens, the same tokenizer used on the essays), gzipped.
Paragraphs under 6 words (headings, list stubs) are skipped, as in ngram_run.wiki_sentences.
usage: uv run python clark2026/wiki_corpus.py  -> clark2026/data/wiki_tokens.txt.gz
"""
import glob, gzip, os, re, sys
from multiprocessing import Pool
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ngrams import tokens  # noqa: E402

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def shard(path):
    out, n = [], 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2000):
        for text in batch.column(0).to_pylist():
            for para in text.split("\n"):
                if len(para.split()) < 6:
                    continue
                for s in SENT.split(para):
                    t = tokens(s)
                    if t:
                        out.append(" ".join(t))
                        n += len(t)
    part = path + ".tok.gz"
    with gzip.open(part, "wt", encoding="utf-8", compresslevel=3) as f:
        f.write("\n".join(out) + "\n")
    return part, n


if __name__ == "__main__":
    shards = sorted(glob.glob(os.path.join(HERE, "data", "wikipedia", "20231101.en", "*.parquet")))
    print(f"{len(shards)} shards", flush=True)
    total = 0
    with Pool(12) as p:
        for part, n in p.imap_unordered(shard, shards):
            total += n
            print(f"{os.path.basename(part)}: {n/1e6:.0f}M words (total {total/1e9:.2f}B)", flush=True)
    # gzip members concatenate into one valid gzip stream
    with open(os.path.join(HERE, "data", "wiki_tokens.txt.gz"), "wb") as out:
        for s in shards:
            with open(s + ".tok.gz", "rb") as f:
                while chunk := f.read(1 << 24):
                    out.write(chunk)
            os.remove(s + ".tok.gz")
    print(f"done: {total/1e9:.2f}B words", flush=True)
