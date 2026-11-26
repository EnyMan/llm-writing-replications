"""N-gram measures for the essay setting, scored against a Wikipedia trigram reference.

The paper does not say how large its Wikipedia reference was, and the unattested-pair rate depends on it.
`calibrate` scores the human essays against references of several sizes; `run SIZE` scores every source
against the chosen size and writes out/ngram_rates.csv.
usage: uv run python clark2026/ngram_run.py calibrate | run 20000000 | export
"""
import glob, json, os, re, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import clean_essay, sentences  # noqa: E402
from ngrams import KN3  # noqa: E402

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def wiki_sentences(n_words):
    w = pd.read_parquet(os.path.join(HERE, "data", "wikipedia", "20231101.en", "train-00001-of-00041.parquet"))  # pinned shard
    w = w.sample(frac=1, random_state=1)
    out, n = [], 0
    for text in w.text:
        for para in text.split("\n"):
            para = para.strip()
            if len(para.split()) < 6:  # headings, list stubs
                continue
            for s in SENT.split(para):
                out.append(s)
                n += len(s.split())
        if n >= n_words:
            return out
    return out


def source_sentences():
    h = pd.read_parquet(os.path.join(HERE, "data", "persuade_indep_score56.parquet"))
    yield "human-persuade", [s for t in h.full_text for s in sentences(clean_essay(t, False))]
    for f in sorted(glob.glob(os.path.join(HERE, "out", "essays", "*.jsonl"))):
        rows = list({(r["prompt_name"], r["rep"]): r for r in map(json.loads, open(f, encoding="utf-8")) if r["text"].strip()}.values())
        if len(rows) >= 70:
            yield os.path.basename(f)[:-6], [s for r in rows for s in sentences(clean_essay(r["text"], True))]


def calibrate():
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    human = next(source_sentences())[1]
    log(f"human sentences: {len(human)}")
    for n in [2_000_000, 5_000_000, 10_000_000, 20_000_000, 40_000_000]:
        ref = wiki_sentences(n)
        log(f"wikipedia {n/1e6:.0f}M: {len(ref)} sentences")
        lm = KN3(ref)
        log("  model built")
        u, ppl = lm.score(human)
        print(f"reference {n/1e6:>4.0f}M words: human unattested {u:5.1f}/1k pairs, perplexity {ppl:6.1f}  (paper: 15.0, 247)", flush=True)


def run(n):
    lm = KN3(wiki_sentences(n))
    rows = [{"source": s, "unattested": u, "perplexity": p} for s, sents in source_sentences() for u, p in [lm.score(sents)]]
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "out", "ngram_rates.csv"), index=False)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


def export():
    """Tokenized sentences per source for kenlm_score.py (same tokenizer as the reference corpus)."""
    from ngrams import tokens
    d = os.path.join(HERE, "out", "ngram_inputs")
    os.makedirs(d, exist_ok=True)
    for src, sents in source_sentences():
        with open(os.path.join(d, src + ".txt"), "w", encoding="utf-8") as f:
            f.writelines(" ".join(t) + "\n" for t in map(tokens, sents) if t)
        print(src, len(sents))


def export_parsed(folder, out_name):
    """Tokenized sentences per source straight from a parsed folder (chat: already cleaned by features.clean_chat)."""
    from ngrams import tokens
    d = os.path.join(HERE, "out", out_name)
    os.makedirs(d, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(HERE, folder, "*.parquet"))):
        src = os.path.basename(f)[:-8]
        sents = pd.read_parquet(f, columns=["text"]).text
        with open(os.path.join(d, src + ".txt"), "w", encoding="utf-8") as out:
            out.writelines(" ".join(t) + "\n" for t in map(tokens, sents) if t)
        print(src, len(sents))


if __name__ == "__main__":
    {"calibrate": calibrate, "export": export, "export_chat": lambda: export_parsed("parsed_chat", "ngram_inputs_chat"),
     "export_code": lambda: export_parsed("parsed_code", "ngram_inputs_code")}.get(
        sys.argv[1], lambda: run(int(sys.argv[2])))()
