"""Over/under-represented words per source vs human chunk 2 (paper Table 1 / S7 / S8 analogue).

Lowercased word tokens (no lemmatization, unlike the paper's Fig 4). Uses the same filtered texts as tag_all.py.
rate_ratio = (llm per-million rate) / (human per-million rate), add-one smoothed.
over:  ratio >= 10 and in >= 2% of the model's outputs, ordered by fraction of outputs containing the word (S7)
under: human count >= 30, ordered by ratio ascending (S8)
Writes results/vocab_over.csv, results/vocab_under.csv
"""
import collections, re
import pandas as pd
from tag_all import sources

WORD = re.compile(r"[a-z]+(?:['’][a-z]+)?")
rows = []
counts, docfreq, totals, ndocs = {}, {}, {}, {}
for src, d in sources():
    c, df = collections.Counter(), collections.Counter()
    for t in d.text:
        toks = WORD.findall(t.lower().replace("’", "'"))  # curly vs straight apostrophe is typography, not vocab
        c.update(toks)
        df.update(set(toks))
    counts[src], docfreq[src], totals[src], ndocs[src] = c, df, sum(c.values()), len(d)

H = "human-chunk-2"
hc, ht = counts[H], totals[H]
over, under = [], []
for src in counts:
    if src == H:
        continue
    c, t, n = counts[src], totals[src], ndocs[src]
    ratio = {w: ((c[w] + 1) / t) / ((hc[w] + 1) / ht) for w in c}
    o = [(w, ratio[w], docfreq[src][w] / n) for w in c if ratio[w] >= 10 and docfreq[src][w] / n >= 0.02]
    over += [(src, w, r, f) for w, r, f in sorted(o, key=lambda x: -x[2])[:15]]
    u = [(w, ((c[w] + 1) / t) / ((hc[w] + 1) / ht)) for w in hc if hc[w] >= 30]
    under += [(src, w, r) for w, r in sorted(u, key=lambda x: x[1])[:15]]

pd.DataFrame(over, columns=["source", "word", "rate_ratio", "doc_frac"]).to_csv("results/vocab_over.csv", index=False)
pd.DataFrame(under, columns=["source", "word", "rate_ratio"]).to_csv("results/vocab_under.csv", index=False)
o = pd.DataFrame(over, columns=["source", "word", "rate_ratio", "doc_frac"])
for s, g in o.groupby("source", sort=False):
    print(f"{s:42s} " + ", ".join(f"{w} {f:.0%}" for w, f in zip(g.word[:8], g.doc_frac[:8])))
