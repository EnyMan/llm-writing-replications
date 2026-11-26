"""Does adding the NOCOUNT sentence to the prompt shift Biber features?

Paired (same doc, same model) orig vs nocount; Wilcoxon signed-rank + Bonferroni over features,
paired Cohen's d = mean(diff) / sd(diff). Same test the paper uses for LLM vs human.
"""
import json
import pandas as pd
from scipy.stats import wilcoxon
from features import tag

MODELS = ["deepseek__deepseek-v4-flash", "tencent__hy4-flash"]


def load(path):
    d = pd.DataFrame(map(json.loads, open(path, encoding="utf-8")))
    d = d[d.text.str.split().str.len() >= 250]  # drop empty/short, as the paper did
    return d.drop_duplicates("doc_id", keep="last")


for m in MODELS:
    a, b = load(f"out/{m}.jsonl"), load(f"out/{m}@nocount.jsonl")
    ids = sorted(set(a.doc_id) & set(b.doc_id))
    fa = tag(a[a.doc_id.isin(ids)]).loc[ids]
    fb = tag(b[b.doc_id.isin(ids)]).loc[ids]
    diff = fb - fa
    rows = []
    for c in fa.columns:
        p = wilcoxon(fa[c], fb[c], zero_method="zsplit").pvalue if diff[c].abs().sum() else 1.0
        sd = diff[c].std()
        rows.append((c, fa[c].mean(), fb[c].mean(), diff[c].mean() / sd if sd else 0.0, min(p * len(fa.columns), 1)))
    r = pd.DataFrame(rows, columns=["feature", "orig", "nocount", "d", "p_bonf"]).set_index("feature")
    print(f"\n== {m}: n={len(ids)} paired docs; words orig {a.text.str.split().str.len().median():.0f} "
          f"vs nocount {b.text.str.split().str.len().median():.0f} (median)")
    print(f"significant after Bonferroni: {(r.p_bonf < 0.05).sum()} / {len(r)}; |d| > 0.2: {(r.d.abs() > 0.2).sum()}")
    print(r[(r.p_bonf < 0.05) | (r.d.abs() > 0.2)].round(3).to_string())
