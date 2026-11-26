"""Paired comparison of two sources on the same documents, from data/features.parquet.

Default pairs: Claude Code default system prompt vs minimal, per model, the Haiku route check, and finetunes vs parents
(OpenRouter API vs claude -p minimal; note the API run had reasoning off, claude -p thinks ~5k tokens).
Per feature: paired Wilcoxon + Bonferroni and paired Cohen's d = mean(b - a) / sd(b - a).
Also: does the condition move a model toward human? mean |d vs human| for a and b.
Writes results/conditions.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PAIRS = [("claude-haiku-5-5@minimal", "claude-haiku-5-5@default"),
         ("claude-sonnet-5-5@minimal", "claude-sonnet-5-5@default"),
         ("claude-opus-5-5@minimal", "claude-opus-5-5@default"),
         ("claude-fable-5-2@minimal", "claude-fable-5-2@default"),
         ("anthropic__claude-haiku-5.5", "claude-haiku-5-5@minimal"),
         # finetune vs its parent
         ("z-ai__glm-5.3-flash", "nanogpt_z-ai__glm-5.3-flash-uncensored"),
         ("deepseek__deepseek-v4.1-flash", "nanogpt_deepseek__deepseek-v4.1-flash-uncensored"),
         ("google__gemma-4-31b-it", "nanogpt_Gemma-4-31B-Gembrain-uncensored-heretic"),
         ("nanogpt_qwen__qwen3.8-27b", "nanogpt_qwen__qwen3.8-27b-antislop"),
         ("nanogpt_qwen__qwen3.8-27b", "ollama_richardyoung__hemmingway-1_Q4_K_M")]

f = pd.read_parquet("data/features.parquet")
feats = [c for c in f.columns if c.startswith("f_")]
by = {s: g.set_index("doc_id")[feats] for s, g in f.groupby("source")}
hum = by["human-chunk-2"]
vs_h = pd.read_csv("results/features_vs_human_all.csv").groupby("source").d.apply(lambda x: x.abs().mean())

rows = []
for a, b in PAIRS:
    if a not in by or b not in by:
        print(f"skip {a} vs {b}: not tagged yet")
        continue
    ids = by[a].index.intersection(by[b].index).intersection(hum.index)
    A, B = by[a].loc[ids], by[b].loc[ids]
    res = []
    for c in feats:
        diff = B[c] - A[c]
        p = wilcoxon(A[c], B[c], zero_method="zsplit").pvalue if diff.abs().sum() else 1.0
        res.append((c, diff.mean() / diff.std() if diff.std() else 0.0, min(p * len(feats), 1.0)))
    r = pd.DataFrame(res, columns=["feature", "d", "p_bonf"])
    sig = r[r.p_bonf < .05].reindex(r.d.abs().sort_values(ascending=False).index).dropna()
    rows.append({"a": a, "b": b, "docs": len(ids), "features_sig": len(sig), "max_abs_d": round(r.d.abs().max(), 2),
                 "mean_abs_d_vs_human_a": round(vs_h.get(a, np.nan), 3), "mean_abs_d_vs_human_b": round(vs_h.get(b, np.nan), 3),
                 "top_shifts": "; ".join(f"{x.feature[5:]} {x.d:+.2f}" for x in sig.head(6).itertuples())})
    print(f"\n{a}  ->  {b}  ({len(ids)} paired docs)")
    print(f"  {len(sig)}/{len(feats)} features differ (Bonferroni); largest |d| {r.d.abs().max():.2f}")
    print(f"  mean |d| vs human: {vs_h.get(a, np.nan):.3f} -> {vs_h.get(b, np.nan):.3f}")
    for x in sig.head(8).itertuples():
        print(f"    {x.feature:32s} d = {x.d:+.2f}")
pd.DataFrame(rows).to_csv("results/conditions.csv", index=False)
