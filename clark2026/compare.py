"""Our measures vs the values recovered from the paper's figure (paper_values.csv), per setting.

CLARK_SETTING=essay (default) or chat. Pooled per source; 95% bootstrap intervals resample essays within prompt
(essays) or whole conversations (chat), the paper's schemes. Merges n-gram and Jev results when present.
"""
import glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import pooled, arc_weights  # noqa: E402

SETTING = os.environ.get("CLARK_SETTING", "essay")
FOLDER, SUFFIX = {"essay": ("parsed", ""), "chat": ("parsed_chat", "_chat"), "coding": ("parsed_code", "_code")}[SETTING]
PAPER_NAME = {"human-persuade": "[human]", "human-opensubtitles": "[human]", "human-stackexchange": "[human]", "wikipedia": "[Wikipedia]"}
for prefix in ("", "nanogpt_"):
    PAPER_NAME.update({prefix + "openai__gpt-5.6-luna": "Luna", prefix + "z-ai__glm-5": "GLM", prefix + "moonshotai__kimi-k2.5": "Kimi",
                       prefix + "meta-llama__llama-3.1-8b-instruct": "Llama Inst."})
MEASURES = ["orc_rel", "orc_norel", "denominal", "inanimate", "dep_len", "stacked", "unattested", "perplexity"]
STRUCT = ["orc_rel", "orc_norel", "stacked", "stacked_common", "dep_len", "dep_len_raw"]

parsed = {os.path.basename(f)[:-8]: pd.read_parquet(f) for f in glob.glob(os.path.join(HERE, FOLDER, "*.parquet"))}
W = arc_weights(pd.read_parquet(os.path.join(HERE, "parsed", "wikipedia.parquet")).to_dict("records"))  # always Wikipedia
rng = np.random.default_rng(0)
rows = []
for src, df in parsed.items():
    recs = df.to_dict("records")
    point = pooled(recs, W)
    # bootstrap over essays within prompt (Wikipedia: over articles)
    by_essay = {}
    for r in recs:
        key = "all" if SETTING != "essay" else r["prompt"]  # chat/coding: resample whole conversations / sessions
        by_essay.setdefault((key, r["essay"]), []).append(r)
    by_prompt = {}
    for (p, e), rs in by_essay.items():
        by_prompt.setdefault(p, []).append(rs)
    boots = []
    for _ in range(200):
        sample = [r for essays in by_prompt.values() for i in rng.integers(0, len(essays), len(essays)) for r in essays[i]]
        boots.append(pooled(sample, W))
    for m in STRUCT:
        lo, hi = np.percentile([b[m] for b in boots], [2.5, 97.5])
        rows.append({"source": src, "measure": m, "ours": point[m], "lo": lo, "hi": hi})
ours = pd.DataFrame(rows)
for extra in [f"ngram_rates{SUFFIX}.csv", f"jev_rates{SUFFIX}.csv"]:
    p = os.path.join(HERE, "out", extra)
    if os.path.exists(p):
        e = pd.read_csv(p).melt(id_vars="source", var_name="measure", value_name="ours")
        ours = pd.concat([ours, e[e.measure.isin(MEASURES)]])

paper = pd.read_csv(os.path.join(HERE, "paper_values.csv"))
paper = paper[paper.setting == SETTING].rename(columns={"model": "paper_name", "value": "paper"})
ours["paper_name"] = ours.source.map(PAPER_NAME)
ours["base"] = ours.measure.str.replace("_common", "").str.replace("_raw", "")  # variants compare to the same paper panel
m = ours.merge(paper[["paper_name", "measure", "paper"]].rename(columns={"measure": "base"}), on=["paper_name", "base"], how="left")
m.to_csv(os.path.join(HERE, "out", f"comparison_{SETTING}.csv"), index=False)
m["cell"] = m.apply(lambda r: f"{r.ours:.2f}" + (f" ({r.paper:.2f})" if pd.notna(r.paper) else ""), axis=1)
order = [s for s in PAPER_NAME if s in set(m.source)] + sorted(set(m.source) - set(PAPER_NAME))
t = m.pivot_table(index="source", columns="measure", values="cell", aggfunc="first")
print("ours (paper)")
cols = [c for c in MEASURES + ["stacked_common", "dep_len_raw"] if c in t.columns]
print(t.reindex(order)[cols].to_string())
