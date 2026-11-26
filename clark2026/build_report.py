"""Report data for the Clark et al. (2026) replication page -> clark2026/report/*.json.

Run after compare.py (and ngram / Torchcast steps). Re-run and re-upload when new settings or models arrive.
"""
import json, os
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report")
os.makedirs(OUT, exist_ok=True)

MEASURES = {  # ours -> (label, unit, paper panel)
    "orc_rel": ("Object relatives with a relativizer", "per 1,000 words", "orc_rel"),
    "orc_norel": ("Object relatives without a relativizer", "per 1,000 words", "orc_norel"),
    "denominal": ("Denominal verbs", "per 1,000 sentences", "denominal"),
    "inanimate": ("Inanimate subjects with animate verbs", "per 1,000 sentences", "inanimate"),
    "dep_len_raw": ("Dependency length", "words, standardized", "dep_len"),
    # common nouns only: with proper nouns, subtitle name chains inflate the human chat rate 4.6x (1.85 vs the paper's 0.40)
    "stacked_common": ("Stacked nouns", "per 1,000 words", "stacked"),
    "unattested": ("Word pairs never seen in the reference corpus", "per 1,000 pairs", "unattested"),
    "perplexity": ("Trigram perplexity against the reference corpus", "perplexity", "perplexity"),
}
LABELS = {"human-persuade": "Student essays (PERSUADE)", "wikipedia": "Wikipedia sample",
          "openai__gpt-5.6-luna": "GPT-5.6 Luna", "z-ai__glm-5": "GLM 5", "moonshotai__kimi-k2.5": "Kimi K2.5",
          "meta-llama__llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
          "nanogpt_z-ai__glm-5.3-flash-uncensored": "GLM 5.3 Flash Uncensored",
          "nanogpt_deepseek__deepseek-v4.1-flash-uncensored": "DeepSeek v4.1 Flash Uncensored",
          "nanogpt_Gemma-4-31B-Gembrain-uncensored-heretic": "Gemma 4 31B Gembrain (heretic)",
          "nanogpt_qwen__qwen3.8-27b-antislop": "Qwen 3.8 27B Antislop", "nanogpt_qwen__qwen3.8-27b": "Qwen 3.8 27B",
          "ollama_richardyoung__hemmingway-1_Q4_K_M": "Hemmingway-1 (Qwen 3.8 27B finetune)",
          "mistralai__mistral-small-3.1-24b-instruct": "Mistral Small 3.1 24B"}
PAPER_MODELS = ["openai__gpt-5.6-luna", "z-ai__glm-5", "moonshotai__kimi-k2.5", "meta-llama__llama-3.1-8b-instruct"]
PAPER_MODELS += ["nanogpt_" + x for x in PAPER_MODELS]  # chat ran the paper's models through NanoGPT
LABELS.update({"nanogpt_" + k: v for k, v in list(LABELS.items()) if not k.startswith(("nanogpt_", "human", "wikipedia"))})
LABELS["human-opensubtitles"] = "OpenSubtitles (human)"
LABELS["human-stackexchange"] = "StackExchange (human)"
LABELS.update({"Gemma-4-31B-Gembrain-uncensored-heretic": "Gemma 4 31B Gembrain (heretic)",  # coding: no provider prefix
               "deepseek__deepseek-v4.1-flash-uncensored": "DeepSeek v4.1 Flash Uncensored",
               "z-ai__glm-5.3-flash-uncensored": "GLM 5.3 Flash Uncensored", "qwen__qwen3.8-27b-antislop": "Qwen 3.8 27B Antislop",
               "qwen__qwen3.8-27b": "Qwen 3.8 27B", "hemmingway-1-96k": "Hemmingway-1 (Qwen 3.8 27B finetune)"})


def group(src):
    if src in ("human-persuade", "human-opensubtitles", "human-stackexchange"):
        return "Human"
    if src == "wikipedia":
        return "Wikipedia"
    return "In the paper" if src in PAPER_MODELS else "Our addition"


rows = []
for setting in ("essay", "chat", "coding"):
    path = os.path.join(HERE, "out", f"comparison_{setting}.csv")
    if not os.path.exists(path):
        continue
    m = pd.read_csv(path)
    m = m[m.measure.isin(MEASURES)]
    for r in m.itertuples():
        label, unit, panel = MEASURES[r.measure]
        rows.append({"setting": setting, "source": r.source, "label": LABELS.get(r.source, r.source), "group": group(r.source),
                     "measure": panel, "measure_label": label, "unit": unit, "ours": round(float(r.ours), 3),
                     "lo": None if pd.isna(getattr(r, "lo", float("nan"))) else round(float(r.lo), 3),
                     "hi": None if pd.isna(getattr(r, "hi", float("nan"))) else round(float(r.hi), 3),
                     "paper": None if pd.isna(r.paper) else round(float(r.paper), 3)})
values = pd.DataFrame(rows)
values.to_json(os.path.join(OUT, "values.json"), orient="records", force_ascii=False)

agree = []
for (setting, meas), g in values.groupby(["setting", "measure"], sort=False):
    h = g[g.group == "Human"]
    mod = g[g.source.isin(PAPER_MODELS) & g.paper.notna()]
    if len(mod) < 3:
        continue
    has_h = len(h) and pd.notna(h.iloc[0].paper)  # the paper draws no human line for some chat measures
    agree.append({"setting": setting, "measure": meas, "measure_label": g.measure_label.iloc[0],
                  "rank_rho": round(float(spearmanr(mod.ours, mod.paper).statistic), 2),
                  "same_side": int(((mod.ours > h.iloc[0].ours) == (mod.paper > h.iloc[0].paper)).sum()) if has_h else None,
                  "models": len(mod),
                  "human_ours": round(float(h.iloc[0].ours), 2) if len(h) else None,
                  "human_paper": round(float(h.iloc[0].paper), 2) if has_h else None})
pd.DataFrame(agree).to_json(os.path.join(OUT, "agreement.json"), orient="records", force_ascii=False)
print(pd.DataFrame(agree).to_string(index=False))
print(len(values), "value rows;", values.source.nunique(), "sources")
