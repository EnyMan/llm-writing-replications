"""Build the report's data files (report/*.csv) from results/ and out/.

Run after tag_all.py, analysis.py all, analysis.py (paper), compare_paper.py, vocab.py.
"""
import glob, json, os, re
import pandas as pd
from compare_paper import NAME2F

os.makedirs("report", exist_ok=True)
F2NAME = {v: k for k, v in NAME2F.items()} | {"f_43_type_token": "Type-token ratio"}
PAPER_LABELS = {"gpt-4o-2024-08-06": "GPT-4o", "gpt-4o-mini-2024-07-18": "GPT-4o mini",
                "llama-3-70B-Instruct": "Llama 3 70B Instruct", "llama-3-8B-Instruct": "Llama 3 8B Instruct",
                "llama-3-70B": "Llama 3 70B (base)", "llama-3-8B": "Llama 3 8B (base)"}
NICE = {"gpt": "GPT", "glm": "GLM", "hy": "Hy", "mimo": "MiMo", "vl": "VL", "it": "IT", "b": "B", "a": "A"}


LABELS = {"nanogpt_z-ai__glm-5.3-flash-uncensored": "GLM 5.3 Flash Uncensored",
          "nanogpt_deepseek__deepseek-v4.1-flash-uncensored": "DeepSeek v4.1 Flash Uncensored",
          "nanogpt_Gemma-4-31B-Gembrain-uncensored-heretic": "Gemma 4 31B Gembrain (heretic)",
          "nanogpt_qwen__qwen3.8-27b-antislop": "Qwen 3.8 27B Antislop", "nanogpt_qwen__qwen3.8-27b": "Qwen 3.8 27B",
          "ollama_richardyoung__hemmingway-1_Q4_K_M": "Hemmingway-1 (Qwen 3.8 27B, local Q4)",
          "mistralai__mistral-small-3.1-24b-instruct": "Mistral Small 3.1 24B",
          "google__gemini-3.8-flash_floor": "Gemini 3.8 Flash", "baidu__ernie-5.5-flash-preview_free@nocount": "ERNIE 5.5 Flash"}


def label(src):
    if src in PAPER_LABELS:
        return PAPER_LABELS[src]
    if src in LABELS:
        return LABELS[src]
    name, _, variant = src.partition("@")
    if name.startswith("claude-"):  # claude-opus-5-5@minimal -> Claude Opus 5.5 [minimal]
        _, fam, major, minor = name.split("-")
        return f"Claude {fam.capitalize()} {major}.{minor} [{variant}]"
    model = name.split("__", 1)[-1].removesuffix("_free")
    words = [NICE.get(w, w.capitalize()) if not re.match(r"^v?\d", w) else w for w in model.split("-")]
    out = " ".join(words).replace(" Flash 1017", " Flash (1017)").replace(" Flash 0731", " Flash (0731)")
    for a, b in [("Deepseek", "DeepSeek"), ("Minimax", "MiniMax"), ("GPT 6", "GPT-6"), ("31b", "31B"),
                 ("550b A55b", "550B"), ("Mini4", "Mini 4"), ("Qwen3.8", "Qwen 3.8"), ("Ernie", "ERNIE")]:
        out = out.replace(a, b)
    return out + (f" [{variant}]" if variant and variant != "nocount" else "")


def group(src):
    if src.startswith("llama-3") and "Instruct" not in src:
        return "2024 base model"
    if src in PAPER_LABELS:
        return "2024 instruction-tuned"
    if src.startswith("claude-"):
        return f"Claude via claude -p ({src.partition('@')[2]})"
    if re.search(r"uncensored|gembrain|antislop|hemmingway", src, re.I):
        return "2026 community finetune"
    return "2026 general model"


# per-model generation facts from out/
gen = {}
for f in glob.glob("out/*.jsonl"):
    name = os.path.basename(f).removesuffix(".jsonl")
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    gen[name] = {"cost_usd": round(sum(r.get("cost") or 0 for r in rows), 2),
                 "attempted": len({r["doc_id"] for r in rows})}

feats = pd.read_csv("results/features_vs_human_all.csv")
pw = pd.read_csv("results/pairwise_all.csv").set_index("source")
tagged = pd.read_parquet("data/features.parquet").groupby("source").size()
models = []
for src, g in feats.groupby("source", sort=False):
    att = gen.get(src, {}).get("attempted", 900)
    models.append({"model": src, "label": label(src), "group": group(src),
                   "texts_used": int(tagged[src]), "dropped_pct": round(100 * (1 - tagged[src] / max(att, tagged[src])), 1),
                   "mean_abs_d": round(g.d.abs().mean(), 3), "features_large_d": int((g.d.abs() > 0.5).sum()),
                   "rf_acc": round(100 * pw.loc[src, "rf_acc"], 1), "lasso_acc": round(100 * pw.loc[src, "lasso_acc"], 1),
                   "gen_cost_usd": gen.get(src, {}).get("cost_usd", "")})
pd.DataFrame(models).sort_values("mean_abs_d").to_csv("report/models.csv", index=False)

imp = pd.read_csv("results/importance_paper.csv", index_col=0).importance.rank(ascending=False).astype(int)
lab = {m["model"]: m["label"] for m in models}
grp = {m["model"]: m["group"] for m in models}
feats = feats.assign(label=feats.source.map(lab), group=feats.source.map(grp),
                     feature_label=feats.feature.map(F2NAME), paper_importance_rank=feats.feature.map(imp),
                     rate=feats.rate.round(3), d=feats.d.round(3), significant=(feats.p_bonf < 0.05).astype(int))
feats.rename(columns={"source": "model"})[["model", "label", "group", "feature", "feature_label", "paper_importance_rank",
                                           "rate", "d", "significant"]].to_csv("report/features.csv", index=False)

rep = pd.read_csv("results/replication_vs_paper.csv")
rep.assign(label=rep.source.map(lab), feature_label=rep.feature.map(F2NAME),
           d_paper=rep.d_paper.round(2), d_ours=rep.d_ours.round(3))[
    ["label", "feature_label", "d_paper", "d_ours"]].to_csv("report/replication.csv", index=False)

KIND = {"claude-haiku-5-5@minimal": "Claude Code prompt", "claude-sonnet-5-5@minimal": "Claude Code prompt",
        "claude-opus-5-5@minimal": "Claude Code prompt", "claude-fable-5-2@minimal": "Claude Code prompt",
        "anthropic__claude-haiku-5.5": "Route: API vs claude -p"}
c = pd.read_csv("results/conditions.csv")
c.assign(comparison=c.a.map(KIND).fillna("Finetune vs parent"), from_label=c.a.map(lab), to_label=c.b.map(lab),
         from_d=c.mean_abs_d_vs_human_a, to_d=c.mean_abs_d_vs_human_b,
         from_rf=c.a.map(lambda s: round(100 * pw.loc[s, "rf_acc"], 1)), to_rf=c.b.map(lambda s: round(100 * pw.loc[s, "rf_acc"], 1)))[
    ["comparison", "from_label", "to_label", "docs", "features_sig", "max_abs_d", "from_d", "to_d", "from_rf", "to_rf", "top_shifts"]
].to_csv("report/pairs.csv", index=False)

v = pd.read_csv("results/vocab_over.csv")
v = v.groupby("source", sort=False).head(8)
v.assign(label=v.source.map(lab), rate_ratio=v.rate_ratio.round(1), doc_pct=(100 * v.doc_frac).round(1))[
    ["label", "word", "doc_pct", "rate_ratio"]].to_csv("report/vocab.csv", index=False)
print(pd.read_csv("report/models.csv").to_string())

# The dashboard splits CSV at every comma (no quoting), and some labels contain commas: ship JSON rows instead.
for name in ["models", "features", "replication", "vocab", "pairs"]:
    pd.read_csv(f"report/{name}.csv", keep_default_na=False).to_json(f"report/{name}.json", orient="records", force_ascii=False)
