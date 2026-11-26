"""Replicate the paper's analyses on data/features.parquet.

1. Random forest classifying source (paper: 7-way acc 66%; 4.2% LLM->human, 9.8% human->LLM)
2. Rate vs human, paired Cohen's d, Wilcoxon + Bonferroni per feature (paper Fig 3, Tables S5-S6)
3. Pairwise RF human vs each LLM (paper: 93-98%)
usage: uv run python analysis.py [all | source ...]   (default: the paper's 6 LLMs; all = every tagged source)
Multi-way RF uses docs completed by every source; per-model comparisons use each model's own completed docs.
Writes results/<name>.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PAPER = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "llama-3-70B-Instruct",
         "llama-3-8B-Instruct", "llama-3-70B", "llama-3-8B"]
HUMAN = "human-chunk-2"
PAPER_FIG3 = {"gpt-4o-2024-08-06": {"f_25_present_participle": (5.3, 1.38), "f_29_that_subj": (2.6, 0.77),
                                    "f_14_nominalizations": (2.1, 1.23), "f_64_phrasal_coordination": (1.9, 0.81)}}
os.makedirs("results", exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn 1.8 LogisticRegressionCV API churn

fa = pd.read_parquet("data/features.parquet")
llms = sys.argv[1:] or PAPER
if llms == ["all"]:
    llms = PAPER + sorted(set(fa.source) - set(PAPER) - {HUMAN})
NAME = "paper" if llms == PAPER else "all" if len(llms) > len(PAPER) else "custom"
fa = fa[fa.source.isin(llms + [HUMAN])].reset_index(drop=True)
docs = set.intersection(*(set(g.doc_id) for _, g in fa.groupby("source")))  # docs completed by every source
f = fa[fa.doc_id.isin(docs)]
feats = [c for c in f.columns if c.startswith("f_")]
print(f"{len(docs)} paired docs, {len(llms)} LLMs, {len(feats)} features")

# 75/25 split by document, stratified by genre (paper SI); a human text and its LLM continuations stay together
rng = np.random.default_rng(0)
all_docs = pd.Series(sorted(set(fa.doc_id)))
test_docs = set()
for _, g in all_docs.groupby(all_docs.str.split("_").str[0]):
    test_docs |= set(rng.choice(g.values, size=len(g) // 4, replace=False))


def rf(data):
    m = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)
    te_mask = data.doc_id.isin(test_docs)
    tr, te = data[~te_mask], data[te_mask]
    m.fit(tr[feats], tr.source)
    return m, te.source, m.predict(te[feats])


# 1. multi-way classifier
m, y, p = rf(f)
labels = [HUMAN] + llms
cm = pd.DataFrame(confusion_matrix(y, p, labels=labels), index=labels, columns=labels)
llm_rows = y != HUMAN
print(f"\n1) {len(labels)}-way RF accuracy {np.mean(y == p):.1%} (chance {1/len(labels):.1%}); "
      f"LLM->human {np.mean(p[llm_rows.values] == HUMAN):.1%}, human->LLM {np.mean(p[~llm_rows.values] != HUMAN):.1%}")
print(cm.to_string())
imp = pd.Series(m.feature_importances_, feats).sort_values(ascending=False)
top15 = list(imp.index[:15])

# 2. per-feature comparison against the human chunk
h_all = fa[fa.source == HUMAN].set_index("doc_id")[feats]
rows = []
for s in llms:
    l = fa[fa.source == s].set_index("doc_id")[feats]
    ids = l.index.intersection(h_all.index).sort_values()
    l, h = l.loc[ids], h_all.loc[ids]
    for c in feats:
        diff = l[c] - h[c]
        p = wilcoxon(l[c], h[c], zero_method="zsplit").pvalue if diff.abs().sum() else 1.0
        rows.append((s, c, l[c].mean() / h[c].mean() if h[c].mean() else np.nan,
                     diff.mean() / diff.std() if diff.std() else 0.0, min(p * len(feats), 1.0)))
res = pd.DataFrame(rows, columns=["source", "feature", "rate", "d", "p_bonf"])
res.to_csv(f"results/features_vs_human_{NAME}.csv", index=False)
print("\n2) Rate vs human (top 15 RF features; * = Bonferroni p < .05)")
tab = res[res.feature.isin(top15)].assign(cell=lambda r: r.rate.map("{:.2f}".format) + np.where(r.p_bonf < .05, "*", " "))
print(tab.pivot(index="feature", columns="source", values="cell").loc[top15, llms].to_string())
for s, ref in PAPER_FIG3.items():
    if s in llms:
        print(f"\n   {s} vs paper (rate, d):")
        for c, (r0, d0) in ref.items():
            r = res[(res.source == s) & (res.feature == c)].iloc[0]
            print(f"   {c:28s} ours {r.rate:.2f}x d={r.d:.2f} | paper {r0}x d={d0}")

# 3. pairwise human vs each LLM: random forest and lasso logistic regression (paper Tables S9, S10)
# ponytail: lasso C picked by 10-fold CV best score, not glmnet's 1-SE rule; slightly less regularized
print("\n3) Pairwise test accuracy, human vs LLM:     RF     lasso")
pair_rows = []
for s in llms:
    d = fa[fa.source.isin([HUMAN, s])]
    d = d[d.doc_id.isin(set(d[d.source == s].doc_id))]  # pairs only
    _, y, p = rf(d)
    te_mask = d.doc_id.isin(test_docs)
    tr, te = d[~te_mask], d[te_mask]
    lasso = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=20, cv=10, penalty="l1", solver="liblinear"))
    acc_l = lasso.fit(tr[feats], tr.source).score(te[feats], te.source)
    pair_rows.append((s, np.mean(y == p), acc_l))
    print(f"   {s:40s} {np.mean(y == p):6.1%} {acc_l:6.1%}")
pd.DataFrame(pair_rows, columns=["source", "rf_acc", "lasso_acc"]).to_csv(
    f"results/pairwise_{NAME}.csv", index=False)
imp.rename("importance").to_csv(f"results/importance_{NAME}.csv")
cm.to_csv(f"results/confusion_{NAME}.csv")
