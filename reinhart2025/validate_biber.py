"""Check that pybiber reproduces the published HAP-E Biber features on a sample."""
import sys
import pandas as pd
import polars as pl
import spacy
import pybiber

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
D = "data/human-ai-parallel-corpus"
FEATS = "data/human-ai-parallel-corpus-biber/biber_data"

text = pd.read_parquet(f"{D}/text_data/hape-text_human-chunk-2.parquet")
pub = pd.read_parquet(f"{FEATS}/hape-biber_human-chunk-2.parquet")
pub = pub.set_index(pub.doc_id.str.split("@").str[0]).drop(columns="doc_id")
text = text.assign(doc_id=text.doc_id.str.split("@").str[0])
text = text[text.doc_id.isin(pub.index)].groupby(text.doc_id.str[:4]).head(N // 6)

nlp = spacy.load("en_core_web_sm", disable=["ner"])
tokens = pybiber.CorpusProcessor().process_corpus(pl.from_pandas(text[["doc_id", "text"]]), nlp_model=nlp)
ours = pybiber.biber(tokens, force_ttr=True, strict_be_main_verb=False).to_pandas().set_index("doc_id")

pub = pub.loc[ours.index, ours.columns]
rows = []
for c in ours.columns:
    a, b = ours[c], pub[c]
    rows.append((c, a.corr(b), a.mean(), b.mean(), (a - b).abs().mean()))
r = pd.DataFrame(rows, columns=["feature", "pearson_r", "ours_mean", "pub_mean", "mean_abs_diff"]).set_index("feature")
pd.set_option("display.width", 200, "display.max_rows", 100)
print(r.round(3))
print(f"\nn={len(ours)} docs; median r={r.pearson_r.median():.3f}; features with r<0.9: {(r.pearson_r < 0.9).sum()}")
