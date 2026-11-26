"""Build the 900-doc working set: HAP-E mini doc IDs x (chunk 1, chunk 2, paper's 6 LLMs)."""
import glob, os
import pandas as pd

mini = pd.read_parquet("data/human-ai-parallel-corpus-mini/hape_mini-text.parquet")
ids = set(mini.doc_id.str.split("_", n=1).str[1])  # gpt4_acad_0005 -> acad_0005

frames = []
for f in glob.glob("data/human-ai-parallel-corpus/text_data/*.parquet"):
    src = os.path.basename(f).removeprefix("hape-text_").removesuffix(".parquet")
    d = pd.read_parquet(f)
    d["doc_id"] = d.doc_id.str.split("@").str[0]
    frames.append(d[d.doc_id.isin(ids)].assign(source=src))
df = pd.concat(frames)[["doc_id", "source", "text"]]
df.to_parquet("data/mini_parallel.parquet", index=False)
print(len(ids), "ids"); print(df.groupby("source").size())
