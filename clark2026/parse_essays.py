"""Parse essays (human PERSUADE, model essays, Wikipedia sample) with en_core_web_trf on GPU.

Writes clark2026/parsed/<source>.parquet: one row per sentence with essay id, prompt, words, arcs and counts.
Incremental: sources already parsed are skipped. Model essay files with fewer than 70 essays are skipped.
usage (repo root): uv run python clark2026/parse_essays.py
"""
import glob, json, os, sys
import pandas as pd
import spacy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import clean_essay, sentences, sent_features  # noqa: E402

OUT = os.path.join(HERE, "parsed")
WIKI_WORDS = 500_000


def sources():
    h = pd.read_parquet(os.path.join(HERE, "data", "persuade_indep_score56.parquet"))
    yield "human-persuade", [(r.essay_id, r.prompt_name, r.full_text) for r in h.itertuples()], False
    w = pd.read_parquet(os.path.join(HERE, "data", "wikipedia", "20231101.en", "train-00001-of-00041.parquet"))  # pinned shard
    w = w.sample(frac=1, random_state=0)
    picked, n = [], 0
    for r in w.itertuples():
        picked.append((r.id, "wikipedia", r.text))
        n += len(r.text.split())
        if n >= WIKI_WORDS:
            break
    yield "wikipedia", picked, False
    for f in sorted(glob.glob(os.path.join(HERE, "out", "essays", "*.jsonl"))):
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        rows = list({(r["prompt_name"], r["rep"]): r for r in rows if r["text"].strip()}.values())
        if len(rows) < 70:
            continue
        name = os.path.basename(f).removesuffix(".jsonl")
        yield name, [(f"{r['prompt_name']}#{r['rep']}", r["prompt_name"], r["text"]) for r in rows], True


def main():
    os.makedirs(OUT, exist_ok=True)
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["ner"])
    for name, essays, is_model in sources():
        path = os.path.join(OUT, name + ".parquet")
        if os.path.exists(path):
            continue
        meta, texts = [], []
        for eid, prompt, text in essays:
            for s in sentences(clean_essay(text, is_model)):
                meta.append((eid, prompt))
                texts.append(s)
        rows = []
        for (eid, prompt), s, doc in zip(meta, texts, nlp.pipe(texts, batch_size=128)):
            for sent in doc.sents:  # syntok decided the sentence; spaCy may still split it further
                r = sent_features(sent)
                rows.append({"essay": eid, "prompt": prompt, "text": sent.text, **r})
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"{name}: {len(essays)} essays, {len(rows)} sentences, {sum(r['words'] for r in rows)} words", flush=True)


if __name__ == "__main__":
    main()
