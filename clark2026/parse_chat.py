"""Parse the chat setting: turn-final model replies and the OpenSubtitles human sample, with en_core_web_trf on GPU.

Writes clark2026/parsed_chat/<source>.parquet, one row per sentence (essay = conversation id, prompt = topic seed,
so the bootstrap resamples whole conversations). Headings are left out of sentence measures; no sentence crosses a
heading, list item, table cell or paragraph (features.clean_chat). Incremental; conversations files with < 10
conversations are skipped.
usage (repo root): uv run python clark2026/parse_chat.py   (CLARK_SETTING=coding: coding replies + StackExchange -> parsed_code/)
"""
import glob, json, os, sys
import pandas as pd
import spacy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import clean_chat, sentences, sent_features  # noqa: E402

CODING = os.environ.get("CLARK_SETTING") == "coding"
OUT = os.path.join(HERE, "parsed_code" if CODING else "parsed_chat")


def code_sources():
    """Coding: the StackExchange sample (unit = thread) and turn-final Pi replies (unit = agent session)."""
    sample = os.path.join(HERE, "data", "stackexchange", "sample.jsonl")
    if os.path.exists(sample):
        yield "human-stackexchange", [(r["id"], "stackexchange", clean_chat(r["text"])) for r in map(json.loads, open(sample, encoding="utf-8"))]
    for f in sorted(glob.glob(os.path.join(HERE, "out", "code", "*.jsonl"))):
        sessions = [json.loads(l) for l in open(f, encoding="utf-8")]
        if len(sessions) < 4:  # Antislop and Gembrain lost a session each to loops/provider errors
            continue
        yield os.path.basename(f)[:-6], [(s["repo_id"], s["repo_id"], clean_chat(e["reply"]))
                                         for s in sessions for e in s["episodes"] if (e.get("reply") or "").strip()]


def sources():
    if CODING:
        yield from code_sources()
        return
    sample = os.path.join(HERE, "data", "opensubtitles", "sample.txt")
    if os.path.exists(sample):
        lines = [l.strip() for l in open(sample, encoding="utf-8") if l.strip()]
        # each subtitle line is its own stretch; bootstrap unit: blocks of 50 lines
        yield "human-opensubtitles", [(f"subs{i // 50}", "opensubtitles", [(l, False)]) for i, l in enumerate(lines)]
    for f in sorted(glob.glob(os.path.join(HERE, "out", "chat", "*.jsonl"))):
        convs = [json.loads(l) for l in open(f, encoding="utf-8")]
        if len(convs) < 10:
            continue
        rows = []
        for c in convs:
            for t in c["turns"]:
                if t["role"] == "assistant" and t["text"].strip():
                    rows.append((c["seed"], c["seed"], clean_chat(t["text"])))
        yield os.path.basename(f)[:-6], rows


def main():
    os.makedirs(OUT, exist_ok=True)
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["ner"])
    for name, items in sources():
        path = os.path.join(OUT, name + ".parquet")
        if os.path.exists(path):
            continue
        meta, texts = [], []
        for eid, prompt, segments in items:
            for s in sentences(segments):
                meta.append((eid, prompt))
                texts.append(s)
        rows = []
        for (eid, prompt), doc in zip(meta, nlp.pipe(texts, batch_size=128)):
            for sent in doc.sents:
                rows.append({"essay": eid, "prompt": prompt, "text": sent.text, **sent_features(sent)})
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"{name}: {len(set(m[0] for m in meta))} units, {len(rows)} sentences, {sum(r['words'] for r in rows)} words", flush=True)


if __name__ == "__main__":
    main()
