"""Judge exported candidates with Torchcast Decision 4B locally (run inside WSL with /opt/tcvenv).

Uses the bundled torchcast_decision package (prompt format jev_render_v3, bidirectional full attention) but reads the
raw yes probability from engine.probabilities(), skipping the server's 0.2-0.8 "noul floor", so borderline cases keep
their real probability.
usage: /opt/tcvenv/bin/python decide_local.py MODEL_DIR CANDIDATES.jsonl QUESTIONS.json VERSION OUT.jsonl
"""
import json, os, sys, time

model_dir, cand_path, q_path, version, out_path = sys.argv[1:6]
sys.path.insert(0, model_dir)  # the bundled torchcast_decision package
from torchcast_decision import TorchcastDecision  # noqa: E402
from torchcast_decision import render as R  # noqa: E402

Q = json.load(open(q_path, encoding="utf-8"))[version]
done = set()
if os.path.exists(out_path):
    done = {(r["kind"], r["sentence"], r["word"], r["subj"]) for r in map(json.loads, open(out_path, encoding="utf-8"))}
cands = [c for c in map(json.loads, open(cand_path, encoding="utf-8")) if (c["kind"], c["sentence"], c["word"], c["subj"]) not in done]
print(f"{len(cands)} to judge ({len(done)} done), questions {version}", flush=True)

# max_batch_tokens: the package default (65,536) filled all 24 GB and spilled into shared memory (~4/s instead of ~30/s)
engine = TorchcastDecision(model_dir, images=False, max_batch_tokens=int(os.environ.get("TC_BATCH_TOKENS", "8192")))
t0, BATCH = time.time(), 256
with open(out_path, "a", encoding="utf-8") as out:
    for i in range(0, len(cands), BATCH):
        chunk = cands[i:i + BATCH]
        qs = []
        for c in chunk:
            spec = Q[c["kind"]]
            qs.append(R.question({"sentence": c["sentence"]}, {"type": "noul", "criteria": spec["criteria"],  # as sent to Jev
                                                 "instructions": spec["instructions"].format(word=c["word"], subj=c["subj"])}))
        probs, _ = engine.probabilities(qs)
        for c, q, p in zip(chunk, qs, probs):
            out.write(json.dumps({**c, "p": p[q.ids.index("true")], "model": "torchcast-decision-4b-v1.3", "questions": version}) + "\n")
        out.flush()
        if (i // BATCH) % 20 == 0:
            rate = (i + len(chunk)) / (time.time() - t0)
            print(f"{i + len(chunk)}/{len(cands)}  {rate:.0f}/s", flush=True)
print("done", flush=True)
