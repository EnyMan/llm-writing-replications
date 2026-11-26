"""Ask a generative model the decision question with an explanation, to see why borderline cases tip yes or no.

Uses the same prompt layout as Torchcast's jev_render_v3 (system line; Evidence / Question / Options A) yes B) no),
plus a request to explain before answering. Thinking off, temperature 0. Picks diagnostic cases from the test sample:
validation items it got wrong, human-essay items where Torchcast says yes but Jev says no, and the probes.
usage: uv run python clark2026/explain.py VERSION TC_SAMPLE.jsonl [ollama-model]  -> out/explain_<VERSION>.md
"""
import json, os, random, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jev import VALIDATION, PROBES  # noqa: E402

version, tc_path = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "qwen3.5:4b-bf16"
Q = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))[version]
SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option."


def load(p):
    return {(r["kind"], r["sentence"], r["word"], r["subj"]): r["p"] for r in map(json.loads, open(p, encoding="utf-8"))}


def ask(kind, sentence, word, subj):
    spec = Q[kind]
    q = spec["instructions"].format(word=word, subj=subj)
    user = (f"Evidence:\n{json.dumps({'sentence': sentence}, ensure_ascii=False)}\n\nQuestion: {q}\nOptions:\n"
            f"A) yes: {spec['criteria']['true']}\nB) no: {spec['criteria']['false']}\n\n"
            "First explain in one or two sentences why, naming what you take the subject and verb to mean here. "
            "Then give the letter on its own last line.")
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 200}}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]["content"].strip()


tc, jev = load(tc_path), load(os.path.join(HERE, "out", "jev_cache.jsonl"))
labelled = {(k, x, v, su): want for (k, want), rows in VALIDATION.items() for x, su, v in rows}
cases = [(key, f"validation, should be {'yes' if want else 'no'}") for key, want in labelled.items()
         if key in tc and (tc[key] > 0.5) != want]
cases += [(p, "probe") for p in PROBES]
random.seed(0)
for kind in ("denominal", "inanimate"):
    excess = [k for k in tc if k[0] == kind and k not in labelled and k in jev and tc[k] > 0.5 and jev[k] < 0.5]
    cases += [(k, f"Torchcast yes ({tc[k]:.2f}), Jev no ({jev[k]:.2f})") for k in random.sample(excess, min(30, len(excess)))]

out = os.path.join(HERE, "out", f"explain_{version}.md")
with open(out, "w", encoding="utf-8") as f:
    f.write(f"# Explanations, questions {version}, model {model}\n\n")
    for (kind, sentence, word, subj), why in cases:
        ans = ask(kind, sentence, word, subj)
        f.write(f"## {kind}: {subj + ' -> ' if subj else ''}{word}  [{why}; Torchcast {tc.get((kind, sentence, word, subj), float('nan')):.2f}]\n"
                f"> {sentence}\n\n{ans}\n\n")
        print(".", end="", flush=True)
print(f"\n{len(cases)} cases -> {out}")
