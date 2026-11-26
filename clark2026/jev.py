"""Jev-judged features of Clark et al. (2026): denominal verbs and inanimate subjects with animate verbs.

Candidates come from the parse (features.sent_features):
  denominal   a verb whose lemma is used mostly as a noun in the setting's human corpus (PERSUADE for essays)
  inanimate   an active verb whose subject is a noun or it/this/that/these/those/which
Each candidate is judged in its sentence by Jev (typesafe/jev-1.13); accepted when P > 0.5. Rates per 1,000 sentences.
The paper's exact questions are unpublished; ours are below, checked by `validate` on the paper's example sentences.

usage: uv run python clark2026/jev.py validate | run | export | rates JUDGMENTS.jsonl [OUT.csv] | calibrated JUDGMENTS.jsonl [OUT.csv]
"""
import glob, json, os, sys, threading, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import KEY  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "out", "jev_cache.jsonl")
MODEL = "typesafe/jev-1.13"
Q = {
    "denominal": ("Is \"{word}\" in this sentence a denominal verb: a verb that is basically a noun put to work as a verb "
                  "(a thing-word turned into an action, like \"bottle the wine\", \"mirror a trend\", \"cushion a fall\", "
                  "\"mask a problem\", \"pool resources\", \"stage changes\", \"footnote it\")? Ordinary verbs that happen to "
                  "share a form with a noun but are primarily verbs (need, plan, study, help, work) do not count.",
                  {"true": "yes: a noun put to work as a verb", "false": "no: a primarily verbal word"}),
    "inanimate": ("In this sentence the subject \"{subj}\" does the action \"{word}\". Is the subject a thing, idea or "
                  "object (not a person, animal or group of people) while \"{word}\" is a verb that normally needs a "
                  "living, thinking subject (like promise, hope, forget, expect, ask, bite, swallow, misunderstand)?",
                  {"true": "yes: a lifeless subject doing something only a living being normally does",
                   "false": "no: the verb is normal for this kind of subject, or the subject is a person or group"}),
}
lock = threading.Lock()


def ask(kind, sentence, word, subj=""):
    instr, crit = Q[kind]
    body = {"model": MODEL, "state": {"sentence": sentence},
            "questions": {"q": {"type": "noul", "instructions": instr.format(word=word, subj=subj), "criteria": crit}}}
    req = urllib.request.Request("https://openrouter.ai/api/alpha/decisions", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            return d["answers"]["q"]["noul"], d.get("usage", {}).get("cost", 0)
        except Exception:
            if attempt == 2:
                return None, 0


VALIDATION = {  # (sentence, subject or "", verb): the paper's own example sentences (+) and plain cases (-)
    ("inanimate", True): [("The docstring promises they're skipped when they appear anywhere in the path.", "docstring", "promises"),
                          ("Sometimes chaos expects you to remember it independently.", "chaos", "expects"),
                          ("A song doesn't ask you to turn melancholy into a calendar item.", "song", "ask"),
                          ("These are the exact skills the worksheet hoped to teach.", "worksheet", "hoped"),
                          ("The time objection misunderstands how service actually functions.", "objection", "misunderstands"),
                          ("The existing check forgot its sibling type bytes.", "check", "forgot")],
    ("inanimate", False): [("The policy allows students to bring phones to school.", "policy", "allows"),
                           ("The car hit the wall at full speed.", "car", "hit"), ("The rain fell all night.", "rain", "fell"),
                           ("This idea makes sense for most students.", "idea", "makes"),
                           ("The data shows a steady increase.", "data", "shows"),
                           ("The committee decided to cancel the trip.", "committee", "decided")],
    ("denominal", True): [("Austin has high inventory, which cushions price spikes.", "", "cushions"),
                          ("Don't try to get her to categorize movies.", "", "categorize"),
                          ("The fallback is masking that inconsistency.", "", "masking"),
                          ("Pooling independent judgments filters out individual errors.", "", "pooling"),
                          ("This process mirrors real-world challenges.", "", "mirrors"), ("Just footnote it.", "", "footnote")],
    ("denominal", False): [("Students need more time to finish.", "", "need"), ("I think we should go.", "", "think"),
                           ("She will study for the test.", "", "study"), ("They plan to leave early.", "", "plan"),
                           ("He made a decision.", "", "made"), ("We want to help others.", "", "help")],
}
PROBES = [("inanimate", "Policy 1 teaches students responsibility.", "teaches", "policy"),  # Luna-style, unlabelled
          ("inanimate", "Sports teach teamwork, discipline, responsibility, and perseverance.", "teach", "sports")]


def validate():
    pos_inan, neg_inan = VALIDATION[("inanimate", True)], VALIDATION[("inanimate", False)]
    pos_den = [(x, v) for x, _, v in VALIDATION[("denominal", True)]]
    neg_den = [(x, v) for x, _, v in VALIDATION[("denominal", False)]]
    for name, rows, kind, want in [("inanimate +", pos_inan, "inanimate", True), ("inanimate -", neg_inan, "inanimate", False),
                                   ("denominal +", pos_den, "denominal", True), ("denominal -", neg_den, "denominal", False)]:
        ps = [ask(kind, r[0], r[-1], r[1] if kind == "inanimate" else "")[0] for r in rows]
        ok = sum((p > 0.5) == want for p in ps if p is not None)
        print(f"{name}: {ok}/{len(rows)} correct  " + " ".join(f"{r[-1]}={p:.2f}" for r, p in zip(rows, ps)))


def run():
    parsed = {os.path.basename(f)[:-8]: pd.read_parquet(f) for f in glob.glob(os.path.join(HERE, "parsed", "*.parquet"))}
    human = parsed["human-persuade"]
    nouns, verbs = Counter(w for ws in human.nouns for w in ws), Counter(w for ws in human.verbs for w in ws)
    mostly_noun = {w for w, n in nouns.items() if n > verbs.get(w, 0)}
    cache = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            r = json.loads(l)
            cache[(r["kind"], r["sentence"], r["word"], r["subj"])] = r["p"]
    todo = set()
    for src, df in parsed.items():
        if src == "wikipedia":
            continue
        for s, vs, sj in zip(df.text, df.verbs, df.subj):
            todo |= {("denominal", s, v, "") for v in vs if v in mostly_noun}
            todo |= {("inanimate", s, x.split("|")[2], x.split("|")[0]) for x in sj}
    todo = [k for k in todo if k not in cache]
    print(f"{len(todo)} candidates to judge ({len(cache)} cached)")

    def one(k):
        p, cost = ask(k[0], k[1], k[2], k[3])
        if p is None:
            return
        line = json.dumps({"kind": k[0], "sentence": k[1], "word": k[2], "subj": k[3], "p": p, "cost": cost}) + "\n"
        with lock:
            for attempt in range(20):  # Windows: an indexer / antivirus can hold the file for a moment
                try:
                    with open(CACHE, "a", encoding="utf-8") as f:
                        f.write(line)
                    break
                except PermissionError:
                    time.sleep(0.5)
            cache[k] = p
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, todo))
    rows = []
    for src, df in parsed.items():
        if src == "wikipedia":
            continue
        den = sum(cache.get(("denominal", s, v, ""), 0) > 0.5 for s, vs in zip(df.text, df.verbs) for v in vs if v in mostly_noun)
        ina = sum(cache.get(("inanimate", s, x.split("|")[2], x.split("|")[0]), 0) > 0.5 for s, sj in zip(df.text, df.subj) for x in sj)
        rows.append({"source": src, "sentences": len(df), "denominal": 1000 * den / len(df), "inanimate": 1000 * ina / len(df)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "out", "jev_rates.csv"), index=False)
    print(out.round(2).to_string(index=False))


# setting -> (parsed folder, human corpus that defines "mostly a noun", candidates file); CLARK_SETTING picks one
SETTINGS = {"essay": ("parsed", "human-persuade", "candidates.jsonl"),
            "chat": ("parsed_chat", "human-opensubtitles", "candidates_chat.jsonl"),
            "coding": ("parsed_code", "human-stackexchange", "candidates_code.jsonl")}
SETTING = os.environ.get("CLARK_SETTING", "essay")


def candidates(setting=None):
    """{source: dataframe}, the set of mostly-noun lemmas, and every unique candidate (kind, sentence, word, subj)."""
    folder, human_name, _ = SETTINGS[setting or SETTING]
    parsed = {os.path.basename(f)[:-8]: pd.read_parquet(f) for f in glob.glob(os.path.join(HERE, folder, "*.parquet"))
              if not f.endswith("wikipedia.parquet")}
    human = parsed[human_name]
    nouns, verbs = Counter(w for ws in human.nouns for w in ws), Counter(w for ws in human.verbs for w in ws)
    mostly_noun = {w for w, n in nouns.items() if n > verbs.get(w, 0)}
    todo = set()
    for df in parsed.values():
        for s, vs, sj in zip(df.text, df.verbs, df.subj):
            todo |= {("denominal", s, v, "") for v in vs if v in mostly_noun}
            todo |= {("inanimate", s, x.split("|")[2], x.split("|")[0]) for x in sj}
    return parsed, mostly_noun, todo


def export():
    """Candidates + validation items as JSONL for decide_local.py."""
    _, _, todo = candidates()
    extra = {(k, x, v, su) for (k, _), rows in VALIDATION.items() for x, su, v in rows} | set(PROBES)
    path = os.path.join(HERE, "out", SETTINGS[SETTING][2])
    with open(path, "w", encoding="utf-8") as f:
        for k, s, w, su in sorted(todo | extra):
            f.write(json.dumps({"kind": k, "sentence": s, "word": w, "subj": su}) + "\n")
    print(f"{len(todo)} candidates + {len(extra)} validation items -> {path}")


def rates(path, threshold=0.5):
    """Rates per 1,000 sentences from a judgment file (jev_cache.jsonl or a decide_local.py output); validation too."""
    judged = {(r["kind"], r["sentence"], r["word"], r["subj"]): r["p"] for r in map(json.loads, open(path, encoding="utf-8"))}
    for (k, want), rows in VALIDATION.items():
        ps = [judged.get((k, x, v, su)) for x, su, v in rows]
        ok = sum((p > threshold) == want for p in ps if p is not None)
        print(f"validation {k} {'+' if want else '-'}: {ok}/{len(rows)}  " + " ".join(f"{v}={p:.2f}" for (x, su, v), p in zip(rows, ps) if p is not None))
    for k, x, v, su in PROBES:
        if (k, x, v, su) in judged:
            print(f"probe: {x!r} -> {judged[(k, x, v, su)]:.2f}")
    parsed, mostly_noun, _ = candidates()
    rows = []
    for src, df in parsed.items():
        den = sum(judged.get(("denominal", s, v, ""), 0) > threshold for s, vs in zip(df.text, df.verbs) for v in vs if v in mostly_noun)
        ina = sum(judged.get(("inanimate", s, x.split("|")[2], x.split("|")[0]), 0) > threshold for s, sj in zip(df.text, df.subj) for x in sj)
        rows.append({"source": src, "sentences": len(df), "denominal": 1000 * den / len(df), "inanimate": 1000 * ina / len(df)})
    out = pd.DataFrame(rows)
    print(out.round(2).to_string(index=False))
    return out


PAPER_HUMAN = {"denominal": 41.3, "inanimate": 15.2}  # PERSUADE essays, recovered from the paper's figure


def calibrated(path, out_csv=None, essay_path=None):
    """Per feature: the threshold at which the human ESSAYS match the paper's human rate (the paper reports no chat or
    code human rate for these two features), then every source's rate in the current setting at that threshold."""
    import numpy as np
    judged = {(r["kind"], r["sentence"], r["word"], r["subj"]): r["p"] for r in map(json.loads, open(path, encoding="utf-8"))}
    if essay_path:  # chat: thresholds come from the essay judgments
        judged = {**{(r["kind"], r["sentence"], r["word"], r["subj"]): r["p"] for r in map(json.loads, open(essay_path, encoding="utf-8"))}, **judged}
    e_parsed, e_noun, _ = candidates("essay")
    parsed, mostly_noun, _ = candidates()

    def probs(df, kind, noun=None):
        noun = mostly_noun if noun is None else noun
        if kind == "denominal":
            return [judged.get(("denominal", s, v, ""), 0.0) for s, vs in zip(df.text, df.verbs) for v in vs if v in noun]
        return [judged.get(("inanimate", s, x.split("|")[2], x.split("|")[0]), 0.0) for s, sj in zip(df.text, df.subj) for x in sj]

    h = e_parsed["human-persuade"]
    thr = {}
    for kind, anchor in PAPER_HUMAN.items():
        ps = np.sort(np.array(probs(h, kind, e_noun)))[::-1]
        k = int(round(anchor * len(h) / 1000))  # number of human positives the paper's rate implies
        thr[kind] = float(ps[k]) if k < len(ps) else 0.0
        print(f"{kind}: threshold {thr[kind]:.3f} gives the paper's human rate {anchor}/1k ({k} of {len(ps)} human candidates)")
    rows = []
    for src, df in parsed.items():
        r = {"source": src, "sentences": len(df)}
        for kind in PAPER_HUMAN:
            ps = np.array(probs(df, kind))
            r[kind] = 1000 * (ps > thr[kind]).sum() / len(df)
            r[kind + "_at_0.5"] = 1000 * (ps > 0.5).sum() / len(df)
        rows.append(r)
    out = pd.DataFrame(rows)
    print(out.round(2).to_string(index=False))
    if out_csv:
        out.to_csv(out_csv, index=False)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "calibrated":
        calibrated(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "rates":
        out = rates(sys.argv[2])
        if len(sys.argv) > 3:
            out.to_csv(sys.argv[3], index=False)
    else:
        {"validate": validate, "run": run, "export": export}[cmd]()
