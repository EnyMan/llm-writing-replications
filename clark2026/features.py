"""Structural features of Clark, Martin & Mahowald (2026), essay setting (appendix "Processing" and "Analysis details").

clean_essay(text, is_model) -> list of (paragraph, is_heading)
parse_docs(texts) -> per-essay counts + per-sentence dependency arcs, from spaCy en_core_web_trf
Measures (pooled per source):
  orc_rel / orc_norel  object relative clauses with / without a relativizer, per 1,000 words
  stacked              stacked noun compounds (3+ nouns in a compound chain), per 1,000 words
  dep_len              mean dependency length, directly standardized to Wikipedia's sentence-length mix
"""
import re
import numpy as np
from syntok.segmenter import split as syntok_split
from syntok.tokenizer import Tokenizer

STAND_IN = "Zzname"  # replaces masked names / [Your Name] slots; excluded from noun counts and n-grams
PLACEHOLDER = re.compile(r"\b(?:Generic_[A-Za-z_]+|PROPER_NAME|LOCATION_NAME|STUDENT_NAME|TEACHER_NAME|RESTAURANT_NAME)\b"
                         r"|\[(?:your|insert|student|teacher|principal|school)[^\]]{0,40}\]", re.I)
BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30)]
RELATIVIZERS = {"that", "which", "who", "whom", "whose", "when", "where", "why"}
# "the reason (why) they came", "the time (when) we left": adverbial relatives, not object relatives
ADVERBIAL_HEADS = {"reason", "time", "way", "day", "moment", "place", "year", "week", "period", "night", "morning", "point"}
_tok = Tokenizer()


def _normalize(t):
    t = t.replace("\r\n", "\n").replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    t = t.replace(" ", " ").replace("—", " -- ").replace("–", "-")
    t = re.sub(r"[ \t]+", " ", t)
    return PLACEHOLDER.sub(STAND_IN, t)


def _strip_markdown(t):
    t = re.split(r"\n\s*(?:-{3,}|\*{3,}|_{3,})\s*\n", t)[0]  # text set apart by a horizontal rule (a note) is removed
    t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", t)  # links: keep text
    t = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", t, flags=re.M)  # list markers
    return t


def clean_essay(text, is_model):
    """Return [(paragraph, is_heading)]."""
    t = _normalize(text)
    if is_model:
        t = _strip_markdown(t)
    lines = [l.strip() for l in t.split("\n")]
    paras = []
    for l in lines:
        if not l:
            continue
        heading = bool(re.match(r"^#{1,6}\s", l))
        paras.append((re.sub(r"^#{1,6}\s*", "", l), heading))
    # drop a leading heading (model title), then up to four leading title / name / date lines
    if is_model and paras and paras[0][1]:
        paras = paras[1:]
    for _ in range(4):
        if paras and len(paras[0][0].split()) <= 8 and not re.search(r"[.!?:,]$", paras[0][0]):
            paras = paras[1:]
        else:
            break
    # a short line with no final punctuation in a model essay is a section heading
    if is_model:
        paras = [(p, h or (len(p.split()) <= 8 and not re.search(r"[.!?:;,\"')]$", p))) for p, h in paras]
    return paras


CODE_WORD = "Zzcode"  # placeholder for inline code, URLs, paths, identifiers, inline math; n-grams with it are skipped
_FENCE = re.compile(r"```.*?```|~~~.*?~~~", re.S)
_HTML = re.compile(r"<[^>\n]+>")
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_INLINE = re.compile(r"`[^`\n]+`|\$[^$\n]+\$|\\\(.+?\\\)|https?://\S+|(?<!\w)(?:[\w.-]+/)+[\w.-]+|\b\w+\(\)|\b[a-z]+_[a-z0-9_]+\b|\b[a-z]+[A-Z]\w*\b")
_SYMBOLS = re.compile(r"[←-⇿➔➡⬅-⬇\U0001F000-\U0001FAFF☀-➿=<>±×÷≤≥≈∞]+")


def clean_chat(text):
    """A chat or code reply -> [(segment, is_heading)] following the paper's appendix "Processing": emphasis and links
    keep their text; fenced code, HTML and images go; inline code / URLs / paths / identifiers / math become one
    placeholder word; arrows, emoji and stray math symbols are removed and end the stretch; nothing crosses a heading,
    list item, table cell or paragraph break."""
    t = _normalize(_FENCE.sub("\n\n", text))
    t = _IMG.sub(" ", _HTML.sub(" ", t))
    t = _INLINE.sub(CODE_WORD, t)
    t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", t)
    out = []
    for line in t.split("\n"):
        line = line.strip()
        if not line or re.fullmatch(r"[-*_|: ]{3,}", line):  # rules, table separators
            continue
        heading = bool(re.match(r"^#{1,6}\s", line))
        line = re.sub(r"^#{1,6}\s*|^\s*(?:[-*+]|\d+[.)])\s+|^>\s*", "", line)
        for cell in line.split("|"):  # table cells are separate stretches
            for stretch in _SYMBOLS.split(cell):  # a stray symbol ends the stretch
                stretch = stretch.strip()
                if re.search(r"[A-Za-z]", stretch):
                    out.append((stretch, heading))
    return out


def sentences(paras):
    """Sentences (as strings) from non-heading paragraphs; none crosses a paragraph."""
    out = []
    for p, heading in paras:
        if heading:
            continue
        for s in syntok_split(_tok.tokenize(p)):  # split() yields sentences (lists of tokens)
            out.append("".join(t.spacing + t.value for t in s).strip())
    return [s for s in out if s]


def _is_word(t):
    return not t.is_punct and not t.is_space


def sent_features(sent):
    """Counts and dependency arcs for one parsed sentence (spaCy Span)."""
    words = [t for t in sent if _is_word(t)]
    pos = {t.i: k for k, t in enumerate(words)}  # word positions, punctuation not counted
    arcs = [abs(pos[t.i] - pos[t.head.i]) for t in words if t.dep_ != "ROOT" and t.head.i in pos and t.head is not t]
    # variant: distance in raw token positions (punctuation between the two ends counts)
    arcs_raw = [abs(t.i - t.head.i) for t in words if t.dep_ != "ROOT" and t.head.i in pos and t.head is not t]
    orc_rel = orc_norel = stacked = stacked_common = 0
    for t in sent:
        if t.dep_ == "relcl":
            kids = list(t.children)
            relz = [c for c in kids if c.tag_ in ("WDT", "WP", "WP$", "WRB") or c.lower_ in RELATIVIZERS]
            if relz:
                if any(c.dep_ == "dobj" and c.tag_ != "WRB" for c in relz):
                    orc_rel += 1  # "the policy that you are considering"
            elif (any(c.dep_ in ("nsubj", "nsubjpass") for c in kids)
                  and not any(c.dep_ in ("dobj", "ccomp", "xcomp") for c in kids)
                  and not any(c.dep_ == "prep" and not any(g.dep_ in ("pobj", "pcomp") for g in c.children) for c in t.subtree)  # stranded prep: oblique
                  and not any(c.dep_ == "aux" and c.lower_ == "to" for c in kids)  # infinitival relative
                  and not any(c.dep_ == "mark" for c in kids)
                  and t.head.lemma_.lower() not in ADVERBIAL_HEADS):
                orc_norel += 1  # reduced: overt subject, object gap ("the job you hate")
        if t.pos_ in ("NOUN", "PROPN") and t.dep_ != "compound" and t.text != STAND_IN:
            for only_common in (False, True):
                ok = ("NOUN",) if only_common else ("NOUN", "PROPN")
                if t.pos_ not in ok:
                    continue
                n, frontier = 1, [t]
                while frontier:
                    nxt = [c for x in frontier for c in x.children if c.dep_ == "compound" and c.pos_ in ok and c.text != STAND_IN]
                    n += len(nxt)
                    frontier = nxt
                if only_common:
                    stacked_common += n >= 3
                else:
                    stacked += n >= 3
    # candidates for the Jev-judged features:
    #   verbs: lemma of every verb (denominal if the lemma is mostly a noun in the human corpus)
    #   subj: active verb with an inanimate-looking subject (a noun, or it/this/that/these/those/which)
    verbs = [t.lemma_.lower() for t in sent if t.pos_ == "VERB" and not _modifier_participle(t)]
    subj = [f"{t.lower_}|{t.head.lemma_.lower()}|{t.head.text}" for t in sent
            if t.dep_ == "nsubj" and t.head.pos_ == "VERB" and t.text != STAND_IN and _active_clause_verb(t.head)
            and (t.pos_ == "NOUN" or t.lower_ in ("it", "this", "that", "these", "those", "which"))]
    return {"words": len(words), "arcs": arcs, "arcs_raw": arcs_raw, "orc_rel": orc_rel, "orc_norel": orc_norel,
            "stacked": stacked, "stacked_common": stacked_common,
            "verbs": verbs, "nouns": [t.lemma_.lower() for t in sent if t.pos_ == "NOUN"], "subj": subj}


def _modifier_participle(v):
    """A participle modifying a noun ("documented differences", "distanced learning"): not a verb in use."""
    return v.dep_ in ("amod", "compound") or (v.tag_ in ("VBN", "VBG") and v.dep_ == "acl")


def _active_clause_verb(v):
    """The verb heads an active clause: finite, or with an auxiliary (can feel, has forgotten, is reaching);
    not passive, not a bare participle (student-designed, half ... doing)."""
    kids = list(v.children)
    if _modifier_participle(v) or any(c.dep_ == "auxpass" for c in kids):
        return False
    return v.tag_ in ("VBD", "VBP", "VBZ") or any(c.dep_ == "aux" for c in kids)


def bin_of(n):
    for k, (lo, hi) in enumerate(BINS):
        if lo <= n <= hi:
            return k
    return None  # > 30 words: dropped from dependency length


def pooled(rows, weights):
    """rows: per-sentence dicts (words, arcs, orc_rel, orc_norel, stacked). Pooled per-source measures."""
    words = sum(r["words"] for r in rows)

    def std_len(key):
        sums, cnts = np.zeros(len(BINS)), np.zeros(len(BINS))
        for r in rows:
            b = bin_of(r["words"])
            if b is not None and len(r[key]):  # arcs may be a numpy array after parquet
                sums[b] += sum(r[key])
                cnts[b] += len(r[key])
        means = np.divide(sums, cnts, out=np.full(len(BINS), np.nan), where=cnts > 0)
        return float(np.nansum(means * weights) / weights[~np.isnan(means)].sum())

    per1k = lambda k: 1000 * sum(r[k] for r in rows if k in r) / words
    return {"words": words, "orc_rel": per1k("orc_rel"), "orc_norel": per1k("orc_norel"),
            "stacked": per1k("stacked"), "stacked_common": per1k("stacked_common"),
            "dep_len": std_len("arcs"), "dep_len_raw": std_len("arcs_raw") if "arcs_raw" in rows[0] else np.nan}


def arc_weights(rows):
    """Wikipedia standard: share of all dependency arcs that come from sentences in each length bin."""
    cnts = np.zeros(len(BINS))
    for r in rows:
        b = bin_of(r["words"])
        if b is not None:
            cnts[b] += len(r["arcs"])
    return cnts / cnts.sum()


if __name__ == "__main__":  # self-check on hand-made cases (needs en_core_web_trf)
    import spacy
    nlp = spacy.load("en_core_web_trf")
    f = lambda s: sent_features(next(nlp(s).sents))
    assert f("That's the job you hate.")["orc_norel"] == 1
    assert f("These are causes they are passionate about.")["orc_norel"] == 0  # stranded preposition
    assert f("There are times when students need rest.")["orc_norel"] == 0
    assert f("That is the reason they come to school.")["orc_norel"] == 0
    assert f("It is a park for all residents to enjoy.")["orc_norel"] == 0
    assert f("This is the policy that you are considering.")["orc_rel"] == 1
    assert f("The man who likes popcorn left.")["orc_rel"] == 0 and f("The man who likes popcorn left.")["orc_norel"] == 0
    assert f("We need a community service honor cord.")["stacked"] == 1
    assert f("We need a coffee shop.")["stacked"] == 0
    p = clean_essay("# My Essay\n\nDear Principal,\n\nPhones help **students**. Generic_Name agrees.\n\n---\n*Word count: 500*", True)
    assert all("Word count" not in x for x, _ in p) and p[0][0].startswith("Dear") and STAND_IN in p[1][0], p
    assert sentences(p) == ["Dear Principal,", "Phones help students.", f"{STAND_IN} agrees."], sentences(p)
    c = clean_chat("## Why it works\n\nThe **docstring** promises a fix → see `is_file()` at https://x.io.\n\n- first item\n- second item\n\n```py\nprint(1)\n```")
    assert ("Why it works", True) in c and ("first item", False) in c and not any("print" in s for s, _ in c), c
    assert any(CODE_WORD in s for s, _ in c) and not any("→" in s or "https" in s for s, _ in c), c
    subj = lambda s: [x.split("|")[1] for x in f(s)["subj"]]
    assert subj("The docstring promises they're skipped.") == ["promise"]
    assert subj("School can feel like a rut.") == ["feel"]
    assert "design" not in subj("Making summer projects student-designed would enable students to grow.")
    assert subj("The projects were designed by students.") == []
    assert "document" not in f("They balance jobs, caregiving, or documented learning differences.")["verbs"]  # amod
    print("ok")
